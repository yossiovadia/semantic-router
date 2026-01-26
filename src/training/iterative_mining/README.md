# Iterative Hard-Negative Mining for Domain-Aware Embeddings

Implementation of domain-aware embedding training using iterative hard-negative mining with LLM-as-judge, based on the paper ["Adaptation of Embedding Models to Financial Filings via LLM Distillation"](https://arxiv.org/pdf/2512.08088).

## Quick Start for New Session

**Current Status:** POC validated. Anti-forgetting fixes work. Best result: +0.68% MRR@5 improvement on held-out test set (stable, no degradation).

**AWS Instance:** g5.12xlarge at `18.116.29.245` (may need to restart if stopped)
- SSH key: `~/.ssh/router-team-us-east2.pem`
- vLLM server should be started manually if needed
- Models saved at: `~/semantic-router/src/training/iterative_mining/models/anti_forgetting/`

### To Resume Work

```bash
# SSH to AWS instance
ssh -i ~/.ssh/router-team-us-east2.pem ubuntu@18.116.29.245

# Navigate to project
cd ~/semantic-router/src/training/iterative_mining

# Start vLLM server (if not running)
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 &

# Check existing models
ls -la models/anti_forgetting/
```

---

## Overview

This POC trains a medical domain-specialized embedding model using the MedQuAD dataset and iterative hard-negative mining.

## Methodology

1. **Retrieve** top-k candidates for each query using current embedding model
2. **Judge** each (query, candidate) pair with an LLM (1-4 relevance score)
3. **Mine** hard examples:
   - Hard positives: LLM says relevant (score >= 3) but model ranked low (rank > 10)
   - Hard negatives: LLM says irrelevant (score <= 2) but model ranked high (rank <= 5)
4. **Fine-tune** model on triplets (query, positive, negative) using TripletLoss
5. **Repeat** for multiple iterations

---

## Key Findings

### Bug Fixes Applied

1. **Wrong loss function** (CRITICAL): Original used `label=0.0` for negatives with CosineSimilarityLoss, which targets neutral similarity instead of pushing apart. Fixed by switching to TripletLoss.

2. **Learning rate 40,000x too high**: Original used `2e-5`, paper uses `5e-7`. Even `5e-7` caused issues at scale.

3. **Catastrophic forgetting at scale**: 1000 queries x 10 iterations caused -34% degradation by iteration 9.

### Anti-Forgetting Fixes (in `robust_iterative_miner.py`)

```python
# In fine_tune() method:
self.model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,                    # Reduced from 2
    optimizer_params={'lr': 1e-8},  # 50x lower than paper
    weight_decay=0.01,           # L2 regularization
)

# TripletLoss with larger margin
train_loss = losses.TripletLoss(
    model=self.model,
    triplet_margin=0.2  # Increased from 0.1
)

# Cap triplets per iteration
MAX_TRIPLETS = 500  # Was 5000+
```

### Results Summary

| Run | Configuration | Best Result | Status |
|-----|---------------|-------------|--------|
| Initial (buggy) | 100q x 3i | -69% to +3% | Fixed bugs |
| Full scale | 1000q x 10i, lr=5e-7 | **-34.42%** | Catastrophic forgetting |
| Anti-forgetting | 1000q x 5i, lr=1e-8 | **+0.68%** | Stable, no degradation |

### Key Insight

**Trade-off between stability and improvement magnitude:**
- Aggressive parameters (higher LR, more triplets) -> larger gains but catastrophic forgetting
- Conservative parameters (lower LR, fewer triplets) -> stable but modest gains (+0.68%)

The paper's +15-25% gains may require:
- Larger corpus (financial domain vs our medical)
- Better LLM judge (GPT-4 vs Qwen-7B)
- Different base model with more room to improve
- Domain-specific tuning of mining criteria

---

## Files

### Main Scripts

- **`robust_iterative_miner.py`** - Production-ready miner with checkpointing and anti-forgetting fixes
- **`iterative_miner.py`** - Original implementation (kept for reference)

### Analysis Utilities

- **`analyze_dataset.py`** - Analyze query diversity, corpus coverage, hard example potential
- **`analyze_hard_examples.py`** - Analyze mined hard examples across iterations

### Data Preparation

- `prepare_corpus.py` - Load MedQuAD and prepare corpus
- `prepare_queries.py` - Prepare queries from MedQuAD questions
- `evaluate.py` - Evaluation script
- `requirements.txt` - Python dependencies

### Documentation

- **`README.md`** - This file
- **`ANALYSIS_SUMMARY.md`** - Detailed analysis of all experiments and results

---

## Running Training

### Quick Test (verify setup)
```bash
python robust_iterative_miner.py \
  --data-dir data \
  --num-queries 20 \
  --num-iterations 2 \
  --llm-endpoint http://localhost:8000/v1 \
  --output-dir models/quick_test \
  --checkpoint-dir checkpoints/quick_test
```

### Full Training
```bash
python robust_iterative_miner.py \
  --data-dir data \
  --num-queries 1000 \
  --num-iterations 5 \
  --llm-endpoint http://localhost:8000/v1 \
  --output-dir models/full \
  --checkpoint-dir checkpoints/full
```

### Resume from Checkpoint
```bash
python robust_iterative_miner.py \
  --start-iteration 3 \
  [other args]
```

---

## Evaluation

```python
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load test data
with open('data/test_queries.pkl', 'rb') as f:
    test_queries = pickle.load(f)
with open('data/corpus_chunks.pkl', 'rb') as f:
    corpus_chunks = pickle.load(f)

# Load model
model = SentenceTransformer('models/anti_forgetting/iteration_3')

# Compute embeddings
query_embs = model.encode([q['query'] for q in test_queries])
corpus_embs = model.encode([c['text'] for c in corpus_chunks])

# Compute MRR@5
similarities = np.dot(query_embs, corpus_embs.T)
mrr = 0.0
for i, query in enumerate(test_queries):
    gt_ids = set(query['ground_truth_chunk_ids'])
    ranked = np.argsort(similarities[i])[::-1][:5]
    for rank, idx in enumerate(ranked, 1):
        if idx in gt_ids:
            mrr += 1.0 / rank
            break
mrr /= len(test_queries)
print(f'MRR@5: {mrr:.4f}')
```

---

## Next Steps (if continuing this work)

1. **Try different base model**: BGE, E5, or larger models may have more room to improve
2. **Better LLM judge**: GPT-4 or Claude instead of Qwen-7B
3. **Hyperparameter search**: Find optimal point between aggressive (forgetting) and conservative (limited gains)
4. **Different domain**: Financial/legal domains may show larger improvements (per paper)
5. **More sophisticated anti-forgetting**: EWC, knowledge distillation, or replay buffers

---

## Data Format

### Queries (train_queries.pkl, test_queries.pkl)
```python
[
    {
        'query': 'What are the symptoms of diabetes?',
        'ground_truth_chunk_ids': [42, 156, 789]  # Indices into corpus_chunks
    },
    ...
]
```

### Corpus (corpus_chunks.pkl)
```python
[
    {
        'text': 'Diabetes symptoms include increased thirst...',
        'source': 'MedQuAD',
        # ... other metadata
    },
    ...
]
```
