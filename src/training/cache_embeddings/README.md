# Cache Embedding Training Pipeline

**Research-validated pipeline for training domain-specific cache embedding models using LLM-generated synthetic data.**

Based on: [arXiv:2504.02268v1](https://arxiv.org/pdf/2504.02268v1)

## Overview

This pipeline trains specialized embedding models for semantic caching by:
1. Taking unlabeled domain queries (medical, legal, history, etc.)
2. Using LLM to generate paraphrases (positive pairs) and hard negatives
3. Training LoRA-adapted models with Multiple Negatives Ranking loss
4. Producing lightweight, domain-specialized cache embeddings

## Quick Start

### Prerequisites

- **Ollama** installed locally with a model (e.g., `qwen2.5:1.5b`)
- Python 3.11+ with dependencies: `transformers`, `torch`, `peft`, `sentence-transformers`

### Complete Workflow (Medical Domain Example)

```bash
# 1. Prepare unlabeled queries (44K medical queries provided)
ls data/cache_embeddings/medical/unlabeled_queries.jsonl

# 2. Generate training data using LLM (Ollama)
# This creates paraphrases + hard negatives for each query
# ~12 min for 500 queries, ~4 hours for full 44K on CPU
python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/medical/unlabeled_queries.jsonl \
  --output data/cache_embeddings/medical/augmented.jsonl \
  --model qwen2.5:1.5b \
  --paraphrases 3 \
  --negatives 2 \
  --workers 10

# 3. Convert to triplet format for training
python3 << 'EOF'
import json

# Read augmented data
with open('data/cache_embeddings/medical/augmented.jsonl') as f:
    data = [json.loads(line) for line in f]

# Group by positive (canonical query)
positive_groups = {}
for item in data:
    if 'positive' in item:
        positive = item['positive']
        if positive not in positive_groups:
            positive_groups[positive] = {'paraphrases': [], 'negatives': []}
        positive_groups[positive]['paraphrases'].append(item['anchor'])

# Add negatives
for item in data:
    if 'hard_negative' in item:
        anchor = item['anchor']
        for positive, group in positive_groups.items():
            if anchor in group['paraphrases'] or anchor == positive:
                group['negatives'].append(item['hard_negative'])
                break

# Create triplets
triplets = []
for positive, group in positive_groups.items():
    for para in group['paraphrases']:
        for neg in group['negatives']:
            triplets.append({
                'anchor': para,
                'positive': positive,
                'hard_negative': neg
            })

# Save
with open('data/cache_embeddings/medical/training.jsonl', 'w') as f:
    for t in triplets:
        f.write(json.dumps(t) + '\n')

print(f"Created {len(triplets)} triplets")
EOF

# 4. Train LoRA model (~8 min for 2.8K samples on CPU, ~2 hours for full dataset)
python3 src/training/cache_embeddings/lora_trainer.py \
  --train-data data/cache_embeddings/medical/training.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/medical-cache-lora \
  --epochs 1 \
  --batch-size 32 \
  --lr 2e-5 \
  --temperature 0.05

# 5. Model ready! (582KB LoRA adapter)
ls models/medical-cache-lora/
```

## Methodology

### 1. Data Generation (`generate_training_data.py`)

**Input:** Unlabeled domain queries
```json
{"query": "How to diagnose Pertussis?"}
```

**Process:** Uses Ollama LLM to generate:
- **Paraphrases** (positive pairs): Same meaning, different wording
- **Hard negatives**: Related but different queries

**Output:** Augmented training samples
```json
{"anchor": "What are the methods for diagnosing pertussis?", "positive": "How to diagnose Pertussis?", "is_duplicate": 1}
{"anchor": "How to diagnose Pertussis?", "hard_negative": "What are symptoms of pertussis?", "is_duplicate": 0}
```

**Key Parameters:**
- `--paraphrases 3`: Generate 3 paraphrases per query
- `--negatives 2`: Generate 2 hard negatives per query
- `--workers 10`: Parallel workers for faster processing
- `--max-queries N`: Limit for testing (e.g., 500)

**Augmentation Factor:** ~5x (500 queries → 2,449 samples → 2,854 triplets)

### 2. LoRA Training (`lora_trainer.py`)

**Input:** Triplets with anchor, positive, hard_negative

**Training:**
- **Base model:** `sentence-transformers/all-MiniLM-L12-v2` (33.5M params)
- **LoRA adapter:** Only 147K trainable params (0.44%)
- **Loss:** Multiple Negatives Ranking (MNR) with temperature=0.05
- **Epochs:** 1 (per paper recommendation)

**Output:** Lightweight domain-specific model (582KB)

**Performance:**
- Training time: ~1.7s per batch (32 samples)
- Final loss: ~0.086 (lower is better)
- Model size: 582KB vs 130MB full model

## Pipeline Architecture

```
Unlabeled Queries (44K medical questions)
    ↓
LLM Augmentation (Ollama qwen2.5:1.5b)
    ↓
Augmented Data (~220K samples: paraphrases + hard negatives)
    ↓
Triplet Creation (~220K triplets with anchor/positive/negative)
    ↓
LoRA Training (1 epoch, MNR loss, temp=0.05)
    ↓
Domain-Specific Cache Model (582KB, 0.44% trainable params)
```

## Files

| File | Purpose | Usage |
|------|---------|-------|
| `generate_training_data.py` | LLM-based data augmentation | Generate paraphrases + hard negatives |
| `lora_trainer.py` | LoRA fine-tuning | Train domain-specific models |
| `hard_negative_miner.py` | Similarity-based mining | Alternative to LLM negatives |
| `evaluate_model.py` | Model evaluation | Compare LoRA vs baseline |

## Validated Dataset: Medical Domain

**Source:** MedQuAD (Medical Question Answering Dataset)
- **Queries:** 44,603 medical questions
- **License:** CC BY 4.0
- **Format:** `{"query": "medical question"}`
- **File:** `data/cache_embeddings/medical/unlabeled_queries.jsonl`

**Example queries:**
- "How to diagnose Pertussis?"
- "What are the symptoms of diabetes?"
- "What treatments are available for hypertension?"

## Training Configuration

### LLM Augmentation
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | qwen2.5:1.5b | Fast, good quality |
| Paraphrases | 3 | Sufficient diversity |
| Hard negatives | 2 | Challenging but learnable |
| Workers | 10 | Parallel processing |

### LoRA Training
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | all-MiniLM-L12-v2 | Strong baseline |
| LoRA rank | 8 | Efficient adaptation |
| LoRA alpha | 32 | 4x rank (standard) |
| Epochs | 1 | Per paper recommendation |
| Batch size | 32 | Good for 8K+ samples |
| Learning rate | 2e-5 | Standard for BERT |
| Temperature | 0.05 | MNR loss scaling |

## Scaling to Full Dataset (4090 GPU)

For full 44K queries on GPU:

```bash
# Expected output: ~220K training samples
# Expected time: ~4 hours for augmentation, ~2 hours for training

python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/medical/unlabeled_queries.jsonl \
  --output data/cache_embeddings/medical/augmented_full.jsonl \
  --model qwen2.5:1.5b \
  --paraphrases 3 \
  --negatives 2 \
  --workers 20

# Then follow triplet creation + training steps above
```

## Next Steps

1. **Validate on full medical dataset** (44K queries on 4090 GPU)
2. **Expand to other domains:** history, legal, finance, etc.
3. **Production integration:** Deploy models in semantic router
4. **Evaluation framework:** Build proper test sets with held-out queries

## Key Insights

### ✅ What Works
- **LLM-generated synthetic data:** High-quality paraphrases and hard negatives
- **LoRA fine-tuning:** Efficient (0.44% params) and effective
- **Single epoch training:** Sufficient per paper validation
- **Domain specialization:** Unlabeled queries from specific domains

### ❌ Previous Approach (Deprecated)
- ~~Manual template-based data~~ → Replaced with LLM generation
- ~~Hardcoded topic lists~~ → Replaced with real unlabeled queries
- ~~11 small manually-created domains~~ → Focused on quality over quantity

## References

- **Paper:** [Semantic Cache Embeddings (arXiv:2504.02268v1)](https://arxiv.org/pdf/2504.02268v1)
- **Base Model:** [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- **Dataset:** [MedQuAD](https://github.com/abachaa/MedQuAD) (CC BY 4.0)
- **LoRA:** [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- **MNR Loss:** [Sentence-BERT](https://arxiv.org/abs/1908.10084)

## License

- Code: Follow semantic-router repository license
- MedQuAD dataset: CC BY 4.0
