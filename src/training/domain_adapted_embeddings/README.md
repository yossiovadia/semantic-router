# Domain-Adapted Embedding Fine-Tuning

Fine-tune embedding models for improved retrieval in specific domains using iterative hard-negative mining.

**Based on:** ["Distilling an LLM's Wisdom: A Framework for Creating Domain Adapted Financial Embedding Models"](https://arxiv.org/abs/2512.08088)

**Result:** +71.18% MRR@5 improvement on MedQuAD medical dataset.

---

## Pre-trained Models

| Domain | Model | Base Model | Improvement |
|--------|-------|------------|-------------|
| Medical | [mmbert-embed-medical](https://huggingface.co/llm-semantic-router/mmbert-embed-medical) | mmbert-embed-32k-2d-matryoshka | +71% MRR@5 |

```python
# Use pre-trained domain-adapted model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("llm-semantic-router/mmbert-embed-medical", trust_remote_code=True)
embeddings = model.encode(["What are the symptoms of diabetes?"])
```

---

## Train Your Own

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (using MedQuAD as example)
python prepare_data.py --source huggingface --dataset keivalya/MedQuad-MedicalQnADataset

# 3. Train
python train.py --data-dir data --output-dir models/trained

# 4. Use trained model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("models/trained/best", trust_remote_code=True)
embeddings = model.encode(["Your query here"])
```

---

## How It Works

### The Problem
General-purpose embedding models work well across domains but may underperform on specialized content (medical, legal, financial). Fine-tuning on domain-specific data improves retrieval quality.

### The Solution
**Iterative hard-negative mining:**

1. **Mine triplets** from current model's rankings using ground-truth labels
   - **Hard triplets:** Ground-truth docs ranked low (5-100) + non-GT docs ranked high (1-15)
   - **Easy triplets:** Ground-truth docs already ranked high (anti-forgetting)

2. **Fine-tune** with TripletLoss (margin=0.1)

3. **Repeat** for 2 iterations (accumulating triplets)

> **Note:** The original paper uses an LLM-as-judge to score relevance. Our implementation
> uses ground-truth labels instead (which doc answers which query), achieving similar results
> without requiring an LLM server. This works when you have labeled Q&A data.

```
Iteration 1: Mine -> Train -> Evaluate
Iteration 2: Mine -> Accumulate -> Train -> Evaluate
            |
      Best model saved
```

---

## Data Preparation

### Option 1: HuggingFace Dataset

```bash
python prepare_data.py --source huggingface --dataset keivalya/MedQuad-MedicalQnADataset
```

### Option 2: Custom JSON File

Create a JSON file with Q&A pairs:
```json
[
  {"question": "What are the symptoms of diabetes?", "answer": "Diabetes symptoms include..."},
  {"question": "How is hypertension treated?", "answer": "Hypertension treatment involves..."}
]
```

Then run:
```bash
python prepare_data.py --source json --input-file your_data.json
```

### Option 3: JSONL File

```bash
python prepare_data.py --source jsonl --input-file your_data.jsonl
```

### Output Files

```
data/
├── corpus_chunks.pkl      # Document chunks for retrieval
├── train_queries.pkl      # Training queries with ground-truth
├── test_queries.pkl       # Test queries for evaluation
├── corpus_sample.json     # Sample for inspection
└── queries_sample.json    # Sample for inspection
```

---

## Training

### Basic Training

```bash
python train.py --data-dir data --output-dir models/trained
```

### Custom Configuration

```bash
python train.py \
    --data-dir data \
    --output-dir models/trained \
    --base-model llm-semantic-router/mmbert-embed-32k-2d-matryoshka \
    --iterations 2 \
    --learning-rate 5e-5 \
    --margin 0.1
```

### All Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | (required) | Directory with prepared data |
| `--output-dir` | `models/trained` | Where to save trained model |
| `--base-model` | `llm-semantic-router/mmbert-embed-32k-2d-matryoshka` | Base model to fine-tune |
| `--iterations` | `2` | Number of training iterations |
| `--learning-rate` | `5e-5` | Learning rate (**critical - use 5e-5, not lower**) |
| `--epochs` | `2` | Epochs per iteration |
| `--batch-size` | `8` | Training batch size |
| `--margin` | `0.1` | TripletLoss margin |
| `--easy-to-hard-ratio` | `2` | Ratio of easy:hard triplets |
| `--num-queries` | all | Limit training queries |

---

## Critical Hyperparameters

These settings are essential for good results:

| Parameter | Value | Why It Matters |
|-----------|-------|----------------|
| **learning-rate** | `5e-5` | Lower values (5e-7) result in minimal improvement |
| **margin** | `0.1` | Default margin (~5.0) causes severe forgetting |
| easy:hard ratio | `2:1` | Easy triplets reinforce existing knowledge |
| Accumulate triplets | Yes | Don't replace, add to training set each iteration |
| iterations | `2` | Second iteration provides additional gains |

---

## Results

### MedQuAD Medical Dataset (13,125 training queries)

| Metric | Baseline | Iteration 1 | Iteration 2 | Improvement |
|--------|----------|-------------|-------------|-------------|
| **MRR@5** | 0.4354 | 0.6528 (+49.93%) | **0.7453** | **+71.18%** |
| **Recall@5** | 0.4896 | 0.5918 | **0.6900** | **+40.92%** |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | llm-semantic-router/mmbert-embed-32k-2d-matryoshka |
| Learning rate | 5e-5 |
| Batch size | 8 |
| Epochs per iteration | 2 |
| Margin | 0.1 |
| Easy:Hard ratio | 2:1 |
| GPU | NVIDIA L4 (24GB) |

---

## Using the Trained Model

```python
from sentence_transformers import SentenceTransformer

# Load trained model
model = SentenceTransformer("models/trained/best", trust_remote_code=True)

# Encode queries and documents
query_embedding = model.encode("What are the symptoms of diabetes?")
doc_embeddings = model.encode([
    "Diabetes symptoms include increased thirst...",
    "The weather today is sunny...",
])

# Compute similarity
similarities = query_embedding @ doc_embeddings.T
print(similarities)  # Higher score = more relevant
```

---

## Adapting to Other Domains

1. **Collect Q&A pairs** for your domain
   - Question: Natural language query
   - Answer: Document/passage that answers the question

2. **Prepare data**
   ```bash
   python prepare_data.py --source json --input-file your_domain_qa.json
   ```

3. **Train**
   ```bash
   python train.py --data-dir data --output-dir models/your_domain
   ```

4. **Evaluate** - Check training_summary.json for metrics

---

## Files

```
domain_adapted_embeddings/
├── README.md           # This file
├── train.py            # Main training script
├── prepare_data.py     # Data preparation (JSON, JSONL, HuggingFace)
├── requirements.txt    # Python dependencies
└── .gitignore          # Ignore data/models
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- sentence-transformers 2.2+
- GPU recommended (NVIDIA L4 or better with 24GB+ VRAM)

Install:
```bash
pip install -r requirements.txt
```

---

## References

- **Paper:** [arXiv:2512.08088](https://arxiv.org/abs/2512.08088) - "Distilling an LLM's Wisdom"
- **Base Model:** [llm-semantic-router/mmbert-embed-32k-2d-matryoshka](https://huggingface.co/llm-semantic-router/mmbert-embed-32k-2d-matryoshka)
- **Dataset:** [MedQuAD](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)
