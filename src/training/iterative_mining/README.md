# Iterative Hard-Negative Mining for Domain-Aware Embeddings

Implementation of domain-aware embedding training using iterative hard-negative mining with LLM-as-judge, based on the paper ["Adaptation of Embedding Models to Financial Filings via LLM Distillation"](https://arxiv.org/pdf/2512.08088).

## Overview

This POC trains a medical domain-specialized embedding model using the MedQuAD dataset and iterative hard-negative mining.

## Methodology

1. **Start** with generic `all-MiniLM-L12-v2` model
2. **Iterate** (2 rounds):
   - Retrieve candidate chunks for medical queries
   - LLM judges relevance (1-4 scores)
   - Mine hard positives/negatives
   - Fine-tune model with triplet loss
3. **Evaluate** with retrieval metrics (MRR@5, DCG@5, Precision@5)

## Files

- `prepare_corpus.py` - Load MedQuAD and prepare corpus
- `prepare_queries.py` - Prepare queries from MedQuAD questions
- `iterative_miner.py` - Main training pipeline
- `evaluate.py` - Evaluation script
- `requirements.txt` - Python dependencies

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data
python prepare_corpus.py
python prepare_queries.py

# Train specialized model
python iterative_miner.py

# Evaluate
python evaluate.py
```

## Expected Results

- Baseline (all-MiniLM-L12-v2): MRR@5 ~0.15, DCG@5 ~0.20
- Specialized (after 2 iterations): MRR@5 ~0.20-0.25, DCG@5 ~0.25-0.30
- Target improvement: +15-25% in retrieval metrics
