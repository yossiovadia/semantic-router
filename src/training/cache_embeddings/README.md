# Cache-Specific Embedding Training Pipeline

**Status:** 🚧 Work in Progress (Phase 1 - Coding Domain)

## Overview

This package implements domain-specific embedding model training for semantic caching, following the methodology from the research paper "Enhancing Semantic Caching with Domain-Specific Embeddings" ([arXiv:2504.02268v1](https://arxiv.org/abs/2504.02268v1)).

### Key Findings from Research

- **Small, specialized models outperform large general-purpose ones** for cache matching
- **1 epoch fine-tuning** is sufficient with proper dataset
- **Synthetic data generation** is crucial for performance
- **Precision/recall balance** is more important than raw accuracy

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Training Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Dataset Collection                                        │
│     ├─ Real Data (Stack Overflow, GitHub, Docs)              │
│     └─ Synthetic Data Generation                             │
│         ├─ Paraphrasing (T5/BART)                            │
│         ├─ Hard Negative Mining                              │
│         └─ Query Reformulation                               │
│                                                               │
│  2. Contrastive Learning                                      │
│     ├─ Triplet Loss                                          │
│     ├─ InfoNCE Loss                                          │
│     └─ Multiple Negatives Ranking (MNR) ✅ Recommended       │
│                                                               │
│  3. LoRA Fine-Tuning                                          │
│     ├─ Base: bert-base-uncased (110M params)                 │
│     ├─ Trainable: ~0.02% (LoRA adapters)                     │
│     └─ Output: 384-dim embeddings                            │
│                                                               │
│  4. Evaluation                                                │
│     ├─ Precision@K (K=0.80, 0.85, 0.90, 0.95)                │
│     ├─ Recall@K                                              │
│     ├─ MRR (Mean Reciprocal Rank)                            │
│     └─ False Positive/Negative Rates                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
src/training/cache_embeddings/
├── __init__.py                     # Package initialization
├── common_utils.py                 # ✅ Shared utilities (logging, metrics, I/O)
├── losses.py                       # ✅ Contrastive loss functions
├── dataset_builder.py              # 🚧 Real & synthetic data collection
├── synthetic_data_generator.py     # 🔜 Paraphrasing & hard negative mining
├── train_cache_embedding_lora.py   # 🔜 Main training script
├── evaluate_cache_model.py         # 🔜 Evaluation framework
└── README.md                       # 📄 This file

datasets/cache_training/
└── coding/
    ├── real_data.jsonl             # Stack Overflow, GitHub issues
    ├── synthetic_data.jsonl        # Generated training pairs
    └── test_set.jsonl              # Held-out evaluation set
```

## Components

### ✅ Completed

#### 1. Common Utilities (`common_utils.py`)
- Logging setup
- Random seed setting
- Device selection (CPU/GPU)
- Similarity metrics (cosine, euclidean)
- Cache-specific metrics calculation
- Model artifact saving/loading
- JSONL I/O utilities

#### 2. Contrastive Losses (`losses.py`)

**TripletLoss**
```python
loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
```
- Margin: 1.0 (default)
- Distance: Cosine or Euclidean

**InfoNCELoss**
```python
# Normalized temperature-scaled cross-entropy
# Used in SimCLR
```
- Temperature: 0.07 (default)
- Encourages clustering of similar pairs

**MultipleNegativesRankingLoss (MNR)** ⭐ **Recommended**
```python
# Uses in-batch negatives efficiently
# All other positives serve as negatives
```
- Temperature: 0.05 (default)
- Most efficient for cache learning
- Recommended by research paper

**CosineEmbeddingLoss**
```python
# Simple pairwise cosine loss
```
- Margin: 0.5 (default)

**HardTripletLoss**
```python
# Mines hardest examples in batch
# More efficient than random triplets
```
- Strategies: hardest, semi-hard, all

### 🚧 In Progress

#### 3. Dataset Builder (`dataset_builder.py`)
- [ ] Stack Overflow Q&A loader
- [ ] GitHub issues/discussions loader
- [ ] Coding documentation scraper
- [ ] Data preprocessing & filtering
- [ ] Train/val/test splitting
- [ ] JSONL export

### 🔜 Planned

#### 4. Synthetic Data Generator (`synthetic_data_generator.py`)
- [ ] Paraphrase generation (T5-small/BART)
- [ ] Hard negative mining (clustering-based)
- [ ] Query reformulation
- [ ] Back-translation augmentation

#### 5. Training Script (`train_cache_embedding_lora.py`)
- [ ] LoRA configuration for embeddings
- [ ] Training loop with contrastive loss
- [ ] Checkpoint management
- [ ] TensorBoard logging
- [ ] Distributed training support

#### 6. Evaluation Framework (`evaluate_cache_model.py`)
- [ ] Precision@K calculation
- [ ] Recall@K calculation
- [ ] MRR (Mean Reciprocal Rank)
- [ ] False positive/negative rates
- [ ] Latency benchmarking
- [ ] A/B comparison with baseline

## Training Data Format

### Triplet Format (for Triplet/InfoNCE/MNR losses)

```jsonl
{
  "anchor": "How do I reverse a string in Python?",
  "positive": "What's the method to reverse a Python string?",
  "hard_negative": "How do I reverse a list in Python?",
  "domain": "coding",
  "similarity_score": 0.95,
  "source": "stackoverflow"
}
```

### Pair Format (for Cosine loss)

```jsonl
{
  "query1": "How to sort a Python list?",
  "query2": "Python list sorting methods",
  "label": 1,
  "domain": "coding",
  "similarity_score": 0.92,
  "source": "synthetic"
}
```

## Usage (Coming Soon)

### Training

```bash
# Train coding domain model with MNR loss
python train_cache_embedding_lora.py \
  --domain coding \
  --base-model bert-base-uncased \
  --loss-type mnr \
  --epochs 1 \
  --batch-size 32 \
  --lora-rank 16 \
  --output-dir models/cache_coding_bert-base_lora

# Train with triplet loss
python train_cache_embedding_lora.py \
  --domain coding \
  --loss-type triplet \
  --margin 1.0 \
  --epochs 3
```

### Evaluation

```bash
# Evaluate against baseline
python evaluate_cache_model.py \
  --model models/cache_coding_bert-base_lora \
  --baseline models/all-MiniLM-L12-v2 \
  --test-data datasets/cache_training/coding/test_set.jsonl \
  --thresholds 0.80 0.85 0.90 0.95
```

### Synthetic Data Generation

```bash
# Generate synthetic training data
python synthetic_data_generator.py \
  --input datasets/cache_training/coding/real_data.jsonl \
  --output datasets/cache_training/coding/synthetic_data.jsonl \
  --num-paraphrases 5 \
  --num-hard-negatives 3
```

## Success Criteria (Phase 1 - Coding Domain)

| Metric | Target | Baseline (all-MiniLM-L12-v2) |
|--------|--------|------------------------------|
| Precision@0.90 | +10% improvement | TBD |
| Recall@0.90 | +5% improvement | TBD |
| False Positive Rate | < 2% | TBD |
| Embedding Latency | < 10ms | ~5ms |
| Model Size | < 150MB | 127MB |

## Integration with Multi-Domain Cache

Once trained, the model integrates with the `semantic_cache` branch's domain-specific caching:

```yaml
# config/config.yaml
semantic_cache:
  enabled: true
  backend_type: "redis"

  domains:
    coding:
      namespace: "coding"
      embedding_model_path: "models/cache_coding_bert-base_lora"  # ← New!
      similarity_threshold: 0.92
      ttl_seconds: 7200
```

## References

1. **Research Paper:** "Enhancing Semantic Caching with Domain-Specific Embeddings" ([arXiv:2504.02268v1](https://arxiv.org/abs/2504.02268v1))
2. **Semantic Cache Overview:** [website/docs/tutorials/semantic-cache/overview.md](../../../website/docs/tutorials/semantic-cache/overview.md)
3. **Redis Cache Tutorial:** [website/docs/tutorials/semantic-cache/redis-cache.md](../../../website/docs/tutorials/semantic-cache/redis-cache.md)
4. **LoRA Training Reference:** [training_lora/classifier_model_fine_tuning_lora/ft_linear_lora.py](../training_lora/classifier_model_fine_tuning_lora/ft_linear_lora.py)

## Contributing

This is Phase 1 (Coding Domain). Future phases:
- Phase 2: Medical domain
- Phase 3: Math domain
- Phase 4: Business domain
- Phase 5: Multi-task model (domain-conditioned)

## License

Apache 2.0 (same as parent project)

---

**Last Updated:** 2024-12-17
**Status:** Phase 1 Implementation in Progress
