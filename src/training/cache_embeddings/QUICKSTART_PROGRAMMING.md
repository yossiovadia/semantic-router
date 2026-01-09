# Programming Domain Cache Embeddings - Quick Start

Train domain-specific LoRA cache embeddings for programming queries to improve semantic caching performance.

**Results**: +5.7% margin improvement over generic embeddings

## Prerequisites

```bash
pip install datasets sentence-transformers peft transformers torch
```

## Step 1: Prepare Data

Download and prepare the CoNaLa dataset (Python programming queries):

```bash
python3 src/training/cache_embeddings/prepare_programming_data.py \
  --output data/cache_embeddings/programming/unlabeled_queries.jsonl
```

**Output**: ~71,761 unique programming queries

## Step 2: Generate Training Triplets

Use an LLM to generate paraphrases and hard negatives:

```bash
python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/programming/unlabeled_queries.jsonl \
  --output data/cache_embeddings/programming/triplets.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --domain programming \
  --paraphrases 3 \
  --negatives 2 \
  --batch-size 32 \
  --tensor-parallel 4
```

**Time**: ~4 hours (AWS g5.12xlarge with 4x A10G GPUs)
**Output**: ~194,000 training triplets

## Step 3: Train LoRA Adapter

```bash
python3 src/training/cache_embeddings/lora_trainer.py \
  --train-data data/cache_embeddings/programming/triplets.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/programming-cache-lora \
  --epochs 1 \
  --batch-size 128 \
  --lr 2e-5 \
  --temperature 0.05
```

**Time**: ~11 minutes (GPU) or ~45 minutes (CPU)
**Output**: LoRA adapter (~582 KB)

## Step 4: Test the Model

```bash
python3 src/training/cache_embeddings/test_programming_model.py
```

**Expected Results:**
- Baseline margin: ~0.41
- LoRA margin: ~0.44
- **Improvement: +5.7%**

## Use in Production

The LoRA adapter is a small file (~582 KB) that gets applied to the base model (384 MB).

**Terminology:**
- **Base model**: `all-MiniLM-L12-v2` (384 MB) - the generic embedding model
- **LoRA adapter**: `programming-cache-lora` (582 KB) - domain-specific patch with 147K trainable parameters
- **Final model**: Base model + LoRA adapter = domain-optimized embedding model

```python
from sentence_transformers import SentenceTransformer
from peft import PeftModel

# Load base model (384 MB)
base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# Apply LoRA adapter (582 KB) - this patches the base model for programming domain
base_model[0].auto_model = PeftModel.from_pretrained(
    base_model[0].auto_model,
    "models/programming-cache-lora"
)

# Now use for programming query caching
query = "How to implement a context manager in Python?"
embedding = base_model.encode(query)  # Uses base + LoRA
```

**Storage breakdown:**
- Base model: 384 MB (downloaded once, shared across all domains)
- Programming LoRA adapter: 582 KB
- Medical LoRA adapter: 582 KB
- Total for 2 domains: 384 MB + (2 × 0.6 MB) = ~385 MB

## Dataset Details

**CoNaLa (Code/Natural Language Challenge)**
- Source: Stack Overflow Python questions
- Queries: 71,761 unique programming intents
- Language: Python-focused (dataset), but negatives use multi-language for diversity
- Quality: Human-annotated code-to-natural-language mappings

## Training Configuration

**LoRA Settings:**
- Rank (r): 8
- Alpha: 16
- Target modules: `query`, `value` (attention layers)
- Dropout: 0.1
- Trainable parameters: 147,456 (0.44% of base model)

**Training Hyperparameters:**
- Optimizer: AdamW
- Learning rate: 2e-5
- Batch size: 128
- **Epochs: 1** (optimal - more epochs lead to overfitting)
- Temperature: 0.05
- Loss: Multiple Negatives Ranking (MNR)

## Why 1 Epoch?

Testing shows that 1 epoch achieves better generalization than 3 epochs:
- 1 epoch: +5.7% improvement (final loss: 0.0541)
- 3 epochs: +4.8% improvement (final loss: 0.0459)

Despite lower training loss, 3 epochs showed signs of overfitting to specific phrasings.

## Hardware Requirements

**Data Generation:**
- Recommended: AWS g5.12xlarge (4x A10G GPUs, 24GB VRAM each)
- Alternative: Any GPU with 16GB+ VRAM (slower)
- CPU-only: Possible but very slow (~40+ hours)

**LoRA Training:**
- GPU: ~11 minutes (RTX 3090 or equivalent)
- CPU: ~45 minutes (16-core processor)
- Memory: 8GB RAM minimum

## Cost Estimate (AWS)

Using g5.12xlarge ($5.67/hour):
1. Data generation: 4 hours = $22.68
2. LoRA training: 0.2 hours = $1.13

**Total**: ~$24 for complete training pipeline

## Performance Comparison

| Domain | Baseline | LoRA | Improvement | Dataset Size |
|--------|----------|------|-------------|--------------|
| Medical | 0.17 | 0.21 | +21.4% | 47K queries |
| Programming | 0.41 | 0.44 | +5.7% | 71K queries |

Programming shows smaller gains because the baseline model already understands code well. Medical domain benefits more from specialization.

## Citation

This methodology is based on:
```
@article{cache-embeddings-2025,
  title={Domain-Specific Cache Embedding Optimization},
  journal={arXiv preprint arXiv:2504.02268v1},
  year={2025}
}
```
