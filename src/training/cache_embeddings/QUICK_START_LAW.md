# Quick Start: Law Domain (with Validation!)

**Critical**: ALWAYS test with 50 samples before full generation!

## 1. SSH to AWS VM

```bash
ssh ubuntu@3.142.79.17
cd semantic-router
```

## 2. Prepare Data (1-2 hours)

```bash
# Create law data directory
mkdir -p data/cache_embeddings/law

# Download and prepare ~60k legal queries
python3 src/training/cache_embeddings/prepare_law_data.py \
  --output data/cache_embeddings/law/unlabeled_queries.jsonl

# ✅ CHECKPOINT: Verify ~60k queries
wc -l data/cache_embeddings/law/unlabeled_queries.jsonl
head -5 data/cache_embeddings/law/unlabeled_queries.jsonl | jq .
```

## 3. TEST with 50 Queries (5-10 mins) ⚠️ CRITICAL!

```bash
# Generate ONLY 50 triplets for validation
python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/law/unlabeled_queries.jsonl \
  --output data/cache_embeddings/law/triplets_test_50.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --domain law \
  --paraphrases 3 \
  --negatives 2 \
  --max-queries 50 \
  --batch-size 8 \
  --tensor-parallel 4
```

### 3.1 INSPECT QUALITY MANUALLY! ⚠️

```bash
# Check triplet count (should be ~300: 50 queries × 3 paraphrases × 2 negatives)
wc -l data/cache_embeddings/law/triplets_test_50.jsonl

# Look at samples
head -10 data/cache_embeddings/law/triplets_test_50.jsonl | jq .

# Check negatives specifically
jq -r '.negative' data/cache_embeddings/law/triplets_test_50.jsonl | head -20
```

### 3.2 Quality Checks ✅ vs ❌

**✅ GOOD negative** (different legal topic):
```
Anchor: "What constitutes a breach of contract?"
Negative: "What elements must be proven to establish fraudulent misrepresentation?"
```

**❌ BAD negative** (too similar - reject!):
```
Anchor: "What constitutes a breach of contract?"
Negative: "What is considered a breach of contract?"
```

**❌ BAD negative** (incomplete - reject!):
```
Anchor: "What constitutes a breach of contract?"
Negative: "negligence"
```

**IF BAD**: Fix `src/training/cache_embeddings/domains/prompts.yaml` → Re-run Step 3

**IF GOOD**: Proceed to Step 4 ✅

## 4. Generate Full Dataset (2-3 hours)

**ONLY run this if test triplets look perfect!**

```bash
python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/law/unlabeled_queries.jsonl \
  --output data/cache_embeddings/law/triplets_full.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --domain law \
  --paraphrases 3 \
  --negatives 2 \
  --batch-size 32 \
  --tensor-parallel 4

# ✅ CHECKPOINT: ~200k triplets, ~50MB file
wc -l data/cache_embeddings/law/triplets_full.jsonl
ls -lh data/cache_embeddings/law/triplets_full.jsonl

# Spot check middle of file
sed -n '100000,100010p' data/cache_embeddings/law/triplets_full.jsonl | jq .
```

## 5. Train LoRA (30 mins)

```bash
python3 src/training/cache_embeddings/lora_trainer.py \
  --train-data data/cache_embeddings/law/triplets_full.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/law-cache-lora \
  --epochs 1 \
  --batch-size 32 \
  --lr 2e-5

# ✅ CHECKPOINT: Verify adapter created (~582KB)
ls -lh models/law-cache-lora/adapter_model.safetensors
```

## 6. Evaluate

```bash
python3 src/training/cache_embeddings/test_law_model.py

# Target: > 10% margin improvement (medical achieved 14.8%)
```

---

## Files Updated

- ✅ `src/training/cache_embeddings/domains/prompts.yaml` - Added law prompts
- ✅ `src/training/cache_embeddings/domains/law.yaml` - Domain config
- ✅ `LAW_TRAINING_WORKFLOW.md` - Detailed workflow with checkpoints

## Key Lessons from Programming Failure

1. **Test with 50 samples first** ⚠️
2. **Manually inspect negatives** ⚠️
3. **Verify prompts.yaml has law domain** ⚠️
4. **Don't skip checkpoints!** ⚠️
