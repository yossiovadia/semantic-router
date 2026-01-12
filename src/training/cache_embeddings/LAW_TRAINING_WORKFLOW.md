# Law Domain Training Workflow - Step by Step

**CRITICAL**: This workflow includes small-sample validation at EVERY step to catch issues early.

**Lessons Learned from Programming Domain Failure**:
- ❌ Used hard-coded medical prompts (wasted hours)
- ❌ Didn't validate triplet quality on small sample first
- ❌ Ran full 200k triplet generation before checking quality
- ✅ **This time**: Validate with 30-50 samples FIRST, then scale up

---

## Prerequisites

**AWS VM**: g5.12xlarge (4x A10G GPUs) - Already running at `3.142.79.17`

```bash
# SSH into the VM
ssh ubuntu@3.142.79.17

# Navigate to project
cd semantic-router
git pull
```

---

## Step 1: Prepare Law Dataset (~1-2 hours)

### 1.1 Create Data Preparation Script

Create `src/training/cache_embeddings/prepare_law_data.py`:

```python
#!/usr/bin/env python3
"""
Prepare law domain dataset for cache embedding training.

Downloads and formats legal queries from:
- CaseHOLD (53k legal case holdings)
- LegalQA from StackExchange Law
"""

import json
from datasets import load_dataset

# Download CaseHOLD
print("Downloading CaseHOLD dataset...")
dataset = load_dataset("casehold/casehold", split="train")

# Extract legal holdings as queries
queries = []
for item in dataset:
    # CaseHOLD has 'holding' field with legal text
    if 'citing_prompt' in item:  # This is the legal question
        queries.append({
            "query": item['citing_prompt'],
            "source": "casehold"
        })

# Save to JSONL
output_file = "data/cache_embeddings/law/unlabeled_queries.jsonl"
with open(output_file, 'w') as f:
    for q in queries:
        f.write(json.dumps(q) + '\n')

print(f"Saved {len(queries)} legal queries to {output_file}")
```

### 1.2 Run Data Preparation

```bash
# Create output directory
mkdir -p data/cache_embeddings/law

# Run data preparation
python3 src/training/cache_embeddings/prepare_law_data.py \
  --output data/cache_embeddings/law/unlabeled_queries.jsonl

# Verify output
wc -l data/cache_embeddings/law/unlabeled_queries.jsonl
# Expected: ~60,000 lines

# Inspect first 5 queries
head -5 data/cache_embeddings/law/unlabeled_queries.jsonl | jq .
```

**✅ CHECKPOINT 1**: Verify queries look like proper legal questions, not fragments or code.

---

## Step 2: VALIDATE Triplet Generation on SMALL SAMPLE (30-50 queries)

**THIS IS CRITICAL - DO NOT SKIP!**

### 2.1 Test with 50 Queries First

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

# This should take ~5-10 minutes
```

### 2.2 MANUALLY INSPECT TRIPLET QUALITY

**Critical checks**:

```bash
# 1. Check triplet count (should be 50 queries × 3 paraphrases × 2 negatives = 300 triplets)
wc -l data/cache_embeddings/law/triplets_test_50.jsonl

# 2. Inspect first 10 triplets
head -10 data/cache_embeddings/law/triplets_test_50.jsonl | jq .

# 3. Look for quality issues
jq -r '.anchor' data/cache_embeddings/law/triplets_test_50.jsonl | head -20
jq -r '.positive' data/cache_embeddings/law/triplets_test_50.jsonl | head -20
jq -r '.negative' data/cache_embeddings/law/triplets_test_50.jsonl | head -20
```

**✅ CHECKPOINT 2 - Verify Quality Manually**:

**Good paraphrase example** (preserve legal meaning):
```json
{
  "anchor": "What constitutes a breach of contract?",
  "positive": "How is a breach of contract legally defined?",
  "negative": "What elements must be proven to establish fraudulent misrepresentation?"
}
```

**Bad examples to watch for** (❌ REJECT if you see these):

1. **Negative is just a paraphrase** (too similar):
```json
{
  "anchor": "What constitutes a breach of contract?",
  "positive": "How is breach of contract defined?",
  "negative": "What is considered a breach of contract?" ❌ TOO SIMILAR!
}
```

2. **Negative is incomplete** (fragments):
```json
{
  "anchor": "What constitutes a breach of contract?",
  "positive": "...",
  "negative": "negligence" ❌ INCOMPLETE!
}
```

3. **Negative uses wrong domain** (non-legal):
```json
{
  "anchor": "What constitutes a breach of contract?",
  "positive": "...",
  "negative": "How do I debug Python code?" ❌ WRONG DOMAIN!
}
```

4. **Placeholder text**:
```json
{
  "anchor": "What constitutes a breach of contract?",
  "positive": "...",
  "negative": "question 1?" ❌ PLACEHOLDER!
}
```

### 2.3 Fix Prompts if Needed

**If quality is bad**, edit `src/training/cache_embeddings/domains/prompts.yaml`:

- Strengthen negative_guidelines
- Add more negative_examples
- Adjust role or topic_name

Then **re-run Step 2.1** with new prompts until quality is good.

**✅ CHECKPOINT 3**: Do NOT proceed to full generation until test triplets look perfect!

---

## Step 3: Generate Full Triplet Dataset (~2-3 hours)

**ONLY run this after validating test triplets!**

```bash
# Generate full triplet dataset (~200k triplets)
python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/law/unlabeled_queries.jsonl \
  --output data/cache_embeddings/law/triplets_full.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --domain law \
  --paraphrases 3 \
  --negatives 2 \
  --batch-size 32 \
  --tensor-parallel 4

# Monitor progress (should take ~2-3 hours)
# Expected output: ~180k-200k triplets (~50MB file)
```

**✅ CHECKPOINT 4**: Verify triplet count and file size:

```bash
wc -l data/cache_embeddings/law/triplets_full.jsonl
# Expected: ~180,000-200,000 lines

ls -lh data/cache_embeddings/law/triplets_full.jsonl
# Expected: ~40-60MB
```

### 3.1 Spot Check Full Dataset

```bash
# Random sample check (look at middle, not just start)
sed -n '50000,50010p' data/cache_embeddings/law/triplets_full.jsonl | jq .
sed -n '100000,100010p' data/cache_embeddings/law/triplets_full.jsonl | jq .
```

---

## Step 4: Train LoRA Adapter (~30 mins GPU)

```bash
# Train the LoRA adapter
python3 src/training/cache_embeddings/lora_trainer.py \
  --train-data data/cache_embeddings/law/triplets_full.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/law-cache-lora \
  --epochs 1 \
  --batch-size 32 \
  --lr 2e-5 \
  --temperature 0.05

# Watch for training metrics
# Should see loss decreasing over ~30 minutes
```

**✅ CHECKPOINT 5**: Verify model files:

```bash
ls -lh models/law-cache-lora/
# Expected files:
# - adapter_model.safetensors (~582KB)
# - adapter_config.json
# - config.json
# - special_tokens_map.json
# - tokenizer_config.json
```

---

## Step 5: Create Evaluation Dataset

Create a small law test set (30-50 examples) manually or from held-out data:

```bash
# Hold out first 100 queries for testing
head -100 data/cache_embeddings/law/unlabeled_queries.jsonl > \
  data/cache_embeddings/law/test_queries.jsonl
```

---

## Step 6: Evaluate Model

Create `src/training/cache_embeddings/test_law_model.py` (copy from `test_medical_model.py` and adapt).

```bash
python3 src/training/cache_embeddings/test_law_model.py

# Expected output:
# - Baseline margin: ~0.53 (from medical baseline)
# - LoRA margin: ~0.61+ (target: > 10% improvement)
# - Precision@1: should be comparable or better
```

**✅ CHECKPOINT 6 - SUCCESS CRITERIA**:

Compare to medical results:
- Medical: 14.8% margin improvement ✅
- Programming: -3% (worse) ❌
- **Law target**: > 10% margin improvement

If margin improvement < 5%, investigate:
1. Are negatives too easy? (check samples)
2. Is domain too broad? (might need narrower focus like "contract law")
3. Did prompts work correctly? (re-check triplets)

---

## Summary Checklist

Before scaling up, validate EVERY step:

- [ ] **Step 1**: 60k legal queries prepared and inspected
- [ ] **Step 2.1**: Generate 50 test triplets
- [ ] **Step 2.2**: MANUALLY inspect test triplets (paraphrases + negatives)
- [ ] **Step 2.3**: Fix prompts if needed, re-test until perfect
- [ ] **Step 3**: Generate full 200k triplets (only after test looks good)
- [ ] **Step 3.1**: Spot check full dataset (random samples)
- [ ] **Step 4**: Train LoRA adapter
- [ ] **Step 5**: Create evaluation dataset
- [ ] **Step 6**: Evaluate and compare to baseline

**Time Estimates**:
- With validation: 4-6 hours total ✅
- Without validation (programming mistake): 8+ hours wasted ❌

---

## Common Issues and Fixes

### Issue: Negatives are too similar to anchors

**Symptom**: Hard negatives look like paraphrases
```json
{"anchor": "What is negligence?", "negative": "What does negligence mean?"}
```

**Fix**: Update `prompts.yaml` law domain:
- Add more examples showing clear topic separation
- Strengthen "STRICTLY FORBIDDEN" section
- Add penalty examples

### Issue: Negatives are incomplete fragments

**Symptom**: Short phrases or single words
```json
{"anchor": "What is negligence?", "negative": "tort law"}
```

**Fix**: Add to negative_guidelines:
```yaml
MINIMUM LENGTH: 8 words, complete sentence with subject and verb
```

### Issue: Wrong domain in prompts

**Symptom**: Script uses medical prompts for law

**Fix**: Verify script loads correct domain:
```python
# In generate_training_data.py
domain_config = load_domain_prompts(args.domain)  # Should load 'law'
print(f"Using domain: {domain_config['domain']}")  # Should print "law"
```

---

## Next Steps After Success

1. Upload to HuggingFace: `models/law-cache-lora` → `your-org/semantic-router-law-cache`
2. Update `DOMAIN_SELECTION.md` with results
3. Document in blog post (like medical domain)
4. Choose next domain (physics, economics, or biology)
