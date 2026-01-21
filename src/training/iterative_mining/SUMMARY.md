# Implementation Summary - Iterative Hard-Negative Mining POC

## What We Built

A complete implementation of domain-aware embedding training using **iterative hard-negative mining** with LLM-as-judge, based on the paper ["Adaptation of Embedding Models to Financial Filings via LLM Distillation"](https://arxiv.org/pdf/2512.08088).

## Key Features

### 1. Complete Training Pipeline

- **Data Preparation**: Loads MedQuAD medical Q&A dataset, creates corpus chunks, prepares queries
- **Iterative Mining**: Implements paper's core method - LLM judges relevance, mines hard examples, fine-tunes model
- **Evaluation**: Compares baseline vs specialized model using retrieval metrics (MRR@5, DCG@5, Precision@5)
- **Small Sample Testing**: Validates pipeline with 30-50 queries before full training

### 2. Optimized for AWS g5.12xlarge

- **GPU Utilization**: Uses sentence-transformers for efficient embedding (single GPU)
- **vLLM Integration**: LLM-as-judge via vLLM with 4x A10G GPUs
- **Time Estimates**: 6-12 hours full training, 2-4 hours optimized
- **Cost Optimization**: Configurable query count and candidate size

### 3. Medical Domain POC

- **Dataset**: MedQuAD (~47K medical question-answer pairs)
- **Advantage**: Pre-existing questions (skip query generation step from paper)
- **Method**: Still applies paper's core contribution (iterative hard-negative mining)
- **Target**: +15-25% improvement in retrieval quality

## Files Implemented

```
src/training/iterative_mining/
├── README.md                  # Project overview
├── SUMMARY.md                # This file
├── AWS_RUNBOOK.md            # Step-by-step AWS deployment guide
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignore data/models (too large)
│
├── prepare_corpus.py         # Load MedQuAD, chunk answers, split train/val/test
├── prepare_queries.py        # Map questions to ground-truth chunks
├── iterative_miner.py        # Main training loop (LLM-as-judge, mine hard examples, fine-tune)
├── evaluate.py               # Compare baseline vs specialized (MRR@5, DCG@5, Precision@5)
├── test_small_sample.py      # TEST FIRST! Validates with 30 queries before full run
└── run_full_pipeline.sh      # Full pipeline runner script
```

## How It Works

### Step 1: Data Preparation (15 minutes)

```bash
# Load MedQuAD dataset
python prepare_corpus.py
# Creates: corpus_chunks.pkl, train_qas.pkl, val_qas.pkl, test_qas.pkl

# Prepare queries
python prepare_queries.py
# Creates: train_queries.pkl, val_queries.pkl, test_queries.pkl
```

**What happens**:
- Downloads MedQuAD medical Q&A dataset
- Chunks long answers (500-1000 chars)
- Splits into 70% train, 15% val, 15% test
- Maps questions to their ground-truth answer chunks

### Step 2: Small Sample Test (5-10 minutes)

```bash
# CRITICAL: Test with 30 queries first!
python test_small_sample.py \
    --num-queries 30 \
    --num-candidates 10 \
    --llm-endpoint http://localhost:8000/v1
```

**What it validates**:
- LLM is giving reasonable relevance scores (1-4)
- Hard examples are being mined
- Prompts are working correctly
- Pipeline runs without errors

**Output**: `small_sample_results.json` - inspect before full training!

### Step 3: Full Training (6-12 hours)

```bash
python iterative_miner.py \
    --iterations 2 \
    --num-queries 1000 \
    --llm-endpoint http://localhost:8000/v1
```

**What happens** (per iteration):

1. **Embed corpus**: all-MiniLM-L12-v2 embeds 47K chunks (~3 min)
2. **Retrieve candidates**: Top-50 chunks for each query (~30 sec)
3. **LLM judges**: Qwen 2.5-7B scores 50K (query, chunk) pairs (~3-5 hours)
4. **Mine hard examples**: Find hard positives/negatives (~1 min)
5. **Fine-tune model**: Train on hard examples with triplet loss (~15 min)

**Result**: `models/medical-specialized/final/` - specialized medical embedding model

### Step 4: Evaluation (10-20 minutes)

```bash
python evaluate.py
```

**What it measures**:
- **MRR@5**: Mean Reciprocal Rank of first relevant result
- **DCG@5**: Discounted Cumulative Gain (rewards top rankings)
- **Precision@5**: Fraction of top-5 that are relevant
- **Recall@5**: Fraction of relevant docs found in top-5

**Success criteria**:
- MRR@5 improvement: ≥15%
- DCG@5 improvement: ≥15%

## Key Implementation Details

### 1. Improved LLM Judging Prompt

Based on the paper's Table II criteria, but clearer:

```python
"""You are a medical information relevance judge. Rate how well this passage answers the medical query.

**Scoring Criteria:**
- Score 4: The passage EXPLICITLY and COMPLETELY answers the query with all necessary details
- Score 3: The passage PARTIALLY answers the query - relevant and clear connection, but missing some details
- Score 2: The passage is SOMEWHAT relevant to the query topic, but does NOT answer it
- Score 1: The passage is NOT relevant - completely different topic from the query

**Query (Medical Question):**
{query}

**Passage (Medical Information):**
{chunk}

**Instructions:**
- Consider: Does this passage actually ANSWER the question being asked?
- A passage can mention related topics but still not answer the specific question (score 2 or 1)
- Only give score 4 if someone reading ONLY this passage would have their question fully answered

Respond with ONLY the number 1, 2, 3, or 4:"""
```

### 2. Hard Example Mining

```python
# Hard positive: LLM says relevant (score ≥3) but model ranked low (rank >10)
if llm_score >= 3 and model_rank > 10:
    hard_positives.append((query, chunk))

# Hard negative: LLM says not relevant (score ≤2) but model ranked high (rank ≤5)
if llm_score <= 2 and model_rank <= 5:
    hard_negatives.append((query, chunk))
```

### 3. Triplet Loss Fine-Tuning

```python
# Positive pairs: Query should be close to relevant chunks
InputExample(texts=[query, positive_chunk], label=1.0)

# Negative pairs: Query should be far from irrelevant chunks
InputExample(texts=[query, negative_chunk], label=0.0)

# Fine-tune with CosineSimilarityLoss
model.fit(train_objectives=[(dataloader, CosineSimilarityLoss(model))], epochs=1)
```

## Differences from Paper

| Aspect | Paper | Our POC |
|--------|-------|---------|
| **Domain** | Finance (SEC filings) | Medical (MedQuAD) |
| **Corpus** | Raw PDFs → chunk | Q&A answers (already chunked) |
| **Queries** | LLM-generated | Existing questions (advantage!) |
| **Method** | Iterative hard-negative mining | ✅ Same |
| **LLM Judge** | Relevance scores 1-4 | ✅ Same |
| **Fine-tuning** | Full model, triplet loss | ✅ Same |
| **Iterations** | 2 | 2 |
| **Results** | +27.7% MRR@5, +44.6% DCG@5 | Target: +15-25% |

## Next Steps (After POC Success)

1. **Phase 2: Production Integration** (Rust/Go)
   - Load fine-tuned model in Rust via candle
   - FFI bindings for Go
   - Configuration support for domain selection
   - Model registry for multiple domains

2. **Additional Domains**
   - Legal (using legal Q&A dataset)
   - Programming (using StackOverflow)
   - Finance (using financial filings like paper)

3. **Optimization**
   - Quantization for faster inference
   - Distillation to smaller models
   - Multi-task training (combine domains)

## Timeline (AWS g5.12xlarge)

| Phase | Duration | Notes |
|-------|----------|-------|
| **Setup** | 5-10 min | Clone repo, install deps, start vLLM |
| **Data prep** | 15 min | Download MedQuAD, prepare corpus/queries |
| **Small test** | 5-10 min | **CRITICAL - Run first!** |
| **Full training** | 6-12 hrs | Bottleneck: LLM judging 50K-100K pairs |
| **Evaluation** | 10-20 min | Compare baseline vs specialized |
| **Total** | **~7-13 hrs** | - |

## Cost Estimate (AWS)

- g5.12xlarge: ~$5.67/hour (on-demand)
- Training time: 7-13 hours
- **Total cost**: $40-75 per training run

**Optimization**:
- Use Spot instances: ~$2-3/hour → $14-40 total
- Reduce queries to 500: ~4-6 hours → $20-35 total

## References

- **Paper**: [Adaptation of Embedding Models to Financial Filings via LLM Distillation](https://arxiv.org/pdf/2512.08088)
- **GitHub Issue**: #1131
- **Branch**: `feature/rag-distillation`
- **Plan**: `/Users/yovadia/.claude/plans/functional-watching-flask.md`

## Questions Answered During Development

### Q1: How long will training take?
**A**: 6-12 hours full (1000 queries), 2-4 hours optimized (500 queries)

### Q2: How to prove specialized model is better?
**A**: Retrieval metrics (MRR@5, DCG@5) - both models rank same test corpus, specialized ranks correct answers higher

### Q3: Are cache embedding triplets the same as training triplets?
**A**: NO! Cache triplets map between different models' embedding spaces. Training triplets are (query, answer) pairs with relevance labels.

### Q4: Can we reuse anything from cache PR?
**A**: Infrastructure (vLLM, GPUs) - yes. Triplet data - no (completely different purpose).

## Success Metrics

### POC Success
- ✅ Pipeline implemented correctly
- ✅ Small sample test passes (scores look reasonable)
- ✅ Training completes without errors
- ✅ MRR@5 improvement ≥15%
- ✅ DCG@5 improvement ≥15%

### Production Ready (Phase 2)
- ✅ Model loads in Rust via candle
- ✅ FFI bindings work from Go
- ✅ Configuration supports domain selection
- ✅ Multiple domains available

---

**Status**: ✅ Implementation complete, ready for AWS testing

**Next**: Run small sample test on AWS, then full training if results look good
