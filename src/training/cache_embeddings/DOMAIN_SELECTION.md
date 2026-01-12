# Domain Selection for Cache Embedding LoRA Training

**Date**: 2025-01-12
**Current Status**: Selecting next domain for training

---

## Available Domains in Semantic Router

From `models/lora_intent_classifier_bert-base-uncased_model/category_mapping.json`:

1. **biology** (idx: 0)
2. **business** (idx: 1)
3. **chemistry** (idx: 2)
4. **computer science** (idx: 3) ✅ **DONE** (failed - no improvement)
5. **economics** (idx: 4)
6. **engineering** (idx: 5)
7. **health** (idx: 6) ✅ **DONE** (medical - showed 14.8% margin improvement)
8. **history** (idx: 7)
9. **law** (idx: 8)
10. **math** (idx: 9)
11. **other** (idx: 10)
12. **philosophy** (idx: 11)
13. **physics** (idx: 12)
14. **psychology** (idx: 13)

---

## Training Results Summary

### ✅ Medical/Health Domain (SUCCESS)
- **Baseline precision@1**: 92.5%
- **Fine-tuned**: 89.5% (slightly worse on precision, but...)
- **Embedding quality margin improvement**: +14.8% ✅
- **Conclusion**: LoRA specialization worked well

### ❌ Programming/Computer Science (FAILED)
- **Easy test**: Both base and LoRA achieved 100% accuracy
- **Hard test**: Base = 91.67%, LoRA = 88.89% (worse!)
- **Conclusion**: Domain too easy for base model, or training data not distinctive enough

---

## Domain Selection Criteria

For the next domain, we need:

1. **Semantic Complexity**: Not too easy (avoid programming mistake)
2. **Vocabulary Distinctiveness**: Domain-specific terminology that benefits from specialization
3. **Large Dataset Availability**: 50k+ examples with permissive license
4. **Natural Semantic Variations**: Good for contrastive learning (paraphrases, similar concepts)
5. **Cache Use Case**: Domain where caching is valuable (repeated similar queries)

---

## Domain Analysis & Recommendations

### 🥇 TOP RECOMMENDATION: Law

**Why Law is Ideal**:
- ✅ **High semantic complexity**: Legal language is nuanced (contracts, statutes, case law)
- ✅ **Distinctive vocabulary**: Legal jargon, Latin terms, specific phrasing
- ✅ **Strong cache use case**: Legal queries are often repeated (precedents, common legal questions)
- ✅ **Not too easy**: Base models struggle with legal nuances
- ✅ **Large datasets available**

**Available Datasets** (targeting ~45-70k queries like medical):

| Dataset | Size (Queries) | License | Quality | Link |
|---------|---------------|---------|---------|------|
| **LegalAdvice (Reddit r/legaladvice)** | ~190k questions | Public/Reddit | High | Kaggle/Reddit API |
| **CaseHOLD** | 53k legal holdings | CC BY 4.0 | High | HuggingFace: `casehold` |
| **LegalQA (StackExchange Law)** | ~30k Q&A pairs | CC BY-SA 4.0 | High | Archive.org |
| **CUAD (Contract clauses)** | 13k+ clauses | CC BY 4.0 | High | HuggingFace: `cuad` |
| **LeXFiles (Legal Questions)** | ~40k legal questions | Check license | Medium | GitHub research repos |

**BEST FIT**: Combine **CaseHOLD (53k)** + samples from other sources = ~60-70k legal queries

**Training Data Generation Strategy**:
```python
# Positive pairs (similar legal concepts):
- "breach of contract" ↔ "violation of contractual agreement"
- "force majeure clause" ↔ "act of God provision"
- "joint and several liability" ↔ "collective responsibility"

# Hard negatives (legal but different):
- "breach of contract" ❌ "breach of warranty" (related but distinct)
- "negligence" ❌ "strict liability" (both torts, different standards)
```

**Expected Outcome**:
- Legal terminology benefits from specialized embeddings
- Base models often conflate similar legal terms
- ✅ **High likelihood of improvement**

---

### 🥈 SECOND CHOICE: Physics

**Why Physics Could Work**:
- ✅ **Technical vocabulary**: Quantum mechanics, thermodynamics, relativity
- ✅ **Mathematical notation**: Equations, formulas (distinct from pure math)
- ✅ **Conceptual complexity**: Abstract concepts with precise meanings
- ⚠️ **Risk**: Might overlap too much with general STEM knowledge

**Available Datasets**:

| Dataset | Size | License | Quality | Link |
|---------|------|---------|---------|------|
| **arXiv Physics Papers** | 2M+ papers | CC BY 4.0 | High | Kaggle: `arXiv Dataset` |
| **PhysicsQA** | 10K+ Q&A pairs | MIT | Medium | GitHub: various repos |
| **SciQ** | 13K science questions (physics subset) | CC BY-SA 4.0 | Medium | HuggingFace: `sciq` |

**Concern**:
- Physics overlap with general scientific knowledge might limit improvement
- Less distinctive than legal domain

---

### 🥉 THIRD CHOICE: Economics

**Why Economics is Interesting**:
- ✅ **Specialized terminology**: Macroeconomics, econometrics, market theory
- ✅ **Policy/theory distinction**: Keynesian vs Austrian vs Chicago schools
- ✅ **Numerical + conceptual**: GDP, inflation, trade (contextual numbers)
- ✅ **Good cache use case**: Financial analysis, market queries

**Available Datasets**:

| Dataset | Size | License | Quality | Link |
|---------|------|---------|---------|------|
| **EconBiz** | Large economics database | Check license | High | Web scraping needed |
| **Federal Reserve Papers** | Thousands of papers | Public domain (US Gov) | High | FRED, Fed websites |
| **NBER Working Papers** | 30K+ papers | Open access | High | NBER website |

**Concern**:
- Harder to get clean triplet data
- May require more manual curation

---

### ⚠️ NOT RECOMMENDED

**Math** (idx: 9):
- ❌ Too similar to programming - base models handle well
- ❌ Risk of same failure as programming domain

**Philosophy** (idx: 11):
- ❌ Too abstract - hard to create good hard negatives
- ❌ Less distinctive vocabulary than law

**Chemistry** (idx: 2):
- ⚠️ Highly specialized but dataset quality concerns
- ⚠️ Overlap with general chemistry knowledge

**History** (idx: 7):
- ⚠️ Very broad domain - hard to define scope
- ⚠️ Base models already good at historical facts

---

## Recommended Next Steps

### Option 1: Law Domain (RECOMMENDED) 🏆

**Justification**:
1. Highest likelihood of success (complex, distinctive, not too easy)
2. Excellent dataset availability (MultiLegalPile, CUAD, LegalBench)
3. Strong real-world cache use case
4. Clear semantic boundaries (unlike philosophy)

**Implementation Plan** (following medical domain workflow):

**Step 1: Prepare Law Dataset (~1-2 hours)**
```bash
python3 src/training/cache_embeddings/prepare_law_data.py \
  --output data/cache_embeddings/law/unlabeled_queries.jsonl
```
- Download CaseHOLD (53k holdings) from HuggingFace
- Download LegalQA from StackExchange (~15k)
- Clean and format → Target: ~60k legal queries
- Output: `unlabeled_queries.jsonl` (~6MB, similar to medical's 2.8MB)

**Step 2: Generate Training Triplets (~2-3 hours on AWS g5.12xlarge)**
```bash
python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/law/unlabeled_queries.jsonl \
  --output data/cache_embeddings/law/triplets.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --domain law \
  --paraphrases 3 \
  --negatives 2 \
  --batch-size 32 \
  --tensor-parallel 4
```
- Uses vLLM to generate paraphrases and hard negatives
- Output: ~180-200k training triplets (~50MB file)

**Step 3: Train LoRA Adapter (~30 mins GPU)**
```bash
python3 src/training/cache_embeddings/lora_trainer.py \
  --train-data data/cache_embeddings/law/triplets.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/law-cache-lora \
  --epochs 1 \
  --batch-size 32
```
- LoRA adapter: ~582KB (same as medical)

**Step 4: Evaluate**
```bash
python3 src/training/cache_embeddings/test_law_model.py
```

**Total Estimated Timeline**: 4-6 hours (not days!)

---

### Option 2: Physics Domain (BACKUP)

If law proves too difficult to curate, fall back to physics:
1. Use arXiv physics papers (filter by subject: hep-th, cond-mat, etc.)
2. Generate triplets from paper abstracts
3. Focus on specialized subfields (quantum, relativity, particle physics)

---

## Dataset Preparation Checklist

For whichever domain we choose:

- [ ] Download raw dataset (aim for 100k+ documents)
- [ ] License verification (must be Apache/MIT/CC-BY compatible)
- [ ] Filter for quality (remove short/low-quality texts)
- [ ] Generate anchor texts (queries/prompts)
- [ ] Create positive pairs (paraphrases, similar concepts)
- [ ] Create hard negatives (same domain, different concept)
- [ ] Split into train (50k), val (5k), test (5k)
- [ ] Verify triplet quality (manual sampling)

---

## Expected Performance Targets

Based on medical domain results:

| Metric | Target |
|--------|--------|
| **Training time** | 1-3 epochs (12-36 hours on g5.12xlarge) |
| **Embedding margin improvement** | > 10% (like medical: 14.8%) |
| **Precision@1** | > baseline or within 5% |
| **Precision@5** | > baseline |

---

## Decision

**Recommended Domain**: **Law** (idx: 8)

**Datasets to Use**:
1. Primary: MultiLegalPile (legal contracts, case law subset)
2. Secondary: CUAD (contracts)
3. Validation: LegalBench tasks

**Next Action**:
- [ ] SSH into AWS VM (already started: i-0f8b657990eb4413e)
- [ ] Download MultiLegalPile sample (10GB)
- [ ] Set up data generation pipeline
- [ ] Generate 50k law triplets
- [ ] Train law LoRA adapter

---

## Backup Domains (if Law doesn't work)

1. **Physics** - Good datasets, technical vocabulary
2. **Economics** - Specialized but requires more curation
3. **Biology** - Similar to medical, might work well
