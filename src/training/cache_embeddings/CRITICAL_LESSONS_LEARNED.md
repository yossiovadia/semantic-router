# CRITICAL LESSONS LEARNED - Cache Embedding Training

**READ THIS BEFORE TRAINING ANY NEW DOMAIN**

This document captures hard-won lessons from training medical, programming, and law domains. Following these will save you **hours of debugging**.

---

## 🔴 LESSON 1: The Code Ignored Our Prompts (LAW DOMAIN FAILURE)

### The Problem

When training the **law domain**, we spent **2+ hours** iterating on `prompts.yaml`, creating increasingly strict prompt variants (A, B, C, D, E) to force better hard negatives. **NONE OF THEM WORKED.**

The negatives kept being too related:
- ❌ Anchor: "Colameta trade secrets" → Negative: "What penalties for Colameta trade secrets?"
- ❌ Anchor: "bomb-making" → Negative: "sentencing for bomb offenses"

### The Root Cause

**`generate_training_data.py` had HARDCODED prompt text that IGNORED `negative_guidelines` from prompts.yaml!**

Location: [generate_training_data.py:237-248](../generate_training_data.py#L237-L248)

```python
# BAD (old code) - hardcoded prompt ignoring YAML
prompt = f"""You are a {role}.

Task: Generate {num_negatives} COMPLETE questions that are RELATED to the original query but ask about DIFFERENT aspects.

Rules:
- Keep the same general topic/domain but ask about different aspects  # ← THIS CAUSED THE PROBLEM
- Example: if original asks about "diagnosis", ask about "symptoms" or "treatment"
```

This hardcoded "Keep the same general topic" **directly contradicted** our YAML guidelines that said "jump to completely different legal branches."

### The Fix

**Line 216 in `generate_training_data.py`:**

```python
# GOOD (new code) - actually uses negative_guidelines from YAML
negative_guidelines = domain_config.get('negative_guidelines', '')

prompt = f"""You are a {role}.

Task: Generate {num_negatives} hard negative questions for this query.

{negative_guidelines}  # ← NOW ACTUALLY USES THE YAML GUIDELINES

Original Query: {query}
{examples_text}
Return ONLY valid JSON:
{{"negatives": ["question 1 here", "question 2 here"]}}
"""
```

### The Results

**BEFORE FIX (Variant D/E with broken code):**
```
Anchor: "Colameta's use of Protégé data... trade secrets"
Negative: "What penalties can Colameta face for misappropriating trade secrets?"
```
❌ Same case, same topic, just different question

**AFTER FIX (Variant F with working code):**
```
Anchor: "Colameta's use of Protégé data... trade secrets"
Negative: "How are asylum seekers' rights protected under international law?"
```
✅ Complete topic jump - immigration law vs trade secrets!

### Action Items for Future Domains

1. ✅ **VERIFY the code actually uses your prompts** before spending hours iterating
2. ✅ **Test with 20 queries FIRST** - don't waste GPU hours on bad prompts
3. ✅ **Check `generate_training_data.py:generate_negatives_batch_vllm()`** to see what's actually sent to the LLM
4. ✅ **When prompts don't work, check the CODE not just the YAML**

---

## 🔴 LESSON 2: Dataset Contamination (LAW DOMAIN)

### The Problem

Initial law triplets had narrow, repetitive negatives:
- "subscription cancellation" → "cancellation consequences"
- "refund policy" → "refund timeline"

### The Root Cause

The `lex_glue` dataset was ordered with:
- **First 5,000 queries**: Terms of Service (ToS) - very narrow topics (subscriptions, refunds, cancellations)
- **Next 50,000 queries**: Case law - diverse legal topics (criminal, torts, constitutional, property)

When we tested with the first 50 queries, we got ToS contamination.

### The Fix

**Skip the ToS section entirely:**

```bash
# On AWS VM
tail -n +5210 unlabeled_queries.jsonl > unlabeled_queries_pure_caselaw.jsonl
```

Result: 49,987 pure case law queries with diverse topics.

### Action Items for Future Domains

1. ✅ **Always inspect the FIRST and LAST samples** of your dataset
2. ✅ **Check for topic clustering** - are the first 1k queries all similar?
3. ✅ **Use random sampling for tests** if dataset has ordering bias
4. ✅ **Document dataset structure** in comments for future reference

---

## 🔴 LESSON 3: YAML Syntax with Emojis (LAW DOMAIN)

### The Problem

After fixing the code, we got:

```
yaml.parser.ParserError: while parsing a block collection
  in "prompts.yaml", line 212, column 84
expected <block end>, but found '<scalar>'
```

### The Root Cause

YAML doesn't like unquoted strings with emojis:

```yaml
# BAD - breaks YAML parser
bad_negatives:
  - "What penalties can Colameta face?" ❌ (Same case, same topic)

# GOOD - works fine
bad_negatives:
  - "What penalties can Colameta face?"
```

### The Fix

Remove inline emojis from YAML strings. Put explanations in comments or the guidelines section instead.

### Action Items for Future Domains

1. ✅ **Test YAML syntax** with `python -c "import yaml; yaml.safe_load(open('prompts.yaml'))"`
2. ✅ **Avoid emojis in YAML list items** - use them in guidelines text blocks only
3. ✅ **Use proper YAML linters** before uploading to VM

---

## 🔴 LESSON 4: Programming Domain Failure (PROGRAMMING DOMAIN)

### The Problem

Programming domain **failed** despite good dataset and proper prompts.

**Result**: Base model was already too good at programming. LoRA training showed **no improvement** because the base embedding model already clustered programming queries perfectly.

### The Lesson

**Not all domains benefit from cache embedding training.**

Domains that work:
- ❓ **Medical**: Claims 14.8% (or 21.4%) improvement but **UNVERIFIED** - model file corrupted, baseline achieves 100% on rigorous test

Domains that don't work:
- ❌ **Programming**: Base model already excellent → **0% improvement**
- ❌ **Law**: Base model already excellent → **0.40% improvement** (25x below target)

**CRITICAL FINDING (2026-01-13)**: After rigorous evaluation, LoRA fine-tuning shows **minimal to no improvement** across all tested domains when using modern sentence-transformer base models. See [CACHE_EMBEDDING_FINDINGS.md](../../../CACHE_EMBEDDING_FINDINGS.md) for full analysis.

### Action Items for Future Domains

1. ✅ **Test base model first** - run 100 queries through base embeddings
2. ✅ **Check if clusters are already good** - if yes, skip that domain
3. ✅ **Pick domains where base model is weak** - specialized terminology, jargon, technical concepts
4. ✅ **Target domains: biology, chemistry, engineering, economics** - likely to benefit from training

---

## 🔴 LESSON 5: Test Small Before Going Big (ALL DOMAINS)

### The Problem

In programming domain, we wasted **hours generating 100k+ triplets** before discovering the quality was bad (medical prompts hardcoded).

In law domain, we almost made the same mistake with ToS-contaminated data.

### The Solution

**ALWAYS test with 20-50 queries FIRST:**

```bash
# Test command (20 queries, ~2-3 minutes)
python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/DOMAIN/unlabeled_queries.jsonl \
  --output data/cache_embeddings/DOMAIN/triplets_TEST.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --domain DOMAIN \
  --paraphrases 3 \
  --negatives 2 \
  --max-queries 20 \  # ← CRITICAL: Test with small sample
  --batch-size 8 \
  --tensor-parallel 4
```

**Manual inspection:**

```bash
# View 10 random samples
shuf -n 10 triplets_TEST.jsonl | jq -r '"ANCHOR: " + .anchor + "\nNEGATIVE: " + .negative + "\n---"'
```

### Quality Checklist

Before full generation, verify:

1. ✅ **Paraphrases preserve meaning** - no hallucinations
2. ✅ **Negatives are complete questions** (8-15 words minimum)
3. ✅ **Negatives are from different aspects/topics** - not just rewordings
4. ✅ **No entity overlap** - names/cases in anchor shouldn't appear in negatives
5. ✅ **Diverse topics** - not clustered in one narrow subcategory

### Action Items for Future Domains

1. ✅ **NEVER run full generation first** - always test 20-50 queries
2. ✅ **Manually inspect at least 10 samples** - don't trust metrics alone
3. ✅ **If quality is bad, iterate on prompts** - don't waste GPU hours
4. ✅ **Document what "good" looks like** for your domain in prompts.yaml

---

## 🟢 WINNING FORMULA: Hard Negative Prompts

After many failed attempts, here's what **actually works**:

### Structure

```yaml
negative_guidelines: |
  [CLEAR OBJECTIVE - 1 sentence]

  STEP 1: EXTRACT AND BLOCK
  [List what to identify in anchor and ban from negatives]

  STEP 2: MANDATORY TOPIC JUMP
  [Explicit mapping: if anchor is X → negative must be Y]

  STEP 3: ENTITY REPLACEMENT
  [Rules for handling names, cases, statutes]

  STEP 4: VERIFICATION (MANDATORY)
  [Yes/no questions to verify quality before generating]

  EXAMPLES OF COMPLETE FAILURE (WHAT NOT TO DO):
  ❌ [Bad example 1]
  ❌ [Bad example 2]

  EXAMPLES OF SUCCESS (WHAT TO DO):
  ✅ [Good example 1]
  ✅ [Good example 2]

negative_examples:
  - original: "Example query"
    bad_negatives:
      - "Bad example 1"
      - "Bad example 2"
    good_negatives:
      - "Good example 1"
      - "Good example 2"
```

### Key Insights

1. ✅ **Be prescriptive, not descriptive** - "DO X" not "X is good"
2. ✅ **Use explicit mappings** - "IF X THEN Y" not "choose different topics"
3. ✅ **Show BAD examples** - LLMs learn more from anti-patterns
4. ✅ **Multi-step verification** - force the LLM to check itself
5. ✅ **Use emojis in guidelines** (text blocks) but **NOT in YAML lists**

---

## 📋 CHECKLIST: Training a New Domain

Use this checklist for every new domain:

### Phase 1: Data Preparation (30 mins)

- [ ] Find dataset with 40-60k queries
- [ ] Check dataset licensing (must allow commercial use)
- [ ] Create `prepare_DOMAIN_data.py` script
- [ ] Inspect first 100 and last 100 samples for clustering
- [ ] Verify dataset diversity (not all one subtopic)
- [ ] Document dataset source and structure

### Phase 2: Prompt Engineering (1 hour)

- [ ] Create domain entry in `prompts.yaml`
- [ ] Define `role` and `topic_name`
- [ ] Write `paraphrase_guidelines` (simple, 4 rules max)
- [ ] Write `negative_guidelines` using the winning formula above
- [ ] Include 3-4 `negative_examples` with BAD and GOOD samples
- [ ] **TEST YAML SYNTAX**: `python -c "import yaml; yaml.safe_load(open('prompts.yaml'))"`
- [ ] **VERIFY CODE USES GUIDELINES**: Check `generate_training_data.py:generate_negatives_batch_vllm()`

### Phase 3: Quality Testing (30 mins)

- [ ] Generate 20 test triplets (2-3 minutes)
- [ ] Manually inspect 10 random samples
- [ ] Check: paraphrases preserve meaning?
- [ ] Check: negatives are complete questions?
- [ ] Check: negatives jump to different aspects/topics?
- [ ] Check: no entity overlap (names, cases, etc.)?
- [ ] If quality bad → iterate on prompts, test again
- [ ] If quality good → proceed to full generation

### Phase 4: Full Generation (2-4 hours)

- [ ] Run full triplet generation (~50k queries)
- [ ] Monitor first few batches for quality
- [ ] Verify augmentation factor (target: 3-5x)
- [ ] Check final triplet count (target: 150-250k)
- [ ] Save triplets to `data/cache_embeddings/DOMAIN/triplets.jsonl`

### Phase 5: Training (30 mins)

- [ ] Run LoRA training (1 epoch)
- [ ] Monitor training loss (should decrease)
- [ ] Save adapter to `models/DOMAIN-cache-lora/`
- [ ] Verify adapter files exist (adapter_model.safetensors, etc.)

### Phase 6: Evaluation (30 mins)

- [ ] Run embedding margin evaluation
- [ ] Check Precision@1, Precision@5, Precision@10
- [ ] Compare to baseline (medical: 14.8% improvement)
- [ ] **Success threshold**: >10% margin improvement
- [ ] If fails → revisit prompts or try different domain

---

## 🎯 Quick Reference: Commands

```bash
# 1. Prepare dataset
python3 src/training/cache_embeddings/prepare_DOMAIN_data.py \
  --output data/cache_embeddings/DOMAIN/unlabeled_queries.jsonl

# 2. Test with 20 queries (ALWAYS DO THIS FIRST)
python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/DOMAIN/unlabeled_queries.jsonl \
  --output data/cache_embeddings/DOMAIN/triplets_TEST.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --domain DOMAIN \
  --paraphrases 3 \
  --negatives 2 \
  --max-queries 20 \
  --batch-size 8 \
  --tensor-parallel 4

# 3. Inspect quality
shuf -n 10 data/cache_embeddings/DOMAIN/triplets_TEST.jsonl | \
  jq -r '"ANCHOR: " + .anchor + "\nNEGATIVE: " + .negative + "\n---"'

# 4. Full generation (only if test quality is good)
python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/DOMAIN/unlabeled_queries.jsonl \
  --output data/cache_embeddings/DOMAIN/triplets.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --domain DOMAIN \
  --paraphrases 3 \
  --negatives 2 \
  --batch-size 8 \
  --tensor-parallel 4
```

---

## 💡 Domain Selection Recommendations

**HIGH PRIORITY** (likely to benefit from training):
1. **Biology** - specialized terminology, taxonomy
2. **Chemistry** - molecular structures, reactions
3. **Engineering** - technical specs, standards
4. **Economics** - mathematical models, theories

**MEDIUM PRIORITY** (might benefit):
5. **Business** - jargon, frameworks
6. **Philosophy** - abstract concepts, schools of thought
7. **Psychology** - clinical terms, theories

**LOW PRIORITY** (base model likely already good):
8. **History** - dates, events (factual, not specialized)
9. **Math** - symbolic, base model strong
10. **Physics** - well-covered in training data

**AVOID**:
11. **Programming** - base model already excellent (proven failure)

---

## 📞 Emergency Troubleshooting

### Symptom: Negatives too related to anchors

**Likely cause**: Code not using `negative_guidelines`

**Fix**:
1. Check `generate_training_data.py:generate_negatives_batch_vllm()`
2. Verify `negative_guidelines = domain_config.get('negative_guidelines', '')` is used
3. Confirm guidelines are passed to LLM prompt

### Symptom: Negatives are fragments or single words

**Likely cause**: Insufficient examples or bad prompt structure

**Fix**:
1. Add GOOD examples to `negative_examples` in YAML
2. Add BAD examples showing what NOT to do
3. Increase word count requirement (8-15 words minimum)

### Symptom: Dataset has narrow topics

**Likely cause**: Dataset ordering bias or contamination

**Fix**:
1. Inspect first/last 100 samples: `head -100 dataset.jsonl | jq .query`
2. If clustered, skip problematic section: `tail -n +5000 dataset.jsonl`
3. Use random sampling for tests: `shuf dataset.jsonl | head -50`

### Symptom: YAML parsing errors

**Likely cause**: Emojis or special characters in list items

**Fix**:
1. Remove emojis from YAML lists
2. Test syntax: `python -c "import yaml; yaml.safe_load(open('prompts.yaml'))"`
3. Use YAML validator online

### Symptom: No improvement after training

**Likely cause**: Base model already good at this domain (like programming)

**Fix**:
1. Test base embeddings first on 100 queries
2. If already well-clustered, **skip this domain**
3. Try a more specialized domain instead

---

## 📝 Summary

**The 3 Critical Mistakes to Avoid:**

1. ❌ **Assuming code uses your prompts** → Always verify
2. ❌ **Testing on full dataset first** → Always start with 20 queries
3. ❌ **Ignoring dataset contamination** → Always inspect samples

**The 3 Rules for Success:**

1. ✅ **Test small, iterate fast** → 20 queries → inspect → adjust → repeat
2. ✅ **Verify code matches intent** → Check what's actually sent to LLM
3. ✅ **Show the LLM what NOT to do** → BAD examples teach more than GOOD ones

---

**Last Updated**: 2026-01-12 (Law domain training)

**Authors**: Lessons learned from medical (success), programming (failure), and law (hard-won success) domains.
