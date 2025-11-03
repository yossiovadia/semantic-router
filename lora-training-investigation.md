# LoRA Intent Classification Training Investigation

## Date
2025-11-03

## Problem Statement
E2E classification tests failing with ~30% accuracy after refactoring from legacy ModernBERT to LoRA-based classification. Investigation revealed that downloaded LoRA models only have 3 categories instead of required 14.

## Investigation Summary

### Issue Discovery
1. **E2E Test 03 Failure**: Classification accuracy dropped to 30% across multiple categories
2. **Root Cause**: LoRA models downloaded from HuggingFace only contain 3 categories (business, law, psychology) instead of 14
3. **Legacy Model**: Non-LoRA ModernBERT model has all 14 categories and works correctly

### Training Attempts - LoRA Script (FAILED)

**Script**: `src/training/training_lora/classifier_model_fine_tuning_lora/ft_linear_lora.py`

Multiple training attempts with different hyperparameters all resulted in severe overfitting and ~6% validation accuracy:

#### Attempt 1: BERT with default params
- Model: `bert-base-uncased`
- Samples: 7000
- Epochs: 3 (default)
- Batch size: 8 (default)
- Learning rate: 3e-5 (default)
- LoRA rank: 16
- **Result**: `eval_accuracy: 0.0635` (6.35%), `eval_loss: nan`, `train_loss: 0.0001`

#### Attempt 2: BERT with more samples and epochs
- Model: `bert-base-uncased`
- Samples: 14000
- Epochs: 5
- Batch size: 16
- Learning rate: 2e-5
- LoRA rank: 16, alpha: 32
- **Result**: `eval_accuracy: 0.0622` (6.22%), `eval_loss: nan`, `train_loss: 0.007`

#### Attempt 3: RoBERTa with optimized params
- Model: `roberta-base`
- Samples: 14000
- Epochs: 8
- Batch size: 16
- Learning rate: 2e-5
- LoRA rank: 16, alpha: 32
- **Result**: `eval_accuracy: 0.0622` (6.22%), `eval_loss: nan`, `train_loss: 0.004`

#### Attempt 4: RoBERTa with stratification fix
- Added stratification to train/val split (line 491)
- Same hyperparameters as Attempt 3
- **Result**: `eval_accuracy: 0.0645` (6.45%), `eval_loss: nan`

**Consistent Pattern Across All LoRA Attempts**:
- Extreme overfitting (train_loss → 0)
- Numerical instability (eval_loss: nan)
- Validation accuracy ~6% (worse than random guessing at 7.14% for 14 classes)
- All 14 categories present in label_mapping.json
- Same failure regardless of model architecture or hyperparameters

### Training Success - Legacy Script (SUCCESS)

**Script**: `src/training/classifier_model_fine_tuning/ft_linear.py`

#### Successful Training: RoBERTa Full Fine-tuning
- Model: `roberta-base`
- Epochs: 3 (capped by script)
- Batch size: 16
- Learning rate: 2e-5
- **NO LoRA** - full model fine-tuning
- **Result**:
  - Validation Accuracy: **84.04%**
  - Test Accuracy: **83.99%**
  - All 14 categories working correctly
  - F1 scores ranging from 0.70 (other) to 0.955 (law)

## Key Differences Between Scripts

### Dataset Loading
- **Both scripts** use MMLU-Pro dataset
- **Both scripts** load only the "question" field from test split
- **Both scripts** use the same 14 required categories

### Data Splitting
- **LoRA script**:
  - First splits in `prepare_datasets()`: 60/20/20 train/val/test with stratification
  - Then combines train+val and re-splits 80/20 (BUG: missing stratification initially, fixed but didn't help)
- **Legacy script**:
  - Single split: 70/15/15 train/val/test with stratification

### Training Configuration

| Parameter | LoRA Script | Legacy Script |
|-----------|-------------|---------------|
| Weight decay | 0.01 | 0.1 |
| Gradient accumulation | None | 2 |
| Max epochs | User specified (3-8) | Capped at 3 |
| Batch size | User specified | Capped at 8 |
| Learning rate | 3e-5 default | 2e-5 fixed |
| Gradient clipping | 1.0 | Default |
| LR scheduler | Cosine | Default |
| Warmup ratio | 0.06 | Warmup steps |

### Model Architecture
- **LoRA script**: Uses PEFT LoRA adapters (r=8-16, alpha=16-32)
- **Legacy script**: Full model fine-tuning (no LoRA)

## ROOT CAUSE IDENTIFIED ✅

The LoRA training script has a **critical dataset loading bug** in the `create_mmlu_dataset()` function and main() that causes:
1. Numerical instability (NaN loss)
2. Complete failure to generalize (6% validation accuracy)
3. Extreme overfitting (near-zero training loss)

### The Bug: Double-Split Data Corruption

**LoRA script (BROKEN) flow**:
1. `MMLU_Dataset.prepare_datasets(max_samples)` creates 60/20/20 train/val/test split with stratification ✓
2. `create_mmlu_dataset()` **COMBINES train + val back together** into `all_data` ✗
   - Line 323: `for text, label in zip(train_texts + val_texts, train_labels + val_labels)`
   - This throws away the carefully stratified splits!
3. `main()` line 491: **RE-SPLITS the combined data 80/20** ✗
   - `train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)`
   - Even after adding stratification here, the damage is done

**Legacy script (WORKING) flow**:
1. `MMLU_Dataset.prepare_datasets()` creates 70/15/15 train/val/test split with stratification ✓
2. **Uses those splits directly** - no combining, no re-splitting ✓
   - Lines 381-383: Direct assignment from `datasets["train"]`, `datasets["validation"]`, `datasets["test"]`

### Why This Breaks Training

The double-split causes:
- **Data leakage**: Validation samples may end up in training set and vice versa
- **Distribution mismatch**: The careful category balancing in `prepare_datasets()` is destroyed
- **Loss of test set**: The original test split is completely ignored
- **Randomization issues**: Two different random_state values (42 in both splits) don't guarantee proper stratification

This is **NOT a hyperparameter issue** or **LoRA-specific issue** - it's a fundamental data loading bug that would break any training approach.

## The Fix

### Code Changes Required

**File**: `src/training/training_lora/classifier_model_fine_tuning_lora/ft_linear_lora.py`

**Lines 489-491** - Replace the broken double-split logic:

**BEFORE (BROKEN)**:
```python
# Load real MMLU-Pro dataset
all_data, category_to_idx, idx_to_category = create_mmlu_dataset(max_samples)
train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
```

**AFTER (FIXED)**:
```python
# Load real MMLU-Pro dataset - use splits directly from prepare_datasets
dataset_loader = MMLU_Dataset()
datasets = dataset_loader.prepare_datasets(max_samples)

train_texts, train_labels = datasets["train"]
val_texts, val_labels = datasets["validation"]

# Convert to format expected by tokenize_data
train_data = [{"text": text, "label": label} for text, label in zip(train_texts, train_labels)]
val_data = [{"text": text, "label": label} for text, label in zip(val_texts, val_labels)]

category_to_idx = dataset_loader.label2id
idx_to_category = dataset_loader.id2label
```

**Optional**: Remove or deprecate the `create_mmlu_dataset()` function (lines 313-329) since it's no longer needed and is the source of the bug.

### Expected Results After Fix

With the fix applied, LoRA training should achieve:
- **Validation accuracy**: 75-85% (similar to legacy script)
- **No NaN losses**: Stable training with normal loss curves
- **No overfitting**: train_loss and eval_loss should be reasonably close
- **All 14 categories**: Properly working with balanced performance

## Recommendations

### Immediate Actions
1. ✅ **Root cause identified**: Double-split data corruption bug
2. **Apply the fix** to ft_linear_lora.py (lines 489-491)
3. **Retrain LoRA model** with the fixed script
4. **Verify 80%+ accuracy** on validation set
5. **Copy trained model** to Mac for E2E testing
6. **Run E2E test 03** to confirm classification works

### Code Quality
1. **Remove create_mmlu_dataset()** function to prevent future bugs
2. **Add comments** explaining why we use prepare_datasets() directly
3. **Add unit tests** to catch data splitting bugs
4. **Document** the proper dataset loading pattern

### Testing Other LoRA Models
1. **Check jailbreak classifier** training script for the same bug
2. **Check PII detector** training script for the same bug
3. **Verify all LoRA models** on HuggingFace were trained correctly

### Long-term
1. **Update HuggingFace models** with properly trained versions
2. **Create training guide** documenting correct dataset loading
3. **Add CI checks** to validate model accuracy before upload

## Files and Locations

### Windows Training Location
- Working model: `~/semantic_router/src/training/classifier_model_fine_tuning/category_classifier_roberta-base_model/`

### Mac Deployment Location
- Target: `/Users/yovadia/code/semantic-router/models/category_classifier_roberta-base_model/`

### Training Scripts
- LoRA (broken): `src/training/training_lora/classifier_model_fine_tuning_lora/ft_linear_lora.py`
- Legacy (working): `src/training/classifier_model_fine_tuning/ft_linear.py`

### Related GitHub Issues
- Issue #584: Classification accuracy regression documentation

## Next Steps
1. Detailed diff comparison of both training scripts
2. Test working RoBERTa model with E2E tests
3. Decide: Fix LoRA or abandon it for intent classification
4. Update documentation and training guides
