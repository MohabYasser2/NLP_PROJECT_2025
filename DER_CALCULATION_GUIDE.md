# DER (Diacritic Error Rate) Calculation Guide

## What is DER?

**Diacritic Error Rate (DER)** is the primary metric for evaluating Arabic diacritization models. It measures the percentage of characters where the predicted diacritic is incorrect.

```
DER = (Number of Incorrect Diacritics) / (Total Number of Characters)
DER = 1 - Accuracy
```

## How to Calculate Your DER

### Option 1: Using the Dev/Validation Set (Before Test Release)

Since `test.txt` won't be available until **1 day before final delivery**, you should evaluate your model on the **dev/validation set** during development.

**Your current results from the notebook:**
```
Dev Set Results:
  Accuracy: 0.3529  (35.29%)
  DER: 0.6471       (64.71%)
  Correct: 143,789 / 407,434
```

This means:
- ✅ **Your model correctly predicted 35.29% of diacritics**
- ❌ **Your DER is 64.71%** - this is what you report in your project document

### Option 2: Using the Test Set (Final Submission)

When `test.txt` becomes available (1 day before deadline):

1. **The notebook will automatically detect it** and generate predictions
2. **Submit to Kaggle** to get your official ranking DER
3. **Your Kaggle DER** will be different from dev set DER (could be better or worse)

## Understanding Your Results

### Current Performance (DER = 64.71%)

| DER Range | Performance Level | Status |
|-----------|------------------|---------|
| < 10% | Excellent | 🏆 |
| 10-20% | Very Good | 🥇 |
| 20-30% | Good | 🥈 |
| 30-50% | Acceptable (Baseline) | 🥉 |
| **> 50%** | **Needs Improvement** | ⚠️ **← You are here** |

### Why is Your DER High?

Your current DER of **64.71%** is high because you're using **ultra-light mode** for Kaggle memory constraints:

- ✅ **Only 10k training samples** (20% of full dataset)
- ✅ **Only 1000 features** (very small vocabulary)
- ✅ **Only 50 epochs** (minimal training)
- ✅ **Window size = 3** (limited context: ±1 character)

This is a **baseline model** designed to run on Kaggle without crashing.

## How to Improve Your DER

### Strategy 1: Train Locally (Recommended)

If you have a local machine with more resources:

```python
# Use full dataset and better parameters
run_logreg_training(
    train_file='data/train.txt',        # Full 50k samples
    dev_file='data/val.txt',
    max_features=10000,                 # ↑ Larger vocabulary
    max_iter=500,                       # ↑ More training
    batch_size=64,                      # ↓ Smaller batches (better learning)
    ngram_range=(1, 3),                 # ↑ Include trigrams
    window_size=5                       # ↑ More context (±2 characters)
)
```

**Expected improvement:** DER could drop to **25-35%**

### Strategy 2: Use Better Features

The project requires **at least 3 different features**. You currently have:

1. ✅ **TF-IDF** (currently using)
2. ❌ **Word Embeddings** (to implement)
3. ❌ **Contextual Features** (to implement)

### Strategy 3: Use Better Models

The project requires **at least 2 models**. You have:

1. ✅ **Logistic Regression** (implemented)
2. ✅ **CRF** (implemented but not tested yet)

Try training the CRF model - it's designed for sequence labeling and should perform better!

### Strategy 4: Ensemble Multiple Models

Combine predictions from multiple models:
- Logistic Regression (your current model)
- CRF (your second model)
- Use voting or weighted averaging

## What to Report in Your Project Document

### Section: Evaluation Results

```
Model: Logistic Regression (From Scratch)
Training Configuration:
  - Training Samples: 10,000 (subsampled due to memory)
  - Features: TF-IDF with character n-grams (1-2)
  - Max Features: 1,000
  - Window Size: 3 (±1 character context)
  - Batch Size: 1,024
  - Learning Rate: 0.01
  - Iterations: 50

Dev Set Results:
  - Accuracy: 35.29%
  - DER: 64.71%
  - Total Characters: 407,434
  - Correct Predictions: 143,789

Analysis:
  - This is a baseline model with intentionally limited resources
  - DER is high due to ultra-light configuration (memory constraints)
  - Expected to improve significantly with full dataset and better features
```

### Section: Model Comparison

Create a table comparing all your trials:

| Model | Features | Training Samples | DER (Dev) | Notes |
|-------|----------|-----------------|-----------|-------|
| LogReg | TF-IDF | 10k | 64.71% | Ultra-light baseline |
| LogReg | TF-IDF | 50k | TBD | Full dataset (local) |
| CRF | TF-IDF | 10k | TBD | Sequence model |
| LogReg | Word2Vec | 50k | TBD | With embeddings |

## Next Steps

1. ✅ **Record your current DER: 64.71%**
2. 🔄 **Train CRF model** (use `notebooks/kaggle_train_crf.ipynb`)
3. 🔄 **Try training locally with full dataset**
4. 🔄 **Implement additional features** (word embeddings, contextual)
5. ⏰ **Wait for test.txt** (1 day before deadline)
6. 📤 **Generate final predictions and submit to Kaggle**

## Calculating DER Manually (Optional)

If you want to calculate DER yourself:

```python
from src.preprocessing import load_dataset, clean_arabic_text, extract_labels_simple
from pathlib import Path

# Load dev set with ground truth
dev_sentences = load_dataset(Path('data/val.txt'))

# Count total characters and errors
total_chars = 0
errors = 0

for sent in dev_sentences:
    clean_sent = clean_arabic_text(sent)
    clean_text, true_labels = extract_labels_simple(clean_sent)
    
    # Get predictions from your model
    pred_labels = model.predict([clean_text], window_size=3)[0]
    
    # Count errors
    errors += sum(t != p for t, p in zip(true_labels, pred_labels))
    total_chars += len(true_labels)

der = errors / total_chars
accuracy = 1 - der

print(f"DER: {der:.4f} ({der*100:.2f}%)")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
```

## FAQ

**Q: Is 64.71% DER acceptable?**  
A: It's a valid baseline result, but you should aim for < 30% DER to be competitive.

**Q: How do I know my Kaggle ranking?**  
A: When test.txt is released, submit your predictions to Kaggle. Your rank will be based on DER on the test set.

**Q: Should I report dev DER or test DER?**  
A: Report BOTH! Dev DER shows your model selection process. Test DER is your final ranking.

**Q: Can I use the dev set for training?**  
A: NO! Dev set is for validation and model selection only. Using it for training is cheating.

**Q: What if my test DER is worse than dev DER?**  
A: This is called overfitting. Increase regularization or use more training data.

---

**Good luck with your project! 🚀**
