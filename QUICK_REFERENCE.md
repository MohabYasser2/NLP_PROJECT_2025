# Quick Reference Guide - Arabic Diacritization

## 🚀 Quick Start (30 seconds)

```bash
# 1. Test everything works
python test_pipeline.py

# 2. Train Logistic Regression (10-20 min)
python -m src.training.train_logreg

# 3. Train CRF (15-30 min)
python -m src.training.train_crf
```

## 📊 For Kaggle

### Option 1: Use Pre-made Notebooks
1. Open `notebooks/kaggle_train_logreg.ipynb` OR `notebooks/kaggle_train_crf.ipynb`
2. Upload to Kaggle
3. Add your dataset as input
4. Run all cells
5. Download `submission.csv`

### Option 2: Custom Notebook
```python
# Cell 1: Setup paths
TRAIN_FILE = '/kaggle/input/your-dataset/train.txt'
DEV_FILE = '/kaggle/input/your-dataset/dev.txt'
TEST_FILE = '/kaggle/input/your-dataset/test.txt'
OUTPUT_FILE = '/kaggle/working/submission.csv'

# Cell 2: Train and predict
from src.training import run_logreg_training
run_logreg_training(TRAIN_FILE, DEV_FILE, TEST_FILE, OUTPUT_FILE)
```

## 🔧 Common Tasks

### Load and Predict with Trained Model

**Logistic Regression:**
```python
from src.models.logreg_model import LogisticRegressionModel

model = LogisticRegressionModel()
model.load('models/logreg_model.pkl')

texts = ["مرحبا بك في بلدنا"]
predictions = model.predict(texts)
diacritized = model.apply_diacritics(texts[0], predictions[0])
print(diacritized)
```

**CRF:**
```python
from src.models.crf_model import CRFModel, CRFFeatureExtractor
import pickle

# Load model and features
model = CRFModel(num_labels=15, feature_dim=0)
model.load('models/crf_model.pkl')

with open('models/crf_features.pkl', 'rb') as f:
    extractor = pickle.load(f)

# Predict
texts = ["مرحبا بك في بلدنا"]
features = extractor.transform(texts)
predictions = model.predict(features)
diacritized = model.apply_diacritics(texts[0], predictions[0])
print(diacritized)
```

### Evaluate on Test Set

```python
from src.evaluate import calculate_der, calculate_accuracy
from src.preprocessing import prepare_dataset

# Load data
test_texts, test_labels = prepare_dataset('data/test.txt', 'data/test')

# Get predictions (using your trained model)
predictions = model.predict(test_texts)

# Calculate metrics
der = calculate_der(predictions, test_labels)
accuracy = calculate_accuracy(predictions, test_labels)

print(f"DER: {der:.4f} ({der*100:.2f}%)")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
```

### Preprocess New Data

```python
from src.preprocessing import clean_arabic_text, extract_labels_simple

# Clean text
text = "مَرْحَباً بِكَ فِي بَلَدِنَا"
clean = clean_arabic_text(text)

# Extract labels
clean_text, labels = extract_labels_simple(text)
print(f"Clean: {clean_text}")
print(f"Labels: {labels}")
```

## 🎯 Model Comparison

| Model | DER | Train Time | Memory | Best For |
|-------|-----|------------|--------|----------|
| **Logistic Regression** | ~10-15% | 10-20 min | ~500 MB | Fast baseline, interpretable |
| **CRF** | ~8-12% | 15-30 min | ~1 GB | Better accuracy, sequence modeling |

## 📁 File Structure Quick Reference

```
src/
├── config.py              # Load this for paths & constants
├── preprocessing.py       # clean_arabic_text, extract_labels_simple
├── features.py           # CharacterIndexer
├── evaluate.py           # calculate_der, calculate_accuracy
├── models/
│   ├── logreg_model.py  # LogisticRegressionModel
│   └── crf_model.py     # CRFModel, CRFFeatureExtractor
└── training/
    ├── train_logreg.py  # run_logreg_training
    └── train_crf.py     # run_crf_training
```

## 🔑 Key Functions

### Preprocessing
```python
from src.preprocessing import (
    clean_arabic_text,      # Normalize text
    strip_diacritics,       # Remove diacritics
    extract_labels_simple,  # Get diacritic labels
    prepare_dataset         # Full pipeline
)
```

### Training
```python
from src.training import (
    run_logreg_training,    # Train LogReg end-to-end
    run_crf_training        # Train CRF end-to-end
)
```

### Evaluation
```python
from src.evaluate import (
    calculate_der,          # Diacritic Error Rate
    calculate_accuracy      # Character accuracy
)
```

## ⚙️ Hyperparameter Tuning

### Logistic Regression
```python
from src.training import train_logreg

model = train_logreg(
    learning_rate=0.02,      # Default: 0.01
    max_iter=1000,           # Default: 500
    regularization=0.005,    # Default: 0.01
    max_features=10000,      # Default: 5000
    ngram_range=(1, 4),      # Default: (1, 3)
    batch_size=128           # Default: 64
)
```

### CRF
```python
from src.training import train_crf

model, extractor = train_crf(
    lr=0.02,                 # Default: 0.01
    max_iter=100,            # Default: 50
    l2_penalty=0.05          # Default: 0.1
)
```

## 🐛 Troubleshooting

### Import Error: "No module named 'src'"
```bash
# Solution: Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### FileNotFoundError: pickle resources
```bash
# Solution: Ensure utils/ directory has:
# - diacritics.pickle
# - diacritic2id.pickle
# - arabic_letters.pickle
```

### Memory Error during training
```python
# Solution: Reduce batch size
model = LogisticRegressionModel(batch_size=32)  # or 16
```

### Slow training
```python
# Solution 1: Use fewer features
model = LogisticRegressionModel(max_features=2000)

# Solution 2: Fewer iterations
model = LogisticRegressionModel(max_iter=200)

# Solution 3: For CRF, reduce max_iter
model = CRFModel(..., max_iter=20)
```

## 📊 Monitoring Training

Both models print progress during training:

**Logistic Regression:**
```
Epoch 100/500, Loss: 0.4523
Epoch 200/500, Loss: 0.3891
...
```

**CRF:**
```
Iteration 10/50, Avg Log-Likelihood: -2.3456
Iteration 20/50, Avg Log-Likelihood: -1.8923
...
```

## 💡 Tips

1. **Start with small data** to test pipeline
2. **Cache preprocessing** results (automatic with `prepare_dataset`)
3. **LogReg is faster** - use for quick experiments
4. **CRF is more accurate** - use for final submission
5. **Monitor DER on dev set** during training
6. **Save models frequently** to avoid losing progress

## 📝 Submission Format

The training functions automatically generate `submission.csv`:

```csv
id,diacritized_text
0,مَرْحَباً بِكَ فِي بَلَدِنَا
1,كَيْفَ حَالُكَ الْيَوْمَ
...
```

## 🎓 Understanding Output

### DER (Diacritic Error Rate)
- **Lower is better**
- 0.10 = 10% error rate (90% correct)
- Typical range: 8-15%

### Accuracy
- **Higher is better**
- 0.90 = 90% of characters correct
- Inverse of DER

## 🔗 Related Files

- `IMPLEMENTATION_GUIDE.md` - Detailed documentation
- `IMPLEMENTATION_SUMMARY.md` - What was implemented
- `TODO_CHRONOLOGICAL.md` - Original requirements
- `test_pipeline.py` - Verify installation

---

**Need help?** Check `IMPLEMENTATION_GUIDE.md` for detailed explanations!
