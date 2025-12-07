# Arabic Diacritization Project - Complete Pipeline

This repository contains a **production-ready, modular codebase** for Arabic text diacritization using machine learning models implemented **100% from scratch** (no sklearn, no ML frameworks—only Python + NumPy).

## 🎯 Project Overview

**Goal**: Restore missing diacritics in Arabic text using ML models  
**Evaluation Metric**: Diacritic Error Rate (DER)  
**Models Implemented**:
- Logistic Regression (Softmax Regression)
- Linear-chain Conditional Random Fields (CRF)

## 🏗️ Architecture

```
NLP_PROJECT_2025/
├── src/
│   ├── config.py                    # Environment detection & paths
│   ├── preprocessing.py             # Text cleaning, label extraction
│   ├── features.py                  # CharacterIndexer, feature extraction
│   ├── evaluate.py                  # DER calculation, evaluation
│   ├── infer.py                     # Inference utilities
│   ├── utils.py                     # Helper functions
│   ├── models/
│   │   ├── logreg_model.py         # Logistic Regression (from scratch)
│   │   └── crf_model.py            # CRF (from scratch)
│   └── training/
│       ├── train_logreg.py         # LogReg training pipeline
│       └── train_crf.py            # CRF training pipeline
├── data/
│   ├── train.txt                    # Training data
│   ├── val.txt (or dev.txt)        # Validation data
│   └── test.txt                     # Test data
├── notebooks/
│   ├── kaggle_train_logreg.ipynb   # Kaggle notebook for LogReg (<1MB)
│   └── kaggle_train_crf.ipynb      # Kaggle notebook for CRF (<1MB)
├── models/                          # Saved models (generated)
├── outputs/                         # Logs and results (generated)
└── utils/
    ├── diacritics.pickle           # Arabic diacritics
    ├── diacritic2id.pickle         # Diacritic to ID mapping
    └── arabic_letters.pickle       # Arabic letters vocabulary
```

## ⚡ Quick Start

### 1. Test the Pipeline

```bash
python test_pipeline.py
```

This will verify all imports and basic functionality.

### 2. Train Logistic Regression Model

```bash
python -m src.training.train_logreg
```

Or in Python:

```python
from src.training import run_logreg_training
run_logreg_training()
```

### 3. Train CRF Model

```bash
python -m src.training.train_crf
```

Or in Python:

```python
from src.training import run_crf_training
run_crf_training()
```

## 📊 Model Details

### Logistic Regression (Softmax Regression)

**Architecture**:
- **Feature Extraction**: TF-IDF on character n-grams (1-3)
- **Context Window**: Size 5 (±2 characters around target)
- **Classifier**: Multinomial logistic regression with softmax
- **Training**: Mini-batch gradient descent (batch size: 64)
- **Regularization**: L2 penalty (λ = 0.01)

**Implementation** (100% from scratch):
- Custom TF-IDF vectorizer
- Numerically stable softmax
- Cross-entropy loss with regularization
- Gradient descent optimizer

**Hyperparameters**:
```python
learning_rate = 0.01
max_iter = 500
regularization = 0.01
max_features = 5000
ngram_range = (1, 3)
batch_size = 64
```

### Conditional Random Field (CRF)

**Architecture**:
- **Type**: Linear-chain CRF for sequence labeling
- **Features**: Rich contextual features per character
  - Current character
  - Previous 1-2 characters
  - Next 1-2 characters
  - Character bigrams and trigrams
  - Character type (letter/space/digit)
  - Word boundaries (start/end)
- **Parameters**: 
  - Emission weights: (feature_dim, num_labels)
  - Transition weights: (num_labels, num_labels)

**Training** (100% from scratch):
1. **Forward Algorithm**: Compute α (forward probabilities) in log-space
2. **Backward Algorithm**: Compute β (backward probabilities) in log-space
3. **Gradient Computation**: Expected - observed feature counts
4. **Parameter Update**: Gradient ascent with L2 regularization

**Inference**:
- **Viterbi Decoding**: Find most likely label sequence using dynamic programming

**Hyperparameters**:
```python
learning_rate = 0.01
max_iter = 50
l2_penalty = 0.1
```

## 🔬 Implementation Details

### No External ML Libraries!

All models are implemented from scratch using **only**:
- Python standard library
- NumPy (for numerical operations)

**No usage of**:
- scikit-learn
- sklearn-crfsuite
- TensorFlow/PyTorch (for these baseline models)
- HuggingFace
- spaCy, CAMeL Tools, Farasa, etc.

### Preprocessing Pipeline

```python
from src.preprocessing import clean_arabic_text, extract_labels_simple

# Clean text
text = "مَرْحَباً بِكَ"
clean = clean_arabic_text(text)

# Extract labels
clean_text, labels = extract_labels_simple(text)
# clean_text: "مرحبا بك"
# labels: [diacritic_id_1, diacritic_id_2, ...]
```

### Feature Extraction

**For Logistic Regression**:
```python
from src.models.logreg_model import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
vectorizer.fit(train_texts)
features = vectorizer.transform(test_texts)
```

**For CRF**:
```python
from src.models.crf_model import CRFFeatureExtractor

extractor = CRFFeatureExtractor()
features = extractor.fit_transform(train_texts)
```

### Training

Both models expose simple training APIs:

```python
# Logistic Regression
from src.models.logreg_model import LogisticRegressionModel

model = LogisticRegressionModel(learning_rate=0.01, max_iter=500)
model.fit(train_texts, train_labels)
predictions = model.predict(test_texts)
model.save('models/logreg_model.pkl')

# CRF
from src.models.crf_model import CRFModel, CRFFeatureExtractor

extractor = CRFFeatureExtractor()
train_features = extractor.fit_transform(train_texts)

model = CRFModel(num_labels=NUM_CLASSES, feature_dim=extractor.feature_count)
model.fit(train_features, train_labels)
predictions = model.predict(test_features)
model.save('models/crf_model.pkl')
```

### Evaluation

```python
from src.evaluate import calculate_der, calculate_accuracy

der = calculate_der(predictions, targets)
accuracy = calculate_accuracy(predictions, targets)

print(f"DER: {der:.4f}")
print(f"Accuracy: {accuracy:.4f}")
```

## 📓 Kaggle Notebooks

The repository includes two ultra-lightweight Kaggle notebooks (<1MB each):

### `kaggle_train_logreg.ipynb`
- Trains Logistic Regression model
- Evaluates on dev set
- Generates predictions for test set
- Saves `submission.csv`

### `kaggle_train_crf.ipynb`
- Trains CRF model
- Evaluates on dev set
- Generates predictions for test set
- Saves `submission.csv`

**Usage on Kaggle**:
1. Upload the entire repository as a Kaggle dataset
2. Create a new notebook
3. Add the dataset as input
4. Copy the cells from `kaggle_train_logreg.ipynb` or `kaggle_train_crf.ipynb`
5. Run the notebook

The notebooks contain **minimal code**—all heavy logic is in the `/src` modules.

## 🎯 Design Philosophy

1. **Modular**: Each component (preprocessing, features, models, training) is in its own module
2. **Testable**: All functions are unit-testable
3. **Reusable**: Import and use any component independently
4. **Production-Ready**: Proper error handling, logging, and documentation
5. **Educational**: Clear implementations showing how models work from first principles
6. **Lightweight**: Kaggle notebooks are minimal wrappers around training functions

## 📝 Configuration

The `src/config.py` module auto-detects the environment:

```python
from src.config import IS_KAGGLE, TRAIN_FILE, MODEL_DIR

if IS_KAGGLE:
    # Uses /kaggle/input/ and /kaggle/working/
else:
    # Uses local data/ and models/ directories
```

## 🔍 Testing

Run the test pipeline:

```bash
python test_pipeline.py
```

This verifies:
- ✓ Config and pickle resources load correctly
- ✓ Preprocessing functions work
- ✓ Feature extraction works
- ✓ Models initialize correctly
- ✓ Training modules import successfully
- ✓ Evaluation functions work
- ✓ Data files exist

## 📊 Expected Performance

| Model | DER (Dev Set) | Training Time |
|-------|---------------|---------------|
| Logistic Regression | ~10-15% | 10-20 min |
| CRF | ~8-12% | 15-30 min |

*Performance depends on dataset size and hardware*

## 🚀 Advanced Usage

### Custom Training

```python
from src.training import train_logreg, train_crf

# Custom hyperparameters for LogReg
model = train_logreg(
    train_file='data/train.txt',
    dev_file='data/val.txt',
    learning_rate=0.02,
    max_iter=1000,
    max_features=10000
)

# Custom hyperparameters for CRF
model, extractor = train_crf(
    train_file='data/train.txt',
    dev_file='data/val.txt',
    lr=0.02,
    max_iter=100,
    l2_penalty=0.05
)
```

### Inference

```python
from src.models.logreg_model import LogisticRegressionModel

# Load trained model
model = LogisticRegressionModel()
model.load('models/logreg_model.pkl')

# Predict
texts = ["مرحبا بك", "كيف حالك"]
predictions = model.predict(texts)

# Apply diacritics
for text, pred in zip(texts, predictions):
    diacritized = model.apply_diacritics(text, pred)
    print(f"{text} -> {diacritized}")
```

## 🔧 Troubleshooting

### Import Errors

If you get import errors, ensure you're running from the project root:

```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python test_pipeline.py
```

### Pickle Resources Not Found

Ensure the `utils/` directory contains:
- `diacritics.pickle`
- `diacritic2id.pickle`
- `arabic_letters.pickle`

### Memory Issues

For large datasets, adjust batch size:

```python
model = LogisticRegressionModel(batch_size=32)  # Reduce from 64
```

## 📚 Code Structure

### Core Modules

- **`config.py`**: Environment detection, paths, constants
- **`preprocessing.py`**: Text cleaning, tokenization, label extraction
- **`features.py`**: Feature extraction (TF-IDF, CRF features, character indexing)
- **`evaluate.py`**: DER calculation, accuracy, per-class metrics
- **`utils.py`**: Logging, seed setting, helper functions

### Model Modules

- **`models/logreg_model.py`**: 
  - `TfidfVectorizer`: Custom TF-IDF implementation
  - `LogisticRegressionModel`: Softmax regression from scratch

- **`models/crf_model.py`**:
  - `CRFFeatureExtractor`: Contextual feature extraction
  - `CRFModel`: Linear-chain CRF with forward-backward and Viterbi

### Training Modules

- **`training/train_logreg.py`**: End-to-end LogReg training pipeline
- **`training/train_crf.py`**: End-to-end CRF training pipeline

## 🎓 Educational Value

This codebase demonstrates:

1. **From-Scratch ML**: How to implement ML models without libraries
2. **CRF Internals**: Forward-backward algorithm, Viterbi decoding
3. **Softmax Regression**: Multinomial logistic regression details
4. **Feature Engineering**: TF-IDF, contextual features
5. **Gradient Descent**: Mini-batch training, regularization
6. **Production ML**: Modular design, error handling, logging

## 📄 License

This is an educational project for NLP coursework.

## 👥 Contributors

- Course: Natural Language Processing
- Year: 2025

## 🙏 Acknowledgments

- Arabic diacritization dataset
- NumPy community
- Course instructors and TAs

---

**For questions or issues, please create a GitHub issue or contact the repository maintainers.**
