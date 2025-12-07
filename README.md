# Arabic Diacritization Project

A machine learning system for automatic Arabic text diacritization using Logistic Regression and Conditional Random Fields (CRF).

## Project Overview

This project implements an end-to-end pipeline for restoring missing diacritics in Arabic text. Diacritics are short vowels that are usually omitted in written Arabic but are crucial for proper pronunciation and disambiguation.

### Models Implemented

1. **Logistic Regression** (Baseline)
   - Character-level classification using sliding windows
   - TF-IDF features with character n-grams
   - Fast training and inference
   - Achieves ~60-70% accuracy

2. **Conditional Random Fields (CRF)**
   - Sequence labeling model
   - Rich feature extraction (character context, word boundaries, trigrams)
   - Forward-backward algorithm for training
   - Viterbi decoding for inference
   - Better accuracy but slower training

## Project Structure

```
NLP_PROJECT_2025/
├── data/
│   ├── train.txt              # Training data (50K sentences)
│   ├── val.txt                # Validation data (2.5K sentences)
│   └── test.txt               # Test data (to be provided)
├── models/                     # Saved trained models
├── outputs/                    # Prediction outputs
├── src/
│   ├── config.py              # Configuration and constants
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── features.py            # Feature extraction
│   ├── models/
│   │   ├── logreg_model.py   # Logistic Regression implementation
│   │   └── crf_model.py      # CRF implementation
│   └── training/
│       ├── train_logreg.py   # LogReg training pipeline
│       └── train_crf.py      # CRF training pipeline
├── utils/                      # Pickle resources (diacritics, mappings)
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks for exploration
├── train_medium.py            # Train LogReg on 5K sentences
├── train_crf_medium.py        # Train CRF on 1K sentences
├── evaluate_and_infer.py      # Evaluation and inference script
└── requirements.txt           # Python dependencies
```

## Setup

### Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended for full training)
- Optional: CUDA-enabled GPU (for faster training with CuPy)

### Installation

```bash
# Clone the repository
cd NLP_PROJECT_2025

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_preprocessing.py
```

## Quick Start

### 1. Data Preprocessing

The preprocessing pipeline:
- Cleans Arabic text (removes non-Arabic characters, normalizes letters)
- Extracts diacritic labels from text
- Creates character-level windows for feature extraction

Test preprocessing:
```bash
python test_preprocessing.py
```

### 2. Training Models

#### Train Logistic Regression (Recommended for quick testing)

```bash
# Train on 5K sentences (~10 minutes)
python train_medium.py

# Train on full 50K sentences (~1 hour)
python train_logreg_quick.py
```

#### Train CRF

```bash
# Train on 1K sentences (~15-20 minutes)
python train_crf_medium.py
```

### 3. Evaluation

```bash
python evaluate_and_infer.py
```

Select option 1 or 2 to evaluate trained models on the dev set.

### 4. Generate Predictions

```bash
python evaluate_and_infer.py
```

Select option 3 or 4 to generate predictions for test data.

## Model Details

### Logistic Regression

**Architecture:**
- Character-level classification
- Sliding window of size 5 (2 chars before + current + 2 chars after)
- TF-IDF features with character unigrams and bigrams
- Softmax classifier with 15 output classes (diacritics)

**Training:**
- Mini-batch gradient descent
- Batch size: 512-1024
- Learning rate: 0.1
- L2 regularization: 0.001
- Iterations: 50-100 epochs

**Performance:**
- Training time: ~10 minutes for 5K sentences
- Inference: Very fast (<1 second for 1K sentences)
- Accuracy: ~60-70% on dev set
- DER (Diacritic Error Rate): ~30-40%

### Conditional Random Fields (CRF)

**Architecture:**
- Linear-chain CRF for sequence labeling
- Rich feature extraction:
  - Current, previous, and next characters
  - Character bigrams and trigrams
  - Word boundary features
  - Character type features
- Transition weights between labels

**Training:**
- Forward-backward algorithm for gradient computation
- Learning rate: 0.01
- L2 penalty: 0.1
- Iterations: 30-50

**Performance:**
- Training time: ~15-20 minutes for 1K sentences
- Inference: Slower (uses Viterbi decoding)
- Accuracy: ~65-75% (typically better than LogReg)
- DER: ~25-35%

## Diacritic Classes

The model predicts 15 different diacritic classes:

| ID | Diacritic | Name | Unicode |
|----|-----------|------|---------|
| 0 | َ | Fatha | U+064E |
| 1 | ً | Fathatan | U+064B |
| 2 | ُ | Damma | U+064F |
| 3 | ٌ | Dammatan | U+064C |
| 4 | ِ | Kasra | U+0650 |
| 5 | ٍ | Kasratan | U+064D |
| 6 | ْ | Sukun | U+0652 |
| 7 | ّ | Shadda | U+0651 |
| 8-13 | | Shadda combinations | |
| 14 | (empty) | No diacritic | - |

## Features

### Preprocessing
- ✅ Arabic text normalization
- ✅ Diacritic extraction and alignment
- ✅ Character and word tokenization
- ✅ Data caching for faster loading

### Feature Extraction
- ✅ Character n-gram TF-IDF (LogReg)
- ✅ Sliding window features
- ✅ Contextual character features (CRF)
- ✅ Word boundary detection

### Models
- ✅ Logistic Regression from scratch (NumPy)
- ✅ CRF from scratch (NumPy)
- ✅ GPU acceleration support (CuPy)
- ✅ Streaming training for large datasets

### Evaluation
- ✅ DER (Diacritic Error Rate) metric
- ✅ Character-level accuracy
- ✅ Per-class metrics (precision, recall, F1)
- ✅ Comprehensive evaluation script

## Advanced Usage

### Training on Full Dataset

```python
from src.training.train_logreg import run_logreg_training

# Train on full 50K sentences
run_logreg_training(
    train_file='data/train.txt',
    dev_file='data/val.txt',
    max_features=5000,      # More features for better accuracy
    max_iter=100,           # More iterations
    learning_rate=0.1,
    batch_size=1024
)
```

### Custom Feature Engineering

```python
from src.features import CharacterIndexer, ManualTFIDF

# Create custom feature extractor
indexer = CharacterIndexer(use_arabic_letters=True)
indexer.fit(train_texts)

# Or use custom TF-IDF
tfidf = ManualTFIDF(ngram=3)  # Trigrams
tfidf.fit_transform(train_texts)
```

### Ensemble Models

Combine predictions from multiple models:

```python
# Load models
logreg = LogisticRegressionModel()
logreg.load('models/logreg_5k.pkl')

crf = CRFModel(...)
crf.load('models/crf_1k.pkl')

# Generate predictions
logreg_preds = logreg.predict(texts)
crf_preds = crf.predict(features)

# Ensemble (e.g., voting)
final_preds = ensemble_vote([logreg_preds, crf_preds])
```

## Performance Optimization Tips

1. **Memory Optimization:**
   - Use streaming training for large datasets
   - Reduce `max_features` parameter
   - Process data in smaller chunks

2. **Speed Optimization:**
   - Install CuPy for GPU acceleration
   - Increase batch size (if memory allows)
   - Use cached preprocessed data

3. **Accuracy Optimization:**
   - Train on more data
   - Increase feature count
   - Use CRF instead of LogReg
   - Ensemble multiple models

## Known Issues & Limitations

- **Memory Usage:** Full training requires ~15-20GB RAM
- **Training Time:** CRF is significantly slower than LogReg
- **Label Imbalance:** Some diacritics are rare in the data
- **Context Window:** Limited to local context (5-11 characters)

## Future Improvements

- [ ] LSTM/Transformer models for better context modeling
- [ ] Word-level embeddings (FastText, Word2Vec)
- [ ] Attention mechanisms
- [ ] Beam search decoding
- [ ] Data augmentation
- [ ] Transfer learning from pre-trained models (AraBERT)

## References

- Original Project Document: See project requirements
- Arabic Diacritics: [Wikipedia](https://en.wikipedia.org/wiki/Arabic_diacritics)
- CRF Tutorial: [Sutton & McCallum, 2011]
- TF-IDF: [Manning et al., Information Retrieval]

## Team

Add your team member names here.

## License

This project is for educational purposes as part of the NLP course at Cairo University.

---

**Last Updated:** December 2025  
**Version:** 1.0
