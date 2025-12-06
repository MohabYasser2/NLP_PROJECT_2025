# Arabic Diacritization - Chronological Implementation Guide

**Project Goal**: Restore missing diacritics in Arabic text using ML/DL models  
**Evaluation**: Diacritic Error Rate (DER)  
**Timeline**: 6 weeks (Week 6-12)

---

## Phase 1: Data Preparation (Week 6-7)

### Step 1: Explore the Dataset
**File**: `notebooks/01_eda.ipynb`

**Actions**:
- Load `data/train.txt` and `data/val.txt`
- Count: total lines, characters, words
- Analyze diacritic frequency distribution
- Check class imbalance
- Visualize text length distribution
- Document statistics

**Output**: EDA report with visualizations

---

### Step 2: Text Cleaning
**File**: `src/preprocessing.py`  
**Function**: `clean_arabic_text(text)`

**Actions**:
- Remove non-Arabic characters (keep Arabic letters + diacritics)
- Normalize whitespace
- Handle empty strings
- Test on sample texts

**Test**:
```python
from src.preprocessing import clean_arabic_text
result = clean_arabic_text("مرحباً123!!")
print(result)  # Should output: "مرحباً"
```

---

### Step 3: Diacritic Removal
**File**: `src/preprocessing.py`  
**Function**: `strip_diacritics(text)`

**Actions**:
- Load `ARABIC_DIACRITICS` from `src/config.py` (auto-loaded from pickle)
- Remove all diacritics from input text
- Return clean Arabic text without diacritics

**Test**:
```python
from src.preprocessing import strip_diacritics
result = strip_diacritics("مَرْحَباً")
print(result)  # Should output: "مرحبا"
```

---

### Step 4: Label Extraction
**File**: `src/preprocessing.py`  
**Function**: `extract_labels_simple(text)`

**Actions**:
- Extract diacritics as labels for each character
- Use `DIACRITIC_TO_ID` from `src/config.py` (auto-loaded from pickle)
- Handle Shadda combinations (ّ + َ / ُ / ِ)
- Align labels with characters
- Return: `(clean_text, labels)`

**Test**:
```python
from src.preprocessing import extract_labels_simple
clean, labels = extract_labels_simple("مَرْحَباً")
print(clean)   # "مرحبا"
print(labels)  # [diacritic_ids...]
```

---

### Step 5: Dataset Preparation Pipeline
**File**: `src/preprocessing.py`  
**Function**: `prepare_dataset(input_file, output_prefix)`

**Actions**:
- Read raw text from `input_file`
- Apply `clean_arabic_text()` to each line
- Apply `extract_labels_simple()` to get (input, target) pairs
- Save processed data as pickle files
- Create train/validation splits if needed

**Run**:
```bash
python -c "from src.preprocessing import prepare_dataset; prepare_dataset('data/train.txt', 'data/processed_train')"
python -c "from src.preprocessing import prepare_dataset; prepare_dataset('data/val.txt', 'data/processed_val')"
```

**Output**: `data/processed_train.pkl`, `data/processed_val.pkl`

---

## Phase 2: Feature Engineering (Week 7)

### Step 6: Character Indexing
**File**: `src/features.py`  
**Class**: `CharacterIndexer`

**Actions**:
- Load `ARABIC_LETTERS` from `src/config.py` (36 letters from pickle)
- Add special tokens: `<PAD>` (0), `<UNK>` (1)
- Build `char2idx` and `idx2char` dictionaries
- Implement `encode(text)`: convert text to list of indices
- Implement `decode(indices)`: convert indices back to text
- Implement `encode_batch(texts, max_len)`: batch encoding with padding

**Test**:
```python
from src.features import CharacterIndexer
indexer = CharacterIndexer()
indices = indexer.encode("مرحبا")
decoded = indexer.decode(indices)
print(indices)  # [14, 25, 7, 3, 1]
print(decoded)  # "مرحبا"
```

---

### Step 7: TF-IDF Features
**File**: `src/features.py`  
**Class**: `TfidfFeatureExtractor`

**Actions**:
- Initialize `TfidfVectorizer` with `max_features=5000`
- Use character n-grams `(1, 3)`
- Implement `fit(texts)`: build vocabulary from training data
- Implement `transform(texts)`: convert texts to TF-IDF vectors
- Save fitted vectorizer to `models/tfidf_vectorizer.pkl`

**Test**:
```python
from src.features import TfidfFeatureExtractor
extractor = TfidfFeatureExtractor()
extractor.fit(["مرحبا", "أهلا", "كيف حالك"])
features = extractor.transform(["مرحبا"])
print(features.shape)  # (1, max_features)
```

**Usage**: For Logistic Regression and SVM models

---

### Step 8: Contextual Features (CRF)
**File**: `src/features.py`  
**Class**: `ContextualFeatureExtractor`

**Actions**:
- Implement `char_features(text, position)`: extract features for character at position
  - Current character
  - Previous 2 characters
  - Next 2 characters
  - Character type (letter/space)
  - Word boundaries (BOS/EOS)
- Implement `text_to_features(text)`: convert entire text to feature list
- Return: list of feature dicts for CRF

**Test**:
```python
from src.features import ContextualFeatureExtractor
extractor = ContextualFeatureExtractor()
features = extractor.text_to_features("مرحبا")
print(features[0])  # Feature dict for first character
```

---

## Phase 3: Baseline Models (Week 8)

### Step 9: Logistic Regression Baseline
**File**: `src/models/ml_baseline.py`  
**Class**: `LogisticRegressionModel`

**Actions**:
- Load processed data from Step 5
- Extract TF-IDF features using Step 7
- Initialize `LogisticRegression(C=1.0, max_iter=1000)`
- Train on training data
- Predict diacritic per character
- Save model to `models/lr_baseline.pkl`

**Run**:
```bash
python src/train.py --model logistic_regression --features tfidf
```

**Evaluate**: Calculate DER on validation set

---

### Step 10: SVM Baseline
**File**: `src/models/ml_baseline.py`  
**Class**: `SVMModel`

**Actions**:
- Load processed data from Step 5
- Extract TF-IDF features using Step 7
- Initialize `LinearSVC(C=1.0, max_iter=1000)`
- Train on training data
- Evaluate on validation set
- Save model to `models/svm_baseline.pkl`

**Run**:
```bash
python src/train.py --model svm --features tfidf
```

---

### Step 11: CRF Model
**File**: `src/models/crf_baseline.py`  
**Class**: `CRFModel`

**Actions**:
- Load processed data from Step 5
- Extract contextual features using Step 8
- Initialize `CRF(c1=0.1, c2=0.1, max_iterations=100)`
- Train CRF on sequence data
- Save model to `models/crf_baseline.pkl`

**Run**:
```bash
python src/train.py --model crf
```

---

### Step 12: Compare Baselines
**File**: `notebooks/02_baseline_training.ipynb`

**Actions**:
- Load all baseline models (LR, SVM, CRF)
- Evaluate each on validation set
- Calculate DER for each model
- Create comparison table
- Visualize results

**Output**: Baseline comparison report

---

## Phase 4: Deep Learning Model (Week 9-10)

### Step 13: Character Sequence Preparation
**File**: `src/preprocessing.py`  
**Function**: `create_char_sequences(texts, labels, max_len=100)`

**Actions**:
- Use `CharacterIndexer` from Step 6
- Convert texts to character indices
- Pad sequences to `max_len`
- Convert labels to class IDs using `DIACRITIC_TO_ID`
- Return: `(X_padded, y_padded, lengths)`

**Test**:
```python
from src.preprocessing import create_char_sequences
X, y, lengths = create_char_sequences(["مرحبا"], [[0,1,2,3,4]], max_len=10)
print(X.shape)  # (1, 10)
print(y.shape)  # (1, 10)
```

---

### Step 14: BiLSTM Architecture
**File**: `src/models/lstm_char.py`  
**Class**: `LSTMCharModel`

**Actions**:
- Implement `__init__()`:
  - Embedding layer: `vocab_size → embedding_dim=128`
  - BiLSTM: 2 layers, `hidden_dim=256`, dropout=0.3
  - Linear output: `hidden_dim*2 → num_diacritic_classes`
  
- Implement `forward(x, lengths)`:
  - Embed input
  - Pack padded sequence
  - Pass through BiLSTM
  - Unpack sequence
  - Apply linear layer
  - Return logits

- Implement `compute_loss(predictions, targets, mask)`:
  - CrossEntropyLoss with masking
  - Ignore padded positions

- Implement `predict(x, lengths)`:
  - Run forward pass
  - Apply argmax
  - Return predicted diacritic IDs

**Load config**:
```python
from src.config import HYPERPARAMS
lstm_params = HYPERPARAMS['lstm']
```

---

### Step 15: Training Loop
**File**: `src/train.py`  
**Function**: `train_lstm_model()`

**Actions**:
- Load prepared sequences from Step 13
- Initialize `LSTMCharModel` from Step 14
- Set up optimizer: Adam with `lr=0.001`
- Training loop (20 epochs):
  - Batch iteration
  - Forward pass
  - Compute loss with masking
  - Backward pass
  - Update weights
  - Log training loss
- Validation after each epoch:
  - Calculate validation DER
  - Save best model to `models/lstm_char_best.pt`
  - Early stopping (patience=3)
- Log final results

**Run**:
```bash
python src/train.py --model lstm_char --epochs 20 --batch_size 32 --lr 0.001
```

**Output**: 
- `models/lstm_char_best.pt`
- `outputs/training.log`

---

## Phase 5: Evaluation (Week 10)

### Step 16: DER Metric Implementation
**File**: `src/evaluate.py`  
**Function**: `calculate_der(predictions, targets)`

**Actions**:
- Count total characters (excluding padding)
- Count diacritic mismatches
- Calculate: `DER = errors / total * 100`
- Handle edge cases (empty sequences)

**Test**:
```python
from src.evaluate import calculate_der
predictions = [0, 1, 2, 3]
targets = [0, 1, 3, 3]  # 1 error
der = calculate_der(predictions, targets)
print(der)  # 25.0
```

---

### Step 17: Model Evaluation
**File**: `src/evaluate.py`  
**Function**: `evaluate_model(model, test_loader)`

**Actions**:
- Load model checkpoint
- Run inference on test set
- Calculate overall DER
- Calculate per-diacritic accuracy
- Generate confusion matrix
- Save results to `outputs/evaluation_results.json`

**Run**:
```bash
python src/evaluate.py --model models/lstm_char_best.pt --test_file data/test.txt
```

---

### Step 18: Compare All Models
**File**: `src/evaluate.py`

**Actions**:
- Load all models: LR, SVM, CRF, BiLSTM
- Evaluate each on same test set
- Create comparison table with DER scores
- Generate visualization (bar chart)
- Save to `outputs/model_comparison.png`

**Run**:
```bash
python src/evaluate.py --compare_all --test_file data/test.txt
```

---

## Phase 6: Inference (Week 11)

### Step 19: Inference Pipeline
**File**: `src/infer.py`  
**Function**: `predict_diacritics(text, model_path)`

**Actions**:
- Load trained model from `model_path`
- Preprocess input text (clean + encode)
- Run model prediction
- Decode predictions to diacritics
- Restore diacritics to original text positions
- Return diacritized text

**Test**:
```python
from src.infer import predict_diacritics
result = predict_diacritics("مرحبا", "models/lstm_char_best.pt")
print(result)  # "مَرْحَباً"
```

---

### Step 20: Interactive Mode
**File**: `src/infer.py`

**Actions**:
- Implement interactive loop
- Read user input from console
- Call `predict_diacritics()`
- Display result
- Continue until 'exit'

**Run**:
```bash
python src/infer.py --model models/lstm_char_best.pt --interactive
```

---

### Step 21: Batch File Processing
**File**: `src/infer.py`

**Actions**:
- Read input file line by line
- Apply `predict_diacritics()` to each line
- Write results to output file
- Show progress bar

**Run**:
```bash
python src/infer.py --model models/lstm_char_best.pt --input input.txt --output output.txt
```

---

## Phase 7: Optimization (Week 11-12)

### Step 22: Hyperparameter Tuning
**File**: Create `notebooks/04_hyperparameter_tuning.ipynb`

**Actions**:
- Define parameter grid:
  - `learning_rate`: [0.0001, 0.001, 0.01]
  - `hidden_dim`: [128, 256, 512]
  - `num_layers`: [1, 2, 3]
  - `dropout`: [0.2, 0.3, 0.5]
- Train models with different configs
- Track validation DER for each
- Select best configuration
- Save to `outputs/best_hyperparams.json`

**Output**: Best model configuration

---

### Step 23: Error Analysis
**File**: Create `notebooks/05_error_analysis.ipynb`

**Actions**:
- Load best model predictions
- Identify most common errors
- Analyze patterns:
  - Which diacritics are confused?
  - Position-based errors (start/middle/end)?
  - Word length correlation?
- Visualize error distribution
- Document findings

**Output**: Error analysis report

---

### Step 24: Data Augmentation (Optional)
**File**: `src/preprocessing.py`  
**Function**: `augment_text(text, augmentation_factor=2)`

**Actions**:
- Implement augmentation strategies:
  - Random character substitution (similar shapes)
  - Synthetic diacritic noise
  - Back-translation (if available)
- Apply to training data
- Retrain model with augmented data
- Compare DER improvement

---

### Step 25: Ensemble Model (Optional)
**File**: Create `src/models/ensemble.py`

**Actions**:
- Load predictions from: CRF + BiLSTM
- Implement voting mechanism:
  - Hard voting (majority)
  - Weighted voting (confidence-based)
- Evaluate ensemble DER
- Compare with individual models

**Run**:
```bash
python src/train.py --model ensemble --models crf,lstm_char
```

---

## Phase 8: Kaggle Deployment (Week 12)

### Step 26: Kaggle Environment Setup
**File**: `notebooks/03_kaggle_run.ipynb`

**Actions**:
- Check `IS_KAGGLE` flag from `src/config.py` (auto-detected)
- Verify paths:
  - Data: `/kaggle/input/arabic-diacritization-dataset/`
  - Output: `/kaggle/working/`
- Install requirements (if needed)
- Test data loading

---

### Step 27: Full Training Pipeline
**File**: `notebooks/03_kaggle_run.ipynb`

**Actions**:
- Load training data from Kaggle dataset
- Preprocess data (Steps 2-5)
- Train best model configuration (from Step 22)
- Save model checkpoint
- Validate on dev set
- Log metrics

---

### Step 28: Generate Kaggle Submission
**File**: `notebooks/03_kaggle_run.ipynb`

**Actions**:
- Load test data (released 1 day before deadline)
- Run inference on entire test set
- Format predictions as CSV:
  ```
  id,diacritized_text
  0,مَرْحَباً
  1,كَيْفَ حَالُكَ
  ```
- Save to `submission.csv`
- Verify format
- Submit to Kaggle

**Run**:
```bash
# In Kaggle notebook
python src/infer.py --model models/lstm_char_best.pt --input /kaggle/input/test.txt --output submission.csv --format kaggle
```

---

## Testing Checklist

### Unit Tests
- [ ] Run `pytest tests/test_preprocessing.py` - text cleaning, diacritic extraction
- [ ] Run `pytest tests/test_features.py` - character indexing, feature extraction
- [ ] Run `pytest tests/test_models.py` - model initialization, forward pass

### Integration Tests
- [ ] Test full pipeline: data → preprocessing → training → inference
- [ ] Verify pickle resources load: `python test_pickle_loading.py`
- [ ] Test Kaggle path detection

### Model Validation
- [ ] DER < 15% on validation set (baseline target)
- [ ] BiLSTM outperforms CRF
- [ ] No overfitting (train DER ≈ val DER)

---

## Quick Reference

| Step | File | Function/Class | Command |
|------|------|----------------|---------|
| 1 | `notebooks/01_eda.ipynb` | - | Open notebook |
| 2 | `src/preprocessing.py` | `clean_arabic_text()` | - |
| 3 | `src/preprocessing.py` | `strip_diacritics()` | - |
| 4 | `src/preprocessing.py` | `extract_labels_simple()` | - |
| 5 | `src/preprocessing.py` | `prepare_dataset()` | `python -c "..."` |
| 6 | `src/features.py` | `CharacterIndexer` | - |
| 7 | `src/features.py` | `TfidfFeatureExtractor` | - |
| 8 | `src/features.py` | `ContextualFeatureExtractor` | - |
| 9 | `src/models/ml_baseline.py` | `LogisticRegressionModel` | `python src/train.py --model logistic_regression` |
| 10 | `src/models/ml_baseline.py` | `SVMModel` | `python src/train.py --model svm` |
| 11 | `src/models/crf_baseline.py` | `CRFModel` | `python src/train.py --model crf` |
| 12 | `notebooks/02_baseline_training.ipynb` | - | Open notebook |
| 13 | `src/preprocessing.py` | `create_char_sequences()` | - |
| 14 | `src/models/lstm_char.py` | `LSTMCharModel` | - |
| 15 | `src/train.py` | `train_lstm_model()` | `python src/train.py --model lstm_char` |
| 16 | `src/evaluate.py` | `calculate_der()` | - |
| 17 | `src/evaluate.py` | `evaluate_model()` | `python src/evaluate.py` |
| 18 | `src/evaluate.py` | - | `python src/evaluate.py --compare_all` |
| 19 | `src/infer.py` | `predict_diacritics()` | - |
| 20 | `src/infer.py` | - | `python src/infer.py --interactive` |
| 21 | `src/infer.py` | - | `python src/infer.py --input in.txt --output out.txt` |
| 22 | `notebooks/04_hyperparameter_tuning.ipynb` | - | Open notebook |
| 23 | `notebooks/05_error_analysis.ipynb` | - | Open notebook |
| 24 | `src/preprocessing.py` | `augment_text()` | Optional |
| 25 | `src/models/ensemble.py` | - | Optional |
| 26-28 | `notebooks/03_kaggle_run.ipynb` | - | Run in Kaggle |

---

## Key Files & Resources

**Configuration**:
- `src/config.py` - Auto-loads: `ARABIC_DIACRITICS`, `DIACRITIC_TO_ID`, `ARABIC_LETTERS`

**Pickle Resources** (loaded automatically):
- `utils/diacritics.pickle` - 8 Arabic diacritics
- `utils/diacritic2id.pickle` - 15 diacritic→ID mappings
- `utils/arabic_letters.pickle` - 36 Arabic letters

**Verify**: `python test_pickle_loading.py`

---

## Success Criteria

✅ All preprocessing functions implemented and tested  
✅ All 4 models trained: LR, SVM, CRF, BiLSTM  
✅ BiLSTM achieves DER < 15% on validation set  
✅ Inference pipeline working (interactive + batch)  
✅ Kaggle submission generated and submitted  
✅ All unit tests passing  

---

## Notes for AI Agents

1. **Always check `src/config.py`** - Pickle resources are auto-loaded, don't hardcode values
2. **Use `IS_KAGGLE` flag** - Handles path differences automatically
3. **Follow TODO comments** - All incomplete functions marked with `# TODO:`
4. **Save everything** - Models to `models/`, outputs to `outputs/`
5. **Test incrementally** - Run tests after each step
6. **Log experiments** - Track all hyperparameters and results
