# Arabic Diacritization Project - Complete Setup and Execution Guide

**Date:** December 6, 2025  
**Project:** Arabic Diacritization NLP System

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Data Preparation](#data-preparation)
4. [Environment Configuration](#environment-configuration)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Training Models](#training-models)
7. [Model Evaluation](#model-evaluation)
8. [Inference and Prediction](#inference-and-prediction)
9. [Running Tests](#running-tests)
10. [Kaggle Deployment](#kaggle-deployment)
11. [Troubleshooting](#troubleshooting)
12. [Advanced Usage](#advanced-usage)

---

## Prerequisites

### System Requirements

- **Python:** 3.8 or higher
- **RAM:** Minimum 8GB (16GB recommended for LSTM training)
- **Storage:** At least 2GB free space
- **OS:** Windows, macOS, or Linux

### Required Knowledge

- Basic Python programming
- Command line/terminal usage
- Understanding of machine learning concepts (helpful but not required)

---

## Initial Setup

### Step 1: Navigate to Project Directory

```powershell
cd "c:\Users\Asus\Desktop\UNI\Courses\Year5_1st_Term\NLP\Final Project"
```

### Step 2: Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Expected Installation Time:** 5-10 minutes depending on internet speed

### Step 4: Verify Installation

```powershell
# Test imports
python -c "import torch; import sklearn; import numpy; print('All packages installed successfully!')"
```

---

## Data Preparation

### Step 1: Verify Dataset Files

Your dataset should be in the `data/` directory with these files:

- `train.txt` - Training data with diacritized Arabic text
- `dev.txt` (or `val.txt`) - Development/validation data
- `test.txt` - Test data

**Note:** The project currently has `dataset/train.txt` and `dataset/val.txt`. You need to:

```powershell
# Create data directory if it doesn't exist
New-Item -ItemType Directory -Force -Path data

# Copy or move your files
Copy-Item "dataset\train.txt" -Destination "data\train.txt"
Copy-Item "dataset\val.txt" -Destination "data\dev.txt"

# If you have test data:
# Copy-Item "dataset\test.txt" -Destination "data\test.txt"

# If you don't have test.txt, create a small test split from val.txt
# (You can do this manually or write a script)
```

### Step 2: Verify Data Format

Check that each line contains Arabic text with diacritics:

```powershell
# View first few lines of training data
Get-Content data\train.txt -Head 5
```

Expected format:

```
مَرْحَبًا بِكَ فِي الْمَشْرُوعِ
أَهْلًا وَسَهْلًا
```

### Step 3: Check Data Statistics

```powershell
# Count lines in each file
(Get-Content data\train.txt).Count
(Get-Content data\dev.txt).Count
(Get-Content data\test.txt).Count
```

---

## Environment Configuration

### Step 1: Review Configuration

Open `src/config.py` and verify:

```python
# Check if paths are correct
DATA_DIR = BASE_DIR / 'data'
TRAIN_FILE = DATA_DIR / 'train.txt'
DEV_FILE = DATA_DIR / 'dev.txt'
TEST_FILE = DATA_DIR / 'test.txt'
```

### Step 2: Test Configuration

```powershell
python -c "from src.config import TRAIN_FILE, DEV_FILE, TEST_FILE; print(f'Train: {TRAIN_FILE.exists()}\nDev: {DEV_FILE.exists()}\nTest: {TEST_FILE.exists()}')"
```

All should return `True`.

---

## Exploratory Data Analysis

### Step 1: Launch Jupyter Notebook

```powershell
# Start Jupyter
jupyter notebook
```

This will open your browser automatically.

### Step 2: Open EDA Notebook

Navigate to: `notebooks/01_eda.ipynb`

### Step 3: Run All Cells

1. Click **Cell** → **Run All**
2. Wait for all cells to execute (2-5 minutes)

### Step 4: Review Outputs

The notebook will show you:

- Dataset size and statistics
- Sequence length distributions
- Diacritic frequency analysis
- Character vocabulary
- Data quality checks

**Key Insights to Note:**

- Total number of training samples
- Average sequence length
- Most common diacritics
- Class imbalance issues

---

## Training Models

### Option A: Train CRF Model (Recommended for First Run)

CRF is faster and good for understanding the pipeline.

```powershell
# Train CRF model
python src/train.py --model crf --seed 42

# Expected time: 10-30 minutes depending on dataset size
```

**What happens:**

1. Loads training and dev data
2. Extracts contextual features
3. Trains CRF model
4. Evaluates on dev set
5. Saves best model to `models/crf_best.pkl`

### Option B: Train Logistic Regression Baseline

Fast baseline for comparison.

```powershell
# Train Logistic Regression with TF-IDF features
python src/train.py --model logistic_regression --features tfidf --seed 42

# Expected time: 5-15 minutes
```

Model saved to: `models/logistic_regression_tfidf_best.pkl`

### Option C: Train SVM Baseline

```powershell
# Train SVM with TF-IDF features
python src/train.py --model svm --features tfidf --seed 42

# Expected time: 10-20 minutes
```

Model saved to: `models/svm_tfidf_best.pkl`

### Option D: Train LSTM Model (Advanced)

Requires more computational resources.

```powershell
# Train LSTM model
python src/train.py --model lstm_char --epochs 20 --batch_size 32 --learning_rate 0.001 --seed 42

# Expected time: 1-3 hours depending on hardware
# With GPU: 20-40 minutes
```

**Monitor Training:**

- Watch the loss decreasing
- Dev DER should improve
- Early stopping will trigger if no improvement

Model saved to: `models/lstm_char_best.pt`

### Step 5: Compare Multiple Models (Optional)

Train all models and compare:

```powershell
# Train all baselines
python src/train.py --model crf
python src/train.py --model logistic_regression --features tfidf
python src/train.py --model svm --features tfidf

# Then compare results (see evaluation section)
```

---

## Model Evaluation

### Step 1: Evaluate Trained Model

```powershell
# Evaluate CRF model
python src/evaluate.py --model_path models/crf_best.pkl --model_type crf --test_file data/test.txt

# Evaluate Logistic Regression
python src/evaluate.py --model_path models/logistic_regression_tfidf_best.pkl --model_type logistic_regression

# Evaluate LSTM
python src/evaluate.py --model_path models/lstm_char_best.pt --model_type lstm_char
```

### Step 2: Understand Output Metrics

The evaluation will show:

1. **Overall Metrics:**

   - **DER (Diacritic Error Rate):** Primary metric - lower is better
   - **Accuracy:** Percentage of correctly predicted diacritics

2. **Per-Class Metrics:**

   - Precision, Recall, F1 for each diacritic
   - Support (number of occurrences)

3. **Macro Averages:**
   - Average performance across all classes

**Example Output:**

```
EVALUATION RESULTS
================================================================================

Overall Metrics:
  Diacritic Error Rate (DER): 0.1234 (12.34%)
  Accuracy: 0.8766 (87.66%)

Per-Class Metrics:
Class           Precision    Recall       F1-Score     Support
--------------------------------------------------------------------------------
Fatha           0.9012       0.8923       0.8967       15234
Damma           0.8756       0.8634       0.8695       12456
...
```

### Step 3: Using Jupyter Notebook for Evaluation

Open `notebooks/02_baseline_training.ipynb` for visual comparison:

```powershell
jupyter notebook notebooks/02_baseline_training.ipynb
```

Run all cells to see:

- Side-by-side model comparison
- Bar charts of performance
- Detailed error analysis

---

## Inference and Prediction

### Mode 1: Interactive Mode (Real-time Diacritization)

```powershell
# Start interactive mode
python src/infer.py --model_path models/crf_best.pkl --model_type crf --interactive
```

**Usage:**

```
>>> مرحبا بك
Diacritized: مَرْحَبًا بِكَ

>>> اهلا وسهلا
Diacritized: أَهْلًا وَسَهْلًا

>>> quit
Goodbye!
```

### Mode 2: Single Text Diacritization

```powershell
# Diacritize a single sentence
python src/infer.py --model_path models/crf_best.pkl --model_type crf --text "مرحبا بك في المشروع"
```

**Output:**

```
Input:  مرحبا بك في المشروع
Output: مَرْحَبًا بِكَ فِي الْمَشْرُوعِ
```

### Mode 3: File Processing

```powershell
# Create input file with undiacritized text
Set-Content -Path input.txt -Value @"
مرحبا بك
اهلا وسهلا
كيف حالك
"@

# Diacritize entire file
python src/infer.py --model_path models/crf_best.pkl --model_type crf --input_file input.txt --output_file output.txt

# View results
Get-Content output.txt
```

**Expected output.txt:**

```
مَرْحَبًا بِكَ
أَهْلًا وَسَهْلًا
كَيْفَ حَالُكَ
```

### Step 4: Batch Processing for Submission

```powershell
# Process test file for submission
python src/infer.py --model_path models/crf_best.pkl --model_type crf --input_file data/test.txt --output_file submission.txt
```

---

## Running Tests

### Step 1: Run All Unit Tests

```powershell
# Run all tests
python -m unittest discover tests/

# Expected output: OK (X tests)
```

### Step 2: Run Specific Test Files

```powershell
# Test preprocessing
python -m unittest tests/test_preprocessing.py

# Test features
python -m unittest tests/test_features.py

# Test models
python -m unittest tests/test_models.py
```

### Step 3: Run Specific Test Cases

```powershell
# Run a specific test
python -m unittest tests.test_preprocessing.TestStripDiacritics.test_strip_simple
```

### Step 4: Run Tests with Verbose Output

```powershell
python -m unittest discover tests/ -v
```

---

## Kaggle Deployment

### Step 1: Prepare Code for Kaggle

```powershell
# Create a zip file with your code
Compress-Archive -Path src/,notebooks/,requirements.txt -DestinationPath kaggle_code.zip
```

### Step 2: Upload to Kaggle

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets)
2. Click **New Dataset**
3. Upload `kaggle_code.zip`
4. Name it: `arabic-diacritization-code`
5. Make it public or private

### Step 3: Upload Your Data

1. Create a new dataset
2. Upload `train.txt`, `dev.txt`, `test.txt`
3. Name it: `arabic-diacritization-dataset`

### Step 4: Create Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **New Notebook**
3. Click **File** → **Upload Notebook**
4. Upload `notebooks/03_kaggle_run.ipynb`

### Step 5: Add Datasets to Notebook

1. In your Kaggle notebook, click **Add Data**
2. Search and add:
   - `arabic-diacritization-code`
   - `arabic-diacritization-dataset`

### Step 6: Update Paths in Kaggle Notebook

The notebook should auto-detect Kaggle environment. Verify the first cell shows:

```
Running on Kaggle: True
```

### Step 7: Run Kaggle Notebook

1. Click **Run All**
2. Wait for training to complete (30-60 minutes)
3. Download `submission.txt` from output

### Step 8: Alternative - Direct Code Execution

If you prefer running scripts directly on Kaggle:

```python
# In Kaggle notebook cell:
import sys
sys.path.append('/kaggle/input/arabic-diacritization-code')

# Run training
!python /kaggle/input/arabic-diacritization-code/src/train.py --model crf
```

---

## Troubleshooting

### Issue 1: ModuleNotFoundError

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**

```powershell
# Make sure you're in the project root directory
cd "c:\Users\Asus\Desktop\UNI\Courses\Year5_1st_Term\NLP\Final Project"

# Add current directory to Python path
$env:PYTHONPATH = "."

# Or run with:
python -m src.train --model crf
```

### Issue 2: FileNotFoundError for Data

**Error:** `FileNotFoundError: Dataset file not found`

**Solution:**

```powershell
# Check if data files exist
Test-Path data/train.txt
Test-Path data/dev.txt
Test-Path data/test.txt

# If False, copy from dataset folder
Copy-Item dataset/* data/
```

### Issue 3: CUDA/GPU Not Available

**Error:** LSTM training is slow

**Solution:**

```powershell
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch:
# Visit: https://pytorch.org/get-started/locally/
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 4: Out of Memory (LSTM)

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**

```powershell
# Reduce batch size
python src/train.py --model lstm_char --batch_size 16

# Or use smaller sequences
python src/train.py --model lstm_char --batch_size 8
```

### Issue 5: Virtual Environment Not Activating

**Error:** PowerShell execution policy restriction

**Solution:**

```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine

# Or bypass for current session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Then activate
.\venv\Scripts\Activate.ps1
```

### Issue 6: Jupyter Notebook Won't Start

**Solution:**

```powershell
# Reinstall Jupyter
pip install --upgrade jupyter notebook

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```

### Issue 7: sklearn-crfsuite Installation Fails

**Solution:**

```powershell
# Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or use conda:
conda install -c conda-forge sklearn-crfsuite
```

---

## Advanced Usage

### Custom Hyperparameter Tuning

```powershell
# LSTM with custom parameters
python src/train.py --model lstm_char --epochs 30 --batch_size 64 --learning_rate 0.0001 --seed 42

# Save to specific location
python src/train.py --model crf --save_path models/crf_custom.pkl
```

### Using Pre-trained Embeddings

Edit `src/config.py` and add embedding path:

```python
EMBEDDING_PATH = Path('embeddings/fasttext_ar.vec')
```

Then modify `src/models/lstm_char.py` to load embeddings.

### Ensemble Models

Train multiple models and combine predictions:

```python
from src.models import CRFModel, MLBaselineModel

# Load models
crf = CRFModel.load('models/crf_best.pkl')
lr = MLBaselineModel.load('models/logistic_regression_tfidf_best.pkl')

# Implement voting or averaging
# (Custom implementation required)
```

### Cross-Validation

Modify training script to use k-fold cross-validation:

```python
from sklearn.model_selection import KFold

# Add to train.py
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
    # Train model on fold
    pass
```

---

## Complete Workflow Summary

### Quick Start (Minimal Steps)

```powershell
# 1. Setup
cd "c:\Users\Asus\Desktop\UNI\Courses\Year5_1st_Term\NLP\Final Project"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Prepare Data
Copy-Item dataset/* data/

# 3. Train Model
python src/train.py --model crf

# 4. Evaluate
python src/evaluate.py --model_path models/crf_best.pkl --model_type crf

# 5. Inference
python src/infer.py --model_path models/crf_best.pkl --model_type crf --interactive
```

### Full Workflow (Complete Analysis)

```powershell
# 1. Setup environment
cd "c:\Users\Asus\Desktop\UNI\Courses\Year5_1st_Term\NLP\Final Project"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Prepare data
New-Item -ItemType Directory -Force -Path data
Copy-Item dataset/train.txt data/
Copy-Item dataset/val.txt data/dev.txt

# 3. Run EDA
jupyter notebook notebooks/01_eda.ipynb
# (Run all cells, review insights)

# 4. Train multiple models
python src/train.py --model crf --seed 42
python src/train.py --model logistic_regression --features tfidf --seed 42
python src/train.py --model svm --features tfidf --seed 42

# 5. Compare models in notebook
jupyter notebook notebooks/02_baseline_training.ipynb

# 6. Evaluate best model
python src/evaluate.py --model_path models/crf_best.pkl --model_type crf

# 7. Run tests
python -m unittest discover tests/

# 8. Generate predictions
python src/infer.py --model_path models/crf_best.pkl --model_type crf --input_file data/test.txt --output_file submission.txt

# 9. Review submission
Get-Content submission.txt -Head 10
```

---

## Performance Benchmarks

Expected performance on standard Arabic diacritization datasets:

| Model               | Training Time | DER (Test) | Accuracy |
| ------------------- | ------------- | ---------- | -------- |
| Logistic Regression | 5-10 min      | ~15-20%    | ~80-85%  |
| SVM                 | 10-20 min     | ~12-18%    | ~82-88%  |
| CRF                 | 15-30 min     | ~8-12%     | ~88-92%  |
| BiLSTM              | 1-3 hours     | ~5-8%      | ~92-95%  |

_Note: Actual results depend on dataset size and quality_

---

## Next Steps and Improvements

### Immediate Enhancements

1. **Add test.txt to data/** if not present
2. **Run full EDA** to understand your specific dataset
3. **Train all baseline models** for comparison
4. **Fine-tune best model** by adjusting hyperparameters

### Future Improvements

1. Implement LSTM + CRF hybrid
2. Add Transformer-based models (AraBERT)
3. Implement data augmentation
4. Add model ensemble
5. Optimize inference speed
6. Create web API for deployment

---

## Support and Resources

### Documentation

- Project README: `README.md`
- Code documentation: Docstrings in each module
- Configuration: `src/config.py`

### Getting Help

- Check error messages carefully
- Review troubleshooting section
- Inspect logs in `outputs/training.log`
- Use `--help` flag: `python src/train.py --help`

### Learning Resources

- Arabic NLP basics
- Sequence labeling with CRF
- PyTorch LSTM tutorial
- sklearn documentation

---

## Checklist

Use this checklist to track your progress:

- [ ] Virtual environment created and activated
- [ ] Dependencies installed successfully
- [ ] Data files in `data/` directory
- [ ] Configuration verified
- [ ] EDA notebook executed
- [ ] At least one model trained
- [ ] Model evaluation completed
- [ ] Inference tested (interactive or file)
- [ ] Unit tests passing
- [ ] Submission file generated
- [ ] (Optional) Kaggle deployment completed

---

**Project Status:** Ready for Training and Experimentation  
**Last Updated:** December 6, 2025

**Good luck with your Arabic Diacritization project! 🚀**
