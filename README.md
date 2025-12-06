# Arabic Diacritization NLP Project

A comprehensive Arabic diacritization system using machine learning and deep learning approaches. This project supports training on Kaggle and provides modular, production-ready code for Arabic text diacritization.

## 📋 Features

- **Multiple Model Architectures**:
  - BiLSTM Character-Level Model
  - CRF (Conditional Random Fields) Baseline
  - ML Baselines (Logistic Regression, SVM)
  
- **Kaggle Integration**: Auto-detects Kaggle environment with proper path handling
- **Modular Design**: Clean separation of preprocessing, features, models, and evaluation
- **Comprehensive Evaluation**: DER (Diacritic Error Rate) metric and per-class analysis
- **Interactive Inference**: Diacritize text via command-line or file processing
- **Reproducible**: Seeded random states and documented hyperparameters

## 🗂️ Project Structure

```
arabic-diacritization/
│
├── data/                      # Dataset files
│   ├── train.txt
│   ├── dev.txt
│   └── test.txt
│
├── utils/                     # Pre-existing resources (pickle files)
│   ├── arabic_letters.pickle  # Set of Arabic letters
│   ├── diacritic2id.pickle    # Diacritic to ID mappings
│   └── diacritics.pickle      # Set of Arabic diacritics
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── config.py              # Configuration and paths (loads pickle resources)
│   ├── preprocessing.py       # Text preprocessing utilities
│   ├── features.py            # Feature extraction
│   ├── utils.py               # General utilities
│   │
│   ├── models/                # Model implementations
│   │   ├── __init__.py
│   │   ├── lstm_char.py       # BiLSTM model
│   │   ├── crf_baseline.py    # CRF model
│   │   └── ml_baseline.py     # ML baselines
│   │
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── infer.py               # Inference script
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_baseline_training.ipynb
│   └── 03_kaggle_run.ipynb   # Kaggle training pipeline
│
├── tests/                     # Unit tests
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
│
├── models/                    # Saved models (created automatically)
│
├── test_pickle_loading.py    # Test script for pickle resources
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                # Git ignore rules
```

## 📦 Pre-existing Resources

The project includes pre-existing pickle resources in the `utils/` folder that are automatically loaded by `src/config.py`:

- **`arabic_letters.pickle`**: Set of 36 Arabic letters used in the dataset
- **`diacritic2id.pickle`**: Dictionary mapping diacritics to integer IDs (15 mappings)
- **`diacritics.pickle`**: Set of 8 Arabic diacritics used for labeling

These resources are loaded automatically when you import from `src/config.py`. To verify they load correctly:

```bash
python test_pickle_loading.py
```

The configuration module includes fallback values if pickle files are not found.

## 🚀 Quick Start

### Local Setup

1. **Clone the repository**:
```bash
git clone https://github.com/YOUR_USERNAME/arabic-diacritization.git
cd arabic-diacritization
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare your data**:
Place your dataset files in the `data/` directory:
- `train.txt`: Training data
- `dev.txt`: Development/validation data
- `test.txt`: Test data

### Training Models

#### Train CRF Model
```bash
python src/train.py --model crf
```

#### Train Logistic Regression
```bash
python src/train.py --model logistic_regression --features tfidf
```

#### Train LSTM Model
```bash
python src/train.py --model lstm_char --epochs 20 --batch_size 32
```

### Evaluation

```bash
python src/evaluate.py --model_path models/crf_best.pkl --model_type crf
```

### Inference

#### Interactive Mode
```bash
python src/infer.py --model_path models/crf_best.pkl --model_type crf --interactive
```

#### Process a File
```bash
python src/infer.py --model_path models/crf_best.pkl --model_type crf \
    --input_file input.txt --output_file output.txt
```

#### Diacritize Single Text
```bash
python src/infer.py --model_path models/crf_best.pkl --model_type crf \
    --text "مرحبا بك"
```

## 🏆 Kaggle Training

### Option 1: Using Notebook

1. Upload the repository to Kaggle as a dataset
2. Create a new Kaggle notebook
3. Add your data as a dataset
4. Run `notebooks/03_kaggle_run.ipynb`

### Option 2: Using Scripts

1. Upload code to Kaggle dataset
2. In Kaggle notebook:

```python
# Clone or copy your code
import sys
sys.path.append('/kaggle/input/your-code-dataset')

# Run training
!python src/train.py --model crf
```

## 📊 Evaluation Metrics

### Diacritic Error Rate (DER)
The primary metric: percentage of characters with incorrect diacritic predictions.

```
DER = (Number of incorrect diacritics) / (Total characters)
```

Lower is better. State-of-the-art systems achieve DER < 5%.

### Additional Metrics
- Character-level accuracy
- Per-class precision, recall, F1
- Confusion matrix analysis

## 🔧 Configuration

Edit `src/config.py` to customize:

- **Data paths**: Auto-detects Kaggle vs local
- **Hyperparameters**: Learning rates, batch sizes, etc.
- **Model parameters**: Hidden dimensions, layers, dropout
- **Diacritic mappings**: Arabic diacritic definitions

## 🧪 Running Tests

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test file
python -m unittest tests/test_preprocessing.py

# Run specific test
python -m unittest tests.test_preprocessing.TestStripDiacritics.test_strip_simple
```

## 📈 Model Architectures

### 1. BiLSTM Character Model
- Character-level embeddings
- Bidirectional LSTM layers
- Per-character diacritic prediction
- Masking for variable-length sequences

### 2. CRF Model
- Contextual feature extraction
- Sequence labeling with CRF
- N-gram character features
- Position-based features

### 3. ML Baselines
- TF-IDF character n-gram features
- Logistic Regression or SVM classifier
- Fast training and inference

## 🛣️ Roadmap

### Immediate TODOs
- [ ] Add pre-trained embedding support (FastText, Word2Vec)
- [ ] Implement LSTM + CRF hybrid model
- [ ] Add data augmentation strategies
- [ ] Optimize feature extraction speed
- [ ] Add model ensemble methods

### Future Enhancements
- [ ] Transformer-based models (AraBERT, mBERT)
- [ ] Attention mechanism for LSTM
- [ ] Multi-task learning (diacritics + POS)
- [ ] Beam search decoding
- [ ] Web API for inference
- [ ] Model quantization for mobile deployment

## 📚 Dataset Format

Each line in the dataset files should contain a single Arabic sentence with diacritics:

```
مَرْحَبًا بِكَ فِي الْمَشْرُوعِ
أَهْلًا وَسَهْلًا
```

The system will automatically:
- Strip diacritics for input
- Extract diacritics as labels
- Align characters with their diacritics

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

[Specify your license here]

## 📧 Contact

[Your contact information]

## 🙏 Acknowledgments

- Dataset source: [Specify if applicable]
- Inspired by state-of-the-art Arabic NLP research
- Built with PyTorch, scikit-learn, and sklearn-crfsuite

## 📖 References

[Add relevant papers and resources]

---

**Note**: This is a research/educational project. For production use, consider:
- Additional testing and validation
- Performance optimization
- Error handling improvements
- Security audits
- Deployment infrastructure
