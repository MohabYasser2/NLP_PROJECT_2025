"""
Configuration Module
Auto-detects Kaggle vs Local environment and sets paths accordingly.
"""

import os
import pickle
from pathlib import Path


def is_kaggle_environment():
    """Detect if running on Kaggle."""
    return os.path.exists('/kaggle/input')


def load_pickle_resource(filename):
    """
    Load a pickle resource from the utils/ directory.
    
    Args:
        filename: Name of the pickle file (e.g., 'diacritics.pickle')
    
    Returns:
        Loaded object from pickle file
    """
    resource_path = Path(__file__).parent.parent / 'utils' / filename
    if not resource_path.exists():
        raise FileNotFoundError(f"Resource file not found: {resource_path}")
    
    with open(resource_path, 'rb') as f:
        return pickle.load(f)


# Environment Detection
IS_KAGGLE = is_kaggle_environment()

# Base Paths
if IS_KAGGLE:
    BASE_DIR = Path('/kaggle/working')
    DATA_DIR = Path('/kaggle/input/arabic-diacritization-dataset')  
    # TODO: Update dataset name to match your Kaggle dataset slug
else:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'

# Data Paths
TRAIN_FILE = DATA_DIR / 'train.txt'
DEV_FILE = DATA_DIR / 'val.txt'  # Using val.txt as dev set
TEST_FILE = DATA_DIR / 'test.txt'

# Model Save Directory
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

# Output Directory
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# Hyperparameters
HYPERPARAMS = {
    # General
    'random_seed': 42,
    'batch_size': 32,
    'num_epochs': 20,
    'early_stopping_patience': 3,
    
    # LSTM Model
    'lstm': {
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': True,
        'learning_rate': 0.001,
    },
    
    # CRF Model
    'crf': {
        'c1': 0.1,  # L1 regularization coefficient
        'c2': 0.1,  # L2 regularization coefficient
        'max_iterations': 100,
        'all_possible_transitions': True,
    },
    
    # ML Baselines
    'ml_baseline': {
        'logistic_regression': {
            'C': 1.0,
            'max_iter': 1000,
            'solver': 'lbfgs',
        },
        'svm': {
            'C': 1.0,
            'kernel': 'linear',
            'max_iter': 1000,
        },
    },
    
    # Feature Extraction
    'features': {
        'tfidf_max_features': 5000,
        'char_ngram_range': (1, 3),
        'word_ngram_range': (1, 2),
    },
}

# Load resources from pickle files
try:
    ARABIC_DIACRITICS = load_pickle_resource('diacritics.pickle')
    # Convert to list if it's a set for consistency
    if isinstance(ARABIC_DIACRITICS, set):
        ARABIC_DIACRITICS = sorted(list(ARABIC_DIACRITICS))
    
    DIACRITIC_TO_ID = load_pickle_resource('diacritic2id.pickle')
    
    ARABIC_LETTERS = load_pickle_resource('arabic_letters.pickle')
    # Convert to list if it's a set for easier iteration
    if isinstance(ARABIC_LETTERS, set):
        ARABIC_LETTERS = sorted(list(ARABIC_LETTERS))
        
except FileNotFoundError as e:
    print(f"Warning: Could not load pickle resources: {e}")
    print("Falling back to hardcoded values...")
    # Fallback to hardcoded values
    ARABIC_DIACRITICS = [
        '\u064B',  # Fathatan
        '\u064C',  # Dammatan
        '\u064D',  # Kasratan
        '\u064E',  # Fatha
        '\u064F',  # Damma
        '\u0650',  # Kasra
        '\u0651',  # Shadda
        '\u0652',  # Sukun
        '\u0653',  # Maddah
        '\u0654',  # Hamza Above
        '\u0655',  # Hamza Below
        '\u0656',  # Subscript Alef
        '\u0657',  # Inverted Damma
        '\u0658',  # Mark Noon Ghunna
    ]
    DIACRITIC_TO_ID = {diac: idx for idx, diac in enumerate(ARABIC_DIACRITICS)}
    DIACRITIC_TO_ID['_'] = len(ARABIC_DIACRITICS)
    ARABIC_LETTERS = None

# Inverse mapping
ID_TO_DIACRITIC = {idx: diac for diac, idx in DIACRITIC_TO_ID.items()}

# Number of diacritic classes
NUM_DIACRITIC_CLASSES = len(DIACRITIC_TO_ID)

# Logging Configuration
LOG_FILE = OUTPUT_DIR / 'training.log'
LOG_LEVEL = 'INFO'

# TODO: Add configuration for Transformer-based models (AraBERT, mBERT, etc.)
# TODO: Add configuration for ensemble methods
# TODO: Add configuration for data augmentation strategies
# TODO: Add W&B or TensorBoard integration configs

print(f"Running on {'Kaggle' if IS_KAGGLE else 'Local'} environment")
print(f"Data directory: {DATA_DIR}")
print(f"Model directory: {MODEL_DIR}")
