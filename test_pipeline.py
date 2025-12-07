"""
Test Script - Verify Training Pipeline
Quick test to ensure all imports and basic functionality work.
"""

import sys
from pathlib import Path

print("="*60)
print("TESTING ARABIC DIACRITIZATION PIPELINE")
print("="*60)

# Test 1: Import config
print("\n[1/8] Testing config imports...")
try:
    from src.config import (
        IS_KAGGLE, TRAIN_FILE, DEV_FILE, MODEL_DIR,
        ARABIC_DIACRITICS, DIACRITIC_TO_ID, ID_TO_DIACRITIC,
        NUM_DIACRITIC_CLASSES
    )
    print(f"  ✓ Config loaded")
    print(f"  ✓ Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
    print(f"  ✓ Number of diacritic classes: {NUM_DIACRITIC_CLASSES}")
    print(f"  ✓ Diacritics loaded: {len(ARABIC_DIACRITICS)}")
except Exception as e:
    print(f"  ✗ Config import failed: {e}")
    sys.exit(1)

# Test 2: Import preprocessing
print("\n[2/8] Testing preprocessing imports...")
try:
    from src.preprocessing import (
        clean_arabic_text, strip_diacritics, extract_labels_simple,
        prepare_dataset, load_dataset
    )
    print("  ✓ Preprocessing module loaded")
    
    # Test basic functions
    test_text = "مَرْحَباً"
    clean = strip_diacritics(test_text)
    print(f"  ✓ strip_diacritics: '{test_text}' -> '{clean}'")
    
    clean_text, labels = extract_labels_simple(test_text)
    print(f"  ✓ extract_labels_simple: {len(labels)} labels extracted")
    
except Exception as e:
    print(f"  ✗ Preprocessing import failed: {e}")
    sys.exit(1)

# Test 3: Import features
print("\n[3/8] Testing features imports...")
try:
    from src.features import CharacterIndexer
    print("  ✓ Features module loaded")
    
    indexer = CharacterIndexer()
    indices = indexer.encode("مرحبا")
    print(f"  ✓ CharacterIndexer: encoded {len(indices)} characters")
    
except Exception as e:
    print(f"  ✗ Features import failed: {e}")
    sys.exit(1)

# Test 4: Import Logistic Regression model
print("\n[4/8] Testing LogisticRegression model...")
try:
    from src.models.logreg_model import LogisticRegressionModel, TfidfVectorizer
    print("  ✓ LogisticRegression model imported")
    
    # Test TF-IDF
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    vectorizer.fit(["مرحبا", "أهلا"])
    features = vectorizer.transform(["مرحبا"])
    print(f"  ✓ TfidfVectorizer: {features.shape[1]} features extracted")
    
    # Test model initialization
    model = LogisticRegressionModel(max_iter=10)
    print(f"  ✓ LogisticRegressionModel initialized")
    
except Exception as e:
    print(f"  ✗ LogisticRegression import failed: {e}")
    sys.exit(1)

# Test 5: Import CRF model
print("\n[5/8] Testing CRF model...")
try:
    from src.models.crf_model import CRFModel, CRFFeatureExtractor
    print("  ✓ CRF model imported")
    
    # Test feature extractor
    extractor = CRFFeatureExtractor()
    features = extractor.text_to_features("مرحبا")
    print(f"  ✓ CRFFeatureExtractor: {len(features)} positions, {len(features[0])} features per position")
    
    # Test model initialization
    model = CRFModel(num_labels=10, feature_dim=100, max_iter=5)
    print(f"  ✓ CRFModel initialized")
    
except Exception as e:
    print(f"  ✗ CRF import failed: {e}")
    sys.exit(1)

# Test 6: Import training modules
print("\n[6/8] Testing training modules...")
try:
    from src.training import run_logreg_training, run_crf_training
    print("  ✓ Training modules imported")
    print("  ✓ run_logreg_training available")
    print("  ✓ run_crf_training available")
    
except Exception as e:
    print(f"  ✗ Training import failed: {e}")
    sys.exit(1)

# Test 7: Import evaluation
print("\n[7/8] Testing evaluation module...")
try:
    from src.evaluate import calculate_der, calculate_accuracy
    print("  ✓ Evaluation module imported")
    
    # Test DER calculation
    pred = [[0, 1, 2]]
    target = [[0, 1, 3]]
    der = calculate_der(pred, target)
    print(f"  ✓ DER calculation: {der:.4f}")
    
except Exception as e:
    print(f"  ✗ Evaluation import failed: {e}")
    sys.exit(1)

# Test 8: Check data files
print("\n[8/8] Checking data files...")
try:
    if Path(TRAIN_FILE).exists():
        print(f"  ✓ Training file found: {TRAIN_FILE}")
    else:
        print(f"  ⚠ Training file not found: {TRAIN_FILE}")
    
    if Path(DEV_FILE).exists():
        print(f"  ✓ Dev file found: {DEV_FILE}")
    else:
        print(f"  ⚠ Dev file not found: {DEV_FILE}")
    
    print(f"  ✓ Model directory: {MODEL_DIR}")
    
except Exception as e:
    print(f"  ✗ Data check failed: {e}")

print("\n" + "="*60)
print("ALL TESTS PASSED! ✓")
print("="*60)
print("\nThe pipeline is ready to use!")
print("\nTo train models:")
print("  • Logistic Regression: python -m src.training.train_logreg")
print("  • CRF: python -m src.training.train_crf")
print("\nFor Kaggle:")
print("  • Open notebooks/kaggle_train_logreg.ipynb")
print("  • Open notebooks/kaggle_train_crf.ipynb")
print("="*60)
