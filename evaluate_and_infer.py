"""
Comprehensive Evaluation and Inference Module
Evaluates trained models and generates predictions for test data
"""

import sys
from pathlib import Path
import pickle
from typing import List, Tuple
import csv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import MODEL_DIR, DEV_FILE, TEST_FILE, ID_TO_DIACRITIC
from src.preprocessing import load_dataset, clean_arabic_text
from src.models.logreg_model import LogisticRegressionModel
from src.models.crf_model import CRFModel, CRFFeatureExtractor


def evaluate_logreg(model_path: str, data_file: str = None):
    """
    Evaluate Logistic Regression model.
    
    Args:
        model_path: Path to saved model
        data_file: Path to evaluation data (default: dev set)
    """
    print("="*70)
    print(" LOGISTIC REGRESSION EVALUATION")
    print("="*70)
    
    if data_file is None:
        data_file = DEV_FILE
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = LogisticRegressionModel()
    model.load(model_path)
    
    # Load data
    print(f"Loading data from {data_file}...")
    from src.training.train_logreg import load_or_prepare_data
    texts, labels = load_or_prepare_data(Path(data_file))
    print(f"  ✓ Loaded {len(texts):,} sentences")
    
    # Evaluate
    print("\nEvaluating...")
    metrics = model.evaluate(texts, labels, window_size=5)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("-"*70)
    print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   DER:      {metrics['der']:.4f}")
    print(f"   Correct:  {metrics['correct']:,} / {metrics['total']:,}")
    print("="*70)
    
    return metrics


def evaluate_crf(model_path: str, features_path: str, data_file: str = None):
    """
    Evaluate CRF model.
    
    Args:
        model_path: Path to saved model
        features_path: Path to saved feature extractor
        data_file: Path to evaluation data (default: dev set)
    """
    print("="*70)
    print(" CRF EVALUATION")
    print("="*70)
    
    if data_file is None:
        data_file = DEV_FILE
    
    # Load model and features
    print(f"\nLoading model from {model_path}...")
    from src.config import NUM_DIACRITIC_CLASSES
    model = CRFModel(num_labels=NUM_DIACRITIC_CLASSES, feature_dim=0)
    model.load(model_path)
    
    print(f"Loading feature extractor from {features_path}...")
    with open(features_path, 'rb') as f:
        feature_extractor = pickle.load(f)
    
    # Load data
    print(f"Loading data from {data_file}...")
    from src.training.train_crf import load_or_prepare_data
    import numpy as np
    texts, labels = load_or_prepare_data(Path(data_file))
    print(f"  ✓ Loaded {len(texts):,} sentences")
    
    # Extract features
    print("Extracting features...")
    features = feature_extractor.transform(texts)
    labels_np = [np.array(l, dtype=np.int32) for l in labels]
    
    # Evaluate
    print("Evaluating...")
    metrics = model.evaluate(features, labels_np)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("-"*70)
    print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   DER:      {metrics['der']:.4f}")
    print(f"   Correct:  {metrics['correct']:,} / {metrics['total']:,}")
    print("="*70)
    
    return metrics


def generate_predictions_logreg(
    model_path: str,
    input_file: str,
    output_file: str,
    window_size: int = 5
):
    """
    Generate predictions for test data using Logistic Regression.
    
    Args:
        model_path: Path to saved model
        input_file: Path to input text file (undiacritized)
        output_file: Path to save predictions (CSV format)
        window_size: Context window size
    """
    print("="*70)
    print(" GENERATING PREDICTIONS - LOGISTIC REGRESSION")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = LogisticRegressionModel()
    model.load(model_path)
    
    # Load test data
    print(f"Loading test data from {input_file}...")
    sentences = load_dataset(Path(input_file))
    texts = [clean_arabic_text(sent) for sent in sentences]
    print(f"  ✓ Loaded {len(texts):,} sentences")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict(texts, window_size=window_size)
    
    # Apply diacritics
    print("Applying diacritics...")
    diacritized_texts = []
    for text, pred_labels in zip(texts, predictions):
        diacritized = model.apply_diacritics(text, pred_labels)
        diacritized_texts.append(diacritized)
    
    # Save to CSV
    print(f"\nSaving predictions to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text'])
        for idx, text in enumerate(diacritized_texts):
            writer.writerow([idx, text])
    
    print(f"  ✓ Saved {len(diacritized_texts):,} predictions")
    print("\n" + "="*70)
    print("✓ Predictions generated successfully!")
    print("="*70)


def generate_predictions_crf(
    model_path: str,
    features_path: str,
    input_file: str,
    output_file: str
):
    """
    Generate predictions for test data using CRF.
    
    Args:
        model_path: Path to saved model
        features_path: Path to saved feature extractor
        input_file: Path to input text file (undiacritized)
        output_file: Path to save predictions (CSV format)
    """
    print("="*70)
    print(" GENERATING PREDICTIONS - CRF")
    print("="*70)
    
    # Load model and features
    print(f"\nLoading model from {model_path}...")
    from src.config import NUM_DIACRITIC_CLASSES
    model = CRFModel(num_labels=NUM_DIACRITIC_CLASSES, feature_dim=0)
    model.load(model_path)
    
    print(f"Loading feature extractor from {features_path}...")
    with open(features_path, 'rb') as f:
        feature_extractor = pickle.load(f)
    
    # Load test data
    print(f"Loading test data from {input_file}...")
    sentences = load_dataset(Path(input_file))
    texts = [clean_arabic_text(sent) for sent in sentences]
    print(f"  ✓ Loaded {len(texts):,} sentences")
    
    # Extract features
    print("\nExtracting features...")
    features = feature_extractor.transform(texts)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(features)
    
    # Apply diacritics
    print("Applying diacritics...")
    diacritized_texts = []
    for text, pred_labels in zip(texts, predictions):
        diacritized = model.apply_diacritics(text, pred_labels)
        diacritized_texts.append(diacritized)
    
    # Save to CSV
    print(f"\nSaving predictions to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text'])
        for idx, text in enumerate(diacritized_texts):
            writer.writerow([idx, text])
    
    print(f"  ✓ Saved {len(diacritized_texts):,} predictions")
    print("\n" + "="*70)
    print("✓ Predictions generated successfully!")
    print("="*70)


def main():
    """Interactive menu for evaluation and prediction"""
    print("="*70)
    print(" ARABIC DIACRITIZATION - EVALUATION & INFERENCE")
    print("="*70)
    
    print("\nAvailable options:")
    print("  1. Evaluate Logistic Regression model")
    print("  2. Evaluate CRF model")
    print("  3. Generate predictions with Logistic Regression")
    print("  4. Generate predictions with CRF")
    print("  5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        model_path = input("Enter model path (or press Enter for default): ").strip()
        if not model_path:
            model_path = str(MODEL_DIR / 'logreg_5k.pkl')
        evaluate_logreg(model_path)
    
    elif choice == '2':
        model_path = input("Enter model path (or press Enter for default): ").strip()
        features_path = input("Enter features path (or press Enter for default): ").strip()
        if not model_path:
            model_path = str(MODEL_DIR / 'crf_1k.pkl')
        if not features_path:
            features_path = str(MODEL_DIR / 'crf_features_1k.pkl')
        evaluate_crf(model_path, features_path)
    
    elif choice == '3':
        model_path = input("Enter model path (or press Enter for default): ").strip()
        input_file = input("Enter input file path: ").strip()
        output_file = input("Enter output file path: ").strip()
        if not model_path:
            model_path = str(MODEL_DIR / 'logreg_5k.pkl')
        generate_predictions_logreg(model_path, input_file, output_file)
    
    elif choice == '4':
        model_path = input("Enter model path (or press Enter for default): ").strip()
        features_path = input("Enter features path (or press Enter for default): ").strip()
        input_file = input("Enter input file path: ").strip()
        output_file = input("Enter output file path: ").strip()
        if not model_path:
            model_path = str(MODEL_DIR / 'crf_1k.pkl')
        if not features_path:
            features_path = str(MODEL_DIR / 'crf_features_1k.pkl')
        generate_predictions_crf(model_path, features_path, input_file, output_file)
    
    elif choice == '5':
        print("Goodbye!")
        return
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
