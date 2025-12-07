"""
Training Module for CRF
Provides end-to-end training pipeline for CRF diacritization model.
"""

import os
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import pickle
import time

from src.config import TRAIN_FILE, DEV_FILE, MODEL_DIR, NUM_DIACRITIC_CLASSES
from src.preprocessing import prepare_dataset, load_dataset, clean_arabic_text, extract_labels_simple
from src.models.crf_model import CRFModel, CRFFeatureExtractor


def load_or_prepare_data(data_file: Path, use_cache: bool = True) -> Tuple[List[str], List[List[int]]]:
    """
    Load or prepare dataset.
    
    Args:
        data_file: Path to data file
        use_cache: Whether to use cached pickle file
    
    Returns:
        Tuple of (texts, labels)
    """
    # Use /kaggle/working for cache on Kaggle (writable), otherwise use same directory
    if os.path.exists('/kaggle/input'):
        cache_dir = Path('/kaggle/working')
    else:
        cache_dir = data_file.parent
    
    cache_file = cache_dir / f"{data_file.stem}_processed.pkl"
    
    if use_cache and cache_file.exists():
        print(f"Loading cached data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data['texts'], data['labels']
    
    # Process raw data
    print(f"Processing data from {data_file}...")
    texts, labels = prepare_dataset(str(data_file), str(cache_file.with_suffix('')))
    
    return texts, labels


def train_crf(
    train_file: str = None,
    dev_file: str = None,
    lr: float = 0.01,
    max_iter: int = 50,
    l2_penalty: float = 0.1,
    save_path: str = None,
    save_features: bool = True
) -> Tuple[CRFModel, CRFFeatureExtractor]:
    """
    Train CRF model for Arabic diacritization.
    
    Args:
        train_file: Path to training data file
        dev_file: Path to dev data file
        lr: Learning rate
        max_iter: Maximum number of iterations
        l2_penalty: L2 regularization coefficient
        save_path: Path to save trained model
        save_features: Whether to save feature extractor
    
    Returns:
        Tuple of (trained CRFModel, CRFFeatureExtractor)
    """
    print("="*60)
    print("CRF MODEL TRAINING")
    print("="*60)
    
    # Use default paths if not provided
    if train_file is None:
        train_file = TRAIN_FILE
    if dev_file is None:
        dev_file = DEV_FILE
    if save_path is None:
        save_path = MODEL_DIR / 'crf_model.pkl'
    
    # Load data
    print("\n[1/5] Loading datasets...")
    start_time = time.time()
    
    train_texts, train_labels = load_or_prepare_data(Path(train_file))
    dev_texts, dev_labels = load_or_prepare_data(Path(dev_file))
    
    print(f"  Train: {len(train_texts)} sequences")
    print(f"  Dev: {len(dev_texts)} sequences")
    print(f"  Time: {time.time() - start_time:.2f}s")
    
    # Extract features
    print("\n[2/5] Extracting CRF features...")
    start_time = time.time()
    
    feature_extractor = CRFFeatureExtractor()
    train_features = feature_extractor.fit_transform(train_texts)
    dev_features = feature_extractor.transform(dev_texts)
    
    print(f"  Feature dimension: {feature_extractor.feature_count}")
    print(f"  Time: {time.time() - start_time:.2f}s")
    
    # Convert labels to numpy arrays
    train_labels_np = [np.array(labels, dtype=np.int32) for labels in train_labels]
    dev_labels_np = [np.array(labels, dtype=np.int32) for labels in dev_labels]
    
    # Initialize model
    print("\n[3/5] Initializing CRF model...")
    model = CRFModel(
        num_labels=NUM_DIACRITIC_CLASSES,
        feature_dim=feature_extractor.feature_count,
        lr=lr,
        max_iter=max_iter,
        l2_penalty=l2_penalty
    )
    
    # Train model
    print("\n[4/5] Training CRF...")
    start_time = time.time()
    model.fit(train_features, train_labels_np)
    print(f"  Training time: {time.time() - start_time:.2f}s")
    
    # Evaluate on dev set
    print("\n[5/5] Evaluating on dev set...")
    dev_metrics = model.evaluate(dev_features, dev_labels_np)
    
    print("\nDev Set Results:")
    print(f"  Accuracy: {dev_metrics['accuracy']:.4f}")
    print(f"  DER: {dev_metrics['der']:.4f}")
    print(f"  Correct: {dev_metrics['correct']:,} / {dev_metrics['total']:,}")
    
    # Save model
    print(f"\nSaving model to {save_path}...")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    
    # Save feature extractor
    if save_features:
        features_path = Path(save_path).parent / 'crf_features.pkl'
        with open(features_path, 'wb') as f:
            pickle.dump(feature_extractor, f)
        print(f"âœ“ Feature extractor saved to {features_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return model, feature_extractor


def predict_with_crf(
    model_path: str,
    features_path: str,
    texts: List[str]
) -> List[np.ndarray]:
    """
    Make predictions using trained CRF model.
    
    Args:
        model_path: Path to saved model
        features_path: Path to saved feature extractor
        texts: List of input texts (without diacritics)
    
    Returns:
        List of predicted label sequences
    """
    # Load model
    model = CRFModel(num_labels=NUM_DIACRITIC_CLASSES, feature_dim=0)
    model.load(model_path)
    
    # Load feature extractor
    with open(features_path, 'rb') as f:
        feature_extractor = pickle.load(f)
    
    # Extract features
    features = feature_extractor.transform(texts)
    
    # Predict
    predictions = model.predict(features)
    
    return predictions


def run_crf_training(
    train_file: str = None,
    dev_file: str = None,
    test_file: str = None,
    output_file: str = None
):
    """
    Complete training and evaluation pipeline for Kaggle notebooks.
    
    Args:
        train_file: Path to training data
        dev_file: Path to dev data
        test_file: Path to test data (optional)
        output_file: Path to save predictions (optional)
    """
    # Train model
    model, feature_extractor = train_crf(
        train_file=train_file,
        dev_file=dev_file,
        lr=0.01,
        max_iter=50,
        l2_penalty=0.1
    )
    
    # If test file provided, generate predictions
    if test_file is not None and output_file is not None:
        print("\nGenerating predictions for test set...")
        
        # Load test data
        test_sentences = load_dataset(Path(test_file))
        test_texts = [clean_arabic_text(sent) for sent in test_sentences]
        
        # Extract features
        test_features = feature_extractor.transform(test_texts)
        
        # Predict
        predictions = model.predict(test_features)
        
        # Apply diacritics to create diacritized text
        diacritized_texts = []
        for text, pred_labels in zip(test_texts, predictions):
            diacritized = model.apply_diacritics(text, pred_labels)
            diacritized_texts.append(diacritized)
        
        # Save to CSV
        import csv
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'diacritized_text'])
            for idx, text in enumerate(diacritized_texts):
                writer.writerow([idx, text])
        
        print(f"âœ“ Predictions saved to {output_file}")


if __name__ == '__main__':
    # Run training
    run_crf_training()

