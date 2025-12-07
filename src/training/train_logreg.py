"""
Training Module for Logistic Regression
Provides end-to-end training pipeline for logistic regression diacritization model.
"""

import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import pickle
import time

from src.config import TRAIN_FILE, DEV_FILE, MODEL_DIR, NUM_DIACRITIC_CLASSES
from src.preprocessing import prepare_dataset, load_dataset, clean_arabic_text, extract_labels_simple
from src.models.logreg_model import LogisticRegressionModel


def load_or_prepare_data(data_file: Path, use_cache: bool = True) -> Tuple[List[str], List[List[int]]]:
    """
    Load or prepare dataset.
    
    Args:
        data_file: Path to data file
        use_cache: Whether to use cached pickle file
    
    Returns:
        Tuple of (texts, labels)
    """
    # Check for cached pickle file
    cache_file = data_file.parent / f"{data_file.stem}_processed.pkl"
    
    if use_cache and cache_file.exists():
        print(f"Loading cached data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data['texts'], data['labels']
    
    # Process raw data
    print(f"Processing data from {data_file}...")
    texts, labels = prepare_dataset(str(data_file), str(cache_file.with_suffix('')))
    
    return texts, labels


def train_logreg(
    train_file: str = None,
    dev_file: str = None,
    window_size: int = 5,
    learning_rate: float = 0.01,
    max_iter: int = 500,
    regularization: float = 0.01,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 3),
    batch_size: int = 64,
    save_path: str = None
) -> LogisticRegressionModel:
    """
    Train logistic regression model for Arabic diacritization.
    
    Args:
        train_file: Path to training data file
        dev_file: Path to dev data file
        window_size: Context window size
        learning_rate: Learning rate for gradient descent
        max_iter: Maximum number of iterations
        regularization: L2 regularization coefficient
        max_features: Maximum number of TF-IDF features
        ngram_range: N-gram range for TF-IDF
        batch_size: Mini-batch size
        save_path: Path to save trained model
    
    Returns:
        Trained LogisticRegressionModel
    """
    print("="*60)
    print("LOGISTIC REGRESSION TRAINING")
    print("="*60)
    
    # Use default paths if not provided
    if train_file is None:
        train_file = TRAIN_FILE
    if dev_file is None:
        dev_file = DEV_FILE
    if save_path is None:
        save_path = MODEL_DIR / 'logreg_model.pkl'
    
    # Load data
    print("\n[1/4] Loading datasets...")
    start_time = time.time()
    
    train_texts, train_labels = load_or_prepare_data(Path(train_file))
    dev_texts, dev_labels = load_or_prepare_data(Path(dev_file))
    
    print(f"  Train: {len(train_texts)} sequences")
    print(f"  Dev: {len(dev_texts)} sequences")
    print(f"  Time: {time.time() - start_time:.2f}s")
    
    # Initialize model
    print("\n[2/4] Initializing model...")
    model = LogisticRegressionModel(
        learning_rate=learning_rate,
        max_iter=max_iter,
        regularization=regularization,
        max_features=max_features,
        ngram_range=ngram_range,
        batch_size=batch_size
    )
    
    # Train model
    print("\n[3/4] Training model...")
    start_time = time.time()
    model.fit(train_texts, train_labels, window_size=window_size)
    print(f"  Training time: {time.time() - start_time:.2f}s")
    
    # Evaluate on dev set
    print("\n[4/4] Evaluating on dev set...")
    dev_metrics = model.evaluate(dev_texts, dev_labels, window_size=window_size)
    
    print("\nDev Set Results:")
    print(f"  Accuracy: {dev_metrics['accuracy']:.4f}")
    print(f"  DER: {dev_metrics['der']:.4f}")
    print(f"  Correct: {dev_metrics['correct']:,} / {dev_metrics['total']:,}")
    
    # Save model
    print(f"\nSaving model to {save_path}...")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return model


def predict_with_logreg(
    model_path: str,
    texts: List[str],
    window_size: int = 5
) -> List[List[int]]:
    """
    Make predictions using trained logistic regression model.
    
    Args:
        model_path: Path to saved model
        texts: List of input texts (without diacritics)
        window_size: Context window size
    
    Returns:
        List of predicted label sequences
    """
    # Load model
    model = LogisticRegressionModel()
    model.load(model_path)
    
    # Predict
    predictions = model.predict(texts, window_size=window_size)
    
    return predictions


def run_logreg_training(
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
    model = train_logreg(
        train_file=train_file,
        dev_file=dev_file,
        window_size=5,
        learning_rate=0.01,
        max_iter=500,
        regularization=0.01,
        max_features=5000,
        ngram_range=(1, 3),
        batch_size=64
    )
    
    # If test file provided, generate predictions
    if test_file is not None and output_file is not None:
        print("\nGenerating predictions for test set...")
        
        # Load test data
        test_sentences = load_dataset(Path(test_file))
        test_texts = [clean_arabic_text(sent) for sent in test_sentences]
        
        # Predict
        predictions = model.predict(test_texts, window_size=5)
        
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
        
        print(f"✓ Predictions saved to {output_file}")


if __name__ == '__main__':
    # Run training
    run_logreg_training()
