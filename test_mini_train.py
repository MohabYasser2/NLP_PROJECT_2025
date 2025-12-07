"""
Mini Training Script - Test on small subset
Tests the training pipeline on just 100 sentences
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.train_logreg import load_or_prepare_data
from src.models.logreg_model import LogisticRegressionModel
from src.config import TRAIN_FILE, DEV_FILE, MODEL_DIR

def main():
    """Train on a tiny subset to verify everything works"""
    print("="*70)
    print(" MINI TEST - Logistic Regression (100 sentences)")
    print("="*70)
    
    # Load small subset
    print("\n[1/4] Loading tiny dataset...")
    train_texts, train_labels = load_or_prepare_data(Path(TRAIN_FILE), use_cache=False)
    dev_texts, dev_labels = load_or_prepare_data(Path(DEV_FILE), use_cache=False)
    
    # Take only first 100 train and 20 dev
    train_texts = train_texts[:100]
    train_labels = train_labels[:100]
    dev_texts = dev_texts[:20]
    dev_labels = dev_labels[:20]
    
    print(f"   Train: {len(train_texts)} sentences")
    print(f"   Dev: {len(dev_texts)} sentences")
    
    # Initialize model
    print("\n[2/4] Initializing model...")
    model = LogisticRegressionModel(
        learning_rate=0.1,
        max_iter=10,              # Just 10 iterations for quick test
        regularization=1e-3,
        max_features=1000,        # Reduced features
        ngram_range=(1, 2),
        batch_size=64
    )
    
    # Train model
    print("\n[3/4] Training model...")
    start_time = time.time()
    model.fit(train_texts, train_labels, window_size=5, streaming=False)
    print(f"   Training time: {time.time() - start_time:.2f}s")
    
    # Evaluate on dev set
    print("\n[4/4] Evaluating on dev set...")
    dev_metrics = model.evaluate(dev_texts, dev_labels, window_size=5)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("-"*70)
    print(f"   Accuracy: {dev_metrics['accuracy']:.4f} ({dev_metrics['accuracy']*100:.2f}%)")
    print(f"   DER:      {dev_metrics['der']:.4f}")
    print(f"   Correct:  {dev_metrics['correct']:,} / {dev_metrics['total']:,}")
    print("="*70)
    
    # Test prediction
    print("\nTesting prediction on first dev sentence:")
    print("-"*70)
    test_text = dev_texts[0]
    print(f"Input (no diacritics): {test_text[:80]}...")
    
    predictions = model.predict([test_text], window_size=5)
    diacritized = model.apply_diacritics(test_text, predictions[0])
    print(f"Output (with predicted diacritics): {diacritized[:80]}...")
    
    print("\n✓ Mini test completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
