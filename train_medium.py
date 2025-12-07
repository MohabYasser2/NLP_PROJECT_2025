"""
Training Script - Medium Dataset (5000 sentences)
Trains on 5K sentences for reasonable results in ~10 minutes
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
    """Train on 5000 sentences"""
    print("="*70)
    print(" MEDIUM TRAINING - Logistic Regression (5K sentences)")
    print("="*70)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    train_texts, train_labels = load_or_prepare_data(Path(TRAIN_FILE))
    dev_texts, dev_labels = load_or_prepare_data(Path(DEV_FILE))
    
    # Use 5K training sentences
    train_texts = train_texts[:5000]
    train_labels = train_labels[:5000]
    
    print(f"   Train: {len(train_texts):,} sentences")
    print(f"   Dev: {len(dev_texts):,} sentences")
    
    # Initialize model
    print("\n[2/4] Initializing model...")
    model = LogisticRegressionModel(
        learning_rate=0.1,
        max_iter=50,              # 50 iterations for better learning
        regularization=1e-3,
        max_features=3000,        # 3K features for balance
        ngram_range=(1, 2),       # Unigrams and bigrams
        batch_size=512
    )
    
    # Train model
    print("\n[3/4] Training model...")
    print("   This will take approximately 5-10 minutes...")
    start_time = time.time()
    model.fit(train_texts, train_labels, window_size=5, streaming=False)
    training_time = time.time() - start_time
    print(f"   ✓ Training time: {training_time/60:.2f} minutes")
    
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
    
    # Save model
    save_path = MODEL_DIR / 'logreg_5k.pkl'
    print(f"\nSaving model to {save_path}...")
    model.save(str(save_path))
    
    # Test prediction on a few examples
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS:")
    print("-"*70)
    for i in range(min(3, len(dev_texts))):
        test_text = dev_texts[i]
        print(f"\nExample {i+1}:")
        print(f"  Input:  {test_text[:60]}...")
        
        predictions = model.predict([test_text], window_size=5)
        diacritized = model.apply_diacritics(test_text, predictions[0])
        print(f"  Output: {diacritized[:60]}...")
    
    print("\n✓ Training completed successfully!")
    print(f"  Model saved to: {save_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
