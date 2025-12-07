"""
FULL TRAINING SCRIPT - Train on Complete 50K Dataset
Optimized for best accuracy with reasonable training time
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
    """Train Logistic Regression on FULL 50K training set"""
    print("="*80)
    print(" FULL TRAINING - Logistic Regression (50K sentences)")
    print("="*80)
    print("\n⚠️  IMPORTANT NOTES:")
    print("  - This will train on ALL 50,000 training sentences")
    print("  - Estimated time: 20-40 minutes (depending on hardware)")
    print("  - Memory usage: ~10-15 GB RAM")
    print("  - Expected accuracy: 60-70% on dev set")
    print("\n")
    
    # Load dataset
    print("[1/4] Loading full dataset...")
    train_texts, train_labels = load_or_prepare_data(Path(TRAIN_FILE))
    dev_texts, dev_labels = load_or_prepare_data(Path(DEV_FILE))
    
    print(f"   Train: {len(train_texts):,} sentences")
    print(f"   Dev: {len(dev_texts):,} sentences")
    
    # Configuration
    config = {
        'learning_rate': 0.1,
        'max_iter': 100,
        'regularization': 1e-3,
        'max_features': 5000,      # Increased for better accuracy
        'ngram_range': (1, 2),
        'batch_size': 1024,
        'window_size': 5
    }
    
    print("\n[2/4] Training Configuration:")
    print("-"*80)
    for key, value in config.items():
        print(f"   {key:20s}: {value}")
    print("-"*80)
    
    # Initialize model
    print("\n[3/4] Training model...")
    print("   This will take approximately 20-40 minutes...")
    print("   Progress will be shown every 10 epochs.")
    print()
    
    model = LogisticRegressionModel(
        learning_rate=config['learning_rate'],
        max_iter=config['max_iter'],
        regularization=config['regularization'],
        max_features=config['max_features'],
        ngram_range=config['ngram_range'],
        batch_size=config['batch_size']
    )
    
    start_time = time.time()
    
    # Use streaming training for memory efficiency
    model.fit(
        train_texts, 
        train_labels, 
        window_size=config['window_size'],
        streaming=True,      # Enable streaming for large dataset
        chunk_size=1000      # Process 1K sentences at a time
    )
    
    training_time = time.time() - start_time
    print(f"\n   ✓ Training completed in {training_time/60:.2f} minutes")
    
    # Evaluate on dev set
    print("\n[4/4] Evaluating on dev set...")
    dev_metrics = model.evaluate(dev_texts, dev_labels, window_size=config['window_size'])
    
    print("\n" + "="*80)
    print(" FINAL RESULTS:")
    print("="*80)
    print(f"   Dataset: 50,000 training sentences")
    print(f"   Training Time: {training_time/60:.2f} minutes")
    print(f"   ")
    print(f"   Dev Set Performance:")
    print(f"   -------------------")
    print(f"   Accuracy: {dev_metrics['accuracy']:.4f} ({dev_metrics['accuracy']*100:.2f}%)")
    print(f"   DER:      {dev_metrics['der']:.4f}")
    print(f"   Correct:  {dev_metrics['correct']:,} / {dev_metrics['total']:,}")
    print("="*80)
    
    # Save model
    save_path = MODEL_DIR / 'logreg_full_50k.pkl'
    print(f"\nSaving model to {save_path}...")
    model.save(str(save_path))
    print(f"✓ Model saved successfully!")
    
    # Test predictions
    print("\n" + "="*80)
    print(" SAMPLE PREDICTIONS:")
    print("="*80)
    
    for i in range(min(5, len(dev_texts))):
        test_text = dev_texts[i]
        print(f"\nExample {i+1}:")
        print(f"  Original (no diacritics):")
        print(f"    {test_text[:80]}...")
        
        predictions = model.predict([test_text], window_size=config['window_size'])
        diacritized = model.apply_diacritics(test_text, predictions[0])
        print(f"  Predicted (with diacritics):")
        print(f"    {diacritized[:80]}...")
    
    print("\n" + "="*80)
    print(" ✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nModel location: {save_path}")
    print("\nNext steps:")
    print("  1. Evaluate this model: python evaluate_and_infer.py")
    print("  2. Train CRF for comparison: python train_crf_medium.py")
    print("  3. Generate test predictions for Kaggle submission")
    print()

if __name__ == "__main__":
    try:
        # Confirm before starting
        response = input("\nProceed with full training? This will take 20-40 minutes. (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Training cancelled.")
            exit()
        
        print("\n🚀 Starting training...\n")
        main()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
