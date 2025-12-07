"""
Quick Training Script for Logistic Regression Model
Trains on a smaller subset for quick testing
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.train_logreg import train_logreg
from src.config import TRAIN_FILE, DEV_FILE, MODEL_DIR

def main():
    """Train Logistic Regression model with optimized parameters"""
    print("="*70)
    print(" ARABIC DIACRITIZATION - LOGISTIC REGRESSION TRAINING")
    print("="*70)
    
    # Training parameters - optimized for speed and memory
    params = {
        'train_file': str(TRAIN_FILE),
        'dev_file': str(DEV_FILE),
        'window_size': 5,           # Context window around each character
        'learning_rate': 0.1,        # Faster convergence
        'max_iter': 100,             # 100 epochs
        'regularization': 1e-3,      # L2 regularization
        'max_features': 3000,        # Reduced for memory efficiency
        'ngram_range': (1, 2),       # Unigrams and bigrams
        'batch_size': 1024,          # Large batches for efficiency
        'save_path': str(MODEL_DIR / 'logreg_model.pkl')
    }
    
    print("\nTraining Configuration:")
    print("-" * 70)
    for key, value in params.items():
        print(f"  {key:20s}: {value}")
    print("-" * 70)
    
    # Confirm training
    print("\n⚠️  This will train on the FULL training set (50k sentences)")
    print("    Estimated time: 30-60 minutes depending on your hardware")
    print("    Memory usage: ~15-20 GB RAM")
    
    response = input("\nProceed with training? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Training cancelled.")
        return
    
    # Start training
    start_time = time.time()
    
    try:
        model = train_logreg(**params)
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✓ TRAINING COMPLETED SUCCESSFULLY")
        print(f"  Total time: {total_time/60:.2f} minutes")
        print(f"  Model saved to: {params['save_path']}")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
