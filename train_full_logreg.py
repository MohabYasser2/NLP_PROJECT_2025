"""
Train Logistic Regression with OPTIMIZED PARAMETERS
This uses the full dataset (50k samples) with aggressive optimization for best results
Expected to run for 30-60 minutes for maximum accuracy
"""

from src.training.train_logreg import run_logreg_training
import time

print("=" * 70)
print("OPTIMIZED LOGISTIC REGRESSION TRAINING")
print("=" * 70)
print("\n🚀 AGGRESSIVE OPTIMIZATION MODE")
print("   • Full 50k training samples")
print("   • 20,000 features (maximum vocabulary)")
print("   • Trigram support (1-3 grams)")
print("   • Large context window (±3 characters)")
print("   • 500 training iterations")
print("   • Small batch size for better convergence")
print("\n⏱️  Expected time: 30-60 minutes")
print("💾 Memory required: ~10-15 GB RAM")
print("\n🎯 Target DER: 20-30% (70-80% accuracy)")
print("=" * 70)
print("\n")

start_time = time.time()

# Train with OPTIMIZED parameters for best accuracy
run_logreg_training(
    train_file='data/train.txt',        # Full 50k samples
    dev_file='data/val.txt',
    test_file=None,                     # No test file yet
    output_file='submission.csv',
    
    # AGGRESSIVE FEATURE EXTRACTION
    max_features=20000,                 # 20x more features (maximum vocab)
    ngram_range=(1, 3),                 # Full trigram support
    window_size=7,                      # Large context: ±3 characters
    
    # OPTIMIZED TRAINING
    max_iter=500,                       # Maximum iterations for convergence
    batch_size=32,                      # Small batches = better gradient estimates
    learning_rate=0.01,                 # Standard rate (stable)
    regularization=0.005                # Light regularization (less constraint)
)

elapsed_time = time.time() - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"\n⏱️  Total training time: {minutes}m {seconds}s")
print("\n📊 Expected Results:")
print("   • Your ultra-light model: 64.71% DER")
print("   • This optimized model: 20-30% DER (70-80% accuracy)")
print("   • Improvement: ~35-45% reduction in error rate")
print("\n✓ Model saved to: models/logreg_model.pkl")
print("✓ You can now:")
print("   1. Upload this model to GitHub (git add/commit/push)")
print("   2. Use it in Kaggle notebook (loads from repository)")
print("   3. Generate test predictions when test.txt is released")
print("\n💡 For even better results, try the CRF model: python train_crf.py")
print("=" * 70)
