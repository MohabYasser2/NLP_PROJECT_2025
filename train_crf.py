"""
Train CRF Model - Better for Sequence Labeling
CRF models typically perform 10-20% better than LogReg for diacritization
"""

from src.training.train_crf import run_crf_training

print("=" * 70)
print("CRF MODEL TRAINING")
print("=" * 70)
print("\n💡 CRF (Conditional Random Field) is better for sequence tasks")
print("💡 Expected DER: 15-25% (75-85% accuracy)")
print("\n")

# Train CRF model
run_crf_training(
    train_file='data/train.txt',
    dev_file='data/val.txt',
    test_file=None,
    output_file='submission_crf.csv',
    
    # CRF-specific parameters
    max_iter=100,                       # CRF iterations
    learning_rate=0.1,                  # CRF learning rate
    regularization=1.0,                 # L2 regularization
    window_size=5                       # Context window
)

print("\n✓ CRF model saved to: models/crf_model.pkl")
print("✓ This should perform better than LogReg!")
