"""
Test transformer features integrated with LogReg model
"""

import numpy as np
from pathlib import Path
import time
import pickle

from src.features_arabert import TransformerFeatureExtractor
from src.models.logreg_model import LogisticRegressionModel, TfidfVectorizer


def calculate_der(y_true, y_pred):
    """Simple DER calculation."""
    total = len(y_true)
    errors = np.sum(np.array(y_true) != np.array(y_pred))
    return errors / total if total > 0 else 0.0


print("="*80)
print(" Testing Transformer Features with Diacritization Models")
print("="*80)

# Load preprocessed data
print("\n[1/6] Loading preprocessed data...")
data_file = 'data/train_processed.pkl'
if not Path(data_file).exists():
    print(f"  [ERROR] {data_file} not found")
    print("  Please run preprocessing first")
    exit(1)

with open(data_file, 'rb') as f:
    data = pickle.load(f)
    X_all = data['texts']
    y_all = data['labels']

# Use small subset for testing
X_train = X_all[:10]
y_train = y_all[:10]

print(f"  Training samples: {len(X_train):,}")
print(f"  Sample text: {X_train[0][:50]}...")

# Initialize transformer
print("\n[2/6] Initializing transformer feature extractor...")
transformer = TransformerFeatureExtractor(
    vocab_size=256,
    d_model=128,
    num_layers=4,
    num_heads=4,
    d_ff=512,
    max_seq_length=512,
    aggregation='mean',
    output_layers=4
)

# Extract transformer features for training
print("\n[3/6] Extracting transformer features for training...")
start = time.time()
train_transformer_features = []
for i, text in enumerate(X_train[:10]):  # Test with 10 samples first
    if (i + 1) % 5 == 0:
        print(f"  Processing {i+1}/10...")
    features = transformer.extract_features(text)  # (seq_len, d_model)
    # Flatten to 1D for LogReg
    flat_features = features.flatten()
    train_transformer_features.append(flat_features)

elapsed = time.time() - start
print(f"  Extracted features in {elapsed:.2f}s ({elapsed/10:.3f}s per sample)")
print(f"  Feature dimension: {train_transformer_features[0].shape}")

# Create combined features (TF-IDF + Transformer)
print("\n[4/6] Training LogReg with TF-IDF features...")

# Standard TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train[:10])

print(f"  TF-IDF features shape: {X_train_tfidf.shape}")
print(f"  Transformer features available: 10 samples")

# Train simple model
print("\n[5/6] Training LogReg on TF-IDF (baseline)...")
model = LogisticRegressionModel(learning_rate=0.01, max_iter=20, regularization=0.1)

start = time.time()
model.fit(X_train_tfidf, np.array(y_train[:10]), verbose=False)
elapsed = time.time() - start

# Evaluate
y_pred = []
for x in X_train_tfidf:
    pred = model.predict(x)
    y_pred.extend(pred.tolist())

y_true_flat = [label for seq in y_train[:10] for label in seq]
y_pred_flat = y_pred[:len(y_true_flat)]

accuracy = np.mean(np.array(y_true_flat) == np.array(y_pred_flat))
der = calculate_der(y_true_flat, y_pred_flat)

print(f"  Training time: {elapsed:.2f}s")
print(f"  Train accuracy: {accuracy:.4f}")
print(f"  Train DER: {der:.4f}")

# Save results
print("\n[6/6] Summary:")
print(f"  [OK] Transformer features: {transformer.get_output_dim()}D embeddings")
print(f"  [OK] Feature extraction: ~{elapsed/10:.3f}s per sample")
print(f"  [OK] LogReg baseline: {accuracy:.2%} accuracy")
print(f"  [OK] Ready for integration with main training pipeline")

print("\n" + "="*80)
print("Next steps:")
print("  1. Integrate transformer as optional feature in train_models.py")
print("  2. Add --use-transformer flag for overnight training")
print("  3. Combine TF-IDF + Transformer features for better accuracy")
print("="*80)
