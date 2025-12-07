"""
Training Script for CRF Model - Medium Dataset
Trains CRF on 1000 sentences (CRF is much slower than LogReg)
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.train_crf import load_or_prepare_data
from src.models.crf_model import CRFModel, CRFFeatureExtractor
from src.config import TRAIN_FILE, DEV_FILE, MODEL_DIR, NUM_DIACRITIC_CLASSES
import numpy as np

def main():
    """Train CRF on 1000 sentences"""
    print("="*70)
    print(" CRF TRAINING - Medium Dataset (1K sentences)")
    print("="*70)
    print("\nNote: CRF training is much slower than LogReg due to")
    print("      forward-backward algorithm complexity.")
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    train_texts, train_labels = load_or_prepare_data(Path(TRAIN_FILE))
    dev_texts, dev_labels = load_or_prepare_data(Path(DEV_FILE))
    
    # Use only 1K training sentences (CRF is slow)
    train_texts = train_texts[:1000]
    train_labels = train_labels[:1000]
    # Use subset of dev for faster evaluation
    dev_texts = dev_texts[:200]
    dev_labels = dev_labels[:200]
    
    print(f"   Train: {len(train_texts):,} sentences")
    print(f"   Dev: {len(dev_texts):,} sentences")
    
    # Extract features
    print("\n[2/5] Extracting CRF features...")
    print("   This extracts character-level features with context windows...")
    start_time = time.time()
    
    feature_extractor = CRFFeatureExtractor()
    train_features = feature_extractor.fit_transform(train_texts)
    dev_features = feature_extractor.transform(dev_texts)
    
    print(f"   ✓ Feature dimension: {feature_extractor.feature_count:,}")
    print(f"   Time: {time.time() - start_time:.2f}s")
    
    # Convert labels to numpy arrays
    train_labels_np = [np.array(labels, dtype=np.int32) for labels in train_labels]
    dev_labels_np = [np.array(labels, dtype=np.int32) for labels in dev_labels]
    
    # Initialize model
    print("\n[3/5] Initializing CRF model...")
    model = CRFModel(
        num_labels=NUM_DIACRITIC_CLASSES,
        feature_dim=feature_extractor.feature_count,
        lr=0.01,              # Learning rate
        max_iter=30,          # Reduced iterations for speed
        l2_penalty=0.1        # L2 regularization
    )
    
    # Train model
    print("\n[4/5] Training CRF...")
    print("   This will take approximately 10-20 minutes...")
    print("   (CRF uses forward-backward algorithm which is O(T*L^2))")
    start_time = time.time()
    model.fit(train_features, train_labels_np)
    training_time = time.time() - start_time
    print(f"   ✓ Training time: {training_time/60:.2f} minutes")
    
    # Evaluate on dev set
    print("\n[5/5] Evaluating on dev set...")
    dev_metrics = model.evaluate(dev_features, dev_labels_np)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("-"*70)
    print(f"   Accuracy: {dev_metrics['accuracy']:.4f} ({dev_metrics['accuracy']*100:.2f}%)")
    print(f"   DER:      {dev_metrics['der']:.4f}")
    print(f"   Correct:  {dev_metrics['correct']:,} / {dev_metrics['total']:,}")
    print("="*70)
    
    # Save model
    save_path = MODEL_DIR / 'crf_1k.pkl'
    features_path = MODEL_DIR / 'crf_features_1k.pkl'
    
    print(f"\nSaving model...")
    model.save(str(save_path))
    
    import pickle
    with open(features_path, 'wb') as f:
        pickle.dump(feature_extractor, f)
    print(f"   ✓ Model saved to: {save_path}")
    print(f"   ✓ Features saved to: {features_path}")
    
    # Test prediction on a few examples
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS:")
    print("-"*70)
    for i in range(min(3, len(dev_texts))):
        test_text = dev_texts[i]
        print(f"\nExample {i+1}:")
        print(f"  Input:  {test_text[:60]}...")
        
        test_features = feature_extractor.transform([test_text])
        predictions = model.predict(test_features)
        diacritized = model.apply_diacritics(test_text, predictions[0])
        print(f"  Output: {diacritized[:60]}...")
    
    print("\n✓ Training completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
