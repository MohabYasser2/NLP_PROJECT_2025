"""
COMPREHENSIVE TRAINING SCRIPT
Train both Logistic Regression and CRF models with optimal settings
"""

import sys
from pathlib import Path
import time
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.train_logreg import load_or_prepare_data
from src.models.logreg_model import LogisticRegressionModel
from src.models.crf_model import CRFModel, CRFFeatureExtractor
from src.config import TRAIN_FILE, DEV_FILE, MODEL_DIR, NUM_DIACRITIC_CLASSES
import numpy as np


def train_logistic_regression(num_sentences=None, use_streaming=True):
    """
    Train Logistic Regression model
    
    Args:
        num_sentences: Number of training sentences (None = all)
        use_streaming: Use streaming training for memory efficiency
    """
    print("="*80)
    print(" TRAINING LOGISTIC REGRESSION")
    print("="*80)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    train_texts, train_labels = load_or_prepare_data(Path(TRAIN_FILE))
    dev_texts, dev_labels = load_or_prepare_data(Path(DEV_FILE))
    
    # Subset if specified
    if num_sentences is not None:
        train_texts = train_texts[:num_sentences]
        train_labels = train_labels[:num_sentences]
    
    print(f"   Train: {len(train_texts):,} sentences")
    print(f"   Dev: {len(dev_texts):,} sentences")
    
    # Configuration
    config = {
        'learning_rate': 0.1,
        'max_iter': 100,
        'regularization': 1e-3,
        'max_features': 5000,
        'ngram_range': (1, 2),
        'batch_size': 1024,
        'window_size': 5
    }
    
    print("\n[2/4] Configuration:")
    for key, value in config.items():
        print(f"   {key:20s}: {value}")
    
    # Initialize and train
    print("\n[3/4] Training...")
    model = LogisticRegressionModel(
        learning_rate=config['learning_rate'],
        max_iter=config['max_iter'],
        regularization=config['regularization'],
        max_features=config['max_features'],
        ngram_range=config['ngram_range'],
        batch_size=config['batch_size']
    )
    
    start_time = time.time()
    model.fit(
        train_texts, 
        train_labels, 
        window_size=config['window_size'],
        streaming=use_streaming,
        chunk_size=1000
    )
    training_time = time.time() - start_time
    
    # Evaluate
    print("\n[4/4] Evaluating...")
    dev_metrics = model.evaluate(dev_texts, dev_labels, window_size=config['window_size'])
    
    print("\n" + "="*80)
    print(" LOGISTIC REGRESSION RESULTS:")
    print("="*80)
    print(f"   Training sentences: {len(train_texts):,}")
    print(f"   Training time: {training_time/60:.2f} minutes")
    print(f"   ")
    print(f"   Dev Set Performance:")
    print(f"   Accuracy: {dev_metrics['accuracy']:.4f} ({dev_metrics['accuracy']*100:.2f}%)")
    print(f"   DER:      {dev_metrics['der']:.4f}")
    print(f"   Correct:  {dev_metrics['correct']:,} / {dev_metrics['total']:,}")
    print("="*80)
    
    # Save
    model_name = f"logreg_{len(train_texts)}.pkl"
    save_path = MODEL_DIR / model_name
    model.save(str(save_path))
    print(f"\n[OK] Model saved to: {save_path}")
    
    return model, dev_metrics


def train_crf(num_sentences=None):
    """
    Train CRF model
    
    Args:
        num_sentences: Number of training sentences (None = all, recommended: 1000-5000)
    """
    print("\n\n" + "="*80)
    print(" TRAINING CRF")
    print("="*80)
    print("\nNote: CRF is much slower than LogReg. Recommended: 1000-5000 sentences")
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    train_texts, train_labels = load_or_prepare_data(Path(TRAIN_FILE))
    dev_texts, dev_labels = load_or_prepare_data(Path(DEV_FILE))
    
    # Subset (CRF is slow, use smaller dataset)
    if num_sentences is None:
        num_sentences = 1000
    train_texts = train_texts[:num_sentences]
    train_labels = train_labels[:num_sentences]
    dev_texts = dev_texts[:200]  # Subset dev for faster evaluation
    dev_labels = dev_labels[:200]
    
    print(f"   Train: {len(train_texts):,} sentences")
    print(f"   Dev: {len(dev_texts):,} sentences")
    
    # Extract features
    print("\n[2/5] Extracting CRF features...")
    start_time = time.time()
    
    feature_extractor = CRFFeatureExtractor()
    train_features = feature_extractor.fit_transform(train_texts)
    dev_features = feature_extractor.transform(dev_texts)
    
    print(f"   Feature dimension: {feature_extractor.feature_count:,}")
    print(f"   Time: {time.time() - start_time:.2f}s")
    
    # Convert labels
    train_labels_np = [np.array(labels, dtype=np.int32) for labels in train_labels]
    dev_labels_np = [np.array(labels, dtype=np.int32) for labels in dev_labels]
    
    # Initialize model
    print("\n[3/5] Initializing CRF...")
    model = CRFModel(
        num_labels=NUM_DIACRITIC_CLASSES,
        feature_dim=feature_extractor.feature_count,
        lr=0.01,
        max_iter=30,
        l2_penalty=0.1
    )
    
    # Train
    print("\n[4/5] Training CRF...")
    print("   This may take 10-30 minutes depending on dataset size...")
    start_time = time.time()
    model.fit(train_features, train_labels_np)
    training_time = time.time() - start_time
    
    # Evaluate
    print("\n[5/5] Evaluating...")
    dev_metrics = model.evaluate(dev_features, dev_labels_np)
    
    print("\n" + "="*80)
    print(" CRF RESULTS:")
    print("="*80)
    print(f"   Training sentences: {len(train_texts):,}")
    print(f"   Training time: {training_time/60:.2f} minutes")
    print(f"   ")
    print(f"   Dev Set Performance:")
    print(f"   Accuracy: {dev_metrics['accuracy']:.4f} ({dev_metrics['accuracy']*100:.2f}%)")
    print(f"   DER:      {dev_metrics['der']:.4f}")
    print(f"   Correct:  {dev_metrics['correct']:,} / {dev_metrics['total']:,}")
    print("="*80)
    
    # Save
    model_name = f"crf_{len(train_texts)}.pkl"
    features_name = f"crf_features_{len(train_texts)}.pkl"
    
    save_path = MODEL_DIR / model_name
    features_path = MODEL_DIR / features_name
    
    model.save(str(save_path))
    
    import pickle
    with open(features_path, 'wb') as f:
        pickle.dump(feature_extractor, f)
    
    print(f"\n[OK] Model saved to: {save_path}")
    print(f"[OK] Features saved to: {features_path}")
    
    return model, feature_extractor, dev_metrics


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Arabic Diacritization Models')
    parser.add_argument('--model', type=str, choices=['logreg', 'crf', 'both'], default='both',
                      help='Which model to train (default: both)')
    parser.add_argument('--logreg-sentences', type=int, default=None,
                      help='Number of sentences for LogReg (default: all 50K)')
    parser.add_argument('--crf-sentences', type=int, default=1000,
                      help='Number of sentences for CRF (default: 1000)')
    parser.add_argument('--quick', action='store_true',
                      help='Quick mode: LogReg=5K, CRF=500 sentences')
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.logreg_sentences = 5000
        args.crf_sentences = 500
        print("\n[QUICK MODE] Training with reduced dataset sizes")
    
    print("="*80)
    print(" ARABIC DIACRITIZATION - COMPREHENSIVE TRAINING")
    print("="*80)
    print(f"\nModels to train: {args.model.upper()}")
    if args.model in ['logreg', 'both']:
        sent_text = f"{args.logreg_sentences:,}" if args.logreg_sentences else "ALL (50K)"
        print(f"  LogReg: {sent_text} sentences")
    if args.model in ['crf', 'both']:
        print(f"  CRF: {args.crf_sentences:,} sentences")
    
    results = {}
    
    # Train Logistic Regression
    if args.model in ['logreg', 'both']:
        try:
            logreg_model, logreg_metrics = train_logistic_regression(
                num_sentences=args.logreg_sentences,
                use_streaming=True
            )
            results['logreg'] = logreg_metrics
        except Exception as e:
            print(f"\n[ERROR] LogReg training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Train CRF
    if args.model in ['crf', 'both']:
        try:
            crf_model, crf_features, crf_metrics = train_crf(
                num_sentences=args.crf_sentences
            )
            results['crf'] = crf_metrics
        except Exception as e:
            print(f"\n[ERROR] CRF training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n\n" + "="*80)
    print(" TRAINING SUMMARY")
    print("="*80)
    
    if 'logreg' in results:
        print("\nLogistic Regression:")
        print(f"  Accuracy: {results['logreg']['accuracy']:.4f} ({results['logreg']['accuracy']*100:.2f}%)")
        print(f"  DER: {results['logreg']['der']:.4f}")
    
    if 'crf' in results:
        print("\nCRF:")
        print(f"  Accuracy: {results['crf']['accuracy']:.4f} ({results['crf']['accuracy']*100:.2f}%)")
        print(f"  DER: {results['crf']['der']:.4f}")
    
    if len(results) == 2:
        logreg_acc = results['logreg']['accuracy']
        crf_acc = results['crf']['accuracy']
        better = 'CRF' if crf_acc > logreg_acc else 'LogReg'
        diff = abs(crf_acc - logreg_acc) * 100
        print(f"\n{better} performed better by {diff:.2f}%")
    
    print("\n" + "="*80)
    print(" [OK] TRAINING COMPLETED!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Evaluate models: python evaluate_and_infer.py")
    print("  2. Generate test predictions for Kaggle submission")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
