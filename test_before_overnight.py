"""
QUICK TEST - Verify both models work before overnight training
Tests LogReg and CRF on tiny datasets (2-3 minutes total)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.train_logreg import load_or_prepare_data
from src.models.logreg_model import LogisticRegressionModel
from src.models.crf_model import CRFModel, CRFFeatureExtractor
from src.config import TRAIN_FILE, DEV_FILE, MODEL_DIR, NUM_DIACRITIC_CLASSES
import numpy as np
import time

def test_logistic_regression():
    """Test LogReg on 200 sentences"""
    print("="*80)
    print(" TEST 1: LOGISTIC REGRESSION")
    print("="*80)
    
    print("\n[1/3] Loading 200 training sentences...")
    train_texts, train_labels = load_or_prepare_data(Path(TRAIN_FILE))
    dev_texts, dev_labels = load_or_prepare_data(Path(DEV_FILE))
    
    train_texts = train_texts[:200]
    train_labels = train_labels[:200]
    dev_texts = dev_texts[:50]
    dev_labels = dev_labels[:50]
    
    print(f"   Train: {len(train_texts)} sentences")
    print(f"   Dev: {len(dev_texts)} sentences")
    
    print("\n[2/3] Training LogReg (max_iter=20 for speed)...")
    model = LogisticRegressionModel(
        learning_rate=0.1,
        max_iter=20,  # Fast test
        regularization=1e-3,
        max_features=1000,
        ngram_range=(1, 2),
        batch_size=64
    )
    
    start = time.time()
    model.fit(train_texts, train_labels, window_size=5, streaming=False)
    train_time = time.time() - start
    
    print(f"\n[3/3] Evaluating...")
    metrics = model.evaluate(dev_texts, dev_labels, window_size=5)
    
    print(f"\n   Results:")
    print(f"   - Training time: {train_time:.1f}s")
    print(f"   - Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   - DER: {metrics['der']:.4f}")
    
    # Test prediction
    print(f"\n   Sample prediction:")
    test_text = dev_texts[0][:60]
    pred = model.predict([test_text], window_size=5)
    diacritized = model.apply_diacritics(test_text, pred[0])
    print(f"   Input:  {test_text}")
    print(f"   Output: {diacritized}")
    
    print("\n   [OK] LogReg test passed!")
    return True

def test_crf():
    """Test CRF on 50 sentences"""
    print("\n\n" + "="*80)
    print(" TEST 2: CRF")
    print("="*80)
    
    print("\n[1/4] Loading 50 training sentences...")
    train_texts, train_labels = load_or_prepare_data(Path(TRAIN_FILE))
    dev_texts, dev_labels = load_or_prepare_data(Path(DEV_FILE))
    
    train_texts = train_texts[:50]
    train_labels = train_labels[:50]
    dev_texts = dev_texts[:20]
    dev_labels = dev_labels[:20]
    
    print(f"   Train: {len(train_texts)} sentences")
    print(f"   Dev: {len(dev_texts)} sentences")
    
    print("\n[2/4] Extracting CRF features...")
    start = time.time()
    feature_extractor = CRFFeatureExtractor()
    train_features = feature_extractor.fit_transform(train_texts)
    dev_features = feature_extractor.transform(dev_texts)
    print(f"   Feature dim: {feature_extractor.feature_count}, Time: {time.time()-start:.1f}s")
    
    train_labels_np = [np.array(l, dtype=np.int32) for l in train_labels]
    dev_labels_np = [np.array(l, dtype=np.int32) for l in dev_labels]
    
    print("\n[3/4] Training CRF (max_iter=10 for speed)...")
    model = CRFModel(
        num_labels=NUM_DIACRITIC_CLASSES,
        feature_dim=feature_extractor.feature_count,
        lr=0.01,
        max_iter=10,  # Fast test
        l2_penalty=0.1
    )
    
    start = time.time()
    model.fit(train_features, train_labels_np)
    train_time = time.time() - start
    
    print(f"\n[4/4] Evaluating...")
    metrics = model.evaluate(dev_features, dev_labels_np)
    
    print(f"\n   Results:")
    print(f"   - Training time: {train_time:.1f}s")
    print(f"   - Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   - DER: {metrics['der']:.4f}")
    
    # Test prediction
    print(f"\n   Sample prediction:")
    test_text = dev_texts[0][:60]
    test_feat = feature_extractor.transform([test_text])
    pred = model.predict(test_feat)
    diacritized = model.apply_diacritics(test_text, pred[0])
    print(f"   Input:  {test_text}")
    print(f"   Output: {diacritized}")
    
    print("\n   [OK] CRF test passed!")
    return True

def main():
    print("\n" + "="*80)
    print(" QUICK VERIFICATION TEST")
    print(" Testing both models before overnight training")
    print("="*80)
    print("\n   This will take approximately 2-3 minutes")
    print("   LogReg: 200 sentences (~1 min)")
    print("   CRF: 50 sentences (~1-2 min)")
    
    input("\nPress Enter to start tests...")
    
    start_time = time.time()
    
    try:
        # Test LogReg
        logreg_ok = test_logistic_regression()
        
        # Test CRF
        crf_ok = test_crf()
        
        total_time = time.time() - start_time
        
        # Summary
        print("\n\n" + "="*80)
        print(" TEST SUMMARY")
        print("="*80)
        print(f"\n   Total test time: {total_time/60:.2f} minutes")
        print(f"\n   [OK] LogReg: {'PASSED' if logreg_ok else 'FAILED'}")
        print(f"   [OK] CRF:    {'PASSED' if crf_ok else 'FAILED'}")
        
        if logreg_ok and crf_ok:
            print("\n" + "="*80)
            print(" ALL TESTS PASSED! Ready for overnight training")
            print("="*80)
            print("\n RECOMMENDED OVERNIGHT TRAINING COMMANDS:")
            print("-"*80)
            print("\n 1. BEST OPTION - Train both models:")
            print("    python train_models.py --model both --logreg-sentences 50000 --crf-sentences 3000")
            print("    Estimated time: 2-3 hours")
            print("    Expected: LogReg ~60-65%, CRF ~65-70%")
            
            print("\n 2. FAST OPTION - LogReg only (full 50K):")
            print("    python train_models.py --model logreg")
            print("    Estimated time: 30-40 minutes")
            print("    Expected: LogReg ~60-65%")
            
            print("\n 3. BALANCED - LogReg full + CRF medium:")
            print("    python train_models.py --model both --logreg-sentences 50000 --crf-sentences 1500")
            print("    Estimated time: 1-1.5 hours")
            print("    Expected: LogReg ~60-65%, CRF ~63-68%")
            
            print("\n" + "="*80)
            return True
        else:
            print("\n[ERROR] Some tests failed! Fix errors before overnight training.")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[!] Tests interrupted by user")
        sys.exit(1)
