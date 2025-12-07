"""
OVERNIGHT TRAINING SCRIPT
Trains models with full datasets and saves detailed logs
Run this before going to sleep - it will complete in 2-3 hours
"""

import sys
from pathlib import Path
import time
import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Redirect output to log file
log_file = Path(__file__).parent / 'outputs' / f'training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
log_file.parent.mkdir(exist_ok=True)

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
sys.stderr = sys.stdout

print("="*80)
print(" OVERNIGHT TRAINING - Arabic Diacritization")
print("="*80)
print(f"\nStart time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {log_file}")
print("\nThis script will train:")
print("  1. Logistic Regression on 50,000 sentences (~30-40 min)")
print("  2. CRF on 3,000 sentences (~1.5-2 hours)")
print("  Total estimated time: 2-3 hours")
print("="*80)

# Import after logger setup
from train_models import train_logistic_regression, train_crf

def main():
    overall_start = time.time()
    results = {}
    
    try:
        # Train Logistic Regression on FULL 50K dataset
        print("\n\n" + "="*80)
        print(" PHASE 1: LOGISTIC REGRESSION (50K sentences)")
        print("="*80)
        print(f"\nPhase 1 start: {datetime.datetime.now().strftime('%H:%M:%S')}")
        
        logreg_model, logreg_metrics = train_logistic_regression(
            num_sentences=None,  # Use all 50K
            use_streaming=True
        )
        results['logreg'] = logreg_metrics
        print(f"\nPhase 1 complete: {datetime.datetime.now().strftime('%H:%M:%S')}")
        
        # Train CRF on 3K sentences
        print("\n\n" + "="*80)
        print(" PHASE 2: CRF (3,000 sentences)")
        print("="*80)
        print(f"\nPhase 2 start: {datetime.datetime.now().strftime('%H:%M:%S')}")
        
        crf_model, crf_features, crf_metrics = train_crf(
            num_sentences=3000
        )
        results['crf'] = crf_metrics
        print(f"\nPhase 2 complete: {datetime.datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"\n\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    total_time = time.time() - overall_start
    
    print("\n\n" + "="*80)
    print(" OVERNIGHT TRAINING COMPLETE!")
    print("="*80)
    print(f"\nEnd time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    
    print("\n" + "-"*80)
    print(" FINAL RESULTS")
    print("-"*80)
    
    if 'logreg' in results:
        print("\nLogistic Regression (50K sentences):")
        print(f"  Accuracy: {results['logreg']['accuracy']:.4f} ({results['logreg']['accuracy']*100:.2f}%)")
        print(f"  DER:      {results['logreg']['der']:.4f}")
        print(f"  Model:    models/logreg_50000.pkl")
    
    if 'crf' in results:
        print("\nCRF (3K sentences):")
        print(f"  Accuracy: {results['crf']['accuracy']:.4f} ({results['crf']['accuracy']*100:.2f}%)")
        print(f"  DER:      {results['crf']['der']:.4f}")
        print(f"  Model:    models/crf_3000.pkl")
        print(f"  Features: models/crf_features_3000.pkl")
    
    if len(results) == 2:
        better = 'CRF' if results['crf']['accuracy'] > results['logreg']['accuracy'] else 'LogReg'
        diff = abs(results['crf']['accuracy'] - results['logreg']['accuracy']) * 100
        print(f"\nBest model: {better} (better by {diff:.2f}%)")
    
    print("\n" + "="*80)
    print(" NEXT STEPS:")
    print("="*80)
    print("1. Check results above")
    print("2. Evaluate models: python evaluate_and_infer.py")
    print("3. Generate test predictions for Kaggle")
    print(f"4. View full log: {log_file}")
    print("\n" + "="*80)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
