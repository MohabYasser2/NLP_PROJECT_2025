"""
START HERE - Main Training Launcher with proper console encoding
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        # Try to set console to UTF-8
        os.system('chcp 65001 >nul 2>&1')
        # Set Python's default encoding
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Now import the actual training module
from train_models import main

if __name__ == "__main__":
    print("="*80)
    print(" ARABIC DIACRITIZATION - TRAINING LAUNCHER")
    print("="*80)
    print("\nThis script will train your Arabic diacritization models.")
    print("Choose from the following options:\n")
    print("  1. Quick Test (LogReg: 1K sentences, ~1 minute)")
    print("  2. Medium Training (LogReg: 5K, CRF: 500, ~10 minutes)")
    print("  3. Full LogReg (50K sentences, ~30 minutes)")
    print("  4. Both Models (LogReg: 50K, CRF: 1K, ~45 minutes)")
    print("  5. Custom (specify parameters)")
    print("  6. Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        print("\n[Quick Test Mode] Training LogReg on 1K sentences...")
        sys.argv = ['train_models.py', '--model', 'logreg', '--logreg-sentences', '1000']
        
    elif choice == '2':
        print("\n[Medium Mode] Training both models with reduced datasets...")
        sys.argv = ['train_models.py', '--quick']
        
    elif choice == '3':
        print("\n[Full LogReg] Training on ALL 50K sentences...")
        print("This will take approximately 30 minutes.")
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Cancelled.")
            sys.exit(0)
        sys.argv = ['train_models.py', '--model', 'logreg']
        
    elif choice == '4':
        print("\n[Both Models] Training LogReg (50K) and CRF (1K)...")
        print("This will take approximately 45 minutes.")
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Cancelled.")
            sys.exit(0)
        sys.argv = ['train_models.py', '--model', 'both']
        
    elif choice == '5':
        print("\nCustom training:")
        model = input("  Model (logreg/crf/both): ").strip().lower()
        if model == 'logreg' or model == 'both':
            logreg_sent = input("  LogReg sentences (default: ALL): ").strip()
            if logreg_sent:
                sys.argv = ['train_models.py', '--model', model, '--logreg-sentences', logreg_sent]
            else:
                sys.argv = ['train_models.py', '--model', model]
        else:
            crf_sent = input("  CRF sentences (default: 1000): ").strip()
            if crf_sent:
                sys.argv = ['train_models.py', '--model', 'crf', '--crf-sentences', crf_sent]
            else:
                sys.argv = ['train_models.py', '--model', 'crf']
        
    elif choice == '6':
        print("Goodbye!")
        sys.exit(0)
        
    else:
        print("Invalid choice!")
        sys.exit(1)
    
    # Launch training
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
