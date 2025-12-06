"""
Quick script to test pickle resource loading from utils/ folder.
Run this to verify the pickle files are loaded correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import (
    ARABIC_DIACRITICS, 
    DIACRITIC_TO_ID, 
    ARABIC_LETTERS,
    load_pickle_resource
)

def test_pickle_loading():
    """Test that pickle resources are loaded correctly."""
    print("=" * 60)
    print("TESTING PICKLE RESOURCE LOADING")
    print("=" * 60)
    
    # Test 1: Check ARABIC_DIACRITICS
    print("\n1. ARABIC_DIACRITICS:")
    print(f"   Type: {type(ARABIC_DIACRITICS)}")
    print(f"   Length: {len(ARABIC_DIACRITICS) if ARABIC_DIACRITICS else 'N/A'}")
    if ARABIC_DIACRITICS:
        if isinstance(ARABIC_DIACRITICS, set):
            print(f"   Sample (first 3): {list(ARABIC_DIACRITICS)[:3]}")
        else:
            print(f"   Sample (first 3): {ARABIC_DIACRITICS[:3]}")
    
    # Test 2: Check DIACRITIC_TO_ID
    print("\n2. DIACRITIC_TO_ID:")
    print(f"   Type: {type(DIACRITIC_TO_ID)}")
    print(f"   Length: {len(DIACRITIC_TO_ID) if DIACRITIC_TO_ID else 'N/A'}")
    if DIACRITIC_TO_ID:
        print(f"   Sample items (first 3): {list(DIACRITIC_TO_ID.items())[:3]}")
    
    # Test 3: Check ARABIC_LETTERS
    print("\n3. ARABIC_LETTERS:")
    print(f"   Type: {type(ARABIC_LETTERS)}")
    if ARABIC_LETTERS is not None:
        if isinstance(ARABIC_LETTERS, (list, tuple)):
            print(f"   Length: {len(ARABIC_LETTERS)}")
            print(f"   Sample (first 5): {ARABIC_LETTERS[:5]}")
        elif isinstance(ARABIC_LETTERS, dict):
            print(f"   Length: {len(ARABIC_LETTERS)}")
            print(f"   Sample keys (first 5): {list(ARABIC_LETTERS.keys())[:5]}")
        else:
            print(f"   Value: {ARABIC_LETTERS}")
    else:
        print("   Not loaded (None)")
    
    # Test 4: Direct pickle loading
    print("\n4. Testing direct pickle loading:")
    try:
        diacritics = load_pickle_resource('diacritics.pickle')
        print(f"   ✓ diacritics.pickle loaded successfully")
        print(f"     Type: {type(diacritics)}, Length: {len(diacritics) if hasattr(diacritics, '__len__') else 'N/A'}")
    except Exception as e:
        print(f"   ✗ Error loading diacritics.pickle: {e}")
    
    try:
        diacritic2id = load_pickle_resource('diacritic2id.pickle')
        print(f"   ✓ diacritic2id.pickle loaded successfully")
        print(f"     Type: {type(diacritic2id)}, Length: {len(diacritic2id) if hasattr(diacritic2id, '__len__') else 'N/A'}")
    except Exception as e:
        print(f"   ✗ Error loading diacritic2id.pickle: {e}")
    
    try:
        arabic_letters = load_pickle_resource('arabic_letters.pickle')
        print(f"   ✓ arabic_letters.pickle loaded successfully")
        print(f"     Type: {type(arabic_letters)}, Length: {len(arabic_letters) if hasattr(arabic_letters, '__len__') else 'N/A'}")
    except Exception as e:
        print(f"   ✗ Error loading arabic_letters.pickle: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    # Summary
    print("\n📋 SUMMARY:")
    all_loaded = all([
        ARABIC_DIACRITICS is not None,
        DIACRITIC_TO_ID is not None,
        ARABIC_LETTERS is not None
    ])
    
    if all_loaded:
        print("✓ All pickle resources loaded successfully!")
    else:
        print("⚠ Some resources not loaded - using fallback values")
    
    return all_loaded


if __name__ == '__main__':
    success = test_pickle_loading()
    sys.exit(0 if success else 1)
