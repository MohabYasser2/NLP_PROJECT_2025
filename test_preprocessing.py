"""
Quick test script to verify data loading and preprocessing
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import load_dataset, clean_arabic_text, extract_labels_simple
from src.config import TRAIN_FILE, DEV_FILE, DIACRITIC_TO_ID, ID_TO_DIACRITIC

def test_data_loading():
    """Test loading data files"""
    print("=" * 60)
    print("Testing Data Loading")
    print("=" * 60)
    
    print(f"\n1. Loading train file: {TRAIN_FILE}")
    train_sentences = load_dataset(TRAIN_FILE)
    print(f"   ✓ Loaded {len(train_sentences)} training sentences")
    print(f"   First sentence: {train_sentences[0][:100]}...")
    
    print(f"\n2. Loading dev file: {DEV_FILE}")
    dev_sentences = load_dataset(DEV_FILE)
    print(f"   ✓ Loaded {len(dev_sentences)} dev sentences")
    print(f"   First sentence: {dev_sentences[0][:100]}...")
    
    return train_sentences[:10], dev_sentences[:10]

def test_preprocessing(sentences):
    """Test preprocessing functions"""
    print("\n" + "=" * 60)
    print("Testing Preprocessing Functions")
    print("=" * 60)
    
    test_sentence = sentences[0]
    print(f"\n1. Original: {test_sentence}")
    
    # Test cleaning
    cleaned = clean_arabic_text(test_sentence)
    print(f"2. Cleaned: {cleaned}")
    
    # Test label extraction
    clean_text, labels = extract_labels_simple(cleaned)
    print(f"3. Clean text (no diacritics): {clean_text}")
    print(f"4. Label IDs: {labels[:20]}...")
    print(f"5. Number of labels: {len(labels)}")
    print(f"6. Number of non-space chars in clean text: {sum(1 for c in clean_text if not c.isspace())}")
    
    # Verify alignment
    non_space_count = sum(1 for c in clean_text if not c.isspace())
    print(f"\n✓ Label count matches non-space chars: {len(labels) == non_space_count}")
    
    return clean_text, labels

def test_diacritic_mappings():
    """Test diacritic dictionaries"""
    print("\n" + "=" * 60)
    print("Testing Diacritic Mappings")
    print("=" * 60)
    
    print(f"\nNumber of diacritic classes: {len(DIACRITIC_TO_ID)}")
    print("\nDiacritic to ID mapping (first 10):")
    for i, (diac, idx) in enumerate(list(DIACRITIC_TO_ID.items())[:10]):
        print(f"   '{diac}' (U+{ord(diac[0]):04X} if diac else 'EMPTY') -> {idx}")
    
    print("\nID to Diacritic mapping (first 10):")
    for i, (idx, diac) in enumerate(list(ID_TO_DIACRITIC.items())[:10]):
        print(f"   {idx} -> '{diac}' (U+{ord(diac[0]):04X} if diac else 'EMPTY')")

if __name__ == "__main__":
    try:
        # Test data loading
        train_sample, dev_sample = test_data_loading()
        
        # Test preprocessing
        clean_text, labels = test_preprocessing(train_sample)
        
        # Test mappings
        test_diacritic_mappings()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
