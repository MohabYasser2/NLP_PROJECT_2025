"""
Test script for Phase 1 preprocessing functions
Tests from TODO_CHRONOLOGICAL.md
"""

from src.preprocessing import clean_arabic_text, strip_diacritics, extract_labels_simple
from src.config import ID_TO_DIACRITIC, DIACRITIC_TO_ID

print("=" * 60)
print("PHASE 1 PREPROCESSING TESTS")
print("=" * 60)

# Test 1: clean_arabic_text
print("\n1. Testing clean_arabic_text()")
print("-" * 60)
test_text = "مرحباً123!!"
result = clean_arabic_text(test_text)
print(f"Input:    {test_text}")
print(f"Output:   {result}")
print(f"Expected: مرحباً")
print(f"✓ PASS" if "مرحبا" in result and "123" not in result and "!!" not in result else "✗ FAIL")

# Test 2: strip_diacritics
print("\n2. Testing strip_diacritics()")
print("-" * 60)
test_text = "مَرْحَباً"
result = strip_diacritics(test_text)
print(f"Input:    {test_text}")
print(f"Output:   {result}")
print(f"Expected: مرحبا")
print(f"✓ PASS" if result == "مرحبا" else "✗ FAIL")

# Test 3: extract_labels_simple - basic
print("\n3. Testing extract_labels_simple() - basic")
print("-" * 60)
test_text = "مَرْحَباً"
clean, labels = extract_labels_simple(test_text)
print(f"Input:          {test_text}")
print(f"Clean text:     {clean}")
print(f"Labels (IDs):   {labels}")
print(f"Labels (chars): {[ID_TO_DIACRITIC[l] for l in labels]}")
print(f"Length match:   clean={len(clean)}, labels={len(labels)}")
print(f"✓ PASS" if len(clean) == len(labels) and clean == "مرحبا" else "✗ FAIL")

# Test 4: extract_labels_simple - multi-diacritic (shadda combinations)
print("\n4. Testing extract_labels_simple() - multi-diacritic")
print("-" * 60)
test_text = "مَّرَحبَا"  # م with fatha+shadda, ر with fatha, ح with no diacritic, ب with fatha, ا with no diacritic
clean, labels = extract_labels_simple(test_text)
print(f"Input:          {test_text}")
print(f"Clean text:     {clean}")
print(f"Labels (IDs):   {labels}")
print(f"Labels (chars): {[repr(ID_TO_DIACRITIC[l]) for l in labels]}")
print(f"Length match:   clean={len(clean)}, labels={len(labels)}")

# Check if multi-diacritic is captured
has_multi = any(len(ID_TO_DIACRITIC[l]) > 1 for l in labels if ID_TO_DIACRITIC[l] != '')
print(f"Multi-diacritic captured: {has_multi}")
print(f"✓ PASS" if len(clean) == len(labels) and has_multi else "✗ FAIL")

# Test 5: Check available multi-diacritics in mapping
print("\n5. Available multi-diacritic combinations in DIACRITIC_TO_ID:")
print("-" * 60)
multi_diacs = {k: v for k, v in DIACRITIC_TO_ID.items() if len(k) > 1}
for diac, idx in sorted(multi_diacs.items(), key=lambda x: x[1]):
    print(f"  ID {idx}: {repr(diac)}")
print(f"Total: {len(multi_diacs)} combinations")

# Test 6: Empty and edge cases
print("\n6. Testing edge cases")
print("-" * 60)
test_cases = [
    ("", "empty string"),
    ("مرحبا", "text without diacritics"),
    ("َّ", "only diacritics"),
    ("   ", "only spaces"),
]

for test_text, description in test_cases:
    try:
        clean, labels = extract_labels_simple(test_text)
        print(f"  {description:25s} → clean='{clean}', labels={labels}")
    except Exception as e:
        print(f"  {description:25s} → ERROR: {e}")

print("\n" + "=" * 60)
print("TESTS COMPLETE")
print("=" * 60)
