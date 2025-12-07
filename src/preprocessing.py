"""
Preprocessing Module
Handles Arabic text cleaning, diacritic extraction, and tokenization.
"""

import re
from typing import List, Tuple, Dict
from pathlib import Path
from src.config import ARABIC_DIACRITICS, DIACRITIC_TO_ID


def strip_diacritics(text: str) -> str:
    """
    Remove all Arabic diacritics from text.
    
    Args:
        text: Arabic text with diacritics
        
    Returns:
        Text without diacritics
    """
    for diacritic in ARABIC_DIACRITICS:
        text = text.replace(diacritic, '')
    return text


def extract_labels(text: str) -> Tuple[str, List[str]]:
    """
    Extract diacritics as labels and return clean text.
    
    Args:
        text: Diacritized Arabic text
        
    Returns:
        Tuple of (clean_text, labels) where labels align with characters
    """
    clean_chars = []
    labels = []

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        # If current char is a diacritic (unexpected leading diacritic), skip it
        if ch in ARABIC_DIACRITICS:
            i += 1
            continue

        # Regular character: collect it
        clean_chars.append(ch)

        # Gather all following diacritics that belong to this character
        diac_seq = ''
        j = i + 1
        while j < n and text[j] in ARABIC_DIACRITICS:
            diac_seq += text[j]
            j += 1

        if diac_seq == '':
            labels.append('_')
        else:
            # store combined diacritics as a single string (order preserved)
            labels.append(diac_seq)

        # Advance to next base character position
        i = j

    clean_text = ''.join(clean_chars)

    return clean_text, labels


def normalize_diacritic_order(diac_seq: str) -> str:
    """
    Normalize the order of diacritics to match the canonical order.
    Arabic canonical order: Shadda (ّ U+0651) comes before vowel diacritics.
    
    Args:
        diac_seq: Sequence of diacritics
        
    Returns:
        Normalized diacritic sequence
    """
    if len(diac_seq) <= 1:
        return diac_seq
    
    # Shadda character
    SHADDA = '\u0651'
    
    # If sequence contains shadda, ensure it comes first
    if SHADDA in diac_seq:
        # Separate shadda and other diacritics
        other_diacs = diac_seq.replace(SHADDA, '')
        # Return shadda first, then others
        return SHADDA + other_diacs
    
    return diac_seq


def extract_labels_simple(text: str) -> Tuple[str, List[int]]:
    """
    Extract diacritics as label IDs (supports multiple diacritics per char).
    
    Args:
        text: Diacritized Arabic text
        
    Returns:
        Tuple of (clean_text, label_ids)
    """
    clean_chars = []
    label_ids = []
    
    i = 0
    n = len(text)
    while i < n:
        char = text[i]

        # Skip stray diacritics
        if char in ARABIC_DIACRITICS:
            i += 1
            continue

        # Regular character: collect it
        clean_chars.append(char)

        # Skip spaces - don't create labels for them (matches window creation logic)
        if char.isspace():
            i += 1
            continue

        # Gather all following diacritics that belong to this character
        diac_seq = ''
        j = i + 1
        while j < n and text[j] in ARABIC_DIACRITICS:
            diac_seq += text[j]
            j += 1

        # Normalize diacritic order before lookup
        diac_seq = normalize_diacritic_order(diac_seq)

        # Map the combined diacritic sequence to an ID
        if diac_seq == '':
            label_ids.append(DIACRITIC_TO_ID[''])
        else:
            # Use combined diacritics as key, fallback to '' (no diacritic) if not in mapping
            label_ids.append(DIACRITIC_TO_ID.get(diac_seq, DIACRITIC_TO_ID['']))

        # Advance to next base character position
        i = j
    
    clean_text = ''.join(clean_chars)
    
    return clean_text, label_ids


def tokenize_characters(text: str) -> List[str]:
    """
    Tokenize text into characters.
    
    Args:
        text: Input text
        
    Returns:
        List of characters
    """
    return list(text)


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into words (split by whitespace).
    
    Args:
        text: Input text
        
    Returns:
        List of words
    """
    # special-case empty string to match expected behavior in tests
    if text == "":
        return ['']

    return text.split()


def clean_arabic_text(text: str) -> str:
    """
    Clean Arabic text using regex patterns.
    
    Args:
        text: Raw Arabic text
        
    Returns:
        Cleaned text
    """
    # Use a more thorough normalization that reduces letter shape variations
    # and removes tatweel while preserving diacritics.
    return clean_sentence(text)


def normalize_arabic(text: str) -> str:
    """Normalize common Arabic letter shapes."""
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    # Remove tatweel
    text = text.replace("ـ", "")
    return text


# Allow common base Arabic letters (no diacritics here)
ARABIC_LETTERS = "ءاأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي"


def clean_sentence(sentence: str) -> str:
    """
    Cleans Arabic sentence while preserving diacritics defined in ARABIC_DIACRITICS.
    Steps:
        1) Normalize Arabic letters (Reduce variations)
        2) Remove English digits & characters
        3) Remove punctuation / symbols
        4) Keep ONLY Arabic letters + diacritics + spaces
        5) Compress multiple spaces
    """

    # (1) normalize
    sentence = normalize_arabic(sentence)
    # (2) Remove English & numbers
    sentence = re.sub(r"[A-Za-z0-9]+", " ", sentence)

    # (3) Remove punctuation, brackets, symbols
    sentence = re.sub(r"[«»()\[\]{}<>؛:;/\\\-–—_.,!?+*=]", " ", sentence)

    # (4) Allow only Arabic + diacritics + whitespace
    allowed_pattern = rf"[^{ARABIC_LETTERS}{''.join(ARABIC_DIACRITICS)}\s]"
    sentence = re.sub(allowed_pattern, " ", sentence)

    # (5) Remove duplicate spaces
    sentence = re.sub(r"\s+", " ", sentence).strip()

    return sentence


def load_dataset(file_path: Path) -> List[str]:
    """
    Load dataset from text file.
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        List of sentences
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Clean and filter empty lines
    sentences = [line.strip() for line in lines if line.strip()]
    
    return sentences


def prepare_dataset(input_file: str, output_prefix: str) -> Tuple[List[str], List[List[int]]]:
    """
    Load and prepare dataset for training, then save as pickle files.
    Complete preprocessing pipeline:
    1. Load sentences from file
    2. Clean Arabic text (normalize, remove non-Arabic chars)
    3. Extract diacritic labels (before cleaning removes them)
    4. Save processed data as pickle file
    
    Args:
        input_file: Path to input dataset file (str or Path)
        output_prefix: Path prefix for output pickle file (e.g., 'data/processed_train')
        
    Returns:
        Tuple of (clean_texts, label_sequences)
    """
    import pickle
    
    file_path = Path(input_file)
    sentences = load_dataset(file_path)
    
    clean_texts = []
    label_sequences = []
    
    print(f"Processing {len(sentences)} sentences from {file_path}...")
    
    for i, sentence in enumerate(sentences):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(sentences)} sentences...")
        
        # Step 1: Clean the Arabic text (normalize, remove non-Arabic)
        # This preserves diacritics while cleaning everything else
        cleaned_sentence = clean_arabic_text(sentence)
        
        # Skip empty sentences after cleaning
        if not cleaned_sentence:
            continue
        
        # Step 2: Extract labels from the cleaned diacritized text
        clean_text, labels = extract_labels_simple(cleaned_sentence)
        
        # Skip if no content after label extraction
        if not clean_text:
            continue
        
        clean_texts.append(clean_text)
        label_sequences.append(labels)
    
    # Save to pickle file
    output_file = Path(f"{output_prefix}.pkl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'texts': clean_texts,
        'labels': label_sequences
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Saved {len(clean_texts)} processed sentences to {output_file}")
    
    return clean_texts, label_sequences


def create_char_vocabulary(texts: List[str]) -> Dict[str, int]:
    """
    Create character vocabulary from texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary mapping characters to indices
    """
    chars = set()
    for text in texts:
        chars.update(text)
    
    # Add special tokens
    char_to_idx = {'<PAD>': 0, '<UNK>': 1}
    
    for idx, char in enumerate(sorted(chars), start=2):
        char_to_idx[char] = idx
    
    return char_to_idx


def encode_sequences(texts: List[str], char_to_idx: Dict[str, int], 
                     max_len: int = None) -> List[List[int]]:
    """
    Encode text sequences to integer sequences.
    
    Args:
        texts: List of text strings
        char_to_idx: Character to index mapping
        max_len: Maximum sequence length (for padding)
        
    Returns:
        List of encoded sequences
    """
    encoded = []
    
    for text in texts:
        seq = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in text]

        if max_len is not None:
            if len(seq) > max_len:
                # truncate
                seq = seq[:max_len]
            elif len(seq) < max_len:
                # pad with PAD token
                pad_id = char_to_idx.get('<PAD>', 0)
                seq = seq + [pad_id] * (max_len - len(seq))

        encoded.append(seq)

    return encoded


# TODO: Add function for data augmentation (character substitution, etc.)
# TODO: Add function for handling imbalanced diacritic distribution
# TODO: Add support for word-level features alongside character features
# TODO: Add caching mechanism for preprocessed data
