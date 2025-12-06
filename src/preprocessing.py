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
    
    for char in text:
        if char in ARABIC_DIACRITICS:
            # Diacritic found - add to labels for previous character
            if labels:  # Append to last label if exists
                labels[-1] = labels[-1] + char if labels[-1] != '_' else char
            else:
                labels.append(char)
        else:
            # Regular character
            clean_chars.append(char)
            labels.append('_')  # No diacritic by default
    
    clean_text = ''.join(clean_chars)
    
    # TODO: Handle multiple diacritics per character more robustly
    # TODO: Consider Shadda combinations (Shadda + Fatha, etc.)
    
    return clean_text, labels


def extract_labels_simple(text: str) -> Tuple[str, List[int]]:
    """
    Extract diacritics as label IDs (simplified: one diacritic per char).
    
    Args:
        text: Diacritized Arabic text
        
    Returns:
        Tuple of (clean_text, label_ids)
    """
    clean_chars = []
    label_ids = []
    
    i = 0
    while i < len(text):
        char = text[i]
        
        if char in ARABIC_DIACRITICS:
            i += 1
            continue
            
        clean_chars.append(char)
        
        # Check if next character is a diacritic
        if i + 1 < len(text) and text[i + 1] in ARABIC_DIACRITICS:
            diacritic = text[i + 1]
            label_ids.append(DIACRITIC_TO_ID[diacritic])
        else:
            label_ids.append(DIACRITIC_TO_ID['_'])  # No diacritic
        
        i += 1
    
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
    return text.split()


def clean_arabic_text(text: str) -> str:
    """
    Clean Arabic text using regex patterns.
    
    Args:
        text: Raw Arabic text
        
    Returns:
        Cleaned text
    """
    # Remove non-Arabic characters except spaces and diacritics
    # Arabic Unicode range: \u0600-\u06FF
    # Keep diacritics for now
    pattern = r'[^\u0600-\u06FF\s]'
    text = re.sub(pattern, '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # TODO: Add Tatweel removal (ـ)
    # TODO: Normalize Alef variations (أ، إ، آ -> ا)
    # TODO: Normalize Taa Marbuta (ة -> ه)
    
    return text


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


def prepare_dataset(file_path: Path) -> Tuple[List[str], List[List[int]]]:
    """
    Load and prepare dataset for training.
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        Tuple of (clean_texts, label_sequences)
    """
    sentences = load_dataset(file_path)
    
    clean_texts = []
    label_sequences = []
    
    for sentence in sentences:
        clean_text, labels = extract_labels_simple(sentence)
        clean_texts.append(clean_text)
        label_sequences.append(labels)
    
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
        encoded.append(seq)
    
    # TODO: Add padding if max_len is specified
    # TODO: Add truncation for sequences longer than max_len
    
    return encoded


# TODO: Add function for data augmentation (character substitution, etc.)
# TODO: Add function for handling imbalanced diacritic distribution
# TODO: Add support for word-level features alongside character features
# TODO: Add caching mechanism for preprocessed data
