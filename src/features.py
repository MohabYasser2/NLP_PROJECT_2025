"""
Feature Extraction Module
Provides feature extraction utilities for ML and DL models.
"""

import json
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler

from src.config import HYPERPARAMS

# ============================================================
# CharacterIndexer
# ============================================================
class CharacterIndexer:
    """Character to index mapping for neural models."""

    def __init__(self, use_arabic_letters: bool = True):
        """
        Initialize CharacterIndexer.
        
        Args:
            use_arabic_letters: If True, pre-populate with ARABIC_LETTERS from config
        """
        self.char_to_idx: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_char: Dict[int, str] = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size: int = 2
        
        # Pre-populate with Arabic letters if requested
        if use_arabic_letters:
            from src.config import ARABIC_LETTERS
            if ARABIC_LETTERS:
                self.fit(ARABIC_LETTERS)
            # Always include space character
            self.fit([' '])

    # ---------- Fitting / Vocabulary ----------

    def fit(self, texts: List[str]) -> None:
        """
        Build vocabulary from texts.

        Args:
            texts: List of text strings
        """
        chars = set()
        for text in texts:
            chars.update(text)

        for char in sorted(chars):
            if char not in self.char_to_idx:
                idx = self.vocab_size
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
                self.vocab_size += 1

    def get_vocab_size(self) -> int:
        """Return current vocabulary size."""
        return self.vocab_size

    # ---------- Encoding / Decoding ----------

    def encode(self, text: str) -> List[int]:
        """Encode text to indices."""
        return [self.char_to_idx.get(char, self.char_to_idx['<UNK>'])
                for char in text]

    def decode(self, indices: List[int]) -> str:
        """Decode indices to text."""
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in indices])

    def encode_batch(self, texts: List[str], max_len: Optional[int] = None) -> np.ndarray:
        """
        Encode batch of texts with padding.

        Args:
            texts: List of texts
            max_len: Maximum sequence length

        Returns:
            Padded array of shape (batch_size, max_len)
        """
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.int32)

        if max_len is None:
            max_len = max(len(text) for text in texts) if texts else 0

        batch = np.zeros((len(texts), max_len), dtype=np.int32)

        for i, text in enumerate(texts):
            encoded = self.encode(text)[:max_len]
            batch[i, :len(encoded)] = encoded

        return batch

    # ---------- Persistence ----------

    def save(self, path: str) -> None:
        """Save vocabulary to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char_to_idx, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Load vocabulary from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            self.char_to_idx = json.load(f)
        self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

# ============================================================
# TF-IDF Feature Extractor
# ============================================================

class TfidfFeatureExtractor:
    """TF-IDF feature extraction for ML baselines."""
    
    def __init__(self, max_features: int = None, ngram_range: Tuple[int, int] = (1, 3)):
        """
        Initialize TF-IDF extractor.
        
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range for character n-grams
        """
        self.max_features = max_features or HYPERPARAMS['features']['tfidf_max_features']
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=self.ngram_range,
            max_features=self.max_features
        )
    
    def fit(self, texts: List[str]):
        """Fit vectorizer on texts."""
        self.vectorizer.fit(texts)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF features."""
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.vectorizer.fit_transform(texts).toarray()
    def get_feature_names(self) -> List[str]:
        """Return the list of TF-IDF feature names (char n-grams)."""
        return list(self.vectorizer.get_feature_names_out())

    # TODO: Add feature importance analysis
    # TODO: Add feature name extraction

# ============================================================
# Bag-of-Characters Extractor
# ============================================================
class BagOfCharactersExtractor:
    """Bag of characters feature extraction."""
    
    def __init__(self, max_features: int = None):
        """
        Initialize bag of characters extractor.
        
        Args:
            max_features: Maximum number of features
        """
        self.max_features = max_features
        self.vectorizer = CountVectorizer(
            analyzer='char',
            max_features=self.max_features
        )
    
    def fit(self, texts: List[str]):
        """Fit vectorizer on texts."""
        self.vectorizer.fit(texts)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to bag of characters features."""
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.vectorizer.fit_transform(texts).toarray()
    def get_feature_names(self) -> List[str]:
        """Return the character feature names."""
        return list(self.vectorizer.get_feature_names_out())
# ============================================================
# Contextual Feature Extractor (for CRF / ML models)
# ============================================================
class ContextualFeatureExtractor:
    """Extract contextual features for CRF and traditional ML models."""
    @staticmethod
    def _char_to_word_index(text: str, i: int) -> int:
        """
        Approximate mapping from character index to word index
        by counting spaces before position i.
        """
        if not text:
            return 0
        # count spaces before i, but ignore leading spaces
        prefix = text[:i + 1]
        return len(prefix.split()) - 1 if prefix.strip() else 0
    @staticmethod
    def char_features(text: str, i: int) -> Dict[str, any]:
        """
        Extract features for character at position i.
        
        Args:
            text: Input text
            i: Character position
            
        Returns:
            Dictionary of features
        """
        char = text[i]
        
        features = {
            'char': char,
            'is_space': char.isspace(),
            'is_digit': char.isdigit(),
            'is_alpha': char.isalpha(),
            'position_norm': i / max(len(text), 1),  # Normalized position
            'is_first': (i == 0),
            'is_last': (i == len(text) - 1),
        }
        
        # Previous character features
        if i > 0:
            features['prev_char'] = text[i-1]
            features['prev_is_space'] = text[i-1].isspace()
        else:
            features['BOS'] = True  # Beginning of sequence
        
        # Next character features
        if i < len(text) - 1:
            features['next_char'] = text[i+1]
            features['next_is_space'] = text[i+1].isspace()
        else:
            features['EOS'] = True  # End of sequence
        
        # Bi-gram features
        if i > 0:
            features['bigram_prev'] = text[i - 1:i + 1]

        if i < len(text) - 1:
            features['bigram_next'] = text[i:i + 2]

        # Tri-gram features
        if i > 1:
            features['trigram_prev'] = text[i - 2:i + 1]

        if i < len(text) - 2:
            features['trigram_next'] = text[i:i + 3]
        
        # TODO: Add word-level features
        # TODO: Add morphological features
        # TODO: Add POS tag features (if available)
                # Word-level features (approximate)
        words = text.split()
        if words:
            w_idx = ContextualFeatureExtractor._char_to_word_index(text, i)
            w_idx = max(0, min(w_idx, len(words) - 1))
            word = words[w_idx]

            features['word'] = word
            features['word_len'] = len(word)
            features['word_prefix2'] = word[:2]
            features['word_prefix3'] = word[:3]
            features['word_suffix2'] = word[-2:]
            features['word_suffix3'] = word[-3:]
            features['is_first_word'] = (w_idx == 0)
            features['is_last_word'] = (w_idx == len(words) - 1)

        return features
    
    @staticmethod
    def text_to_features(text: str) -> List[Dict[str, any]]:
        """
        Extract features for all characters in text.
        
        Args:
            text: Input text
            
        Returns:
            List of feature dictionaries
        """
        return [ContextualFeatureExtractor.char_features(text, i) 
                for i in range(len(text))]

# ============================================================
# Embedding Loader (Word2Vec / FastText)
# ============================================================

class EmbeddingLoader:
    """Loader for pre-trained embeddings (FastText, Word2Vec, etc.)."""
    
    def __init__(self, embedding_path: str = None):
        """
        Initialize embedding loader.
        
        Args:
            embedding_path: Path to embedding file
        """
        self.embedding_path = embedding_path
        self.embeddings = {} # gensim KeyedVectors or dict-like
        self.embedding_dim = None
    
    def load_word2vec(self, binary: bool = True) -> None:
        """
        Load Word2Vec embeddings using gensim.

        Args:
            binary: Whether file is in binary word2vec format.
        """
        from gensim.models import KeyedVectors
        if self.embedding_path is None:
            raise ValueError("embedding_path is not set")

        model = KeyedVectors.load_word2vec_format(self.embedding_path, binary=binary)
        self.embeddings = model
        self.embedding_dim = model.vector_size
        
    def load_fasttext(self) -> None:
        """
        Load FastText embeddings using gensim.

        Note: For .bin FastText models, you can also use gensim's
        FastText.load_fasttext_format if needed.
        """
        from gensim.models import KeyedVectors
        if self.embedding_path is None:
            raise ValueError("embedding_path is not set")

        # Many FastText embeddings are provided in .vec (text) format.
        model = KeyedVectors.load_word2vec_format(self.embedding_path, binary=False)
        self.embeddings = model
        self.embedding_dim = model.vector_size 
    def get_embedding(self, token: str) -> np.ndarray:
        """
        Get embedding for a token.
        
        Args:
            token: Input token
            
        Returns:
            Embedding vector
        """
        if self.embeddings is None or self.embedding_dim is None:
            raise ValueError("Embeddings not loaded. Call load_word2vec or load_fasttext first.")

        if token in self.embeddings:
            return self.embeddings[token]
        else:
            # Return zero vector for unknown tokens
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
    def build_embedding_matrix(self, vocab: Dict[str, int]) -> np.ndarray:
        """
        Build embedding matrix aligned with given vocabulary.

        Args:
            vocab: token -> index mapping

        Returns:
            Embedding matrix of shape (len(vocab), embedding_dim)
        """
        if self.embeddings is None or self.embedding_dim is None:
            raise ValueError("Embeddings not loaded. Call load_word2vec or load_fasttext first.")

        matrix = np.zeros((len(vocab), self.embedding_dim), dtype=np.float32)
        for token, idx in vocab.items():
            if token in self.embeddings:
                matrix[idx] = self.embeddings[token]
            else:
                # leave as zeros for PAD/UNK or unseen tokens
                pass
        return matrix


# ============================================================
# Utility Functions for Feature Matrices
# ============================================================

def merge_features(*arrays: np.ndarray) -> np.ndarray:
    """
    Merge multiple feature matrices along the last axis.

    All arrays must have the same number of samples (first dimension).

    Args:
        *arrays: numpy arrays of shape (n_samples, n_features_i)

    Returns:
        merged: array of shape (n_samples, sum_i n_features_i)
    """
    valid_arrays = [a for a in arrays if a is not None]
    if not valid_arrays:
        return np.array([])

    # Ensure they all have the same number of samples
    n_samples = valid_arrays[0].shape[0]
    for a in valid_arrays:
        if a.shape[0] != n_samples:
            raise ValueError("All feature arrays must have same number of samples")

    return np.concatenate(valid_arrays, axis=-1)


def normalize_features(X: np.ndarray) -> np.ndarray:
    """
    Normalize features to zero-mean, unit-variance (per column).

    Args:
        X: feature matrix of shape (n_samples, n_features)

    Returns:
        normalized X (same shape)
    """
    if X.size == 0:
        return X
    scaler = StandardScaler()
    return scaler.fit_transform(X)

    # TODO: Add method to create embedding matrix for vocabulary
    # TODO: Add support for subword embeddings
    # TODO: Add embedding fine-tuning option


# TODO: Add feature selection methods
# TODO: Add feature normalization utilities
# TODO: Add feature combination strategies
# TODO: Add support for contextualized embeddings (BERT, AraBERT)
