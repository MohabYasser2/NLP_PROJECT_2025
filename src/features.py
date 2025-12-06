"""
Feature Extraction Module
Provides feature extraction utilities for ML and DL models.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.config import HYPERPARAMS


class CharacterIndexer:
    """Character to index mapping for neural models."""
    
    def __init__(self):
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_char = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
    def fit(self, texts: List[str]):
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
    
    def encode(self, text: str) -> List[int]:
        """Encode text to indices."""
        return [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) 
                for char in text]
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices to text."""
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in indices])
    
    def encode_batch(self, texts: List[str], max_len: int = None) -> np.ndarray:
        """
        Encode batch of texts with padding.
        
        Args:
            texts: List of texts
            max_len: Maximum sequence length
            
        Returns:
            Padded array of shape (batch_size, max_len)
        """
        if max_len is None:
            max_len = max(len(text) for text in texts)
        
        batch = np.zeros((len(texts), max_len), dtype=np.int32)
        
        for i, text in enumerate(texts):
            encoded = self.encode(text)[:max_len]
            batch[i, :len(encoded)] = encoded
        
        return batch
    
    # TODO: Add save/load methods for vocabulary persistence
    # TODO: Add method to get vocabulary size


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
    
    # TODO: Add feature importance analysis
    # TODO: Add feature name extraction


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


class ContextualFeatureExtractor:
    """Extract contextual features for CRF and traditional ML models."""
    
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
            'position': i / len(text),  # Normalized position
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
            features['bigram'] = text[i-1:i+1]
        
        if i < len(text) - 1:
            features['bigram_next'] = text[i:i+2]
        
        # Tri-gram features
        if i > 1:
            features['trigram_prev'] = text[i-2:i+1]
        
        if i < len(text) - 2:
            features['trigram_next'] = text[i:i+3]
        
        # TODO: Add word-level features
        # TODO: Add morphological features
        # TODO: Add POS tag features (if available)
        
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


class EmbeddingLoader:
    """Loader for pre-trained embeddings (FastText, Word2Vec, etc.)."""
    
    def __init__(self, embedding_path: str = None):
        """
        Initialize embedding loader.
        
        Args:
            embedding_path: Path to embedding file
        """
        self.embedding_path = embedding_path
        self.embeddings = {}
        self.embedding_dim = None
    
    def load_fasttext(self):
        """
        Load FastText embeddings.
        
        TODO: Implement FastText loading
        - Use gensim or fasttext library
        - Handle binary and text formats
        - Create character-level embeddings if needed
        """
        raise NotImplementedError("FastText loading not implemented yet")
    
    def load_word2vec(self):
        """
        Load Word2Vec embeddings.
        
        TODO: Implement Word2Vec loading
        """
        raise NotImplementedError("Word2Vec loading not implemented yet")
    
    def get_embedding(self, token: str) -> np.ndarray:
        """
        Get embedding for a token.
        
        Args:
            token: Input token
            
        Returns:
            Embedding vector
        """
        if token in self.embeddings:
            return self.embeddings[token]
        else:
            # Return zero vector for unknown tokens
            return np.zeros(self.embedding_dim)
    
    # TODO: Add method to create embedding matrix for vocabulary
    # TODO: Add support for subword embeddings
    # TODO: Add embedding fine-tuning option


# TODO: Add feature selection methods
# TODO: Add feature normalization utilities
# TODO: Add feature combination strategies
# TODO: Add support for contextualized embeddings (BERT, AraBERT)
