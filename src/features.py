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
# CharacterIndexer → integer encoding & vocabulary building
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
# TF-IDF Feature Extractor → baseline ML features
# ============================================================

class ManualTFIDF:
    """
    Pure Python TF-IDF without sklearn.
    Works on characters or n-grams depending on tokenizer.
    """

    def _init_(self, ngram:int=1):
        self.ngram = ngram
        self.idf_dict = {}
        self.vocab = []

    # --------------------- VOCAB BUILD -------------------------
    def _tokenize(self, text:str)->List[str]:
        return [text[i:i+self.ngram] for i in range(len(text)-self.ngram+1)]

    def build_vocab(self, corpus:List[str]) -> None:
        vocab_set = set()
        for doc in corpus:
            tokens = self._tokenize(doc)
            vocab_set.update(tokens)
        self.vocab = sorted(list(vocab_set))  # stable order

    # ---------------------- IDF COMPUTATION ---------------------
    def fit(self, corpus:List[str]) -> None:
        if not self.vocab:
            self.build_vocab(corpus)

        N = len(corpus)
        df = {term:0 for term in self.vocab}

        for doc in corpus:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                df[t]+=1
        
        # IDF = log(N / (1 + df))
        self.idf_dict = {t:np.log(N/(1+df[t])) for t in self.vocab}

    # ----------------------- TRANSFORM -------------------------
    def transform(self, docs:List[str]) -> np.ndarray:
        """Return TF-IDF vectors."""
        vectors = []

        for doc in docs:
            tokens = self._tokenize(doc)
            tf = {t:tokens.count(t)/max(len(tokens),1) for t in self.vocab}
            row = np.array([ tf[t] * self.idf_dict[t] for t in self.vocab ])
            vectors.append(row)

        return np.vstack(vectors)

    def fit_transform(self, corpus:List[str]):
        self.fit(corpus)
        return self.transform(corpus)
class SkipGramCharEmbedding:
    """
    Train character embeddings from scratch using a true Skip-Gram neural network:
    
    Architecture:
        - Input: one-hot vector for center character (size = vocab_size)
        - Hidden layer: embedding matrix W_in (vocab_size x embed_dim)
        - Output layer: W_out (vocab_size x embed_dim)
        - Softmax over all characters to predict a context character.
    
    Training objective:
        For each (center, context) pair:
            minimize -log P(context | center)
        where P is computed by softmax(W_out * W_in[center])
    """

    def _init_(self, embed_dim: int = 50, window: int = 3, lr: float = 0.01):
        self.embed_dim = embed_dim
        self.window = window
        self.lr = lr

        self.vocab: List[str] = []
        self.char2id: dict = {}
        self.id2char: dict = {}

        # Will be initialized after build_vocab
        self.W_in: np.ndarray = None   # (vocab_size, embed_dim)
        self.W_out: np.ndarray = None  # (vocab_size, embed_dim)

    # -------------------------------------------------
    # 1) Vocabulary building
    # -------------------------------------------------
    def build_vocab(self, corpus: List[str]) -> None:
        """
        Build character vocabulary from corpus and initialize weights.
        """
        chars = set("".join(corpus))
        self.vocab = sorted(list(chars))
        self.char2id = {c: i for i, c in enumerate(self.vocab)}
        self.id2char = {i: c for c, i in self.char2id.items()}

        vocab_size = len(self.vocab)

        # Initialize weights with small random values
        self.W_in = 0.01 * np.random.randn(vocab_size, self.embed_dim)
        self.W_out = 0.01 * np.random.randn(vocab_size, self.embed_dim)

    # -------------------------------------------------
    # 2) Softmax helper
    # -------------------------------------------------
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax.
        """
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)

    # -------------------------------------------------
    # 3) Generate training pairs (center, context)
    # -------------------------------------------------
    def _generate_pairs(self, corpus: List[str]) -> List[tuple]:
        """
        Generate (center_char_idx, context_char_idx) pairs
        using the given window size.
        """
        pairs = []

        for text in corpus:
            # Skip empty strings
            if not text:
                continue

            for i, ch in enumerate(text):
                if ch not in self.char2id:
                    continue

                center_id = self.char2id[ch]

                # Window around the center
                start = max(0, i - self.window)
                end = min(len(text), i + self.window + 1)

                for j in range(start, end):
                    if j == i:
                        continue
                    ctx_ch = text[j]
                    if ctx_ch not in self.char2id:
                        continue
                    context_id = self.char2id[ctx_ch]

                    pairs.append((center_id, context_id))

        return pairs

    # -------------------------------------------------
    # 4) One training step for a single pair (center, context)
    # -------------------------------------------------
    def _train_single_pair(self, center_id: int, context_id: int) -> float:
        """
        Perform forward + backward pass on a single (center, context) pair.
        Returns the loss for monitoring.
        """

        # ---- Forward ----
        # v_c: embedding of center character (D,)
        v_c = self.W_in[center_id]  # (embed_dim,)

        # scores for all output characters: u_k^T * v_c  (size = vocab_size)
        scores = self.W_out @ v_c   # (vocab_size,)

        # probabilities P(context | center)
        y_pred = self._softmax(scores)  # (vocab_size,)

        # true distribution (one-hot)
        vocab_size = len(self.vocab)
        y_true = np.zeros(vocab_size)
        y_true[context_id] = 1.0

        # cross-entropy loss: -log P(correct context)
        loss = -np.log(y_pred[context_id] + 1e-12)

        # ---- Backward ----
        # error = y_pred - y_true
        error = y_pred - y_true  # (vocab_size,)

        # Gradients:
        # dL/dW_out = outer(error, v_c)  (vocab_size x embed_dim)
        grad_W_out = np.outer(error, v_c)

        # dL/dv_c = error^T * W_out   (embed_dim,)
        grad_v_c = error @ self.W_out  # (embed_dim,)

        # ---- Parameter update ----
        self.W_out -= self.lr * grad_W_out
        self.W_in[center_id] -= self.lr * grad_v_c

        return loss

    # -------------------------------------------------
    # 5) Full training loop
    # -------------------------------------------------
    def train(self, corpus: List[str], epochs: int = 5, shuffle: bool = True) -> None:
        """
        Train Skip-Gram on the given corpus.

        Args:
            corpus: list of strings (sentences or full lines)
            epochs: number of passes over the training pairs
            shuffle: whether to shuffle pairs each epoch
        """
        if self.W_in is None or self.W_out is None:
            self.build_vocab(corpus)

        pairs = self._generate_pairs(corpus)

        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(pairs)

            total_loss = 0.0

            for center_id, context_id in pairs:
                loss = self._train_single_pair(center_id, context_id)
                total_loss += loss

            avg_loss = total_loss / max(len(pairs), 1)
            print(f"Epoch {epoch+1}/{epochs} - avg loss: {avg_loss:.4f}")

    # -------------------------------------------------
    # 6) Get embedding of a single character
    # -------------------------------------------------
    def get_char_embedding(self, ch: str) -> np.ndarray:
        """
        Return the embedding vector of a character.
        """
        if ch not in self.char2id:
            # Unknown char → zero vector
            return np.zeros(self.embed_dim)
        return self.W_in[self.char2id[ch]]

    # -------------------------------------------------
    # 7) Encode sentence → mean of character embeddings
    # -------------------------------------------------
    def encode_sentence(self, text: str) -> np.ndarray:
        """
        Encode a whole sentence by averaging its character embeddings.
        """
        vecs = [self.get_char_embedding(c) for c in text if c in self.char2id]
        if not vecs:
            return np.zeros(self.embed_dim)
        return np.mean(vecs, axis=0)
# ============================================================
# Bag-of-Characters Extractor → simple frequency model
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
        """Transform texts to Bag-of-Characters features, fallback to fit if required."""
        try:
            return self.vectorizer.transform(texts).toarray()
        except:
            return self.fit_transform(texts)

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

# ============================================================
#merge_features + normalize_features → unify ML feature spaces
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
