"""
Logistic Regression Model - FROM SCRATCH
Pure NumPy implementation for NLP course.
No sklearn, no pre-built ML libraries.
Supports CUDA GPU acceleration via CuPy when available.
"""

import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import Counter
from scipy.sparse import csr_matrix, vstack as sparse_vstack

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
    GPU_AVAILABLE = True
    print("✓ CuPy available - GPU acceleration enabled")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("✓ Running in CPU mode (NumPy)")

from src.config import DIACRITIC_TO_ID, ID_TO_DIACRITIC


class TfidfVectorizer:
    """
    Custom TF-IDF implementation from scratch.
    Uses character n-grams for Arabic diacritization.
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 3)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary = {}  # ngram -> index
        self.idf = None
        self.is_fitted = False
    
    def _extract_ngrams(self, text: str) -> List[str]:
        """Extract character n-grams from text."""
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i+n])
        return ngrams
    
    def fit(self, texts: List[str]):
        """Build vocabulary and compute IDF values."""
        print(f"Building vocabulary from {len(texts)} texts...")
        
        # Count document frequency
        df_counter = Counter()
        
        for text in tqdm(texts, desc="Building vocabulary", unit="text"):
            ngrams = set(self._extract_ngrams(text))
            for ngram in ngrams:
                df_counter[ngram] += 1
        
        # Select top features by frequency
        print(f"  Selecting top {self.max_features} features...")
        most_common = df_counter.most_common(self.max_features)
        self.vocabulary = {ngram: idx for idx, (ngram, _) in enumerate(most_common)}
        print(f"  ✓ Vocabulary built: {len(self.vocabulary)} features")
        
        # Compute IDF: log((N + 1) / (df + 1)) + 1
        num_docs = len(texts)
        self.idf = np.zeros(len(self.vocabulary))
        for ngram, idx in self.vocabulary.items():
            df = df_counter[ngram]
            self.idf[idx] = np.log((num_docs + 1) / (df + 1)) + 1
        
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str], batch_size: int = 50000, silent: bool = False):
        """Transform texts to TF-IDF matrix (memory-optimized for GPU)."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        num_features = len(self.vocabulary)
        num_texts = len(texts)
        num_batches = (num_texts + batch_size - 1) // batch_size
        
        if not silent:
            print(f"Transforming {num_texts:,} texts to TF-IDF ({num_batches} batches)...")
            print(f"Pre-allocating matrix: {num_texts} × {num_features} = {num_texts * num_features * 4 / 1e9:.2f} GB")
        
        # Pre-allocate ENTIRE matrix at once (avoids vstack memory spike)
        matrix = np.zeros((num_texts, num_features), dtype=np.float32)
        
        # Fill matrix in batches
        iterator = tqdm(range(num_batches), desc="Building TF-IDF", unit="batch", disable=silent)
        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_texts)
            
            # Fill matrix directly (no intermediate batch array)
            for global_idx in range(start_idx, end_idx):
                text = texts[global_idx]
                ngrams = self._extract_ngrams(text)
                tf_counter = Counter(ngrams)
                doc_length = len(ngrams) if ngrams else 1
                
                for ngram, count in tf_counter.items():
                    if ngram in self.vocabulary:
                        feat_idx = self.vocabulary[ngram]
                        matrix[global_idx, feat_idx] = (count / doc_length) * self.idf[feat_idx]
        
        if not silent:
            print(f"✓ TF-IDF matrix built: {matrix.shape}, {matrix.nbytes / 1e9:.2f} GB")
        return matrix
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)


class LogisticRegressionModel:
    """
    Logistic Regression FROM SCRATCH using only NumPy.
    Multinomial logistic regression with gradient descent.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                 regularization: float = 0.01, max_features: int = 5000, 
                 ngram_range: Tuple[int, int] = (1, 3), batch_size: int = 32):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularization = regularization  # L2 regularization
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.batch_size = batch_size
        
        self.vectorizer = TfidfVectorizer(max_features, ngram_range)
        self.weights = None  # Will be (num_features, num_classes)
        self.bias = None  # Will be (num_classes,)
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.is_fitted = False
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _prepare_windows(self, texts: List[str], window_size: int = 5) -> List[str]:
        """Create character windows (skip spaces)."""
        windows = []
        for text in texts:
            padded = ' ' * window_size + text + ' ' * window_size
            for i in range(len(text)):
                if text[i].isspace():
                    continue
                center = i + window_size
                window = padded[center - window_size:center + window_size + 1]
                windows.append(window)
        return windows
    
    def fit(self, texts: List[str], label_sequences: List[List[int]], 
            window_size: int = 5, streaming: bool = True, chunk_size: int = 500):
        """
        Train logistic regression using gradient descent with streaming support.
        
        Args:
            texts: Input texts without diacritics
            label_sequences: Diacritic labels for each non-space character
            window_size: Context window size
            streaming: If True, use streaming training (recommended for large datasets)
            chunk_size: Number of sentences to process per chunk (only used if streaming=True)
        """
        if streaming and len(texts) > chunk_size:
            return self._fit_streaming(texts, label_sequences, window_size, chunk_size)
        else:
            return self._fit_standard(texts, label_sequences, window_size)
    
    def _fit_standard(self, texts: List[str], label_sequences: List[List[int]], window_size: int = 5):
        """Standard non-streaming fit (loads all data to GPU at once)."""
        print("Creating character windows...")
        windows = self._prepare_windows(texts, window_size)
        
        print(f"Extracting TF-IDF features from {len(windows)} windows...")
        X = self.vectorizer.fit_transform(windows)
        
        # Flatten labels
        y = np.array([label for seq in label_sequences for label in seq])
        
        # Build label mapping
        unique_labels = np.unique(y)
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Convert labels to indices
        y_idx = np.array([self.label_to_idx[label] for label in y])
        
        # Initialize weights
        num_features = X.shape[1]
        num_classes = len(self.label_to_idx)
        
        # Move to GPU if available
        if GPU_AVAILABLE:
            print("🚀 Moving data to GPU...")
            self.weights = cp.random.randn(num_features, num_classes).astype(cp.float32) * 0.01
            self.bias = cp.zeros(num_classes, dtype=cp.float32)
            y_idx_gpu = cp.array(y_idx)
            print(f"✓ Weights on GPU: {self.weights.device}")
        else:
            self.weights = np.random.randn(num_features, num_classes).astype(np.float32) * 0.01
            self.bias = np.zeros(num_classes, dtype=np.float32)
            y_idx_gpu = y_idx
        
        print(f"Training on {X.shape[0]} samples, {num_features} features, {num_classes} classes...")
        print(f"  Batch size: {self.batch_size}, Iterations: {self.max_iter}, LR: {self.learning_rate}")
        
        # Convert sparse to dense EFFICIENTLY
        print("Preparing data for training...")
        if GPU_AVAILABLE:
            print(f"🚀 Transferring {X.nbytes / 1e9:.2f} GB to GPU...")
            X = cp.asarray(X, dtype=cp.float32)
            print(f"✓ Data on GPU: {X.device}")
        else:
            X = X.astype(np.float32)
        
        # Training loop with mini-batch gradient descent
        num_samples = X.shape[0]
        xp = cp if GPU_AVAILABLE else np  # Use CuPy or NumPy depending on GPU availability
        
        for epoch in tqdm(range(self.max_iter), desc="Training epochs", unit="epoch"):
            # Shuffle data
            indices = xp.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y_idx_gpu[indices]
            
            total_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass (X already dense, no conversion needed)
                logits = X_batch @ self.weights + self.bias
                probs = self._softmax(logits)
                
                # Cross-entropy loss
                batch_size_actual = X_batch.shape[0]
                log_probs = xp.log(probs[xp.arange(batch_size_actual), y_batch] + 1e-10)
                loss = -xp.mean(log_probs)
                
                # Add L2 regularization
                loss += 0.5 * self.regularization * xp.sum(self.weights ** 2)
                total_loss += float(loss) if GPU_AVAILABLE else loss
                num_batches += 1
                
                # Backward pass
                # Gradient of cross-entropy + softmax
                grad_logits = probs.copy()
                grad_logits[xp.arange(batch_size_actual), y_batch] -= 1
                grad_logits /= batch_size_actual
                
                # Gradients
                grad_weights = X_batch.T @ grad_logits + self.regularization * self.weights
                grad_bias = xp.sum(grad_logits, axis=0)
                
                # Update parameters
                self.weights -= self.learning_rate * grad_weights
                self.bias -= self.learning_rate * grad_bias
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.max_iter:
                avg_loss = total_loss / num_batches
                tqdm.write(f"  Epoch {epoch+1}/{self.max_iter}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        
        # Move weights back to CPU for saving/prediction
        if GPU_AVAILABLE:
            print("Moving weights back to CPU...")
            self.weights = cp.asnumpy(self.weights)
            self.bias = cp.asnumpy(self.bias)
        
        print("✓ Training completed")
        return self
    
    def _fit_streaming(self, texts: List[str], label_sequences: List[List[int]], 
                      window_size: int = 5, chunk_size: int = 500):
        """
        Streaming fit that processes data in chunks to avoid memory overflow.
        
        This method:
        1. Fits vocabulary on a SAMPLE of data (to build feature space)
        2. Trains on STREAMING CHUNKS (never loads all data at once)
        3. Uses GPU mini-batches for efficient training
        """
        import gc
        xp = cp if GPU_AVAILABLE else np
        
        print(f"🔄 STREAMING TRAINING MODE (chunk_size={chunk_size})")
        print(f"Total sentences: {len(texts):,}")
        
        # PHASE 1: Build vocabulary from SAMPLE
        print("\n[Phase 1/3] Building vocabulary from sample...")
        vocab_sample_size = min(10000, len(texts) // 10)  # 10k contexts or 10% of data
        sample_texts = texts[:vocab_sample_size]
        sample_windows = self._prepare_windows(sample_texts, window_size)
        print(f"  Sampling {len(sample_windows):,} windows from {vocab_sample_size:,} sentences")
        
        self.vectorizer.fit_transform(sample_windows)  # Build vocabulary
        num_features = len(self.vectorizer.vocabulary)
        print(f"  ✓ Vocabulary: {num_features:,} features")
        del sample_windows
        gc.collect()
        
        # PHASE 2: Build label mapping from ALL labels
        print("\n[Phase 2/3] Building label mapping...")
        all_labels = set()
        for seq in label_sequences:
            all_labels.update(seq)
        unique_labels = sorted(all_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        num_classes = len(self.label_to_idx)
        print(f"  ✓ Classes: {num_classes}")
        
        # Initialize weights on GPU
        if GPU_AVAILABLE:
            print("🚀 Initializing weights on GPU...")
            self.weights = cp.random.randn(num_features, num_classes).astype(cp.float32) * 0.01
            self.bias = cp.zeros(num_classes, dtype=cp.float32)
        else:
            self.weights = np.random.randn(num_features, num_classes).astype(np.float32) * 0.01
            self.bias = np.zeros(num_classes, dtype=np.float32)
        
        # PHASE 3: Streaming training
        print(f"\n[Phase 3/3] Streaming training ({self.max_iter} epochs)...")
        print(f"  Batch size: {self.batch_size}, LR: {self.learning_rate}")
        
        for epoch in tqdm(range(self.max_iter), desc="Training", unit="epoch"):
            total_loss = 0
            num_batches = 0
            
            # Process data in chunks
            for chunk_start in range(0, len(texts), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(texts))
                chunk_texts = texts[chunk_start:chunk_end]
                chunk_labels = label_sequences[chunk_start:chunk_end]
                
                # Transform chunk
                chunk_windows = self._prepare_windows(chunk_texts, window_size)
                X_chunk_sparse = self.vectorizer.transform(chunk_windows, silent=True)
                
                # Convert labels
                y_chunk = np.array([self.label_to_idx[label] for seq in chunk_labels for label in seq])
                
                # Convert sparse to dense first, then move to GPU
                X_chunk_dense = X_chunk_sparse.toarray().astype(np.float32)
                
                if GPU_AVAILABLE:
                    X_chunk = cp.asarray(X_chunk_dense)
                    y_chunk_gpu = cp.array(y_chunk)
                else:
                    X_chunk = X_chunk_dense
                    y_chunk_gpu = y_chunk
                
                # Mini-batch training on this chunk
                num_samples = X_chunk.shape[0]
                indices = xp.random.permutation(num_samples)
                X_chunk = X_chunk[indices]
                y_chunk_gpu = y_chunk_gpu[indices]
                
                for start_idx in range(0, num_samples, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, num_samples)
                    X_batch = X_chunk[start_idx:end_idx]
                    y_batch = y_chunk_gpu[start_idx:end_idx]
                    
                    # Forward pass
                    logits = X_batch @ self.weights + self.bias
                    probs = self._softmax(logits)
                    
                    # Loss
                    batch_size_actual = X_batch.shape[0]
                    log_probs = xp.log(probs[xp.arange(batch_size_actual), y_batch] + 1e-10)
                    loss = -xp.mean(log_probs) + 0.5 * self.regularization * xp.sum(self.weights ** 2)
                    total_loss += float(loss) if GPU_AVAILABLE else loss
                    num_batches += 1
                    
                    # Backward pass
                    grad_logits = probs.copy()
                    grad_logits[xp.arange(batch_size_actual), y_batch] -= 1
                    grad_logits /= batch_size_actual
                    
                    grad_weights = X_batch.T @ grad_logits + self.regularization * self.weights
                    grad_bias = xp.sum(grad_logits, axis=0)
                    
                    # Update
                    self.weights -= self.learning_rate * grad_weights
                    self.bias -= self.learning_rate * grad_bias
                
                # Clean up chunk
                del X_chunk, y_chunk_gpu, chunk_windows
                gc.collect()
            
            # Log progress
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.max_iter:
                avg_loss = total_loss / num_batches
                tqdm.write(f"  Epoch {epoch+1}/{self.max_iter}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        
        # Move weights back to CPU
        if GPU_AVAILABLE:
            print("\nMoving weights back to CPU...")
            self.weights = cp.asnumpy(self.weights)
            self.bias = cp.asnumpy(self.bias)
        
        print("✓ Streaming training completed")
        return self
    
    def predict(self, texts: List[str], window_size: int = 5) -> List[List[int]]:
        """Predict diacritic labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        windows = self._prepare_windows(texts, window_size)
        X = self.vectorizer.transform(windows)
        
        # Weights are already on CPU after training
        # Forward pass (on CPU for inference)
        logits = X @ self.weights + self.bias
        probs = self._softmax(logits)
        y_pred_idx = np.argmax(probs, axis=1)
        
        # Convert back to original labels
        y_pred = np.array([self.idx_to_label[idx] for idx in y_pred_idx])
        
        # Reshape to sequences
        predictions = []
        offset = 0
        for text in texts:
            seq_len = sum(1 for c in text if not c.isspace())
            predictions.append(list(y_pred[offset:offset + seq_len]))
            offset += seq_len
        
        return predictions
    
    def evaluate(self, texts: List[str], label_sequences: List[List[int]], 
                 window_size: int = 5) -> Dict[str, float]:
        """Evaluate model accuracy."""
        predictions = self.predict(texts, window_size)
        
        y_true = np.array([label for seq in label_sequences for label in seq])
        y_pred = np.array([label for seq in predictions for label in seq])
        
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'der': 1.0 - accuracy,
            'correct': int(correct),
            'total': int(total)
        }
    
    def save(self, path: str):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'bias': self.bias,
                'vectorizer': self.vectorizer,
                'label_to_idx': self.label_to_idx,
                'idx_to_label': self.idx_to_label,
                'hyperparams': {
                    'learning_rate': self.learning_rate,
                    'max_iter': self.max_iter,
                    'regularization': self.regularization,
                    'max_features': self.max_features,
                    'ngram_range': self.ngram_range,
                    'batch_size': self.batch_size
                }
            }, f)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.weights = data['weights']
        self.bias = data['bias']
        self.vectorizer = data['vectorizer']
        self.label_to_idx = data['label_to_idx']
        self.idx_to_label = data['idx_to_label']
        
        hp = data['hyperparams']
        self.learning_rate = hp['learning_rate']
        self.max_iter = hp['max_iter']
        self.regularization = hp['regularization']
        self.max_features = hp['max_features']
        self.ngram_range = hp['ngram_range']
        self.batch_size = hp['batch_size']
        
        self.is_fitted = True
        print(f"Model loaded from {path}")
        return self
    
    def apply_diacritics(self, text: str, predicted_labels: List[int]) -> str:
        """Apply predicted diacritics to text."""
        result = []
        label_idx = 0
        
        for char in text:
            if char.isspace():
                result.append(char)
            else:
                result.append(char)
                if predicted_labels[label_idx] != 0:
                    diacritic = ID_TO_DIACRITIC.get(predicted_labels[label_idx], '')
                    result.append(diacritic)
                label_idx += 1
        
        return ''.join(result)
