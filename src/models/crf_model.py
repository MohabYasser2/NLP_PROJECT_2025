"""
CRF Model - FROM SCRATCH
Pure NumPy implementation for NLP course.
No sklearn-crfsuite, no ML libraries.
Linear-chain CRF with forward-backward algorithm and Viterbi decoding.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle
from collections import Counter, defaultdict

from src.config import DIACRITIC_TO_ID, ID_TO_DIACRITIC


class CRFFeatureExtractor:
    """
    Feature extractor for CRF model.
    Converts text sequences into sparse feature representations.
    """
    
    def __init__(self):
        self.feature_to_idx = {}
        self.feature_count = 0
        self.is_fitted = False
    
    def _extract_char_features(self, text: str, position: int) -> List[str]:
        """
        Extract features for a character at given position.
        
        Features include:
        - Current character
        - Previous 1-2 characters
        - Next 1-2 characters
        - Character type (letter/space/digit)
        - Position features (beginning/end of word)
        """
        features = []
        n = len(text)
        
        # Current character
        curr_char = text[position] if 0 <= position < n else '<PAD>'
        features.append(f'char={curr_char}')
        
        # Previous characters
        prev1 = text[position-1] if position > 0 else '<BOS>'
        prev2 = text[position-2] if position > 1 else '<BOS>'
        features.append(f'prev1={prev1}')
        features.append(f'prev2={prev2}')
        features.append(f'prev_bigram={prev1}{curr_char}')
        
        # Next characters
        next1 = text[position+1] if position < n-1 else '<EOS>'
        next2 = text[position+2] if position < n-2 else '<EOS>'
        features.append(f'next1={next1}')
        features.append(f'next2={next2}')
        features.append(f'next_bigram={curr_char}{next1}')
        
        # Character type
        if curr_char.isalpha():
            features.append('type=letter')
        elif curr_char.isspace():
            features.append('type=space')
        elif curr_char.isdigit():
            features.append('type=digit')
        else:
            features.append('type=other')
        
        # Word boundaries
        is_word_start = (position == 0) or text[position-1].isspace()
        is_word_end = (position == n-1) or text[position+1].isspace()
        
        if is_word_start:
            features.append('word_start=True')
        if is_word_end:
            features.append('word_end=True')
        
        # Trigram features
        if position > 0 and position < n-1:
            trigram = prev1 + curr_char + next1
            features.append(f'trigram={trigram}')
        
        return features
    
    def text_to_features(self, text: str) -> List[List[str]]:
        """Convert text to list of feature lists (one per character)."""
        return [self._extract_char_features(text, i) for i in range(len(text))]
    
    def fit(self, texts: List[str]):
        """Build feature vocabulary from training texts."""
        feature_counter = Counter()
        
        for text in texts:
            feature_lists = self.text_to_features(text)
            for feat_list in feature_lists:
                feature_counter.update(feat_list)
        
        # Build feature to index mapping
        self.feature_to_idx = {}
        for idx, (feature, _) in enumerate(feature_counter.most_common()):
            self.feature_to_idx[feature] = idx
        
        self.feature_count = len(self.feature_to_idx)
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> List[np.ndarray]:
        """
        Transform texts to sparse feature matrices.
        Returns list of (seq_len, num_features) arrays.
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        results = []
        for text in texts:
            feature_lists = self.text_to_features(text)
            seq_len = len(feature_lists)
            
            # Create sparse representation
            feature_matrix = np.zeros((seq_len, self.feature_count), dtype=np.float32)
            
            for pos, feat_list in enumerate(feature_lists):
                for feat in feat_list:
                    if feat in self.feature_to_idx:
                        feat_idx = self.feature_to_idx[feat]
                        feature_matrix[pos, feat_idx] = 1.0
            
            results.append(feature_matrix)
        
        return results
    
    def fit_transform(self, texts: List[str]) -> List[np.ndarray]:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)


class CRFModel:
    """
    Linear-chain CRF model from scratch.
    Implements forward-backward algorithm, gradient computation, and Viterbi decoding.
    """
    
    def __init__(self, num_labels: int, feature_dim: int, 
                 lr: float = 0.01, max_iter: int = 50, 
                 l2_penalty: float = 0.1):
        """
        Initialize CRF model.
        
        Args:
            num_labels: Number of possible labels
            feature_dim: Dimension of feature vectors
            lr: Learning rate
            max_iter: Maximum training iterations
            l2_penalty: L2 regularization coefficient
        """
        self.num_labels = num_labels
        self.feature_dim = feature_dim
        self.lr = lr
        self.max_iter = max_iter
        self.l2_penalty = l2_penalty
        
        # Parameters
        # Emission weights: (feature_dim, num_labels)
        self.emission_weights = np.random.randn(feature_dim, num_labels) * 0.01
        
        # Transition weights: (num_labels, num_labels)
        # transitions[i, j] = score for transitioning from label i to label j
        self.transition_weights = np.random.randn(num_labels, num_labels) * 0.01
        
        self.is_fitted = False
    
    def _compute_emission_scores(self, features: np.ndarray) -> np.ndarray:
        """
        Compute emission scores for each position.
        
        Args:
            features: (seq_len, feature_dim)
        
        Returns:
            emission_scores: (seq_len, num_labels)
        """
        return features @ self.emission_weights
    
    def _forward_algorithm(self, emission_scores: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm in log space.
        
        Args:
            emission_scores: (seq_len, num_labels)
        
        Returns:
            alpha: (seq_len, num_labels) - forward probabilities
            log_Z: log partition function
        """
        seq_len = emission_scores.shape[0]
        alpha = np.zeros((seq_len, self.num_labels))
        
        # Initialize: alpha[0] = emission_scores[0]
        alpha[0] = emission_scores[0]
        
        # Forward pass
        for t in range(1, seq_len):
            for j in range(self.num_labels):
                # alpha[t, j] = log(sum_i exp(alpha[t-1, i] + transition[i, j])) + emission[t, j]
                scores = alpha[t-1] + self.transition_weights[:, j]
                alpha[t, j] = self._log_sum_exp(scores) + emission_scores[t, j]
        
        # Partition function
        log_Z = self._log_sum_exp(alpha[-1])
        
        return alpha, log_Z
    
    def _backward_algorithm(self, emission_scores: np.ndarray) -> np.ndarray:
        """
        Backward algorithm in log space.
        
        Args:
            emission_scores: (seq_len, num_labels)
        
        Returns:
            beta: (seq_len, num_labels) - backward probabilities
        """
        seq_len = emission_scores.shape[0]
        beta = np.zeros((seq_len, self.num_labels))
        
        # Initialize: beta[-1] = 0 (log(1) = 0)
        beta[-1] = 0
        
        # Backward pass
        for t in range(seq_len - 2, -1, -1):
            for i in range(self.num_labels):
                # beta[t, i] = log(sum_j exp(transition[i, j] + emission[t+1, j] + beta[t+1, j]))
                scores = self.transition_weights[i, :] + emission_scores[t+1] + beta[t+1]
                beta[t, i] = self._log_sum_exp(scores)
        
        return beta
    
    def _log_sum_exp(self, scores: np.ndarray) -> float:
        """Numerically stable log-sum-exp."""
        max_score = np.max(scores)
        return max_score + np.log(np.sum(np.exp(scores - max_score)))
    
    def _compute_log_likelihood(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute log-likelihood for a single sequence.
        
        Args:
            features: (seq_len, feature_dim)
            labels: (seq_len,) - ground truth labels
        
        Returns:
            log_likelihood
        """
        seq_len = len(labels)
        emission_scores = self._compute_emission_scores(features)
        
        # Score of true path
        score = 0.0
        for t in range(seq_len):
            score += emission_scores[t, labels[t]]
            if t > 0:
                score += self.transition_weights[labels[t-1], labels[t]]
        
        # Partition function
        _, log_Z = self._forward_algorithm(emission_scores)
        
        # Log-likelihood = score - log(Z)
        return score - log_Z
    
    def _compute_gradients(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients of log-likelihood w.r.t. parameters.
        
        Args:
            features: (seq_len, feature_dim)
            labels: (seq_len,)
        
        Returns:
            grad_emission: (feature_dim, num_labels)
            grad_transition: (num_labels, num_labels)
        """
        seq_len = len(labels)
        emission_scores = self._compute_emission_scores(features)
        
        # Forward-backward
        alpha, log_Z = self._forward_algorithm(emission_scores)
        beta = self._backward_algorithm(emission_scores)
        
        # Compute marginal probabilities
        # p(y_t = j | x) = exp(alpha[t, j] + beta[t, j] - log_Z)
        marginals = np.exp(alpha + beta - log_Z)
        
        # Compute pairwise marginals for transitions
        # p(y_{t-1} = i, y_t = j | x)
        pairwise_marginals = np.zeros((self.num_labels, self.num_labels))
        
        for t in range(1, seq_len):
            for i in range(self.num_labels):
                for j in range(self.num_labels):
                    score = (alpha[t-1, i] + 
                            self.transition_weights[i, j] + 
                            emission_scores[t, j] + 
                            beta[t, j] - log_Z)
                    pairwise_marginals[i, j] += np.exp(score)
        
        # Gradient of emission weights
        # Expected feature counts under model - observed feature counts
        grad_emission = np.zeros_like(self.emission_weights)
        
        for t in range(seq_len):
            # Expected counts
            for j in range(self.num_labels):
                grad_emission[:, j] -= marginals[t, j] * features[t]
            
            # Observed counts
            grad_emission[:, labels[t]] += features[t]
        
        # Gradient of transition weights
        grad_transition = np.zeros_like(self.transition_weights)
        
        # Expected counts
        grad_transition -= pairwise_marginals
        
        # Observed counts
        for t in range(1, seq_len):
            grad_transition[labels[t-1], labels[t]] += 1
        
        return grad_emission, grad_transition
    
    def fit(self, sequences: List[np.ndarray], labels: List[np.ndarray]):
        """
        Train CRF using gradient ascent.
        
        Args:
            sequences: List of feature matrices, each (seq_len, feature_dim)
            labels: List of label sequences, each (seq_len,)
        """
        print(f"Training CRF: {len(sequences)} sequences, {self.num_labels} labels, {self.feature_dim} features")
        
        for iteration in range(self.max_iter):
            total_ll = 0.0
            
            # Accumulate gradients
            total_grad_emission = np.zeros_like(self.emission_weights)
            total_grad_transition = np.zeros_like(self.transition_weights)
            
            for features, label_seq in zip(sequences, labels):
                # Compute log-likelihood
                ll = self._compute_log_likelihood(features, label_seq)
                total_ll += ll
                
                # Compute gradients
                grad_emission, grad_transition = self._compute_gradients(features, label_seq)
                
                total_grad_emission += grad_emission
                total_grad_transition += grad_transition
            
            # Average gradients
            total_grad_emission /= len(sequences)
            total_grad_transition /= len(sequences)
            
            # Add L2 regularization gradient
            total_grad_emission -= self.l2_penalty * self.emission_weights
            total_grad_transition -= self.l2_penalty * self.transition_weights
            
            # Update parameters (gradient ascent for log-likelihood)
            self.emission_weights += self.lr * total_grad_emission
            self.transition_weights += self.lr * total_grad_transition
            
            avg_ll = total_ll / len(sequences)
            
            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration+1}/{self.max_iter}, Avg Log-Likelihood: {avg_ll:.4f}")
        
        self.is_fitted = True
        print("[OK] CRF training completed")
        return self
    
    def viterbi_decode(self, features: np.ndarray) -> np.ndarray:
        """
        Viterbi algorithm for finding most likely label sequence.
        
        Args:
            features: (seq_len, feature_dim)
        
        Returns:
            best_path: (seq_len,) - most likely label sequence
        """
        seq_len = features.shape[0]
        emission_scores = self._compute_emission_scores(features)
        
        # Viterbi variables
        viterbi = np.zeros((seq_len, self.num_labels))
        backpointer = np.zeros((seq_len, self.num_labels), dtype=np.int32)
        
        # Initialize
        viterbi[0] = emission_scores[0]
        
        # Forward pass
        for t in range(1, seq_len):
            for j in range(self.num_labels):
                # viterbi[t, j] = max_i (viterbi[t-1, i] + transition[i, j]) + emission[t, j]
                scores = viterbi[t-1] + self.transition_weights[:, j]
                best_prev = np.argmax(scores)
                viterbi[t, j] = scores[best_prev] + emission_scores[t, j]
                backpointer[t, j] = best_prev
        
        # Backtrack
        best_path = np.zeros(seq_len, dtype=np.int32)
        best_path[-1] = np.argmax(viterbi[-1])
        
        for t in range(seq_len - 2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]
        
        return best_path
    
    def predict(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """
        Predict label sequences using Viterbi decoding.
        
        Args:
            sequences: List of feature matrices
        
        Returns:
            List of predicted label sequences
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        predictions = []
        for features in sequences:
            pred = self.viterbi_decode(features)
            predictions.append(pred)
        
        return predictions
    
    def evaluate(self, sequences: List[np.ndarray], labels: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate model accuracy.
        
        Args:
            sequences: List of feature matrices
            labels: List of true label sequences
        
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(sequences)
        
        total_chars = 0
        correct_chars = 0
        
        for pred, true in zip(predictions, labels):
            min_len = min(len(pred), len(true))
            total_chars += min_len
            correct_chars += np.sum(pred[:min_len] == true[:min_len])
        
        accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'der': 1.0 - accuracy,
            'correct': int(correct_chars),
            'total': int(total_chars)
        }
    
    def save(self, path: str):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'emission_weights': self.emission_weights,
                'transition_weights': self.transition_weights,
                'num_labels': self.num_labels,
                'feature_dim': self.feature_dim,
                'hyperparams': {
                    'lr': self.lr,
                    'max_iter': self.max_iter,
                    'l2_penalty': self.l2_penalty
                }
            }, f)
        print(f"CRF model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.emission_weights = data['emission_weights']
        self.transition_weights = data['transition_weights']
        self.num_labels = data['num_labels']
        self.feature_dim = data['feature_dim']
        
        hp = data['hyperparams']
        self.lr = hp['lr']
        self.max_iter = hp['max_iter']
        self.l2_penalty = hp['l2_penalty']
        
        self.is_fitted = True
        print(f"CRF model loaded from {path}")
        return self
    
    def apply_diacritics(self, text: str, predicted_labels: np.ndarray) -> str:
        """Apply predicted diacritics to text."""
        result = []
        label_idx = 0
        
        for char in text:
            if char.isspace():
                result.append(char)
            else:
                result.append(char)
                if label_idx < len(predicted_labels):
                    label = predicted_labels[label_idx]
                    diacritic = ID_TO_DIACRITIC.get(label, '')
                    # Only add non-empty diacritics
                    if diacritic:
                        result.append(diacritic)
                    label_idx += 1
        
        return ''.join(result)
