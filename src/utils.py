"""
Utility Functions Module
General-purpose utilities for the project.
"""

import random
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Any
from src.config import LOG_FILE, LOG_LEVEL


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # TODO: Add TensorFlow seed setting if needed


def setup_logging(log_file: Path = LOG_FILE, level: str = LOG_LEVEL):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def pad_sequences(sequences: List[List[int]], max_len: int = None, 
                  padding_value: int = 0) -> np.ndarray:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of sequences
        max_len: Maximum length (if None, use longest sequence)
        padding_value: Value to use for padding
        
    Returns:
        Padded array of shape (num_sequences, max_len)
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded = np.full((len(sequences), max_len), padding_value, dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    
    return padded


def create_batches(data: List[Any], batch_size: int, shuffle: bool = True) -> List[List[Any]]:
    """
    Create batches from data.
    
    Args:
        data: Input data
        batch_size: Size of each batch
        shuffle: Whether to shuffle data
        
    Returns:
        List of batches
    """
    if shuffle:
        data = data.copy()
        random.shuffle(data)
    
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i + batch_size])
    
    return batches


def save_model(model: Any, path: Path):
    """
    Save model to disk.
    
    Args:
        model: Model object
        path: Save path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # TODO: Implement model-specific saving logic
    # For PyTorch: torch.save(model.state_dict(), path)
    # For sklearn: joblib.dump(model, path)
    
    raise NotImplementedError("Model saving not implemented yet")


def load_model(path: Path, model_class: Any = None) -> Any:
    """
    Load model from disk.
    
    Args:
        path: Model path
        model_class: Model class (for PyTorch)
        
    Returns:
        Loaded model
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # TODO: Implement model-specific loading logic
    
    raise NotImplementedError("Model loading not implemented yet")


def print_stats(data: List[str], name: str = "Dataset"):
    """
    Print statistics about dataset.
    
    Args:
        data: List of text samples
        name: Dataset name
    """
    num_samples = len(data)
    lengths = [len(text) for text in data]
    
    print(f"\n{name} Statistics:")
    print(f"  Number of samples: {num_samples}")
    print(f"  Min length: {min(lengths)}")
    print(f"  Max length: {max(lengths)}")
    print(f"  Mean length: {np.mean(lengths):.2f}")
    print(f"  Median length: {np.median(lengths):.2f}")
    print(f"  Total characters: {sum(lengths)}")


def calculate_class_weights(labels: List[int], num_classes: int) -> np.ndarray:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: List of label indices
        num_classes: Number of classes
        
    Returns:
        Array of class weights
    """
    # Count occurrences of each class
    counts = np.bincount(labels, minlength=num_classes)
    
    # Avoid division by zero
    counts = np.maximum(counts, 1)
    
    # Calculate weights (inverse frequency)
    total = len(labels)
    weights = total / (num_classes * counts)
    
    return weights


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_device():
    """
    Get available device (CUDA, MPS, or CPU).
    
    Returns:
        Device string
    """
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'


# TODO: Add function for progress bar display
# TODO: Add function for model checkpoint management
# TODO: Add function for early stopping logic
# TODO: Add function for learning rate scheduling
# TODO: Add function for experiment tracking (W&B, MLflow)
