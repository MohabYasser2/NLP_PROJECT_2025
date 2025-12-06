"""
Training Script
Train diacritization models with various configurations.
"""

import argparse
import time
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    TRAIN_FILE, DEV_FILE, MODEL_DIR, HYPERPARAMS, 
    NUM_DIACRITIC_CLASSES
)
from src.preprocessing import prepare_dataset, create_char_vocabulary, encode_sequences
from src.features import CharacterIndexer
from src.models import LSTMCharModel, CRFModel, MLBaselineModel
from src.evaluate import evaluate_model, calculate_der
from src.utils import set_seed, setup_logging, print_stats, get_device, format_time
import logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Arabic Diacritization Model')
    
    parser.add_argument('--model', type=str, default='lstm_char',
                       choices=['lstm_char', 'crf', 'logistic_regression', 'svm'],
                       help='Model type to train')
    
    parser.add_argument('--features', type=str, default='embeddings',
                       choices=['embeddings', 'tfidf', 'bag_of_chars'],
                       help='Feature type (for ML models)')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save trained model')
    
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    
    return parser.parse_args()


def train_lstm_model(args):
    """
    Train LSTM character model.
    
    Args:
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)
    logger.info("Training LSTM Character Model")
    
    # Load and prepare data
    logger.info("Loading datasets...")
    train_texts, train_labels = prepare_dataset(TRAIN_FILE)
    dev_texts, dev_labels = prepare_dataset(DEV_FILE)
    
    print_stats(train_texts, "Training Set")
    print_stats(dev_texts, "Dev Set")
    
    # Create vocabulary
    logger.info("Building vocabulary...")
    char_indexer = CharacterIndexer()
    char_indexer.fit(train_texts + dev_texts)
    vocab_size = char_indexer.vocab_size
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Encode sequences
    logger.info("Encoding sequences...")
    train_encoded = char_indexer.encode_batch(train_texts)
    dev_encoded = char_indexer.encode_batch(dev_texts)
    
    # Pad labels to match encoded sequences
    from src.utils import pad_sequences
    train_labels_padded = pad_sequences(train_labels, max_len=train_encoded.shape[1])
    dev_labels_padded = pad_sequences(dev_labels, max_len=dev_encoded.shape[1])
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.LongTensor(train_encoded),
        torch.LongTensor(train_labels_padded)
    )
    dev_dataset = TensorDataset(
        torch.LongTensor(dev_encoded),
        torch.LongTensor(dev_labels_padded)
    )
    
    # Create data loaders
    batch_size = args.batch_size or HYPERPARAMS['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = args.device or get_device()
    logger.info(f"Using device: {device}")
    
    model = LSTMCharModel(vocab_size=vocab_size)
    model = model.to(device)
    
    # Optimizer
    learning_rate = args.learning_rate or HYPERPARAMS['lstm']['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    num_epochs = args.epochs or HYPERPARAMS['num_epochs']
    best_der = float('inf')
    patience = HYPERPARAMS['early_stopping_patience']
    patience_counter = 0
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Create mask for padding
            mask = (inputs != 0).float()
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(inputs)
            loss = model.compute_loss(logits, targets, mask)
            
            # Backward pass
            loss.backward()
            
            # TODO: Add gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Batch [{batch_idx+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluation on dev set
        model.eval()
        dev_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in dev_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                mask = (inputs != 0).float()
                
                logits = model(inputs)
                loss = model.compute_loss(logits, targets, mask)
                dev_loss += loss.item()
                
                predictions = model.predict(inputs)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_dev_loss = dev_loss / len(dev_loader)
        
        # Calculate DER
        der = calculate_der(all_predictions, all_targets, all_predictions)
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] completed in {format_time(epoch_time)}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Dev Loss: {avg_dev_loss:.4f}, DER: {der:.4f}")
        
        # Save best model
        if der < best_der:
            best_der = der
            patience_counter = 0
            
            save_path = args.save_path or MODEL_DIR / 'lstm_char_best.pt'
            model.save(str(save_path))
            logger.info(f"Best model saved with DER: {best_der:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info("Early stopping triggered!")
            break
    
    logger.info(f"Training completed! Best DER: {best_der:.4f}")


def train_crf_model(args):
    """
    Train CRF model.
    
    Args:
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)
    logger.info("Training CRF Model")
    
    # Load data
    logger.info("Loading datasets...")
    train_texts, train_labels = prepare_dataset(TRAIN_FILE)
    dev_texts, dev_labels = prepare_dataset(DEV_FILE)
    
    print_stats(train_texts, "Training Set")
    print_stats(dev_texts, "Dev Set")
    
    # Initialize model
    model = CRFModel()
    
    # Train
    logger.info("Training CRF...")
    model.fit(train_texts, train_labels)
    
    # Evaluate
    logger.info("Evaluating on dev set...")
    predictions = model.predict(dev_texts)
    der = calculate_der(predictions, dev_labels, train_labels)
    
    logger.info(f"Dev DER: {der:.4f}")
    
    # Save model
    save_path = args.save_path or MODEL_DIR / 'crf_best.pkl'
    model.save(str(save_path))
    logger.info(f"Model saved to {save_path}")


def train_ml_model(args):
    """
    Train ML baseline model.
    
    Args:
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Training {args.model.upper()} Model with {args.features} features")
    
    # Load data
    logger.info("Loading datasets...")
    train_texts, train_labels = prepare_dataset(TRAIN_FILE)
    dev_texts, dev_labels = prepare_dataset(DEV_FILE)
    
    print_stats(train_texts, "Training Set")
    print_stats(dev_texts, "Dev Set")
    
    # Initialize model
    model = MLBaselineModel(model_type=args.model, feature_type=args.features)
    
    # Train
    logger.info("Training model...")
    model.fit(train_texts, train_labels)
    
    # Evaluate
    logger.info("Evaluating on dev set...")
    predictions = model.predict(dev_texts)
    der = calculate_der(predictions, dev_labels, train_labels)
    
    logger.info(f"Dev DER: {der:.4f}")
    
    # Save model
    save_path = args.save_path or MODEL_DIR / f'{args.model}_{args.features}_best.pkl'
    model.save(str(save_path))
    logger.info(f"Model saved to {save_path}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Arabic Diacritization Training")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Features: {args.features}")
    logger.info(f"Random Seed: {args.seed}")
    
    # Train based on model type
    if args.model == 'lstm_char':
        train_lstm_model(args)
    elif args.model == 'crf':
        train_crf_model(args)
    elif args.model in ['logistic_regression', 'svm']:
        train_ml_model(args)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    logger.info("Training script finished!")


if __name__ == '__main__':
    main()


# TODO: Add support for distributed training
# TODO: Add TensorBoard/W&B logging
# TODO: Implement learning rate scheduling
# TODO: Add model checkpointing at regular intervals
# TODO: Add support for resuming training from checkpoint
# TODO: Implement data augmentation during training
# TODO: Add mixed precision training for faster training
# TODO: Add support for multi-GPU training
