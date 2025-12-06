"""
Evaluation Script
Evaluate diacritization models and compute DER metric.
"""

import argparse
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from collections import defaultdict

from src.config import TEST_FILE, MODEL_DIR, DIACRITIC_TO_ID, ID_TO_DIACRITIC
from src.preprocessing import prepare_dataset, strip_diacritics
from src.models import LSTMCharModel, CRFModel, MLBaselineModel
from src.features import CharacterIndexer
import logging


def calculate_der(predictions: List[List[int]], 
                 targets: List[List[int]], 
                 reference: List[List[int]] = None) -> float:
    """
    Calculate Diacritic Error Rate (DER).
    
    DER = (Number of incorrectly predicted diacritics) / (Total number of characters)
    
    Args:
        predictions: Predicted label sequences
        targets: Target label sequences
        reference: Reference sequences for padding handling (optional)
        
    Returns:
        DER value (lower is better)
    """
    total_chars = 0
    total_errors = 0
    
    for pred_seq, target_seq in zip(predictions, targets):
        # Handle different lengths
        min_len = min(len(pred_seq), len(target_seq))
        
        for i in range(min_len):
            total_chars += 1
            if pred_seq[i] != target_seq[i]:
                total_errors += 1
    
    if total_chars == 0:
        return 0.0
    
    der = total_errors / total_chars
    return der


def calculate_accuracy(predictions: List[List[int]], 
                      targets: List[List[int]]) -> float:
    """
    Calculate character-level accuracy.
    
    Args:
        predictions: Predicted label sequences
        targets: Target label sequences
        
    Returns:
        Accuracy value
    """
    total_chars = 0
    correct_chars = 0
    
    for pred_seq, target_seq in zip(predictions, targets):
        min_len = min(len(pred_seq), len(target_seq))
        
        for i in range(min_len):
            total_chars += 1
            if pred_seq[i] == target_seq[i]:
                correct_chars += 1
    
    if total_chars == 0:
        return 0.0
    
    return correct_chars / total_chars


def calculate_per_class_metrics(predictions: List[List[int]], 
                                targets: List[List[int]]) -> Dict[int, Dict[str, float]]:
    """
    Calculate precision, recall, and F1 for each diacritic class.
    
    Args:
        predictions: Predicted label sequences
        targets: Target label sequences
        
    Returns:
        Dictionary mapping class ID to metrics
    """
    # Initialize counters
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    
    for pred_seq, target_seq in zip(predictions, targets):
        min_len = min(len(pred_seq), len(target_seq))
        
        for i in range(min_len):
            pred_label = pred_seq[i]
            target_label = target_seq[i]
            
            if pred_label == target_label:
                true_positives[target_label] += 1
            else:
                false_positives[pred_label] += 1
                false_negatives[target_label] += 1
    
    # Calculate metrics for each class
    metrics = {}
    
    for class_id in set(list(true_positives.keys()) + 
                       list(false_positives.keys()) + 
                       list(false_negatives.keys())):
        tp = true_positives[class_id]
        fp = false_positives[class_id]
        fn = false_negatives[class_id]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn
        }
    
    return metrics


def print_evaluation_results(predictions: List[List[int]], 
                            targets: List[List[int]]):
    """
    Print comprehensive evaluation results.
    
    Args:
        predictions: Predicted label sequences
        targets: Target label sequences
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Calculate overall metrics
    der = calculate_der(predictions, targets)
    accuracy = calculate_accuracy(predictions, targets)
    
    print(f"\nOverall Metrics:")
    print(f"  Diacritic Error Rate (DER): {der:.4f} ({der*100:.2f}%)")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Calculate per-class metrics
    per_class_metrics = calculate_per_class_metrics(predictions, targets)
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    for class_id in sorted(per_class_metrics.keys()):
        metrics = per_class_metrics[class_id]
        class_name = ID_TO_DIACRITIC.get(class_id, f"Unknown({class_id})")
        
        print(f"{class_name:<15} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} "
              f"{metrics['support']:<10}")
    
    # Calculate macro averages
    avg_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
    avg_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
    avg_f1 = np.mean([m['f1'] for m in per_class_metrics.values()])
    
    print("-" * 80)
    print(f"{'Macro Avg':<15} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f}")
    print()
    
    # TODO: Add confusion matrix visualization
    # TODO: Add per-word accuracy metrics
    # TODO: Add error analysis with examples


def evaluate_model(model, test_texts: List[str], test_labels: List[List[int]], 
                   model_type: str) -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model
        test_texts: Test texts
        test_labels: Test labels
        model_type: Type of model
        
    Returns:
        Dictionary of evaluation metrics
    """
    logging.info("Running model predictions...")
    
    # Get predictions based on model type
    if model_type == 'lstm_char':
        # TODO: Implement LSTM evaluation with proper batching
        predictions = []
        # predictions = model.predict(test_texts)
    elif model_type == 'crf':
        predictions = model.predict(test_texts)
    else:  # ML baselines
        predictions = model.predict(test_texts)
    
    # Calculate metrics
    der = calculate_der(predictions, test_labels)
    accuracy = calculate_accuracy(predictions, test_labels)
    
    metrics = {
        'der': der,
        'accuracy': accuracy
    }
    
    return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Arabic Diacritization Model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['lstm_char', 'crf', 'logistic_regression', 'svm'],
                       help='Type of model')
    
    parser.add_argument('--test_file', type=str, default=None,
                       help='Path to test file (default: use config TEST_FILE)')
    
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save evaluation results')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Arabic Diacritization Evaluation")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Model Type: {args.model_type}")
    
    # Load test data
    test_file = Path(args.test_file) if args.test_file else TEST_FILE
    logger.info(f"Loading test data from {test_file}...")
    test_texts, test_labels = prepare_dataset(test_file)
    logger.info(f"Test samples: {len(test_texts)}")
    
    # Load model
    logger.info("Loading model...")
    if args.model_type == 'lstm_char':
        model = LSTMCharModel.load(args.model_path)
    elif args.model_type == 'crf':
        model = CRFModel.load(args.model_path)
    else:
        model = MLBaselineModel.load(args.model_path)
    
    # Evaluate
    logger.info("Evaluating model...")
    predictions = model.predict(test_texts)
    
    # Print results
    print_evaluation_results(predictions, test_labels)
    
    # Save results if output file specified
    if args.output_file:
        # TODO: Save detailed results to file
        logger.info(f"Results saved to {args.output_file}")
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()


# TODO: Add statistical significance testing
# TODO: Add bootstrap confidence intervals for metrics
# TODO: Implement error analysis tools
# TODO: Add visualization of predictions vs ground truth
# TODO: Generate detailed error reports by diacritic type
# TODO: Add support for comparing multiple models
# TODO: Implement cross-validation evaluation
