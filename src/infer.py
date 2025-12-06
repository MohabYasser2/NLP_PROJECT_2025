"""
Inference Script
Diacritize raw Arabic text using trained models.
"""

import argparse
import torch
from pathlib import Path
from typing import List

from src.config import MODEL_DIR, ID_TO_DIACRITIC
from src.preprocessing import strip_diacritics, clean_arabic_text
from src.models import LSTMCharModel, CRFModel, MLBaselineModel
from src.features import CharacterIndexer


def diacritize_sentence(model, text: str, model_type: str, 
                       char_indexer: CharacterIndexer = None) -> str:
    """
    Diacritize a single sentence.
    
    Args:
        model: Trained model
        text: Input text (without diacritics)
        model_type: Type of model
        char_indexer: Character indexer (for LSTM models)
        
    Returns:
        Diacritized text
    """
    # Clean and strip diacritics
    text = clean_arabic_text(text)
    text = strip_diacritics(text)
    
    if len(text) == 0:
        return text
    
    # Get predictions based on model type
    if model_type == 'lstm_char':
        if char_indexer is None:
            raise ValueError("char_indexer is required for LSTM models")
        
        # Encode text
        encoded = char_indexer.encode_batch([text])
        encoded_tensor = torch.LongTensor(encoded)
        
        # Predict
        model.eval()
        with torch.no_grad():
            predictions = model.predict(encoded_tensor)
        
        # Get predicted labels for the sentence
        pred_labels = predictions[0][:len(text)].cpu().numpy().tolist()
    
    elif model_type == 'crf':
        predictions = model.predict([text])
        pred_labels = predictions[0]
    
    else:  # ML baselines
        predictions = model.predict([text])
        pred_labels = predictions[0]
    
    # Reconstruct diacritized text
    diacritized = reconstruct_diacritized_text(text, pred_labels)
    
    return diacritized


def reconstruct_diacritized_text(text: str, labels: List[int]) -> str:
    """
    Reconstruct diacritized text from characters and labels.
    
    Args:
        text: Clean text without diacritics
        labels: Predicted diacritic labels
        
    Returns:
        Diacritized text
    """
    result = []
    
    for i, char in enumerate(text):
        result.append(char)
        
        # Add diacritic if not 'no diacritic' label
        if i < len(labels):
            label_id = labels[i]
            if label_id in ID_TO_DIACRITIC:
                diacritic = ID_TO_DIACRITIC[label_id]
                if diacritic != '_':  # '_' means no diacritic
                    result.append(diacritic)
    
    return ''.join(result)


def diacritize_file(model, input_path: Path, output_path: Path, 
                   model_type: str, char_indexer: CharacterIndexer = None):
    """
    Diacritize all sentences in a file.
    
    Args:
        model: Trained model
        input_path: Input file path (raw text)
        output_path: Output file path (diacritized text)
        model_type: Type of model
        char_indexer: Character indexer (for LSTM models)
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Reading from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Diacritizing {len(lines)} lines...")
    diacritized_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if len(line) == 0:
            diacritized_lines.append('')
            continue
        
        diacritized = diacritize_sentence(model, line, model_type, char_indexer)
        diacritized_lines.append(diacritized)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(lines)} lines...")
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in diacritized_lines:
            f.write(line + '\n')
    
    print("Diacritization completed!")


def interactive_mode(model, model_type: str, char_indexer: CharacterIndexer = None):
    """
    Interactive diacritization mode.
    
    Args:
        model: Trained model
        model_type: Type of model
        char_indexer: Character indexer (for LSTM models)
    """
    print("\n" + "=" * 80)
    print("Interactive Diacritization Mode")
    print("=" * 80)
    print("Enter Arabic text to diacritize (or 'quit' to exit)")
    print()
    
    while True:
        text = input(">>> ")
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if len(text.strip()) == 0:
            continue
        
        try:
            diacritized = diacritize_sentence(model, text, model_type, char_indexer)
            print(f"Diacritized: {diacritized}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("Goodbye!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Diacritize Arabic Text')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['lstm_char', 'crf', 'logistic_regression', 'svm'],
                       help='Type of model')
    
    parser.add_argument('--input_file', type=str, default=None,
                       help='Input file with raw text')
    
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for diacritized text')
    
    parser.add_argument('--text', type=str, default=None,
                       help='Single text to diacritize')
    
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to vocabulary file (for LSTM models)')
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    print("=" * 80)
    print("Arabic Diacritization Inference")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Model Type: {args.model_type}")
    
    # Load model
    print("Loading model...")
    if args.model_type == 'lstm_char':
        model = LSTMCharModel.load(args.model_path)
        
        # TODO: Load vocabulary from saved file
        # For now, create a new indexer (in production, save/load vocab)
        char_indexer = CharacterIndexer()
        print("Warning: Using new character indexer. In production, load saved vocabulary.")
    elif args.model_type == 'crf':
        model = CRFModel.load(args.model_path)
        char_indexer = None
    else:
        model = MLBaselineModel.load(args.model_path)
        char_indexer = None
    
    print("Model loaded successfully!")
    
    # Choose mode
    if args.interactive:
        interactive_mode(model, args.model_type, char_indexer)
    
    elif args.text:
        # Single text
        diacritized = diacritize_sentence(model, args.text, args.model_type, char_indexer)
        print(f"\nInput:  {args.text}")
        print(f"Output: {diacritized}\n")
    
    elif args.input_file and args.output_file:
        # File processing
        input_path = Path(args.input_file)
        output_path = Path(args.output_file)
        diacritize_file(model, input_path, output_path, args.model_type, char_indexer)
    
    else:
        print("Error: Please specify --interactive, --text, or --input_file and --output_file")
        return
    
    print("Inference completed!")


if __name__ == '__main__':
    main()


# TODO: Add batch processing for faster inference
# TODO: Implement beam search for better predictions
# TODO: Add confidence scores for predictions
# TODO: Support for processing multiple files at once
# TODO: Add API endpoint for web service integration
# TODO: Implement caching for repeated text
# TODO: Add support for streaming inference
# TODO: Optimize inference speed with quantization/pruning
