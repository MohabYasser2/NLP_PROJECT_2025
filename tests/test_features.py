"""
Unit Tests for Feature Extraction Module
"""

import unittest
import numpy as np
from src.features import (
    CharacterIndexer, TfidfFeatureExtractor, BagOfCharactersExtractor,
    ContextualFeatureExtractor
)


class TestCharacterIndexer(unittest.TestCase):
    """Test CharacterIndexer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.indexer = CharacterIndexer()
        self.texts = ["مرحبا", "شكرا"]
    
    def test_fit(self):
        """Test vocabulary fitting."""
        self.indexer.fit(self.texts)
        
        # Check that vocab size is correct
        unique_chars = set(''.join(self.texts))
        expected_size = len(unique_chars) + 2  # +2 for <PAD> and <UNK>
        self.assertEqual(self.indexer.vocab_size, expected_size)
    
    def test_encode(self):
        """Test text encoding."""
        self.indexer.fit(self.texts)
        
        text = "مرحبا"
        encoded = self.indexer.encode(text)
        
        self.assertEqual(len(encoded), len(text))
        self.assertTrue(all(isinstance(x, int) for x in encoded))
    
    def test_decode(self):
        """Test sequence decoding."""
        self.indexer.fit(self.texts)
        
        text = "مرحبا"
        encoded = self.indexer.encode(text)
        decoded = self.indexer.decode(encoded)
        
        self.assertEqual(text, decoded)
    
    def test_encode_batch(self):
        """Test batch encoding with padding."""
        self.indexer.fit(self.texts)
        
        batch = self.indexer.encode_batch(self.texts)
        
        self.assertEqual(batch.shape[0], len(self.texts))
        self.assertTrue(isinstance(batch, np.ndarray))
    
    # TODO: Add test for unknown characters
    # TODO: Add test for empty sequences


class TestTfidfFeatureExtractor(unittest.TestCase):
    """Test TF-IDF feature extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = TfidfFeatureExtractor(max_features=100)
        self.texts = ["مرحبا بك", "شكرا لك", "أهلا"]
    
    def test_fit_transform(self):
        """Test fit and transform."""
        features = self.extractor.fit_transform(self.texts)
        
        self.assertEqual(features.shape[0], len(self.texts))
        self.assertTrue(isinstance(features, np.ndarray))
    
    def test_transform(self):
        """Test transform on new data."""
        self.extractor.fit(self.texts)
        
        new_texts = ["مرحبا"]
        features = self.extractor.transform(new_texts)
        
        self.assertEqual(features.shape[0], len(new_texts))
    
    # TODO: Add test for feature dimensionality
    # TODO: Add test for empty texts


class TestBagOfCharactersExtractor(unittest.TestCase):
    """Test Bag of Characters feature extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = BagOfCharactersExtractor(max_features=100)
        self.texts = ["مرحبا", "شكرا", "أهلا"]
    
    def test_fit_transform(self):
        """Test fit and transform."""
        features = self.extractor.fit_transform(self.texts)
        
        self.assertEqual(features.shape[0], len(self.texts))
        self.assertTrue(isinstance(features, np.ndarray))
    
    # TODO: Add more test cases


class TestContextualFeatureExtractor(unittest.TestCase):
    """Test contextual feature extraction for CRF."""
    
    def test_char_features(self):
        """Test character feature extraction."""
        text = "مرحبا"
        features = ContextualFeatureExtractor.char_features(text, 0)
        
        # Check that features are extracted
        self.assertIsInstance(features, dict)
        self.assertIn('char', features)
    
    def test_text_to_features(self):
        """Test feature extraction for entire text."""
        text = "مرحبا"
        features = ContextualFeatureExtractor.text_to_features(text)
        
        self.assertEqual(len(features), len(text))
        self.assertTrue(all(isinstance(f, dict) for f in features))
    
    def test_boundary_features(self):
        """Test beginning/end of sequence features."""
        text = "مرحبا"
        
        # First character should have BOS
        first_features = ContextualFeatureExtractor.char_features(text, 0)
        self.assertTrue(first_features.get('BOS', False))
        
        # Last character should have EOS
        last_features = ContextualFeatureExtractor.char_features(text, len(text) - 1)
        self.assertTrue(last_features.get('EOS', False))
    
    # TODO: Add tests for n-gram features
    # TODO: Add tests for contextual features


if __name__ == '__main__':
    unittest.main()


# TODO: Add integration tests
# TODO: Add performance benchmarks
# TODO: Add tests for edge cases
# TODO: Add tests for embedding loader
