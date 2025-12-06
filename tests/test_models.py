"""
Unit Tests for Model Modules
"""

import unittest
import numpy as np
import torch
from src.models import LSTMCharModel, CRFModel, MLBaselineModel
from src.config import NUM_DIACRITIC_CLASSES


class TestLSTMCharModel(unittest.TestCase):
    """Test LSTM character model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 50
        self.batch_size = 4
        self.seq_len = 10
        self.model = LSTMCharModel(vocab_size=self.vocab_size)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, LSTMCharModel)
        self.assertEqual(self.model.vocab_size, self.vocab_size)
        self.assertEqual(self.model.num_classes, NUM_DIACRITIC_CLASSES)
    
    def test_forward_pass(self):
        """Test forward pass shape."""
        # Create dummy input
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Forward pass
        logits = self.model(x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, NUM_DIACRITIC_CLASSES)
        self.assertEqual(logits.shape, expected_shape)
    
    def test_predict(self):
        """Test prediction."""
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        predictions = self.model.predict(x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len)
        self.assertEqual(predictions.shape, expected_shape)
        
        # Check that predictions are valid class indices
        self.assertTrue(torch.all(predictions >= 0))
        self.assertTrue(torch.all(predictions < NUM_DIACRITIC_CLASSES))
    
    def test_compute_loss(self):
        """Test loss computation."""
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        targets = torch.randint(0, NUM_DIACRITIC_CLASSES, (self.batch_size, self.seq_len))
        
        logits = self.model(x)
        loss = self.model.compute_loss(logits, targets)
        
        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)
        
        # Check that loss is positive
        self.assertGreater(loss.item(), 0)
    
    def test_compute_loss_with_mask(self):
        """Test loss computation with padding mask."""
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        targets = torch.randint(0, NUM_DIACRITIC_CLASSES, (self.batch_size, self.seq_len))
        mask = torch.ones(self.batch_size, self.seq_len)
        mask[:, 5:] = 0  # Mask half of the sequence
        
        logits = self.model(x)
        loss = self.model.compute_loss(logits, targets, mask)
        
        self.assertGreater(loss.item(), 0)
    
    # TODO: Add test for save/load functionality
    # TODO: Add test for gradient flow
    # TODO: Add test for different configurations


class TestCRFModel(unittest.TestCase):
    """Test CRF model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = CRFModel()
        self.texts = ["مرحبا", "شكرا"]
        self.labels = [[1, 2, 3, 4, 5], [1, 2, 3, 4]]
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, CRFModel)
    
    def test_prepare_features(self):
        """Test feature preparation."""
        features = self.model.prepare_features(self.texts)
        
        self.assertEqual(len(features), len(self.texts))
        self.assertTrue(all(isinstance(f, list) for f in features))
    
    def test_prepare_labels(self):
        """Test label preparation."""
        labels = self.model.prepare_labels(self.labels)
        
        self.assertEqual(len(labels), len(self.labels))
        self.assertTrue(all(isinstance(seq, list) for seq in labels))
        self.assertTrue(all(isinstance(label, str) for seq in labels for label in seq))
    
    # TODO: Add test for training
    # TODO: Add test for prediction
    # TODO: Add test for save/load


class TestMLBaselineModel(unittest.TestCase):
    """Test ML baseline models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.texts = ["مرحبا بك", "شكرا لك"]
        self.labels = [[1, 2, 3, 4, 5, 0, 0, 0], [1, 2, 3, 4, 0, 0, 0, 0]]
    
    def test_logistic_regression_initialization(self):
        """Test Logistic Regression initialization."""
        model = MLBaselineModel(model_type='logistic_regression', feature_type='tfidf')
        self.assertIsInstance(model, MLBaselineModel)
        self.assertEqual(model.model_type, 'logistic_regression')
    
    def test_svm_initialization(self):
        """Test SVM initialization."""
        model = MLBaselineModel(model_type='svm', feature_type='tfidf')
        self.assertIsInstance(model, MLBaselineModel)
        self.assertEqual(model.model_type, 'svm')
    
    def test_prepare_data(self):
        """Test data preparation."""
        model = MLBaselineModel(model_type='logistic_regression', feature_type='tfidf')
        model.feature_extractor.fit(self.texts)
        
        X, y = model.prepare_data(self.texts, self.labels)
        
        self.assertEqual(X.shape[0], len(self.texts))
        self.assertIsNotNone(y)
    
    # TODO: Add test for training
    # TODO: Add test for prediction
    # TODO: Add test for different feature types


if __name__ == '__main__':
    unittest.main()


# TODO: Add integration tests with full training pipeline
# TODO: Add tests for model serialization
# TODO: Add performance benchmarks
# TODO: Add tests for edge cases
# TODO: Add tests for memory usage
