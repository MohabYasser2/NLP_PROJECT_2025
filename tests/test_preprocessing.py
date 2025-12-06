"""
Unit Tests for Preprocessing Module
"""

import unittest
from src.preprocessing import (
    strip_diacritics, extract_labels, extract_labels_simple,
    tokenize_characters, tokenize_words, clean_arabic_text,
    create_char_vocabulary, encode_sequences
)
from src.config import ARABIC_DIACRITICS, DIACRITIC_TO_ID


class TestStripDiacritics(unittest.TestCase):
    """Test diacritic stripping functionality."""
    
    def test_strip_simple(self):
        """Test stripping diacritics from simple text."""
        text = "مَرْحَبًا"
        expected = "مرحبا"
        result = strip_diacritics(text)
        self.assertEqual(result, expected)
    
    def test_strip_no_diacritics(self):
        """Test text without diacritics."""
        text = "مرحبا"
        result = strip_diacritics(text)
        self.assertEqual(result, text)
    
    def test_strip_empty(self):
        """Test empty string."""
        text = ""
        result = strip_diacritics(text)
        self.assertEqual(result, "")
    
    # TODO: Add more test cases
    # TODO: Test with different types of diacritics
    # TODO: Test with mixed Arabic and English text


class TestExtractLabels(unittest.TestCase):
    """Test label extraction functionality."""
    
    def test_extract_simple(self):
        """Test simple label extraction."""
        text = "مَرْحَبًا"
        clean_text, labels = extract_labels_simple(text)
        
        # Check that diacritics are removed
        self.assertNotIn('\u064E', clean_text)  # Fatha
        
        # Check that labels are extracted
        self.assertEqual(len(clean_text), len(labels))
    
    def test_extract_no_diacritics(self):
        """Test text without diacritics."""
        text = "مرحبا"
        clean_text, labels = extract_labels_simple(text)
        
        self.assertEqual(clean_text, text)
        self.assertEqual(len(labels), len(text))
        
        # All labels should be 'no diacritic'
        no_diac_id = DIACRITIC_TO_ID['_']
        self.assertTrue(all(label == no_diac_id for label in labels))
    
    # TODO: Add more test cases
    # TODO: Test with multiple diacritics per character
    # TODO: Test edge cases


class TestTokenization(unittest.TestCase):
    """Test tokenization functions."""
    
    def test_character_tokenization(self):
        """Test character-level tokenization."""
        text = "مرحبا"
        tokens = tokenize_characters(text)
        
        self.assertEqual(len(tokens), len(text))
        self.assertEqual(tokens, list(text))
    
    def test_word_tokenization(self):
        """Test word-level tokenization."""
        text = "مرحبا بك"
        tokens = tokenize_words(text)
        
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens, ["مرحبا", "بك"])
    
    def test_empty_tokenization(self):
        """Test tokenization of empty string."""
        tokens_char = tokenize_characters("")
        tokens_word = tokenize_words("")
        
        self.assertEqual(len(tokens_char), 0)
        self.assertEqual(len(tokens_word), 1)  # split() returns [''] for empty string
    
    # TODO: Add more test cases


class TestCleanArabicText(unittest.TestCase):
    """Test Arabic text cleaning."""
    
    def test_clean_basic(self):
        """Test basic cleaning."""
        text = "مرحبا  بك"
        result = clean_arabic_text(text)
        
        """
        Unit Tests for Preprocessing Module
        """

        import unittest
        from src.preprocessing import (
            strip_diacritics, extract_labels, extract_labels_simple,
            tokenize_characters, tokenize_words, clean_arabic_text,
            create_char_vocabulary, encode_sequences
        )
        from src.config import ARABIC_DIACRITICS, DIACRITIC_TO_ID


        class TestStripDiacritics(unittest.TestCase):
            """Test diacritic stripping functionality."""
    
            def test_strip_simple(self):
                """Test stripping diacritics from simple text."""
                text = "مَرْحَبًا"
                expected = "مرحبا"
                result = strip_diacritics(text)
                self.assertEqual(result, expected)
    
            def test_strip_no_diacritics(self):
                """Test text without diacritics."""
                text = "مرحبا"
                result = strip_diacritics(text)
                self.assertEqual(result, text)
    
            def test_strip_empty(self):
                """Test empty string."""
                text = ""
                result = strip_diacritics(text)
                self.assertEqual(result, "")
    


        class TestExtractLabels(unittest.TestCase):
            """Test label extraction functionality."""
    
            def test_extract_simple(self):
                """Test simple label extraction."""
                text = "مَرْحَبًا"
                clean_text, labels = extract_labels_simple(text)
        
                # Check that diacritics are removed
                self.assertNotIn('\u064E', clean_text)  # Fatha
        
                # Check that labels are extracted
                self.assertEqual(len(clean_text), len(labels))
    
            def test_extract_no_diacritics(self):
                """Test text without diacritics."""
                text = "مرحبا"
                clean_text, labels = extract_labels_simple(text)
        
                self.assertEqual(clean_text, text)
                self.assertEqual(len(labels), len(text))
        
                # All labels should be 'no diacritic'
                no_diac_id = DIACRITIC_TO_ID['_']
                self.assertTrue(all(label == no_diac_id for label in labels))


        class TestTokenization(unittest.TestCase):
            """Test tokenization functions."""
    
            def test_character_tokenization(self):
                """Test character-level tokenization."""
                text = "مرحبا"
                tokens = tokenize_characters(text)
        
                self.assertEqual(len(tokens), len(text))
                self.assertEqual(tokens, list(text))
    
            def test_word_tokenization(self):
                """Test word-level tokenization."""
                text = "مرحبا بك"
                tokens = tokenize_words(text)
        
                self.assertEqual(len(tokens), 2)
                self.assertEqual(tokens, ["مرحبا", "بك"])
    
            def test_empty_tokenization(self):
                """Test tokenization of empty string."""
                tokens_char = tokenize_characters("")
                tokens_word = tokenize_words("")
        
                self.assertEqual(len(tokens_char), 0)
                self.assertEqual(len(tokens_word), 1)  # split() returns [''] for empty string


        class TestCleanArabicText(unittest.TestCase):
            """Test Arabic text cleaning."""
    
            def test_clean_basic(self):
                """Test basic cleaning."""
                text = "مرحبا  بك"
                result = clean_arabic_text(text)
        
                # Check whitespace normalization
                self.assertNotIn("  ", result)
    
            def test_remove_non_arabic(self):
                """Test removal of non-Arabic characters."""
                text = "مرحبا123"
                result = clean_arabic_text(text)
        
                self.assertNotIn("1", result)
                self.assertNotIn("2", result)
                self.assertNotIn("3", result)


        class TestVocabularyAndEncoding(unittest.TestCase):
            """Test vocabulary creation and sequence encoding."""
    
            def test_create_vocabulary(self):
                """Test vocabulary creation."""
                texts = ["مرحبا", "شكرا"]
                vocab = create_char_vocabulary(texts)
        
                # Check special tokens
                self.assertIn('<PAD>', vocab)
                self.assertIn('<UNK>', vocab)
        
                # Check that all characters are included
                all_chars = set(''.join(texts))
                for char in all_chars:
                    self.assertIn(char, vocab)
    
            def test_encode_sequences(self):
                """Test sequence encoding."""
                texts = ["مرحبا"]
                vocab = create_char_vocabulary(texts)
                encoded = encode_sequences(texts, vocab)
        
                self.assertEqual(len(encoded), len(texts))
                self.assertEqual(len(encoded[0]), len(texts[0]))


        if __name__ == '__main__':
            unittest.main()
