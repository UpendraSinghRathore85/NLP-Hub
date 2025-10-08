"""
Tests for NLP-Hub modules
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp_hub.preprocessing import TextPreprocessor, Tokenizer
from nlp_hub.sentiment import SentimentAnalyzer
from nlp_hub.chatbot import ChatBot
from nlp_hub.visualization import NLPVisualizer


class TestTextPreprocessing(unittest.TestCase):
    """Test text preprocessing functionality"""
    
    def setUp(self):
        self.preprocessor = TextPreprocessor()
        self.tokenizer = Tokenizer()
    
    def test_clean_text(self):
        text = "Hello, World! This has 123 numbers."
        cleaned = self.preprocessor.clean_text(text, remove_punctuation=True, remove_numbers=True)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("123", cleaned)
        self.assertIn("hello", cleaned.lower())
    
    def test_preprocess_text(self):
        text = "This is a test sentence with some words."
        tokens = self.preprocessor.preprocess_text(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
    
    def test_word_tokenize(self):
        text = "Hello world! How are you?"
        tokens = self.tokenizer.word_tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertIn("Hello", tokens)
    
    def test_sentence_tokenize(self):
        text = "First sentence. Second sentence! Third sentence?"
        sentences = self.tokenizer.sentence_tokenize(text)
        self.assertEqual(len(sentences), 3)


class TestSentimentAnalysis(unittest.TestCase):
    """Test sentiment analysis functionality"""
    
    def setUp(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_textblob_sentiment(self):
        text = "I love this product!"
        result = self.analyzer.textblob_sentiment(text)
        self.assertIn('sentiment', result)
        self.assertIn('polarity', result)
        self.assertEqual(result['sentiment'], 'positive')
    
    def test_vader_sentiment(self):
        text = "This is terrible!"
        result = self.analyzer.vader_sentiment(text)
        self.assertIn('sentiment', result)
        self.assertIn('compound', result)
        self.assertEqual(result['sentiment'], 'negative')
    
    def test_batch_analyze(self):
        texts = ["Great product!", "Terrible service!", "It's okay."]
        results = self.analyzer.batch_analyze(texts, method='textblob')
        self.assertEqual(len(results), 3)
        self.assertIn('sentiment', results[0])


class TestChatBot(unittest.TestCase):
    """Test chatbot functionality"""
    
    def setUp(self):
        self.bot = ChatBot("Test Bot")
    
    def test_detect_intent(self):
        intent = self.bot.detect_intent("Hello there!")
        self.assertEqual(intent, 'greeting')
        
        intent = self.bot.detect_intent("Goodbye!")
        self.assertEqual(intent, 'farewell')
    
    def test_chat(self):
        response = self.bot.chat("Hello!")
        self.assertIn('response', response)
        self.assertIn('analysis', response)
        self.assertIsInstance(response['response'], str)
    
    def test_conversation_history(self):
        self.bot.chat("Hello!")
        self.bot.chat("How are you?")
        history = self.bot.get_conversation_history()
        self.assertEqual(len(history), 2)


class TestVisualization(unittest.TestCase):
    """Test visualization functionality"""
    
    def setUp(self):
        self.visualizer = NLPVisualizer()
    
    def test_sentiment_distribution(self):
        sentiments = ['positive', 'negative', 'neutral', 'positive']
        fig = self.visualizer.sentiment_distribution(sentiments)
        self.assertIsNotNone(fig)
    
    def test_word_cloud(self):
        text = "natural language processing machine learning artificial intelligence"
        fig = self.visualizer.word_cloud(text)
        self.assertIsNotNone(fig)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTextPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestChatBot))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualization))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")