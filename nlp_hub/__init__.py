"""
NLP-Hub: A comprehensive Natural Language Processing toolkit

This package provides various NLP utilities including:
- Text preprocessing and tokenization
- Sentiment analysis
- Topic modeling and classification
- Transformer-based models
- Chatbot framework
- Data visualization tools
"""

__version__ = "1.0.0"
__author__ = "NLP-Hub Contributors"

# Import main classes from each module
try:
    from .preprocessing import TextPreprocessor, Tokenizer
except ImportError:
    from nlp_hub.preprocessing import TextPreprocessor, Tokenizer

try:
    from .sentiment import SentimentAnalyzer
except ImportError:
    from nlp_hub.sentiment import SentimentAnalyzer

try:
    from .classification import TextClassifier, TopicModeler
except ImportError:
    from nlp_hub.classification import TextClassifier, TopicModeler

try:
    from .transformers import TransformerModels
except ImportError:
    from nlp_hub.transformers import TransformerModels

try:
    from .chatbot import ChatBot, ConversationManager
except ImportError:
    from nlp_hub.chatbot import ChatBot, ConversationManager

try:
    from .visualization import NLPVisualizer
except ImportError:
    from nlp_hub.visualization import NLPVisualizer

__all__ = [
    "TextPreprocessor",
    "Tokenizer", 
    "SentimentAnalyzer",
    "TextClassifier",
    "TopicModeler",
    "TransformerModels",
    "ChatBot",
    "ConversationManager",
    "NLPVisualizer"
]