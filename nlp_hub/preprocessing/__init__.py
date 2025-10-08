"""
Text preprocessing and tokenization utilities
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy
from typing import List, Optional, Dict, Any


class TextPreprocessor:
    """
    Comprehensive text preprocessing utility class
    """
    
    def __init__(self, language: str = 'english'):
        self.language = language
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            
        try:
            nltk.data.find('chunkers/maxent_ne_chunker')
        except LookupError:
            nltk.download('maxent_ne_chunker')
            
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
        
        self.stop_words = set(stopwords.words(language))
        
        # Load spaCy model if available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clean_text(self, text: str, remove_punctuation: bool = True, 
                   remove_numbers: bool = False, lowercase: bool = True) -> str:
        """
        Basic text cleaning operations
        
        Args:
            text: Input text to clean
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
            lowercase: Whether to convert to lowercase
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Remove punctuation
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        return text
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens without stopwords
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to tokens
        
        Args:
            tokens: List of tokens to stem
            
        Returns:
            Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens
        
        Args:
            tokens: List of tokens to lemmatize
            
        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True,
                       apply_stemming: bool = False, apply_lemmatization: bool = True,
                       remove_punctuation: bool = True, remove_numbers: bool = False) -> List[str]:
        """
        Complete text preprocessing pipeline
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            apply_stemming: Whether to apply stemming
            apply_lemmatization: Whether to apply lemmatization
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
            
        Returns:
            List of processed tokens
        """
        # Clean text
        cleaned_text = self.clean_text(text, remove_punctuation, remove_numbers)
        
        # Tokenize
        tokens = word_tokenize(cleaned_text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Apply stemming or lemmatization
        if apply_stemming:
            tokens = self.stem_tokens(tokens)
        elif apply_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        
        return tokens


class Tokenizer:
    """
    Advanced tokenization utilities
    """
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def word_tokenize(self, text: str) -> List[str]:
        """
        Basic word tokenization
        
        Args:
            text: Input text
            
        Returns:
            List of word tokens
        """
        return word_tokenize(text)
    
    def sentence_tokenize(self, text: str) -> List[str]:
        """
        Sentence tokenization
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        return sent_tokenize(text)
    
    def advanced_tokenize(self, text: str) -> Dict[str, Any]:
        """
        Advanced tokenization with POS tagging and NER
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with tokens, POS tags, and named entities
        """
        if self.nlp is None:
            # Fallback to NLTK
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)
            
            return {
                'tokens': tokens,
                'pos_tags': pos_tags,
                'named_entities': str(named_entities)
            }
        else:
            # Use spaCy for advanced processing
            doc = self.nlp(text)
            
            return {
                'tokens': [token.text for token in doc],
                'lemmas': [token.lemma_ for token in doc],
                'pos_tags': [(token.text, token.pos_) for token in doc],
                'named_entities': [(ent.text, ent.label_) for ent in doc.ents],
                'dependencies': [(token.text, token.dep_, token.head.text) for token in doc]
            }
    
    def extract_ngrams(self, tokens: List[str], n: int = 2) -> List[tuple]:
        """
        Extract n-grams from tokens
        
        Args:
            tokens: List of tokens
            n: N-gram size
            
        Returns:
            List of n-grams
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def extract_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text
        
        Args:
            text: Input text
            
        Returns:
            List of noun phrases
        """
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]