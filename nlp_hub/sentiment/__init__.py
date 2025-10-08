"""
Sentiment analysis utilities using multiple approaches
"""

import numpy as np
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List, Union, Optional, Tuple
import pickle
import os


class SentimentAnalyzer:
    """
    Multi-approach sentiment analysis class
    """
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.ml_model = None
        self.vectorizer = None
        self.transformer_pipeline = None
        
        # Initialize transformer model if available
        try:
            self.transformer_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
        except Exception as e:
            print(f"Could not load transformer model: {e}")
            self.transformer_pipeline = None
    
    def textblob_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with polarity and subjectivity scores
        """
        blob = TextBlob(text)
        
        return {
            'polarity': blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
            'subjectivity': blob.sentiment.subjectivity,  # 0 (objective) to 1 (subjective)
            'sentiment': 'positive' if blob.sentiment.polarity > 0 else 'negative' if blob.sentiment.polarity < 0 else 'neutral'
        }
    
    def vader_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with VADER sentiment scores
        """
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine overall sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        scores['sentiment'] = sentiment
        return scores
    
    def transformer_sentiment(self, text: str) -> Dict[str, Union[str, float, List]]:
        """
        Analyze sentiment using transformer model
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with transformer sentiment predictions
        """
        if self.transformer_pipeline is None:
            return {'error': 'Transformer model not available'}
        
        try:
            results = self.transformer_pipeline(text)
            
            # Convert to standard format
            sentiment_map = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive'
            }
            
            processed_results = []
            for result in results[0]:
                label = sentiment_map.get(result['label'], result['label'].lower())
                processed_results.append({
                    'sentiment': label,
                    'confidence': result['score']
                })
            
            # Get the highest confidence prediction
            best_prediction = max(processed_results, key=lambda x: x['confidence'])
            
            return {
                'sentiment': best_prediction['sentiment'],
                'confidence': best_prediction['confidence'],
                'all_scores': processed_results
            }
        except Exception as e:
            return {'error': f'Transformer prediction failed: {str(e)}'}
    
    def train_ml_model(self, texts: List[str], labels: List[str], 
                      test_size: float = 0.2) -> Dict[str, float]:
        """
        Train a machine learning model for sentiment classification
        
        Args:
            texts: List of training texts
            labels: List of sentiment labels
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.ml_model = LogisticRegression(random_state=42)
        self.ml_model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.ml_model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def ml_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict sentiment using trained ML model
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with ML model prediction
        """
        if self.ml_model is None or self.vectorizer is None:
            return {'error': 'ML model not trained. Call train_ml_model first.'}
        
        text_vec = self.vectorizer.transform([text])
        prediction = self.ml_model.predict(text_vec)[0]
        probabilities = self.ml_model.predict_proba(text_vec)[0]
        
        # Get confidence (max probability)
        confidence = max(probabilities)
        
        return {
            'sentiment': prediction,
            'confidence': confidence
        }
    
    def ensemble_sentiment(self, text: str, methods: List[str] = ['textblob', 'vader']) -> Dict[str, Union[str, float, Dict]]:
        """
        Combine multiple sentiment analysis methods
        
        Args:
            text: Input text
            methods: List of methods to use ['textblob', 'vader', 'ml', 'transformer']
            
        Returns:
            Dictionary with ensemble results
        """
        results = {}
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        if 'textblob' in methods:
            tb_result = self.textblob_sentiment(text)
            results['textblob'] = tb_result
            sentiment_scores[tb_result['sentiment']] += 1
        
        if 'vader' in methods:
            vader_result = self.vader_sentiment(text)
            results['vader'] = vader_result
            sentiment_scores[vader_result['sentiment']] += 1
        
        if 'ml' in methods and self.ml_model is not None:
            ml_result = self.ml_sentiment(text)
            if 'error' not in ml_result:
                results['ml'] = ml_result
                sentiment_scores[ml_result['sentiment']] += 1
        
        if 'transformer' in methods and self.transformer_pipeline is not None:
            transformer_result = self.transformer_sentiment(text)
            if 'error' not in transformer_result:
                results['transformer'] = transformer_result
                sentiment_scores[transformer_result['sentiment']] += 1
        
        # Determine ensemble prediction
        ensemble_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        ensemble_confidence = sentiment_scores[ensemble_sentiment] / sum(sentiment_scores.values())
        
        return {
            'ensemble_sentiment': ensemble_sentiment,
            'ensemble_confidence': ensemble_confidence,
            'individual_results': results,
            'vote_counts': sentiment_scores
        }
    
    def analyze_text_emotions(self, text: str) -> Dict[str, float]:
        """
        Analyze emotions in text using TextBlob and VADER
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with emotion analysis
        """
        vader_scores = self.vader_sentiment(text)
        textblob_scores = self.textblob_sentiment(text)
        
        # Combine scores for emotion analysis
        emotions = {
            'joy': max(0, vader_scores['pos']),
            'anger': max(0, vader_scores['neg']),
            'fear': max(0, vader_scores['neg'] * 0.7),
            'sadness': max(0, vader_scores['neg'] * 0.8),
            'surprise': abs(textblob_scores['polarity']) * textblob_scores['subjectivity'],
            'disgust': max(0, vader_scores['neg'] * 0.6)
        }
        
        return emotions
    
    def batch_analyze(self, texts: List[str], method: str = 'ensemble') -> List[Dict]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            method: Analysis method to use
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        for text in texts:
            if method == 'textblob':
                result = self.textblob_sentiment(text)
            elif method == 'vader':
                result = self.vader_sentiment(text)
            elif method == 'ml':
                result = self.ml_sentiment(text)
            elif method == 'transformer':
                result = self.transformer_sentiment(text)
            elif method == 'ensemble':
                result = self.ensemble_sentiment(text)
            else:
                result = {'error': f'Unknown method: {method}'}
            
            results.append(result)
        
        return results
    
    def save_model(self, filepath: str):
        """
        Save trained ML model and vectorizer
        
        Args:
            filepath: Path to save the model
        """
        if self.ml_model is None or self.vectorizer is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.ml_model,
            'vectorizer': self.vectorizer
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """
        Load trained ML model and vectorizer
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ml_model = model_data['model']
        self.vectorizer = model_data['vectorizer']