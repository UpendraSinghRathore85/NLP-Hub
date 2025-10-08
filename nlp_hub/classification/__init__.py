"""
Text classification and topic modeling utilities
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import gensim
from gensim import corpora
from gensim.models import LdaModel, Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import os


class TextClassifier:
    """
    Comprehensive text classification and topic modeling class
    """
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'naive_bayes': MultinomialNB(),
            'svm': SVC(probability=True, random_state=42),
            'random_forest': RandomForestClassifier(random_state=42)
        }
        
    def prepare_data(self, texts: List[str], labels: List[str], 
                    vectorizer_type: str = 'tfidf', max_features: int = 10000) -> Tuple[Any, Any]:
        """
        Prepare text data for classification
        
        Args:
            texts: List of text documents
            labels: List of corresponding labels
            vectorizer_type: Type of vectorizer ('tfidf' or 'count')
            max_features: Maximum number of features
            
        Returns:
            Tuple of (vectorized_texts, labels)
        """
        # Initialize vectorizer
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
        
        # Transform texts
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        return X, y
    
    def train_classifier(self, texts: List[str], labels: List[str], 
                        model_name: str = 'logistic_regression',
                        test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train a text classifier
        
        Args:
            texts: List of training texts
            labels: List of labels
            model_name: Name of the model to use
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training results
        """
        # Prepare data
        X, y = self.prepare_data(texts, labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Select and train model
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.models[model_name]
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_names': self.vectorizer.get_feature_names_out()[:100].tolist()
        }
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict labels for new texts
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of predictions with confidence scores
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train_classifier first.")
        
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'prediction': predictions[i],
                'confidence': max(probabilities[i]),
                'all_probabilities': dict(zip(self.model.classes_, probabilities[i]))
            }
            results.append(result)
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get most important features for each class
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with top features for each class
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained.")
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        if hasattr(self.model, 'coef_'):
            # For linear models
            importance_dict = {}
            for i, class_name in enumerate(self.model.classes_):
                if len(self.model.coef_.shape) == 1:
                    # Binary classification
                    coefficients = self.model.coef_
                else:
                    # Multi-class classification
                    coefficients = self.model.coef_[i]
                
                feature_importance = list(zip(feature_names, coefficients))
                feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                importance_dict[str(class_name)] = feature_importance[:top_n]
            
            return importance_dict
        elif hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            feature_importance = list(zip(feature_names, self.model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            return {'all_classes': feature_importance[:top_n]}
        else:
            return {'error': 'Feature importance not available for this model type'}


class TopicModeler:
    """
    Topic modeling utilities using LDA and other techniques
    """
    
    def __init__(self):
        self.lda_model = None
        self.dictionary = None
        self.corpus = None
        self.vectorizer = None
        self.sklearn_lda = None
    
    def preprocess_for_lda(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess texts for LDA topic modeling
        
        Args:
            texts: List of text documents
            
        Returns:
            List of tokenized and cleaned documents
        """
        processed_docs = []
        
        for text in texts:
            # Simple preprocessing
            words = text.lower().split()
            # Remove short words and common stopwords
            words = [word for word in words if len(word) > 3]
            processed_docs.append(words)
        
        return processed_docs
    
    def train_lda_gensim(self, texts: List[str], num_topics: int = 5, 
                        passes: int = 10) -> Dict[str, Any]:
        """
        Train LDA model using Gensim
        
        Args:
            texts: List of text documents
            num_topics: Number of topics to extract
            passes: Number of passes through the corpus
            
        Returns:
            Dictionary with model results
        """
        # Preprocess texts
        processed_docs = self.preprocess_for_lda(texts)
        
        # Create dictionary and corpus
        self.dictionary = corpora.Dictionary(processed_docs)
        self.dictionary.filter_extremes(no_below=2, no_above=0.8)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        
        # Train LDA model
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=passes,
            alpha='auto',
            per_word_topics=True
        )
        
        # Get topics
        topics = []
        for idx in range(num_topics):
            topic = self.lda_model.show_topic(idx, topn=10)
            topics.append({
                'topic_id': idx,
                'words': [word for word, prob in topic],
                'probabilities': [prob for word, prob in topic]
            })
        
        return {
            'topics': topics,
            'coherence': self.calculate_coherence(),
            'perplexity': self.lda_model.log_perplexity(self.corpus)
        }
    
    def train_lda_sklearn(self, texts: List[str], num_topics: int = 5) -> Dict[str, Any]:
        """
        Train LDA model using scikit-learn
        
        Args:
            texts: List of text documents
            num_topics: Number of topics to extract
            
        Returns:
            Dictionary with model results
        """
        # Vectorize texts
        self.vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True
        )
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        
        # Train LDA
        self.sklearn_lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=10
        )
        self.sklearn_lda.fit(doc_term_matrix)
        
        # Get topics
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.sklearn_lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_probs = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'probabilities': top_probs
            })
        
        return {
            'topics': topics,
            'perplexity': self.sklearn_lda.perplexity(doc_term_matrix)
        }
    
    def calculate_coherence(self) -> float:
        """
        Calculate coherence score for Gensim LDA model
        
        Returns:
            Coherence score
        """
        if self.lda_model is None:
            return 0.0
        
        try:
            from gensim.models import CoherenceModel
            coherence_model = CoherenceModel(
                model=self.lda_model,
                corpus=self.corpus,
                dictionary=self.dictionary,
                coherence='c_v'
            )
            return coherence_model.get_coherence()
        except:
            return 0.0
    
    def predict_topic(self, text: str) -> Dict[str, Any]:
        """
        Predict topic distribution for a new text
        
        Args:
            text: Input text
            
        Returns:
            Topic distribution
        """
        if self.lda_model is None:
            raise ValueError("LDA model not trained.")
        
        # Preprocess text
        processed_text = self.preprocess_for_lda([text])[0]
        bow = self.dictionary.doc2bow(processed_text)
        
        # Get topic distribution
        topic_dist = self.lda_model.get_document_topics(bow)
        
        return {
            'topic_distribution': dict(topic_dist),
            'dominant_topic': max(topic_dist, key=lambda x: x[1])[0]
        }
    
    def cluster_documents(self, texts: List[str], num_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster documents using K-means
        
        Args:
            texts: List of text documents
            num_clusters: Number of clusters
            
        Returns:
            Clustering results
        """
        # Vectorize texts
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Analyze clusters
        clusters = {}
        for i in range(num_clusters):
            cluster_docs = [texts[j] for j in range(len(texts)) if cluster_labels[j] == i]
            clusters[f'cluster_{i}'] = {
                'documents': cluster_docs,
                'size': len(cluster_docs)
            }
        
        return {
            'clusters': clusters,
            'labels': cluster_labels.tolist(),
            'inertia': kmeans.inertia_
        }
    
    def save_model(self, filepath: str, model_type: str = 'gensim'):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
            model_type: Type of model ('gensim' or 'sklearn')
        """
        if model_type == 'gensim' and self.lda_model is not None:
            self.lda_model.save(filepath)
        elif model_type == 'sklearn' and self.sklearn_lda is not None:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.sklearn_lda,
                    'vectorizer': self.vectorizer
                }, f)
        else:
            raise ValueError("No trained model to save or invalid model type")
    
    def load_model(self, filepath: str, model_type: str = 'gensim'):
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
            model_type: Type of model ('gensim' or 'sklearn')
        """
        if model_type == 'gensim':
            self.lda_model = LdaModel.load(filepath)
        elif model_type == 'sklearn':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.sklearn_lda = data['model']
                self.vectorizer = data['vectorizer']
        else:
            raise ValueError("Invalid model type")