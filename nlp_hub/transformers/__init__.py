"""
Transformer-based models integration (BERT, GPT, etc.)
"""

import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering, AutoModelForTokenClassification,
    pipeline, BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer,
    DistilBertTokenizer, DistilBertModel
)
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
import json
import os


class TransformerModels:
    """
    Wrapper class for various transformer models and tasks
    """
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        # Pre-defined model configurations
        self.model_configs = {
            'bert_base': {
                'model_name': 'bert-base-uncased',
                'tasks': ['feature_extraction', 'classification', 'question_answering']
            },
            'distilbert': {
                'model_name': 'distilbert-base-uncased',
                'tasks': ['feature_extraction', 'classification']
            },
            'roberta_sentiment': {
                'model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'tasks': ['sentiment_analysis']
            },
            'gpt2': {
                'model_name': 'gpt2',
                'tasks': ['text_generation']
            },
            'bert_ner': {
                'model_name': 'dbmdz/bert-large-cased-finetuned-conll03-english',
                'tasks': ['named_entity_recognition']
            }
        }
    
    def load_model(self, model_key: str, task: str = 'feature_extraction') -> bool:
        """
        Load a transformer model for a specific task
        
        Args:
            model_key: Key from model_configs or custom model name
            task: Specific task to load the model for
            
        Returns:
            True if model loaded successfully
        """
        try:
            if model_key in self.model_configs:
                model_name = self.model_configs[model_key]['model_name']
            else:
                model_name = model_key
            
            # Load tokenizer
            self.tokenizers[model_key] = AutoTokenizer.from_pretrained(model_name)
            
            # Load model based on task
            if task == 'feature_extraction':
                self.models[model_key] = AutoModel.from_pretrained(model_name)
            elif task == 'classification':
                self.models[model_key] = AutoModelForSequenceClassification.from_pretrained(model_name)
            elif task == 'question_answering':
                self.models[model_key] = AutoModelForQuestionAnswering.from_pretrained(model_name)
            elif task == 'token_classification':
                self.models[model_key] = AutoModelForTokenClassification.from_pretrained(model_name)
            else:
                self.models[model_key] = AutoModel.from_pretrained(model_name)
            
            return True
            
        except Exception as e:
            print(f"Error loading model {model_key}: {e}")
            return False
    
    def load_pipeline(self, task: str, model_name: str = None) -> bool:
        """
        Load a pre-built pipeline for common tasks
        
        Args:
            task: Task name (sentiment-analysis, text-generation, etc.)
            model_name: Optional model name, uses default if not specified
            
        Returns:
            True if pipeline loaded successfully
        """
        try:
            if model_name:
                self.pipelines[task] = pipeline(task, model=model_name)
            else:
                self.pipelines[task] = pipeline(task)
            return True
        except Exception as e:
            print(f"Error loading pipeline for {task}: {e}")
            return False
    
    def extract_features(self, texts: List[str], model_key: str = 'bert_base') -> np.ndarray:
        """
        Extract features from texts using transformer model
        
        Args:
            texts: List of input texts
            model_key: Model to use for feature extraction
            
        Returns:
            Feature matrix
        """
        if model_key not in self.models or model_key not in self.tokenizers:
            if not self.load_model(model_key, 'feature_extraction'):
                raise ValueError(f"Could not load model {model_key}")
        
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        
        features = []
        
        for text in texts:
            # Tokenize
            inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                             padding=True, max_length=512)
            
            # Extract features
            with torch.no_grad():
                outputs = model(**inputs)
                # Use CLS token representation for sentence-level features
                cls_features = outputs.last_hidden_state[:, 0, :].numpy()
                features.append(cls_features[0])
        
        return np.array(features)
    
    def classify_text(self, texts: List[str], model_key: str = 'roberta_sentiment') -> List[Dict[str, Any]]:
        """
        Classify texts using transformer model
        
        Args:
            texts: List of input texts
            model_key: Model to use for classification
            
        Returns:
            List of classification results
        """
        task = 'sentiment-analysis' if 'sentiment' in model_key else 'text-classification'
        
        if task not in self.pipelines:
            # Try to load appropriate model
            if model_key in self.model_configs:
                model_name = self.model_configs[model_key]['model_name']
                if not self.load_pipeline(task, model_name):
                    raise ValueError(f"Could not load pipeline for {task}")
            else:
                if not self.load_pipeline(task):
                    raise ValueError(f"Could not load pipeline for {task}")
        
        results = []
        pipeline_model = self.pipelines[task]
        
        for text in texts:
            try:
                result = pipeline_model(text)
                if isinstance(result, list):
                    result = result[0]
                results.append({
                    'text': text,
                    'label': result['label'],
                    'confidence': result['score']
                })
            except Exception as e:
                results.append({
                    'text': text,
                    'error': str(e)
                })
        
        return results
    
    def generate_text(self, prompt: str, model_key: str = 'gpt2', 
                     max_length: int = 100, num_return_sequences: int = 1) -> List[str]:
        """
        Generate text using transformer model
        
        Args:
            prompt: Input prompt
            model_key: Model to use for generation
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated texts
        """
        if 'text-generation' not in self.pipelines:
            if model_key in self.model_configs:
                model_name = self.model_configs[model_key]['model_name']
                if not self.load_pipeline('text-generation', model_name):
                    raise ValueError("Could not load text generation pipeline")
            else:
                if not self.load_pipeline('text-generation'):
                    raise ValueError("Could not load text generation pipeline")
        
        generator = self.pipelines['text-generation']
        
        results = generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=0.7
        )
        
        return [result['generated_text'] for result in results]
    
    def answer_questions(self, contexts: List[str], questions: List[str]) -> List[Dict[str, Any]]:
        """
        Answer questions based on given contexts
        
        Args:
            contexts: List of context paragraphs
            questions: List of questions
            
        Returns:
            List of answers with confidence scores
        """
        if 'question-answering' not in self.pipelines:
            if not self.load_pipeline('question-answering'):
                raise ValueError("Could not load question answering pipeline")
        
        qa_pipeline = self.pipelines['question-answering']
        results = []
        
        for context, question in zip(contexts, questions):
            try:
                result = qa_pipeline(question=question, context=context)
                results.append({
                    'question': question,
                    'context': context,
                    'answer': result['answer'],
                    'confidence': result['score'],
                    'start': result['start'],
                    'end': result['end']
                })
            except Exception as e:
                results.append({
                    'question': question,
                    'context': context,
                    'error': str(e)
                })
        
        return results
    
    def named_entity_recognition(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Perform named entity recognition on texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of NER results
        """
        if 'ner' not in self.pipelines:
            if not self.load_pipeline('ner'):
                # Try with token classification
                if not self.load_pipeline('token-classification'):
                    raise ValueError("Could not load NER pipeline")
                pipeline_key = 'token-classification'
            else:
                pipeline_key = 'ner'
        else:
            pipeline_key = 'ner'
        
        ner_pipeline = self.pipelines[pipeline_key]
        results = []
        
        for text in texts:
            try:
                entities = ner_pipeline(text)
                processed_entities = []
                
                for entity in entities:
                    processed_entities.append({
                        'text': entity['word'],
                        'label': entity['entity'],
                        'confidence': entity['score'],
                        'start': entity.get('start', 0),
                        'end': entity.get('end', 0)
                    })
                
                results.append({
                    'text': text,
                    'entities': processed_entities
                })
            except Exception as e:
                results.append({
                    'text': text,
                    'error': str(e)
                })
        
        return results
    
    def similarity_search(self, query: str, documents: List[str], 
                         model_key: str = 'bert_base', top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar documents to query using transformer embeddings
        
        Args:
            query: Search query
            documents: List of documents to search
            model_key: Model to use for embeddings
            top_k: Number of top results to return
            
        Returns:
            List of most similar documents with scores
        """
        # Extract features for query and documents
        all_texts = [query] + documents
        features = self.extract_features(all_texts, model_key)
        
        query_features = features[0]
        doc_features = features[1:]
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_features], doc_features)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': documents[idx],
                'similarity_score': float(similarities[idx]),
                'rank': len(results) + 1
            })
        
        return results
    
    def batch_process(self, texts: List[str], task: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple texts for a given task
        
        Args:
            texts: List of input texts
            task: Task to perform
            **kwargs: Additional arguments for specific tasks
            
        Returns:
            List of results
        """
        if task == 'sentiment':
            return self.classify_text(texts, kwargs.get('model_key', 'roberta_sentiment'))
        elif task == 'classification':
            return self.classify_text(texts, kwargs.get('model_key', 'bert_base'))
        elif task == 'ner':
            return self.named_entity_recognition(texts)
        elif task == 'features':
            features = self.extract_features(texts, kwargs.get('model_key', 'bert_base'))
            return [{'text': text, 'features': feat.tolist()} 
                   for text, feat in zip(texts, features)]
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models and available configurations
        
        Returns:
            Dictionary with model information
        """
        return {
            'loaded_models': list(self.models.keys()),
            'loaded_tokenizers': list(self.tokenizers.keys()),
            'loaded_pipelines': list(self.pipelines.keys()),
            'available_configs': self.model_configs
        }
    
    def clear_models(self):
        """
        Clear all loaded models to free memory
        """
        self.models.clear()
        self.tokenizers.clear()
        self.pipelines.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()