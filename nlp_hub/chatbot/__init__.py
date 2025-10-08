"""
Chatbot and conversational AI framework
"""

import json
import random
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import nltk
from nltk.chat.util import Chat, reflections
from transformers import pipeline
import spacy


class ChatBot:
    """
    Advanced chatbot with multiple response strategies
    """
    
    def __init__(self, name: str = "NLP-Hub Bot"):
        self.name = name
        self.conversation_history = []
        self.user_profile = {}
        self.context = {}
        
        # Initialize transformer for advanced responses
        try:
            self.generator = pipeline('text-generation', model='microsoft/DialoGPT-medium')
            self.has_transformer = True
        except Exception as e:
            print(f"Could not load transformer model: {e}")
            self.has_transformer = False
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.has_spacy = True
        except OSError:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.has_spacy = False
        
        # Rule-based patterns
        self.patterns = [
            (r'hi|hello|hey|greetings', ['Hello!', 'Hi there!', 'Hey! How can I help you?', 'Greetings!']),
            (r'how are you', ['I\'m doing well, thank you!', 'I\'m great! How are you?', 'All good here!']),
            (r'what is your name', [f'My name is {self.name}', f'I\'m {self.name}', f'You can call me {self.name}']),
            (r'bye|goodbye|see you', ['Goodbye!', 'See you later!', 'Take care!', 'Until next time!']),
            (r'thank you|thanks', ['You\'re welcome!', 'Happy to help!', 'Anytime!', 'Glad I could assist!']),
            (r'what can you do|help', [
                'I can help with various NLP tasks like sentiment analysis, text classification, and more!',
                'I\'m here to assist with natural language processing questions and tasks.',
                'I can chat with you and help with NLP-related queries!'
            ]),
            (r'weather', ['I don\'t have access to weather data, but I can help with text analysis!',
                         'For weather info, try a weather service. I specialize in NLP tasks!']),
            (r'joke', ['Why did the NLP model break up? It couldn\'t find the right tokens!',
                      'What do you call a chatbot that tells jokes? A pun-bot!',
                      'Why don\'t neural networks ever get tired? They always have enough layers to rest on!']),
        ]
        
        # Initialize NLTK chat
        self.rule_based_chat = Chat(self.patterns, reflections)
        
        # Intent recognition patterns
        self.intents = {
            'greeting': [r'hi|hello|hey|greetings'],
            'farewell': [r'bye|goodbye|see you|exit'],
            'question': [r'what|how|when|where|why|who'],
            'help': [r'help|assist|support'],
            'sentiment': [r'sentiment|feeling|emotion|mood'],
            'classification': [r'classify|categorize|predict'],
            'nlp': [r'nlp|natural language|text analysis|processing']
        }
    
    def detect_intent(self, message: str) -> str:
        """
        Detect the intent of the user message
        
        Args:
            message: User input message
            
        Returns:
            Detected intent
        """
        message_lower = message.lower()
        
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent
        
        return 'general'
    
    def extract_entities(self, message: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from message
        
        Args:
            message: Input message
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        if self.has_spacy:
            doc = self.nlp(message)
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_)
                })
        
        return entities
    
    def analyze_sentiment(self, message: str) -> Dict[str, Any]:
        """
        Analyze sentiment of the message
        
        Args:
            message: Input message
            
        Returns:
            Sentiment analysis result
        """
        # Simple rule-based sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'joy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed']
        
        message_lower = message.lower()
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = positive_count / (positive_count + negative_count + 1)
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = negative_count / (positive_count + negative_count + 1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def generate_response(self, message: str, use_transformer: bool = True) -> str:
        """
        Generate a response to the user message
        
        Args:
            message: User input message
            use_transformer: Whether to use transformer model for generation
            
        Returns:
            Generated response
        """
        intent = self.detect_intent(message)
        
        # Context-aware responses based on intent
        if intent == 'greeting':
            responses = [
                f"Hello! I'm {self.name}, your NLP assistant. How can I help you today?",
                "Hi there! Ready to explore some natural language processing?",
                "Hey! What NLP task would you like to work on?"
            ]
            return random.choice(responses)
        
        elif intent == 'farewell':
            responses = [
                "Goodbye! It was great chatting about NLP with you!",
                "See you later! Keep exploring the world of natural language processing!",
                "Take care! Come back anytime for more NLP assistance!"
            ]
            return random.choice(responses)
        
        elif intent == 'help':
            return """I can help you with various NLP tasks:
            
🔤 Text preprocessing and tokenization
😊 Sentiment analysis
📊 Text classification and topic modeling
🤖 Transformer-based models (BERT, GPT, etc.)
📈 Text visualization and insights
💬 Conversational AI

Just ask me about any of these topics or try out some text analysis!"""
        
        elif intent == 'sentiment':
            sentiment_result = self.analyze_sentiment(message)
            return f"I detect a {sentiment_result['sentiment']} sentiment in your message (confidence: {sentiment_result['confidence']:.2f}). How can I help you analyze sentiment in other texts?"
        
        elif intent == 'nlp':
            return "Great! You're interested in Natural Language Processing. I can help with text analysis, sentiment detection, classification, and much more. What specific NLP task would you like to explore?"
        
        # Try transformer generation if available
        if use_transformer and self.has_transformer:
            try:
                response = self.generator(f"Human: {message}\nBot:", max_length=100, num_return_sequences=1)
                generated_text = response[0]['generated_text']
                # Extract bot response
                bot_response = generated_text.split("Bot:")[-1].strip()
                if bot_response and len(bot_response) > 10:
                    return bot_response
            except Exception as e:
                print(f"Transformer generation failed: {e}")
        
        # Fallback to rule-based
        rule_response = self.rule_based_chat.respond(message)
        if rule_response:
            return rule_response
        
        # Default responses
        default_responses = [
            "That's interesting! Can you tell me more?",
            "I'd love to help you with that. Could you provide more details?",
            "Let me think about that. What specific aspect interests you most?",
            "That's a great question! How can I assist you with NLP analysis?",
            "I'm here to help with your natural language processing needs!"
        ]
        
        return random.choice(default_responses)
    
    def chat(self, message: str) -> Dict[str, Any]:
        """
        Main chat function that processes message and returns comprehensive response
        
        Args:
            message: User input message
            
        Returns:
            Dictionary with response and analysis
        """
        timestamp = datetime.now().isoformat()
        
        # Analyze the message
        intent = self.detect_intent(message)
        entities = self.extract_entities(message)
        sentiment = self.analyze_sentiment(message)
        
        # Generate response
        response = self.generate_response(message)
        
        # Store in conversation history
        conversation_entry = {
            'timestamp': timestamp,
            'user_message': message,
            'bot_response': response,
            'intent': intent,
            'entities': entities,
            'sentiment': sentiment
        }
        
        self.conversation_history.append(conversation_entry)
        
        return {
            'response': response,
            'analysis': {
                'intent': intent,
                'entities': entities,
                'sentiment': sentiment
            },
            'timestamp': timestamp
        }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history
        
        Returns:
            List of conversation entries
        """
        return self.conversation_history
    
    def clear_history(self):
        """
        Clear the conversation history
        """
        self.conversation_history.clear()
    
    def save_conversation(self, filepath: str):
        """
        Save conversation history to file
        
        Args:
            filepath: Path to save the conversation
        """
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def load_conversation(self, filepath: str):
        """
        Load conversation history from file
        
        Args:
            filepath: Path to load the conversation from
        """
        with open(filepath, 'r') as f:
            self.conversation_history = json.load(f)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics
        
        Returns:
            Dictionary with conversation stats
        """
        if not self.conversation_history:
            return {'total_messages': 0}
        
        total_messages = len(self.conversation_history)
        intents = [entry['intent'] for entry in self.conversation_history]
        sentiments = [entry['sentiment']['sentiment'] for entry in self.conversation_history]
        
        from collections import Counter
        intent_counts = Counter(intents)
        sentiment_counts = Counter(sentiments)
        
        return {
            'total_messages': total_messages,
            'intent_distribution': dict(intent_counts),
            'sentiment_distribution': dict(sentiment_counts),
            'avg_message_length': sum(len(entry['user_message']) for entry in self.conversation_history) / total_messages,
            'entities_extracted': sum(len(entry['entities']) for entry in self.conversation_history)
        }


class ConversationManager:
    """
    Manage multiple chatbot conversations and sessions
    """
    
    def __init__(self):
        self.sessions = {}
        self.active_session = None
    
    def create_session(self, session_id: str, bot_name: str = "NLP-Hub Bot") -> ChatBot:
        """
        Create a new chat session
        
        Args:
            session_id: Unique session identifier
            bot_name: Name of the chatbot
            
        Returns:
            ChatBot instance
        """
        self.sessions[session_id] = ChatBot(bot_name)
        self.active_session = session_id
        return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> Optional[ChatBot]:
        """
        Get an existing chat session
        
        Args:
            session_id: Session identifier
            
        Returns:
            ChatBot instance or None
        """
        return self.sessions.get(session_id)
    
    def switch_session(self, session_id: str) -> bool:
        """
        Switch to a different session
        
        Args:
            session_id: Session to switch to
            
        Returns:
            True if successful
        """
        if session_id in self.sessions:
            self.active_session = session_id
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session
        
        Args:
            session_id: Session to delete
            
        Returns:
            True if successful
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.active_session == session_id:
                self.active_session = None
            return True
        return False
    
    def list_sessions(self) -> List[str]:
        """
        List all active sessions
        
        Returns:
            List of session IDs
        """
        return list(self.sessions.keys())
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all sessions
        
        Returns:
            Dictionary with stats for each session
        """
        return {session_id: bot.get_stats() 
                for session_id, bot in self.sessions.items()}