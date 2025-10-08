"""
Simple examples demonstrating NLP-Hub functionality
"""

from nlp_hub import TextPreprocessor, SentimentAnalyzer, ChatBot, NLPVisualizer


def text_preprocessing_example():
    """Demonstrate text preprocessing capabilities"""
    print("=== Text Preprocessing Example ===")
    
    preprocessor = TextPreprocessor()
    
    text = "Hello World! This is a SAMPLE text with punctuation, numbers 123, and UPPERCASE letters."
    print(f"Original: {text}")
    
    # Clean text
    cleaned = preprocessor.clean_text(text)
    print(f"Cleaned: {cleaned}")
    
    # Full preprocessing
    processed = preprocessor.preprocess_text(text)
    print(f"Processed tokens: {processed}")
    print()


def sentiment_analysis_example():
    """Demonstrate sentiment analysis capabilities"""
    print("=== Sentiment Analysis Example ===")
    
    analyzer = SentimentAnalyzer()
    
    texts = [
        "I love this product! It's amazing!",
        "This is terrible. I hate it.",
        "It's an okay product, nothing special."
    ]
    
    for text in texts:
        # TextBlob analysis
        textblob_result = analyzer.textblob_sentiment(text)
        print(f"Text: {text}")
        print(f"TextBlob sentiment: {textblob_result['sentiment']} (polarity: {textblob_result['polarity']:.2f})")
        
        # VADER analysis
        vader_result = analyzer.vader_sentiment(text)
        print(f"VADER sentiment: {vader_result['sentiment']} (compound: {vader_result['compound']:.2f})")
        print()


def chatbot_example():
    """Demonstrate chatbot capabilities"""
    print("=== Chatbot Example ===")
    
    bot = ChatBot("Demo Bot")
    
    messages = [
        "Hello!",
        "What can you do?",
        "Analyze this text: I'm feeling great today!",
        "Thank you!",
        "Goodbye!"
    ]
    
    for message in messages:
        response = bot.chat(message)
        print(f"User: {message}")
        print(f"Bot: {response['response']}")
        print(f"Intent: {response['analysis']['intent']}")
        print()


def visualization_example():
    """Demonstrate visualization capabilities"""
    print("=== Visualization Example ===")
    
    visualizer = NLPVisualizer()
    
    # Sample data
    texts = [
        "I love this amazing product!",
        "Terrible experience, very disappointed.",
        "It's an average item, nothing special.",
        "Fantastic quality and great service!",
        "Not worth the money, poor quality."
    ]
    
    sentiments = ["positive", "negative", "neutral", "positive", "negative"]
    
    # Create word cloud
    print("Creating word cloud...")
    wordcloud_fig = visualizer.word_cloud(texts)
    wordcloud_fig.savefig("../examples/wordcloud_example.png")
    print("Word cloud saved as 'wordcloud_example.png'")
    
    # Create sentiment distribution
    print("Creating sentiment distribution...")
    sentiment_fig = visualizer.sentiment_distribution(sentiments)
    sentiment_fig.savefig("../examples/sentiment_distribution.png")
    print("Sentiment distribution saved as 'sentiment_distribution.png'")
    print()


def main():
    """Run all examples"""
    print("NLP-Hub Examples Demo")
    print("=" * 50)
    
    try:
        text_preprocessing_example()
        sentiment_analysis_example()
        chatbot_example()
        visualization_example()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Some dependencies might be missing. Please install requirements.txt")


if __name__ == "__main__":
    main()