"""
Data visualization utilities for NLP insights
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import networkx as nx
from typing import Dict, List, Union, Optional, Any, Tuple
import io
import base64


class NLPVisualizer:
    """
    Comprehensive visualization toolkit for NLP data and insights
    """
    
    def __init__(self, style: str = 'default'):
        """
        Initialize the visualizer
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 10)
    
    def word_cloud(self, text: Union[str, List[str]], 
                   width: int = 800, height: int = 400,
                   background_color: str = 'white',
                   max_words: int = 100,
                   colormap: str = 'viridis') -> plt.Figure:
        """
        Generate a word cloud from text
        
        Args:
            text: Input text or list of texts
            width: Width of the word cloud
            height: Height of the word cloud
            background_color: Background color
            max_words: Maximum number of words to display
            colormap: Color map for the words
            
        Returns:
            Matplotlib figure
        """
        if isinstance(text, list):
            text = ' '.join(text)
        
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            colormap=colormap,
            relative_scaling=0.5,
            random_state=42
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        return fig
    
    def sentiment_distribution(self, sentiments: List[str], 
                             title: str = "Sentiment Distribution") -> plt.Figure:
        """
        Visualize sentiment distribution
        
        Args:
            sentiments: List of sentiment labels
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        sentiment_counts = Counter(sentiments)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot
        labels = list(sentiment_counts.keys())
        counts = list(sentiment_counts.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99'][:len(labels)]
        
        ax1.bar(labels, counts, color=colors)
        ax1.set_title(f'{title} - Bar Chart')
        ax1.set_ylabel('Count')
        
        # Pie chart
        ax2.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
        ax2.set_title(f'{title} - Pie Chart')
        
        plt.tight_layout()
        return fig
    
    def topic_visualization(self, topics: List[Dict[str, Any]], 
                           title: str = "Topic Modeling Results") -> plt.Figure:
        """
        Visualize topic modeling results
        
        Args:
            topics: List of topic dictionaries with 'words' and 'probabilities'
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        n_topics = len(topics)
        fig, axes = plt.subplots(2, (n_topics + 1) // 2, figsize=(15, 8))
        if n_topics == 1:
            axes = [axes]
        axes = axes.flatten()
        
        for i, topic in enumerate(topics):
            if i < len(axes):
                words = topic['words'][:10]  # Top 10 words
                probs = topic['probabilities'][:10]
                
                axes[i].barh(words, probs, color=self.colors[i % len(self.colors)])
                axes[i].set_title(f'Topic {i+1}')
                axes[i].set_xlabel('Probability')
        
        # Hide extra subplots
        for i in range(len(topics), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def confusion_matrix(self, y_true: List[str], y_pred: List[str], 
                        labels: Optional[List[str]] = None,
                        title: str = "Confusion Matrix") -> plt.Figure:
        """
        Visualize confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label names
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import confusion_matrix as cm
        
        matrix = cm(y_true, y_pred, labels=labels)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        return fig
    
    def text_length_distribution(self, texts: List[str], 
                               title: str = "Text Length Distribution") -> plt.Figure:
        """
        Visualize distribution of text lengths
        
        Args:
            texts: List of texts
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        lengths = [len(text.split()) for text in texts]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title(f'{title} - Histogram')
        ax1.set_xlabel('Number of Words')
        ax1.set_ylabel('Frequency')
        
        # Box plot
        ax2.boxplot(lengths)
        ax2.set_title(f'{title} - Box Plot')
        ax2.set_ylabel('Number of Words')
        
        plt.tight_layout()
        return fig
    
    def feature_importance(self, features: List[str], importance: List[float],
                          title: str = "Feature Importance", top_n: int = 20) -> plt.Figure:
        """
        Visualize feature importance
        
        Args:
            features: List of feature names
            importance: List of importance scores
            title: Plot title
            top_n: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1][:top_n]
        sorted_features = [features[i] for i in sorted_idx]
        sorted_importance = [importance[i] for i in sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(sorted_features))
        
        ax.barh(y_pos, sorted_importance, color='lightgreen')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        ax.invert_yaxis()  # Top feature at the top
        
        return fig
    
    def interactive_sentiment_timeline(self, dates: List[str], 
                                     sentiments: List[str],
                                     title: str = "Sentiment Over Time") -> go.Figure:
        """
        Create interactive sentiment timeline using Plotly
        
        Args:
            dates: List of dates
            sentiments: List of sentiment labels
            title: Plot title
            
        Returns:
            Plotly figure
        """
        df = pd.DataFrame({'date': dates, 'sentiment': sentiments})
        df['date'] = pd.to_datetime(df['date'])
        
        # Count sentiments by date
        sentiment_counts = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
        
        fig = px.line(sentiment_counts, x='date', y='count', color='sentiment',
                     title=title, markers=True)
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Count",
            hovermode='x unified'
        )
        
        return fig
    
    def interactive_word_frequency(self, text: str, top_n: int = 20,
                                  title: str = "Word Frequency") -> go.Figure:
        """
        Create interactive word frequency chart
        
        Args:
            text: Input text
            top_n: Number of top words to show
            title: Plot title
            
        Returns:
            Plotly figure
        """
        words = text.lower().split()
        word_counts = Counter(words)
        
        # Get top N words
        top_words = word_counts.most_common(top_n)
        words, counts = zip(*top_words)
        
        fig = go.Figure(data=[
            go.Bar(x=list(words), y=list(counts), 
                  marker_color='lightblue',
                  text=list(counts),
                  textposition='auto')
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Words",
            yaxis_title="Frequency",
            xaxis_tickangle=-45
        )
        
        return fig
    
    def network_graph(self, entities: List[Tuple[str, str]], 
                     title: str = "Entity Relationship Network") -> plt.Figure:
        """
        Create a network graph of entity relationships
        
        Args:
            entities: List of entity pairs (entity1, entity2)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        G = nx.Graph()
        
        # Add edges
        for entity1, entity2 in entities:
            G.add_edge(entity1, entity2)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw network
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue',
                node_size=1000, font_size=8, font_weight='bold',
                edge_color='gray', alpha=0.7)
        
        ax.set_title(title)
        ax.axis('off')
        
        return fig
    
    def classification_report_heatmap(self, classification_report: Dict[str, Any],
                                    title: str = "Classification Report") -> plt.Figure:
        """
        Visualize classification report as heatmap
        
        Args:
            classification_report: sklearn classification report dictionary
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Extract relevant metrics
        classes = [k for k in classification_report.keys() 
                  if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        metrics = ['precision', 'recall', 'f1-score']
        data = []
        
        for metric in metrics:
            row = [classification_report[cls][metric] for cls in classes]
            data.append(row)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=classes, yticklabels=metrics, ax=ax)
        ax.set_title(title)
        
        return fig
    
    def sentiment_radar_chart(self, sentiment_scores: Dict[str, float],
                            title: str = "Sentiment Radar Chart") -> go.Figure:
        """
        Create a radar chart for sentiment analysis
        
        Args:
            sentiment_scores: Dictionary with sentiment dimensions and scores
            title: Plot title
            
        Returns:
            Plotly figure
        """
        categories = list(sentiment_scores.keys())
        values = list(sentiment_scores.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Sentiment Scores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=title
        )
        
        return fig
    
    def save_figure(self, fig, filepath: str, format: str = 'png', **kwargs):
        """
        Save figure to file
        
        Args:
            fig: Matplotlib or Plotly figure
            filepath: Path to save the figure
            format: File format ('png', 'pdf', 'svg', 'html')
            **kwargs: Additional arguments for saving
        """
        if hasattr(fig, 'write_html'):  # Plotly figure
            if format == 'html':
                fig.write_html(filepath, **kwargs)
            else:
                fig.write_image(filepath, format=format, **kwargs)
        else:  # Matplotlib figure
            fig.savefig(filepath, format=format, **kwargs)
    
    def create_dashboard_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary dashboard with multiple visualizations
        
        Args:
            data: Dictionary containing various data for visualization
            
        Returns:
            Dictionary with multiple figures
        """
        dashboard = {}
        
        # Word cloud if text data is available
        if 'texts' in data:
            dashboard['wordcloud'] = self.word_cloud(data['texts'])
        
        # Sentiment analysis if available
        if 'sentiments' in data:
            dashboard['sentiment_dist'] = self.sentiment_distribution(data['sentiments'])
        
        # Text length distribution
        if 'texts' in data:
            dashboard['text_lengths'] = self.text_length_distribution(data['texts'])
        
        # Topic modeling if available
        if 'topics' in data:
            dashboard['topics'] = self.topic_visualization(data['topics'])
        
        # Feature importance if available
        if 'features' in data and 'importance' in data:
            dashboard['feature_importance'] = self.feature_importance(
                data['features'], data['importance']
            )
        
        return dashboard
    
    def figure_to_base64(self, fig) -> str:
        """
        Convert matplotlib figure to base64 string for web display
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        
        return image_base64