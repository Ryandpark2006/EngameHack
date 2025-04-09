import requests
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TokenMetricsAPI:
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv('TOKEN_METRICS_API_KEY')
        logger.debug(f"API Key loaded: {'Yes' if self.api_key else 'No'}")
        if not self.api_key:
            logger.error("No API key found. Please set TOKEN_METRICS_API_KEY in your .env file or pass it directly.")
            raise ValueError("Token Metrics API key not found")
        
        self.base_url = "https://api.tokenmetrics.com/v2/"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Token {self.api_key}',
            'Content-Type': 'application/json'
        })
        logger.debug("API client initialized successfully")
    
    def get_token_metrics(self, symbol: str) -> Dict:
        """Get token metrics data for a specific symbol."""
        endpoint = f"{self.base_url}/tokens/{symbol}/metrics"
        params = {
            'symbol': symbol,
            'limit': 100
        }
        logger.debug(f"Fetching token metrics from {endpoint}")
        response = self.session.get(endpoint, params=params)
        logger.debug(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            return response.json()
        logger.error(f"Failed to fetch token metrics. Status code: {response.status_code}")
        logger.error(f"Response content: {response.text}")
        return {}
    
    def get_market_sentiment(self, symbol: str) -> Dict:
        """Get market sentiment data for a specific symbol."""
        endpoint = f"{self.base_url}/sentiment/{symbol}"
        logger.debug(f"Fetching market sentiment from {endpoint}")
        response = self.session.get(endpoint)
        logger.debug(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            return response.json()
        logger.error(f"Failed to fetch market sentiment. Status code: {response.status_code}")
        return {}
    
    def get_technical_indicators(self, symbol: str) -> Dict:
        """Get technical indicators for a specific symbol."""
        endpoint = f"{self.base_url}/technical/{symbol}"
        logger.debug(f"Fetching technical indicators from {endpoint}")
        response = self.session.get(endpoint)
        logger.debug(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            return response.json()
        logger.error(f"Failed to fetch technical indicators. Status code: {response.status_code}")
        return {}

class TokenMetricsDataProcessor:
    def __init__(self):
        self.api = TokenMetricsAPI()
    
    def fetch_and_process_data(self, symbol: str) -> pd.DataFrame:
        """Fetch and process token metrics data."""
        metrics = self.api.get_token_metrics(symbol)
        sentiment = self.api.get_market_sentiment(symbol)
        technical = self.api.get_technical_indicators(symbol)
        
        # Combine all data into a DataFrame
        data = {
            'timestamp': [],
            'price': [],
            'volume': [],
            'market_cap': [],
            'sentiment_score': [],
            'technical_score': []
        }
        
        # Process metrics data
        if metrics:
            data['timestamp'].extend(metrics.get('timestamps', []))
            data['price'].extend(metrics.get('prices', []))
            data['volume'].extend(metrics.get('volumes', []))
            data['market_cap'].extend(metrics.get('market_caps', []))
        
        # Process sentiment data
        if sentiment:
            data['sentiment_score'].extend(sentiment.get('scores', []))
        
        # Process technical data
        if technical:
            data['technical_score'].extend(technical.get('scores', []))
        
        return pd.DataFrame(data)
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various metrics from the data."""
        metrics = {
            'avg_sentiment': df['sentiment_score'].mean(),
            'avg_technical_score': df['technical_score'].mean(),
            'price_volatility': df['price'].std(),
            'volume_trend': df['volume'].pct_change().mean()
        }
        return metrics

class TokenMetricsAnalyzer:
    def __init__(self):
        self.processor = TokenMetricsDataProcessor()
    
    def analyze_token(self, symbol: str) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """Perform comprehensive analysis of a token."""
        # Fetch and process data
        df = self.processor.fetch_and_process_data(symbol)
        metrics = self.processor.calculate_metrics(df)
        
        # Calculate additional analysis
        analysis = {
            'data': df,
            'metrics': metrics,
            'summary': self.generate_summary(df, metrics)
        }
        return analysis
    
    def generate_summary(self, df: pd.DataFrame, metrics: Dict) -> Dict[str, str]:
        """Generate a summary of the analysis."""
        summary = {
            'trend': 'bullish' if metrics['avg_technical_score'] > 0.5 else 'bearish',
            'sentiment': 'positive' if metrics['avg_sentiment'] > 0.5 else 'negative',
            'volatility': 'high' if metrics['price_volatility'] > 0.1 else 'low',
            'volume': 'increasing' if metrics['volume_trend'] > 0 else 'decreasing'
        }
        return summary 