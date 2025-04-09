import os
import requests
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class TokenMetricsAPI:
    """Class for interacting with the Token Metrics API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TOKEN_METRICS_API_KEY")
        if not self.api_key:
            raise ValueError("Token Metrics API key not found")
        
        self.base_url = "https://api.tokenmetrics.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical price and indicator data."""
        endpoint = f"{self.base_url}/historical"
        params = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval
        }
        
        response = requests.get(endpoint, headers=self.headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        return pd.DataFrame(data)
    
    def get_indicators(self, symbol: str) -> List[Dict]:
        """Get available indicators for a symbol."""
        endpoint = f"{self.base_url}/indicators/{symbol}"
        response = requests.get(endpoint, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def get_realtime_data(self, symbol: str) -> Dict:
        """Get real-time data for a symbol."""
        endpoint = f"{self.base_url}/realtime/{symbol}"
        response = requests.get(endpoint, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def validate_api_key(self) -> bool:
        """Validate the API key."""
        try:
            self.get_indicators("BTC")
            return True
        except requests.exceptions.RequestException:
            return False 