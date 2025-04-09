import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    """Class for preprocessing and feature engineering of trading data."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.target_column: str = "target"
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and engineer features from raw data."""
        df = df.copy()
        
        # Calculate technical indicators
        df = self._add_technical_indicators(df)
        
        # Calculate returns and volatility
        df = self._add_returns_and_volatility(df)
        
        # Create target variable (next day's return)
        df[self.target_column] = df['close'].pct_change().shift(-1)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Store feature columns
        self.feature_columns = [col for col in df.columns if col != self.target_column]
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        return df
    
    def _add_returns_and_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add returns and volatility features."""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']).diff()
        
        # Volatility
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_50'] = df['returns'].rolling(window=50).std()
        
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using the configured scaler."""
        if not self.feature_columns:
            raise ValueError("Feature columns not set. Call prepare_features first.")
        
        df_scaled = df.copy()
        df_scaled[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        return df_scaled
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance from the model if available."""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(self.feature_columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(zip(self.feature_columns, np.abs(model.coef_)))
        else:
            return {} 