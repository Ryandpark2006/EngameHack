import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime

from app.models.base_model import BaseTradingModel
from app.models.trading_strategy import BaseStrategy
from app.models.risk_management import RiskManager

class TradingSimulator:
    """Class for simulating trading strategies."""
    
    def __init__(
        self,
        model: BaseTradingModel,
        initial_balance: float = 10000.0,
        commission: float = 0.001
    ):
        self.model = model
        self.initial_balance = initial_balance
        self.commission = commission
        self.reset()
    
    def reset(self):
        """Reset the simulator state."""
        self.balance = self.initial_balance
        self.position = 0
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [self.initial_balance]
    
    def backtest(
        self,
        data: pd.DataFrame,
        features: List[str],
        target: str
    ) -> Dict:
        """Run backtest on historical data."""
        self.reset()
        
        for i in range(len(data)):
            current_data = data.iloc[i]
            current_price = current_data['close']
            
            # Prepare features for prediction
            X = pd.DataFrame([current_data[features]])
            
            # Get model prediction
            prediction = self.model.predict(X)[0]
            
            # Execute trade based on prediction
            if prediction > 0.5 and self.position <= 0:  # Buy signal
                self._execute_trade('buy', current_price, current_data.name)
            elif prediction < -0.5 and self.position >= 0:  # Sell signal
                self._execute_trade('sell', current_price, current_data.name)
            
            # Update equity curve
            current_equity = self.balance + (self.position * current_price)
            self.equity_curve.append(current_equity)
        
        return self._generate_performance_metrics()
    
    def _execute_trade(self, action: str, price: float, timestamp: datetime):
        """Execute a trade and update position."""
        if action == 'buy':
            shares = self.balance / price
            cost = shares * price * (1 + self.commission)
            if cost <= self.balance:
                self.position = shares
                self.balance -= cost
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': price,
                    'shares': shares
                })
        
        elif action == 'sell':
            if self.position > 0:
                proceeds = self.position * price * (1 - self.commission)
                self.balance += proceeds
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'sell',
                    'price': price,
                    'shares': self.position
                })
                self.position = 0
    
    def _generate_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        equity_curve = pd.Series(self.equity_curve)
        returns = equity_curve.pct_change().dropna()
        
        return {
            'total_return': (equity_curve.iloc[-1] / self.initial_balance) - 1,
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std(),
            'max_drawdown': (equity_curve / equity_curve.cummax() - 1).min(),
            'num_trades': len(self.trades),
            'win_rate': len([t for t in self.trades if t['action'] == 'sell']) / len(self.trades) if self.trades else 0
        } 