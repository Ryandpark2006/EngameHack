import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from app.models.model_factory import ModelFactory
from app.backend.trading_simulator import TradingSimulator

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(100) * 10 + 100,
        'high': np.random.randn(100) * 10 + 105,
        'low': np.random.randn(100) * 10 + 95,
        'close': np.random.randn(100) * 10 + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    return data

@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    model = ModelFactory.create_model("decision_tree", {})
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    model.train(X, y)
    return model

def test_trading_simulator_initialization(sample_model):
    """Test trading simulator initialization."""
    simulator = TradingSimulator(sample_model, initial_balance=10000.0, commission=0.001)
    assert simulator.initial_balance == 10000.0
    assert simulator.commission == 0.001
    assert simulator.balance == 10000.0
    assert simulator.position == 0
    assert len(simulator.trades) == 0
    assert len(simulator.equity_curve) == 1
    assert simulator.equity_curve[0] == 10000.0

def test_trading_simulator_reset(sample_model):
    """Test trading simulator reset."""
    simulator = TradingSimulator(sample_model)
    simulator.balance = 5000.0
    simulator.position = 1.0
    simulator.trades = [{'action': 'buy', 'price': 100.0}]
    simulator.equity_curve = [10000.0, 9000.0]
    
    simulator.reset()
    assert simulator.balance == simulator.initial_balance
    assert simulator.position == 0
    assert len(simulator.trades) == 0
    assert len(simulator.equity_curve) == 1
    assert simulator.equity_curve[0] == simulator.initial_balance

def test_trading_simulator_backtest(sample_model, sample_data):
    """Test trading simulator backtest."""
    simulator = TradingSimulator(sample_model)
    features = ['open', 'high', 'low', 'close', 'volume']
    
    metrics = simulator.backtest(sample_data, features, 'close')
    
    assert isinstance(metrics, dict)
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'num_trades' in metrics
    assert 'win_rate' in metrics
    
    assert len(simulator.equity_curve) == len(sample_data) + 1
    assert simulator.equity_curve[0] == simulator.initial_balance

def test_trading_simulator_execute_trade(sample_model):
    """Test trading simulator trade execution."""
    simulator = TradingSimulator(sample_model)
    price = 100.0
    timestamp = datetime.now()
    
    # Test buy trade
    simulator._execute_trade('buy', price, timestamp)
    assert simulator.position > 0
    assert simulator.balance < simulator.initial_balance
    assert len(simulator.trades) == 1
    assert simulator.trades[0]['action'] == 'buy'
    
    # Test sell trade
    simulator._execute_trade('sell', price, timestamp)
    assert simulator.position == 0
    assert len(simulator.trades) == 2
    assert simulator.trades[1]['action'] == 'sell'

def test_trading_simulator_performance_metrics(sample_model, sample_data):
    """Test trading simulator performance metrics calculation."""
    simulator = TradingSimulator(sample_model)
    features = ['open', 'high', 'low', 'close', 'volume']
    
    simulator.backtest(sample_data, features, 'close')
    metrics = simulator._generate_performance_metrics()
    
    assert isinstance(metrics, dict)
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'num_trades' in metrics
    assert 'win_rate' in metrics
    
    # Test specific metric calculations
    equity_curve = pd.Series(simulator.equity_curve)
    returns = equity_curve.pct_change().dropna()
    
    expected_total_return = (equity_curve.iloc[-1] / simulator.initial_balance) - 1
    assert abs(metrics['total_return'] - expected_total_return) < 1e-10
    
    expected_sharpe = np.sqrt(252) * returns.mean() / returns.std()
    assert abs(metrics['sharpe_ratio'] - expected_sharpe) < 1e-10
    
    expected_drawdown = (equity_curve / equity_curve.cummax() - 1).min()
    assert abs(metrics['max_drawdown'] - expected_drawdown) < 1e-10 