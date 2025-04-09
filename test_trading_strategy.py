import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from app.models.trading_strategy import (
    StrategyManager,
    SignalGenerator,
    Backtester,
    StrategyOptimizer
)

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(45000, 55000, len(dates)),
        'high': np.random.uniform(46000, 56000, len(dates)),
        'low': np.random.uniform(44000, 54000, len(dates)),
        'close': np.random.uniform(45000, 55000, len(dates)),
        'volume': np.random.uniform(1000, 5000, len(dates)),
        'rsi': np.random.uniform(30, 70, len(dates)),
        'macd': np.random.uniform(-100, 100, len(dates)),
        'macd_signal': np.random.uniform(-100, 100, len(dates)),
        'macd_hist': np.random.uniform(-50, 50, len(dates)),
        'sentiment_score': np.random.uniform(-1, 1, len(dates)),
        'sentiment_magnitude': np.random.uniform(0, 1, len(dates))
    })

@pytest.fixture
def sample_signals():
    """Create sample trading signals for testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', end='2023-01-31', freq='D'),
        'signal': np.random.choice([-1, 0, 1], size=31),
        'strength': np.random.uniform(0, 1, size=31),
        'confidence': np.random.uniform(0, 1, size=31)
    })

def test_strategy_manager_initialization():
    """Test StrategyManager initialization."""
    strategy_manager = StrategyManager(
        strategy_type='trend_following',
        parameters={'window': 20, 'threshold': 0.02}
    )
    
    assert strategy_manager.strategy_type == 'trend_following'
    assert strategy_manager.parameters == {'window': 20, 'threshold': 0.02}

def test_signal_generation(sample_data):
    """Test SignalGenerator."""
    signal_generator = SignalGenerator(
        strategy_type='trend_following',
        parameters={'window': 20, 'threshold': 0.02}
    )
    
    # Test signal generation
    signals = signal_generator.generate_signals(sample_data)
    
    assert isinstance(signals, pd.DataFrame)
    assert 'signal' in signals.columns
    assert 'strength' in signals.columns
    assert 'confidence' in signals.columns
    assert all(s in [-1, 0, 1] for s in signals['signal'])
    assert all(0 <= s <= 1 for s in signals['strength'])
    assert all(0 <= c <= 1 for c in signals['confidence'])
    
    # Test signal filtering
    filtered_signals = signal_generator.filter_signals(
        signals,
        min_strength=0.5,
        min_confidence=0.7
    )
    
    assert isinstance(filtered_signals, pd.DataFrame)
    assert len(filtered_signals) <= len(signals)

def test_backtesting(sample_data, sample_signals):
    """Test Backtester."""
    backtester = Backtester(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    
    # Test backtest execution
    results = backtester.run_backtest(
        data=sample_data,
        signals=sample_signals
    )
    
    assert isinstance(results, dict)
    assert 'returns' in results
    assert 'positions' in results
    assert 'trades' in results
    assert 'performance_metrics' in results
    
    # Test performance metrics
    metrics = results['performance_metrics']
    assert isinstance(metrics, dict)
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics
    
    # Test trade analysis
    trade_analysis = backtester.analyze_trades(results['trades'])
    
    assert isinstance(trade_analysis, dict)
    assert 'avg_trade_return' in trade_analysis
    assert 'avg_winning_trade' in trade_analysis
    assert 'avg_losing_trade' in trade_analysis
    assert 'profit_factor' in trade_analysis

def test_strategy_optimization(sample_data):
    """Test StrategyOptimizer."""
    optimizer = StrategyOptimizer(
        strategy_type='trend_following',
        parameter_ranges={
            'window': (10, 50),
            'threshold': (0.01, 0.05)
        }
    )
    
    # Test parameter optimization
    best_params = optimizer.optimize_parameters(
        data=sample_data,
        metric='sharpe_ratio'
    )
    
    assert isinstance(best_params, dict)
    assert 'window' in best_params
    assert 'threshold' in best_params
    assert 10 <= best_params['window'] <= 50
    assert 0.01 <= best_params['threshold'] <= 0.05
    
    # Test walk-forward optimization
    walk_forward_results = optimizer.walk_forward_optimization(
        data=sample_data,
        train_size=0.7,
        test_size=0.3
    )
    
    assert isinstance(walk_forward_results, dict)
    assert 'best_parameters' in walk_forward_results
    assert 'test_performance' in walk_forward_results
    assert 'stability_metrics' in walk_forward_results

def test_strategy_validation(sample_data, sample_signals):
    """Test strategy validation."""
    strategy_manager = StrategyManager(
        strategy_type='trend_following',
        parameters={'window': 20, 'threshold': 0.02}
    )
    
    # Test strategy validation
    validation_results = strategy_manager.validate_strategy(
        data=sample_data,
        signals=sample_signals
    )
    
    assert isinstance(validation_results, dict)
    assert 'performance_metrics' in validation_results
    assert 'risk_metrics' in validation_results
    assert 'stability_metrics' in validation_results
    
    # Test robustness check
    robustness_results = strategy_manager.check_robustness(
        data=sample_data,
        signals=sample_signals,
        n_simulations=10
    )
    
    assert isinstance(robustness_results, dict)
    assert 'avg_performance' in robustness_results
    assert 'performance_std' in robustness_results
    assert 'success_rate' in robustness_results

def test_strategy_monitoring(sample_data, sample_signals):
    """Test strategy monitoring."""
    strategy_manager = StrategyManager(
        strategy_type='trend_following',
        parameters={'window': 20, 'threshold': 0.02}
    )
    
    # Test performance monitoring
    performance_metrics = strategy_manager.monitor_performance(
        data=sample_data,
        signals=sample_signals
    )
    
    assert isinstance(performance_metrics, dict)
    assert 'current_performance' in performance_metrics
    assert 'performance_trend' in performance_metrics
    assert 'risk_metrics' in performance_metrics
    
    # Test strategy health check
    health_status = strategy_manager.check_strategy_health(
        data=sample_data,
        signals=sample_signals
    )
    
    assert isinstance(health_status, dict)
    assert 'status' in health_status
    assert 'issues' in health_status
    assert 'recommendations' in health_status 