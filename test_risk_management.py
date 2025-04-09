import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from app.models.risk_management import (
    RiskManager,
    PositionSizer,
    RiskCalculator,
    RiskMonitor
)

@pytest.fixture
def sample_portfolio():
    """Create sample portfolio data for testing."""
    return {
        'BTC': {
            'quantity': 1.5,
            'avg_price': 50000,
            'current_price': 52000,
            'allocation': 0.6
        },
        'ETH': {
            'quantity': 10,
            'avg_price': 3000,
            'current_price': 3100,
            'allocation': 0.4
        }
    }

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'BTC_price': np.random.uniform(45000, 55000, len(dates)),
        'ETH_price': np.random.uniform(2800, 3200, len(dates)),
        'BTC_volume': np.random.uniform(1000, 5000, len(dates)),
        'ETH_volume': np.random.uniform(500, 2500, len(dates))
    })

@pytest.fixture
def sample_risk_metrics():
    """Create sample risk metrics for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'portfolio_value': np.random.uniform(100000, 150000, len(dates)),
        'drawdown': np.random.uniform(0, 0.2, len(dates)),
        'volatility': np.random.uniform(0.1, 0.3, len(dates)),
        'sharpe_ratio': np.random.uniform(0.5, 2.0, len(dates))
    })

def test_risk_manager_initialization():
    """Test RiskManager initialization."""
    risk_manager = RiskManager(
        max_position_size=0.1,
        max_drawdown=0.2,
        risk_per_trade=0.02,
        max_leverage=3.0
    )
    
    assert risk_manager.max_position_size == 0.1
    assert risk_manager.max_drawdown == 0.2
    assert risk_manager.risk_per_trade == 0.02
    assert risk_manager.max_leverage == 3.0

def test_position_sizing(sample_portfolio, sample_market_data):
    """Test PositionSizer."""
    position_sizer = PositionSizer(
        max_position_size=0.1,
        risk_per_trade=0.02
    )
    
    # Test position size calculation
    position_size = position_sizer.calculate_position_size(
        portfolio=sample_portfolio,
        symbol='BTC',
        entry_price=52000,
        stop_loss=50000
    )
    
    assert isinstance(position_size, float)
    assert 0 <= position_size <= 0.1
    
    # Test portfolio rebalancing
    new_allocations = position_sizer.rebalance_portfolio(
        portfolio=sample_portfolio,
        target_allocations={'BTC': 0.5, 'ETH': 0.5}
    )
    
    assert isinstance(new_allocations, dict)
    assert 'BTC' in new_allocations
    assert 'ETH' in new_allocations
    assert abs(new_allocations['BTC'] + new_allocations['ETH'] - 1.0) < 1e-6
    
    # Test leverage calculation
    leverage = position_sizer.calculate_leverage(
        portfolio=sample_portfolio,
        symbol='BTC',
        position_size=0.1
    )
    
    assert isinstance(leverage, float)
    assert 1.0 <= leverage <= 3.0

def test_risk_calculation(sample_portfolio, sample_market_data, sample_risk_metrics):
    """Test RiskCalculator."""
    risk_calculator = RiskCalculator()
    
    # Test value at risk calculation
    var = risk_calculator.calculate_var(
        portfolio=sample_portfolio,
        market_data=sample_market_data,
        confidence_level=0.95
    )
    
    assert isinstance(var, float)
    assert var < 0
    
    # Test expected shortfall calculation
    es = risk_calculator.calculate_expected_shortfall(
        portfolio=sample_portfolio,
        market_data=sample_market_data,
        confidence_level=0.95
    )
    
    assert isinstance(es, float)
    assert es < 0
    
    # Test portfolio risk metrics
    risk_metrics = risk_calculator.calculate_portfolio_risk(
        portfolio=sample_portfolio,
        market_data=sample_market_data
    )
    
    assert isinstance(risk_metrics, dict)
    assert 'volatility' in risk_metrics
    assert 'sharpe_ratio' in risk_metrics
    assert 'sortino_ratio' in risk_metrics
    assert 'max_drawdown' in risk_metrics
    
    # Test correlation analysis
    correlations = risk_calculator.analyze_correlations(
        portfolio=sample_portfolio,
        market_data=sample_market_data
    )
    
    assert isinstance(correlations, pd.DataFrame)
    assert 'BTC' in correlations.columns
    assert 'ETH' in correlations.columns

def test_risk_monitoring(sample_portfolio, sample_market_data, sample_risk_metrics):
    """Test RiskMonitor."""
    risk_monitor = RiskMonitor(
        max_drawdown=0.2,
        risk_threshold=0.1
    )
    
    # Test risk threshold monitoring
    threshold_alerts = risk_monitor.monitor_risk_thresholds(
        portfolio=sample_portfolio,
        risk_metrics=sample_risk_metrics
    )
    
    assert isinstance(threshold_alerts, list)
    for alert in threshold_alerts:
        assert isinstance(alert, dict)
        assert 'type' in alert
        assert 'message' in alert
        assert 'severity' in alert
    
    # Test portfolio health check
    health_report = risk_monitor.check_portfolio_health(
        portfolio=sample_portfolio,
        market_data=sample_market_data
    )
    
    assert isinstance(health_report, dict)
    assert 'overall_health' in health_report
    assert 'risk_factors' in health_report
    assert 'recommendations' in health_report
    
    # Test risk exposure analysis
    exposure_report = risk_monitor.analyze_risk_exposure(
        portfolio=sample_portfolio,
        market_data=sample_market_data
    )
    
    assert isinstance(exposure_report, dict)
    assert 'market_exposure' in exposure_report
    assert 'sector_exposure' in exposure_report
    assert 'concentration_risk' in exposure_report

def test_risk_mitigation(sample_portfolio, sample_market_data):
    """Test risk mitigation strategies."""
    risk_manager = RiskManager(
        max_position_size=0.1,
        max_drawdown=0.2,
        risk_per_trade=0.02,
        max_leverage=3.0
    )
    
    # Test stop loss calculation
    stop_loss = risk_manager.calculate_stop_loss(
        portfolio=sample_portfolio,
        symbol='BTC',
        entry_price=52000,
        risk_amount=1000
    )
    
    assert isinstance(stop_loss, float)
    assert stop_loss < 52000
    
    # Test take profit calculation
    take_profit = risk_manager.calculate_take_profit(
        portfolio=sample_portfolio,
        symbol='BTC',
        entry_price=52000,
        risk_reward_ratio=2.0,
        stop_loss=50000
    )
    
    assert isinstance(take_profit, float)
    assert take_profit > 52000
    
    # Test position adjustment
    adjusted_portfolio = risk_manager.adjust_positions(
        portfolio=sample_portfolio,
        market_data=sample_market_data,
        risk_metrics={'drawdown': 0.25}
    )
    
    assert isinstance(adjusted_portfolio, dict)
    assert 'BTC' in adjusted_portfolio
    assert 'ETH' in adjusted_portfolio
    assert adjusted_portfolio['BTC']['quantity'] <= sample_portfolio['BTC']['quantity']
    assert adjusted_portfolio['ETH']['quantity'] <= sample_portfolio['ETH']['quantity'] 