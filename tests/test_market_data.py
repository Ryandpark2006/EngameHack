import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from app.models.market_data import (
    MarketDataManager,
    DataFetcher,
    DataProcessor,
    DataValidator
)

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(45000, 55000, len(dates)),
        'high': np.random.uniform(46000, 56000, len(dates)),
        'low': np.random.uniform(44000, 54000, len(dates)),
        'close': np.random.uniform(45000, 55000, len(dates)),
        'volume': np.random.uniform(1000, 5000, len(dates))
    })

@pytest.fixture
def sample_technical_indicators():
    """Create sample technical indicators for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'rsi': np.random.uniform(30, 70, len(dates)),
        'macd': np.random.uniform(-100, 100, len(dates)),
        'macd_signal': np.random.uniform(-100, 100, len(dates)),
        'macd_hist': np.random.uniform(-50, 50, len(dates)),
        'bb_upper': np.random.uniform(46000, 56000, len(dates)),
        'bb_middle': np.random.uniform(45000, 55000, len(dates)),
        'bb_lower': np.random.uniform(44000, 54000, len(dates))
    })

@pytest.fixture
def sample_sentiment_data():
    """Create sample sentiment data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'sentiment_score': np.random.uniform(-1, 1, len(dates)),
        'sentiment_magnitude': np.random.uniform(0, 1, len(dates)),
        'news_count': np.random.randint(0, 100, len(dates)),
        'social_volume': np.random.randint(0, 1000, len(dates))
    })

def test_market_data_manager_initialization():
    """Test MarketDataManager initialization."""
    data_manager = MarketDataManager(
        data_sources=['binance', 'coingecko'],
        update_interval=300
    )
    
    assert data_manager.data_sources == ['binance', 'coingecko']
    assert data_manager.update_interval == 300

def test_data_fetching(sample_market_data):
    """Test DataFetcher."""
    data_fetcher = DataFetcher(
        source='binance',
        symbol='BTC/USDT',
        timeframe='1d'
    )
    
    # Test market data fetching
    market_data = data_fetcher.fetch_market_data(
        start_time='2023-01-01',
        end_time='2023-01-31'
    )
    
    assert isinstance(market_data, pd.DataFrame)
    assert all(col in market_data.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Test technical indicators fetching
    indicators = data_fetcher.fetch_technical_indicators(
        data=sample_market_data,
        indicators=['rsi', 'macd', 'bollinger_bands']
    )
    
    assert isinstance(indicators, pd.DataFrame)
    assert all(col in indicators.columns for col in ['rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower'])
    
    # Test sentiment data fetching
    sentiment_data = data_fetcher.fetch_sentiment_data(
        symbol='BTC',
        start_time='2023-01-01',
        end_time='2023-01-31'
    )
    
    assert isinstance(sentiment_data, pd.DataFrame)
    assert all(col in sentiment_data.columns for col in ['timestamp', 'sentiment_score', 'sentiment_magnitude', 'news_count', 'social_volume'])

def test_data_processing(sample_market_data, sample_technical_indicators, sample_sentiment_data):
    """Test DataProcessor."""
    data_processor = DataProcessor()
    
    # Test data cleaning
    cleaned_data = data_processor.clean_data(sample_market_data)
    
    assert isinstance(cleaned_data, pd.DataFrame)
    assert not cleaned_data.isnull().any().any()
    assert not cleaned_data.duplicated().any()
    
    # Test data normalization
    normalized_data = data_processor.normalize_data(sample_market_data)
    
    assert isinstance(normalized_data, pd.DataFrame)
    assert all(0 <= val <= 1 for col in normalized_data.columns if col != 'timestamp' for val in normalized_data[col])
    
    # Test feature engineering
    features = data_processor.create_features(
        market_data=sample_market_data,
        technical_indicators=sample_technical_indicators,
        sentiment_data=sample_sentiment_data
    )
    
    assert isinstance(features, pd.DataFrame)
    assert 'timestamp' in features.columns
    assert len(features.columns) > len(sample_market_data.columns)
    
    # Test data aggregation
    aggregated_data = data_processor.aggregate_data(
        data=sample_market_data,
        timeframe='1W'
    )
    
    assert isinstance(aggregated_data, pd.DataFrame)
    assert len(aggregated_data) < len(sample_market_data)

def test_data_validation(sample_market_data, sample_technical_indicators, sample_sentiment_data):
    """Test DataValidator."""
    data_validator = DataValidator()
    
    # Test data quality check
    quality_report = data_validator.check_data_quality(sample_market_data)
    
    assert isinstance(quality_report, dict)
    assert 'completeness' in quality_report
    assert 'accuracy' in quality_report
    assert 'consistency' in quality_report
    assert 'timeliness' in quality_report
    
    # Test data consistency check
    consistency_report = data_validator.check_data_consistency(
        market_data=sample_market_data,
        technical_indicators=sample_technical_indicators,
        sentiment_data=sample_sentiment_data
    )
    
    assert isinstance(consistency_report, dict)
    assert 'timestamp_alignment' in consistency_report
    assert 'value_ranges' in consistency_report
    assert 'correlations' in consistency_report
    
    # Test anomaly detection
    anomalies = data_validator.detect_anomalies(sample_market_data)
    
    assert isinstance(anomalies, pd.DataFrame)
    assert 'timestamp' in anomalies.columns
    assert 'anomaly_score' in anomalies.columns
    assert 'anomaly_type' in anomalies.columns

def test_data_storage(sample_market_data, sample_technical_indicators, sample_sentiment_data):
    """Test data storage functionality."""
    data_manager = MarketDataManager(
        data_sources=['binance'],
        update_interval=300
    )
    
    # Test data storage
    storage_result = data_manager.store_data(
        market_data=sample_market_data,
        technical_indicators=sample_technical_indicators,
        sentiment_data=sample_sentiment_data
    )
    
    assert isinstance(storage_result, dict)
    assert 'success' in storage_result
    assert 'message' in storage_result
    
    # Test data retrieval
    retrieved_data = data_manager.retrieve_data(
        start_time='2023-01-01',
        end_time='2023-01-31',
        data_types=['market', 'technical', 'sentiment']
    )
    
    assert isinstance(retrieved_data, dict)
    assert 'market_data' in retrieved_data
    assert 'technical_indicators' in retrieved_data
    assert 'sentiment_data' in retrieved_data
    
    # Test data update
    update_result = data_manager.update_data(
        new_data=sample_market_data,
        data_type='market'
    )
    
    assert isinstance(update_result, dict)
    assert 'success' in update_result
    assert 'rows_updated' in update_result 