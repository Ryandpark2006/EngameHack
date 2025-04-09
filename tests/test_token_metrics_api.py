import pytest
import os
from datetime import datetime, timedelta
from app.backend.token_metrics_api import TokenMetricsAPI

@pytest.fixture
def api():
    """Create a TokenMetricsAPI instance for testing."""
    api_key = os.getenv('TOKEN_METRICS_API_KEY')
    if not api_key:
        pytest.skip("TOKEN_METRICS_API_KEY not set")
    return TokenMetricsAPI(api_key)

def test_get_historical_data(api):
    """Test getting historical data."""
    # Test with valid parameters
    data = api.get_historical_data(
        symbol='BTC',
        start_date='2023-01-01',
        end_date='2023-01-31',
        interval='1d'
    )
    
    assert isinstance(data, list)
    if data:  # If we got data back
        assert all(isinstance(item, dict) for item in data)
        assert all('timestamp' in item for item in data)
        assert all('open' in item for item in data)
        assert all('high' in item for item in data)
        assert all('low' in item for item in data)
        assert all('close' in item for item in data)
        assert all('volume' in item for item in data)

def test_get_technical_indicators(api):
    """Test getting technical indicators."""
    # Test with valid parameters
    indicators = api.get_technical_indicators(
        symbol='BTC',
        start_date='2023-01-01',
        end_date='2023-01-31',
        interval='1d'
    )
    
    assert isinstance(indicators, list)
    if indicators:  # If we got indicators back
        assert all(isinstance(item, dict) for item in indicators)
        assert all('timestamp' in item for item in indicators)
        assert all('rsi' in item for item in indicators)
        assert all('macd' in item for item in indicators)
        assert all('macd_signal' in item for item in indicators)
        assert all('macd_hist' in item for item in indicators)

def test_get_sentiment_analysis(api):
    """Test getting sentiment analysis."""
    # Test with valid parameters
    sentiment = api.get_sentiment_analysis(
        symbol='BTC',
        start_date='2023-01-01',
        end_date='2023-01-31'
    )
    
    assert isinstance(sentiment, list)
    if sentiment:  # If we got sentiment data back
        assert all(isinstance(item, dict) for item in sentiment)
        assert all('timestamp' in item for item in sentiment)
        assert all('sentiment_score' in item for item in sentiment)
        assert all('sentiment_magnitude' in item for item in sentiment)

def test_get_market_overview(api):
    """Test getting market overview."""
    # Test with valid parameters
    overview = api.get_market_overview(
        symbols=['BTC', 'ETH'],
        start_date='2023-01-01',
        end_date='2023-01-31'
    )
    
    assert isinstance(overview, list)
    if overview:  # If we got overview data back
        assert all(isinstance(item, dict) for item in overview)
        assert all('symbol' in item for item in overview)
        assert all('price' in item for item in overview)
        assert all('market_cap' in item for item in overview)
        assert all('volume_24h' in item for item in overview)

def test_get_ai_predictions(api):
    """Test getting AI predictions."""
    # Test with valid parameters
    predictions = api.get_ai_predictions(
        symbol='BTC',
        start_date='2023-01-01',
        end_date='2023-01-31'
    )
    
    assert isinstance(predictions, list)
    if predictions:  # If we got predictions back
        assert all(isinstance(item, dict) for item in predictions)
        assert all('timestamp' in item for item in predictions)
        assert all('predicted_price' in item for item in predictions)
        assert all('confidence' in item for item in predictions)

def test_invalid_api_key():
    """Test behavior with invalid API key."""
    api = TokenMetricsAPI('invalid_key')
    
    with pytest.raises(Exception):
        api.get_historical_data(
            symbol='BTC',
            start_date='2023-01-01',
            end_date='2023-01-31'
        )

def test_invalid_parameters(api):
    """Test behavior with invalid parameters."""
    with pytest.raises(ValueError):
        api.get_historical_data(
            symbol='BTC',
            start_date='invalid_date',
            end_date='2023-01-31'
        )
    
    with pytest.raises(ValueError):
        api.get_historical_data(
            symbol='BTC',
            start_date='2023-01-01',
            end_date='2023-01-31',
            interval='invalid_interval'
        )

def test_rate_limiting(api):
    """Test rate limiting behavior."""
    # Make multiple requests in quick succession
    for _ in range(5):
        try:
            api.get_historical_data(
                symbol='BTC',
                start_date='2023-01-01',
                end_date='2023-01-02'
            )
        except Exception as e:
            if "rate limit" in str(e).lower():
                pytest.skip("Rate limit reached")
            raise 