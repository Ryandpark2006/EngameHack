import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from app.models.ml_models import (
    DecisionTreeModel,
    RandomForestModel,
    XGBoostModel,
    LSTMModel,
    TransformerModel
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(100, 10, len(dates)),
        'high': np.random.normal(105, 10, len(dates)),
        'low': np.random.normal(95, 10, len(dates)),
        'close': np.random.normal(100, 10, len(dates)),
        'volume': np.random.normal(1000, 100, len(dates)),
        'rsi': np.random.normal(50, 10, len(dates)),
        'macd': np.random.normal(0, 1, len(dates)),
        'macd_signal': np.random.normal(0, 1, len(dates)),
        'macd_hist': np.random.normal(0, 1, len(dates)),
        'sentiment_score': np.random.normal(0, 1, len(dates)),
        'sentiment_magnitude': np.random.normal(1, 0.1, len(dates))
    })
    return data

@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    return [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'sentiment_score', 'sentiment_magnitude'
    ]

def test_decision_tree_model(sample_data, sample_features):
    """Test DecisionTreeModel."""
    model = DecisionTreeModel()
    
    # Test training
    model.train(sample_data, sample_features)
    assert model.model is not None
    
    # Test prediction
    prediction = model.predict(sample_data.iloc[-1:])
    assert isinstance(prediction, float)
    
    # Test feature importance
    importance = model.get_feature_importance()
    assert isinstance(importance, dict)
    assert all(f in importance for f in sample_features)

def test_random_forest_model(sample_data, sample_features):
    """Test RandomForestModel."""
    model = RandomForestModel()
    
    # Test training
    model.train(sample_data, sample_features)
    assert model.model is not None
    
    # Test prediction
    prediction = model.predict(sample_data.iloc[-1:])
    assert isinstance(prediction, float)
    
    # Test feature importance
    importance = model.get_feature_importance()
    assert isinstance(importance, dict)
    assert all(f in importance for f in sample_features)

def test_xgboost_model(sample_data, sample_features):
    """Test XGBoostModel."""
    model = XGBoostModel()
    
    # Test training
    model.train(sample_data, sample_features)
    assert model.model is not None
    
    # Test prediction
    prediction = model.predict(sample_data.iloc[-1:])
    assert isinstance(prediction, float)
    
    # Test feature importance
    importance = model.get_feature_importance()
    assert isinstance(importance, dict)
    assert all(f in importance for f in sample_features)

def test_lstm_model(sample_data, sample_features):
    """Test LSTMModel."""
    model = LSTMModel()
    
    # Test training
    model.train(sample_data, sample_features)
    assert model.model is not None
    
    # Test prediction
    prediction = model.predict(sample_data.iloc[-1:])
    assert isinstance(prediction, float)
    
    # Test sequence prediction
    sequence = model.predict_sequence(sample_data.iloc[-10:])
    assert isinstance(sequence, list)
    assert all(isinstance(p, float) for p in sequence)

def test_transformer_model(sample_data, sample_features):
    """Test TransformerModel."""
    model = TransformerModel()
    
    # Test training
    model.train(sample_data, sample_features)
    assert model.model is not None
    
    # Test prediction
    prediction = model.predict(sample_data.iloc[-1:])
    assert isinstance(prediction, float)
    
    # Test sequence prediction
    sequence = model.predict_sequence(sample_data.iloc[-10:])
    assert isinstance(sequence, list)
    assert all(isinstance(p, float) for p in sequence)

def test_model_save_load(sample_data, sample_features):
    """Test model save and load functionality."""
    models = [
        DecisionTreeModel(),
        RandomForestModel(),
        XGBoostModel(),
        LSTMModel(),
        TransformerModel()
    ]
    
    for model in models:
        # Train model
        model.train(sample_data, sample_features)
        
        # Save model
        model.save('test_model')
        
        # Create new instance and load
        new_model = type(model)()
        new_model.load('test_model')
        
        # Test prediction
        original_pred = model.predict(sample_data.iloc[-1:])
        loaded_pred = new_model.predict(sample_data.iloc[-1:])
        
        assert abs(original_pred - loaded_pred) < 1e-6

def test_model_validation(sample_data, sample_features):
    """Test model validation functionality."""
    models = [
        DecisionTreeModel(),
        RandomForestModel(),
        XGBoostModel(),
        LSTMModel(),
        TransformerModel()
    ]
    
    for model in models:
        # Train model
        model.train(sample_data, sample_features)
        
        # Validate model
        metrics = model.validate(sample_data)
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert all(isinstance(v, float) for v in metrics.values())

def test_model_hyperparameter_tuning(sample_data, sample_features):
    """Test model hyperparameter tuning."""
    models = [
        DecisionTreeModel(),
        RandomForestModel(),
        XGBoostModel(),
        LSTMModel(),
        TransformerModel()
    ]
    
    for model in models:
        # Tune hyperparameters
        best_params = model.tune_hyperparameters(sample_data, sample_features)
        
        assert isinstance(best_params, dict)
        
        # Train with best parameters
        model.train(sample_data, sample_features, **best_params)
        assert model.model is not None 