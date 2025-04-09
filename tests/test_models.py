import pytest
import numpy as np
import pandas as pd
from app.models.model_factory import ModelFactory
from app.models.base_model import BaseTradingModel

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    return X, y

def test_decision_tree_model(sample_data):
    """Test decision tree model."""
    X, y = sample_data
    model = ModelFactory.create_model("decision_tree", {})
    
    # Test training
    model.train(X, y)
    assert model.is_trained
    
    # Test prediction
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert all(-1 <= p <= 1 for p in predictions)
    
    # Test feature importance
    importance = model.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) == X.shape[1]

def test_mlp_model(sample_data):
    """Test MLP model."""
    X, y = sample_data
    model = ModelFactory.create_model("mlp", {
        "input_size": X.shape[1],
        "hidden_sizes": [32, 16],
        "output_size": 1
    })
    
    # Test training
    model.train(X, y)
    assert model.is_trained
    
    # Test prediction
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert all(-1 <= p <= 1 for p in predictions)
    
    # Test feature importance
    importance = model.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) == X.shape[1]

def test_transformer_model(sample_data):
    """Test transformer model."""
    X, y = sample_data
    model = ModelFactory.create_model("transformer", {
        "input_size": X.shape[1],
        "sequence_length": 10,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 2
    })
    
    # Test training
    model.train(X, y)
    assert model.is_trained
    
    # Test prediction
    predictions = model.predict(X)
    assert len(predictions) == len(X) - model.sequence_length + 1
    assert all(-1 <= p <= 1 for p in predictions)
    
    # Test feature importance
    importance = model.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) == X.shape[1]

def test_model_factory():
    """Test model factory."""
    # Test available models
    available_models = ModelFactory.get_available_models()
    assert isinstance(available_models, dict)
    assert "decision_tree" in available_models
    assert "mlp" in available_models
    assert "transformer" in available_models
    
    # Test default configs
    for model_type in available_models.keys():
        config = ModelFactory.get_default_config(model_type)
        assert isinstance(config, dict)
        assert len(config) > 0
    
    # Test model creation
    for model_type in available_models.keys():
        model = ModelFactory.create_model(model_type, {})
        assert isinstance(model, BaseTradingModel)
        assert not model.is_trained 