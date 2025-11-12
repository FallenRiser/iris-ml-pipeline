# tests/test_model.py
import pytest
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os

MODEL_PATH = "model_iris_bq.joblib"

@pytest.fixture(scope="module")
def model():
    """Load model once for all tests"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file {MODEL_PATH} not found. Run 'dvc pull' first.")
    return joblib.load(MODEL_PATH)

@pytest.fixture
def sample_data():
    """Generate sample test data"""
    data = {
        'sepal_length': [5.1, 4.9, 6.7, 5.8, 6.3, 5.0],
        'sepal_width': [3.5, 3.0, 3.1, 2.7, 2.8, 3.4],
        'petal_length': [1.4, 1.4, 4.7, 5.1, 5.1, 1.5],
        'petal_width': [0.2, 0.2, 1.5, 1.9, 1.5, 0.2]
    }
    y_true = [0, 0, 1, 2, 2, 0]
    return pd.DataFrame(data), y_true

class TestModelExistence:
    """Test model file and loading"""
    
    def test_model_file_exists(self):
        """Verify model file exists"""
        assert os.path.exists(MODEL_PATH), f"Model file {MODEL_PATH} not found"
    
    def test_model_loads(self, model):
        """Verify model loads correctly"""
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

class TestModelPredictions:
    """Test model prediction functionality"""
    
    def test_prediction_shape(self, model, sample_data):
        """Verify predictions have correct shape"""
        X, _ = sample_data
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert predictions.dtype in [np.int32, np.int64]
    
    def test_prediction_range(self, model, sample_data):
        """Verify predictions are valid class labels"""
        X, _ = sample_data
        predictions = model.predict(X)
        assert all(p in [0, 1, 2] for p in predictions), \
            "Predictions must be 0, 1, or 2"
    
    def test_no_nan_predictions(self, model, sample_data):
        """Verify no NaN predictions"""
        X, _ = sample_data
        predictions = model.predict(X)
        assert not np.isnan(predictions).any(), \
            "Model produced NaN predictions"
    
    def test_probability_predictions(self, model, sample_data):
        """Verify probability predictions are valid"""
        X, _ = sample_data
        probas = model.predict_proba(X)
        
        # Check shape
        assert probas.shape == (len(X), 3), \
            f"Expected shape ({len(X)}, 3), got {probas.shape}"
        
        # Check probabilities sum to 1
        assert np.allclose(probas.sum(axis=1), 1.0), \
            "Probabilities don't sum to 1"
        
        # Check all probabilities are between 0 and 1
        assert ((probas >= 0) & (probas <= 1)).all(), \
            "Probabilities out of [0,1] range"

class TestModelPerformance:
    """Test model performance metrics"""
    
    def test_minimum_accuracy(self, model, sample_data):
        """Verify model meets minimum accuracy threshold"""
        X, y_true = sample_data
        y_pred = model.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n📊 Model accuracy: {accuracy:.2%}")
        assert accuracy >= 0.7, \
            f"Model accuracy {accuracy:.2%} below minimum 70%"
    
    def test_consistent_predictions(self, model, sample_data):
        """Verify predictions are consistent across calls"""
        X, _ = sample_data
        pred1 = model.predict(X)
        pred2 = model.predict(X)
        
        assert np.array_equal(pred1, pred2), \
            "Model predictions are not deterministic"
