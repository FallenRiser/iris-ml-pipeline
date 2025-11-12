# tests/test_data.py
import pytest
import pandas as pd
import os

DATA_PATH = "iris_data_adapted_for_feast.csv"

@pytest.fixture(scope="module")
def data():
    """Load data once for all tests"""
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Data file {DATA_PATH} not found. Run 'dvc pull' first.")
    return pd.read_csv(DATA_PATH)

class TestDataExistence:
    """Test data file existence and loading"""
    
    def test_data_file_exists(self):
        """Verify data file exists"""
        assert os.path.exists(DATA_PATH), \
            f"Data file {DATA_PATH} not found"
    
    def test_data_loads(self, data):
        """Verify data loads correctly"""
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0, "Dataset is empty"
        print(f"\n📊 Dataset contains {len(data)} rows")

class TestDataSchema:
    """Test data schema and structure"""
    
    def test_required_columns(self, data):
        """Verify all required columns exist"""
        required = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in required:
            assert col in data.columns, f"Missing column: {col}"
    
    def test_data_types(self, data):
        """Verify column data types are correct"""
        numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(data[col]), \
                f"Column {col} is not numeric"

class TestDataQuality:
    """Test data quality"""
    
    def test_no_missing_values(self, data):
        """Verify no missing values in feature columns"""
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in feature_cols:
            missing = data[col].isna().sum()
            assert missing == 0, \
                f"Column {col} has {missing} missing values"
    
    def test_positive_measurements(self, data):
        """Verify all measurements are non-negative"""
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in feature_cols:
            min_val = data[col].min()
            assert min_val >= -0.5, \
                f"Column {col} has negative value: {min_val}"
    
    def test_reasonable_ranges(self, data):
        """Verify measurements are within reasonable ranges"""
        ranges = {
            'sepal_length': (4.0, 8.0),
            'sepal_width': (2.0, 5.0),
            'petal_length': (0.5, 7.0),
            'petal_width': (-0.5, 3.0)
        }
        
        for col, (min_val, max_val) in ranges.items():
            assert data[col].between(min_val, max_val).all(), \
                f"Column {col} has values outside [{min_val}, {max_val}]"

class TestDataStatistics:
    """Test basic data statistics"""
    
    def test_sufficient_samples(self, data):
        """Verify sufficient number of samples"""
        assert len(data) >= 30, \
            f"Dataset has only {len(data)} samples (minimum 30 recommended)"
    
    def test_feature_variance(self, data):
        """Verify features have non-zero variance"""
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in feature_cols:
            variance = data[col].var()
            assert variance > 0, \
                f"Column {col} has zero variance"
