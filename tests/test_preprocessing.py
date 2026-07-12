import joblib
import pandas as pd
import pytest

@pytest.fixture
def preprocessor():
    return joblib.load("models/preprocessor.pkl")

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "age": [55.0], "trestbps": [130.0], "chol": [250.0],
        "thalach": [150.0], "oldpeak": [1.0],
        "sex": [1.0], "cp": [2.0], "fbs": [0.0], "restecg": [1.0],
        "exang": [0.0], "slope": [2.0], "ca": [0.0], "thal": [3.0],
    })

def test_preprocessor_output_shape(preprocessor, sample_data):
    result = preprocessor.transform(sample_data)
    assert result.shape[0] == 1

def test_preprocessor_no_nans(preprocessor, sample_data):
    result = preprocessor.transform(sample_data)
    assert not pd.isna(result).any()