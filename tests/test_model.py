import joblib
import pandas as pd
import pytest

@pytest.fixture
def model():
    return joblib.load("models/final_model.pkl")

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "age": [55.0], "trestbps": [130.0], "chol": [250.0],
        "thalach": [150.0], "oldpeak": [1.0],
        "sex": [1.0], "cp": [2.0], "fbs": [0.0], "restecg": [1.0],
        "exang": [0.0], "slope": [2.0], "ca": [0.0], "thal": [3.0],
    })

def test_model_predicts_binary(model, sample_data):
    pred = model.predict(sample_data)
    assert pred[0] in [0, 1]

def test_model_predicts_valid_probability(model, sample_data):
    proba = model.predict_proba(sample_data)[0][1]
    assert 0.0 <= proba <= 1.0