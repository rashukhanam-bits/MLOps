import logging
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-disease-api")

app = FastAPI(title="Heart Disease Prediction API")
model = joblib.load("../models/final_model.pkl")

Instrumentator().instrument(app).expose(app)

class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.dict()])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    logger.info(f"Prediction={prediction} Probability={probability:.4f}")
    return {"prediction": prediction, "probability": probability}

@app.get("/health")
def health():
    return {"status": "ok"}