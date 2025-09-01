# src/inference/service.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from src.common.features import add_derived_features, select_features, LABEL, TIME

# Load preprocessor + model
preprocessor = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/model.joblib")

app = FastAPI(title="Fraud Detection API")

# Input schema
class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float

@app.post("/predict")
def predict(tx: Transaction):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([tx.dict()])

        # Add derived features
        df = add_derived_features(df)
        df = select_features(df)

        # Drop label/time if present
        if LABEL in df: 
            df = df.drop(columns=[LABEL])
        if TIME in df:
            df = df.drop(columns=[TIME])

        # Transform + predict
        X = preprocessor.transform(df)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0,1]

        return {
            "prediction": int(pred),
            "fraud_probability": float(proba)
        }
    except Exception as e:
        return {"error": str(e)}
