# src/serving/inference.py
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "xgb_model.joblib"
PREPROCESSOR_PATH = Path(__file__).resolve().parents[2] / "models" / "preprocessor.joblib"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# These are the columns your preprocessor was fit on
EXPECTED_FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "org_balance_delta",
    "dest_balance_delta",
    "balance_error",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df["org_balance_delta"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["dest_balance_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["balance_error"] = (df["oldbalanceOrg"] - df["newbalanceOrig"]) - df["amount"]
    return df

def _select_expected(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only the features that the preprocessor knows
    missing = set(EXPECTED_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"columns are missing: {missing}")
    return df[EXPECTED_FEATURES]

def score_record(record: dict, explain: bool = False) -> dict:
    """
    Takes a raw transaction (same fields as PredictionRequest), engineers features,
    transforms, and returns prediction (and optional explanation placeholder).
    """
    df = pd.DataFrame([record]) # Engineer features to match training
    df = add_derived_features(df)   # 👈 Add missing features
    # Select exact columns the preprocessor expects
    X = _select_expected(df)

    # Transform + predict
    Xt = preprocessor.transform(X)
    pred = model.predict(Xt)[0]

    result = {"fraud_prediction": int(pred)}
    if explain:
        # Placeholder – plug SHAP here later
        result["explanation"] = "Explanation not yet implemented"
    return result