import sys
import os
import pandas as pd
from joblib import load
from src.explain.shap_utils import shap_topk
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

pre = load("models/preprocessor.joblib")
model = load("models/xgb_model.joblib")

def score_record(record: dict, explain: bool = True):
    raw = pd.DataFrame([record])
    proba = float(model.predict_proba(pre.transform(raw))[0,1])
    label = int(proba >= 0.5)
    expl = shap_topk(pre, model, raw, k=5) if explain else []
    return {"score": proba, "label": label, "explanations": expl}
