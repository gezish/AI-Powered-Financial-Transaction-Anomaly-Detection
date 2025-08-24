import shap, numpy as np
from joblib import load

# Note: use a small background dataset for Kernel/TreeExplainer speed
def load_artifacts():
    pre = load("models/preprocessor.joblib")
    model = load("models/xgb_model.joblib")
    return pre, model

def shap_topk(pre, model, raw_df_row, k=5):
    X = pre.transform(raw_df_row)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)[0]   # shap values for first row
    feature_names = (pre.get_feature_names_out()).tolist()
    pairs = sorted(zip(feature_names, sv), key=lambda t: abs(t[1]), reverse=True)[:k]
    return [{"feature": f, "contribution": float(v)} for f,v in pairs]
