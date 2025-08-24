import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
from xgboost import XGBClassifier
from src.common.features import LABEL, TIME

train = pd.read_parquet("data/train.parquet")
test  = pd.read_parquet("data/test.parquet")
pre = load("models/preprocessor.joblib")

X_train = pre.transform(train.drop(columns=[LABEL, TIME]))
y_train = train[LABEL].values
X_test  = pre.transform(test.drop(columns=[LABEL, TIME]))
y_test  = test[LABEL].values

# handle imbalance: set scale_pos_weight = (neg/pos)
pos = y_train.sum()
neg = len(y_train) - pos
spw = max(1.0, neg / max(1.0, pos))

clf = XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    tree_method="hist",
    scale_pos_weight=spw,
    n_jobs=-1
)
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)[:,1]
preds = (probs >= 0.5).astype(int)

print("ROC-AUC:", roc_auc_score(y_test, probs))
print("PR-AUC:", average_precision_score(y_test, probs))
print("F1@0.5:", f1_score(y_test, preds))

dump(clf, "models/xgb_model.joblib")
