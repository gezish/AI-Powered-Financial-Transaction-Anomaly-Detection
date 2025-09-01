import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import sklearn.compose._column_transformer as ct
# Patch missing class
ct._RemainderColsList = list

# Load test data
test_df = pd.read_parquet("data/test.parquet")

# Separate features and label
X_test = test_df.drop(columns=["isFraud"])
y_test = test_df["isFraud"]

# Load preprocessor and model
preprocessor = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/xgb_model.joblib")

# Transform test data
X_test_transformed = preprocessor.transform(X_test)

# Predictions
y_pred = model.predict(X_test_transformed)
y_proba = model.predict_proba(X_test_transformed)[:, 1]

# Evaluation
print("\n✅ Model Evaluation on Test Data")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
