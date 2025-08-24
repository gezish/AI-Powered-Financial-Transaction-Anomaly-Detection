import pandas as pd
import numpy as np

CATEGORICAL = ["type"]
NUMERICAL = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
LABEL = "isFraud"
TIME = "step"   # optional if time-based features are added

def add_derived_features(df):
    # Example: difference in balances
    df["deltaOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["deltaDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    return df

def select_features(df):
    return df[CATEGORICAL + NUMERICAL + ["deltaOrig", "deltaDest"]]

