import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import dump

# Define column groups
CATEGORICAL = ["type"]
NUMERICAL   = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
LABEL       = "isFraud"
TIME        = "step"

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add new useful features based on balances."""
    df["org_balance_delta"] = df["newbalanceOrig"] - (df["oldbalanceOrg"] - df["amount"])
    df["dest_balance_delta"] = df["newbalanceDest"] - (df["oldbalanceDest"] + df["amount"])
    df["balance_error"] = (df["newbalanceOrig"] + df["amount"]) - df["oldbalanceOrg"]
    return df

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only required columns + new engineered ones."""
    keep_cols = CATEGORICAL + NUMERICAL + [LABEL, TIME, "org_balance_delta", "dest_balance_delta", "balance_error"]
    return df[keep_cols]

def load_paysim(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df = add_derived_features(df)
    df = select_features(df)

    # time-aware split (avoid leakage)
    cutoff = int(df[TIME].quantile(0.8))
    train = df[df[TIME] <= cutoff]
    test  = df[df[TIME] > cutoff]
    return train, test

def build_preprocessor():
    """Create preprocessing pipeline with OHE for categorical + passthrough for numerical."""
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pre = ColumnTransformer(
        transformers=[
            ("cat", ohe, CATEGORICAL),
            ("num", "passthrough", NUMERICAL + ["org_balance_delta","dest_balance_delta","balance_error"])
        ]
    )
    return pre

if __name__ == "__main__":
    train, test = load_paysim("data/PS_20174392719_1491204439457_log.csv")

    pre = build_preprocessor()
    X_train = train.drop(columns=[LABEL, TIME])  # fit only on features
    pre.fit(X_train)

    # Save processor + datasets
    dump(pre, "models/preprocessor.joblib")
    train.to_parquet("data/train.parquet")
    test.to_parquet("data/test.parquet")

    print("✅ Data preparation complete: train/test splits and preprocessor saved.")
