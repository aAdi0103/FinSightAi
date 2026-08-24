import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler


# Default path to the raw PaySim CSV.
DATA_PATH: str = os.getenv(
    "PAYSIM_DATA_PATH",
    r"Data\PS_20174392719_1491204439457_log.csv",
)

# Transaction types known in PaySim.
KNOWN_TYPES: set = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}

# Threshold used by the PaySim simulator to flag large transfers.
LARGE_TRANSFER_THRESHOLD: float = 200_000.0

# Numerical columns to be scaled.
NUMERICAL_COLS_TO_SCALE: list = [
    "amount",
    "log_amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_diff_sender",
    "balance_diff_receiver",
    "sender_error",
    "receiver_error",
    "amount_vs_sender_balance_ratio",
]


COLUMNS_TO_DROP: list = ["nameOrig", "nameDest", "isFraud", "isFlaggedFraud"]

# LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# 1. DATA LOADING

def load_data(path: str = DATA_PATH) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Set the PAYSIM_DATA_PATH environment variable or pass the "
            "correct path to load_data()."
        )

    logger.info("Loading dataset from: %s", path)

    dtype_map = {
        "step": "int32",
        "type": "category",
        "amount": "float32",
        "nameOrig": "object",
        "oldbalanceOrg": "float32",
        "newbalanceOrig": "float32",
        "nameDest": "object",
        "oldbalanceDest": "float32",
        "newbalanceDest": "float32",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }

    df = pd.read_csv(path, dtype=dtype_map)
    logger.info("Dataset loaded - shape: %s", df.shape)
    return df


# 2. DATA AUDITING

def audit_data(df: pd.DataFrame) -> dict:
    
    logger.info("Auditing dataset ...")

    n_rows, n_cols = df.shape
    null_counts = df.isnull().sum().to_dict()
    n_duplicates = int(df.duplicated().sum())

    fraud_counts = df["isFraud"].value_counts().to_dict()
    fraud_pct = round(fraud_counts.get(1, 0) / n_rows * 100, 4)

    type_counts = df["type"].value_counts().to_dict()
    unknown_types = set(df["type"].unique()) - KNOWN_TYPES
    summary_stats = df.describe().round(2).to_dict()

    audit = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": null_counts,
        "n_duplicates": n_duplicates,
        "fraud_counts": fraud_counts,
        "fraud_percentage": fraud_pct,
        "type_distribution": type_counts,
        "unknown_types": list(unknown_types),
        "summary_statistics": summary_stats,
    }

    logger.info("  Rows         : %d", n_rows)
    logger.info("  Columns      : %d", n_cols)
    logger.info("  Missing vals : %s", null_counts)
    logger.info("  Duplicates   : %d", n_duplicates)
    logger.info("  Fraud %%      : %.4f%%", fraud_pct)
    logger.info("  Txn types    : %s", type_counts)

    return audit


# 3. FEATURE ENGINEERING

import gc

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering features ...")
    # Accounting Accuracy Features
    df["balance_diff_sender"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_receiver"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["sender_error"] = (
        (df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"])
        .abs()
        .astype("float32")
    )
    df["receiver_error"] = (
        (df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"])
        .abs()
        .astype("float32")
    )

    # Balance State Features
    df["sender_balance_drained"] = (
        (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)
    ).astype("int8")
    df["receiver_balance_unchanged"] = (
        df["oldbalanceDest"] == df["newbalanceDest"]
    ).astype("int8")

    # Proportional Risk Features
    df["amount_vs_sender_balance_ratio"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1.0)
    ).astype("float32")
    df["high_amount_flag"] = (
        df["amount"] > LARGE_TRANSFER_THRESHOLD
    ).astype("int8")

    # Destination Type Feature
    df["is_merchant_dest"] = df["nameDest"].astype(str).str.startswith("M").astype("int8")

    # Log-Transformed Amount
    df["log_amount"] = np.log1p(df["amount"]).astype("float32")

    # Temporal Features
    df["hour_of_day"] = (df["step"] % 24).astype("int8")
    df["day_of_sim"] = (df["step"] // 24).astype("int8")

    logger.info("Feature engineering complete - %d total columns.", df.shape[1])
    return df


# 4. CATEGORICAL ENCODING

def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    logger.info("Encoding categorical column: 'type' ...")
    le = LabelEncoder()
    # Fit label encoder on unique types
    unique_types = sorted(df["type"].astype(str).unique())
    le.fit(unique_types)
    df["type"] = le.transform(df["type"].astype(str)).astype("int8")

    type_mapping = {int(code): name for code, name in enumerate(le.classes_)}
    logger.info("Type encoding map: %s", type_mapping)

    return df, type_mapping


# 5. FEATURE SCALING

def scale_features(
    df: pd.DataFrame,
    cols: list = None,
    scaler: RobustScaler = None,
) -> Tuple[pd.DataFrame, RobustScaler]:
    if cols is None:
        cols = NUMERICAL_COLS_TO_SCALE

    logger.info("Scaling %d numerical features with RobustScaler ...", len(cols))
    if scaler is None:
        scaler = RobustScaler()
        df[cols] = scaler.fit_transform(df[cols]).astype("float32")
        logger.info("RobustScaler fitted and applied (training mode).")
    else:
        df[cols] = scaler.transform(df[cols]).astype("float32")
        logger.info("Pre-fitted RobustScaler applied (inference mode).")

    return df, scaler


# 6. FEATURE / LABEL SEPARATION

def separate_features_labels(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Separating features (X) from labels (y) ...")
    y = df["isFraud"].copy()

    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    X = df

    logger.info("X shape: %s | y shape: %s", X.shape, y.shape)
    logger.info("Feature columns: %s", list(X.columns))

    return X, y


# 7. FULL PIPELINE (CONVENIENCE WRAPPER)

def run_full_pipeline(
    path: str = DATA_PATH,
    scaler: RobustScaler = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict, RobustScaler]:
    logger.info("=" * 60)
    logger.info("PaySim Preprocessing Pipeline - START")
    logger.info("=" * 60)

    df = load_data(path)
    _audit_report = audit_data(df)
    df = engineer_features(df)
    df, type_mapping = encode_categoricals(df)
    df, fitted_scaler = scale_features(df, scaler=scaler)
    X, y = separate_features_labels(df)
    gc.collect()

    logger.info("=" * 60)
    logger.info("Pipeline complete - X: %s | y: %s", X.shape, y.shape)
    logger.info("=" * 60)

    return X, y, None, type_mapping, fitted_scaler


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    X, y, df_eng, type_map, scaler = run_full_pipeline(csv_path)

    print("\n--- Pipeline Summary ---")
    print(f"Feature matrix shape : {X.shape}")
    print(f"Label vector shape   : {y.shape}")
    print(f"Type mapping         : {type_map}")
    print(f"Feature columns      :\n  {list(X.columns)}")
    print(f"\nFraud label distribution:\n{y.value_counts()}")
    print(f"\nFirst 3 rows of X:\n{X.head(3)}")
