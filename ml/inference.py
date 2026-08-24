
import json
import logging
import os
from typing import Any, Dict, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MODELS_DIR = os.getenv("MODELS_DIR", "models")
MODEL_FILE = os.path.join(MODELS_DIR, "isolation_forest.joblib")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.joblib")
METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")

# Threshold for large transfer flagging
LARGE_TRANSFER_THRESHOLD: float = 200_000.0

# Columns scaled during training
NUMERICAL_COLS_TO_SCALE = [
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ARTIFACT LOADER (LAZY SINGLETON PATTERN)
# ---------------------------------------------------------------------------

class ArtifactStore:
    """Singleton container to load and cache model artifacts in memory."""
    _instance = None

    def __init__(self):
        self.model: IsolationForest = None
        self.scaler: RobustScaler = None
        self.metadata: Dict[str, Any] = None
        self.feature_names: list = None
        self.type_to_code: Dict[str, int] = None
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_FILE}'. "
                "Please train the model first by running `python -m ml.train`."
            )
        if not os.path.exists(SCALER_FILE):
            raise FileNotFoundError(
                f"Scaler file not found at '{SCALER_FILE}'. "
                "Please train the model first by running `python -m ml.train`."
            )
        if not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(
                f"Metadata file not found at '{METADATA_FILE}'. "
                "Please train the model first by running `python -m ml.train`."
            )

        logger.info("Loading serialized artifacts from '%s'...", MODELS_DIR)
        self.model = joblib.load(MODEL_FILE)
        self.scaler = joblib.load(SCALER_FILE)

        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata.get("feature_names", [])
        
        # Invert type mapping (saved as code -> name, convert to name -> code)
        raw_type_map = self.metadata.get("type_mapping", {})
        self.type_to_code = {name.upper(): int(code) for code, name in raw_type_map.items()}
        logger.info("Artifacts loaded successfully. Features: %d", len(self.feature_names))

    @classmethod
    def get_instance(cls) -> "ArtifactStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# ---------------------------------------------------------------------------
# INFERENCE PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess_single_transaction(
    transaction: Union[Dict[str, Any], pd.DataFrame],
    artifacts: ArtifactStore,
) -> pd.DataFrame:
    """
    Apply feature engineering, categorical encoding, and scaling to a single
    transaction record or a mini-batch DataFrame.

    Args:
        transaction: Dictionary or DataFrame containing raw transaction fields:
          - step (int)
          - type (str): 'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'
          - amount (float)
          - nameOrig (str, optional)
          - oldbalanceOrg (float)
          - newbalanceOrig (float)
          - nameDest (str, optional)
          - oldbalanceDest (float)
          - newbalanceDest (float)
        artifacts: Pre-loaded ArtifactStore instance.

    Returns:
        pd.DataFrame containing engineered and scaled features matching model input.
    """
    if isinstance(transaction, dict):
        df = pd.DataFrame([transaction])
    else:
        df = transaction.copy()

    # Fill defaults for optional ID columns if missing
    if "nameOrig" not in df.columns:
        df["nameOrig"] = "C_UNKNOWN"
    if "nameDest" not in df.columns:
        df["nameDest"] = "C_UNKNOWN"

    # Ensure correct data types
    df["step"] = df["step"].astype("int32")
    df["amount"] = df["amount"].astype("float32")
    df["oldbalanceOrg"] = df["oldbalanceOrg"].astype("float32")
    df["newbalanceOrig"] = df["newbalanceOrig"].astype("float32")
    df["oldbalanceDest"] = df["oldbalanceDest"].astype("float32")
    df["newbalanceDest"] = df["newbalanceDest"].astype("float32")

    # 1. Feature Engineering
    df["balance_diff_sender"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_receiver"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["sender_error"] = (
        (df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]).abs().astype("float32")
    )
    df["receiver_error"] = (
        (df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]).abs().astype("float32")
    )
    df["sender_balance_drained"] = (
        (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)
    ).astype("int8")
    df["receiver_balance_unchanged"] = (
        df["oldbalanceDest"] == df["newbalanceDest"]
    ).astype("int8")
    df["amount_vs_sender_balance_ratio"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1.0)
    ).astype("float32")
    df["high_amount_flag"] = (
        df["amount"] > LARGE_TRANSFER_THRESHOLD
    ).astype("int8")
    df["is_merchant_dest"] = df["nameDest"].astype(str).str.startswith("M").astype("int8")
    df["log_amount"] = np.log1p(df["amount"]).astype("float32")
    df["hour_of_day"] = (df["step"] % 24).astype("int8")
    df["day_of_sim"] = (df["step"] // 24).astype("int8")

    # 2. Categorical Encoding (type)
    type_str = df["type"].astype(str).str.upper()
    df["type"] = type_str.map(lambda t: artifacts.type_to_code.get(t, 0)).astype("int32")

    # 3. Scale continuous numerical columns
    cols_to_scale = [c for c in NUMERICAL_COLS_TO_SCALE if c in df.columns]
    df[cols_to_scale] = artifacts.scaler.transform(df[cols_to_scale])

    # 4. Select and order features strictly matching training
    X = df[artifacts.feature_names].copy()
    return X


# ---------------------------------------------------------------------------
# PUBLIC INFERENCE INTERFACE
# ---------------------------------------------------------------------------

def predict_transaction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score an incoming transaction using the trained Isolation Forest model.

    Args:
        transaction: Dictionary containing transaction attributes:
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "nameOrig": "C123456789",
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C987654321",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0
            }

    Returns:
        Dictionary strictly conforming to Phase 2 requirements:
            {
                "prediction": "Normal" | "Suspicious",
                "anomaly_score": float
            }
    """
    artifacts = ArtifactStore.get_instance()
    X = preprocess_single_transaction(transaction, artifacts)

    # raw_pred: 1 = Inlier (Normal), -1 = Outlier (Suspicious)
    raw_pred = int(artifacts.model.predict(X)[0])
    prediction_label = "Suspicious" if raw_pred == -1 else "Normal"

    # Invert decision function so higher score = higher anomaly
    raw_score = float(artifacts.model.decision_function(X)[0])
    anomaly_score = round(-raw_score, 4)

    return {
        "prediction": prediction_label,
        "anomaly_score": anomaly_score,
    }


# ---------------------------------------------------------------------------
# CLI / TESTING ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test cases: 1 Normal transaction and 1 Obvious Fraud transaction
    sample_normal = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 9839.64,
        "nameOrig": "C1231006815",
        "oldbalanceOrg": 170136.0,
        "newbalanceOrig": 160296.36,
        "nameDest": "M1979787155",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }

    sample_fraud = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "nameOrig": "C1305486145",
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C553264065",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }

    print("\n--- Testing Single Transaction Inference ---")
    try:
        res_normal = predict_transaction(sample_normal)
        print("Sample Normal Transaction Result:")
        print(json.dumps(res_normal, indent=2))

        res_fraud = predict_transaction(sample_fraud)
        print("\nSample Fraud Transaction Result:")
        print(json.dumps(res_fraud, indent=2))
    except FileNotFoundError as e:
        print(f"[Warning] {e}")
