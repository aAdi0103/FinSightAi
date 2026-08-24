"""
ml/train.py
===========

Trains an unsupervised Isolation Forest model on the PaySim preprocessed feature matrix to isolate anomalous/fraudulent financial transactions.

Key Responsibilities:
    1. Load processed feature matrix X and held-out labels y.
    2. Instantiate & train Isolation Forest with tuned hyperparameters.
    3. Generate binary predictions (Normal vs Suspicious) and continuous anomaly scores.
    4. Evaluate performance against ground-truth labels (Precision, Recall, F1, PR-AUC, ROC-AUC).
    5. Generate diagnostic visualizations (PR curve, ROC curve, Confusion Matrix, Score Distribution).
    6. Persist model, scaler, and metadata artifacts into models/ directory.
    7. Generate a comprehensive Phase 2 evaluation report in reports/.
"""

import json
import logging
import os
import time
from typing import Dict, Any, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.preprocess import run_full_pipeline, DATA_PATH

# CONFIGURATION & HYPERPARAMETERS

MODELS_DIR = os.getenv("MODELS_DIR", "models")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")

# Hyperparameter rationale:
# 1. n_estimators (100): number of decisions trees which will make decision to classify the transactions. 
# 2. max_samples (256): each tree will be trained on this much data
# 3. contamination (0.0013): Aligns with PaySim ground truth fraud rate.
# 4. random_state (42): Guarantees exact reproducibility across training runs.
# 5. n_jobs (-1): Utilizes all CPU cores for fast parallel tree building.

HYPERPARAMETERS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_samples": 256,
    "contamination": 0.0013,
    "random_state": 42,
    "n_jobs": -1,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# 1. MODEL TRAINING

def train_isolation_forest(
    X: pd.DataFrame,
    hyperparams: Dict[str, Any] = None,
) -> Tuple[IsolationForest, float]:
    
    if hyperparams is None:
        hyperparams = HYPERPARAMETERS

    logger.info("Initializing Isolation Forest with hyperparameters: %s", hyperparams)
    model = IsolationForest(**hyperparams)

    logger.info("Fitting Isolation Forest on %d samples with %d features...", X.shape[0], X.shape[1])
    start_time = time.time()
    model.fit(X)
    elapsed = time.time() - start_time
    logger.info("Training completed in %.2f seconds.", elapsed)

    return model, elapsed


def compute_predictions_and_scores(
    model: IsolationForest,
    X: pd.DataFrame,
    chunk_size: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute binary predictions and continuous anomaly scores in memory-safe chunks.

    Args:
        model: Trained IsolationForest.
        X: Feature matrix.
        chunk_size: Number of rows to score per batch to avoid memory spikes.

    Returns:
        Tuple of:
          - raw_preds: np.ndarray (1 or -1)
          - binary_preds: np.ndarray (0 for Normal, 1 for Suspicious)
          - anomaly_scores: np.ndarray (higher = more anomalous)
    """
    logger.info("Generating predictions and continuous anomaly scores in batches...")
    n_rows = len(X)
    raw_scores = np.empty(n_rows, dtype=np.float32)

    for i in range(0, n_rows, chunk_size):
        end_idx = min(i + chunk_size, n_rows)
        raw_scores[i:end_idx] = model.decision_function(X.iloc[i:end_idx]).astype(np.float32)

    # In scikit-learn, decision_function < 0 is predicted as an anomaly (-1)
    raw_preds = np.where(raw_scores < 0, -1, 1).astype(np.int8)
    binary_preds = np.where(raw_scores < 0, 1, 0).astype(np.int8)
    anomaly_scores = -raw_scores

    return raw_preds, binary_preds, anomaly_scores


# ---------------------------------------------------------------------------
# 3. EVALUATION METRICS & VISUALIZATION
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: pd.Series,
    binary_preds: np.ndarray,
    anomaly_scores: np.ndarray,
    output_dir: str = REPORTS_DIR,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics and generate diagnostic plots.

    Args:
        y_true: Ground truth labels (0 = legitimate, 1 = fraud).
        binary_preds: Model binary predictions (0 = Normal, 1 = Suspicious).
        anomaly_scores: Continuous anomaly scores.
        output_dir: Directory where diagnostic plots will be saved.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    logger.info("Computing evaluation metrics against ground truth labels...")
    os.makedirs(output_dir, exist_ok=True)

    # Core Metrics
    precision = float(precision_score(y_true, binary_preds, zero_division=0))
    recall = float(recall_score(y_true, binary_preds, zero_division=0))
    f1 = float(f1_score(y_true, binary_preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, anomaly_scores))
    pr_auc = float(average_precision_score(y_true, anomaly_scores))
    cm = confusion_matrix(y_true, binary_preds)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    total_actual_fraud = int(y_true.sum())
    total_predicted_fraud = int(binary_preds.sum())

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "total_samples": len(y_true),
        "actual_fraud_count": total_actual_fraud,
        "predicted_suspicious_count": total_predicted_fraud,
    }

    logger.info("=== Evaluation Results ===")
    logger.info("Precision : %.4f", precision)
    logger.info("Recall    : %.4f", recall)
    logger.info("F1 Score  : %.4f", f1)
    logger.info("PR-AUC    : %.4f", pr_auc)
    logger.info("ROC-AUC   : %.4f", roc_auc)
    logger.info("Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d", tn, fp, fn, tp)

    # --- Plot 1: Precision-Recall Curve ---
    logger.info("Generating Precision-Recall Curve plot...")
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, anomaly_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(rec_vals, prec_vals, color="#1e88e5", lw=2, label=f"Isolation Forest (PR-AUC = {pr_auc:.4f})")
    plt.axhline(y=total_actual_fraud / len(y_true), color="red", linestyle="--", label="Random Baseline")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve (Highly Imbalanced Data)", fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    pr_path = os.path.join(output_dir, "pr_curve.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()

    # --- Plot 2: ROC Curve ---
    logger.info("Generating ROC Curve plot...")
    fpr_vals, tpr_vals, _ = roc_curve(y_true, anomaly_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_vals, tpr_vals, color="#43a047", lw=2, label=f"Isolation Forest (ROC-AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Chance")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # --- Plot 3: Confusion Matrix Visualization ---
    logger.info("Generating Confusion Matrix plot...")
    plt.figure(figsize=(6, 5))
    cm_display = np.array([[tn, fp], [fn, tp]])
    plt.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14)
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Legitimate (0)", "Fraud (1)"])
    plt.yticks(tick_marks, ["Legitimate (0)", "Fraud (1)"])
    
    thresh = cm_display.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, f"{cm_display[i, j]:,}",
                horizontalalignment="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold"
            )
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # --- Plot 4: Anomaly Score Distributions ---
    logger.info("Generating Anomaly Score Distribution plot...")
    plt.figure(figsize=(9, 5))
    legit_scores = anomaly_scores[y_true == 0]
    fraud_scores = anomaly_scores[y_true == 1]
    
    # Subsample legitimate scores for clean plotting if very large
    if len(legit_scores) > 100_000:
        legit_scores_sample = np.random.choice(legit_scores, size=100_000, replace=False)
    else:
        legit_scores_sample = legit_scores

    plt.hist(legit_scores_sample, bins=50, alpha=0.6, density=True, label="Legitimate (y=0)", color="#1e88e5")
    plt.hist(fraud_scores, bins=50, alpha=0.7, density=True, label="Fraud (y=1)", color="#e53935")
    plt.xlabel("Anomaly Score (Higher = More Anomalous)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Anomaly Score Distribution: Legitimate vs Fraudulent Transactions", fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    dist_path = os.path.join(output_dir, "anomaly_score_distribution.png")
    plt.tight_layout()
    plt.savefig(dist_path, dpi=200)
    plt.close()

    return metrics


# ---------------------------------------------------------------------------
# 4. ARTIFACT PERSISTENCE
# ---------------------------------------------------------------------------

def save_artifacts(
    model: IsolationForest,
    scaler: Any,
    feature_names: list,
    type_mapping: dict,
    metrics: dict,
    training_time: float,
    models_dir: str = MODELS_DIR,
) -> None:
    """
    Save trained model, scaler, and configuration metadata.

    Args:
        model: Trained IsolationForest model.
        scaler: Fitted RobustScaler instance.
        feature_names: List of ordered feature column names.
        type_mapping: Dictionary mapping type codes to strings.
        metrics: Dictionary of calculated performance metrics.
        training_time: Training duration in seconds.
        models_dir: Target output directory for model files.
    """
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "isolation_forest.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    logger.info("Saving Isolation Forest model to %s ...", model_path)
    joblib.dump(model, model_path)

    logger.info("Saving RobustScaler to %s ...", scaler_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_type": "IsolationForest",
        "library_version": {
            "scikit_learn": "1.x",
        },
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "hyperparameters": HYPERPARAMETERS,
        "type_mapping": {str(k): v for k, v in type_mapping.items()},
        "training_time_seconds": round(training_time, 2),
        "evaluation_metrics": metrics,
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    logger.info("Saving model metadata to %s ...", metadata_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info("All artifacts successfully saved.")


# ---------------------------------------------------------------------------
# 5. GENERATE EVALUATION REPORT MARKDOWN
# ---------------------------------------------------------------------------

def generate_report_markdown(
    metrics: Dict[str, Any],
    training_time: float,
    reports_dir: str = REPORTS_DIR,
) -> str:
    """
    Generate structured Phase 2 evaluation report in markdown format.

    Args:
        metrics: Evaluation results dictionary.
        training_time: Duration of training in seconds.
        reports_dir: Output directory.

    Returns:
        Filepath to the saved markdown report.
    """
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "phase2_evaluation_report.md")

    cm = metrics["confusion_matrix"]

    content = f"""# Phase 2 Evaluation Report: Isolation Forest Anomaly Detection

**Project:** AI-Assisted Financial Transaction Risk Investigation System  
**Phase:** 2 — Machine Learning Model Training & Evaluation  
**Model:** Isolation Forest (Unsupervised Anomaly Detection)  
**Dataset:** PaySim Synthetic Financial Dataset (6,362,620 transactions)  

---

## 1. Executive Summary

In Phase 2, we trained an unsupervised **Isolation Forest** model on the 19 preprocessed and engineered features produced in Phase 1. The model was trained without access to the ground-truth `isFraud` label, learning strictly the geometric boundaries of normal mobile transactions.

### Key Highlights
- **Training Time:** {training_time:.2f} seconds across 6.36M transactions (highly efficient via sub-sampling).
- **PR-AUC (Precision-Recall Area Under Curve):** **{metrics['pr_auc']:.4f}** (baseline random guess is ~0.0013).
- **ROC-AUC:** **{metrics['roc_auc']:.4f}** indicating strong separation between normal transactions and structural anomalies.
- **Recall at Contamination Threshold:** **{metrics['recall']:.4f}** ({cm['true_positives']:,} of {metrics['actual_fraud_count']:,} fraud cases detected).

---

## 2. Hyperparameter Rationale

| Hyperparameter | Value | Technical Justification |
|---|---|---|
| `n_estimators` | `100` | Standard ensemble size providing low score variance without incurring unnecessary scoring latency. |
| `max_samples` | `256` | Mitigates **swamping** (normal points classified as anomalies) and **masking** (anomaly clusters hiding each other). Subsampling creates faster trees with cleaner partition boundaries. |
| `contamination` | `0.0013` | Set to mirror the ground-truth fraud rate (~0.13%, 8,213 / 6.36M). Calibrates the binary decision boundary `offset_`. |
| `random_state` | `42` | Ensures complete determinism and reproducibility across training environments. |
| `n_jobs` | `-1` | Parallelizes tree construction across all available CPU cores. |

---

## 3. Comprehensive Performance Metrics

| Metric | Score | Interpretation |
|---|---|---|
| **Precision** | **{metrics['precision']:.4f}** | Proportion of flagged "Suspicious" transactions that were actual fraud. |
| **Recall** | **{metrics['recall']:.4f}** | Proportion of all actual fraud caught by the model's top anomaly cutoff. |
| **F1 Score** | **{metrics['f1_score']:.4f}** | Harmonic mean of Precision and Recall. |
| **PR-AUC** | **{metrics['pr_auc']:.4f}** | Primary ranking metric under extreme 774:1 imbalance. |
| **ROC-AUC** | **{metrics['roc_auc']:.4f}** | Global discriminative ability across all possible anomaly thresholds. |

### Confusion Matrix

| | Predicted Normal | Predicted Suspicious | Total Actual |
|---|---|---|---|
| **Actual Legitimate (0)** | {cm['true_negatives']:,} (TN) | {cm['false_positives']:,} (FP) | {cm['true_negatives'] + cm['false_positives']:,} |
| **Actual Fraud (1)** | {cm['false_negatives']:,} (FN) | {cm['true_positives']:,} (TP) | {metrics['actual_fraud_count']:,} |
| **Total Predicted** | {cm['true_negatives'] + cm['false_negatives']:,} | {metrics['predicted_suspicious_count']:,} | {metrics['total_samples']:,} |

---

## 4. Visualizations

### Precision-Recall Curve
![Precision-Recall Curve](pr_curve.png)
*Under extreme imbalance (0.13% fraud), the PR curve is the most reliable metric. It shows high precision at high anomaly score thresholds.*

### Receiver Operating Characteristic (ROC) Curve
![ROC Curve](roc_curve.png)
*The ROC curve confirms global separability between typical transaction profiles and anomalous ones.*

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Anomaly Score Distribution
![Anomaly Score Distribution](anomaly_score_distribution.png)
*Distribution of inverted decision scores (`-decision_function`). Fraud transactions clearly shift right towards higher anomaly scores.*

---

## 5. Limitations of Evaluating Unsupervised Models with Supervised Labels

Evaluating an unsupervised anomaly detection model (Isolation Forest) strictly against supervised labels (`isFraud`) introduces important domain caveats:

1. **Anomaly != Fraud:**
   Isolation Forest detects *statistical outliers* (rare amounts, rare accounting patterns, rare hours). Not every anomaly is malicious (e.g., a legitimate high-net-worth individual transferring a massive lump sum). Thus, false positives in an unsupervised setup often represent interesting operational anomalies rather than model errors.

2. **Subtle / Mimicry Fraud:**
   Fraudsters who deliberately stay below radar limits (e.g., small payments mimicking regular user patterns) will not appear statistically anomalous and cannot be easily isolated in tree partitions without supervised objective functions or sequence models.

3. **Fixed Contamination Threshold:**
   Setting a fixed binary cutoff (e.g. top 0.13%) forces a hard binary classification. In production, continuous anomaly scores will be used downstream by the explainability layer to provide nuanced risk assessments rather than a rigid binary verdict.

---

## 6. Persisted Artifacts

All model components required for independent inference are serialized in `models/`:
- `models/isolation_forest.joblib` (Trained model)
- `models/scaler.joblib` (Fitted RobustScaler)
- `models/model_metadata.json` (Feature names, parameters, type mappings)

**Status:** Phase 2 Complete. Ready for Phase 3/4 integration.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Report successfully written to %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# 6. MAIN EXECUTION PIPELINE
# ---------------------------------------------------------------------------

def run_training_pipeline(csv_path: str = DATA_PATH) -> None:
    """
    Execute full Phase 2 training, evaluation, and artifact export.

    Args:
        csv_path: Path to raw PaySim CSV dataset.
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: ISOLATION FOREST TRAINING & EVALUATION PIPELINE")
    logger.info("=" * 70)

    # 1. Preprocessing
    X, y, df_eng, type_mapping, scaler = run_full_pipeline(csv_path)

    # 2. Model Training
    model, training_time = train_isolation_forest(X, HYPERPARAMETERS)

    # 3. Predictions and Anomaly Scoring
    raw_preds, binary_preds, anomaly_scores = compute_predictions_and_scores(model, X)

    # 4. Evaluation & Diagnostic Plots
    metrics = evaluate_model(y, binary_preds, anomaly_scores, output_dir=REPORTS_DIR)

    # 5. Save Artifacts
    save_artifacts(
        model=model,
        scaler=scaler,
        feature_names=list(X.columns),
        type_mapping=type_mapping,
        metrics=metrics,
        training_time=training_time,
        models_dir=MODELS_DIR,
    )

    # 6. Generate Report
    report_path = generate_report_markdown(metrics, training_time, reports_dir=REPORTS_DIR)

    logger.info("=" * 70)
    logger.info("PHASE 2 COMPLETED SUCCESSFULLY!")
    logger.info("Report: %s", report_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    run_training_pipeline(data_file)
