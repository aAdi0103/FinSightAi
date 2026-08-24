"""
api/routes/investigate.py
=========================
FastAPI route handlers for transaction prediction, deep investigation,
and batch CSV upload.

Author: AI-Assisted Financial Transaction Risk Investigation System
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from api.schemas import (
    InvestigationResponse,
    SingleTransactionInput,
)
from ml.inference import predict_transaction
from rag.investigator import investigate_transaction

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Investigation & Scoring"])


import datetime

def resolve_step_and_timestamp(txn_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert user-friendly transaction date/time into simulation step & temporal attributes
    while preserving original timestamp for report generation.
    """
    ts_val = txn_dict.get("timestamp")
    step_val = txn_dict.get("step")

    if ts_val:
        try:
            clean_ts = str(ts_val).replace("Z", "").strip()
            dt = datetime.datetime.fromisoformat(clean_ts)
            # Map day of month (1-31) and hour (0-23) to PaySim simulation step (1-744)
            day_idx = (dt.day - 1) % 31
            hour = dt.hour
            calc_step = (day_idx * 24) + hour + 1
            txn_dict["step"] = int(max(1, min(744, calc_step)))
            txn_dict["hour_of_day"] = hour
            txn_dict["formatted_time"] = dt.strftime("%b %d, %Y at %I:%M %p")
        except Exception:
            if not step_val:
                txn_dict["step"] = 1
    elif step_val is not None:
        txn_dict["step"] = int(step_val)
    else:
        txn_dict["step"] = 1

    return txn_dict


# ---------------------------------------------------------------------------
# 1. SINGLE TRANSACTION INVESTIGATION
# ---------------------------------------------------------------------------

@router.post(
    "/investigate",
    response_model=InvestigationResponse,
    summary="Investigate Single Transaction",
    description="Executes the full pipeline: ML Anomaly Scoring, Risk Indicator Extraction, Regulatory RAG Retrieval, and Narrative Report Generation.",
)
async def investigate_single_transaction(payload: SingleTransactionInput):
    try:
        txn_dict = payload.model_dump()
        txn_dict = resolve_step_and_timestamp(txn_dict)
        result = investigate_transaction(txn_dict)
        return result
    except Exception as e:
        logger.exception("Error during transaction investigation: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Investigation failed: {str(e)}")


# ---------------------------------------------------------------------------
# 2. QUICK ML ANOMALY SCORING
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    summary="Quick ML Anomaly Score",
    description="Fast ML-only inference returning binary prediction and raw continuous anomaly score.",
)
async def predict_single_transaction(payload: SingleTransactionInput):
    try:
        txn_dict = payload.model_dump()
        txn_dict = resolve_step_and_timestamp(txn_dict)
        result = predict_transaction(txn_dict)
        return result
    except Exception as e:
        logger.exception("Error during transaction prediction: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



