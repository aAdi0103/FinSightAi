"""
rag/investigator.py
===================
Evidence-Backed Fraud Investigation Report Generator.

Orchestrates the entire AI investigation pipeline:
1. Scores transaction via Isolation Forest (ML Layer).
2. Extracts business risk indicators (Feature Reasoning Layer).
3. Retrieves relevant RBI regulations and historical cases (RAG Layer).
4. Generates an evidence-backed narrative investigation report (LLM Layer).

Author: AI-Assisted Financial Transaction Risk Investigation System
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from ml.inference import predict_transaction
from ml.indicators import extract_risk_indicators
from rag.retriever import retrieve_evidence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SYNTHESIS & REPORT GENERATION ENGINE
# ---------------------------------------------------------------------------

def generate_llm_investigation_report(
    transaction: Dict[str, Any],
    ml_result: Dict[str, Any],
    risk_profile: Dict[str, Any],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Synthesize ML anomaly scores, risk indicators, and retrieved evidence
    into an analyst-grade investigation report.

    Args:
        transaction: Transaction attributes.
        ml_result: Output from Isolation Forest inference.
        risk_profile: Business risk indicators and severity score.
        evidence: Retrieved regulatory articles and historical cases.

    Returns:
        Structured narrative report dictionary.
    """
    txn_type = str(transaction.get("type", "UNKNOWN"))
    amount = float(transaction.get("amount", 0.0))
    sender_id = str(transaction.get("nameOrig", "UNKNOWN_SENDER"))
    dest_id = str(transaction.get("nameDest", "UNKNOWN_RECIPIENT"))
    old_orig = float(transaction.get("oldbalanceOrg", 0.0))
    new_orig = float(transaction.get("newbalanceOrig", 0.0))
    old_dest = float(transaction.get("oldbalanceDest", 0.0))
    new_dest = float(transaction.get("newbalanceDest", 0.0))
    risk_level = risk_profile["risk_level"]
    indicators = risk_profile["indicators"]
    regulations = evidence.get("regulations", [])
    cases = evidence.get("historical_cases", [])

    # If transaction is safe and no risk indicators triggered
    if risk_level == "LOW" and not ml_result.get("prediction") == "Suspicious":
        return {
            "status": "CLEARED",
            "executive_summary": f"Transaction of ₹{amount:,.2f} via {txn_type} is within normal behavioral baseline parameters. No regulatory escalation required.",
            "anomalous_characteristics": ["Transaction amount and account balance ratios conform to legitimate user profiles."],
            "regulatory_compliance_notes": [
                {
                    "code": "RBI-KYC-BASE",
                    "authority": "Reserve Bank of India",
                    "title": "Standard KYC & Transaction Monitoring Guidelines",
                    "clause": "RBI Master Directions on KYC, 2016 — Section 4",
                    "relevance_summary": "Meets standard KYC and transaction monitoring guidelines under RBI Master Directions. No escalation required.",
                }
            ],
            "similar_precedents": [],
            "recommended_actions": [
                "Proceed with standard settlement.",
                "No immediate analyst intervention necessary."
            ],
        }

    # 1. Executive Summary
    formatted_time = transaction.get("formatted_time")
    time_phrase = f" on {formatted_time}" if formatted_time else ""
    if old_orig > 0 and new_orig == 0:
        draining_phrase = "accompanied by complete (100%) balance exhaustion"
    else:
        draining_phrase = f"with a remaining sender balance of ₹{new_orig:,.2f}"

    executive_summary = (
        f"Suspicious {txn_type} of ₹{amount:,.2f} initiated{time_phrase} from account {sender_id} to beneficiary {dest_id} "
        f"assessed at {risk_level} risk severity. "
        f"The transaction was executed {draining_phrase}, fitting known patterns of unauthorized account diversion."
    )

    # 2. Key Anomalous Characteristics
    anomalous_chars = [f"• {ind}" for ind in indicators]

    # 3. Regulatory Compliance Citations
    reg_citations = []
    for reg in regulations[:2]:
        reg_citations.append({
            "code": reg["doc_id"],
            "authority": reg["authority"],
            "title": reg["title"],
            "clause": reg.get("section", ""),
            "relevance_summary": reg.get("content", ""),
        })

    # 4. Historical Precedent Matches
    precedent_matches = []
    for case in cases[:2]:
        precedent_matches.append({
            "case_id": case["doc_id"],
            "title": case["title"],
            "typology": case.get("typology", ""),
            "similarity_score": case.get("similarity_score", 0.0),
            "historical_outcome": case.get("resolution", ""),
        })

    # 5. Recommended Analyst Actions
    recommended_actions = []
    if risk_level in ["CRITICAL", "HIGH"]:
        recommended_actions.append(f"Place an immediate temporary administrative hold on destination account {dest_id}.")
        recommended_actions.append(f"Initiate out-of-band biometric or voice re-verification with customer {sender_id}.")
        recommended_actions.append("Inspect session logs for sudden IP/device identifier shifts or SIM swap indicators.")
        if amount >= 200_000.0:
            recommended_actions.append("File Suspicious Activity Report (SAR) under RBI Section 4.1 EWS guidelines.")
    else:
        recommended_actions.append(f"Flag account {sender_id} for 48-hour enhanced velocity monitoring.")
        recommended_actions.append(f"Verify beneficiary {dest_id} relationship via SMS confirmation prompt.")

    return {
        "status": "ESCALATED_FOR_INVESTIGATION",
        "executive_summary": executive_summary,
        "anomalous_characteristics": anomalous_chars,
        "regulatory_compliance_notes": reg_citations,
        "similar_precedents": precedent_matches,
        "recommended_actions": recommended_actions,
    }


# ---------------------------------------------------------------------------
# PUBLIC FULL INVESTIGATION INTERFACE
# ---------------------------------------------------------------------------

def investigate_transaction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run complete end-to-end investigation on a financial transaction.

    Args:
        transaction: Dictionary containing transaction fields:
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181000.0,
                "nameOrig": "C1305486145",
                "oldbalanceOrg": 181000.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C553264065",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0
            }

    Returns:
        Consolidated investigation report payload containing:
          - transaction_details
          - ml_analysis (prediction, anomaly_score)
          - risk_indicators (risk_level, severity_score, triggered_flags)
          - retrieved_evidence (regulations, historical cases)
          - llm_investigation_report (narrative explanation & actions)
    """
    logger.info("Starting investigation for transaction amount: %s, type: %s ...",
                transaction.get("amount"), transaction.get("type"))

    # 1. ML Anomaly Detection Layer
    ml_result = predict_transaction(transaction)

    # 2. Risk Indicator Extraction Layer
    is_suspicious = (ml_result["prediction"] == "Suspicious")
    anomaly_score = ml_result["anomaly_score"]
    risk_profile = extract_risk_indicators(
        transaction=transaction,
        anomaly_score=anomaly_score,
        is_suspicious=is_suspicious,
    )

    # 3. RAG Retrieval Layer
    evidence = retrieve_evidence(query=risk_profile["search_query"], top_k=2)

    # 4. LLM / Evidence-Backed Report Synthesis Layer
    investigation_report = generate_llm_investigation_report(
        transaction=transaction,
        ml_result=ml_result,
        risk_profile=risk_profile,
        evidence=evidence,
    )

    return {
        "transaction": transaction,
        "ml_analysis": {
            "prediction": ml_result["prediction"],
            "anomaly_score": ml_result["anomaly_score"],
            "decision_boundary_status": "Outlier" if is_suspicious else "Inlier",
        },
        "risk_stratification": {
            "risk_level": risk_profile["risk_level"],
            "severity_score": risk_profile["severity_score"],
            "triggered_indicators": risk_profile["indicators"],
        },
        "retrieved_evidence": {
            "regulations": evidence["regulations"],
            "historical_cases": evidence["historical_cases"],
        },
        "investigation_report": investigation_report,
    }


# ---------------------------------------------------------------------------
# CLI / TESTING
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_fraudulent_drain = {
        "step": 3,
        "type": "TRANSFER",
        "amount": 250000.0,
        "nameOrig": "C841298412",
        "oldbalanceOrg": 250000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C991823122",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }

    print("\n" + "=" * 70)
    print("END-TO-END INVESTIGATION TEST (SUSPICIOUS ACCOUNT DRAINING)")
    print("=" * 70)
    full_report = investigate_transaction(sample_fraudulent_drain)
    print(json.dumps(full_report, indent=2))
