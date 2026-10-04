from typing import Any, Dict, List, Tuple

# this method analyzes the transaction data and extracts the risk indicators
# it returns the risk level (LOW, MEDIUM, HIGH, or CRITICAL)
# risk_level - This tells us how dangerous the transaction is.
# severity_score - A numerical score from 0 to 100 showing how severe the risk is. Higher is riskier.
# is_suspicious - A simple True or False flag indicating if the transaction is flagged as suspicious.
# anomaly_score - A continuous anomaly score from Isolation Forest.
# indicators - List of triggered risk strings.
# search_query - Synthesized query string for RAG retrieval.
# summary_attributes - Key quantitative signals extracted.

def extract_risk_indicators(
    transaction: Dict[str, Any],
    anomaly_score: float = 0.0,
    is_suspicious: bool = False,
) -> Dict[str, Any]:

    txn_type = str(transaction.get("type", "UNKNOWN")).upper()
    amount = float(transaction.get("amount", 0.0))
    old_orig = float(transaction.get("oldbalanceOrg", 0.0))
    new_orig = float(transaction.get("newbalanceOrig", 0.0))
    old_dest = float(transaction.get("oldbalanceDest", 0.0))
    new_dest = float(transaction.get("newbalanceDest", 0.0))
    dest_name = str(transaction.get("nameDest", ""))
    step = int(transaction.get("step", 1))
    hour = int(transaction.get("hour_of_day", step % 24))
    formatted_time = transaction.get("formatted_time")
    sender_delta = old_orig - new_orig
    receiver_delta = new_dest - old_dest
    sender_error = abs(old_orig - new_orig - amount)
    receiver_error = abs(old_dest + amount - new_dest)

    triggered_indicators: List[str] = []
    risk_points = 0

    # 1. Account Draining Indicator (Crucial PaySim Fraud Pattern)
    if old_orig > 0 and new_orig == 0.0:
        triggered_indicators.append(
            "Account Drained to Zero: The sender's entire account balance was depleted in a single transaction."
        )
        risk_points += 35

    # 2. Large Value Transfer Indicator
    if amount >= 200_000.0:
        triggered_indicators.append(
            f"High-Value Outflow: Transaction amount (₹{amount:,.2f}) exceeds the mandatory high-value monitoring threshold of ₹200,000."
        )
        risk_points += 20
    elif amount >= 50_000.0:
        triggered_indicators.append(
            f"Substantial Value Outflow: Significant transaction amount of ₹{amount:,.2f}."
        )
        risk_points += 10

    # 3. High Proportional Balance Depletion
    if old_orig > 0:
        depletion_ratio = amount / (old_orig + 1.0)
        if depletion_ratio >= 0.90 and new_orig > 0:
            triggered_indicators.append(
                f"Severe Balance Depletion: Transferred {depletion_ratio*100:.1f}% of total available balance in one transaction."
            )
            risk_points += 15

    # 4. Accounting Mismatch on Sender Side
    if sender_error > 1.0:
        triggered_indicators.append(
            f"Sender Accounting Discrepancy: Expected debit of ₹{amount:,.2f} but actual sender balance change was ₹{sender_delta:,.2f} (Discrepancy: ₹{sender_error:,.2f})."
        )
        risk_points += 25

    # 5. Accounting Mismatch on Receiver Side (Uncredited Funds / Simulator Cancellation)
    if receiver_error > 1.0 and not dest_name.startswith("M"):
        if old_dest == new_dest and amount > 0:
            triggered_indicators.append(
                "Uncredited Recipient Balance: Recipient balance remained unchanged (₹0.00 change) despite transfer of funds."
            )
            risk_points += 30
        else:
            triggered_indicators.append(
                f"Recipient Accounting Discrepancy: Expected credit of ₹{amount:,.2f} but recipient balance changed by ₹{receiver_delta:,.2f}."
            )
            risk_points += 15

    # 6. High-Risk Transaction Channel (TRANSFER or CASH_OUT)
    if txn_type in ["TRANSFER", "CASH_OUT"]:
        if txn_type == "TRANSFER":
            triggered_indicators.append(
                "High-Risk Electronic Transfer: Transaction executed via peer-to-peer wire transfer, the primary channel for unauthorized account takeovers."
            )
            risk_points += 10
        elif txn_type == "CASH_OUT":
            triggered_indicators.append(
                "Rapid Cash Liquidation: Funds immediately moved via CASH_OUT, indicating potential money mule off-ramping."
            )
            risk_points += 10

    # 7. Off-Hours Temporal Anomaly
    if hour in [1, 2, 3, 4, 5]:
        time_suffix = f" on {formatted_time}" if formatted_time else f" during late-night hours ({hour:02d}:00)"
        triggered_indicators.append(
            f"Off-Hours Execution: Transaction initiated{time_suffix}, outside typical legitimate user operating windows."
        )
        risk_points += 10

    # Factor in ML Anomaly Score
    if is_suspicious or anomaly_score > -0.20:
        risk_points += 20

    # Normalize severity score (0 to 100)
    severity_score = min(100.0, max(5.0, float(risk_points)))

    if severity_score >= 70.0:
        risk_level = "CRITICAL"
    elif severity_score >= 45.0:
        risk_level = "HIGH"
    elif severity_score >= 25.0:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"


    # this section creates a search query for RAG (Retrieval-Augmented Generation)
    # it is used to search for similar transactions in the knowledge base
    # and retrieve relevant information to help with the investigation.

    query_parts = []
    if old_orig > 0 and new_orig == 0:
        query_parts.append("rapid account draining to zero balance")
    if amount >= 200_000:
        query_parts.append("high value wire transfer threshold monitoring")
    if receiver_error > 1.0 or (old_dest == new_dest and amount > 0):
        query_parts.append("uncredited beneficiary balance accounting mismatch mule account")
    if txn_type in ["TRANSFER", "CASH_OUT"]:
        query_parts.append(f"{txn_type} unauthorized account takeover fund diversion")
    if hour in [1, 2, 3, 4, 5]:
        query_parts.append("off-hours unusual transaction velocity")

    search_query = (
        " ".join(query_parts)
        if query_parts
        else f"{txn_type} transaction amount {amount} anomaly investigation"
    )

    return {
        "risk_level": risk_level,
        "severity_score": round(severity_score, 1),
        "is_suspicious": is_suspicious or (risk_level in ["HIGH", "CRITICAL"]),
        "anomaly_score": anomaly_score,
        "indicators": triggered_indicators if triggered_indicators else ["Standard transaction profile within baseline parameters."],
        "search_query": search_query,
        "summary_attributes": {
            "transaction_type": txn_type,
            "amount": amount,
            "hour_of_day": hour,
            "sender_account_drained": (old_orig > 0 and new_orig == 0.0),
            "sender_error": round(sender_error, 2),
            "receiver_error": round(receiver_error, 2),
            "is_merchant": dest_name.startswith("M"),
        },
    }
