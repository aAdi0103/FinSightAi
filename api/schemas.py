
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. TRANSACTION INPUT SCHEMAS
# ---------------------------------------------------------------------------

class SingleTransactionInput(BaseModel):
    timestamp: Optional[str] = Field(None, description="Transaction ISO date-time string e.g. '2026-08-22T03:00'")
    step: Optional[int] = Field(None, description="Simulation timeline step / hour (1 to 744)")
    type: str = Field(..., description="Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER")
    amount: float = Field(..., description="Transaction amount in monetary units", gt=0)
    nameOrig: Optional[str] = Field("C_CUSTOMER", description="Originating customer/account identifier")
    oldbalanceOrg: float = Field(..., description="Initial sender balance before transaction", ge=0)
    newbalanceOrig: float = Field(..., description="New sender balance after transaction", ge=0)
    nameDest: Optional[str] = Field("C_RECIPIENT", description="Destination recipient/merchant identifier")
    oldbalanceDest: float = Field(0.0, description="Initial recipient balance before transaction", ge=0)
    newbalanceDest: float = Field(0.0, description="New recipient balance after transaction", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2026-08-22T03:00",
                "type": "TRANSFER",
                "amount": 250000.0,
                "nameOrig": "C123456789",
                "oldbalanceOrg": 250000.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C987654321",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0
            }
        }


# ---------------------------------------------------------------------------
# 2. EVIDENCE & CITATION SCHEMAS
# ---------------------------------------------------------------------------

class RegulatoryCitation(BaseModel):
    code: str
    authority: str
    title: str
    clause: str
    relevance_summary: str


class PrecedentCaseMatch(BaseModel):
    case_id: str
    title: str
    typology: str
    similarity_score: float
    historical_outcome: str


# ---------------------------------------------------------------------------
# 3. REPORT OUTPUT SCHEMAS
# ---------------------------------------------------------------------------

class MLAnalysisResult(BaseModel):
    prediction: str
    anomaly_score: float
    decision_boundary_status: str


class RiskStratificationResult(BaseModel):
    risk_level: str
    severity_score: float
    triggered_indicators: List[str]


class InvestigationNarrative(BaseModel):
    status: str
    executive_summary: str
    anomalous_characteristics: List[str]
    regulatory_compliance_notes: List[RegulatoryCitation]
    similar_precedents: List[PrecedentCaseMatch]
    recommended_actions: List[str]


class InvestigationResponse(BaseModel):
    transaction: Dict[str, Any]
    ml_analysis: MLAnalysisResult
    risk_stratification: RiskStratificationResult
    retrieved_evidence: Dict[str, Any]
    investigation_report: InvestigationNarrative


class BatchInvestigationSummary(BaseModel):
    total_transactions_processed: int
    safe_transactions_count: int
    suspicious_transactions_count: int
    critical_alerts_count: int
    suspicious_investigations: List[InvestigationResponse]
