"""
tests/test_api.py
=================
Integration and smoke tests for FastAPI backend routes.
"""

import io
import json
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "READY"
    assert data["services"]["ml_isolation_forest"] == "LOADED"
    assert data["services"]["rag_knowledge_index"] == "LOADED"
    print("\n[PASSED] Health Check Test")


def test_predict_endpoint():
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "nameOrig": "C123",
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C456",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "anomaly_score" in data
    print(f"\n[PASSED] Quick Predict Test: {data}")


def test_investigate_endpoint():
    payload = {
        "step": 3,
        "type": "TRANSFER",
        "amount": 250000.0,
        "nameOrig": "C841298412",
        "oldbalanceOrg": 250000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C991823122",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }
    response = client.post("/api/v1/investigate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_stratification"]["risk_level"] in ["HIGH", "CRITICAL"]
    assert len(data["retrieved_evidence"]["regulations"]) > 0
    assert len(data["retrieved_evidence"]["historical_cases"]) > 0
    assert "recommended_actions" in data["investigation_report"]
    print("\n[PASSED] Full Investigation Test")
    print(f"Executive Summary: {data['investigation_report']['executive_summary']}")


def test_knowledge_endpoints():
    r1 = client.get("/api/v1/knowledge/regulations")
    assert r1.status_code == 200
    assert r1.json()["total"] > 0

    r2 = client.get("/api/v1/knowledge/cases")
    assert r2.status_code == 200
    assert r2.json()["total"] > 0

    r3 = client.get("/api/v1/knowledge/search?q=account+draining")
    assert r3.status_code == 200
    print("\n[PASSED] Knowledge Base Endpoints Test")


def test_csv_upload_endpoint():
    csv_content = """step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest
1,PAYMENT,100.0,C111,500.0,400.0,M222,0.0,0.0
2,TRANSFER,300000.0,C333,300000.0,0.0,C444,0.0,0.0
"""
    files = {
        "file": ("test_transactions.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    }
    response = client.post("/api/v1/upload-csv", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions_processed"] == 2
    assert data["safe_transactions_count"] == 1
    assert data["suspicious_transactions_count"] == 1
    assert len(data["suspicious_investigations"]) == 1
    print(f"\n[PASSED] CSV Upload Batch Test: {data['total_transactions_processed']} processed, {data['suspicious_transactions_count']} suspicious.")


if __name__ == "__main__":
    test_health_check()
    test_predict_endpoint()
    test_investigate_endpoint()
    test_knowledge_endpoints()
    test_csv_upload_endpoint()
    print("\n" + "="*50 + "\nALL BACKEND API TESTS PASSED SUCCESSFULLY!\n" + "="*50)
