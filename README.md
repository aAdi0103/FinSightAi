
FinSight AI is a financial transaction fraud investigation assistant that helps analysts prioritize suspicious transactions. It does not automatically declare a transaction as fraud. 
Instead, it combines machine learning, rule-based risk analysis, regulatory guidance, and similar historical cases to generate an investigation report that supports analyst decision-making.

---

## Features

- Transaction risk analysis using an **Isolation Forest** model
- **19-feature** transaction risk assessment pipeline
- Explainable **rule-based risk engine**
- Severity scoring from **0–100**
- Local **RAG-style** knowledge retrieval using **TF-IDF** and **cosine similarity**
- Regulatory references from **RBI**, **FATF**, and **FinCEN**
- Historical fraud case retrieval
- Investigation reports with risk indicators and recommended analyst actions
- Regulatory knowledge base search
- Light/Dark mode dashboard

---

## Tech Stack

### Backend
- FastAPI
- Python
- scikit-learn
- Pandas
- NumPy

### Machine Learning
- Isolation Forest
- PaySim Dataset (6.36M synthetic financial transactions)

### Knowledge Retrieval
- TF-IDF
- Cosine Similarity

### Frontend
- HTML
- CSS
- JavaScript

---

## System Workflow

```text
User Transaction
       │
       ▼
Feature Engineering (19 Features)
       │
       ▼
Isolation Forest
       │
       ├── Normal
       │
       └── Suspicious + Anomaly Score
                │
                ▼
Rule-Based Risk Engine
                │
                ▼
Severity Score (0–100)
                │
                ▼
Knowledge Retrieval
(TF-IDF + Cosine Similarity)
                │
                ▼
RBI / FATF / FinCEN Guidance
+ Historical Fraud Cases
                │
                ▼
Investigation Report
                │
                ▼
Analyst Review
```

---

## Feature Engineering

Each transaction is transformed into the same feature format used during model training.

Features include:

- Transaction type
- Transaction amount
- Sender balance before transaction
- Sender balance after transaction
- Receiver balance before transaction
- Receiver balance after transaction
- Sender account drained flag
- Receiver balance update failure
- Sender accounting mismatch
- Receiver accounting mismatch
- Amount relative to sender balance
- High-value transaction flag (> ₹200,000)
- Merchant destination flag
- Log-transformed amount
- Hour of transaction
- Day information
- Additional engineered numerical features used during model training

---

## Machine Learning Model

FinSight AI uses an **Isolation Forest**, an unsupervised anomaly detection algorithm.

### Dataset

- PaySim synthetic financial transaction dataset
- Approximately **6.36 million** transactions

### Model Output

- Normal / Suspicious
- Anomaly Score

The model learns normal transaction behavior and identifies unusual transactions.

Since Isolation Forest is an **unsupervised** algorithm, fraud labels were **not** used during training. Fraud labels were used only for evaluation.

**Model Performance**

- ROC-AUC: **0.9112**

The anomaly score is used as a **triage signal**, not a final fraud decision.

---

## Rule-Based Risk Engine

The rule engine evaluates observable fraud indicators including:

- Account draining
- High-value transfers
- Balance inconsistencies
- Receiver balance update failures
- Nearly complete balance transfers
- TRANSFER and CASH_OUT transaction types
- Off-hours activity (1 AM–5 AM)
- Machine learning anomaly result

The total risk points are converted into a severity score.

| Score | Severity |
|-------:|----------|
| 0–24 | LOW |
| 25–44 | MEDIUM |
| 45–69 | HIGH |
| 70+ | CRITICAL |

---

## Knowledge Retrieval

FinSight AI uses a **local RAG-style retrieval system**.

Instead of vector embeddings and an LLM, the project performs document retrieval using:

- TF-IDF
- Cosine Similarity

The knowledge base contains:

- RBI fraud guidance
- FATF recommendations
- FinCEN fraud typologies
- Historical fraud investigation cases

Retrieved documents are used to generate deterministic investigation reports.

---

## Investigation Report

For suspicious transactions, the system generates a structured report containing:

- Executive Summary
- Triggered Risk Indicators
- Regulatory References
- Similar Historical Cases
- Recommended Analyst Actions

Depending on the calculated severity, suggested actions may include:

- Enhanced monitoring
- Beneficiary verification
- Temporary hold on destination account
- Sender re-verification
- Device/IP investigation
- SIM swap verification
- SAR consideration

---

## Dashboard

The web dashboard provides:

- Investigation Desk
- Transaction input form
- Detailed investigation results
- Regulatory Knowledge Base search
- Light/Dark theme support
