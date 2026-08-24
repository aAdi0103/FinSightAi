# EDA Report: PaySim Financial Transactions Dataset

**Project:** AI-Assisted Financial Transaction Risk Investigation System  
**Phase:** 1 — Data Understanding & Preprocessing  
**Generated:** 2026-08-19  
**Dataset:** PS_20174392719_1491204439457_log.csv  

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| **Total Rows** | 6,362,620 |
| **Total Columns** | 11 (raw) → 23 (after feature engineering) |
| **File Size** | ~493 MB |
| **Simulation Period** | 30 days (744 hours, 1 step = 1 hour) |
| **Missing Values** | **0** — dataset is complete |
| **Duplicate Rows** | **0** — all transactions are unique |

---

## 2. Feature Descriptions

| Feature | Type | Plain Language Description | Anomaly Signal |
|---|---|---|---|
| `step` | int | Hour of simulation (1–744) | Fraud may cluster at unusual hours |
| `type` | categorical | Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER | **Only TRANSFER and CASH_OUT ever contain fraud** |
| `amount` | float | Transaction value in local currency | Fraudulent amounts skew large |
| `nameOrig` | string | Sender customer ID (prefix C = customer) | High-cardinality; dropped from model |
| `oldbalanceOrg` | float | Sender balance BEFORE transaction | Starting balance context |
| `newbalanceOrig` | float | Sender balance AFTER transaction | Should equal oldbalanceOrg − amount |
| `nameDest` | string | Recipient ID (prefix M = merchant, C = customer) | Merchant = no balance data |
| `oldbalanceDest` | float | Receiver balance BEFORE transaction | Should increase by amount |
| `newbalanceDest` | float | Receiver balance AFTER transaction | **Unchanged balance = strongest fraud signal** |
| `isFraud` | int | Ground truth label (1 = fraud, 0 = legitimate) | Target label; never enters model |
| `isFlaggedFraud` | int | Simulator rule flag for transfers > 200k | Derived variable; excluded (data leakage) |

---

## 3. Key Observations

### 3.1 Class Imbalance (Critical)

| Class | Count | Percentage |
|---|---|---|
| Legitimate (0) | 6,354,407 | **99.87%** |
| Fraud (1) | 8,213 | **0.13%** |
| **Imbalance Ratio** | — | **~774:1** |

This is one of the most imbalanced real-world datasets used in ML research.  
**Implication for modelling:** Accuracy is a misleading metric — a model that predicts "always legitimate" achieves 99.87% accuracy but detects zero fraud. We will evaluate on **anomaly score distributions** and use the label only for post-hoc validation.

### 3.2 Transaction Type Distribution

| Type | Count | Fraud Count | Fraud Rate |
|---|---|---|---|
| CASH_OUT | 2,237,500 | ~4,116 | ~0.18% |
| PAYMENT | 2,151,495 | 0 | 0.00% |
| CASH_IN | 1,399,284 | 0 | 0.00% |
| TRANSFER | 532,909 | ~4,097 | ~0.77% |
| DEBIT | 41,432 | 0 | 0.00% |

> **Critical finding:** Fraud occurs **exclusively** in TRANSFER and CASH_OUT transactions. PAYMENT, CASH_IN, and DEBIT have zero fraud. This is a domain insight that Isolation Forest will leverage through the encoded `type` feature.

### 3.3 Amount Distribution

- Raw `amount` is **heavily right-skewed** (range: ~0 to ~92M)
- Fraudulent transactions have a **higher median amount** than legitimate ones
- The right tail is dominated by TRANSFER and CASH_OUT transactions
- **log1p transformation** normalises this distribution significantly

### 3.4 Balance Accounting Anomalies

The strongest fraud signals in the dataset are **balance discrepancies**:

- **sender_balance_drained:** Sender goes from non-zero to exactly 0 in one transaction → extremely high fraud rate
- **receiver_balance_unchanged:** Receiver balance does not increase despite funds being "sent" → hallmark of fraudulent TRANSFER cancellation by the simulator
- **sender_error & receiver_error:** Absolute accounting mismatches — these are close to 0 for all legitimate transactions and strongly non-zero for fraud

### 3.5 Merchant vs. Customer Destinations

Transactions to merchants (`nameDest` starts with `M`) always show `oldbalanceDest = 0` and `newbalanceDest = 0`. This is expected — merchant balance data is not tracked in PaySim. Without the `is_merchant_dest` flag, the model would incorrectly treat these as anomalies.

---

## 4. Interesting Patterns

1. **All fraud is in TRANSFER + CASH_OUT** — domain fact, not noise.
2. **Fraudulent TRANSFER transactions almost always drain the sender to zero** — one-shot account emptying.
3. **Fraudulent transactions show zero balance update on the receiver side** — the PaySim simulator cancels the transfer after flagging but records the transaction attempt.
4. **isFlaggedFraud only captures 16 of the 8,213 actual fraud cases** — the rule-based flag is nearly useless and is correctly excluded from the model.
5. **No missing values or duplicates** — PaySim is a clean synthetic dataset; in production data, missing balance fields would be a red flag themselves.

---

## 5. Potential Challenges

| Challenge | Description | Mitigation |
|---|---|---|
| **Extreme class imbalance (774:1)** | Unsupervised model must detect anomalies without labels | Isolation Forest is designed for this; anomaly score is continuous, not binary |
| **Balance column data leakage** | In PaySim, fraud transactions have zeroed balances due to simulator cancellation | We engineered `sender_error` and `receiver_error` to capture this signal explicitly; the raw balance columns are still included but the model learns patterns, not the exact simulator logic |
| **isFlaggedFraud leakage** | This column is derived by the simulator from `amount` | Excluded entirely from `X`; we recreate the signal ourselves via `high_amount_flag` |
| **High-cardinality IDs** | `nameOrig` and `nameDest` have millions of unique values | Dropped from `X`; `is_merchant_dest` binary captures the only useful signal from `nameDest` |
| **Right-skewed amounts** | Amounts span 0 to 92M with extreme outliers | `log_amount` normalisation + `RobustScaler` mitigates this |
| **Synthetic dataset limitations** | PaySim may not perfectly replicate real fraud patterns | Isolation Forest + RAG grounding mitigates model over-reliance on synthetic signatures |

---

## 6. Engineered Features Summary

| Feature | Formula | What It Captures |
|---|---|---|
| `balance_diff_sender` | `oldbalanceOrg − newbalanceOrig` | Expected to equal `amount`; divergence = anomaly |
| `balance_diff_receiver` | `newbalanceDest − oldbalanceDest` | Expected to equal `amount`; divergence = anomaly |
| `sender_error` | `|oldbalanceOrg − newbalanceOrig − amount|` | Absolute accounting mismatch on sender side |
| `receiver_error` | `|oldbalanceDest + amount − newbalanceDest|` | Absolute accounting mismatch on receiver side |
| `sender_balance_drained` | 1 if `oldbalanceOrg > 0` AND `newbalanceOrig == 0` | Account emptied in one transaction |
| `receiver_balance_unchanged` | 1 if `oldbalanceDest == newbalanceDest` | Receiver balance not updated despite transfer |
| `amount_vs_sender_balance_ratio` | `amount / (oldbalanceOrg + 1)` | Fraction of account balance consumed |
| `high_amount_flag` | 1 if `amount > 200,000` | Large transfer indicator |
| `is_merchant_dest` | 1 if `nameDest.startswith('M')` | Prevents false balance anomaly for merchants |
| `log_amount` | `log1p(amount)` | Log-normalised transaction size |
| `hour_of_day` | `step % 24` | Intra-day temporal pattern |
| `day_of_sim` | `step // 24` | Day-level temporal pattern |

---

## 7. Recommended Preprocessing Pipeline

```
Raw CSV (6,362,620 rows × 11 cols)
         │
         ▼
load_data()
  - Memory-efficient dtypes (float32, int32, category)
  - Validates file existence
         │
         ▼
audit_data()
  - Shape, dtypes, nulls, duplicates
  - Fraud %, type distribution
         │
         ▼
engineer_features()
  - +12 engineered columns
  - Total: 23 columns
         │
         ▼
encode_categoricals()
  - LabelEncoder on 'type' (5 classes → 0–4)
  - Saves mapping dict for decoding
         │
         ▼
scale_features()
  - RobustScaler on 11 continuous numerical cols
  - Reason: resistant to extreme financial outliers
  - Binary flags and temporal ints NOT scaled
         │
         ▼
separate_features_labels()
  - X: 19 features (drop nameOrig, nameDest, isFraud, isFlaggedFraud)
  - y: isFraud series (for post-hoc evaluation only)
         │
         ▼
X (6,362,620 × 19) ready for Phase 2: Isolation Forest
```

### Final Feature List for Model Input (19 features)

| # | Feature | Scaled? |
|---|---|---|
| 1 | step | No |
| 2 | type (encoded) | No |
| 3 | amount | **Yes** |
| 4 | oldbalanceOrg | **Yes** |
| 5 | newbalanceOrig | **Yes** |
| 6 | oldbalanceDest | **Yes** |
| 7 | newbalanceDest | **Yes** |
| 8 | balance_diff_sender | **Yes** |
| 9 | balance_diff_receiver | **Yes** |
| 10 | sender_error | **Yes** |
| 11 | receiver_error | **Yes** |
| 12 | amount_vs_sender_balance_ratio | **Yes** |
| 13 | sender_balance_drained | No (binary) |
| 14 | receiver_balance_unchanged | No (binary) |
| 15 | high_amount_flag | No (binary) |
| 16 | is_merchant_dest | No (binary) |
| 17 | log_amount | **Yes** |
| 18 | hour_of_day | No (int 0-23) |
| 19 | day_of_sim | No (int 0-30) |

---

## 8. Phase 1 Completion Checklist

- [x] Dataset loaded and validated (6,362,620 rows × 11 columns)
- [x] Full data audit (nulls, duplicates, type distribution, fraud %)
- [x] All 11 features explained with anomaly detection relevance
- [x] 12 features engineered with full rationale
- [x] `type` column encoded (LabelEncoder, mapping saved)
- [x] 11 numerical features scaled (RobustScaler, rationale documented)
- [x] X and y separated (X: 19 features, y: isFraud)
- [x] Reusable pipeline in `ml/preprocess.py`
- [x] EDA notebook in `notebooks/eda.ipynb`
- [x] This markdown report in `reports/eda_report.md`

**Ready for Phase 2: Isolation Forest Model Training**
