# FraudShield — Real-Time Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> A real-time fraud detection system trained on 285,000+ real transactions from the Kaggle Credit Card Fraud dataset. Achieves ROC-AUC of 0.9802 using XGBoost, with a persistent database layer, a full test suite, and a REST API.

---

## Features

- **Real-Time Predictions** — Sub-5ms fraud scoring via Uvicorn ASGI server
- **Live Dashboard** — Interactive charts (probability history, session donut) powered by Chart.js
- **XGBoost Model** — Trained on 285K real Kaggle transactions with class-imbalance handling via `scale_pos_weight`
- **Risk Tiering** — 4-level risk engine: `LOW -> MEDIUM -> HIGH -> CRITICAL` with automated recommendations
- **Feature Importance** — Top 10 XGBoost features visualized with live bar charts
- **Persistent History** — Every scored transaction is written to a relational database, queryable by risk level and time window
- **Batch Scoring** — `/predict/batch` endpoint supports up to 100 transactions in one call
- **Tested** — 23 pytest tests covering endpoint validation, risk-tier boundaries, batch limits, and DB persistence

---

## Model Performance

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **0.9802** |
| **Precision (Fraud)** | **0.8617** |
| **Recall (Fraud)** | 0.84 |
| **F1 Score (Fraud)** | 0.85 |
| **Training Samples** | 284,807 |
| **Fraud Samples** | 492 (0.17%) |
| **Model** | XGBoost (`scale_pos_weight` for imbalance) |
| **Dataset** | [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

---

## System Architecture

```
+-----------------------------------------------------------+
|                    FraudShield System                     |
+-----------------------------------------------------------+
|                                                             |
|   Browser Dashboard (Chart.js + Vanilla JS)                |
|         |                                                   |
|         v                                                   |
|   FastAPI Server (Uvicorn ASGI)                             |
|   +-- GET  /             -> Serve HTML Dashboard            |
|   +-- POST /predict      -> Single transaction scoring      |
|   +-- POST /predict/batch-> Batch scoring (<=100)           |
|   +-- GET  /api/history  -> Recent scored transactions      |
|   +-- GET  /api/flagged  -> Filter by risk level + window   |
|   +-- GET  /api/stats    -> Aggregate scoring stats         |
|   +-- GET  /api/meta     -> Model metadata + stats          |
|   +-- GET  /health       -> Health check                    |
|         |                                                   |
|         v                                                   |
|   Prediction Pipeline                                       |
|   +-- StandardScaler  (Amount + Time normalization)         |
|   +-- XGBoost Model   (fraud probability)                   |
|   +-- Risk Engine     (LOW/MEDIUM/HIGH/CRITICAL)             |
|         |                                                   |
|         v                                                   |
|   Persistence Layer (SQLAlchemy)                             |
|   +-- transactions    (amount, time, PCA features)          |
|   +-- risk_scores     (FK -> transactions, one-to-many)      |
|       SQLite by default, swap via DATABASE_URL env var       |
|                                                             |
+-----------------------------------------------------------+
```

---

## Project Structure

```
FRAUDSHIELD/
|
├── app/
│   ├── main.py                  # FastAPI app - serves UI + API endpoints
│   ├── database.py              # SQLAlchemy engine/session setup
│   └── models.py                # ORM models: Transaction, RiskScore
|
├── model/
│   └── train_model.py           # Model training script (supports real + synthetic data)
|
├── saved_model/
│   ├── model.pkl                # Trained XGBoost model
│   ├── scaler.pkl               # StandardScaler for Amount + Time
│   ├── feature_cols.pkl         # Feature column order
│   └── meta.json                # Model performance metadata (AUC, precision, etc.)
|
├── templates/
│   └── index.html               # Full dashboard UI (Chart.js, vanilla JS)
|
├── tests/
│   └── test_api.py              # 23 pytest tests - endpoints, boundaries, persistence
|
├── requirements.txt
└── README.md
```

---

## Local Setup

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/Arshpreet-Singh-2005/FRAUDSHIELD.git
cd FRAUDSHIELD
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. (Optional) Add Real Kaggle Dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

Without it, the script uses synthetic data automatically.

### 4. Train the Model
```bash
python model/train_model.py
```

Expected output:
```
Found creditcard.csv - using real Kaggle dataset
284807 transactions | 492 fraudulent (0.172%)
Training XGBoost model...
ROC-AUC Score: 0.9802
All saved to saved_model/
```

### 5. Start the Server
```bash
uvicorn app.main:app --reload
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

A `fraudshield.db` SQLite file is created automatically on first run — no setup needed. To point at Postgres/MySQL instead, set `DATABASE_URL` before starting the server.

### 6. Run the Tests
```bash
pytest tests/ -v
```
23 tests covering input validation, risk-tier boundaries, batch limits, and DB persistence — all run against a temporary SQLite database, never your real data.

---

## API Reference

### `POST /predict` — Single Transaction

**Request:**
```json
{
  "V1": -1.36, "V2": -0.07, "V3": 2.54,
  "...": "...",
  "V28": -0.02,
  "Amount": 149.62,
  "Time": 0.0
}
```

**Response:**
```json
{
  "transaction_id": 1,
  "is_fraud": false,
  "fraud_probability": 0.0021,
  "risk_level": "LOW",
  "confidence": "High confidence - legitimate",
  "recommendation": "Approve transaction",
  "response_time_ms": 1.54
}
```
Every call persists the transaction and its score to the database — `transaction_id` in the response can be used to trace it in `/api/history`.

### Risk Level Mapping

| Fraud Probability | Risk Level | Action |
|---|---|---|
| `< 0.30` | LOW | Approve transaction |
| `0.30 - 0.50` | MEDIUM | Flag for manual review |
| `0.50 - 0.75` | HIGH | Block and alert customer |
| `> 0.75` | CRITICAL | Block immediately and escalate |

### `GET /health`
```json
{ "status": "healthy", "model": "XGBoost", "features": 30 }
```

### `GET /api/meta`
Returns model performance stats (AUC, precision, recall, F1, training size, feature importances).

### `GET /api/history?limit=50`
Most recently scored transactions, newest first.

### `GET /api/flagged?risk_level=HIGH&hours=24`
Transactions at or above a given risk tier, scored within the last N hours. Demonstrates filtered, time-windowed queries over the persisted history.

### `GET /api/stats`
Aggregate stats over all scoring history: total scored, fraud rate, average response time, and a breakdown by risk tier.
```json
{
  "total_scored": 142,
  "fraud_flagged": 6,
  "fraud_rate": 0.0423,
  "avg_response_time_ms": 2.11,
  "by_risk_level": { "LOW": 128, "MEDIUM": 8, "HIGH": 5, "CRITICAL": 1 }
}
```

---

## Database Design

Two tables, linked by a foreign key:

- **`transactions`** — one row per scored transaction. `Amount` and `Time` are first-class columns since they're what get filtered and aggregated on; the 28 PCA-anonymized `V1`-`V28` features are stored as a single JSON column rather than 28 near-meaningless float columns.
- **`risk_scores`** — one row per scoring event, FK'd to `transactions`. Kept separate rather than merged into `transactions` so a transaction could be re-scored by a future model version without losing scoring history — a one-to-many relationship by design.

Indexed on `risk_level` + `scored_at` (composite) to keep the `/api/flagged` time-windowed query fast as history grows. Runs on SQLite locally with zero setup; swapping to Postgres or MySQL in production is a one-line environment variable change (`DATABASE_URL`), with no code changes required.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **ML Model** | XGBoost 2.0 |
| **Preprocessing** | scikit-learn StandardScaler |
| **Backend** | FastAPI + Uvicorn |
| **Database** | SQLAlchemy ORM (SQLite / Postgres / MySQL) |
| **Testing** | Pytest, FastAPI TestClient |
| **Validation** | Pydantic v2 |
| **Frontend** | Vanilla JS + Chart.js 4.4 |
| **Dataset** | Kaggle Credit Card Fraud (ULB) |

---

## Key ML Decisions

**Why XGBoost?**
Gradient boosting excels on tabular, imbalanced datasets. The `scale_pos_weight` parameter (approximately 578x for this dataset) ensures the minority fraud class is weighted appropriately during training without oversampling artifacts.

**Why StandardScaler only on Amount + Time?**
Features V1-V28 are already PCA-transformed by the dataset authors (zero mean, unit variance). Only `Amount` and `Time` need normalization.

**Why 0.5 threshold?**
Default probability threshold optimized for balanced precision/recall on this dataset. In production, lowering to 0.3 would increase recall (catch more fraud) at the cost of more false positives.

---

## Author

**Arshpreet Singh**

- **LinkedIn:** [linkedin.com/in/arshpreet-singh-56089531a](https://www.linkedin.com/in/arshpreet-singh-56089531a)
- **GitHub:** [github.com/Arshpreet-Singh-2005](https://github.com/Arshpreet-Singh-2005)
- **Email:** sarshpreet653@gmail.com

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
