# 💳 Fraud Detection API

A real-time credit card fraud detection system built with **XGBoost** and **FastAPI**. Submit transaction features and instantly receive fraud probability, risk level, and recommended action.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

##  Live Demo

> API Docs: https://fraudshield-sbko.onrender.com/

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.97 |
| Precision (Fraud) | ~0.93 |
| Recall (Fraud) | ~0.91 |
| Model | XGBoost (scale_pos_weight for imbalance) |
| Dataset | Credit Card Fraud Detection (Kaggle / Synthetic) |

---

## 🏗️ Architecture

```
Transaction Data
      │
      ▼
  FastAPI /predict
      │
      ▼
StandardScaler (Amount + Time)
      │
      ▼
XGBoost Classifier
      │
      ▼
  Fraud Probability → Risk Level → Recommendation
```

---

## 🛠️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/Arshpreet-Singh-2005/fraud-detection-api
cd fraud-detection-api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates saved_model/ folder)
python model/train_model.py

# 4. Start the API server
uvicorn app.main:app --reload
```

API will be live at: **http://localhost:8000**
Interactive docs at: **http://localhost:8000/docs**

---

## 📡 API Endpoints

### `POST /predict` — Single Transaction
```json
// Request Body
{
  "V1": -1.36, "V2": -0.07, ..., "V28": -0.02,
  "Amount": 149.62,
  "Time": 0.0
}

// Response
{
  "is_fraud": true,
  "fraud_probability": 0.8923,
  "risk_level": "CRITICAL",
  "confidence": "Very high confidence — likely fraud",
  "recommendation": "🚨 Block immediately and escalate",
  "response_time_ms": 4.2
}
```

### `POST /predict/batch` — Batch of Transactions (max 100)
```json
// Response
{
  "total_transactions": 5,
  "flagged_as_fraud": 1,
  "fraud_rate": "20.0%",
  "results": [...]
}
```

### `GET /health` — Health Check
```json
{ "status": "healthy", "model_loaded": true }
```

### `GET /model/info` — Model Metadata
```json
{ "model_type": "XGBClassifier", "n_features": 30, "threshold": 0.5 }
```

---

## ☁️ Deploy to Render (Free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set **Build Command:** `pip install -r requirements.txt && python model/train_model.py`
5. Set **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Click **Deploy** ✅

---

## 🧰 Tech Stack

- **FastAPI** — REST API framework
- **XGBoost** — Gradient boosted trees for fraud classification
- **scikit-learn** — Preprocessing & evaluation
- **Pydantic** — Request/response validation
- **Uvicorn** — ASGI server

---

## 👤 Author

**Arshpreet Singh** — [GitHub](https://github.com/Arshpreet-Singh-2005) · [LinkedIn](https://linkedin.com/in/arshpreet-singh-56089531a)
