# 🛡️ FraudShield — Real-Time Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://fraudshield-sbko.onrender.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> A production-grade, real-time fraud detection system trained on **285,000+ real transactions** from the Kaggle Credit Card Fraud dataset. Achieves **ROC-AUC of 0.9802** using XGBoost with a live interactive dashboard and REST API.

---

## 🌐 Live Demo

### 👉 [https://fraudshield-sbko.onrender.com](https://fraudshield-sbko.onrender.com)

>  Hosted on Render free tier — may take 30–60 seconds to wake up on first visit.

---

## ✨ Features

- **⚡ Real-Time Predictions** — Sub-5ms fraud scoring via Uvicorn ASGI server
- **📊 Live Dashboard** — Interactive charts (probability history, session donut) powered by Chart.js
- **🧠 XGBoost Model** — Trained on 285K real Kaggle transactions with class-imbalance handling via `scale_pos_weight`
- **🎯 Risk Tiering** — 4-level risk engine: `LOW → MEDIUM → HIGH → CRITICAL` with automated recommendations
- **📈 Feature Importance** — Top 10 XGBoost features visualized with live bar charts
- **🗂️ Session History** — Every transaction logged with amount, probability, risk level, and response time
- **🔁 Batch Scoring** — `/predict/batch` endpoint supports up to 100 transactions in one call
- **☁️ Cloud Deployed** — Live on Render with CI/CD via GitHub push

---

## 📊 Model Performance

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

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FraudShield System                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Browser Dashboard (Chart.js + Vanilla JS)             │
│         │                                               │
│         ▼                                               │
│   FastAPI Server (Uvicorn ASGI)                         │
│   ├── GET  /           → Serve HTML Dashboard           │
│   ├── POST /predict    → Single transaction scoring     │
│   ├── GET  /api/meta   → Model metadata + stats         │
│   └── GET  /health     → Health check                   │
│         │                                               │
│         ▼                                               │
│   Prediction Pipeline                                   │
│   ├── StandardScaler  (Amount + Time normalization)     │
│   ├── XGBoost Model   (fraud probability)               │
│   └── Risk Engine     (LOW/MEDIUM/HIGH/CRITICAL)        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
FRAUDSHIELD/
│
├── app/
│   └── main.py                  # FastAPI app — serves UI + API endpoints
│
├── model/
│   └── train_model.py           # Model training script (supports real + synthetic data)
│
├── saved_model/
│   ├── model.pkl                # Trained XGBoost model
│   ├── scaler.pkl               # StandardScaler for Amount + Time
│   ├── feature_cols.pkl         # Feature column order
│   └── meta.json                # Model performance metadata (AUC, precision, etc.)
│
├── templates/
│   └── index.html               # Full dashboard UI (Chart.js, vanilla JS)
│
├── requirements.txt
├── Procfile                     # Render deployment config
└── README.md
```

---

## 🛠️ Local Setup

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
✅ Found creditcard.csv — using REAL Kaggle dataset!
📦 284807 transactions | 492 fraudulent (0.172%)
🤖 Training XGBoost model...
ROC-AUC Score: 0.9802
✅ All saved to saved_model/
```

### 5. Start the Server
```bash
uvicorn app.main:app --reload
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 📡 API Reference

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
  "is_fraud": false,
  "fraud_probability": 0.0021,
  "risk_level": "LOW",
  "confidence": "High confidence — legitimate",
  "recommendation": "✅ Approve transaction",
  "response_time_ms": 1.54
}
```

### Risk Level Mapping

| Fraud Probability | Risk Level | Action |
|---|---|---|
| `< 0.30` | 🟢 LOW | Approve transaction |
| `0.30 – 0.50` | 🟡 MEDIUM | Flag for manual review |
| `0.50 – 0.75` | 🟠 HIGH | Block and alert customer |
| `> 0.75` | 🔴 CRITICAL | Block immediately & escalate |

### `GET /health`
```json
{ "status": "healthy", "model": "XGBoost", "features": 30 }
```

### `GET /api/meta`
Returns model performance stats (AUC, precision, recall, F1, training size, feature importances).

---

## ☁️ Deploy to Render (Free)

1. Fork this repo to your GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set these values:

| Field | Value |
|---|---|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Free |

5. Click **Deploy** ✅

> Note: `creditcard.csv` is excluded via `.gitignore`. The pre-trained `saved_model/` is included in the repo so Render loads it directly without retraining.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **ML Model** | XGBoost 2.0 |
| **Preprocessing** | scikit-learn StandardScaler |
| **Backend** | FastAPI + Uvicorn |
| **Validation** | Pydantic v2 |
| **Frontend** | Vanilla JS + Chart.js 4.4 |
| **Deployment** | Render (free tier) |
| **Dataset** | Kaggle Credit Card Fraud (ULB) |

---

## 🔬 Key ML Decisions

**Why XGBoost?**
Gradient boosting excels on tabular, imbalanced datasets. The `scale_pos_weight` parameter (≈578x for this dataset) ensures the minority fraud class is weighted appropriately during training without oversampling artifacts.

**Why StandardScaler only on Amount + Time?**
Features V1–V28 are already PCA-transformed by the dataset authors (zero mean, unit variance). Only `Amount` and `Time` need normalization.

**Why 0.5 threshold?**
Default probability threshold optimized for balanced precision/recall on this dataset. In production, lowering to 0.3 would increase recall (catch more fraud) at the cost of more false positives.

---

## 👤 Author

**Arshpreet Singh**

- 🌐 **Live App:** [fraudshield-sbko.onrender.com](https://fraudshield-sbko.onrender.com)
- 💼 **LinkedIn:** [linkedin.com/in/arshpreet-singh-56089531a](https://www.linkedin.com/in/arshpreet-singh-56089531a)
- 🐙 **GitHub:** [github.com/Arshpreet-Singh-2005](https://github.com/Arshpreet-Singh-2005)
- 📧 **Email:** sarshpreet653@gmail.com

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this project useful, please consider starring the repo!**
