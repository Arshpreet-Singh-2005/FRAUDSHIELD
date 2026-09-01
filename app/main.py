"""
FraudShield - FastAPI Backend
Serves the web dashboard + REST API, persists every scored transaction
to a relational DB, and exposes query endpoints over that history.
"""

from datetime import datetime, timedelta, timezone
import pickle
import time
import json
import os

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db, init_db
from app import models

# ── Resolve paths relative to this file, not the working directory ──
# (This is the actual fix the "Fix absolute paths for Render deployment"
#  commit was meant to make — the previous attempt wiped the file instead.)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(SAVED_MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(SAVED_MODEL_DIR, "feature_cols.pkl")
META_PATH = os.path.join(SAVED_MODEL_DIR, "meta.json")

# ── Load model artifacts ──
for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]:
    if not os.path.exists(path):
        raise RuntimeError(f"Missing {path} — run: python model/train_model.py")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(FEATURES_PATH, "rb") as f:
    feature_cols = pickle.load(f)

meta = {}
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        meta = json.load(f)

print("✅ FraudShield model loaded")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="FraudShield", version="2.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Serve dashboard ──
@app.get("/", response_class=HTMLResponse)
def ui():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    with open(index_path, encoding="utf-8") as f:
        return f.read()


# ── Model metadata for dashboard ──
@app.get("/api/meta")
def get_meta():
    return meta


# ── Schema ──
class Transaction(BaseModel):
    V1: float = 0; V2: float = 0; V3: float = 0; V4: float = 0; V5: float = 0; V6: float = 0
    V7: float = 0; V8: float = 0; V9: float = 0; V10: float = 0; V11: float = 0; V12: float = 0
    V13: float = 0; V14: float = 0; V15: float = 0; V16: float = 0; V17: float = 0; V18: float = 0
    V19: float = 0; V20: float = 0; V21: float = 0; V22: float = 0; V23: float = 0; V24: float = 0
    V25: float = 0; V26: float = 0; V27: float = 0; V28: float = 0
    Amount: float = Field(..., ge=0)
    Time: float = Field(..., ge=0)


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=100)


def risk_meta(p: float):
    if p < 0.3:
        return "LOW", "High confidence — legitimate", "✅ Approve transaction"
    if p < 0.5:
        return "MEDIUM", "Borderline — manual review advised", "⚠️ Flag for manual review"
    if p < 0.75:
        return "HIGH", "High fraud likelihood", "🔴 Block and alert customer"
    return "CRITICAL", "Very likely fraud", "🚨 Block immediately & escalate"


def score_transaction(t: Transaction, db: Session) -> dict:
    """Scores one transaction, persists it + its risk score, returns the API response."""
    start = time.time()
    raw = np.array([[getattr(t, c) for c in feature_cols]])
    scaled = raw.copy()
    ai, ti = feature_cols.index("Amount"), feature_cols.index("Time")
    scaled[:, [ai, ti]] = scaler.transform(raw[:, [ai, ti]])
    prob = float(model.predict_proba(scaled)[0][1])
    risk, confidence, recommendation = risk_meta(prob)
    response_time_ms = round((time.time() - start) * 1000, 2)

    # Persist: one transaction row + one risk_score row, linked by FK
    v_features = {c: getattr(t, c) for c in feature_cols if c not in ("Amount", "Time")}
    db_txn = models.Transaction(amount=t.Amount, time_offset=t.Time, features=v_features)
    db.add(db_txn)
    db.flush()  # get db_txn.id without committing yet

    db_score = models.RiskScore(
        transaction_id=db_txn.id,
        fraud_probability=round(prob, 4),
        risk_level=risk,
        is_fraud=prob >= 0.5,
        response_time_ms=response_time_ms,
    )
    db.add(db_score)
    db.commit()

    return {
        "transaction_id": db_txn.id,
        "is_fraud": prob >= 0.5,
        "fraud_probability": round(prob, 4),
        "risk_level": risk,
        "confidence": confidence,
        "recommendation": recommendation,
        "response_time_ms": response_time_ms,
    }


@app.post("/predict")
def predict(t: Transaction, db: Session = Depends(get_db)):
    return score_transaction(t, db)


@app.post("/predict/batch")
def predict_batch(req: BatchRequest, db: Session = Depends(get_db)):
    return {"results": [score_transaction(t, db) for t in req.transactions]}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost", "features": len(feature_cols)}


# ── History / query endpoints — demonstrate real read access over persisted data ──

@app.get("/api/history")
def get_history(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    """Most recent scored transactions, newest first."""
    rows = (
        db.query(models.RiskScore, models.Transaction)
        .join(models.Transaction, models.RiskScore.transaction_id == models.Transaction.id)
        .order_by(models.RiskScore.scored_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "transaction_id": txn.id,
            "amount": txn.amount,
            "fraud_probability": score.fraud_probability,
            "risk_level": score.risk_level,
            "is_fraud": score.is_fraud,
            "scored_at": score.scored_at.isoformat(),
        }
        for score, txn in rows
    ]


@app.get("/api/flagged")
def get_flagged(
    risk_level: str = Query("HIGH", pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$"),
    hours: int = Query(24, ge=1, le=720),
    db: Session = Depends(get_db),
):
    """Transactions at or above a given risk level in the last N hours."""
    tier_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    min_index = tier_order.index(risk_level)
    eligible_tiers = tier_order[min_index:]
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    rows = (
        db.query(models.RiskScore, models.Transaction)
        .join(models.Transaction, models.RiskScore.transaction_id == models.Transaction.id)
        .filter(models.RiskScore.risk_level.in_(eligible_tiers))
        .filter(models.RiskScore.scored_at >= since)
        .order_by(models.RiskScore.scored_at.desc())
        .all()
    )
    return {
        "count": len(rows),
        "window_hours": hours,
        "min_risk_level": risk_level,
        "transactions": [
            {
                "transaction_id": txn.id,
                "amount": txn.amount,
                "fraud_probability": score.fraud_probability,
                "risk_level": score.risk_level,
                "scored_at": score.scored_at.isoformat(),
            }
            for score, txn in rows
        ],
    }


@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    """Aggregate stats over all scoring history — total scored, fraud rate, avg latency."""
    total = db.query(func.count(models.RiskScore.id)).scalar() or 0
    fraud_count = db.query(func.count(models.RiskScore.id)).filter(models.RiskScore.is_fraud.is_(True)).scalar() or 0
    avg_latency = db.query(func.avg(models.RiskScore.response_time_ms)).scalar()
    by_tier = (
        db.query(models.RiskScore.risk_level, func.count(models.RiskScore.id))
        .group_by(models.RiskScore.risk_level)
        .all()
    )
    return {
        "total_scored": total,
        "fraud_flagged": fraud_count,
        "fraud_rate": round(fraud_count / total, 4) if total else 0.0,
        "avg_response_time_ms": round(avg_latency, 2) if avg_latency else None,
        "by_risk_level": {tier: count for tier, count in by_tier},
    }
