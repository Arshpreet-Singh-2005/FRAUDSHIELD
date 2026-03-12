"""
FraudShield - FastAPI Backend
Serves the web dashboard + REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pickle, numpy as np, os, time, json

# ── Load model artifacts ──
for path in ["saved_model/model.pkl", "saved_model/scaler.pkl", "saved_model/feature_cols.pkl"]:
    if not os.path.exists(path):
        raise RuntimeError(f"Missing {path} — run: python model/train_model.py")

with open("saved_model/model.pkl",        "rb") as f: model        = pickle.load(f)
with open("saved_model/scaler.pkl",       "rb") as f: scaler       = pickle.load(f)
with open("saved_model/feature_cols.pkl", "rb") as f: feature_cols = pickle.load(f)

meta = {}
if os.path.exists("saved_model/meta.json"):
    with open("saved_model/meta.json") as f: meta = json.load(f)

print("✅ FraudShield model loaded")

app = FastAPI(title="FraudShield", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Serve dashboard ──
@app.get("/", response_class=HTMLResponse)
def ui():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()

# ── Model metadata for dashboard ──
@app.get("/api/meta")
def get_meta():
    return meta

# ── Schema ──
class Transaction(BaseModel):
    V1:float=0; V2:float=0; V3:float=0; V4:float=0; V5:float=0; V6:float=0
    V7:float=0; V8:float=0; V9:float=0; V10:float=0; V11:float=0; V12:float=0
    V13:float=0; V14:float=0; V15:float=0; V16:float=0; V17:float=0; V18:float=0
    V19:float=0; V20:float=0; V21:float=0; V22:float=0; V23:float=0; V24:float=0
    V25:float=0; V26:float=0; V27:float=0; V28:float=0
    Amount: float = Field(..., ge=0)
    Time:   float = Field(..., ge=0)

def risk_meta(p):
    if p < 0.3:  return "LOW",      "High confidence — legitimate",      "✅ Approve transaction"
    if p < 0.5:  return "MEDIUM",   "Borderline — manual review advised", "⚠️ Flag for manual review"
    if p < 0.75: return "HIGH",     "High fraud likelihood",             "🔴 Block and alert customer"
    return             "CRITICAL",  "Very likely fraud",                 "🚨 Block immediately & escalate"

@app.post("/predict")
def predict(t: Transaction):
    start = time.time()
    raw    = np.array([[getattr(t, c) for c in feature_cols]])
    scaled = raw.copy()
    ai, ti = feature_cols.index("Amount"), feature_cols.index("Time")
    scaled[:, [ai, ti]] = scaler.transform(raw[:, [ai, ti]])
    prob = float(model.predict_proba(scaled)[0][1])
    risk, confidence, recommendation = risk_meta(prob)
    return {
        "is_fraud":          prob >= 0.5,
        "fraud_probability": round(prob, 4),
        "risk_level":        risk,
        "confidence":        confidence,
        "recommendation":    recommendation,
        "response_time_ms":  round((time.time()-start)*1000, 2),
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost", "features": len(feature_cols)}
