"""
FraudShield - Model Training Script
Supports real Kaggle creditcard.csv dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import pickle
import os
import json

KAGGLE_CSV = "creditcard.csv"

# ── Load Data ──
if os.path.exists(KAGGLE_CSV):
    print(f"✅ Found {KAGGLE_CSV} — using REAL Kaggle dataset!")
    df = pd.read_csv(KAGGLE_CSV)
    total = len(df)
    fraud_count = int(df['Class'].sum())
    print(f"📦 {total:,} transactions | {fraud_count:,} fraudulent ({df['Class'].mean()*100:.3f}%)")
else:
    print("⚠️  creditcard.csv not found — using synthetic data.")
    print("   Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n")
    np.random.seed(42)
    n_samples, n_fraud = 50000, 500
    legit = pd.DataFrame(np.random.randn(n_samples, 28), columns=[f"V{i}" for i in range(1, 29)])
    legit["Amount"] = np.abs(np.random.exponential(88, n_samples))
    legit["Time"]   = np.random.uniform(0, 172800, n_samples)
    legit["Class"]  = 0
    fraud = pd.DataFrame(np.random.randn(n_fraud, 28) * 1.8 + 2, columns=[f"V{i}" for i in range(1, 29)])
    fraud["Amount"] = np.abs(np.random.exponential(300, n_fraud))
    fraud["Time"]   = np.random.uniform(0, 172800, n_fraud)
    fraud["Class"]  = 1
    df = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    total = len(df)
    fraud_count = n_fraud
    print(f"📦 Synthetic: {total:,} transactions | {fraud_count} fraudulent")

# ── Preprocess ──
feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
X = df[feature_cols].copy()
y = df["Class"]

scaler = StandardScaler()
X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── Train ──
print("\n🤖 Training XGBoost (this may take ~1 min on real data)...")
scale = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── Evaluate ──
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
report  = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], output_dict=True)
auc     = roc_auc_score(y_test, y_proba)
cm      = confusion_matrix(y_test, y_pred).tolist()

print("\n📈 Model Performance:")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
print(f"ROC-AUC Score: {auc:.4f}")

# ── Save model + metadata ──
os.makedirs("saved_model", exist_ok=True)

with open("saved_model/model.pkl",        "wb") as f: pickle.dump(model,        f)
with open("saved_model/scaler.pkl",       "wb") as f: pickle.dump(scaler,       f)
with open("saved_model/feature_cols.pkl", "wb") as f: pickle.dump(feature_cols, f)

# Save stats for dashboard
meta = {
    "roc_auc":       round(auc, 4),
    "precision":     round(report["Fraud"]["precision"], 4),
    "recall":        round(report["Fraud"]["recall"], 4),
    "f1":            round(report["Fraud"]["f1-score"], 4),
    "total_trained": total,
    "fraud_trained": fraud_count,
    "confusion_matrix": cm,
    "feature_importances": dict(zip(feature_cols, model.feature_importances_.tolist()))
}
with open("saved_model/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ All saved to saved_model/")
print("   Run: uvicorn app.main:app --reload")
print("   Open: http://localhost:8000")
