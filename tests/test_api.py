"""
FraudShield - Test suite

Uses a fresh temp SQLite DB per test session (never touches the real
fraudshield.db) and monkeypatches the risk-scoring function directly for
the boundary tests so they don't depend on the actual trained model's
exact probability outputs.
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Point at a temp DB *before* importing the app, so init_db() creates
# tables there instead of the real fraudshield.db
TMP_DB_FD, TMP_DB_PATH = tempfile.mkstemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{TMP_DB_PATH}"

from app.main import app, risk_meta, feature_cols  # noqa: E402
from app.database import Base, engine, get_db  # noqa: E402

client = TestClient(app)


def _sample_transaction(amount=100.0, time_offset=5000.0, **overrides):
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["Amount"] = amount
    payload["Time"] = time_offset
    payload.update(overrides)
    return payload


@pytest.fixture(autouse=True)
def clean_db():
    """Wipe and recreate all tables before every test for isolation."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


def teardown_module(module):
    # Release SQLAlchemy's pooled connections before deleting the file —
    # required on Windows, which locks open file handles (Linux doesn't).
    engine.dispose()
    os.close(TMP_DB_FD)
    try:
        os.remove(TMP_DB_PATH)
    except PermissionError:
        pass  # best-effort cleanup; harmless if the OS still holds a handle


# ── Health & meta ──

def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["features"] == len(feature_cols)


def test_meta_endpoint_returns_model_stats():
    resp = client.get("/api/meta")
    assert resp.status_code == 200
    assert "roc_auc" in resp.json()


# ── /predict — validation ──

def test_predict_rejects_missing_required_fields():
    resp = client.post("/predict", json={"V1": 0.1})  # missing Amount/Time
    assert resp.status_code == 422


def test_predict_rejects_negative_amount():
    payload = _sample_transaction(amount=-50.0)
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_negative_time():
    payload = _sample_transaction(time_offset=-1.0)
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_accepts_valid_transaction():
    payload = _sample_transaction()
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert "transaction_id" in body
    assert body["response_time_ms"] >= 0


# ── Risk-tier boundary logic (pure unit test, no HTTP/model involved) ──

@pytest.mark.parametrize(
    "probability, expected_tier",
    [
        (0.0, "LOW"),
        (0.29, "LOW"),
        (0.3, "MEDIUM"),      # lower bound of MEDIUM is inclusive
        (0.49, "MEDIUM"),
        (0.5, "HIGH"),        # lower bound of HIGH is inclusive
        (0.74, "HIGH"),
        (0.75, "CRITICAL"),   # lower bound of CRITICAL is inclusive
        (1.0, "CRITICAL"),
    ],
)
def test_risk_tier_boundaries(probability, expected_tier):
    tier, _, _ = risk_meta(probability)
    assert tier == expected_tier


# ── /predict/batch ──

def test_batch_scores_multiple_transactions():
    payload = {"transactions": [_sample_transaction(amount=a) for a in (10.0, 500.0, 2000.0)]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 3
    assert all("risk_level" in r for r in results)


def test_batch_rejects_empty_list():
    resp = client.post("/predict/batch", json={"transactions": []})
    assert resp.status_code == 422


def test_batch_rejects_over_100_transactions():
    payload = {"transactions": [_sample_transaction() for _ in range(101)]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 422


# ── Persistence: does scoring actually write to the DB? ──

def test_predict_persists_transaction_and_score():
    payload = _sample_transaction(amount=777.0)
    resp = client.post("/predict", json=payload)
    txn_id = resp.json()["transaction_id"]

    history = client.get("/api/history", params={"limit": 10}).json()
    assert any(row["transaction_id"] == txn_id and row["amount"] == 777.0 for row in history)


def test_history_orders_newest_first():
    for amount in (1.0, 2.0, 3.0):
        client.post("/predict", json=_sample_transaction(amount=amount))

    history = client.get("/api/history").json()
    amounts = [row["amount"] for row in history]
    assert amounts == list(reversed(amounts[::-1]))  # sanity: it's a list
    # Most recently scored (amount=3.0) should appear before amount=1.0
    assert amounts.index(3.0) < amounts.index(1.0)


# ── /api/flagged ──

def test_flagged_filters_by_risk_level_and_window():
    client.post("/predict", json=_sample_transaction())
    resp = client.get("/api/flagged", params={"risk_level": "LOW", "hours": 24})
    assert resp.status_code == 200
    body = resp.json()
    assert body["min_risk_level"] == "LOW"
    assert body["count"] >= 1


def test_flagged_rejects_invalid_risk_level():
    resp = client.get("/api/flagged", params={"risk_level": "EXTREME"})
    assert resp.status_code == 422


# ── /api/stats ──

def test_stats_reflects_scored_transactions():
    client.post("/predict", json=_sample_transaction())
    client.post("/predict", json=_sample_transaction())

    stats = client.get("/api/stats").json()
    assert stats["total_scored"] == 2
    assert 0.0 <= stats["fraud_rate"] <= 1.0
    assert isinstance(stats["by_risk_level"], dict)


def test_stats_on_empty_db_does_not_error():
    stats = client.get("/api/stats").json()
    assert stats["total_scored"] == 0
    assert stats["fraud_rate"] == 0.0
    assert stats["avg_response_time_ms"] is None
