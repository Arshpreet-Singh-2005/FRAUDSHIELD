"""
FraudShield - ORM models

Schema design:
  transactions   — one row per scored transaction. V1-V28 are PCA-anonymized
                   features from the source dataset; storing them as a single
                   JSON column avoids 28 near-meaningless float columns while
                   keeping Amount/Time first-class, since those are the two
                   fields we actually filter and aggregate on.
  risk_scores    — one row per scoring event, FK'd to transactions.
                   Kept separate (not merged into transactions) so a
                   transaction could be re-scored by a newer model version
                   later without losing scoring history — one-to-many by design.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, ForeignKey, JSON, Index
)
from sqlalchemy.orm import relationship

from app.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float, nullable=False)
    time_offset = Column(Float, nullable=False)
    features = Column(JSON, nullable=False)  # {"V1": ..., "V2": ..., ..., "V28": ...}
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    risk_scores = relationship("RiskScore", back_populates="transaction", cascade="all, delete-orphan")


class RiskScore(Base):
    __tablename__ = "risk_scores"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    fraud_probability = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False, index=True)
    is_fraud = Column(Boolean, nullable=False)
    model_version = Column(String(32), default="xgboost-v2.0")
    response_time_ms = Column(Float, nullable=False)
    scored_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    transaction = relationship("Transaction", back_populates="risk_scores")


# Composite index for the common query pattern: "flagged transactions in the last N hours"
Index("ix_risk_level_scored_at", RiskScore.risk_level, RiskScore.scored_at)
