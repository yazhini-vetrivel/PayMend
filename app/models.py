"""
app/models.py

SQLAlchemy ORM models for the three core tables:
  - Payment          -> raw + classified payment records
  - RecoveryAction    -> every action the agent takes to try to recover money
  - AuditLog          -> full explainable trail of every decision/event

Uses SQLAlchemy 2.0-style declarative models.
"""

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, Enum
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class PaymentStatus(str, enum.Enum):
    CREATED = "created"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    FAILED = "failed"
    REFUNDED = "refunded"
    RECOVERED = "recovered"          # failed, then a later retry succeeded
    RECOVERY_EXHAUSTED = "recovery_exhausted"  # failed, all recovery attempts used up


class RootCause(str, enum.Enum):
    TIMEOUT = "timeout"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    INVALID_CARD = "invalid_card"
    EXPIRED_CARD = "expired_card"
    AUTH_FAILURE = "auth_failure"
    RISK_BLOCK = "risk_block"
    OTHER = "other"
    UNCLASSIFIED = "unclassified"


class ActionType(str, enum.Enum):
    RETRY = "retry"
    NOTIFY = "notify"
    SWITCH_METHOD = "switch_method"
    ESCALATE = "escalate"
    NO_ACTION = "no_action"


class ActionStatus(str, enum.Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class Payment(Base):
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    razorpay_payment_id = Column(String(64), unique=True, nullable=False, index=True)
    order_id = Column(String(64), nullable=True, index=True)

    amount = Column(Integer, nullable=False)          # smallest currency unit (paise)
    currency = Column(String(8), nullable=False, default="INR")

    status = Column(String(32), nullable=False, default=PaymentStatus.FAILED.value)
    failure_code = Column(String(64), nullable=True)
    failure_description = Column(Text, nullable=True)

    root_cause = Column(String(32), nullable=True, default=RootCause.UNCLASSIFIED.value)
    root_cause_confidence = Column(Float, nullable=True)   # if using ML classifier
    root_cause_method = Column(String(16), nullable=True)  # "rule" or "ml"

    method = Column(String(32), nullable=True)          # card, upi, netbanking, etc.
    customer_id = Column(String(64), nullable=True, index=True)

    is_recovered = Column(Boolean, default=False)
    recovered_amount = Column(Integer, default=0)

    # Set True once the recovery pipeline has picked this payment up,
    # so we don't process the same failed payment twice (Step 6).
    processed_for_recovery = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    recovery_actions = relationship(
        "RecoveryAction", back_populates="payment", cascade="all, delete-orphan"
    )
    audit_logs = relationship(
        "AuditLog", back_populates="payment", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return (f"<Payment {self.razorpay_payment_id} amount={self.amount} "
                f"status={self.status} root_cause={self.root_cause}>")


class RecoveryAction(Base):
    __tablename__ = "recovery_actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    payment_id = Column(Integer, ForeignKey("payments.id"), nullable=False, index=True)

    action_type = Column(String(32), nullable=False)
    attempt_number = Column(Integer, nullable=False, default=1)
    status = Column(String(16), nullable=False, default=ActionStatus.PENDING.value)

    result_details = Column(Text, nullable=True)  # JSON-encoded string
    idempotency_key = Column(String(64), nullable=True, unique=True)  # prevents duplicate execution

    created_at = Column(DateTime(timezone=True), default=utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    payment = relationship("Payment", back_populates="recovery_actions")

    def __repr__(self):
        return (f"<RecoveryAction payment_id={self.payment_id} "
                f"type={self.action_type} attempt={self.attempt_number} status={self.status}>")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    payment_id = Column(Integer, ForeignKey("payments.id"), nullable=True, index=True)

    event_type = Column(String(64), nullable=False)   # e.g. "payment_received", "root_cause_assigned"
    root_cause = Column(String(32), nullable=True)
    action_type = Column(String(32), nullable=True)

    details = Column(Text, nullable=True)  # JSON-encoded string with free-form context
    created_at = Column(DateTime(timezone=True), default=utcnow, index=True)

    payment = relationship("Payment", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog event={self.event_type} payment_id={self.payment_id}>"
