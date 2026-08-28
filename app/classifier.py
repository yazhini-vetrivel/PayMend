"""
app/classifier.py

Root-cause classification: maps a failed Payment's raw failure_code /
failure_description into a business-level root cause label.

  classify_failure_rule_based(payment) -> str   (must-have, always available)
  classify_failure_ml(payment)         -> str   (optional, only if a
                                                   trained model file exists)
  classify_failure(payment)            -> str   (what the rest of the app
                                                   calls — rule-based primary,
                                                   falls back / compares to ML)

Root causes: timeout, insufficient_funds, invalid_card, expired_card,
             auth_failure, risk_block, other
"""

import os
import re

from app.models import RootCause

# ----------------------------------------------------------------------
# 7.1 Rule-based classifier (must-do)
# ----------------------------------------------------------------------
# Ordered list of (keyword patterns, root_cause). Checked against
# failure_code + failure_description (lowercased). First match wins,
# so put more specific rules before generic ones.
#
# Based on Razorpay's documented error codes/descriptions:
# https://razorpay.com/docs/payments/payments/failures/
_RULES = [
    # --- timeout ---
    (r"timed?\s*out|timeout|gateway.*(no response|unreachable)", RootCause.TIMEOUT),

    # --- insufficient funds ---
    (r"insufficient\s*(funds|balance)|not enough (funds|balance)", RootCause.INSUFFICIENT_FUNDS),

    # --- expired card (check before generic "invalid card") ---
    (r"expired|expiry.*(invalid|past|elapsed)", RootCause.EXPIRED_CARD),

    # --- invalid card ---
    (r"invalid card|card number is invalid|incorrect (cvv|card)|invalid cvv|invalid expiry", RootCause.INVALID_CARD),

    # --- auth failure (3DS / OTP) ---
    (r"3d\s*secure|3ds|authentication failed|otp.*(fail|incorrect|invalid)|otp not verified", RootCause.AUTH_FAILURE),

    # --- risk / fraud block ---
    (r"risk|fraud|blocked by (bank|issuer)|flagged|suspicious", RootCause.RISK_BLOCK),
]

# Fast-path on Razorpay's failure_code field alone, when description is
# missing or too generic to keyword-match reliably.
_CODE_DEFAULTS = {
    "GATEWAY_ERROR": RootCause.TIMEOUT,        # most GATEWAY_ERROR failures in test mode are timeout/bank-side
    "BAD_REQUEST_ERROR": RootCause.INVALID_CARD,  # most BAD_REQUEST_ERROR are malformed/invalid input
    "SERVER_ERROR": RootCause.TIMEOUT,
}


def classify_failure_rule_based(payment) -> str:
    """
    payment: a Payment ORM row (or any object with .failure_code and
    .failure_description attributes).
    Returns one of the RootCause string values.
    """
    code = (getattr(payment, "failure_code", None) or "").strip()
    description = (getattr(payment, "failure_description", None) or "").strip().lower()

    # 1. Try description-based keyword rules first — most specific.
    for pattern, cause in _RULES:
        if description and re.search(pattern, description):
            return cause.value

    # 2. Fall back to failure_code defaults.
    if code in _CODE_DEFAULTS:
        return _CODE_DEFAULTS[code].value

    # 3. Nothing matched.
    return RootCause.OTHER.value


# ----------------------------------------------------------------------
# 7.3 Optional ML classifier
# ----------------------------------------------------------------------
_ML_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", "notebooks/root_cause_model.joblib")
_ml_model = None
_ml_encoders = None


def _load_ml_model():
    """Lazily loads the trained model + encoders. Returns None if not trained yet."""
    global _ml_model, _ml_encoders
    if _ml_model is not None:
        return _ml_model, _ml_encoders

    if not os.path.exists(_ML_MODEL_PATH):
        return None, None

    import joblib
    bundle = joblib.load(_ML_MODEL_PATH)
    _ml_model = bundle["model"]
    _ml_encoders = bundle["encoders"]
    return _ml_model, _ml_encoders


def classify_failure_ml(payment):
    """
    Returns (label: str, confidence: float) or (None, None) if no trained
    model is available yet (train_classifier.ipynb hasn't been run).
    """
    model, encoders = _load_ml_model()
    if model is None:
        return None, None

    import pandas as pd

    row = {
        "failure_code": getattr(payment, "failure_code", None) or "UNKNOWN",
        "method": getattr(payment, "method", None) or "unknown",
        "amount_bucket": _amount_bucket(getattr(payment, "amount", 0)),
    }
    df = pd.DataFrame([row])

    for col, encoder in encoders.items():
        if col == "root_cause":
            continue
        # unseen category at inference time -> fall back to a neutral value
        df[col] = df[col].apply(lambda v: v if v in encoder.classes_ else encoder.classes_[0])
        df[col] = encoder.transform(df[col])

    pred = model.predict(df)[0]
    proba = max(model.predict_proba(df)[0])
    label = encoders["root_cause"].inverse_transform([pred])[0]
    return label, float(proba)


def _amount_bucket(amount: int) -> str:
    """Coarse amount bucketing so the ML model isn't overfit to exact paise values."""
    if amount < 10000:
        return "low"
    if amount < 100000:
        return "mid"
    return "high"


# ----------------------------------------------------------------------
# Combined entry point — what recovery.py / policy.py actually call
# ----------------------------------------------------------------------
def classify_failure(payment, prefer: str = "rule") -> dict:
    """
    Returns:
        {
            "root_cause": str,
            "method": "rule" | "ml",
            "confidence": float | None,
            "rule_based_label": str,
            "ml_label": str | None,
            "ml_confidence": float | None,
        }

    prefer="rule" (default): rule-based is the source of truth (deterministic,
    auditable, no training data dependency). ML result is computed too (if a
    model exists) purely for comparison/logging — this is how you produce the
    "ML vs rule-based accuracy" comparison the project asks for.
    """
    rule_label = classify_failure_rule_based(payment)
    ml_label, ml_confidence = classify_failure_ml(payment)

    if prefer == "ml" and ml_label is not None:
        final_label, final_method, final_conf = ml_label, "ml", ml_confidence
    else:
        final_label, final_method, final_conf = rule_label, "rule", None

    return {
        "root_cause": final_label,
        "method": final_method,
        "confidence": final_conf,
        "rule_based_label": rule_label,
        "ml_label": ml_label,
        "ml_confidence": ml_confidence,
    }


if __name__ == "__main__":
    # quick smoke test: python -m app.classifier
    class FakePayment:
        def __init__(self, code, desc):
            self.failure_code = code
            self.failure_description = desc
            self.method = "card"
            self.amount = 49900

    tests = [
        ("GATEWAY_ERROR", "The card issuing bank server timed out."),
        ("BAD_REQUEST_ERROR", "Insufficient funds in the account."),
        ("BAD_REQUEST_ERROR", "The card number is invalid."),
        ("BAD_REQUEST_ERROR", "The card has expired."),
        ("GATEWAY_ERROR", "3D Secure authentication failed."),
        ("GATEWAY_ERROR", "Payment flagged by risk engine."),
        ("SERVER_ERROR", "Something unexpected happened."),
    ]
    for code, desc in tests:
        result = classify_failure(FakePayment(code, desc))
        print(f"{code:20s} | {desc:45s} -> {result['root_cause']}")
