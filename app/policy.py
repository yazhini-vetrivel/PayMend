"""
app/policy.py

Policy engine: given a root_cause + context, decides WHAT to do about it.
This is deliberately separate from HOW to do it (that's recovery.py) so
the decision logic is easy to read, test, and explain in an audit log.

    choose_action(root_cause, context) -> action_plan dict

action_plan shape:
    {
        "action_type": "retry" | "notify" | "escalate" | "no_action",
        "params": {...},       # everything recovery.py needs to execute it
        "reason": "...",       # human-readable explanation (goes in AuditLog)
    }
"""

from app.models import ActionType, RootCause

# ----------------------------------------------------------------------
# Bounded limits — these are what make every money action "bounded and
# gated" rather than an agent that can retry forever or notify endlessly.
# ----------------------------------------------------------------------
MAX_RETRIES_TIMEOUT = 3
MAX_RETRIES_AUTH_FAILURE = 1
INSUFFICIENT_FUNDS_RETRY_DELAY_HOURS = 24


def choose_action(root_cause: str, context: dict) -> dict:
    """
    root_cause: one of the RootCause values (string).
    context: {
        "attempt_count": int,      # how many recovery attempts already made for this payment
        "method": str,             # card, upi, netbanking, etc.
        "amount": int,             # paise
        "customer_id": str | None,
        "customer_history": dict | None,  # optional: e.g. {"past_failures": 2}
    }
    """
    attempt_count = context.get("attempt_count", 0)

    # ------------------------------------------------------------------
    # timeout -> retry with backoff, bounded at MAX_RETRIES_TIMEOUT
    # ------------------------------------------------------------------
    if root_cause == RootCause.TIMEOUT.value:
        if attempt_count < MAX_RETRIES_TIMEOUT:
            backoff_seconds = _exponential_backoff(attempt_count)
            return {
                "action_type": ActionType.RETRY.value,
                "params": {
                    "delay_seconds": backoff_seconds,
                    "max_retries": MAX_RETRIES_TIMEOUT,
                },
                "reason": (
                    f"Gateway/network timeout — transient, so retry "
                    f"(attempt {attempt_count + 1}/{MAX_RETRIES_TIMEOUT}) "
                    f"after {backoff_seconds}s backoff."
                ),
            }
        return _escalate(f"Timeout persisted after {MAX_RETRIES_TIMEOUT} retries — needs manual review.")

    # ------------------------------------------------------------------
    # insufficient_funds -> notify now, optional single delayed retry
    # ------------------------------------------------------------------
    if root_cause == RootCause.INSUFFICIENT_FUNDS.value:
        if attempt_count == 0:
            return {
                "action_type": ActionType.NOTIFY.value,
                "params": {
                    "channel": "email",
                    "template": "insufficient_funds",
                    "schedule_retry_after_hours": INSUFFICIENT_FUNDS_RETRY_DELAY_HOURS,
                },
                "reason": (
                    "Insufficient funds — can't fix by retrying immediately. "
                    "Notify customer and schedule one delayed retry "
                    f"in {INSUFFICIENT_FUNDS_RETRY_DELAY_HOURS}h in case balance is topped up."
                ),
            }
        # delayed retry window already used once — don't nag/retry indefinitely
        return _escalate("Insufficient funds persisted after notify + delayed retry window.")

    # ------------------------------------------------------------------
    # invalid_card / expired_card -> notify to update card, never auto-retry same card
    # ------------------------------------------------------------------
    if root_cause in (RootCause.INVALID_CARD.value, RootCause.EXPIRED_CARD.value):
        if attempt_count == 0:
            return {
                "action_type": ActionType.NOTIFY.value,
                "params": {
                    "channel": "email",
                    "template": "update_card",
                },
                "reason": (
                    f"{root_cause} — retrying the same card will fail again by definition. "
                    "Ask customer to update card details instead."
                ),
            }
        return _escalate(f"{root_cause} — customer didn't update card after notify. Escalating.")

    # ------------------------------------------------------------------
    # auth_failure -> retry once (proper 3DS flow), then notify
    # ------------------------------------------------------------------
    if root_cause == RootCause.AUTH_FAILURE.value:
        if attempt_count < MAX_RETRIES_AUTH_FAILURE:
            return {
                "action_type": ActionType.RETRY.value,
                "params": {
                    "delay_seconds": 0,
                    "max_retries": MAX_RETRIES_AUTH_FAILURE,
                    "force_3ds": True,
                },
                "reason": "3DS/auth failure can be a one-off glitch — retry once with proper 3DS flow.",
            }
        return {
            "action_type": ActionType.NOTIFY.value,
            "params": {
                "channel": "email",
                "template": "auth_failure_retry_failed",
            },
            "reason": "Auth failure persisted after one retry — notify customer to complete authentication manually.",
        }

    # ------------------------------------------------------------------
    # risk_block -> never auto-retry, straight to escalate
    # ------------------------------------------------------------------
    if root_cause == RootCause.RISK_BLOCK.value:
        return _escalate("Flagged by risk/fraud engine — auto-retry could look like abuse. Manual review required.")

    # ------------------------------------------------------------------
    # other / unclassified -> safest default is escalate, not silent retry
    # ------------------------------------------------------------------
    return _escalate(f"Root cause '{root_cause}' has no defined policy — routing to manual review rather than guessing.")


def _escalate(reason: str) -> dict:
    return {
        "action_type": ActionType.ESCALATE.value,
        "params": {},
        "reason": reason,
    }


def _exponential_backoff(attempt_count: int, base_seconds: int = 30) -> int:
    """attempt 0 -> 30s, attempt 1 -> 60s, attempt 2 -> 120s ..."""
    return base_seconds * (2 ** attempt_count)


# ----------------------------------------------------------------------
# Optional advanced bit: stats-informed action preference.
# ----------------------------------------------------------------------
# Tracks, per (root_cause, action_type), how often that action has
# actually succeeded historically. This doesn't override the bounded
# rules above (safety limits stay hard limits) — it's meant to inform
# a future version where, when a root cause has more than one
# reasonable action, you pick the historically better-performing one.
class ActionStatsTracker:
    def __init__(self):
        # {(root_cause, action_type): {"success": int, "total": int}}
        self._stats = {}

    def record(self, root_cause: str, action_type: str, success: bool):
        key = (root_cause, action_type)
        entry = self._stats.setdefault(key, {"success": 0, "total": 0})
        entry["total"] += 1
        if success:
            entry["success"] += 1

    def success_rate(self, root_cause: str, action_type: str) -> float:
        entry = self._stats.get((root_cause, action_type))
        if not entry or entry["total"] == 0:
            return 0.0
        return entry["success"] / entry["total"]

    def load_from_db(self, session):
        """Rebuild stats from the RecoveryAction table (call at startup)."""
        from app.models import Payment, RecoveryAction, ActionStatus

        self._stats = {}
        rows = (
            session.query(Payment.root_cause, RecoveryAction.action_type, RecoveryAction.status)
            .join(RecoveryAction, RecoveryAction.payment_id == Payment.id)
            .all()
        )
        for root_cause, action_type, status in rows:
            if root_cause is None:
                continue
            self.record(root_cause, action_type, success=(status == ActionStatus.SUCCESS.value))


# Module-level singleton so recovery.py can share one tracker across a batch run.
stats_tracker = ActionStatsTracker()


if __name__ == "__main__":
    # quick smoke test: python -m app.policy
    scenarios = [
        (RootCause.TIMEOUT.value, {"attempt_count": 0}),
        (RootCause.TIMEOUT.value, {"attempt_count": 3}),
        (RootCause.INSUFFICIENT_FUNDS.value, {"attempt_count": 0}),
        (RootCause.INVALID_CARD.value, {"attempt_count": 0}),
        (RootCause.AUTH_FAILURE.value, {"attempt_count": 0}),
        (RootCause.AUTH_FAILURE.value, {"attempt_count": 1}),
        (RootCause.RISK_BLOCK.value, {"attempt_count": 0}),
        (RootCause.OTHER.value, {"attempt_count": 0}),
    ]
    for root_cause, ctx in scenarios:
        plan = choose_action(root_cause, ctx)
        print(f"{root_cause:20s} attempt={ctx['attempt_count']} -> {plan['action_type']:10s} | {plan['reason']}")
