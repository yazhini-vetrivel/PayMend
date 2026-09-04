"""
app/policy.py
...(unchanged docstring)...
"""

from app.models import ActionType, RootCause

MAX_RETRIES_TIMEOUT = 3
MAX_RETRIES_AUTH_FAILURE = 1
INSUFFICIENT_FUNDS_RETRY_DELAY_HOURS = 24

# ----------------------------------------------------------------------
# NEW: follow-up simulation config.
#
# A NOTIFY action isn't a dead end — in real life, some % of customers
# who get notified actually act on it (update their card, top up their
# balance). This project runs in test mode with no real time delay, so
# process_followups() in recovery.py uses these probabilities to
# simulate "what happens after the notify" in the SAME batch run,
# instead of leaving every notified payment stuck at 0% forever.
#
# root_cause NOT listed here (risk_block) intentionally has no
# follow-up simulation: escalation requires an actual human decision,
# so it must never self-resolve. That 0% is correct and permanent.
# ----------------------------------------------------------------------
FOLLOWUP_SIMULATION_PROBABILITY = {
    RootCause.INSUFFICIENT_FUNDS.value: 0.35,  # balance topped up before delayed retry
    RootCause.EXPIRED_CARD.value: 0.45,        # customer updates card, pays via new order
    RootCause.INVALID_CARD.value: 0.40,        # customer corrects card, pays via new order
    RootCause.AUTH_FAILURE.value: 0.30,        # customer completes 3DS manually after notify
    # RootCause.RISK_BLOCK.value: intentionally omitted
}


def get_followup_probability(root_cause: str) -> float:
    return FOLLOWUP_SIMULATION_PROBABILITY.get(root_cause, 0.0)


def choose_action(root_cause: str, context: dict) -> dict:
    attempt_count = context.get("attempt_count", 0)

    if root_cause == RootCause.TIMEOUT.value:
        if attempt_count < MAX_RETRIES_TIMEOUT:
            backoff_seconds = _exponential_backoff(attempt_count)
            return {
                "action_type": ActionType.RETRY.value,
                "params": {"delay_seconds": backoff_seconds, "max_retries": MAX_RETRIES_TIMEOUT},
                "reason": (
                    f"Gateway/network timeout — transient, so retry "
                    f"(attempt {attempt_count + 1}/{MAX_RETRIES_TIMEOUT}) "
                    f"after {backoff_seconds}s backoff."
                ),
            }
        return _escalate(f"Timeout persisted after {MAX_RETRIES_TIMEOUT} retries — needs manual review.")

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
        return _escalate("Insufficient funds persisted after notify + delayed retry window.")

    if root_cause in (RootCause.INVALID_CARD.value, RootCause.EXPIRED_CARD.value):
        if attempt_count == 0:
            return {
                "action_type": ActionType.NOTIFY.value,
                "params": {"channel": "email", "template": "update_card"},
                "reason": (
                    f"{root_cause} — retrying the same card will fail again by definition. "
                    "Ask customer to update card details instead."
                ),
            }
        return _escalate(f"{root_cause} — customer didn't update card after notify. Escalating.")

    if root_cause == RootCause.AUTH_FAILURE.value:
        if attempt_count < MAX_RETRIES_AUTH_FAILURE:
            return {
                "action_type": ActionType.RETRY.value,
                "params": {"delay_seconds": 0, "max_retries": MAX_RETRIES_AUTH_FAILURE, "force_3ds": True},
                "reason": "3DS/auth failure can be a one-off glitch — retry once with proper 3DS flow.",
            }
        return {
            "action_type": ActionType.NOTIFY.value,
            "params": {"channel": "email", "template": "auth_failure_retry_failed"},
            "reason": "Auth failure persisted after one retry — notify customer to complete authentication manually.",
        }

    if root_cause == RootCause.RISK_BLOCK.value:
        return _escalate("Flagged by risk/fraud engine — auto-retry could look like abuse. Manual review required.")

    return _escalate(f"Root cause '{root_cause}' has no defined policy — routing to manual review rather than guessing.")


def _escalate(reason: str) -> dict:
    return {"action_type": ActionType.ESCALATE.value, "params": {}, "reason": reason}


def _exponential_backoff(attempt_count: int, base_seconds: int = 30) -> int:
    return base_seconds * (2 ** attempt_count)


class ActionStatsTracker:
    def __init__(self):
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
