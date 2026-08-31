"""
app/recovery.py

The recovery executor — ties classifier.py + policy.py + razorpay_client.py
together into one closed-loop function per payment:

    execute_recovery(payment_id) -> summary dict

    Fetch payment -> classify_failure() -> choose_action() -> execute it
    -> record RecoveryAction -> write AuditLog -> update Payment status.

Also provides run_recovery_batch() which loops over every pending failed
payment (Step 6's get_pending_failed_payments) and calls execute_recovery
on each — this is what scripts/run_recovery_batch.py will call to produce
the batch metrics (Step 7 in your original plan / metrics step).
"""

import json
import random
from datetime import datetime, timezone

from app.db import get_session
from app.models import Payment, RecoveryAction, AuditLog, ActionType, ActionStatus, PaymentStatus
from app.classifier import classify_failure
from app.policy import choose_action
from app.ingestion import mark_processed


# ----------------------------------------------------------------------
# TEST-MODE outcome simulation for RETRY actions.
# ----------------------------------------------------------------------
# Real production code would call razorpay_client.retry_as_new_order()
# and then wait for the actual webhook/poll result. In test mode there's
# no real card being charged on retry, so we simulate an outcome with a
# success probability that varies by root cause — timeouts are mostly
# transient (high retry success), risk_block/invalid_card would never be
# retried by policy.py anyway so they don't need an entry here.
RETRY_SUCCESS_PROBABILITY = {
    "timeout": 0.70,
    "auth_failure": 0.55,
    "insufficient_funds": 0.20,  # only reached via the rare delayed-retry path
}


def _simulate_retry_outcome(root_cause: str) -> bool:
    p = RETRY_SUCCESS_PROBABILITY.get(root_cause, 0.30)
    return random.random() < p


def _mock_send_notification(payment: Payment, template: str, channel: str) -> dict:
    """
    Stand-in for a real email/SMS/WhatsApp API call. Logs what would have
    been sent and always "succeeds" (delivery isn't the interesting part
    for this project — the decision to notify, and what you say, is).
    """
    message = {
        "insufficient_funds": "Your payment didn't go through due to insufficient funds. We'll retry in 24h, or you can pay now via another method.",
        "update_card": "Your payment failed because of a card issue. Please update your card details to complete the purchase.",
        "auth_failure_retry_failed": "We couldn't verify your card (3D Secure). Please try again and complete the authentication step.",
    }.get(template, "Your recent payment failed. Please try again.")

    print(f"[MOCK NOTIFY] channel={channel} to customer={payment.customer_id} :: {message}")
    return {"delivered": True, "channel": channel, "template": template, "message": message}


def _log_audit(session, payment_id, event_type, root_cause=None, action_type=None, details=None):
    entry = AuditLog(
        payment_id=payment_id,
        event_type=event_type,
        root_cause=root_cause,
        action_type=action_type,
        details=json.dumps(details or {}),
    )
    session.add(entry)
    session.commit()
    return entry


def _acquire_lock(session, payment_id: int) -> bool:
    """
    Atomic compare-and-swap: only succeeds if recovery_in_progress was
    False, and flips it to True in the same UPDATE statement. This is
    safe under concurrent callers (unlike 'read the flag, then write it'
    in separate steps, which is exactly the race that caused three
    inconsistent RecoveryAction rows for one payment before this fix).
    Returns True if the lock was acquired.
    """
    rows_updated = (
        session.query(Payment)
        .filter(Payment.id == payment_id, Payment.recovery_in_progress.is_(False))
        .update({Payment.recovery_in_progress: True})
    )
    session.commit()
    return rows_updated > 0


def _release_lock(session, payment_id: int):
    session.query(Payment).filter(Payment.id == payment_id).update({Payment.recovery_in_progress: False})
    session.commit()


def execute_recovery(payment_id: int) -> dict:
    """
    payment_id: the internal Payment.id (primary key), not the Razorpay
    payment_id string.

    Returns a summary dict describing what happened — useful for
    run_recovery_batch()'s metrics rollup.
    """
    with get_session() as lock_session:
        if not _acquire_lock(lock_session, payment_id):
            return {"payment_id": payment_id, "skipped": True, "reason": "recovery already in progress for this payment"}

    try:
        return _execute_recovery_locked(payment_id)
    finally:
        with get_session() as unlock_session:
            _release_lock(unlock_session, payment_id)


def _execute_recovery_locked(payment_id: int) -> dict:
    """Original execute_recovery body — now only ever runs while the lock is held."""
    with get_session() as session:
        payment = session.query(Payment).filter(Payment.id == payment_id).one_or_none()
        if payment is None:
            raise ValueError(f"No payment found with id={payment_id}")

        _log_audit(session, payment.id, "payment_received", details={
            "razorpay_payment_id": payment.razorpay_payment_id,
            "amount": payment.amount,
            "status": payment.status,
        })

        if payment.status != PaymentStatus.FAILED.value:
            _log_audit(session, payment.id, "skipped_not_failed", details={"status": payment.status})
            return {"payment_id": payment.id, "skipped": True, "reason": "not in failed status"}

        # ------------------------------------------------------------
        # 1. Classify
        # ------------------------------------------------------------
        classification = classify_failure(payment)
        payment.root_cause = classification["root_cause"]
        payment.root_cause_confidence = classification["confidence"]
        payment.root_cause_method = classification["method"]
        session.commit()

        _log_audit(
            session, payment.id, "root_cause_assigned",
            root_cause=classification["root_cause"],
            details=classification,
        )

        # ------------------------------------------------------------
        # 2. Choose action
        # ------------------------------------------------------------
        attempt_count = len(payment.recovery_actions)  # how many actions already taken on this payment
        context = {
            "attempt_count": attempt_count,
            "method": payment.method,
            "amount": payment.amount,
            "customer_id": payment.customer_id,
        }
        action_plan = choose_action(classification["root_cause"], context)

        _log_audit(
            session, payment.id, "action_chosen",
            root_cause=classification["root_cause"],
            action_type=action_plan["action_type"],
            details={"reason": action_plan["reason"], "params": action_plan["params"], "context": context},
        )

        # ------------------------------------------------------------
        # 3. Idempotency guard — don't execute the same (payment, action,
        #    attempt) twice, e.g. if this function gets called again for
        #    a payment that's already mid-processing (duplicate webhook,
        #    overlapping batch run, etc.)
        # ------------------------------------------------------------
        attempt_number = attempt_count + 1
        idempotency_key = f"{payment.id}:{action_plan['action_type']}:{attempt_number}"

        existing_action = (
            session.query(RecoveryAction)
            .filter(RecoveryAction.idempotency_key == idempotency_key)
            .one_or_none()
        )
        if existing_action:
            _log_audit(session, payment.id, "duplicate_action_skipped", details={"idempotency_key": idempotency_key})
            return {"payment_id": payment.id, "skipped": True, "reason": "duplicate idempotency key"}

        recovery_action = RecoveryAction(
            payment_id=payment.id,
            action_type=action_plan["action_type"],
            attempt_number=attempt_number,
            status=ActionStatus.PENDING.value,
            idempotency_key=idempotency_key,
        )
        session.add(recovery_action)
        session.commit()
        session.refresh(recovery_action)

        # ------------------------------------------------------------
        # 4. Execute
        # ------------------------------------------------------------
        result_details = {}

        if action_plan["action_type"] == ActionType.RETRY.value:
            succeeded = _simulate_retry_outcome(classification["root_cause"])
            result_details = {
                "simulated": True,
                "delay_seconds": action_plan["params"].get("delay_seconds", 0),
                "outcome": "success" if succeeded else "failed",
            }
            if succeeded:
                recovery_action.status = ActionStatus.SUCCESS.value
                payment.status = PaymentStatus.RECOVERED.value
                payment.is_recovered = True
                payment.recovered_amount = payment.amount
            else:
                recovery_action.status = ActionStatus.FAILED.value
                max_retries = action_plan["params"].get("max_retries", 1)
                if attempt_number >= max_retries:
                    payment.status = PaymentStatus.RECOVERY_EXHAUSTED.value

        elif action_plan["action_type"] == ActionType.NOTIFY.value:
            notify_result = _mock_send_notification(
                payment,
                template=action_plan["params"].get("template"),
                channel=action_plan["params"].get("channel", "email"),
            )
            result_details = notify_result
            recovery_action.status = ActionStatus.SUCCESS.value  # "notify" succeeds if it was sent
            if action_plan["params"].get("schedule_retry_after_hours"):
                result_details["next_retry_at"] = _future_timestamp(
                    action_plan["params"]["schedule_retry_after_hours"]
                )
                # Leave processed_for_recovery False so it's picked up again
                # after the delay window in a later batch run, instead of
                # marking it done here.

        elif action_plan["action_type"] == ActionType.ESCALATE.value:
            payment.needs_manual_review = True
            recovery_action.status = ActionStatus.SUCCESS.value  # "escalate" succeeds if it was flagged
            result_details = {"flagged_for_manual_review": True}

        else:  # NO_ACTION / unknown — shouldn't happen given policy.py's coverage, but stay safe
            recovery_action.status = ActionStatus.SKIPPED.value
            result_details = {"note": "no action taken"}

        recovery_action.result_details = json.dumps(result_details)
        recovery_action.completed_at = datetime.now(timezone.utc)
        session.commit()

        _log_audit(
            session, payment.id, "action_executed",
            root_cause=classification["root_cause"],
            action_type=action_plan["action_type"],
            details={"result": result_details, "recovery_action_status": recovery_action.status},
        )

        # ------------------------------------------------------------
        # 5. Mark processed (Step 6) — unless we deliberately left a
        #    delayed-retry window open above.
        # ------------------------------------------------------------
        left_open_for_delayed_retry = (
            action_plan["action_type"] == ActionType.NOTIFY.value
            and action_plan["params"].get("schedule_retry_after_hours")
        )
        left_open_for_more_retries = (
            action_plan["action_type"] == ActionType.RETRY.value
            and recovery_action.status == ActionStatus.FAILED.value
            and payment.status != PaymentStatus.RECOVERY_EXHAUSTED.value
        )
        if not (left_open_for_delayed_retry or left_open_for_more_retries):
            mark_processed(session, payment)

        _log_audit(session, payment.id, "recovery_cycle_complete", details={
            "final_payment_status": payment.status,
            "is_recovered": payment.is_recovered,
        })

        return {
            "payment_id": payment.id,
            "razorpay_payment_id": payment.razorpay_payment_id,
            "root_cause": classification["root_cause"],
            "action_type": action_plan["action_type"],
            "action_status": recovery_action.status,
            "final_payment_status": payment.status,
            "recovered_amount": payment.recovered_amount,
        }


def _future_timestamp(hours_from_now: int) -> str:
    from datetime import timedelta
    return (datetime.now(timezone.utc) + timedelta(hours=hours_from_now)).isoformat()


def run_recovery_batch(limit: int = None) -> list:
    """
    Loops over every pending failed payment and runs execute_recovery on
    each. Returns the list of summary dicts — scripts/run_recovery_batch.py
    uses this to compute the Step 7 metrics (total at risk, recovered,
    recovery rate, breakdown by root cause / action type).
    """
    from app.ingestion import get_pending_failed_payments

    with get_session() as session:
        pending = get_pending_failed_payments(session, limit=limit)
        payment_ids = [p.id for p in pending]  # snapshot ids before the session closes

    results = []
    for pid in payment_ids:
        results.append(execute_recovery(pid))
    return results


if __name__ == "__main__":
    # quick smoke test: python -m app.recovery
    from app.db import init_db
    init_db()

    with get_session() as session:
        one_pending = session.query(Payment).filter(
            Payment.status == PaymentStatus.FAILED.value,
            Payment.processed_for_recovery.is_(False),
        ).first()
        target_id = one_pending.id if one_pending else None

    if target_id is None:
        print("No pending failed payments found. Run: python -m app.ingestion --synthetic")
    else:
        result = execute_recovery(target_id)
        print(json.dumps(result, indent=2))
