"""
app/ingestion.py

Ingestion layer: gets payment events INTO your Payments table.

We implement Option A (polling) as the primary path — simpler, and
enough for the Buildathon. Two entry points:

  1. poll_payments()          -> pulls REAL recent payments from your
                                  Razorpay test account via the API.
  2. ingest_synthetic_batch() -> generates and stores SYNTHETIC failed
                                  payments (via razorpay_client.simulate_payment_failure)
                                  so you have bulk data to run the rest
                                  of the pipeline against without
                                  needing hundreds of real failed
                                  payments in your test account.

Both funnel through the same _upsert_payment_record() so downstream
code (classifier, policy, recovery) never has to care where a payment
record originally came from.

Why polling over webhooks for now: webhooks need a publicly reachable
HTTPS endpoint (ngrok/deployed server) + signature verification, which
is extra infra for a Buildathon demo. Polling list_recent_payments()
on a timer gets you the same data with a `python -m app.ingestion`
call. In production you'd switch to webhooks because polling has
built-in lag (you only see a failure on your next poll cycle) and
wastes API calls when nothing changed — but the upsert logic below is
identical either way, so swapping later is a small, contained change
(just add a FastAPI route that calls _upsert_payment_record with the
webhook payload instead of a polled one).
"""

import argparse
import time

from app.db import get_session, init_db
from app.models import Payment, PaymentStatus
from app.razorpay_client import RazorpayClient


def _upsert_payment_record(session, raw: dict) -> Payment:
    """
    Insert a new Payment row, or update an existing one if we've already
    seen this razorpay_payment_id (idempotent — safe to call repeatedly
    on the same payment as its status changes, e.g. authorized -> failed).

    `raw` is a dict shaped like Razorpay's payment entity (works for both
    real API responses and our synthetic simulate_payment_failure() output
    since we kept the field names aligned).
    """
    payment_id = raw["id"]

    existing = (
        session.query(Payment)
        .filter(Payment.razorpay_payment_id == payment_id)
        .one_or_none()
    )

    fields = dict(
        order_id=raw.get("order_id"),
        amount=raw["amount"],
        currency=raw.get("currency", "INR"),
        status=raw.get("status", "failed"),
        failure_code=raw.get("error_code"),
        failure_description=raw.get("error_description"),
        method=raw.get("method"),
        customer_id=raw.get("customer_id"),
    )

    if existing:
        for key, value in fields.items():
            setattr(existing, key, value)
        session.commit()
        return existing

    payment = Payment(razorpay_payment_id=payment_id, **fields)
    session.add(payment)
    session.commit()
    session.refresh(payment)
    return payment


def poll_payments(client: RazorpayClient = None, limit: int = 50) -> list:
    """
    Pulls the last `limit` real payments from your Razorpay test account
    and upserts each into the Payments table. Returns the list of
    Payment rows touched.

    Run standalone: python -m app.ingestion --poll
    """
    client = client or RazorpayClient()
    raw_payments = client.list_recent_payments(limit=limit)

    touched = []
    with get_session() as session:
        for raw in raw_payments:
            touched.append(_upsert_payment_record(session, raw))

    print(f"[poll_payments] fetched={len(raw_payments)} upserted={len(touched)}")
    return touched


def ingest_synthetic_batch(scenarios: dict, client: RazorpayClient = None) -> list:
    """
    scenarios: dict like {"timeout": 40, "insufficient_funds": 60, ...}
    mapping scenario name -> how many synthetic failed payments to generate.

    Generates and stores them so you have realistic bulk data to drive
    Steps 6/7 without needing hundreds of real failed test payments.

    Run standalone: python -m app.ingestion --synthetic
    """
    client = client or RazorpayClient()
    touched = []

    with get_session() as session:
        for scenario, count in scenarios.items():
            for _ in range(count):
                raw = client.simulate_payment_failure(scenario)
                touched.append(_upsert_payment_record(session, raw))

    print(f"[ingest_synthetic_batch] inserted {len(touched)} synthetic failed payments")
    return touched


# ----------------------------------------------------------------------
# Step 6: Failure detection
# ----------------------------------------------------------------------
def get_pending_failed_payments(session, limit: int = None) -> list:
    """
    Returns Payment rows that are failed AND haven't been picked up by the
    recovery pipeline yet. This is what classifier.py / policy.py /
    recovery.py will loop over.

    Doesn't mark anything as processed here — that happens once you've
    actually classified + acted on the payment (do it in recovery.py after
    a successful classify+act cycle, so a crash mid-processing doesn't
    silently drop the payment).
    """
    query = (
        session.query(Payment)
        .filter(Payment.status == PaymentStatus.FAILED.value)
        .filter(Payment.processed_for_recovery.is_(False))
        .order_by(Payment.created_at.asc())
    )
    if limit:
        query = query.limit(limit)
    return query.all()


def mark_processed(session, payment: Payment):
    """Call this once recovery.py has finished acting on a payment."""
    payment.processed_for_recovery = True
    session.commit()


def poll_loop(interval_seconds: int = 30):
    """A simple cron-like loop. Ctrl+C to stop. For real polling deployments."""
    client = RazorpayClient()
    print(f"[poll_loop] polling every {interval_seconds}s. Ctrl+C to stop.")
    while True:
        poll_payments(client=client)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    init_db()
    parser = argparse.ArgumentParser(description="Ingestion CLI")
    parser.add_argument("--poll", action="store_true", help="One-off poll of recent real payments")
    parser.add_argument("--loop", type=int, default=None, help="Poll continuously every N seconds")
    parser.add_argument("--synthetic", action="store_true", help="Generate a default synthetic batch")
    args = parser.parse_args()

    if args.loop:
        poll_loop(interval_seconds=args.loop)
    elif args.synthetic:
        default_scenarios = {
            "timeout": 15,
            "insufficient_funds": 15,
            "invalid_card": 15,
            "expired_card": 10,
            "auth_failure": 10,
            "risk_block": 5,
        }
        ingest_synthetic_batch(default_scenarios)
    else:
        poll_payments()
