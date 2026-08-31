"""
scripts/run_recovery_batch.py

The core demo script. Runs the recovery agent over every pending failed
payment, repeating rounds until the queue converges (nothing left that
can still be retried), then prints + saves a metrics report:

    Total failed amount, recovered amount, recovery rate,
    breakdown by root_cause, breakdown by action_type.

Run: python scripts/run_recovery_batch.py
Options: --max-rounds N   (default 10, safety cap so a policy bug can't loop forever)
         --limit N        (only pull N pending payments per round)
         --json-out PATH  (default reports/recovery_summary_<timestamp>.json)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db import get_session, init_db
from app.models import Payment, RecoveryAction
from app.recovery import run_recovery_batch


def run_until_convergence(max_rounds: int = 10, limit: int = None) -> list:
    """Repeats run_recovery_batch() until a round processes 0 payments."""
    all_results = []
    for round_num in range(1, max_rounds + 1):
        round_results = run_recovery_batch(limit=limit)
        print(f"Round {round_num}: processed {len(round_results)} payment(s)")
        all_results.extend(round_results)
        if not round_results:
            break
    else:
        print(f"WARNING: hit max_rounds={max_rounds} without full convergence. "
              f"Some payments may still be retryable — this is a safety cap, not a bug.")
    return all_results


def compute_metrics() -> dict:
    """
    Computes metrics from the current DB state rather than just the
    in-memory results list, so this also works if you re-run the script
    against a DB that already has some history in it.
    """
    with get_session() as session:
        payments = session.query(Payment).all()

        total_failed_amount = sum(p.amount for p in payments)
        recovered_amount = sum(p.recovered_amount for p in payments if p.is_recovered)
        recovery_rate = (recovered_amount / total_failed_amount) if total_failed_amount else 0.0

        # breakdown by root cause
        by_cause = {}
        for p in payments:
            cause = p.root_cause or "unclassified"
            entry = by_cause.setdefault(cause, {"count": 0, "total_amount": 0, "recovered_amount": 0})
            entry["count"] += 1
            entry["total_amount"] += p.amount
            if p.is_recovered:
                entry["recovered_amount"] += p.recovered_amount
        for cause, entry in by_cause.items():
            entry["recovery_rate"] = (
                entry["recovered_amount"] / entry["total_amount"] if entry["total_amount"] else 0.0
            )

        # breakdown by action type (success rate per action)
        actions = session.query(RecoveryAction).all()
        by_action = {}
        for a in actions:
            entry = by_action.setdefault(a.action_type, {"count": 0, "success": 0})
            entry["count"] += 1
            if a.status == "success":
                entry["success"] += 1
        for action_type, entry in by_action.items():
            entry["success_rate"] = entry["success"] / entry["count"] if entry["count"] else 0.0

        manual_review_count = sum(1 for p in payments if p.needs_manual_review)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_payments": len(payments),
            "total_failed_amount": total_failed_amount,
            "recovered_amount": recovered_amount,
            "recovery_rate": recovery_rate,
            "needs_manual_review_count": manual_review_count,
            "by_root_cause": by_cause,
            "by_action_type": by_action,
        }


def _fmt_amount(paise: int) -> str:
    return f"Rs.{paise / 100:,.2f}"


def print_report(metrics: dict):
    print()
    print("=" * 60)
    print("RECOVERY BATCH SUMMARY")
    print("=" * 60)
    print(f"Processed {metrics['total_payments']} failed payments, "
          f"recovered {metrics['recovery_rate']:.1%} "
          f"({_fmt_amount(metrics['recovered_amount'])} out of {_fmt_amount(metrics['total_failed_amount'])})")
    print(f"Flagged for manual review: {metrics['needs_manual_review_count']}")
    print()

    print("-- Breakdown by root cause --")
    for cause, entry in sorted(metrics["by_root_cause"].items()):
        print(f"  {cause:20s} count={entry['count']:3d}  "
              f"recovered={_fmt_amount(entry['recovered_amount']):>15s} / {_fmt_amount(entry['total_amount']):>15s}  "
              f"({entry['recovery_rate']:.1%})")
    print()

    print("-- Breakdown by action type --")
    for action, entry in sorted(metrics["by_action_type"].items()):
        print(f"  {action:12s} attempts={entry['count']:3d}  success_rate={entry['success_rate']:.1%}")
    print("=" * 60)


def save_report(metrics: dict, json_out: str = None) -> str:
    json_out = json_out or os.path.join(
        os.path.dirname(__file__), "..", "reports",
        f"recovery_summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(metrics, f, indent=2)
    return json_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the recovery agent over all pending failed payments")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    init_db()
    run_until_convergence(max_rounds=args.max_rounds, limit=args.limit)

    metrics = compute_metrics()
    print_report(metrics)
    saved_path = save_report(metrics, args.json_out)
    print(f"\nSaved JSON report to: {saved_path}")
