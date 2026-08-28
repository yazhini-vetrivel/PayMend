"""
app/razorpay_client.py

Thin wrapper around Razorpay's REST API (Orders + Payments) using `requests`
and Basic Auth with your test-mode key_id / key_secret.

All Razorpay-facing HTTP calls live here so the rest of the app never
touches `requests` or auth directly.
"""

import os
import time
import random
import uuid
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

RAZORPAY_BASE_URL = "https://api.razorpay.com/v1"
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    raise RuntimeError(
        "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET missing. "
        "Add them to your .env file (use your TEST mode keys)."
    )


class RazorpayError(Exception):
    """Raised when the Razorpay API returns a non-2xx response."""

    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.payload = payload
        super().__init__(f"Razorpay API error [{status_code}]: {payload}")


class RazorpayClient:
    def __init__(self, key_id: str = None, key_secret: str = None, timeout: int = 15):
        self.auth = (key_id or RAZORPAY_KEY_ID, key_secret or RAZORPAY_KEY_SECRET)
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal request helper
    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, params=None, json_body=None):
        url = f"{RAZORPAY_BASE_URL}{path}"
        try:
            resp = requests.request(
                method=method,
                url=url,
                auth=self.auth,
                params=params,
                json=json_body,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            # network-level failure (timeout, DNS, connection refused, etc.)
            raise RazorpayError(status_code=None, payload=str(e))

        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except ValueError:
                payload = resp.text
            raise RazorpayError(status_code=resp.status_code, payload=payload)

        return resp.json()

    # ------------------------------------------------------------------
    # Orders API
    # ------------------------------------------------------------------
    def create_order(self, amount: int, currency: str = "INR", customer_id: str = None,
                      receipt: str = None, notes: dict = None):
        """
        amount is in the smallest currency unit (paise for INR).
        e.g. amount=50000 -> ₹500.00
        """
        body = {
            "amount": amount,
            "currency": currency,
            "receipt": receipt or f"rcpt_{uuid.uuid4().hex[:12]}",
            "payment_capture": 1,
        }
        if notes:
            body["notes"] = notes
        if customer_id:
            body.setdefault("notes", {})["customer_id"] = customer_id

        return self._request("POST", "/orders", json_body=body)

    def fetch_order(self, order_id: str):
        return self._request("GET", f"/orders/{order_id}")

    def fetch_order_payments(self, order_id: str):
        return self._request("GET", f"/orders/{order_id}/payments")

    # ------------------------------------------------------------------
    # Payments API
    # ------------------------------------------------------------------
    def fetch_payment(self, payment_id: str):
        return self._request("GET", f"/payments/{payment_id}")

    def list_recent_payments(self, limit: int = 50, skip: int = 0, from_ts: int = None, to_ts: int = None):
        """
        Fetch last N payments across the account (used for polling ingestion).
        limit is capped at 100 by Razorpay's API.
        """
        params = {"count": min(limit, 100), "skip": skip}
        if from_ts:
            params["from"] = from_ts
        if to_ts:
            params["to"] = to_ts

        data = self._request("GET", "/payments", params=params)
        return data.get("items", [])

    def capture_payment(self, payment_id: str, amount: int, currency: str = "INR"):
        """Manually capture an authorized payment (needed if payment_capture=0)."""
        body = {"amount": amount, "currency": currency}
        return self._request("POST", f"/payments/{payment_id}/capture", json_body=body)

    # ------------------------------------------------------------------
    # Retry: create a fresh order for the same logical purchase and
    # return it so the caller can drive the customer through checkout
    # (or, in test mode, through the simulated flow below).
    # ------------------------------------------------------------------
    def retry_as_new_order(self, original_order: dict, notes: dict = None):
        merged_notes = dict(original_order.get("notes") or {})
        if notes:
            merged_notes.update(notes)
        merged_notes["retry_of_order_id"] = original_order["id"]

        return self.create_order(
            amount=original_order["amount"],
            currency=original_order["currency"],
            notes=merged_notes,
        )

    # ------------------------------------------------------------------
    # TEST-MODE SIMULATION
    # ------------------------------------------------------------------
    # Razorpay has no server-side "force this payment to fail" endpoint —
    # real failed payments only happen via Checkout using their published
    # test card numbers (e.g. 4000000000000002 = card declined). This
    # method exists for two purposes:
    #
    #  1. Generate synthetic-but-realistic payment records (same shape as
    #     Razorpay's actual payment object) for bulk classifier training
    #     data, when you don't want to click through Checkout 300 times.
    #
    #  2. Optionally create a *real* order via the API that you then complete
    #     manually through Checkout.js using a known test failure card, so
    #     you get an authentic failed payment in your dashboard + webhook.
    # ------------------------------------------------------------------

    # Razorpay published test failure scenarios (test mode only).
    # See: Razorpay Docs > Payments > Test Card Numbers
    TEST_FAILURE_SCENARIOS = {
        "insufficient_funds": {
            "card": "4000000000000002",
            "failure_code": "BAD_REQUEST_ERROR",
            "failure_description": "Insufficient funds in the account.",
        },
        "invalid_card": {
            "card": "4000000000000010",
            "failure_code": "BAD_REQUEST_ERROR",
            "failure_description": "The card number is invalid.",
        },
        "expired_card": {
            "card": "4000000000000069",
            "failure_code": "BAD_REQUEST_ERROR",
            "failure_description": "The card has expired.",
        },
        "auth_failure": {
            "card": "4000000000000119",
            "failure_code": "GATEWAY_ERROR",
            "failure_description": "3D Secure authentication failed.",
        },
        "timeout": {
            "card": "4000000000000101",
            "failure_code": "GATEWAY_ERROR",
            "failure_description": "The card issuing bank server timed out.",
        },
        "risk_block": {
            "card": "4100000000000019",
            "failure_code": "GATEWAY_ERROR",
            "failure_description": "Payment flagged by risk engine.",
        },
    }

    def create_test_order_for_scenario(self, scenario: str, amount: int, currency: str = "INR"):
        """
        Creates a REAL order via the API tagged with the scenario name in
        `notes`, so you know which test card to use on Checkout to
        reproduce this failure. Returns the order + the card to use.
        """
        if scenario not in self.TEST_FAILURE_SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Options: {list(self.TEST_FAILURE_SCENARIOS)}")

        order = self.create_order(amount=amount, currency=currency,
                                   notes={"simulated_scenario": scenario})
        return {
            "order": order,
            "test_card": self.TEST_FAILURE_SCENARIOS[scenario]["card"],
            "expected_failure_code": self.TEST_FAILURE_SCENARIOS[scenario]["failure_code"],
            "expected_failure_description": self.TEST_FAILURE_SCENARIOS[scenario]["failure_description"],
        }

    def simulate_payment_failure(self, scenario: str, amount: int = None, currency: str = "INR",
                                  method: str = "card", customer_id: str = None):
        """
        Generates a SYNTHETIC payment record (not a live API call) shaped
        exactly like Razorpay's payment object, for bulk classifier
        training / demoing the pipeline without clicking through Checkout.

        Use create_test_order_for_scenario() instead when you want a real,
        auditable failed payment tied to an actual Razorpay order.
        """
        if scenario not in self.TEST_FAILURE_SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Options: {list(self.TEST_FAILURE_SCENARIOS)}")

        spec = self.TEST_FAILURE_SCENARIOS[scenario]
        amount = amount or random.choice([9900, 19900, 49900, 99900, 149900])
        now = int(time.time())

        return {
            "id": f"pay_SIM{uuid.uuid4().hex[:14]}",
            "order_id": f"order_SIM{uuid.uuid4().hex[:14]}",
            "entity": "payment",
            "amount": amount,
            "currency": currency,
            "status": "failed",
            "method": method,
            "captured": False,
            "error_code": spec["failure_code"],
            "error_description": spec["failure_description"],
            "error_reason": scenario,
            "customer_id": customer_id or f"cust_SIM{uuid.uuid4().hex[:10]}",
            "created_at": now,
            "_simulated": True,
            "_scenario": scenario,
        }


if __name__ == "__main__":
    # quick manual smoke test — run: python -m app.razorpay_client
    client = RazorpayClient()
    fake = client.simulate_payment_failure("timeout", amount=50000)
    print("Synthetic failed payment:")
    print(fake)
