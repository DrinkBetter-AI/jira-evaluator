"""Read-only client for the Medusa storefront's admin API: orders, nothing else.

The dashboard reports how the shop is doing beside how engineering is doing, so
it needs the order book - and only the order book. Every call here is a GET
against ``/admin/orders``; the key it uses is a full admin key because Medusa
issues no narrower one, which is exactly why this module has no other verb.
"""

from __future__ import annotations

import base64
import datetime as _dt
import os

import pandas as pd
import requests

DEFAULT_BASE_URL = "https://merchants.vinovoss.com"

# Medusa authenticates a secret API key as HTTP Basic with the key as username
# and an empty password; the bearer form its docs suggest is for JWT sessions
# and is rejected for an api key.
_KEY_ENV_VARS = ("MEDUSA_ADMIN_API_KEY", "MEDUSA_API_KEY")

# One page per hundred orders keeps a thirty-day window to two or three calls.
_PAGE_SIZE = 100
# A month of a growing shop, with headroom; a runaway loop would otherwise walk
# the whole order history a hundred rows at a time.
_MAX_PAGES = 40

# Medusa returns the order's line items whatever `fields` asks for, so the list
# is about the columns the metrics need, not about the response size.
_ORDER_FIELDS = "id,display_id,created_at,status,payment_status,total,currency_code"

# An order that was paid, including one since partly refunded: the refund is a
# correction to revenue already earned, not a sale that never happened.
PAID_PAYMENT_STATUSES = frozenset({"captured", "partially_refunded", "refunded"})

COLUMNS = [
    "id",
    "display_id",
    "created_at",
    "status",
    "payment_status",
    "total",
    "currency_code",
]


class MedusaConfigError(RuntimeError):
    """Raised when the CRM credential is missing or refused."""


def load_medusa_env() -> tuple[str, str] | None:
    """Return ``(api_key, base_url)`` from the environment, or ``None`` when unset."""
    key = ""
    for name in _KEY_ENV_VARS:
        key = os.getenv(name, "").strip()
        if key:
            break
    if not key:
        return None
    base = os.getenv("MEDUSA_ADMIN_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    return key, base.rstrip("/")


def _auth_header(api_key: str) -> dict[str, str]:
    token = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}", "Accept": "application/json"}


def fetch_orders(since: _dt.datetime, api_key: str, base_url: str) -> pd.DataFrame:
    """Every order created at or after ``since``, oldest first.

    Filtered server-side: asking for a month of a shop that has years of history
    behind it should cost a month of rows.
    """
    headers = _auth_header(api_key)
    url = f"{base_url}/admin/orders"
    rows: list[dict] = []
    offset = 0
    for _ in range(_MAX_PAGES):
        response = requests.get(
            url,
            params={
                "limit": _PAGE_SIZE,
                "offset": offset,
                "order": "created_at",
                "created_at[$gte]": since.astimezone(_dt.timezone.utc).isoformat(),
                "fields": _ORDER_FIELDS,
            },
            headers=headers,
            timeout=30,
        )
        if response.status_code in (401, 403):
            raise MedusaConfigError(
                f"The CRM refused the API key ({response.status_code}). "
                "Create a fresh secret key in Medusa and set MEDUSA_ADMIN_API_KEY."
            )
        response.raise_for_status()
        payload = response.json()
        page = payload.get("orders") or []
        rows.extend(page)
        offset += len(page)
        if not page or offset >= int(payload.get("count") or 0):
            break

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.DataFrame(rows)
    # Revenue is defined by payment state, so an order book without it is not a
    # smaller answer, it is a wrong one - every order would read as unpaid.
    if "payment_status" not in frame.columns:
        raise MedusaConfigError(
            "The CRM returned orders with no payment_status, so paid and unpaid "
            "orders cannot be told apart. Check the Medusa version behind "
            f"{base_url}."
        )
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[COLUMNS].copy()
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    frame["total"] = pd.to_numeric(frame["total"], errors="coerce").fillna(0.0)
    for column in ("status", "payment_status", "currency_code"):
        frame[column] = frame[column].fillna("").astype(str).str.strip().str.lower()
    return frame.sort_values("created_at").reset_index(drop=True)


__all__ = [
    "COLUMNS",
    "DEFAULT_BASE_URL",
    "MedusaConfigError",
    "PAID_PAYMENT_STATUSES",
    "fetch_orders",
    "load_medusa_env",
]
