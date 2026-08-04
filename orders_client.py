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
from dataclasses import dataclass

import pandas as pd
import requests

DEFAULT_BASE_URL = "https://merchants.vinovoss.com"

# Medusa authenticates a secret API key as HTTP Basic with the key as username
# and an empty password; the bearer form its docs suggest is for JWT sessions
# and is rejected for an api key.
_KEY_ENV_VARS = ("MEDUSA_ADMIN_API_KEY", "MEDUSA_API_KEY")

# One page per hundred orders keeps a thirty-day window to two or three calls.
_PAGE_SIZE = 100
# The first read covers a year, which is ~1,500 orders today, so this is a guard
# against a runaway loop rather than a budget: it has to stay well clear of the
# real volume, because everything past it is silently the oldest orders missing.
_MAX_PAGES = 200

# Medusa returns the order's line items whatever `fields` asks for, so the list
# is about the columns the metrics need, not about the response size.
_ORDER_FIELDS = (
    "id,display_id,created_at,updated_at,status,payment_status,total,"
    "refunded_total,currency_code"
)

# Re-reading a year of orders on every refresh costs seconds and gets slower
# every month the shop trades. Instead the book is kept and topped up: Medusa can
# filter on `updated_at`, which moves both when an order is placed and when it is
# later paid or cancelled, so one incremental read catches new sales and
# corrections to old ones alike. The overlap covers clock skew between this
# process and the CRM.
_SYNC_OVERLAP = _dt.timedelta(minutes=10)

# Line-item columns: what was bought, how many, and for how much.
ITEM_COLUMNS = [
    "order_id",
    "item_id",
    "created_at",
    "status",
    "payment_status",
    "currency_code",
    "title",
    "product_handle",
    "quantity",
    "revenue",
    "refunded",
]

# An order that was paid, including one since refunded: the refund is netted off
# the total rather than the order being struck out, so a wholly refunded sale
# contributes nothing and a half-refunded one contributes half.
PAID_PAYMENT_STATUSES = frozenset({"captured", "partially_refunded", "refunded"})

# Without any one of these the arithmetic is not merely incomplete, it is wrong:
# no total is $0 of revenue, no payment state is every order unpaid, no date is
# an empty window.
REQUIRED_FIELDS = ("created_at", "payment_status", "total")

COLUMNS = [
    "id",
    "display_id",
    "created_at",
    "updated_at",
    "status",
    "payment_status",
    "total",
    "refunded_total",
    "currency_code",
]


@dataclass(frozen=True)
class OrderBook:
    """Orders and their line items, plus when they were last read.

    Two frames rather than one: the tiles count orders, the wine and merchant
    tables count what was in them, and a shared order may hold wine from more
    than one merchant.
    """

    orders: pd.DataFrame
    items: pd.DataFrame
    window_days: int
    synced_at: _dt.datetime
    truncated: bool


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
    # The key travels in an Authorization header, so the host it is sent to has
    # to be deliberate and the connection encrypted: a mistyped variable would
    # otherwise hand an admin credential to whatever answers.
    if not base.lower().startswith("https://"):
        raise MedusaConfigError(
            "MEDUSA_ADMIN_URL must be an https:// address; the admin key is sent "
            "to it as a credential and will not be sent in the clear."
        )
    return key, base.rstrip("/")


def _auth_header(api_key: str) -> dict[str, str]:
    token = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}", "Accept": "application/json"}


def _fetch_pages(
    api_key: str, base_url: str, filters: dict[str, str]
) -> tuple[list[dict], bool]:
    """Every order matching ``filters``, newest first, and whether paging ran out."""
    headers = _auth_header(api_key)
    url = f"{base_url}/admin/orders"
    rows: list[dict] = []
    offset = 0
    truncated = False
    for page_number in range(_MAX_PAGES):
        response = requests.get(
            url,
            params={
                "limit": _PAGE_SIZE,
                "offset": offset,
                # Newest first, so a shop that outgrows the page ceiling loses the
                # oldest comparison window rather than the week being headlined.
                "order": "-created_at",
                "fields": _ORDER_FIELDS,
                **filters,
            },
            headers=headers,
            timeout=60,
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
        # `count` is the window's real size and settles it. Without it, a page
        # shorter than asked for is the end - though a deployment that caps the
        # page size below the ask and reports no count cannot be told apart from
        # a window that simply ended, which is why count is preferred.
        reported = int(payload.get("count") or 0)
        exhausted = (
            offset >= reported if reported else len(page) < _PAGE_SIZE
        )
        if not page or exhausted:
            break
        truncated = page_number == _MAX_PAGES - 1
    return rows, truncated


def _order_frame(rows: list[dict], base_url: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    # Offset paging over a list the shop is still adding to can hand back the
    # same order twice: a sale made mid-read pushes every row down a place.
    frame = pd.DataFrame(rows)
    if "id" in frame.columns:
        frame = frame.drop_duplicates(subset="id")
    # Every figure is defined by these three. Filling a missing one in would not
    # give a smaller answer, it would give a wrong one - a confident $0 for a
    # shop that took money - so the section refuses instead.
    missing = [name for name in REQUIRED_FIELDS if name not in frame.columns]
    if missing:
        raise MedusaConfigError(
            f"The CRM returned orders with no {', '.join(missing)}, so the order "
            f"figures cannot be trusted. Check the Medusa version behind {base_url}."
        )
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[COLUMNS].copy()
    for column in ("created_at", "updated_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    for column in ("total", "refunded_total"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in ("status", "payment_status", "currency_code"):
        frame[column] = frame[column].fillna("").astype(str).str.strip().str.lower()
    return frame.sort_values("created_at").reset_index(drop=True)


def _with_refunds(lines: list[dict], order: dict) -> list[dict]:
    """Spread an order's refund across its lines, in proportion to their value.

    A refund is recorded against the order, not against the bottle that went
    back, so the merchant who sold it cannot be known. Splitting it by value at
    least keeps a merchant's revenue from counting money the customer got back,
    and keeps the per-merchant total reconcilable with the revenue tile.
    """
    refunded = pd.to_numeric(order.get("refunded_total"), errors="coerce")
    refunded = 0.0 if pd.isna(refunded) else float(refunded)
    gross = sum(line["revenue"] for line in lines)
    if refunded <= 0 or gross <= 0:
        return lines
    for line in lines:
        line["refunded"] = round(
            min(refunded * line["revenue"] / gross, line["revenue"]), 2
        )
    return lines


def _item_frame(rows: list[dict]) -> pd.DataFrame:
    """One row per line item, carrying its order's date and payment state.

    Denormalised on purpose: every question asked of the items - what sold, whose
    wine it was, what it earned - is also filtered by when the order was placed
    and whether it stood, and a per-item frame answers those without a join.
    """
    records: list[dict] = []
    for order in rows:
        order_id = order.get("id")
        lines: list[dict] = []
        for position, item in enumerate(order.get("items") or []):
            quantity = pd.to_numeric(item.get("quantity"), errors="coerce")
            quantity = 0 if pd.isna(quantity) else int(quantity)
            unit_price = pd.to_numeric(item.get("unit_price"), errors="coerce")
            unit_price = 0.0 if pd.isna(unit_price) else float(unit_price)
            lines.append(
                {
                    "order_id": order_id,
                    # Identity per line, so paging that returns an order twice
                    # collapses to one copy while an order genuinely listing the
                    # same wine on two lines keeps both. Falls back to position
                    # when Medusa omits the id.
                    "item_id": str(item.get("id") or f"{order_id}#{position}"),
                    "created_at": order.get("created_at"),
                    "status": order.get("status"),
                    "payment_status": order.get("payment_status"),
                    "currency_code": order.get("currency_code"),
                    "title": item.get("product_title") or item.get("title") or "",
                    "product_handle": item.get("product_handle") or "",
                    "quantity": quantity,
                    "revenue": round(unit_price * quantity, 2),
                    "refunded": 0.0,
                }
            )
        records.extend(_with_refunds(lines, order))
    if not records:
        return pd.DataFrame(columns=ITEM_COLUMNS)
    frame = pd.DataFrame(records)[ITEM_COLUMNS]
    frame = frame.drop_duplicates(subset=["order_id", "item_id"])
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    for column in ("revenue", "refunded"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in ("status", "payment_status", "currency_code", "product_handle"):
        frame[column] = frame[column].fillna("").astype(str).str.strip().str.lower()
    frame["title"] = frame["title"].fillna("").astype(str).str.strip()
    return frame.reset_index(drop=True)


def fetch_orders(since: _dt.datetime, api_key: str, base_url: str) -> pd.DataFrame:
    """Every order created at or after ``since``, oldest first.

    Filtered server-side: asking for a month of a shop that has years of history
    behind it should cost a month of rows.
    """
    rows, truncated = _fetch_pages(
        api_key,
        base_url,
        {"created_at[$gte]": since.astimezone(_dt.timezone.utc).isoformat()},
    )
    frame = _order_frame(rows, base_url)
    # Read by the UI to say so, rather than reporting a partial book as the book.
    frame.attrs["truncated"] = truncated
    return frame


def _stack(kept: pd.DataFrame, fresh: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Concatenate two frames, ignoring an empty one so its dtypes cannot win.

    A refresh that finds nothing new hands over an all-empty frame, and pandas
    reads its columns as objects when deciding the result's dtypes.
    """
    parts = [frame for frame in (kept, fresh) if not frame.empty]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def sync_order_book(
    previous: OrderBook | None,
    api_key: str,
    base_url: str,
    days: int,
    now: _dt.datetime | None = None,
) -> OrderBook:
    """The last ``days`` days of orders, reading only what has changed since
    ``previous`` was read.

    A first call reads the whole window. Later calls ask only for orders touched
    since the last read and graft them onto the book, replacing the earlier copy
    of any order that changed - an order captured or cancelled hours after it was
    placed has to overwrite its old row, not appear twice.
    Orders that have aged out of the window are dropped on the way through.
    """
    read_at = now or _dt.datetime.now(_dt.timezone.utc)
    window_start = read_at - _dt.timedelta(days=days)
    incremental = previous is not None and previous.window_days == days
    if incremental:
        assert previous is not None  # narrowed by `incremental`
        filters = {
            "updated_at[$gte]": (
                (previous.synced_at - _SYNC_OVERLAP)
                .astimezone(_dt.timezone.utc)
                .isoformat()
            )
        }
    else:
        filters = {"created_at[$gte]": window_start.isoformat()}

    rows, truncated = _fetch_pages(api_key, base_url, filters)
    fresh_orders = _order_frame(rows, base_url)
    fresh_items = _item_frame(rows)

    if incremental and previous is not None:
        changed = set(fresh_orders["id"].tolist())
        kept_orders = previous.orders[~previous.orders["id"].isin(changed)]
        kept_items = previous.items[~previous.items["order_id"].isin(changed)]
        order_frame = _stack(kept_orders, fresh_orders, COLUMNS)
        item_frame = _stack(kept_items, fresh_items, ITEM_COLUMNS)
        truncated = truncated or previous.truncated
    else:
        order_frame, item_frame = fresh_orders, fresh_items

    # An incremental read is filtered on when an order changed, so it also brings
    # back orders placed before the window: the window is applied here instead.
    order_frame = order_frame[order_frame["created_at"].ge(window_start)]
    item_frame = item_frame[item_frame["created_at"].ge(window_start)]
    order_frame = (
        order_frame.drop_duplicates(subset="id", keep="last")
        .sort_values("created_at")
        .reset_index(drop=True)
    )
    item_frame = item_frame.sort_values("created_at").reset_index(drop=True)
    order_frame.attrs["truncated"] = truncated
    return OrderBook(
        orders=order_frame,
        items=item_frame,
        window_days=days,
        synced_at=read_at,
        truncated=truncated,
    )


def _prefix_aliases() -> dict[str, str]:
    """Retired handle prefixes mapped to the merchant they belonged to.

    A merchant that changed its prefix keeps the old one on everything it has
    already sold, and the store record only remembers the current one, so that
    history would otherwise be attributed to nobody. Set as
    ``MEDUSA_STORE_PREFIX_ALIASES="oldprefix=Store Name,other=Other Store"``.
    """
    aliases: dict[str, str] = {}
    for entry in os.getenv("MEDUSA_STORE_PREFIX_ALIASES", "").split(","):
        prefix, _, name = entry.partition("=")
        if prefix.strip() and name.strip():
            aliases[prefix.strip().lower()] = name.strip()
    return aliases


def fetch_stores(api_key: str, base_url: str) -> dict[str, str]:
    """Each merchant's handle prefix mapped to its name.

    Every product a merchant lists is slugged with that prefix, and the order
    line keeps the product handle, so the prefix is what ties a bottle sold back
    to the shop that sold it. The admin API exposes no order-to-store link to an
    API-key caller.
    """
    response = requests.get(
        f"{base_url}/admin/stores",
        params={"limit": 200},
        headers=_auth_header(api_key),
        timeout=30,
    )
    if response.status_code in (401, 403):
        raise MedusaConfigError(
            f"The CRM refused the API key ({response.status_code}) for the store list."
        )
    response.raise_for_status()
    prefixes: dict[str, str] = {}
    for store in response.json().get("stores") or []:
        metadata = store.get("metadata") or {}
        prefix = str(metadata.get("store_prefix") or "").strip().lower()
        name = str(store.get("name") or "").strip()
        if prefix and name:
            prefixes[prefix] = name
    # Configured aliases lose to a live store prefix: a prefix the CRM is using
    # today belongs to whoever it says, not to a stale setting.
    return {**_prefix_aliases(), **prefixes}


__all__ = [
    "COLUMNS",
    "ITEM_COLUMNS",
    "REQUIRED_FIELDS",
    "DEFAULT_BASE_URL",
    "MedusaConfigError",
    "OrderBook",
    "PAID_PAYMENT_STATUSES",
    "fetch_orders",
    "fetch_stores",
    "load_medusa_env",
    "sync_order_book",
]
