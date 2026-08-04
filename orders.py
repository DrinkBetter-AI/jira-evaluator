"""Order counts, revenue and AOV over the last 7 and 30 days.

Two decisions worth stating, because they change the numbers: a cancelled order
is not a sale and is counted separately rather than inside revenue, and revenue
is money actually captured, so an order placed but not yet paid raises the order
count without raising revenue. AOV follows revenue - captured money over the
orders that produced it - since dividing captured money by every order placed
would understate the basket of a shop with a payment backlog.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import pandas as pd

from orders_client import PAID_PAYMENT_STATUSES

CANCELED_STATUSES = frozenset({"canceled", "cancelled"})

# Windows the wine and merchant tables offer: a month reads as what is selling
# now, a year as what the shop actually sells.
LOOKBACK_WINDOWS = (30, 90, 180, 360)

# Sold alongside the wine and shipped with it, but nobody's bottle. Left in the
# merchant tables, where it is real money, and out of the wine table, where it
# would otherwise outrank every wine in the shop.
NON_WINE_HANDLES = frozenset({"ice-pack"})

UNATTRIBUTED = "Unattributed"
# Where the add-ons' money goes in the merchant table: the platform sells them,
# so crediting them to nobody would read as an attribution failure.
PLATFORM = "VinoVoss (add-ons)"
TOP_WINES_LIMIT = 15


@dataclass(frozen=True)
class WindowMetrics:
    """What one window (7 or 30 days) of the order book looks like."""

    days: int
    orders: int
    canceled: int
    paid_orders: int
    revenue: float
    aov: float
    # The same figures for the window immediately before this one, so a number
    # can be read as rising or falling rather than in isolation.
    prev_orders: int
    prev_revenue: float

    @property
    def unpaid_orders(self) -> int:
        """Placed, not cancelled, and not yet captured."""
        return max(self.orders - self.canceled - self.paid_orders, 0)

    @property
    def orders_delta(self) -> int:
        return self.orders - self.prev_orders

    @property
    def revenue_delta(self) -> float:
        return round(self.revenue - self.prev_revenue, 2)


def _slice(orders: pd.DataFrame, start: _dt.datetime, end: _dt.datetime) -> pd.DataFrame:
    if orders.empty:
        return orders
    created = orders["created_at"]
    return orders[created.ge(start) & created.lt(end)]


def single_currency(orders: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    """The order book in its main currency, that currency, and any others found.

    Totals in different currencies cannot be added, and the shop has only ever
    billed in one. Should that change, the tiles keep meaning something - they
    report the main currency - and the caller names what was set aside.
    """
    if orders.empty or "currency_code" not in orders.columns:
        return orders, "", []
    found = orders["currency_code"].value_counts()
    main = str(found.index[0])
    others = sorted(str(code) for code in found.index[1:])
    if not others:
        return orders, main, []
    return orders[orders["currency_code"].eq(main)], main, others


def _canceled(orders: pd.DataFrame) -> pd.Series:
    if orders.empty:
        return pd.Series(dtype=bool)
    return orders["status"].isin(CANCELED_STATUSES)


def _paid(orders: pd.DataFrame) -> pd.DataFrame:
    """Orders whose money the shop kept: captured, and not cancelled.

    Payment state alone is not enough. A cancelled order is usually refunded,
    and a refund is only a correction to revenue when the sale stood; on a
    cancelled order it means the sale did not happen at all.
    """
    if orders.empty:
        return orders
    return orders[
        orders["payment_status"].isin(PAID_PAYMENT_STATUSES) & ~_canceled(orders)
    ]


def _kept(orders: pd.DataFrame) -> pd.Series:
    """Per order, the money the shop kept: the total less anything refunded."""
    if orders.empty:
        return pd.Series(dtype=float)
    refunded = orders["refunded_total"] if "refunded_total" in orders.columns else 0.0
    return (orders["total"] - refunded).clip(lower=0)


def window_metrics(
    orders: pd.DataFrame,
    days: int,
    now: _dt.datetime | None = None,
) -> WindowMetrics:
    """Metrics for the last ``days`` days, and the ``days`` before those."""
    end = now or _dt.datetime.now(_dt.timezone.utc)
    start = end - _dt.timedelta(days=days)
    previous_start = start - _dt.timedelta(days=days)

    current = _slice(orders, start, end)
    previous = _slice(orders, previous_start, start)
    paid = _paid(current)
    kept = _kept(paid)
    revenue = float(kept.sum()) if not paid.empty else 0.0
    # An order refunded down to nothing is still a paid order - it is not
    # awaiting payment - but it produced no basket, so averaging over it would
    # drag AOV towards zero for a sale that was undone rather than made small.
    earning = int((kept > 0).sum()) if not paid.empty else 0
    canceled = int(_canceled(current).sum()) if not current.empty else 0
    return WindowMetrics(
        days=days,
        orders=int(len(current)),
        canceled=canceled,
        paid_orders=int(len(paid)),
        revenue=round(revenue, 2),
        aov=round(revenue / earning, 2) if earning else 0.0,
        prev_orders=int(len(previous)),
        prev_revenue=round(float(_kept(_paid(previous)).sum()), 2),
    )


def daily_orders(
    orders: pd.DataFrame,
    days: int = 30,
    now: _dt.datetime | None = None,
) -> pd.DataFrame:
    """Orders and captured revenue per day, with quiet days present as zeroes.

    A day with no orders is a fact about the shop; leaving it out of the series
    would draw a flat line through it and hide the gap.
    """
    columns = ["date", "orders", "revenue"]
    end = now or _dt.datetime.now(_dt.timezone.utc)
    # Whole days only: counting from this hour ``days`` ago would draw a first
    # bar covering the tail of a day and read as a collapse in orders.
    start = pd.Timestamp(end).tz_convert("UTC").normalize() - pd.Timedelta(
        days=days - 1
    )
    window = _slice(orders, start.to_pydatetime(), end)
    index = pd.date_range(start, pd.Timestamp(end).tz_convert("UTC").normalize(), freq="D")
    if window.empty:
        return pd.DataFrame({"date": index.date, "orders": 0, "revenue": 0.0})[columns]

    placed = window.copy()
    placed["date"] = placed["created_at"].dt.normalize()
    counts = placed.groupby("date")["id"].count().reindex(index, fill_value=0)
    paid = _paid(placed)
    revenue = (
        paid.assign(kept=_kept(paid))
        .groupby("date")["kept"]
        .sum()
        .reindex(index, fill_value=0.0)
        if not paid.empty
        else pd.Series(0.0, index=index)
    )
    return pd.DataFrame(
        {
            "date": index.date,
            "orders": counts.astype(int).to_numpy(),
            "revenue": revenue.round(2).to_numpy(),
        }
    )[columns]


def _standing(items: pd.DataFrame) -> pd.DataFrame:
    """Line items from orders that were not cancelled.

    Deliberately not limited to captured orders, unlike revenue: what sells is a
    question about demand, and an order awaiting payment still shows what a
    customer chose.
    """
    if items.empty:
        return items
    return items[~items["status"].isin(CANCELED_STATUSES)]


def top_wines(
    items: pd.DataFrame,
    days: int,
    now: _dt.datetime | None = None,
    limit: int = TOP_WINES_LIMIT,
) -> pd.DataFrame:
    """The wines sold most often in the last ``days`` days, by bottles.

    Bottles, not revenue: the question is which wine is popular, and ranking by
    revenue would answer a different one and put the most expensive bottle on
    top of a list of the most wanted.
    """
    columns = ["wine", "bottles", "orders", "revenue"]
    end = now or _dt.datetime.now(_dt.timezone.utc)
    window = _standing(_slice(items, end - _dt.timedelta(days=days), end))
    if window.empty:
        return pd.DataFrame(columns=columns)
    wines = window[~window["product_handle"].isin(NON_WINE_HANDLES)]
    if wines.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        wines.groupby("title")
        .agg(
            bottles=("quantity", "sum"),
            orders=("order_id", "nunique"),
            revenue=("revenue", "sum"),
        )
        .reset_index()
        .rename(columns={"title": "wine"})
    )
    grouped["revenue"] = grouped["revenue"].round(2)
    return (
        grouped.sort_values(["bottles", "revenue"], ascending=False)
        .head(limit)
        .reset_index(drop=True)[columns]
    )


def merchant_of(handle: str, prefixes: dict[str, str]) -> str:
    """The merchant whose product handle starts with one of ``prefixes``.

    Longest prefix first: one merchant's prefix can begin with another's, and the
    shorter would then claim the longer's wine. A handle that matches nothing is
    named as unattributed rather than guessed at - a merchant that has since
    changed its prefix would otherwise have its history quietly credited
    elsewhere.
    """
    text = str(handle or "").strip().lower()
    if text in NON_WINE_HANDLES:
        return PLATFORM
    for prefix in sorted(prefixes, key=len, reverse=True):
        if text.startswith(f"{prefix}-"):
            return prefixes[prefix]
    return UNATTRIBUTED


def merchant_breakdown(
    items: pd.DataFrame,
    prefixes: dict[str, str],
    days: int,
    now: _dt.datetime | None = None,
) -> pd.DataFrame:
    """Revenue, orders and cancellations per merchant.

    Revenue is per line, since one order can carry wine from several merchants,
    but the order counts are per order: an order split between two merchants is a
    cancellation for both, because both had a sale fall through. Summing the
    order columns therefore over-counts, which is why the table shows no total.
    """
    columns = ["merchant", "revenue", "orders", "canceled"]
    end = now or _dt.datetime.now(_dt.timezone.utc)
    window = _slice(items, end - _dt.timedelta(days=days), end)
    if window.empty:
        return pd.DataFrame(columns=columns)

    window = window.assign(
        merchant=[merchant_of(handle, prefixes) for handle in window["product_handle"]]
    )
    canceled = window["status"].isin(CANCELED_STATUSES)
    paid = window["payment_status"].isin(PAID_PAYMENT_STATUSES) & ~canceled
    earned = (
        window[paid].groupby("merchant")["revenue"].sum().round(2)
        if paid.any()
        else pd.Series(dtype=float)
    )
    # Per merchant per order, so an order holding three of a merchant's wines is
    # one order for them rather than three.
    pairs = window[["merchant", "order_id"]].drop_duplicates()
    order_counts = pairs.groupby("merchant")["order_id"].nunique()
    canceled_counts = (
        window[canceled][["merchant", "order_id"]]
        .drop_duplicates()
        .groupby("merchant")["order_id"]
        .nunique()
        if canceled.any()
        else pd.Series(dtype=int)
    )

    table = pd.DataFrame({"merchant": sorted(pairs["merchant"].unique())})
    table["revenue"] = table["merchant"].map(earned).fillna(0.0)
    table["orders"] = table["merchant"].map(order_counts).fillna(0).astype(int)
    table["canceled"] = table["merchant"].map(canceled_counts).fillna(0).astype(int)
    return table.sort_values("revenue", ascending=False).reset_index(drop=True)[columns]


__all__ = [
    "CANCELED_STATUSES",
    "LOOKBACK_WINDOWS",
    "NON_WINE_HANDLES",
    "PLATFORM",
    "TOP_WINES_LIMIT",
    "UNATTRIBUTED",
    "WindowMetrics",
    "daily_orders",
    "merchant_breakdown",
    "merchant_of",
    "single_currency",
    "top_wines",
    "window_metrics",
]
