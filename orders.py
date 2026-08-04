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


def _paid(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return orders
    return orders[orders["payment_status"].isin(PAID_PAYMENT_STATUSES)]


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
    revenue = float(paid["total"].sum()) if not paid.empty else 0.0
    canceled = (
        int(current["status"].isin(CANCELED_STATUSES).sum()) if not current.empty else 0
    )
    return WindowMetrics(
        days=days,
        orders=int(len(current)),
        canceled=canceled,
        paid_orders=int(len(paid)),
        revenue=round(revenue, 2),
        aov=round(revenue / len(paid), 2) if len(paid) else 0.0,
        prev_orders=int(len(previous)),
        prev_revenue=round(float(_paid(previous)["total"].sum()) if not previous.empty else 0.0, 2),
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
    start = end - _dt.timedelta(days=days)
    window = _slice(orders, start, end)
    index = pd.date_range(start.date(), end.date(), freq="D", tz="UTC").normalize()
    if window.empty:
        return pd.DataFrame({"date": index.date, "orders": 0, "revenue": 0.0})[columns]

    placed = window.copy()
    placed["date"] = placed["created_at"].dt.normalize()
    counts = placed.groupby("date")["id"].count().reindex(index, fill_value=0)
    paid = _paid(placed)
    revenue = (
        paid.groupby("date")["total"].sum().reindex(index, fill_value=0.0)
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


__all__ = ["CANCELED_STATUSES", "WindowMetrics", "daily_orders", "window_metrics"]
