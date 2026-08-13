"""Offline checks for the order book's window figures.

Kept outside the repository for the same reason as the other check scripts: it
exercises the module directly, with no Medusa and no Streamlit.

    python3 -m pytest tests/test_orders_windows.py -q
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orders as ob  # noqa: E402

NOW = dt.datetime(2026, 8, 6, tzinfo=dt.timezone.utc)


def book(rows: list[tuple[int, str, str, float]]) -> pd.DataFrame:
    """``(days_ago, status, payment_status, total)`` as an order book."""
    return pd.DataFrame(
        [
            {
                "created_at": NOW - dt.timedelta(days=ago, hours=1),
                "status": status,
                "payment_status": payment,
                "total": total,
                "refunded_total": 0.0,
                "currency_code": "usd",
            }
            for ago, status, payment, total in rows
        ]
    )


def test_the_previous_window_counts_paid_orders_on_the_same_basis_as_this_one():
    # Two paid orders in each window, plus one cancelled and one awaiting
    # payment in the earlier one. Counting those two would report a fall in
    # paid orders that never happened.
    metrics = ob.window_metrics(
        book(
            [
                (1, "pending", "captured", 100.0),
                (2, "pending", "captured", 100.0),
                (8, "pending", "captured", 100.0),
                (9, "pending", "captured", 100.0),
                (10, "canceled", "refunded", 100.0),
                (11, "pending", "awaiting", 100.0),
            ]
        ),
        7,
        now=NOW,
    )
    assert metrics.paid_orders == 2
    assert metrics.prev_paid_orders == 2
    # The all-orders figure is unchanged and still counts the other two.
    assert metrics.prev_orders == 4


def test_an_empty_previous_window_has_no_paid_orders_rather_than_an_error():
    metrics = ob.window_metrics(
        book([(1, "pending", "captured", 100.0)]), 7, now=NOW
    )
    assert metrics.prev_paid_orders == 0
    assert metrics.prev_revenue == 0.0


def test_a_delta_is_read_as_a_number_rather_than_by_its_first_characters():
    # `_unmoved` lives in app.py, which runs a whole dashboard when imported, so
    # the function is lifted out of the source rather than imported.
    import ast

    source = open(Path(__file__).resolve().parents[1] / "app.py").read()
    tree = ast.parse(source)
    wanted = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_unmoved"
    )
    namespace: dict = {"re": __import__("re")}
    exec(compile(ast.Module([wanted], []), "app.py", "exec"), namespace)
    unmoved = namespace["_unmoved"]

    assert unmoved("flat")
    assert unmoved("+0 people")
    assert unmoved("+0.00")
    # A twentieth of a unit is nothing beside a spend figure and everything
    # beside a return per unit spent; either way it moved.
    assert not unmoved("+0.05")
    assert not unmoved("-0.05")
    assert not unmoved("+$0.05")
    assert not unmoved("+1,204 people")


import orders_client as oc  # noqa: E402


class _Cursor:
    """Enough of a psycopg2 cursor to see what was asked and answer it."""

    def __init__(self, rows, seen):
        self._rows = rows
        self._seen = seen

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, sql, params=None):
        self._seen.append((sql, params))

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _Connection:
    def __init__(self, rows, seen):
        self._rows = rows
        self._seen = seen

    def cursor(self):
        return _Cursor(self._rows, self._seen)

    def close(self):
        pass


def test_an_offer_names_every_merchant_listing_it(monkeypatch):
    """The same bottle is stocked by several merchants, and all of them matter."""
    seen: list = []
    rows = [
        ("001037450", "capital-croft-port-purple-velvet-0"),
        ("001037450", "yiannis-croft-port-finest-reserve-0"),
        ("  ", "orphan-handle"),
    ]
    monkeypatch.setattr(oc, "_connect", lambda config: _Connection(rows, seen))
    config = oc.DbConfig("host", 5432, "db", "user", "secret")
    found = oc.fetch_offer_handles(config, ["001037450", "001037450", " "])
    assert found == {
        "001037450": (
            "capital-croft-port-purple-velvet-0",
            "yiannis-croft-port-finest-reserve-0",
        )
    }
    # Asked for once, as a parameter rather than as interpolated SQL.
    sql, params = seen[0]
    assert params == {"offers": ["001037450"]}
    assert "external_id = any(%(offers)s)" in sql

    prefixes = {"capital": "Capital Fine Wine", "yiannis": "Yiannis Wine Shop"}
    names = sorted(
        ob.merchant_of(handle, prefixes) for handle in found["001037450"]
    )
    assert names == ["Capital Fine Wine", "Yiannis Wine Shop"]


def test_no_offers_asks_the_database_nothing(monkeypatch):
    def _refuse(_config):
        raise AssertionError("the database was opened for an empty list")

    monkeypatch.setattr(oc, "_connect", _refuse)
    config = oc.DbConfig("host", 5432, "db", "user", "secret")
    assert oc.fetch_offer_handles(config, []) == {}


class _Column:
    def __init__(self, name):
        self.name = name


class _FrameCursor(_Cursor):
    """A cursor that also describes its columns, as ``_frame`` needs."""

    def __init__(self, rows, seen, columns):
        super().__init__(rows, seen)
        self.description = [_Column(name) for name in columns]


class _FrameConnection(_Connection):
    def __init__(self, rows, seen, columns):
        super().__init__(rows, seen)
        self._columns = columns

    def cursor(self):
        return _FrameCursor(self._rows, self._seen, self._columns)


def test_the_bottles_sold_are_counted_per_google_offer(monkeypatch):
    """What the merchant is shown beside its price: its own paid sales."""
    seen: list = []
    rows = [
        ("001037450", "Yiannis-Croft-Port", "7", 129.5),
        ("  ", "orphan", "3", 10.0),
    ]
    monkeypatch.setattr(
        oc,
        "_connect",
        lambda config: _FrameConnection(
            rows, seen, ["offer", "handle", "bottles", "revenue"]
        ),
    )
    config = oc.DbConfig("host", 5432, "db", "user", "secret")
    now = dt.datetime(2026, 8, 13, tzinfo=dt.timezone.utc)
    frame = oc.fetch_offer_sales(config, 90, now)
    # The offer with no id is not a sale of anything Google prices.
    assert list(frame["offer"]) == ["001037450"]
    assert list(frame["bottles"]) == [7]
    assert list(frame["revenue"]) == [129.5]
    assert list(frame["handle"]) == ["yiannis-croft-port"]

    sql, params = seen[0]
    assert params["since"] == now - dt.timedelta(days=90)
    # Paid orders only, and passed as a parameter rather than built into the SQL.
    assert params["paid"] == sorted(oc.PAID_PAYMENT_STATUSES)
    assert "p.payment_status = any(%(paid)s)" in sql
    # A cancelled order that was captured and refunded reads as a paid status,
    # so the paid filter alone would count bottles nobody kept.
    assert "p.status <> 'canceled'" in sql
    # One row per offer, however many merchants list the wine.
    assert "group by pr.external_id" in sql


def test_a_shop_with_no_sales_yet_is_an_empty_frame_not_a_crash(monkeypatch):
    seen: list = []
    monkeypatch.setattr(
        oc,
        "_connect",
        lambda config: _FrameConnection([], seen, ["offer", "handle", "bottles",
                                                   "revenue"]),
    )
    config = oc.DbConfig("host", 5432, "db", "user", "secret")
    frame = oc.fetch_offer_sales(config, 30)
    assert frame.empty
    assert list(frame.columns) == ["offer", "handle", "bottles", "revenue"]
