"""The Delivery page's two computed views.

Cycle-time-by-status and the masked-days stale table are the page's claims
about time. Both are derived from the changelog, so both are pinned against
the ways a changelog can mislead: open intervals inflating medians, thin
samples posing as statistics, and edit-freshness masking a stalled ticket.
"""

from __future__ import annotations

import pandas as pd

import app


def _ticket(key: str, transitions: list[tuple[str, str, str]], **fields) -> dict:
    """A ticket whose changelog moves through the given (ts, from, to) states."""
    return {
        "key": key,
        "assignee": fields.get("assignee", "Tam"),
        "status": fields.get("status", "In Progress"),
        "summary": fields.get("summary", "work"),
        "idle_days": fields.get("idle_days", 1.0),
        "created": pd.Timestamp("2026-01-01T00:00:00Z"),
        "changelog": [
            {
                "created": ts,
                "items": [
                    {"field": "status", "fromString": src, "toString": dst}
                ],
                "author": {"displayName": "Tam"},
            }
            for ts, src, dst in transitions
        ],
    }


def test_cycle_time_needs_five_intervals_before_it_speaks():
    """Two closed intervals is an anecdote; the chart must not print it."""
    rows = [
        _ticket(
            f"ENG-{i}",
            [
                (f"2026-02-0{i}T00:00:00.000+0000", "To Do", "In Progress"),
                (f"2026-02-0{i}T12:00:00.000+0000", "In Progress", "Done"),
            ],
        )
        for i in range(1, 3)
    ]
    out = app._cycle_by_status(pd.DataFrame(rows))
    assert out.empty


def test_cycle_time_reports_the_median_of_closed_intervals():
    rows = [
        _ticket(
            f"ENG-{i}",
            [
                (f"2026-02-{i:02d}T00:00:00.000+0000", "To Do", "In Progress"),
                (f"2026-02-{i+10:02d}T00:00:00.000+0000", "In Progress", "Done"),
            ],
        )
        for i in range(1, 7)
    ]
    out = app._cycle_by_status(pd.DataFrame(rows))
    row = out[out["status"] == "In Progress"]
    assert not row.empty
    assert float(row["median_days"].iloc[0]) == 10.0
    assert int(row["n"].iloc[0]) == 6


def test_an_unparseable_changelog_omits_the_chart_rather_than_crashing():
    df = pd.DataFrame({"key": ["ENG-1"], "changelog": [object()]})
    out = app._cycle_by_status(df)
    assert out.empty


def test_the_stale_table_surfaces_masked_days_per_row():
    """Status age 200+, touched yesterday: the row the whole table exists for."""
    df = pd.DataFrame(
        [
            _ticket(
                "ENG-1",
                [("2026-01-02T00:00:00.000+0000", "To Do", "In Progress")],
                idle_days=1.0,
                summary="merchant importer",
            )
        ]
    )
    stale = app._stale_with_masked(df)
    assert not stale.empty
    row = stale.iloc[0]
    assert row["status_age_days"] > 180
    assert row["masked_days"] > 180
    assert "merchant importer" in row["summary"]


def test_the_stale_table_is_ordered_by_status_age_not_edit_age():
    old_moved_recently_touched = _ticket(
        "ENG-OLD",
        [("2026-01-02T00:00:00.000+0000", "To Do", "In Progress")],
        idle_days=0.5,
    )
    young = _ticket(
        "ENG-NEW",
        [("2026-08-01T00:00:00.000+0000", "To Do", "In Progress")],
        idle_days=50.0,
    )
    stale = app._stale_with_masked(pd.DataFrame([young, old_moved_recently_touched]))
    assert stale.iloc[0]["key"] == "ENG-OLD"


def _moved_days_ago(key: str, days: float, **fields) -> dict:
    moved = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    return _ticket(
        key,
        [(moved.strftime("%Y-%m-%dT%H:%M:%S.000+0000"), "To Do", "In Progress")],
        **fields,
    )


def test_a_ticket_that_moved_this_week_is_not_called_stale():
    """A board where everything is moving has an empty stale table, not a top 12."""
    df = pd.DataFrame(
        [_moved_days_ago(f"ENG-{i}", float(i)) for i in range(1, 6)]
    )
    assert app._stale_with_masked(df).empty


def test_the_stale_table_keeps_only_rows_past_the_stalled_clock():
    df = pd.DataFrame(
        [
            _moved_days_ago("ENG-FRESH", 2.0),
            _moved_days_ago("ENG-STALE", app.TODAY_STALLED_DAYS + 5.0),
        ]
    )
    stale = app._stale_with_masked(df)
    assert list(stale["key"]) == ["ENG-STALE"]
