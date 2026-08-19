"""The Delivery page's two computed views.

Cycle-time-by-status and the masked-days stale table are the page's claims
about time. Both are derived from the changelog, so both are pinned against
the ways a changelog can mislead: open intervals inflating medians, thin
samples posing as statistics, and edit-freshness masking a stalled ticket.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pandas as pd

import app
import data_layer
import integrity
from pages import delivery as delivery_page


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


def test_a_cut_stale_queue_carries_how_much_was_cut():
    """Fourteen stale tickets shown as twelve owes the reader the other two."""
    df = pd.DataFrame(
        [_moved_days_ago(f"ENG-{i}", 40.0 + i) for i in range(1, 15)]
    )
    stale = app._stale_with_masked(df)
    assert len(stale) == 12
    assert stale.attrs["stale_total"] == 14
    assert "of 14" in app._truncation_note(stale.attrs["stale_total"], len(stale))


def test_an_unreadable_history_is_not_reported_as_a_clean_board(monkeypatch):
    """Empty because it failed, not empty because nothing is stale."""

    def boom(*_args, **_kwargs):
        raise ValueError("changelog")

    monkeypatch.setattr(app.integrity, "status_age_days", boom)
    stale = app._stale_with_masked(pd.DataFrame([_moved_days_ago("ENG-1", 90.0)]))
    assert stale.empty
    assert stale.attrs.get("stale_unreadable") is True


def test_a_board_with_nothing_stale_is_not_flagged_as_unreadable():
    df = pd.DataFrame([_moved_days_ago("ENG-1", 3.0)])
    stale = app._stale_with_masked(df)
    assert stale.empty
    assert not stale.attrs.get("stale_unreadable")


def test_a_label_only_edit_does_not_remove_a_ticket_from_the_stale_queue():
    """The exploit this table exists to close: touching a field cannot buy a ticket out of it."""
    gamed = _ticket(
        "ENG-GAMED",
        [("2026-01-05T00:00:00.000+0000", "To Do", "In Progress")],
        idle_days=0.2,  # "last touched" an hour ago
    )
    # A cosmetic-only entry, months after the status transition, is exactly
    # what resets idle_days without moving the ticket - append it so the
    # fixture carries the edit itself, not just its consequence (idle_days).
    gamed["changelog"].append(
        {
            "created": "2026-08-18T00:00:00.000+0000",
            "items": [{"field": "labels", "fromString": "", "toString": "needs-triage"}],
            "author": {"displayName": "Tam"},
        }
    )
    stale = app._stale_with_masked(pd.DataFrame([gamed]))
    assert "ENG-GAMED" in list(stale["key"])
    row = stale[stale["key"] == "ENG-GAMED"].iloc[0]
    assert row["status_age_days"] > 180
    assert row["masked_days"] > 180


# ---------------------------------------------------------------------------
# Task 3C: the three replaced tiles, their deltas, and the full-page render.
#
# These render ``_render_delivery_page()`` end to end against a stubbed
# ``_EngineeringData``/``_EngineeringView`` (no Streamlit test harness exists
# in this repo) and capture every ``st.markdown`` fragment it writes, since
# that HTML is the page's actual, observable output.
# ---------------------------------------------------------------------------


def _issue(
    key: str,
    *histories: dict,
    status: str = "In Progress",
    assignee: str = "Ana",
    created: str = "2026-01-01T00:00:00.000+0000",
    summary: str = "work",
    idle_days: float = 1.0,
) -> dict:
    return {
        "key": key,
        "status": status,
        "assignee": assignee,
        "summary": summary,
        "created": created,
        "idle_days": idle_days,
        "changelog": {"histories": list(histories)},
    }


def _hist(ts: str, author: str, field: str, frm: str, to: str) -> dict:
    return {
        "created": ts,
        "author": {"displayName": author},
        "items": [{"field": field, "fromString": frm, "toString": to}],
    }


def _ago(now: pd.Timestamp, days: float) -> str:
    return (now - pd.Timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.000+0000")


def _render_page(monkeypatch, raw_df: pd.DataFrame, *, data: dict | None = None) -> str:
    """Render the Delivery page against a stub bundle; return every fragment it wrote."""
    events = integrity.changelog_events(raw_df)
    bundle = data_layer._EngineeringData(
        data=data or {},
        errors={},
        raw_df=raw_df,
        df=raw_df,
        events=events,
        github_ready=True,
        github_error="",
        open_prs=pd.DataFrame(),
        merged_prs=pd.DataFrame(),
        pr_count_7=0,
        pr_count_30=0,
        open_count_exact=0,
        assignees=[],
        statuses=[],
        priorities=[],
        max_results=100,
        page_size=100,
    )
    view = data_layer._EngineeringView(
        scope="Organization",
        selected_assignees=[],
        selected_statuses=[],
        selected_priorities=[],
        min_idle=0,
        min_age=0,
        include_backlogs=False,
        color_by=None,
        allow_writes=False,
        filtered=raw_df,
        unscoped=raw_df,
    )
    slot = SimpleNamespace(download_button=lambda *a, **k: None)

    captured: list[str] = []
    monkeypatch.setattr(delivery_page, "_engineering_context", lambda: (bundle, view, slot))
    monkeypatch.setattr(delivery_page, "_download_report", lambda *a, **k: None)
    monkeypatch.setattr(delivery_page.st, "markdown", lambda body, **k: captured.append(body))
    delivery_page._render_delivery_page()
    return "".join(captured)


def _tile_chunk(html_out: str, label: str) -> str:
    """The one ``<div class="tile">...`` fragment whose label is ``label``."""
    for chunk in html_out.split('<div class="tile"'):
        if f'<div class="lbl">{label}</div>' in chunk:
            return chunk
    raise AssertionError(f"tile {label!r} not found in rendered output")


def test_the_three_new_tiles_render_with_labels_and_an_explicit_delta_direction(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    raw_df = pd.DataFrame(
        [
            _issue(
                "ENG-1",
                _hist(_ago(now, 40), "Ana", "status", "To Do", "In Progress"),
                _hist(_ago(now, 20), "Ana", "status", "In Progress", "Review in Staging"),
                _hist(_ago(now, 18), "Bob", "status", "Review in Staging", "In Progress"),
                _hist(_ago(now, 5), "Ana", "status", "In Progress", "Review in Staging"),
                status="Review in Staging",
            )
        ]
    )
    html_out = _render_page(
        monkeypatch, raw_df, data={"resolved_count_7": 5, "resolved_30": pd.DataFrame()}
    )
    for label in ("Staging round-trips", "Reopened · 30d", "Unattributed"):
        chunk = _tile_chunk(html_out, label)
        delta_match = re.search(r'<div class="delta[^"]*">(.*?)</div>', chunk)
        assert delta_match, f"{label} tile carries no delta line"
        assert delta_match.group(1).strip().startswith(("▲", "▼", "—"))


def test_reopened_renders_its_denominator(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    raw_df = pd.DataFrame(
        [
            _issue(
                "ENG-1",
                _hist(_ago(now, 10), "Ana", "status", "To Do", "In Progress"),
                _hist(_ago(now, 5), "Ana", "status", "In Progress", "Review in Staging"),
                _hist(_ago(now, 2), "Bob", "status", "Review in Staging", "In Progress"),
                status="In Progress",
            )
        ]
    )
    html_out = _render_page(
        monkeypatch, raw_df, data={"resolved_count_7": 1, "resolved_30": pd.DataFrame()}
    )
    chunk = _tile_chunk(html_out, "Reopened · 30d")
    assert "of 1 resolved" in chunk
    assert "— of 1 resolved" not in chunk


def test_reopened_renders_an_em_dash_not_zero_percent_when_nothing_resolved(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    raw_df = pd.DataFrame(
        [
            _issue(
                "ENG-1",
                _hist(_ago(now, 10), "Ana", "priority", "Low", "High"),  # never touches status
                status="In Progress",
            )
        ]
    )
    html_out = _render_page(
        monkeypatch, raw_df, data={"resolved_count_7": 0, "resolved_30": pd.DataFrame()}
    )
    chunk = _tile_chunk(html_out, "Reopened · 30d")
    assert "— of 0 resolved" in chunk
    assert "0% of 0 resolved" not in chunk
    assert "0.0% of 0 resolved" not in chunk


def test_the_rendered_stale_table_ranks_by_status_age_not_last_touched(monkeypatch):
    """Same fixture shape as the dataframe-level ordering test, checked in the actual HTML."""
    now = pd.Timestamp.now(tz="UTC")
    old_touched_recently = _issue(
        "ENG-OLD",
        _hist(_ago(now, 200), "Ana", "status", "To Do", "In Progress"),
        idle_days=0.5,
    )
    young_but_idle_looking = _issue(
        "ENG-NEW",
        _hist(_ago(now, 35), "Bob", "status", "To Do", "In Progress"),
        idle_days=50.0,
    )
    raw_df = pd.DataFrame([young_but_idle_looking, old_touched_recently])
    html_out = _render_page(
        monkeypatch, raw_df, data={"resolved_count_7": 0, "resolved_30": pd.DataFrame()}
    )
    assert "ENG-OLD" in html_out and "ENG-NEW" in html_out
    assert html_out.index("ENG-OLD") < html_out.index("ENG-NEW")


def test_the_page_renders_end_to_end_on_an_empty_frame_without_raising(monkeypatch):
    """No changelog, no ticket frame: the three new tiles say so instead of printing 0."""
    html_out = _render_page(monkeypatch, pd.DataFrame(), data={})
    for label in ("Staging round-trips", "Reopened · 30d", "Unattributed"):
        chunk = _tile_chunk(html_out, label)
        assert "no changelog data read" in chunk
        assert '<div class="val">—' in chunk
