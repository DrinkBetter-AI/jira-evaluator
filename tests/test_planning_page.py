"""Task 3E: the Planning page's HTML-kit surfaces, offline.

Every case here is one of 3E's acceptance bullets made concrete. Two
functions in ``render_shared.py`` are exercised directly rather than through
Streamlit's ``AppTest`` harness: ``_render_hourly_capacity`` (plain, not
fragment-wrapped) and ``_render_sprint_capacity`` via its ``__wrapped__``
attribute - it carries ``@st.fragment``, which silently no-ops when called
bare outside a real script run (no exception, nothing rendered), and
``functools.wraps`` inside Streamlit's own ``fragment()`` decorator leaves
the undecorated function reachable at ``.__wrapped__`` for exactly this
reason. Every ``theme_html`` entry point the page reaches for is monkeypatched
to capture its arguments instead of drawing anything, mirroring the pattern
``tests/test_code_page.py`` already uses.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import render_shared as rs  # noqa: E402
import theme_html  # noqa: E402
import planning_metrics  # noqa: E402


def _ticket(
    key,
    assignee,
    sprint_name,
    board_id,
    state,
    hours,
    start=None,
    end=None,
    *,
    status="To Do",
    status_category="To Do",
    priority="High",
):
    """One synthetic row carrying every column ``_render_sprint_capacity`` reads."""
    return dict(
        key=key,
        assignee=assignee,
        sprint_name=sprint_name,
        sprint_id=f"{board_id}-{sprint_name}",
        sprint_board_id=board_id,
        sprint_state=state,
        sprint_start=start,
        sprint_end=end,
        original_estimate_sec=hours * 3600.0 if hours is not None else None,
        original_estimate="" if hours is None else f"{hours}h",
        status=status,
        status_category=status_category,
        priority=priority,
        issue_type="Story",
        summary=f"summary {key}",
        reporter="QA",
        logged_time="",
        time_spent_sec=0,
        ticket_age_days=10,
        idle_days=1,
        created="2026-08-01",
        updated="2026-08-10",
        epic_summary="",
        assignee_account_id="",
    )


class _Capture:
    """Records every call theme_html's kit functions receive, draws nothing."""

    def __init__(self, monkeypatch):
        self.sprint_card = []
        self.callout = []
        self.render = []
        self.hbars = []
        self.table = []
        monkeypatch.setattr(
            theme_html, "sprint_card", lambda *a, **k: self.sprint_card.append(a) or "<sprint_card/>"
        )
        monkeypatch.setattr(
            theme_html, "callout", lambda *a, **k: self.callout.append(a) or "<callout/>"
        )
        monkeypatch.setattr(theme_html, "render", lambda *frags: self.render.append(frags))
        monkeypatch.setattr(
            theme_html, "hbars", lambda bars, **k: self.hbars.append(list(bars)) or "<hbars/>"
        )
        monkeypatch.setattr(
            theme_html,
            "table",
            lambda columns, rows, **k: self.table.append((list(columns), list(rows))) or "<table/>",
        )

    def rendered_html(self) -> str:
        """Every fragment string handed to ``render()``, concatenated."""
        return "".join(frag for call in self.render for frag in call)


# ---------------------------------------------------------------------------
# Three sprint cards, no selectbox gate
# ---------------------------------------------------------------------------


def test_three_sprint_cards_render_simultaneously_with_no_selectbox_gate(monkeypatch):
    capture = _Capture(monkeypatch)
    selectbox_labels = []
    _real_selectbox = rs.st.selectbox

    def _recording_selectbox(label, *a, **k):
        selectbox_labels.append(label)
        return _real_selectbox(label, *a, **k)

    monkeypatch.setattr(rs.st, "selectbox", _recording_selectbox)

    df = pd.DataFrame(
        [
            _ticket("APP-1", "Farid", "App Sprint 1", 1, "active", 10, "2026-08-03", "2026-08-14"),
            _ticket("MKT-1", "Anouar", "Mkt Sprint 1", 2, "active", 6, "2026-08-03", "2026-08-14"),
            _ticket("ML-1", "Tam", "ML Sprint 5", 3, "active", 5, "2026-08-03", "2026-08-14"),
        ]
    )
    monkeypatch.setattr(rs, "WEEKLY_HOURS", {"Farid": 20.0, "Anouar": 20.0, "Tam": 20.0})

    rs._render_sprint_capacity.__wrapped__(df)

    assert len(capture.sprint_card) == 3
    boards = {call[0] for call in capture.sprint_card}
    assert boards == {"App", "Marketplace", "ML"}
    # The old page-level sprint picker is gone; nothing gates the three cards
    # behind a "Sprint" dropdown any more.
    assert "Sprint" not in selectbox_labels


# ---------------------------------------------------------------------------
# The dateless-sprint callout
# ---------------------------------------------------------------------------


def test_a_dateless_sprint_produces_the_page_level_callout_naming_it(monkeypatch):
    capture = _Capture(monkeypatch)

    df = pd.DataFrame(
        [
            _ticket("APP-1", "Farid", "App Sprint 1", 1, "active", 10, "2026-08-03", "2026-08-14"),
            # ML carries no sprint_start/sprint_end - the real ML-board shape.
            _ticket("ML-1", "Tam", "ML Sprint 5", 3, "active", 5),
        ]
    )
    monkeypatch.setattr(rs, "WEEKLY_HOURS", {"Farid": 20.0, "Tam": 20.0})

    rs._render_sprint_capacity.__wrapped__(df)

    assert len(capture.callout) == 1
    tone, title, body = capture.callout[0]
    assert tone == "warn"
    assert "ML Sprint 5" in title
    assert "excluded from the totals" in body
    assert "counted as zero" in body


# ---------------------------------------------------------------------------
# _render_hourly_capacity: cross-board totals, exclusion, two-rows-one-total,
# and the "unknown" coverage sentinel. Called directly - it is not
# fragment-wrapped.
# ---------------------------------------------------------------------------


def test_capacity_totals_exclude_a_dateless_sprint_not_count_it_as_zero(monkeypatch):
    capture = _Capture(monkeypatch)
    df = pd.DataFrame(
        [
            _ticket("APP-1", "Farid", "App Sprint 1", 1, "active", 10, "2026-08-03", "2026-08-14"),
            _ticket("ML-1", "Farid", "ML Sprint 5", 3, "active", 40),  # no dates
        ]
    )
    monkeypatch.setattr(rs, "WEEKLY_HOURS", {"Farid": 20.0})

    rs._render_hourly_capacity(df)

    assert len(capture.table) == 2  # capacity table, then coverage table
    columns, rows = capture.table[0]
    totals = [r for r in rows if str(r[1].value).startswith("Total")]
    assert len(totals) == 1
    # The total sums only the dated sprint (App, 10h) - the dateless ML
    # sprint's 40h never enters it, whether as a real number or a zero.
    assert totals[0][3].value == 10.0
    rendered = capture.rendered_html()
    assert "ML Sprint 5" in rendered
    assert "Excluded from these totals" in rendered


def test_a_person_on_two_boards_renders_two_rows_and_one_total_row(monkeypatch):
    capture = _Capture(monkeypatch)
    df = pd.DataFrame(
        [
            _ticket("APP-1", "Farid", "App Sprint 1", 1, "active", 10, "2026-08-03", "2026-08-14"),
            _ticket("MKT-1", "Farid", "Mkt Sprint 1", 2, "active", 6, "2026-08-03", "2026-08-14"),
        ]
    )
    # Only Farid is declared, and only two dated sprints exist in the whole
    # frame, so nothing can add a third "declared but idle" row for anyone.
    monkeypatch.setattr(rs, "WEEKLY_HOURS", {"Farid": 20.0})

    rs._render_hourly_capacity(df)

    columns, rows = capture.table[0]
    farid_rows = [r for r in rows if r[0].value == "Farid"]
    assert len(farid_rows) == 3  # two per-sprint rows ...
    per_sprint = [r for r in farid_rows if not str(r[1].value).startswith("Total")]
    totals = [r for r in farid_rows if str(r[1].value).startswith("Total")]
    assert len(per_sprint) == 2  # ... one per sprint ...
    assert len(totals) == 1  # ... and exactly one total row.
    assert {r[1].value for r in per_sprint} == {"App Sprint 1", "Mkt Sprint 1"}
    # The row that matters is the total, and it is the correct sum (10 + 6).
    assert totals[0][3].value == 16.0


def test_the_unestimated_cell_renders_unknown_never_0_percent(monkeypatch):
    capture = _Capture(monkeypatch)
    df = pd.DataFrame(
        [
            _ticket("ML-1", "Mehdi", "ML Sprint 5", 3, "active", None, "2026-08-03", "2026-08-14"),
            _ticket("ML-2", "Mehdi", "ML Sprint 5", 3, "active", None, "2026-08-03", "2026-08-14"),
        ]
    )
    monkeypatch.setattr(rs, "WEEKLY_HOURS", {"Mehdi": 20.0})

    rs._render_hourly_capacity(df)

    assert len(capture.table) == 2
    _, coverage_rows = capture.table[1]
    mehdi_row = next(r for r in coverage_rows if r[1].value == "Mehdi")
    coverage_cell = mehdi_row[4].value
    assert coverage_cell == "unknown"
    assert coverage_cell != "0%"


# ---------------------------------------------------------------------------
# Eight board-hygiene bars, ghost-assigned as a legible zero
# ---------------------------------------------------------------------------


def test_eight_hygiene_bars_render_with_ghost_assigned_as_a_legible_zero(monkeypatch):
    capture = _Capture(monkeypatch)

    df = pd.DataFrame(
        [
            _ticket("APP-1", "Farid", "App Sprint 1", 1, "active", 10, "2026-08-03", "2026-08-14"),
        ]
    )
    monkeypatch.setattr(rs, "WEEKLY_HOURS", {"Farid": 20.0})

    rs._render_sprint_capacity.__wrapped__(df, triage_stuck_count=2)

    assert len(capture.hbars) == 1
    bars = capture.hbars[0]
    assert len(bars) == 8
    names = [bar.name for bar in bars]
    assert "Ghost-assigned" in names
    ghost = next(bar for bar in bars if bar.name == "Ghost-assigned")
    # No former-staff assignee is present in this board, so ghost-assigned is
    # a real, measured zero - it must still render as a labelled bar, not be
    # dropped or read as "n/a"/missing.
    assert ghost.value == "0"
    assert ghost.dim is False
    rendered = capture.rendered_html()
    assert "genuine zero" in rendered


def test_ghost_assigned_bar_shows_the_real_count_when_nonzero(monkeypatch):
    capture = _Capture(monkeypatch)
    df = pd.DataFrame(
        [
            _ticket("APP-1", "Farid", "App Sprint 1", 1, "active", 10, "2026-08-03", "2026-08-14"),
            _ticket("OLD-1", "Sai Shankar", "App Sprint 1", 1, "active", 5, "2026-08-03", "2026-08-14"),
        ]
    )
    monkeypatch.setattr(rs, "WEEKLY_HOURS", {"Farid": 20.0})

    rs._render_sprint_capacity.__wrapped__(df)

    bars = capture.hbars[0]
    ghost = next(bar for bar in bars if bar.name == "Ghost-assigned")
    assert ghost.value == "1"
    assert "OLD-1" in capture.rendered_html()


# ---------------------------------------------------------------------------
# The triage flow: still Streamlit-native, not migrated to the HTML kit.
# ---------------------------------------------------------------------------


def test_triage_card_uses_streamlit_widgets_not_the_html_kit(monkeypatch):
    calls = {"markdown": 0, "link_button": 0}
    monkeypatch.setattr(rs.st, "markdown", lambda *a, **k: calls.__setitem__("markdown", calls["markdown"] + 1))
    monkeypatch.setattr(
        rs.st, "link_button", lambda *a, **k: calls.__setitem__("link_button", calls["link_button"] + 1)
    )

    def _boom(*a, **k):
        raise AssertionError("theme_html must not be touched by the triage card")

    monkeypatch.setattr(theme_html, "render", _boom)
    monkeypatch.setattr(theme_html, "table", _boom)
    monkeypatch.setattr(theme_html, "hbars", _boom)
    monkeypatch.setattr(theme_html, "sprint_card", _boom)

    row = pd.Series(
        {
            "key": "VIN-1",
            "summary": "Checkout fails",
            "assignee": "",
            "status": "Backlog",
            "priority": "",
            "issue_type": "Bug",
            "epic_summary": "",
            "ticket_age_days": 400,
            "idle_days": 200,
            "suggested": "Close",
            "why": "old and idle",
        }
    )
    rs._render_triage_card(row)

    assert calls["markdown"] == 1  # the hand-rolled HTML card
    assert calls["link_button"] == 1  # the "Open in Jira" widget


def test_the_triage_flow_source_has_no_html_kit_dependency():
    """Structural guarantee that Task 3E did not touch the triage flow.

    ``_render_cleanup``/``_render_triage_card`` are outside this task's file
    ownership and MODE: HYBRID says they stay Streamlit-native - reading their
    source is a stronger, more durable check than one captured render, since
    it fails the day anyone routes either function through ``theme_html``.
    """
    source = inspect.getsource(rs._render_triage_card) + inspect.getsource(rs._render_cleanup)
    assert "theme_html" not in source
    assert ".button(" in source
    assert "st.progress(" in source
    assert "st.link_button(" in source


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
