"""Offline checks for the whole-board snapshot.

Offline like the other check scripts: the recorder is fed the calls a page
would make, with no Streamlit running, and the file it produces is read.

    python3 -m pytest tests/test_snapshot.py -q
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import snapshot  # noqa: E402

WHEN = dt.datetime(2026, 8, 6, 9, 30, tzinfo=dt.timezone.utc)


def board(*calls) -> snapshot.Snapshot:
    """A recording of the given ``(call, args, kwargs)`` drawings."""
    recorded = snapshot.Snapshot("Engineering")
    for call, args, kwargs in calls:
        recorded.observe(call, args, kwargs)
    return recorded


def test_a_page_that_drew_nothing_but_rules_is_empty():
    assert board().empty
    assert board(("divider", (), {})).empty
    assert not board(("header", ("Ticket health",), {})).empty


def test_the_board_keeps_the_page_in_the_order_it_was_drawn():
    page = board(
        ("title", ("Engineering board",), {}),
        ("header", ("Ticket health",), {}),
        ("metric", ("Open tickets", 13), {}),
        ("caption", ("Follows the sidebar scope.",), {}),
    ).html(now=WHEN)

    assert page.index("Engineering board") < page.index("Ticket health")
    assert page.index("Ticket health") < page.index("Open tickets")
    assert page.index("Open tickets") < page.index("Follows the sidebar scope.")


def test_a_row_of_tiles_carries_its_labels_values_and_deltas():
    page = board(
        ("metric", ("Open tickets", 13), {"delta": "+2"}),
        ("metric", ("Stalled 30d+", 3), {}),
    ).html(now=WHEN)

    for expected in ("Open tickets", "13", "+2", "Stalled 30d+"):
        assert expected in page
    # One row on screen is one row on paper, not two stacks.
    assert page.count('class="tiles"') == 1


def test_a_table_is_printed_with_its_rows_and_its_headings():
    frame = pd.DataFrame({"key": ["MB-1", "MB-2"], "idle_days": [3, 40]})
    page = board(("dataframe", (frame,), {"hide_index": True})).html(now=WHEN)

    assert "<table" in page
    assert "MB-2" in page and "idle_days" in page


def test_a_ticket_titled_like_markup_is_text_on_paper_and_not_markup():
    frame = pd.DataFrame({"summary": ["<script>alert(1)</script> broke"]})
    styled = frame.style.set_properties(**{"background-color": "#fee"})
    page = board(("dataframe", (styled,), {})).html(now=WHEN)

    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_a_table_longer_than_the_page_says_how_much_of_it_is_shown():
    frame = pd.DataFrame({"key": [f"MB-{n}" for n in range(snapshot.MAX_ROWS + 40)]})
    page = board(("dataframe", (frame,), {})).html(now=WHEN)

    assert f"First {snapshot.MAX_ROWS} of {snapshot.MAX_ROWS + 40} rows." in page


def test_the_page_s_own_cards_are_kept_as_the_cards_they_are():
    card = '<div class="kpi-strip"><div class="kpi-card">Ready</div></div>'
    page = board(("markdown", (card,), {"unsafe_allow_html": True})).html(now=WHEN)

    assert 'class="kpi-card"' in page


def test_markdown_written_for_the_screen_reads_as_prose_on_paper():
    page = board(
        (
            "markdown",
            ("**\\$2,402 bought 120 orders** - read `google_ads_live` for it.",),
            {},
        ),
        ("markdown", ("- first thing\n- second thing",), {}),
    ).html(now=WHEN)

    assert "<strong>$2,402 bought 120 orders</strong>" in page
    assert "\\$" not in page
    assert "<code>google_ads_live</code>" in page
    assert page.count("<li>") == 2


def test_a_tinted_note_is_tinted_the_way_the_screen_tints_it():
    page = board(
        ("info", ("Jira editing is off.",), {}),
        ("warning", ("No estimate on 4 tickets.",), {}),
    ).html(now=WHEN)

    assert 'class="note info"' in page
    assert 'class="note warning"' in page


def test_a_section_folded_behind_a_label_still_says_what_it_was_called():
    page = board(
        ("expander", ("Change History and Revert",), {}),
        ("tabs", (["Top wines", "By merchant"],), {}),
    ).html(now=WHEN)

    assert "Change History and Revert" in page
    assert "Top wines" in page and "By merchant" in page


def test_a_chart_is_drawn_into_the_page_as_a_picture():
    frame = pd.DataFrame({"day": [1, 2, 3], "orders": [4, 9, 2]})
    page = board(("bar_chart", (frame,), {"x": "day", "y": "orders"})).html(now=WHEN)

    assert "data:image/png;base64," in page


def test_a_board_with_no_chart_renderer_loses_the_chart_and_not_the_page(monkeypatch):
    monkeypatch.setattr(snapshot, "_printable", lambda figure: (_ for _ in ()).throw(RuntimeError))
    frame = pd.DataFrame({"day": [1, 2], "orders": [4, 9]})
    page = board(
        ("header", ("Orders",), {}),
        ("bar_chart", (frame,), {}),
    ).html(now=WHEN)

    assert "Orders" in page
    assert "chart missing" in page


def test_the_file_is_named_for_the_page_and_the_day():
    recorded = snapshot.Snapshot("Business")
    assert recorded.filename("pdf", now=WHEN) == "business-board-2026-08-06.pdf"
    assert recorded.filename("html", now=WHEN) == "business-board-2026-08-06.html"


def test_a_deployment_with_no_pdf_library_is_offered_the_page_itself(monkeypatch):
    monkeypatch.setattr(snapshot, "_weasyprint", lambda: None)
    assert snapshot.to_pdf("<html><body>x</body></html>") is None


def test_the_board_is_a_pdf_a_reader_can_open():
    pytest.importorskip("weasyprint")
    frame = pd.DataFrame({"key": ["MB-1"], "idle_days": [3]})
    printed = board(
        ("title", ("Engineering board",), {}),
        ("metric", ("Open tickets", 13), {}),
        ("dataframe", (frame,), {}),
    ).pdf(now=WHEN)

    assert printed and printed.startswith(b"%PDF-")


def test_nothing_the_sidebar_draws_belongs_to_the_board():
    class Container:
        def __init__(self, root: int) -> None:
            self._root_container = root

    assert snapshot._in_sidebar(Container(1))
    assert not snapshot._in_sidebar(Container(0))
