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


def _has_browser() -> bool:
    """Whether a browser the chart renderer can drive is on this machine."""
    import os
    import shutil

    named = os.environ.get("BROWSER_PATH")
    if named and os.path.exists(named):
        return True
    return any(
        shutil.which(name)
        for name in ("chromium", "chromium-browser", "google-chrome", "chrome")
    )


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


def test_an_assignee_named_like_markup_is_text_even_as_a_row_label():
    frame = pd.DataFrame({"tickets": [3]}, index=["<img src=x onerror=1>"])
    styled = frame.style.set_properties(**{"background-color": "#fee"})
    page = board(("dataframe", (styled,), {})).html(now=WHEN)

    assert "<img src=x" not in page
    assert "&lt;img src=x" in page


def test_a_column_is_called_what_the_screen_calls_it_and_hidden_where_it_hides():
    import streamlit as st

    frame = pd.DataFrame({"key_url": ["http://j/MB-1"], "idle_days": [3.0], "tier": ["Red"]})
    page = board(
        (
            "dataframe",
            (frame,),
            {
                "hide_index": True,
                "column_config": {
                    "idle_days": st.column_config.NumberColumn("Idle (days)", format="%.1f"),
                    "key_url": None,
                },
            },
        )
    ).html(now=WHEN)

    assert "Idle (days)" in page
    assert "idle_days" not in page
    assert "key_url" not in page and "http://j/MB-1" not in page


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
    # A chart image is drawn by a headless browser, which is in the deployed
    # image but not on every machine that runs the checks.
    pytest.importorskip("kaleido")
    if not _has_browser():
        pytest.skip("no browser for the chart renderer to draw in")
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


def test_the_printed_board_fetches_nothing_from_outside_itself():
    """A typesetter given an address goes and gets it; the board carries its own."""
    with pytest.raises(ValueError):
        snapshot._inline_only("file:///etc/passwd")
    with pytest.raises(ValueError):
        snapshot._inline_only("http://169.254.169.254/")


def test_a_line_between_sections_survives_the_tiles_above_it():
    page = board(
        ("divider", (), {}),
        ("metric", ("Open tickets", 13), {}),
        ("divider", (), {}),
        ("header", ("Ticket health",), {}),
    ).html(now=WHEN)

    assert page.count("<hr>") == 2


def test_a_board_that_cannot_be_laid_out_is_offered_as_the_page(monkeypatch):
    """A PDF library that fails mid-layout must not lose the reader their board."""

    class Refuses:
        def __init__(self, *args, **kwargs):
            pass

        def write_pdf(self):
            raise RuntimeError("no such font")

    monkeypatch.setattr(snapshot, "_weasyprint", lambda: Refuses)

    assert snapshot.to_pdf("<html></html>") is None


def test_a_link_with_search_terms_in_it_still_points_where_it_pointed():
    page = board(("markdown", ("[Open in Jira](https://j/issues?jql=a=1&b=2)",), {})).html(now=WHEN)

    assert 'href="https://j/issues?jql=a=1&amp;b=2"' in page
    assert "&amp;amp;" not in page


def test_a_line_the_page_wrote_once_is_printed_once():
    """``st.write`` draws by calling markdown, and both must not be kept."""
    import streamlit as st

    with snapshot.recording("Engineering") as recorded:
        st.write("Closing MB-1, MB-2")

    assert [block.text for block in recorded.blocks] == ["Closing MB-1, MB-2"]


def test_two_readers_at_once_each_get_their_own_whole_board():
    """One session finishing must not stop another's page being recorded."""
    import threading

    import streamlit as st

    started = threading.Event()
    finished = threading.Event()
    theirs: list[snapshot.Snapshot] = []

    def other() -> None:
        with snapshot.recording("Business") as recorded:
            st.header("Orders")
            started.set()
            theirs.append(recorded)
        finished.set()

    with snapshot.recording("Engineering") as mine:
        st.header("Before")
        reader = threading.Thread(target=other)
        reader.start()
        started.wait(5)
        finished.wait(5)
        st.header("After")
    reader.join()

    assert [block.text for block in mine.blocks] == ["Before", "After"]
    assert [block.text for block in theirs[0].blocks] == ["Orders"]


def test_nothing_the_sidebar_draws_belongs_to_the_board():
    class Container:
        def __init__(self, root: int) -> None:
            self._root_container = root

    assert snapshot._in_sidebar(Container(1))
    assert not snapshot._in_sidebar(Container(0))
