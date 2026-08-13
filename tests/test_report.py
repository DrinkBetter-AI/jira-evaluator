"""Offline checks for the printable tab report.

Kept outside the repository like the other check scripts: it exercises the
module directly, with no Streamlit running.

    python3 -m pytest tests/test_report.py -q
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import report  # noqa: E402

WHEN = dt.datetime(2026, 8, 6, 9, 30, tzinfo=dt.timezone.utc)


def test_a_report_that_gathered_nothing_is_empty():
    assert report.Report("Business").empty


def test_a_tile_that_read_nothing_is_left_out_rather_than_printed_as_a_dash():
    built = report.Report("Business")
    built.figure("Ads", "Commission per dollar spent", "\u2014")
    assert built.empty


def test_the_page_carries_every_figure_and_the_sentence_that_explains_it():
    built = report.Report("Business")
    built.figure("Ads", "Spend (30d)", "$176.00", "+$20.00")
    built.note("Ads", "**$0.84 back** for every dollar of ad spend.")
    page = built.html(now=WHEN)

    assert "Spend (30d)" in page
    assert "$176.00" in page
    assert "+$20.00" in page
    # The figure a sentence is about stays bold on paper, and the markdown
    # asterisks that carried it do not.
    assert "<strong>$0.84 back</strong>" in page
    assert "**" not in page
    assert "06 August 2026, 09:30 UTC" in page


def test_a_section_that_gathered_nothing_prints_no_empty_heading():
    built = report.Report("Engineering")
    built.figure("Ticket health", "Open tickets", "42")
    built.note("PR hygiene", "")
    page = built.html(now=WHEN)

    assert "Ticket health" in page
    assert "PR hygiene" not in page


def test_a_value_from_the_dashboard_cannot_inject_markup_into_the_page():
    built = report.Report("Engineering")
    built.figure("Ticket health", "Owner", "<script>alert(1)</script>")
    page = built.html(now=WHEN)

    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_sections_print_in_the_order_the_tab_drew_them():
    built = report.Report("Business")
    built.figure("Orders", "Orders (7 days)", "12")
    built.figure("Ads", "Spend (30d)", "$176.00")
    built.figure("Orders", "Revenue (7 days)", "$1,226.00")
    page = built.html(now=WHEN)

    assert page.index("Orders</h2>") < page.index("Ads</h2>")
    # A section reopened later keeps its first position and gathers into it.
    assert "Revenue (7 days)" in page.split("Ads</h2>")[0]


def test_the_file_is_named_for_the_tab_and_the_day_it_was_taken():
    assert report.Report("Business").filename(now=WHEN) == "business-2026-08-06.html"
    assert (
        report.Report("Engineering").filename(now=WHEN) == "engineering-2026-08-06.html"
    )
