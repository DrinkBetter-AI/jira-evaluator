"""What the merchant's one-page picture must never get wrong.

It leaves the building: it is emailed to a shop and used to ask them for money,
so a slice that misstates their range, or a rate invented for a band nobody
clicked, is worse than no page at all.
"""

from __future__ import annotations

import datetime as _dt
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import merchant_letter as ml  # noqa: E402


def bands():
    return [
        ml.Band("Cheaper than the market", 12, 300, 60, 20.0),
        ml.Band("About the market", 4, 100, 5, 5.0),
        ml.Band("Up to 25% more expensive", 20, 400, 12, 3.0),
        ml.Band("More than 25% more expensive", 64, 200, 2, 1.0),
    ]


def test_every_band_gets_a_slice_and_its_own_colour():
    svg = ml.pie_svg(bands())
    assert svg.count("<path") == 4, svg
    for colour in ml.BAND_COLOURS:
        assert colour in svg, colour


def test_a_band_with_no_wines_is_not_drawn_as_a_hairline():
    empty = [ml.Band("Cheaper than the market", 0, 0, 0, None), bands()[3]]
    svg = ml.pie_svg(empty)
    assert svg.count("<path") == 1, svg


def test_one_band_holding_everything_is_a_whole_circle():
    svg = ml.pie_svg([ml.Band("More than 25% more expensive", 40, 100, 1, 1.0)])
    # An arc of exactly 360 degrees draws nothing, so that case is a circle.
    assert "a 108 108 0 1 1" in svg, svg


def test_the_ring_is_a_ring_rather_than_a_pie():
    assert "<circle" in ml.pie_svg(bands())
    # And an empty shop gets no stray hole floating on the page.
    assert "<circle" not in ml.pie_svg([])


def test_the_page_says_the_shares_are_of_the_compared_wines_only():
    # Google benchmarks only bottles other shops also sell, so "your range" on
    # its own would overstate how much of the catalogue is red.
    page = ml.one_pager("Yiannis", bands(), sales_days=90, demand_days=30)
    assert "the wines Google could compare" in page, page[-900:]
    assert "Every wine you list is compared" not in page


def test_a_shop_with_nothing_in_it_draws_no_ring_at_all():
    assert ml.pie_svg([]) == ""
    assert ml.pie_svg([ml.Band("Cheaper than the market", 0, 0, 0, None)]) == ""


def test_a_band_nobody_clicked_gets_no_bar_rather_than_a_zero_one():
    unclicked = bands()[:2] + [ml.Band("Up to 25% more expensive", 9, 0, 0, None)]
    bars = ml.bars_svg(unclicked)
    assert bars.count("<rect") == 2, bars
    assert "Up to 25% more expensive" not in bars, bars


def test_the_longest_bar_is_the_best_selling_band():
    widths = [float(w) for w in re.findall(r'<rect [^>]*width="([\d.]+)"', ml.bars_svg(bands()))]
    assert widths[0] == max(widths), widths
    assert widths[0] > widths[-1], widths


def test_the_headline_names_both_ends_in_bottles_a_merchant_recognises():
    said = ml.headline(bands())
    assert "20 bottles" in said, said
    assert said.endswith("sold 1.") or "sold 1." in said, said


def test_no_headline_is_claimed_when_an_end_of_it_never_sold():
    quiet = [
        ml.Band("Cheaper than the market", 12, 300, 0, 0.0),
        ml.Band("More than 25% more expensive", 64, 200, 2, 1.0),
    ]
    assert ml.headline(quiet) == ""
    assert ml.headline([]) == ""


def test_the_page_carries_the_shop_s_name_and_no_network_at_all():
    page = ml.one_pager(
        "Yiannis Wine Shop", bands(), sales_days=90, demand_days=30
    )
    assert "Yiannis Wine Shop" in page
    assert "<svg" in page
    for outside in ("http://", "https://", "<script"):
        assert outside not in page, outside


def test_a_name_with_a_comma_or_an_ampersand_survives_as_itself():
    page = ml.one_pager(
        "Black Bear Wines & Spirits, Inc", bands(), sales_days=90, demand_days=30
    )
    assert "Black Bear Wines &amp; Spirits, Inc" in page
    assert "&amp;amp;" not in page
    assert ml.filename("Black Bear Wines & Spirits, Inc", now=_dt.date(2026, 8, 1)) == (
        "black-bear-wines-spirits-inc-your-prices-2026-08-01.html"
    )


def test_the_windows_the_two_halves_come_from_are_on_the_page():
    page = ml.one_pager("Yiannis", bands(), sales_days=90, demand_days=30)
    assert "last 30 days" in page and "last 90 days" in page, page[-800:]
    # And that it is a comparison rather than a trial, which is the honest part.
    assert "rather than being a trial" in page


def test_a_page_with_no_rates_still_shows_the_range():
    blind = [ml.Band(band.name, band.listings, 0, 0, None) for band in bands()]
    page = ml.one_pager("Yiannis", blind, sales_days=90, demand_days=30)
    assert "What each of those sold" not in page
    assert "<svg" in page


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_"):
            case()
            print(f"{name}: ok")
    print("the merchant's page says only what the numbers say")
