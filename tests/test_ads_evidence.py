"""Offline checks for the per-wine ad ledger: no BigQuery, no Merchant Center.

Every figure here is one a CTO is expected to argue with, so what is tested is
the arithmetic and the joins rather than the rendering: which wines a claim
counts, what happens to spend on a wine Google cannot benchmark, and that a
claim about return per dollar is never made from a window with no spend in it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ads_evidence as ae  # noqa: E402
import merchant_client as mc  # noqa: E402


def prices(rows: list[tuple[str, float, float]]) -> mc.Prices:
    """Offers with their price and the market's, the gap worked out as read."""
    frame = pd.DataFrame(rows, columns=["offer", "price", "benchmark"])
    frame["title"] = "Wine " + frame["offer"]
    frame["gap"] = frame["price"] / frame["benchmark"] - 1
    return mc.Prices(frame, "USD")


def sales(rows: list[tuple[str, int, float]]) -> mc.Sales:
    return mc.Sales(pd.DataFrame(rows, columns=["offer", "bottles", "revenue"]), 90)


def unread() -> mc.Sales:
    """The order book as it arrives when Postgres cannot be reached at all."""
    return mc.Sales(pd.DataFrame(), 90, read=False)


def ads(rows: list[tuple[str, float, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["offer", "spend", "clicks"])
    return frame.assign(impressions=frame["clicks"] * 10, ad_conversions=0.0)


def test_a_wine_with_no_benchmark_keeps_its_spend():
    """Four offers in five have no benchmark; an inner join would lose them."""
    ledger = ae.ledger(
        ads([("a", 5.0, 10), ("unknown", 3.0, 6)]),
        prices([("a", 10.0, 10.0)]),
        sales([("a", 2, 20.0)]),
    )
    assert len(ledger) == 2
    assert float(ledger["spend"].sum()) == 8.0
    unpriced = ledger[ledger["offer"] == "unknown"].iloc[0]
    assert pd.isna(unpriced["gap"])
    assert int(unpriced["bottles"]) == 0


def test_spend_is_split_by_whether_the_wine_sold_at_all():
    ledger = ae.ledger(
        ads([("a", 5.0, 10), ("b", 7.0, 20), ("c", 1.0, 3)]),
        prices([("a", 10.0, 10.0), ("b", 20.0, 10.0), ("c", 9.0, 10.0)]),
        sales([("a", 2, 20.0)]),
    )
    split = ae.spend_split(ledger)
    sold = split[split["outcome"] == ae.SOLD].iloc[0]
    nothing = split[split["outcome"] == ae.NOTHING].iloc[0]
    assert (int(sold["wines"]), float(sold["spend"])) == (1, 5.0)
    assert (int(nothing["wines"]), float(nothing["spend"])) == (2, 8.0)
    assert float(nothing["revenue"]) == 0.0


def test_bands_use_the_same_edges_as_the_sales_evidence():
    """One panel calling a wine expensive and another not is how a merchant is
    told two different things about the same bottle."""
    ledger = ae.ledger(
        ads([("cheap", 1.0, 10), ("same", 1.0, 10), ("bit", 1.0, 10), ("lots", 1.0, 10)]),
        prices(
            [
                ("cheap", 7.0, 10.0),
                ("same", 10.0, 10.0),
                ("bit", 11.5, 10.0),
                ("lots", 20.0, 10.0),
            ]
        ),
        sales([("cheap", 4, 40.0)]),
    )
    bands = ae.by_band(ledger)
    assert list(bands["band"].astype(str)) == list(mc.BAND_NAMES)
    assert int(bands.loc[bands["band"] == mc.BAND_NAMES[0], "wines"].iloc[0]) == 1
    # $1 spent on the cheap wine returned $40 of it; the rest returned nothing.
    assert float(bands.loc[bands["band"] == mc.BAND_NAMES[0], "per_dollar"].iloc[0]) == 40
    assert float(bands.loc[bands["band"] == mc.BAND_NAMES[3], "per_dollar"].iloc[0]) == 0


def test_a_band_nobody_advertised_has_no_return_rather_than_zero():
    ledger = ae.ledger(
        ads([("cheap", 2.0, 10)]),
        prices([("cheap", 7.0, 10.0), ("lots", 20.0, 10.0)]),
        sales([("cheap", 1, 10.0)]),
    )
    bands = ae.by_band(ledger)
    empty = bands[bands["band"] == mc.BAND_NAMES[3]].iloc[0]
    assert int(empty["wines"]) == 0
    assert pd.isna(empty["per_dollar"])


def test_the_waste_list_is_expensive_clicked_and_unsold():
    ledger = ae.ledger(
        ads([("expensive", 5.0, 10), ("unclicked", 1.0, 0), ("sold", 2.0, 8)]),
        prices(
            [
                ("expensive", 20.0, 10.0),
                ("unclicked", 20.0, 10.0),
                ("sold", 20.0, 10.0),
            ]
        ),
        sales([("sold", 1, 20.0)]),
    )
    assert list(ae.waste(ledger)["offer"]) == ["expensive"]


def test_the_sale_price_feed_offers_the_market_price_and_nothing_lower():
    ledger = ae.ledger(
        ads([("a", 9.0, 10), ("b", 1.0, 2), ("keen", 4.0, 5)]),
        prices([("a", 20.0, 10.0), ("b", 30.0, 10.0), ("keen", 8.0, 10.0)]),
        sales([]),
    )
    feed = ae.sale_price_feed(ledger)
    # The keenly priced wine is left alone; the costliest is offered first.
    assert list(feed["id"]) == ["a", "b"]
    assert list(feed["sale_price"]) == [10.0, 10.0]
    assert list(ae.sale_price_feed(ledger, limit=1)["id"]) == ["a"]


def test_a_wine_google_cannot_benchmark_is_never_put_on_sale():
    """A suggested price needs a market price; without one there is no offer."""
    ledger = ae.ledger(ads([("unknown", 9.0, 10)]), prices([]), sales([]))
    assert ae.sale_price_feed(ledger).empty
    assert ae.by_band(ledger).empty


def test_most_clicked_is_by_clicks_rather_than_by_spend():
    ledger = ae.ledger(
        ads([("costly", 50.0, 3), ("popular", 2.0, 90)]),
        prices([("costly", 10.0, 10.0), ("popular", 10.0, 10.0)]),
        sales([]),
    )
    assert list(ae.most_clicked(ledger, 1)["offer"]) == ["popular"]


def test_the_claims_name_the_share_that_sold_nothing():
    ledger = ae.ledger(
        ads([("a", 25.0, 10), ("b", 75.0, 30)]),
        prices([("a", 10.0, 10.0), ("b", 10.0, 10.0)]),
        sales([("a", 1, 10.0)]),
    )
    said = " ".join(claim for _, claim in ae.verdicts(ledger))
    assert "75% of the ad spend went to 1 wines" not in said
    assert "sold nothing" in said
    assert "$75 of $100" in said


def test_nothing_is_claimed_about_a_window_with_no_spend_in_it():
    ledger = ae.ledger(
        ads([("a", 0.0, 0)]), prices([("a", 10.0, 10.0)]), sales([("a", 1, 10.0)])
    )
    assert ae.verdicts(ledger) == []
    assert ae.verdicts(ae.ledger(pd.DataFrame(), prices([]), sales([]))) == []


def test_an_unread_order_book_is_never_reported_as_wines_that_sold_nothing():
    """The mistake worth a test of its own: with Postgres down, every wine joins
    to no sale, and filled with zeroes that reads as the whole budget wasted."""
    ledger = ae.ledger(
        ads([("a", 25.0, 10), ("b", 75.0, 30)]),
        prices([("a", 20.0, 10.0), ("b", 10.0, 10.0)]),
        unread(),
    )
    assert not ae.sold_known(ledger)
    assert float(ledger["spend"].sum()) == 100.0
    assert ledger["bottles"].isna().all()
    assert ae.spend_split(ledger).empty
    assert ae.by_band(ledger)["per_dollar"].isna().all()
    assert ae.waste(ledger).empty
    tags = [tag for tag, _ in ae.verdicts(ledger)]
    assert ae.WASTED not in tags and ae.BY_PRICE not in tags
    # The feed does not need the order book: it is a price argument.
    assert list(ae.sale_price_feed(ledger)["id"]) == ["a"]


def test_each_claim_names_the_wines_it_is_about_rather_than_its_position():
    """The band claim is dropped whenever one band earned nothing, and the claim
    behind it must not inherit the wines that claim would have had."""
    ledger = ae.ledger(
        ads([("priced", 10.0, 10), ("unknown", 90.0, 40)]),
        prices([("priced", 20.0, 10.0)]),
        sales([]),
    )
    tags = [tag for tag, _ in ae.verdicts(ledger)]
    assert ae.BY_PRICE not in tags
    assert tags == [ae.WASTED, ae.NO_BENCHMARK]


def test_a_merchant_is_named_beside_each_wine_where_the_catalogue_knows_one():
    ledger = ae.ledger(
        ads([("a", 1.0, 1), ("b", 1.0, 1)]),
        prices([("a", 10.0, 10.0), ("b", 10.0, 10.0)]),
        sales([]),
        {"a": ("Yiannis", "Capital Fine Wine")},
    )
    named = dict(zip(ledger["offer"], ledger["merchant"]))
    assert named["a"] == "Yiannis, Capital Fine Wine"
    assert named["b"] == ""
