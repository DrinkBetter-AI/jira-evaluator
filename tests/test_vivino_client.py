"""Offline checks for the Vivino price comparison.

No Vivino and no Streamlit: the fetch is exercised against a fake session that
serves canned explore pages, and the matching against frames built here.

    python3 -m pytest tests/test_vivino_client.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vivino_client as vv  # noqa: E402


def listing(name: str, year: int, price: float) -> pd.DataFrame:
    frame = pd.DataFrame([{"name": name, "year": year, "price": price}])
    frame["key"] = [vv.wine_key(name, year)]
    return frame


def shop(rows: list[tuple[str, int, float]], listed: int | None = None, complete=True):
    frame = pd.DataFrame(rows, columns=["name", "year", "price"])
    if not frame.empty:
        frame["key"] = [
            vv.wine_key(name, year) for name, year in zip(frame["name"], frame["year"])
        ]
    else:
        frame = pd.DataFrame(columns=["name", "year", "price", "key"])
    return vv.Shop(
        slug="a-shop",
        listings=frame,
        listed=listed if listed is not None else len(frame),
        complete=complete,
    )


def ours(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["title", "year", "price"])


# --- the key both catalogues are matched on ---------------------------------


def test_the_key_survives_accents_punctuation_and_case():
    assert vv.wine_key("Château Margaux", 2015) == vv.wine_key("chateau margaux", 2015)


def test_a_year_embedded_in_the_name_does_not_double_the_vintage():
    # Vivino names carry the vintage; ours may not. Both must land on one key.
    assert vv.wine_key("Opus One 2019", 2019) == vv.wine_key("Opus One", 2019)


def test_different_vintages_are_different_wines():
    assert vv.wine_key("Opus One", 2019) != vv.wine_key("Opus One", 2018)


def test_a_non_vintage_wine_only_matches_non_vintage():
    assert vv.wine_key("Krug Grande Cuvee", 0) != vv.wine_key("Krug Grande Cuvee", 2015)


def test_vivinos_nv_vintage_reads_as_no_year_rather_than_dying():
    assert vv.vintage_year("N.V.") == 0
    assert vv.vintage_year(None) == 0
    assert vv.vintage_year(2019) == 2019
    assert vv.vintage_year("2019") == 2019


def test_the_year_is_read_from_a_name_that_carries_one():
    assert vv.year_of("Opus One 2019") == 2019
    assert vv.year_of("Krug Grande Cuvee") == 0


# --- matching and the arithmetic on it ---------------------------------------


def test_a_matched_wine_reports_both_prices_and_the_gap():
    result = vv.compare(
        ours([("Quinta do Vallado Vintage Port", 2018, 128.09)]),
        shop([("Quinta Do Vallado Vintage Port 2018", 2018, 74.99)]),
    )
    assert result.matched == 1
    row = result.rows.iloc[0]
    assert row["ours"] == 128.09
    assert row["theirs"] == 74.99
    assert row["gap"] == pytest.approx((74.99 - 128.09) / 128.09)


def test_cheaper_same_and_dearer_are_split_at_the_rounding_margin():
    result = vv.compare(
        ours(
            [
                ("Wine A", 2020, 100.0),
                ("Wine B", 2020, 100.0),
                ("Wine C", 2020, 100.0),
            ]
        ),
        shop(
            [
                ("Wine A 2020", 2020, 80.0),  # cheaper there
                ("Wine B 2020", 2020, 100.5),  # the same price, differently rounded
                ("Wine C 2020", 2020, 120.0),  # more expensive there
            ]
        ),
    )
    assert result.matched == 3
    assert len(result.cheaper_there) == 1
    assert result.cheaper_there.iloc[0]["wine"] == "Wine A"
    assert result.same == 1
    assert result.dearer_there == 1


def test_our_wines_with_no_vivino_listing_are_counted_not_guessed():
    result = vv.compare(
        ours([("Wine A", 2020, 50.0), ("Wine B", 2021, 60.0)]),
        shop([("Wine A 2020", 2020, 50.0)]),
    )
    assert result.matched == 1
    assert result.unmatched_ours == 1


def test_an_empty_vivino_shop_matches_nothing_and_says_how_many_it_left_out():
    result = vv.compare(ours([("Wine A", 2020, 50.0)]), shop([]))
    assert result.matched == 0
    assert result.unmatched_ours == 1


def test_duplicate_listings_of_one_wine_compare_at_the_cheapest():
    result = vv.compare(
        ours([("Wine A", 2020, 90.0), ("Wine A", 2020, 80.0)]),
        shop([("Wine A 2020", 2020, 70.0)]),
    )
    assert result.matched == 1
    assert result.rows.iloc[0]["ours"] == 80.0


def test_the_worst_gap_leads_the_cheaper_list():
    result = vv.compare(
        ours([("Wine A", 2020, 100.0), ("Wine B", 2020, 100.0)]),
        shop([("Wine A 2020", 2020, 90.0), ("Wine B 2020", 2020, 60.0)]),
    )
    assert result.cheaper_there.iloc[0]["wine"] == "Wine B"


# --- reading the shop from the feed ------------------------------------------


class FakeReply:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload

    @property
    def text(self):
        return self.payload


class FakeSession:
    """Serves the shop page, then explore pages keyed on (price_from, page)."""

    def __init__(self, pages: dict, merchant: int | None = 26035):
        self.pages = pages
        self.merchant = merchant
        self.calls = 0

    def get(self, url, **kwargs):
        if "merchants/" in url:
            body = f"merchant_id&quot;:{self.merchant}" if self.merchant else "<html>"
            return FakeReply(body)
        self.calls += 1
        params = kwargs["params"]
        key = (params["price_range_min"], params["page"])
        return FakeReply({"explore_vintage": self.pages.get(key, {"matches": []})})


def match(name: str, year: int, price: float, volume=750, vintage_id=None, quantity=1):
    return {
        "vintage": {"id": vintage_id or hash((name, year)) % 10**6, "name": name, "year": year},
        "price": {
            "amount": price,
            "bottle_type": {"volume_ml": volume},
            "bottle_quantity": quantity,
        },
    }


def test_only_single_standard_bottles_are_kept():
    session = FakeSession(
        {
            (0.0, 1): {
                "records_matched": 3,
                "matches": [
                    match("Wine A 2020", 2020, 50.0),
                    match("Magnum B 2020", 2020, 90.0, volume=1500),
                    match("Half C 2020", 2020, 20.0, volume=375),
                ],
            }
        }
    )
    read = vv.fetch_shop("a-shop", session)
    assert list(read.listings["name"]) == ["Wine A 2020"]
    assert read.listed == 3


def test_the_read_walks_pages_and_then_the_price_axis():
    # Page 2 at price 0 is empty although more wines exist - the feed's page
    # ceiling - so the read must start again from the highest price it saw.
    session = FakeSession(
        {
            (0.0, 1): {
                "records_matched": 2,
                "matches": [match("Wine A 2020", 2020, 50.0)],
            },
            (50.0, 1): {
                "records_matched": 2,
                "matches": [match("Wine B 2020", 2020, 75.0)],
            },
        }
    )
    read = vv.fetch_shop("a-shop", session)
    assert sorted(read.listings["name"]) == ["Wine A 2020", "Wine B 2020"]


def test_a_case_price_quoted_per_bottle_is_not_a_single_bottle_price():
    # A merchant can quote Vivino $9 a bottle for a 12-bottle case; comparing
    # that against a single bottle's price here is apples and oranges, so
    # only prices a shopper pays for one bottle are kept.
    session = FakeSession(
        {
            (0.0, 1): {
                "records_matched": 3,
                "matches": [
                    match("Wine A 2020", 2020, 9.0, quantity=12),
                    match("Wine B 2020", 2020, 12.0, quantity=6),
                    match("Wine C 2020", 2020, 17.0),
                ],
            },
            (17.0, 1): {"records_matched": 3, "matches": []},
        }
    )
    read = vv.fetch_shop("a-shop", session)
    assert list(read.listings["name"]) == ["Wine C 2020"]
    assert read.packs == 2
    assert read.complete


def test_a_wine_sold_both_singly_and_by_the_case_is_compared_not_left_out():
    session = FakeSession(
        {
            (0.0, 1): {
                "records_matched": 2,
                "matches": [
                    match("Wine A 2020", 2020, 9.0, vintage_id=7, quantity=12),
                    match("Wine A 2020", 2020, 17.0, vintage_id=7),
                ],
            },
            (17.0, 1): {"records_matched": 2, "matches": []},
        }
    )
    read = vv.fetch_shop("a-shop", session)
    assert list(read.listings["price"]) == [17.0]
    assert read.packs == 0


def test_no_match_names_the_pack_prices_rather_than_blaming_the_overlap():
    # A shop quoted entirely per bottle of a case has nothing comparable, and
    # the verdict must say so instead of claiming the cellars fail to overlap.
    ours = pd.DataFrame({"title": ["Wine A 2020"], "year": [2020], "price": [20.0]})
    result = vv.compare(
        ours,
        vv.Shop(slug="a-shop", listings=pd.DataFrame(), listed=40, complete=True, packs=40),
    )
    (line,) = vv.verdicts("A Shop", result)
    assert "40 of them are priced per bottle" in line
    assert "should overlap" not in line


def test_a_partial_no_match_read_admits_the_shop_was_not_fully_read():
    # A read that failed after seeing only pack prices must not present the
    # missing overlap as a finding about the whole shop.
    ours = pd.DataFrame({"title": ["Wine A 2020"], "year": [2020], "price": [20.0]})
    result = vv.compare(
        ours,
        vv.Shop(slug="a-shop", listings=pd.DataFrame(), listed=40, complete=False, packs=12),
    )
    lines = vv.verdicts("A Shop", result)
    assert any("stopped before the whole shop was read" in line for line in lines)


def test_a_blank_year_field_falls_back_to_the_vintage_in_the_name():
    # Some listings leave the vintage field empty though the name carries the
    # year; the bottle must still match the shop's own by name and vintage.
    session = FakeSession(
        {
            (0.0, 1): {
                "records_matched": 1,
                "matches": [match("Opus One 2019", None, 60.0)],
            },
            (60.0, 1): {"records_matched": 1, "matches": []},
        }
    )
    read = vv.fetch_shop("a-shop", session)
    assert list(read.listings["year"]) == [2019]


def test_a_spent_time_budget_keeps_what_was_read_and_admits_the_rest(monkeypatch):
    # A slow feed must end the read with the listings in hand, marked partial,
    # rather than run past the server's request limit and lose everything.
    session = FakeSession(
        {
            (0.0, 1): {
                "records_matched": 2,
                "matches": [match("Wine A 2020", 2020, 50.0)],
            },
            (50.0, 1): {
                "records_matched": 2,
                "matches": [match("Wine B 2020", 2020, 75.0)],
            },
        }
    )
    ticks = iter([0.0, 1.0, 1.0, 10.0])
    monkeypatch.setattr(vv.time, "monotonic", lambda: next(ticks, 10.0))
    monkeypatch.setattr(vv, "_TIME_BUDGET_SECONDS", 5)
    read = vv.fetch_shop("a-shop", session)
    assert list(read.listings["name"]) == ["Wine A 2020"]
    assert not read.complete


def test_a_band_of_magnums_does_not_end_the_read_or_fake_completeness():
    # A walk can serve nothing comparable - magnums, halves - while raising
    # the price floor; the read must climb past them to the standard bottles
    # above, not stop there and call the shop fully read.
    session = FakeSession(
        {
            (0.0, 1): {
                "records_matched": 3,
                "matches": [match("Wine A 2020", 2020, 50.0)],
            },
            (50.0, 1): {
                "records_matched": 3,
                "matches": [
                    match("Wine A 2020", 2020, 50.0),
                    match("Magnum B 2020", 2020, 90.0, volume=1500),
                ],
            },
            (90.0, 1): {
                "records_matched": 3,
                "matches": [match("Wine C 2020", 2020, 120.0)],
            },
            (120.0, 1): {
                "records_matched": 3,
                "matches": [match("Wine C 2020", 2020, 120.0)],
            },
            (120.0, 2): {"records_matched": 3, "matches": []},
        }
    )
    read = vv.fetch_shop("a-shop", session)
    assert sorted(read.listings["name"]) == ["Wine A 2020", "Wine C 2020"]
    assert read.complete


def test_the_price_floor_holds_still_while_a_walk_turns_its_pages():
    # Raising the floor between pages would shift the result set under the
    # page number and skip listings for good; page 2 must be asked for at the
    # price the walk started from.
    session = FakeSession(
        {
            (0.0, 1): {
                "records_matched": 3,
                "matches": [
                    match("Wine A 2020", 2020, 10.0),
                    match("Wine B 2020", 2020, 20.0),
                ],
            },
            (0.0, 2): {
                "records_matched": 3,
                "matches": [match("Wine C 2020", 2020, 30.0)],
            },
        }
    )
    read = vv.fetch_shop("a-shop", session)
    assert sorted(read.listings["name"]) == [
        "Wine A 2020",
        "Wine B 2020",
        "Wine C 2020",
    ]
    assert read.complete


def test_a_walk_that_drains_with_nothing_new_is_the_top_of_the_shop():
    # The boundary walk re-serves only the top-priced listing already read and
    # then runs out of pages: everything at or above the floor is known, so
    # the read is complete rather than caveated on every real shop.
    session = FakeSession(
        {
            (0.0, 1): {
                "records_matched": 1,
                "matches": [match("Wine A 2020", 2020, 50.0)],
            },
            (50.0, 1): {
                "records_matched": 1,
                "matches": [match("Wine A 2020", 2020, 50.0)],
            },
            (50.0, 2): {"records_matched": 1, "matches": []},
        }
    )
    read = vv.fetch_shop("a-shop", session)
    assert read.complete
    assert list(read.listings["name"]) == ["Wine A 2020"]


def test_a_shop_with_no_listings_reads_as_empty_and_complete():
    session = FakeSession({(0.0, 1): {"records_matched": 0, "matches": []}})
    read = vv.fetch_shop("a-shop", session)
    assert read.listings.empty
    assert read.listed == 0
    assert read.complete


def test_a_page_without_a_merchant_id_is_a_failed_read_not_an_empty_shop():
    with pytest.raises(vv.VivinoError):
        vv.fetch_shop("a-shop", FakeSession({}, merchant=None))


def test_a_deployment_with_a_proxy_sends_vivino_traffic_through_it(monkeypatch):
    monkeypatch.setenv("VIVINO_PROXY", "http://user:pw@10.0.0.5:8899")
    session = vv._session()
    assert session.proxies == {
        "http": "http://user:pw@10.0.0.5:8899",
        "https": "http://user:pw@10.0.0.5:8899",
    }


def test_a_failure_through_the_proxy_never_names_its_credential():
    class RefusingSession(FakeSession):
        def get(self, url, **kwargs):
            raise requests.ConnectionError(
                "Cannot connect to proxy http://user:secret@10.0.0.5:8899"
            )

    with pytest.raises(vv.VivinoError) as caught:
        vv.fetch_shop("unknown-shop", RefusingSession({}))
    assert "secret" not in str(caught.value)
    assert "10.0.0.5" in str(caught.value)


def test_without_a_proxy_vivino_is_reached_directly(monkeypatch):
    monkeypatch.delenv("VIVINO_PROXY", raising=False)
    assert vv._session().proxies == {}


def test_a_shop_already_known_is_not_asked_for_its_page_again():
    # Vivino's shop pages refuse cloud addresses (403) more readily than the
    # listings API, so a known merchant id must not depend on the page at all.
    class PageRefusingSession(FakeSession):
        def get(self, url, **kwargs):
            if "merchants/" in url:
                raise requests.HTTPError("403 Client Error: Forbidden")
            return super().get(url, **kwargs)

    session = PageRefusingSession(
        {
            (0.0, 1): {"records_matched": 1, "matches": [match("Wine A 2020", 2020, 9.0)]},
            (9.0, 1): {"records_matched": 1, "matches": []},
        }
    )
    read = vv.fetch_shop("yiannis-wine-shop", session)
    assert list(read.listings["name"]) == ["Wine A 2020"]


def test_a_renumbered_merchant_is_followed_when_the_page_can_be_read():
    # The page names the shop's current id and wins over the table, so a
    # renumbering never silently reads the wrong merchant's prices.
    class RenumberedSession(FakeSession):
        def get(self, url, **kwargs):
            if "merchants/" in url:
                return FakeReply("merchant_id&quot;:999")
            if kwargs["params"]["merchant_id"] != 999:
                return FakeReply({"explore_vintage": {"records_matched": 0, "matches": []}})
            return super().get(url, **kwargs)

    session = RenumberedSession(
        {
            (0.0, 1): {"records_matched": 1, "matches": [match("Wine A 2020", 2020, 9.0)]},
            (9.0, 1): {"records_matched": 1, "matches": []},
        }
    )
    read = vv.fetch_shop("yiannis-wine-shop", session)
    assert list(read.listings["name"]) == ["Wine A 2020"]


def test_a_feed_that_refuses_outright_raises_rather_than_reporting_nothing():
    class RefusingSession(FakeSession):
        def get(self, url, **kwargs):
            if "merchants/" in url:
                return super().get(url, **kwargs)
            raise requests.ConnectionError("refused")

    with pytest.raises(vv.VivinoError):
        vv.fetch_shop("a-shop", RefusingSession({}))


def test_a_feed_that_dies_mid_read_keeps_what_it_read_and_says_so():
    class DyingSession(FakeSession):
        def get(self, url, **kwargs):
            if "merchants/" in url:
                return super().get(url, **kwargs)
            if self.calls >= 1:
                raise requests.ConnectionError("gone")
            return super().get(url, **kwargs)

    session = DyingSession(
        {
            (0.0, 1): {
                "records_matched": 5,
                "matches": [match("Wine A 2020", 2020, 50.0)],
            }
        }
    )
    read = vv.fetch_shop("a-shop", session)
    assert list(read.listings["name"]) == ["Wine A 2020"]
    assert not read.complete


# --- the sentences put in front of the merchant ------------------------------


def test_an_empty_shop_is_reported_as_nothing_to_compare():
    lines = vv.verdicts("Capital Fine Wine", vv.compare(ours([]), shop([], listed=0)))
    assert any("no live listings" in line for line in lines)


def test_no_matches_across_a_stocked_shop_is_flagged_not_celebrated():
    result = vv.compare(
        ours([("Wine A", 2020, 50.0)]), shop([("Other 2019", 2019, 9.0)], listed=400)
    )
    lines = vv.verdicts("Yiannis Wine Shop", result)
    assert any("matched" in line.lower() for line in lines)


def test_an_unread_catalogue_is_admitted_rather_than_blamed_on_matching():
    # Our side had nothing priced to compare, which is a catalogue problem to
    # fix, not a finding that the merchant's two cellars share no wines.
    result = vv.compare(ours([]), shop([("Other 2019", 2019, 9.0)], listed=400))
    lines = vv.verdicts("Yiannis Wine Shop", result)
    assert any("nothing on our side" in line for line in lines)
    assert not any("should overlap" in line for line in lines)


def test_wines_cheaper_on_vivino_lead_the_verdict():
    result = vv.compare(
        ours([("Wine A", 2020, 100.0)]),
        shop([("Wine A 2020", 2020, 60.0)], listed=10),
    )
    lines = vv.verdicts("Yiannis Wine Shop", result)
    assert "cheaper on" in lines[0]
    assert "Wine A" in lines[0]


def test_a_clean_sheet_says_so_without_a_cheaper_list():
    result = vv.compare(
        ours([("Wine A", 2020, 100.0)]),
        shop([("Wine A 2020", 2020, 100.0)], listed=10),
    )
    lines = vv.verdicts("Yiannis Wine Shop", result)
    assert "no better price" in lines[0]


def test_a_partial_read_is_admitted_in_the_verdict():
    result = vv.compare(
        ours([("Wine A", 2020, 100.0)]),
        shop([("Wine A 2020", 2020, 100.0)], listed=10, complete=False),
    )
    lines = vv.verdicts("Yiannis Wine Shop", result)
    assert any("stopped before" in line for line in lines)
