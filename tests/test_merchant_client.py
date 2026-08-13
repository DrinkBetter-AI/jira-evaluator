"""Offline checks for merchant_client: no network.

The reports themselves are stubbed; what is tested is the shaping - which offers
count as dearer, which currency wins, what the sentences claim - because that is
where a wrong figure would be quoted in a leadership meeting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import merchant_client as mc  # noqa: E402

ACCOUNT = mc.Merchant("5304150122")


def micros(amount: float, currency: str = "USD") -> dict:
    return {"amountMicros": str(int(amount * mc.MICROS)), "currencyCode": currency}


def view(
    offer: str,
    price: float,
    benchmark: float,
    currency: str = "USD",
    title: str | None = None,
) -> dict:
    return {
        "priceCompetitivenessProductView": {
            "offerId": offer,
            "title": title or f"Wine {offer}",
            "brand": "Brand",
            "price": micros(price, currency),
            "benchmarkPrice": micros(benchmark, currency),
        }
    }


def stub_search(monkeypatch, rows: list[dict], truncated: bool = False) -> None:
    monkeypatch.setattr(mc, "_search", lambda *_args, **_kw: (rows, truncated))


def test_the_account_id_must_be_the_numeric_merchant_id(monkeypatch):
    """The obvious wrong answer is the account's name, and it goes in a URL."""
    monkeypatch.setenv("GOOGLE_MERCHANT_ID", "vinovoss")
    with pytest.raises(mc.MerchantConfigError, match="numeric"):
        mc.load_merchant_env()


def test_no_merchant_id_is_not_an_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_MERCHANT_ID", raising=False)
    assert mc.load_merchant_env() is None
    monkeypatch.setenv("GOOGLE_MERCHANT_ID", " 5304150122 ")
    monkeypatch.delenv("GCP_BIGQUERY_READONLY_KEY", raising=False)
    assert mc.load_merchant_env() == mc.Merchant("5304150122", None)


def test_dearer_is_counted_against_the_benchmark(monkeypatch):
    stub_search(
        monkeypatch,
        [
            view("a", 20.0, 10.0),
            view("b", 12.0, 10.0),
            view("c", 8.0, 10.0),
            view("d", 10.0, 10.0),
        ],
    )
    prices = mc.price_gaps(ACCOUNT, "token")
    assert (prices.counted, prices.dearer, prices.cheaper) == (4, 2, 1)
    assert prices.dear_share == 0.5
    assert prices.median_gap == pytest.approx(0.1)


def test_a_penny_apart_is_the_same_price(monkeypatch):
    """A rounding difference is not a pricing problem, and there are thousands."""
    stub_search(monkeypatch, [view("a", 10.01, 10.0), view("b", 9.99, 10.0)])
    prices = mc.price_gaps(ACCOUNT, "token")
    assert (prices.dearer, prices.cheaper) == (0, 0)


def test_an_offer_with_no_benchmark_is_not_competitive(monkeypatch):
    """Google publishes one only where others sell it; nought is not a price."""
    stub_search(monkeypatch, [view("a", 20.0, 0.0), view("b", 20.0, 10.0)])
    prices = mc.price_gaps(ACCOUNT, "token")
    assert prices.counted == 1
    assert prices.dear_share == 1.0


def test_two_currencies_are_never_compared_together(monkeypatch):
    """A euro gap and a dollar gap are both fractions, which hides the mixing."""
    stub_search(
        monkeypatch,
        [
            view("a", 20.0, 10.0),
            view("b", 20.0, 10.0),
            view("c", 5.0, 10.0, currency="EUR"),
        ],
    )
    prices = mc.price_gaps(ACCOUNT, "token")
    assert (prices.currency, prices.other_currencies) == ("USD", ("EUR",))
    assert prices.counted == 2


def test_an_empty_feed_says_nothing_rather_than_nought_percent(monkeypatch):
    stub_search(monkeypatch, [])
    prices = mc.price_gaps(ACCOUNT, "token")
    assert prices.counted == 0
    assert prices.dear_share == 0.0
    assert "has a benchmark yet" in " ".join(mc.verdicts(prices))


def test_the_worst_offenders_come_dearest_first(monkeypatch):
    stub_search(
        monkeypatch,
        [
            view("a", 12.0, 10.0, title="Mildly dear"),
            view("b", 30.0, 10.0, title="Three times the market"),
        ],
    )
    worst = mc.price_gaps(ACCOUNT, "token").worst
    assert list(worst["title"]) == ["Three times the market", "Mildly dear"]


def test_a_read_that_hit_its_ceiling_says_so(monkeypatch):
    stub_search(monkeypatch, [view("a", 20.0, 10.0)], truncated=True)
    lines = mc.verdicts(mc.price_gaps(ACCOUNT, "token"))
    assert any("read rather than all of them" in line for line in lines)


def test_the_headline_names_the_share_and_the_typical_gap(monkeypatch):
    stub_search(monkeypatch, [view("a", 20.0, 10.0), view("b", 8.0, 10.0)])
    lines = mc.verdicts(mc.price_gaps(ACCOUNT, "token"))
    assert "50% of 2 priced products" in lines[0]
    assert "cheaper than the market" in " ".join(lines)


def test_only_suggestions_to_charge_less_are_insights(monkeypatch):
    """A suggestion to charge more is not what a competitiveness panel is for."""
    rows = [
        {
            "priceInsightsProductView": {
                "offerId": "a",
                "title": "Cut me",
                "price": micros(20.0),
                "suggestedPrice": micros(15.0),
                "predictedClicksChangeFraction": 0.4,
                "predictedConversionsChangeFraction": 0.2,
            }
        },
        {
            "priceInsightsProductView": {
                "offerId": "b",
                "title": "Raise me",
                "price": micros(10.0),
                "suggestedPrice": micros(12.0),
                "predictedClicksChangeFraction": 0.1,
                "predictedConversionsChangeFraction": 0.1,
            }
        },
    ]
    monkeypatch.setattr(mc, "_search", lambda *_a, **_k: (rows, False))
    insights = mc.price_insights(ACCOUNT, "token")
    assert list(insights.offers["offer"]) == ["a"]
    assert insights.clicks_gain == pytest.approx(0.4)
    assert insights.conversions_gain == pytest.approx(0.2)


def test_the_suggestions_are_reported_with_what_they_would_buy(monkeypatch):
    prices = mc.Prices(
        pd.DataFrame(
            [{"offer": "a", "title": "t", "brand": "b", "price": 20.0,
              "benchmark": 10.0, "currency": "USD", "gap": 1.0}]
        ),
        "USD",
    )
    insights = mc.Insights(
        pd.DataFrame(
            [{"offer": "a", "title": "t", "price": 20.0, "suggested": 15.0,
              "clicks_change": 0.5, "conversions_change": 0.3}]
        )
    )
    lines = " ".join(mc.verdicts(prices, insights))
    assert "cut the price on 1" in lines
    assert "+50% clicks" in lines


def test_a_refused_credential_says_what_to_do_about_it(monkeypatch):
    """The fix is a click in Merchant Center, not a code change."""

    class Refused:
        status_code = 403
        text = "nope"

        def json(self):  # pragma: no cover - never reached
            return {}

    monkeypatch.setattr(mc.requests, "post", lambda *_a, **_k: Refused())
    with pytest.raises(mc.MerchantConfigError, match="People and access"):
        mc._search(ACCOUNT, "token", "SELECT 1")


def test_every_page_is_read_and_the_ceiling_is_reported(monkeypatch):
    """A catalogue is tens of thousands of rows, and the API pages at a thousand."""
    pages = {"count": 0}

    class Page:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            pages["count"] += 1
            return {
                "results": [view(str(pages["count"]), 20.0, 10.0)],
                "nextPageToken": "more",
            }

    monkeypatch.setattr(mc.requests, "post", lambda *_a, **_k: Page())
    rows, truncated = mc._search(ACCOUNT, "token", "SELECT 1")
    assert (len(rows), truncated) == (mc._MAX_PAGES, True)


def test_a_last_page_ends_the_read(monkeypatch):
    class Page:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [view("a", 20.0, 10.0)]}

    monkeypatch.setattr(mc.requests, "post", lambda *_a, **_k: Page())
    rows, truncated = mc._search(ACCOUNT, "token", "SELECT 1")
    assert (len(rows), truncated) == (1, False)


def test_a_key_that_is_not_json_says_which_variable_to_fix(monkeypatch):
    with pytest.raises(mc.MerchantConfigError, match="GCP_BIGQUERY_READONLY_KEY"):
        mc.access_token(mc.Merchant("5304150122", "not json"))


def test_the_reports_are_read_from_v1_and_ask_for_the_id_column(monkeypatch):
    """v1beta was switched off in February 2026, and v1 rejects a query
    on either view unless ``id`` is selected."""
    seen: dict[str, object] = {}

    class Page:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"results": []}

    def post(url, headers=None, json=None, timeout=None):  # noqa: A002
        seen["url"] = url
        seen["query"] = (json or {}).get("query", "")
        return Page()

    monkeypatch.setattr(mc.requests, "post", post)
    mc.price_gaps(ACCOUNT, "token")
    assert "/reports/v1/accounts/5304150122/" in str(seen["url"])
    assert str(seen["query"]).startswith("SELECT id, offer_id")
    mc.price_insights(ACCOUNT, "token")
    assert str(seen["query"]).startswith("SELECT id, offer_id")


def test_a_country_that_is_not_a_country_never_reaches_the_query(monkeypatch):
    """It is interpolated into a quoted literal, as the account id is into a path."""
    stub_search(monkeypatch, [])
    with pytest.raises(mc.MerchantConfigError, match="two-letter"):
        mc.price_gaps(ACCOUNT, "token", country="US' OR '1'='1")


def test_a_capped_suggestions_read_says_so_rather_than_under_counting(monkeypatch):
    """The count is of what was read, and a partial read admits it."""
    rows = [
        {
            "priceInsightsProductView": {
                "offerId": "a",
                "title": "t",
                "price": micros(20.0),
                "suggestedPrice": micros(15.0),
                "predictedClicksChangeFraction": 0.5,
                "predictedConversionsChangeFraction": 0.3,
            }
        }
    ]
    monkeypatch.setattr(mc, "_search", lambda *_a, **_k: (rows, True))
    insights = mc.price_insights(ACCOUNT, "token")
    assert insights.truncated
    prices = mc.Prices(
        pd.DataFrame(
            [{"offer": "a", "title": "t", "brand": "b", "price": 20.0,
              "benchmark": 10.0, "currency": "USD", "gap": 1.0}]
        ),
        "USD",
    )
    assert "products read rather than all of them" in " ".join(
        mc.verdicts(prices, insights)
    )


def test_the_feed_country_is_configurable_and_checked(monkeypatch):
    """Benchmarks are per country, so the wrong one reads as an empty feed."""
    monkeypatch.setenv("GOOGLE_MERCHANT_ID", "5304150122")
    monkeypatch.delenv("GCP_BIGQUERY_READONLY_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_MERCHANT_COUNTRY", raising=False)
    assert mc.load_merchant_env().country == "US"
    monkeypatch.setenv("GOOGLE_MERCHANT_COUNTRY", "gb")
    config = mc.load_merchant_env()
    assert config.country == "GB"
    monkeypatch.setenv("GOOGLE_MERCHANT_COUNTRY", "United Kingdom")
    with pytest.raises(mc.MerchantConfigError, match="two-letter"):
        mc.load_merchant_env()

    seen: dict[str, str] = {}

    def search(_config, _token, query):
        seen["query"] = query
        return [], False

    monkeypatch.setattr(mc, "_search", search)
    mc.price_gaps(config, "token")
    assert "report_country_code = 'GB'" in seen["query"]


def test_a_suggestion_on_a_product_nobody_compared_is_not_counted(monkeypatch):
    """The insights report has no benchmark, country or currency filter, so
    counting it whole could claim a cut on more products than were compared."""
    prices = mc.Prices(
        pd.DataFrame(
            [{"offer": "a", "title": "t", "brand": "b", "price": 20.0,
              "benchmark": 10.0, "currency": "USD", "gap": 1.0}]
        ),
        "USD",
    )
    insights = mc.Insights(
        pd.DataFrame(
            [
                {"offer": "a", "title": "t", "price": 20.0, "suggested": 15.0,
                 "clicks_change": 0.5, "conversions_change": 0.3},
                {"offer": "no-benchmark", "title": "u", "price": 30.0,
                 "suggested": 25.0, "clicks_change": 9.0,
                 "conversions_change": 9.0},
            ]
        )
    )
    assert insights.within(prices.offers).counted == 1
    lines = " ".join(mc.verdicts(prices, insights))
    assert "cut the price on 1 of them" in lines
    assert "+50% clicks" in lines  # not the 9.0 outsider's average


def priced(rows: list[tuple[str, float, float]]) -> mc.Prices:
    """A comparison frame in the shape ``price_gaps`` returns."""
    frame = pd.DataFrame(
        [
            {
                "offer": offer,
                "title": f"Wine {offer}",
                "brand": "Brand",
                "price": price,
                "benchmark": benchmark,
                "currency": "USD",
            }
            for offer, price, benchmark in rows
        ]
    )
    frame = frame.assign(
        gap=(frame["price"] - frame["benchmark"]) / frame["benchmark"]
    )
    return mc.Prices(frame, "USD")


def demand(rows: list[tuple[str, int, int]]) -> mc.Demand:
    return mc.Demand(
        pd.DataFrame(rows, columns=["offer", "clicks", "impressions"])
    )


def test_clicks_are_read_per_offer_over_the_last_month(monkeypatch):
    """One row per offer, not one per marketing method, and only the clicked."""
    seen: dict[str, str] = {}

    def search(_config, _token, query):
        seen["query"] = query
        return [
            {
                "productPerformanceView": {
                    "offerId": "a",
                    "clicks": "12",
                    "impressions": "300",
                }
            }
        ], False

    monkeypatch.setattr(mc, "_search", search)
    read = mc.product_demand(ACCOUNT, "token")
    assert "marketing_method" not in seen["query"]
    assert "LAST_30_DAYS" in seen["query"] and "clicks > 0" in seen["query"]
    assert read.clicks == 12
    assert list(read.offers["offer"]) == ["a"]


def test_an_offer_nobody_clicked_is_kept_with_no_clicks():
    """The performance report omits them, and dropping them would drop the wine."""
    prices = priced([("a", 20.0, 10.0), ("b", 30.0, 10.0)])
    joined = demand([("a", 5, 100)]).against(prices.offers)
    assert list(joined["clicks"]) == [5, 0]
    assert len(joined) == 2


def test_the_ask_list_ranks_demand_against_the_gap():
    """A wine nobody looks at is not the one to spend a merchant's goodwill on."""
    prices = priced(
        [
            ("quiet", 100.0, 10.0),  # a 900% gap, one click
            ("wanted", 30.0, 20.0),  # a 50% gap, a hundred clicks
            ("fair", 10.1, 10.0),  # within the tolerance, not on the list
            ("cheap", 8.0, 10.0),
        ]
    )
    wines = mc.ask_list(prices, demand([("quiet", 1, 10), ("wanted", 100, 900)]))
    assert list(wines["offer"]) == ["wanted", "quiet"]
    row = wines.iloc[0]
    # What it would take to reach the market price, off the shop's own price.
    assert round(float(row["cut"]), 3) == round(1 - 20 / 30, 3)
    assert round(float(row["overpay"]), 2) == 10.0


def test_googles_suggestion_rides_along_where_it_has_one():
    """Its own recommendation covers a few hundred offers, not the catalogue."""
    prices = priced([("a", 20.0, 10.0), ("b", 30.0, 10.0)])
    insights = mc.Insights(
        pd.DataFrame(
            [
                {
                    "offer": "a",
                    "title": "Wine a",
                    "price": 20.0,
                    "suggested": 18.0,
                    "clicks_change": 0.5,
                    "conversions_change": 0.3,
                }
            ]
        )
    )
    wines = mc.ask_list(prices, demand([("a", 10, 100), ("b", 5, 50)]), insights)
    row = wines[wines["offer"] == "a"].iloc[0]
    assert round(float(row["google_cut"]), 2) == 0.10
    assert pd.isna(wines[wines["offer"] == "b"].iloc[0]["google_cut"])


def test_the_bargains_are_the_cheaper_half_by_demand():
    prices = priced([("a", 20.0, 10.0), ("cheap", 8.0, 10.0), ("fair", 10.1, 10.0)])
    wines = mc.bargains(prices, demand([("cheap", 7, 70)]))
    assert list(wines["offer"]) == ["cheap"]
    assert int(wines["clicks"].iloc[0]) == 7


def test_a_cut_says_which_bottles_it_would_bring_to_market():
    """The negotiation is one percentage over a list, not a price per bottle."""
    prices = priced([("a", 11.0, 10.0), ("b", 30.0, 10.0)])
    wines = mc.ask_list(prices, demand([("a", 5, 50), ("b", 5, 50)]))
    after = mc.after_cut(wines, 0.10)
    assert mc.beats_market(after) == 1
    assert mc.beats_market(mc.after_cut(wines, 0.0)) == 0
    assert round(float(after[after["offer"] == "a"].iloc[0]["cut_price"]), 2) == 9.9


def test_the_verdict_says_what_share_of_demand_the_list_covers():
    prices = priced([("a", 20.0, 10.0), ("b", 30.0, 10.0)])
    lines = " ".join(
        mc.verdicts(prices, None, demand([("a", 30, 300), ("b", 70, 700)]))
    )
    assert "took 100 of the 100 clicks" in lines
    assert "100% of that demand" in lines


def test_no_clicks_at_all_costs_the_ranking_and_not_the_panel():
    """The performance report is optional: the headline does not depend on it."""
    prices = priced([("a", 20.0, 10.0)])
    empty = mc.Demand(pd.DataFrame())
    wines = mc.ask_list(prices, empty)
    assert list(wines["clicks"]) == [0]
    assert "cost more here than the market" in " ".join(mc.verdicts(prices, None, empty))


def test_a_click_report_that_could_not_be_read_does_not_fake_a_ranking():
    """Zero clicks on every row would sort the hundred arbitrarily and silently."""
    prices = priced([("small", 11.0, 10.0), ("big", 30.0, 10.0)])
    blind = mc.Demand(pd.DataFrame())
    assert not blind.measured
    wines = mc.ask_list(prices, blind)
    # Ranked by the gap instead, and in a defined order rather than whatever a
    # sort on a constant column happens to return.
    assert list(wines["offer"]) == ["big", "small"]
    assert list(mc.bargains(priced([("a", 9.0, 10.0), ("b", 5.0, 10.0)]), blind)[
        "offer"
    ]) == ["b", "a"]
    assert demand([("small", 3, 30)]).measured


def test_a_wine_the_report_splits_in_two_is_one_wine(monkeypatch):
    """Segmented rows would otherwise rank the same bottle at half its demand."""
    monkeypatch.setattr(
        mc,
        "_search",
        lambda *_args: (
            [
                {"productPerformanceView": {"offerId": "a", "clicks": "6",
                                            "impressions": "60"}},
                {"productPerformanceView": {"offerId": "a", "clicks": "4",
                                            "impressions": "40"}},
            ],
            False,
        ),
    )
    read = mc.product_demand(ACCOUNT, "token")
    assert list(read.offers["offer"]) == ["a"]
    assert read.clicks == 10
    assert int(read.offers["impressions"].iloc[0]) == 100


def test_a_suggestion_for_a_wine_this_panel_never_compared_is_not_shown():
    """The insights report has no country or currency filter of its own."""
    prices = priced([("a", 20.0, 10.0)])
    insights = mc.Insights(
        pd.DataFrame(
            [
                {"offer": "a", "title": "Wine a", "price": 20.0, "suggested": 18.0,
                 "clicks_change": 0.5, "conversions_change": 0.3},
                # The same offer twice, as a second feed label would give it.
                {"offer": "a", "title": "Wine a", "price": 20.0, "suggested": 18.0,
                 "clicks_change": 0.5, "conversions_change": 0.3},
                {"offer": "elsewhere", "title": "Wine b", "price": 90.0,
                 "suggested": 10.0, "clicks_change": 9.0, "conversions_change": 9.0},
            ]
        )
    )
    wines = mc.ask_list(prices, demand([("a", 4, 40)]), insights)
    # One row for the one wine compared: the duplicate suggestion does not take
    # a second place on the list, and the outsider takes none.
    assert list(wines["offer"]) == ["a"]
    assert round(float(wines["google_cut"].iloc[0]), 2) == 0.10


def test_a_report_that_never_arrived_is_not_a_shop_nobody_clicked():
    """Both leave an empty frame; only one is a finding about the wines."""
    unread = mc.Demand(pd.DataFrame(), read=False)
    quiet = mc.Demand(pd.DataFrame(columns=["offer", "clicks", "impressions"]))
    assert not unread.measured and not quiet.measured
    assert unread.read is False and quiet.read is True


def test_one_bottle_listed_twice_is_still_one_bottle(monkeypatch):
    """The view is keyed by feed label and language, not by the offer."""
    def view(price):
        return {
            "priceCompetitivenessProductView": {
                "offerId": "a",
                "title": "Wine a",
                "price": {"amountMicros": str(int(price * mc.MICROS)),
                          "currencyCode": "USD"},
                "benchmarkPrice": {"amountMicros": str(int(10 * mc.MICROS)),
                                   "currencyCode": "USD"},
            }
        }

    monkeypatch.setattr(mc, "_search", lambda *_args: ([view(20.0), view(20.0)], False))
    prices = mc.price_gaps(ACCOUNT, "token", "US")
    assert prices.counted == 1
    # And so the ask list offers the wine once, with its clicks counted once.
    wines = mc.ask_list(prices, demand([("a", 5, 50)]))
    assert len(wines) == 1
    assert int(wines["clicks"].iloc[0]) == 5


def test_a_bottle_listed_in_two_currencies_keeps_its_own_currency_row(monkeypatch):
    """De-duplicating before the currency split could keep the wrong row."""
    def view(offer, price, code):
        return {
            "priceCompetitivenessProductView": {
                "offerId": offer,
                "title": f"Wine {offer}",
                "price": {"amountMicros": str(int(price * mc.MICROS)),
                          "currencyCode": code},
                "benchmarkPrice": {"amountMicros": str(int(10 * mc.MICROS)),
                                   "currencyCode": code},
            }
        }

    monkeypatch.setattr(
        mc,
        "_search",
        # The euro row comes back first, and the same bottle is also priced in
        # the currency most of the catalogue uses.
        lambda *_args: (
            [view("a", 18.0, "EUR"), view("a", 20.0, "USD"),
             view("b", 30.0, "USD"), view("c", 40.0, "USD")],
            False,
        ),
    )
    prices = mc.price_gaps(ACCOUNT, "token", "US")
    assert prices.currency == "USD"
    assert prices.other_currencies == ("EUR",)
    # The bottle is still compared, once, at its dollar price.
    assert list(prices.offers["offer"]) == ["a", "b", "c"]
    assert float(prices.offers["price"].iloc[0]) == 20.0


def test_the_clicks_are_read_for_the_country_the_prices_were_compared_in(monkeypatch):
    """A click from another country never saw the price being argued about."""
    seen = {}

    def search(_config, _token, query):
        seen["query"] = query
        return [], False

    monkeypatch.setattr(mc, "_search", search)
    mc.product_demand(ACCOUNT, "token", "GB")
    assert "customer_country_code = 'GB'" in seen["query"]
    mc.product_demand(ACCOUNT, "token")
    assert f"customer_country_code = '{ACCOUNT.country}'" in seen["query"]
    with pytest.raises(mc.MerchantConfigError):
        mc.product_demand(ACCOUNT, "token", "United States")


def sold(rows: list[tuple[str, int, float]]) -> mc.Sales:
    return mc.Sales(pd.DataFrame(rows, columns=["offer", "bottles", "revenue"]))


def test_a_wine_that_sold_nothing_is_a_zero_and_not_a_missing_row():
    """The order book holds only what sold; the catalogue is what was offered."""
    frame = sold([("a", 3, 90.0)]).against(priced([("a", 20.0, 10.0),
                                                   ("b", 20.0, 10.0)]).offers)
    assert list(frame["bottles"]) == [3, 0]
    assert list(frame["sold_revenue"]) == [90.0, 0.0]


def test_an_unread_order_book_is_not_a_shop_that_sold_nothing():
    assert not mc.Sales(pd.DataFrame(), read=False).measured
    assert not mc.Sales(pd.DataFrame()).measured
    assert sold([("a", 1, 10.0)]).measured


def test_the_bands_count_bottles_against_the_clicks_that_saw_the_price():
    """The evidence: a rate per band, not a total that the biggest band wins."""
    prices = priced(
        [
            ("cheap", 8.0, 10.0),
            ("same", 10.1, 10.0),
            ("dear", 12.0, 10.0),
            ("dearest", 20.0, 10.0),
        ]
    )
    bands = mc.price_bands(
        prices,
        demand([("cheap", 100, 1000), ("same", 100, 1000),
                ("dear", 100, 1000), ("dearest", 100, 1000)]),
        sold([("cheap", 30, 240.0), ("same", 10, 101.0), ("dear", 5, 60.0)]),
    )
    rates = dict(zip(bands["band"], bands["per_100_clicks"]))
    assert rates["Cheaper than the market"] == 30
    assert rates["About the market"] == 10
    assert rates["Up to 25% dearer"] == 5
    # Nobody bought the dearest wine, which is a nought rather than a blank.
    assert rates["More than 25% dearer"] == 0
    assert list(bands["listings"]) == [1, 1, 1, 1]


def test_a_band_nobody_clicked_has_no_rate_rather_than_nought():
    """A wine Google never showed did not fail to sell; it was never offered."""
    bands = mc.price_bands(
        priced([("cheap", 8.0, 10.0), ("dear", 20.0, 10.0)]),
        demand([("cheap", 50, 500)]),
        sold([("cheap", 10, 80.0)]),
    )
    rates = dict(zip(bands["band"], bands["per_100_clicks"]))
    assert rates["Cheaper than the market"] == 20
    assert pd.isna(rates["More than 25% dearer"])


def test_the_evidence_sentence_compares_under_the_market_with_over_it():
    prices = priced([("cheap", 8.0, 10.0), ("dear", 20.0, 10.0)])
    lines = mc.sales_verdicts(
        prices,
        demand([("cheap", 100, 1000), ("dear", 100, 1000)]),
        sold([("cheap", 30, 240.0), ("dear", 10, 200.0)]),
    )
    assert lines, lines
    assert "30 bottles per 100 clicks" in lines[0]
    assert "sold 10" in lines[0]
    assert "90 days of orders against the last 30 days of clicks" in lines[0]
    assert "3.0x" in lines[0]
    # And it says what kind of evidence it is, rather than claiming a cause.
    assert "comparison rather than an experiment" in lines[1]


def test_no_sales_and_no_clicks_make_no_claim_at_all():
    """Half the evidence is not an argument, and silence is the honest output."""
    prices = priced([("cheap", 8.0, 10.0), ("dear", 20.0, 10.0)])
    clicks = demand([("cheap", 100, 1000), ("dear", 100, 1000)])
    assert mc.sales_verdicts(prices, clicks, mc.Sales(pd.DataFrame(), read=False)) == []
    assert mc.sales_verdicts(prices, mc.Demand(pd.DataFrame()),
                             sold([("cheap", 30, 240.0)])) == []
    # Nothing sold above the market is not a division by nought.
    assert mc.sales_verdicts(prices, clicks, sold([("cheap", 30, 240.0)])) == []


def test_a_merchants_own_wines_are_what_counts_as_measured():
    """The shop selling wine says nothing about the shop on screen.

    Filter the panel to a merchant whose handles never matched the catalogue
    and every row joins to nought - which must read as a failed match, not as a
    merchant whose wines nobody buys.
    """
    shop = sold([("mine", 12, 240.0)])
    assert shop.measured
    assert shop.measured_against(priced([("mine", 8.0, 10.0)]).offers)
    assert not shop.measured_against(priced([("theirs", 8.0, 10.0)]).offers)


def test_a_merchant_whose_bottles_never_matched_gets_no_verdict():
    theirs = priced([("theirs-cheap", 8.0, 10.0), ("theirs-dear", 20.0, 10.0)])
    clicks = demand([("theirs-cheap", 100, 1000), ("theirs-dear", 100, 1000)])
    assert mc.sales_verdicts(theirs, clicks, sold([("mine", 30, 240.0)])) == []
