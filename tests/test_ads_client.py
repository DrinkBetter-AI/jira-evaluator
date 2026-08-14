"""Offline checks for ads_client: no BigQuery, no network.

Everything here exercises the shaping - what a window folds to, what the campaign
table drops, what the sentences say - because that is where a wrong number would
be believed. The BigQuery reads themselves were verified against the live
dataset (w266-project-329918.google_ads).
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ads_client as ac  # noqa: E402

TODAY = dt.date(2026, 8, 6)
YESTERDAY = dt.date(2026, 8, 5)


def stats(rows: list[tuple[int, dt.date, float, int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "campaign_id",
            "day",
            "cost",
            "clicks",
            "conversions",
            "conversion_value",
        ],
    ).assign(impressions=100)


def names(rows: list[tuple[int, str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["campaign_id", "campaign", "status", "channel", "budget"]
    )


def test_no_environment_at_all_is_not_an_error(monkeypatch):
    for var in (
        "GOOGLE_ADS_BQ_PROJECT",
        "GOOGLE_ADS_BQ_DATASET",
        "GOOGLE_ADS_CUSTOMER_ID",
        "GCP_BIGQUERY_READONLY_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GCP_PROJECT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(ac, "default_project", lambda: "")
    assert ac.load_ads_env() is None


def test_a_project_alone_is_enough_and_the_dataset_has_a_default(monkeypatch):
    monkeypatch.setenv("GOOGLE_ADS_BQ_PROJECT", "w266-project-329918")
    monkeypatch.delenv("GOOGLE_ADS_BQ_DATASET", raising=False)
    monkeypatch.delenv("GCP_BIGQUERY_READONLY_KEY", raising=False)
    config = ac.load_ads_env()
    assert config.project == "w266-project-329918"
    assert config.dataset == ac.DEFAULT_DATASET
    assert config.customer_id is None


def test_a_key_names_its_own_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_ADS_BQ_PROJECT", raising=False)
    monkeypatch.setenv(
        "GCP_BIGQUERY_READONLY_KEY", json.dumps({"project_id": "from-the-key"})
    )
    assert ac.load_ads_env().project == "from-the-key"


def test_a_key_that_is_not_json_says_so(monkeypatch):
    monkeypatch.setenv("GCP_BIGQUERY_READONLY_KEY", "not json")
    with pytest.raises(ac.AdsConfigError, match="not valid JSON"):
        ac.load_ads_env()


def test_a_customer_id_may_be_written_the_way_google_prints_it(monkeypatch):
    monkeypatch.setenv("GOOGLE_ADS_BQ_PROJECT", "p")
    monkeypatch.delenv("GCP_BIGQUERY_READONLY_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_ADS_CUSTOMER_ID", "887-686-4797")
    assert ac.load_ads_env().customer_id == "8876864797"


def test_a_customer_id_that_is_not_one_is_refused(monkeypatch):
    monkeypatch.setenv("GOOGLE_ADS_BQ_PROJECT", "p")
    monkeypatch.delenv("GCP_BIGQUERY_READONLY_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_ADS_CUSTOMER_ID", "drop table")
    with pytest.raises(ac.AdsConfigError, match="customer ID"):
        ac.load_ads_env()


def test_a_table_name_cannot_be_smuggled_in_through_the_customer_id():
    config = ac.AdsConfig("p", "google_ads", None, None)
    with pytest.raises(ac.AdsConfigError):
        ac._table(config, "ads_Campaign", "8876864797` UNION SELECT")


def test_a_project_that_is_not_one_is_refused(monkeypatch):
    # The project reaches the same backquoted table reference as the dataset, and
    # arrives from the environment or from a pasted key's own project_id.
    monkeypatch.delenv("GCP_BIGQUERY_READONLY_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_ADS_BQ_PROJECT", "p`; DROP")
    with pytest.raises(ac.AdsConfigError, match="project id"):
        ac.load_ads_env()
    monkeypatch.delenv("GOOGLE_ADS_BQ_PROJECT")
    monkeypatch.setenv(
        "GCP_BIGQUERY_READONLY_KEY", json.dumps({"project_id": "p` UNION SELECT"})
    )
    with pytest.raises(ac.AdsConfigError, match="project id"):
        ac.load_ads_env()


def test_the_ambient_project_is_asked_for_once_per_process(monkeypatch):
    # Streamlit reruns the script on every click, and on Cloud Run this branch
    # reaches the metadata server, so the answer has to be remembered.
    for var in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    ac._adc_project.cache_clear()
    calls = []

    def _once():
        calls.append(1)
        return "discovered-project"

    monkeypatch.setattr(ac, "_ask_adc_project", _once)
    assert ac.default_project() == "discovered-project"
    assert ac.default_project() == "discovered-project"
    assert len(calls) == 1
    ac._adc_project.cache_clear()


def test_a_dataset_name_that_is_not_one_is_refused(monkeypatch):
    monkeypatch.setenv("GOOGLE_ADS_BQ_PROJECT", "p")
    monkeypatch.delenv("GCP_BIGQUERY_READONLY_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_ADS_BQ_DATASET", "google_ads`; DROP")
    with pytest.raises(ac.AdsConfigError, match="dataset name"):
        ac.load_ads_env()


def test_the_window_ends_yesterday_because_today_is_half_recorded():
    frame = stats(
        [
            (1, TODAY, 999.0, 1, 0.0, 0.0),
            (1, YESTERDAY, 10.0, 5, 1.0, 30.0),
        ]
    )
    spend = ac.window(frame, 7, now=TODAY)
    assert spend.cost == 10.0
    assert spend.last_day == YESTERDAY


def test_the_previous_period_is_the_same_length_immediately_before():
    frame = stats(
        [
            (1, YESTERDAY, 10.0, 5, 1.0, 30.0),
            (1, YESTERDAY - dt.timedelta(days=6), 5.0, 2, 0.0, 0.0),
            # One day before the previous 7-day window: out of both.
            (1, YESTERDAY - dt.timedelta(days=7), 4.0, 2, 2.0, 0.0),
            (1, YESTERDAY - dt.timedelta(days=13), 3.0, 1, 1.0, 0.0),
            (1, YESTERDAY - dt.timedelta(days=14), 100.0, 1, 1.0, 0.0),
        ]
    )
    spend = ac.window(frame, 7, now=TODAY)
    assert spend.cost == 15.0
    assert spend.prev_cost == 7.0
    assert spend.prev_conversions == 3.0
    assert spend.cost_change == 8.0


def test_a_window_whose_history_has_not_arrived_says_how_much_has():
    frame = stats([(1, YESTERDAY, 80.07, 458, 3.97, 69.84)])
    spend = ac.window(frame, 30, now=TODAY, history_start=YESTERDAY)
    assert spend.days_with_data == 1
    assert spend.partial
    assert spend.days_loaded == 1
    assert spend.history_start == YESTERDAY
    assert not ac.window(frame, 1, now=TODAY, history_start=YESTERDAY).partial


def test_days_on_which_nothing_ran_are_not_a_broken_feed():
    # The transfer writes no row for a day with no activity, so a window whose
    # history goes back far enough is complete even with gaps in the middle.
    frame = stats(
        [
            (1, YESTERDAY, 10.0, 5, 1.0, 30.0),
            (1, YESTERDAY - dt.timedelta(days=25), 10.0, 5, 1.0, 30.0),
        ]
    )
    spend = ac.window(
        frame, 7, now=TODAY, history_start=YESTERDAY - dt.timedelta(days=200)
    )
    assert spend.days_with_data == 1
    assert not spend.partial, "a quiet fortnight is not missing history"
    assert spend.days_loaded == 7


def test_a_long_pause_is_not_read_as_history_that_never_arrived():
    # The rows themselves start inside the window, because nothing ran before
    # that; the transfer's own history is what says whether days are loaded.
    frame = stats([(1, YESTERDAY, 10.0, 5, 1.0, 30.0)])
    loaded = YESTERDAY - dt.timedelta(days=200)
    assert not ac.window(frame, 30, now=TODAY, history_start=loaded).partial


def test_an_unknown_history_is_reported_complete_rather_than_partial():
    # Better a missing warning than one that fires on every quiet account.
    frame = stats([(1, YESTERDAY, 10.0, 5, 1.0, 30.0)])
    spend = ac.window(frame, 30, now=TODAY)
    assert not spend.partial
    assert spend.days_loaded == 30


def test_a_window_with_no_spend_at_all_is_quiet_rather_than_partial():
    # Everything the read returned falls in the previous period: the account
    # stopped spending, which must not be reported as data that never arrived.
    frame = stats([(1, YESTERDAY - dt.timedelta(days=8), 50.0, 5, 1.0, 30.0)])
    spend = ac.window(
        frame, 7, now=TODAY, history_start=YESTERDAY - dt.timedelta(days=200)
    )
    assert spend.cost == 0.0
    assert spend.prev_cost == 50.0
    assert not spend.partial
    assert spend.first_day is None
    assert spend.window_end == YESTERDAY


def test_an_empty_read_is_a_zero_window_rather_than_an_exception():
    spend = ac.window(stats([]), 30, now=TODAY)
    assert (spend.cost, spend.conversions, spend.days_with_data) == (0.0, 0.0, 0)
    assert spend.first_day is None
    assert spend.cost_per_conversion == 0.0
    assert spend.roas == 0.0


def test_cost_per_conversion_and_roas_come_out_of_the_live_figures():
    frame = stats([(1, YESTERDAY, 80.07, 458, 3.97, 69.84)])
    spend = ac.window(frame, 7, now=TODAY)
    assert spend.cost_per_conversion == 20.17
    assert spend.roas == 0.87


def test_campaigns_are_named_dearest_first_and_the_silent_ones_are_dropped():
    frame = stats(
        [
            (1, YESTERDAY, 13.49, 118, 3.0, 56.93),
            (2, YESTERDAY, 66.58, 458, 0.97, 12.91),
            (3, YESTERDAY, 0.0, 0, 0.0, 0.0),
        ]
    )
    table = ac.by_campaign(
        frame,
        names(
            [
                (1, "Sales-Shopping-sept17", "ENABLED", "SHOPPING", 60.0),
                (2, "AI-PMax-Bestsellers-FeedOnly", "ENABLED", "PERFORMANCE_MAX", 80.0),
                (3, "Thanksgiving_2025_pmax", "PAUSED", "PERFORMANCE_MAX", 20.0),
            ]
        ),
        7,
        now=TODAY,
    )
    assert list(table["campaign"]) == [
        "AI-PMax-Bestsellers-FeedOnly",
        "Sales-Shopping-sept17",
    ]
    assert table.iloc[0]["cost_per_conversion"] == 68.64
    assert table.iloc[1]["roas"] == 4.22


def test_spend_on_a_campaign_with_no_snapshot_is_still_reported():
    frame = stats([(4242, YESTERDAY, 12.0, 3, 0.0, 0.0)])
    table = ac.by_campaign(frame, names([]), 7, now=TODAY)
    assert list(table["campaign"]) == ["Campaign 4242"]
    assert list(table["status"]) == ["UNKNOWN"]


def test_wasted_is_the_money_that_bought_nothing():
    frame = stats(
        [
            (1, YESTERDAY, 66.58, 458, 0.0, 0.0),
            (2, YESTERDAY, 13.49, 118, 3.0, 56.93),
        ]
    )
    table = ac.by_campaign(
        frame,
        names(
            [
                (1, "AI-PMax-Bestsellers-FeedOnly", "ENABLED", "PERFORMANCE_MAX", 80.0),
                (2, "Sales-Shopping-sept17", "ENABLED", "SHOPPING", 60.0),
            ]
        ),
        7,
        now=TODAY,
    )
    assert list(ac.wasted(table)["campaign"]) == ["AI-PMax-Bestsellers-FeedOnly"]
    assert ac.paused_spenders(table).empty


def test_paused_spenders_are_the_ones_no_longer_running():
    frame = stats([(1, YESTERDAY, 30.0, 10, 1.0, 5.0)])
    table = ac.by_campaign(
        frame, names([(1, "vivi-PMax-sept-29", "PAUSED", "PERFORMANCE_MAX", 50.0)]), 7, now=TODAY
    )
    assert list(ac.paused_spenders(table)["campaign"]) == ["vivi-PMax-sept-29"]


def two_campaigns() -> pd.DataFrame:
    frame = stats(
        [
            (1, YESTERDAY, 66.58, 458, 0.0, 0.0),
            (2, YESTERDAY, 13.49, 118, 3.0, 56.93),
        ]
    )
    return ac.by_campaign(
        frame,
        names(
            [
                (1, "AI-PMax-Bestsellers-FeedOnly", "ENABLED", "PERFORMANCE_MAX", 80.0),
                (2, "Sales-Shopping-sept17", "ENABLED", "SHOPPING", 60.0),
            ]
        ),
        7,
        now=TODAY,
    )


def test_nothing_spent_says_only_that():
    spend = ac.window(stats([]), 30, now=TODAY)
    assert ac.verdicts(spend, pd.DataFrame(), None) == [
        "Nothing was spent on Google Ads in this window."
    ]


def test_the_verdicts_quote_the_crm_not_google():
    frame = stats([(1, YESTERDAY, 100.0, 458, 20.0, 500.0)])
    spend = ac.window(frame, 7, now=TODAY)
    lines = ac.verdicts(spend, two_campaigns(), ac.Sales(orders=4, revenue=800.0))
    joined = " ".join(lines)
    assert "$25 of ad spend per order" in joined
    assert "8.0x on money actually captured" in joined
    # Google claimed 20 against the shop's 4, which is worth saying.
    assert "Google claims 20 conversions where the CRM has 4 orders" in joined


def test_a_small_attribution_gap_is_left_alone():
    frame = stats([(1, YESTERDAY, 100.0, 458, 4.0, 500.0)])
    spend = ac.window(frame, 7, now=TODAY)
    lines = ac.verdicts(spend, two_campaigns(), ac.Sales(orders=4, revenue=800.0))
    assert not any("Google claims" in line for line in lines)


def test_money_that_bought_nothing_is_named():
    frame = stats([(1, YESTERDAY, 80.07, 576, 3.0, 56.93)])
    spend = ac.window(frame, 7, now=TODAY)
    lines = ac.verdicts(spend, two_campaigns(), None)
    assert any(
        "went to 1 campaign that recorded no conversion at all" in line
        and "AI-PMax-Bestsellers-FeedOnly" in line
        for line in lines
    ), lines


def test_a_shop_with_no_orders_against_real_spend_is_flagged():
    frame = stats([(1, YESTERDAY, 100.0, 458, 0.0, 0.0)])
    spend = ac.window(frame, 7, now=TODAY)
    lines = ac.verdicts(spend, two_campaigns(), ac.Sales(orders=0, revenue=0.0))
    assert "recorded no orders" in lines[0]


def test_a_ten_percent_move_in_spend_is_reported_and_a_smaller_one_is_not():
    def lines_for(previous: float) -> str:
        frame = stats(
            [
                (1, YESTERDAY, 110.0, 10, 1.0, 10.0),
                (1, YESTERDAY - dt.timedelta(days=7), previous, 10, 1.0, 10.0),
            ]
        )
        spend = ac.window(frame, 7, now=TODAY)
        return " ".join(ac.verdicts(spend, two_campaigns(), None))

    assert "Spend is up $10 (10%)" in lines_for(100.0)
    assert "Spend is up" not in lines_for(105.0)


def test_one_campaign_taking_the_budget_is_worth_a_sentence():
    frame = stats([(1, YESTERDAY, 80.07, 576, 3.0, 56.93)])
    spend = ac.window(frame, 7, now=TODAY)
    lines = " ".join(ac.verdicts(spend, two_campaigns(), None))
    assert "AI-PMax-Bestsellers-FeedOnly is 83% of the spend" in lines


def test_the_verdicts_quote_the_currency_the_account_bills_in():
    # The tiles read the symbol off the ad account, so the sentences beneath
    # them must not say dollars for a euro account.
    frame = stats([(1, YESTERDAY, 100.0, 458, 4.0, 500.0)])
    spend = ac.window(frame, 7, now=TODAY)
    lines = ac.verdicts(
        spend, two_campaigns(), ac.Sales(orders=4, revenue=800.0, currency="eur"), "EUR"
    )
    joined = " ".join(lines)
    assert "\u20ac25 of ad spend per order" in joined
    assert "$" not in joined


def test_a_currency_with_no_symbol_is_named_instead():
    frame = stats([(1, YESTERDAY, 100.0, 458, 4.0, 500.0)])
    spend = ac.window(frame, 7, now=TODAY)
    lines = ac.verdicts(spend, two_campaigns(), ac.Sales(orders=4, revenue=0.0), "CAD")
    assert "25 CAD of ad spend per order" in " ".join(lines)


def test_the_commission_on_a_dollar_of_spend_is_the_figure_leadership_asked_for():
    # Angel's worked example: 1,226 of revenue at 12% against 176 of spend,
    # which is 0.8359 - quoted as 0.83 by hand, and rounded here rather than cut.
    assert ac.commission_return(1226.0, 176.0, 0.12) == pytest.approx(0.84)
    assert ac.commission_return(1226.0, 0.0, 0.12) == 0.0


def test_the_commission_line_leads_the_verdicts_and_names_the_shortfall():
    frame = stats(
        [
            (1, YESTERDAY, 176.0, 500, 4.0, 900.0),
            (1, YESTERDAY - dt.timedelta(days=7), 176.0, 500, 4.0, 900.0),
        ]
    )
    spend = ac.window(frame, 7, now=TODAY)
    lines = ac.verdicts(
        spend,
        two_campaigns(),
        ac.Sales(orders=9, revenue=1226.0, prev_revenue=613.0),
        "USD",
    )
    assert "$0.84 of commission back for every $1 of ad spend" in lines[0]
    assert "1,226 of revenue at 12%" in lines[0]
    assert "short by $0.16 on every $1" in lines[0]
    # Half the revenue for the same spend last window: the trend must say so.
    assert "That return is up" in lines[1]
    assert "$0.42 per $1" in lines[1]


def test_a_return_at_break_even_is_not_reported_as_a_shortfall():
    frame = stats([(1, YESTERDAY, 100.0, 500, 4.0, 900.0)])
    spend = ac.window(frame, 7, now=TODAY)
    said = " ".join(
        ac.verdicts(spend, two_campaigns(), ac.Sales(orders=9, revenue=10000.0), "USD")
    )
    assert "pay for themselves" in said
    assert "short by" not in said


def test_the_commission_rate_is_read_from_the_environment_in_any_of_three_forms(
    monkeypatch,
):
    monkeypatch.delenv("MARKETPLACE_COMMISSION_RATE", raising=False)
    assert ac.commission_rate() == ac.DEFAULT_COMMISSION_RATE
    for written in ("15", "15%", "0.15", " 15 % "):
        monkeypatch.setenv("MARKETPLACE_COMMISSION_RATE", written)
        assert ac.commission_rate() == pytest.approx(0.15)
    for wrong in ("twelve", "0", "-4", "250"):
        monkeypatch.setenv("MARKETPLACE_COMMISSION_RATE", wrong)
        with pytest.raises(ac.AdsConfigError):
            ac.commission_rate()


def test_commission_actually_charged_beats_a_rate_that_fits_no_merchant():
    # Merchants sit on 10% and 12% agreements, so a flat rate is an estimate of
    # something Stripe already counted. 147 of commission against 176 of spend.
    frame = stats(
        [
            (1, YESTERDAY, 176.0, 500, 4.0, 900.0),
            (1, YESTERDAY - dt.timedelta(days=7), 176.0, 500, 4.0, 900.0),
        ]
    )
    spend = ac.window(frame, 7, now=TODAY)
    charged = ac.Commission(now=147.0, before=73.0, measured=True)
    lines = ac.verdicts(
        spend,
        two_campaigns(),
        ac.Sales(orders=9, revenue=1226.0, prev_revenue=613.0),
        "USD",
        commission=charged,
    )
    assert "$0.84 of commission back for every $1 of ad spend" in lines[0]
    assert "$147 of commission actually charged" in lines[0]
    # The assumed rate is not quoted once the real one has been counted.
    assert "12%" not in lines[0]
    assert "That return is up" in lines[1]


def test_commission_charged_is_read_even_where_the_crm_has_no_revenue():
    # The CRM's takings and Stripe's ledger are different feeds; one being out
    # must not silently drop the only income figure on the panel.
    frame = stats([(1, YESTERDAY, 100.0, 500, 4.0, 900.0)])
    spend = ac.window(frame, 7, now=TODAY)
    lines = ac.verdicts(
        spend,
        two_campaigns(),
        ac.Sales(orders=0, revenue=0.0),
        "USD",
        commission=ac.Commission(now=120.0, before=0.0, measured=True),
    )
    assert "$1.20 of commission back for every $1 of ad spend" in lines[0]
    assert "pay for themselves" in lines[0]


def test_a_commission_return_divides_only_when_there_was_spend():
    assert ac.earned_return(147.0, 176.0) == pytest.approx(0.84)
    assert ac.earned_return(147.0, 0.0) == 0.0


class FakeQuery:
    """A BigQuery client that records the SQL and answers with a frame."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.sql = ""
        self.params: dict = {}

    def query(self, sql, job_config=None):  # noqa: D401 - mimics the client
        self.sql = sql
        self.params = {
            param.name: param.value for param in (job_config.query_parameters or [])
        }
        frame = self.frame

        class Job:
            def result(self):
                class Rows:
                    def to_dataframe(self):
                        return frame

                return Rows()

        return Job()


PRODUCTS = pd.DataFrame(
    [
        ("wine-1", 12.5, 40, 900, 1.0),
        (" wine-2 ", 3.25, 8, 120, 0.0),
        ("", 1.0, 2, 30, 0.0),
    ],
    columns=["offer", "spend", "clicks", "impressions", "ad_conversions"],
)


def test_product_spend_is_read_per_offer_over_the_window_asked_for():
    client = FakeQuery(PRODUCTS)
    config = ac.AdsConfig("w266-project-329918", "google_ads", None, None)
    frame = ac.product_stats(client, config, "8876864797", 90, now=TODAY)
    # Yesterday backwards, today excluded: today is a part-day and would read as
    # spend collapsing.
    assert client.params == {"first": dt.date(2026, 5, 8), "last": YESTERDAY}
    assert "ads_ShoppingProductStats_8876864797" in client.sql
    # An offer id with nothing in it joins to every wine and to none, so it goes.
    assert list(frame["offer"]) == ["wine-1", "wine-2"]
    assert float(frame["spend"].iloc[0]) == 12.5


def test_spend_with_no_product_id_on_it_is_not_a_wine_called_none():
    """BigQuery groups NULL as a key of its own, so the product read can come
    back with a row for spend that has no offer id; cast to a string first it
    became a bottle named "None" that Google could publish no benchmark for."""
    client = FakeQuery(
        pd.DataFrame(
            [("wine-1", 12.5, 40, 900, 1.0), (None, 4.0, 3, 60, 0.0)],
            columns=list(ac.PRODUCT_COLUMNS),
        )
    )
    config = ac.AdsConfig("w266-project-329918", "google_ads", None, None)
    frame = ac.product_stats(client, config, "8876864797", 30, now=TODAY)
    assert list(frame["offer"]) == ["wine-1"]


def test_no_shopping_rows_at_all_reads_as_no_products_rather_than_an_error():
    client = FakeQuery(pd.DataFrame())
    config = ac.AdsConfig("w266-project-329918", "google_ads", None, None)
    frame = ac.product_stats(client, config, "8876864797", 30, now=TODAY)
    assert frame.empty
    assert list(frame.columns) == list(ac.PRODUCT_COLUMNS)


class FakeRows:
    """A client that records the SQL and answers with rows rather than a frame."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.sql = ""

    def query(self, sql, job_config=None):  # noqa: D401 - mimics the client
        self.sql = sql
        rows = self.rows

        class Job:
            def result(self):
                return rows

        return Job()


def test_which_table_a_history_is_asked_of_is_the_callers_to_choose():
    """The Shopping product report is transferred separately from the campaign
    one and is routinely switched on months later, so the campaign table's first
    day says nothing about how much per-wine spend is loaded."""
    client = FakeRows([{"first_day": dt.date(2026, 7, 6)}])
    config = ac.AdsConfig("w266-project-329918", "google_ads", None, None)
    assert ac.loaded_from(client, config, "8876864797") == dt.date(2026, 7, 6)
    assert "ads_CampaignBasicStats_8876864797" in client.sql
    ac.loaded_from(client, config, "8876864797", TODAY, ac.PRODUCT_TABLE)
    assert "ads_ShoppingProductStats_8876864797" in client.sql


def test_a_window_begins_the_day_after_it_is_long_and_never_includes_today():
    assert ac.window_first_day(90, TODAY) == dt.date(2026, 5, 8)
    assert ac.window_first_day(1, TODAY) == YESTERDAY


def test_a_customer_id_cannot_smuggle_a_table_name_into_the_product_read():
    client = FakeQuery(PRODUCTS)
    config = ac.AdsConfig("w266-project-329918", "google_ads", None, None)
    with pytest.raises(ac.AdsConfigError):
        ac.product_stats(client, config, "8876864797` UNION ALL SELECT", 30, now=TODAY)
