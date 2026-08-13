"""Offline checks for cost_client: no network.

Everything here exercises the shaping - what a window folds to, what the line
table says, what the sentences claim - because that is where a wrong number would
be believed. The OpenAI read itself was verified against the live organization
cost endpoint (org "Voss", $763 over 30 days).
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cost_client as cc  # noqa: E402

TODAY = dt.date(2026, 8, 6)


def costs(rows: list[tuple[dt.date, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["day", "project", "line_item", "cost"]
    ).assign(currency="usd")


def flat(days: int, amount: float, start: dt.date | None = None) -> pd.DataFrame:
    first = start or TODAY - dt.timedelta(days=days - 1)
    return costs(
        [
            (first + dt.timedelta(days=offset), "Vinovoss", "gpt-5.6-terra, input", amount)
            for offset in range(days)
        ]
    )


def test_key_must_be_an_admin_key(monkeypatch):
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-proj-abc123")
    with pytest.raises(cc.CostConfigError, match="project key"):
        cc.load_openai_env()


def test_no_key_is_not_an_error(monkeypatch):
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    assert cc.load_openai_env() is None
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "  sk-admin-xyz  ")
    assert cc.load_openai_env() == "sk-admin-xyz"


def test_window_splits_current_from_previous():
    frame = flat(60, 10.0, start=TODAY - dt.timedelta(days=59))
    burn = cc.window(frame, 30, now=TODAY)
    assert burn.cost == pytest.approx(300.0)
    assert burn.prev_cost == pytest.approx(300.0)
    assert burn.cost_change == pytest.approx(0.0)
    assert burn.days_with_data == 30
    assert burn.first_day == TODAY - dt.timedelta(days=29)
    # Today counts: a provider bills as it goes, unlike an ad transfer that has
    # simply not loaded the day yet.
    assert burn.last_day == TODAY


def test_monthly_rate_divides_by_the_window_not_the_busy_days():
    # Charges on two days of seven. A month at this rate is four times $50, not
    # thirty times $25, which is what dividing by busy days would claim.
    frame = costs(
        [
            (TODAY, "Vinovoss", "input", 25.0),
            (TODAY - dt.timedelta(days=3), "Vinovoss", "input", 25.0),
        ]
    )
    burn = cc.window(frame, 7, now=TODAY)
    assert burn.cost == pytest.approx(50.0)
    assert burn.days_with_data == 2
    assert burn.monthly == pytest.approx(50.0 / 7 * 30)


def test_empty_frame_is_a_zero_not_a_crash():
    burn = cc.window(pd.DataFrame(), 30, now=TODAY)
    assert burn.cost == 0.0 and burn.prev_cost == 0.0
    assert burn.lines.empty and burn.first_day is None
    assert cc.verdicts(burn) == ["No OpenAI charges in the last 30 days."]


def test_lines_are_dearest_first_with_shares():
    frame = costs(
        [
            (TODAY, "Vinovoss", "gpt-5.6-terra, cached input", 60.0),
            (TODAY, "Vinovoss", "gpt-5.6-terra, input", 30.0),
            (TODAY - dt.timedelta(days=1), "Vinovoss", "gpt-5.6-terra, input", 10.0),
        ]
    )
    lines = cc.by_line(frame)
    assert lines["line_item"].tolist() == [
        "gpt-5.6-terra, cached input",
        "gpt-5.6-terra, input",
    ]
    assert lines["cost"].tolist() == [60.0, 40.0]
    assert lines["share"].tolist() == pytest.approx([0.6, 0.4])


def test_projects_are_dearest_first():
    frame = costs(
        [
            (TODAY, "Default project", "input", 3.0),
            (TODAY, "Vinovoss", "input", 700.0),
        ]
    )
    assert cc.by_project(frame)["project"].tolist() == ["Vinovoss", "Default project"]


def test_cached_share_counts_cache_traffic_only():
    frame = costs(
        [
            (TODAY, "Vinovoss", "gpt-5.6-terra, cached input, long context", 50.0),
            (TODAY, "Vinovoss", "gpt-5.6-terra, cache writes", 25.0),
            (TODAY, "Vinovoss", "gpt-5.6-terra, input", 25.0),
        ]
    )
    assert cc.cached_share(cc.by_line(frame)) == pytest.approx(0.75)
    assert cc.cached_share(pd.DataFrame()) == 0.0
    # A frame of zeroes divides by nothing rather than raising.
    assert cc.cached_share(costs([(TODAY, "V", "input", 0.0)])) == 0.0


def test_currency_is_read_rather_than_assumed():
    frame = flat(3, 5.0).assign(currency="eur")
    assert cc.window(frame, 3, now=TODAY).currency == "eur"


def test_a_rise_is_named_with_both_the_amount_and_the_share():
    frame = pd.concat(
        [
            flat(7, 20.0, start=TODAY - dt.timedelta(days=6)),
            flat(7, 10.0, start=TODAY - dt.timedelta(days=13)),
        ]
    )
    burn = cc.window(frame, 7, now=TODAY)
    said = " ".join(cc.verdicts(burn))
    assert "$140 on OpenAI in 7 days" in said
    assert "$600 a month at this rate" in said
    assert "up $70 (+100%)" in said


def test_a_fall_is_named_as_a_fall():
    frame = pd.concat(
        [
            flat(7, 10.0, start=TODAY - dt.timedelta(days=6)),
            flat(7, 20.0, start=TODAY - dt.timedelta(days=13)),
        ]
    )
    said = " ".join(cc.verdicts(cc.window(frame, 7, now=TODAY)))
    assert "down $70 (-50%)" in said


def test_spend_that_started_inside_the_window_is_not_a_spike():
    # The month before came to almost nothing, so a percentage would read as a
    # 25,000% jump - true, and useless.
    frame = pd.concat(
        [
            flat(30, 25.0, start=TODAY - dt.timedelta(days=29)),
            costs([(TODAY - dt.timedelta(days=45), "Vinovoss", "input", 3.0)]),
        ]
    )
    said = " ".join(cc.verdicts(cc.window(frame, 30, now=TODAY)))
    assert "This spend is new" in said
    assert "%)" not in said


def test_a_quiet_change_is_left_unsaid():
    frame = pd.concat(
        [
            flat(7, 21.0, start=TODAY - dt.timedelta(days=6)),
            flat(7, 20.0, start=TODAY - dt.timedelta(days=13)),
        ]
    )
    said = " ".join(cc.verdicts(cc.window(frame, 7, now=TODAY)))
    assert "up" not in said and "down" not in said


def test_cache_heavy_bills_say_so_and_price_it():
    frame = pd.concat(
        [
            costs(
                [
                    (TODAY, "Vinovoss", "gpt-5.6-terra, cached input", 700.0),
                    (TODAY, "Vinovoss", "gpt-5.6-terra, input", 300.0),
                ]
            ),
            flat(30, 34.0, start=TODAY - dt.timedelta(days=59)),
        ]
    )
    said = " ".join(cc.verdicts(cc.window(frame, 30, now=TODAY)))
    assert "70% of it is cached context" in said
    assert "$700 spent re-sending prompts" in said


def test_a_dominant_line_item_is_named():
    frame = pd.concat(
        [
            costs(
                [
                    (TODAY, "Vinovoss", "gpt-realtime-2.1 audio, output", 800.0),
                    (TODAY, "Vinovoss", "gpt-5.6-terra, input", 200.0),
                ]
            ),
            flat(30, 34.0, start=TODAY - dt.timedelta(days=59)),
        ]
    )
    said = " ".join(cc.verdicts(cc.window(frame, 30, now=TODAY)))
    assert "gpt-realtime-2.1 audio, output is 80% of the bill" in said
    assert "$800" in said


def test_money_rounds_pennies_away_only_when_they_are_noise():
    assert cc._money(763.0) == "$763"
    assert cc._money(2.95) == "$2.95"
    assert cc._money(-70.4) == "$-70"


def test_windows_offered_are_the_leadership_ones():
    assert cc.LOOKBACK_WINDOWS == (7, 30)


def entries(rows: list[tuple[dt.date, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["day", "type", "amount", "fee"]
    ).assign(category="", currency="usd")


def test_the_commission_change_is_net_of_refunds_not_gross():
    """A refunded month must not read as a rise under a figure that fell."""
    frame = entries(
        [
            (TODAY, "application_fee", 1000.0, 0.0),
            (TODAY, "application_fee_refund", -400.0, 0.0),
            (TODAY - dt.timedelta(days=10), "application_fee", 900.0, 0.0),
        ]
    )
    ledger = cc.ledger_window(frame, 7, now=TODAY)
    assert ledger.earnings_change == pytest.approx(100.0)
    assert ledger.net_change == pytest.approx(-300.0)
    assert "down" in " ".join(cc.stripe_verdicts(ledger))


def test_a_platform_with_no_fees_of_its_own_says_so():
    frame = entries([(TODAY, "application_fee", 500.0, 0.0)])
    ledger = cc.ledger_window(frame, 7, now=TODAY)
    assert ledger.fees == 0.0
    said = " ".join(cc.stripe_verdicts(ledger))
    assert "charged this account nothing" in said
    assert "merchants' own accounts" in said


def test_an_empty_ledger_still_carries_the_dispute_count():
    ledger = cc.ledger_window(pd.DataFrame(), 7, disputes=3, now=TODAY)
    assert ledger.disputes == 3
    assert ledger.net == 0.0 and ledger.prev_net == 0.0


class _Response:
    def __init__(self, payload: dict) -> None:
        self.status_code = 200
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_the_ledger_read_reports_when_stripe_still_had_more(monkeypatch):
    """The dropped rows are the oldest, so silence would read as growth."""
    calls: list[dict] = []

    def always_more(url, params=None, headers=None, timeout=None):
        calls.append(dict(params or []))
        return _Response(
            {
                "has_more": True,
                "data": [
                    {
                        "id": f"txn_{len(calls)}",
                        "created": int(
                            dt.datetime(2026, 8, 6, tzinfo=dt.timezone.utc).timestamp()
                        ),
                        "type": "application_fee",
                        "amount": 100,
                        "fee": 0,
                        "currency": "usd",
                    }
                ],
            }
        )

    monkeypatch.setattr(cc.requests, "get", always_more)
    frame, truncated = cc.stripe_ledger("rk_test", 30, now=TODAY)
    assert truncated is True
    assert len(calls) == cc._STRIPE_MAX_PAGES
    assert len(frame) == cc._STRIPE_MAX_PAGES


def test_a_ledger_that_fits_is_not_called_truncated(monkeypatch):
    def one_page(url, params=None, headers=None, timeout=None):
        return _Response(
            {
                "has_more": False,
                "data": [
                    {
                        "id": "txn_1",
                        "created": int(
                            dt.datetime(2026, 8, 6, tzinfo=dt.timezone.utc).timestamp()
                        ),
                        "type": "application_fee",
                        "amount": 2500,
                        "fee": 0,
                        "currency": "usd",
                    }
                ],
            }
        )

    monkeypatch.setattr(cc.requests, "get", one_page)
    frame, truncated = cc.stripe_ledger("rk_test", 30, now=TODAY)
    assert truncated is False
    assert float(frame["amount"].sum()) == pytest.approx(25.0)


def test_disputes_are_counted_past_the_first_hundred(monkeypatch):
    pages = [
        {"has_more": True, "data": [{"id": f"dp_{i}"} for i in range(100)]},
        {"has_more": False, "data": [{"id": "dp_x"}, {"id": "dp_y"}]},
    ]

    def paged(url, params=None, headers=None, timeout=None):
        return _Response(pages.pop(0))

    monkeypatch.setattr(cc.requests, "get", paged)
    assert cc.stripe_disputes("rk_test", 30, now=TODAY) == 102


def test_a_full_secret_key_is_refused(monkeypatch):
    monkeypatch.setenv("STRIPE_READONLY_API_KEY", "sk_live_pretend")
    with pytest.raises(cc.CostConfigError):
        cc.load_stripe_env()


def test_sentences_are_written_in_the_billing_currency():
    frame = costs([(TODAY, "Vinovoss", "gpt-5.6-terra, input", 400.0)]).assign(
        currency="eur"
    )
    said = " ".join(cc.verdicts(cc.window(frame, 7, now=TODAY)))
    assert "€400" in said and "$" not in said


def test_a_second_currency_is_set_aside_rather_than_added():
    frame = pd.concat(
        [
            flat(7, 30.0, start=TODAY - dt.timedelta(days=6)),
            costs([(TODAY, "Vinovoss", "gpt-5.6-terra, input", 900.0)]).assign(
                currency="eur"
            ),
        ]
    )
    burn = cc.window(frame, 7, now=TODAY)
    assert burn.currency == "usd"
    assert burn.other_currencies == ("eur",)
    assert burn.cost == pytest.approx(210.0)


def test_a_ledger_in_two_currencies_reports_only_the_main_one():
    frame = pd.concat(
        [
            entries([(TODAY, "application_fee", 500.0, 0.0)] * 3),
            entries([(TODAY, "application_fee", 900.0, 0.0)]).assign(currency="gbp"),
        ]
    )
    ledger = cc.ledger_window(frame, 7, now=TODAY)
    assert ledger.currency == "usd"
    assert ledger.other_currencies == ("gbp",)
    assert ledger.earnings == pytest.approx(1500.0)
    assert "£" not in " ".join(cc.stripe_verdicts(ledger))


def test_an_ordinary_account_keeps_its_charges_not_a_negative():
    """No application fees means no platform: the charge is the income."""
    frame = entries(
        [
            (TODAY, "charge", 500.0, 17.5),
            (TODAY, "refund", -50.0, 0.0),
            (TODAY - dt.timedelta(days=10), "charge", 400.0, 14.0),
        ]
    )
    ledger = cc.ledger_window(frame, 7, now=TODAY)
    assert ledger.platform is False
    assert ledger.earnings == pytest.approx(500.0)
    assert ledger.net == pytest.approx(432.5)
    said = " ".join(cc.stripe_verdicts(ledger))
    assert "of payments kept" in said
    # The nil-fee sentence is about a platform's merchants; this account paid.
    assert "charged this account nothing" not in said


def test_a_platform_is_still_read_as_a_platform_when_it_also_takes_charges():
    frame = entries(
        [(TODAY, "application_fee", 500.0, 0.0), (TODAY, "charge", 900.0, 30.0)]
    )
    ledger = cc.ledger_window(frame, 7, now=TODAY)
    assert ledger.platform is True
    assert ledger.earnings == pytest.approx(500.0)
    assert "of commission kept" in " ".join(cc.stripe_verdicts(ledger))


def test_disputes_are_reported_even_when_the_window_earned_nothing():
    ledger = cc.ledger_window(pd.DataFrame(), 30, disputes=2)
    said = " ".join(cc.stripe_verdicts(ledger))
    assert "2 disputes opened" in said


# --- Google Cloud's billing export -----------------------------------------


class _FakeJob:
    def __init__(self, rows: list[dict], frame: pd.DataFrame | None = None):
        self._rows = rows
        self._frame = frame

    def result(self):
        return self

    def __iter__(self):
        return iter(self._rows)

    def to_dataframe(self):
        return self._frame if self._frame is not None else pd.DataFrame()


class _FakeBigQuery:
    """Enough of a BigQuery client to read the calls the billing reader makes."""

    def __init__(self, tables: list[str], rows=None, frame=None):
        self._tables = tables
        self._rows = rows or []
        self._frame = frame
        self.sql: list[str] = []
        self.params: list[object] = []

    def list_tables(self, dataset: str):
        self.dataset = dataset
        return [type("T", (), {"table_id": name})() for name in self._tables]

    def query(self, sql: str, job_config=None):
        self.sql.append(sql)
        if job_config is not None:
            self.params.append(job_config.query_parameters)
        return _FakeJob(self._rows, self._frame)


def test_the_export_table_is_found_by_its_prefix_not_configured():
    """The suffix is the billing account's id, which nobody should look up."""
    client = _FakeBigQuery(
        ["some_other_table", "gcp_billing_export_v1_01DD38_399CFC_E301B4"]
    )
    tables = cc.billing_tables(client, cc.Billing("proj", "billing_export"))
    assert tables == ["proj.billing_export.gcp_billing_export_v1_01DD38_399CFC_E301B4"]


def test_two_billing_accounts_exporting_to_one_dataset_are_both_reported():
    """Both are named so the panel can say whose bill is on screen."""
    client = _FakeBigQuery(
        ["gcp_billing_export_v1_B_2", "gcp_billing_export_v1_A_1", "other"]
    )
    tables = cc.billing_tables(client, cc.Billing("proj", "ds"))
    assert [name.rsplit(".", 1)[1] for name in tables] == [
        "gcp_billing_export_v1_A_1",
        "gcp_billing_export_v1_B_2",
    ]


def test_a_dataset_with_no_export_table_yet_is_not_an_error():
    """Google takes hours to write the first one, which is normal, not broken."""
    assert cc.billing_tables(_FakeBigQuery([]), cc.Billing("proj", "ds")) == []


def test_a_dataset_nobody_has_created_reads_as_no_export_rather_than_an_error():
    """Where every deployment starts, and a 404 is not worth a warning banner."""
    from google.api_core import exceptions

    class Missing(_FakeBigQuery):
        def list_tables(self, dataset):
            raise exceptions.NotFound("404 Not found: Dataset proj:billing_export")

    assert cc.billing_tables(Missing([]), cc.Billing("proj", "billing_export")) == []


def test_an_unreadable_billing_dataset_names_itself():
    class Denied(_FakeBigQuery):
        def list_tables(self, dataset):
            raise RuntimeError("403 Access Denied: User does not have permission")

    with pytest.raises(cc.CostConfigError) as caught:
        cc.billing_tables(Denied([]), cc.Billing("proj", "billing_export"))
    assert "proj.billing_export" in str(caught.value)


def test_cloud_costs_reads_twice_the_window_ending_where_it_is_told():
    frame = pd.DataFrame(
        [(dt.date(2026, 7, 2), "proj", "Cloud Run", "usd", 12.0)],
        columns=["day", "project", "line_item", "currency", "cost"],
    )
    client = _FakeBigQuery(["x"], frame=frame)
    read = cc.cloud_costs(client, "p.d.t", 7, now=dt.date(2026, 7, 2))
    assert list(read["line_item"]) == ["Cloud Run"]
    first, last = (p.value for p in client.params[0])
    # Fourteen days inclusive, ending on the day asked for rather than today.
    assert last == dt.date(2026, 7, 2)
    assert first == dt.date(2026, 6, 19)
    # Credits are netted against the charge they belong to, in the query.
    assert "credits" in client.sql[0]


def test_a_day_whose_credits_cancel_it_is_not_a_line_on_the_bill():
    frame = pd.DataFrame(
        [
            (dt.date(2026, 7, 2), "proj", "Cloud Run", "usd", 12.0),
            (dt.date(2026, 7, 2), "proj", "Free tier thing", "usd", 0.0),
        ],
        columns=["day", "project", "line_item", "currency", "cost"],
    )
    read = cc.cloud_costs(_FakeBigQuery(["x"], frame=frame), "p.d.t", 7)
    assert list(read["line_item"]) == ["Cloud Run"]


def test_the_export_reports_both_ends_of_what_it_covers():
    rows = [{"first": dt.date(2026, 6, 1), "last": dt.date(2026, 7, 2)}]
    first, last = cc.billing_coverage(_FakeBigQuery(["x"], rows=rows), "p.d.t")
    assert (first, last) == (dt.date(2026, 6, 1), dt.date(2026, 7, 2))


def test_a_window_whose_credits_cancel_its_charges_is_not_negative_zero():
    """-0.0 formats as "$-0.00", which reads as a refund that never happened."""
    costs = pd.DataFrame(
        [
            (TODAY, "proj", "Cloud Run", 5.0),
            (TODAY, "proj", "Committed use discount", -5.0),
            # A credit worth a fraction of a cent, which is what the export
            # actually holds for a day nobody used anything on.
            (TODAY, "proj", "AlloyDB", -0.000159),
        ],
        columns=["day", "project", "line_item", "cost"],
    ).assign(currency="usd")
    burn = cc.window(costs, 7, provider="Google Cloud", now=TODAY)
    assert burn.cost == 0.0
    assert not str(burn.cost).startswith("-")
    assert "-" not in " ".join(cc.verdicts(burn))


def test_the_ads_settings_cannot_switch_the_cloud_bill_off(monkeypatch):
    """An unrelated Ads typo used to hide the Cloud panel behind its message."""
    monkeypatch.setenv("GCP_BILLING_BQ_PROJECT", "w266-project-329918")
    monkeypatch.setenv("GOOGLE_ADS_CUSTOMER_ID", "not-a-customer-id")
    monkeypatch.delenv("GCP_BIGQUERY_READONLY_KEY", raising=False)
    config = cc.load_billing_env()
    assert config is not None
    assert (config.project, config.dataset) == (
        "w266-project-329918",
        cc.DEFAULT_BILLING_DATASET,
    )


def test_a_backquote_in_the_dataset_name_is_refused(monkeypatch):
    """Both names are interpolated into a backquoted table reference."""
    monkeypatch.setenv("GCP_BILLING_BQ_PROJECT", "proj")
    monkeypatch.setenv("GCP_BILLING_BQ_DATASET", "billing_export`) UNION ALL (SELECT 1")
    with pytest.raises(cc.CostConfigError) as caught:
        cc.load_billing_env()
    assert "GCP_BILLING_BQ_DATASET" in str(caught.value)


def test_the_billing_project_falls_back_to_the_key_that_reads_it(monkeypatch):
    monkeypatch.delenv("GCP_BILLING_BQ_PROJECT", raising=False)
    monkeypatch.delenv("GCP_BILLING_BQ_DATASET", raising=False)
    monkeypatch.setenv(
        "GCP_BIGQUERY_READONLY_KEY", '{"project_id": "w266-project-329918"}'
    )
    config = cc.load_billing_env()
    assert config is not None and config.project == "w266-project-329918"


def test_a_source_that_covers_part_of_the_window_is_averaged_over_what_it_covers():
    """A five-day-old export over a month is not five days of month-long spend."""
    frame = pd.DataFrame(
        [
            (dt.date(2026, 7, 2) - dt.timedelta(days=n), "p", "Cloud Run", "usd", 50.0)
            for n in range(5)
        ],
        columns=["day", "project", "line_item", "currency", "cost"],
    )
    burn = cc.window(frame, 30, provider="Google Cloud", now=dt.date(2026, 7, 2))
    assert burn.per_day == pytest.approx(250.0 / 30)
    loaded = cc.window(
        frame, 30, provider="Google Cloud", now=dt.date(2026, 7, 2), loaded=5
    )
    assert loaded.cost == 250.0
    assert loaded.per_day == pytest.approx(50.0)
    assert loaded.monthly == pytest.approx(1500.0)


def test_a_source_that_covers_more_than_the_window_is_averaged_over_the_window():
    """The window is the shorter of the two, always."""
    frame = pd.DataFrame(
        [
            (dt.date(2026, 7, 2) - dt.timedelta(days=n), "p", "Cloud Run", "usd", 10.0)
            for n in range(7)
        ],
        columns=["day", "project", "line_item", "currency", "cost"],
    )
    burn = cc.window(
        frame, 7, provider="Google Cloud", now=dt.date(2026, 7, 2), loaded=400
    )
    assert burn.days_loaded == 7
    assert burn.per_day == pytest.approx(10.0)


def test_a_partly_held_previous_period_draws_no_trend_at_all():
    """Two days of last month against a whole month is a rise nobody saw."""
    frame = pd.DataFrame(
        [
            (dt.date(2026, 7, 2) - dt.timedelta(days=n), "p", "Cloud Run", "usd", 10.0)
            for n in range(32)
        ],
        columns=["day", "project", "line_item", "currency", "cost"],
    )
    drawn = cc.verdicts(
        cc.window(frame, 30, provider="Google Cloud", now=dt.date(2026, 7, 2))
    )
    assert any("Spend is up" in line for line in drawn), drawn
    silent = cc.verdicts(
        cc.window(
            frame,
            30,
            provider="Google Cloud",
            now=dt.date(2026, 7, 2),
            loaded=32,
            comparable=False,
        )
    )
    assert not any("Spend is" in line for line in silent), silent
    assert not any("This spend is new" in line for line in silent), silent
    # The bill itself is still reported, and still by service.
    assert "$300 on Google Cloud in 30 days" in silent[0], silent[0]


def test_a_cloud_service_with_cache_in_its_name_is_not_context_sent_again():
    """Memorystore for Memcached is a cache; it is not a re-sent prompt."""
    frame = pd.DataFrame(
        [
            (dt.date(2026, 7, 2), "p", "Cloud Memorystore for Memcached", "usd", 80.0),
            (dt.date(2026, 7, 2), "p", "Cloud Run", "usd", 20.0),
        ],
        columns=["day", "project", "line_item", "currency", "cost"],
    )
    cloud = cc.verdicts(
        cc.window(frame, 7, provider="Google Cloud", now=dt.date(2026, 7, 2))
    )
    assert not any("cached context" in line for line in cloud), cloud
    tokens = cc.verdicts(cc.window(frame, 7, now=dt.date(2026, 7, 2)))
    assert any("cached context" in line for line in tokens), tokens


def test_coverage_ignores_the_rounding_rows_dated_at_the_billing_period_start():
    """The live export's MIN is 1 June and its usage starts on the 28th."""
    captured = {}

    class Coverage(_FakeBigQuery):
        def query(self, sql, job_config=None):
            captured["sql"] = sql
            return _FakeJob([{"first": dt.date(2026, 6, 28), "last": dt.date(2026, 7, 2)}])

    first, last = cc.billing_coverage(Coverage([]), "p.d.t")
    assert (first, last) == (dt.date(2026, 6, 28), dt.date(2026, 7, 2))
    # A day is only history where it holds a charge worth a cent.
    assert "HAVING ABS(SUM(cost)) >= 0.01" in captured["sql"], captured["sql"]


def test_a_credit_that_cancels_a_service_leaves_no_minus_zero_row():
    """A free tier nets to a fraction below zero, and printed as "$-0.00"."""

    class Charged(_FakeBigQuery):
        def query(self, sql, job_config=None):
            return _FakeJob(
                [],
                pd.DataFrame(
                    [
                        (dt.date(2026, 7, 2), "p", "Cloud Storage", "usd", -0.000159),
                        (dt.date(2026, 7, 2), "p", "Cloud Run", "usd", 12.0),
                    ],
                    columns=["day", "project", "line_item", "currency", "cost"],
                ),
            )

    frame = cc.cloud_costs(Charged([]), "p.d.t", 7, now=dt.date(2026, 7, 2))
    assert list(frame["line_item"]) == ["Cloud Run"], frame


def test_a_truncated_stripe_read_draws_no_trend_against_the_days_it_missed():
    entries = pd.DataFrame(
        [
            (dt.date(2026, 7, 2) - dt.timedelta(days=n), "application_fee", "", 100.0, 0.0, "usd")
            for n in range(14)
        ],
        columns=["day", "type", "category", "amount", "fee", "currency"],
    )
    whole = cc.ledger_window(entries, 7, now=dt.date(2026, 7, 2))
    cut = cc.ledger_window(entries, 7, now=dt.date(2026, 7, 2), comparable=False)
    assert whole.prev_net == cut.prev_net == 700.0
    assert not any("on the 7 days before" in line for line in cc.stripe_verdicts(cut))


def test_a_charge_with_no_service_name_still_appears_in_the_breakdown():
    """An adjustment row carries no service, and would leave the table short."""
    captured = {}

    class Named(_FakeBigQuery):
        def query(self, sql, job_config=None):
            captured["sql"] = sql
            return _FakeJob([], pd.DataFrame(columns=["day", "project", "line_item", "currency", "cost"]))

    cc.cloud_costs(Named([]), "p.d.t", 7, now=dt.date(2026, 7, 2))
    assert "COALESCE(service.description, 'unattributed')" in captured["sql"]


def test_whether_a_capped_read_reaches_past_the_window_it_is_asked_about():
    """10,000 entries is two months of a small platform and a week of a big one."""
    entries = pd.DataFrame(
        [(dt.date(2026, 7, 2) - dt.timedelta(days=n), "application_fee", "", 1.0, 0.0, "usd")
         for n in range(20)],
        columns=["day", "type", "category", "amount", "fee", "currency"],
    )
    assert cc.reaches_past(entries, 7, now=dt.date(2026, 7, 2))
    # The read stops inside the window it is asked about: its own figures are
    # short of sales, not merely its comparison.
    assert not cc.reaches_past(entries, 30, now=dt.date(2026, 7, 2))
    assert not cc.reaches_past(entries.iloc[:0], 7, now=dt.date(2026, 7, 2))


def test_the_window_never_ends_on_a_day_the_export_is_still_writing():
    """Hours of today's charges are not a day, and would read as a cheap one."""

    class Reached(_FakeBigQuery):
        def query(self, sql, job_config=None):
            return _FakeJob(
                [{"first": dt.date(2026, 6, 1), "last": dt.date(2026, 7, 2)}]
            )

    first, last = cc.billing_coverage(Reached([]), "p.d.t", now=dt.date(2026, 7, 2))
    assert (first, last) == (dt.date(2026, 6, 1), dt.date(2026, 7, 1))
    # An export whose only day is today has no whole day in it yet.
    class Fresh(_FakeBigQuery):
        def query(self, sql, job_config=None):
            return _FakeJob(
                [{"first": dt.date(2026, 7, 2), "last": dt.date(2026, 7, 2)}]
            )

    assert cc.billing_coverage(Fresh([]), "p.d.t", now=dt.date(2026, 7, 2)) == (
        None,
        None,
    )
