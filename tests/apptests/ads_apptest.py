"""AppTest smoke: the Business tab renders the ads panel, with and without a dataset.

BigQuery is stubbed at the query boundary - the SQL itself was verified against
the live dataset - so what this proves is the rendering, the config branches, and
that CRM orders and Google's own conversions are aligned on the same days.
"""

from __future__ import annotations

import os
import re
import sys

from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
# The harness is generated below and read back by AppTest, which runs it in
# this process; the env var is how it finds the same checkout.
os.environ["DASHBOARD_REPO"] = REPO

HARNESS = str(Path(__file__).resolve().parent / "_ads_harness.py")

HARNESS_SOURCE = '''
import datetime as dt
import os
import sys

import pandas as pd

sys.path.insert(0, os.environ["DASHBOARD_REPO"])

import app as dashboard
import ads_client
import orders_client

MODE = os.getenv("ADS_MODE", "live")

# Stripe's ledger, stubbed everywhere except the mode that is about a refused
# key: the commission figure is asserted, and a live ledger moves it daily.
# The harness runs in the same process on every mode, so the real reader is
# stashed on the module the first time round and put back for that one mode.
if not hasattr(dashboard, "_REAL_STRIPE_ENV"):
    dashboard._REAL_STRIPE_ENV = dashboard.cost_client.load_stripe_env

if MODE == "badstripe":
    dashboard.cost_client.load_stripe_env = dashboard._REAL_STRIPE_ENV
else:
    dashboard.cost_client.load_stripe_env = lambda: "sk_test_stub"

    def _ledger(days):
        rows = [
            {
                "day": dt.date.today() - dt.timedelta(days=n),
                "type": "application_fee",
                "category": "connect_collection_transfer",
                "amount": 100.0,
                "fee": 0.0,
                "currency": "usd",
            }
            for n in range(2 * days)
        ]
        columns = ["day", "type", "category", "amount", "fee", "currency"]
        return pd.DataFrame(rows, columns=columns), False

    dashboard._stripe_ledger_cached = _ledger
    # Disputes are a separate call, and the Payments section makes it.
    dashboard._stripe_disputes_cached = lambda days: 0

    # OpenAI's bill likewise: the printable report asserts the AI section is in
    # it, and that section is only drawn when an admin key answers.
    dashboard.cost_client.load_openai_env = lambda: "sk-admin-stub"

    def _ai_costs(days):
        rows = [
            {
                "day": dt.date.today() - dt.timedelta(days=n),
                "project": "vinovoss",
                "line_item": "gpt-4o-mini, inputs",
                "cost": 4.0,
                "currency": "usd",
            }
            for n in range(days)
        ]
        columns = ["day", "project", "line_item", "cost", "currency"]
        return pd.DataFrame(rows, columns=columns)

    dashboard._openai_costs_cached = _ai_costs



TODAY = dt.date.today()
YESTERDAY = TODAY - dt.timedelta(days=1)


def _tickets(*a, **k):
    now = pd.Timestamp.now(tz="UTC")
    return pd.DataFrame(
        [
            {
                "key": "MB-1",
                "summary": "A ticket",
                "description": "x" * 200,
                "status": "In Progress",
                "status_category": "In Progress",
                "priority": "High",
                "assignee": "Ali",
                "assignee_account_id": "a1",
                "reporter": "Angel",
                "created": now - pd.Timedelta(days=5),
                "updated": now - pd.Timedelta(days=1),
                "last_meaningful_activity": now - pd.Timedelta(days=1),
                "due_date": pd.NaT,
                "issue_type": "Task",
                "project_key": "MB",
                "project_name": "Marketplace",
                "parent_key": None,
                "parent_type": None,
                "epic_key": None,
                "epic_summary": None,
                "epic_status": None,
                "labels": "",
                "resolution": None,
                "status_category_changed_date": now - pd.Timedelta(days=1),
                "original_estimate": 3600,
                "sprint_id": None,
                "sprint_name": None,
                "sprint_state": None,
                "carry_over_count": 0,
            }
        ]
    )


dashboard.fetch_tickets = _tickets
dashboard.fetch_all_priorities = lambda *a, **k: ["Highest", "High", "Normal"]
dashboard.fetch_all_users = lambda *a, **k: {}
dashboard.fetch_available_transition_statuses = lambda *a, **k: ["Done"]
dashboard.github_client.load_github_env = lambda: None
dashboard.amplitude_client.load_amplitude_env = lambda: None


def _order_book():
    """Four captured orders a day for the last 40 days, $200 each.

    Enough that the CRM comparison has to pick out the right days rather than
    happening to be right.
    """
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for day in range(40):
        for index in range(4):
            rows.append(
                {
                    "id": f"o{day}-{index}",
                    "display_id": day * 10 + index,
                    "created_at": now - pd.Timedelta(days=day, hours=6),
                    "status": "completed",
                    "payment_status": "captured",
                    "fulfillment_status": "fulfilled",
                    "total": 200.0,
                    "refunded_total": 0.0,
                    # The ad account bills in USD, so "eur" makes the two
                    # incomparable, which the panel must decline rather than divide.
                    # In "gbp" the shop is kept in step with the ad account,
                    # where what is checked is the symbol, not the mismatch.
                    "currency_code": {"eur": "eur", "gbp": "gbp"}.get(MODE, "usd"),
                    "email": "a@b.c",
                    "sales_channel_id": "sc",
                }
            )
    return orders_client.OrderBook(
        orders=pd.DataFrame(rows),
        items=pd.DataFrame(
            columns=[
                "order_id",
                "created_at",
                "title",
                "product_title",
                "variant_title",
                "quantity",
                "unit_price",
                "total",
                "currency_code",
            ]
        ),
        window_days=365,
        synced_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
    )


if MODE == "noorders":
    dashboard.orders_client.load_medusa_env = lambda: None
else:
    dashboard.orders_client.load_medusa_env = lambda: orders_client.DbConfig(
        "crm.example", "medusa", "reader", "secret", 5432
    )
    dashboard._order_book = lambda source, days: _order_book()
    dashboard.fetch_store_prefixes_cached = lambda *a, **k: {}

if MODE == "nodataset":
    dashboard.ads_client.load_ads_env = lambda: None
elif MODE == "badconfig":
    def _bad():
        raise ads_client.AdsConfigError("GOOGLE_ADS_CUSTOMER_ID is not a customer ID")

    dashboard.ads_client.load_ads_env = _bad
elif MODE == "unreadable":
    dashboard.ads_client.load_ads_env = lambda: ads_client.AdsConfig(
        "w266-project-329918", "google_ads", None, None
    )

    def _explode(*a, **k):
        raise RuntimeError("404 Not found: Dataset w266-project-329918:google_ads")

    dashboard._ads_cached = _explode
else:
    dashboard.ads_client.load_ads_env = lambda: ads_client.AdsConfig(
        "w266-project-329918", f"google_ads_{MODE}", None, None
    )

    def _stats(days: int) -> pd.DataFrame:
        rows = []
        # One day only in "partial": a fresh transfer, before the backfill. In
        # "quiet" every day falls in the previous period: the account stopped.
        span = 1 if MODE == "partial" else 2 * days
        start = days + 1 if MODE == "quiet" else 1
        for back in range(start, span + 1):
            day = YESTERDAY - dt.timedelta(days=back - 1)
            # The second campaign spends and converts nothing: money with nothing
            # to show for it, which is the line the panel exists to print.
            # Google credits fewer conversions than the shop has orders, which
            # is the gap the panel is meant to notice.
            rows.append((1, day, 13.49, 118, 1.0, 200.0))
            rows.append((2, day, 66.58, 458, 0.0, 0.0))
        frame = pd.DataFrame(
            rows,
            columns=[
                "campaign_id",
                "day",
                "cost",
                "clicks",
                "conversions",
                "conversion_value",
            ],
        )
        return frame.assign(impressions=1000)

    NAMES = pd.DataFrame(
        [
            (1, "Sales-Shopping-sept17", "ENABLED", "SHOPPING", 60.0),
            (2, "AI-PMax-Bestsellers-FeedOnly", "ENABLED", "PERFORMANCE_MAX", 80.0),
        ],
        columns=["campaign_id", "campaign", "status", "channel", "budget"],
    )

    # The panel reads the widest window once and cuts the narrower one out of
    # that frame, keyed on the day rather than the window.
    WIDEST = max(ads_client.LOOKBACK_WINDOWS)

    def _cached(project: str, dataset: str, today: dt.date):
        days = WIDEST
        if MODE == "empty":
            return dashboard.AdsRead(
                pd.DataFrame(), pd.DataFrame(), "", "USD", None, []
            )
        if MODE in ("paused", "fresh"):
            # No rows either way. In "paused" the transfer has years behind it,
            # so the account stopped; in "fresh" it began two days ago, so most
            # of the window was never loaded and nothing ran in the rest.
            return dashboard.AdsRead(
                stats=pd.DataFrame(),
                names=pd.DataFrame(),
                account="Vinovoss.com New Google Ads account",
                currency="USD",
                history_start=YESTERDAY
                - dt.timedelta(days=400 if MODE == "paused" else 1),
                other_currencies=[],
            )
        # Where the transfer's history begins, which is what tells a window that
        # has not loaded from an account that simply paused.
        history = YESTERDAY if MODE == "partial" else YESTERDAY - dt.timedelta(days=400)
        return dashboard.AdsRead(
            stats=_stats(days),
            names=NAMES,
            account="Vinovoss.com New Google Ads account",
            currency="GBP" if MODE == "gbp" else "USD",
            history_start=history,
            other_currencies=["EUR"] if MODE == "gbp" else [],
        )

    dashboard._ads_cached = _cached

# main() builds an st.navigation over two pages and runs whichever one the
# browser asked for, which an AppTest has no browser to do: the page function
# is where the sections are, and one page per run is what the app does too.
dashboard.inject_styles()
dashboard._reset_reports()
if os.getenv("HARNESS_PAGE", "business") == "engineering":
    dashboard._render_engineering_page()
else:
    dashboard._render_business()
'''

with open(HARNESS, "w") as handle:
    handle.write(HARNESS_SOURCE)

from streamlit.testing.v1 import AppTest  # noqa: E402


STRIPE_KEY = os.environ.get("STRIPE_READONLY_API_KEY", "")


def run(mode: str) -> AppTest:
    os.environ["ADS_MODE"] = mode
    # A key that can move money: the payments reader refuses it, and the ads
    # panel must fall back to the configured rate rather than take the tab down.
    # Restored either side, since the harness runs in this same process.
    if mode == "badstripe":
        os.environ["STRIPE_READONLY_API_KEY"] = "sk_live_thiswouldbeafullsecretkey"
    elif STRIPE_KEY:
        os.environ["STRIPE_READONLY_API_KEY"] = STRIPE_KEY
    test = AppTest.from_file(HARNESS, default_timeout=180)
    test.run()
    if "Read the shop's figures" in [b.label for b in test.button]:
        test = test.button(key="business_open").click().run()
    assert not test.exception, (mode, [e.value for e in test.exception])
    return test


def texts(test: AppTest) -> str:
    parts = (
        [c.value for c in test.caption]
        + [m.value for m in test.markdown]
        + [i.value for i in test.info]
        + [w.value for w in test.warning]
    )
    return "\n".join(str(p) for p in parts)


# The panel is always announced, so the reader can see the figures are missing
# rather than assume nothing was spent.
for mode, expect in (
    ("nodataset", "GOOGLE_ADS_BQ_PROJECT"),
    ("badconfig", "misconfigured"),
    ("unreadable", "Could not read `w266-project-329918.google_ads`"),
    ("empty", "holds no spend yet"),
):
    test = run(mode)
    assert "Ads Spend & Return" in [h.value for h in test.subheader], mode
    body = texts(test)
    assert expect in body, (mode, body[-900:])
    print(f"{mode}: ok ({expect!r} shown)")

test = run("live")
metrics = {m.label: m.value for m in test.metric}
print({k: v for k, v in metrics.items() if k in
       ("Spend (30d)", "Orders (CRM)", "Ad spend per order",
        "Revenue per $1 spent", "Google's own conversions")})
# 30 days at $80.07 a day.
assert metrics["Spend (30d)"] == "$2,402.10", metrics["Spend (30d)"]
# Four captured orders a day over the same 30 days, at $200 each.
assert metrics["Orders (CRM)"] == "120", metrics["Orders (CRM)"]
assert metrics["Ad spend per order"] == "$20.02", metrics["Ad spend per order"]
assert metrics["Revenue per $1 spent"] == "10.0x", metrics["Revenue per $1 spent"]
assert metrics["Google's own conversions"] == "30", metrics["Google's own conversions"]
# The commission is Stripe's own, read over the same days the spend covers -
# ending yesterday, where the ads end - so the figure moves with the real
# ledger rather than an assumed 12% of revenue; merchants are on their own
# rates and this counts what each of them actually paid. Asserted by shape and
# against the sentence beneath it, because the ledger is live.
earned = metrics["Commission per $1 spent"]
assert re.fullmatch(r"\d+\.\d\d", earned), earned
live_body = texts(test)
assert f"${earned} back for every $1 of ad spend" in live_body, live_body[-900:]
assert "Goal 1.00" in live_body, live_body[-900:]
assert "pay for themselves" in live_body, live_body[-900:]
assert "what Stripe charged across every merchant" in live_body, live_body[-900:]
assert "of commission actually charged" in live_body, live_body[-900:]

frames = [df.value for df in test.dataframe]
campaigns = next(f for f in frames if "Cost per conversion" in list(f.columns))
print(campaigns.to_string(index=False))
# Dearest first, whatever it returned.
assert campaigns["Campaign"].tolist() == [
    "AI-PMax-Bestsellers-FeedOnly",
    "Sales-Shopping-sept17",
], campaigns["Campaign"].tolist()
assert campaigns["Cost per conversion"].iloc[0] == "\u2014", campaigns.to_string()

body = texts(test)
assert "$1,997 went to 1 campaign that recorded no conversion at all" in body, body[-1500:]
assert "Google claims 30 conversions where the CRM has 120 orders" in body, body[-1500:]
assert "of ad spend per order" in body
print("\n".join(line for line in body.splitlines() if line.startswith("- **")))
print("live: ok")

# A window that is mostly not there yet must say so rather than report a month of
# spend that is really one day of it.
partial = run("partial")
partial_body = texts(partial)
assert "Only 1 of these 30 days have been loaded" in partial_body, partial_body[-900:]
assert "history starts on" in partial_body, partial_body[-900:]
partial_metrics = {m.label: m.value for m in partial.metric}
assert partial_metrics["Spend (30d)"] == "$80.07", partial_metrics["Spend (30d)"]
# The CRM side is held to the same single day, so cost per order stays honest:
# four orders that day, not a month of them.
assert partial_metrics["Orders (CRM)"] == "4", partial_metrics["Orders (CRM)"]
assert partial_metrics["Ad spend per order"] == "$20.02"
print("partial: ok")

# An account that stopped spending is a quiet account, not a broken feed: the
# transfer writes no row for a day on which nothing ran.
quiet = run("quiet")
quiet_body = texts(quiet)
assert "No spend recorded in the last 30 days" in quiet_body, quiet_body[-900:]
assert "days have been loaded" not in quiet_body, quiet_body[-900:]
print("quiet: ok")

# No rows at all is a paused account when the transfer has history behind it,
# and a transfer that has not run when it has none.
paused = run("paused")
paused_body = texts(paused)
assert "stopped advertising rather than missing figures" in paused_body, (
    paused_body[-900:]
)
assert "holds no spend yet" not in paused_body, paused_body[-900:]
print("paused: ok")

# A Stripe key the payments reader refuses is a reason to quote the rate, not a
# reason for the Business tab to disappear.
bad_stripe = run("badstripe")
bad_metrics = {m.label: m.value for m in bad_stripe.metric}
bad_body = texts(bad_stripe)
assert bad_metrics["Commission per $1 spent"] == "1.20", bad_metrics
assert "$1.20 back for every $1 of ad spend" in bad_body, bad_body[-900:]
assert "Commission is assumed at 12%" in bad_body, bad_body[-900:]
print("badstripe: ok")

# A transfer two days old with nothing spent in those two days is neither: most
# of the window never arrived, which is the thing worth saying.
fresh_body = texts(run("fresh"))
assert "Only 2 of these 30 days have been loaded" in fresh_body, fresh_body[-900:]
assert "stopped advertising" not in fresh_body, fresh_body[-900:]
print("fresh: ok")

# Takings in another currency cannot be divided by spend in this one.
eur = run("eur")
eur_metrics = {m.label: m.value for m in eur.metric}
eur_body = texts(eur)
assert eur_metrics["Orders (CRM)"] == "120", eur_metrics["Orders (CRM)"]
assert eur_metrics["Revenue per $1 spent"] == "\u2014", eur_metrics
# Commission survives it, and should: it is what Stripe charged in dollars
# against spend in dollars, whatever currency the shop's own takings are in.
assert eur_metrics["Commission per $1 spent"] == earned, eur_metrics
assert f"${earned} back for every $1 of ad spend" in eur_body, eur_body[-900:]
assert "of revenue at 12%" not in eur_body, eur_body[-900:]
assert "takings are in EUR" in eur_body, eur_body[-900:]
assert "on money actually captured" not in eur_body, eur_body[-900:]
assert "$20 of ad spend per order" in eur_body, eur_body[-900:]
assert "average basket" not in eur_body, eur_body[-900:]
print("eur: ok")

# A non-USD account must be quoted in its own currency, tiles and sentences
# alike, and accounts billing in something else are left out of the total.
gbp = run("gbp")
gbp_metrics = {m.label: m.value for m in gbp.metric}
gbp_body = texts(gbp)
assert gbp_metrics["Spend (30d)"] == "\u00a32,402.10", gbp_metrics["Spend (30d)"]
assert "Revenue per \u00a31 spent" in gbp_metrics, list(gbp_metrics)
assert "\u00a320 of ad spend per order" in gbp_body, gbp_body[-900:]
assert "$" not in gbp_body.split("What this means")[-1][:1200], gbp_body[-900:]
assert "Only the accounts billing in GBP" in gbp_body, gbp_body[-900:]
print("gbp: ok")

# No CRM key: the ads still read, and the CRM columns say unknown rather than nil.
noorders = run("noorders")
noorders_metrics = {m.label: m.value for m in noorders.metric}
assert noorders_metrics["Orders (CRM)"] == "\u2014", noorders_metrics
assert noorders_metrics["Ad spend per order"] == "\u2014", noorders_metrics
assert noorders_metrics["Spend (30d)"] == "$2,402.10", noorders_metrics
noorders_body = texts(noorders)
assert "bought" not in noorders_body.split("What this means")[-1][:400]
print("no CRM: ok")

import app as dashboard  # noqa: E402

# A fall in spend must be drawn as a fall: Streamlit reads the first character,
# so a currency symbol in front of the sign would colour every cut as a rise.
assert dashboard._money_delta(-412.9, "usd") == "-$412.90"
assert dashboard._money_delta(412.9, "usd") == "+$412.90"
assert dashboard._money_delta(0.0, "usd") == "flat"
assert dashboard._delta_arrow(dashboard._money_delta(0.0, "usd"))["delta_color"] == "off"
assert dashboard._delta_arrow("-$412.90")["delta_color"] == "normal"
print("edge cases: ok")
