"""AppTest smoke: the Business tab renders the Burn panel in each config state.

The provider reads are stubbed at the cache boundary - both were verified against
the live OpenAI organization cost endpoint and the live Stripe account - so what
this proves is the rendering, the config branches, and that the sentences say
what the numbers do.
"""

from __future__ import annotations

import os
import subprocess
import sys

from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
# The harness is generated below and read back by AppTest, which runs it in
# this process; the env var is how it finds the same checkout.
os.environ["DASHBOARD_REPO"] = REPO

HARNESS = str(Path(__file__).resolve().parent / "_burn_harness.py")

HARNESS_SOURCE = '''
import datetime as dt
import os
import sys

import pandas as pd

sys.path.insert(0, os.environ["DASHBOARD_REPO"])

import app as dashboard
import cost_client

MODE = os.getenv("BURN_MODE", "live")

TODAY = dt.date.today()


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
dashboard.orders_client.load_medusa_env = lambda: None
dashboard.ads_client.load_ads_env = lambda: None
# Merchant Center is drawn by the same page and is not what this check is about.
dashboard.merchant_client.load_merchant_env = lambda: None


def _openai_costs(days: int) -> pd.DataFrame:
    """Sixty days of charges: $30 a day now, $10 a day before, half of it cache."""
    rows = []
    for back in range(2 * days):
        day = TODAY - dt.timedelta(days=back)
        recent = back < days
        rows.append((day, "Vinovoss", "gpt-5.6-terra, cached input", 15.0 if recent else 5.0))
        rows.append((day, "Vinovoss", "gpt-5.6-terra, input", 14.0 if recent else 4.5))
        rows.append((day, "Default project", "gpt-5.4, input", 1.0 if recent else 0.5))
    return pd.DataFrame(rows, columns=["day", "project", "line_item", "cost"]).assign(
        currency="usd"
    )


def _lapsed_costs(days: int) -> pd.DataFrame:
    """Charges in the period before the window and none inside it."""
    rows = [
        (TODAY - dt.timedelta(days=days + back), "Vinovoss", "gpt-5.6-terra, input", 20.0)
        for back in range(days)
    ]
    return pd.DataFrame(rows, columns=["day", "project", "line_item", "cost"]).assign(
        currency="usd"
    )


def _mixed_currency_costs(days: int) -> pd.DataFrame:
    """Dollars from one project and a larger euro bill from another."""
    frame = _openai_costs(days)
    euros = pd.DataFrame(
        [(TODAY, "Euro project", "gpt-5.6-terra, input", 99999.0)],
        columns=["day", "project", "line_item", "cost"],
    ).assign(currency="eur")
    return pd.concat([frame, euros], ignore_index=True)


def _cloud_costs(days: int) -> pd.DataFrame:
    """Sixty days of Cloud charges: $50 a day now, $40 a day before."""
    rows = []
    for back in range(2 * days):
        day = TODAY - dt.timedelta(days=back)
        recent = back < days
        rows.append((day, "w266", "Cloud Run", 30.0 if recent else 24.0))
        rows.append((day, "w266", "AlloyDB", 15.0 if recent else 12.0))
        rows.append((day, "w266", "Cloud Storage", 5.0 if recent else 4.0))
    return pd.DataFrame(rows, columns=["day", "project", "line_item", "cost"]).assign(
        currency="usd"
    )


def _plain_entries(days: int) -> pd.DataFrame:
    """An ordinary account's ledger: its own charges, and Stripe's fee on them."""
    rows = [
        (TODAY - dt.timedelta(days=back), "charge", "charge", 200.0, 7.0)
        for back in range(2 * days)
    ]
    return pd.DataFrame(
        rows, columns=["day", "type", "category", "amount", "fee"]
    ).assign(currency="usd")


def _stripe_entries(days: int) -> pd.DataFrame:
    """A Connect platform's ledger: commission in, a refund, a payout, no fees."""
    rows = []
    for back in range(2 * days):
        day = TODAY - dt.timedelta(days=back)
        rows.append((day, "application_fee", "platform_earning", 100.0 if back < days else 80.0, 0.0))
    rows.append((TODAY, "application_fee_refund", "platform_earning_refund", -25.0, 0.0))
    rows.append((TODAY, "payout", "payout", -1500.0, 0.0))
    if MODE == "fees":
        # A platform that also took a charge of its own, which Stripe did bill for.
        rows.append((TODAY, "charge", "charge", 500.0, 17.5))
    return pd.DataFrame(
        rows, columns=["day", "type", "category", "amount", "fee"]
    ).assign(currency="usd")


# Google Cloud, whose export is a dataset rather than a key: configured and
# written to unless the mode is about what happens when it is not.
if MODE == "nocloud":
    dashboard.cost_client.load_billing_env = lambda: None
else:
    dashboard.cost_client.load_billing_env = lambda: cost_client.Billing(
        "w266-project-329918", "billing_export"
    )
    EXPORT = "w266-project-329918.billing_export.gcp_billing_export_v1_01DD38"
    if MODE == "cloudempty":
        dashboard._cloud_costs_cached = lambda days, today: dashboard.CloudRead(
            pd.DataFrame(), None, None, ()
        )
    elif MODE == "cloudnew":
        # Switched on the day before yesterday: three days of the thirty.
        dashboard._cloud_costs_cached = lambda days, today: dashboard.CloudRead(
            _cloud_costs(days)[
                _cloud_costs(days)["day"] >= TODAY - dt.timedelta(days=2)
            ],
            TODAY - dt.timedelta(days=2),
            TODAY,
            (EXPORT,),
        )
    elif MODE == "cloudexact":
        # History exactly as long as the window: nothing at all behind it, which
        # is not a comparison built on nought days.
        dashboard._cloud_costs_cached = lambda days, today: dashboard.CloudRead(
            _cloud_costs(days)[
                _cloud_costs(days)["day"] > TODAY - dt.timedelta(days=days)
            ],
            TODAY - dt.timedelta(days=days - 1),
            TODAY,
            (EXPORT,),
        )
    elif MODE == "cloudpartial":
        # Thirty-two days of export under a thirty-day window: the live case,
        # where the earlier period holds two days and no trend can be drawn.
        dashboard._cloud_costs_cached = lambda days, today: dashboard.CloudRead(
            _cloud_costs(days)[
                _cloud_costs(days)["day"] > TODAY - dt.timedelta(days=32)
            ],
            TODAY - dt.timedelta(days=31),
            TODAY,
            (EXPORT,),
        )
    elif MODE == "cloudeur":
        # A second billing currency, which is set aside rather than added.
        dashboard._cloud_costs_cached = lambda days, today: dashboard.CloudRead(
            pd.concat(
                [
                    _cloud_costs(days),
                    _cloud_costs(days).head(3).assign(currency="eur"),
                ],
                ignore_index=True,
            ),
            TODAY - dt.timedelta(days=200),
            TODAY,
            (EXPORT,),
        )
    elif MODE == "cloudtwo":
        # Two billing accounts exporting into one dataset.
        dashboard._cloud_costs_cached = lambda days, today: dashboard.CloudRead(
            _cloud_costs(days),
            TODAY - dt.timedelta(days=200),
            TODAY,
            (EXPORT, EXPORT + "_OTHER"),
        )
    elif MODE == "cloudlag":
        # Written in arrears and still backfilling: a fortnight behind today,
        # with the window ending where the charges do.
        behind = TODAY - dt.timedelta(days=14)
        dashboard._cloud_costs_cached = lambda days, today: dashboard.CloudRead(
            _cloud_costs(days).assign(
                day=lambda frame: frame["day"] - dt.timedelta(days=14)
            ),
            behind - dt.timedelta(days=200),
            behind,
            (EXPORT,),
        )
    else:
        dashboard._cloud_costs_cached = lambda days, today: dashboard.CloudRead(
            _cloud_costs(days),
            TODAY - dt.timedelta(days=200),
            TODAY,
            (EXPORT,),
        )

if MODE == "nokeys":
    dashboard.cost_client.load_openai_env = lambda: None
    dashboard.cost_client.load_stripe_env = lambda: None
elif MODE == "badkeys":
    def _bad_openai():
        raise cost_client.CostConfigError("OPENAI_ADMIN_KEY looks like a project key")

    def _bad_stripe():
        raise cost_client.CostConfigError("STRIPE_READONLY_API_KEY is a full secret key")

    dashboard.cost_client.load_openai_env = _bad_openai
    dashboard.cost_client.load_stripe_env = _bad_stripe
elif MODE == "refused":
    dashboard.cost_client.load_openai_env = lambda: "sk-admin-x"
    dashboard.cost_client.load_stripe_env = lambda: "rk_live_x"

    def _refused_openai(days):
        raise cost_client.CostConfigError("OpenAI refused the key for organization costs.")

    def _refused_stripe(days):
        raise cost_client.CostConfigError("Stripe refused the restricted key.")

    dashboard._openai_costs_cached = _refused_openai
    dashboard._stripe_cached = _refused_stripe
elif MODE == "mixed":
    dashboard.cost_client.load_openai_env = lambda: "sk-admin-x"
    dashboard.cost_client.load_stripe_env = lambda: None
    dashboard._openai_costs_cached = _mixed_currency_costs
elif MODE == "plain":
    dashboard.cost_client.load_openai_env = lambda: None
    dashboard.cost_client.load_stripe_env = lambda: "rk_live_x"
    dashboard._stripe_cached = lambda days: (_plain_entries(days), False, 0)
elif MODE == "lapsed":
    dashboard.cost_client.load_openai_env = lambda: "sk-admin-x"
    dashboard.cost_client.load_stripe_env = lambda: None
    dashboard._openai_costs_cached = _lapsed_costs
elif MODE == "quietweek":
    # Busy two months ago, nothing since: the download is full, the window empty.
    dashboard.cost_client.load_openai_env = lambda: None
    dashboard.cost_client.load_stripe_env = lambda: "rk_live_x"
    dashboard._stripe_cached = lambda days: (
        _stripe_entries(days).assign(day=TODAY - dt.timedelta(days=55)),
        False,
        0,
    )
elif MODE == "quietdisputes":
    dashboard.cost_client.load_openai_env = lambda: None
    dashboard.cost_client.load_stripe_env = lambda: "rk_live_x"
    dashboard._stripe_cached = lambda days: (pd.DataFrame(), False, 3)
elif MODE == "quiet":
    dashboard.cost_client.load_openai_env = lambda: "sk-admin-x"
    dashboard.cost_client.load_stripe_env = lambda: "rk_live_x"
    dashboard._openai_costs_cached = lambda days: pd.DataFrame()
    dashboard._stripe_cached = lambda days: (pd.DataFrame(), False, 0)
else:
    dashboard.cost_client.load_openai_env = lambda: "sk-admin-x"
    dashboard.cost_client.load_stripe_env = lambda: "rk_live_x"
    dashboard._openai_costs_cached = _openai_costs
    dashboard._stripe_cached = lambda days: (
        _stripe_entries(days),
        MODE == "truncated",
        2 if MODE == "disputed" else 0,
    )

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


def run(mode: str) -> AppTest:
    os.environ["BURN_MODE"] = mode
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


# Every state says what it can and cannot read, rather than leaving a reader to
# assume nothing was spent.
for mode, expect in (
    ("nokeys", "Set OPENAI_ADMIN_KEY to show it"),
    ("badkeys", "looks like a project key"),
    ("refused", "OpenAI refused the key"),
    ("quiet", "reports no charges for this organization"),
):
    test = run(mode)
    assert "Burn" in [h.value for h in test.subheader], mode
    body = texts(test)
    assert expect in body, (mode, body[-900:])
    print(f"{mode}: ok ({expect!r} shown)")

# A missing key must not silence the other provider.
nokeys_body = texts(run("nokeys"))
assert "Set STRIPE_READONLY_API_KEY" in nokeys_body, nokeys_body[-900:]
assert "full secret key" in texts(run("badkeys")), "stripe key check not reported"
assert "Stripe refused" in texts(run("refused")), "stripe refusal not reported"
assert "Stripe recorded no money moving" in texts(run("quiet")), "quiet stripe not reported"
print("both providers reported in every state: ok")

test = run("live")
metrics = {m.label: m.value for m in test.metric}
print(metrics)
# 30 days at $30 a day.
assert metrics["OpenAI (30d)"] == "$900.00", metrics["OpenAI (30d)"]
assert metrics["At this rate, a month"] == "$900.00", metrics["At this rate, a month"]
# $15 of every $30 is cached context.
assert metrics["Cached context"] == "50%", metrics["Cached context"]
# $3,000 of commission less $25 refunded, and nothing charged by Stripe.
assert metrics["Commission kept (30d)"] == "$2,975.00", metrics["Commission kept (30d)"]
assert metrics["Refunded"] == "$25.00", metrics["Refunded"]
assert metrics["Stripe's own fees"] == "$0.00", metrics["Stripe's own fees"]
assert metrics["Paid out to the bank"] == "$1,500.00", metrics["Paid out to the bank"]

frames = [df.value for df in test.dataframe]
lines = next(f for f in frames if "Line item" in list(f.columns))
print(lines.to_string(index=False))
# Dearest line first, and the project split named beneath.
assert lines["Line item"].iloc[0] == "gpt-5.6-terra, cached input", lines.to_string()

body = texts(test)
assert "$900 on OpenAI in 30 days" in body, body[-1500:]
assert "up $600 (+200%)" in body, body[-1500:]
assert "50% of it is cached context" in body, body[-1500:]
assert "By project: Vinovoss" in body, body[-1500:]
assert "$2,975 of commission kept in 30 days" in body, body[-1500:]
# Net of the refund, like the tile above it: $2,975 against $2,400.
assert "Commission is up $575 (+24%)" in body, body[-1500:]
assert "charged this account nothing to process it" in body, body[-1500:]
print("\n".join(line for line in body.splitlines() if line.startswith("- **")))
print("live: ok")

# A week reads the same shape without mixing windows: one radio governs both.
week = run("live")
week = week.radio(key="burn_window_days").set_value(7).run()
week_metrics = {m.label: m.value for m in week.metric}
assert week_metrics["OpenAI (7d)"] == "$210.00", week_metrics
assert week_metrics["Commission kept (7d)"] == "$675.00", week_metrics
print("7-day window: ok")

# Disputes are counted, never added to a total: the outcome is still open.
disputed = texts(run("disputed"))
assert "2 disputes opened" in disputed, disputed[-900:]
print("disputes: ok")

# On an account that pays its own card fees, the nil-fees sentence must not appear.
fees = run("fees")
fees_body = texts(fees)
assert {m.label: m.value for m in fees.metric}["Stripe's own fees"] == "$17.50", [
    (m.label, m.value) for m in fees.metric
]
assert "charged this account nothing" not in fees_body, fees_body[-900:]
print("own fees: ok")

# A ledger read that hit its ceiling must say so: Stripe returns newest first,
# so the missing rows are the comparison period and silence would read as growth.
cut = texts(run("truncated"))
assert "oldest days of the comparison period are missing" in cut, cut[-900:]
assert "oldest days of the comparison" not in texts(run("live"))
print("truncated ledger: ok")

# Spend that stopped before the window: the tiles read zero, so no breakdown of
# the earlier period may appear beneath them.
lapsed = run("lapsed")
lapsed_metrics = {m.label: m.value for m in lapsed.metric}
assert lapsed_metrics["OpenAI (30d)"] == "$0.00", lapsed_metrics
assert "By project" not in texts(lapsed), texts(lapsed)[-900:]
print("lapsed spend: ok")

# Euros are never added to dollars, in the project line as in the totals.
mixed = run("mixed")
mixed_body = texts(mixed)
assert {m.label: m.value for m in mixed.metric}["OpenAI (30d)"] == "$900.00"
assert "Euro project" not in mixed_body, mixed_body[-900:]
assert "also billed in EUR" in mixed_body, mixed_body[-900:]
print("a second currency: ok")

# An account that is not a platform takes charges rather than commission, and
# must not read as having kept a negative amount of money.
plain = run("plain")
plain_metrics = {m.label: m.value for m in plain.metric}
assert plain_metrics["Payments kept (30d)"] == "$5,790.00", plain_metrics
assert plain_metrics["Stripe's own fees"] == "$210.00", plain_metrics
assert "of payments kept" in texts(plain), texts(plain)[-900:]
print("a plain account: ok")

# Chargebacks are money at risk, and a window that earned nothing is where they
# matter most: they must not disappear with the takings.
risk = texts(run("quietdisputes"))
assert "3 disputes were opened" in risk, risk[-900:]
print("disputes without takings: ok")

# One Stripe read now serves both panels and spans two months, so a quiet week
# inside a busy quarter must still be reported as quiet rather than as zeroes.
quiet_week = run("quietweek")
quiet_week = quiet_week.radio(key="burn_window_days").set_value(7).run()
assert "Stripe recorded no money moving in the last 7 days" in texts(quiet_week), (
    texts(quiet_week)[-900:]
)
print("a quiet week inside a busy quarter: ok")

# Google Cloud, which is usually the largest of the three bills.
cloud = run("live")
cloud_metrics = {m.label: m.value for m in cloud.metric}
# 30 days at $50 a day, against $40 a day before it.
assert cloud_metrics["Google Cloud (30d)"] == "$1,500.00", cloud_metrics
assert cloud_metrics["A day of Cloud"] == "$50.00", cloud_metrics
assert cloud_metrics["Dearest service"] == "Cloud Run", cloud_metrics
cloud_body = texts(cloud)
assert "$1,500 on Google Cloud in 30 days" in cloud_body, cloud_body[-1200:]
assert "up $300 (+25%)" in cloud_body, cloud_body[-1200:]
assert "Cloud Run is 60% of the bill" in cloud_body, cloud_body[-1200:]
services = next(f for f in [df.value for df in cloud.dataframe] if "Service" in f)
assert services["Service"].iloc[0] == "Cloud Run", services.to_string()
print("google cloud: ok")

# The export is not retroactive, so a window older than it is a shorter period
# wearing a longer label, and the panel has to say which.
young = run("cloudnew")
new = texts(young)
assert "The export only goes back to" in new, new[-1200:]
assert "3 days of charges" in new, new[-1200:]
assert "The export only goes back to" not in cloud_body
# Three days of charges over three days, not over the thirty the export was
# not switched on for: a fifth of the real daily burn is worse than no figure.
young_metrics = {m.label: m.value for m in young.metric}
assert young_metrics["Google Cloud (30d)"] == "$150.00", young_metrics
assert young_metrics["A day of Cloud"] == "$50.00", young_metrics
print("a young export: ok")

# The export is written in arrears, so the days it has not reached are days it
# has no charges for: a window ending today would average real spend over free
# days and report the bill as falling.
lag = run("cloudlag")
lag_metrics = {m.label: m.value for m in lag.metric}
assert lag_metrics["Google Cloud (30d)"] == "$1,500.00", lag_metrics
assert lag_metrics["A day of Cloud"] == "$50.00", lag_metrics
lag_body = texts(lag)
assert "The export has only reached" in lag_body, lag_body[-1200:]
assert "14 days behind today" in lag_body, lag_body[-1200:]
assert "The export has only reached" not in cloud_body
print("an export running behind: ok")

# Thirty-two days of history under a thirty-day window - the export as it
# actually stands - leaves two days to compare a month with, which would read
# as a rise of a thousand per cent. No trend at all is the honest answer.
partial = run("cloudpartial")
partial_body = texts(partial)
assert "too few to compare with" in partial_body, partial_body[-1200:]
# The Cloud rise specifically - the AI panel beside it has a real one.
assert "up $300 (+25%)" not in partial_body, partial_body[-1200:]
assert "This spend is new" not in partial_body, partial_body[-1200:]
assert not [
    m for m in partial.metric if m.label == "Google Cloud (30d)" and m.delta
], [(m.label, m.delta) for m in partial.metric]
print("a previous period the export barely holds: ok")

# An export whose whole history is the window has nothing behind it: saying the
# comparison rests on nought days would be worse than saying there is none.
exact_body = texts(run("cloudexact"))
assert "there is no earlier period to compare them with" in exact_body, exact_body[-800:]
assert "the 0 of them it holds" not in exact_body
print("history exactly as long as the window: ok")

# Two billing accounts in one dataset are two bills, and adding them would be
# the currency mistake in another costume.
two_body = texts(run("cloudtwo"))
assert "The dataset holds 2 billing exports" in two_body, two_body[-800:]
assert "The dataset holds" not in cloud_body
print("two exports in one dataset: ok")

# A second currency in the export is set aside, and the reader is told, as
# everywhere else on the page: two currencies are never added.
eur_body = texts(run("cloudeur"))
assert "Google Cloud also billed in EUR" in eur_body, eur_body[-800:]
print("cloud charges in a second currency: ok")

# The page holding the Cloud bill has to be offered when the billing export is
# the only thing configured, or the bill is set up and never seen. In a clean
# interpreter, since the Ads loader asks the ambient credentials for a project.
probe = subprocess.run(
    [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, %r);"
        "import app;"
        "app.ads_client.default_project = lambda: '';"
        "print(app._business_readable())" % REPO,
    ],
    capture_output=True,
    text=True,
    env={
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "GCP_BILLING_BQ_PROJECT": "w266-project-329918",
    },
)
assert probe.stdout.strip().endswith("True"), (probe.stdout, probe.stderr[-800:])
print("billing alone offers the Business page: ok")

# No table in the dataset - because it was created minutes ago, or because the
# export was never switched on, or because the dataset is not there at all.
# One sentence for all three, and none of them a warning.
empty = texts(run("cloudempty"))
assert "There is no billing export in" in empty, empty[-1200:]
assert "standard usage cost" in empty, empty[-1200:]
print("an export with no table yet: ok")

# And with no export configured at all, the other two providers still report.
none = texts(run("nocloud"))
assert "GCP_BILLING_BQ_PROJECT" in none, none[-1200:]
assert "$900 on OpenAI in 30 days" in none, none[-1200:]
print("no billing export configured: ok")
