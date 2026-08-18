"""AppTest smoke: the Business tab renders the funnel, with and without keys.

Amplitude is stubbed, so this proves the rendering path and the config branches,
not the REST contract (that needs the real keys).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
# The harness is generated below and read back by AppTest, which runs it in
# this process; the env var is how it finds the same checkout.
os.environ["DASHBOARD_REPO"] = REPO

HARNESS = str(Path(__file__).resolve().parent / "_funnel_harness.py")

HARNESS_SOURCE = '''
import os
import sys

import pandas as pd

sys.path.insert(0, os.environ["DASHBOARD_REPO"])

import app as dashboard
import data_layer
import amplitude_client

MODE = os.getenv("FUNNEL_MODE", "live")


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


data_layer.fetch_tickets = _tickets
data_layer.fetch_all_priorities = lambda *a, **k: ["Highest", "High", "Normal"]
data_layer.fetch_all_users = lambda *a, **k: {}
data_layer.fetch_available_transition_statuses = lambda *a, **k: ["Done"]
dashboard.github_client.load_github_env = lambda: None
# The CRM is out of scope here, and a missing key is one of its own branches.
dashboard.orders_client.load_medusa_env = lambda: None
# The funnel is one panel on a page of them, and rendering the page renders the
# rest: refused here rather than left to whatever keys the machine has, so this
# check reads no money source it is not about.
dashboard.ads_client.load_ads_env = lambda: None
dashboard.cost_client.load_openai_env = lambda: None
dashboard.cost_client.load_stripe_env = lambda: None
dashboard.cost_client.load_billing_env = lambda: None
dashboard.merchant_client.load_merchant_env = lambda: None

if MODE == "nokeys":
    dashboard.amplitude_client.load_amplitude_env = lambda: None
elif MODE == "halfkeys":
    def _half():
        raise amplitude_client.AmplitudeConfigError("only one key set")

    dashboard.amplitude_client.load_amplitude_env = _half
else:
    # A different base URL per mode, because @st.cache_data lives in the process
    # rather than the script run: the same key would serve the previous mode's
    # numbers back to this one.
    dashboard.amplitude_client.load_amplitude_env = lambda: (
        "k",
        "s",
        f"https://x-{MODE}",
    )

    NOW = [28000, 600, 220, 150, 105]
    if MODE == "notop":
        # The product-view event renamed in the storefront: no funnel at all,
        # while tens of thousands of people are still using the shop.
        NOW = [0, 0, 0, 0, 0]
    if MODE == "hollow":
        # A window in which nobody got past the product page: every rate below it
        # is over nobody, and must not read as a total collapse.
        NOW = [28000, 0, 0, 0, 0]
    # The period before: fewer product-page viewers, but a better cart step, so
    # the trend column has to be able to say worse and better in one table.
    BEFORE = [21000, 600, 200, 140, 90]

    def _funnel(creds, steps, days, offset_days=0):
        users = (BEFORE if offset_days else NOW)[: len(steps)]
        frame = pd.DataFrame(
            {
                "step": [s.label for s in steps],
                "event": [s.event for s in steps],
                "users": users,
            }
        )
        return frame.assign(
            from_previous=(frame["users"] / frame["users"].shift(1)).fillna(0.0),
            from_start=frame["users"] / frame["users"].iloc[0],
            lost=(frame["users"].shift(1) - frame["users"]).fillna(0).astype(int),
        )

    dashboard.amplitude_client.funnel = _funnel
    # Keyed by event rather than by position: the dashboard now fetches a tab's
    # event counts as separate concurrent calls (one event per call, batched
    # through _parallel) rather than one call carrying the whole list, and a
    # real Amplitude count depends only on which event was asked for, not on
    # how many others rode along with it or in what order.
    EVENT_COUNTS = {
        "_active": 38482,  # "Used the site", read alone as context for the funnel.
        "app_error_occurred": 1587,
        "cart_product_add_failed": 12,
        "cart_checkout_blocked": 4,
        "checkout_payment_failed": 3,
        "search_zero_results": 39,
        "vossai_opened": 800,
        "vossai_query_submitted": 250,
        "vossai_zero_results": 40,
    }

    # Recorded on the amplitude_client module itself, not a harness-local list,
    # because the harness re-executes as a script on every run() while the
    # outer test process's `import amplitude_client` shares the same module
    # object - this is how the assertions below can see what was actually
    # asked for, per run, without the harness and the outer script needing a
    # channel of their own.
    amplitude_client.event_user_calls = []

    def _event_users(creds, events, days, offset_days=0):
        amplitude_client.event_user_calls.extend(e.event for e in events)
        return pd.DataFrame(
            {
                "label": [e.label for e in events],
                "event": [e.event for e in events],
                "users": [EVENT_COUNTS.get(e.event, 0) for e in events],
            }
        )

    dashboard.amplitude_client.event_users = _event_users

    BREAKDOWNS = {
        "page_type": [("pdp", 2100), ("other", 240), ("checkout", 30)],
        "message": [("Loading chunk # failed", 1900), ("Network request failed", 470)],
    }

    def _breakdown(creds, event, prop, days):
        return pd.DataFrame(BREAKDOWNS[prop], columns=["value", "events"])

    dashboard.amplitude_client.event_breakdown = _breakdown

# The Business page directly, as the other smoke tests do: main() renders the
# Engineering page first and would want Jira credentials to do it.
dashboard.inject_styles()
dashboard._reset_reports()
dashboard._render_business()
'''

with open(HARNESS, "w") as handle:
    handle.write(HARNESS_SOURCE)

from streamlit.testing.v1 import AppTest  # noqa: E402


def run(mode: str) -> AppTest:
    os.environ["FUNNEL_MODE"] = mode
    test = AppTest.from_file(HARNESS, default_timeout=120)
    test.run()
    return test


def texts(test: AppTest) -> str:
    parts = [c.value for c in test.caption] + [
        m.value for m in test.markdown
    ]
    return "\n".join(str(p) for p in parts)


for mode, expect in (
    ("nokeys", "AMPLITUDE_API_KEY"),
    ("halfkeys", "misconfigured"),
):
    test = run(mode)
    assert not test.exception, (mode, [e.value for e in test.exception])
    body = texts(test)
    assert expect in body, (mode, body[-800:])
    print(f"{mode}: ok ({expect!r} shown)")

test = run("live")
assert not test.exception, [e.value for e in test.exception]
heads = [h.value for h in test.subheader]
assert "Product Funnel & Friction" in heads, heads
frames = [df.value for df in test.dataframe]
funnel = next(f for f in frames if "From previous step" in list(f.columns))
print(funnel.to_string(index=False))
metrics = {m.label: m.value for m in test.metric}
print({k: v for k, v in metrics.items() if "drop" in k.lower() or "order" in k.lower()})
assert metrics["Used the site (30d)"] == "38,482", metrics
assert metrics["Product page to order placed"] == "0.4%", metrics
assert metrics["Biggest drop-off"] == "Added to cart", metrics["Biggest drop-off"]
friction = next(f for f in frames if "Of all visitors" in list(f.columns))
print(friction.to_string(index=False))
assert friction["People"].max() == 1587
# Errors are quoted against everybody who used the site, not only the people who
# got as far as the funnel's first step.
assert friction["Of all visitors"].iloc[0] == "4.1%", friction.to_string(index=False)
assert funnel["Lost here"].iloc[0] == "\u2014", funnel["Lost here"].tolist()

# Gating: "What went wrong" is the default-open tab, so its events (plus the
# lone "Used the site" read beside the funnel) must have been fetched - but
# Voss AI is the tab nobody opened, and asking it anything at all would be
# exactly the request this change was meant to stop.
import amplitude_client as ac_gate  # noqa: E402

called = set(ac_gate.event_user_calls)
assert called == {
    "_active",
    "app_error_occurred",
    "cart_product_add_failed",
    "cart_checkout_blocked",
    "checkout_payment_failed",
    "search_zero_results",
}, called
assert not called & {"vossai_opened", "vossai_query_submitted", "vossai_zero_results"}, called
print("tab gating: ok (Voss AI not fetched while its tab is closed)")

# Trend: the previous window had 21,000 product-page viewers against this one's
# 28,000, and a cart step that kept 2.9% against this one's 2.1% - so the cart
# step must read as worse even though more people came.
trend_column = next(c for c in funnel.columns if c.startswith("vs previous"))
assert funnel[trend_column].iloc[0] == "\u2014", funnel[trend_column].tolist()
print("trend column:", funnel[trend_column].tolist())
assert metrics["Product page (30d)"] == "28,000", metrics
cart_trend = funnel[trend_column].iloc[1]
assert cart_trend.startswith("-"), cart_trend

# Verdicts: a sentence per step, in people rather than percentages.
body = texts(test)
assert "of every 100 people who got as far as" in body, body[-1200:]
assert "The one to fix first is" in body, body[-1200:]
print(
    "\n".join(
        line
        for line in body.splitlines()
        if "every 100 people" in line or "fix first" in line
    )
)

# Error breakdowns: where, and what it said.
where = next(f for f in frames if "Where it happened" in list(f.columns))
what = next(f for f in frames if "What it said" in list(f.columns))
assert where["Times"].tolist() == ["2,100", "240", "30"], where["Times"].tolist()
assert what["What it said"].iloc[0] == "Loading chunk # failed"
print(where.to_string(index=False))
print(what.to_string(index=False))
print("live: ok")

# A window with an empty mid-funnel step: neither the table nor the sentences may
# claim that a screen nobody reached lost everybody.
hollow = run("hollow")
if "Read the shop's figures" in [b.label for b in hollow.button]:
    hollow = hollow.button(key="business_open").click().run()
assert not hollow.exception, [e.value for e in hollow.exception]
hollow_frames = [df.value for df in hollow.dataframe]
hollow_funnel = next(
    f for f in hollow_frames if "From previous step" in list(f.columns)
)
print(hollow_funnel.to_string(index=False))
rates = hollow_funnel["From previous step"].tolist()
assert rates[1] == "0.0%", rates
assert rates[2:] == ["\u2014"] * 3, rates
hollow_body = texts(hollow)
assert "nobody reached" in hollow_body, hollow_body[-1200:]
print(
    "\n".join(line for line in hollow_body.splitlines() if "nobody reached" in line)
)
print("hollow: ok")

# The funnel's own first step empty: the funnel goes, the errors stay. They are
# the likeliest explanation of why it went, so they must not go with it.
notop = run("notop")
if "Read the shop's figures" in [b.label for b in notop.button]:
    notop = notop.button(key="business_open").click().run()
assert not notop.exception, [e.value for e in notop.exception]
notes = [str(i.value) for i in notop.info]
assert any("no funnel to draw" in n for n in notes), notes
notop_metrics = {m.label: m.value for m in notop.metric}
assert notop_metrics["Used the site (30d)"] == "38,482", notop_metrics
notop_frames = [df.value for df in notop.dataframe]
assert not any("From previous step" in list(f.columns) for f in notop_frames)
notop_friction = next(f for f in notop_frames if "Of all visitors" in list(f.columns))
assert notop_friction["People"].max() == 1587, notop_friction.to_string(index=False)
assert any("What it said" in list(f.columns) for f in notop_frames), "breakdown gone"
print(notop_friction.to_string(index=False))
print("no first step: ok")

import app as dashboard  # noqa: E402
import amplitude_client as ac  # noqa: E402


def _frame(users: list[int]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "step": [f"S{i}" for i in range(len(users))],
            "event": [f"e{i}" for i in range(len(users))],
            "users": users,
        }
    )
    return frame.assign(
        from_previous=(frame["users"] / frame["users"].shift(1)).fillna(0.0).fillna(0.0),
        from_start=(frame["users"] / frame["users"].iloc[0]) if users[0] else 0.0,
        lost=(frame["users"].shift(1) - frame["users"]).fillna(0).astype(int),
    )


now = _frame([1000, 500, 200])
# The previous period had nobody at the middle step, so its 200/0 rate is unknown
# rather than 0%: reporting the difference would announce an improvement that was
# only an absence of data.
before = _frame([1000, 0, 0])
assert dashboard._rate_delta(now, before, 2, "from_previous") is None
assert dashboard._rate_delta(now, before, 1, "from_previous") == "+50.0pp"

# A real few must not be rounded away to nobody, in either direction.
assert dashboard._per_hundred(0.021) == "2 of every 100"
assert dashboard._per_hundred(0.004) == "fewer than 1 of every 100"
assert dashboard._per_hundred(0.0) == "0 of every 100"
assert dashboard._per_hundred(0.996) == "more than 99 of every 100"
assert dashboard._per_hundred(1.0) == "100 of every 100"

# Amplitude series carrying an object per interval must not raise.
assert ac._as_ints([{"value": 3}, {"count": 4}, {"other": 1}, 5, None]) == [3, 4, 0, 5, 0]

# Standing still must not be drawn as a green arrow upwards.
assert dashboard._delta_arrow(None) == {}
assert dashboard._delta_arrow("flat")["delta_color"] == "off"
assert dashboard._delta_arrow("+0 people")["delta_color"] == "off"
assert dashboard._delta_arrow("+3.3pp")["delta_color"] == "normal"
assert dashboard._delta_arrow("-1,000 people")["delta_color"] == "normal"
print("edge cases: ok")
