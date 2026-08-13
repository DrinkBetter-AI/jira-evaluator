"""AppTest smoke: the Business page's price-competitiveness panel, per state.

The Merchant Center read itself is stubbed at the cache boundary; what this
proves is the rendering, the config branches, and that the tiles and sentences
say what the offers do.
"""

from __future__ import annotations

import datetime as _dt
import os
import sys

from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
# The harness is generated below and read back by AppTest, which runs it in
# this process; the env var is how it finds the same checkout.
os.environ["DASHBOARD_REPO"] = REPO

import pandas as pd  # noqa: E402
import app as dashboard  # noqa: E402
import merchant_client  # noqa: E402

# The harness runs in this process and replaces the sales read with a stub, so
# the real one is held here while it is still the real one.
_SALES_READ = dashboard._offer_sales_cached

HARNESS = str(Path(__file__).resolve().parent / "_benchmark_harness.py")

HARNESS_SOURCE = '''
import os
import sys

import pandas as pd

sys.path.insert(0, os.environ["DASHBOARD_REPO"])

import app as dashboard
import merchant_client

MODE = os.getenv("BENCHMARK_MODE", "live")

dashboard.github_client.load_github_env = lambda: None
dashboard.amplitude_client.load_amplitude_env = lambda: None
dashboard.ads_client.load_ads_env = lambda: None
dashboard.cost_client.load_openai_env = lambda: None
dashboard.cost_client.load_stripe_env = lambda: None
dashboard.cost_client.load_billing_env = lambda: None


def offers(rows):
    frame = pd.DataFrame(
        rows, columns=["offer", "title", "brand", "price", "benchmark", "currency"]
    )
    return frame.assign(
        gap=(frame["price"] - frame["benchmark"]) / frame["benchmark"]
    )


DEAR = offers(
    [
        ("a", "Atalon Cabernet Sauvignon", "Atalon", 84.99, 28.42, "USD"),
        ("b", "Villa Pozzi Pinot Grigio", "Villa Pozzi", 16.99, 5.74, "USD"),
        ("c", "A fair wine", "Fair", 20.40, 20.00, "USD"),
        ("d", "A cheap wine", "Cheap", 8.00, 10.00, "USD"),
    ]
)

CLICKS = pd.DataFrame(
    [
        {"offer": "a", "clicks": 4, "impressions": 400},
        {"offer": "b", "clicks": 90, "impressions": 2000},
        {"offer": "d", "clicks": 30, "impressions": 900},
    ]
)

NO_CLICKS = merchant_client.Demand(pd.DataFrame())

# The shop's own order book, keyed on the same offers: the wine under the market
# sells on a third of its clicks, the dear one on a tenth.
SOLD = pd.DataFrame(
    [
        {"offer": "d", "bottles": 10, "revenue": 80.0},
        {"offer": "b", "bottles": 9, "revenue": 152.91},
    ]
)

# Whose wine each offer is, as the catalogue would answer it. One of the shops
# has a comma in its name, which is a name and not two shops, and one bottle is
# stocked by two of them.
LISTED = {
    "a": ("Little International Wine, Inc",),
    "b": ("Yiannis",),
    "c": ("Yiannis",),
    "d": ("Blackbear", "Yiannis"),
}

if MODE in ("sold", "merchant", "nomatch"):
    dashboard.orders_client.load_medusa_env = lambda: dashboard.orders_client.DbConfig(
        "host", "db", "user", "secret", 5432
    )
    dashboard._offer_sales_cached = lambda source, days, today: (
        SOLD.iloc[0:0] if MODE == "nomatch" else SOLD
    )
    dashboard._offer_merchants_cached = lambda source, offers: {
        offer: LISTED[offer] for offer in offers if offer in LISTED
    }
else:
    dashboard.orders_client.load_medusa_env = lambda: None

SUGGESTED = pd.DataFrame(
    [
        {
            "offer": "a",
            "title": "Atalon Cabernet Sauvignon",
            "price": 84.99,
            "suggested": 45.00,
            "clicks_change": 0.6,
            "conversions_change": 0.4,
        }
    ]
)

if MODE == "unset":
    dashboard.merchant_client.load_merchant_env = lambda: None
elif MODE == "misconfigured":
    def _bad():
        raise merchant_client.MerchantConfigError(
            "GOOGLE_MERCHANT_ID should be the numeric Merchant Center id"
        )

    dashboard.merchant_client.load_merchant_env = _bad
else:
    dashboard.merchant_client.load_merchant_env = lambda: merchant_client.Merchant(
        "5304150122", None
    )
    if MODE == "refused":
        def _refused(account, country):
            raise merchant_client.MerchantConfigError(
                "Merchant Center refused the credential. Add the service account "
                "as a user of the account under Settings, People and access."
            )

        dashboard._price_benchmark_cached = _refused
    elif MODE == "broken":
        def _broken(account, country):
            raise RuntimeError("503 Service Unavailable")

        dashboard._price_benchmark_cached = _broken
    elif MODE == "empty":
        dashboard._price_benchmark_cached = lambda account, country: dashboard.BenchmarkRead(
            merchant_client.Prices(offers([]), "", ()),
            merchant_client.Insights(pd.DataFrame()),
            NO_CLICKS,
        )
    elif MODE == "capped":
        dashboard._price_benchmark_cached = lambda account, country: dashboard.BenchmarkRead(
            merchant_client.Prices(DEAR, "USD", (), True),
            merchant_client.Insights(pd.DataFrame()),
            NO_CLICKS,
        )
    elif MODE == "blind":
        dashboard._price_benchmark_cached = lambda account, country: dashboard.BenchmarkRead(
            merchant_client.Prices(DEAR, "USD", ()),
            merchant_client.Insights(SUGGESTED),
            NO_CLICKS,
        )
    elif MODE == "unread":
        dashboard._price_benchmark_cached = lambda account, country: dashboard.BenchmarkRead(
            merchant_client.Prices(DEAR, "USD", ()),
            merchant_client.Insights(SUGGESTED),
            merchant_client.Demand(pd.DataFrame(), read=False),
        )
    elif MODE == "eur":
        dashboard._price_benchmark_cached = lambda account, country: dashboard.BenchmarkRead(
            merchant_client.Prices(DEAR, "USD", ("EUR",)),
            merchant_client.Insights(pd.DataFrame()),
            NO_CLICKS,
        )
    else:
        dashboard._price_benchmark_cached = lambda account, country: dashboard.BenchmarkRead(
            merchant_client.Prices(DEAR, "USD", ()),
            merchant_client.Insights(SUGGESTED),
            merchant_client.Demand(CLICKS),
        )

dashboard.inject_styles()
dashboard._reset_reports()
dashboard._render_business()
'''

with open(HARNESS, "w") as handle:
    handle.write(HARNESS_SOURCE)

from streamlit.testing.v1 import AppTest  # noqa: E402


def run(mode: str) -> AppTest:
    os.environ["BENCHMARK_MODE"] = mode
    test = AppTest.from_file(HARNESS, default_timeout=180)
    test.run()
    assert not test.exception, (mode, [e.value for e in test.exception])
    assert "Price competitiveness" in [h.value for h in test.subheader], mode
    return test


def texts(test: AppTest) -> str:
    parts = (
        [c.value for c in test.caption]
        + [m.value for m in test.markdown]
        + [w.value for w in test.warning]
    )
    return "\n".join(str(part) for part in parts)


# Every state that cannot show a figure says why, rather than leaving a reader to
# read a missing panel as competitive pricing.
for mode, expect in (
    ("unset", "Set GOOGLE_MERCHANT_ID"),
    ("misconfigured", "numeric Merchant Center id"),
    ("refused", "People and access"),
    ("broken", "Could not read the price benchmarks"),
    ("empty", "has a benchmark yet"),
):
    body = texts(run(mode))
    assert expect in body, (mode, body[-900:])
    print(f"{mode}: ok ({expect!r} shown)")

# An empty feed names the country it read, which is the usual reason for one.
empty = texts(run("empty"))
assert "Read for US" in empty and "GOOGLE_MERCHANT_COUNTRY" in empty, empty[-600:]
print("an empty feed names the country it read: ok")

live = run("live")
metrics = {m.label: m.value for m in live.metric}
print(metrics)
assert metrics["More expensive than the market"] == "50%", metrics
# Two of the four sit near 2x, so the median of the four gaps is high: a median
# rather than a mean, but a median of a tiny frame is still a big number.
assert metrics["Typical gap"] == "+99%", metrics
assert metrics["Priced products compared"] == "4", metrics

body = texts(live)
assert "50% of 4 priced products cost more here than the market" in body, body[-1200:]
assert "1 is cheaper than the market" in body, body[-1200:]
assert "cut the price on 1" in body, body[-1200:]
assert "+60% clicks" in body, body[-1200:]
assert str(__import__("datetime").date.today()) in body, body[-1200:]
print("live: tiles, table and verdicts agree with the offers")

tables = [frame.value for frame in live.dataframe]


def table_with(*columns: str):
    for frame in tables:
        if set(columns) <= set(frame.columns):
            return frame
    raise AssertionError((columns, [list(f.columns) for f in tables]))


dearest = table_with("Wine", "Brand", "Our price", "Market", "Gap")
assert dearest["Wine"].iloc[0] == "Atalon Cabernet Sauvignon", dearest.to_string()
assert dearest["Gap"].iloc[0] == "+199%", dearest.to_string()
print("dearest offer leads the table: ok")

# The ask list is ranked on demand as well as the gap, so the wine with 90
# clicks and a 196% gap outranks the one with 4 clicks and a 199% gap.
ask = table_with("Cut to match", "Clicks 30d")
assert ask["Wine"].iloc[0] == "Villa Pozzi Pinot Grigio", ask.to_string()
assert ask["Clicks 30d"].iloc[0] == "90", ask.to_string()
assert ask["Cut to match"].iloc[0] == "-66%", ask.to_string()
# Google's suggestion rides along where it has one and is blank where it does not.
assert ask["Google suggests"].iloc[0] == "", ask.to_string()
assert set(ask["Google suggests"]) == {"", "-47%"}, ask.to_string()
# A merchant column needs the catalogue, which this harness has switched off.
assert "Merchant" not in ask.columns, ask.to_string()
print("the ask list is ranked by clicks against the gap: ok")

# The other side of the same read: the one wine under the market, with its
# clicks, and nothing that is over it.
bargain = [
    frame
    for frame in tables
    if "Clicks 30d" in frame.columns and "Cut to match" not in frame.columns
][0]
assert list(bargain["Wine"]) == ["A cheap wine"], bargain.to_string()
assert bargain["Clicks 30d"].iloc[0] == "30", bargain.to_string()
print("the cheaper-than-market list is ranked by clicks: ok")

# The slider is the negotiation: a tenth off leaves both of these dearer still.
slider = live.slider[0]
assert slider.value == 10, slider.value
assert "At 10% off, 0 of these 2 would be at or below the market price" in body, body[
    -1200:
]
# And it moves the table, not only the sentence: each bottle shows what it would
# cost at that percentage off and how far over the market that still leaves it.
assert ask["At -10%"].iloc[0] == "$15.29", ask.to_string()
assert ask["Gap then"].iloc[0] == "+166%", ask.to_string()
after = slider.set_value(30).run()
assert "At 30% off, 0 of these 2" in texts(after), texts(after)[-1200:]
moved = [f.value for f in after.dataframe if "Cut to match" in f.value.columns][0]
assert "At -30%" in moved.columns, moved.to_string()
assert moved["At -30%"].iloc[0] == "$11.89", moved.to_string()
print("the cut slider re-prices the list without re-reading Merchant Center: ok")

assert any(
    "Download the ask list" == button.label for button in after.get("download_button")
), [button.label for button in after.get("download_button")]
print("the ask list can be taken to a meeting: ok")

assert "took 94 of the 94 clicks" in body or "of that demand is one" in body, body[
    -1200:
]
print("the verdict says how much demand a hundred calls covers: ok")

# A performance report that could not be read must not leave a column of zeros
# reading as "nobody wants these", nor an arbitrary hundred posing as a shortlist.
blind = run("blind")
blind_body = texts(blind)
assert "Shopping reported no clicks" in blind_body, blind_body[-1200:]
assert "performance report could not be read" not in blind_body, blind_body[-1200:]
blind_ask = [f.value for f in blind.dataframe if "Cut to match" in f.value.columns][0]
assert "Clicks 30d" not in blind_ask.columns, blind_ask.to_string()
assert blind_ask["Wine"].iloc[0] == "Atalon Cabernet Sauvignon", blind_ask.to_string()
print("a shop nobody clicked is disclosed, not faked: ok")

# And a report that never arrived says that instead of reporting no demand.
unread_body = texts(run("unread"))
assert "performance report could not be read" in unread_body, unread_body[-1200:]
assert "unknown rather than none" in unread_body, unread_body[-1200:]
assert "Shopping reported no clicks" not in unread_body, unread_body[-1200:]
print("an unread report is not reported as no demand: ok")

# The file leaves with the columns the screen shows and no others: a spreadsheet
# is read further from its caption than a screen is, so neither a zero meaning
# "not measured" nor Google's predicted orders may walk into a meeting alone.
unmeasured = dashboard._visible(
    pd.DataFrame(columns=["title", "clicks", "impressions", "price"]),
    ("title", "clicks", "impressions", "price"),
    False,
)
assert unmeasured == ["title", "price"], unmeasured
# The bytes behind the download button are not in the AppTest proto, so the
# export is checked where it is decided: the column list the table is built
# from, against every column the ask list actually carries.
priced = pd.DataFrame(
    [("a", "Atalon", "Atalon", 84.99, 28.42, "USD")],
    columns=["offer", "title", "brand", "price", "benchmark", "currency"],
).assign(gap=(84.99 - 28.42) / 28.42)
suggestion = pd.DataFrame(
    [
        {
            "offer": "a",
            "title": "Atalon",
            "price": 84.99,
            "suggested": 45.0,
            "clicks_change": 0.6,
            "conversions_change": 0.4,
        }
    ]
)
carried = set(
    merchant_client.after_cut(
        merchant_client.ask_list(
            merchant_client.Prices(priced, "USD", ()),
            merchant_client.Demand(
                pd.DataFrame([{"offer": "a", "clicks": 4, "impressions": 40}])
            ),
            merchant_client.Insights(suggestion),
        ),
        0.1,
    ).columns
) | {"merchant"}
for working in ("conversions_change", "clicks_change", "impact", "google_price"):
    assert working in carried, working
exported = dashboard._visible(
    pd.DataFrame(columns=sorted(carried)), dashboard._ASK_COLUMNS, True
)
assert "conversions_change" not in exported, exported
assert "impact" not in exported, exported
assert {"title", "clicks", "price", "cut_price"} <= set(exported), exported
print("the export carries the table's columns and not the working out: ok")

# The order book beside the benchmark: what each price actually sold, and the
# comparison a merchant is meant to be shown.
sold_run = run("sold")
sold_body = texts(sold_run)
sold_tables = [frame.value for frame in sold_run.dataframe]


def sold_table_with(*columns: str):
    for frame in sold_tables:
        if set(columns) <= set(frame.columns):
            return frame
    raise AssertionError((columns, [list(f.columns) for f in sold_tables]))


evidence = sold_table_with("Against the market", "Bottles per 100 clicks")
rates = dict(zip(evidence["Against the market"], evidence["Bottles per 100 clicks"]))
# 10 bottles on the cheap wine's 30 clicks, 9 on the dear one's 90.
assert rates["Cheaper than the market"] == "33", evidence.to_string()
assert rates["More than 25% more expensive"] == "10", evidence.to_string()
# Nobody clicked the wine priced at the market, so it has no rate to report.
assert rates["About the market"] == "\u2014", evidence.to_string()
assert "sold 33 bottles per 100 clicks" in sold_body, sold_body[-1500:]
assert "sold 10." in sold_body, sold_body[-1500:]
assert "comparison rather than an experiment" in sold_body, sold_body[-1500:]
print("the evidence tab and verdict compare sales per click by price band: ok")

# The page a merchant is actually sent: their name on it, and the same bands.
letters = [b.label for b in sold_run.download_button if "page for" in b.label]
assert letters == ["Download a page for The shop"], [
    b.label for b in sold_run.download_button
]
print("the whole shop's picture is offered as a page to send: ok")

# And the same figure per wine, beside the price it was asked to justify.
sold_ask = sold_table_with("Cut to match", "Sold 90d")
assert sold_ask["Wine"].iloc[0] == "Villa Pozzi Pinot Grigio", sold_ask.to_string()
assert sold_ask["Sold 90d"].iloc[0] == "9", sold_ask.to_string()
assert sold_ask["Merchant"].iloc[0] == "Yiannis", sold_ask.to_string()
# A wine that sold nothing is a nought, not a blank the reader has to interpret.
assert "0" in set(sold_ask["Sold 90d"]), sold_ask.to_string()
print("each wine carries what it sold, beside its merchant: ok")

# Without the order book the column goes rather than reading as no sales at all.
assert "Sold 90d" not in ask.columns, ask.to_string()
assert "order book could not be read" in body, body[-1500:]
print("an unreadable order book loses the column, not the truth: ok")

# One merchant at a time, because that is what gets sent to one merchant.
picked = run("merchant")
chooser = picked.selectbox[0]
assert chooser.value == "Every merchant", chooser.value
assert "Yiannis" in chooser.options, chooser.options
# A comma in a shop's name is a name, not two shops nobody stocks anything for.
assert "Little International Wine, Inc" in chooser.options, chooser.options
assert "Inc" not in chooser.options, chooser.options
mine = chooser.select("Yiannis").run()
assert not mine.exception, [e.value for e in mine.exception]
figures = {m.label: m.value for m in mine.metric}
# Yiannis lists three of the four wines - one well over the market, one within
# the 2% that counts as the same price, one under it - so its own headline is a
# third rather than the shop's half.
assert figures["Priced products compared"] == "3", figures
assert figures["More expensive than the market"] == "33%", figures
only_mine = [f.value for f in mine.dataframe if "Merchant" in f.value.columns]
assert only_mine, [list(f.value.columns) for f in mine.dataframe]
for frame in only_mine:
    for names in frame["Merchant"]:
        assert "Yiannis" in names.split(", "), frame.to_string()
# The wine two of them stock names both, and is still Yiannis' to answer for.
shared = [f for f in only_mine if "A cheap wine" in set(f["Wine"])]
assert shared, [f.to_string() for f in only_mine]
assert "Blackbear, Yiannis" in set(shared[0]["Merchant"]), shared[0].to_string()
print("the panel can be cut to one merchant, figures and files alike: ok")

# And the page to send carries whoever was picked, not the whole shop.
theirs = [b.label for b in mine.download_button if "page for" in b.label]
assert theirs == ["Download a page for Yiannis"], [
    b.label for b in mine.download_button
]
print("the page to send is named for the merchant it is about: ok")

# An order book that opened and matched nothing is not a shop that sold nothing.
nothing = texts(run("nomatch"))
assert "No bottles in the order book match these listings" in nothing, nothing[-900:]
assert "Sold 90d" not in nothing, nothing[-900:]
assert "bottles per 100 clicks" not in nothing.lower(), nothing[-900:]
print("a join that matched nothing says so rather than printing zeros: ok")

capped = texts(run("capped"))
assert "read rather than all of them" in capped, capped[-900:]
assert "read rather than all of them" not in body
print("a capped read is disclosed: ok")

eur = texts(run("eur"))
assert "also quotes EUR" in eur, eur[-900:]
assert "also quotes" not in body
print("a second currency is set aside rather than mixed: ok")

# An ice pack ships with the wine and is not one, so it never reaches a band.
_read = dashboard.orders_client.fetch_offer_sales
_env = dashboard.orders_client.load_medusa_env
try:
    dashboard.orders_client.load_medusa_env = (
        lambda: dashboard.orders_client.DbConfig("host", "db", "user", "secret", 5432)
    )
    dashboard.orders_client.fetch_offer_sales = lambda config, days: pd.DataFrame(
        [
            {"offer": "d", "handle": "yiannis-a-cheap-wine", "bottles": 10, "revenue": 80.0},
            {"offer": "i", "handle": "ice-pack", "bottles": 400, "revenue": 1200.0},
        ]
    )
    kept = _SALES_READ(
        dashboard.orders_client.load_medusa_env().label, 90, _dt.date.today()
    )
finally:
    dashboard.orders_client.fetch_offer_sales = _read
    dashboard.orders_client.load_medusa_env = _env
assert list(kept["handle"]) == ["yiannis-a-cheap-wine"], kept.to_string()
print("an add-on shipped with the wine is not counted as bottles sold: ok")

print("all price-competitiveness scenarios passed")
