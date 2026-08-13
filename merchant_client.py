"""Read-only client for Merchant Center's price reports, via the Merchant API.

What a bottle costs against what everyone else charges for the same bottle is a
number the shop cannot see in its own data: it needs Google's benchmark, which
is the median price other merchants sell that product at. Merchant Center shows
it on a page nobody opens; this module reads it so the dashboard can say it.

Not read from BigQuery, though a Merchant Center transfer once wrote these rows
into ``gmc_raw``: Google deprecated ``export_price_benchmarks``, that transfer
stopped in May 2026, and benchmarks cannot reach BigQuery that way any more. The
Merchant API serves them directly and daily, which is why this module speaks
HTTP rather than SQL like its neighbours.

Two reports are read, both a snapshot of now rather than a series:

``price_competitiveness_product_view``
    Each offer's price and Google's benchmark for it, per country.
``price_insights_product_view``
    Google's suggested price for an offer, and what it predicts a move to that
    price would do to clicks and conversions.

The credential is a service account added to Merchant Center as a reader. There
is no write scope on it and this module has no write verb.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass, field

import pandas as pd
import requests

_MERCHANT_ENV_VAR = "GOOGLE_MERCHANT_ID"
_COUNTRY_ENV_VAR = "GOOGLE_MERCHANT_COUNTRY"
_KEY_ENV_VAR = "GCP_BIGQUERY_READONLY_KEY"

# Benchmarks are published per country, so reading a country the feed does not
# target returns nothing at all rather than an error.
DEFAULT_COUNTRY = "US"

# v1, not v1beta: the beta endpoint was switched off on 28 February 2026 and
# answers every request with a 409 telling you so.
_BASE_URL = "https://merchantapi.googleapis.com/reports/v1"
_SCOPE = "https://www.googleapis.com/auth/content"
_TIMEOUT_SECONDS = 90

# Merchant Center account ids are numeric and about ten digits. Checked because
# it is interpolated into the request path.
_MERCHANT_ID_PATTERN = re.compile(r"^\d{6,16}$")

# Likewise the country, which goes into a quoted literal in the report query.
_COUNTRY_PATTERN = re.compile(r"^[A-Za-z]{2}$")

# The API pages at 1,000 rows and the catalogue is tens of thousands of offers,
# so a ceiling is needed or one refresh walks the whole feed. 25 pages is 25,000
# offers, which covers the catalogue with room to grow.
_PAGE_SIZE = 1000
_MAX_PAGES = 25

# Prices arrive as micros - millionths of a currency unit - as everywhere in
# Google's APIs.
MICROS = 1_000_000

# A price within this of the benchmark is the same price to a shopper, and
# counting it as dearer would make a rounding difference look like a problem.
_SAME_PRICE = 0.02

# How far above the benchmark is worth naming in a table of the worst offenders.
DEAR_GAP = 0.25

# The window the click figures cover. Merchant Center's performance report
# offers fixed ranges rather than an arbitrary number of days, and a month is
# the shortest of them wide enough that a wine selling one bottle a fortnight
# still shows demand.
DEMAND_DAYS = 30

# How many wines a negotiation list holds. A merchant will not reprice five
# thousand bottles; a hundred is a conversation, and the tail below it is worth
# a rounding error of the clicks.
ASK_LIST = 100

# How far back the order book is read for the sales beside each price. A
# quarter rather than the month the clicks cover: a wine sells a few bottles a
# month at best, and a month of orders would put a zero against most of the
# catalogue and call it evidence.
SALES_DAYS = 90

# The bands the evidence is grouped into, as the top of each band and its name.
# Cut at the same 2% that counts as the same price everywhere else, and again
# at the 25% the verdicts already call the band a shopper does not come back
# from.
_BANDS = (
    (-_SAME_PRICE, "Cheaper than the market"),
    (_SAME_PRICE, "About the market"),
    (DEAR_GAP, f"Up to {DEAR_GAP:.0%} dearer"),
)
_LAST_BAND = f"More than {DEAR_GAP:.0%} dearer"

_BAND_COLUMNS = ("band", "listings", "clicks", "bottles", "per_100_clicks")


class MerchantConfigError(RuntimeError):
    """Raised when Merchant Center is unconfigured, or refuses the credential."""


@dataclass(frozen=True)
class Merchant:
    """Which Merchant Center account to read, and who to read it as."""

    account: str
    key_json: str | None = None
    country: str = DEFAULT_COUNTRY


@dataclass(frozen=True)
class Prices:
    """The catalogue's prices against Google's benchmarks for them.

    ``offers`` is one row per offer per country, with the gap already worked out
    as a fraction of the benchmark, so a table of the worst can be cut from it
    without repeating the arithmetic.
    """

    offers: pd.DataFrame
    currency: str = ""
    other_currencies: tuple[str, ...] = field(default=())
    # True when the read hit its page ceiling, so shares below describe the
    # offers that were read rather than the whole catalogue.
    truncated: bool = False

    @property
    def counted(self) -> int:
        """Offers with a benchmark to compare against."""
        return len(self.offers)

    @property
    def dearer(self) -> int:
        """Offers priced meaningfully above the benchmark."""
        return int((self.offers["gap"] > _SAME_PRICE).sum()) if self.counted else 0

    @property
    def cheaper(self) -> int:
        """Offers priced meaningfully below it."""
        return int((self.offers["gap"] < -_SAME_PRICE).sum()) if self.counted else 0

    @property
    def dear_share(self) -> float:
        """The headline: what fraction of the catalogue is dearer than the rest."""
        return self.dearer / self.counted if self.counted else 0.0

    @property
    def median_gap(self) -> float:
        """The typical gap, which is not the average: a few offers sit at 2x."""
        return float(self.offers["gap"].median()) if self.counted else 0.0

    @property
    def worst(self) -> pd.DataFrame:
        """The offers furthest above benchmark, dearest first."""
        if not self.counted:
            return self.offers
        return self.offers.sort_values("gap", ascending=False).reset_index(drop=True)


@dataclass(frozen=True)
class Insights:
    """Where Google thinks a price cut would pay for itself."""

    offers: pd.DataFrame
    # As on ``Prices``: True when the read stopped at its page ceiling, so the
    # count below is of the suggestions read rather than all of them.
    truncated: bool = False

    @property
    def counted(self) -> int:
        return len(self.offers)

    @property
    def clicks_gain(self) -> float:
        """Predicted extra clicks, as a fraction, over the offers it names."""
        return float(self.offers["clicks_change"].mean()) if self.counted else 0.0

    @property
    def conversions_gain(self) -> float:
        return float(self.offers["conversions_change"].mean()) if self.counted else 0.0

    def within(self, offers: pd.DataFrame) -> "Insights":
        """The suggestions for offers that were actually compared.

        The two reports are not one population: this view carries no country,
        benchmark or currency filter, so a count taken straight off it could
        claim a cut on more products than the panel says it compared.
        """
        if self.offers.empty or "offer" not in offers.columns:
            return self
        kept = self.offers[self.offers["offer"].isin(set(offers["offer"]))]
        return Insights(kept.reset_index(drop=True), self.truncated)


@dataclass(frozen=True)
class Demand:
    """What shoppers actually did with each offer, over the last month.

    Clicks are the only demand signal Merchant Center has that the shop cannot
    get from its own order book: they count the shoppers who chose this bottle
    out of a page of competing ones, including every shopper who then bought it
    somewhere cheaper. Conversions are in the report too and are left out - the
    feed carries no conversion tracking, so they are zero on every row and a
    zero that means "not measured" is worse than no column.
    """

    offers: pd.DataFrame
    truncated: bool = False
    # False when the report could not be read at all. An empty frame otherwise
    # means the shop was shown and nobody clicked, which is a finding; a report
    # that never arrived is not, and the panel must not report one as the other.
    read: bool = True

    @property
    def clicks(self) -> int:
        return int(self.offers["clicks"].sum()) if len(self.offers) else 0

    @property
    def measured(self) -> bool:
        """Whether there is any demand to rank by."""
        return self.read and bool(len(self.offers)) and self.clicks > 0

    def against(self, offers: pd.DataFrame) -> pd.DataFrame:
        """``offers`` with clicks and impressions on it, zero where unseen.

        An offer nobody clicked is absent from the performance report rather
        than present with a zero, so the join has to fill rather than drop: a
        wine with no clicks is a real row with no demand, not a missing one.
        """
        if "offer" not in offers.columns:
            return offers
        if self.offers.empty:
            return offers.assign(clicks=0, impressions=0)
        joined = offers.merge(self.offers, on="offer", how="left")
        joined["clicks"] = joined["clicks"].fillna(0).astype(int)
        joined["impressions"] = joined["impressions"].fillna(0).astype(int)
        return joined


@dataclass(frozen=True)
class Sales:
    """What the shop actually sold of each offer, from its own order book.

    The half of the argument Google cannot supply. Merchant Center reports no
    conversions on this account, so the only record of a click that became a
    bottle is the shop's own, and it is the record that matters to a merchant
    being asked to come down: its wine, its sales, at its price.
    """

    offers: pd.DataFrame
    days: int = SALES_DAYS
    # False when the order book could not be read, which is not the same as a
    # wine nobody bought - see ``Demand.read``.
    read: bool = True

    @property
    def bottles(self) -> int:
        return int(self.offers["bottles"].sum()) if len(self.offers) else 0

    @property
    def measured(self) -> bool:
        return self.read and bool(len(self.offers)) and self.bottles > 0

    def measured_against(self, offers: pd.DataFrame) -> bool:
        """Whether any of *these* offers sold anything.

        The shop as a whole selling wine says nothing about the wines on
        screen: filter the panel to one merchant whose handles never matched
        the catalogue and every row joins to zero, which would be the panel
        telling that merchant its wines do not sell on the strength of our own
        failed join.
        """
        if not self.measured:
            return False
        joined = self.against(offers)
        if "bottles" not in joined.columns:
            return False
        return int(joined["bottles"].sum()) > 0

    def against(self, offers: pd.DataFrame) -> pd.DataFrame:
        """``offers`` with bottles sold on it, zero where none were.

        Filled rather than dropped, for the reason ``Demand.against`` fills: a
        wine that sold nothing is the observation, not a missing row.
        """
        if "offer" not in offers.columns:
            return offers
        if self.offers.empty:
            return offers.assign(bottles=0, sold_revenue=0.0)
        columns = self.offers[["offer", "bottles", "revenue"]].rename(
            columns={"revenue": "sold_revenue"}
        )
        joined = offers.merge(
            columns.drop_duplicates(subset="offer"), on="offer", how="left"
        )
        joined["bottles"] = joined["bottles"].fillna(0).astype(int)
        joined["sold_revenue"] = joined["sold_revenue"].fillna(0.0)
        return joined


def price_bands(prices: Prices, demand: Demand, sales: Sales) -> pd.DataFrame:
    """How each price band did, from cheaper than the market to well above it.

    The evidence a merchant asks for, and the only form of it this data can
    honestly carry. Per-wine conversion rates cannot: a bottle bought on two
    clicks would read as a wine that converts every other shopper. Grouped into
    bands the counts are large enough to mean something, and bottles are counted
    per hundred clicks rather than in total so a band with more listings in it
    does not win by being bigger.

    It is a comparison, not an experiment: a wine priced under the market may
    also be a wine people want, and nothing here separates the two.
    """
    if not prices.counted:
        return pd.DataFrame(columns=list(_BAND_COLUMNS))
    frame = sales.against(demand.against(prices.offers))
    edges = [-float("inf")] + [edge for edge, _ in _BANDS] + [float("inf")]
    labels = [name for _, name in _BANDS] + [_LAST_BAND]
    frame = frame.assign(
        band=pd.cut(frame["gap"], bins=edges, labels=labels, right=True)
    )
    grouped = (
        frame.groupby("band", observed=False)
        .agg(
            listings=("offer", "size"),
            clicks=("clicks", "sum"),
            bottles=("bottles", "sum"),
        )
        .reset_index()
    )
    # Per hundred clicks, and blank rather than zero where nobody clicked: a
    # band Google never showed has no rate, and printing 0 would read as a band
    # shoppers saw and refused.
    grouped["per_100_clicks"] = (
        100 * grouped["bottles"] / grouped["clicks"].where(grouped["clicks"] > 0)
    )
    return grouped[list(_BAND_COLUMNS)]


def sales_verdicts(prices: Prices, demand: Demand, sales: Sales) -> list[str]:
    """Whether the shop's own sales say a keener price sells more of it."""
    if not prices.counted:
        return []
    if not (sales.measured_against(prices.offers) and demand.measured):
        return []
    bands = price_bands(prices, demand, sales)
    cheap = bands[bands["band"] == _BANDS[0][1]]
    dear = bands[bands["band"].isin([name for _, name in _BANDS[2:]] + [_LAST_BAND])]
    cheap_clicks = int(cheap["clicks"].sum())
    dear_clicks = int(dear["clicks"].sum())
    if not (cheap_clicks and dear_clicks):
        return []
    cheap_rate = 100 * int(cheap["bottles"].sum()) / cheap_clicks
    dear_rate = 100 * int(dear["bottles"].sum()) / dear_clicks
    if not dear_rate:
        return []
    lines = [
        f"**Wines priced under the market sold {cheap_rate:.0f} bottles per 100 "
        f"clicks - {sales.days} days of orders against the last {DEMAND_DAYS} "
        f"days of clicks - where wines priced above it sold {dear_rate:.0f}.** "
        f"Same shop, same shoppers, {cheap_rate / dear_rate:.1f}x the sales for "
        "the same attention."
    ]
    lines.append(
        "That is a comparison rather than an experiment - a keenly priced wine "
        "may also be a wine people want - but it is the shop's own order book, "
        "which is the number a merchant will argue with."
    )
    return lines


def load_merchant_env() -> Merchant | None:
    """Which Merchant Center account to read, or ``None`` when unconfigured."""
    account = os.getenv(_MERCHANT_ENV_VAR, "").strip()
    if not account:
        return None
    if not _MERCHANT_ID_PATTERN.match(account):
        raise MerchantConfigError(
            f"{_MERCHANT_ENV_VAR} should be the numeric Merchant Center id, "
            "shown at the top right of merchants.google.com."
        )
    country = os.getenv(_COUNTRY_ENV_VAR, "").strip() or DEFAULT_COUNTRY
    if not _COUNTRY_PATTERN.match(country):
        raise MerchantConfigError(
            f"{_COUNTRY_ENV_VAR} should be a two-letter country code, the one "
            "the feed targets in Merchant Center."
        )
    return Merchant(
        account, os.getenv(_KEY_ENV_VAR, "").strip() or None, country.upper()
    )


def access_token(config: Merchant) -> str:
    """A bearer token for the content scope, from the service account or the VM.

    On Cloud Run there is no key: the service is the credential, and the account
    it runs as is the one that has to be added to Merchant Center.
    """
    try:
        import google.auth  # noqa: PLC0415 - optional at import time
        import google.auth.transport.requests as transport  # noqa: PLC0415
        from google.oauth2 import service_account  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - only without the libraries
        raise MerchantConfigError(
            "google-auth is needed to read Merchant Center."
        ) from exc

    if config.key_json:
        try:
            info = json.loads(config.key_json)
        except ValueError as exc:
            raise MerchantConfigError(
                f"{_KEY_ENV_VAR} is not valid JSON: paste the whole service "
                "account key file, braces included."
            ) from exc
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=[_SCOPE]
        )
    else:
        creds, _ = google.auth.default(scopes=[_SCOPE])
    creds.refresh(transport.Request())
    return str(creds.token)


def _search(config: Merchant, token: str, query: str) -> tuple[list[dict], bool]:
    """Every row the report returns, and whether the page ceiling was hit."""
    url = f"{_BASE_URL}/accounts/{config.account}/reports:search"
    headers = {"Authorization": f"Bearer {token}"}
    body: dict[str, object] = {"query": query, "pageSize": _PAGE_SIZE}
    rows: list[dict] = []
    for _ in range(_MAX_PAGES):
        response = requests.post(
            url, headers=headers, json=body, timeout=_TIMEOUT_SECONDS
        )
        if response.status_code in (401, 403):
            raise MerchantConfigError(
                "Merchant Center refused the credential. Add the service "
                "account as a user of the account under Settings, People and "
                "access, with read access."
            )
        response.raise_for_status()
        payload = response.json()
        rows.extend(payload.get("results", []))
        page = payload.get("nextPageToken")
        if not page:
            return rows, False
        body["pageToken"] = page
    return rows, True


def _money(value: dict | None) -> float:
    """A Money field as a number, or zero when the report left it out."""
    if not value:
        return 0.0
    return int(value.get("amountMicros", 0) or 0) / MICROS


def _currency(value: dict | None) -> str:
    return str((value or {}).get("currencyCode", "") or "").upper()


def price_gaps(config: Merchant, token: str, country: str = "") -> Prices:
    """Each offer's price against Google's benchmark for the same product.

    Offers without a benchmark are dropped rather than counted as competitive:
    Google publishes one only where enough other merchants sell the product, and
    a wine nobody else lists has no market price to be dearer than.
    """
    country = country or config.country
    if not _COUNTRY_PATTERN.match(country):
        raise MerchantConfigError(
            f"{country!r} is not a two-letter country code."
        )
    # ``id`` is not wanted, but v1 refuses a query on this view without it.
    query = (
        "SELECT id, offer_id, title, brand, price, benchmark_price, "
        "report_country_code FROM price_competitiveness_product_view "
        f"WHERE report_country_code = '{country.upper()}'"
    )
    rows, truncated = _search(config, token, query)
    frame = pd.DataFrame(
        [
            {
                "offer": str(view.get("offerId", "")),
                "title": str(view.get("title", "")),
                "brand": str(view.get("brand", "")),
                "price": _money(view.get("price")),
                "benchmark": _money(view.get("benchmarkPrice")),
                "currency": _currency(view.get("price")),
            }
            for view in (
                row.get("priceCompetitivenessProductView", {}) for row in rows
            )
        ],
        columns=["offer", "title", "brand", "price", "benchmark", "currency"],
    )
    frame = frame[(frame["benchmark"] > 0) & (frame["price"] > 0)]
    frame, currency, others = main_currency(frame)
    # One row per bottle, and after the currency is chosen rather than before:
    # the view's key is the whole REST id - channel, language, feed label and
    # offer - so a catalogue listed twice would be counted twice in the share
    # above the market and take two places on the ask list with the same wine,
    # but dropping the duplicates first could keep the row in the wrong currency
    # and lose the bottle to the filter below it.
    frame = frame.drop_duplicates(subset="offer")
    frame = frame.assign(
        gap=(frame["price"] - frame["benchmark"]) / frame["benchmark"]
    ).reset_index(drop=True)
    return Prices(frame, currency, others, truncated)


def main_currency(frame: pd.DataFrame) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    """The rows in the currency most of the catalogue is priced in.

    Prices in two currencies are never compared or counted together, and the
    excluded ones are named rather than dropped silently.
    """
    if frame.empty:
        return frame, "", ()
    counts = frame["currency"].value_counts()
    main = str(counts.index[0])
    others = tuple(sorted(str(code) for code in counts.index[1:]))
    return frame[frame["currency"] == main], main, others


def price_insights(config: Merchant, token: str) -> Insights:
    """Where Google suggests a lower price, and what it predicts that would do."""
    query = (
        "SELECT id, offer_id, title, price, suggested_price, "
        "predicted_clicks_change_fraction, "
        "predicted_conversions_change_fraction "
        "FROM price_insights_product_view"
    )
    rows, truncated = _search(config, token, query)
    frame = pd.DataFrame(
        [
            {
                "offer": str(view.get("offerId", "")),
                "title": str(view.get("title", "")),
                "price": _money(view.get("price")),
                "suggested": _money(view.get("suggestedPrice")),
                "clicks_change": float(
                    view.get("predictedClicksChangeFraction", 0.0) or 0.0
                ),
                "conversions_change": float(
                    view.get("predictedConversionsChangeFraction", 0.0) or 0.0
                ),
            }
            for view in (row.get("priceInsightsProductView", {}) for row in rows)
        ],
        columns=[
            "offer",
            "title",
            "price",
            "suggested",
            "clicks_change",
            "conversions_change",
        ],
    )
    # Only the offers Google would actually move, and only downwards: a
    # suggestion to charge more is not what this panel is for.
    frame = frame[(frame["suggested"] > 0) & (frame["suggested"] < frame["price"])]
    return Insights(frame.reset_index(drop=True), truncated)


def product_demand(config: Merchant, token: str, country: str = "") -> Demand:
    """Clicks and impressions per offer over the last month.

    Only the offers with a click: the report holds a row for every product that
    was ever shown, the catalogue is tens of thousands of them, and the ones
    nobody clicked carry no demand to rank by. Marketing method is left out of
    the select on purpose - naming it would segment the report into an ads row
    and an organic row per offer, and demand is demand.

    The shopper's country is not: the prices are compared against one country's
    benchmark, so a click from another one is demand that never saw the price
    being argued about.
    """
    country = country or config.country
    if not _COUNTRY_PATTERN.match(country):
        raise MerchantConfigError(f"{country!r} is not a two-letter country code.")
    query = (
        "SELECT offer_id, clicks, impressions FROM product_performance_view "
        "WHERE date DURING LAST_30_DAYS AND clicks > 0 "
        f"AND customer_country_code = '{country.upper()}'"
    )
    rows, truncated = _search(config, token, query)
    frame = pd.DataFrame(
        [
            {
                "offer": str(view.get("offerId", "")),
                "clicks": int(view.get("clicks", 0) or 0),
                "impressions": int(view.get("impressions", 0) or 0),
            }
            for view in (row.get("productPerformanceView", {}) for row in rows)
        ],
        columns=["offer", "clicks", "impressions"],
    )
    # One row per offer whatever the report segments by: a wine split across two
    # rows is one wine's demand, and left split it would rank as half of it.
    if not frame.empty:
        frame = frame.groupby("offer", as_index=False)[["clicks", "impressions"]].sum()
    return Demand(frame, truncated)


def bargains(prices: Prices, demand: Demand, limit: int = ASK_LIST) -> pd.DataFrame:
    """The wines already cheaper than the market, the most wanted ones first.

    The mirror of the ask list and the half of it nobody asks for: these need no
    merchant's agreement, only a bigger share of the ad budget, because the
    price comparison a shopper does in the next tab comes out in the shop's
    favour.
    """
    if not prices.counted:
        return prices.offers
    cheaper = prices.offers[prices.offers["gap"] < -_SAME_PRICE]
    if cheaper.empty:
        return cheaper.assign(clicks=0, impressions=0)
    frame = demand.against(cheaper)
    frame = frame.assign(under=-frame["gap"] * frame["benchmark"])
    order = ["clicks", "under"] if demand.measured else ["under"]
    return (
        frame.sort_values(order, ascending=False).head(limit).reset_index(drop=True)
    )


def ask_list(
    prices: Prices,
    demand: Demand,
    insights: Insights | None = None,
    limit: int = ASK_LIST,
) -> pd.DataFrame:
    """The bottles worth asking a merchant to reprice, best argument first.

    Ranked on clicks times the gap, which is the demand actually seen times how
    far over the market that demand was asked to pay. It deliberately does not
    predict extra orders: the feed reports no conversions, so any figure in
    orders would be a model of a model. Google's own suggestion is carried
    alongside where it has one, as the second opinion it is, for the few hundred
    offers it covers.

    ``cut`` is what it would take to reach the market price, so a merchant can
    be asked for a number rather than for less.
    """
    if not prices.counted:
        return prices.offers
    dear = prices.offers[prices.offers["gap"] > _SAME_PRICE]
    if dear.empty:
        return dear.assign(clicks=0, impressions=0)
    frame = demand.against(dear)
    frame = frame.assign(
        cut=1 - frame["benchmark"] / frame["price"],
        overpay=frame["price"] - frame["benchmark"],
        impact=frame["clicks"] * frame["gap"],
    )
    if insights is not None:
        # The same discipline the verdicts keep: the insights report carries no
        # country, currency or benchmark filter, so a suggestion is only ours if
        # it names an offer this panel compared - otherwise the percentage below
        # could divide one currency by another. One row per offer, too: an offer
        # id repeated across feed labels would otherwise take several of the
        # hundred places with the same wine.
        advice = insights.within(dear)
        if advice.counted:
            columns = advice.offers[
                ["offer", "suggested", "clicks_change", "conversions_change"]
            ].rename(columns={"suggested": "google_price"})
            frame = frame.merge(
                columns.drop_duplicates(subset="offer"), on="offer", how="left"
            )
            frame["google_cut"] = 1 - frame["google_price"] / frame["price"]
    # Without clicks there is no demand to weigh the gap by, and a ranking by a
    # column of zeros is an arbitrary hundred wines dressed as a shortlist. The
    # gap alone is a worse argument but an honest one; the panel says which it
    # is showing.
    order = ["impact", "clicks"] if demand.measured else ["gap"]
    return (
        frame.sort_values(order, ascending=False).head(limit).reset_index(drop=True)
    )


def after_cut(offers: pd.DataFrame, cut: float) -> pd.DataFrame:
    """The same offers repriced by ``cut``, with the gap that would leave.

    Merchants agree to a percentage off a range, not to a price per bottle, so
    the question a negotiation actually asks is what one percentage would do to
    the whole list.
    """
    if offers.empty or "price" not in offers.columns:
        return offers
    price = offers["price"] * (1 - cut)
    return offers.assign(
        cut_price=price, cut_gap=(price - offers["benchmark"]) / offers["benchmark"]
    )


def beats_market(offers: pd.DataFrame) -> int:
    """How many of the repriced offers would no longer be dearer than the rest."""
    if offers.empty or "cut_gap" not in offers.columns:
        return 0
    return int((offers["cut_gap"] <= _SAME_PRICE).sum())


def verdicts(
    prices: Prices,
    insights: Insights | None = None,
    demand: Demand | None = None,
) -> list[str]:
    """What the prices mean, in the sentences a leadership meeting would use."""
    if not prices.counted:
        return [
            "No offer in the feed has a benchmark yet. Google publishes one only "
            "where enough other merchants sell the same product."
        ]
    lines = [
        f"**{prices.dear_share:.0%} of {prices.counted:,} priced products cost "
        f"more here than the market**, typically {prices.median_gap:+.0%} against "
        "Google's benchmark for the same bottle."
    ]
    dear = prices.offers[prices.offers["gap"] > DEAR_GAP]
    if not dear.empty:
        lines.append(
            f"{len(dear):,} of them are more than {DEAR_GAP:.0%} above it, which "
            "is the band where a shopper comparing two tabs does not come back."
        )
    if prices.cheaper:
        lines.append(
            f"{prices.cheaper:,} "
            + ("is" if prices.cheaper == 1 else "are")
            + " cheaper than the market, and those are the ones worth "
            "advertising."
        )
    # What a hundred phone calls would actually cover. The point of the ask list
    # is that demand is not spread evenly over the catalogue, and the share below
    # is the argument for negotiating a short list rather than a price policy.
    if demand is not None:
        wanted = ask_list(prices, demand, None, ASK_LIST)
        dear = demand.against(prices.offers[prices.offers["gap"] > _SAME_PRICE])
        dear_clicks = int(dear["clicks"].sum()) if not dear.empty else 0
        asked = int(wanted["clicks"].sum()) if not wanted.empty else 0
        if dear_clicks and asked:
            lines.append(
                f"The {len(wanted)} bottles worth asking about took {asked:,} of "
                f"the {dear_clicks:,} clicks that went to a dearer-than-market "
                f"wine, so {asked / dear_clicks:.0%} of that demand is one "
                "conversation with a handful of merchants."
            )
    # Only the suggestions for offers this panel compared: the insights report
    # is a wider population and "of them" is a claim about this one.
    if insights is not None:
        insights = insights.within(prices.offers)
    if insights is not None and insights.counted:
        lines.append(
            f"Google would cut the price on {insights.counted:,} of them, and "
            f"predicts {insights.clicks_gain:+.0%} clicks and "
            f"{insights.conversions_gain:+.0%} conversions if you did."
        )
    if prices.truncated or (insights is not None and insights.truncated):
        lines.append(
            "The feed has more priced products than one read carries, so these "
            "figures describe the products read rather than all of them."
        )
    return lines


def as_of(now: _dt.date | None = None) -> _dt.date:
    """The day these figures describe: Merchant Center serves today's snapshot."""
    return now or _dt.date.today()


__all__ = [
    "ASK_LIST",
    "DEAR_GAP",
    "DEFAULT_COUNTRY",
    "DEMAND_DAYS",
    "Demand",
    "Insights",
    "Merchant",
    "MerchantConfigError",
    "Prices",
    "SALES_DAYS",
    "Sales",
    "access_token",
    "after_cut",
    "as_of",
    "ask_list",
    "bargains",
    "beats_market",
    "load_merchant_env",
    "main_currency",
    "price_bands",
    "price_gaps",
    "price_insights",
    "product_demand",
    "sales_verdicts",
    "verdicts",
]
