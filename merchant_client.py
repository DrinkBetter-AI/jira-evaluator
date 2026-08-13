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


def verdicts(prices: Prices, insights: Insights | None = None) -> list[str]:
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
    "DEAR_GAP",
    "DEFAULT_COUNTRY",
    "Insights",
    "Merchant",
    "MerchantConfigError",
    "Prices",
    "access_token",
    "as_of",
    "load_merchant_env",
    "main_currency",
    "price_gaps",
    "price_insights",
    "verdicts",
]
