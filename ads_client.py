"""Read-only client for Google Ads figures, read out of BigQuery.

Google will not issue this account an Ads API developer token: the API Center is
open to manager accounts only, and 887-686-4797 has no manager above it. Google
will, however, push the same data into BigQuery daily for nothing and without a
token, so that is where the dashboard reads it from. BigQuery is the delivery
route as much as the store.

The transfer writes about 190 tables per account per day. Four are enough here:

``ads_CampaignBasicStats_<customer>``
    Cost, clicks, impressions, conversions and conversion value per campaign per
    day, segmented by network, device and slot - so every read groups.
``ads_Campaign_<customer>``
    One snapshot row per campaign per day: name, status, budget.
``ads_Customer_<customer>``
    The account's name and the currency its money is quoted in.
``ads_CampaignConversionStats_<customer>``
    Conversions split by which conversion action fired them. Not read yet.

Two things about the data shape that change how it must be queried. Costs arrive
as micros - millionths of a currency unit - so every figure is divided by a
million on the way out. And the entity tables are daily snapshots rather than a
current state, so a campaign renamed yesterday has a row under each name; the
name is therefore taken from the newest snapshot rather than joined per day.

Every statement is a SELECT. The credential this runs under has BigQuery's
dataViewer and jobUser roles and nothing else, so it cannot write here even by
accident, and it has no access to Google Ads itself at all.
"""

from __future__ import annotations

import datetime as _dt
import functools
import json
import os
import re
from dataclasses import dataclass

import pandas as pd

_DATASET_ENV_VAR = "GOOGLE_ADS_BQ_DATASET"
_PROJECT_ENV_VAR = "GOOGLE_ADS_BQ_PROJECT"
_CUSTOMER_ENV_VAR = "GOOGLE_ADS_CUSTOMER_ID"
_KEY_ENV_VAR = "GCP_BIGQUERY_READONLY_KEY"

# What `bq mk -d google_ads` created. Named rather than searched for, because
# guessing at datasets is how a dashboard ends up reading somebody else's.
DEFAULT_DATASET = "google_ads"

# The stats table's grain is a day, and the account's own timezone decides which
# day a click belongs to, so "yesterday" here means yesterday in Los Angeles.
# Today is excluded throughout: it is a part-day and reads as a collapse.
_STATS_TABLE = "ads_CampaignBasicStats"
_CAMPAIGN_TABLE = "ads_Campaign"
_CUSTOMER_TABLE = "ads_Customer"
# Shopping's own grain: one row per product per day, keyed by the offer id the
# Merchant Center feed uses, which is what lets ad spend be set beside the price
# of the bottle it was spent on.
_PRODUCT_TABLE = "ads_ShoppingProductStats"

MICROS = 1_000_000

# Windows the panel offers. A week is what a campaign change shows up in; 30
# days is what a monthly budget is set against.
LOOKBACK_WINDOWS = (7, 30)

_CUSTOMER_ID_PATTERN = re.compile(r"^\d{6,12}$")
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,1024}$")
# Project ids are letters, digits and hyphens, optionally domain-prefixed for the
# oldest projects (`example.com:project`). Checked for the same reason as the
# dataset: both are interpolated into a backquoted table reference, which a
# backquote in either would close.
_PROJECT_PATTERN = re.compile(r"^[A-Za-z0-9\-.]{1,63}(:[A-Za-z0-9\-]{1,63})?$")


class AdsConfigError(RuntimeError):
    """Raised when the BigQuery side is missing, misnamed or unreadable."""


@dataclass(frozen=True)
class AdsConfig:
    """Where the Ads tables are, and who to read them as."""

    project: str
    dataset: str
    # None means "discover it from the table names", which is the normal case:
    # the transfer suffixes every table with the account it came from, so the
    # dashboard needs no configuration to find a newly added account.
    customer_id: str | None
    # A service-account JSON, when one was supplied. Cloud Run supplies none and
    # authenticates as itself.
    key_json: str | None


@dataclass(frozen=True)
class Spend:
    """One window of the account, and the window immediately before it."""

    days: int
    cost: float
    clicks: int
    impressions: int
    # Google Ads' own count, from its own attribution, and not the same thing as
    # an order in the CRM: it credits a conversion to the day of the *click*,
    # counts view-throughs, and can fractionally credit one sale to several ads.
    # Reported beside CRM orders rather than instead of them.
    conversions: float
    conversion_value: float
    prev_cost: float
    prev_conversions: float
    # Days in the window that actually have data. The transfer backfills one day
    # per run, so a fresh setup has one day of history and a 30-day figure that
    # would otherwise silently mean "one day".
    days_with_data: int
    first_day: _dt.date | None
    last_day: _dt.date | None
    # The window asked for, and the earliest day the transfer has loaded at all -
    # read from the table rather than inferred from these rows, since a paused
    # account has no rows for days that are perfectly well loaded. Between them
    # they tell a window whose history has not arrived from one that was quiet.
    window_start: _dt.date | None = None
    window_end: _dt.date | None = None
    history_start: _dt.date | None = None

    @property
    def cost_per_conversion(self) -> float:
        return round(self.cost / self.conversions, 2) if self.conversions else 0.0

    @property
    def roas(self) -> float:
        """Ads' own value per unit of spend. See ``conversions`` on the caveat."""
        return round(self.conversion_value / self.cost, 2) if self.cost else 0.0

    @property
    def cost_change(self) -> float:
        return round(self.cost - self.prev_cost, 2)

    @property
    def partial(self) -> bool:
        """Whether the window starts before the transfer's history does.

        Deliberately not "fewer days have rows than the window is long". The
        stats table has no row for a day on which nothing ran, so an account
        paused for a fortnight would otherwise be reported as a broken feed. What
        makes a figure an understatement is history that begins mid-window.
        """
        if self.history_start is None or self.window_start is None:
            return False
        return self.history_start > self.window_start

    @property
    def days_loaded(self) -> int:
        """Days of the window the transfer has actually loaded."""
        if not self.partial:
            return self.days
        assert self.history_start is not None  # narrowed by `partial`
        assert self.window_start is not None
        return max(self.days - (self.history_start - self.window_start).days, 0)


def load_ads_env() -> AdsConfig | None:
    """Return where to read Ads figures from, or ``None`` when unconfigured.

    A missing key is not an error: on Cloud Run there is no key, because the
    service authenticates as its own service account. A missing project is,
    since nothing can be read without knowing which project holds the dataset -
    unless a key was supplied, which names its project itself.
    """
    dataset = os.getenv(_DATASET_ENV_VAR, "").strip() or DEFAULT_DATASET
    project = os.getenv(_PROJECT_ENV_VAR, "").strip()
    key_json = os.getenv(_KEY_ENV_VAR, "").strip() or None
    customer = os.getenv(_CUSTOMER_ENV_VAR, "").strip().replace("-", "") or None

    if key_json:
        try:
            info = json.loads(key_json)
        except ValueError as exc:
            raise AdsConfigError(
                f"{_KEY_ENV_VAR} is not valid JSON: paste the whole service "
                "account key file, braces included."
            ) from exc
        project = project or str(info.get("project_id", "")).strip()

    if not project:
        project = default_project()
    if not project:
        return None
    if customer and not _CUSTOMER_ID_PATTERN.match(customer):
        raise AdsConfigError(
            f"{_CUSTOMER_ENV_VAR} should be a Google Ads customer ID such as "
            "887-686-4797 or 8876864797."
        )
    if not _NAME_PATTERN.match(dataset):
        raise AdsConfigError(f"{_DATASET_ENV_VAR} is not a valid dataset name.")
    if not _PROJECT_PATTERN.match(project):
        raise AdsConfigError(
            f"{project!r} is not a GCP project id; check "
            f"{_PROJECT_ENV_VAR} and the key's project_id."
        )
    return AdsConfig(
        project=project, dataset=dataset, customer_id=customer, key_json=key_json
    )


def valid_project(project: str) -> bool:
    """Whether this is a project id, and so safe in a table reference."""
    return bool(_PROJECT_PATTERN.match(project))


def valid_name(name: str) -> bool:
    """Whether this is a dataset or table name, and so safe in a reference."""
    return bool(_NAME_PATTERN.match(name))


def default_project() -> str:
    """The project Google's own libraries would default to, if any.

    On Cloud Run this is the project the service runs in, which is where the
    dataset lives, so the deployed dashboard needs no configuration at all.
    """
    for var in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT"):
        value = os.getenv(var, "").strip()
        if value:
            return value
    return _adc_project()


@functools.lru_cache(maxsize=1)
def _adc_project() -> str:
    """The project the ambient credentials belong to, asked once per process.

    On Cloud Run none of the project variables are set, so this is the branch that
    answers - and it reaches the metadata server to do it. It is called on every
    Streamlit rerun, which is every click, so the answer is remembered: the
    identity a process runs as does not change underneath it.
    """
    return _ask_adc_project()


def _ask_adc_project() -> str:
    try:
        import google.auth  # noqa: PLC0415 - optional dependency

        _, project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/bigquery"]
        )
    except Exception:  # noqa: BLE001 - no credentials is a normal outcome here
        return ""
    return str(project or "")


def build_client(config: AdsConfig):
    """A BigQuery client for ``config``, read-only by virtue of its credential."""
    try:
        from google.cloud import bigquery  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - exercised by deployment
        raise AdsConfigError(
            "google-cloud-bigquery is not installed, so the Google Ads figures "
            "cannot be read."
        ) from exc

    if not config.key_json:
        return bigquery.Client(project=config.project)

    from google.oauth2 import service_account  # noqa: PLC0415

    credentials = service_account.Credentials.from_service_account_info(
        json.loads(config.key_json),
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )
    return bigquery.Client(project=config.project, credentials=credentials)


def customer_ids(client, config: AdsConfig) -> list[str]:
    """Accounts present in the dataset, newest transfer first.

    Read from the table names rather than configured, so linking a second Ads
    account to the transfer needs no deployment. When one is named in the
    environment it wins, which is how a shared dataset gets narrowed.
    """
    if config.customer_id:
        return [config.customer_id]
    prefix = f"{_STATS_TABLE}_"
    found = []
    for table in client.list_tables(f"{config.project}.{config.dataset}"):
        name = table.table_id
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if _CUSTOMER_ID_PATTERN.match(suffix):
                found.append(suffix)
    return sorted(set(found))


def _table(config: AdsConfig, table: str, customer_id: str) -> str:
    if not _CUSTOMER_ID_PATTERN.match(customer_id):
        raise AdsConfigError(f"{customer_id!r} is not a Google Ads customer ID.")
    return f"`{config.project}.{config.dataset}.{table}_{customer_id}`"


def account(client, config: AdsConfig, customer_id: str) -> tuple[str, str]:
    """The account's own name and the currency it bills in, e.g. ``USD``.

    Both from the newest snapshot. The currency matters more than it looks: the
    CRM's revenue is in dollars, and dividing it by spend quoted in something
    else would produce a confident, wrong ROAS.
    """
    sql = f"""
        SELECT customer_descriptive_name AS name, customer_currency_code AS currency
        FROM {_table(config, _CUSTOMER_TABLE, customer_id)}
        ORDER BY _DATA_DATE DESC
        LIMIT 1
    """
    rows = list(client.query(sql).result())
    if not rows:
        return "", "USD"
    return str(rows[0]["name"] or ""), str(rows[0]["currency"] or "USD")


def daily_stats(
    client,
    config: AdsConfig,
    customer_id: str,
    days: int,
    now: _dt.date | None = None,
) -> pd.DataFrame:
    """Per campaign per day over the last ``2 * days`` days, to allow comparison.

    Twice the window in one query rather than two queries: the previous period is
    only ever wanted beside the current one, and BigQuery bills per byte scanned,
    so one pass over a wider slice is cheaper than two over halves.
    """
    end = now or _dt.date.today()
    # Yesterday is the last complete day. Today's rows exist from the moment the
    # transfer runs and would read as a collapse in spend.
    last = end - _dt.timedelta(days=1)
    first = last - _dt.timedelta(days=2 * days - 1)
    sql = f"""
        SELECT
          stats.campaign_id AS campaign_id,
          stats.segments_date AS day,
          SUM(stats.metrics_cost_micros) / {MICROS} AS cost,
          SUM(stats.metrics_clicks) AS clicks,
          SUM(stats.metrics_impressions) AS impressions,
          SUM(stats.metrics_conversions) AS conversions,
          SUM(stats.metrics_conversions_value) AS conversion_value
        FROM {_table(config, _STATS_TABLE, customer_id)} AS stats
        WHERE stats.segments_date BETWEEN @first AND @last
        GROUP BY campaign_id, day
    """
    frame = _run(client, sql, first=first, last=last)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "campaign_id",
                "day",
                "cost",
                "clicks",
                "impressions",
                "conversions",
                "conversion_value",
            ]
        )
    frame["day"] = pd.to_datetime(frame["day"]).dt.date
    return frame


PRODUCT_COLUMNS = ("offer", "spend", "clicks", "impressions", "ad_conversions")


def product_stats(
    client,
    config: AdsConfig,
    customer_id: str,
    days: int,
    now: _dt.date | None = None,
) -> pd.DataFrame:
    """What each advertised product cost and drew over the last ``days`` days.

    One row per offer id, which is the same id the Merchant Center feed and the
    catalogue's ``external_id`` carry, so the spend on a bottle can be read
    beside its price against the market and the bottles it actually sold.

    ``ad_conversions`` is Google's own attribution and is reported as such: the
    account records a few dozen against the CRM's hundreds of orders, so it is
    evidence about the ad, not about the shop.
    """
    end = now or _dt.date.today()
    last = end - _dt.timedelta(days=1)
    first = last - _dt.timedelta(days=days - 1)
    sql = f"""
        SELECT
          segments_product_item_id AS offer,
          SUM(metrics_cost_micros) / {MICROS} AS spend,
          SUM(metrics_clicks) AS clicks,
          SUM(metrics_impressions) AS impressions,
          SUM(metrics_conversions) AS ad_conversions
        FROM {_table(config, _PRODUCT_TABLE, customer_id)}
        WHERE segments_date BETWEEN @first AND @last
        GROUP BY offer
    """
    frame = _run(client, sql, first=first, last=last)
    if frame.empty:
        return pd.DataFrame(columns=list(PRODUCT_COLUMNS))
    frame["offer"] = frame["offer"].astype(str).str.strip()
    for column in ("spend", "clicks", "impressions", "ad_conversions"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return (
        frame[frame["offer"] != ""]
        .sort_values("spend", ascending=False)
        .reset_index(drop=True)
    )


def campaign_names(client, config: AdsConfig, customer_id: str) -> pd.DataFrame:
    """Each campaign's current name, status and daily budget.

    From the newest snapshot only. The table holds one row per campaign per day,
    so joining it to the stats by date would multiply a renamed campaign into two
    rows and split its spend between them.
    """
    sql = f"""
        WITH newest AS (
          SELECT MAX(_DATA_DATE) AS day
          FROM {_table(config, _CAMPAIGN_TABLE, customer_id)}
        )
        SELECT
          campaign_id,
          campaign_name AS campaign,
          campaign_status AS status,
          campaign_advertising_channel_type AS channel,
          campaign_budget_amount_micros / {MICROS} AS budget
        FROM {_table(config, _CAMPAIGN_TABLE, customer_id)}
        WHERE _DATA_DATE = (SELECT day FROM newest)
    """
    frame = _run(client, sql)
    if frame.empty:
        return pd.DataFrame(
            columns=["campaign_id", "campaign", "status", "channel", "budget"]
        )
    return frame


def _run(client, sql: str, **params) -> pd.DataFrame:
    from google.cloud import bigquery  # noqa: PLC0415 - optional dependency

    types = {_dt.date: "DATE", int: "INT64", str: "STRING"}
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(name, types[type(value)], value)
            for name, value in params.items()
        ]
    )
    return client.query(sql, job_config=job_config).result().to_dataframe()


def loaded_from(
    client, config: AdsConfig, customer_id: str, now: _dt.date | None = None
) -> _dt.date | None:
    """The earliest day the transfer has loaded, over the whole table.

    Asked separately, and unbounded by any window, because it is the only honest
    way to tell "these days have not arrived" from "nothing ran on these days":
    the stats table has no row for a day with no activity, so the earliest row
    *within* a window says nothing about whether the window is loaded. Reads one
    column of a partitioned table, so it is the cheapest query here.
    """
    sql = f"""
        SELECT MIN(segments_date) AS first_day
        FROM {_table(config, _STATS_TABLE, customer_id)}
    """
    rows = list(client.query(sql).result())
    if not rows or rows[0]["first_day"] is None:
        return None
    first = rows[0]["first_day"]
    day = first.date() if isinstance(first, _dt.datetime) else first
    # A transfer part-way through a backfill can hold a day beyond the window;
    # today is never a complete day, so it is not treated as history either.
    return min(day, (now or _dt.date.today()) - _dt.timedelta(days=1))


def window(
    stats: pd.DataFrame,
    days: int,
    now: _dt.date | None = None,
    history_start: _dt.date | None = None,
) -> Spend:
    """Fold ``daily_stats`` into one window and the window before it.

    ``history_start`` comes from :func:`loaded_from`. Without it the window is
    reported as complete: a missing figure is better than a warning that fires
    every time an account pauses.
    """
    end = now or _dt.date.today()
    last = end - _dt.timedelta(days=1)
    first = last - _dt.timedelta(days=days - 1)
    previous_first = first - _dt.timedelta(days=days)

    if stats.empty:
        return Spend(
            days=days,
            cost=0.0,
            clicks=0,
            impressions=0,
            conversions=0.0,
            conversion_value=0.0,
            prev_cost=0.0,
            prev_conversions=0.0,
            days_with_data=0,
            first_day=None,
            last_day=None,
            window_start=first,
            window_end=last,
            history_start=history_start,
        )

    current = stats[(stats["day"] >= first) & (stats["day"] <= last)]
    previous = stats[(stats["day"] >= previous_first) & (stats["day"] < first)]
    days_present = sorted(current["day"].unique()) if not current.empty else []
    return Spend(
        days=days,
        cost=round(float(current["cost"].sum()), 2),
        clicks=int(current["clicks"].sum()) if not current.empty else 0,
        impressions=int(current["impressions"].sum()) if not current.empty else 0,
        conversions=round(float(current["conversions"].sum()), 2),
        conversion_value=round(float(current["conversion_value"].sum()), 2),
        prev_cost=round(float(previous["cost"].sum()), 2),
        prev_conversions=round(float(previous["conversions"].sum()), 2),
        days_with_data=len(days_present),
        first_day=days_present[0] if days_present else None,
        last_day=days_present[-1] if days_present else None,
        window_start=first,
        window_end=last,
        history_start=history_start,
    )


def by_campaign(
    stats: pd.DataFrame,
    names: pd.DataFrame,
    days: int,
    now: _dt.date | None = None,
) -> pd.DataFrame:
    """Spend and what it bought, per campaign, dearest first.

    Campaigns with no spend in the window are dropped rather than listed at
    zero: the account holds twelve, two of which are running, and a table of ten
    silent rows buries the two that cost money.
    """
    columns = [
        "campaign",
        "status",
        "channel",
        "cost",
        "clicks",
        "conversions",
        "conversion_value",
        "cost_per_conversion",
        "roas",
        "budget",
    ]
    if stats.empty:
        return pd.DataFrame(columns=columns)

    end = now or _dt.date.today()
    last = end - _dt.timedelta(days=1)
    first = last - _dt.timedelta(days=days - 1)
    current = stats[(stats["day"] >= first) & (stats["day"] <= last)]
    if current.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        current.groupby("campaign_id", as_index=False)[
            ["cost", "clicks", "impressions", "conversions", "conversion_value"]
        ]
        .sum()
        .round(2)
    )
    merged = grouped.merge(names, on="campaign_id", how="left")
    # A campaign that spent money but whose snapshot is missing is still money
    # spent, so it is named by its id rather than dropped.
    merged["campaign"] = merged["campaign"].fillna(
        merged["campaign_id"].astype(str).radd("Campaign ")
    )
    for column, default in (("status", "UNKNOWN"), ("channel", "UNKNOWN")):
        merged[column] = merged[column].fillna(default)
    # A merge against an empty snapshot table leaves this column typed as object,
    # which pandas would then downcast noisily on the way to a number.
    merged["budget"] = pd.to_numeric(merged["budget"], errors="coerce").fillna(0.0)
    merged["cost_per_conversion"] = [
        round(cost / conversions, 2) if conversions else 0.0
        for cost, conversions in zip(merged["cost"], merged["conversions"])
    ]
    merged["roas"] = [
        round(value / cost, 2) if cost else 0.0
        for value, cost in zip(merged["conversion_value"], merged["cost"])
    ]
    merged = merged[merged["cost"] > 0]
    return merged.sort_values("cost", ascending=False)[columns].reset_index(drop=True)


@dataclass(frozen=True)
class Sales:
    """What the CRM says happened over the same days the spend covers.

    Kept separate from ``Spend`` because the two disagree by design: Google
    counts a conversion against the day of the *click* and credits fractions of
    one sale to several ads, while the CRM counts money captured on the day it
    arrived. The CRM is the one to quote to a board; Google's is the one that
    splits spend across campaigns.
    """

    orders: int
    revenue: float
    # The currency that revenue is in. Money in two currencies cannot be added,
    # so the caller takes the shop's main one - and if it is not the currency the
    # ad account bills in, dividing one by the other means nothing.
    currency: str = ""
    # The same window immediately before this one, so the return on a dollar can
    # be read as moving towards break-even rather than as a figure on its own.
    prev_orders: int = 0
    prev_revenue: float = 0.0


# Below this, wine's margin does not cover the ad that sold it. A rough floor
# rather than a law: it assumes roughly a third of the bottle price is gross
# margin, so three dollars back per dollar spent is about break-even.
BREAK_EVEN_ROAS = 3.0

# What the marketplace actually keeps of a sale. The revenue an ad wins is the
# merchant's; only the commission on it is income here, so gross ROAS flatters
# the account by roughly eight times and this is the figure to steer by.
DEFAULT_COMMISSION_RATE = 0.12
_COMMISSION_ENV_VAR = "MARKETPLACE_COMMISSION_RATE"

# A dollar of commission for a dollar of spend. Not a profit: it is where the ad
# pays for itself before anybody's time, packaging or card fees.
BREAK_EVEN_RETURN = 1.0


def commission_rate() -> float:
    """The share of revenue the marketplace keeps, as a fraction.

    Set ``MARKETPLACE_COMMISSION_RATE`` to ``12``, ``12%`` or ``0.12`` when the
    rate changes; a rate the dashboard cannot parse is an error rather than a
    silent fallback, because every return figure on the page is built on it.
    """
    raw = os.getenv(_COMMISSION_ENV_VAR, "").strip().rstrip("%").strip()
    if not raw:
        return DEFAULT_COMMISSION_RATE
    try:
        rate = float(raw)
    except ValueError as exc:
        raise AdsConfigError(
            f"{_COMMISSION_ENV_VAR}={raw!r} is not a number. Use 12, 12% or 0.12."
        ) from exc
    if rate > 1:
        rate /= 100
    if not 0 < rate <= 1:
        raise AdsConfigError(
            f"{_COMMISSION_ENV_VAR}={raw!r} is not a commission share between "
            "0 and 100%."
        )
    return rate


def commission_return(revenue: float, cost: float, rate: float) -> float:
    """Commission earned per unit of ad spend: ``revenue * rate / cost``.

    1,226 of revenue at 12% against 176 of spend is 0.83 - eighty-three cents
    back for every dollar out, which is a loss on the ad before anything else
    is counted.
    """
    return round(revenue * rate / cost, 2) if cost else 0.0


@dataclass(frozen=True)
class Commission:
    """The commission a window earned, and whether it was counted or assumed.

    Merchants are on different agreements - 10% for some, 12% for others - so a
    single rate estimates a figure Stripe already holds exactly: the application
    fee it took from each sale. When that is readable, the return is money the
    marketplace actually charged; when it is not, the configured rate stands in
    and the panel says which of the two it is showing.
    """

    now: float
    before: float
    measured: bool


def earned_return(commission: float, cost: float) -> float:
    """Commission per unit of ad spend, from commission already counted."""
    return round(commission / cost, 2) if cost else 0.0


# How far Google's conversion count may sit from the CRM's order count before it
# is worth saying out loud. Some gap is normal - different days, different
# attribution - but a large one means one of the two is not measuring the shop.
ATTRIBUTION_TOLERANCE = 0.25


def verdicts(
    spend: Spend,
    campaigns: pd.DataFrame,
    sales: Sales | None,
    currency: str = "USD",
    rate: float | None = None,
    commission: Commission | None = None,
) -> list[str]:
    """The tables again, in sentences somebody can act on.

    Deliberately not a score. Each line either names money that bought nothing,
    or a number that has moved enough to be worth a question in a meeting.
    """
    lines: list[str] = []
    if spend.cost <= 0:
        return ["Nothing was spent on Google Ads in this window."]

    def money(amount: float) -> str:
        return _money(amount, currency)

    keep = commission_rate() if rate is None else rate
    unit = _money(1, currency).replace("1.00", "1")
    counted = commission is not None and commission.measured
    if counted or (sales is not None and sales.revenue):
        # First, because it is the only line here that is income rather than
        # turnover: the revenue belongs to the merchants and this is our share.
        if counted:
            now = earned_return(commission.now, spend.cost)
            before = earned_return(commission.before, spend.prev_cost)
            basis = f"{money(commission.now)} of commission actually charged"
        else:
            now = commission_return(sales.revenue, spend.cost, keep)
            before = commission_return(sales.prev_revenue, spend.prev_cost, keep)
            basis = f"{money(sales.revenue)} of revenue at {keep:.0%}"
        lines.append(
            f"**{money(now)} of commission back for every {unit} of ad "
            f"spend** - {basis} "
            f"against {money(spend.cost)} spent. Break even is "
            f"{BREAK_EVEN_RETURN:.2f}, so the ads "
            + (
                "pay for themselves."
                if now >= BREAK_EVEN_RETURN
                else f"are short by {money(BREAK_EVEN_RETURN - now)} on every "
                f"{unit} spent, before anybody's time, packaging or card fees."
            )
        )
        if before:
            word = "up" if now > before else "down" if now < before else "flat"
            lines.append(
                f"**That return is {word}** on the previous {spend.days} days, "
                f"which returned {money(before)} per {unit}."
            )

    if sales is not None and sales.orders:
        per_order = spend.cost / sales.orders
        # The basket is only quoted when there is revenue to quote: a caller that
        # cannot compare the two currencies passes the orders without it.
        basket = (
            f", against an average basket of {money(sales.revenue / sales.orders)}"
            if sales.revenue
            else ""
        )
        lines.append(
            f"**{money(spend.cost)} bought {sales.orders:,} orders** - "
            f"{money(per_order)} of ad spend per order{basket}."
        )
        if sales.revenue:
            true_roas = sales.revenue / spend.cost
            verdict = (
                "comfortably above"
                if true_roas >= BREAK_EVEN_ROAS
                else "below the rough break-even of"
            )
            lines.append(
                f"**{true_roas:.1f}x on money actually captured** - "
                f"{money(sales.revenue)} of revenue per {money(spend.cost)} "
                f"spent, {verdict} {BREAK_EVEN_ROAS:.0f}x. Every order in the "
                "window counts here, including the ones ads had nothing to do "
                "with, so read it as a ceiling."
            )
    elif sales is not None:
        lines.append(
            f"**{money(spend.cost)} spent and the CRM recorded no orders at "
            "all in these days**, which is either a very bad window or a break "
            "in the order feed."
        )

    if sales is not None and sales.orders and spend.conversions:
        gap = abs(spend.conversions - sales.orders) / sales.orders
        if gap > ATTRIBUTION_TOLERANCE:
            direction = "more" if spend.conversions > sales.orders else "fewer"
            lines.append(
                f"**Google claims {spend.conversions:,.0f} conversions where the "
                f"CRM has {sales.orders:,} orders** - {direction} than the shop "
                "saw. Google counts the day of the click and splits one sale "
                "across ads; a gap this size usually means its conversion "
                "tracking needs a look."
            )

    silent = wasted(campaigns)
    if not silent.empty:
        cost = float(silent["cost"].sum())
        names = ", ".join(silent["campaign"].head(3))
        more = f" and {len(silent) - 3} more" if len(silent) > 3 else ""
        plural = "campaign" if len(silent) == 1 else "campaigns"
        lines.append(
            f"**{money(cost)} went to {len(silent)} {plural} that recorded no "
            f"conversion at all**: {names}{more}."
        )

    if spend.prev_cost:
        change = spend.cost - spend.prev_cost
        share = abs(change) / spend.prev_cost
        if share >= 0.1:
            word = "up" if change > 0 else "down"
            lines.append(
                f"**Spend is {word} {money(abs(change))} ({share:.0%}) on the "
                f"previous {spend.days} days.**"
            )

    if not campaigns.empty:
        top = campaigns.iloc[0]
        share = float(top["cost"]) / spend.cost
        if share >= 0.5:
            lines.append(
                f"**{top['campaign']} is {share:.0%} of the spend.** If one "
                "campaign is the budget, its settings are the strategy."
            )
    return lines


def _money(amount: float, currency: str = "USD") -> str:
    """Whole units above ten, because cents in a spend figure are noise.

    In the ad account's own currency, which is not always dollars: the tiles read
    it off the account and the sentences beneath them must agree.
    """
    symbol = {"usd": "$", "eur": "\u20ac", "gbp": "\u00a3"}.get(currency.lower(), "")
    figure = f"{amount:,.0f}" if abs(amount) >= 10 else f"{amount:,.2f}"
    return f"{symbol}{figure}" if symbol else f"{figure} {currency.upper()}"


def wasted(campaigns: pd.DataFrame) -> pd.DataFrame:
    """Campaigns that spent money and recorded nothing for it.

    The one table in the panel that is a to-do list rather than a report.
    """
    if campaigns.empty:
        return campaigns
    return campaigns[campaigns["conversions"] <= 0]


def paused_spenders(campaigns: pd.DataFrame) -> pd.DataFrame:
    """Campaigns billed for in the window that are now switched off.

    Not a fault - a campaign paused mid-window spent real money before it was -
    but it explains a spend figure that no live campaign accounts for.
    """
    if campaigns.empty:
        return campaigns
    return campaigns[~campaigns["status"].isin(("ENABLED",))]


__all__ = [
    "ATTRIBUTION_TOLERANCE",
    "AdsConfig",
    "AdsConfigError",
    "BREAK_EVEN_RETURN",
    "BREAK_EVEN_ROAS",
    "Commission",
    "DEFAULT_COMMISSION_RATE",
    "DEFAULT_DATASET",
    "LOOKBACK_WINDOWS",
    "MICROS",
    "Sales",
    "Spend",
    "account",
    "build_client",
    "by_campaign",
    "campaign_names",
    "commission_rate",
    "commission_return",
    "customer_ids",
    "default_project",
    "daily_stats",
    "earned_return",
    "load_ads_env",
    "paused_spenders",
    "verdicts",
    "wasted",
    "window",
]
