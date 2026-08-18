"""Read-only clients for what the business spends, provider by provider.

The shop's revenue is only half a sentence. This module reads the other half from
the providers themselves, so the dashboard can put spend beside takings instead of
asking somebody to remember the invoices.

Today that means OpenAI, whose organization cost endpoint is the only credential
that reports it (project API keys are refused), and Stripe, which turns out to
report the other direction: this account is a Connect platform, so its ledger is
the marketplace's own earnings rather than a cost. Google Cloud follows its
billing export. Everything here is a GET.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import pathlib
import tempfile
from dataclasses import dataclass, field

import pandas as pd
import requests

OPENAI_BASE_URL = "https://api.openai.com"
_OPENAI_KEY_ENV_VAR = "OPENAI_ADMIN_KEY"
_COSTS_PATH = "/v1/organization/costs"

STRIPE_BASE_URL = "https://api.stripe.com"
_STRIPE_KEY_ENV_VAR = "STRIPE_READONLY_API_KEY"
_BALANCE_PATH = "/v1/balance_transactions"

_TIMEOUT_SECONDS = 60

# The endpoint's own ceiling; asking for more is a 400. One bucket is one day, so
# this caps a single call at roughly six months.
_MAX_BUCKETS = 180

# Enough pages to cover the widest window at the smallest sane page, and a stop
# in case the cursor ever fails to advance.
_MAX_PAGES = 20

LOOKBACK_WINDOWS = (7, 30)

# What the panel counts as an admin key. Deliberately loose - OpenAI has issued
# several prefixes - but tight enough to catch a project key pasted by mistake,
# which returns a bare 401 and looks like an outage.
_PROJECT_KEY_HINT = "sk-proj-"


class CostConfigError(RuntimeError):
    """Raised when a cost credential is missing, malformed or refused."""


def load_openai_env() -> str | None:
    """Return the OpenAI admin key, or ``None`` when there is none set.

    Names the project-key mistake rather than letting it become a 401 later: the
    cost endpoint is organization-scoped and refuses the ordinary key that every
    other OpenAI integration uses, so pasting that one is the likely error.
    """
    key = os.getenv(_OPENAI_KEY_ENV_VAR, "").strip()
    if not key:
        return None
    if key.startswith(_PROJECT_KEY_HINT):
        raise CostConfigError(
            f"{_OPENAI_KEY_ENV_VAR} looks like a project key. The cost endpoint "
            "is organization-scoped: create an admin key under Organization "
            "settings, API keys, Admin keys."
        )
    return key


@dataclass(frozen=True)
class Burn:
    """What one provider cost over a window, and over the one before it."""

    days: int
    provider: str
    cost: float
    prev_cost: float
    currency: str
    days_with_data: int
    first_day: _dt.date | None
    last_day: _dt.date | None
    # Named lines, dearest first: "gpt-5.6-terra, cached input" and the like.
    lines: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Currencies set aside rather than added to the figure above, which is what
    # the caller names so the reader knows the total is not the whole bill.
    other_currencies: tuple[str, ...] = ()
    # Days of the window the source can actually answer for, where that is
    # fewer than the window: a billing export switched on last week holds a
    # week, and averaging that week over a month is a discount nobody gave.
    days_loaded: int | None = None
    # Whether the period before this one is whole. A source that starts inside
    # it holds a day or two of a month, and a month measured against two days
    # of it is a rise of a thousand per cent that never happened.
    comparable: bool = True

    @property
    def cost_change(self) -> float:
        return self.cost - self.prev_cost

    @property
    def per_day(self) -> float:
        """Averaged over the days the source covers, not the days with charges.

        A quiet weekend is part of the monthly bill, so dividing by days with
        charges would overstate what the next month is likely to cost. Days the
        source has no answer for are the other error, and understate it: they
        are not free days, they are days nobody can see.
        """
        span = self.days_loaded or self.days
        return self.cost / span if span else 0.0

    @property
    def monthly(self) -> float:
        """The window's rate carried out to a month, which is how bills arrive."""
        return self.per_day * 30


def openai_costs(key: str, days: int, now: _dt.date | None = None) -> pd.DataFrame:
    """Daily OpenAI cost by project and line item, over ``2 * days``.

    Twice the window because every figure the panel prints is a comparison with
    the period before it. Grouped by line item because "$765 on OpenAI" is not
    actionable and "$227 of it re-sending cached context" is.
    """
    today = now or _dt.date.today()
    span = min(2 * days, _MAX_BUCKETS)
    start = today - _dt.timedelta(days=span - 1)
    rows: list[dict] = []
    page: str | None = None
    for _ in range(_MAX_PAGES):
        payload = _get_costs(key, start, span, page)
        for bucket in payload.get("data") or []:
            day = _bucket_day(bucket)
            for result in bucket.get("results") or []:
                amount = (result or {}).get("amount") or {}
                rows.append(
                    {
                        "day": day,
                        # The name where the endpoint returns one, the id where it
                        # does not: the grouping is by id, and an unnamed project
                        # is still worth telling apart from the rest of the bill.
                        "project": result.get("project_name")
                        or result.get("project_id")
                        or "",
                        "line_item": result.get("line_item") or "",
                        "cost": float(amount.get("value") or 0.0),
                        "currency": str(amount.get("currency") or "usd").lower(),
                    }
                )
        page = payload.get("next_page")
        if not payload.get("has_more") or not page:
            break
    frame = pd.DataFrame(
        rows, columns=["day", "project", "line_item", "cost", "currency"]
    )
    # Buckets with no charges arrive empty rather than absent, and a day the
    # provider has not closed yet arrives as zero; neither is worth a row.
    return frame[frame["cost"] != 0.0].reset_index(drop=True)


def _get_costs(key: str, start: _dt.date, span: int, page: str | None) -> dict:
    params: list[tuple[str, str]] = [
        ("start_time", str(_midnight(start))),
        ("bucket_width", "1d"),
        ("limit", str(min(span, _MAX_BUCKETS))),
        ("group_by", "project_id"),
        ("group_by", "line_item"),
    ]
    if page:
        params.append(("page", page))
    response = requests.get(
        f"{OPENAI_BASE_URL}{_COSTS_PATH}",
        params=params,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        timeout=_TIMEOUT_SECONDS,
    )
    if response.status_code in (401, 403):
        raise CostConfigError(
            "OpenAI refused the key for organization costs. It has to be an "
            "admin key from Organization settings, API keys, Admin keys - a "
            "project key cannot read the organization's spend."
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise CostConfigError("OpenAI returned an unexpected response shape.")
    return payload


def _midnight(day: _dt.date) -> int:
    """The day's start as a UTC timestamp, which is what the endpoint takes."""
    moment = _dt.datetime.combine(day, _dt.time.min, tzinfo=_dt.timezone.utc)
    return int(moment.timestamp())


def _bucket_day(bucket: dict) -> _dt.date:
    stamp = bucket.get("start_time")
    return _dt.datetime.fromtimestamp(int(stamp or 0), _dt.timezone.utc).date()


def window(
    costs: pd.DataFrame,
    days: int,
    provider: str = "OpenAI",
    now: _dt.date | None = None,
    loaded: int | None = None,
    comparable: bool = True,
) -> Burn:
    """Split a daily cost frame into this window and the one before it.

    Today is included, unlike the ad panel: a provider bills as it goes and the
    day's charges so far are real money, where an ad transfer simply has not
    loaded the day yet.
    """
    today = now or _dt.date.today()
    if costs.empty:
        return Burn(days, provider, 0.0, 0.0, "usd", 0, None, None, pd.DataFrame())
    frame, money, others = main_currency(costs.copy())
    frame = frame.copy()
    frame["day"] = pd.to_datetime(frame["day"]).dt.date
    start = today - _dt.timedelta(days=days - 1)
    current = frame[frame["day"] >= start]
    previous = frame[(frame["day"] < start) & (frame["day"] >= start - _dt.timedelta(days=days))]
    return Burn(
        days=days,
        provider=provider,
        # To the cent, and adding zero, because neither is a no-op here: a
        # credit of a hundredth of a cent against nothing else leaves a
        # fraction below zero, and both it and -0.0 format as "$-0.00" - a
        # refund of nothing, on a bill that had nothing on it.
        cost=round(float(current["cost"].sum()), 2) + 0.0,
        prev_cost=round(float(previous["cost"].sum()), 2) + 0.0,
        currency=money,
        days_with_data=int(current["day"].nunique()),
        first_day=min(current["day"]) if not current.empty else None,
        last_day=max(current["day"]) if not current.empty else None,
        lines=by_line(current),
        other_currencies=others,
        days_loaded=min(loaded, days) if loaded else None,
        comparable=comparable,
    )


def main_currency(frame: pd.DataFrame) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    """The rows billed in the commonest currency, that currency, and the rest.

    Euros are never added to dollars, here as in the order book and the ad
    account: the totals stay in one currency and the caller names what was left
    out. Both providers bill one account in one currency today, so this is a
    guard rather than a feature.
    """
    if frame.empty or "currency" not in frame.columns:
        return frame, "usd", ()
    counted = frame["currency"].value_counts()
    main = str(counted.index[0])
    others = tuple(sorted(str(code) for code in counted.index[1:]))
    if not others:
        return frame, main, ()
    return frame[frame["currency"].eq(main)], main, others


def by_line(costs: pd.DataFrame) -> pd.DataFrame:
    """Cost per named line item, dearest first, with its share of the total."""
    if costs.empty:
        return pd.DataFrame(columns=["line_item", "cost", "share"])
    grouped = (
        costs.groupby("line_item", as_index=False)["cost"]
        .sum()
        .sort_values("cost", ascending=False)
        .reset_index(drop=True)
    )
    total = float(grouped["cost"].sum())
    grouped["share"] = grouped["cost"] / total if total else 0.0
    return grouped


def by_project(costs: pd.DataFrame) -> pd.DataFrame:
    """Cost per project, dearest first: which application is spending."""
    if costs.empty:
        return pd.DataFrame(columns=["project", "cost"])
    return (
        costs.groupby("project", as_index=False)["cost"]
        .sum()
        .sort_values("cost", ascending=False)
        .reset_index(drop=True)
    )


# A line item whose name says the tokens were served from cache or written to it.
# Worth separating because it is the one part of an AI bill that is usually a
# choice rather than a workload: it is context sent again.
_CACHE_WORDS = ("cached", "cache write", "cache read")

# The providers whose line items are tokens, and so where a cache line means
# context re-sent rather than a service that happens to be a cache.
TOKEN_PROVIDERS = ("OpenAI", "Anthropic")


def cached_share(costs: pd.DataFrame) -> float:
    """The fraction of the bill that is cache traffic rather than new work."""
    if costs.empty or "line_item" not in costs:
        return 0.0
    total = float(costs["cost"].sum())
    if not total:
        return 0.0
    names = costs["line_item"].fillna("").str.lower()
    cache = costs[names.str.contains("|".join(_CACHE_WORDS))]
    return float(cache["cost"].sum()) / total


# Below this, the period before the window is too small to be a comparison: the
# usage started inside the window, and a percentage would read as a spike.
_NEW_SPEND_SHARE = 0.05


def verdicts(burn: Burn) -> list[str]:
    """The bill in sentences: what it runs to a month, and what it is made of."""
    if burn.cost <= 0:
        return [f"No {burn.provider} charges in the last {burn.days} days."]
    money = burn.currency
    lines = [
        f"**{_money(burn.cost, money)} on {burn.provider} in {burn.days} days** - "
        f"{_money(burn.monthly, money)} a month at this rate."
    ]
    # Silent where the earlier period is only partly held: the caller says why,
    # and a trend drawn against two days of a month is worse than no trend.
    if burn.comparable:
        if burn.prev_cost < burn.cost * _NEW_SPEND_SHARE:
            lines.append(
                f"**This spend is new** - the {burn.days} days before came to "
                f"{_money(burn.prev_cost, money)}, so there is no earlier period "
                "to read it against yet."
            )
        else:
            share = burn.cost_change / burn.prev_cost
            if abs(share) >= 0.1:
                word = "up" if share > 0 else "down"
                lines.append(
                    f"**Spend is {word} {_money(abs(burn.cost_change), money)} "
                    f"({share:+.0%}) on the {burn.days} days before**, so this is "
                    "a change in usage rather than the usual bill."
                )
    # Only of a token bill: a Cloud line called "Cloud Memorystore for
    # Memcached" is a cache, but it is not context sent again.
    cached = cached_share(burn.lines) if burn.provider in TOKEN_PROVIDERS else 0.0
    if cached >= 0.25:
        lines.append(
            f"**{cached:.0%} of it is cached context** - "
            f"{_money(burn.cost * cached, money)} spent re-sending prompts rather "
            "than on new work, which is usually the cheapest thing to cut."
        )
    if not burn.lines.empty:
        top = burn.lines.iloc[0]
        if float(top["share"]) >= 0.3:
            lines.append(
                f"**{top['line_item']} is {float(top['share']):.0%} of the "
                f"bill** at {_money(float(top['cost']), money)}."
            )
    return lines


def _money(amount: float, currency: str = "usd") -> str:
    """Whole units above ten; cents in a monthly bill are noise.

    In the provider's own billing currency, because the tiles read it off the
    account and the sentences beneath them have to agree.
    """
    symbol = {"usd": "$", "eur": "\u20ac", "gbp": "\u00a3"}.get(currency.lower(), "")
    figure = f"{amount:,.0f}" if abs(amount) >= 10 else f"{amount:,.2f}"
    return f"{symbol}{figure}" if symbol else f"{figure} {currency.upper()}"


# A live secret key can write; the panel only ever reads, and a key that can do
# more than the panel needs should not be sitting in the environment.
_STRIPE_LIVE_SECRET_PREFIX = "sk_"


def load_stripe_env() -> str | None:
    """Return the Stripe restricted key, or ``None`` when there is none set.

    Refuses a full secret key outright rather than quietly using it: this panel
    needs four read permissions, and a key that can also issue refunds has no
    business being here.
    """
    key = os.getenv(_STRIPE_KEY_ENV_VAR, "").strip()
    if not key:
        return None
    if key.startswith(_STRIPE_LIVE_SECRET_PREFIX):
        raise CostConfigError(
            f"{_STRIPE_KEY_ENV_VAR} is a full secret key, which can move money. "
            "Create a restricted key with read access to balance transactions, "
            "charges, disputes and payouts instead."
        )
    return key


# Stripe returns at most this many objects per call, and the ledger of a busy
# platform runs to hundreds a month, so the reads page.
_STRIPE_PAGE = 100

# Ten thousand entries, which is a couple of years of this platform's ledger at
# its current rate. Stripe returns newest first, so a cap that bites drops the
# *oldest* rows - the period being compared against - and the read says so
# rather than letting the gap be reported as a rise.
_STRIPE_MAX_PAGES = 100

# The ledger entries this panel understands. A Connect platform's own income
# arrives as an application fee taken from the merchant's charge, and Stripe's
# processing fees are charged on that merchant's account rather than here.
_PLATFORM_EARNING = "application_fee"
_PLATFORM_REFUND = "application_fee_refund"
_PAYOUT = "payout"

# How an ordinary (non-platform) account takes money instead: the sale itself,
# with Stripe's processing fee charged on the same entry.
_CHARGE = "charge"
_REFUND = "refund"


@dataclass(frozen=True)
class Ledger:
    """What Stripe's own books say happened over a window.

    Named for what the account really holds rather than what the panel wanted:
    on a Connect platform the takings are commission, and ``fees`` is the money
    Stripe charged *this* account, which is nil while every charge belongs to a
    connected merchant.
    """

    days: int
    earnings: float
    refunds: float
    fees: float
    paid_out: float
    disputes: int
    currency: str
    prev_earnings: float
    prev_refunds: float
    prev_fees: float
    first_day: _dt.date | None
    last_day: _dt.date | None
    other_currencies: tuple[str, ...] = ()
    # Whether the takings above are commission from connected merchants or the
    # account's own sales. The two are not the same money, and only one of them
    # has Stripe's processing fee charged against it here.
    platform: bool = True
    # Whether the period before this one is whole. Stripe returns newest first,
    # so a read that hit its ceiling is missing precisely those older days.
    comparable: bool = True

    @property
    def net(self) -> float:
        """Commission kept: earnings less refunds less anything Stripe charged."""
        return self.earnings - abs(self.refunds) - abs(self.fees)

    @property
    def prev_net(self) -> float:
        return self.prev_earnings - abs(self.prev_refunds) - abs(self.prev_fees)

    @property
    def earnings_change(self) -> float:
        return self.earnings - self.prev_earnings

    @property
    def net_change(self) -> float:
        """Like for like with :attr:`net`, which is what the tile prints.

        Comparing gross takings under a figure that is net of refunds shows a
        rise beneath a number that fell, in the month it matters most.
        """
        return self.net - self.prev_net


def stripe_ledger(
    key: str, days: int, now: _dt.date | None = None
) -> tuple[pd.DataFrame, bool]:
    """Balance-transaction lines over ``2 * days``, and whether any were missed.

    The balance transaction is the only object that covers every kind of money
    movement at once - commission, refunds, Stripe's own fees, payouts - so the
    panel reads that rather than stitching charges and payouts together.
    """
    today = now or _dt.date.today()
    start = today - _dt.timedelta(days=2 * days - 1)
    rows: list[dict] = []
    after: str | None = None
    truncated = False
    for page in range(_STRIPE_MAX_PAGES):
        payload = _get_balance(key, start, after)
        data = payload.get("data") or []
        for entry in data:
            rows.append(
                {
                    "day": _stamp_day(entry.get("created")),
                    "type": str(entry.get("type") or ""),
                    "category": str(entry.get("reporting_category") or ""),
                    # Stripe counts in the currency's smallest unit.
                    "amount": float(entry.get("amount") or 0) / 100.0,
                    "fee": float(entry.get("fee") or 0) / 100.0,
                    "currency": str(entry.get("currency") or "usd").lower(),
                }
            )
        if not payload.get("has_more") or not data:
            break
        after = data[-1].get("id")
        if not after:  # pragma: no cover - defensive: no cursor, no next page
            break
        truncated = page == _STRIPE_MAX_PAGES - 1
    return (
        pd.DataFrame(
            rows, columns=["day", "type", "category", "amount", "fee", "currency"]
        ),
        truncated,
    )


def _get_balance(key: str, start: _dt.date, after: str | None) -> dict:
    params: list[tuple[str, str]] = [
        ("limit", str(_STRIPE_PAGE)),
        ("created[gte]", str(_midnight(start))),
    ]
    if after:
        params.append(("starting_after", after))
    response = requests.get(
        f"{STRIPE_BASE_URL}{_BALANCE_PATH}",
        params=params,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        timeout=_TIMEOUT_SECONDS,
    )
    if response.status_code in (401, 403):
        raise CostConfigError(
            "Stripe refused the restricted key for balance transactions. Give it "
            "read access to balance transactions, charges, disputes and payouts."
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise CostConfigError("Stripe returned an unexpected response shape.")
    return payload


def _stamp_day(stamp) -> _dt.date:
    return _dt.datetime.fromtimestamp(int(stamp or 0), _dt.timezone.utc).date()


def stripe_disputes(key: str, days: int, now: _dt.date | None = None) -> int:
    """How many disputes were opened in the window.

    Counted rather than summed: a dispute is money at risk with an outcome still
    to come, so adding it to a cost total would report a loss that may not happen.
    Pages like the ledger: a hundred disputes is a bad month rather than an
    impossible one, and a single page would report it as exactly a hundred.
    """
    today = now or _dt.date.today()
    start = today - _dt.timedelta(days=days - 1)
    counted = 0
    after: str | None = None
    for _ in range(_STRIPE_MAX_PAGES):
        params: list[tuple[str, str]] = [
            ("limit", str(_STRIPE_PAGE)),
            ("created[gte]", str(_midnight(start))),
        ]
        if after:
            params.append(("starting_after", after))
        response = requests.get(
            f"{STRIPE_BASE_URL}/v1/disputes",
            params=params,
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            timeout=_TIMEOUT_SECONDS,
        )
        if response.status_code in (401, 403):
            raise CostConfigError(
                "Stripe refused the restricted key for disputes; give it read "
                "access to disputes or the panel cannot report them."
            )
        response.raise_for_status()
        payload = response.json() or {}
        data = payload.get("data") or []
        counted += len(data)
        if not payload.get("has_more") or not data:
            break
        after = data[-1].get("id")
        if not after:  # pragma: no cover - defensive: no cursor, no next page
            break
    return counted


def reaches_past(entries: pd.DataFrame, days: int, now: _dt.date | None = None) -> bool:
    """Whether the oldest row read is older than this window's first day.

    A capped read is missing its oldest days, and how much that matters depends
    on where the cap fell: past the window's start and the window itself is
    whole, short of it and every figure in the window is missing sales too.
    """
    if entries.empty or "day" not in entries:
        return False
    today = now or _dt.date.today()
    return min(entries["day"]) < today - _dt.timedelta(days=days - 1)


def ledger_window(
    entries: pd.DataFrame,
    days: int,
    disputes: int = 0,
    now: _dt.date | None = None,
    comparable: bool = True,
) -> Ledger:
    """Fold Stripe's ledger into this window and the takings of the one before."""
    today = now or _dt.date.today()
    if entries.empty:
        return Ledger(
            days, 0.0, 0.0, 0.0, 0.0, disputes, "usd", 0.0, 0.0, 0.0, None, None
        )
    frame, money, others = main_currency(entries.copy())
    frame = frame.copy()
    frame["day"] = pd.to_datetime(frame["day"]).dt.date
    start = today - _dt.timedelta(days=days - 1)
    current = frame[frame["day"] >= start]
    previous = frame[
        (frame["day"] < start) & (frame["day"] >= start - _dt.timedelta(days=days))
    ]

    # An account with no application fees at all is not a platform: its income is
    # the charge itself, and Stripe's processing fee is charged against it here.
    # Counting only application fees on such an account leaves the fees taken off
    # nothing, which reports the takings as a loss.
    platform = bool((frame["type"] == _PLATFORM_EARNING).any())
    earning = _PLATFORM_EARNING if platform else _CHARGE
    refund = _PLATFORM_REFUND if platform else _REFUND

    def takings(rows: pd.DataFrame) -> float:
        return float(rows.loc[rows["type"] == earning, "amount"].sum())

    def refunded(rows: pd.DataFrame) -> float:
        return float(rows.loc[rows["type"] == refund, "amount"].sum())

    return Ledger(
        days=days,
        earnings=takings(current),
        refunds=refunded(current),
        # What Stripe charged this account, which is nil while every charge sits
        # on a connected merchant's books.
        fees=float(current["fee"].sum()),
        paid_out=float(current.loc[current["type"] == _PAYOUT, "amount"].sum()),
        disputes=disputes,
        currency=money,
        prev_earnings=takings(previous),
        prev_refunds=refunded(previous),
        prev_fees=float(previous["fee"].sum()),
        first_day=min(current["day"]) if not current.empty else None,
        last_day=max(current["day"]) if not current.empty else None,
        other_currencies=others,
        platform=platform,
        comparable=comparable,
    )


def stripe_verdicts(ledger: Ledger) -> list[str]:
    """The account's own take, in sentences, and what Stripe is silent about."""
    # A platform keeps commission from other people's sales; an ordinary account
    # keeps its own takings less what Stripe charged to process them.
    kept = "commission" if ledger.platform else "payments"
    if not ledger.earnings:
        # Still say what is at risk: a quiet window with chargebacks in it is
        # exactly the window where a hidden dispute count matters.
        return [f"Stripe recorded no {kept} in the last {ledger.days} days."] + (
            [_dispute_line(ledger)] if ledger.disputes else []
        )
    money = ledger.currency
    lines = [
        f"**{_money(ledger.net, money)} of {kept} kept in {ledger.days} days** "
        f"({_money(ledger.earnings, money)} earned"
        + (
            f", {_money(abs(ledger.refunds), money)} refunded"
            if ledger.refunds
            else ""
        )
        + ")."
    ]
    if ledger.prev_net and ledger.comparable:
        share = ledger.net_change / ledger.prev_net
        if abs(share) >= 0.1:
            word = "up" if share > 0 else "down"
            lines.append(
                f"**{kept.capitalize()} {'is' if ledger.platform else 'are'} "
                f"{word} {_money(abs(ledger.net_change), money)} "
                f"({share:+.0%}) on the {ledger.days} days before.**"
            )
    if ledger.platform and not ledger.fees:
        lines.append(
            "**Stripe charged this account nothing to process it** - the "
            "platform takes a fee from each merchant's charge, so the card fees "
            "sit on the merchants' own accounts and are not a cost here."
        )
    if ledger.disputes:
        lines.append(_dispute_line(ledger))
    return lines


def _dispute_line(ledger: Ledger) -> str:
    """Chargebacks opened in the window, said the same way wherever they appear."""
    return (
        f"**{ledger.disputes} dispute"
        f"{'s' if ledger.disputes != 1 else ''} opened** in the window, "
        "which is money at risk rather than money lost."
    )


# Google's own name for the standard usage cost export, suffixed with the
# billing account's id. Found by prefix rather than configured: the suffix is
# not something anybody should have to look up, and a project has one of these.
BILLING_TABLE_PREFIX = "gcp_billing_export_v1_"
DEFAULT_BILLING_DATASET = "billing_export"
_BILLING_DATASET_ENV_VAR = "GCP_BILLING_BQ_DATASET"
_BILLING_PROJECT_ENV_VAR = "GCP_BILLING_BQ_PROJECT"
# The same credential the Ads dataset is read with: one service account reads
# both, and on Cloud Run neither is set because the service is the credential.
_BQ_KEY_ENV_VAR = "GCP_BIGQUERY_READONLY_KEY"


@dataclass(frozen=True)
class Billing:
    """Where the Cloud billing export lives, and what reads it."""

    project: str
    dataset: str
    key_json: str | None = None


def load_billing_env() -> Billing | None:
    """Where to read Google Cloud costs from, or ``None`` when unconfigured.

    Nothing here is read from the Ads settings, though both read BigQuery under
    the same credential: an invalid Ads customer id is not a reason to stop
    reporting the Cloud bill, and it once was. The project comes from this
    panel's own variable, else the key's own project, else the project Google's
    libraries would default to - which on Cloud Run is the one the export is in.
    """
    import ads_client  # noqa: PLC0415 - avoids a cycle at import time

    project = os.getenv(_BILLING_PROJECT_ENV_VAR, "").strip()
    dataset = (
        os.getenv(_BILLING_DATASET_ENV_VAR, "").strip() or DEFAULT_BILLING_DATASET
    )
    key_json = os.getenv(_BQ_KEY_ENV_VAR, "").strip() or None
    if not project and key_json:
        try:
            info = json.loads(key_json)
        except ValueError as exc:
            raise CostConfigError(
                f"{_BQ_KEY_ENV_VAR} is not valid JSON: paste the whole service "
                "account key file, braces included."
            ) from exc
        project = str(info.get("project_id", "")).strip()
    project = project or ads_client.default_project()
    if not project:
        return None
    # Both are interpolated into a backquoted table reference, where a backquote
    # would end the identifier and leave the rest of the value as SQL.
    if not ads_client.valid_name(dataset):
        raise CostConfigError(
            f"{_BILLING_DATASET_ENV_VAR} is not a valid dataset name."
        )
    if not ads_client.valid_project(project):
        raise CostConfigError(
            f"{project!r} is not a GCP project id; check "
            f"{_BILLING_PROJECT_ENV_VAR} and the key's project_id."
        )
    return Billing(project, dataset, key_json)


def build_billing_client(config: Billing):
    """A BigQuery client for the billing export, read-only by its credential."""
    import ads_client  # noqa: PLC0415

    try:
        return ads_client.build_client(config)
    except ads_client.AdsConfigError as exc:
        raise CostConfigError(str(exc)) from exc


def billing_tables(client, config: Billing) -> list[str]:
    """The export tables in the dataset, oldest name first.

    Usually one, and empty in three cases that are all the same sentence to a
    reader: the dataset does not exist, the export was never enabled, or Google
    has not written the first table in the hours since it was. None of them is
    an error. More than one means more than one billing account exports here,
    and the panel reads the first and says so rather than adding two accounts'
    bills into one figure.
    """
    try:
        tables = list(client.list_tables(f"{config.project}.{config.dataset}"))
    except Exception as exc:  # noqa: BLE001 - the caller words this for a reader
        if _absent(exc):
            return []
        raise CostConfigError(
            f"Could not read `{config.project}.{config.dataset}`: {str(exc)[:200]}"
        ) from exc
    return [
        f"{config.project}.{config.dataset}.{name}"
        for name in sorted(
            table.table_id
            for table in tables
            if table.table_id.startswith(BILLING_TABLE_PREFIX)
        )
    ]


def _absent(exc: Exception) -> bool:
    """Whether BigQuery said there is no such dataset, rather than refusing it.

    A dataset nobody has created is where every deployment starts, and reads as
    a 404; a dataset the credential may not see is a 403 and worth a warning.
    """
    try:
        from google.api_core import exceptions  # noqa: PLC0415 - optional
    except ImportError:  # pragma: no cover - only without the client library
        return False
    return isinstance(exc, exceptions.NotFound)


# How long a correction row can land after the usage day it corrects: the
# export is written in arrears and backfills over hours according to Google's
# own guidance, not weeks, so a few days of slack past the usage window catches
# late-arriving rows without giving up the partition pruning the bound exists
# for.
_PARTITION_LAG_DAYS = 3


def cloud_costs(
    client, table: str, days: int, now: _dt.date | None = None
) -> pd.DataFrame:
    """Daily Google Cloud cost by project and service, over ``2 * days``.

    Net of credits, because that is what the invoice says: a committed-use
    discount or a free tier is applied against the line it belongs to, and the
    gross figure would report money that was never charged. Twice the window in
    one pass, as everywhere else here, since every figure is a comparison.

    The export is partitioned on ingestion time, not on ``usage_start_time``, so
    the usage-day filter below prunes no partitions by itself: unfiltered, every
    partition the table has ever written is a candidate. ``_PARTITIONTIME`` is
    filtered too, trailing the usage window by ``_PARTITION_LAG_DAYS`` so a
    correction ingested a day or two after the usage it corrects is still read.
    """
    today = now or _dt.date.today()
    first = today - _dt.timedelta(days=2 * days - 1)
    sql = f"""
        SELECT
          DATE(usage_start_time) AS day,
          COALESCE(project.id, 'unattributed') AS project,
          -- Nullable, unlike a service's own name: rounding rows and invoice
          -- adjustments carry no service, and pandas would drop the group and
          -- leave the breakdown adding up to less than the total above it.
          COALESCE(service.description, 'unattributed') AS line_item,
          LOWER(currency) AS currency,
          SUM(cost + IFNULL(
            (SELECT SUM(credit.amount) FROM UNNEST(credits) AS credit), 0
          )) AS cost
        FROM `{table}`
        WHERE DATE(usage_start_time) BETWEEN @first AND @last
          AND _PARTITIONTIME BETWEEN TIMESTAMP(@first) AND @partition_last
        GROUP BY day, project, line_item, currency
    """
    from google.cloud import bigquery  # noqa: PLC0415 - optional dependency

    partition_last = _dt.datetime.combine(
        today + _dt.timedelta(days=_PARTITION_LAG_DAYS), _dt.time.max
    )
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("first", "DATE", first),
                bigquery.ScalarQueryParameter("last", "DATE", today),
                bigquery.ScalarQueryParameter(
                    "partition_last", "TIMESTAMP", partition_last
                ),
            ]
        ),
    )
    frame = job.result().to_dataframe()
    if frame.empty:
        return pd.DataFrame(
            columns=["day", "project", "line_item", "cost", "currency"]
        )
    frame["day"] = pd.to_datetime(frame["day"]).dt.date
    # A refund or a credit larger than the day's usage nets to zero or below;
    # neither is a charge, and both would read as a line item on the bill. To
    # the cent, since a credit that cancels usage leaves a fraction behind that
    # survives an exact test and prints as "$-0.00" against a service name.
    frame["cost"] = frame["cost"].astype(float).round(2) + 0.0
    return frame[frame["cost"] != 0.0].reset_index(drop=True)


# Below this a day holds no charges worth the name. The export dates rounding
# corrections at the start of the billing period rather than at the usage they
# correct, so a table five days old carries June 1st rows worth a billionth of
# a cent - and a MIN over every row would call that a month of history.
_BILLING_DAY_FLOOR = 0.01

# Where the day's answer is written once it has been asked, keyed on the table
# and the day: a directory rather than a file, since more than one billing
# account can export into one dataset and each gets its own entry.
_BILLING_COVERAGE_CACHE_DIR = (
    pathlib.Path(tempfile.gettempdir()) / "jira-evaluator-billing-coverage"
)


def _billing_coverage_cache_path(
    directory: pathlib.Path, table: str, day: _dt.date
) -> pathlib.Path:
    key = hashlib.sha256(f"{table}:{day.isoformat()}".encode()).hexdigest()
    return directory / f"{key}.json"


def _read_billing_coverage_cache(
    path: pathlib.Path,
) -> tuple[_dt.date | None, _dt.date | None] | None:
    """The cached answer, or ``None`` where there is none yet to trust.

    Any way the file could fail to be a clean answer - not written yet,
    written by a different version, truncated by a crash mid-write - is read
    the same as no cache at all, and answered from BigQuery instead. A cache
    is a speed-up; it must never be the reason a figure comes back wrong.
    """
    try:
        payload = json.loads(path.read_text())
        first = _dt.date.fromisoformat(payload["first"]) if payload["first"] else None
        last = _dt.date.fromisoformat(payload["last"]) if payload["last"] else None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return first, last


def _write_billing_coverage_cache(
    path: pathlib.Path, result: tuple[_dt.date | None, _dt.date | None]
) -> None:
    first, last = result
    payload = {
        "first": first.isoformat() if first else None,
        "last": last.isoformat() if last else None,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except OSError:
        pass  # A cache that fails to write costs a slower tomorrow, not today.


def billing_coverage(
    client,
    table: str,
    now: _dt.date | None = None,
    cache_dir: str | pathlib.Path | None = None,
) -> tuple[_dt.date | None, _dt.date | None]:
    """The first and last days the export covers.

    Both ends matter and neither is today. The export is not retroactive, so it
    holds nothing from before it was switched on: a 30-day figure over a
    week-old export is a week's spend wearing a month's label. And it is
    written in arrears and backfills over hours, so its last day trails the
    calendar - a window ending today would read the missing days as free.

    The range asked for here is exactly the thing no date predicate can be
    handed in advance - finding it is the query's job - so the scan itself is
    unbounded over the whole export. What is bounded instead is how often it
    runs: the answer is cached to disk keyed by the table and the day, so a
    second call the same day, whether from the next Streamlit rerun or a fresh
    instance after a redeploy, reads yesterday's file rather than paying for
    the scan again.
    """
    today = now or _dt.date.today()
    directory = (
        pathlib.Path(cache_dir) if cache_dir is not None else _BILLING_COVERAGE_CACHE_DIR
    )
    cache_path = _billing_coverage_cache_path(directory, table, today)
    cached = _read_billing_coverage_cache(cache_path)
    if cached is not None:
        return cached
    job = client.query(
        "SELECT MIN(day) AS first, MAX(day) AS last FROM ("
        "SELECT DATE(usage_start_time) AS day, SUM(cost) AS total "
        f"FROM `{table}` GROUP BY day "
        f"HAVING ABS(SUM(cost)) >= {_BILLING_DAY_FLOOR})"
    )
    rows = list(job.result())
    if not rows:
        result = (None, None)
    else:
        first, last = rows[0]["first"], rows[0]["last"]
        # Never today, wherever the export has reached it: a day still being
        # written is hours of charges wearing a whole day's label, and ends the
        # window on a cheap day nobody had. The ads reader excludes today too.
        yesterday = today - _dt.timedelta(days=1)
        if last is None or first is None or first > yesterday:
            result = (None, None)
        else:
            result = (first, min(last, yesterday))
    _write_billing_coverage_cache(cache_path, result)
    return result


__all__ = [
    "Billing",
    "Burn",
    "Ledger",
    "CostConfigError",
    "DEFAULT_BILLING_DATASET",
    "LOOKBACK_WINDOWS",
    "billing_coverage",
    "billing_tables",
    "build_billing_client",
    "by_line",
    "cloud_costs",
    "load_billing_env",
    "by_project",
    "cached_share",
    "ledger_window",
    "load_openai_env",
    "main_currency",
    "load_stripe_env",
    "openai_costs",
    "stripe_ledger",
    "stripe_verdicts",
    "verdicts",
    "window",
]
