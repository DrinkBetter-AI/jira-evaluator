"""Phase 8: the Business page - orders, wine and merchant pricing,
Google Ads, Cloud/OpenAI/Stripe burn, and the Amplitude funnel. Split out
of app.py as its own module: nothing outside this page reads any name
defined here.
"""

from __future__ import annotations

import collections
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing, contextmanager
import datetime as _dt
import hashlib
import html
import logging
import os
import pickle
import re
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable, NamedTuple
from urllib.parse import quote, unquote, urlencode

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from dotenv import load_dotenv

# Local settings come from a .env file when one is present, so `streamlit run
# app.py` needs no exported variables. This runs before the local imports below
# because some of them read the environment at import time (change_audit picks
# its log path there). Existing variables win, so a real deployment's injected
# environment is never overridden by a file that happens to be lying around.
load_dotenv(override=False)

import read_log
from change_audit import (
    append_operation,
    finalize_operation,
    load_operations,
    new_operation_record,
    summarize_operations,
)
from jira_client import (
    DEFAULT_CREDS_PATH,
    DEFAULT_FIELDS,
    DEFAULT_PROFILE_NAME,
    MAX_PARALLEL_REQUESTS,
    JiraClient,
    JiraConfigError,
    normalize_base_url,
    load_jira_env,
    load_jira_profile,
)
from access_gate import render_sign_out, require_password
import focus
import github_client
import kpi
import integrity
import next_actions
import pr_hygiene
import pr_quality
import theme_html
from capacity import (
    capacity_table,
    same_person,
    match_weekly_hours,
    parse_weekly_hours,
    working_days,
)
import cleanup
from cleanup import is_unowned
import engineer_letter
import epic_organization
from epics import epic_health_flags, epic_rollup
from teams import (
    NO_OWNER_TEAM,
    DEFAULT_TEAM_PEOPLE,
    add_team,
    parse_team_people,
    parse_team_projects,
    team_summary,
)
import theme
import theme_tokens
from theme import inject_styles
import report as reporting
import snapshot as board
from hygiene import (
    CONTAINER_ISSUE_TYPES,
    DEFAULT_STALE_DAYS,
    estimate_policy,
    policy_compliance_by_owner,
    stale_candidates,
)
import ads_client
import ads_evidence
import cost_client
import amplitude_client
import merchant_client
import merchant_letter
import crawler_prices
import vivino_client
import orders
import orders_client
from prioritization import add_priority_score, assignee_rollup
import sprint_planner
import ticket_quality
from transformations import add_ticket_health_fields
import write_access
from contextlib import contextmanager



from data_layer import (
    _gather,
    _parallel,
)

from page_shared import (
    TAB_BUSINESS,
    _download_report,
    _number_or,
    _report,
    _said,
    _text_or,
    _tile,
    _unmathed,
)

logger = logging.getLogger(__name__)




# A year of orders, so the wine and merchant tables can look back 360 days, plus
# the days a 30-day figure needs to be shown as up or down on the month before.
ORDER_BOOK_DAYS = 390
ORDER_BOOK_TTL_SECONDS = 900


@read_log.logged_read("app._order_book")
@st.cache_data(ttl=ORDER_BOOK_TTL_SECONDS, show_spinner=False, refresh_mode="background")
def _order_book(source: str, days: int) -> orders_client.OrderBook:
    """The year of orders, re-read whole when the cache lapses.

    Keyed on the source's label rather than the config, so the password never
    becomes part of a cache key. Reading the year outright costs a single
    sub-second query, which is why there is no incremental top-up to go wrong.
    """
    read_log.mark_executed()
    config = orders_client.load_medusa_env()
    if config is None or config.label != source:
        raise orders_client.MedusaConfigError(
            "The order database configuration changed while it was being read."
        )
    return orders_client.read_order_book(config, days)


@read_log.logged_read("app.fetch_store_prefixes_cached")
@st.cache_data(ttl=3600, show_spinner=False, refresh_mode="background")
def fetch_store_prefixes_cached(source: str) -> dict[str, str]:
    read_log.mark_executed()
    config = orders_client.load_medusa_env()
    if config is None or config.label != source:
        raise orders_client.MedusaConfigError(
            "The order database configuration changed while it was being read."
        )
    return orders_client.fetch_stores(config)


def _money(amount: float, currency: str = "usd") -> str:
    symbol = {"usd": "$", "eur": "\u20ac", "gbp": "\u00a3"}.get(currency.lower(), "")
    return f"{symbol}{amount:,.2f}" + (f" {currency.upper()}" if not symbol else "")


def _business_readable() -> bool:
    """Whether the CRM, Amplitude, Ads, the cost APIs or the billing export read.

    Reading the environment is close enough to free that a deployment with no
    keys keeps the Business page out of the navigation entirely, rather than
    offering a link to a page that goes on to admit it cannot read anything.
    Close enough rather than free: with no project variable set, the Ads loader
    asks the ambient credentials which project they belong to, which is a
    metadata call - answered once per process, and swallowed when there is
    nothing to answer.
    """
    loaders = (
        orders_client.load_medusa_env,
        amplitude_client.load_amplitude_env,
        ads_client.load_ads_env,
        cost_client.load_openai_env,
        cost_client.load_stripe_env,
        cost_client.load_billing_env,
        merchant_client.load_merchant_env,
    )
    for load in loaders:
        try:
            if load() is not None:
                return True
        except (
            orders_client.MedusaConfigError,
            amplitude_client.AmplitudeConfigError,
            ads_client.AdsConfigError,
            cost_client.CostConfigError,
            merchant_client.MerchantConfigError,
        ):
            # Configured but wrongly - which is still worth rendering, because
            # the section is where the error message belongs.
            return True
    return False


def _render_business() -> None:
    """The shop's numbers, and how far visitors get towards being one of them.

    No longer behind a button. It was gated because Streamlit runs the body of
    every tab on every rerun, so the reads happened whichever tab the browser was
    showing; now this is its own page and nothing here runs until somebody asks
    for it. The year of orders costs about a sixth of a second anyway - the
    button was guarding the cheapest read on the dashboard.
    """
    business_slot = st.columns([5, 1])[1]
    _prefetch_ads()
    order_book = _render_business_sections()
    st.divider()
    # Straight after what sold: whether the shop is dearer than the rest of the
    # market is the first thing to ask of a week that sold less than the last.
    _render_price_benchmark()
    st.divider()
    # After the shop's own figures and before the funnel: what the orders cost to
    # win only means anything beside the orders themselves.
    _render_ads(order_book)
    st.divider()
    # Spend that is not advertising, after the spend that is: the two together
    # are what the revenue above has to cover.
    _render_burn()
    st.divider()
    _render_product_funnel()
    _download_report(business_slot, TAB_BUSINESS)


def _render_business_sections() -> orders_client.OrderBook | None:
    """The order book's own sections, and the book itself for other panels."""
    try:
        config = orders_client.load_medusa_env()
    except orders_client.MedusaConfigError as exc:
        st.subheader("Orders, Revenue & AOV")
        st.caption(f"Order figures are misconfigured: {exc}")
        return None
    if config is None:
        st.subheader("Orders, Revenue & AOV")
        st.caption(
            "Order figures need the order database's password. Set "
            "POSTGRES_PASSWORD (or MEDUSA_DB_PASSWORD) to the credential for "
            f"{orders_client.DEFAULT_USER} on {orders_client.DEFAULT_HOST}."
        )
        return None

    try:
        with st.spinner("Reading the order book..."):
            order_book = _order_book(config.label, ORDER_BOOK_DAYS)
    except Exception as exc:  # noqa: BLE001
        st.subheader("Orders, Revenue & AOV")
        st.warning(f"Could not read the order book: {str(exc)[:400]}")
        return None

    _render_orders(order_book, config.label)
    st.divider()
    _render_wines_and_merchants(order_book, config.label)
    return order_book


def _render_orders(order_book: orders_client.OrderBook, source: str) -> None:
    """Orders, revenue and AOV for the last 7 and 30 days, straight from the CRM."""
    st.subheader("Orders, Revenue & AOV")
    # Totals in different currencies cannot be added; the shop bills in one, and
    # if that ever stops being true the tiles report the main one and say so.
    book, currency, other_currencies = orders.single_currency(order_book.orders)
    week = orders.window_metrics(book, 7)
    month = orders.window_metrics(book, 30)
    for window, label in ((week, "7 days"), (month, "30 days")):
        tiles = st.columns(4)
        shop = "Orders, Revenue & AOV"
        _tile(
            tiles[0],
            TAB_BUSINESS,
            shop,
            f"Orders ({label})",
            f"{window.orders:,}",
            delta=f"{window.orders_delta:+,}",
        )
        _tile(
            tiles[1],
            TAB_BUSINESS,
            shop,
            f"Revenue ({label})",
            _money(window.revenue, currency),
            delta=(
                f"{'+' if window.revenue_delta >= 0 else '-'}"
                f"{_money(abs(window.revenue_delta), currency)}"
            ),
        )
        _tile(
            tiles[2], TAB_BUSINESS, shop, f"AOV ({label})", _money(window.aov, currency)
        )
        # Cancelled and unpaid are the gap between "orders" and "revenue", and
        # the reason the two tiles do not divide into each other.
        _tile(
            tiles[3],
            TAB_BUSINESS,
            shop,
            f"Cancelled / unpaid ({label})",
            f"{window.canceled} / {window.unpaid_orders}",
        )

    trend = orders.daily_orders(book, 30)
    if trend["orders"].sum():
        figure = px.bar(trend, x="date", y="orders", title="Orders per day (30 days)")
        figure.update_layout(height=260, margin=dict(l=0, r=0, t=40, b=0))
        theme.apply_palette(figure)
        theme.plot(figure, width="stretch", key="orders_daily")

    st.caption(
        "Revenue and AOV count captured payments only, so an order placed but not "
        "yet paid raises the order count and not the revenue; cancelled orders are "
        "excluded from both, and anything refunded is netted off. Deltas compare "
        "with the equivalent window before it, and the daily bars break the day at "
        f"UTC midnight. Read read-only from {source}, {ORDER_BOOK_DAYS} days at a "
        "time, straight from the CRM's own tables rather than its API."
    )
    if other_currencies:
        st.caption(
            f"Figures cover {currency.upper()} orders only; orders in "
            f"{', '.join(code.upper() for code in other_currencies)} are left out "
            "rather than added to a total in another currency."
        )


def _render_wines_and_merchants(
    order_book: orders_client.OrderBook, source: str
) -> None:
    """What sold, and how each merchant did, over a window the reader picks."""
    st.subheader("Best Sellers & Merchants")
    days = st.radio(
        "Window",
        options=list(orders.LOOKBACK_WINDOWS),
        format_func=lambda value: f"{value} days",
        index=0,
        horizontal=True,
        key="business_window_days",
    )

    # Same guard as the tiles: lines billed in another currency are set aside
    # rather than added into a column labelled with this one.
    _, currency, other_currencies = orders.single_currency(order_book.orders)
    items = orders.main_currency_items(order_book.items, currency)
    wines_tab, merchants_tab = st.tabs(["Top wines", "By merchant"])

    with wines_tab:
        wines = orders.top_wines(items, days)
        if wines.empty:
            st.info(f"No wine sold in the last {days} days.")
        else:
            st.dataframe(
                wines.rename(
                    columns={
                        "wine": "Wine",
                        "bottles": "Bottles",
                        "orders": "Orders",
                        "revenue": "Revenue",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                f"Top {orders.TOP_WINES_LIMIT} by bottles sold in the last {days} "
                "days, cancelled orders excluded. Bottles count what customers "
                "chose, so an order still awaiting payment counts; revenue is the "
                "line's own price, which is why it does not add up to the captured "
                "revenue above. Ice packs are left out."
            )

    with merchants_tab:
        try:
            prefixes = fetch_store_prefixes_cached(source)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not read the merchant list: {str(exc)[:200]}")
            return
        table = orders.merchant_breakdown(items, prefixes, days)
        if table.empty:
            st.info(f"No orders in the last {days} days.")
            return
        st.dataframe(
            table.rename(
                columns={
                    "merchant": "Merchant",
                    "revenue": f"Revenue ({currency.upper()})" if currency else "Revenue",
                    "orders": "Orders",
                    "canceled": "Cancelled",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        unattributed = table[table["merchant"].eq(orders.UNATTRIBUTED)]
        st.caption(
            "A merchant is read from the product handle on each line, so an order "
            "split between two merchants counts once for each and the order "
            "columns do not sum to the shop's totals. Revenue counts captured "
            "lines only, at the price on the line, less a share of anything the "
            "order was refunded."
        )
        if not unattributed.empty:
            st.caption(
                "'Unattributed' is wine whose handle matches no current merchant "
                "prefix, usually a shop that has since been renamed; it is shown "
                "rather than credited to a guess."
            )
        if other_currencies:
            st.caption(
                f"Both tables cover {currency.upper()} orders only; "
                f"{', '.join(code.upper() for code in other_currencies)} is left "
                "out rather than added to a total in another currency."
            )


# Merchant Center recomputes benchmarks daily, so a read held for six hours is
# as fresh as the data can be, and the catalogue is tens of thousands of rows
# fetched over HTTP - not a read to repeat because somebody moved the window on
# another panel.
BENCHMARK_TTL_SECONDS = 6 * 3600

# How many of the dearest offers to name. Enough for a pricing conversation to
# start with, few enough that the panel is not a spreadsheet.
_WORST_OFFERS = 15


# An order book that could not be read, which is not a catalogue nobody bought
# from: every table below keeps the two apart.
_NO_SALES = merchant_client.Sales(pd.DataFrame(), read=False)


class BenchmarkRead(NamedTuple):
    """The catalogue's prices against the market, and Google's advice on them."""

    prices: merchant_client.Prices
    insights: merchant_client.Insights
    demand: merchant_client.Demand
    # The shop's own sales per offer. Read separately from the three above and
    # from another system entirely, so that a CRM that cannot be reached costs
    # the evidence column rather than the whole panel.
    sales: merchant_client.Sales = _NO_SALES


@read_log.logged_read("app._price_benchmark_cached")
@st.cache_data(
    ttl=BENCHMARK_TTL_SECONDS, show_spinner=False, refresh_mode="background"
)
def _price_benchmark_cached(account: str, country: str) -> BenchmarkRead:
    """What Merchant Center says the shop's prices look like against the market.

    Keyed on the account rather than on the config, so that Streamlit hashes a
    string: the credential is read again inside, as the billing client is, and
    never becomes part of a cache key.
    """
    read_log.mark_executed()
    config = merchant_client.load_merchant_env()
    if config is None or (config.account, config.country) != (account, country):
        raise merchant_client.MerchantConfigError(
            "The Merchant Center configuration changed while it was being read."
        )
    token = merchant_client.access_token(config)
    # Three independent Merchant Center reports, so they go out together rather
    # than one after another. Only the prices are load-bearing, same as before:
    # a bad reply for either of the other two costs its own column, not the
    # whole panel, so their failures are swallowed here exactly as they were
    # when read in series.
    answers, errors = _gather(
        {
            "prices": lambda: merchant_client.price_gaps(config, token),
            "insights": lambda: merchant_client.price_insights(config, token),
            "demand": lambda: merchant_client.product_demand(config, token, country),
        }
    )
    if "prices" in errors:
        raise errors["prices"]
    insights = (
        answers["insights"]
        if "insights" not in errors
        else merchant_client.Insights(pd.DataFrame())
    )
    demand = (
        answers["demand"]
        if "demand" not in errors
        # Marked unread rather than empty: an empty report says nobody clicked,
        # and the panel would otherwise print that as a finding about the shop.
        else merchant_client.Demand(pd.DataFrame(), read=False)
    )
    return BenchmarkRead(answers["prices"], insights, demand)


# The catalogue moves slower than the prices in it, and the whole feed's
# merchants are read once and then handed to every table below, so it is held
# for a day.
_OFFER_MERCHANTS_TTL_SECONDS = 24 * 3600

# Orders arrive all day, but a quarter of them is a slow-moving figure and this
# read groups the whole quarter, so it is held on the benchmarks' own cycle.
_OFFER_SALES_TTL_SECONDS = 6 * 3600


@read_log.logged_read("app._offer_sales_cached")
@st.cache_data(
    ttl=_OFFER_SALES_TTL_SECONDS, show_spinner=False, refresh_mode="background"
)
def _offer_sales_cached(source: str, days: int, today: _dt.date) -> pd.DataFrame:
    """Bottles sold per Google offer, from the shop's own order book.

    Keyed on the day as well as the window so the quarter rolls forward with
    the calendar rather than whenever the cache happens to expire.
    """
    read_log.mark_executed()
    del today
    config = orders_client.load_medusa_env()
    if config is None or config.label != source:
        raise orders_client.MedusaConfigError(
            "The order database configuration changed while it was being read."
        )
    sold = orders_client.fetch_offer_sales(config, days)
    if sold.empty:
        return sold
    # An ice pack ships one per order and is nobody's bottle, so it would carry
    # a wine-sized count into whichever price band it landed in. The wine table
    # drops it the same way.
    return sold[~sold["handle"].map(orders.is_add_on)].reset_index(drop=True)


# Ad spend per wine is read over the same quarter as the sales it is set beside:
# a month of spend against a quarter of orders would read as a return the ads
# never earned.
_AD_PRODUCTS_TTL_SECONDS = 6 * 3600


class AdProducts(NamedTuple):
    """Spend per offer, and the currency it was actually billed in."""

    frame: pd.DataFrame
    currency: str
    # Accounts left out because they bill in some other currency, named so the
    # reader knows the total is not the whole dataset.
    other_currencies: list[str]
    # False when the report could not be read at all, which is not an account
    # that spent nothing - the same distinction ``Sales.read`` keeps.
    read: bool = True
    # The earliest day the Shopping product table holds, asked of that table
    # rather than of the campaign one beside it: the two are transferred
    # separately, and only this one says how much of a per-wine window is real.
    history_start: _dt.date | None = None
    # Accounts whose product table could not be read while another's could, so
    # the total below is short by whatever they spent.
    unread_accounts: int = 0


def _no_ad_products(read: bool = True) -> AdProducts:
    return AdProducts(
        pd.DataFrame(columns=list(ads_client.PRODUCT_COLUMNS)), "", [], read
    )


@read_log.logged_read("app._ad_products_cached")
@st.cache_data(ttl=_AD_PRODUCTS_TTL_SECONDS, show_spinner=False)
def _ad_products_cached(
    project: str, dataset: str, days: int, today: _dt.date
) -> AdProducts:
    """What each advertised offer cost, over the accounts that bill alike.

    Keyed on the day for the reason the campaign reads are: the window has to
    roll forward with the calendar rather than with the cache's timer.

    A dataset can hold several ad accounts, and dollars and euros are never
    added: the most common billing currency wins and the rest are set aside, the
    same rule the campaign read and the order book follow.
    """
    read_log.mark_executed()
    config = _ads_config(project, dataset)
    client = _ads_bigquery_client(project, dataset)
    accounts = _ads_accounts(project, dataset, today)
    if not accounts:
        return _no_ad_products()
    counted = collections.Counter(account.currency for account in accounts)
    main = counted.most_common(1)[0][0]
    others = sorted({code for code in counted if code != main})
    billing = [account for account in accounts if account.currency == main]
    read = _parallel(
        {
            account.customer_id: (
                lambda customer_id=account.customer_id: _one_account_products(
                    client, config, customer_id, days, today
                )
            )
            for account in billing
        }
    )
    got = [answer for answer in read.values() if answer is not None]
    if not got:
        return _no_ad_products(read=False)
    starts = [start for _, start in got if start is not None]
    return AdProducts(
        _offers_together([frame for frame, _ in got]),
        main,
        others,
        # The latest of the accounts' first days, for the reason the campaign
        # read takes it: a total is only wholly loaded once every account in it
        # has reached back that far.
        history_start=max(starts) if starts else None,
        unread_accounts=len(billing) - len(got),
    )


def _one_account_products(
    client, config, customer_id: str, days: int, today: _dt.date
) -> tuple[pd.DataFrame, _dt.date | None] | None:
    """One account's per-offer spend, and the first day its product table holds.

    ``None`` rather than an exception when that account cannot be read. Accounts
    are found from the campaign tables, and the Shopping product report is
    transferred separately and often switched on months later, so a dataset can
    hold an account with no product table at all - and raising here took the
    spending account's whole tab down with it, which is the thing
    ``_offers_together`` exists to prevent. Every account failing is still a
    report that could not be read, and the caller says so.

    The history is asked of the product table itself: the campaign table's first
    day, already read for the account, says nothing about how much per-wine spend
    is loaded.
    """
    try:
        return (
            ads_client.product_stats(client, config, customer_id, days, today),
            ads_client.loaded_from(
                client, config, customer_id, today, ads_client.PRODUCT_TABLE
            ),
        )
    except Exception:  # noqa: BLE001 - one account's read, not the panel's
        return None


def _offers_together(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Several accounts' product rows as one row per offer.

    One row per offer even where two accounts advertised the same bottle: the
    panel's subject is the wine, and the same wine twice would halve its apparent
    return.

    Only the accounts that have rows go into the concatenation. An empty frame's
    columns are all object dtype, and concatenating one promotes clicks and
    impressions to object, where a numeric aggregation drops them and the ledger
    loses two of the columns it is built from - one account at rest breaking the
    tab of the account that is spending.
    """
    spending = [frame for frame in frames if not frame.empty]
    if not spending:
        return pd.DataFrame(columns=list(ads_client.PRODUCT_COLUMNS))
    return (
        pd.concat(spending, ignore_index=True)
        .groupby("offer", as_index=False)
        .sum(numeric_only=True)
        .sort_values("spend", ascending=False)
        .reset_index(drop=True)
    )


def _ad_products(days: int) -> AdProducts:
    """Ad spend per offer, or nothing at all when Ads cannot be read.

    Empty rather than raised: this is one tab of a panel whose other tabs do not
    need Google Ads, and a dataset nobody has set up is not an error on a page
    about prices.
    """
    try:
        config = ads_client.load_ads_env()
        if config is None:
            return _no_ad_products()
        return _ad_products_cached(
            config.project, config.dataset, days, _dt.date.today()
        )
    except Exception:  # noqa: BLE001
        # A refused credential, an absent Shopping table or a BigQuery nobody
        # can reach is not an account that spent nothing, and this tab exists to
        # argue about spend: say it could not be read.
        return _no_ad_products(read=False)


def _ads_configured() -> bool:
    """Whether the dashboard has been pointed at an Ads dataset at all.

    An empty tab has two causes worth telling apart: nobody configured Google
    Ads, or it is configured and these wines took no money. Reading environment
    variables out at somebody who has already set them is the panel being wrong
    about itself.
    """
    try:
        return ads_client.load_ads_env() is not None
    except Exception:  # noqa: BLE001
        return False


def _offer_sales() -> merchant_client.Sales:
    """What the shop sold, or an unread ``Sales`` when the CRM is unreachable."""
    try:
        config = orders_client.load_medusa_env()
        if config is None:
            return _NO_SALES
        frame = _offer_sales_cached(
            config.label, merchant_client.SALES_DAYS, _dt.date.today()
        )
    except Exception:  # noqa: BLE001
        return _NO_SALES
    return merchant_client.Sales(frame, merchant_client.SALES_DAYS)


# How far a merchant might be asked to come down. Past a third off, the question
# stops being a price negotiation and becomes a question about the wine.
_MAX_CUT_PERCENT = 30
_DEFAULT_CUT_PERCENT = 10


@read_log.logged_read("app._offer_merchants_cached")
@st.cache_data(
    ttl=_OFFER_MERCHANTS_TTL_SECONDS, show_spinner=False, refresh_mode="background"
)
def _offer_merchants_cached(
    source: str, offers: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    """Which merchants list each of these Google offers.

    Each offer's merchants are kept apart rather than joined into a string: a
    shop is free to have a comma in its name, and a name split back out of one
    would be a merchant that matches nothing.

    Google knows the bottle and its price; only the catalogue knows whose
    listing that is, and a bottle several merchants stock names all of them -
    the one to ask is the one whose price is the one in the feed, and this is
    the panel saying who to start with rather than deciding for you.
    """
    read_log.mark_executed()
    config = orders_client.load_medusa_env()
    if config is None or config.label != source:
        raise orders_client.MedusaConfigError(
            "The order database configuration changed while it was being read."
        )
    # The offer listings and the store prefixes are two unrelated reads of the
    # same order database, so they go out together rather than one after another.
    read = _parallel(
        {
            "handles": lambda: orders_client.fetch_offer_handles(config, list(offers)),
            "prefixes": lambda: orders_client.fetch_stores(config),
        }
    )
    handles, prefixes = read["handles"], read["prefixes"]
    return {
        offer: tuple(
            sorted({orders.merchant_of(handle, prefixes) for handle in listings})
        )
        for offer, listings in handles.items()
    }


# The choice that means no filter at all, and how a wine stocked by two
# merchants is named on the page.
_EVERY_MERCHANT = "Every merchant"
_MERCHANT_SEPARATOR = merchant_client.MERCHANT_SEPARATOR

# The merchants actually trading. The catalogue still carries wine from shops
# that have been switched off, and counting them was distorting the one number
# this whole section exists to state: "87% of products cost more here than the
# market" was measured across inventory nobody can buy. A disabled shop's
# prices are nobody's decision, so they belong outside the denominator rather
# than inside it with an asterisk.
#
# ``ACTIVE_MERCHANTS`` is semicolon-separated - not comma-separated - to match
# roles_template.env's own JIRA_ROLES/GITHUB_LOGIN_MAP convention, since a
# merchant's catalogue name is free to carry a comma of its own.
#
# Baked in below, copied verbatim from roles_template.env's own
# ACTIVE_MERCHANTS line (regenerated 19 Aug 2026), for the same reason
# roles.py bakes in JIRA_ROLES: Cloud Run carries no ACTIVE_MERCHANTS env var
# today, that is this deployment's actual state rather than a hypothetical,
# and the page has to be correct anyway. The env var wins when it is set;
# unset, the page now falls back to this list rather than to "every merchant,
# including the ones switched off" - the old fallback silently let a stale
# catalogue answer the one number this section exists to state.
#
# The five names are the vendor panel's Active list (Angel's screenshot):
# Yiannis, Black Bear, Capital Fine Wine, TheWinesGood, World of wine. Little
# International is NOT here - DEVIN_PLAN prereq 3 called it active, but the
# vendor panel is newer and wins (docs/assumptions/3F.md).
_ACTIVE_MERCHANTS_ENV = "ACTIVE_MERCHANTS"
_DEFAULT_ACTIVE_MERCHANTS = (
    "Yiannis;Black Bear;Capital Fine Wine;TheWinesGood;World of wine"
)


def _active_merchant_names() -> frozenset[str] | None:
    """The trading roster: ``ACTIVE_MERCHANTS`` if set, else the baked default.

    Never ``None`` in practice any more - the baked default is a fixed,
    non-empty string - but the return type stays optional so a caller that
    checks ``if active:`` before filtering keeps working unchanged.
    """
    raw = os.getenv(_ACTIVE_MERCHANTS_ENV, "").strip() or _DEFAULT_ACTIVE_MERCHANTS
    names = frozenset(part.strip() for part in raw.split(";") if part.strip())
    return names or None


# A separate, still-open question from the roster above: whether
# ``medusa.store``'s own metadata agrees with ACTIVE_MERCHANTS. Read-only and
# best-effort - this is a provenance note, not a source of truth the filter
# above depends on, so a failed read here degrades to a caption saying so
# rather than touching what counts as trading. See docs/assumptions/3F.md for
# the exact verification query this mirrors.
_STORE_METADATA_SQL = "select name, metadata from medusa.store limit 12"


def _store_metadata_note(rows: list[tuple[str, dict | None]]) -> str:
    """The data-provenance caption: which metadata key decided each row.

    Split from the DB read below so it can be exercised with fabricated rows
    and no database in the loop. Never raises: ``merchant_client.resolve_store_status``
    already doesn't, and this only sorts and joins its answers.
    """
    matched: dict[str, str] = {}
    for name, metadata in rows:
        name = str(name or "").strip()
        if not name:
            continue
        _active, key = merchant_client.resolve_store_status(metadata, name)
        if key:
            matched[name] = key
    if not matched:
        return (
            "Store metadata: none of medusa.store's rows carried "
            + ", ".join(merchant_client.STORE_STATUS_KEYS)
            + f" ({len(rows)} row(s) read). Active/inactive above comes from "
            f"{_ACTIVE_MERCHANTS_ENV} alone."
        )
    keys_used = sorted(set(matched.values()))
    if len(keys_used) == 1:
        return (
            f"Store metadata: every row read used the {keys_used[0]!r} key "
            f"({len(matched)} of {len(rows)}). Safe to trim "
            "merchant_client.STORE_STATUS_KEYS to just that one."
        )
    detail = ", ".join(f"{n}={k}" for n, k in sorted(matched.items()))
    return f"Store metadata: the matching key varies by store - {detail}."


@st.cache_data(ttl=3600, show_spinner=False)
def _store_metadata_note_cached(source: str) -> str:
    """:func:`_store_metadata_note`, over a live sample - or why it could not run.

    Cached for an hour on the DB config's label, same pattern as
    ``_offer_merchants_cached``: this is a provenance sentence, not live
    inventory, and re-reading it every rerun would cost a database round trip
    for text that does not change between page loads.
    """
    config = orders_client.load_medusa_env()
    if config is None or config.label != source:
        return "Store metadata: not checked, the order database is not configured."
    try:
        import psycopg2

        # ``closing``, not a bare ``with``: a psycopg2 connection used as a
        # context manager ends the transaction and leaves the socket open.
        # ``orders_client`` learned this the hard way (see its comment about
        # leaking a connection on every refresh until the server ran out) and
        # every read there goes through ``closing``; this one now does too.
        with closing(
            psycopg2.connect(
                host=config.host,
                port=config.port,
                dbname=config.database,
                user=config.user,
                password=config.password,
                connect_timeout=10,
            )
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(_STORE_METADATA_SQL)
                rows = cursor.fetchall()
    except Exception as exc:  # noqa: BLE001 - a caption, not a page, on failure
        logger.info("store metadata read failed: %s", exc)
        return (
            "Store metadata: not verified, the medusa.store read failed "
            f"({str(exc)[:120]}). Tried, in order: "
            + ", ".join(merchant_client.STORE_STATUS_KEYS)
            + "."
        )
    return _store_metadata_note(rows)


def _store_metadata_provenance() -> str:
    """The caption above, guarded so a surprise here never breaks the page."""
    try:
        config = orders_client.load_medusa_env()
    except Exception:  # noqa: BLE001
        config = None
    if config is None:
        return "Store metadata: not checked, the order database is not configured."
    try:
        return _store_metadata_note_cached(config.label)
    except Exception as exc:  # noqa: BLE001
        return f"Store metadata: not checked ({str(exc)[:120]})."


def _trading_only(
    prices: merchant_client.Prices,
    named: dict[str, tuple[str, ...]],
    active: frozenset[str] | None,
) -> tuple[merchant_client.Prices, dict[str, tuple[str, ...]], int]:
    """Drop the offers that belong only to shops that are switched off.

    Returns the narrowed read, the narrowed offer-to-merchant map, and how many
    offers were set aside, so the page can say what it left out rather than
    quietly reporting a smaller catalogue than the feed holds.
    """
    if not active or not named:
        return prices, named, 0
    kept_names = {
        offer: tuple(name for name in names if name in active)
        for offer, names in named.items()
    }
    trading = {offer: names for offer, names in kept_names.items() if names}
    if not trading:
        # Every name in the roster is a typo, or the catalogue calls these
        # shops something else. Reporting nothing would look like a dead feed,
        # so the honest move is to leave the read alone and let the caption say
        # the roster matched none of it.
        return prices, named, 0
    kept = prices.offers[prices.offers["offer"].isin(trading)].reset_index(drop=True)
    set_aside = int(len(prices.offers) - len(kept))
    return (
        merchant_client.Prices(
            kept, prices.currency, prices.other_currencies, prices.truncated
        ),
        trading,
        set_aside,
    )


def _offer_merchants(offers: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Each offer's merchants, or nothing at all when the catalogue is shut."""
    if offers.empty or "offer" not in offers.columns:
        return {}
    try:
        config = orders_client.load_medusa_env()
        if config is None:
            return {}
        return _offer_merchants_cached(config.label, tuple(offers["offer"]))
    except Exception:  # noqa: BLE001
        return {}


def _one_merchant(
    prices: merchant_client.Prices,
    named: dict[str, tuple[str, ...]],
    merchant: str,
) -> merchant_client.Prices:
    """The same read, cut down to the offers one merchant lists.

    The whole point of the filter: a merchant will not read a five-thousand-row
    catalogue to find its own wine, and the case for repricing is made shop by
    shop. Everything above and below - the share dearer than the market, the
    ask list, the evidence - is then that merchant's, and the file downloaded
    beside it is the one to send them.
    """
    if merchant == _EVERY_MERCHANT or not named:
        return prices
    mine = {offer for offer, names in named.items() if merchant in names}
    kept = prices.offers[prices.offers["offer"].isin(mine)].reset_index(drop=True)
    return merchant_client.Prices(
        kept, prices.currency, prices.other_currencies, prices.truncated
    )


# A whole shop is read from Vivino page by page, so the read is kept for a
# day rather than repeated on every rerun; the refresh button clears it with
# the other reads.
_VIVINO_TTL_SECONDS = 24 * 3600
_CRAWLER_TTL_SECONDS = 3600


@read_log.logged_read("app._crawled_shop_cached")
@st.cache_data(ttl=_CRAWLER_TTL_SECONDS, show_spinner=False)
def _crawled_shop_cached(
    source: str, slug: str
) -> vivino_client.Shop | None:
    """Keep the quick daily read separate from the button-gated live fallback."""
    read_log.mark_executed()
    config = orders_client.load_medusa_env()
    if config is None or config.label != source:
        raise orders_client.MedusaConfigError(
            "The order database configuration changed while it was being read."
        )
    return crawler_prices.fetch_crawled_shop(config, slug)


@read_log.logged_read("app._vivino_comparison_cached")
@st.cache_data(ttl=_VIVINO_TTL_SECONDS, show_spinner=False)
def _vivino_comparison_cached(
    source: str, merchant: str, slug: str
) -> vivino_client.Comparison:
    """One merchant's Vivino prices against their prices here, matched wine by wine."""
    read_log.mark_executed()
    config = orders_client.load_medusa_env()
    if config is None or config.label != source:
        raise orders_client.MedusaConfigError(
            "The order database configuration changed while it was being read."
        )
    prefixes = orders_client.fetch_stores(config)
    # Live store prefixes are merged in after configured aliases, so the last
    # match is the current one; no match at all is a naming problem to report,
    # not a merchant with an empty cellar.
    matching = [pref for pref, name in prefixes.items() if name == merchant]
    if not matching:
        raise orders_client.MedusaConfigError(
            f"No store in the order database is named {merchant}, so their "
            "own catalogue prices cannot be read to compare."
        )
    # Every prefix carrying the name is read - an alias exists exactly
    # because the products still wear a retired prefix - and the union is
    # their catalogue.
    others = tuple(p for p in prefixes if p not in matching)
    pieces = [
        orders_client.fetch_catalog(config, prefix, others=others)
        for prefix in matching
    ]
    ours = pd.concat(pieces, ignore_index=True)
    try:
        crawled = _crawled_shop_cached(source, slug)
    except Exception:  # noqa: BLE001 - the button-gated live read remains available
        crawled = None
        crawler_unavailable = True
    else:
        crawler_unavailable = False
    if crawled is not None:
        return replace(
            vivino_client.compare(ours, crawled),
            crawled_at=crawled.crawled_at,
            from_crawler=True,
        )
    shop = vivino_client.fetch_shop(slug)
    return replace(
        vivino_client.compare(ours, shop),
        crawler_missing=not crawler_unavailable,
        crawler_unavailable=crawler_unavailable,
    )


def _vivino_blocked(exc: Exception) -> bool:
    """Whether ``exc`` is Vivino's 403, as opposed to any other read failure.

    Vivino refuses Cloud Run's shared egress addresses outright (see
    vivino_client.py's own note on ``_PROXY_VAR``); every request this
    deployment makes to it fails the same specific way. ``vivino_client``
    always surfaces that as a ``VivinoError`` or a wrapped
    ``requests.HTTPError`` whose message carries "403" - not a status code
    attribute this function can read directly, since the wrapping loses it -
    so the text is what is checked. A miss here (a timeout, a 5xx, a changed
    page) still falls through to the generic warning below, unchanged.
    """
    return "403" in str(exc)


def _render_vivino(chosen: str, picker: bool = True) -> None:
    """What the chosen merchant charges on Vivino against what they charge here.

    Single 0.75l bottles only, matched by wine name and vintage - all both
    sides publish - and honest about what could not be compared: Vivino's feed
    read short, a shop with no listings, a merchant with no known Vivino page.
    """
    # The order database is what names the merchants, so when it cannot be
    # read there may be no merchant picker on the page at all - that check
    # comes first, or the reader is pointed at a picker that is not there.
    config = orders_client.load_medusa_env()
    if config is None:
        st.caption(
            "The comparison needs the shop's own catalogue prices, which come "
            "from the order database. Set POSTGRES_PASSWORD (or "
            "MEDUSA_DB_PASSWORD) to read them."
        )
        return
    if chosen == _EVERY_MERCHANT:
        if not picker:
            # No merchant chooser was drawn - the store names could not be
            # read - so asking the reader to pick one would point at nothing.
            st.caption(
                "The merchants' names could not be read from the order "
                "database just now, so there is nobody to compare with "
                "their Vivino shop. Refresh once the database is reachable."
            )
            return
        st.caption(
            "Pick a merchant above: Vivino prices are one shop's against the "
            "same shop's prices here, not a market average."
        )
        return
    slug = vivino_client.VIVINO_SHOPS.get(chosen)
    if not slug:
        st.caption(
            f"No Vivino shop is on record for {chosen}. The ones known are "
            + ", ".join(sorted(vivino_client.VIVINO_SHOPS))
            + "; if they open one, add it to VIVINO_SHOPS in vivino_client.py."
        )
        return
    # The read walks the shop's Vivino listings page by page and can take
    # minutes, and every tab's body runs whether or not it is the one open -
    # so it starts on a press, not on a merchant being picked for another tab.
    # The read this session already made is held here with its result, so the
    # gate cannot outlive what it stands for: the shared cache expiring or
    # being refreshed elsewhere never restarts the pull without the button.
    reads = st.session_state.setdefault("vivino_reads", {})
    held = reads.get(slug)
    if held and time.time() - held[0] < _VIVINO_TTL_SECONDS:
        result = held[1]
    else:
        reads.pop(slug, None)
        crawler_unavailable = False
        try:
            crawled = _crawled_shop_cached(config.label, slug)
        except Exception as exc:  # noqa: BLE001 - a bad reply stays in this tab
            st.warning(f"Could not read crawler Vivino prices: {str(exc)[:300]}")
            crawled = None
            crawler_unavailable = True
        if crawled is not None:
            try:
                result = _vivino_comparison_cached(config.label, chosen, slug)
            except Exception as exc:  # noqa: BLE001 - a bad reply stays in this tab
                st.warning(f"Could not compare their Vivino prices: {str(exc)[:300]}")
                return
        else:
            if not st.button(
                f"Read {chosen}'s Vivino shop (takes a few minutes)",
                key="vivino_read",
            ):
                st.caption(
                        "Their whole Vivino shop is read page by page and matched to "
                        "their prices here by wine name and vintage, single 0.75l "
                        "bottles only. "
                        + (
                            "The daily crawler is unavailable; the live read is "
                            "kept behind this button."
                            if crawler_unavailable
                            else "The daily crawler has no rows for this merchant "
                            "yet; the live read is kept behind this button."
                        )
                )
                return
            try:
                with st.spinner(f"Reading {chosen}'s Vivino shop, page by page..."):
                    result = _vivino_comparison_cached(config.label, chosen, slug)
            except Exception as exc:  # noqa: BLE001 - a bad reply stays in this tab
                # Not recorded as read: a failure would otherwise restart the
                # minutes-long pull on every rerun instead of waiting for the button.
                if _vivino_blocked(exc):
                    # 403 from Vivino is not a bug in this page - it is Vivino's
                    # shared-egress block, permanent until VIVINO_PROXY names a
                    # host Vivino answers - and a raw exception dump reads as one.
                    # Saying "unavailable" plainly is honest where a stack trace
                    # is not: nothing here failed, Vivino refused the request.
                    st.info(
                        "Their Vivino price: unavailable — Vivino blocks our "
                        "requests. This is a known, permanent limit for this "
                        "deployment's egress, not a fault in the page; set "
                        "VIVINO_PROXY to a forward proxy Vivino will answer to "
                        "compare here."
                    )
                else:
                    st.warning(f"Could not compare their Vivino prices: {str(exc)[:300]}")
                return
        reads[slug] = (time.time(), result)

    for line in vivino_client.verdicts(chosen, result):
        st.markdown(line)
    if result.from_crawler:
        if result.crawled_at is not None:
            crawled_at = result.crawled_at
            if crawled_at.tzinfo is None:
                crawled_at = crawled_at.replace(tzinfo=_dt.timezone.utc)
            crawled_at = crawled_at.astimezone(_dt.timezone.utc)
            freshness = crawled_at.strftime("%Y-%m-%d %H:%M UTC")
            st.caption(f"Prices from the daily Vivino crawl, last crawled {freshness}.")
        else:
            st.caption("Prices from the daily Vivino crawl; crawl freshness was unavailable.")
    elif result.crawler_missing:
        st.caption(
            f"The daily Vivino crawl has no rows for {chosen} yet; "
            "showing today's live Vivino read."
        )
    elif result.crawler_unavailable:
        st.caption(
            "The daily Vivino crawl was unavailable; showing today's live "
            "Vivino read."
        )
    if not result.matched:
        return

    cheaper = result.cheaper_there
    shown = cheaper if len(cheaper) else result.rows
    st.dataframe(
        shown.assign(
            year=lambda frame: frame["year"].map(
                lambda value: str(int(value)) if value else "NV"
            ),
            ours=lambda frame: frame["ours"].map(lambda value: f"${value:,.2f}"),
            theirs=lambda frame: frame["theirs"].map(lambda value: f"${value:,.2f}"),
            gap=lambda frame: frame["gap"].map(lambda value: f"{value:+.0%}"),
        ).rename(
            columns={
                "wine": "Wine",
                "year": "Vintage",
                "ours": "Price here",
                "theirs": "Price on Vivino",
                "gap": "Vivino against here",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        f"Download {chosen}'s Vivino comparison (CSV)",
        result.rows.to_csv(index=False).encode(),
        file_name=f"vivino-vs-us-{slug}.csv",
        mime="text/csv",
        key="vivino_csv",
    )
    packs_note = (
        f"{result.packs:,} of their Vivino wines are priced only per bottle "
        "of a pack - 3, 6 or 12 bottles bought together - and are left out: "
        "only what one bottle costs is compared with what one bottle costs "
        "here. "
        if result.packs
        else ""
    )
    st.caption(
        "Same wine, same vintage, single 0.75l bottles, both prices in USD "
        "before shipping - Vivino's checkout may add shipping differently. "
        f"{packs_note}"
        f"Vivino read within the last day; {result.unmatched_ours:,} of the "
        "merchant's wines here found no Vivino listing by name and vintage "
        "and are left out rather than guessed at."
    )


def _with_merchants(
    frame: pd.DataFrame, named: dict[str, tuple[str, ...]] | None = None
) -> pd.DataFrame:
    """``frame`` with a merchant column, or unchanged if the catalogue is shut.

    The names are worth a lot to the conversation and nothing to the arithmetic,
    so a CRM that cannot be reached costs the column rather than the table.
    """
    named = _offer_merchants(frame) if named is None else named
    if not named:
        return frame
    return frame.assign(
        merchant=frame["offer"].map(
            lambda offer: _MERCHANT_SEPARATOR.join(named.get(offer, ()))
        )
    )


def _demand_note(demand: merchant_client.Demand) -> str:
    """What the ordering rests on, and what it does not.

    A report that could not be read and a shop nobody clicked leave the same
    empty frame behind, and only one of them is a fact about the wines.
    """
    if demand.measured:
        return f"Clicks are the last {merchant_client.DEMAND_DAYS} days in Shopping."
    if not demand.read:
        return (
            "Ranked by the gap alone: Merchant Center's performance report "
            "could not be read, so how many shoppers each of these lost is "
            "unknown rather than none."
        )
    return (
        "Ranked by the gap alone: Shopping reported no clicks on these "
        f"products in the last {merchant_client.DEMAND_DAYS} days, so there is "
        "no demand to weigh it by."
    )


def _price_columns(money: str) -> dict[str, object]:
    """The formatters the price tables share, so the same column reads the same."""
    return {
        "price": lambda value: _money(value, money),
        "benchmark": lambda value: _money(value, money),
        "gap": lambda value: f"{value:+.0%}",
        "cut": lambda value: f"-{value:.0%}",
        "overpay": lambda value: _money(value, money),
        "clicks": lambda value: f"{int(value):,}",
        "impressions": lambda value: f"{int(value):,}",
        "bottles": lambda value: f"{int(value):,}",
        "cut_price": lambda value: _money(value, money),
        "cut_gap": lambda value: f"{value:+.0%}",
    }


def _visible(
    frame: pd.DataFrame,
    wanted: tuple[str, ...],
    clicked: bool,
    sold: bool = True,
) -> list[str]:
    """Which of ``wanted`` the frame can actually show.

    A clicks column of nothing but zeros reads as a measurement rather than as
    a missing report, so when there is no demand to show the column goes with
    it and the caption says why. The bottles sold go the same way when the order
    book could not be read.
    """
    hidden = set()
    if not clicked:
        hidden |= {"clicks", "impressions"}
    if not sold:
        hidden.add("bottles")
    return [
        column for column in wanted if column in frame.columns and column not in hidden
    ]


def _formatted(frame: pd.DataFrame, money: str) -> pd.DataFrame:
    formatters = _price_columns(money)
    shown = frame.copy()
    for column, formatter in formatters.items():
        if column in shown.columns:
            shown[column] = shown[column].map(formatter)
    return shown


# What the ask list shows, on screen and in the file, in the order a phone call
# would go through them. The frame behind it also carries the working out - the
# impact score, and Google's predicted change in conversions, which this panel
# will not report because the feed measures none - and neither belongs in a
# spreadsheet read a long way from the caption that would have said so.
_ASK_COLUMNS = (
    "title",
    "merchant",
    "clicks",
    "bottles",
    "price",
    "benchmark",
    "gap",
    "cut_price",
    "cut_gap",
    "cut",
    "overpay",
    "google_cut",
)


def _render_ask_list(
    read: BenchmarkRead,
    money: str,
    named: dict[str, tuple[str, ...]] | None = None,
) -> None:
    """The hundred bottles worth taking to a merchant, best argument first.

    Nobody reprices five thousand wines, so the panel's job is to choose the
    argument: the wines shoppers are already clicking on and finding dearer
    here than everywhere else, which is where a percentage off buys the most
    back. Ranked on clicks times the gap - demand seen, times how far over the
    market that demand was asked to pay.
    """
    wines = merchant_client.ask_list(
        read.prices, read.demand, read.insights, merchant_client.ASK_LIST
    )
    if wines.empty:
        st.caption("Nothing in the feed is priced above the market.")
        return
    percent = st.slider(
        "If merchants came down by",
        min_value=0,
        max_value=_MAX_CUT_PERCENT,
        value=_DEFAULT_CUT_PERCENT,
        step=1,
        format="%d%%",
        key="price_ask_cut",
    )
    cut = percent / 100
    priced = merchant_client.after_cut(wines, cut)
    beaten = merchant_client.beats_market(priced)
    demand = read.demand
    st.caption(
        f"At {cut:.0%} off, {beaten} of these {len(priced)} would be at or below "
        f"the market price, and {len(priced) - beaten} would still be above it. "
        + _demand_note(demand)
    )
    if demand.truncated:
        st.caption(
            "The clicks are as far as the performance report was read, so a "
            "wine further down it can read lower here than it was."
        )
    shown = _with_merchants(read.sales.against(priced), named)
    columns = _visible(
        shown, _ASK_COLUMNS, demand.measured, read.sales.measured_against(priced)
    )
    table = _formatted(shown, money)[columns]
    if "google_cut" in table.columns:
        table["google_cut"] = shown["google_cut"].map(
            lambda value: "" if pd.isna(value) else f"-{value:.0%}"
        )
    st.dataframe(
        table.rename(
            columns={
                "title": "Wine",
                "merchant": "Merchant",
                "clicks": "Clicks 30d",
                "bottles": f"Sold {merchant_client.SALES_DAYS}d",
                "price": "Our price",
                "benchmark": "Market",
                "gap": "Gap",
                "cut_price": f"At -{percent}%",
                "cut_gap": "Gap then",
                "cut": "Cut to match",
                "overpay": "Per bottle",
                "google_cut": "Google suggests",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download the ask list",
        # The columns on the screen and no others.
        data=shown[columns].to_csv(index=False).encode("utf-8"),
        file_name=f"price-ask-list-{merchant_client.as_of()}.csv",
        mime="text/csv",
        key="price_ask_download",
    )
    st.caption(
        "Cut to match is what it would take to reach the market price on that "
        "bottle. Google suggests is Google's own recommendation where it has "
        "one, which it publishes for a few hundred products rather than all of "
        "them. No figure here predicts extra orders: the feed carries no "
        "conversion tracking, so an order count would be invented."
        + (
            " Sold is what the shop actually sold of that wine in the last "
            f"{merchant_client.SALES_DAYS} days, from its own order book."
            if "bottles" in columns
            else ""
        )
    )


def _render_bargains(
    read: BenchmarkRead,
    money: str,
    named: dict[str, tuple[str, ...]] | None = None,
) -> None:
    """The wines already cheaper than everyone else, most wanted first.

    The other half of the same read, and the cheaper half to act on: these need
    nobody's agreement, only the ad budget pointed at them.
    """
    wines = merchant_client.bargains(read.prices, read.demand)
    if wines.empty:
        st.caption("Nothing in the feed is priced below the market.")
        return
    shown = _with_merchants(read.sales.against(wines), named)
    columns = _visible(
        shown,
        ("title", "merchant", "clicks", "bottles", "price", "benchmark", "gap"),
        read.demand.measured,
        read.sales.measured_against(wines),
    )
    st.dataframe(
        _formatted(shown, money)[columns].rename(
            columns={
                "title": "Wine",
                "merchant": "Merchant",
                "clicks": "Clicks 30d",
                "bottles": f"Sold {merchant_client.SALES_DAYS}d",
                "price": "Our price",
                "benchmark": "Market",
                "gap": "Gap",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download these",
        data=shown[columns].to_csv(index=False).encode("utf-8"),
        file_name=f"cheaper-than-market-{merchant_client.as_of()}.csv",
        mime="text/csv",
        key="price_bargains_download",
    )
    st.caption(
        "Cheaper here than the median merchant. These are what the ad budget "
        "can be pointed at without asking anybody to change a price"
        + (
            f", most clicked on in the last {merchant_client.DEMAND_DAYS} days "
            "first."
            if read.demand.measured
            else ", ordered by how far under the market they are. "
            + _demand_note(read.demand)
        )
    )


def _band_pictures(bands: pd.DataFrame, merchant: str) -> None:
    """The two figures as a picture, which is the form a merchant reads.

    A wine merchant is not going to be argued out of a price by a column called
    bottles per 100 clicks. The same numbers as a coloured ring and a row of
    bars make the case at a glance: this much of what Google could compare is
    red, and the red is the part that is not selling.
    """
    slices = bands[bands["listings"] > 0]
    if slices.empty:
        return
    colours = {
        band: merchant_letter.BAND_COLOURS[index]
        for index, band in enumerate(bands["band"].astype(str))
    }
    left, right = st.columns(2)
    with left:
        # A pie of four bands still asks the eye to compare angles; the band
        # a shop should worry about is the one with the most wine in it, and a
        # bar sorted by size names that band without asking anyone to guess a
        # slice's share by looking at it. Ascending order, undrawn-first: a
        # horizontal bar chart stacks its first row at the bottom, so the
        # smallest band ends up there and the one to act on lands at the top.
        ranked_bands = slices.sort_values("listings", ascending=True)
        by_band = px.bar(
            ranked_bands,
            x="listings",
            y=ranked_bands["band"].astype(str),
            orientation="h",
            color=ranked_bands["band"].astype(str),
            color_discrete_map=colours,
            # Kept short: the text is drawn larger than it used to be, and a
            # title long enough to be clipped is worse than a terse one.
            title="Wines by price against the market",
            text=ranked_bands["listings"].map(lambda value: f"{int(value):,}"),
        )
        by_band.update_traces(
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:,} wines<extra></extra>",
        )
        by_band.update_layout(
            margin=dict(t=54, b=0, l=0, r=48),
            showlegend=False,
            xaxis_title="",
            yaxis_title="",
        )
        theme.plot(by_band, width="stretch")
    rated = bands[bands["per_100_clicks"].notna()]
    if rated.empty:
        return
    with right:
        bars = px.bar(
            rated,
            x="per_100_clicks",
            y=rated["band"].astype(str),
            orientation="h",
            color=rated["band"].astype(str),
            color_discrete_map=colours,
            title="Bottles sold per 100 shoppers",
            text=rated["per_100_clicks"].map(lambda value: f"{value:.0f}"),
        )
        bars.update_layout(
            margin=dict(t=54, b=0, l=0, r=0),
            showlegend=False,
            xaxis_title="",
            yaxis_title="",
            yaxis=dict(autorange="reversed"),
        )
        theme.plot(bars, width="stretch")


# How many wines the price ladder draws. A merchant reads a page of bottles and
# argues with it; a hundred rows is a spreadsheet they close.
_LADDER_WINES = 20

def _price_sales_scatter(points: pd.DataFrame, merchant: str, money: str) -> None:
    """Every clicked wine as a dot: what it costs against the market, what it sold.

    The chart a merchant asked for. The bands beside it already say that keener
    prices sell more, but a band is four numbers and can be dismissed as our
    grouping; a dot per bottle is their own catalogue, and the slope through it
    is the argument in one line.
    """
    rho, sampled = merchant_client.price_sales_correlation(points)
    # The money is written into the hover rather than formatted by plotly, which
    # would need a currency symbol hard-coded into the template.
    plotted = points.assign(
        ours=points["price"].map(lambda value: _money(value, money)),
        market=points["benchmark"].map(lambda value: _money(value, money)),
    )
    figure = px.scatter(
        plotted,
        x=plotted["gap"] * 100,
        y="per_100_clicks",
        size="clicks",
        color="band",
        color_discrete_map={
            band: merchant_letter.BAND_COLOURS[index]
            for index, band in enumerate(merchant_client.BAND_NAMES)
        },
        custom_data=["title", "ours", "market", "clicks", "bottles"],
        title=f"{merchant}: price against the market, and sales",
        labels={"x": "", "per_100_clicks": ""},
    )
    figure.update_traces(
        hovertemplate=(
            "%{customdata[0]}<br>Our price %{customdata[1]} · "
            "market %{customdata[2]}<br>%{x:.0f}% against the market<br>"
            "%{customdata[3]} clicks, %{customdata[4]} bottles"
            "<extra></extra>"
        )
    )
    # The market itself, so a dot's side of the line is readable without doing
    # arithmetic on the axis.
    figure.add_vline(x=0, line_dash="dash", line_color=theme_tokens.INK["3"])
    # Only where the coefficient is quotable. A fitted line through nine dots
    # looks as confident as one through ninety, and drawn beside a figure that
    # says "not enough wines" it is the chart contradicting its own caption.
    fit = (
        _least_squares(points["gap"] * 100, points["per_100_clicks"])
        if rho is not None
        else None
    )
    if fit is not None:
        figure.add_trace(
            go.Scatter(
                x=fit[0],
                y=fit[1],
                mode="lines",
                name="Trend",
                # 2px, the ceiling every line on the dashboard is held to
                # (docs/assumptions/5A.md) - was 3px.
                line=dict(color=theme_tokens.INK["1"], width=2),
                hoverinfo="skip",
            )
        )
    figure.update_layout(
        margin=dict(t=56, b=48, l=8, r=8),
        xaxis_title="Percent against the market price",
        yaxis_title=f"Bottles sold per 100 shoppers ({merchant_client.SALES_DAYS}d)",
        legend_title_text="",
    )
    theme.plot(figure, width="stretch")

    left, right = st.columns(2)
    left.metric(
        "Wines plotted",
        f"{len(points):,}",
        help=(
            "Wines with a Google benchmark and at least "
            f"{merchant_client.SCATTER_MIN_CLICKS} clicks, so each dot is a rate "
            "rather than a coincidence."
        ),
    )
    right.metric(
        "Correlation, price against sales",
        "not enough wines" if rho is None else f"{rho:+.2f}",
        help=(
            "Spearman correlation of the gap to the market against bottles per "
            "100 shoppers. Negative means the more expensive a wine is, the less "
            "of it sells."
        ),
    )
    if rho is not None and rho < 0:
        st.markdown(
            f"**Across {sampled:,} of {merchant}'s own wines, the more a bottle "
            f"is priced above the market, the fewer of it sells ({rho:+.2f}).** "
            "Same shop, same shoppers, their own order book."
        )
    st.caption(
        f"One dot per wine, sized by how many shoppers chose it. Prices are "
        f"against Google's benchmark for the same bottle; sales are the shop's "
        f"own paid orders over {merchant_client.SALES_DAYS} days against "
        f"{merchant_client.DEMAND_DAYS} days of clicks. It is a correlation and "
        "not an experiment: a keenly priced wine may also be a wine people want, "
        "and the way to separate the two is to try the market price on a few of "
        "these bottles and read this chart again."
    )
    st.download_button(
        "Download the wines behind this chart",
        data=points.to_csv(index=False).encode("utf-8"),
        file_name=f"price-vs-sales-{merchant_client.as_of()}.csv",
        mime="text/csv",
        key="price_scatter_download",
        help="The same dots as a spreadsheet, to send with the chart.",
    )


def _least_squares(x: pd.Series, y: pd.Series) -> tuple[list[float], list[float]] | None:
    """The straight line through the dots, as two endpoints to draw.

    Fitted here rather than by ``px.scatter(trendline="ols")``, which needs
    statsmodels; this is two points from numpy and one fewer dependency in the
    deployment.
    """
    left = pd.to_numeric(x, errors="coerce")
    right = pd.to_numeric(y, errors="coerce")
    both = pd.concat([left, right], axis=1).dropna()
    if len(both) < 3 or both.iloc[:, 0].nunique() < 2:
        return None
    slope, intercept = np.polyfit(both.iloc[:, 0], both.iloc[:, 1], 1)
    ends = [float(both.iloc[:, 0].min()), float(both.iloc[:, 0].max())]
    return ends, [float(slope * end + intercept) for end in ends]


def _distinct_labels(titles: pd.Series) -> pd.Series:
    """Shortened wine names, kept different from each other.

    A row of the ladder is a category, and plotly draws two identical
    categories on one line: two vintages of the same wine agree for the first
    forty-six characters, so truncation alone would pile their prices on top of
    each other and show nineteen wines in a chart claiming twenty.
    """
    short = titles.fillna("").astype(str).str.slice(0, 46)
    seen: dict[str, int] = {}
    out = []
    for label in short:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return pd.Series(out, index=titles.index)


def _price_ladder(points: pd.DataFrame, merchant: str, money: str) -> None:
    """Their price beside the market's, wine by wine, most-clicked first.

    The scatter makes the general case; this one names bottles. A merchant who
    will not discuss a catalogue will discuss the twenty wines their own
    shoppers looked at most, with the gap drawn as the distance between two dots.
    """
    ladder = points.sort_values("clicks", ascending=False).head(_LADDER_WINES)
    if ladder.empty:
        return
    ladder = ladder.iloc[::-1]
    labels = _distinct_labels(ladder["title"])
    figure = go.Figure()
    for row in range(len(ladder)):
        figure.add_trace(
            go.Scatter(
                x=[
                    float(ladder["benchmark"].iloc[row]),
                    float(ladder["price"].iloc[row]),
                ],
                y=[labels.iloc[row], labels.iloc[row]],
                mode="lines",
                line=dict(
                    color=theme_tokens.STATUS["crit"][0]
                    if float(ladder["gap"].iloc[row]) > 0
                    else theme_tokens.STATUS["good"][0],
                    # 2px, the ceiling every line on the dashboard is held to
                    # (docs/assumptions/5A.md) - was 3px.
                    width=2,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=ladder["benchmark"],
            y=labels,
            mode="markers",
            name="The market",
            marker=dict(size=11, color=theme_tokens.INK["3"], symbol="diamond"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ladder["price"],
            y=labels,
            mode="markers",
            name="Our price",
            marker=dict(size=13, color=theme_tokens.STATUS["info"][0]),
        )
    )
    figure.update_layout(
        title=f"{merchant}: the {len(ladder)} most looked-at wines",
        margin=dict(t=56, b=40, l=8, r=8),
        height=max(360, 26 * len(ladder) + 130),
        xaxis_title=f"Price per bottle ({money or 'USD'})",
        yaxis_title="",
        legend_title_text="",
    )
    theme.plot(figure, width="stretch")
    st.caption(
        "Ordered by the shoppers who chose the bottle, so the wines at the top "
        "are the ones a price change would be felt on. A red line is the amount "
        "a shopper saves by buying the same wine somewhere else."
    )


def _merchant_letter(bands: pd.DataFrame, merchant: str, sales_days: int) -> None:
    """The same picture as a page to send, rather than a screen to describe."""
    rows = [
        merchant_letter.Band(
            str(row["band"]),
            int(row["listings"]),
            int(row["clicks"]),
            int(row["bottles"]),
            None if pd.isna(row["per_100_clicks"]) else float(row["per_100_clicks"]),
        )
        for _, row in bands.iterrows()
    ]
    st.download_button(
        f"Download a page for {merchant}",
        data=merchant_letter.one_pager(
            merchant,
            rows,
            sales_days=sales_days,
            demand_days=merchant_client.DEMAND_DAYS,
        ).encode("utf-8"),
        file_name=merchant_letter.filename(merchant),
        mime="text/html",
        key="price_evidence_letter",
        help=(
            "One page, their name on it, no jargon: the ring, the bars and one "
            "sentence. Opens in any browser and prints to a PDF."
        ),
    )


def _ad_ledger_table(frame: pd.DataFrame, money: str, spent: str = "") -> pd.DataFrame:
    """The per-wine ledger as a reader sees it, money and gaps formatted.

    Two currencies, because there are two: Google bills the account in its own,
    and the price beside it is the feed's. Where they differ the columns say so
    rather than being added up under one symbol.
    """
    return frame.assign(
        spend=frame["spend"].map(lambda value: _money(value, spent or money)),
        sold_revenue=frame["sold_revenue"].map(
            lambda value: "\u2014" if pd.isna(value) else _money(value, money)
        ),
        price=frame["price"].map(
            lambda value: "\u2014" if pd.isna(value) else _money(value, money)
        ),
        benchmark=frame["benchmark"].map(
            lambda value: "\u2014" if pd.isna(value) else _money(value, money)
        ),
        gap=frame["gap"].map(
            lambda value: "\u2014" if pd.isna(value) else f"{value:+.0%}"
        ),
        clicks=frame["clicks"].map(lambda value: f"{int(value):,}"),
        impressions=frame["impressions"].map(lambda value: f"{int(value):,}"),
        bottles=frame["bottles"].map(
            lambda value: "\u2014" if pd.isna(value) else f"{int(value):,}"
        ),
    ).rename(
        columns={
            "offer": "Offer",
            "title": "Wine",
            "merchant": "Merchant",
            "spend": "Ad spend",
            "clicks": "Clicks",
            "impressions": "Shown",
            "bottles": f"Bottles {merchant_client.SALES_DAYS}d",
            "sold_revenue": "Revenue",
            "price": "Our price",
            "benchmark": "Market",
            "gap": "Gap",
        }
    )


def _ad_claim(
    label: str, claim: str, wines: pd.DataFrame, money: str, key: str, spent: str = ""
) -> None:
    """One claim with the wines behind it folded up underneath.

    Every sentence this panel makes is opened by clicking it: the argument it is
    part of is with the person who runs the campaign, and a summary he cannot
    drill into is a summary he is right to distrust.

    Kept for the printable report as well as drawn: the tiles this claim
    explains already go into it, and figures are read furthest from their
    caption once they are on paper.
    """
    _report(TAB_BUSINESS).note("Ad spend per wine", claim)
    st.markdown(_unmathed(claim))
    if wines.empty:
        return
    with st.expander(f"{label} - the {len(wines):,} wines behind this", expanded=False):
        st.dataframe(
            _ad_ledger_table(wines.head(_AD_LEDGER_ROWS), money, spent),
            width="stretch",
            hide_index=True,
        )
        if len(wines) > _AD_LEDGER_ROWS:
            st.caption(
                f"The {_AD_LEDGER_ROWS} costliest of {len(wines):,}; the file "
                "below holds all of them."
            )
        st.download_button(
            "Download these",
            data=wines.to_csv(index=False).encode("utf-8"),
            file_name=f"ads-{key}-{merchant_client.as_of()}.csv",
            mime="text/csv",
            key=f"ads_claim_{key}",
        )


def _ad_ledger(
    read: BenchmarkRead, named: dict | None, merchant: str
) -> tuple[pd.DataFrame, AdProducts]:
    """The per-wine ad ledger, cut to one merchant's wines when one is picked.

    The picker above promises every figure below is that merchant's alone, and
    ad spend is no exception: showing a merchant somebody else's wasted spend
    would be the panel arguing with the wrong person.

    Whose wine an offer is comes from the catalogue, which is only asked about
    the offers Merchant Center benchmarks, so one merchant's tab is that
    merchant's benchmarked wines - said in a caption rather than left to be
    inferred from a total that does not match the shop's.
    """
    ads = _ad_products(merchant_client.SALES_DAYS)
    frame = ads.frame
    if merchant != _EVERY_MERCHANT and not frame.empty:
        frame = frame[frame["offer"].isin(set(read.prices.offers["offer"]))]
    return (
        ads_evidence.ledger(frame, read.prices, read.sales, named),
        ads,
    )


def _ad_window_note(ads: AdProducts, days: int = merchant_client.SALES_DAYS) -> None:
    """How much of the window the product report actually holds.

    The tab asks for a quarter of spend and puts it beside a quarter of the
    shop's orders, but the Shopping product report is transferred separately from
    the campaign one and is routinely switched on later: a fortnight of spend
    against a quarter of orders reads as a return the ads never earned. Said only
    when it is short, since a whole window needs no caveat.
    """
    if ads.frame.empty or ads.history_start is None:
        return
    wanted = ads_client.window_first_day(days)
    if ads.history_start <= wanted:
        return
    held = (_dt.date.today() - _dt.timedelta(days=1) - ads.history_start).days + 1
    st.caption(
        f"**Google's product report only goes back to {ads.history_start}**, so "
        f"the spend here is {max(held, 0)} days of it, not {days} - while the "
        "bottles and revenue beside it are the whole window. The return per unit "
        "spent is therefore flattered; the spend, the clicks and which wines took "
        "the money are unaffected."
    )


def _ad_money_notes(ads: AdProducts, money: str, merchant: str) -> None:
    """What the figures above are not: one currency, one merchant, one feed."""
    _ad_window_note(ads)
    if ads.unread_accounts:
        st.caption(
            f"{ads.unread_accounts} ad account"
            + ("" if ads.unread_accounts == 1 else "s")
            + " in this dataset could not be read - most often a transfer that "
            "does not carry the Shopping product report - so the spend below is "
            "the accounts that could, and is short by whatever they spent."
        )
    if ads.other_currencies:
        st.caption(
            "Spend is the "
            + f"{ads.currency} accounts only; the dataset also holds "
            + ", ".join(ads.other_currencies)
            + " accounts, which are left out rather than added to them."
        )
    if ads.currency and money and ads.currency != money:
        st.caption(
            f"Google bills this account in {ads.currency} and the feed prices "
            f"in {money}, so spend and price are not the same money and the "
            "return per unit spent is only as good as the rate between them."
        )
    if merchant != _EVERY_MERCHANT:
        st.caption(
            f"Only {merchant}'s wines that Google publishes a benchmark for: "
            "whose listing an offer is comes from the catalogue, which is asked "
            "about the benchmarked offers, so spend on their other wines is "
            "outside this tab rather than nil. Every merchant shows all of it."
        )


def _render_ad_money(
    read: BenchmarkRead, money: str, named: dict | None, merchant: str
) -> None:
    """Where the ad budget went, wine by wine, against price and against sales.

    The panel's other tabs ask a merchant to change a price. This one asks the
    account itself a cheaper question: of the money already spent, how much went
    to bottles that nobody bought - which needs nobody's agreement to change.
    """
    frame, ads = _ad_ledger(read, named, merchant)
    if frame.empty:
        _no_ad_spend(ads, merchant)
        return
    spent = ads.currency or money
    if not ads_evidence.sold_known(frame):
        st.caption(
            "The shop's own sales could not be put beside these wines - either "
            "the order book could not be read, or it holds none of these offer "
            "ids - so what each one sold is unknown rather than none: the spend "
            "and the clicks below stand, and every figure about what the money "
            "bought is left out rather than shown as nil."
        )
    split = ads_evidence.spend_split(frame)
    tiles = st.columns(3)
    _tile(
        tiles[0],
        TAB_BUSINESS,
        "Ad spend per wine",
        f"Spend {merchant_client.SALES_DAYS}d",
        _money(float(frame["spend"].sum()), spent),
    )
    nothing = split[split["outcome"] == ads_evidence.NOTHING]
    _tile(
        tiles[1],
        TAB_BUSINESS,
        "Ad spend per wine",
        "On wines that sold nothing",
        f"{float(nothing['spend'].iloc[0]) / float(frame['spend'].sum()):.0%}"
        if not nothing.empty and float(frame["spend"].sum()) > 0
        else "\u2014",
    )
    _tile(
        tiles[2],
        TAB_BUSINESS,
        "Ad spend per wine",
        "Wines advertised",
        f"{len(frame):,}",
    )
    _ad_pictures(frame, spent)
    for tag, claim in ads_evidence.verdicts(frame, spent, money):
        wines, label = _ad_claim_wines(frame, tag)
        _ad_claim(label, claim, wines, money, tag, spent)
    stop = ads_evidence.waste(frame)
    if not stop.empty:
        _ad_claim(
            "Clicked, expensive and unsold",
            f"The {len(stop):,} wines to stop paying for first: more than "
            f"{merchant_client.DEAR_GAP:.0%} above the market, clicked, and no "
            f"bottle sold in {merchant_client.SALES_DAYS} days - "
            f"{_money(float(stop['spend'].sum()), spent)} of spend that needs "
            "nobody's agreement to stop.",
            stop,
            money,
            "clicked-expensive-unsold",
            spent,
        )
    _ad_money_notes(ads, money, merchant)
    st.download_button(
        f"Download every advertised wine ({len(frame):,})",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=f"ads-per-wine-{merchant_client.as_of()}.csv",
        mime="text/csv",
        key="ads_ledger_download",
        help=(
            "One row per wine: what Google charged for it, what it was shown "
            "and clicked, what it sold, and its price against the market."
        ),
    )
    st.caption(
        f"Google Ads' own product report, last {merchant_client.SALES_DAYS} days, "
        "beside the same window of the shop's orders. Bottles are every sale of "
        "that wine in the window rather than sales an ad can be shown to have "
        "caused - Google's own attribution records a fraction of the shop's "
        "orders - so a return here is the return on advertising a wine, not the "
        "sales advertising created."
    )
    _ad_advice(frame, spent, money)


def _ad_advice(frame: pd.DataFrame, spent: str, money: str) -> None:
    """What to change on Monday, under everything that argues for it.

    Last, and deliberately: a recommendation above its evidence is an opinion,
    and the reader of this panel runs the campaign it is about.
    """
    said = ads_evidence.advice(frame, spent, money)
    if not said:
        return
    st.markdown("#### What to do about it")
    for index, line in enumerate(said, start=1):
        _report(TAB_BUSINESS).note("Ad spend per wine", f"{index}. {line}")
        st.markdown(_unmathed(f"{index}. {line}"))


def _no_ad_spend(ads: AdProducts, merchant: str) -> None:
    """Why an ads tab is empty, which is not always that Ads is unconfigured.

    Four different emptinesses used to share one caption telling the reader to
    set environment variables: a merchant whose wines took no money would be
    told to configure an account that is already configured and spending, and a
    report that could not be read would be reported as an account at rest.

    Settings are asked about before the read is: a name the Ads client rejects
    fails the read like an absent table does, and sending somebody hunting
    BigQuery permissions over a typo is the worst of the four to get wrong.
    """
    if not _ads_configured():
        st.caption(
            "Per-wine ad spend comes from Google Ads' Shopping product report in "
            "BigQuery. Set GOOGLE_ADS_BQ_PROJECT and GOOGLE_ADS_BQ_DATASET - and "
            "check what they are set to, since a value the Ads client rejects "
            "reads the same from here as one nobody set - and the transfer will "
            "need the Shopping product stats table."
        )
    elif not ads.read:
        st.caption(
            "Google Ads' Shopping product report could not be read, so what each "
            "wine cost is unknown rather than nil. The dataset is configured; "
            "either the transfer is not carrying the Shopping product stats "
            "table, or the credential cannot see it."
        )
    elif merchant != _EVERY_MERCHANT:
        st.caption(
            f"None of {merchant}'s benchmarked wines took ad money in the last "
            f"{merchant_client.SALES_DAYS} days. Every merchant shows the whole "
            "account, including the wines Google publishes no benchmark for."
        )
    else:
        st.caption(
            f"Google Ads is set up and no wine took ad money in the last "
            f"{merchant_client.SALES_DAYS} days. If the account is spending, its "
            "BigQuery transfer is probably not carrying the Shopping product "
            "stats table."
        )


# How many rows a folded-up claim shows before it becomes a file.
_AD_LEDGER_ROWS = 50


def _ad_claim_wines(frame: pd.DataFrame, tag: str) -> tuple[pd.DataFrame, str]:
    """The wines behind one of ``ads_evidence.verdicts``' claims, by which it is.

    Chosen by the claim's own name rather than by its place in the list: any of
    the claims can be left out, and a claim opening onto the wines that happened
    to be in its position is a table that argues with its own sentence.
    """
    by_spend = frame.sort_values("spend", ascending=False)
    if tag == ads_evidence.WASTED:
        return by_spend[by_spend["bottles"] <= 0], "Sold nothing"
    if tag == ads_evidence.BY_PRICE:
        return by_spend[by_spend["gap"].notna()], "Priced against the market"
    return by_spend[by_spend["gap"].isna()], "No benchmark"


def _ad_pictures(frame: pd.DataFrame, money: str) -> None:
    """The two figures as pictures: where the money went, and what came back."""
    split = ads_evidence.spend_split(frame)
    bands = ads_evidence.by_band(frame)
    left, right = st.columns(2)
    if not split.empty and float(split["spend"].sum()) > 0:
        with left:
            # Two slices is still a pie a reader has to measure by angle; the
            # question this picture answers ("how much went to wine that
            # never sold") is a single comparison, which a bar states directly
            # instead of asking for one. Ascending, undrawn-first, so the
            # larger outcome - the one worth a sentence - lands at the top.
            ranked_split = split.sort_values("spend", ascending=True)
            by_outcome = px.bar(
                ranked_split,
                x="spend",
                y="outcome",
                orientation="h",
                color="outcome",
                color_discrete_map=ads_evidence.SPLIT_COLOURS,
                title=f"Ad spend, last {merchant_client.SALES_DAYS} days",
                text=ranked_split["spend"].map(lambda value: _money(value, money)),
            )
            by_outcome.update_traces(
                textposition="outside",
                cliponaxis=False,
                # A written sentence per bar rather than a template over
                # ``customdata``: the money is formatted here anyway, to carry
                # the currency Google billed rather than Plotly's default.
                hovertext=ads_evidence.split_hovers(ranked_split, money),
                hovertemplate="%{hovertext}<extra></extra>",
            )
            by_outcome.update_layout(
                margin=dict(t=54, b=0, l=0, r=48),
                showlegend=False,
                xaxis_title="",
                yaxis_title="",
            )
            theme.plot(by_outcome, width="stretch")
    rated = bands[bands["per_dollar"].notna()]
    if rated.empty:
        return
    with right:
        colours = {
            band: merchant_letter.BAND_COLOURS[index]
            for index, band in enumerate(merchant_client.BAND_NAMES)
        }
        bars = px.bar(
            rated,
            x="per_dollar",
            y=rated["band"].astype(str),
            orientation="h",
            color=rated["band"].astype(str),
            color_discrete_map=colours,
            # Short enough not to be clipped in half a row at the larger size.
            title=f"Revenue per {_money(1, money)} of ads",
            # To the cent while the return is small: a band giving back forty
            # cents a dollar labelled 0 reads as a band that sold nothing.
            text=rated["per_dollar"].map(
                lambda value: f"{value:,.0f}" if value >= 10 else f"{value:,.2f}"
            ),
        )
        bars.update_layout(
            margin=dict(t=54, b=0, l=0, r=0),
            showlegend=False,
            xaxis_title="",
            yaxis_title="",
            yaxis=dict(autorange="reversed"),
        )
        theme.plot(bars, width="stretch")


def _render_most_clicked(
    read: BenchmarkRead, money: str, named: dict | None, merchant: str
) -> None:
    """The wines shoppers chose most, and what each one's price did next.

    Clicks are demand the shop did not have to earn: a shopper on a Shopping row
    has picked this bottle out of a dozen of the same wine. What happened after
    the click - a bottle sold, or nothing - is the whole argument, per wine, with
    the price gap that goes with it.
    """
    frame, ads = _ad_ledger(read, named, merchant)
    if frame.empty:
        _no_ad_spend(ads, merchant)
        return
    wanted = ads_evidence.most_clicked(frame, _MOST_CLICKED)
    if wanted.empty:
        st.caption("Nothing was clicked in the window.")
        return
    st.dataframe(
        _ad_ledger_table(wanted, money, ads.currency),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download the most clicked",
        data=wanted.to_csv(index=False).encode("utf-8"),
        file_name=f"ads-most-clicked-{merchant_client.as_of()}.csv",
        mime="text/csv",
        key="ads_clicked_download",
    )
    st.caption(
        f"The {len(wanted)} most clicked advertised wines of the last "
        f"{merchant_client.SALES_DAYS} days, with the bottles each sold in the "
        "same window and its price against the market. A wine with clicks, no "
        "bottles and a gap well above the market is the case this panel is "
        "making; a wine with clicks, no bottles and a keen price is a different "
        "problem, and worth reading as one."
    )
    _ad_money_notes(ads, money, merchant)


# How many wines the most-clicked table names. Enough to see the pattern without
# becoming the ledger, which is a download rather than a table.
_MOST_CLICKED = 40

# How the sale-price test is sized: in tens, and never more than a few hundred
# wines, which is as large as a price test can be and still be read afterwards.
_SALE_FEED_STEP = 10
_SALE_FEED_MAX = 500


def _render_sale_prices(
    read: BenchmarkRead, money: str, named: dict | None, merchant: str
) -> None:
    """A supplemental feed that tries the market price without changing a price.

    The panel's ask - drop the price - needs a merchant to agree, a shop update
    and a wait. Merchant Center takes a ``sale_price`` per offer in a
    supplemental feed instead, so the same wines can be tried at the market
    price for a fortnight and the result read here.
    """
    frame, ads = _ad_ledger(read, named, merchant)
    spent_known = not frame.empty
    if frame.empty:
        # Without ad spend there is still a feed to make, from the benchmark
        # alone: the wines to try are the expensive ones, spend only orders them.
        frame = ads_evidence.ledger(
            pd.DataFrame(
                {
                    "offer": read.prices.offers["offer"],
                    "spend": 0.0,
                    "clicks": 0,
                    "impressions": 0,
                    "ad_conversions": 0.0,
                }
            ),
            read.prices,
            read.sales,
            named,
        )
    feed = ads_evidence.sale_price_feed(frame)
    if not feed.empty and not spent_known:
        # Every spend in this frame is nought, so ordering by it would be the
        # feed's own order presented as a ranking: order by what each wine is
        # over the market instead, and say that is what the order is.
        feed = (
            feed.assign(over=feed["price"] - feed["sale_price"])
            .sort_values("over", ascending=False)
            .drop(columns="over")
            .reset_index(drop=True)
        )
        st.caption(
            (
                "Google Ads' product report could not be read, so what each of "
                "these wines costs in ad spend is unknown: "
                if not ads.read
                else "No ad spend is recorded against these wines, so: "
            )
            + "the list below is ordered by how far each one is above the "
            "market rather than by what it cost, and the ad spend column is nil "
            "because it is unknown rather than because the wine is free to "
            "advertise."
        )
    if feed.empty:
        st.caption(
            "Nothing here is priced far enough above the market to be worth "
            "putting on sale."
        )
        return
    # A slider needs two ends to it: a handful of wines is the whole test, and
    # asking how many of five to try is a question with one answer.
    if len(feed) > _SALE_FEED_STEP:
        count = st.slider(
            "How many wines to try",
            min_value=_SALE_FEED_STEP,
            max_value=min(_SALE_FEED_MAX, len(feed)),
            value=min(50, len(feed)),
            step=_SALE_FEED_STEP,
            key="ads_sale_feed_count",
            help=(
                "The costliest first, so a small test is a test of the wines the "
                "budget is actually going to."
                if spent_known
                else "Furthest above the market first: without the ad report "
                "there is no spend to rank them by."
            ),
        )
    else:
        count = len(feed)
    trying = feed.head(count)
    st.dataframe(
        trying.assign(
            price=trying["price"].map(lambda value: _money(value, money)),
            sale_price=trying["sale_price"].map(lambda value: _money(value, money)),
            spend=trying["spend"].map(
                lambda value: _money(value, ads.currency or money)
            ),
            clicks=trying["clicks"].map(lambda value: f"{int(value):,}"),
        ).rename(
            columns={
                "id": "Offer",
                "title": "Wine",
                "price": "Our price",
                "sale_price": "Suggested sale price",
                "clicks": "Clicks",
                "spend": "Ad spend",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        f"Download a supplemental feed ({len(trying):,} wines)",
        # Two columns and no more: a supplemental feed overrides every attribute
        # it carries, so a ``price`` column in it would pin the catalogue price
        # to whatever it was the day the file was downloaded - the shop could
        # reprice the wine and Google would keep showing the old figure until
        # somebody deleted the feed. The table above still shows the price,
        # which is what it is for.
        data=trying[["id", "sale_price"]]
        .assign(
            sale_price=trying["sale_price"].map(
                lambda value: f"{value:.2f} {money or 'USD'}"
            ),
        )
        .to_csv(index=False)
        .encode("utf-8"),
        file_name=f"sale-price-feed-{merchant_client.as_of()}.csv",
        mime="text/csv",
        key="ads_sale_feed_download",
        help=(
            "Merchant Center, Data sources, add a supplemental feed and upload "
            "this: it sets sale_price on these offers and leaves everything "
            "else alone."
        ),
    )
    st.caption(
        "The suggested price is Google's benchmark itself - the market, not "
        "under it. A sale price shows as a struck-through price in Shopping and "
        "does not touch the shop's own prices, so it can be reversed by deleting "
        "the feed. Which wines to try is a judgement: "
        + (
            "these are the ones the ad budget is already going to at a price the "
            "market is not paying."
            if spent_known
            else "these are the ones furthest above the market, which is all "
            "that can be said without the ad report."
        )
    )
    if spent_known:
        # The spend column here is the same short window as the ledger's, and it
        # is what the list is ordered by.
        _ad_window_note(ads)


def _render_evidence(read: BenchmarkRead, merchant: str = _EVERY_MERCHANT) -> None:
    """What the shop's own sales say about the price it charged.

    The tab to send a merchant. Everywhere else the panel argues from Google's
    benchmark, which a merchant can dismiss as somebody else's number; this
    argues from the merchant's own bottles: the same shop, the same shoppers,
    and what a keener price did to how many of them bought.
    """
    sales, demand = read.sales, read.demand
    if not sales.read:
        st.caption(
            "The order book could not be read, so what these prices sold is "
            "unknown rather than nothing."
        )
        return
    if not sales.measured_against(read.prices.offers):
        # A join that matched nothing and a shop that sold nothing leave the
        # same empty frame, and printing a zero against every band would be the
        # panel telling a merchant its wines do not sell on our own bad match.
        # Judged on the wines on screen, so a merchant filter that matched none
        # of them says so rather than borrowing the whole shop's bottles.
        st.caption(
            "No bottles in the order book match these listings, so there is "
            "nothing to set beside the prices - which is not the same as "
            "nothing having sold."
        )
        return
    if not demand.measured:
        st.caption(
            "Sales per click need both halves, and " + _demand_note(demand).lower()
        )
        return
    bands = merchant_client.price_bands(read.prices, demand, sales)
    if bands.empty:
        st.caption("Nothing has both a benchmark and a click to compare.")
        return
    named = "The shop" if merchant == _EVERY_MERCHANT else merchant
    _band_pictures(bands, named)

    # Per wine rather than per band: the bands are the summary, and a merchant
    # who disputes our grouping can be shown their own bottles instead.
    points = merchant_client.wine_points(read.prices, demand, sales)
    if points.empty:
        st.caption(
            "No single wine has both a benchmark and enough clicks "
            f"({merchant_client.SCATTER_MIN_CLICKS}) for its own sales rate, so "
            "the bands above are as fine as this evidence goes."
        )
    else:
        _price_sales_scatter(points, named, read.prices.currency)
        st.divider()
        _price_ladder(points, named, read.prices.currency)
        st.divider()

    shown = bands.assign(
        per_100_clicks=bands["per_100_clicks"].map(
            lambda value: "\u2014" if pd.isna(value) else f"{value:.0f}"
        ),
        listings=bands["listings"].map(lambda value: f"{int(value):,}"),
        clicks=bands["clicks"].map(lambda value: f"{int(value):,}"),
        bottles=bands["bottles"].map(lambda value: f"{int(value):,}"),
    )
    st.dataframe(
        shown.rename(
            columns={
                "band": "Against the market",
                "listings": "Wines",
                "clicks": f"Clicks {merchant_client.DEMAND_DAYS}d",
                "bottles": f"Bottles sold {sales.days}d",
                "per_100_clicks": "Bottles per 100 clicks",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download the evidence",
        data=bands.to_csv(index=False).encode("utf-8"),
        file_name=f"price-and-sales-{merchant_client.as_of()}.csv",
        mime="text/csv",
        key="price_evidence_download",
    )
    _merchant_letter(bands, named, sales.days)
    st.caption(
        f"Clicks are Google Shopping's last {merchant_client.DEMAND_DAYS} days; "
        f"bottles are what the shop sold in the last {sales.days} days, paid "
        "orders only. Bottles per 100 clicks is the comparison that survives the "
        "difference in size between the bands - a band with more wines in it "
        "does not sell more per shopper for being bigger. Merchant Center "
        "reports no conversions on this feed, so this is the shop's own order "
        "book rather than Google's attribution."
    )


def _render_price_benchmark() -> None:
    """How the shop's prices compare with everyone else selling the same wine.

    The order book cannot answer this: it holds what the shop charged, not what
    the shop next door charged for the same bottle. Google works that out across
    every merchant in Shopping and calls it a benchmark, and the gap to it is the
    difference between a product page that sells and one that is a price check
    for somebody else's shop.
    """
    section = "Price competitiveness"
    st.subheader(section)
    try:
        config = merchant_client.load_merchant_env()
    except merchant_client.MerchantConfigError as exc:
        st.caption(f"Price benchmarks are misconfigured: {exc}")
        return
    if config is None:
        st.caption(
            "Price benchmarks come from Merchant Center. Set GOOGLE_MERCHANT_ID "
            "to the account id, and add the dashboard's service account under "
            "Settings, People and access with read access."
        )
        return

    try:
        # The benchmark and the shop's own sales are unrelated reads - Merchant
        # Center against the order database - so they go out together rather
        # than one after another.
        with st.spinner("Reading Merchant Center's price benchmarks..."):
            parallel_read = _parallel(
                {
                    "benchmark": lambda: _price_benchmark_cached(
                        config.account, config.country
                    ),
                    "sales": _offer_sales,
                }
            )
    except merchant_client.MerchantConfigError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read the price benchmarks: {str(exc)[:200]}")
        return

    read = parallel_read["benchmark"]._replace(sales=parallel_read["sales"])
    prices, insights = read.prices, read.insights
    # Held before the merchant filter can empty it, so an empty filter is never
    # mistaken below for a feed with no benchmarks in it.
    whole_feed_counted = prices.counted
    filtered_to_nothing = False
    chosen = _EVERY_MERCHANT
    # Whose wine each offer is, read once for the whole catalogue so the same
    # names can both filter it and label the rows below.
    named = _offer_merchants(prices.offers)
    # Shops that have been switched off are dropped before anything is counted,
    # so "Every merchant" means every merchant still trading rather than every
    # merchant the feed remembers.
    active = _active_merchant_names()
    prices, named, set_aside = _trading_only(prices, named, active)
    if set_aside:
        read = read._replace(prices=prices)
        whole_feed_counted = prices.counted
    if named:
        merchants = sorted(
            {name for names in named.values() for name in names if name}
        )
        chosen = st.selectbox(
            "Merchant",
            [_EVERY_MERCHANT, *merchants],
            key="price_merchant",
            help=(
                "Every figure and file below is then that merchant's alone, "
                "which is what to send them."
            ),
        )
        prices = _one_merchant(prices, named, chosen)
        read = read._replace(prices=prices)
        filtered_to_nothing = chosen != _EVERY_MERCHANT and not prices.counted
        if filtered_to_nothing:
            st.caption(
                f"None of {chosen}'s wines has a benchmark: Google publishes "
                "one only where enough other merchants sell the same product."
            )
    if set_aside:
        st.caption(
            f"{set_aside:,} offer(s) left out: they belong only to shops that "
            f"are switched off. Every figure here is the {len(active)} trading "
            f"merchant(s) named in {_ACTIVE_MERCHANTS_ENV} (env var, or the "
            "vendor-panel default baked in when it is unset)."
        )
    elif active and not named:
        st.caption(
            f"{_ACTIVE_MERCHANTS_ENV} names a trading roster, but the "
            "catalogue could not be read to apply it, so every merchant is "
            "counted below."
        )
    elif active:
        st.caption(
            f"{_ACTIVE_MERCHANTS_ENV} names no shop this catalogue knows, so "
            "every merchant is counted below. Check the names match the feed."
        )
    st.caption(_store_metadata_provenance())
    money = prices.currency
    tiles = st.columns(3)
    _tile(
        tiles[0],
        TAB_BUSINESS,
        section,
        "More expensive than the market",
        f"{prices.dear_share:.0%}" if prices.counted else "\u2014",
    )
    _tile(
        tiles[1],
        TAB_BUSINESS,
        section,
        "Typical gap",
        f"{prices.median_gap:+.0%}" if prices.counted else "\u2014",
    )
    _tile(
        tiles[2],
        TAB_BUSINESS,
        section,
        "Priced products compared",
        f"{prices.counted:,}",
    )

    # Against the whole feed, not the merchant's slice of it: an empty filter
    # is already explained above, and telling the reader to change the feed's
    # country for it would send them after a setting that is not the matter.
    if not whole_feed_counted:
        st.caption(
            f"Read for {config.country}, the country the feed is taken to "
            "target. Benchmarks are published per country, so set "
            "GOOGLE_MERCHANT_COUNTRY if this feed targets another one."
        )

    if prices.counted:
        (
            ask_tab,
            bargain_tab,
            evidence_tab,
            ads_tab,
            clicked_tab,
            feed_tab,
            dear_tab,
            vivino_tab,
        ) = st.tabs(
            [
                f"Ask the merchants ({merchant_client.ASK_LIST})",
                "Cheaper than the market",
                "What price did to sales",
                "Where the ad money went",
                "Most clicked",
                "Try a sale price",
                "Most expensive bottles",
                "Their Vivino price",
            ]
        )
        with ask_tab:
            _render_ask_list(read, money, named)
        with bargain_tab:
            _render_bargains(read, money, named)
        with evidence_tab:
            _render_evidence(read, chosen)
        with ads_tab:
            _render_ad_money(read, money, named, chosen)
        with clicked_tab:
            _render_most_clicked(read, money, named, chosen)
        with feed_tab:
            _render_sale_prices(read, money, named, chosen)
        with vivino_tab:
            _render_vivino(chosen, picker=bool(named))
        with dear_tab:
            st.dataframe(
                prices.worst.head(_WORST_OFFERS)
                .assign(
                    price=lambda frame: frame["price"].map(
                        lambda value: _money(value, money)
                    ),
                    benchmark=lambda frame: frame["benchmark"].map(
                        lambda value: _money(value, money)
                    ),
                    gap=lambda frame: frame["gap"].map(lambda value: f"{value:+.0%}"),
                )[["title", "brand", "price", "benchmark", "gap"]]
                .rename(
                    columns={
                        "title": "Wine",
                        "brand": "Brand",
                        "price": "Our price",
                        "benchmark": "Market",
                        "gap": "Gap",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "The furthest above the market in percentage terms, whether or "
                "not anybody is looking at them. What to do about them is the "
                "first tab, which weighs the same gap by the shoppers it lost."
            )
    elif named:
        # The Vivino comparison needs no Google benchmark - it reads the
        # merchant's own catalogue - so it stays reachable for a merchant
        # none of whose wines Google can price. Without a merchant picker
        # above there is nobody to compare, so the tab stays away too.
        (vivino_tab,) = st.tabs(["Their Vivino price"])
        with vivino_tab:
            _render_vivino(chosen, picker=bool(named))

    if prices.other_currencies:
        st.caption(
            f"These are the {money.upper()} prices only; the feed also quotes "
            + ", ".join(code.upper() for code in prices.other_currencies)
            + ", which is set aside rather than compared against a benchmark in "
            "another currency."
        )
    st.caption(
        "Google's benchmark is the median price other merchants charge for the "
        f"same product, read from Merchant Center for {merchant_client.as_of()}. "
        "Products no other merchant sells have no benchmark and are left out "
        "rather than counted as competitive."
    )

    # A merchant filter that kept nothing is explained above, in terms of that
    # merchant; the verdicts would say it of the whole feed, which has
    # thousands of benchmarked wines in it.
    lines = [] if filtered_to_nothing else merchant_client.verdicts(
        prices, insights, read.demand
    )
    if not filtered_to_nothing:
        lines += merchant_client.sales_verdicts(prices, read.demand, read.sales)
    if lines:
        with st.expander("What the prices say", expanded=True):
            _said(TAB_BUSINESS, section, lines)


# Ad figures are a day old the moment they exist: the grain is a day, the last
# one is yesterday, and Google's transfer writes them once a day. Refreshing
# every quarter of an hour, which is what the order book needs, bought no
# freshness at all here and paid a full round of BigQuery jobs for it. Both
# entries are keyed on the date as well, so they roll over when the transfer
# does rather than at some arbitrary point mid-morning.
ADS_TTL_SECONDS = 6 * 3600
# Names, currencies and the day a transfer's history begins move about once
# ever, so they are held apart from the spend and for a day at a time; on the
# spend's cycle they cost two extra BigQuery jobs per account per refresh.
ADS_ACCOUNTS_TTL_SECONDS = 24 * 3600
# The widest window the panel offers, which is the only one read. `daily_stats`
# fetches twice what it is asked for so the previous period can be compared, so
# the widest option's rows contain every narrower option's, and `window` and
# `by_campaign` both slice by day in pandas - a click on the radio now redraws
# from the frame in hand instead of going back to BigQuery for a subset of what
# it already had.
ADS_WINDOW_DAYS = max(ads_client.LOOKBACK_WINDOWS)


class AdsAccount(NamedTuple):
    """One ad account in the dataset, and the things about it that never move."""

    customer_id: str
    name: str
    currency: str
    # The earliest day the transfer has loaded for this account.
    history_start: _dt.date | None


class AdsRead(NamedTuple):
    """Everything one pass over the Ads dataset yields."""

    stats: pd.DataFrame
    names: pd.DataFrame
    account: str
    currency: str
    # The earliest day the transfer has loaded, from the table itself rather than
    # from these rows: a paused account has no rows for days that are loaded.
    history_start: _dt.date | None
    # Accounts left out because they bill in some other currency, named so the
    # reader knows the total is not the whole dataset.
    other_currencies: list[str]


def _ads_config(project: str, dataset: str) -> ads_client.AdsConfig:
    """The Ads configuration, checked to be the one the cache key was cut from.

    The cached reads take the project and dataset as arguments and the credential
    from the environment, because a service account key has no business in a
    cache key. That leaves one way for the two to disagree - the environment
    changing between the key being cut and the read running - which is caught
    here rather than answered with figures from the wrong dataset.
    """
    config = ads_client.load_ads_env()
    if config is None:  # pragma: no cover - the caller checks first
        raise ads_client.AdsConfigError("Google Ads figures are not configured.")
    if (config.project, config.dataset) != (project, dataset):
        raise ads_client.AdsConfigError(
            "The Google Ads configuration changed while it was being read."
        )
    return config


@st.cache_resource(show_spinner=False)
def _ads_bigquery_client(project: str, dataset: str):
    """The BigQuery client, built once per process rather than once per read.

    Building one loads the credential and opens a session to Google, which is
    work that has nothing to do with how stale the figures are; the client is
    thread-safe and outlives every cache entry that uses it.
    """
    return ads_client.build_client(_ads_config(project, dataset))


@read_log.logged_read("app._ads_accounts")
@st.cache_data(
    ttl=ADS_ACCOUNTS_TTL_SECONDS, show_spinner=False, refresh_mode="background"
)
def _ads_accounts(project: str, dataset: str, today: _dt.date) -> list[AdsAccount]:
    """Every ad account in the dataset, with the things about it that hold still.

    ``today`` is a cache key and not an argument: the earliest loaded day is
    clamped to yesterday, so the entry has to roll over at the day boundary or it
    would go on reporting the day before that.

    The two reads per account are independent and each is a BigQuery job with a
    second or so of latency in front of it, so they go out together rather than
    one after another - as do the accounts.
    """
    read_log.mark_executed()
    config = _ads_config(project, dataset)
    client = _ads_bigquery_client(project, dataset)
    customers = ads_client.customer_ids(client, config)
    if not customers:
        return []
    read = _parallel(
        {
            f"{kind}:{customer_id}": (
                lambda call=call, customer_id=customer_id: call(
                    client, config, customer_id
                )
            )
            for customer_id in customers
            for kind, call in (
                ("account", ads_client.account),
                ("loaded", ads_client.loaded_from),
            )
        }
    )
    return [
        AdsAccount(
            customer_id,
            *read[f"account:{customer_id}"],
            read[f"loaded:{customer_id}"],
        )
        for customer_id in customers
    ]


@read_log.logged_read("app._ads_cached")
@st.cache_data(ttl=ADS_TTL_SECONDS, show_spinner=False, refresh_mode="background")
def _ads_cached(project: str, dataset: str, today: _dt.date) -> AdsRead:
    """Campaign stats by day, campaign names, the account's name and currency.

    Keyed on the dataset rather than the credential: the credential is read from
    the environment and does not change while the process lives, and a service
    account key has no business in a cache key. ``today`` is a key too, so the
    entry expires when the transfer loads a new day rather than on a timer alone.

    A dataset can hold several ad accounts, and their spend can only be added if
    they bill in the same currency, so the most common currency wins and the rest
    are set aside - the same rule the order book follows for takings.
    """
    read_log.mark_executed()
    config = _ads_config(project, dataset)
    client = _ads_bigquery_client(project, dataset)
    accounts = _ads_accounts(project, dataset, today)
    empty = pd.DataFrame()
    if not accounts:
        return AdsRead(empty, empty, "", "USD", None, [])

    counted = collections.Counter(account.currency for account in accounts)
    main = counted.most_common(1)[0][0]
    others = sorted({code for code in counted if code != main})
    billing = [account for account in accounts if account.currency == main]

    tasks: dict[str, Callable[[], Any]] = {}
    for account in billing:
        customer_id = account.customer_id
        tasks[f"stats:{customer_id}"] = (
            lambda customer_id=customer_id: ads_client.daily_stats(
                client, config, customer_id, ADS_WINDOW_DAYS
            )
        )
        tasks[f"names:{customer_id}"] = (
            lambda customer_id=customer_id: ads_client.campaign_names(
                client, config, customer_id
            )
        )
    read = _parallel(tasks)
    stats = [read[f"stats:{account.customer_id}"] for account in billing]
    names = [read[f"names:{account.customer_id}"] for account in billing]
    starts = [
        account.history_start
        for account in billing
        if account.history_start is not None
    ]
    return AdsRead(
        stats=pd.concat(stats, ignore_index=True) if stats else empty,
        names=pd.concat(names, ignore_index=True) if names else empty,
        account=", ".join(account.name or account.customer_id for account in billing),
        currency=main,
        # The latest of the accounts' first days: the window is only wholly
        # loaded once every account it sums has reached it.
        history_start=max(starts) if starts else None,
        other_currencies=others,
    )


def _prefetch_ads() -> None:
    """Start the BigQuery read while the order book is still being read.

    The ads panel is drawn below the shop's own figures, so it was only asked for
    once the order book had come back from Postgres - two networks waited on one
    after the other for no reason but the order they appear on the page. Nothing
    is returned: the read lands in the cache entry the panel goes on to ask for,
    so by the time it does it either finds the answer waiting or waits on the
    query it would have run itself. Failures are left for the panel to report,
    which is where the reader can see them.
    """
    try:
        config = ads_client.load_ads_env()
    except ads_client.AdsConfigError:
        return
    if config is None:
        return
    context = get_script_run_ctx()

    def _read() -> None:
        add_script_run_ctx(threading.current_thread(), context)
        try:
            _ads_cached(config.project, config.dataset, _dt.date.today())
        except Exception as exc:  # noqa: BLE001
            logger.info("Ads prefetch failed; the panel will report it: %s", exc)

    threading.Thread(target=_read, name="ads-prefetch", daemon=True).start()


def _ads_sales(
    order_book: orders_client.OrderBook | None, spend: ads_client.Spend
) -> ads_client.Sales | None:
    """The CRM's orders over exactly the days the spend covers, or ``None``.

    The same days matter more than they look. Ads figures end yesterday and are
    counted in the account's own timezone, so comparing them against a CRM window
    ending now would divide a full month of spend by a month plus today's orders.
    The window is the one the spend was summed over, not the days within it that
    happened to have activity: an account that paused for the first week of the
    month would otherwise have its orders counted over fewer days than its spend.
    """
    if order_book is None or spend.window_end is None:
        return None
    span = spend.days_loaded
    end = _dt.datetime.combine(
        spend.window_end + _dt.timedelta(days=1),
        _dt.time.min,
        tzinfo=_dt.timezone.utc,
    )
    # The shop's main currency only. Every other money section does the same,
    # and adding takings in two currencies would inflate the return on spend
    # quoted in one of them.
    book, currency, _others = orders.single_currency(order_book.orders)
    metrics = orders.window_metrics(book, span, now=end)
    return ads_client.Sales(
        orders=metrics.paid_orders,
        revenue=metrics.revenue,
        currency=currency,
        prev_orders=metrics.prev_paid_orders,
        prev_revenue=metrics.prev_revenue,
    )


def _one(currency: str) -> str:
    """``$1``, or ``1 CAD`` where the currency has no symbol here."""
    return _money(1, currency).replace("1.00", "1")


def _money_delta(change: float, currency: str) -> str:
    """``+$412.90``, signed where Streamlit looks for the sign.

    ``st.metric`` colours a delta by whether the string starts with a minus, and
    a currency symbol in front of it would draw every fall in spend as a rise.
    """
    if not round(change, 2):
        return "flat"
    sign = "+" if change > 0 else "-"
    return f"{sign}{_money(abs(change), currency)}"


def _charged_commission(
    spend: ads_client.Spend, currency: str
) -> ads_client.Commission | None:
    """What the marketplace actually charged over the days the spend covers.

    Commission is the only part of a sale that is income here, and every
    merchant is on their own rate, so the assumed rate is a guess at a figure
    the payments ledger already holds exactly. Read over exactly the spend's own
    days for the same reason the CRM is: ad spend ends yesterday and a partial
    transfer covers fewer days still, so a commission window ending now would
    divide a month of takings by a fraction of a month of spend.

    ``None`` when Stripe cannot be read, when the account takes no commission at
    all, or when it bills in a currency the ads are not billed in - in each case
    a rate is the honest fallback, and the caption says which it used.
    """
    if spend.window_end is None:
        return None
    span = spend.days_loaded
    try:
        if not cost_client.load_stripe_env():
            return None
        # The same read Burn makes, so the tab pages Stripe once. Disputes are
        # not read here: they are a Burn figure and cost another call.
        entries, truncated = _stripe_ledger_cached(STRIPE_LEDGER_DAYS)
        # The fold bounds the window's start but not its end, and today's sales
        # are not in yesterday's spend: without this the return climbs through
        # the day and reads high by a day in every window.
        if not entries.empty:
            day = pd.to_datetime(entries["day"]).dt.date
            entries = entries[day <= spend.window_end]
        ledger = cost_client.ledger_window(entries, span, now=spend.window_end)
    except Exception:  # noqa: BLE001 - the ads panel is not the place to report it
        return None
    if not ledger.platform or ledger.currency != currency:
        return None
    if not ledger.earnings and not ledger.prev_earnings:
        return None
    # Stripe returns newest first, so a read that hit its ceiling is missing its
    # oldest days. Where the cut falls past this window's start the window's own
    # commission is whole and only the comparison goes; where it falls inside,
    # the measured figure is short of sales and the rate is the honest fallback.
    if truncated and not cost_client.reaches_past(entries, span, now=spend.window_end):
        return None
    before = (
        ledger.prev_net
        if not truncated
        or cost_client.reaches_past(entries, 2 * span, now=spend.window_end)
        else 0.0
    )
    return ads_client.Commission(now=ledger.net, before=before, measured=True)


def _render_ads(order_book: orders_client.OrderBook | None) -> None:
    """What the orders cost to win: spend, cost per order and return.

    The order book says what the shop earned and the funnel says how people got
    there; neither says what was paid to bring them. This is the number a
    leadership meeting asks for first and the dashboard could not answer.
    """
    st.subheader("Ads Spend & Return")
    try:
        config = ads_client.load_ads_env()
    except ads_client.AdsConfigError as exc:
        st.caption(f"Google Ads figures are misconfigured: {exc}")
        return
    if config is None:
        st.caption(
            "Ad figures come from Google's own Ads-to-BigQuery transfer, which "
            "needs no Ads API token. Point the dashboard at the dataset with "
            "GOOGLE_ADS_BQ_PROJECT and GOOGLE_ADS_BQ_DATASET."
        )
        return

    days = st.radio(
        "Window",
        options=list(ads_client.LOOKBACK_WINDOWS),
        format_func=lambda value: f"{value} days",
        index=len(ads_client.LOOKBACK_WINDOWS) - 1,
        horizontal=True,
        key="ads_window_days",
    )
    try:
        with st.spinner("Reading ad spend..."):
            # Not keyed on the window: the widest one was read, and both options
            # are cut out of that frame below, so this is only a wait the first
            # time the page is opened.
            read = _ads_cached(config.project, config.dataset, _dt.date.today())
    except ads_client.AdsConfigError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"Could not read `{config.project}.{config.dataset}`: {str(exc)[:200]}"
        )
        return

    spend = ads_client.window(read.stats, days, history_start=read.history_start)

    if read.stats.empty:
        # No rows in the window means three different things, and only the
        # transfer's own history tells them apart: nothing has loaded, some of
        # the window has, or all of it has and the account simply stopped.
        if read.history_start is None:
            st.info(
                "The Ads dataset is readable but holds no spend yet. Google's "
                "transfer loads one day per run and backfills only when asked, "
                "so a new transfer has nothing in it until its first run "
                "completes."
            )
        elif spend.partial:
            st.warning(
                f"Only {spend.days_loaded} of these {days} days have been "
                f"loaded: the transfer's history starts on {spend.history_start}"
                ", and no spend was recorded in the part that has arrived."
            )
        else:
            st.info(
                f"No spend recorded in the last {days} days. The transfer has "
                f"loaded from {read.history_start} onwards, so this is an "
                "account that stopped advertising rather than missing figures."
            )
        return

    campaigns = ads_client.by_campaign(read.stats, read.names, days)
    sales = _ads_sales(order_book, spend)
    currency = read.currency
    money = currency.lower()
    unit = _one(money)

    ads = "Ads Spend & Return"
    tiles = st.columns(6)
    _tile(
        tiles[0],
        TAB_BUSINESS,
        ads,
        f"Spend ({days}d)",
        _money(spend.cost, money),
        **_delta_arrow(
            _money_delta(spend.cost_change, money) if spend.prev_cost else None,
            higher_is_better=False,
        ),
    )
    _tile(
        tiles[1],
        TAB_BUSINESS,
        ads,
        "Orders (CRM)",
        f"{sales.orders:,}" if sales else "\u2014",
    )
    _tile(
        tiles[2],
        TAB_BUSINESS,
        ads,
        "Ad spend per order",
        _money(spend.cost / sales.orders, money) if sales and sales.orders else "\u2014",
    )
    # Comparable only if the shop takes money in the currency the ads are billed
    # in; otherwise the ratio is one currency divided by another.
    comparable = bool(sales and (not sales.currency or sales.currency == money))
    # The one figure on this panel that is income rather than turnover, so it is
    # the one to steer by: the revenue an ad wins belongs to the merchant and
    # only the commission on it is ours. Break-even is 1.00, not 3x.
    try:
        keep = ads_client.commission_rate()
    except ads_client.AdsConfigError as exc:
        st.caption(str(exc))
        keep = ads_client.DEFAULT_COMMISSION_RATE
    # Merchants sit on different agreements, so a single rate is a guess at a
    # number Stripe already holds: what it actually charged each sale. Read that
    # when it is readable, and fall back to the rate when it is not.
    commission = _charged_commission(spend, money)
    if commission is not None:
        earned = ads_client.earned_return(commission.now, spend.cost)
        before = ads_client.earned_return(commission.before, spend.prev_cost)
        basis = (
            "Commission Stripe charged in the window, divided by spend. Every "
            "sale in the window is in it, ads or not, so it is a ceiling rather "
            f"than a return; {ads_client.BREAK_EVEN_RETURN:.2f} is where an ad "
            "pays for itself."
        )
    else:
        earned = (
            ads_client.commission_return(sales.revenue, spend.cost, keep)
            if sales and comparable
            else 0.0
        )
        before = (
            ads_client.commission_return(sales.prev_revenue, spend.prev_cost, keep)
            if sales and comparable
            else 0.0
        )
        basis = (
            f"Revenue in the window at {keep:.0%} commission, divided by spend. "
            "Every sale in the window is in it, ads or not, so it is a ceiling "
            f"rather than a return; {ads_client.BREAK_EVEN_RETURN:.2f} is where "
            "an ad pays for itself."
        )
    _tile(
        tiles[3],
        TAB_BUSINESS,
        ads,
        f"Commission per {unit} spent, at most",
        f"{earned:.2f}" if earned else "\u2014",
        # A ratio, not money: hundredths are the whole movement here, so an
        # unchanged window says so in the word the tiles use rather than
        # printing a zero that reads as a measurement.
        **_delta_arrow(
            (f"{earned - before:+.2f}" if round(earned - before, 2) else "flat")
            if earned and before
            else None
        ),
        help=basis,
    )
    _tile(
        tiles[4],
        TAB_BUSINESS,
        ads,
        f"Revenue per {unit} spent",
        f"{sales.revenue / spend.cost:.1f}x"
        if sales and comparable and spend.cost
        else "\u2014",
        help="Gross, and mostly the merchants': the tile to its left is ours.",
    )
    _tile(
        tiles[5],
        TAB_BUSINESS,
        ads,
        "Google's own conversions",
        f"{spend.conversions:,.0f}",
    )

    # The headline follows the sentences in the expander: a ceiling that was
    # read and came to zero is still a ceiling worth printing - it says the
    # window's sales earned nothing, not that nothing could be measured.
    # And only where money went out: a quiet window can still hold takings and
    # a measured ledger, but a return on nothing spent is not a figure.
    has_ceiling = commission is not None or bool(sales and comparable and sales.revenue)
    if has_ceiling and spend.cost:
        goal = ads_client.BREAK_EVEN_RETURN
        gap = goal - earned
        standing = (
            "Above what an ad needs to pay for itself - at its most flattering."
            if gap <= 0
            else f"{_money(gap, money)} short on every {unit} even at its most "
            f"flattering, which is {_money(spend.cost * gap, money)} over these "
            f"{days} days."
        )
        # The same sum over the sales Google's own attribution recorded, which is
        # the floor under that ceiling. Quoted beside it rather than instead of
        # it: one counts sales the ads had nothing to do with, the other misses
        # sales they did win, and the answer is somewhere between the two. The
        # conversion value is commission already - the site's tag deliberately
        # sends the marketplace's cut of each order - so no rate is applied.
        floor = ads_client.attributed_return(spend.conversion_value, spend.cost)
        # Withheld unless it really is the lower of the two: Google claiming more
        # value than the shop captured, or a measured commission of nothing,
        # would put this above the ceiling it is quoted as sitting under.
        attributed = (
            f" On the sales Google itself claims it is {_money(floor, money)} "
            f"per {unit}."
            if spend.conversion_value and floor < earned
            else ""
        )
        trend = (
            ""
            if not before or earned == before
            else f" {'Up' if earned > before else 'Down'} from "
            f"{_money(before, money)} in the previous {days} days."
        )
        headline = (
            f"### At most {_money(earned, money)} back for every {unit} of ad "
            f"spend\n\n"
            f"**Goal {goal:.2f}.** {standing}{attributed}{trend}"
        )
        _report(TAB_BUSINESS).note(
            ads,
            f"**At most {_money(earned, money)} back for every {unit} of ad "
            f"spend.** Goal {goal:.2f}. {standing}{attributed}{trend}",
        )
        st.markdown(_unmathed(headline))
        st.caption(
            "Commission here is what Stripe charged across every merchant in "
            "the window, so the different rates they are on are already in it, "
            f"rather than {keep:.0%} assumed on captured revenue."
            if commission is not None
            else f"Commission is assumed at {keep:.0%} of captured revenue. "
            "Connect a Stripe key with Application Fees read access and this "
            "becomes what was actually charged, per merchant agreement."
        )
    elif spend.cost and spend.conversion_value:
        # No ceiling could be computed - no Stripe ledger, and either no
        # comparable takings or none captured in the window - but the
        # attributed return needs only the ad account's own figures, which are
        # in its own currency by definition.
        floor = ads_client.attributed_return(spend.conversion_value, spend.cost)
        headline = (
            f"### On the sales Google itself claims, {_money(floor, money)} "
            f"back for every {unit} of ad spend\n\n"
            f"**Goal {ads_client.BREAK_EVEN_RETURN:.2f}.** The tag sends the "
            "marketplace's cut of each order, so this is commission already. "
            "Only Google's own attribution is counted here; no all-channel "
            "ceiling could be read beside it."
        )
        _report(TAB_BUSINESS).note(
            ads,
            f"**On the sales Google itself claims, {_money(floor, money)} back "
            f"for every {unit} of ad spend.** Goal "
            f"{ads_client.BREAK_EVEN_RETURN:.2f}. Only Google's own attribution "
            "is counted; no all-channel ceiling could be read beside it.",
        )
        st.markdown(_unmathed(headline))

    if spend.partial:
        st.warning(
            f"Only {spend.days_loaded} of these {days} days have been loaded: "
            f"the transfer's history starts on {spend.history_start}, so the "
            "spend figure is that much of the window rather than all of it."
        )
    elif not spend.cost:
        st.info(
            f"No spend recorded in the last {days} days. The dataset is loaded "
            "up to date, so this is a quiet account rather than missing figures."
        )
    if sales and not comparable:
        st.caption(
            f"The shop's takings are in {sales.currency.upper()} and the ad "
            f"account bills in {currency.upper()}, so return per unit spent is "
            "left blank rather than dividing one currency by another."
        )

    st.dataframe(
        campaigns.assign(
            cost=campaigns["cost"].map(lambda value: _money(value, money)),
            conversion_value=campaigns["conversion_value"].map(
                lambda value: _money(value, money)
            ),
            cost_per_conversion=campaigns["cost_per_conversion"].map(
                lambda value: _money(value, money) if value else "\u2014"
            ),
            roas=campaigns["roas"].map(lambda value: f"{value:.1f}x" if value else "\u2014"),
            budget=campaigns["budget"].map(lambda value: _money(value, money)),
            conversions=campaigns["conversions"].map(lambda value: f"{value:,.1f}"),
        ).rename(
            columns={
                "campaign": "Campaign",
                "status": "Status",
                "channel": "Type",
                "cost": "Spend",
                "clicks": "Clicks",
                "conversions": "Conversions",
                "conversion_value": "Value",
                "cost_per_conversion": "Cost per conversion",
                "roas": f"Value per {unit}",
                "budget": "Daily budget",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    # Orders can still be compared against spend when the currencies differ -
    # a count is a count - but revenue cannot, so it is withheld rather than
    # divided by a figure in another currency.
    if sales and not comparable:
        sales = ads_client.Sales(
            orders=sales.orders, revenue=0.0, currency=sales.currency
        )
    lines = ads_client.verdicts(
        spend, campaigns, sales, currency, rate=keep, commission=commission
    )
    if lines:
        with st.expander("What this means", expanded=True):
            _said(TAB_BUSINESS, ads, lines)

    if read.other_currencies:
        st.caption(
            f"Only the accounts billing in {currency.upper()} are counted here. "
            f"The dataset also holds {', '.join(read.other_currencies)} accounts, "
            "whose spend cannot be added to this total."
        )

    st.caption(
        f"{read.account or 'Google Ads'}, read from Google's daily transfer into "
        f"`{config.dataset}` rather than the Ads API, which needs a manager "
        "account. Spend ends yesterday and is counted in the ad account's own "
        "timezone. 'Conversions' is Google's own count against the day of the "
        "click and is not the same thing as an order: the CRM's orders are the "
        "figure to quote, and every order in the window counts towards them, "
        "including the ones no ad won."
    )


BURN_TTL_SECONDS = 900


@read_log.logged_read("app._openai_costs_cached")
@st.cache_data(ttl=BURN_TTL_SECONDS, show_spinner=False, refresh_mode="background")
def _openai_costs_cached(days: int) -> pd.DataFrame:
    """Daily OpenAI cost by project and line item.

    Not keyed on the credential: it comes from the environment, does not change
    while the process lives, and an admin key has no business in a cache key.
    """
    read_log.mark_executed()
    key = cost_client.load_openai_env()
    if not key:  # pragma: no cover - the caller checks first
        raise cost_client.CostConfigError("No OpenAI admin key is configured.")
    return cost_client.openai_costs(key, days)


# Every panel that wants the ledger asks for the same days of it, so that all of
# them read one download: the longest window any of them offers, its preceding
# window, and a day of slack for the ads panel, whose window ends yesterday.
# Folding a window narrower than this out of the frame costs nothing; paging a
# busy platform's balance transactions a second time costs a hundred requests.
STRIPE_LEDGER_DAYS = (
    max(cost_client.LOOKBACK_WINDOWS + ads_client.LOOKBACK_WINDOWS) + 1
)


@read_log.logged_read("app._stripe_ledger_cached")
@st.cache_data(ttl=BURN_TTL_SECONDS, show_spinner=False, refresh_mode="background")
def _stripe_ledger_cached(days: int) -> tuple[pd.DataFrame, bool]:
    """Stripe's ledger for the window and the window before.

    Separate from the disputes beside it because two panels want the ledger and
    only one wants the disputes: a busy platform's ledger runs to a hundred
    pages, and the ads panel has no use for a chargeback count.
    """
    read_log.mark_executed()
    key = cost_client.load_stripe_env()
    if not key:  # pragma: no cover - the caller checks first
        raise cost_client.CostConfigError("No Stripe key is configured.")
    return cost_client.stripe_ledger(key, days)


@read_log.logged_read("app._stripe_disputes_cached")
@st.cache_data(ttl=BURN_TTL_SECONDS, show_spinner=False, refresh_mode="background")
def _stripe_disputes_cached(days: int) -> int:
    read_log.mark_executed()
    key = cost_client.load_stripe_env()
    if not key:  # pragma: no cover - the caller checks first
        raise cost_client.CostConfigError("No Stripe key is configured.")
    return cost_client.stripe_disputes(key, days)


def _stripe_cached(days: int) -> tuple[pd.DataFrame, bool, int]:
    """Stripe's ledger for the window and the window before, and its disputes."""
    # Two independent reads off the same key - a ledger page walk and a
    # disputes count - so they go out together rather than one after another.
    read = _parallel(
        {
            "ledger": lambda: _stripe_ledger_cached(STRIPE_LEDGER_DAYS),
            "disputes": lambda: _stripe_disputes_cached(days),
        }
    )
    entries, truncated = read["ledger"]
    return entries, truncated, read["disputes"]


def _burn_reads(days: int) -> dict[str, Callable[[], Any]]:
    """Burn's independent bills, as callables to run together.

    OpenAI, Google Cloud and Stripe are three unrelated providers - reading
    one tells you nothing about another - so asking them one after another
    made the panel as slow as their sum. Only a provider whose configuration
    is actually present is asked at all; a config error surfaces here just as
    it would from the direct call, and simply leaves the task out so the other
    two providers are not held up waiting for it.
    """
    tasks: dict[str, Callable[[], Any]] = {}
    try:
        if cost_client.load_openai_env():
            tasks["openai"] = lambda: _openai_costs_cached(days)
    except cost_client.CostConfigError:
        pass
    try:
        if cost_client.load_billing_env() is not None:
            tasks["cloud"] = lambda: _cloud_costs_cached(
                CLOUD_WINDOW_DAYS, _dt.date.today()
            )
    except cost_client.CostConfigError:
        pass
    try:
        if cost_client.load_stripe_env():
            tasks["stripe"] = lambda: _stripe_cached(days)
    except cost_client.CostConfigError:
        pass
    return tasks


def _render_burn() -> None:
    """What the business spends, and what its own payment ledger says it kept.

    Revenue on its own is half a sentence. This is the other half, provider by
    provider as each one's access arrives: OpenAI reports its organization costs,
    Stripe reports the platform's commission, and Google Cloud follows its
    billing export.
    """
    st.subheader("Burn")
    # One window for every provider here: a leadership reader compares these
    # figures with each other, and two windows on one page invite adding a week
    # of one bill to a month of another.
    days = st.radio(
        "Window",
        options=list(cost_client.LOOKBACK_WINDOWS),
        format_func=lambda value: f"{value} days",
        index=len(cost_client.LOOKBACK_WINDOWS) - 1,
        horizontal=True,
        key="burn_window_days",
    )
    # The three providers' reads go out together; each panel below still
    # renders on its own account of what it got, so one dead provider costs
    # its own section and not the two beside it.
    with st.spinner("Reading what each provider billed..."):
        answers, errors = _gather(_burn_reads(days))
    _render_ai_costs(days, answers.get("openai"), errors.get("openai"))
    _render_cloud(days, answers.get("cloud"), errors.get("cloud"))
    _render_stripe(days, answers.get("stripe"), errors.get("stripe"))


class CloudRead(NamedTuple):
    """Cloud charges, and what the export they came from covers."""

    costs: pd.DataFrame
    history_start: _dt.date | None
    covered_to: _dt.date | None
    # Fully-qualified, and more than one when several billing accounts export
    # into the same dataset. The first is the one read.
    tables: tuple[str, ...] = ()


# The bill moves once a day at best and its last day is yesterday, so a quarter
# of an hour buys no freshness for another BigQuery job: `cloud_costs` now bounds
# its scan with a partition predicate, and `billing_coverage` - which has no
# window to filter on, since finding its own range is the query's job - caches
# its answer to disk by day rather than repeating the scan. Held on the ads
# panel's cycle here too, keyed on the date so it rolls over when the export does.
CLOUD_TTL_SECONDS = 6 * 3600
# The widest window the panel offers, which is the only one read: `window`
# slices narrower ones out of the frame in pandas, so a click on the radio no
# longer sends BigQuery after a subset of rows already in hand.
CLOUD_WINDOW_DAYS = max(cost_client.LOOKBACK_WINDOWS)


@read_log.logged_read("app._cloud_costs_cached")
@st.cache_data(ttl=CLOUD_TTL_SECONDS, show_spinner=False, refresh_mode="background")
def _cloud_costs_cached(days: int, today: _dt.date) -> CloudRead:
    """Google Cloud's billing export, and what the export covers.

    Read to the export's own last day rather than to today: it is written in
    arrears, so the days it has not reached are days it has no charges for, and
    a window ending now would average real spend over imaginary free days.
    """
    read_log.mark_executed()
    config = cost_client.load_billing_env()
    if config is None:  # pragma: no cover - the caller checks first
        raise cost_client.CostConfigError("No billing export is configured.")
    client = _billing_bigquery_client(config.project, config.dataset)
    tables = cost_client.billing_tables(client, config)
    if not tables:
        return CloudRead(pd.DataFrame(), None, None)
    first, last = cost_client.billing_coverage(client, tables[0])
    if last is None:
        return CloudRead(pd.DataFrame(), None, None, tuple(tables))
    return CloudRead(
        cost_client.cloud_costs(client, tables[0], days, now=last),
        first,
        last,
        tuple(tables),
    )


@st.cache_resource(show_spinner=False)
def _billing_bigquery_client(project: str, dataset: str):
    """The billing export's BigQuery client, built once per process.

    Keyed on where it reads so a changed variable builds a new one, as the ads
    client is: a credential loaded and a session opened to Google are not work
    that has anything to do with how stale the figures are.
    """
    config = cost_client.load_billing_env()
    if config is None or (config.project, config.dataset) != (project, dataset):
        raise cost_client.CostConfigError(
            "The billing export configuration changed while it was being read."
        )
    return cost_client.build_billing_client(config)


def _render_cloud(
    days: int,
    read: "CloudRead | None" = None,
    error: Exception | None = None,
) -> None:
    """What Google Cloud charged, service by service.

    The largest bill of the three and the one nobody sees: it arrives monthly,
    by which time a service left running has been running for a month. Read
    from the billing export, which is the only place the figure exists per day.

    ``read``/``error`` are the answer the caller already gathered alongside
    the other providers' reads; when neither is given (a direct call, as the
    tests make) the read happens here instead, exactly as it always did.
    """
    try:
        config = cost_client.load_billing_env()
    except cost_client.CostConfigError as exc:
        st.caption(f"Google Cloud costs are misconfigured: {exc}")
        return
    if config is None:
        st.caption(
            "Google Cloud spend comes from the billing export. Point the "
            "dashboard at the project holding it with GCP_BILLING_BQ_PROJECT."
        )
        return

    if error is not None:
        if isinstance(error, cost_client.CostConfigError):
            st.warning(str(error))
        else:
            st.warning(f"Could not read the billing export: {str(error)[:200]}")
        return
    if read is None:
        try:
            with st.spinner("Reading what Google Cloud charged..."):
                read = _cloud_costs_cached(CLOUD_WINDOW_DAYS, _dt.date.today())
        except cost_client.CostConfigError as exc:
            st.warning(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not read the billing export: {str(exc)[:200]}")
            return

    history_start, covered_to = read.history_start, read.covered_to
    if history_start is None or covered_to is None:
        st.caption(
            f"There is no billing export in `{config.project}.{config.dataset}` "
            "yet. Enable *standard usage cost* export under Billing, Billing "
            "export; Google writes the first table within a few hours, and it "
            "covers nothing from before that."
        )
        return

    # The days the export actually holds, so a fortnight-old export is not
    # averaged over a month it was not switched on for.
    covered = (covered_to - history_start).days + 1
    burn = cost_client.window(
        read.costs,
        days,
        provider="Google Cloud",
        now=covered_to,
        loaded=covered,
        # The period before this one has to be whole to be compared with. Two
        # days of the previous month is not a cheaper month, and reads as one.
        comparable=covered >= 2 * days,
    )
    cloud = "Cloud costs"
    money = burn.currency
    tiles = st.columns(3)
    _tile(
        tiles[0],
        TAB_BUSINESS,
        cloud,
        f"Google Cloud ({days}d)",
        _money(burn.cost, money),
        **_delta_arrow(
            _money_delta(burn.cost_change, money)
            if burn.prev_cost and burn.comparable
            else None,
            higher_is_better=False,
        ),
    )
    # A day rather than a month, unlike the AI tile beside it: Cloud is billed
    # by the hour for things left running, and a daily figure is what a service
    # nobody needed costs while nobody is looking at it.
    _tile(
        tiles[1],
        TAB_BUSINESS,
        cloud,
        "A day of Cloud",
        _money(burn.per_day, money),
    )
    _tile(
        tiles[2],
        TAB_BUSINESS,
        cloud,
        "Most expensive service",
        f"{burn.lines.iloc[0]['line_item']}" if not burn.lines.empty else "\u2014",
    )

    if not burn.lines.empty:
        st.dataframe(
            burn.lines.head(12)
            .assign(
                cost=lambda frame: frame["cost"].map(
                    lambda value: _money(value, money)
                ),
                share=lambda frame: frame["share"].map(lambda value: f"{value:.0%}"),
            )
            .rename(
                columns={"line_item": "Service", "cost": "Cost", "share": "Share"}
            ),
            width="stretch",
            hide_index=True,
        )

    if burn.other_currencies:
        st.caption(
            "These figures are the "
            f"{money.upper()} charges only; Google Cloud also billed in "
            + ", ".join(code.upper() for code in burn.other_currencies)
            + ", which is never added to them."
        )

    lines = cost_client.verdicts(burn)
    # Both ends of the export are said before the verdicts, since either one
    # changes how every figure above reads.
    lag = (_dt.date.today() - covered_to).days
    if lag > 1:
        lines.insert(
            0,
            f"**The export has only reached {covered_to}**, {lag} days behind "
            "today, so this window ends there rather than now. Google writes it "
            "in arrears and backfills over hours after it is switched on.",
        )
    # More than one export in the dataset is more than one billing account, and
    # two accounts' bills are no more addable than two currencies.
    if len(read.tables) > 1:
        lines.insert(
            0,
            f"**The dataset holds {len(read.tables)} billing exports**, one per "
            f"billing account. These figures are `{read.tables[0].rsplit('.', 1)[1]}` "
            "alone.",
        )
    # It is not retroactive either, so a window starting before it was switched
    # on is a shorter period wearing a longer label.
    # Equal is the same case: a window as long as the export's whole history has
    # nothing behind it, and saying the comparison rests on nought days is worse
    # than saying there is no comparison.
    if covered <= days:
        lines.insert(
            0,
            f"**The export only goes back to {history_start}**, so these "
            f"{days} days are {covered} days of charges, and there is no "
            "earlier period to compare them with.",
        )
    elif covered < 2 * days:
        lines.insert(
            0,
            f"**The export only goes back to {history_start}**, so it holds "
            f"{covered - days} of the {days} days before this window - too few "
            "to compare with, and no trend is drawn until it holds them all.",
        )
    if lines:
        with st.expander("What Google Cloud costs", expanded=True):
            _said(TAB_BUSINESS, cloud, lines)


def _render_ai_costs(
    days: int,
    costs: pd.DataFrame | None = None,
    error: Exception | None = None,
) -> None:
    """What the AI providers charged, and what the bill is actually made of.

    ``costs``/``error`` are the answer the caller already gathered alongside
    the other providers' reads; when neither is given (a direct call, as the
    tests make) the read happens here instead, exactly as it always did.
    """
    try:
        key = cost_client.load_openai_env()
    except cost_client.CostConfigError as exc:
        st.caption(f"AI costs are misconfigured: {exc}")
        return
    if not key:
        st.caption(
            "AI spend comes from OpenAI's organization cost endpoint, which "
            "needs an admin key rather than the project key the product uses. "
            "Set OPENAI_ADMIN_KEY to show it."
        )
        return

    if error is not None:
        if isinstance(error, cost_client.CostConfigError):
            st.warning(str(error))
        else:
            st.warning(f"Could not read OpenAI costs: {str(error)[:200]}")
        return
    if costs is None:
        try:
            with st.spinner("Reading what the month cost..."):
                costs = _openai_costs_cached(days)
        except cost_client.CostConfigError as exc:
            st.warning(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not read OpenAI costs: {str(exc)[:200]}")
            return

    burn = cost_client.window(costs, days)
    if not burn.cost and costs.empty:
        st.info(
            "OpenAI reports no charges for this organization in the last "
            f"{days} days."
        )
        return

    money = burn.currency
    ai = "AI costs"
    tiles = st.columns(3)
    _tile(
        tiles[0],
        TAB_BUSINESS,
        ai,
        f"OpenAI ({days}d)",
        _money(burn.cost, money),
        **_delta_arrow(
            _money_delta(burn.cost_change, money) if burn.prev_cost else None,
            higher_is_better=False,
        ),
    )
    _tile(
        tiles[1],
        TAB_BUSINESS,
        ai,
        "At this rate, a month",
        _money(burn.monthly, money),
    )
    # The share of the bill that is context sent again rather than new work,
    # which is the one line on an AI invoice that is usually a choice.
    _tile(
        tiles[2],
        TAB_BUSINESS,
        ai,
        "Cached context",
        f"{cost_client.cached_share(burn.lines):.0%}" if not burn.lines.empty else "\u2014",
    )

    if not burn.lines.empty:
        st.dataframe(
            burn.lines.head(12)
            .assign(
                cost=lambda frame: frame["cost"].map(
                    lambda value: _money(value, money)
                ),
                share=lambda frame: frame["share"].map(lambda value: f"{value:.0%}"),
            )
            .rename(
                columns={
                    "line_item": "Line item",
                    "cost": "Cost",
                    "share": "Share",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    # The window's own rows in the window's own currency, as the totals above:
    # the fetch covers twice the window, and neither the earlier period nor a
    # charge billed in euros belongs under a figure labelled with this one.
    billed, _, _ = cost_client.main_currency(costs)
    projects = cost_client.by_project(
        billed[billed["day"] >= burn.first_day] if burn.first_day else billed.iloc[0:0]
    )
    if len(projects) > 1:
        # Escaped like the verdicts above: a caption is markdown too, and two
        # projects in it are two dollar signs on one line, which Streamlit reads
        # as maths and eats.
        st.caption(
            _unmathed(
                "By project: "
                + ", ".join(
                    # _text_or rather than ``or``: a cloud line with no project
                    # attached comes back as NaN, and NaN is truthy, so this
                    # read "nan $312" rather than "unnamed $312".
                    f"{_text_or(row['project'], 'unnamed')} "
                    f"{_money(_number_or(row['cost']), money)}"
                    for _, row in projects.iterrows()
                )
            )
        )

    if burn.other_currencies:
        st.caption(
            "These figures are the "
            f"{money.upper()} charges only; OpenAI also billed in "
            + ", ".join(code.upper() for code in burn.other_currencies)
            + ", which is never added to them."
        )

    lines = cost_client.verdicts(burn)
    if lines:
        with st.expander("What this means", expanded=True):
            _said(TAB_BUSINESS, ai, lines)

    st.caption(
        "OpenAI's own organization cost report, read with an admin key that can "
        "do nothing but read it. Today counts: a provider bills as it goes, so "
        "the day's charges so far are real money. This is the AI line of the "
        "bill; Google Cloud is the section below it."
    )


def _render_stripe(
    days: int,
    cached: tuple[pd.DataFrame, bool, int] | None = None,
    error: Exception | None = None,
) -> None:
    """What Stripe's books say the platform kept.

    Read expecting a cost - card processing fees - and it is not one: this
    account is a Connect platform, so each sale's fees are charged on the
    merchant's own account and what lands here is the marketplace's commission.
    The panel reports what is there rather than what was hoped for.

    ``cached``/``error`` are the answer the caller already gathered alongside
    the other providers' reads; when neither is given (a direct call, as the
    tests make) the read happens here instead, exactly as it always did.
    """
    try:
        key = cost_client.load_stripe_env()
    except cost_client.CostConfigError as exc:
        st.caption(f"Stripe figures are misconfigured: {exc}")
        return
    if not key:
        st.caption(
            "Payment figures come from Stripe's balance transactions. Set "
            "STRIPE_READONLY_API_KEY to a restricted key with read access to "
            "balance transactions, charges, disputes and payouts."
        )
        return

    if error is not None:
        if isinstance(error, cost_client.CostConfigError):
            st.warning(str(error))
        else:
            st.warning(f"Could not read Stripe: {str(error)[:200]}")
        return
    if cached is None:
        try:
            with st.spinner("Reading Stripe's ledger..."):
                cached = _stripe_cached(days)
        except cost_client.CostConfigError as exc:
            st.warning(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not read Stripe: {str(exc)[:200]}")
            return
    entries, truncated, disputes = cached

    # Where the read's cap fell short of this window's own start, the window is
    # missing sales too, and neither it nor the comparison can be trusted.
    whole = not truncated or cost_client.reaches_past(entries, days)
    ledger = cost_client.ledger_window(
        entries,
        days,
        disputes=disputes,
        # The previous window starts a further ``days`` back, and the read is of
        # two months whatever window is chosen: a cap can lose the older window
        # while leaving a seven-day comparison whole.
        comparable=not truncated or cost_client.reaches_past(entries, 2 * days),
    )
    # The window's own rows, not the download's: one read now covers two months
    # for every panel, so a quiet week inside a busy quarter has to still be
    # able to say that nothing moved.
    if not ledger.earnings and ledger.first_day is None:
        st.info(
            f"Stripe recorded no money moving in the last {days} days."
            + (
                f" {disputes} dispute{'s were' if disputes != 1 else ' was'} opened, "
                "which is money at risk rather than money lost."
                if disputes
                else ""
            )
        )
        return

    money = ledger.currency
    payments = "Payments"
    tiles = st.columns(4)
    # Commission on a platform; on an ordinary account the same tile is its own
    # takings less what Stripe charged to process them, which is not commission.
    kept = "Commission" if ledger.platform else "Payments"
    _tile(
        tiles[0],
        TAB_BUSINESS,
        payments,
        f"{kept} kept ({days}d)",
        _money(ledger.net, money),
        # Compared with the same quantity the tile shows - commission after
        # refunds - so a heavily refunded period cannot read as a rise.
        **_delta_arrow(
            _money_delta(ledger.net_change, money)
            if ledger.prev_net and ledger.comparable
            else None
        ),
    )
    _tile(
        tiles[1], TAB_BUSINESS, payments, "Refunded", _money(abs(ledger.refunds), money)
    )
    # Nil on a platform account, and worth showing as nil rather than omitting:
    # the question "what do the card fees cost us" deserves an answer.
    _tile(
        tiles[2],
        TAB_BUSINESS,
        payments,
        "Stripe's own fees",
        _money(abs(ledger.fees), money),
    )
    _tile(
        tiles[3],
        TAB_BUSINESS,
        payments,
        "Paid out to the bank",
        _money(abs(ledger.paid_out), money),
    )

    if ledger.other_currencies:
        st.caption(
            f"These figures are the {money.upper()} ledger only; Stripe also "
            "settled in "
            + ", ".join(code.upper() for code in ledger.other_currencies)
            + ", which is never added to them."
        )

    if truncated and not whole:
        st.warning(
            f"Stripe had more ledger entries in the last {days} days than one "
            "read carries, and it returns the newest first, so the oldest of "
            "those days are missing from these figures as well as from the "
            "period before them. Read a shorter window to see it whole."
        )
    elif truncated and not ledger.comparable:
        st.warning(
            "Stripe had more ledger entries than one read carries, and it "
            "returns the newest first, so the oldest days of the comparison "
            "period are missing. The window's own figures are whole; no change "
            "against the period before it is drawn, since part of it is unread."
        )

    lines = cost_client.stripe_verdicts(ledger)
    if lines:
        with st.expander("What Stripe says", expanded=True):
            _said(TAB_BUSINESS, payments, lines)

    st.caption(
        "Stripe's balance transactions, read with a restricted key that cannot "
        "move money. This is the platform's own commission on merchants' sales, "
        "not the merchants' takings, and not the same figure as the CRM's "
        "captured revenue above. Card processing fees are charged on the "
        "connected merchant accounts, which this key cannot see."
    )


# How far back the funnel looks. A week is what a sprint changes; a month is
# what a board meeting asks about; a quarter is the only one big enough for the
# checkout steps, which a hundred-odd people a month reach.
FUNNEL_WINDOWS = (7, 30, 90)
FUNNEL_TTL_SECONDS = 900


@read_log.logged_read("app._funnel_cached")
@st.cache_data(ttl=FUNNEL_TTL_SECONDS, show_spinner=False, refresh_mode="background")
def _funnel_cached(
    credentials: tuple[str, str, str],
    funnel_spec: str,
    days: int,
    offset_days: int = 0,
) -> pd.DataFrame:
    read_log.mark_executed()
    steps = amplitude_client.parse_funnel(funnel_spec)
    return amplitude_client.funnel(credentials, steps, days, offset_days)


@read_log.logged_read("app._breakdown_cached")
@st.cache_data(ttl=FUNNEL_TTL_SECONDS, show_spinner=False, refresh_mode="background")
def _breakdown_cached(
    credentials: tuple[str, str, str], event: str, prop: str, days: int
) -> pd.DataFrame:
    read_log.mark_executed()
    return amplitude_client.event_breakdown(credentials, event, prop, days)


@read_log.logged_read("app._event_users_cached")
@st.cache_data(ttl=FUNNEL_TTL_SECONDS, show_spinner=False, refresh_mode="background")
def _event_users_cached(
    credentials: tuple[str, str, str],
    events: tuple[tuple[str, str], ...],
    days: int,
) -> pd.DataFrame:
    """Events as plain pairs, because Streamlit hashes its cache keys itself and
    is not obliged to know how to hash this module's dataclass."""
    read_log.mark_executed()
    steps = tuple(amplitude_client.Step(label, event) for label, event in events)
    return amplitude_client.event_users(credentials, steps, days)


def _points(change: float) -> str:
    """A change in a conversion rate, in percentage points rather than percent.

    A rate that went from 2% to 3% rose by one point and by fifty percent, and
    the second phrasing is how a modest week gets reported as a triumph.
    """
    return f"{change * 100:+.1f}pp"


# Below this, a rate has not really moved: reporting a hundredth of a point as
# progress teaches people to ignore the whole column.
_NOISE_POINTS = 0.001


def _percent(value: float) -> str:
    """A conversion rate, at the precision the number can carry.

    A step two thousandths of the way down the funnel rounds to 0% at one decimal
    place, which reads as broken instrumentation rather than as a hard step.
    """
    if 0 < value < 0.001:
        return "<0.1%"
    return f"{value * 100:.1f}%"


def _render_product_funnel() -> None:
    """How far visitors get towards an order, and what goes wrong on the way.

    The order book says what the shop sold. It cannot say how many people tried
    and gave up, which is the number that says whether the product is working.
    """
    st.subheader("Product Funnel & Friction")
    try:
        credentials = amplitude_client.load_amplitude_env()
    except amplitude_client.AmplitudeConfigError as exc:
        st.caption(f"Product analytics are misconfigured: {exc}")
        return
    if credentials is None:
        st.caption(
            "The funnel needs an Amplitude project API key and secret key "
            "(Settings -> Projects -> your project). Set AMPLITUDE_API_KEY and "
            "AMPLITUDE_SECRET_KEY."
        )
        return

    days = st.radio(
        "Window",
        options=list(FUNNEL_WINDOWS),
        format_func=lambda value: f"{value} days",
        index=1,
        horizontal=True,
        key="funnel_window_days",
    )
    funnel_spec = os.getenv("AMPLITUDE_FUNNEL", "")
    try:
        with st.spinner("Reading the product funnel..."):
            steps = _funnel_cached(credentials, funnel_spec, days)
    except amplitude_client.AmplitudeConfigError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read the funnel from Amplitude: {str(exc)[:200]}")
        return

    active = _active_people(credentials, days)
    if steps.empty or not steps["users"].iloc[0]:
        # No funnel to draw, but errors and Voss AI use do not depend on it and
        # are the numbers most likely to explain why: an event renamed in the
        # storefront empties the funnel while thousands of people are still here.
        st.info(
            f"Amplitude recorded nobody at the funnel's first step in the last "
            f"{days} days, so there is no funnel to draw. Check the events behind "
            "it are still being sent, or name your own with AMPLITUDE_FUNNEL."
        )
        if active:
            # The one figure this panel still has when the funnel is empty, and
            # the reason the report must carry it: a Business pack with no
            # product line at all reads as nobody having visited.
            _tile(
                st,
                TAB_BUSINESS,
                "Product funnel",
                f"{amplitude_client.ACTIVE_STEP.label} ({days}d)",
                f"{active:,}",
            )
            _render_friction_tabs(credentials, days, active, "used the site")
        return

    # The same window again, ending where this one starts. A conversion rate on
    # its own is not a fact anybody can act on - 2% is either a disaster or an
    # improvement depending on last month - so the comparison is worth a second
    # request. It is allowed to fail: a funnel without a trend is still useful,
    # and a project younger than two windows has no previous period at all.
    previous: pd.DataFrame | None = None
    try:
        with st.spinner("Reading the period before it..."):
            candidate = _funnel_cached(credentials, funnel_spec, days, days)
        if not candidate.empty and candidate["users"].iloc[0]:
            previous = candidate
    except Exception:  # noqa: BLE001
        previous = None

    top = steps.iloc[0]
    end = steps.iloc[-1]
    # The step that loses the most people, ignoring the first: it has nothing
    # before it to have lost anybody from.
    worst = steps.iloc[1:].sort_values("lost", ascending=False).iloc[0]
    funnel = "Product funnel"
    tiles = st.columns(5)
    _tile(
        tiles[0],
        TAB_BUSINESS,
        funnel,
        f"{amplitude_client.ACTIVE_STEP.label} ({days}d)",
        f"{active:,}" if active is not None else "\u2014",
    )
    _tile(
        tiles[1],
        TAB_BUSINESS,
        funnel,
        f"{top['step']} ({days}d)",
        f"{int(top['users']):,}",
        **_delta_arrow(_people_delta(top, previous, 0)),
    )
    _tile(
        tiles[2],
        TAB_BUSINESS,
        funnel,
        f"{end['step']} ({days}d)",
        f"{int(end['users']):,}",
        **_delta_arrow(_people_delta(end, previous, len(steps) - 1)),
    )
    _tile(
        tiles[3],
        TAB_BUSINESS,
        funnel,
        f"{top['step']} to {end['step'].lower()}",
        _percent(float(end["from_start"])),
        **_delta_arrow(_rate_delta(steps, previous, len(steps) - 1, "from_start")),
    )
    _tile(
        tiles[4],
        TAB_BUSINESS,
        funnel,
        "Biggest drop-off",
        worst["step"],
        # A magnitude, not a period-over-period change - the sign is only
        # here so st.metric draws a down arrow. "inverse" would colour that
        # arrow green (inverse flips red/negative to green), which is a
        # rising-stall-count-in-green bug for a tile whose whole point is
        # "this many people were lost". "normal" keeps negative red.
        delta=f"-{int(worst['lost']):,} people",
        delta_color="normal",
    )

    figure = px.bar(
        steps,
        x="users",
        y="step",
        orientation="h",
        title=f"People reaching each step ({days} days)",
    )
    figure.update_yaxes(autorange="reversed")
    figure.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
    theme.apply_palette(figure)
    theme.plot(figure, width="stretch", key="funnel_steps")

    table = steps.assign(
        from_previous=steps["from_previous"].map(_percent),
        from_start=steps["from_start"].map(_percent),
        lost=steps["lost"].map(lambda count: f"{int(count):,}"),
        trend=[
            _rate_delta(steps, previous, index, "from_previous") or "\u2014"
            for index in range(len(steps))
        ],
    )
    # The first step has nothing before it, and printing 0% there reads as a step
    # that loses everybody rather than as the top of the funnel. The same goes for
    # any step whose predecessor nobody reached: a rate over nobody is unknown.
    unreached = [table.index[0]] + [
        table.index[index]
        for index in range(1, len(steps))
        if not int(steps.iloc[index - 1]["users"])
    ]
    table.loc[unreached, ["from_previous", "lost", "trend"]] = "\u2014"
    st.dataframe(
        table.rename(
            columns={
                "step": "Step",
                "event": "Event",
                "users": "People",
                "from_previous": "From previous step",
                "trend": f"vs previous {days}d",
                "from_start": "From the start",
                "lost": "Lost here",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    _render_funnel_verdicts(steps, previous, days)
    st.caption(
        "People, not visits, and the steps must happen in this order within "
        f"{amplitude_client.DEFAULT_CONVERSION_DAYS} days of each other - wine is "
        "read about and bought later, so a shorter window would report the shop as "
        "worse than it is. 'From previous step' is the one to act on: it names the "
        "screen costing the most. The window ends yesterday, because today is "
        f"still being recorded and would read as a slump. "
        f"'{amplitude_client.ACTIVE_STEP.label}' is "
        "beside the funnel rather than in it because most people here arrive on a "
        "product page, and a step before their first action is one nothing can "
        "follow. Configure the steps with AMPLITUDE_FUNNEL."
    )

    # Errors and AI use happen anywhere, not only past the first funnel step, so
    # the share they are quoted against is everybody who used the site when that
    # figure arrived, and the funnel's own first step only when it did not.
    _render_friction_tabs(
        credentials,
        days,
        active or int(top["users"]),
        "used the site" if active else f"reached {top['step']}".lower(),
    )


def _render_friction_tabs(
    credentials: tuple[str, str, str],
    days: int,
    everyone: int,
    denominator: str,
) -> None:
    """What went wrong, and whether Voss AI is used, as a share of ``everyone``.

    Kept apart from the funnel because neither depends on it: an empty funnel is
    usually an event that stopped being sent, and the errors are the likeliest
    explanation, so they should not disappear along with it.

    ``on_change="rerun"`` makes the tabs report which one is open instead of
    always reading ``None``, which is what makes the ``.open`` checks below
    mean something: without it Streamlit renders both bodies on every run
    regardless of which tab is showing, and the reads below them fire for a
    tab nobody is looking at.
    """
    friction_tab, ai_tab = st.tabs(["What went wrong", "Voss AI"], on_change="rerun")
    if friction_tab.open:
        with friction_tab:
            _render_event_counts(
                credentials,
                amplitude_client.FRICTION_EVENTS,
                days,
                everyone,
                "Nothing went wrong in this window, which is worth a second look at "
                "whether these events are being sent.",
                f"Share of everyone who {denominator}. An error one person met ten "
                "times is one person here, not ten.",
            )
            _render_error_breakdowns(credentials, days)
    if ai_tab.open:
        with ai_tab:
            _render_event_counts(
                credentials,
                amplitude_client.AI_EVENTS,
                days,
                everyone,
                "Amplitude recorded no Voss AI use in this window.",
                f"Share of everyone who {denominator}, so this is reach rather than "
                "engagement: it says how many people found it, not how much they used "
                "it.",
            )


def _active_people(credentials: tuple[str, str, str], days: int) -> int | None:
    """How many people used the shop at all, or ``None`` if that read failed.

    Beside the funnel rather than at the top of it: Amplitude's ordered funnels
    require each step to happen after the one before, and most of this shop's
    visitors arrive on a product page as their first act, so counting them as a
    first step discards them - see ``amplitude_client.ACTIVE_STEP``. This is
    context, so a failed read blanks one tile rather than losing the funnel.
    """
    step = amplitude_client.ACTIVE_STEP
    try:
        counts = _event_users_cached(credentials, ((step.label, step.event),), days)
    except Exception:  # noqa: BLE001
        return None
    if counts.empty:
        return None
    return int(counts["users"].iloc[0])


def _per_hundred(rate: float) -> str:
    """``2 of every 100``, without rounding a real few away to none.

    A step that keeps 0.4% of its people keeps somebody; ``round`` would report
    that as nobody, and the sentence would then contradict the percentage printed
    two words earlier. Only an exact none is called none.
    """
    people = round(rate * 100)
    if people == 0 and rate > 0:
        return "fewer than 1 of every 100"
    if people == 100 and rate < 1:
        return "more than 99 of every 100"
    return f"{people} of every 100"


def _delta_arrow(change: str | None, *, higher_is_better: bool = True) -> dict:
    """``st.metric`` arguments that do not call standing still an improvement.

    Streamlit colours a tile's delta by whether the string begins with a minus,
    so "flat" and "+0 people" would both be drawn as a green arrow upwards. In a
    table or a sentence those words are read; on a tile only the arrow is.

    ``higher_is_better`` decides the mapping, not the sign alone: a rise in
    people converted is good (green), a rise in cloud spend is not
    (``st.metric``'s ``delta_color="inverse"`` flips red/green so a cost tile
    does not draw the same green-for-up arrow a growth tile does). Callers
    that only ever show growth metrics can leave this at its default; every
    cost tile in this file passes ``higher_is_better=False`` explicitly - see
    docs/assumptions/5A.md for the audit that found the ones that didn't.
    """
    if change is None:
        return {}
    if _unmoved(change):
        return {"delta": change, "delta_color": "off"}
    return {"delta": change, "delta_color": "normal" if higher_is_better else "inverse"}


def _unmoved(change: str) -> bool:
    """Whether a delta amounts to no movement, read as a number not a prefix.

    "+0.05" is nothing next to a spend figure and everything next to a return
    of 0.84 per unit spent, and a test on the leading characters cannot tell
    them apart: it greys out the rise while colouring the identical fall red.
    Callers that mean no movement say so in the word the tiles already use.
    """
    if change == "flat":
        return True
    figure = re.sub(r"[^0-9.]", "", change)
    try:
        return float(figure) == 0
    except ValueError:
        return False


def _people_delta(
    row: pd.Series, previous: pd.DataFrame | None, index: int
) -> str | None:
    """``+1,204 people`` against the same step in the previous window."""
    if previous is None or index >= len(previous):
        return None
    change = int(row["users"]) - int(previous.iloc[index]["users"])
    return f"{change:+,} people"


def _rate_delta(
    steps: pd.DataFrame, previous: pd.DataFrame | None, index: int, column: str
) -> str | None:
    """The move in one rate against the previous window, in points.

    ``None`` when there is nothing to compare to, so the caller can leave the
    space blank rather than print a zero that looks like a measurement.
    """
    if previous is None or index >= len(previous) or index >= len(steps):
        return None
    # Only comparable if the two windows describe the same step: a changed
    # AMPLITUDE_FUNNEL between reads would otherwise subtract one screen's rate
    # from another's.
    if steps.iloc[index]["event"] != previous.iloc[index]["event"]:
        return None
    # A rate over nobody is unknown, not zero: an empty previous period would
    # otherwise make this one look like a triumph, and a step nobody reached now
    # would report the shop as having collapsed.
    denominator = index - 1 if column == "from_previous" else 0
    for frame in (steps, previous):
        if not int(frame.iloc[denominator]["users"]):
            return None
    change = float(steps.iloc[index][column]) - float(previous.iloc[index][column])
    if abs(change) < _NOISE_POINTS:
        return "flat"
    return _points(change)


def _render_funnel_verdicts(
    steps: pd.DataFrame, previous: pd.DataFrame | None, days: int
) -> None:
    """The table again, in sentences.

    A column of percentages is a thing to interpret; this is the interpretation,
    because the person the dashboard is for reads it between meetings and should
    not have to do the arithmetic to find out which screen is losing the shop
    its customers.
    """
    lines: list[str] = []
    for index in range(1, len(steps)):
        row = steps.iloc[index]
        before = steps.iloc[index - 1]["step"]
        if not int(steps.iloc[index - 1]["users"]):
            # Nobody got this far, so there is no rate: a step's 0% here would
            # read as one that loses everybody rather than one nobody saw.
            lines.append(
                f"**{before} \u2192 {row['step']}** \u2014 nobody reached "
                f"{before} in this window, so there is nothing to convert."
            )
            continue
        rate = float(row["from_previous"])
        sentence = (
            f"**{before} \u2192 {row['step']}** \u2014 {_percent(rate)}: "
            f"{_per_hundred(rate)} people who got as far as {before} went on; "
            f"{_per_hundred(1 - rate)} did not."
        )
        trend = _rate_delta(steps, previous, index, "from_previous")
        if trend == "flat":
            sentence += f" Unchanged on the previous {days} days."
        elif trend:
            direction = "Better" if trend.startswith("+") else "Worse"
            sentence += f" {direction} than the previous {days} days ({trend})."
        lines.append(sentence)
    if not lines:
        return
    with st.expander("What each step means", expanded=True):
        _said(TAB_BUSINESS, "Product funnel", lines)
        worst = steps.iloc[1:].sort_values("lost", ascending=False).iloc[0]
        st.markdown(
            f"The one to fix first is **{worst['step']}**: "
            f"{int(worst['lost']):,} people got as far as the step before it and "
            "no further."
        )


def _render_error_breakdowns(
    credentials: tuple[str, str, str], days: int
) -> None:
    """Where the errors happened and what they said.

    The count above says how many people were let down; on its own nobody can
    act on it. These two tables are the difference between "4% saw an error" and
    a ticket with a page and a message in it.
    """
    st.markdown("**Where the errors are**")
    columns = st.columns(len(amplitude_client.ERROR_BREAKDOWNS))
    for column, (title, prop) in zip(columns, amplitude_client.ERROR_BREAKDOWNS):
        with column:
            st.caption(title)
            try:
                with st.spinner("Reading..."):
                    frame = _breakdown_cached(
                        credentials, amplitude_client.ERROR_EVENT, prop, days
                    )
            except Exception as exc:  # noqa: BLE001
                st.caption(f"Could not read this breakdown: {str(exc)[:160]}")
                continue
            if frame.empty:
                st.caption(f"No {prop} recorded on these errors.")
                continue
            shown = frame.head(amplitude_client.BREAKDOWN_ROWS)
            st.dataframe(
                shown.assign(events=shown["events"].map(lambda n: f"{int(n):,}")).rename(
                    columns={"value": title, "events": "Times"}
                ),
                width="stretch",
                hide_index=True,
            )
            if len(frame) > len(shown):
                st.caption(
                    f"{len(frame) - len(shown):,} more values, "
                    f"{int(frame['events'].iloc[len(shown):].sum()):,} times between them."
                )
    st.caption(
        "Counted in times rather than people, so these add up: the same person "
        "meeting the same error twice is two things to fix. Messages carrying a "
        "build hash or an id are collapsed into one line, or a single broken "
        "deploy would fill the table with near-identical rows."
    )


def _render_event_counts(
    credentials: tuple[str, str, str],
    events: tuple[amplitude_client.Step, ...],
    days: int,
    visitors: int,
    empty_note: str,
    caption: str,
) -> None:
    """A count of people per event, beside what share of all visitors that is."""
    if not events:
        counts = pd.DataFrame(columns=["label", "event", "users"])
    else:
        try:
            with st.spinner("Counting..."):
                # One request per event either way - Amplitude has to dedupe each
                # separately - but run through _parallel so a tab's events go out
                # together instead of the one-at-a-time loop this used to leave to
                # amplitude_client.event_users.
                read = _parallel(
                    {
                        step.event: (
                            lambda label=step.label, event=step.event: _event_users_cached(
                                credentials, ((label, event),), days
                            )
                        )
                        for step in events
                    }
                )
                counts = pd.concat(
                    [read[step.event] for step in events], ignore_index=True
                )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not read these counts from Amplitude: {str(exc)[:200]}")
            return
    if counts.empty or not counts["users"].sum():
        st.info(empty_note)
        return
    share = counts["users"] / visitors if visitors else 0.0
    table = counts.assign(share=pd.Series(share).map(_percent)).sort_values(
        "users", ascending=False
    )
    st.dataframe(
        table.rename(
            columns={
                "label": "What happened",
                "event": "Event",
                "users": "People",
                "share": "Of all visitors",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(caption)
