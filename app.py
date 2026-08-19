"""Entry point and page router for the VinoVoss Jira dashboard.

Task 1C split this file: every page's rendering code now lives in its own
module under ``pages/`` (``today.py``, ``people.py``, ``delivery.py``,
``code.py``, ``planning.py``, ``engineering.py``, ``business.py``), and the
widget- and data-shaping helpers more than one page calls live in
``render_shared.py``. What is left here is exactly the navigation
framework: the page registry (``_PageSpec`` / ``_page_specs`` / ``_pages``),
``main()``, and the access gate.

Page modules are imported lazily, inside the dispatch wrapper for each page
below - never at this module's top level - so opening one page does not load
every other page's code, and so importing this module stays cheap. See
``docs/assumptions/1C.md`` for the extraction record and the reasoning
behind what stayed eager versus lazy.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, NamedTuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Local settings come from a .env file when one is present, so `streamlit run
# app.py` needs no exported variables. This runs before the local imports below
# because some of them read the environment at import time (change_audit picks
# its log path there). Existing variables win, so a real deployment's injected
# environment is never overridden by a file that happens to be lying around.
load_dotenv(override=False)

import read_log
import snapshot as board
import theme_html
import integrity
from access_gate import render_sign_out, require_password
from jira_client import JiraClient
from theme import inject_styles
from transformations import add_ticket_health_fields
from prioritization import add_priority_score
from hygiene import estimate_policy

# Re-exported so the apptest harnesses (tests/apptests/_*_harness.py) can
# monkeypatch e.g. ``app.cost_client.load_stripe_env`` before running a
# business-page apptest as a subprocess. These are the same module objects
# pages/business.py imports for itself - patching an attribute here patches
# it everywhere, because Python modules are singletons in sys.modules.
import ads_client
import amplitude_client
import cost_client
import github_client
import merchant_client
import orders_client

from data_layer import (
    SCOPE_INDIVIDUAL,
    SCOPE_ORG,
    SCOPE_TEAM,
    _ENGINEERING_DATA_AS_OF_KEY,
    _BoardDerivation,
    _CARRIED_PREFIX,
    _EngineeringData,
    _PersistedBoard,
    _SNAPSHOT_STALE_LIMIT_SECONDS,
    _board_fingerprint,
    _carried,
    _carried_roster,
    _carry,
    _delete_board_snapshot,
    _derive_board,
    _read_board_snapshot,
    _write_board_snapshot,
    fetch_person_resolved_history,
    requested_person,
)

from page_shared import (
    BOARD_ASKED_KEY,
    BOARD_FILE_KEY,
    BOARD_SLOT_KEY,
    REPORTS_KEY,
    TAB_BUSINESS,
    _deliver_board_snapshot,
    _number_or,
    _page_name,
    _reset_reports,
    _said,
    _text_or,
)

# render_shared is infrastructure shared by every page module, exactly like
# page_shared and data_layer above - not a page module itself, so importing
# it here does not violate "page modules are lazy". Its own top-level
# imports mirror what app.py carried before the split; see the module
# docstring in render_shared.py and docs/assumptions/1C.md for why.
from render_shared import (
    BACKLOG_STATUSES,
    BUSINESS_PAGE_TITLE,
    CODE_PAGE_TITLE,
    DELIVERY_PAGE_TITLE,
    ENGINEERING_PAGE_TITLE,
    PAGE_HEADINGS,
    PEOPLE_PAGE_TITLE,
    PLANNING_PAGE_TITLE,
    TODAY_NO_REVIEWER_DAYS,
    TODAY_PAGE_TITLE,
    TODAY_STALLED_DAYS,
    _TIER_BG,
    _clear_page_caches,
    _dated,
    _exclude_repos,
    _is_bot,
    _jira_ticket_url,
    _metrics_df,
    _NO_VALUE,
    _open_pr_signals,
    _people_only,
    _shown,
    _sprint_label,
    _stalled_rows,
    _truncation_note,
    _weekly_resolved_buckets,
    _workload_hours,
    annotated_board,
    engineer_page,
    person_link,
)

logger = logging.getLogger(__name__)


# Lazy import for pages.business - only imported when the Business page is
# accessed. Loading business.py's 4,300+ lines on every page load was the
# navigation delay PERFORMANCE_FIX.md documents; the same pattern is now
# used for every page below.
def _lazy_import_business():
    """Lazy import of pages.business module to improve performance."""
    from pages import business
    return business


def _business_readable() -> bool:
    """Check if business page is accessible (lazy wrapper)."""
    business = _lazy_import_business()
    return business._business_readable()


def _render_business() -> None:
    """Render the business page (lazy wrapper)."""
    business = _lazy_import_business()
    return business._render_business()


# Re-export business page functions for backward compatibility with tests
def _offers_together(frames):
    """Re-export from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business._offers_together(frames)


def _one_account_products(client, config, customer_id, days, today):
    """Re-export from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business._one_account_products(client, config, customer_id, days, today)


def _distinct_labels(titles):
    """Re-export from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business._distinct_labels(titles)


def _least_squares(x, y):
    """Re-export from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business._least_squares(x, y)


def _rate_delta(now, before, decimals, mode):
    """Re-export from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business._rate_delta(now, before, decimals, mode)


def _delta_arrow(change):
    """Re-export from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business._delta_arrow(change)


def _money_delta(change, currency):
    """Re-export from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business._money_delta(change, currency)


def _no_ad_products(read=True):
    """Re-export from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business._no_ad_products(read)


def _offer_sales_cached(source, days, today):
    """Re-export from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business._offer_sales_cached(source, days, today)


def _per_hundred(rate):
    """Re-export from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business._per_hundred(rate)


def _visible(offers, wines, sales):
    """Re-export from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business._visible(offers, wines, sales)


# Re-export CloudRead, AdsRead, BenchmarkRead, and AdProducts classes for backward compatibility with tests
def CloudRead(*args, **kwargs):
    """Re-export CloudRead from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business.CloudRead(*args, **kwargs)


def AdsRead(*args, **kwargs):
    """Re-export AdsRead from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business.AdsRead(*args, **kwargs)


def BenchmarkRead(*args, **kwargs):
    """Re-export BenchmarkRead from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business.BenchmarkRead(*args, **kwargs)


def AdProducts(*args, **kwargs):
    """Re-export AdProducts from pages.business for backward compatibility."""
    business = _lazy_import_business()
    return business.AdProducts(*args, **kwargs)


# Re-export business page constants for backward compatibility with tests
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


# --- Page dispatch: one lazy wrapper per engineering page --------------------
#
# Each wrapper imports its page module on first call only (the module is
# cached by Python after that), matching _render_business above exactly.
# _page_specs() below passes these functions to st.Page uncalled, so the
# import happens only when a reader actually opens that page.


def _render_today_page() -> None:
    """Render the Today page (lazy wrapper)."""
    from pages import today
    return today._render_today_page()


def _render_people_page() -> None:
    """Render the People page (lazy wrapper)."""
    from pages import people
    return people._render_people_page()


def _render_delivery_page() -> None:
    """Render the Delivery page (lazy wrapper)."""
    from pages import delivery
    return delivery._render_delivery_page()


def _render_code_page() -> None:
    """Render the Code page (lazy wrapper)."""
    from pages import code
    return code._render_code_page()


def _render_planning_page() -> None:
    """Render the Planning page (lazy wrapper)."""
    from pages import planning
    return planning._render_planning_page()


def _render_engineering_page() -> None:
    """Render the legacy combined Engineering page (lazy wrapper)."""
    from pages import engineering
    return engineering._render_engineering_page()


# --- Backward-compatible re-exports for page-private test helpers -----------
#
# _ownerless, _cycle_by_status and friends below are private to one page
# module (pages/today.py, pages/delivery.py, pages/code.py) rather than
# render_shared - nothing else in the app calls them. A handful of tests
# still reach them as app._name, so __getattr__ (PEP 562) resolves those
# names on first access rather than importing the owning page module at
# this module's top level, which would defeat the page's own laziness for
# every reader, not just the tests that ask for it.
_PAGE_PRIVATE_REEXPORTS = {
    "_ownerless": "pages.today",
    "_ownerless_rows": "pages.today",
    "_estimate_coverage": "pages.today",
    "_action_queues": "pages.today",
    "_ACTION_QUEUE_NAMES": "pages.today",
    "_cycle_by_status": "pages.delivery",
    "_stale_with_masked": "pages.delivery",
    "_stalled_count": "pages.delivery",
    "_team_prs": "pages.code",
    "_render_code_kpis": "pages.code",
}


def __getattr__(name: str):
    module_name = _PAGE_PRIVATE_REEXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, name)


class _PageSpec(NamedTuple):
    """One navigation entry, as data.

    The list is built as plain tuples rather than ``st.Page`` objects so the
    invariants worth pinning - exactly one default, no repeated ``url_path``, a
    heading for every title - can be asserted by a test without a Streamlit
    runtime. ``main`` turns them into pages unchanged.
    """

    render: Callable[[], None]
    title: str
    icon: str
    url_path: str
    default: bool = False
    hidden: bool = False


def _page_specs() -> list[_PageSpec]:
    """Every page this dashboard serves, in navigation order."""
    specs = [
        # Today is what a reader lands on. The single engineering page opened on
        # twenty tiles and two pies, and the finding that mattered - most open PRs
        # carry no approving review - sat below all of it. Today asks one question
        # ("what needs a decision now") and the pages behind it keep the detail.
        #
        # The default page is served at "/" and Streamlit forces its `url_path` to
        # "" - it is ignored, not honoured - so the path is stated as "" to say so.
        _PageSpec(_render_today_page, TODAY_PAGE_TITLE, ":material/priority_high:", "", default=True),
        _PageSpec(_render_people_page, PEOPLE_PAGE_TITLE, ":material/group:", "people"),
        _PageSpec(_render_delivery_page, DELIVERY_PAGE_TITLE, ":material/trending_up:", "delivery"),
        _PageSpec(_render_code_page, CODE_PAGE_TITLE, ":material/code:", "code"),
        _PageSpec(_render_planning_page, PLANNING_PAGE_TITLE, ":material/event_note:", "planning"),
        # The former single page, kept whole at the address people already have.
        # Links to /engineering are in Slack and in people's bookmarks.
        _PageSpec(
            _render_engineering_page,
            ENGINEERING_PAGE_TITLE,
            ":material/engineering:",
            "engineering",
        ),
    ]
    if _business_readable():
        specs.append(
            _PageSpec(_render_business, BUSINESS_PAGE_TITLE, ":material/storefront:", "business")
        )
    return specs


def _pages() -> list:
    """The specs as ``st.Page`` objects, for ``st.navigation``."""
    return [
        st.Page(
            spec.render,
            title=spec.title,
            icon=spec.icon,
            url_path=spec.url_path,
            **({"default": True} if spec.default else {}),
            **({"visibility": "hidden"} if spec.hidden else {}),
        )
        for spec in _page_specs()
    ]


def _board_age_caption() -> str | None:
    """How old the board behind the current page's figures is, or None off the Business page.

    Reads ``_ENGINEERING_DATA_AS_OF_KEY`` - the moment the data was actually
    read (a snapshot's written-at stamp, or a live gather's completion), not
    how long this session has merely been trusting it - so a snapshot-served
    cold start says what it is rather than claiming to be fresh.
    """
    as_of = st.session_state.get(_ENGINEERING_DATA_AS_OF_KEY)
    if as_of is None:
        return None
    age_seconds = max(0.0, time.time() - as_of)
    if age_seconds < 90:
        return "Board data: just read"
    minutes = int(age_seconds // 60)
    return f"Board data: {minutes}m old"


def _reads_debug_enabled() -> bool:
    """Off by default: the "reads this page made" expander costs nothing to
    leave in production, but only a reader who sets this env var sees it."""
    return os.environ.get("SHOW_PAGE_READS", "").strip().lower() in {"1", "true", "yes"}


def _render_page_reads(reads: list[dict]) -> None:
    if not reads:
        return
    hits = sum(1 for r in reads if r["cache"] == "hit")
    with st.expander(
        f"Reads this page made ({len(reads)}, {hits} from cache)", expanded=False
    ):
        st.dataframe(pd.DataFrame(reads), hide_index=True, width="stretch")


def main() -> None:
    # Neutral, because the browser tab is shared by both pages and the shop's
    # figures are not Jira ticket health. Each page says what it is in its own
    # heading below.
    st.set_page_config(page_title="VinoVoss Dashboard", layout="wide")
    require_password()
    inject_styles()

    # Pages rather than tabs. Streamlit runs the body of every tab on every
    # rerun, so the engineering sections were being rebuilt for readers looking
    # at the shop's figures and vice versa; a page that is not open does not run
    # at all, which is why the Business page no longer needs a button in front
    # of it and why opening it no longer waits for Jira.
    pages = _pages()
    page = st.navigation(pages, position="top")

    # The login is remembered in the browser for a month, so there has to be a
    # way to hand a shared laptop back without handing over Jira write access.
    render_sign_out()
    # Per page, because "Jira Ticket Health Dashboard" sat above the Business
    # page's orders, revenue and ad spend and was simply untrue there.
    st.title(PAGE_HEADINGS.get(page.title, "VinoVoss Dashboard"))
    if st.button("Refresh data", icon=":material/refresh:"):
        _clear_page_caches(page.title)
    # Reserved before the page runs and filled after: the age is not known
    # until the page's own read has happened, but the caption's *position*
    # belongs right under the button regardless of how long that read takes.
    age_slot = st.empty()

    # Gathered fresh on every run, because every figure the active page draws
    # is. Only one page runs per rerun, so this resets just its own report.
    _reset_reports()
    # Recording costs one call into the snapshot per drawing the page makes -
    # cheap once, not free at the scale of every reader on every rerun who
    # never asks for the board. Arming the button sets BOARD_ASKED_KEY on the
    # rerun the click causes; this run - the very next one - is the first that
    # needs to listen, and every run after it for as long as a file is held,
    # so the offer can still tell a stale board from the one on screen. A
    # reader who never presses the button never pays for any of it.
    board_wanted = bool(st.session_state.get(BOARD_ASKED_KEY)) or bool(
        st.session_state.get(BOARD_FILE_KEY)
    )
    with read_log.track_page_reads() as page_reads:
        if board_wanted:
            # The page draws exactly as it did before; the recorder only
            # listens, so that the reader who asks for the board as a file
            # gets the board they are looking at rather than a second
            # rendering of it.
            with board.recording(_page_name(page)) as recorded:
                page.run()
            _deliver_board_snapshot(recorded)
        else:
            page.run()

    age_caption = _board_age_caption()
    if age_caption:
        age_slot.caption(age_caption)
    if _reads_debug_enabled():
        _render_page_reads(page_reads)


if __name__ == "__main__":
    main()
