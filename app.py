from __future__ import annotations

import collections
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import datetime as _dt
import hashlib
import html
import logging
import os
import re
import threading
import time
from typing import Any, Callable, NamedTuple
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from dotenv import load_dotenv

# Local settings come from a .env file when one is present, so `streamlit run
# app.py` needs no exported variables. This runs before the local imports below
# because some of them read the environment at import time (change_audit picks
# its log path there). Existing variables win, so a real deployment's injected
# environment is never overridden by a file that happens to be lying around.
load_dotenv(override=False)

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
import github_client
import pr_hygiene
from capacity import (
    capacity_table,
    same_person,
    match_weekly_hours,
    parse_weekly_hours,
    working_days,
)
import cleanup
from cleanup import is_unowned
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
from theme import inject_styles, kpi_strip
import report as reporting
from hygiene import (
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
import orders
import orders_client
from prioritization import add_priority_score, assignee_rollup
import sprint_planner
import ticket_quality
from transformations import add_ticket_health_fields
import write_access


logger = logging.getLogger(__name__)


DEFAULT_JQL = """statusCategory != Done
ORDER BY updated ASC"""

# Credentials resolve from JIRA_BASE_URL/JIRA_EMAIL/JIRA_API_TOKEN when set,
# otherwise from this YAML profile. See README.md.
CREDS_PATH = os.getenv("JIRA_CREDS_PATH", DEFAULT_CREDS_PATH)
PROFILE_NAME = os.getenv("JIRA_PROFILE", DEFAULT_PROFILE_NAME)
JQL = os.getenv("JIRA_DASHBOARD_JQL", DEFAULT_JQL)
ORG_TEAM_MEMBERS = [
    name.strip()
    for name in os.getenv("JIRA_TEAM_MEMBERS", "Tam,Mehdi Ordikhani").split(",")
    if name.strip()
]

# Placeholder owner names Jira writes when nobody is assigned.
_NO_OWNER_NAMES = {"", "unassigned", "none"}


def _positive_int(value: str | None, *, default: int) -> int:
    """Read a positive integer setting; a typo costs the setting, not the app."""
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


SCOPE_ORG = "Organization"
SCOPE_TEAM = "Team"
SCOPE_INDIVIDUAL = "Individual"

ENGINEERING_PAGE_TITLE = "Engineering"
BUSINESS_PAGE_TITLE = "Business"

FETCH_SCHEMA_VERSION = 8
JIRA_KEY_DISPLAY_PATTERN = r".*/browse/([^/?#]+)$"

# One Jira request per key, so bound how many the sprint editor asks about.
TRANSITION_LOOKUP_LIMIT = 50
# Upper bound on how many tickets a bulk write-back pre-selects.
BULK_ACTION_DEFAULT_LIMIT = 25
# Slices a composition pie shows before the tail collapses into "Other".
MIX_SLICE_LIMIT = 10
# Ceiling on tickets fetched per run; org-wide JQL can exceed the old fixed 1000.
MAX_RESULTS = _positive_int(os.getenv("JIRA_MAX_RESULTS"), default=1000)
# Tickets per Jira page. Each page is a round trip, and at 100 the open-ticket
# fetch spent about three seconds of its ten just asking again; Jira's documented
# ceiling for the search endpoints is 100 for some fields but accepts larger
# pages here, and it is settable in case a tenant disagrees.
JIRA_PAGE_SIZE = _positive_int(os.getenv("JIRA_PAGE_SIZE"), default=250)
# Statuses the team treats as "resolved" for the top-of-page snapshot. Spans
# more than Jira's Done category (e.g. Ready for Production / Review in Staging
# are In Progress), so it is matched by status name. Override with a
# comma-separated JIRA_RESOLVED_STATUSES.
_DEFAULT_RESOLVED_STATUSES = (
    "Done",
    "Released",
    "Released to Production",
    "Ready for Production",
    "Review in Staging",
)
RESOLVED_STATUSES = tuple(
    s.strip()
    for s in os.getenv(
        "JIRA_RESOLVED_STATUSES", ",".join(_DEFAULT_RESOLVED_STATUSES)
    ).split(",")
    if s.strip()
)
# Triage-only by default: Backlog is a deliberate parking lot, not neglected intake,
# so it stays out of "stuck in triage". Override with a comma-separated
# JIRA_TRIAGE_STATUSES; JIRA_TRIAGE_STUCK_HOURS sets the threshold.
_DEFAULT_TRIAGE_STATUSES = ("Triage",)
TRIAGE_STATUSES = tuple(
    s.strip()
    for s in os.getenv(
        "JIRA_TRIAGE_STATUSES", ",".join(_DEFAULT_TRIAGE_STATUSES)
    ).split(",")
    if s.strip()
)
TRIAGE_STUCK_HOURS = _positive_int(os.getenv("JIRA_TRIAGE_STUCK_HOURS"), default=48)
# Statuses hidden when "Include Backlogs" is off; projects name their backlog differently.
BACKLOG_STATUSES = {
    name.strip().lower()
    for name in os.getenv("JIRA_BACKLOG_STATUSES", "Backlog").split(",")
    if name.strip()
}
# Weekly hours per person ("Tam=10,Jal=20"); Jira does not know who is part-time.
WEEKLY_HOURS = parse_weekly_hours(os.getenv("JIRA_WEEKLY_HOURS", ""))
# Which Jira projects make up each team ("Marketplace=MB;App=AS,OA").
TEAM_PROJECTS = parse_team_projects(os.getenv("JIRA_TEAM_PROJECTS", ""))
# Who sits on which team ("Design=Robert,Alesya;App=Ali,Farid"); people beat
# projects because part-timers here work across several projects.
TEAM_PEOPLE = parse_team_people(os.getenv("JIRA_TEAM_PEOPLE", DEFAULT_TEAM_PEOPLE))
_SCOPE_ASSIGNEES_KEY = "_scope_assignees"


def _default_browse_base() -> str:
    """Derive the ticket link prefix from the site the client actually reads from.

    Resolution mirrors JiraClient.resolve so links never point at a different
    site than the data: the environment only wins when it carries full
    credentials, otherwise the YAML profile does.
    """
    config = load_jira_env()
    if config is None:
        try:
            config = load_jira_profile(CREDS_PATH, PROFILE_NAME)
        except Exception:  # noqa: BLE001
            config = {"base_url": "https://vinovoss.atlassian.net"}
    return f"{normalize_base_url(str(config['base_url']))}/browse"


def _browse_base() -> str:
    """The ticket link prefix, refusing anything that is not plain http(s).

    The value is rendered as a clickable link in several tables, so a scheme
    like ``javascript:`` would turn every ticket key into a trap.
    """
    configured = os.getenv("JIRA_BROWSE_BASE", "").strip().rstrip("/")
    if configured and not configured.lower().startswith(("http://", "https://")):
        configured = ""
    return configured or _default_browse_base()


JIRA_BROWSE_BASE = _browse_base()


def _jira_ticket_url(key: str) -> str:
    """Generate a Jira ticket URL from its key.

    The key is quoted: it arrives from Jira rather than from this code, and a
    stray ``?`` or space would otherwise change which URL the link points at.
    """
    return f"{JIRA_BROWSE_BASE}/{quote(str(key).strip(), safe='')}"


def _normalize_sprint_id(value: object) -> str | None:
    """Convert sprint IDs to canonical string form (e.g. 2693.0 -> 2693)."""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return str(int(value)) if float(value).is_integer() else str(value).strip()

    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


# Every read below is cached the same way, and all of them refresh in the
# background: when the TTL lapses the reader gets the slightly stale answer
# immediately while the new one is fetched behind them, instead of one unlucky
# visitor every five minutes paying the full cold start for everybody else. The
# Refresh button is there for anyone who would rather wait for certainty.
@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_tickets(
    creds_path: str,
    profile_name: str,
    jql: str,
    max_results: int,
    page_size: int,
    schema_version: int,
) -> pd.DataFrame:
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    _ = schema_version
    result = client.search_issues(
        jql=jql,
        fields=DEFAULT_FIELDS,
        max_results=max_results,
        page_size=page_size,
        expand="changelog",
    )
    for col in [
        "sprint_id",
        "sprint_name",
        "sprint_state",
        "sprint_board_id",
        "sprint_start",
        "sprint_end",
        "project_key",
        "epic_key",
        "epic_summary",
    ]:
        if col not in result.columns:
            result[col] = pd.NA
    return result


# The snapshot lists at the top of the page render five columns and an age, and
# the resolved list only feeds the "who resolved tickets" pie. Asking for all
# seventeen default fields - including every ticket's full description - meant
# fetching megabytes to draw a pie chart of names.
LIST_FIELDS = ("summary", "status", "priority", "assignee", "created")
RESOLVED_FIELDS = ("assignee",)


def _jql_status_list(statuses: tuple[str, ...]) -> str:
    """Quote status names for a JQL ``IN``/``CHANGED TO`` clause.

    Escapes backslashes before quotes so a status value cannot break out of the
    quoted literal and inject JQL (values come from JIRA_RESOLVED_STATUSES).
    """
    return ", ".join(
        '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"' for s in statuses
    )


def _resolved_jql(statuses: tuple[str, ...], days: int, ordered: bool = True) -> str:
    """JQL for tickets that entered any ``statuses`` within the last ``days``."""
    jql = f"status CHANGED TO ({_jql_status_list(statuses)}) AFTER -{int(days)}d"
    return jql + " ORDER BY updated DESC" if ordered else jql


@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_resolved_count(
    creds_path: str,
    profile_name: str,
    days: int,
    statuses: tuple[str, ...],
    schema_version: int,
) -> int | None:
    """Jira's server-side count of tickets resolved in the window, never paging-capped.

    Uses ``/search/approximate-count``, which Jira documents as approximate for
    large result sets, so treat it as "Jira's count" rather than a guaranteed
    exact total. Returns ``None`` (rendered as "—") only if the count cannot be
    determined; an empty result is a real ``0``. The caller distinguishes the two.
    """
    _ = schema_version
    if not statuses:
        return 0
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.approximate_count(_resolved_jql(statuses, days, ordered=False))


def _created_jql(days: int, ordered: bool = True) -> str:
    """JQL for tickets created within the last ``days`` (org-wide, all projects)."""
    jql = f"created >= -{int(days)}d"
    return jql + " ORDER BY created DESC" if ordered else jql


def _triage_stuck_jql(
    statuses: tuple[str, ...], hours: int, ordered: bool = True
) -> str:
    """Tickets sitting in a triage status past ``hours``.

    "Stuck" = currently in one of ``statuses``, created more than ``hours`` ago,
    and with no status change in that window (so a just-created ticket, which has
    no status-change event, isn't falsely flagged). Approximates time-in-status
    without walking each changelog.
    """
    jql = (
        f"status in ({_jql_status_list(statuses)}) "
        f"AND created <= -{int(hours)}h AND NOT status CHANGED AFTER -{int(hours)}h"
    )
    return jql + " ORDER BY created ASC" if ordered else jql


@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_created_count(
    creds_path: str,
    profile_name: str,
    days: int,
    schema_version: int,
) -> int | None:
    """Jira's server-side count of tickets created in the window (never paging-capped)."""
    _ = schema_version
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.approximate_count(_created_jql(days, ordered=False))


@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_triage_stuck_count(
    creds_path: str,
    profile_name: str,
    statuses: tuple[str, ...],
    hours: int,
    schema_version: int,
) -> int | None:
    """Server-side count of tickets stuck in triage past ``hours`` (never paging-capped)."""
    _ = schema_version
    if not statuses:
        return 0
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.approximate_count(_triage_stuck_jql(statuses, hours, ordered=False))


@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_triage_stuck_tickets(
    creds_path: str,
    profile_name: str,
    statuses: tuple[str, ...],
    hours: int,
    max_results: int,
    page_size: int,
    schema_version: int,
) -> pd.DataFrame:
    """Tickets stuck in a triage status past ``hours``, oldest first (for the list)."""
    _ = schema_version
    if not statuses:
        return pd.DataFrame()
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.search_issues(
        jql=_triage_stuck_jql(statuses, hours),
        fields=list(LIST_FIELDS),
        max_results=max_results,
        page_size=page_size,
    )


@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_created_tickets(
    creds_path: str,
    profile_name: str,
    days: int,
    max_results: int,
    page_size: int,
    schema_version: int,
) -> pd.DataFrame:
    """Tickets created within the last ``days``, newest first (for the list)."""
    _ = schema_version
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.search_issues(
        jql=_created_jql(days),
        fields=list(LIST_FIELDS),
        max_results=max_results,
        page_size=page_size,
    )


@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_resolved_tickets(
    creds_path: str,
    profile_name: str,
    days: int,
    statuses: tuple[str, ...],
    max_results: int,
    page_size: int,
    schema_version: int,
) -> pd.DataFrame:
    """Tickets that transitioned INTO a "resolved" status within the last ``days``.

    "Resolved" here follows the team's definition (Done / Released / Ready for
    Production / Review in Staging), which spans more than Jira's Done category,
    so the query keys off the status-change event rather than status category.
    ``CHANGED TO ... AFTER`` matches a ticket that entered any resolved status in
    the window even if it later moved on. Used for the per-person pie; the
    headline tiles use :func:`fetch_resolved_count` so they never cap.
    """
    _ = schema_version
    if not statuses:
        return pd.DataFrame()
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.search_issues(
        jql=_resolved_jql(statuses, days),
        fields=list(RESOLVED_FIELDS),
        max_results=max_results,
        page_size=page_size,
    )


@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_open_prs_cached(token: str, org: str, schema_version: int) -> pd.DataFrame:
    _ = schema_version
    return github_client.fetch_open_prs(token, org)


@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_merged_prs_cached(token: str, org: str, days: int, schema_version: int) -> pd.DataFrame:
    _ = schema_version
    return github_client.fetch_merged_prs(token, org, days)


@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_merged_pr_count_cached(token: str, org: str, days: int, schema_version: int) -> int:
    _ = schema_version
    return github_client.merged_pr_count(token, org, days)


@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_open_pr_count_cached(token: str, org: str, schema_version: int) -> int:
    _ = schema_version
    return github_client.open_pr_count(token, org)


@st.cache_data(ttl=3600, show_spinner=False, refresh_mode="background")
def fetch_project_keys(creds_path: str, profile_name: str) -> list[str]:
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.get_project_keys()


@st.cache_data(ttl=600, show_spinner=False, refresh_mode="background")
def fetch_all_priorities(creds_path: str, profile_name: str) -> list[str]:
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.get_all_priorities()


@st.cache_data(ttl=600, show_spinner=False, refresh_mode="background")
def fetch_all_users(creds_path: str, profile_name: str) -> list[dict[str, str]]:
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.get_all_users()


@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_available_transition_statuses(
    creds_path: str,
    profile_name: str,
    issue_keys: tuple[str, ...],
) -> list[str]:
    """Every status the sampled tickets can legally move to.

    One Jira request per key is unavoidable, so they go out together rather
    than one after another; in series this was the single most expensive thing
    on the page.
    """
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    available: set[str] = set()
    for transitions in client.get_issue_transitions_bulk(issue_keys).values():
        for transition in transitions:
            to_status = str(transition.get("to_status", "")).strip()
            if to_status:
                available.add(to_status)
    return sorted(available)


# How often the loading bar is redrawn while the reads are outstanding, and how
# long a load has to be before it is drawn at all - a warm page answers in
# milliseconds and should not flash a progress bar at anybody.
_PROGRESS_TICK_SECONDS = 0.2
_PROGRESS_AFTER_SECONDS = 0.4
# Roughly how long the opening reads take on a cold load, and the only thing
# that paces the bar. It covers the reads rather than the whole page, because
# once the data is in the sections paint themselves down the screen and the
# reader can see that happening; the blank wait is this part. It is a pacing
# hint and never a promise, which is why no duration is shown to anyone: a slow
# Jira makes the bar wait at the ceiling, it does not make the bar lie.
_LOADING_PACE_SECONDS = 8.0
# The bar is not allowed to claim it has finished while anything is outstanding;
# a load that beats the pace waits at this mark rather than sitting full.
_LOADING_CEILING = 0.95


def _load_fraction(finished: int, total: int, elapsed: float) -> float:
    """Where to draw the bar: by the clock, not by the count.

    The reads are wildly unequal - a dozen of them answer inside the first
    second and the open-ticket query holds the page for several more - so a bar
    driven by how many have answered rushes to nine tenths and then sits there
    looking broken for most of the wait. The clock is the honest thing to
    animate; the label alongside it carries the real count.
    """
    if total and finished >= total:
        return _LOADING_CEILING
    return min(elapsed / _LOADING_PACE_SECONDS, _LOADING_CEILING)


def _gather(
    tasks: dict[str, Callable[[], Any]],
    on_progress: Callable[[float, str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Exception]]:
    """Run independent reads at the same time instead of one after another.

    Every query the page opens with is a separate call to Jira or GitHub that
    depends on none of the others, so waiting for each answer before asking the
    next question made the load as slow as their sum (about twenty seconds)
    rather than as slow as the longest of them.

    Each worker is given the calling script's run context, without which the
    ``st.cache_data`` wrappers inside these tasks would log a missing-context
    warning per call and could not read the session's cache. Failures are
    returned rather than raised, so one dead query costs its own section and
    not the page.

    ``on_progress`` is called from this thread - the one Streamlit lets draw -
    with a fraction and a label, so the caller can show how the load is going
    rather than leaving the reader watching an empty page.
    """
    context = get_script_run_ctx()

    def _run(task: Callable[[], Any]) -> Any:
        add_script_run_ctx(threading.current_thread(), context)
        return task()

    results: dict[str, Any] = {}
    errors: dict[str, Exception] = {}
    if not tasks:
        return results, errors

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_REQUESTS, len(tasks))) as pool:
        running = {pool.submit(_run, task): name for name, task in tasks.items()}
        total = len(running)
        pending = set(running)
        while pending:
            # A short wait rather than a blocking join, so the bar can be redrawn
            # while the slowest query is still out.
            done, pending = wait(
                pending, timeout=_PROGRESS_TICK_SECONDS, return_when=FIRST_COMPLETED
            )
            for future in done:
                name = running[future]
                try:
                    results[name] = future.result()
                except Exception as exc:  # noqa: BLE001
                    errors[name] = exc
                    results[name] = None
                    logger.warning("Dashboard read '%s' failed: %s", name, exc)
            elapsed = time.perf_counter() - started
            if on_progress and elapsed >= _PROGRESS_AFTER_SECONDS:
                answered = total - len(pending)
                on_progress(
                    _load_fraction(answered, total, elapsed),
                    f"Reading Jira and GitHub - {answered} of {total} answered",
                )
    logger.info(
        "Opening reads finished in %.2fs (%d of %d succeeded)",
        time.perf_counter() - started,
        len(results) - len(errors),
        len(tasks),
    )
    return results, errors


def _parallel(tasks: dict[str, Callable[[], Any]]) -> dict[str, Any]:
    """Run independent reads together, and raise if any of them fails.

    The sibling of ``_gather`` for a single section rather than a whole page.
    ``_gather`` returns failures instead of raising them, because one dead Jira
    query should cost its own panel and not the twenty beside it; a section that
    already catches and reports its own errors wants the opposite, so the first
    failure comes out here rather than leaving the caller to notice a blank.

    Workers are handed the calling script's run context for the same reason as
    there: without it the ``st.cache_data`` wrappers inside a task cannot see the
    session's cache and log a missing-context warning per call.
    """
    if not tasks:
        return {}
    context = get_script_run_ctx()

    def _run(task: Callable[[], Any]) -> Any:
        add_script_run_ctx(threading.current_thread(), context)
        return task()

    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_REQUESTS, len(tasks))) as pool:
        running = {name: pool.submit(_run, task) for name, task in tasks.items()}
        return {name: future.result() for name, future in running.items()}


def _engineering_reads(max_results: int, page_size: int) -> dict[str, Callable[[], Any]]:
    """The queries the engineering page opens with, as callables to run together."""
    return {
        "tickets": lambda: fetch_tickets(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
            jql=JQL,
            max_results=max_results,
            page_size=page_size,
            schema_version=FETCH_SCHEMA_VERSION,
        ),
        "resolved_count_7": lambda: fetch_resolved_count(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
            days=7,
            statuses=RESOLVED_STATUSES,
            schema_version=FETCH_SCHEMA_VERSION,
        ),
        "resolved_count_30": lambda: fetch_resolved_count(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
            days=30,
            statuses=RESOLVED_STATUSES,
            schema_version=FETCH_SCHEMA_VERSION,
        ),
        "resolved_30": lambda: fetch_resolved_tickets(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
            days=30,
            statuses=RESOLVED_STATUSES,
            max_results=max_results,
            page_size=page_size,
            schema_version=FETCH_SCHEMA_VERSION,
        ),
        "created_count_1": lambda: fetch_created_count(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
            days=1,
            schema_version=FETCH_SCHEMA_VERSION,
        ),
        "created_count_7": lambda: fetch_created_count(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
            days=7,
            schema_version=FETCH_SCHEMA_VERSION,
        ),
        "created_7": lambda: fetch_created_tickets(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
            days=7,
            max_results=max_results,
            page_size=page_size,
            schema_version=FETCH_SCHEMA_VERSION,
        ),
        "triage_stuck_count": lambda: fetch_triage_stuck_count(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
            statuses=TRIAGE_STATUSES,
            hours=TRIAGE_STUCK_HOURS,
            schema_version=FETCH_SCHEMA_VERSION,
        ),
        "triage_stuck": lambda: fetch_triage_stuck_tickets(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
            statuses=TRIAGE_STATUSES,
            hours=TRIAGE_STUCK_HOURS,
            max_results=max_results,
            page_size=page_size,
            schema_version=FETCH_SCHEMA_VERSION,
        ),
    }


# Names of the GitHub reads, so a failure in any of them can put the PR sections
# into their "GitHub could not be read" state rather than their "empty" one.
_GITHUB_READS = (
    "open_prs",
    "open_pr_count",
    "merged_prs",
    "merged_count_7",
    "merged_count_30",
)


def _github_reads(token: str, org: str) -> dict[str, Callable[[], Any]]:
    """The GitHub half of the opening reads, keyed the same way as the Jira half."""
    return {
        "open_prs": lambda: fetch_open_prs_cached(token, org, FETCH_SCHEMA_VERSION),
        "open_pr_count": lambda: fetch_open_pr_count_cached(
            token, org, FETCH_SCHEMA_VERSION
        ),
        "merged_prs": lambda: fetch_merged_prs_cached(token, org, 30, FETCH_SCHEMA_VERSION),
        "merged_count_7": lambda: fetch_merged_pr_count_cached(
            token, org, 7, FETCH_SCHEMA_VERSION
        ),
        "merged_count_30": lambda: fetch_merged_pr_count_cached(
            token, org, 30, FETCH_SCHEMA_VERSION
        ),
    }


def _metrics_df(df: pd.DataFrame, include_backlogs: bool) -> pd.DataFrame:
    if include_backlogs or "status" not in df.columns:
        return df
    statuses = df["status"].fillna("").astype(str).str.strip().str.lower()
    return df[~statuses.isin(BACKLOG_STATUSES)]


TAB_ENGINEERING = "Engineering"
TAB_BUSINESS = "Business"
REPORTS_KEY = "tab_reports"


def _report(tab: str) -> reporting.Report:
    """The report being gathered for one tab on this run.

    Streamlit redraws a tab from scratch on every interaction, so the figures
    are collected afresh each run and the file offered for download is always
    the page as it currently reads.
    """
    reports = st.session_state.setdefault(REPORTS_KEY, {})
    return reports.setdefault(tab, reporting.Report(tab))


def _reset_reports() -> None:
    st.session_state[REPORTS_KEY] = {}


def _tile(column, tab: str, section: str, label: str, value: str, **kwargs) -> None:
    """One metric, drawn on the page and kept for the printable report."""
    _report(tab).figure(section, label, value, str(kwargs.get("delta") or ""))
    column.metric(label, value, **kwargs)


def _kpis(tab: str, section: str, cards: list[tuple[str, str, str, str]]) -> None:
    """A KPI strip, drawn on the page and kept for the printable report."""
    for label, value, note, _accent in cards:
        _report(tab).figure(section, label, value, note)
    kpi_strip(cards)


def _said(tab: str, section: str, lines: list[str]) -> None:
    """Verdict lines, drawn as bullets and kept for the printable report."""
    for line in lines:
        _report(tab).note(section, line)
    # Escaped where it is drawn, plain in the report: these lines are money
    # sentences, and Streamlit reads a pair of dollar signs on one line as inline
    # maths - "$3,669 earned, $39 refunded" rendered as italics with both symbols
    # eaten. The report is HTML and wants the dollars as written.
    st.markdown("\n".join(f"- {_unmathed(line)}" for line in lines))


def _download_report(slot, tab: str) -> None:
    """Offer this tab's figures as a page that prints to a PDF.

    Drawn into a slot reserved before the sections run, because the file can
    only be built once they have: the button sits at the top of the tab where a
    reader looks for it, and holds what the whole tab ended up saying.
    """
    built = _report(tab)
    if built.empty:
        return
    slot.download_button(
        "Download report",
        data=built.html(),
        file_name=built.filename(),
        mime="text/html",
        key=f"download_{tab.lower()}",
        help="A one-page summary of this tab. Open it and print to PDF.",
    )


def _render_metrics(
    df: pd.DataFrame,
    include_backlogs: bool = False,
    *,
    unassigned_source: pd.DataFrame | None = None,
) -> None:
    """The headline numbers for the current scope.

    ``unassigned_source`` is the same data before the assignee scope filter. No
    unowned ticket can match a selected assignee, so within Team or Individual
    scope the unassigned count is structurally zero - a green zero reading as
    "nothing is ownerless" when the truth is "ownerless work is out of view".
    Counting it from the pre-scope frame, and saying so on the card, keeps the
    number honest.
    """
    metrics_df = _metrics_df(df, include_backlogs)
    total_open = int(len(metrics_df))
    avg_idle = float(metrics_df["idle_days"].mean()) if total_open else 0.0
    max_idle = float(metrics_df["idle_days"].max()) if total_open else 0.0
    oldest = float(metrics_df["ticket_age_days"].max()) if total_open else 0.0

    estimated_tickets = 0
    if "estimate_seconds" in metrics_df.columns and total_open:
        estimated_tickets = int(
            pd.to_numeric(metrics_df["estimate_seconds"], errors="coerce").fillna(0).gt(0).sum()
        )
    elif "original_estimate" in metrics_df.columns and total_open:
        estimate_text = metrics_df["original_estimate"].fillna("").astype(str).str.strip()
        estimated_tickets = int(estimate_text.ne("").sum())
    estimate_coverage_pct = (estimated_tickets / total_open * 100.0) if total_open else 0.0

    _LATE_STAGE_STATUSES = {"IN DEV ENV", "Review in Staging", "Ready for Production"}
    _STALE_THRESHOLD_DAYS = 6
    stale_late_stage = 0
    if "status" in metrics_df.columns and "idle_days" in metrics_df.columns and total_open:
        stale_late_stage = int(
            (
                metrics_df["status"].fillna("").astype(str).isin(_LATE_STAGE_STATUSES)
                & (metrics_df["idle_days"] > _STALE_THRESHOLD_DAYS)
            ).sum()
        )

    idle_30 = 0
    if total_open:
        idle_30 = int(pd.to_numeric(metrics_df["idle_days"], errors="coerce").fillna(0).ge(30).sum())

    out_of_scope = unassigned_source is not None
    owner_df = _metrics_df(unassigned_source, include_backlogs) if out_of_scope else metrics_df
    unassigned = 0
    if len(owner_df):
        owners = owner_df["assignee"].fillna("").astype(str).str.strip().str.lower()
        unassigned = int(owners.isin({"", "unassigned", "none"}).sum())

    _kpis(
        TAB_ENGINEERING,
        "Ticket health",
        [
            ("Open tickets", f"{total_open}", "current scope", "neutral"),
            (
                "Stalled 30d+",
                f"{idle_30}",
                f"{idle_30 / total_open * 100:.0f}% of open" if total_open else "—",
                "danger" if idle_30 else "good",
            ),
            (
                "Unassigned",
                f"{unassigned}",
                "no owner set, so outside this scope" if out_of_scope else "no owner set",
                "warning" if unassigned else "good",
            ),
            (
                "Estimate coverage",
                f"{estimate_coverage_pct:.0f}%",
                "tickets carrying an estimate",
                "good" if estimate_coverage_pct >= 80 else "warning",
            ),
            (
                "Stale late stage",
                f"{stale_late_stage}",
                "in dev/staging/prod, idle >6d",
                "danger" if stale_late_stage else "good",
            ),
            ("Oldest ticket", f"{oldest:.0f}d", f"avg idle {avg_idle:.0f}d · max {max_idle:.0f}d", "info"),
        ]
    )

    if "status" in metrics_df.columns and total_open:
        _render_status_pills(metrics_df["status"])


def _roster_matches(roster: list[str], assignees: list[str]) -> list[str]:
    """The Jira assignees named by a configured roster.

    Matched the same loose way as everywhere else, because the roster is written
    by hand: a configured "Mehdi Ordikhani" has to select "Mehdi Ordikhani Fard"
    rather than quietly selecting nobody, which would empty the whole view.
    """
    return [
        name
        for name in assignees
        if any(same_person(member, name) for member in roster)
    ]


def _resolve_scope_assignees(scope: str, assignees: list[str]) -> list[str] | None:
    """Return the assignees to filter on for the selected scope.

    None means "no assignee filter" (organization-wide); a list, including an
    empty one, is an explicit selection.
    """
    if scope == SCOPE_ORG:
        st.caption(f"Organization-wide view across {len(assignees)} assignee(s).")
        return None

    if scope == SCOPE_TEAM:
        defaults = _roster_matches(ORG_TEAM_MEMBERS, assignees)
        selected = st.multiselect("Team members", options=assignees, default=defaults)
        if not selected:
            st.warning("No team members selected - showing no tickets.")
        return selected

    if not assignees:
        st.warning("No assignees available in the current data.")
        return []
    matches = _roster_matches(ORG_TEAM_MEMBERS, assignees)
    default_individual = matches[0] if matches else assignees[0]
    selected = st.selectbox(
        "Assignee",
        options=assignees,
        index=assignees.index(default_individual),
    )
    return [selected]


# --- Per-assignee drill-down: attention tiers --------------------------------

# Four attention tiers colour each engineer's board. Red is most urgent, purple
# is parked low-priority work. Thresholds are deliberately simple so a person
# can reason about their own colour at a glance; tune them here if the team's
# definition of "stale" changes.
_TIER_RED = "Needs attention"
_TIER_YELLOW = "Watch"
_TIER_GREEN = "Healthy"
_TIER_PURPLE = "Low priority"

# Sort order used when ranking by tier: red first, purple last.
_TIER_ORDER = {_TIER_RED: 0, _TIER_YELLOW: 1, _TIER_GREEN: 2, _TIER_PURPLE: 3}

# Light backgrounds so the row text stays readable.
_TIER_BG = {
    _TIER_RED: "#f8d7da",
    _TIER_YELLOW: "#fff3cd",
    _TIER_GREEN: "#d4edda",
    _TIER_PURPLE: "#e7d9f5",
}

# Jira priorities that park a ticket rather than escalate it.
_LOW_PRIORITY_NAMES = {"low", "lowest", "idea"}
_TIER_RED_SCORE = 60.0
_TIER_RED_IDLE = 90.0
_TIER_YELLOW_SCORE = 30.0
_TIER_YELLOW_IDLE = 30.0

# Keyword heuristic for "could Devin pick this up?". Engineering execution work
# with a clear code surface leans yes; product/design/content/coordination work
# leans no. Mixed or empty signals stay "Maybe" so nobody trusts it blindly - it
# is a hint to start the conversation, not an automated assignment.
# Matched as whole words (see ``_kw_hit``) so short tokens do not collide with
# unrelated words ("test" in "latest", "api" in "capital", "search" in
# "research"). Deliberately truncated stems live in ``_DEVIN_*_PREFIXES`` and
# match as prefixes so "migrat" still catches "migration"/"migrate".
_DEVIN_YES_WORDS = (
    "bug", "fix", "error", "crash", "exception", "refactor", "endpoint",
    "api", "backend", "frontend", "unit test", "test", "upgrade", "dependency",
    "ssr", "cache", "database", "postgres", "mongo", "sql", "query",
    "script", "pipeline", "build", "lint", "typing", "performance", "latency",
    "resource", "config", "infra", "deploy", "ranker", "search", "index",
    "schema", "logging", "timeout", "rate limit", "webhook", "parser",
    "validation",
)
_DEVIN_YES_PREFIXES = ("migrat", "integrat")
_DEVIN_NO_WORDS = (
    "design", "mockup", "wireframe", "logo", "brand", "copywriting", "blog",
    "article", "story", "series", "trend", "instagram", "social", "campaign",
    "marketing", "video", "interview", "research", "survey", "meeting",
    "discussion", "strategy", "hiring", "roadmap", "pricing", "content",
)
_DEVIN_NO_PREFIXES: tuple[str, ...] = ()


def _kw_hit(text: str, words: tuple[str, ...], prefixes: tuple[str, ...]) -> bool:
    """True if any whole word in ``words`` or any stem in ``prefixes`` is in text."""
    for p in prefixes:
        if re.search(rf"\b{re.escape(p)}", text):
            return True
    for w in words:
        if re.search(rf"\b{re.escape(w)}\b", text):
            return True
    return False


def _devin_can_handle(row: pd.Series) -> str:
    """Rough hint at whether Devin could take a ticket, from its text signals."""
    issue_type = str(row.get("issue_type") or "").strip().lower()
    summary = str(row.get("summary") or "").lower()
    # YES may use the issue type as a signal (e.g. a Bug), but the NO scan runs
    # on the summary only: Jira type names like "Story"/"Design" collide with the
    # NO keywords and would mislabel ordinary engineering tickets otherwise.
    yes = issue_type == "bug" or _kw_hit(
        f"{summary} {issue_type}", _DEVIN_YES_WORDS, _DEVIN_YES_PREFIXES
    )
    no = _kw_hit(summary, _DEVIN_NO_WORDS, _DEVIN_NO_PREFIXES)
    if yes and not no:
        return "Yes"
    if no and not yes:
        return "No"
    return "Maybe"


def _attention_tier(row: pd.Series) -> str:
    """Bucket a ticket into one of four attention tiers for the drill-down."""
    priority = str(row.get("priority") or "").strip().lower()
    if priority in _LOW_PRIORITY_NAMES:
        return _TIER_PURPLE
    score = float(row.get("priority_score") or 0.0)
    idle = float(row.get("idle_days") or 0.0)
    if score >= _TIER_RED_SCORE or idle >= _TIER_RED_IDLE:
        return _TIER_RED
    if score >= _TIER_YELLOW_SCORE or idle >= _TIER_YELLOW_IDLE:
        return _TIER_YELLOW
    return _TIER_GREEN


def _tier_legend_html() -> str:
    """A small colour legend so the tiers read without guessing."""
    items = [
        (_TIER_BG[_TIER_RED], "Needs attention — high score or idle 90d+"),
        (_TIER_BG[_TIER_YELLOW], "Watch — moderate score or idle 30d+"),
        (_TIER_BG[_TIER_GREEN], "Healthy — recent, low pressure"),
        (_TIER_BG[_TIER_PURPLE], "Low priority — parked (Low / Lowest / Idea)"),
    ]
    spans = "".join(
        '<span style="display:inline-block;margin:0 14px 4px 0;">'
        f'<span style="display:inline-block;width:12px;height:12px;background:{color};'
        'border:1px solid #999;vertical-align:middle;margin-right:5px;"></span>'
        f"{html.escape(label)}</span>"
        for color, label in items
    )
    return f'<div style="font-size:0.85rem;margin:2px 0 10px;">{spans}</div>'


def _render_assignee_detail(df: pd.DataFrame, assignee: str) -> None:
    """The per-person ticket board: tier-coloured and sortable."""
    owners = df["assignee"].fillna("").astype(str).str.strip()
    target = str(assignee).strip()
    if target.lower() in _NO_OWNER_NAMES or target.lower() == "(no owner)":
        mask = owners.str.lower().isin(_NO_OWNER_NAMES)
    else:
        mask = owners == target
    owned = df[mask].copy()
    if owned.empty:
        st.info(f"No tickets for {assignee} in the current scope.")
        return

    if "has_estimate" not in owned.columns:
        owned = estimate_policy(owned, BACKLOG_STATUSES)
    owned["tier"] = owned.apply(_attention_tier, axis=1)
    owned["devin"] = owned.apply(_devin_can_handle, axis=1)
    owned["has_estimate_label"] = owned["has_estimate"].map(
        lambda value: "Yes" if bool(value) else "No"
    )
    owned["key_url"] = owned["key"].map(_jira_ticket_url)
    owned["_tier_order"] = owned["tier"].map(_TIER_ORDER).fillna(9)
    for column in ("created", "updated"):
        if column in owned.columns:
            owned[column] = (
                pd.to_datetime(owned[column], utc=True, errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .fillna("")
            )

    st.markdown(f"#### {assignee} — {len(owned)} open ticket(s)")
    st.markdown(_tier_legend_html(), unsafe_allow_html=True)
    st.caption("Click any column header to sort, or pick a field below. The key links to Jira.")

    sort_options = {
        "Attention tier": "_tier_order",
        "Priority score": "priority_score",
        "Idle (days)": "idle_days",
        "Age (days)": "ticket_age_days",
        "Created": "created",
        "Updated": "updated",
        "Status": "status",
        "Severity (priority)": "priority",
        "Has estimate": "has_estimate_label",
        "Devin-able?": "devin",
    }
    sort_options = {label: col for label, col in sort_options.items() if col in owned.columns}
    sort_col, dir_col = st.columns([3, 1])
    sort_label = sort_col.selectbox(
        "Sort by", list(sort_options), index=0, key=f"detail_sort_{assignee}"
    )
    ascending = (
        dir_col.selectbox("Order", ["asc", "desc"], index=0, key=f"detail_dir_{assignee}")
        == "asc"
    )
    owned = owned.sort_values(sort_options[sort_label], ascending=ascending, kind="stable")

    display_cols = [
        "key_url",
        "summary",
        "status",
        "priority",
        "priority_score",
        "tier",
        "idle_days",
        "ticket_age_days",
        "created",
        "updated",
        "has_estimate_label",
        "devin",
    ]
    display_cols = [c for c in display_cols if c in owned.columns]
    display = owned[display_cols]

    def _row_style(row: pd.Series) -> list[str]:
        background = _TIER_BG.get(row.get("tier", ""), "")
        style = f"background-color: {background}" if background else ""
        return [style] * len(row)

    styled = display.style.apply(_row_style, axis=1)
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        column_config={
            "key_url": st.column_config.LinkColumn("Key", display_text=JIRA_KEY_DISPLAY_PATTERN),
            "summary": st.column_config.TextColumn("Summary", width="large"),
            "status": st.column_config.TextColumn("Status"),
            "priority": st.column_config.TextColumn("Priority"),
            "priority_score": st.column_config.NumberColumn("Score", format="%.1f"),
            "tier": st.column_config.TextColumn("Tier"),
            "idle_days": st.column_config.NumberColumn("Idle (days)", format="%.1f"),
            "ticket_age_days": st.column_config.NumberColumn("Age (days)", format="%.1f"),
            "created": st.column_config.TextColumn("Created"),
            "updated": st.column_config.TextColumn("Updated"),
            "has_estimate_label": st.column_config.TextColumn("Estimate?"),
            "devin": st.column_config.TextColumn(
                "Devin-able?",
                help=(
                    "Heuristic hint from the ticket text: Yes = clear engineering "
                    "execution work, No = product/design/content, Maybe = unclear. "
                    "A starting point, not an automated assignment."
                ),
            ),
        },
    )


@st.fragment
def _render_mix(df: pd.DataFrame) -> None:
    """Composition of the tickets currently in view, as a share rather than a count.

    The other charts answer "how much" and "who"; this answers "of what" - which
    part of the work a reader is looking at before they read any table. It shows
    the filtered scope, Backlog statuses included only when the sidebar is, so it
    always agrees with the headline tiles above it.
    """
    st.subheader("Ticket Composition")
    if df.empty:
        st.info("No tickets in the current scope.")
        return

    # Team is derived, not a Jira field, so it has to be attached before it can
    # be offered as a slice.
    df = add_team(df, TEAM_PROJECTS, TEAM_PEOPLE)
    dimensions = {
        "Status": "status",
        "Team": "team",
        "Priority": "priority",
        "Issue type": "issue_type",
        "Assignee": "assignee",
        "Project": "project_key",
    }
    available = {
        label: column for label, column in dimensions.items() if column in df.columns
    }
    if not available:
        st.info("No breakdown fields available in the current data.")
        return

    label = st.radio(
        "Break down by",
        options=list(available.keys()),
        horizontal=True,
        key="mix_dimension",
    )
    st.caption(
        "Follows the sidebar scope and filters, including whether Backlog "
        "statuses are shown."
    )
    counts = (
        df[available[label]]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace({"": "Unknown"})
        .value_counts()
    )
    # A pie with forty slices is a colour wheel, not a chart: keep the shape of
    # the backlog readable and let the tail sit in one honest "Other" slice.
    top = counts.head(MIX_SLICE_LIMIT)
    remainder = int(counts.iloc[MIX_SLICE_LIMIT:].sum())
    if remainder:
        top = pd.concat(
            [top, pd.Series({f"Other ({len(counts) - MIX_SLICE_LIMIT})": remainder})]
        )

    mix = top.rename_axis(label).reset_index(name="tickets")
    figure = px.pie(mix, names=label, values="tickets", hole=0.45)
    figure.update_traces(textposition="inside", textinfo="percent+label")
    figure.update_layout(height=420, legend_title_text=label)
    left, right = st.columns([3, 2])
    left.plotly_chart(figure, width="stretch")
    right.dataframe(
        mix.assign(share=(mix["tickets"] / mix["tickets"].sum() * 100).round(1)),
        width="stretch",
        hide_index=True,
        column_config={
            "tickets": st.column_config.NumberColumn("Tickets"),
            "share": st.column_config.NumberColumn("Share %", format="%.1f"),
        },
    )


@st.fragment
def _render_scope_breakdown(df: pd.DataFrame, scope: str, include_backlogs: bool) -> None:
    """Render the per-assignee roll-up that backs org-wide and individual views."""
    scoped = _metrics_df(df, include_backlogs)
    rollup = assignee_rollup(scoped)

    st.subheader("Assignee Breakdown")
    if rollup.empty:
        st.info("No tickets in the current scope.")
        return

    if scope == SCOPE_INDIVIDUAL and len(rollup) == 1:
        row = rollup.iloc[0]
        st.markdown(f"**{row['assignee']}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Open Tickets", int(row["open_tickets"]))
        c2.metric("Avg Priority Score", f"{row['avg_priority_score']:.1f}")
        c3.metric("Idle 15d+", int(row["stale_15d_plus"]))
        c4.metric("Unprioritized", int(row["unprioritized"]))
        _render_assignee_detail(scoped, str(row["assignee"]))
        return

    st.caption("Click a row to open that person's ticket board below.")
    event = st.dataframe(
        rollup,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="assignee_rollup_select",
        column_config={
            "assignee": st.column_config.TextColumn("Assignee"),
            "open_tickets": st.column_config.NumberColumn("Open"),
            "avg_priority_score": st.column_config.NumberColumn("Avg Score", format="%.1f"),
            "top_priority_score": st.column_config.NumberColumn("Top Score", format="%.1f"),
            "avg_idle_days": st.column_config.NumberColumn("Avg Idle (days)", format="%.1f"),
            "max_idle_days": st.column_config.NumberColumn("Max Idle (days)", format="%.1f"),
            "stale_15d_plus": st.column_config.NumberColumn("Idle 15d+"),
            "unprioritized": st.column_config.NumberColumn("No Priority"),
        },
    )
    st.bar_chart(rollup.set_index("assignee")["open_tickets"], height=260)

    selected_rows = (event or {}).get("selection", {}).get("rows", [])
    if selected_rows:
        index = selected_rows[0]
        if 0 <= index < len(rollup):
            st.divider()
            _render_assignee_detail(scoped, str(rollup.iloc[index]["assignee"]))
    else:
        st.caption("Select an assignee above to see their tickets.")


@st.fragment
def _render_priority_queue(df: pd.DataFrame, include_backlogs: bool) -> None:
    """Rank tickets by the composite priority score so work can be picked top-down."""
    scoped = _metrics_df(df, include_backlogs)

    st.subheader("Prioritized Queue")
    st.caption(
        "Score (0-100) = Jira priority + idle time + ticket age + sprint carry-over "
        "+ due-date urgency + late-stage staleness. See README.md."
    )
    if scoped.empty:
        st.info("No tickets in the current scope.")
        return

    top_n = st.slider("Tickets to show", min_value=5, max_value=100, value=25, step=5)
    queue = scoped.sort_values("priority_score", ascending=False).head(top_n).copy()
    queue["key_url"] = queue["key"].map(_jira_ticket_url)

    columns = [
        "key_url",
        "summary",
        "assignee",
        "status",
        "priority",
        "priority_score",
        "priority_reasons",
        "idle_days",
        "ticket_age_days",
    ]
    st.dataframe(
        queue[columns],
        width="stretch",
        hide_index=True,
        column_config={
            "key_url": st.column_config.LinkColumn("Key", display_text=JIRA_KEY_DISPLAY_PATTERN),
            "summary": st.column_config.TextColumn("Summary"),
            "assignee": st.column_config.TextColumn("Assignee"),
            "status": st.column_config.TextColumn("Status"),
            "priority": st.column_config.TextColumn("Jira Priority"),
            "priority_score": st.column_config.ProgressColumn(
                "Score",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
            "priority_reasons": st.column_config.TextColumn("Why"),
            "idle_days": st.column_config.NumberColumn("Idle (days)", format="%.1f"),
            "ticket_age_days": st.column_config.NumberColumn("Age (days)", format="%.1f"),
        },
    )
    st.download_button(
        "Download prioritized queue (CSV)",
        data=queue[columns[1:]].assign(key=queue["key"]).to_csv(index=False).encode("utf-8"),
        file_name="jira_prioritized_queue.csv",
        mime="text/csv",
    )


@st.fragment
def _render_team_overview(df: pd.DataFrame) -> None:
    """Per-team load, staffing and sprint state, so each squad is legible alone."""
    st.subheader("Teams")
    if TEAM_PEOPLE:
        fallback = (
            "JIRA_TEAM_PROJECTS" if TEAM_PROJECTS else "the Jira project key"
        )
        st.caption(
            f"A ticket's team follows its assignee (JIRA_TEAM_PEOPLE, "
            f"{len(TEAM_PEOPLE)} people), falling back to {fallback} for anyone "
            "off the roster. Tickets with no owner are grouped as "
            f'"{NO_OWNER_TEAM}" rather than credited to a team.'
        )
    elif TEAM_PROJECTS:
        st.caption("Team membership comes from JIRA_TEAM_PROJECTS.")
    else:
        st.caption(
            "Teams default to Jira project keys. Group them with "
            'JIRA_TEAM_PEOPLE, e.g. "Design=Robert,Alesya;App=Ali,Farid".'
        )

    scored = add_team(estimate_policy(df, BACKLOG_STATUSES), TEAM_PROJECTS, TEAM_PEOPLE)
    summary = team_summary(scored)
    if summary.empty:
        st.info("No tickets in the current scope.")
        return

    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
        column_config={
            "team": st.column_config.TextColumn("Team"),
            "open": st.column_config.NumberColumn("Open"),
            "people": st.column_config.NumberColumn("People"),
            "avg_idle": st.column_config.NumberColumn("Avg idle (days)", format="%.1f"),
            "idle_30d": st.column_config.NumberColumn("Stalled 30d+"),
            "unassigned": st.column_config.NumberColumn("Unassigned"),
            "no_estimate": st.column_config.NumberColumn("No estimate"),
        },
    )

    team_names = summary["team"].tolist()
    chosen = st.selectbox("Current sprint for team", options=team_names)
    team_df = scored[scored["team"] == chosen]

    states = team_df["sprint_state"].fillna("").astype(str).str.lower()
    active = team_df[states.eq("active")]
    if active.empty:
        active = team_df[states.eq("future")]
    if active.empty:
        st.caption(
            f"{chosen} has no ticket in an active or future sprint - "
            f"all {len(team_df)} open tickets sit outside a sprint."
        )
        return

    sprint_label = (
        active["sprint_name"].fillna("").astype(str).mode().iat[0]
        if not active["sprint_name"].dropna().empty
        else "current sprint"
    )
    current = active[active["sprint_name"].fillna("").astype(str).eq(sprint_label)]
    owners = current["assignee"].fillna("").astype(str).str.strip().str.lower()
    unowned = owners.isin(_NO_OWNER_NAMES)
    _kpis(
        TAB_ENGINEERING,
        "Sprint",
        [
            ("Sprint", sprint_label, chosen, "info"),
            ("Tickets", f"{len(current)}", "in this sprint", "neutral"),
            (
                "People",
                f"{current.loc[~unowned, 'assignee'].nunique()}",
                "with sprint work",
                "neutral",
            ),
            (
                "Committed",
                f"{current['estimate_hours'].sum():.0f}h",
                "estimated hours",
                "neutral",
            ),
            (
                "Unassigned",
                f"{int(unowned.sum())}",
                "no owner in sprint",
                "warning" if unowned.any() else "good",
            ),
        ]
    )
    st.bar_chart(
        current.groupby(current["status"].fillna("Unknown").astype(str))["key"].count(),
        height=240,
    )
    st.caption(
        f"{chosen}: {len(team_df)} open tickets total, {len(current)} in {sprint_label}. "
        "Per-person hours are in Availability vs Commitment below."
    )


@st.fragment
def _render_epics(df: pd.DataFrame, organization_source: pd.DataFrame | None = None) -> None:
    """Group open work by epic and name what is wrong with each one."""
    st.subheader("Epics")
    st.caption(
        "Open children only - the dashboard does not load Done tickets, so this is "
        "remaining work per epic, not completion."
    )

    scored = estimate_policy(df, BACKLOG_STATUSES)
    rollup = epic_health_flags(epic_rollup(scored))
    if rollup.empty:
        # The rollup answers a scoped question and can be empty while the instance
        # still has orphans to file, so the unscoped section below outlives it.
        st.info("No tickets in the current scope.")
    else:
        orphans = int(rollup.loc[rollup["epic"] == "No epic", "open_children"].sum())
        drifting = int((rollup["issue_count"] > 0).sum() - (1 if orphans else 0))
        e1, e2, e3 = st.columns(3)
        epics_section = "Epics"
        _tile(
            e1,
            TAB_ENGINEERING,
            epics_section,
            "Epics with open work",
            f"{int((rollup['epic'] != 'No epic').sum())}",
        )
        _tile(
            e2,
            TAB_ENGINEERING,
            epics_section,
            "Epics needing attention",
            f"{max(drifting, 0)}",
        )
        _tile(e3, TAB_ENGINEERING, epics_section, "Tickets with no epic", f"{orphans}")

        # "No epic" is a bucket rather than an issue, so it gets no link to follow.
        display = rollup.copy()
        display["epic_url"] = display["epic_key"].map(
            lambda key: _jira_ticket_url(key) if str(key).strip() else ""
        )
        display = display.drop(columns=["epic_key"])
        display.insert(0, "epic_url", display.pop("epic_url"))
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                "epic_url": st.column_config.LinkColumn(
                    "Key", display_text=JIRA_KEY_DISPLAY_PATTERN
                ),
                "epic": st.column_config.TextColumn("Epic", width="large"),
                "open_children": st.column_config.NumberColumn("Open"),
                "owners": st.column_config.NumberColumn("Owners"),
                "avg_idle": st.column_config.NumberColumn("Avg idle (days)", format="%.1f"),
                "max_idle": st.column_config.NumberColumn("Max idle (days)", format="%.1f"),
                "unassigned": st.column_config.NumberColumn("Unassigned"),
                "no_estimate": st.column_config.NumberColumn("No estimate"),
                "estimated_hours": st.column_config.NumberColumn("Estimated (h)", format="%.1f"),
                "sprints": st.column_config.NumberColumn("Sprints"),
                "issues": st.column_config.TextColumn("What is wrong"),
                "issue_count": st.column_config.NumberColumn("Signals"),
            },
        )
        st.download_button(
            "Download epic rollup (CSV)",
            data=rollup.to_csv(index=False).encode("utf-8"),
            file_name="jira_epic_rollup.csv",
            mime="text/csv",
        )

    # Where a ticket belongs is a question about the whole instance, not about
    # what the sidebar is currently showing: judged on the filtered frame, an
    # epic whose children are all in the Backlog reads as empty, and an epic
    # loses the child summaries its suggestions are built from.
    _render_epic_organization(df if organization_source is None else organization_source)


def _render_epic_organization(df: pd.DataFrame) -> None:
    """Where the parentless tickets belong, and which epics are empty."""
    st.markdown("##### Epic organization")
    suggestions = epic_organization.suggest_parents(df)
    empty = epic_organization.empty_epics(df)
    if suggestions.empty and empty.empty:
        st.success("Every ticket has an epic, and no epic is sitting empty.")
        return

    matched = suggestions[suggestions["suggested_epic_key"].ne("")]
    o1, o2, o3 = st.columns(3)
    orphan_section = "Epic organization"
    _tile(o1, TAB_ENGINEERING, orphan_section, "Tickets with no epic", f"{len(suggestions)}")
    _tile(o2, TAB_ENGINEERING, orphan_section, "With a suggested parent", f"{len(matched)}")
    _tile(o3, TAB_ENGINEERING, orphan_section, "Epics with nothing open", f"{len(empty)}")
    st.caption(
        "Suggestions come from the words a ticket shares with an epic and with the "
        "tickets already in it, scored so that a word common to every epic counts "
        "for nothing. Only epics in the ticket's own project are considered. "
        "Every loaded ticket counts here, backlog included and whatever the scope "
        "and filters are: filing is a question about the whole instance. "
        "Nothing is filed automatically - *Why* is there so the guess can be judged."
    )

    by_epic_tab, ticket_tab, empty_tab = st.tabs(
        ["Suggested parents", "Every orphan", "Empty epics"]
    )
    with by_epic_tab:
        rollup = epic_organization.orphan_summary(suggestions)
        if rollup.empty:
            st.info("No orphan ticket reads clearly enough to point at an epic.")
        else:
            rollup = rollup.assign(
                epic_url=rollup["suggested_epic_key"].map(_jira_ticket_url)
            )
            st.dataframe(
                rollup[["epic_url", "suggested_epic", "tickets", "keys"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "epic_url": st.column_config.LinkColumn(
                        "Epic", display_text=JIRA_KEY_DISPLAY_PATTERN
                    ),
                    "suggested_epic": st.column_config.TextColumn("Name", width="large"),
                    "tickets": st.column_config.NumberColumn("Orphans"),
                    "keys": st.column_config.TextColumn("Tickets", width="large"),
                },
            )
            st.caption("Start at the top: one epic that takes several tickets at once.")
    with ticket_tab:
        display = suggestions.assign(
            key_url=suggestions["key"].map(_jira_ticket_url),
            confidence=suggestions["confidence"] * 100,
        )
        st.dataframe(
            display[
                ["key_url", "summary", "status", "assignee", "suggested_epic", "why", "confidence"]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "key_url": st.column_config.LinkColumn(
                    "Key", display_text=JIRA_KEY_DISPLAY_PATTERN
                ),
                "summary": st.column_config.TextColumn("Summary", width="large"),
                "status": st.column_config.TextColumn("Status"),
                "assignee": st.column_config.TextColumn("Assignee"),
                "suggested_epic": st.column_config.TextColumn("Suggested epic", width="medium"),
                "why": st.column_config.TextColumn("Why"),
                "confidence": st.column_config.NumberColumn("Confidence", format="%.0f%%"),
            },
        )
        st.download_button(
            "Download orphan tickets (CSV)",
            data=suggestions.to_csv(index=False).encode("utf-8"),
            file_name="jira_orphan_tickets.csv",
            mime="text/csv",
        )
    with empty_tab:
        if empty.empty:
            st.success("No epic is sitting empty.")
        else:
            display = empty.assign(key_url=empty["key"].map(_jira_ticket_url))
            st.dataframe(
                display[
                    [column for column in
                     ["key_url", "summary", "status", "assignee", "idle_days", "ticket_age_days"]
                     if column in display.columns]
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "key_url": st.column_config.LinkColumn(
                        "Key", display_text=JIRA_KEY_DISPLAY_PATTERN
                    ),
                    "summary": st.column_config.TextColumn("Epic", width="large"),
                    "status": st.column_config.TextColumn("Status"),
                    "assignee": st.column_config.TextColumn("Owner"),
                    "idle_days": st.column_config.NumberColumn("Idle (days)", format="%.1f"),
                    "ticket_age_days": st.column_config.NumberColumn("Age (days)", format="%.1f"),
                },
            )
            st.caption(
                "No open children left. The dashboard does not load Done tickets, so "
                "these are finished or abandoned - close them either way."
            )


_TRIAGE_DECISIONS_KEY = "_triage_decisions"
_TRIAGE_CURSOR_KEY = "_triage_cursor_key"
_TRIAGE_QUEUE_KEY = "_triage_queue_name"


def _triage_state(queue_name: str, keys: list[str]) -> tuple[dict[str, str], int]:
    """Decisions made so far and the current position for this queue.

    Every decision stays in the session until it is applied or the reviewer
    starts over; the dict returned here is only the part the current queue can
    show. Switching queue, narrowing the page size or tightening a filter
    therefore hides decisions rather than destroying them - the two queues
    overlap heavily, so a peek at the unassigned list must not cost an
    afternoon of triage - and the reviewed count still cannot exceed the
    queue length.

    The place in the queue is remembered as a ticket key rather than an offset,
    because closing five tickets shortens the list above the cursor and an offset
    would silently step over five tickets nobody ever saw. A key that has itself
    left the queue falls back to the first ticket still undecided.
    """
    if st.session_state.get(_TRIAGE_QUEUE_KEY) != queue_name:
        # Only the cursor is queue-local: the other queue's ticket would be an
        # arbitrary starting point here, so the first undecided one wins.
        st.session_state[_TRIAGE_QUEUE_KEY] = queue_name
        st.session_state[_TRIAGE_CURSOR_KEY] = None
    present = set(keys)
    decisions = {
        key: choice
        for key, choice in st.session_state.setdefault(_TRIAGE_DECISIONS_KEY, {}).items()
        if key in present
    }

    cursor = st.session_state.setdefault(_TRIAGE_CURSOR_KEY, None)
    if cursor in present:
        return decisions, keys.index(cursor)
    undecided = [index for index, key in enumerate(keys) if key not in decisions]
    return decisions, undecided[0] if undecided else len(keys)


def _render_triage_card(row: pd.Series) -> None:
    """One ticket, large enough to judge without opening Jira."""
    age = float(row["ticket_age_days"] or 0)
    idle = float(row["idle_days"] or 0)
    owner = "Nobody" if is_unowned(row["assignee"]) else str(row["assignee"])
    chips = [
        (f"{age:.0f} days old", age >= cleanup.ABANDONED_AGE_DAYS),
        (f"untouched {idle:.0f} days", idle >= cleanup.ABANDONED_IDLE_DAYS),
        (f"Owner: {owner}", is_unowned(row["assignee"])),
        (f"Status: {row['status']}", False),
        (f"Epic: {row['epic_summary'] or 'none'}", not str(row["epic_summary"] or "").strip()),
        (f"Priority: {row['priority'] or 'none'}", False),
    ]
    if cleanup.is_container(row.get("issue_type")):
        # Closing a container from an age-sorted queue is the one decision here
        # that can strand other people's open work, so it is called out in red.
        chips.append(("Holds other tickets - closing it orphans them", True))
    chip_html = "".join(
        f'<span class="{"hot" if hot else ""}">{html.escape(str(text))}</span>'
        for text, hot in chips
    )
    st.markdown(
        f'<div class="triage-card">'
        f'<div class="triage-key">{html.escape(str(row["key"]))}</div>'
        f'<div class="triage-summary">{html.escape(str(row["summary"]))}</div>'
        f'<div class="triage-meta">{chip_html}</div>'
        f'<div class="triage-why">Suggested: {html.escape(str(row["suggested"]))} '
        f'- {html.escape(str(row["why"]))}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.link_button(f"Open {row['key']} in Jira", _jira_ticket_url(str(row["key"])))


@st.fragment
def _render_cleanup(
    df: pd.DataFrame,
    *,
    unassigned_source: pd.DataFrame | None = None,
) -> None:
    """Review the oldest tickets one at a time and decide their fate.

    Nothing reaches Jira from here until the decisions are applied at the
    bottom, so a wrong click costs one more click rather than a ticket.

    ``unassigned_source`` is the same data before the assignee scope filter:
    unowned tickets match no assignee, so without it the unassigned queue would
    be empty by construction in Team and Individual scope.
    """
    st.subheader("Backlog Cleanup")
    st.caption(
        "The oldest open tickets, one at a time, Backlog included regardless of "
        "the sidebar toggle, and the unassigned queue ignores the scope filter "
        "because nobody's tickets belong to no team. Decisions are held in this "
        "session and only written to Jira when you apply them at the bottom."
    )
    if df.empty and (unassigned_source is None or unassigned_source.empty):
        st.info("No tickets in the current scope.")
        return

    controls = st.columns([2, 1])
    queue_choice = controls[0].radio(
        "Queue",
        options=["Oldest open tickets", "Oldest unassigned tickets"],
        horizontal=True,
    )
    size = controls[1].selectbox("How many", options=[25, 50, 100, 200], index=2)

    unassigned_only = queue_choice.startswith("Oldest unassigned")
    source = df
    if unassigned_only and unassigned_source is not None:
        source = unassigned_source
    queue = cleanup.build_queue(source, unassigned_only=unassigned_only, limit=int(size))
    if queue.empty:
        st.success("Nothing in this queue.")
        return

    # Size is not part of the identity: the queues are prefixes of one another,
    # so widening the list keeps every decision and narrowing it prunes the ones
    # that fell off, which is what the reviewer expects from a page-size control.
    decisions, position = _triage_state(queue_choice, queue["key"].astype(str).tolist())
    position = max(0, min(position, len(queue)))
    counts = cleanup.decision_summary(decisions)

    st.progress(
        min(len(decisions) / len(queue), 1.0),
        text=(
            f"{len(decisions)} of {len(queue)} reviewed - "
            f"{counts[cleanup.CLOSE]} to close, {counts[cleanup.KEEP]} keep, "
            f"{counts[cleanup.NEEDS_OWNER]} need an owner, {counts[cleanup.SKIP]} skipped"
        ),
    )

    if position >= len(queue):
        st.success("End of the queue - review the decisions below.")
    else:
        row = queue.iloc[position]
        _render_triage_card(row)

        keys = queue["key"].astype(str).tolist()

        def _decide(choice: str) -> None:
            st.session_state[_TRIAGE_DECISIONS_KEY][row["key"]] = choice
            st.session_state[_TRIAGE_CURSOR_KEY] = (
                keys[position + 1] if position + 1 < len(keys) else None
            )

        buttons = st.columns(5)
        buttons[0].button(
            "Close it",
            width="stretch",
            type="primary" if row["suggested"] == cleanup.CLOSE else "secondary",
            on_click=_decide,
            args=(cleanup.CLOSE,),
            key=f"triage_close_{row['key']}",
        )
        buttons[1].button(
            "Keep",
            width="stretch",
            type="primary" if row["suggested"] == cleanup.KEEP else "secondary",
            on_click=_decide,
            args=(cleanup.KEEP,),
            key=f"triage_keep_{row['key']}",
        )
        buttons[2].button(
            "Needs an owner",
            width="stretch",
            on_click=_decide,
            args=(cleanup.NEEDS_OWNER,),
            key=f"triage_owner_{row['key']}",
        )
        buttons[3].button(
            "Skip",
            width="stretch",
            on_click=_decide,
            args=(cleanup.SKIP,),
            key=f"triage_skip_{row['key']}",
        )
        if buttons[4].button("Back", width="stretch", disabled=position == 0):
            st.session_state[_TRIAGE_CURSOR_KEY] = keys[position - 1]
            st.rerun()

        st.caption(f"Ticket {position + 1} of {len(queue)}")

    _render_triage_decisions(queue, decisions)


def _render_triage_decisions(queue: pd.DataFrame, decisions: dict[str, str]) -> None:
    """The decisions taken so far, and the one place they can reach Jira."""
    # Decisions outlive the queue they were made in, so count the closures waiting
    # out of sight rather than letting Apply look like the whole outstanding job.
    visible = set(queue["key"].astype(str))
    elsewhere = sum(
        1
        for key, choice in st.session_state.get(_TRIAGE_DECISIONS_KEY, {}).items()
        if choice == cleanup.CLOSE and key not in visible
    )
    if elsewhere:
        st.caption(
            f"{elsewhere} more ticket(s) marked Close sit outside this queue and "
            "are untouched by Apply here; switch queue or widen the filters to "
            "reach them."
        )
    if not decisions:
        return

    decided = queue[queue["key"].isin(decisions)].copy()
    decided["decision"] = decided["key"].map(decisions)
    decided["key_url"] = decided["key"].map(_jira_ticket_url)

    with st.expander(f"Decisions so far ({len(decided)})", expanded=False):
        st.dataframe(
            decided[["key_url", "decision", "summary", "assignee", "ticket_age_days", "idle_days"]],
            width="stretch",
            hide_index=True,
            column_config={
                "key_url": st.column_config.LinkColumn(
                    "Key", display_text=JIRA_KEY_DISPLAY_PATTERN
                ),
                "decision": st.column_config.TextColumn("Decision"),
                "summary": st.column_config.TextColumn("Summary"),
                "assignee": st.column_config.TextColumn("Assignee"),
                "ticket_age_days": st.column_config.NumberColumn("Age (days)", format="%.0f"),
                "idle_days": st.column_config.NumberColumn("Idle (days)", format="%.0f"),
            },
        )
        st.download_button(
            "Download decisions (CSV)",
            data=decided[["key", "decision", "summary", "assignee", "status", "ticket_age_days"]]
            .to_csv(index=False)
            .encode("utf-8"),
            file_name="jira_cleanup_decisions.csv",
            mime="text/csv",
        )
        if st.button("Start over", key="triage_reset"):
            st.session_state[_TRIAGE_DECISIONS_KEY] = {}
            st.session_state[_TRIAGE_CURSOR_KEY] = None
            st.rerun()

    to_close = cleanup.pending_closures(queue, decisions)
    if not to_close:
        return

    st.markdown(f"##### Apply {len(to_close)} closure(s) to Jira")
    if not write_access.writes_enabled():
        st.info(
            f"{len(to_close)} ticket(s) marked Close are held as notes only. "
            + write_access.READ_ONLY_MESSAGE
        )
        return
    st.caption(
        "Each project runs its own workflow, so the closing status is resolved per "
        "ticket from the transitions Jira actually offers it, preferring "
        + " / ".join(cleanup.CLOSING_STATUS_PREFERENCE[:4])
        + " over Done so cleanup stays distinguishable from real completions."
    )
    # Named, not just counted: decisions outlive the queue they were made in, so
    # the reviewer has to be able to see exactly which tickets a click will move.
    with st.expander(f"Which {len(to_close)} tickets", expanded=False):
        st.write(", ".join(to_close))
    apply_columns = st.columns([3, 1])
    confirmed = apply_columns[0].checkbox(
        f"Yes, write these {len(to_close)} changes to Jira", value=False
    )
    if not apply_columns[1].button("Apply", type="primary", disabled=not confirmed):
        return

    client = JiraClient.resolve(creds_path=CREDS_PATH, profile_name=PROFILE_NAME)
    succeeded: list[str] = []
    failed: dict[str, str] = {}
    progress = st.progress(0.0, text="Resolving transitions...")
    for index, key in enumerate(to_close, start=1):
        progress.progress(index / len(to_close), text=f"Closing {key} ({index}/{len(to_close)})")
        try:
            offered = [t.get("to_status", "") for t in client.get_issue_transitions(key)]
        except Exception as error:  # noqa: BLE001
            failed[key] = f"could not read transitions ({error})"
            continue
        target = cleanup.closing_status(offered)
        if target is None:
            failed[key] = "no closing transition available (offers: " + ", ".join(offered) + ")"
            continue
        moved, errors, _ = _apply_action_with_audit(
            client=client,
            action_type="status",
            selected_keys=[key],
            target=target,
        )
        succeeded.extend(moved)
        failed.update(errors)
    progress.empty()

    if succeeded:
        st.success(f"Closed {len(succeeded)}: {', '.join(succeeded)}")
        for key in succeeded:
            st.session_state[_TRIAGE_DECISIONS_KEY].pop(key, None)
        _clear_page_caches(ENGINEERING_PAGE_TITLE)
    if failed:
        st.error(
            "Could not close "
            + "; ".join(f"{key}: {reason}" for key, reason in failed.items())
        )
        return
    if succeeded:
        # Redraw from fresh Jira data, otherwise the queue keeps offering the
        # tickets that were just closed. Held back when something failed, so the
        # reader gets to read why before the page moves.
        st.rerun()


@st.fragment
def _render_estimate_policy(df: pd.DataFrame) -> None:
    """Who is honouring "estimate it before it leaves Backlog"."""
    st.subheader("Estimate Policy")
    st.caption(
        "Every ticket past Backlog is expected to carry an estimate. Backlog "
        "statuses are exempt (configurable via JIRA_BACKLOG_STATUSES)."
    )

    scored = estimate_policy(df, BACKLOG_STATUSES)
    in_scope = scored[scored["policy_applies"].fillna(False).astype(bool)] if not scored.empty else scored
    if in_scope.empty:
        st.info("No tickets past Backlog in the current scope.")
        return

    violations = in_scope[in_scope["policy_violation"].astype(bool)]
    compliance_pct = (1 - len(violations) / len(in_scope)) * 100.0

    c1, c2, c3 = st.columns(3)
    policy = "Estimate policy"
    _tile(c1, TAB_ENGINEERING, policy, "Policy Compliance", f"{compliance_pct:.0f}%")
    _tile(c2, TAB_ENGINEERING, policy, "Missing Estimate", f"{len(violations)}")
    _tile(
        c3,
        TAB_ENGINEERING,
        policy,
        "Estimated Work",
        f"{in_scope['estimate_hours'].sum():.0f}h",
    )

    rollup = policy_compliance_by_owner(scored)
    if not rollup.empty:
        st.dataframe(
            rollup,
            width="stretch",
            hide_index=True,
            column_config={
                "assignee": st.column_config.TextColumn("Assignee"),
                "past_backlog": st.column_config.NumberColumn("Past Backlog"),
                "missing_estimate": st.column_config.NumberColumn("Missing Estimate"),
                "compliance_pct": st.column_config.ProgressColumn(
                    "Compliance",
                    format="%.0f%%",
                    min_value=0,
                    max_value=100,
                ),
                "estimated_hours": st.column_config.NumberColumn("Estimated (h)", format="%.1f"),
            },
        )

    if violations.empty:
        st.success("Every ticket past Backlog has an estimate.")
        return

    offenders = violations.sort_values(["assignee", "key"]).copy()
    offenders["key_url"] = offenders["key"].map(_jira_ticket_url)
    with st.expander(f"Tickets missing an estimate ({len(offenders)})", expanded=False):
        st.dataframe(
            offenders[["key_url", "summary", "assignee", "status", "idle_days"]],
            width="stretch",
            hide_index=True,
            column_config={
                "key_url": st.column_config.LinkColumn("Key", display_text=JIRA_KEY_DISPLAY_PATTERN),
                "summary": st.column_config.TextColumn("Summary"),
                "assignee": st.column_config.TextColumn("Assignee"),
                "status": st.column_config.TextColumn("Status"),
                "idle_days": st.column_config.NumberColumn("Idle (days)", format="%.1f"),
            },
        )
    st.download_button(
        "Download missing-estimate tickets (CSV)",
        data=offenders[["key", "summary", "assignee", "status", "idle_days"]]
        .to_csv(index=False)
        .encode("utf-8"),
        file_name="jira_missing_estimates.csv",
        mime="text/csv",
    )


@st.fragment
def _render_stale_cleanup(df: pd.DataFrame) -> None:
    """Old tickets ranked by how abandoned they look, so they can be cleared out."""
    st.subheader("Stale & Abandoned")
    st.caption(
        "Tickets idle past the threshold, ranked by how many neglect signals they "
        "carry. The top of this list is the safest to close or send back to Backlog. "
        "Backlog tickets are always included here regardless of the Include Backlogs "
        "filter - that is where rot accumulates."
    )
    if df.empty:
        st.info("No tickets in the current scope.")
        return

    threshold = st.slider(
        "Idle at least (days)",
        min_value=30,
        max_value=365,
        value=DEFAULT_STALE_DAYS,
        step=15,
    )
    candidates = stale_candidates(df, idle_days=threshold, backlog_statuses=BACKLOG_STATUSES)
    if candidates.empty:
        st.success(f"No ticket has been idle for {threshold}+ days.")
        return

    unassigned = candidates["stale_reasons"].str.contains("unassigned").sum()
    never_started = candidates["stale_reasons"].str.contains("never started").sum()
    c1, c2, c3 = st.columns(3)
    abandoned = "Stale & abandoned"
    _tile(c1, TAB_ENGINEERING, abandoned, f"Idle {threshold}d+", f"{len(candidates)}")
    _tile(c2, TAB_ENGINEERING, abandoned, "Unassigned", f"{unassigned}")
    _tile(c3, TAB_ENGINEERING, abandoned, "Never Started", f"{never_started}")

    display = candidates.copy()
    display["key_url"] = display["key"].map(_jira_ticket_url)
    st.dataframe(
        display[
            [
                "key_url",
                "summary",
                "assignee",
                "status",
                "idle_days",
                "ticket_age_days",
                "stale_reasons",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "key_url": st.column_config.LinkColumn("Key", display_text=JIRA_KEY_DISPLAY_PATTERN),
            "summary": st.column_config.TextColumn("Summary"),
            "assignee": st.column_config.TextColumn("Assignee"),
            "status": st.column_config.TextColumn("Status"),
            "idle_days": st.column_config.NumberColumn("Idle (days)", format="%.1f"),
            "ticket_age_days": st.column_config.NumberColumn("Age (days)", format="%.1f"),
            "stale_reasons": st.column_config.TextColumn("Why it looks abandoned"),
        },
    )
    st.download_button(
        "Download stale tickets (CSV)",
        data=display[
            ["key", "summary", "assignee", "status", "idle_days", "ticket_age_days", "stale_reasons"]
        ]
        .to_csv(index=False)
        .encode("utf-8"),
        file_name="jira_stale_tickets.csv",
        mime="text/csv",
    )


def _fmt_seconds(secs: float) -> str:
    """Convert seconds to a human-readable h/m string."""
    if secs <= 0:
        return "—"
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    if h and m:
        return f"{h}h {m}m"
    if h:
        return f"{h}h"
    return f"{m}m"


def _parse_estimate_to_seconds(value: str) -> float | None:
    """Parse Jira estimate text like '2h', '1d 2h', '30m' into seconds."""
    text = str(value or "").strip().lower()
    if not text:
        return None

    # Jira-style units. We use common defaults: 1d = 8h, 1w = 5d.
    unit_seconds = {
        "m": 60,
        "h": 3600,
        "d": 8 * 3600,
        "w": 5 * 8 * 3600,
    }
    tokens = re.findall(r"(\d+)\s*([mhdw])", text)
    if not tokens:
        return None

    matched = " ".join(f"{num}{unit}" for num, unit in tokens)
    normalized = re.sub(r"\s+", "", text)
    if re.sub(r"\s+", "", matched) != normalized:
        return None

    total = 0.0
    for num, unit in tokens:
        total += int(num) * unit_seconds[unit]
    return total


def _transition_sample_keys(editor_df: pd.DataFrame) -> tuple[str, ...]:
    """Pick the tickets to query transitions for, bounded by TRANSITION_LOOKUP_LIMIT.

    One request goes to Jira per key, so an org-wide table cannot query every row.
    One ticket per status that no sprint member holds is reserved first, so the
    Status dropdown keeps offering those moves even for a sprint large enough to
    exhaust the budget on its own; sprint members fill whatever remains.
    """
    frame = editor_df.dropna(subset=["key"]).copy()
    if frame.empty:
        return ()
    frame["key"] = frame["key"].astype(str)
    frame["status"] = frame.get("status", "").fillna("").astype(str).str.strip()

    in_sprint = (
        frame["include"].fillna(False).astype(bool)
        if "include" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    sprint_statuses = set(frame.loc[in_sprint, "status"])
    outside = frame.loc[~in_sprint & ~frame["status"].isin(sprint_statuses)]
    reserved = [
        sorted(group["key"].tolist())[0]
        for _, group in outside.groupby("status", sort=True)
    ][:TRANSITION_LOOKUP_LIMIT]

    budget = TRANSITION_LOOKUP_LIMIT - len(reserved)
    keys = reserved + sorted(frame.loc[in_sprint, "key"].unique().tolist())[:budget]

    return tuple(sorted(set(keys)))


_PLAN_EDITOR_KEY = "_sprint_plan_editor"
_SPRINT_MOVE_BATCH = 50


def _render_sprint_plan(df: pd.DataFrame) -> None:
    """Propose a sprint for one team from a few named goals and real hours."""
    st.caption(
        "A sprint is decided as two or three goals - \"finish onboarding, finalise "
        "the quiz, checkout\" - not as the top rows of a priority list. Name the "
        "goals, and the tickets that serve them are filled into each person's "
        "hours in that order, highest priority first within each goal. Every row "
        "is editable before anything is written."
    )
    scored = add_team(
        add_priority_score(estimate_policy(df, BACKLOG_STATUSES)),
        TEAM_PROJECTS,
        TEAM_PEOPLE,
    )
    teams = sorted(scored["team"].dropna().astype(str).unique())
    if not teams:
        st.info("No teams to plan for. Set JIRA_TEAM_PEOPLE to define them.")
        return

    top = st.columns([2, 3])
    with top[0]:
        team = st.selectbox("Team", options=teams, key="plan_team")
    with top[1]:
        goal_spec = st.text_input(
            "Sprint goals (comma separated, most important first)",
            value=st.session_state.get("plan_goals", ""),
            key="plan_goals",
            placeholder="Onboarding, Quiz, Checkout",
        )
    goals = sprint_planner.parse_goals(goal_spec)

    team_df = scored[scored["team"].astype(str) == team]
    if "issue_type" in team_df.columns:
        # Containers carry no hours of their own, and Jira spells them several
        # ways ("Top-level initiative", "Toplevel"), so the shared test is used
        # rather than a literal list that misses the spellings it does not know.
        team_df = team_df[~team_df["issue_type"].map(cleanup.is_container)]
    if team_df.empty:
        st.info(f"No open tickets for {team}.")
        return

    states = team_df["sprint_state"].fillna("").astype(str).str.lower()
    dated = team_df[states.isin(["active", "future"])]
    start, end = _sprint_window(dated if not dated.empty else team_df)
    knobs = st.columns(4)
    with knobs[0]:
        sprint_days = st.number_input(
            "Sprint length (working days)",
            min_value=1,
            max_value=30,
            # A stale or quarter-long sprint row in Jira would otherwise seed the
            # box with a value its own ceiling rejects, and the section would not
            # render at all.
            value=min(int(working_days(start, end)) or 10, 30),
            key="plan_days",
            help="Taken from the team's current sprint dates when Jira has them.",
        )
    with knobs[1]:
        overhead = st.number_input(
            "Overhead per person (h/week)",
            min_value=0.0,
            max_value=40.0,
            value=sprint_planner.DEFAULT_OVERHEAD_HOURS_PER_WEEK,
            step=0.5,
            key="plan_overhead",
            help="Code review, Slack, meetings - hours spent every week on no ticket.",
        )
    with knobs[2]:
        assumed = st.number_input(
            "Assume unestimated ticket is (h)",
            min_value=0.5,
            max_value=40.0,
            value=sprint_planner.DEFAULT_TICKET_HOURS,
            step=0.5,
            key="plan_assumed",
        )
    with knobs[3]:
        # Streamlit honours `value` only when a keyed widget is first created, so
        # a box keyed once would be stored as unticked from the first render -
        # when no goals had been typed yet - and would stay that way after they
        # were. Naming the widget after whether goals exist gives each state its
        # own default while still remembering a deliberate change within it.
        only_goals = st.checkbox(
            "Only goal work",
            value=bool(goals),
            key=f"plan_only_goals_{bool(goals)}",
            help="Off: tickets serving no goal are planned last with whatever hours remain.",
        )

    if not WEEKLY_HOURS:
        st.warning(
            "Nobody has declared hours, so there is nothing to plan against. Set "
            'JIRA_WEEKLY_HOURS (e.g. "Ali=40,Farid=20") to use this section.'
        )
        return

    # Anyone on the team's roster is planned for, not only whoever already holds
    # a ticket: someone with hours and nothing assigned is the spare capacity the
    # plan exists to find. They join the list under the spelling their hours were
    # declared in, and only when no Jira name already denotes them - listing the
    # roster's "farid" beside Jira's "Farid Shahidi" would make one declaration
    # match two names and be withheld from both as ambiguous.
    known = set(team_df["assignee"].dropna().astype(str))
    roster = [
        name
        for name, owner in TEAM_PEOPLE.items()
        if owner == team and not any(same_person(name, person) for person in known)
    ]
    spare = {
        declared
        for declared in WEEKLY_HOURS
        if any(same_person(declared, name) for name in roster)
    }
    people = sorted(known | spare)
    capacity = sprint_planner.person_capacity(
        people, WEEKLY_HOURS, sprint_days, overhead_per_week=overhead
    )
    if capacity.empty:
        st.warning(
            f"None of {team}'s people appear in JIRA_WEEKLY_HOURS, so their "
            "available hours are unknown. Add them to plan this team."
        )
        return

    candidates = team_df.assign(goal=sprint_planner.match_goals(team_df, goals))
    if goals and only_goals:
        # Work already under way is kept whatever it serves: it is spending this
        # sprint's hours either way, and hiding it would hand those hours out twice.
        candidates = candidates[
            candidates["goal"].ne(sprint_planner.NO_GOAL)
            | sprint_planner.in_flight(candidates)
        ]
    if candidates.empty:
        st.info("No ticket matches those goals. Try different words, or untick *Only goal work*.")
        return

    plan = sprint_planner.plan_sprint(candidates, capacity, goals=goals, default_hours=assumed)

    # The goals belong above the tickets that serve them, but they have to report
    # the reviewer's decision rather than the proposal, so the space is reserved
    # here and filled once the editor below has been read.
    goals_slot = st.container() if goals else None

    st.markdown("**Proposed plan** - change any row; the load below follows your edits.")
    editable = plan.assign(include=plan["plan"].eq(sprint_planner.PLANNED))
    edited = st.data_editor(
        editable[
            ["include", "key", "goal", "summary", "assignee", "status", "hours", "why"]
        ],
        width="stretch",
        hide_index=True,
        # Streamlit remembers edits by row position under the widget's key, so a
        # key that outlived the plan would re-apply a tick made on row 3 of the
        # old plan to whatever ticket is row 3 of the new one - and those are the
        # rows that get written to Jira. Every input that can reorder the plan is
        # therefore part of its identity.
        key="|".join(
            [
                _PLAN_EDITOR_KEY,
                team,
                goal_spec,
                str(sprint_days),
                str(overhead),
                str(assumed),
                str(only_goals),
                # The tickets themselves are part of the identity too: Jira data
                # is refetched on a timer, and a plan reordered by a new ticket
                # would otherwise keep the ticks at their old row positions.
                hashlib.sha1(
                    ",".join(plan["key"].astype(str)).encode("utf-8")
                ).hexdigest(),
            ]
        ),
        column_config={
            "include": st.column_config.CheckboxColumn("In sprint"),
            "key": st.column_config.TextColumn("Key", disabled=True),
            "goal": st.column_config.TextColumn("Goal", disabled=True),
            "summary": st.column_config.TextColumn("Summary", width="large", disabled=True),
            "assignee": st.column_config.TextColumn("Assignee", disabled=True),
            "status": st.column_config.TextColumn("Status", disabled=True),
            "hours": st.column_config.NumberColumn("Hours", format="%.1f"),
            "why": st.column_config.TextColumn("Why", width="medium", disabled=True),
        },
    )

    # The proposal is a starting point, so the load has to answer for what the
    # reviewer decided rather than for what was proposed.
    decided = plan.drop(columns=["hours", "plan"]).merge(
        edited[["key", "include", "hours"]], on="key", how="left"
    )
    decided["plan"] = decided["include"].map(
        lambda chosen: sprint_planner.PLANNED if chosen else sprint_planner.NEXT_UP
    )
    if goals_slot is not None:
        with goals_slot:
            st.markdown("**Goals**")
            st.dataframe(
                sprint_planner.goal_load(decided, goals),
                width="stretch",
                hide_index=True,
                column_config={
                    "goal": st.column_config.TextColumn("Goal", width="large"),
                    "tickets": st.column_config.NumberColumn("Tickets"),
                    "hours": st.column_config.NumberColumn("Hours needed", format="%.1f"),
                    "planned_tickets": st.column_config.NumberColumn("Fits"),
                    "planned_hours": st.column_config.NumberColumn("Hours planned", format="%.1f"),
                    "left_out": st.column_config.NumberColumn("Left over"),
                },
            )

    load = sprint_planner.plan_load(decided, capacity)
    over = load[load["left_hours"].lt(0)]
    st.markdown("**Who it lands on**")
    st.dataframe(
        load,
        width="stretch",
        hide_index=True,
        column_config={
            "assignee": st.column_config.TextColumn("Person"),
            "planning_hours": st.column_config.NumberColumn("Available (h)", format="%.1f"),
            "planned_hours": st.column_config.NumberColumn("Planned (h)", format="%.1f"),
            "left_hours": st.column_config.NumberColumn("Left (h)", format="%.1f"),
            "planned_tickets": st.column_config.NumberColumn("Tickets"),
            "waiting_tickets": st.column_config.NumberColumn("Not in sprint"),
        },
    )
    if not over.empty:
        st.warning(
            "Over their hours: "
            + ", ".join(
                f"{row['assignee']} by {abs(row['left_hours']):.1f}h"
                for _, row in over.iterrows()
            )
        )
    st.caption(
        f"Available hours are each person's weekly hours over {int(sprint_days)} working "
        f"day(s), minus {overhead:g}h/week of overhead. Unestimated tickets are assumed "
        f"to be {assumed:g}h and say so in *Why*."
    )
    st.download_button(
        "Download plan (CSV)",
        data=decided.to_csv(index=False).encode("utf-8"),
        file_name="jira_sprint_plan.csv",
        mime="text/csv",
    )
    _apply_sprint_plan(decided, team_df)


def _apply_sprint_plan(plan: pd.DataFrame, team_df: pd.DataFrame) -> None:
    """Write the chosen tickets into a real sprint, once someone asks for it."""
    chosen = plan[plan["plan"].eq(sprint_planner.PLANNED)]["key"].astype(str).tolist()
    sprints = (
        team_df[team_df["sprint_state"].fillna("").astype(str).str.lower().isin(["future", "active"])]
        [["sprint_id", "sprint_name", "sprint_state"]]
        .dropna(subset=["sprint_id"])
        .drop_duplicates()
    )
    if sprints.empty:
        st.caption(
            "This team has no active or future sprint holding any of its tickets, "
            "so the plan can only be exported. A sprint appears here once it has at "
            "least one of the team's tickets on it."
        )
        return

    labels = {
        f"{row['sprint_name']} ({str(row['sprint_state']).title()})": _normalize_sprint_id(
            row["sprint_id"]
        )
        for _, row in sprints.iterrows()
    }
    columns = st.columns([3, 2])
    with columns[0]:
        label = st.selectbox("Apply to sprint", options=sorted(labels), key="plan_target_sprint")
    sprint_id = labels.get(label)
    # Jira's add-to-sprint moves an issue rather than copying it, so applying a
    # plan to the future sprint takes in-flight work out of the active one - and
    # in-flight work is exactly what the planner ticks first. Say so by name
    # before the button, since the section otherwise promises only to add.
    open_elsewhere = team_df[
        team_df["key"].astype(str).isin(chosen)
        & team_df["sprint_state"].fillna("").astype(str).str.lower().isin(["active", "future"])
        & team_df["sprint_id"].map(_normalize_sprint_id).ne(sprint_id)
    ]
    moving = sorted(set(open_elsewhere["key"].astype(str)))
    if moving:
        leaving = ", ".join(
            sorted(set(open_elsewhere["sprint_name"].dropna().astype(str)))
        )
        st.warning(
            f"{len(moving)} of the chosen tickets are on another open sprint "
            f"({leaving}) and Jira moves rather than copies them, so they would "
            f"leave it: {', '.join(moving[:10])}"
            + (" ..." if len(moving) > 10 else "")
        )
    if not write_access.writes_enabled():
        st.info(write_access.READ_ONLY_MESSAGE)
    with columns[1]:
        apply = st.button(
            f"Add {len(chosen)} ticket(s) to sprint",
            type="primary",
            disabled=not (chosen and sprint_id and write_access.writes_enabled()),
            key="plan_apply",
        )
    if not apply:
        return

    client = JiraClient.resolve(creds_path=CREDS_PATH, profile_name=PROFILE_NAME)
    # Tickets already in the target sprint are left alone: re-adding them is a
    # no-op to Jira but noise in the board's history.
    already = set(
        team_df[team_df["sprint_id"].map(_normalize_sprint_id) == sprint_id]["key"].astype(str)
    )
    to_add = [key for key in chosen if key not in already]
    if not to_add:
        st.info("Every chosen ticket is already in that sprint.")
        return
    written = 0
    with st.spinner(f"Adding {len(to_add)} ticket(s) to {label}..."):
        try:
            # Jira's Agile API takes at most fifty issues per move, and a plan
            # drawn to a whole team's hours passes fifty rows easily.
            for offset in range(0, len(to_add), _SPRINT_MOVE_BATCH):
                batch = to_add[offset : offset + _SPRINT_MOVE_BATCH]
                client.add_issues_to_sprint(sprint_id, batch)
                written += len(batch)
        except Exception as exc:  # noqa: BLE001
            # A later batch failing does not undo the earlier ones: saying the
            # write failed outright would send someone to re-apply tickets that
            # are already on the sprint.
            if written:
                st.error(
                    f"Added {written} of {len(to_add)} ticket(s) to {label}, then "
                    f"the next batch failed: {exc}. Refresh Data before retrying "
                    "so the ones already moved are not applied again."
                )
            else:
                st.error(f"Failed to add tickets to the sprint: {exc}")
            return
    st.success(f"Added {len(to_add)} ticket(s) to {label}. Refresh Data to see it.")


def _render_sprint_capacity(
    df: pd.DataFrame,
    status_source_df: pd.DataFrame | None = None,
    selected_ticket_key: str | None = None,
) -> None:
    """Show sprint capacity breakdown for a selected future/active sprint, grouped by assignee."""
    required_cols = {"sprint_id", "sprint_name", "sprint_state", "sprint_board_id"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        st.info("Sprint data is not fully loaded yet. Refresh data to enable sprint capacity editing.")
        return

    sprint_df = df[df["sprint_name"].notna()].copy() if "sprint_name" in df.columns else pd.DataFrame()
    if sprint_df.empty:
        st.info("No sprint data found. Ensure your Jira board uses sprints and the sprint field is enabled.")
        return

    sprint_df["sprint_state"] = sprint_df["sprint_state"].fillna("").astype(str)
    non_closed = sprint_df[sprint_df["sprint_state"].str.lower().isin(["future", "active"])]
    target_df = non_closed if not non_closed.empty else sprint_df

    state_rank = {"future": 0, "active": 1, "closed": 2, "": 3}
    sprint_options_df = (
        target_df[["sprint_id", "sprint_name", "sprint_state", "sprint_board_id"]]
        .drop_duplicates()
        .assign(
            state_rank=lambda frame: frame["sprint_state"].str.lower().map(state_rank).fillna(9),
            sprint_label=lambda frame: frame["sprint_name"] + " (" + frame["sprint_state"].str.title().replace("", "Unknown") + ")",
        )
        .sort_values(["state_rank", "sprint_name"])
    )
    sprint_labels = sprint_options_df["sprint_label"].tolist()
    default_idx = 0
    selected_label = st.selectbox("Sprint", options=sprint_labels, index=default_idx)
    selected_row = sprint_options_df.loc[sprint_options_df["sprint_label"] == selected_label].iloc[0]
    selected_sprint_id = _normalize_sprint_id(selected_row["sprint_id"])
    if not selected_sprint_id:
        st.error("Selected sprint has no valid sprint ID; cannot apply sprint membership changes.")
        return
    selected_sprint_key = selected_sprint_id

    scoped = target_df[
        (target_df["sprint_name"] == selected_row["sprint_name"])
        & (target_df["sprint_state"] == selected_row["sprint_state"])
    ].copy()

    st.markdown(f"**Selected sprint:** {selected_row['sprint_name']} ({str(selected_row['sprint_state']).title()})")

    is_ml_sprint = str(selected_row["sprint_name"]).startswith("ML Sprint")

    # Two distinct reasons a sprint cannot be edited, kept apart so the reader is
    # told which one applies: a closed sprint is permanent, read-only is a switch.
    sprint_is_open = str(selected_row["sprint_state"]).lower() in {"future", "active"}
    editable = sprint_is_open and write_access.writes_enabled()
    ticket_editor_df = df.copy()

    # Capture epics before filtering them out so we can show them in a separate table.
    _all_epics = ticket_editor_df[
        ticket_editor_df["issue_type"].fillna("").astype(str).str.strip().str.lower() == "epic"
    ].copy()
    epic_sprint_df = _all_epics[
        _all_epics["sprint_id"].map(_normalize_sprint_id) == selected_sprint_id
    ].copy()

    ticket_editor_df = ticket_editor_df[
        ticket_editor_df["issue_type"].fillna("").astype(str).str.strip().str.lower() != "epic"
    ].copy()
    ticket_editor_df["in_selected_sprint"] = (
        ticket_editor_df["sprint_id"].map(_normalize_sprint_id) == selected_sprint_id
    )
    ticket_editor_df["include"] = ticket_editor_df["in_selected_sprint"]
    sprint_ticket_columns = [
        "include",
        "key",
        "summary",
        "status",
        "priority",
        "assignee",
        "original_estimate",
        "reporter",
        "logged_time",
        "ticket_age_days",
        "idle_days",
        "created",
        "updated",
        "issue_type",
    ]
    ticket_editor_df = ticket_editor_df[
        sprint_ticket_columns
    ].sort_values(["include", "assignee", "key"], ascending=[False, True, True])

    # display_editor_df is only used for what the user sees — bubble click narrows rows here only.
    is_bubble_filtered = False
    if selected_ticket_key:
        selected_mask = ticket_editor_df["key"].astype(str) == str(selected_ticket_key)
        if selected_mask.any():
            display_editor_df = ticket_editor_df[selected_mask].copy()
            is_bubble_filtered = True
        else:
            display_editor_df = ticket_editor_df
    else:
        display_editor_df = ticket_editor_df

    sprint_header_col, sprint_action_col = st.columns([6, 1])
    with sprint_header_col:
        st.markdown("##### Sprint Tickets")
    with sprint_action_col:
        if is_bubble_filtered:
            if st.button("Restore table", key=f"restore_table_{selected_sprint_key}"):
                st.session_state["restore_sprint_ticket_table"] = True
                st.rerun()
    editor_key = f"sprint_editor_{selected_sprint_key}"
    editor_version_key = f"{editor_key}_version"
    if editor_version_key not in st.session_state:
        st.session_state[editor_version_key] = 0
    editor_widget_key_base = f"{editor_key}_{st.session_state[editor_version_key]}"
    editor_seed_key = f"{editor_key}_seed_df"

    if editor_seed_key in st.session_state:
        seed_df = st.session_state.pop(editor_seed_key)
        if isinstance(seed_df, pd.DataFrame):
            display_editor_df = seed_df.copy()

    # Optional multi-column sorting for Sprint Tickets (up to 4 levels).
    sort_label_by_col = {
        "include": "In Sprint",
        "key": "Key",
        "summary": "Summary",
        "status": "Status",
        "priority": "Priority",
        "assignee": "Assignee",
        "original_estimate": "Original Estimate",
        "reporter": "Reporter",
        "logged_time": "Logged",
        "ticket_age_days": "Age (days)",
        "idle_days": "Idle (days)",
        "created": "Created at",
        "updated": "Updated at",
        "issue_type": "Type",
    }
    sortable_columns = [c for c in sprint_ticket_columns if c in display_editor_df.columns]
    sort_col_options = ["(none)"] + sortable_columns

    st.caption("Sort Sprint Tickets (up to 4 columns)")
    sort_ui_cols = st.columns(4)

    default_sort_spec = [
        ("include", "desc"),
        ("assignee", "asc"),
        ("key", "asc"),
        ("(none)", "asc"),
    ]

    selected_sort_cols: list[str] = []
    selected_sort_dirs: list[str] = []
    for idx in range(4):
        level = idx + 1
        default_col, default_dir = default_sort_spec[idx]
        with sort_ui_cols[idx]:
            selected_col = st.selectbox(
                f"Sort {level}",
                options=sort_col_options,
                index=sort_col_options.index(default_col) if default_col in sort_col_options else 0,
                format_func=lambda c: "(none)" if c == "(none)" else sort_label_by_col.get(c, c),
                key=f"{editor_key}_sort_col_{level}",
            )
            selected_dir = st.selectbox(
                f"Dir {level}",
                options=["asc", "desc"],
                index=0 if default_dir == "asc" else 1,
                key=f"{editor_key}_sort_dir_{level}",
            )

        if selected_col != "(none)" and selected_col not in selected_sort_cols:
            selected_sort_cols.append(selected_col)
            selected_sort_dirs.append(selected_dir)

    if selected_sort_cols:
        display_editor_df = display_editor_df.sort_values(
            by=selected_sort_cols,
            ascending=[d == "asc" for d in selected_sort_dirs],
            kind="mergesort",
        )

    # Ensure sort changes remount the editor, otherwise Streamlit may keep prior row order/state.
    sort_signature = "__".join(
        f"{col}:{direction}" for col, direction in zip(selected_sort_cols, selected_sort_dirs)
    ) or "none"
    editor_widget_key = f"{editor_widget_key_base}_{sort_signature}"

    current_statuses = sorted(display_editor_df["status"].dropna().astype(str).str.strip().unique().tolist())
    # Asking Jira which moves are legal costs one request per ticket, and the
    # answer is only ever used to widen a dropdown the reader cannot act on
    # while the dashboard is read-only. So it is asked for when edits are armed,
    # and the statuses already on screen stand in for it the rest of the time -
    # which is most page loads, and the reason they no longer wait for it.
    transition_statuses: list[str] = []
    if write_access.writes_enabled():
        visible_keys = _transition_sample_keys(display_editor_df)
        try:
            with st.spinner("Reading which status moves Jira allows..."):
                transition_statuses = fetch_available_transition_statuses(
                    CREDS_PATH,
                    PROFILE_NAME,
                    visible_keys,
                )
        except Exception:
            transition_statuses = []
    _all_statuses = sorted(set(current_statuses) | set(transition_statuses))
    try:
        _all_priorities = fetch_all_priorities(CREDS_PATH, PROFILE_NAME)
    except Exception:
        _all_priorities = ["Highest", "Urgent", "High", "Normal", "Medium", "Low", "Lowest"]

    try:
        all_users = fetch_all_users(CREDS_PATH, PROFILE_NAME)
    except Exception:
        all_users = []

    jira_assignee_names = {
        str(user.get("display_name", "")).strip()
        for user in all_users
        if str(user.get("display_name", "")).strip()
    }
    assignee_options = sorted(
        set(ticket_editor_df["assignee"].dropna().astype(str).str.strip().unique().tolist())
        | jira_assignee_names
        | {"Unassigned"}
    )

    assignee_name_to_account_id = {
        str(user.get("display_name", "")).strip(): str(user.get("account_id", "")).strip()
        for user in all_users
        if str(user.get("display_name", "")).strip() and str(user.get("account_id", "")).strip()
    }
    assignee_name_to_account_id.update(
        (
            df[["assignee", "assignee_account_id"]]
            .dropna(subset=["assignee", "assignee_account_id"])
            .drop_duplicates(subset=["assignee"])
            .assign(
                assignee=lambda frame: frame["assignee"].astype(str).str.strip(),
                assignee_account_id=lambda frame: frame["assignee_account_id"].astype(str).str.strip(),
            )
            .set_index("assignee")["assignee_account_id"]
            .to_dict()
        )
    )

    # Create display dataframe with URL column for LinkColumn
    display_df_for_editor = display_editor_df.copy()
    display_df_for_editor.insert(1, "jira_key_link", display_df_for_editor["key"].apply(_jira_ticket_url))  # Full URL in position 1
    display_df_for_editor = display_df_for_editor.drop(columns=["key"])
    visible_editor_columns = [
        "include",
        "jira_key_link",
        "summary",
        "status",
        "priority",
        "assignee",
        "original_estimate",
        "reporter",
        "logged_time",
        "ticket_age_days",
        "idle_days",
        "created",
        "updated",
        "issue_type",
    ]
    
    edited_output = st.data_editor(
        display_df_for_editor,
        width="stretch",
        hide_index=True,
        column_order=visible_editor_columns,
        disabled=(not editable) or (not is_ml_sprint) or [
            "jira_key_link",
            "summary",
            "reporter",
            "logged_time",
            "ticket_age_days",
            "idle_days",
            "created",
            "updated",
            "issue_type",
        ],
        column_config={
            "include": st.column_config.CheckboxColumn("In Sprint"),
            "jira_key_link": st.column_config.LinkColumn(
                "Key",
                display_text=JIRA_KEY_DISPLAY_PATTERN,
            ),
            "summary": "Summary",
            "status": st.column_config.SelectboxColumn(
                "Status",
                options=_all_statuses,
                help=(
                    "Change status — applied to Jira on Apply sprint selection"
                    if write_access.writes_enabled()
                    else "Statuses currently on the board. Arm 'Allow Jira edits' "
                    "to load every move Jira permits."
                ),
            ),
            "priority": st.column_config.SelectboxColumn(
                "Priority",
                options=_all_priorities,
                help="Change priority — applied to Jira on Apply sprint selection",
            ),
            "assignee": st.column_config.SelectboxColumn(
                "Assignee",
                options=assignee_options,
                help="Change assignee — applied to Jira on Apply sprint selection",
            ),
            "original_estimate": st.column_config.TextColumn(
                "Original Estimate",
                help="Editable Jira estimate format (examples: 2h, 1d 2h, 30m)",
            ),
            "reporter": "Reporter",
            "logged_time": "Logged",
            "ticket_age_days": "Age (days)",
            "idle_days": "Idle (days)",
            "created": "Created at",
            "updated": "Updated at",
            "issue_type": "Type",
        },
        key=editor_widget_key,
    )

    # Restore key column from jira_key_link (extract key from URL)
    edited_tickets = edited_output.copy()
    # unquote mirrors the encoding _jira_ticket_url applies, so the key that goes
    # back to Jira is the key that came out of it.
    edited_tickets["key"] = edited_tickets["jira_key_link"].apply(
        lambda url: unquote(str(url).split("/")[-1])
    )
    edited_tickets = edited_tickets.drop(columns=["jira_key_link"])

    # Build edit dicts directly from the data_editor output (edited_tickets) for the displayed rows,
    # then fall back to ticket_editor_df originals for any rows hidden by bubble-click filtering.
    edited_include_by_key = edited_tickets.set_index("key")["include"].to_dict()
    edited_original_est_by_key_raw = (
        edited_tickets.set_index("key")["original_estimate"].fillna("").astype(str).str.strip().to_dict()
    )
    edited_status_by_key = edited_tickets.set_index("key")["status"].fillna("").astype(str).str.strip().to_dict()
    edited_priority_by_key = edited_tickets.set_index("key")["priority"].fillna("").astype(str).str.strip().to_dict()
    edited_assignee_by_key = edited_tickets.set_index("key")["assignee"].fillna("").astype(str).str.strip().to_dict()

    original_status_by_key = ticket_editor_df.set_index("key")["status"].fillna("").astype(str).str.strip().to_dict()
    original_priority_by_key = ticket_editor_df.set_index("key")["priority"].fillna("").astype(str).str.strip().to_dict()
    original_assignee_by_key = ticket_editor_df.set_index("key")["assignee"].fillna("").astype(str).str.strip().to_dict()

    status_updates = {
        str(k): v for k, v in edited_status_by_key.items()
        if v and v != original_status_by_key.get(str(k), "")
    }
    priority_updates = {
        str(k): v for k, v in edited_priority_by_key.items()
        if v and v != original_priority_by_key.get(str(k), "")
    }
    assignee_updates = {
        str(k): v for k, v in edited_assignee_by_key.items()
        if v and v != original_assignee_by_key.get(str(k), "")
    }

    # Merge: start from full ticket_editor_df, overlay with editor output
    include_by_key = ticket_editor_df.set_index("key")["include"].to_dict()
    include_by_key.update({str(k): v for k, v in edited_include_by_key.items()})

    status_by_key = ticket_editor_df.set_index("key")["status"].fillna("").astype(str).str.strip().to_dict()
    status_by_key.update({str(k): v for k, v in edited_status_by_key.items()})

    original_estimate_by_key = (
        ticket_editor_df.set_index("key")["original_estimate"].fillna("").astype(str).str.strip().to_dict()
    )
    edited_estimate_by_key = dict(original_estimate_by_key)
    edited_estimate_by_key.update({str(k): v for k, v in edited_original_est_by_key_raw.items()})

    desired_in_sprint = {str(k) for k, v in include_by_key.items() if v}
    current_in_sprint = set(
        ticket_editor_df.loc[ticket_editor_df["include"], "key"].astype(str).tolist()
    )
    to_add = sorted(desired_in_sprint - current_in_sprint)
    to_backlog = sorted(current_in_sprint - desired_in_sprint)

    # full_with_edits needed only for changed-row preview table
    full_with_edits = ticket_editor_df.copy()
    full_with_edits["include"] = full_with_edits["key"].astype(str).map(include_by_key).fillna(full_with_edits["include"])
    full_with_edits["status"] = full_with_edits["key"].astype(str).map(status_by_key).fillna(full_with_edits["status"])
    full_with_edits["assignee"] = full_with_edits["key"].astype(str).map(edited_assignee_by_key).fillna(full_with_edits["assignee"])
    full_with_edits["original_estimate"] = (
        full_with_edits["key"].astype(str).map(edited_estimate_by_key).fillna(full_with_edits["original_estimate"])
    )
    parsed_estimate_seconds_by_key: dict[str, float] = {}
    invalid_estimate_keys: list[str] = []
    for key, value in edited_estimate_by_key.items():
        parsed = _parse_estimate_to_seconds(value)
        if value and parsed is None:
            invalid_estimate_keys.append(str(key))
            continue
        if parsed is not None:
            parsed_estimate_seconds_by_key[str(key)] = parsed

    estimate_updates: dict[str, str] = {}
    skipped_blank_estimates: list[str] = []
    for key, new_value in edited_estimate_by_key.items():
        old_value = original_estimate_by_key.get(key, "")
        if new_value == old_value:
            continue
        if not new_value:
            skipped_blank_estimates.append(key)
            continue
        estimate_updates[str(key)] = new_value

    changed_keys = (
        set(to_add)
        | set(to_backlog)
        | set(estimate_updates.keys())
        | set(status_updates.keys())
        | set(priority_updates.keys())
        | set(assignee_updates.keys())
    )
    if changed_keys:
        st.caption("Pending row changes (highlighted)")
        changed_preview = full_with_edits[full_with_edits["key"].astype(str).isin(changed_keys)].copy()

        def _change_type(key: str) -> str:
            parts: list[str] = []
            if key in to_add:
                parts.append("Add to sprint")
            if key in to_backlog:
                parts.append("Move to backlog")
            if key in estimate_updates:
                parts.append("Original estimate edited")
            if key in status_updates:
                parts.append(f"Status → {status_updates[key]}")
            if key in priority_updates:
                parts.append(f"Priority → {priority_updates[key]}")
            if key in assignee_updates:
                parts.append(f"Assignee → {assignee_updates[key]}")
            return " + ".join(parts)

        changed_preview["change_type"] = changed_preview["key"].astype(str).map(_change_type)
        changed_preview = changed_preview[
            [
                "change_type",
                "include",
                "key",
                "summary",
                "status",
                "priority",
                "assignee",
                "reporter",
                "original_estimate",
                "logged_time",
                "ticket_age_days",
                "idle_days",
                "created",
                "updated",
                "issue_type",
            ]
        ]

        reset_actions_df = changed_preview.copy()
        reset_actions_df.insert(0, "reset", False)
        reset_actions = st.data_editor(
            reset_actions_df,
            width="stretch",
            hide_index=True,
            disabled=[col for col in reset_actions_df.columns if col != "reset"],
            column_config={
                "reset": st.column_config.CheckboxColumn(
                    "Reset",
                    help="Tick one or more rows to reset only those pending edits",
                )
            },
            key=f"pending_reset_actions_{selected_sprint_key}_{st.session_state[editor_version_key]}",
        )

        rows_to_reset = reset_actions.loc[reset_actions["reset"], "key"].astype(str).tolist()
        if rows_to_reset:
            updated_display_editor_df = edited_tickets.copy()
            for row_key in rows_to_reset:
                base_row = ticket_editor_df[ticket_editor_df["key"].astype(str) == row_key]
                if base_row.empty:
                    continue
                row0 = base_row.iloc[0]
                reset_mask = updated_display_editor_df["key"].astype(str) == row_key
                if not reset_mask.any():
                    continue
                updated_display_editor_df.loc[reset_mask, "include"] = bool(row0["include"])
                updated_display_editor_df.loc[reset_mask, "status"] = row0["status"]
                updated_display_editor_df.loc[reset_mask, "priority"] = row0["priority"]
                updated_display_editor_df.loc[reset_mask, "assignee"] = row0["assignee"]
                updated_display_editor_df.loc[reset_mask, "original_estimate"] = row0["original_estimate"]

            st.session_state[editor_seed_key] = updated_display_editor_df
            st.session_state[editor_version_key] = int(st.session_state.get(editor_version_key, 0)) + 1
            st.rerun()

    preview_scoped = df[df["key"].isin(desired_in_sprint)].copy()
    all_sprint_tickets = df[df["sprint_name"].notna()].copy()
    status_df = status_source_df if status_source_df is not None else df
    status_all_sprint_tickets = status_df[status_df["sprint_name"].notna()].copy()
    preview_scoped["status_live"] = (
        preview_scoped["key"].astype(str).map(status_by_key).fillna(preview_scoped["status"])
    )
    all_sprint_tickets["status_live"] = (
        all_sprint_tickets["key"].astype(str).map(status_by_key).fillna(all_sprint_tickets["status"])
    )
    preview_scoped["assignee_live"] = (
        preview_scoped["key"].astype(str).map(edited_assignee_by_key).fillna(preview_scoped["assignee"])
    )
    preview_scoped["estimate_seconds_live"] = (
        preview_scoped["key"].astype(str).map(parsed_estimate_seconds_by_key)
        .fillna(preview_scoped["original_estimate_sec"])
        .fillna(0.0)
    )
    all_sprint_tickets["estimate_seconds_live"] = (
        all_sprint_tickets["key"].astype(str).map(parsed_estimate_seconds_by_key)
        .fillna(all_sprint_tickets["original_estimate_sec"])
        .fillna(0.0)
    )

    canonical_status_defaults = ["To Do", "In Progress"]
    discovered_statuses = pd.Index(status_all_sprint_tickets["status"].dropna().unique()).tolist()
    remaining_statuses = sorted([s for s in discovered_statuses if s not in canonical_status_defaults])
    workload_status_options = canonical_status_defaults + remaining_statuses
    default_workload_statuses = canonical_status_defaults.copy()

    workload_statuses_key = f"workload_statuses_{selected_sprint_key}"
    existing_workload_statuses = st.session_state.get(workload_statuses_key)
    if not isinstance(existing_workload_statuses, list):
        st.session_state[workload_statuses_key] = default_workload_statuses
    else:
        normalized_existing = [s for s in existing_workload_statuses if s in workload_status_options]
        if not normalized_existing:
            st.session_state[workload_statuses_key] = default_workload_statuses
        elif normalized_existing != existing_workload_statuses:
            st.session_state[workload_statuses_key] = normalized_existing

    if not sprint_is_open:
        st.info("Sprint membership editing is only available for future or active sprints.")
    elif not write_access.writes_enabled():
        st.info(write_access.READ_ONLY_MESSAGE)
    else:
        st.caption("`Apply sprint selection` writes sprint membership and field edits to Jira.")

    if not is_ml_sprint:
        st.warning(
            f"Sprint editing is restricted to **ML Sprint** boards. "
            f"'{selected_row['sprint_name']}' cannot be modified from this dashboard."
        )

    if skipped_blank_estimates:
        st.caption(
            f"Blank original estimate edits are ignored for {len(skipped_blank_estimates)} ticket(s). "
            "Use a Jira estimate format like `2h` or `1d 2h`."
        )
    if invalid_estimate_keys:
        st.caption(
            f"Invalid original estimate format for {len(invalid_estimate_keys)} ticket(s); "
            "live totals keep previous values for those rows."
        )

    action_col1, action_col2 = st.columns([4, 1])
    with action_col1:
        apply_sprint_selection = st.button(
            f"Apply sprint selection ({len(to_add)} add, {len(to_backlog)} backlog, {len(estimate_updates)} estimates, {len(status_updates)} status, {len(priority_updates)} priority, {len(assignee_updates)} assignee)",

            disabled=(not editable) or (not is_ml_sprint) or (not to_add and not to_backlog and not estimate_updates and not status_updates and not priority_updates and not assignee_updates),
            type="primary",
            key=f"apply_sprint_{selected_sprint_key}",
        )
    with action_col2:
        reset_sprint_changes = st.button(
            "Reset changes",
            disabled=(not editable) or (not is_ml_sprint),
            key=f"reset_sprint_{selected_sprint_key}",
            help="Discard unsaved sprint-ticket edits in Sprint Tickets.",
        )

    if reset_sprint_changes:
        st.session_state[editor_version_key] = int(st.session_state.get(editor_version_key, 0)) + 1
        st.rerun()

    if apply_sprint_selection:
        client = JiraClient.resolve(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
        )
        with st.spinner("Updating sprint membership..."):
            parts: list[str] = []
            had_success = False
            try:
                if to_add:
                    client.add_issues_to_sprint(selected_sprint_id, to_add)
                    parts.append(f"added {len(to_add)}")
                    had_success = True
                if to_backlog:
                    client.move_issues_to_backlog(to_backlog)
                    parts.append(f"moved {len(to_backlog)} to backlog")
                    had_success = True
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to update sprint membership: {exc}")

            estimate_success = 0
            estimate_failed: dict[str, str] = {}
            for key, estimate in estimate_updates.items():
                try:
                    client.update_issue(key, {"timetracking": {"originalEstimate": estimate}})
                    estimate_success += 1
                except Exception as exc:  # noqa: BLE001
                    estimate_failed[key] = str(exc)

            if estimate_success:
                parts.append(f"updated {estimate_success} estimate(s)")
                had_success = True
            if estimate_failed:
                for key, err in estimate_failed.items():
                    st.error(f"Estimate update failed for {key}: {err}")

            status_success = 0
            status_failed: dict[str, str] = {}
            for key, new_status in status_updates.items():
                try:
                    client.transition_issue_to_status(key, new_status)
                    status_success += 1
                except Exception as exc:  # noqa: BLE001
                    status_failed[key] = str(exc)
            if status_success:
                parts.append(f"updated {status_success} status(es)")
                had_success = True
            if status_failed:
                for key, err in status_failed.items():
                    st.error(f"Status update failed for {key}: {err}")

            priority_success = 0
            priority_failed: dict[str, str] = {}
            for key, new_priority in priority_updates.items():
                try:
                    client.set_priority(key, new_priority)
                    priority_success += 1
                except Exception as exc:  # noqa: BLE001
                    priority_failed[key] = str(exc)
            if priority_success:
                parts.append(f"updated {priority_success} priority(ies)")
                had_success = True
            if priority_failed:
                for key, err in priority_failed.items():
                    st.error(f"Priority update failed for {key}: {err}")

            assignee_success = 0
            assignee_failed: dict[str, str] = {}
            for key, new_assignee in assignee_updates.items():
                try:
                    normalized = str(new_assignee).strip()
                    if normalized.lower() == "unassigned":
                        client.update_issue(key, {"assignee": None})
                    else:
                        account_id = assignee_name_to_account_id.get(normalized)
                        if not account_id:
                            raise RuntimeError(
                                f"No account id found for assignee '{normalized}'."
                            )
                        client.update_issue(key, {"assignee": {"accountId": account_id}})
                    assignee_success += 1
                except Exception as exc:  # noqa: BLE001
                    assignee_failed[key] = str(exc)
            if assignee_success:
                parts.append(f"updated {assignee_success} assignee(s)")
                had_success = True
            if assignee_failed:
                for key, err in assignee_failed.items():
                    st.error(f"Assignee update failed for {key}: {err}")

            if parts:
                st.success("Update completed: " + ", ".join(parts))
            if had_success:
                _clear_page_caches(ENGINEERING_PAGE_TITLE)
                st.session_state.pop(editor_seed_key, None)
                st.session_state[editor_version_key] = int(st.session_state.get(editor_version_key, 0)) + 1
                st.rerun()

    # ---- Epics in Sprint ----
    epic_display_cols = [
        c for c in [
            "key", "summary", "status", "priority", "assignee",
            "original_estimate", "reporter", "logged_time", "completion_pct",
            "ticket_age_days", "idle_days", "created", "updated", "issue_type",
        ]
        if c in epic_sprint_df.columns
    ]
    with st.expander(
        f"Epics in Sprint ({len(epic_sprint_df)})",
        expanded=not epic_sprint_df.empty,
    ):
        if epic_sprint_df.empty:
            st.caption("No epics are currently assigned to this sprint.")
        else:
            # Create display dataframe with linked key column in the correct position
            epic_df_display = epic_sprint_df[epic_display_cols].sort_values(["assignee", "key"], ascending=[True, True]).copy()
            epic_df_display["key_url"] = epic_df_display["key"].apply(_jira_ticket_url)
            epic_df_display = epic_df_display.drop(columns=["key"])
            visible_epic_columns = [
                "key_url",
                "summary",
                "status",
                "priority",
                "assignee",
                "original_estimate",
                "reporter",
                "logged_time",
                "completion_pct",
                "ticket_age_days",
                "idle_days",
                "created",
                "updated",
                "issue_type",
            ]
            st.dataframe(
                epic_df_display,
                width="stretch",
                hide_index=True,
                column_order=visible_epic_columns,
                column_config={
                    "key_url": st.column_config.LinkColumn(
                        "Key",
                        display_text=JIRA_KEY_DISPLAY_PATTERN,
                    ),
                    "summary": st.column_config.TextColumn("Summary"),
                    "status": st.column_config.TextColumn("Status"),
                    "priority": st.column_config.TextColumn("Priority"),
                    "assignee": st.column_config.TextColumn("Assignee"),
                    "original_estimate": st.column_config.TextColumn("Estimate"),
                    "reporter": st.column_config.TextColumn("Reporter"),
                    "logged_time": st.column_config.TextColumn("Logged"),
                    "completion_pct": st.column_config.NumberColumn("Done %", format="%.0f%%"),
                    "ticket_age_days": st.column_config.NumberColumn("Age (days)", format="%.1f"),
                    "idle_days": st.column_config.NumberColumn("Idle (days)", format="%.1f"),
                    "created": st.column_config.TextColumn("Created at"),
                    "updated": st.column_config.TextColumn("Updated at"),
                    "issue_type": st.column_config.TextColumn("Type"),
                },
            )

    calc_col1, calc_col2 = st.columns([2, 3])
    workload_statuses = st.multiselect(
        "Statuses counted in hours",
        options=workload_status_options,
        help="Use this to focus sprint effort on work that still needs attention.",
        key=workload_statuses_key,
    )

    if workload_statuses:
        preview_workload = preview_scoped[preview_scoped["status_live"].isin(workload_statuses)].copy()
    else:
        preview_workload = preview_scoped.iloc[0:0].copy()

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        (
            '<div style="font-size: 0.875rem; opacity: 0.85; margin-bottom: 0.2rem;">Tickets in sprint</div>'
            f'<div style="font-size: 2.2rem; font-weight: 700; line-height: 1.05;">'
            f'{len(preview_workload)} '
            '<span style="font-size: 0.75rem; font-weight: 500; opacity: 0.75; vertical-align: middle;">out of</span> '
            f'{len(preview_scoped)}</div>'
        ),
        unsafe_allow_html=True,
    )
    capacity_section = "Sprint capacity"
    _report(TAB_ENGINEERING).figure(
        capacity_section,
        "Tickets in sprint",
        f"{len(preview_workload)}",
        f"out of {len(preview_scoped)}",
    )
    _tile(
        c2,
        TAB_ENGINEERING,
        capacity_section,
        "Total estimated (sprint)",
        _fmt_seconds(preview_workload["estimate_seconds_live"].fillna(0).sum()),
        help="Sum of estimates for In Sprint ✅ tickets matching the selected statuses and assignee filter.",
    )
    _tile(
        c3,
        TAB_ENGINEERING,
        capacity_section,
        "Grand Total (in sprint)",
        _fmt_seconds(preview_scoped["estimate_seconds_live"].fillna(0).sum()),
        help="Sum of estimates for all In Sprint ✅ tickets regardless of status filter.",
    )

    # ---- Status breakdown pills (In Sprint tickets) ----
    _render_status_pills(preview_scoped["status_live"])

    # Per-assignee breakdown
    st.markdown("##### Capacity per Assignee")
    show_logged_details = st.checkbox(
        "Display Logged Time",
        value=False,
        key=f"show_logged_details_{selected_sprint_key}",
    )
    agg = (
        preview_workload.groupby("assignee_live")
        .agg(
            tickets=("key", "count"),
            estimated_sec=("estimate_seconds_live", "sum"),
            logged_sec=("time_spent_sec", "sum"),
        )
        .reset_index()
    )
    agg["Total Estimated"] = agg["estimated_sec"].apply(_fmt_seconds)
    agg["Total Logged"] = agg["logged_sec"].apply(_fmt_seconds)
    agg["Remaining"] = (agg["estimated_sec"] - agg["logged_sec"]).clip(lower=0).apply(_fmt_seconds)
    agg = agg.rename(columns={"assignee_live": "Assignee", "tickets": "Tickets"})
    capacity_columns = ["Assignee", "Tickets", "Total Estimated"]
    if show_logged_details:
        capacity_columns.extend(["Total Logged", "Remaining"])
    st.dataframe(
        agg[capacity_columns],
        width="stretch",
        hide_index=True,
    )

    _render_hourly_capacity(scoped, preview_scoped)


def _sprint_window(sprint_df: pd.DataFrame) -> tuple[object, object]:
    """Start and end of a sprint, taken from a single row so they cannot mismatch."""
    if not {"sprint_start", "sprint_end"}.issubset(sprint_df.columns):
        return None, None
    dated = sprint_df[sprint_df["sprint_start"].notna() & sprint_df["sprint_end"].notna()]
    if dated.empty:
        return None, None
    row = dated.iloc[0]
    start = pd.to_datetime(row["sprint_start"], errors="coerce", utc=True)
    end = pd.to_datetime(row["sprint_end"], errors="coerce", utc=True)
    if pd.isna(start) or pd.isna(end):
        return None, None
    return start, end


def _render_hourly_capacity(sprint_df: pd.DataFrame, in_sprint_df: pd.DataFrame) -> None:
    """Committed hours against each person's declared availability.

    Part-time and hourly engineers make raw committed totals unreadable, so the
    hours per week come from JIRA_WEEKLY_HOURS and are spread across the
    sprint's own working days.
    """
    st.markdown("##### Availability vs Commitment")
    if not WEEKLY_HOURS:
        st.caption(
            "Set JIRA_WEEKLY_HOURS (e.g. \"Tam=10,Jal=20\") to compare committed "
            "hours against what each person is actually available for."
        )
        return

    start, end = _sprint_window(sprint_df)
    days = working_days(start, end)
    if not days:
        st.caption(
            "This sprint has no start/end dates in Jira, so available hours cannot "
            "be derived. Set the sprint dates on the board."
        )
        return

    if in_sprint_df.empty:
        committed = pd.Series(dtype="float64")
    else:
        owners = in_sprint_df["assignee_live"].fillna("Unassigned").astype(str).str.strip()
        committed = (
            pd.to_numeric(in_sprint_df["estimate_seconds_live"], errors="coerce")
            .fillna(0.0)
            .div(3600.0)
            .groupby(owners.mask(owners.eq(""), "Unassigned"))
            .sum()
        )
    # The roster has to follow the scope: outside it a person's tickets are not
    # loaded, so they would read as idle when they are merely filtered out.
    in_scope = st.session_state.get(_SCOPE_ASSIGNEES_KEY)
    roster = (
        WEEKLY_HOURS
        if in_scope is None
        else {
            name: hours
            for name, hours in WEEKLY_HOURS.items()
            # Roster names are short ("Farid"), scope names are Jira display
            # names ("Farid Shahidi"), so compare them the same loose way.
            if any(
                match_weekly_hours(person, {name: hours}) is not None
                for person in in_scope
            )
        }
    )
    table = capacity_table(committed, roster, start, end)
    if table.empty:
        st.caption("No assignees to report on for this sprint.")
        return

    st.caption(
        f"{days:.0f} working day(s) in this sprint "
        f"({pd.Timestamp(start).date()} to {pd.Timestamp(end).date()}). "
        "Committed covers every ticket in the sprint that the current scope and "
        "filters keep, not just the statuses counted in hours above. Utilization is "
        'committed / available; "Unknown" means no weekly hours are declared for '
        'that person, and "Ambiguous roster name" means a declaration like '
        '"Dan=40" matches more than one person in Jira, so it is withheld rather '
        "than handed to both - spell that entry as the full Jira name to fix it."
    )
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Committed (h)": st.column_config.NumberColumn(format="%.1f"),
            "Available (h)": st.column_config.NumberColumn(format="%.1f"),
            "Utilization %": st.column_config.NumberColumn(format="%.0f%%"),
            "Delta (h)": st.column_config.NumberColumn(
                format="%.1f",
                help="Available minus committed; negative means over-committed.",
            ),
        },
    )


# Tinted background with a saturated text colour of the same hue, so the pills
# sit on the light theme the KPI cards are drawn for.
_STAGE_COLORS: dict[str, tuple[str, str]] = {
    # (background, text)
    "Backlog":               ("#eef0f5", "#525c73"),
    "DISCUSSION NEEDED":     ("#fdecec", "#b42318"),
    "To Do":                 ("#e8f0fd", "#1d4ed8"),
    "In Progress":           ("#e6f6ec", "#15803d"),
    "IN DEV ENV":            ("#e6f2fd", "#0369a1"),
    "Code Review":           ("#f1ecfd", "#6d28d9"),
    "Review in Staging":     ("#fdf3e0", "#a16207"),
    "Ready for Production":  ("#e7f8ea", "#15803d"),
    "Review":                ("#fdefe5", "#c2410c"),
}
_DEFAULT_PILL: tuple[str, str] = ("#f1f2f4", "#4b5563")


def _render_status_pills(status_series: pd.Series) -> None:
    """Render a compact row of color-coded status pills with ticket counts."""
    counts = status_series.fillna("Unknown").value_counts().sort_index()
    if counts.empty:
        return
    pills_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 4px 0;">'
    for status, count in counts.items():
        bg, fg = _STAGE_COLORS.get(str(status), _DEFAULT_PILL)
        pills_html += (
            f'<span style="'
            f'background:{bg};color:{fg};'
            f'border-radius:6px;padding:4px 10px;'
            f'font-size:0.78rem;font-weight:600;white-space:nowrap;'
            f'border:1px solid {fg}22;'
            f'">'
            f'{html.escape(str(status))} '
            f'<span style="opacity:0.75;font-weight:400;">({int(count)})</span>'
            f'</span>'
        )
    pills_html += "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)


_PRIORITY_BUCKET_MAP = {
    "low": "Normal",
    "lowest": "Normal",
    "normal": "Normal",
    "medium": "Normal",
    "high": "High",
    "highest": "Urgent",
    "urgent": "Urgent",
}
_BUCKET_COLORS = {"Normal": "#2ECC71", "High": "#F5A623", "Urgent": "#E74C3C"}


def _render_bubble_chart(
    df: pd.DataFrame,
    color_by: str = "priority",
    agg_priority: bool = False,
    chart_key: str = "bubble_chart",
) -> str | None:
    if df.empty:
        st.info("No data available for staleness bubble chart.")
        return None

    plot_df = df.copy()

    STATUS_ORDER = [
        "Backlog",
        "DISCUSSION NEEDED",
        "To Do",
        "In Progress",
        "IN DEV ENV",
        "Review in Staging",
        "Code Review",
        "Ready for Production",
    ]
    statuses = plot_df["status"].fillna("Unknown")
    # Any status not in the fixed list gets appended at the top.
    extra = [s for s in statuses.unique() if s not in STATUS_ORDER]
    full_order = STATUS_ORDER + extra
    status_to_y = {s: i for i, s in enumerate(full_order)}
    rng = np.random.default_rng(seed=42)
    plot_df["y_jitter"] = (
        statuses.map(status_to_y).astype(float)
        + rng.uniform(-0.35, 0.35, size=len(plot_df))
    )
    plot_df["status_label"] = statuses

    age = plot_df["ticket_age_days"].clip(lower=1)
    plot_df["bubble_size"] = ((age - age.min()) / (age.max() - age.min() + 1e-9) * 31 + 3).round(1)

    plot_df["marker_symbol"] = (
        plot_df["issue_type"].fillna("").astype(str).str.strip().str.lower()
        .map(lambda t: "triangle-up" if t == "epic" else "circle")
    )

    if agg_priority and color_by == "priority":
        plot_df["priority_bucket"] = (
            plot_df["priority"].fillna("none").astype(str).str.strip().str.lower()
            .map(_PRIORITY_BUCKET_MAP)
            .fillna("Normal")
        )
        fig = px.scatter(
            plot_df,
            x="idle_days",
            y="y_jitter",
            size="bubble_size",
            color="priority_bucket",
            color_discrete_map=_BUCKET_COLORS,
            category_orders={"priority_bucket": ["Normal", "High", "Urgent"]},
            custom_data=["key", "summary", "assignee", "status_label", "priority", "ticket_age_days", "idle_days", "issue_type"],
            # title="Staleness vs Workflow Status (Aggregated Priority)",
            labels={"idle_days": "Idle Days", "y_jitter": "Status", "priority_bucket": "Priority"},
            size_max=34,
            opacity=0.3,
        )
    else:
        fig = px.scatter(
            plot_df,
            x="idle_days",
            y="y_jitter",
            size="bubble_size",
            color=color_by,
            custom_data=["key", "summary", "assignee", "status_label", "priority", "ticket_age_days", "idle_days", "issue_type"],
            title="Staleness vs Workflow Status",
            labels={"idle_days": "Idle Days", "y_jitter": "Status"},
            size_max=34,
            opacity=0.3,
        )

    for trace in fig.data:
        custom_rows = getattr(trace, "customdata", None)
        if custom_rows is None:
            continue
        trace.marker.symbol = [
            "triangle-up" if str(row[7]).strip().lower() == "epic" else "circle"
            for row in custom_rows
        ]

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br>"
            "Assignee: %{customdata[2]}<br>"
            "Status: %{customdata[3]}<br>"
            "Priority: %{customdata[4]}<br>"
            "Age: %{customdata[5]:.1f} days<br>"
            "Idle: %{customdata[6]:.1f} days<br>"
            "Type: %{customdata[7]}"
            "<extra></extra>"
        )
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(status_to_y.values()),
        ticktext=list(status_to_y.keys()),
        title="Status",
    )
    fig.update_layout(height=560)
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", key=chart_key)
    points = (event or {}).get("selection", {}).get("points", [])
    if points:
        return str(points[0].get("customdata", [None])[0])
    return None

def _apply_action_with_audit(
    client: JiraClient,
    action_type: str,
    selected_keys: list[str],
    target: str,
    source_status: str | None = None,
    parent_operation_id: str | None = None,
) -> tuple[list[str], dict[str, str], dict[str, object]]:
    operation = new_operation_record(
        action_type=action_type,
        target=target,
        selected_keys=selected_keys,
        source_status=source_status,
        parent_operation_id=parent_operation_id,
    )

    items: list[dict[str, object]] = []
    succeeded: list[str] = []
    failed: dict[str, str] = {}

    for key in selected_keys:
        before_snapshot: dict[str, object] = {}
        try:
            before_snapshot = client.get_issue_snapshot(key)

            if action_type == "priority":
                client.set_priority(key, target)
            elif action_type == "status":
                client.transition_issue_to_status(key, target)
            elif action_type == "revert_priority":
                prior_id = str(target).strip()
                if prior_id:
                    client.set_priority_by_id(key, prior_id)
                else:
                    raise RuntimeError("Cannot revert: original priority id is missing.")
            elif action_type == "revert_status":
                client.transition_issue_to_status(key, target)
            else:
                raise RuntimeError(f"Unsupported action type: {action_type}")

            after_snapshot = client.get_issue_snapshot(key)
            items.append(
                {
                    "key": key,
                    "success": True,
                    "before": before_snapshot,
                    "after": after_snapshot,
                }
            )
            succeeded.append(key)
        except Exception as exc:  # noqa: BLE001
            failed[key] = str(exc)
            items.append(
                {
                    "key": key,
                    "success": False,
                    "before": before_snapshot,
                    "error": str(exc),
                }
            )

    operation = finalize_operation(operation, items)
    log_error = append_operation(operation)
    if log_error:
        st.warning(
            "The changes went through, but the audit log could not be written, "
            f"so this batch cannot be reverted from the dashboard - {log_error}"
        )
    return succeeded, failed, operation


def _contribution_pie(
    labels: pd.Series, value_name: str, title: str, unavailable: bool = False
) -> None:
    """Render a 'who did how much' pie from a series of names, or a note if empty.

    ``unavailable`` distinguishes a failed lookup from a genuinely empty window so
    an errored fetch doesn't masquerade as an authoritative "nobody did anything".
    """
    counts = labels.value_counts()
    if counts.empty:
        if unavailable:
            st.caption(f"Could not load {value_name} \u2014 try Refresh Data.")
        else:
            st.caption(f"No {value_name} in the last 30 days.")
        return
    frame = pd.DataFrame({"who": counts.index, value_name: counts.values})
    fig = px.pie(frame, names="who", values=value_name, title=title)
    fig.update_traces(textposition="inside", textinfo="value+label")
    fig.update_layout(height=340, margin=dict(t=48, b=8, l=8, r=8), showlegend=True)
    st.plotly_chart(fig, width="stretch")


def _metric_value(count: int | None) -> str | int:
    """Show a real count, or an em dash when the number is unavailable (not 0)."""
    return "—" if count is None else int(count)


def _render_ticket_list(
    df: pd.DataFrame | None, empty_msg: str, total_count: int | None = None
) -> None:
    """Render a compact clickable ticket table (key/summary/status/assignee/age).

    ``None`` means the fetch failed (distinct from an empty result): say so
    rather than printing a reassuring "nothing here" that hides an outage.
    ``total_count`` is Jira's uncapped count; when the fetched frame is smaller
    (paging cap hit) a caption says so, so the list can't quietly disagree with
    the headline tile.
    """
    if df is None:
        st.caption("Could not load — try Refresh Data.")
        return
    if df.empty:
        st.caption(empty_msg)
        return
    view = df.copy()
    view["key_url"] = view["key"].map(_jira_ticket_url)
    if "created" in view.columns:
        created = pd.to_datetime(view["created"], utc=True, errors="coerce")
        view["age_days"] = (
            pd.Timestamp.now(tz="UTC") - created
        ).dt.total_seconds() / 86400.0
    else:
        view["age_days"] = pd.NA
    cols = ["key_url", "summary", "status", "priority", "assignee", "age_days"]
    cols = [c for c in cols if c in view.columns]
    st.dataframe(
        view[cols],
        width="stretch",
        hide_index=True,
        column_config={
            "key_url": st.column_config.LinkColumn(
                "Key", display_text=JIRA_KEY_DISPLAY_PATTERN
            ),
            "summary": st.column_config.TextColumn("Summary", width="large"),
            "status": st.column_config.TextColumn("Status"),
            "priority": st.column_config.TextColumn("Priority"),
            "assignee": st.column_config.TextColumn("Assignee"),
            "age_days": st.column_config.NumberColumn("Age (days)", format="%.0f"),
        },
    )
    if total_count is not None and len(view) < int(total_count):
        st.caption(
            f"Showing the {len(view)} of {int(total_count)} matching tickets that fit "
            "the fetch limit; the tile above is the exact count."
        )


def _render_new_and_triage(
    new_24h: int | None,
    new_7d: int | None,
    triage_stuck: int | None,
    new_tickets_7d: pd.DataFrame | None,
    triage_tickets: pd.DataFrame | None,
    triage_hours: int,
) -> None:
    """Intake health: brand-new work and tickets sitting in triage too long."""
    st.subheader("New & Untriaged Work")
    c1, c2, c3 = st.columns(3)
    intake = "New & untriaged work"
    _tile(c1, TAB_ENGINEERING, intake, "New tickets (24h)", _metric_value(new_24h))
    _tile(c2, TAB_ENGINEERING, intake, "New tickets (7d)", _metric_value(new_7d))
    _tile(
        c3,
        TAB_ENGINEERING,
        intake,
        f"Stuck in triage (> {triage_hours}h)",
        _metric_value(triage_stuck),
    )

    with st.expander(f"Stuck in triage — {', '.join(TRIAGE_STATUSES)} > {triage_hours}h, oldest first", expanded=True):
        _render_ticket_list(
            triage_tickets,
            f"Nothing has been sitting in {', '.join(TRIAGE_STATUSES)} longer than {triage_hours}h.",
            total_count=triage_stuck,
        )
        st.caption(
            "Stuck = currently in a triage status, created more than "
            f"{triage_hours}h ago, and no status change since. Org-wide. "
            "Configure with JIRA_TRIAGE_STATUSES / JIRA_TRIAGE_STUCK_HOURS."
        )
    with st.expander("New tickets in the last 7 days, newest first", expanded=False):
        _render_ticket_list(
            new_tickets_7d,
            "No tickets created in the last 7 days.",
            total_count=new_7d,
        )


# A year of orders, so the wine and merchant tables can look back 360 days, plus
# the days a 30-day figure needs to be shown as up or down on the month before.
ORDER_BOOK_DAYS = 390
ORDER_BOOK_TTL_SECONDS = 900


@st.cache_data(ttl=ORDER_BOOK_TTL_SECONDS, show_spinner=False, refresh_mode="background")
def _order_book(source: str, days: int) -> orders_client.OrderBook:
    """The year of orders, re-read whole when the cache lapses.

    Keyed on the source's label rather than the config, so the password never
    becomes part of a cache key. Reading the year outright costs a single
    sub-second query, which is why there is no incremental top-up to go wrong.
    """
    config = orders_client.load_medusa_env()
    if config is None or config.label != source:
        raise orders_client.MedusaConfigError(
            "The order database configuration changed while it was being read."
        )
    return orders_client.read_order_book(config, days)


@st.cache_data(ttl=3600, show_spinner=False, refresh_mode="background")
def fetch_store_prefixes_cached(source: str) -> dict[str, str]:
    config = orders_client.load_medusa_env()
    if config is None or config.label != source:
        raise orders_client.MedusaConfigError(
            "The order database configuration changed while it was being read."
        )
    return orders_client.fetch_stores(config)


def _unmathed(text: str) -> str:
    """A sentence with money in it, safe to hand to ``st.markdown``.

    Streamlit reads a pair of dollar signs on one line as inline LaTeX, so a
    sentence saying ``$399 of $1,426`` renders as maths with both symbols eaten
    and the figures between them in italics. Escaped only where it is drawn: the
    same sentence goes into the printable report, which wants the plain text.
    """
    return text.replace("$", "\\$")


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
        st.plotly_chart(figure, width="stretch", key="orders_daily")

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


@st.cache_data(ttl=BENCHMARK_TTL_SECONDS, show_spinner=False)
def _price_benchmark_cached(account: str, country: str) -> BenchmarkRead:
    """What Merchant Center says the shop's prices look like against the market.

    Keyed on the account rather than on the config, so that Streamlit hashes a
    string: the credential is read again inside, as the billing client is, and
    never becomes part of a cache key.
    """
    config = merchant_client.load_merchant_env()
    if config is None or (config.account, config.country) != (account, country):
        raise merchant_client.MerchantConfigError(
            "The Merchant Center configuration changed while it was being read."
        )
    token = merchant_client.access_token(config)
    prices = merchant_client.price_gaps(config, token)
    # Suggestions are a second report and a nice-to-have: an account with price
    # competitiveness but no price insights still gets the headline.
    try:
        insights = merchant_client.price_insights(config, token)
    except Exception:  # noqa: BLE001
        insights = merchant_client.Insights(pd.DataFrame())
    # Likewise the clicks: without them the tables lose their ordering, not
    # their subject, and the headline above them does not depend on demand.
    try:
        demand = merchant_client.product_demand(config, token, country)
    except Exception:  # noqa: BLE001
        # Marked unread rather than empty: an empty report says nobody clicked,
        # and the panel would otherwise print that as a finding about the shop.
        demand = merchant_client.Demand(pd.DataFrame(), read=False)
    return BenchmarkRead(prices, insights, demand)


# The catalogue moves slower than the prices in it, and the whole feed's
# merchants are read once and then handed to every table below, so it is held
# for a day.
_OFFER_MERCHANTS_TTL_SECONDS = 24 * 3600

# Orders arrive all day, but a quarter of them is a slow-moving figure and this
# read groups the whole quarter, so it is held on the benchmarks' own cycle.
_OFFER_SALES_TTL_SECONDS = 6 * 3600


@st.cache_data(ttl=_OFFER_SALES_TTL_SECONDS, show_spinner=False)
def _offer_sales_cached(source: str, days: int, today: _dt.date) -> pd.DataFrame:
    """Bottles sold per Google offer, from the shop's own order book.

    Keyed on the day as well as the window so the quarter rolls forward with
    the calendar rather than whenever the cache happens to expire.
    """
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


@st.cache_data(ttl=_OFFER_MERCHANTS_TTL_SECONDS, show_spinner=False)
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
    config = orders_client.load_medusa_env()
    if config is None or config.label != source:
        raise orders_client.MedusaConfigError(
            "The order database configuration changed while it was being read."
        )
    handles = orders_client.fetch_offer_handles(config, list(offers))
    prefixes = orders_client.fetch_stores(config)
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
        ring = px.pie(
            slices,
            names=slices["band"].astype(str),
            values="listings",
            color=slices["band"].astype(str),
            color_discrete_map=colours,
            hole=0.35,
            title=f"{merchant}: compared wines, by price against the market",
        )
        ring.update_traces(textinfo="percent", hovertemplate="%{label}: %{value} wines")
        ring.update_layout(margin=dict(t=54, b=0, l=0, r=0), legend_title_text="")
        st.plotly_chart(ring, width="stretch")
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
            title="Bottles sold per 100 shoppers who looked",
            text=rated["per_100_clicks"].map(lambda value: f"{value:.0f}"),
        )
        bars.update_layout(
            margin=dict(t=54, b=0, l=0, r=0),
            showlegend=False,
            xaxis_title="",
            yaxis_title="",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(bars, width="stretch")


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
            ring = px.pie(
                split,
                names="outcome",
                values="spend",
                color="outcome",
                color_discrete_map=ads_evidence.SPLIT_COLOURS,
                hole=0.35,
                title=f"Ad spend, last {merchant_client.SALES_DAYS} days",
            )
            ring.update_traces(
                textinfo="percent",
                # A written sentence per slice rather than a template over
                # ``customdata``: the money is formatted here anyway, to carry
                # the currency Google billed rather than Plotly's ``$``.
                hovertext=ads_evidence.split_hovers(split, money),
                hovertemplate="%{hovertext}<extra></extra>",
            )
            ring.update_layout(margin=dict(t=54, b=0, l=0, r=0), legend_title_text="")
            st.plotly_chart(ring, width="stretch")
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
            title=f"Revenue per {_money(1, money)} of ads, by price against market",
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
        st.plotly_chart(bars, width="stretch")


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
        with st.spinner("Reading Merchant Center's price benchmarks..."):
            read = _price_benchmark_cached(config.account, config.country)
    except merchant_client.MerchantConfigError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read the price benchmarks: {str(exc)[:200]}")
        return

    read = read._replace(sales=_offer_sales())
    prices, insights = read.prices, read.insights
    # Held before the merchant filter can empty it, so an empty filter is never
    # mistaken below for a feed with no benchmarks in it.
    whole_feed_counted = prices.counted
    filtered_to_nothing = False
    chosen = _EVERY_MERCHANT
    # Whose wine each offer is, read once for the whole catalogue so the same
    # names can both filter it and label the rows below.
    named = _offer_merchants(prices.offers)
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
        ) = st.tabs(
            [
                f"Ask the merchants ({merchant_client.ASK_LIST})",
                "Cheaper than the market",
                "What price did to sales",
                "Where the ad money went",
                "Most clicked",
                "Try a sale price",
                "Most expensive bottles",
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
            _money_delta(spend.cost_change, money) if spend.prev_cost else None
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

    if earned:
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
    elif (
        commission is None
        and not (sales and comparable and sales.revenue)
        and spend.cost
        and spend.conversion_value
    ):
        # No ceiling could be computed - no Stripe ledger, and either no
        # comparable takings or none captured in the window - but the
        # attributed return needs only the ad account's own figures, which are
        # in its own currency by definition. A measured commission that came
        # to zero is a different story and keeps the block above.
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


@st.cache_data(ttl=BURN_TTL_SECONDS, show_spinner=False)
def _openai_costs_cached(days: int) -> pd.DataFrame:
    """Daily OpenAI cost by project and line item.

    Not keyed on the credential: it comes from the environment, does not change
    while the process lives, and an admin key has no business in a cache key.
    """
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


@st.cache_data(ttl=BURN_TTL_SECONDS, show_spinner=False)
def _stripe_ledger_cached(days: int) -> tuple[pd.DataFrame, bool]:
    """Stripe's ledger for the window and the window before.

    Separate from the disputes beside it because two panels want the ledger and
    only one wants the disputes: a busy platform's ledger runs to a hundred
    pages, and the ads panel has no use for a chargeback count.
    """
    key = cost_client.load_stripe_env()
    if not key:  # pragma: no cover - the caller checks first
        raise cost_client.CostConfigError("No Stripe key is configured.")
    return cost_client.stripe_ledger(key, days)


@st.cache_data(ttl=BURN_TTL_SECONDS, show_spinner=False)
def _stripe_disputes_cached(days: int) -> int:
    key = cost_client.load_stripe_env()
    if not key:  # pragma: no cover - the caller checks first
        raise cost_client.CostConfigError("No Stripe key is configured.")
    return cost_client.stripe_disputes(key, days)


def _stripe_cached(days: int) -> tuple[pd.DataFrame, bool, int]:
    """Stripe's ledger for the window and the window before, and its disputes."""
    entries, truncated = _stripe_ledger_cached(STRIPE_LEDGER_DAYS)
    return entries, truncated, _stripe_disputes_cached(days)


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
    _render_ai_costs(days)
    _render_cloud(days)
    _render_stripe(days)


class CloudRead(NamedTuple):
    """Cloud charges, and what the export they came from covers."""

    costs: pd.DataFrame
    history_start: _dt.date | None
    covered_to: _dt.date | None
    # Fully-qualified, and more than one when several billing accounts export
    # into the same dataset. The first is the one read.
    tables: tuple[str, ...] = ()


# The bill moves once a day at best and its last day is yesterday, so a quarter
# of an hour buys no freshness and pays two whole-table scans of the export for
# it - the coverage probe has no date to filter on, and `DATE(usage_start_time)`
# does not prune the export's ingestion-time partitions either. Held on the ads
# panel's cycle instead, keyed on the date so it rolls over when the export does.
CLOUD_TTL_SECONDS = 6 * 3600
# The widest window the panel offers, which is the only one read: `window`
# slices narrower ones out of the frame in pandas, so a click on the radio no
# longer sends BigQuery after a subset of rows already in hand.
CLOUD_WINDOW_DAYS = max(cost_client.LOOKBACK_WINDOWS)


@st.cache_data(ttl=CLOUD_TTL_SECONDS, show_spinner=False)
def _cloud_costs_cached(days: int, today: _dt.date) -> CloudRead:
    """Google Cloud's billing export, and what the export covers.

    Read to the export's own last day rather than to today: it is written in
    arrears, so the days it has not reached are days it has no charges for, and
    a window ending now would average real spend over imaginary free days.
    """
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


def _render_cloud(days: int) -> None:
    """What Google Cloud charged, service by service.

    The largest bill of the three and the one nobody sees: it arrives monthly,
    by which time a service left running has been running for a month. Read
    from the billing export, which is the only place the figure exists per day.
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
            else None
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


def _render_ai_costs(days: int) -> None:
    """What the AI providers charged, and what the bill is actually made of."""
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
            _money_delta(burn.cost_change, money) if burn.prev_cost else None
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
                    f"{row['project'] or 'unnamed'} {_money(float(row['cost']), money)}"
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


def _render_stripe(days: int) -> None:
    """What Stripe's books say the platform kept.

    Read expecting a cost - card processing fees - and it is not one: this
    account is a Connect platform, so each sale's fees are charged on the
    merchant's own account and what lands here is the marketplace's commission.
    The panel reports what is there rather than what was hoped for.
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

    try:
        with st.spinner("Reading Stripe's ledger..."):
            entries, truncated, disputes = _stripe_cached(days)
    except cost_client.CostConfigError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read Stripe: {str(exc)[:200]}")
        return

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


@st.cache_data(ttl=FUNNEL_TTL_SECONDS, show_spinner=False, refresh_mode="background")
def _funnel_cached(
    credentials: tuple[str, str, str],
    funnel_spec: str,
    days: int,
    offset_days: int = 0,
) -> pd.DataFrame:
    steps = amplitude_client.parse_funnel(funnel_spec)
    return amplitude_client.funnel(credentials, steps, days, offset_days)


@st.cache_data(ttl=FUNNEL_TTL_SECONDS, show_spinner=False, refresh_mode="background")
def _breakdown_cached(
    credentials: tuple[str, str, str], event: str, prop: str, days: int
) -> pd.DataFrame:
    return amplitude_client.event_breakdown(credentials, event, prop, days)


@st.cache_data(ttl=FUNNEL_TTL_SECONDS, show_spinner=False, refresh_mode="background")
def _event_users_cached(
    credentials: tuple[str, str, str],
    events: tuple[tuple[str, str], ...],
    days: int,
) -> pd.DataFrame:
    """Events as plain pairs, because Streamlit hashes its cache keys itself and
    is not obliged to know how to hash this module's dataclass."""
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
        delta=f"-{int(worst['lost']):,} people",
        delta_color="inverse",
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
    st.plotly_chart(figure, width="stretch", key="funnel_steps")

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
    """
    friction_tab, ai_tab = st.tabs(["What went wrong", "Voss AI"])
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


def _delta_arrow(change: str | None) -> dict:
    """``st.metric`` arguments that do not call standing still an improvement.

    Streamlit colours a tile's delta by whether the string begins with a minus,
    so "flat" and "+0 people" would both be drawn as a green arrow upwards. In a
    table or a sentence those words are read; on a tile only the arrow is.
    """
    if change is None:
        return {}
    return {"delta": change, "delta_color": "off" if _unmoved(change) else "normal"}


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
    pairs = tuple((step.label, step.event) for step in events)
    try:
        with st.spinner("Counting..."):
            counts = _event_users_cached(credentials, pairs, days)
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


def _render_resolved_summary(
    ticket_count_7: int | None,
    ticket_count_30: int | None,
    resolved_30: pd.DataFrame | None,
    pr_count_7: int | None,
    pr_count_30: int | None,
    merged_prs: pd.DataFrame,
    github_ready: bool,
    github_error: str = "",
) -> None:
    """Login landing snapshot: tickets and PRs resolved in the last 7 / 30 days,
    with a pie for who resolved tickets and who merged PRs. Tile counts come from
    server-side counts (Jira's approximate count for tickets, GitHub's exact
    issueCount for PRs); the dataframes only drive the pies. A ``None`` count
    means the lookup failed and renders as "—", distinct from a genuine 0."""
    st.subheader("Resolved in the Last 7 / 30 Days")

    c1, c2, c3, c4 = st.columns(4)
    shipped = "Resolved in the last 7 / 30 days"
    _tile(
        c1, TAB_ENGINEERING, shipped, "Tickets resolved (7d)", _metric_value(ticket_count_7)
    )
    _tile(
        c2,
        TAB_ENGINEERING,
        shipped,
        "Tickets resolved (30d)",
        _metric_value(ticket_count_30),
    )
    _tile(c3, TAB_ENGINEERING, shipped, "PRs merged (7d)", _metric_value(pr_count_7))
    _tile(c4, TAB_ENGINEERING, shipped, "PRs merged (30d)", _metric_value(pr_count_30))

    # None means the ticket fetch failed (distinct from an empty 30-day window),
    # so the pie can say "could not load" instead of asserting nobody resolved any.
    tickets_unavailable = resolved_30 is None
    resolved_df = pd.DataFrame() if resolved_30 is None else resolved_30

    left, right = st.columns(2)
    with left:
        ticket_people = (
            resolved_df["assignee"].fillna("Unassigned").astype(str).str.strip().replace("", "Unassigned")
            if "assignee" in resolved_df.columns and not resolved_df.empty
            else pd.Series(dtype=str)
        )
        _contribution_pie(
            ticket_people, "tickets", "Who resolved tickets (30 days)",
            unavailable=tickets_unavailable,
        )
        if (
            not tickets_unavailable
            and ticket_count_30 is not None
            and len(ticket_people) < int(ticket_count_30)
        ):
            st.caption(
                f"Pie shows a {len(ticket_people)}-ticket sample of ~{int(ticket_count_30)} "
                "resolved (fetch limit); ticket tiles are Jira's approximate counts."
            )
    with right:
        if not github_ready:
            st.caption(
                "PR charts need a GitHub token. "
                + (f"({github_error})" if github_error else "Set DASHBOARD_GITHUB_TOKEN.")
            )
        else:
            pr_people = (
                merged_prs["author"].fillna("unknown").astype(str)
                if "author" in merged_prs.columns and not merged_prs.empty
                else pd.Series(dtype=str)
            )
            _contribution_pie(pr_people, "PRs", "Who merged PRs (30 days)")

    st.caption(
        "Ticket resolved = transitioned into Done / Released / Ready for Production / Review in "
        "Staging in the window (credited to current assignee). PR merged = merged anywhere in the "
        "org in the window (credited to the PR author, by GitHub username)."
    )


# --- Pull requests -----------------------------------------------------------

# A stuck PR is open with no approving review. We classify from the actual
# review counts, not reviewDecision: GitHub only populates reviewDecision when
# the base branch *requires* review (branch protection / CODEOWNERS), so in
# repos without that rule it stays null even when approving reviews exist -
# which would otherwise mark reviewed/approved PRs as stuck.
def _pr_review_label(row: pd.Series) -> str:
    if int(row.get("approving_reviews", 0) or 0) > 0:
        return "Approved"
    if int(row.get("changes_reviews", 0) or 0) > 0:
        return "Changes requested"
    if int(row.get("total_reviews", 0) or 0) > 0:
        return "Reviewed, not approved"
    return "No review yet"


@st.fragment
def _render_pr_section(
    open_prs: pd.DataFrame,
    github_ready: bool,
    github_error: str = "",
    open_count_exact: int | None = None,
) -> None:
    """Org-wide PR health: per-person open/stuck counts and the stuck PR queue."""
    st.subheader("Pull Requests")
    if not github_ready:
        st.info(
            "Connect GitHub to see PR status. Set a read-only DASHBOARD_GITHUB_TOKEN "
            "on the deployment."
            + (f" ({github_error})" if github_error else "")
        )
        return
    if open_prs.empty:
        st.success("No open PRs across the organization.")
        return

    prs = open_prs.copy()
    prs["review"] = prs.apply(_pr_review_label, axis=1)
    # Stuck = open, non-draft, with no approving review (the user's definition).
    prs["stuck"] = prs["approving_reviews"].fillna(0).astype(int) == 0

    fetched = int(len(prs))
    # Exact org-wide open count isn't paging-capped; the frame is (max_prs), so
    # fall back to the fetched size only if the exact count is unavailable.
    open_count = fetched if open_count_exact is None else int(open_count_exact)
    stuck = prs[prs["stuck"]]
    no_review = prs[prs["total_reviews"].fillna(0).astype(int) == 0]
    c1, c2, c3 = st.columns(3)
    review = "Pull requests"
    _tile(c1, TAB_ENGINEERING, review, "Open PRs", str(open_count))
    _tile(
        c2,
        TAB_ENGINEERING,
        review,
        "Stuck (no approving review)",
        f"{len(stuck)}",
    )
    _tile(c3, TAB_ENGINEERING, review, "Never reviewed", f"{len(no_review)}")
    if open_count_exact is not None and fetched < open_count_exact:
        st.caption(
            f"Per-person and stuck lists cover the {fetched} oldest of {open_count_exact} "
            "open PRs (fetch limit); the Open PRs tile is exact."
        )

    # Per-person PR status: who is holding open and stuck work, and their oldest.
    by_person = (
        prs.groupby("author")
        .agg(
            open_prs=("number", "size"),
            stuck_prs=("stuck", "sum"),
            oldest_days=("age_days", "max"),
            idle_days=("idle_days", "max"),
        )
        .reset_index()
        .sort_values(["stuck_prs", "oldest_days"], ascending=[False, False])
    )
    by_person["stuck_prs"] = by_person["stuck_prs"].astype(int)
    st.markdown("**PR status by person** (GitHub username)")
    st.dataframe(
        by_person,
        width="stretch",
        hide_index=True,
        column_config={
            "author": st.column_config.TextColumn("Person"),
            "open_prs": st.column_config.NumberColumn("Open"),
            "stuck_prs": st.column_config.NumberColumn("Stuck"),
            "oldest_days": st.column_config.NumberColumn("Oldest (days)", format="%.0f"),
            "idle_days": st.column_config.NumberColumn("Most idle (days)", format="%.0f"),
        },
    )

    # The stuck queue itself, oldest first, so nobody has to hunt for the PRs
    # that have been sitting unreviewed.
    st.markdown("**Stuck PRs — open with no approving review, oldest first**")
    stuck_display = stuck.sort_values("age_days", ascending=False)[
        ["url", "title", "author", "review", "age_days", "idle_days"]
    ]
    st.dataframe(
        stuck_display,
        width="stretch",
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("PR", display_text=r"/pull/(\d+)"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "author": st.column_config.TextColumn("Author"),
            "review": st.column_config.TextColumn("Review"),
            "age_days": st.column_config.NumberColumn("Age (days)", format="%.0f"),
            "idle_days": st.column_config.NumberColumn("Idle (days)", format="%.0f"),
        },
    )
    st.caption(
        "Stuck = open PR without an approving review (no review yet, review pending, or changes "
        "requested). Drafts are excluded. Counts are org-wide."
    )

    # Waiting on a reviewer: open PRs with nobody assigned to review AND no review
    # yet — these fall through the cracks because no one is on the hook for them.
    # "review_requests" may be absent if the client predates the field; treat
    # missing as 0 so an older cache degrades to "no reviewer" rather than crashing.
    requests_series = (
        prs["review_requests"] if "review_requests" in prs.columns else 0
    )
    no_reviewer = prs[
        (pd.Series(requests_series, index=prs.index).fillna(0).astype(int) == 0)
        & (prs["total_reviews"].fillna(0).astype(int) == 0)
    ]
    st.markdown("**Waiting on a reviewer — nobody assigned, no review yet**")
    min_age = st.slider(
        "Only show PRs older than (days)",
        min_value=0,
        max_value=30,
        value=2,
        key="no_reviewer_min_age",
    )
    waiting = no_reviewer[no_reviewer["age_days"] > float(min_age)].sort_values(
        "age_days", ascending=False
    )
    if waiting.empty:
        st.success(
            f"No unassigned, unreviewed PRs older than {min_age} day(s)."
        )
    else:
        st.dataframe(
            waiting[["url", "title", "author", "age_days", "idle_days"]],
            width="stretch",
            hide_index=True,
            column_config={
                "url": st.column_config.LinkColumn("PR", display_text=r"/pull/(\d+)"),
                "title": st.column_config.TextColumn("Title", width="large"),
                "author": st.column_config.TextColumn("Author"),
                "age_days": st.column_config.NumberColumn("Age (days)", format="%.0f"),
                "idle_days": st.column_config.NumberColumn("Idle (days)", format="%.0f"),
            },
        )
        st.caption(
            f"{len(waiting)} open PR(s) older than {min_age} day(s) with no reviewer "
            "requested and no review yet — assign someone so they don't stall."
        )


def _known_project_keys(df: pd.DataFrame) -> list[str]:
    """Project keys a PR may legitimately reference.

    Every project Jira exposes, not just the ones with a ticket on screen: a
    project whose work is all Done still has PRs pointing at it. The tickets are
    a fallback for when the project list cannot be read, and
    JIRA_EXTRA_PROJECT_KEYS covers projects this account cannot see at all.
    Matching only known keys stops strings like "UTF-8" reading as tickets.
    """
    keys = set()
    try:
        keys.update(fetch_project_keys(CREDS_PATH, PROFILE_NAME))
    except Exception:
        # A missing project list only costs precision, so fall back rather than
        # taking the section down.
        pass
    if "project_key" in df.columns:
        keys.update(str(k).strip().upper() for k in df["project_key"].dropna())
    keys.update(
        part.strip().upper()
        for part in os.getenv("JIRA_EXTRA_PROJECT_KEYS", "").split(",")
        if part.strip()
    )
    return sorted(k for k in keys if k)


@st.fragment
def _render_pr_hygiene(
    open_prs: pd.DataFrame,
    github_ready: bool,
    github_error: str = "",
    project_keys: list[str] | None = None,
    tickets: pd.DataFrame | None = None,
) -> None:
    """Open PRs that are untraceable, stalled, or nobody's job to review."""
    st.subheader("PR Hygiene")
    if not github_ready:
        st.info(
            "Connect GitHub to see PR hygiene. Set a read-only DASHBOARD_GITHUB_TOKEN "
            "on the deployment."
            + (f" ({github_error})" if github_error else "")
        )
        return
    if open_prs.empty:
        st.success("No open PRs across the organization.")
        return

    prs = pr_hygiene.add_hygiene_fields(open_prs, project_keys)
    no_key = prs[~prs["has_jira_key"]]
    stale = prs[prs["is_stale"]]
    unowned = prs[prs["is_unowned"]]
    critical = pr_hygiene.critical_in_flight(
        prs, tickets if tickets is not None else pd.DataFrame()
    )

    c1, c2, c3, c4 = st.columns(4)
    hygiene_section = "PR hygiene"
    _tile(c1, TAB_ENGINEERING, hygiene_section, "Critical in flight", f"{len(critical)}")
    _tile(c2, TAB_ENGINEERING, hygiene_section, "No Jira key", f"{len(no_key)}")
    _tile(
        c3,
        TAB_ENGINEERING,
        hygiene_section,
        f"Stale (>{pr_hygiene.STALE_AGE_DAYS:.0f}d old or "
        f">{pr_hygiene.STALE_IDLE_DAYS:.0f}d idle)",
        f"{len(stale)}",
    )
    _tile(c4, TAB_ENGINEERING, hygiene_section, "No reviewer", f"{len(unowned)}")
    st.caption(
        "A key is looked for in the PR title, branch name and description"
        + (f", matched against {len(project_keys)} known project keys." if project_keys else ".")
    )

    columns = ["url", "title", "author", "repo", "age_days", "idle_days"]
    config = {
        "url": st.column_config.LinkColumn("PR", display_text=r"/pull/(\d+)"),
        "title": st.column_config.TextColumn("Title", width="large"),
        "author": st.column_config.TextColumn("Author"),
        "repo": st.column_config.TextColumn("Repo"),
        "age_days": st.column_config.NumberColumn("Age (days)", format="%.0f"),
        "idle_days": st.column_config.NumberColumn("Idle (days)", format="%.0f"),
        "stale_reason": st.column_config.TextColumn("Why"),
        "jira_key": st.column_config.LinkColumn(
            "Jira", display_text=JIRA_KEY_DISPLAY_PATTERN
        ),
        "priority": st.column_config.TextColumn("Priority"),
        "ticket_status": st.column_config.TextColumn("Ticket status"),
    }

    critical_tab, key_tab, stale_tab, owner_tab, person_tab = st.tabs(
        ["Critical in flight", "No Jira key", "Stale", "No reviewer", "By person"]
    )
    with critical_tab:
        if critical.empty:
            st.success(
                "No open PR is carrying high-priority work in dev, review or staging."
            )
        else:
            st.dataframe(
                critical.assign(
                    jira_key=critical["jira_key"].map(_jira_ticket_url)
                ).sort_values(["idle_days", "age_days"], ascending=False)[
                    ["url", "title", "jira_key", "priority", "ticket_status",
                     "author", "repo", "age_days", "idle_days"]
                ],
                width="stretch",
                hide_index=True,
                column_config=config,
            )
        st.caption(
            "High-priority tickets whose PR is still open in dev, code review or "
            "staging: the work nearest to shipping, longest-idle first. A PR with "
            "no Jira key has no priority to read and is listed in the next tab."
        )
    with key_tab:
        if no_key.empty:
            st.success("Every open PR references a Jira ticket.")
        else:
            st.dataframe(
                no_key.sort_values("age_days", ascending=False)[columns],
                width="stretch",
                hide_index=True,
                column_config=config,
            )
            st.caption("Nothing links these back to a ticket, so the work is invisible in Jira.")
    with stale_tab:
        if stale.empty:
            st.success("No stale open PRs.")
        else:
            st.dataframe(
                stale.sort_values("age_days", ascending=False)[columns + ["stale_reason"]],
                width="stretch",
                hide_index=True,
                column_config=config,
            )
    with owner_tab:
        if unowned.empty:
            st.success("Every open PR has a reviewer or a review.")
        else:
            st.dataframe(
                unowned.sort_values("age_days", ascending=False)[columns],
                width="stretch",
                hide_index=True,
                column_config=config,
            )
            st.caption("Nobody was asked to review and nobody has - these stall silently.")
    with person_tab:
        st.dataframe(
            pr_hygiene.hygiene_by_person(prs),
            width="stretch",
            hide_index=True,
            column_config={
                "author": st.column_config.TextColumn("Person"),
                "open_prs": st.column_config.NumberColumn("Open"),
                "no_jira_key": st.column_config.NumberColumn("No Jira key"),
                "stale": st.column_config.NumberColumn("Stale"),
                "unowned": st.column_config.NumberColumn("No reviewer"),
                "problems": st.column_config.NumberColumn("Total flags"),
                "oldest_days": st.column_config.NumberColumn("Oldest (days)", format="%.0f"),
            },
        )

    flagged = prs[~prs["has_jira_key"] | prs["is_stale"] | prs["is_unowned"]]
    st.download_button(
        "Download flagged PRs (CSV)",
        flagged[
            ["repo", "number", "title", "author", "url", "jira_key", "age_days",
             "idle_days", "is_stale", "stale_reason", "is_unowned"]
        ].to_csv(index=False),
        file_name="pr_hygiene.csv",
        mime="text/csv",
        disabled=flagged.empty,
    )


@st.fragment
def _render_ticket_quality(df: pd.DataFrame) -> None:
    """How well tickets are written, and which are clear enough to hand off."""
    st.subheader("Ticket Quality & Ready for Devin")
    if df.empty:
        st.info("No tickets in the current scope.")
        return

    scored = ticket_quality.score_tickets(df)
    gradable = scored[scored["quality_score"].notna()]
    if gradable.empty:
        st.info("No gradable tickets in the current scope (epics and initiatives are exempt).")
        return

    ready = gradable[gradable["devinable"] == "Yes"]
    maybe = gradable[gradable["devinable"] == "Maybe"]
    unclear = gradable[gradable["quality_score"] <= 2]

    c1, c2, c3, c4 = st.columns(4)
    clarity = "Ticket clarity"
    _tile(c1, TAB_ENGINEERING, clarity, "Ready for Devin", f"{len(ready)}")
    _tile(c2, TAB_ENGINEERING, clarity, "Nearly ready", f"{len(maybe)}")
    _tile(c3, TAB_ENGINEERING, clarity, "Unclear (score \u22642)", f"{len(unclear)}")
    _tile(
        c4,
        TAB_ENGINEERING,
        clarity,
        "Average score",
        f"{gradable['quality_score'].mean():.1f} / 5",
    )
    st.caption(
        "Scored out of 5: a summary that says what the work is, a real description, "
        "acceptance criteria, an estimate, and an epic. Epics and initiatives are exempt. "
        "Ready for Devin needs the goal and the finish line written down, and work that "
        "does not hinge on a conversation. Backlog tickets are always included here, "
        "regardless of *Include Backlogs*."
    )

    gradable = gradable.assign(key_url=gradable["key"].map(_jira_ticket_url))
    ready = ready.assign(key_url=ready["key"].map(_jira_ticket_url))
    maybe = maybe.assign(key_url=maybe["key"].map(_jira_ticket_url))
    unclear = unclear.assign(key_url=unclear["key"].map(_jira_ticket_url))

    columns = ["key_url", "summary", "status", "assignee", "reporter", "quality_score", "missing"]
    config = {
        "key_url": st.column_config.LinkColumn("Key", display_text=JIRA_KEY_DISPLAY_PATTERN),
        "summary": st.column_config.TextColumn("Summary", width="large"),
        "status": st.column_config.TextColumn("Status"),
        "assignee": st.column_config.TextColumn("Assignee"),
        "reporter": st.column_config.TextColumn("Reporter"),
        "quality_score": st.column_config.NumberColumn("Score", format="%d"),
        "missing": st.column_config.TextColumn("Missing", width="medium"),
        "devinable": st.column_config.TextColumn("Devin-able?"),
    }

    ready_tab, maybe_tab, unclear_tab, person_tab, all_tab = st.tabs(
        ["Ready for Devin", "Nearly ready", "Unclear", "By reporter", "All scored"]
    )
    with ready_tab:
        if ready.empty:
            st.info("No ticket currently states both its goal and its acceptance criteria.")
        else:
            st.dataframe(
                ready.sort_values("quality_score", ascending=False)[columns],
                width="stretch",
                hide_index=True,
                column_config=config,
            )
            st.caption("Hand these off rather than spending an engineer on them.")
    with maybe_tab:
        if maybe.empty:
            st.success("Nothing sitting just short of ready.")
        else:
            st.dataframe(
                maybe.sort_values("quality_score", ascending=False)[columns],
                width="stretch",
                hide_index=True,
                column_config=config,
            )
            st.caption(
                "One gap away, usually the acceptance criteria - the cheapest tickets to fix."
            )
    with unclear_tab:
        if unclear.empty:
            st.success("No ticket scores 2 or below.")
        else:
            st.dataframe(
                unclear.sort_values("quality_score")[columns],
                width="stretch",
                hide_index=True,
                column_config=config,
            )
            st.caption("Nobody outside the original conversation can pick these up.")
    with person_tab:
        st.dataframe(
            ticket_quality.quality_by_person(scored),
            width="stretch",
            hide_index=True,
            column_config={
                "Reporter": st.column_config.TextColumn("Reporter"),
                "tickets": st.column_config.NumberColumn("Tickets"),
                "avg_score": st.column_config.NumberColumn("Avg score", format="%.1f"),
                "ready_for_devin": st.column_config.NumberColumn("Ready for Devin"),
                "no_acceptance_criteria": st.column_config.NumberColumn("No criteria"),
                "no_description": st.column_config.NumberColumn("No description"),
            },
        )
        st.caption(
            "The tickets currently in scope, grouped by the person who wrote them - "
            "they are the one who can say what done means."
        )
    with all_tab:
        st.dataframe(
            gradable.sort_values("quality_score")[columns + ["devinable"]],
            width="stretch",
            hide_index=True,
            column_config=config,
        )

    st.download_button(
        "Download ticket scores (CSV)",
        gradable[["key"] + columns[1:] + ["devinable"]].to_csv(index=False),
        file_name="ticket_quality.csv",
        mime="text/csv",
    )


def _as_frame(value: Any) -> pd.DataFrame:
    """A frame from a read that may have failed; a failure reads as no rows."""
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _render_engineering_page() -> None:
    st.caption("Visual monitoring for stale, idle, and high-risk tickets.")
    # Reserved before the sections run: the download button can only be built
    # once they have, but the slot has to sit at the top where a reader looks
    # for it.
    engineering_slot = st.columns([5, 1])[1]

    max_results = MAX_RESULTS
    page_size = JIRA_PAGE_SIZE

    # GitHub PR data is optional: without a token the PR views degrade to a hint
    # rather than an error, so the Jira dashboard still works standalone.
    github_error = ""
    try:
        github_env = github_client.load_github_env()
    except Exception as exc:  # noqa: BLE001
        github_env = None
        github_error = str(exc)[:200]

    # Every opening read goes out at once. They share nothing but the page they
    # land on, so the wait is now the slowest of them rather than their sum.
    reads = _engineering_reads(max_results, page_size)
    if github_env is not None:
        reads.update(_github_reads(*github_env))

    # A bar rather than a spinner: a cold load is long enough that a reader
    # deserves to see it moving and roughly how far along it is. The slot stays
    # empty on a warm page, where the reads answer before the bar is due.
    loading_slot = st.empty()
    loading_bar = None

    def _show_progress(fraction: float, label: str) -> None:
        nonlocal loading_bar
        if loading_bar is None:
            loading_bar = loading_slot.progress(fraction, text=label)
        else:
            loading_bar.progress(fraction, text=label)

    data, errors = _gather(reads, on_progress=_show_progress)
    loading_slot.empty()

    if "tickets" in errors:
        failure = errors["tickets"]
        if isinstance(failure, JiraConfigError):
            st.error(f"Configuration error: {failure}")
        else:
            st.error(f"Failed to fetch Jira issues: {failure}")
        st.stop()

    raw_df = _as_frame(data["tickets"])
    if raw_df.empty:
        st.warning("No tickets returned for the current JQL.")
        st.stop()

    if len(raw_df) >= max_results:
        st.warning(
            f"Showing the first {max_results} tickets of a larger result set - the JQL "
            "orders by least recently updated, so newer tickets are missing. Narrow "
            "JIRA_DASHBOARD_JQL or raise JIRA_MAX_RESULTS."
        )

    df = add_priority_score(add_ticket_health_fields(raw_df))

    # A failed GitHub read leaves the PR sections saying so, rather than
    # reporting an empty org as though nobody had opened a pull request.
    github_ready = github_env is not None and not (errors.keys() & set(_GITHUB_READS))
    if github_env is not None and not github_ready:
        first_failure = next(name for name in _GITHUB_READS if name in errors)
        github_error = str(errors[first_failure])[:200]
    open_prs = _as_frame(data.get("open_prs")) if github_ready else pd.DataFrame()
    merged_prs = _as_frame(data.get("merged_prs")) if github_ready else pd.DataFrame()
    pr_count_7 = data.get("merged_count_7") if github_ready else None
    pr_count_30 = data.get("merged_count_30") if github_ready else None
    open_count_exact = data.get("open_pr_count") if github_ready else None

    # None (not an empty frame) marks a read that failed, so a section can say
    # "could not load" instead of an authoritative "there is nothing here".
    _render_resolved_summary(
        data.get("resolved_count_7"),
        data.get("resolved_count_30"),
        data.get("resolved_30"),
        pr_count_7,
        pr_count_30,
        merged_prs,
        github_ready,
        github_error,
    )
    st.divider()

    _render_new_and_triage(
        data.get("created_count_1"),
        data.get("created_count_7"),
        data.get("triage_stuck_count"),
        data.get("created_7"),
        data.get("triage_stuck"),
        TRIAGE_STUCK_HOURS,
    )
    st.divider()

    # "Unassigned" is Jira's placeholder, not a colleague: offering it here would
    # inflate the head-count and let someone "focus" on a person who does not
    # exist. Ownerless work is reached through the cleanup queue and the
    # unassigned KPI instead.
    assignees = sorted(
        name
        for name in df["assignee"].dropna().unique().tolist()
        if str(name).strip().lower() not in _NO_OWNER_NAMES
    )
    statuses = sorted(df["status"].dropna().unique().tolist())
    priorities = sorted(df["priority"].dropna().unique().tolist())

    with st.sidebar:
        st.header("Scope")
        scope = st.radio(
            "View",
            options=[SCOPE_ORG, SCOPE_TEAM, SCOPE_INDIVIDUAL],
            help=(
                "Organization shows every assignee in the JQL scope; "
                "Team pre-selects the configured team members; "
                "Individual focuses on a single assignee."
            ),
        )
        selected_assignees = _resolve_scope_assignees(scope, assignees)
        st.session_state[_SCOPE_ASSIGNEES_KEY] = (
            None if selected_assignees is None else set(selected_assignees)
        )

        # Batched behind a submit button: each of these six used to rerun the
        # whole page on its own, so narrowing a view by status, priority and two
        # sliders cost four full rebuilds to express one thought. Scope stays
        # outside because it decides which widget appears beneath it, which a
        # form cannot do without a second submit.
        st.header("Filters")
        with st.form("engineering_filters", border=False):
            selected_statuses = st.multiselect("Status", options=statuses, default=[])
            selected_priorities = st.multiselect("Priority", options=priorities, default=[])
            min_idle = st.slider("Min idle days", min_value=0, max_value=180, value=0)
            min_age = st.slider("Min ticket age", min_value=0, max_value=365, value=0)
            include_backlogs = st.checkbox("Include Backlogs", value=False)
            color_by = st.segmented_control(
                "Bubble color",
                options=["priority", "assignee"],
                default="priority",
            )
            st.form_submit_button("Apply filters", width="stretch")
        # A cleared segmented control returns None; the chart needs a column.
        color_by = color_by or "priority"

        st.header("Jira writes")
        # Reading the dashboard is the common case; changing Jira is a decision.
        # Off on every page load so a reporting session cannot edit by accident,
        # and re-armed deliberately when the reviewer means it.
        allow_writes = st.toggle(
            "Allow Jira edits",
            value=False,
            help=(
                "Off: the dashboard only reads Jira. On: closures, transitions, "
                "assignee and sprint edits can be applied."
            ),
        )
        write_access.set_writes_enabled(allow_writes)
        if allow_writes:
            st.warning("Edits armed - Apply buttons will change Jira.")
        else:
            st.caption("Read-only. Nothing here can change Jira.")

    filtered = df.copy()
    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]
    if selected_priorities:
        filtered = filtered[filtered["priority"].isin(selected_priorities)]

    filtered = filtered[(filtered["idle_days"] >= min_idle) & (filtered["ticket_age_days"] >= min_age)]

    # Ownerless work belongs to nobody, so no assignee scope can contain it; the
    # cleanup section keeps this pre-scope frame to feed its unassigned queue.
    unscoped = filtered
    if selected_assignees is not None:
        filtered = filtered[filtered["assignee"].isin(selected_assignees)]

    _render_metrics(
        filtered,
        include_backlogs=include_backlogs,
        unassigned_source=unscoped if selected_assignees is not None else None,
    )

    # One backlog-filtered view, shared: four sections asked for the same frame
    # with the same arguments and each rebuilt it.
    metrics_view = _metrics_df(filtered, include_backlogs)

    st.divider()
    _render_mix(metrics_view)

    st.divider()
    _render_team_overview(metrics_view)

    st.divider()
    _render_epics(metrics_view, organization_source=df)

    st.divider()
    # Backlog-inclusive on purpose: the backlog is what this section clears out.
    _render_cleanup(filtered, unassigned_source=unscoped)

    st.divider()
    _render_scope_breakdown(filtered, scope=scope, include_backlogs=include_backlogs)

    st.divider()
    _render_pr_section(open_prs, github_ready, github_error, open_count_exact)

    st.divider()
    # Every ticket, not the scoped slice: a PR belongs to the org whichever team
    # or person the dashboard is currently looking at.
    _render_pr_hygiene(
        open_prs, github_ready, github_error, _known_project_keys(df), tickets=df
    )

    st.divider()
    # Backlog-inclusive on purpose: a backlog ticket is the best kind to hand off,
    # and it is where badly written tickets accumulate unseen.
    _render_ticket_quality(filtered)

    st.divider()
    _render_priority_queue(filtered, include_backlogs=include_backlogs)

    st.divider()
    _render_estimate_policy(filtered)

    st.divider()
    _render_stale_cleanup(filtered)

    restore_requested = bool(st.session_state.pop("restore_sprint_ticket_table", False))
    bubble_chart_version = int(st.session_state.get("bubble_chart_version", 0))
    if restore_requested:
        bubble_chart_version += 1
        st.session_state["bubble_chart_version"] = bubble_chart_version

    agg_priority = st.checkbox(
        "Aggregate Priorities (Normal / High / Urgent)",
        value=False,
        help="Buckets: Normal = None/Low/Normal · High = High · Urgent = Highest/Urgent",
    )
    selected_key = _render_bubble_chart(
        filtered,
        color_by=color_by,
        agg_priority=agg_priority,
        chart_key=f"bubble_chart_{bubble_chart_version}",
    )

    if restore_requested:
        active_sprint_ticket_key = None
    else:
        active_sprint_ticket_key = selected_key if selected_key and selected_key in filtered["key"].values else None

    st.divider()
    st.subheader("Sprint Planner")
    _render_sprint_plan(df)

    st.divider()
    st.subheader("Sprint Capacity")
    _render_sprint_capacity(
        filtered,
        status_source_df=filtered,
        selected_ticket_key=active_sprint_ticket_key,
    )

    st.divider()
    st.subheader("Suggested First Action")

    PRIORITY_OPTIONS = ["Highest", "High", "Normal", "Low", "Lowest"]
    action_type = st.selectbox(
        "Action",
        options=["Set None-priority tickets", "Change status"],
        index=0,
        help="Default action keeps the first cleanup flow: None priority -> Normal.",
    )

    # Bulk writes must not reach tickets the user has hidden with Include Backlogs.
    action_df = metrics_view
    status_options = sorted(action_df["status"].dropna().astype(str).unique().tolist())
    normalized_priority = action_df["priority"].fillna("").astype(str).str.strip().str.lower()
    none_priority_keys = sorted(action_df[normalized_priority.isin(["", "none"])]["key"].tolist())

    with st.container(border=True):
        if action_type == "Set None-priority tickets":
            st.markdown("**Detected tickets without priority**")
            st.caption(
                f"{len(none_priority_keys)} ticket(s) in the current view have no priority set."
            )
            if none_priority_keys:
                preview = ", ".join(none_priority_keys[:15])
                suffix = " ..." if len(none_priority_keys) > 15 else ""
                st.caption(f"Sample: {preview}{suffix}")

            default_keys = none_priority_keys[:BULK_ACTION_DEFAULT_LIMIT]
            if len(none_priority_keys) > len(default_keys):
                st.caption(
                    f"Only the first {BULK_ACTION_DEFAULT_LIMIT} are pre-selected; "
                    "add more explicitly if you mean to update them."
                )
            selected_keys = st.multiselect(
                "Tickets to update",
                options=none_priority_keys,
                default=default_keys,
                help="Remove any tickets you do not want to update.",
            )

            target_priority = st.selectbox(
                "Suggested priority",
                options=PRIORITY_OPTIONS,
                index=2,
                help="Normal is selected by default as the first cleanup action.",
            )
            target_label = f"priority '{target_priority}'"
        else:
            st.markdown("**Change ticket status**")
            if not status_options:
                st.info("No statuses available in the current filtered view.")
                source_status = None
                target_status = None
                selected_keys = []
            else:
                source_status = st.selectbox("From status", options=status_options, index=0)
                to_options = [s for s in status_options if s != source_status] or status_options
                target_status = st.selectbox("To status", options=to_options, index=0)

                source_keys = sorted(action_df[action_df["status"] == source_status]["key"].tolist())
                default_source_keys = source_keys[:BULK_ACTION_DEFAULT_LIMIT]
                if len(source_keys) > len(default_source_keys):
                    st.caption(
                        f"Only the first {BULK_ACTION_DEFAULT_LIMIT} are pre-selected; "
                        "add more explicitly if you mean to update them."
                    )
                selected_keys = st.multiselect(
                    "Tickets to update",
                    options=source_keys,
                    default=default_source_keys,
                    help="Only tickets currently in the selected source status are listed.",
                )
                target_label = f"status '{source_status}' -> '{target_status}'"

        apply_suggestion = st.button(
            f"Apply to {len(selected_keys)} ticket(s)",
            disabled=(not selected_keys) or (not write_access.writes_enabled()),
            type="primary",
        )

    if apply_suggestion and selected_keys:
        client = JiraClient.resolve(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
        )
        with st.spinner(f"Updating {len(selected_keys)} tickets..."):
            if action_type == "Set None-priority tickets":
                succeeded, failed, operation = _apply_action_with_audit(
                    client=client,
                    action_type="priority",
                    selected_keys=selected_keys,
                    target=target_priority,
                )
            else:
                succeeded, failed, operation = _apply_action_with_audit(
                    client=client,
                    action_type="status",
                    selected_keys=selected_keys,
                    target=target_status,
                    source_status=source_status,
                )

        if succeeded:
            st.success(
                f"Updated {len(succeeded)} ticket(s) to {target_label}. Operation ID: {operation['operation_id']}"
            )
        if failed:
            for key, err in failed.items():
                st.error(f"{key}: {err}")

        _clear_page_caches(ENGINEERING_PAGE_TITLE)
        st.rerun()

    st.divider()
    st.subheader("Change History and Revert")
    operations = load_operations(limit=30)
    if not operations:
        st.info("No write operations have been logged yet.")
    else:
        st.dataframe(pd.DataFrame(summarize_operations(operations)), width="stretch")

        op_options = {
            (
                f"{op.get('created_at', '')} | {op.get('action_type', '')} | "
                f"{op.get('target', '')} | success={op.get('success_count', 0)} | "
                f"id={str(op.get('operation_id', ''))[:8]}"
            ): op
            for op in operations
            if op.get("success_count", 0) > 0
        }

        if not op_options:
            st.caption("No successful operation available for revert.")
        else:
            selected_label = st.selectbox(
                "Select operation to revert",
                options=list(op_options.keys()),
            )
            selected_operation = op_options[selected_label]
            confirm_revert = st.checkbox("I understand revert may partially fail due to Jira workflow rules.")

            revert_clicked = st.button(
                "Revert selected operation",
                disabled=(not confirm_revert) or (not write_access.writes_enabled()),
            )

            if revert_clicked:
                client = JiraClient.resolve(
                    creds_path=CREDS_PATH,
                    profile_name=PROFILE_NAME,
                )

                revert_succeeded: list[str] = []
                revert_failed: dict[str, str] = {}
                parent_id = selected_operation.get("operation_id")
                successful_items = [it for it in selected_operation.get("items", []) if it.get("success")]

                with st.spinner(f"Reverting {len(successful_items)} ticket(s)..."):
                    for item in successful_items:
                        key = str(item.get("key", ""))
                        before = item.get("before") or {}
                        try:
                            if selected_operation.get("action_type") == "priority":
                                original_priority_id = before.get("priority_id")
                                if not original_priority_id:
                                    raise RuntimeError("Original priority id missing in audit record.")

                                rev_succeeded, rev_failed, rev_op = _apply_action_with_audit(
                                    client=client,
                                    action_type="revert_priority",
                                    selected_keys=[key],
                                    target=str(original_priority_id),
                                    parent_operation_id=str(parent_id),
                                )
                            elif selected_operation.get("action_type") == "status":
                                original_status = before.get("status")
                                if not original_status:
                                    raise RuntimeError("Original status missing in audit record.")

                                rev_succeeded, rev_failed, rev_op = _apply_action_with_audit(
                                    client=client,
                                    action_type="revert_status",
                                    selected_keys=[key],
                                    target=str(original_status),
                                    parent_operation_id=str(parent_id),
                                )
                            else:
                                raise RuntimeError("Selected operation type is not revertible by this tool.")

                            revert_succeeded.extend(rev_succeeded)
                            revert_failed.update(rev_failed)
                        except Exception as exc:  # noqa: BLE001
                            revert_failed[key] = str(exc)

                if revert_succeeded:
                    st.success(f"Reverted {len(revert_succeeded)} ticket(s).")
                if revert_failed:
                    for key, err in revert_failed.items():
                        st.error(f"Revert failed for {key}: {err}")

                _clear_page_caches(ENGINEERING_PAGE_TITLE)
                st.rerun()

    st.caption(
        "Team member filter uses Jira assignee display names from fetched data. "
        "For stricter JQL filtering, use assignee account IDs in JQL."
    )
    _download_report(engineering_slot, TAB_ENGINEERING)


def _clear_page_caches(page_title: str) -> None:
    """Drop only the reads the page in front of the reader depends on.

    ``st.cache_data.clear()`` emptied every cache in the process, so refreshing
    a ticket count also threw away the year of orders, the funnel, the project
    list and the user directory - and the next visitor to those pages paid for
    it. Clearing per page keeps the button honest about what it refreshes.
    """
    engineering = (
        fetch_tickets,
        fetch_resolved_count,
        fetch_resolved_tickets,
        fetch_created_count,
        fetch_created_tickets,
        fetch_triage_stuck_count,
        fetch_triage_stuck_tickets,
        fetch_open_prs_cached,
        fetch_open_pr_count_cached,
        fetch_merged_prs_cached,
        fetch_merged_pr_count_cached,
        fetch_available_transition_statuses,
        # The sprint editor's dropdowns are drawn from these. They are held for
        # longer than the counts because they move rarely, but "rarely" is not
        # "never": somebody who has just been added to Jira and presses Refresh
        # to find themselves should find themselves.
        fetch_project_keys,
        fetch_all_priorities,
        fetch_all_users,
    )
    business = (
        _order_book,
        fetch_store_prefixes_cached,
        _funnel_cached,
        _breakdown_cached,
        _event_users_cached,
        # Both ads entries, or Refresh would drop the spend and go straight back
        # to a day-old list of accounts to read it for.
        _ads_cached,
        _ads_accounts,
        # Burn holds the three bills, and a quarter of an hour is long enough
        # that a reader who presses Refresh expects them to move too.
        _openai_costs_cached,
        _cloud_costs_cached,
        _stripe_ledger_cached,
        _stripe_disputes_cached,
        # Benchmarks move once a day, but a reader who has just changed a price
        # and pressed Refresh is asking about that price.
        _price_benchmark_cached,
        # And whose listing each of those prices is: a merchant renamed or a
        # product re-slugged would otherwise keep its old name for a day.
        _offer_merchants_cached,
        # And what those prices sold: the evidence beside them is only worth
        # refreshing if the bottles move with it.
        _offer_sales_cached,
    )
    for cached in business if page_title == BUSINESS_PAGE_TITLE else engineering:
        cached.clear()
    logger.info("Cleared cached reads for the %s page", page_title)


def main() -> None:
    st.set_page_config(page_title="Jira Ticket Health Dashboard", layout="wide")
    require_password()
    inject_styles()

    # Pages rather than tabs. Streamlit runs the body of every tab on every
    # rerun, so the engineering sections were being rebuilt for readers looking
    # at the shop's figures and vice versa; a page that is not open does not run
    # at all, which is why the Business page no longer needs a button in front
    # of it and why opening it no longer waits for Jira.
    pages = [
        st.Page(
            _render_engineering_page,
            title=ENGINEERING_PAGE_TITLE,
            icon=":material/engineering:",
            url_path="engineering",
            default=True,
        )
    ]
    if _business_readable():
        pages.append(
            st.Page(
                _render_business,
                title=BUSINESS_PAGE_TITLE,
                icon=":material/storefront:",
                url_path="business",
            )
        )
    page = st.navigation(pages, position="top")

    # The login is remembered in the browser for a month, so there has to be a
    # way to hand a shared laptop back without handing over Jira write access.
    render_sign_out()
    st.title("Jira Ticket Health Dashboard")
    if st.button("Refresh data", icon=":material/refresh:"):
        _clear_page_caches(page.title)

    # Gathered fresh on every run, because every figure the active page draws
    # is. Only one page runs per rerun, so this resets just its own report.
    _reset_reports()
    page.run()


if __name__ == "__main__":
    main()
