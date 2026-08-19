"""Phase 8: the data/derivation layer - every Jira/GitHub read, the
engineering bundle derivation and caching (Phase 1-2), the disk-backed
warm snapshot (Phase 3), and the shared sidebar/scope-filter context every
engineering page opens with. Split out of the page renderers in app.py so
the read/derive layer can be reasoned about (and tested) on its own.
"""

from __future__ import annotations

import collections
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
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
from dataclasses import dataclass
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
from theme import inject_styles, kpi_strip
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
import vivino_client
import orders
import orders_client
from prioritization import add_priority_score, assignee_rollup
import sprint_planner
import ticket_quality
from transformations import add_ticket_health_fields
import write_access
from contextlib import contextmanager



from page_shared import (
    _as_frame,
    _log_stage,
    _positive_int,
)

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


SCOPE_ORG = "Organization"
SCOPE_TEAM = "Team"
SCOPE_INDIVIDUAL = "Individual"

# Bumped for 2A: fetch_resolved_tickets now requests expand="changelog" so
# credited_resolvers can attribute resolutions to the changelog author instead
# of the current assignee. A stale cache entry from before this change carries
# no changelog column at all, so the version bump forces a fresh read rather
# than crediting nothing out of an old, changelog-less cache hit.
FETCH_SCHEMA_VERSION = 9
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
_SCOPE_ASSIGNEES_KEY = "_scope_assignees"


# Every read below is cached the same way, and all of them refresh in the
# background: when the TTL lapses the reader gets the slightly stale answer
# immediately while the new one is fetched behind them, instead of one unlucky
# visitor every five minutes paying the full cold start for everybody else. The
# Refresh button is there for anyone who would rather wait for certainty.
@read_log.logged_read("app.fetch_tickets")
@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_tickets(
    creds_path: str,
    profile_name: str,
    jql: str,
    max_results: int,
    page_size: int,
    schema_version: int,
) -> pd.DataFrame:
    read_log.mark_executed()
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
# ``assignee`` here is the current assignee, kept for comparison against the
# credited resolver - see ``credited_resolvers`` - not the credit itself.
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


@read_log.logged_read("app.fetch_resolved_count")
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
    read_log.mark_executed()
    _ = schema_version
    if not statuses:
        return 0
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.approximate_count(_resolved_jql(statuses, days, ordered=False))


def _person_clause(person: str) -> str:
    """A JQL ``assignee = \"...\"`` clause.

    ``person`` is the Jira account id when the caller knows it (exact, and
    survives two people sharing a display name), else the display name.
    """
    escaped = str(person).strip().replace("\\", "\\\\").replace('"', '\\"')
    return f'assignee = "{escaped}"'


@read_log.logged_read("app.fetch_person_resolved_count")
@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_person_resolved_count(
    creds_path: str,
    profile_name: str,
    person: str,
    days: int,
    statuses: tuple[str, ...],
    schema_version: int,
) -> int | None:
    """One person's resolved count for a window, never paging-capped.

    Counted server-side like the headline tiles, so a 90-day window on a busy
    Jira is not silently truncated to its most recent thousand rows.
    """
    read_log.mark_executed()
    _ = schema_version
    if not statuses or not str(person).strip():
        return 0
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.approximate_count(
        f"{_person_clause(person)} AND {_resolved_jql(statuses, days, ordered=False)}"
    )


@read_log.logged_read("app.fetch_person_reopened_count")
@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_person_reopened_count(
    creds_path: str,
    profile_name: str,
    person: str,
    days: int,
    statuses: tuple[str, ...],
    schema_version: int,
) -> int | None:
    """One person's count of tickets that left a resolved status and stayed out."""
    read_log.mark_executed()
    _ = schema_version
    if not statuses or not str(person).strip():
        return 0
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.approximate_count(
        f"{_person_clause(person)} AND {_reopened_jql(statuses, days, ordered=False)}"
    )


# Estimates and issue type are all the weekly delivery view reads; the rest of
# DEFAULT_FIELDS would be a bigger payload for columns it never shows.
DELIVERY_FIELDS = ("assignee", "issuetype", "status", "summary", "timetracking")


@read_log.logged_read("app.fetch_person_resolved_history")
@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_person_resolved_history(
    creds_path: str,
    profile_name: str,
    person: str,
    weeks: int,
    statuses: tuple[str, ...],
    max_results: int,
    page_size: int,
    schema_version: int,
) -> pd.DataFrame:
    """One person's tickets resolved anywhere in the last ``weeks`` weeks, in one read.

    The weekly chart wants ``weeks`` separate totals, and used to ask Jira
    ``weeks`` times to get them - once per week, each search scoped to that
    week's own date window - which paid for the round trip to Jira as many
    times over for a person who resolves at most a few dozen tickets in three
    months. One search over the whole span answers all of them at once; which
    week each ticket falls into is then a local read of its own history
    (``expand=changelog``) rather than another trip to Jira. See
    :func:`_weekly_resolved_buckets`, which does that split.
    """
    read_log.mark_executed()
    _ = schema_version
    if not statuses or not str(person).strip():
        return pd.DataFrame()
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.search_issues(
        jql=(
            f"{_person_clause(person)} AND "
            f"{_resolved_jql(statuses, int(weeks) * 7, ordered=False)}"
        ),
        fields=list(DELIVERY_FIELDS),
        max_results=max_results,
        page_size=page_size,
        expand="changelog",
    )


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


@read_log.logged_read("app.fetch_created_count")
@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_created_count(
    creds_path: str,
    profile_name: str,
    days: int,
    schema_version: int,
) -> int | None:
    """Jira's server-side count of tickets created in the window (never paging-capped)."""
    read_log.mark_executed()
    _ = schema_version
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.approximate_count(_created_jql(days, ordered=False))


@read_log.logged_read("app.fetch_triage_stuck_count")
@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_triage_stuck_count(
    creds_path: str,
    profile_name: str,
    statuses: tuple[str, ...],
    hours: int,
    schema_version: int,
) -> int | None:
    """Server-side count of tickets stuck in triage past ``hours`` (never paging-capped)."""
    read_log.mark_executed()
    _ = schema_version
    if not statuses:
        return 0
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.approximate_count(_triage_stuck_jql(statuses, hours, ordered=False))


@read_log.logged_read("app.fetch_triage_stuck_tickets")
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
    read_log.mark_executed()
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


@read_log.logged_read("app.fetch_created_tickets")
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
    read_log.mark_executed()
    _ = schema_version
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.search_issues(
        jql=_created_jql(days),
        fields=list(LIST_FIELDS),
        max_results=max_results,
        page_size=page_size,
    )


@read_log.logged_read("app.fetch_resolved_tickets")
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

    Carries the changelog (``expand="changelog"``) so :func:`credited_resolvers`
    can attribute each resolution to the changelog author of the resolving
    transition instead of ``assignee`` - the current assignee, which is what
    this frame's ``assignee`` column has always been and remains, kept for
    comparison rather than as the credit itself. See ``credited_resolvers`` for
    why that distinction is the whole point.
    """
    read_log.mark_executed()
    _ = schema_version
    if not statuses:
        return pd.DataFrame()
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.search_issues(
        jql=_resolved_jql(statuses, days),
        fields=list(RESOLVED_FIELDS),
        max_results=max_results,
        page_size=page_size,
        expand="changelog",
    )


def credited_resolvers(
    resolved_tickets: pd.DataFrame,
    *,
    window_days: float | None = None,
    now: object | None = None,
    resolved_statuses: tuple[str, ...] | None = None,
) -> "integrity.CreditedResolutions":
    """The credit-attribution join: who resolved these tickets, per the changelog.

    ``fetch_resolved_tickets`` (above) has only ever carried the current
    ``assignee`` - not who did the work, just who holds the ticket at read time.
    A ticket resolved by one person and later reassigned credits the new
    assignee for a resolution they may never have touched; a ticket still
    assigned to someone who has since left the company credits them for
    resolutions performed entirely by whoever picks up their old tickets. That
    is exploit #4 in ``KPI_SPEC.md`` - "Sai Shankar, 194 resolved in 30d, second
    highest in the company" for a person no longer on the roster.

    This is a thin join, not a new computation: it flattens
    ``resolved_tickets["changelog"]`` with :func:`integrity.changelog_events` and
    hands the result to :func:`integrity.credited_resolutions`, which does the
    actual attribution (changelog author of the resolving transition) and the
    former-staff flagging (``JIRA_FORMER_STAFF``, see
    ``integrity._former_staff_from_env``). No new Jira reads happen here - the
    changelog ``fetch_resolved_tickets`` now carries is already in memory.

    ``window_days`` defaults to ``None`` (no further narrowing): the caller has
    usually already asked Jira for a specific window via ``days`` on
    ``fetch_resolved_tickets``, and re-windowing here on ``ts`` would silently
    disagree with that if the two ever drift. ``resolved_statuses`` defaults to
    this module's ``RESOLVED_STATUSES`` when not given.

    Returns :class:`integrity.CreditedResolutions` - ``detail`` (one row per
    resolving transition: key, timestamp, credited author, former-staff flag)
    and ``by_person`` (the credited ledger). See that function's docstring for
    the full column list and blind spots; nothing here changes them.
    """
    if resolved_tickets is None or resolved_tickets.empty:
        return integrity.credited_resolutions(integrity.empty_events())
    events = integrity.changelog_events(resolved_tickets)
    statuses = resolved_statuses if resolved_statuses is not None else RESOLVED_STATUSES
    return integrity.credited_resolutions(
        events,
        resolved_tickets,
        window_days=window_days,
        now=now,
        resolved_statuses=statuses,
    )


def _reopened_jql(statuses: tuple[str, ...], days: int, ordered: bool = True) -> str:
    """JQL for tickets that LEFT a resolved status within the last ``days``.

    A ticket that was called done and then moved out of every resolved status
    is rework - usually a bug that came back.
    """
    jql = (
        f"status CHANGED FROM ({_jql_status_list(statuses)}) AFTER -{int(days)}d "
        f"AND status NOT IN ({_jql_status_list(statuses)})"
    )
    return jql + " ORDER BY updated DESC" if ordered else jql


@read_log.logged_read("app.fetch_open_prs_cached")
@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_open_prs_cached(token: str, org: str, schema_version: int) -> pd.DataFrame:
    read_log.mark_executed()
    _ = schema_version
    return github_client.fetch_open_prs(token, org)


@read_log.logged_read("app.fetch_merged_prs_cached")
@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_merged_prs_cached(token: str, org: str, days: int, schema_version: int) -> pd.DataFrame:
    read_log.mark_executed()
    _ = schema_version
    return github_client.fetch_merged_prs(token, org, days)


@read_log.logged_read("app.fetch_merged_pr_count_cached")
@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_merged_pr_count_cached(token: str, org: str, days: int, schema_version: int) -> int:
    read_log.mark_executed()
    _ = schema_version
    return github_client.merged_pr_count(token, org, days)


@read_log.logged_read("app.fetch_open_pr_count_cached")
@st.cache_data(ttl=300, show_spinner=False, refresh_mode="background")
def fetch_open_pr_count_cached(token: str, org: str, schema_version: int) -> int:
    read_log.mark_executed()
    _ = schema_version
    return github_client.open_pr_count(token, org)


@read_log.logged_read("app.fetch_project_keys")
@st.cache_data(ttl=3600, show_spinner=False, refresh_mode="background")
def fetch_project_keys(creds_path: str, profile_name: str) -> list[str]:
    read_log.mark_executed()
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.get_project_keys()


@read_log.logged_read("app.fetch_all_priorities")
@st.cache_data(ttl=600, show_spinner=False, refresh_mode="background")
def fetch_all_priorities(creds_path: str, profile_name: str) -> list[str]:
    read_log.mark_executed()
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.get_all_priorities()


@read_log.logged_read("app.fetch_all_users")
@st.cache_data(ttl=600, show_spinner=False, refresh_mode="background")
def fetch_all_users(creds_path: str, profile_name: str) -> list[dict[str, str]]:
    read_log.mark_executed()
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.get_all_users()


@read_log.logged_read("app.fetch_available_transition_statuses")
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
    read_log.mark_executed()
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


# A link an engineer can be sent, so their page opens on them rather than on
# whoever the dashboard happens to default to.
_PERSON_PARAM = "person"


def requested_person(assignees: list[str]) -> str | None:
    """The engineer a shared link asks for, if it named one of ours.

    Matched loosely, like every other name here: a link written by hand, or one
    whose name came back from Jira cased differently, still has to land on the
    person rather than silently on somebody else.
    """
    wanted = str(st.query_params.get(_PERSON_PARAM) or "").strip()
    if not wanted:
        return None
    for name in assignees:
        if str(name).strip().lower() == wanted.lower():
            return name
    # Loose matching only where it names one person: "Mehdi" fits everyone
    # called Mehdi, and a link that opens a colleague's page is worse than one
    # that opens the usual default.
    loose = [name for name in assignees if same_person(wanted, name)]
    return loose[0] if len(loose) == 1 else None


# --- Sidebar choices that survive a page switch ------------------------------
#
# The six pages are one script, and a page that does not draw the sidebar (Today)
# lets Streamlit forget the widgets on it: a status filter set on Engineering was
# gone on Delivery, and still gone on coming back. A keyed widget is not enough
# for the same reason. So each choice is also kept under a plain session key of
# its own, which nothing garbage-collects, and read back as the widget's default.
_CARRIED_PREFIX = "carried_sidebar_"


def _carry(name: str, value: object) -> None:
    """Remember what the reader chose, for the next page they open."""
    st.session_state[_CARRIED_PREFIX + name] = value


def _carried(name: str, default: Any, options: Sequence[Any] | None = None) -> Any:
    """What they chose last, or ``default`` where there is nothing to carry.

    A carried choice is dropped when the data no longer offers it - a status that
    has left the board, a person off this JQL - because a default Streamlit
    cannot find raises rather than narrowing the view to nothing.
    """
    if _CARRIED_PREFIX + name not in st.session_state:
        return default
    value = st.session_state[_CARRIED_PREFIX + name]
    if options is None:
        return value
    allowed = list(options)
    if isinstance(value, list):
        return [item for item in value if item in allowed]
    return value if value in allowed else default


def _carried_roster(name: str, default: list[str], options: Sequence[str]) -> list[str]:
    """A carried list of names, where an empty one is a decision rather than a loss.

    Three cases, and only the last one is the reader's: nothing carried yet, every
    carried name gone from the board (both fall back to ``default``, or the view
    would open showing no tickets), and a selection the reader deliberately
    emptied, which is kept empty - restoring the whole team under someone who
    just cleared it is the widening this carry exists to prevent.
    """
    key = _CARRIED_PREFIX + name
    if key not in st.session_state:
        return default
    stored = list(st.session_state[key] or [])
    kept = [item for item in stored if item in list(options)]
    return default if stored and not kept else kept


def _resolve_scope_assignees(scope: str, assignees: list[str]) -> list[str] | None:
    """Return the assignees to filter on for the selected scope.

    None means "no assignee filter" (organization-wide); a list, including an
    empty one, is an explicit selection.
    """
    if scope == SCOPE_ORG:
        st.caption(f"Organization-wide view across {len(assignees)} assignee(s).")
        # Clear person parameter when viewing organization-wide
        if _PERSON_PARAM in st.query_params:
            logger.debug("Clearing person parameter from URL (Organization scope)")
            del st.query_params[_PERSON_PARAM]
        return None

    if scope == SCOPE_TEAM:
        defaults = _roster_matches(ORG_TEAM_MEMBERS, assignees)
        selected = st.multiselect(
            "Team members",
            options=assignees,
            default=_carried_roster("team_members", defaults, assignees),
        )
        _carry("team_members", selected)
        if not selected:
            st.warning("No team members selected - showing no tickets.")
        # Update URL to reflect single person selection in Team scope
        if len(selected) == 1:
            if selected[0] != requested_person(assignees):
                logger.debug(f"Setting person parameter in URL to '{selected[0]}' (Team scope, single selection)")
                st.query_params[_PERSON_PARAM] = selected[0]
        else:
            # Clear person parameter when multiple or no people selected
            if _PERSON_PARAM in st.query_params:
                logger.debug(f"Clearing person parameter from URL (Team scope, {len(selected)} selected)")
                del st.query_params[_PERSON_PARAM]
        return selected

    if not assignees:
        st.warning("No assignees available in the current data.")
        return []
    matches = _roster_matches(ORG_TEAM_MEMBERS, assignees)
    default_individual = (
        requested_person(assignees) or (matches[0] if matches else assignees[0])
    )
    # A link that names someone opens on them whatever was carried from the last
    # page: it was sent about that person.
    carried = requested_person(assignees) or _carried(
        "individual", default_individual, assignees
    )
    selected = st.selectbox(
        "Assignee",
        options=assignees,
        index=assignees.index(carried),
    )
    _carry("individual", selected)
    
    # Sync the selected person to the URL so the page can be shared
    # and the URL reflects who is being viewed
    if selected != requested_person(assignees):
        logger.debug(f"Setting person parameter in URL to '{selected}' (Individual scope)")
        st.query_params[_PERSON_PARAM] = selected
    
    return [selected]


def _board_fingerprint(raw_df: pd.DataFrame, jql: str, schema_version: int) -> tuple:
    """A cheap identity for a ticket read - not a hash of its rows.

    Hashing the full frame (nested changelog dicts and all) to decide whether
    the derived board is still fresh would cost what the derivation itself
    costs. Row count plus the newest ``updated`` timestamp changes whenever
    the read does - a ticket added, removed or touched - without walking a
    single changelog to find out.
    """
    if raw_df is None or raw_df.empty or "updated" not in raw_df.columns:
        max_updated = ""
    else:
        max_updated = str(
            pd.to_datetime(raw_df["updated"], utc=True, errors="coerce").max()
        )
    return (0 if raw_df is None else len(raw_df), max_updated, jql, schema_version)


@dataclass(frozen=True)
class _BoardDerivation:
    """The board, shaped once: the reshaped frame and its flattened changelog.

    ``df`` never carries the ``changelog`` column - that is the one field on
    this board expensive enough that copying it into every page's frame would
    be the derivation's whole cost. Anything that genuinely needs raw history
    asks ``events`` instead, which is parsed once here rather than once per
    caller.
    """

    df: pd.DataFrame
    events: pd.DataFrame


@read_log.logged_read("app._derive_board")
@st.cache_data(
    ttl=300,
    show_spinner=False,
    max_entries=4,
    # ``raw_df`` is hashed to a constant on purpose: the fingerprint argument
    # is the real cache key (row count, latest ``updated``, JQL, schema
    # version - see ``_board_fingerprint``), so a cache hit never walks or
    # re-serialises the frame's nested changelog dicts just to decide whether
    # it already has the answer.
    hash_funcs={pd.DataFrame: lambda _df: None},
)
def _derive_board(raw_df: pd.DataFrame, fingerprint: tuple) -> _BoardDerivation:
    """Shape the raw ticket read once: health fields, priority score, changelog events.

    Everything downstream that used to reparse ``df["changelog"]`` on every
    call - ``_stalled_rows``, ``_cycle_by_status``, ``_stale_with_masked`` -
    now reads ``events`` from here instead, computed exactly once per board
    read regardless of how many pages or reruns ask for it.
    """
    read_log.mark_executed()
    _ = fingerprint  # part of the cache key only; the real work reads raw_df
    events = integrity.changelog_events(raw_df)
    shaped = add_priority_score(add_ticket_health_fields(raw_df))
    if "changelog" in shaped.columns:
        shaped = shaped.drop(columns=["changelog"])
    return _BoardDerivation(df=shaped, events=events)


@dataclass(frozen=True)
class _EngineeringData:
    """Everything one engineering page draw needs, read exactly once.

    The reads were inlined in the single engineering page. Six pages now share
    them, and a page that re-derived its own copy would re-run the whole gather
    on every navigation. The caches make the repeat cheap; this makes it once.
    """

    data: Any
    errors: Any
    raw_df: Any
    df: Any
    events: Any
    github_ready: Any
    github_error: Any
    open_prs: Any
    merged_prs: Any
    pr_count_7: Any
    pr_count_30: Any
    open_count_exact: Any
    assignees: Any
    statuses: Any
    priorities: Any
    max_results: Any
    page_size: Any


# --------------------------------------------------------------------------
# Warm board snapshot: a cold start reads this instead of an empty screen and
# an eight-second wait, while a background thread brings both it and the
# session's own cache up to date behind the reader.
# --------------------------------------------------------------------------

_SNAPSHOT_PATH = Path(tempfile.gettempdir()) / "jira_dashboard_board_snapshot.pkl"
# Generous past the ~5-15 minutes a healthy background refresh cycle keeps it
# at: a bound against a stopped or failing refresh serving a board nobody
# would recognise, not the cadence itself.
_SNAPSHOT_STALE_LIMIT_SECONDS = 1800.0
# Comfortably above a real board's derived frame (no changelog column, no raw
# issue payloads), well below "a snapshot large enough to be its own outage".
_SNAPSHOT_MAX_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class _PersistedBoard:
    """What actually goes to disk: the shaped board, not the raw read.

    ``raw_df`` (and the changelog it carries) never reaches this - the same
    reason Phase 1's bundle drops it from every page's frame applies doubly to
    a file every cold start reads back.
    """

    written_at: float
    df: pd.DataFrame
    events: pd.DataFrame
    data: dict
    github_ready: bool
    github_error: str
    open_prs: pd.DataFrame
    merged_prs: pd.DataFrame
    pr_count_7: Any
    pr_count_30: Any
    open_count_exact: Any
    assignees: list
    statuses: list
    priorities: list
    max_results: int
    page_size: int


def _write_board_snapshot(bundle: "_EngineeringData") -> None:
    """Persist the derived bundle to local disk, with a written-at stamp.

    Written atomically (a temp file, then a rename) so a reader mid-read of
    the real path never sees a half-written pickle. Errors here are logged
    and swallowed - a snapshot write is a courtesy to the *next* cold start,
    not something the run that triggered it depends on.
    """
    try:
        persisted = _PersistedBoard(
            written_at=time.time(),
            df=bundle.df,
            events=bundle.events,
            data=bundle.data,
            github_ready=bundle.github_ready,
            github_error=bundle.github_error,
            open_prs=bundle.open_prs,
            merged_prs=bundle.merged_prs,
            pr_count_7=bundle.pr_count_7,
            pr_count_30=bundle.pr_count_30,
            open_count_exact=bundle.open_count_exact,
            assignees=bundle.assignees,
            statuses=bundle.statuses,
            priorities=bundle.priorities,
            max_results=bundle.max_results,
            page_size=bundle.page_size,
        )
        _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Unique per writer: a live gather's own write and a background
        # refresh's write can land in the same few seconds, and two writers
        # sharing one temp name means the first to rename it away can leave
        # the second stat()-ing a file that is no longer there.
        tmp_path = _SNAPSHOT_PATH.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp_path, "wb") as fh:
            pickle.dump(persisted, fh, protocol=pickle.HIGHEST_PROTOCOL)
        if tmp_path.stat().st_size > _SNAPSHOT_MAX_BYTES:
            tmp_path.unlink(missing_ok=True)
            logger.warning(
                "Board snapshot exceeded %d bytes; not written", _SNAPSHOT_MAX_BYTES
            )
            return
        tmp_path.replace(_SNAPSHOT_PATH)
    except Exception:  # noqa: BLE001 - a failed write must not break the page that triggered it
        logger.exception("Could not write the board snapshot")


def _read_board_snapshot() -> tuple["_EngineeringData", float] | None:
    """The on-disk snapshot and its age, or None on anything but a good, fresh read.

    Fails open, always: a missing file, a snapshot older than the staleness
    limit, and a corrupt or otherwise unreadable file all take the same path
    back to the caller - None, meaning "read live" - rather than raising into
    a page that only wanted a fast start.
    """
    try:
        if not _SNAPSHOT_PATH.exists():
            return None
        with open(_SNAPSHOT_PATH, "rb") as fh:
            persisted: _PersistedBoard = pickle.load(fh)
        age = time.time() - persisted.written_at
        if age < 0 or age > _SNAPSHOT_STALE_LIMIT_SECONDS:
            return None
        bundle = _EngineeringData(
            data=persisted.data,
            errors={},
            raw_df=persisted.df,
            df=persisted.df,
            events=persisted.events,
            github_ready=persisted.github_ready,
            github_error=persisted.github_error,
            open_prs=persisted.open_prs,
            merged_prs=persisted.merged_prs,
            pr_count_7=persisted.pr_count_7,
            pr_count_30=persisted.pr_count_30,
            open_count_exact=persisted.open_count_exact,
            assignees=persisted.assignees,
            statuses=persisted.statuses,
            priorities=persisted.priorities,
            max_results=persisted.max_results,
            page_size=persisted.page_size,
        )
        return bundle, persisted.written_at
    except Exception:  # noqa: BLE001 - any failure to read falls back to a live read
        logger.exception("Could not read the board snapshot; falling back to a live read")
        return None


def _delete_board_snapshot() -> None:
    """Drop the on-disk snapshot, so a stale file cannot outlive the reads it came from."""
    try:
        _SNAPSHOT_PATH.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - deleting a snapshot must not break Refresh
        logger.exception("Could not delete the board snapshot")


_ENGINEERING_BG_REFRESH_KEY = "_engineering_bg_refresh_running"


def _start_background_board_refresh(max_results: int, page_size: int) -> None:
    """Bring the live reads and the on-disk snapshot up to date behind a snapshot-served reader.

    Fire-and-forget: no Streamlit element is drawn from this thread, so it
    needs no progress bar, and nothing it does can raise into the page that is
    already rendering from the snapshot. At most one refresh runs per session
    at a time.
    """
    if st.session_state.get(_ENGINEERING_BG_REFRESH_KEY):
        return
    st.session_state[_ENGINEERING_BG_REFRESH_KEY] = True
    ctx = get_script_run_ctx()

    def _run() -> None:
        if ctx is not None:
            add_script_run_ctx(threading.current_thread(), ctx)
        try:
            outcome = _engineering_gather_and_shape(max_results, page_size)
            if outcome.bundle is not None:
                _write_board_snapshot(outcome.bundle)
                fingerprint = _board_fingerprint(outcome.bundle.raw_df, JQL, FETCH_SCHEMA_VERSION)
                try:
                    now = time.time()
                    st.session_state[_ENGINEERING_BUNDLE_KEY] = (
                        fingerprint,
                        now,
                        outcome.bundle,
                    )
                    st.session_state[_ENGINEERING_DATA_AS_OF_KEY] = now
                except Exception:  # noqa: BLE001 - the session may be gone by now
                    logger.exception(
                        "Could not update the session's board cache after a background refresh"
                    )
            else:
                logger.warning(
                    "Background board refresh produced no usable board: %s",
                    outcome.fatal_error,
                )
        except Exception:  # noqa: BLE001 - a failed background refresh must surface nowhere
            logger.exception("Background board refresh failed")
        finally:
            try:
                st.session_state[_ENGINEERING_BG_REFRESH_KEY] = False
            except Exception:  # noqa: BLE001 - the session may be gone by now
                pass

    threading.Thread(target=_run, daemon=True, name="board-snapshot-refresh").start()


_ENGINEERING_BUNDLE_KEY = "_engineering_bundle_cache"
# Matches fetch_tickets' own TTL: the session-held bundle should not outlive
# the reads it was built from by more than the reads themselves would.
_ENGINEERING_BUNDLE_TTL_SECONDS = 300.0
# When the data behind the held bundle was actually read - the snapshot's
# written-at stamp, or the moment a live gather finished. Separate from the
# timestamp inside _ENGINEERING_BUNDLE_KEY, which times how long *this
# session* has trusted what it is holding, not how old the data itself is;
# the caption next to Refresh reads this one.
_ENGINEERING_DATA_AS_OF_KEY = "_engineering_data_as_of"


def _engineering_data() -> "_EngineeringData":
    """The current bundle: the session's held one if it still matches, a fresh gather otherwise.

    Every one of the six engineering pages calls this on its own navigation,
    each a separate script rerun. The per-read caches (``fetch_tickets`` and
    the other fourteen) already make a warm *read* cheap; what still ran on
    every navigation before this was the orchestration around them - the
    thread pool that gathers all fifteen together - and the reshape that
    follows. ``fetch_tickets`` is safe to call speculatively here: it is the
    exact cache the real read below would hit, so probing it first is not an
    extra read, and its fingerprint (the Phase 1 key) says whether the board
    has actually moved since the session last paid for the rest.
    """
    max_results = MAX_RESULTS
    page_size = JIRA_PAGE_SIZE

    try:
        probe_df = fetch_tickets(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
            jql=JQL,
            max_results=max_results,
            page_size=page_size,
            schema_version=FETCH_SCHEMA_VERSION,
        )
    except Exception:  # noqa: BLE001 - the full read below reports this properly
        probe_df = None

    if probe_df is not None:
        probe_fingerprint = _board_fingerprint(probe_df, JQL, FETCH_SCHEMA_VERSION)
        held = st.session_state.get(_ENGINEERING_BUNDLE_KEY)
        if held is not None:
            held_fingerprint, held_at, held_bundle = held
            if held_fingerprint == probe_fingerprint and (
                time.time() - held_at < _ENGINEERING_BUNDLE_TTL_SECONDS
            ):
                return held_bundle

    # A cold session (nothing held yet) reads the on-disk snapshot before
    # paying for a live gather - a board a few minutes old beats an empty
    # screen and an eight-second wait. A background thread brings the
    # session's own cache and the snapshot up to date behind this reader
    # without making them wait a second time. A missing, stale-beyond-limit
    # or unreadable snapshot falls back to the live path exactly - the same
    # path a reader with no snapshot at all has always taken.
    if st.session_state.get(_ENGINEERING_BUNDLE_KEY) is None:
        snapshot = _read_board_snapshot()
        if snapshot is not None:
            snapshot_bundle, written_at = snapshot
            snapshot_fingerprint = _board_fingerprint(
                snapshot_bundle.raw_df, JQL, FETCH_SCHEMA_VERSION
            )
            # Held from *now*, not from the snapshot's own age: the TTL above
            # guards how long this session trusts what it is holding, and the
            # background refresh below needs the room to finish before that
            # guard would otherwise force a second, synchronous live gather
            # on this same session's very next navigation.
            st.session_state[_ENGINEERING_BUNDLE_KEY] = (
                snapshot_fingerprint,
                time.time(),
                snapshot_bundle,
            )
            st.session_state[_ENGINEERING_DATA_AS_OF_KEY] = written_at
            _start_background_board_refresh(max_results, page_size)
            return snapshot_bundle

    bundle = _engineering_data_uncached(max_results, page_size)
    fresh_fingerprint = _board_fingerprint(bundle.raw_df, JQL, FETCH_SCHEMA_VERSION)
    now = time.time()
    st.session_state[_ENGINEERING_BUNDLE_KEY] = (fresh_fingerprint, now, bundle)
    st.session_state[_ENGINEERING_DATA_AS_OF_KEY] = now
    _write_board_snapshot(bundle)
    return bundle


class _EngineeringReadOutcome(NamedTuple):
    """What one gather-and-shape pass produced, with no Streamlit calls in it.

    ``fatal_error``/``empty_board`` are messages, not exceptions, precisely so
    a caller with no page to draw on - the background snapshot refresh - can
    tell a real failure from a usable board without a ``try`` around
    ``st.stop()``.
    """

    bundle: "_EngineeringData | None"
    fatal_error: str | None
    empty_board: bool
    truncated: bool


def _engineering_gather_and_shape(
    max_results: int,
    page_size: int,
    *,
    on_progress: Callable[[float, str], None] | None = None,
) -> _EngineeringReadOutcome:
    """The reads and the reshape, free of Streamlit element calls.

    Split out of the page-facing path so the background snapshot refresh (see
    ``_start_background_board_refresh``) can run the identical gather-and-shape
    work off the script thread, where a progress bar or ``st.stop()`` would be
    meaningless at best and unsafe at worst.
    """
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

    data, errors = _gather(reads, on_progress=on_progress)

    if "tickets" in errors:
        failure = errors["tickets"]
        message = (
            f"Configuration error: {failure}"
            if isinstance(failure, JiraConfigError)
            else f"Failed to fetch Jira issues: {failure}"
        )
        return _EngineeringReadOutcome(None, message, False, False)

    raw_df = _as_frame(data["tickets"])
    if raw_df.empty:
        return _EngineeringReadOutcome(None, None, True, False)

    truncated = len(raw_df) >= max_results

    with _log_stage("changelog:derive_board"):
        fingerprint = _board_fingerprint(raw_df, JQL, FETCH_SCHEMA_VERSION)
        derived = _derive_board(raw_df, fingerprint)
    df = derived.df

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

    bundle = _EngineeringData(
        data=data,
        errors=errors,
        raw_df=raw_df,
        df=df,
        events=derived.events,
        github_ready=github_ready,
        github_error=github_error,
        open_prs=open_prs,
        merged_prs=merged_prs,
        pr_count_7=pr_count_7,
        pr_count_30=pr_count_30,
        open_count_exact=open_count_exact,
        assignees=assignees,
        statuses=statuses,
        priorities=priorities,
        max_results=max_results,
        page_size=page_size,
    )
    return _EngineeringReadOutcome(bundle, None, False, truncated)


def _engineering_data_uncached(max_results: int, page_size: int) -> "_EngineeringData":
    """Load and shape the engineering reads. Calls st.stop() on a fatal read.

    Always does the full gather and reshape - ``_engineering_data`` is the
    entry point every page calls, and is what decides whether this runs at
    all. This is the UI shell around ``_engineering_gather_and_shape``: the
    progress bar and the error/warning/stop calls that only make sense on the
    script thread actually serving a reader.
    """
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

    outcome = _engineering_gather_and_shape(max_results, page_size, on_progress=_show_progress)
    loading_slot.empty()

    if outcome.fatal_error:
        st.error(outcome.fatal_error)
        st.stop()

    if outcome.empty_board:
        st.warning("No tickets returned for the current JQL.")
        st.stop()

    if outcome.truncated:
        st.warning(
            f"Showing the first {max_results} tickets of a larger result set - the JQL "
            "orders by least recently updated, so newer tickets are missing. Narrow "
            "JIRA_DASHBOARD_JQL or raise JIRA_MAX_RESULTS."
        )

    assert outcome.bundle is not None  # every other outcome branch returned above
    return outcome.bundle


@dataclass(frozen=True)
class _EngineeringView:
    """The sidebar's answer: what this reader is looking at, and the frames for it.

    Six pages share one sidebar. Rendering it per page is correct - Streamlit runs
    only the open page, so the widgets have to be declared by whichever page that
    is - but the filtering that follows must not be re-derived six different ways.
    """

    scope: Any
    selected_assignees: Any
    selected_statuses: Any
    selected_priorities: Any
    min_idle: Any
    min_age: Any
    include_backlogs: Any
    color_by: Any
    allow_writes: Any
    filtered: Any
    unscoped: Any


def _engineering_filters(bundle: "_EngineeringData") -> "_EngineeringView":
    """Draw the scope and filter sidebar, then apply it to the ticket frame."""
    df = bundle.df
    assignees, statuses, priorities = bundle.assignees, bundle.statuses, bundle.priorities

    with st.sidebar:
        st.header("Scope")
        scopes = [SCOPE_ORG, SCOPE_TEAM, SCOPE_INDIVIDUAL]
        scope = st.radio(
            "View",
            options=scopes,
            # A link that names an engineer opens on that engineer, or the page
            # they were sent would greet them with the whole organization. Any
            # other run opens on the scope the reader last chose, so walking from
            # Engineering to People does not silently widen the view back to the
            # whole organization.
            index=scopes.index(
                SCOPE_INDIVIDUAL
                if requested_person(assignees)
                else _carried("scope", SCOPE_ORG, scopes)
            ),
            help=(
                "Organization shows every assignee in the JQL scope; "
                "Team pre-selects the configured team members; "
                "Individual focuses on a single assignee."
            ),
        )
        _carry("scope", scope)
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
            selected_statuses = st.multiselect(
                "Status", options=statuses, default=_carried("statuses", [], statuses)
            )
            selected_priorities = st.multiselect(
                "Priority",
                options=priorities,
                default=_carried("priorities", [], priorities),
            )
            min_idle = st.slider(
                "Min idle days", min_value=0, max_value=180, value=int(_carried("min_idle", 0))
            )
            min_age = st.slider(
                "Min ticket age", min_value=0, max_value=365, value=int(_carried("min_age", 0))
            )
            include_backlogs = st.checkbox(
                "Include Backlogs", value=bool(_carried("include_backlogs", False))
            )
            color_by = st.segmented_control(
                "Bubble color",
                options=["priority", "assignee"],
                default=_carried("color_by", "priority", ["priority", "assignee"]),
            )
            st.form_submit_button("Apply filters", width="stretch")
        # A cleared segmented control returns None; the chart needs a column.
        color_by = color_by or "priority"
        for name, value in (
            ("statuses", selected_statuses),
            ("priorities", selected_priorities),
            ("min_idle", min_idle),
            ("min_age", min_age),
            ("include_backlogs", include_backlogs),
            ("color_by", color_by),
        ):
            _carry(name, value)

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

    return _EngineeringView(
        scope=scope,
        selected_assignees=selected_assignees,
        selected_statuses=selected_statuses,
        selected_priorities=selected_priorities,
        min_idle=min_idle,
        min_age=min_age,
        include_backlogs=include_backlogs,
        color_by=color_by,
        allow_writes=allow_writes,
        filtered=filtered,
        unscoped=unscoped,
    )


def _engineering_context() -> tuple["_EngineeringData", "_EngineeringView", Any]:
    """One page's worth of setup: the reads, the sidebar, and the report slot.

    Every engineering page calls this first. The reads are cached, so the second
    page a reader opens pays for the sidebar and nothing else.
    """
    slot = st.columns([5, 1])[1]
    bundle = _engineering_data()
    view = _engineering_filters(bundle)
    return bundle, view, slot
