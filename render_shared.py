"""Widget- and data-shaping helpers shared by two or more of the six focused
engineering pages (Today, People, Delivery, Code, Planning) and the legacy
combined Engineering page. Split out of app.py in Task 1C: a page module
must not import another page module, so anything more than one page calls
lives here instead.

Extracted mechanically - the code below is unchanged from app.py except for
where a symbol lives. See docs/assumptions/1C.md for the extraction record.

Frozen public surface (Task 1C hands this to seven parallel agents - treat
every name below as an API, not an implementation detail):

Page-title / navigation constants: TODAY_PAGE_TITLE, PEOPLE_PAGE_TITLE,
DELIVERY_PAGE_TITLE, CODE_PAGE_TITLE, PLANNING_PAGE_TITLE,
ENGINEERING_PAGE_TITLE, BUSINESS_PAGE_TITLE, PAGE_HEADINGS, BACKLOG_STATUSES,
WEEKLY_HOURS, TEAM_PROJECTS, TEAM_PEOPLE, JIRA_BROWSE_BASE,
TODAY_NO_REVIEWER_DAYS, TODAY_STALLED_DAYS.

Identity / links: _is_bot, _people_only, person_link, _jira_ticket_url,
_browse_base, _default_browse_base.

Board shaping: _metrics_df, _normalize_sprint_id, annotated_board,
engineer_page, _workload_hours.

Per-assignee drill-down: _render_assignee_detail, _render_workload_metrics,
_render_engineer_handout, _attention_tier, _tier_legend_html, _devin_can_handle,
_sprint_label, _kw_hit.

Section renderers used by two or more pages: _render_scope_breakdown,
_render_priority_queue, _render_team_overview, _render_epics,
_render_epic_organization, _render_cleanup, _render_triage_card,
_render_triage_decisions, _triage_state, _render_estimate_policy,
_render_stale_cleanup, _render_sprint_plan, _apply_sprint_plan,
_render_sprint_capacity, _render_hourly_capacity, _sprint_window,
_render_status_pills, _apply_action_with_audit, _render_ticket_list,
_render_new_and_triage, _render_pr_hygiene, _render_ticket_quality,
_render_personal_prs, _render_scorecard, _render_weekly_delivery,
_render_individual_page, _one_person_instead.

Small data helpers: _metric_value, _shown, _dated, _known_project_keys,
_jql_identity, _weekly_resolved_buckets, _open_pr_signals, _stalled_rows,
_exclude_repos, _truncation_note, _fmt_seconds, _parse_estimate_to_seconds,
_transition_sample_keys, _as_number, _estimated.

Page-cache bookkeeping: _clear_page_caches.

Import discipline: this module keeps the same top-level import block app.py
carried before the split (data_layer, page_shared, the analytics and client
modules, pandas, plotly, streamlit) rather than deferring them into each
function. See docs/assumptions/1C.md for why: page_shared.py, a hard
dependency of app.py's own remaining code, already imports every one of
these eagerly, so deferring them here would not change what "import app"
loads - only the six page modules (pages/today.py etc.) get the
function-local treatment, because those genuinely are the files Task 1C
asks to stay light.
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
    AMBIGUOUS_AVAILABILITY,
    AT_CAPACITY,
    HAS_ROOM,
    NO_AVAILABILITY,
    OVER_COMMITTED,
    UNALLOCATED,
    UNKNOWN_AVAILABILITY,
    capacity_table,
    capacity_table_by_sprint,
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
import planning_metrics
from teams import (
    NO_OWNER_TEAM,
    DEFAULT_TEAM_PEOPLE,
    add_team,
    parse_team_people,
    parse_team_projects,
    team_summary,
)
import theme
from theme import inject_styles
import theme_tokens
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



from data_layer import (
    DEFAULT_JQL,
    CREDS_PATH,
    PROFILE_NAME,
    JQL,
    ORG_TEAM_MEMBERS,
    _NO_OWNER_NAMES,
    SCOPE_ORG,
    SCOPE_TEAM,
    SCOPE_INDIVIDUAL,
    FETCH_SCHEMA_VERSION,
    MAX_RESULTS,
    JIRA_PAGE_SIZE,
    _DEFAULT_RESOLVED_STATUSES,
    RESOLVED_STATUSES,
    _DEFAULT_TRIAGE_STATUSES,
    TRIAGE_STATUSES,
    TRIAGE_STUCK_HOURS,
    _SCOPE_ASSIGNEES_KEY,
    fetch_tickets,
    LIST_FIELDS,
    RESOLVED_FIELDS,
    _jql_status_list,
    _resolved_jql,
    fetch_resolved_count,
    _person_clause,
    fetch_person_resolved_count,
    fetch_person_reopened_count,
    DELIVERY_FIELDS,
    fetch_person_resolved_history,
    _created_jql,
    _triage_stuck_jql,
    fetch_created_count,
    fetch_triage_stuck_count,
    fetch_triage_stuck_tickets,
    fetch_created_tickets,
    fetch_resolved_tickets,
    _reopened_jql,
    fetch_open_prs_cached,
    fetch_merged_prs_cached,
    fetch_merged_pr_count_cached,
    fetch_open_pr_count_cached,
    fetch_project_keys,
    fetch_all_priorities,
    fetch_all_users,
    fetch_available_transition_statuses,
    _PROGRESS_TICK_SECONDS,
    _PROGRESS_AFTER_SECONDS,
    _LOADING_PACE_SECONDS,
    _LOADING_CEILING,
    _load_fraction,
    _gather,
    _parallel,
    _engineering_reads,
    _GITHUB_READS,
    _github_reads,
    _roster_matches,
    _PERSON_PARAM,
    requested_person,
    _CARRIED_PREFIX,
    _carry,
    _carried,
    _carried_roster,
    _resolve_scope_assignees,
    _board_fingerprint,
    _BoardDerivation,
    _derive_board,
    _EngineeringData,
    _SNAPSHOT_PATH,
    _SNAPSHOT_STALE_LIMIT_SECONDS,
    _SNAPSHOT_MAX_BYTES,
    _PersistedBoard,
    _write_board_snapshot,
    _read_board_snapshot,
    _delete_board_snapshot,
    _ENGINEERING_BG_REFRESH_KEY,
    _start_background_board_refresh,
    _ENGINEERING_BUNDLE_KEY,
    _ENGINEERING_BUNDLE_TTL_SECONDS,
    _ENGINEERING_DATA_AS_OF_KEY,
    _engineering_data,
    _EngineeringReadOutcome,
    _engineering_gather_and_shape,
    _engineering_data_uncached,
    _EngineeringView,
    _engineering_filters,
    _engineering_context,
)

from page_shared import (
    _log_stage,
    _text_or,
    _number_or,
    _positive_int,
    TAB_ENGINEERING,
    TAB_BUSINESS,
    REPORTS_KEY,
    BOARD_ASKED_KEY,
    BOARD_SLOT_KEY,
    BOARD_FILE_KEY,
    _report,
    _reset_reports,
    _tile,
    _said,
    _download_report,
    _offer_board_snapshot,
    _page_name,
    _build_board_file,
    _deliver_board_snapshot,
    _unmathed,
    _as_frame,
)


logger = logging.getLogger(__name__)

# GitHub logins that are not people. They open and merge real pull requests, so
# they belong in every org-wide total - the work happened - but they are not
# engineers and a per-person table that ranks Devin above a human is a table
# nobody can act on. Matched on the login with any "[bot]" suffix stripped,
# because GitHub spells the same account both ways depending on the API.
# Extend with a comma-separated GITHUB_BOT_LOGINS rather than editing this list.
_DEFAULT_BOT_LOGINS = (
    "devin-ai-integration",
    "github-actions",
    "dependabot",
    "dependabot-preview",
    "renovate",
    "renovate-bot",
    "codecov",
    "sonarcloud",
)
BOT_LOGINS = {
    login.strip().lower()
    for login in os.getenv(
        "GITHUB_BOT_LOGINS", ",".join(_DEFAULT_BOT_LOGINS)
    ).split(",")
    if login.strip()
}


def _is_bot(login: object) -> bool:
    """Whether this GitHub login belongs to an automation rather than a person.

    Args:
        login: a GitHub login (any type; coerced to ``str``), with or
            without the ``[bot]`` suffix GitHub sometimes appends.

    Returns:
        True if the lower-cased, suffix-stripped login is in ``BOT_LOGINS``.
    """
    name = str(login or "").strip().lower()
    if name.endswith("[bot]"):
        name = name[: -len("[bot]")]
    return name in BOT_LOGINS


def _people_only(frame: pd.DataFrame, column: str = "author") -> tuple[pd.DataFrame, int]:
    """The rows written by people, and how many bot rows were set aside.

    Args:
        frame: any frame with a login column (PRs, reviews, ...).
        column: the name of that login column. Defaults to ``"author"``.

    Returns:
        A ``(frame_without_bots, bot_row_count)`` tuple. ``frame`` is
        returned unchanged if it is empty or lacks ``column``.

    The count comes back with the frame so that the caller can say so on the
    page. A table that silently drops half the pull requests is worse than one
    that includes the bots: the reader compares it against an org-wide tile,
    finds the two disagree, and stops trusting both.
    """
    if frame.empty or column not in frame.columns:
        return frame, 0
    is_bot = frame[column].map(_is_bot)
    return frame[~is_bot], int(is_bot.sum())

TODAY_PAGE_TITLE = "Today"
PEOPLE_PAGE_TITLE = "People"
DELIVERY_PAGE_TITLE = "Delivery"
CODE_PAGE_TITLE = "Code"
PLANNING_PAGE_TITLE = "Planning"
ENGINEERING_PAGE_TITLE = "Engineering"
BUSINESS_PAGE_TITLE = "Business"

# What each page calls itself in its own heading. The navigation labels above are
# one word each because a top tab strip has no room for more; a heading has, and
# the two pages are about entirely different things.
PAGE_HEADINGS = {
    TODAY_PAGE_TITLE: "Today — What Needs a Decision",
    PEOPLE_PAGE_TITLE: "People — Scorecards Within Role",
    DELIVERY_PAGE_TITLE: "Delivery — Throughput & Queues",
    CODE_PAGE_TITLE: "Code — Pull Request Health",
    PLANNING_PAGE_TITLE: "Planning — Sprints, Capacity & Hygiene",
    ENGINEERING_PAGE_TITLE: "Engineering — Ticket & PR Health",
    BUSINESS_PAGE_TITLE: "Business — Orders, Revenue & Spend",
}
JIRA_KEY_DISPLAY_PATTERN = r".*/browse/([^/?#]+)$"

# One Jira request per key, so bound how many the sprint editor asks about.
TRANSITION_LOOKUP_LIMIT = 50
# Upper bound on how many tickets a bulk write-back pre-selects.
BULK_ACTION_DEFAULT_LIMIT = 25
# Rows the composition chart draws at most, the last of which is the collapsed
# "Other" when there are more categories than that.
MIX_SLICE_LIMIT = 10
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

    Args:
        key: a Jira issue key such as ``"ENG-123"``.

    Returns:
        ``JIRA_BROWSE_BASE`` joined to the URL-quoted key, e.g.
        ``"https://vinovoss.atlassian.net/browse/ENG-123"``.

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


def _metrics_df(df: pd.DataFrame, include_backlogs: bool) -> pd.DataFrame:
    """``df`` with Backlog-family statuses dropped, unless the reader asked to keep them.

    Args:
        df: a board frame with a ``status`` column (any other shape is
            returned unchanged).
        include_backlogs: the sidebar's "Include Backlogs" toggle. True
            returns ``df`` as-is; False drops rows whose status (lower-cased,
            trimmed) is in ``BACKLOG_STATUSES``.

    Returns:
        ``df``, or a filtered copy. Every page calls this before drawing a
        headline count so Today, Delivery, People, Code, Planning and the
        Engineering page cannot disagree about what "open" means.
    """
    if include_backlogs or "status" not in df.columns:
        return df
    statuses = df["status"].fillna("").astype(str).str.strip().str.lower()
    return df[~statuses.isin(BACKLOG_STATUSES)]


def person_link(person: str) -> str:
    """The URL that opens this person's own page.

    Args:
        person: the assignee/author name to link to (used as-is in the
            ``_PERSON_PARAM`` query string, so callers do not pre-encode it).

    Returns:
        The current page's URL with ``person`` set as its ``_PERSON_PARAM``
        query parameter, or ``"?<query>"`` alone when the browser URL is not
        available (e.g. outside a running Streamlit session).

    Built from the browser's own address rather than a configured hostname: the
    dashboard is reached by several (Cloud Run, a proxy, localhost) and a link
    that names the wrong one is worse than no link.
    """
    query = urlencode({_PERSON_PARAM: str(person)})
    current = str(getattr(st.context, "url", "") or "")
    if not current:
        return f"?{query}"
    return f"{current.split('?')[0]}?{query}"


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

# The semantic tone behind each tier - theme_tokens.STATUS's four-tone
# vocabulary rather than a fifth invented hue. Feeds both the row-background
# map below (light backgrounds so the row text stays readable) and the
# legend (_tier_legend_html): a purple tier used to mean nothing outside
# this dict, and now reads as the same "gray" every other muted chip on the
# dashboard uses. See docs/assumptions/5A.md.
_TIER_TONE = {
    _TIER_RED: "crit",
    _TIER_YELLOW: "warn",
    _TIER_GREEN: "good",
    _TIER_PURPLE: "gray",
}
_TIER_BG = {
    tier: theme_tokens.STATUS_BG.get(tone, theme_tokens.MUTED_BG)
    for tier, tone in _TIER_TONE.items()
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
    """A small colour legend so the tiers read without guessing.

    Built from theme_html.legend() - the one legend component the dashboard
    draws everywhere else, four series or four tiers alike - rather than its
    own row of ``<span style="background:...">`` swatches coloured from a
    four-entry hex map that matched none of theme_tokens' own tones (a
    Bootstrap alert palette, not this app's). Wrapped in the ``.vv`` scope
    itself, same as ``theme_html``'s own CSS variables need, since the call
    site hands this straight to ``st.markdown`` rather than through
    ``theme_html.render()``. See docs/assumptions/5A.md.
    """
    fragment = theme_html.legend(
        [
            ("Needs attention — high score or idle 90d+", _TIER_TONE[_TIER_RED]),
            ("Watch — moderate score or idle 30d+", _TIER_TONE[_TIER_YELLOW]),
            ("Healthy — recent, low pressure", _TIER_TONE[_TIER_GREEN]),
            ("Low priority — parked (Low / Lowest / Idea)", _TIER_TONE[_TIER_PURPLE]),
        ]
    )
    return f'<div class="vv">{fragment}</div>'


def _sprint_label(row: pd.Series) -> str:
    """Where a ticket sits in planning: its sprint, or Backlog, or its status."""
    name = str(row.get("sprint_name") or "").strip()
    state = str(row.get("sprint_state") or "").strip().lower()
    if name and state in {"active", "future"}:
        return f"{name} ({state})"
    status = str(row.get("status") or "").strip()
    if status.lower() in BACKLOG_STATUSES:
        return "Backlog"
    return f"No sprint ({status})" if status else "No sprint"


_URGENT_PRIORITIES = {"highest", "high", "urgent", "critical", "blocker"}


def _workload_hours(owned: pd.DataFrame) -> dict[str, float | int]:
    """The estimated-hour totals behind the one-on-one tiles."""
    hours = pd.to_numeric(
        owned.get("estimate_hours", pd.Series(index=owned.index, dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    state = owned.get("sprint_state", pd.Series(index=owned.index, dtype=object))
    state = state.fillna("").astype(str).str.strip().str.lower()
    priority = owned.get("priority", pd.Series(index=owned.index, dtype=object))
    priority = priority.fillna("").astype(str).str.strip().str.lower()
    # "No estimate" follows the same flag as the Estimate? column - a ticket
    # estimated only in words counts as estimated there, so it must here too.
    # Containers (epics, initiatives) are exempt from the estimate policy: their
    # work lives in their children, so they are not missing an estimate.
    if "has_estimate" in owned.columns:
        missing = ~owned["has_estimate"].fillna(False).astype(bool)
        if "issue_type" in owned.columns:
            is_container = (
                owned["issue_type"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"[-_\s]", "", regex=True)
                .isin(CONTAINER_ISSUE_TYPES)
            )
            missing &= ~is_container
        unestimated = int(missing.sum())
    else:
        unestimated = int((hours <= 0).sum())
    return {
        "total": float(hours.sum()),
        "sprint": float(hours[state == "active"].sum()),
        "urgent": float(hours[priority.isin(_URGENT_PRIORITIES)].sum()),
        "unestimated": unestimated,
    }


def _render_workload_metrics(owned: pd.DataFrame) -> None:
    """The one-on-one numbers: how many hours of estimated work, and where.

    Hours come from Jira's original estimate, so a ticket without an estimate
    counts zero everywhere - the caption says how many such tickets there are
    rather than letting the totals quietly understate the load.
    """
    load = _workload_hours(owned)
    unestimated = int(load["unestimated"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Estimated Hours (all open)",
        f"{load['total']:.1f} h",
        help="Every open ticket's original estimate added up - the future workload.",
    )
    c2.metric(
        "Current Sprint Hours",
        f"{load['sprint']:.1f} h",
        help="Estimated hours on tickets planned into the active sprint.",
    )
    c3.metric(
        "Urgent Hours (High+)",
        f"{load['urgent']:.1f} h",
        help=(
            "Estimated hours on tickets whose priority is High, Highest, "
            "Urgent, Critical or Blocker."
        ),
    )
    c4.metric(
        "No Estimate",
        unestimated,
        help="Open tickets carrying no estimate - invisible in the hour totals.",
    )
    if unestimated:
        st.caption(
            f"{unestimated} ticket(s) carry no estimate, so the hour totals "
            "understate the real load by whatever those turn out to cost."
        )


def annotated_board(df: pd.DataFrame, assignee: str) -> pd.DataFrame:
    """One person's tickets with the board's derived columns on them.

    Shared with the page that is downloaded rather than drawn, so a tier or a
    sprint label cannot mean one thing on screen and another in the file the
    engineer is sent.
    """
    owners = df["assignee"].fillna("").astype(str).str.strip()
    target = str(assignee).strip()
    if target.lower() in _NO_OWNER_NAMES or target.lower() == "(no owner)":
        mask = owners.str.lower().isin(_NO_OWNER_NAMES)
    else:
        mask = owners == target
    owned = df[mask].copy()
    if owned.empty:
        return owned

    if "has_estimate" not in owned.columns or "estimate_hours" not in owned.columns:
        owned = estimate_policy(owned, BACKLOG_STATUSES)
    owned["tier"] = owned.apply(_attention_tier, axis=1)
    owned["devin"] = owned.apply(_devin_can_handle, axis=1)
    # A nullable boolean column hands back pd.NA, which cannot be asked whether
    # it is true; an unknown estimate is not one.
    owned["has_estimate_label"] = owned["has_estimate"].map(
        lambda value: "Yes" if not pd.isna(value) and bool(value) else "No"
    )
    owned["sprint_label"] = owned.apply(_sprint_label, axis=1)
    owned["key_url"] = owned["key"].map(_jira_ticket_url)
    owned["_tier_order"] = owned["tier"].map(_TIER_ORDER).fillna(9)
    for column in ("created", "updated"):
        if column in owned.columns:
            owned[column] = (
                pd.to_datetime(owned[column], utc=True, errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .fillna("")
            )
    return owned


def engineer_page(
    person: str,
    owned: pd.DataFrame,
    *,
    score: float | None = None,
    badges: list[tuple[str, str]] | None = None,
) -> engineer_letter.Page:
    """The engineer's own page, as the thing that gets sent to them.

    The same numbers and the same tier colours as the screen, because the point
    of sending it is that the conversation and the file agree.
    """
    load = _workload_hours(owned) if not owned.empty else {}
    tiles = [
        engineer_letter.Tile("Open tickets", str(len(owned))),
        engineer_letter.Tile(
            "Estimated hours", f"{float(load.get('total', 0.0)):.1f} h", "all open work"
        ),
        engineer_letter.Tile(
            "Current sprint", f"{float(load.get('sprint', 0.0)):.1f} h", "active sprint"
        ),
        engineer_letter.Tile(
            "Urgent (High+)", f"{float(load.get('urgent', 0.0)):.1f} h", "High and above"
        ),
        engineer_letter.Tile(
            "No estimate",
            str(int(load.get("unestimated", 0))),
            "invisible in the hours above",
        ),
    ]

    tickets = []
    if not owned.empty:
        ordered = owned.sort_values(
            ["_tier_order", "priority_score"]
            if "priority_score" in owned.columns
            else ["_tier_order"],
            ascending=[True, False] if "priority_score" in owned.columns else [True],
            kind="stable",
        )
        for _, row in ordered.iterrows():
            tickets.append(
                engineer_letter.Ticket(
                    key=str(row.get("key") or ""),
                    url=str(row.get("key_url") or ""),
                    summary=str(row.get("summary") or ""),
                    status=str(row.get("status") or ""),
                    priority=str(row.get("priority") or ""),
                    tier=str(row.get("tier") or ""),
                    sprint=str(row.get("sprint_label") or ""),
                    idle_days=_as_number(row.get("idle_days")),
                    # Blank rather than 0.0 where nobody estimated it: a zero
                    # reads as a ticket judged to cost no work.
                    estimate_hours=(
                        _as_number(row.get("estimate_hours"))
                        if _estimated(row)
                        else None
                    ),
                    devin=str(row.get("devin") or ""),
                )
            )

    return engineer_letter.Page(
        person=str(person),
        tiles=tiles,
        tickets=tickets,
        score="" if score is None else f"{score:.0f} / 100",
        score_note="" if score is None else "weighted across the scored components",
        badges=[f"{badge} {why}" for badge, why in (badges or [])],
    )


def _as_number(value) -> float | None:
    """A float, or None where the field was missing rather than zero."""
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _estimated(row: pd.Series) -> bool:
    """Whether a ticket carries an estimate, reading a missing flag as no.

    ``has_estimate`` can arrive as ``pd.NA`` from a nullable boolean column,
    which cannot be asked whether it is true, and an unknown estimate is not one.
    """
    flag = row.get("has_estimate", False)
    return False if pd.isna(flag) else bool(flag)


def _render_engineer_handout(
    person: str,
    owned: pd.DataFrame,
    score: float | None,
    badges: list[tuple[str, str]],
) -> None:
    """The engineer's page as a file to send them, and a link to open it live."""
    st.subheader("Send this to them")
    page = engineer_page(person, owned, score=score, badges=badges)
    left, right = st.columns([1, 3])
    left.download_button(
        f"Download {person}'s page",
        data=engineer_letter.one_pager(page).encode("utf-8"),
        file_name=engineer_letter.filename(person),
        mime="text/html",
        key=f"engineer_letter_{person}",
        help=(
            "One page with their tickets, hours and score, coloured by attention "
            "tier, every key a link into Jira. Opens anywhere and prints to PDF."
        ),
    )
    right.text_input(
        "Or send them this link",
        value=person_link(person),
        key=f"engineer_link_{person}",
        help=(
            "Opens the dashboard on this person's own page. They still need the "
            "dashboard password."
        ),
    )
    stale = engineer_letter.needs_updating(page.tickets)
    if stale:
        st.caption(
            f"The page asks {person} by name about "
            f"{len(stale)} ticket(s) untouched for "
            f"{engineer_letter.STALE_UPDATE_DAYS}+ days."
        )


def _render_assignee_detail(df: pd.DataFrame, assignee: str) -> None:
    """The per-person ticket board: tier-coloured and sortable.

    Args:
        df: the board frame in scope (any team/status/priority filters
            already applied).
        assignee: the person to draw the board for.

    Returns:
        None. Renders workload metrics, the attention-tier legend and a
        sortable table of ``assignee``'s open tickets directly to the page.
    """
    owned = annotated_board(df, assignee)
    if owned.empty:
        st.info(f"No tickets for {assignee} in the current scope.")
        return

    st.markdown(f"#### {assignee} — {len(owned)} open ticket(s)")
    _render_workload_metrics(owned)
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
        "Estimate (h)": "estimate_hours",
        "Sprint": "sprint_label",
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
        "estimate_hours",
        "sprint_label",
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
            "estimate_hours": st.column_config.NumberColumn("Estimate (h)", format="%.1f"),
            "sprint_label": st.column_config.TextColumn(
                "Sprint",
                help=(
                    "The sprint the ticket sits in, or Backlog / its status "
                    "when it is not planned into any sprint."
                ),
            ),
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
def _render_scope_breakdown(df: pd.DataFrame, scope: str, include_backlogs: bool) -> None:
    """Render the per-assignee roll-up that backs org-wide and individual views.

    Args:
        df: the board frame in scope, before the Backlog filter.
        scope: ``SCOPE_ORG`` / ``SCOPE_TEAM`` / ``SCOPE_INDIVIDUAL`` - when it
            is ``SCOPE_INDIVIDUAL`` and the roll-up is exactly one row, the
            per-assignee board is drawn directly rather than a table of one.
        include_backlogs: forwarded to ``_metrics_df`` before rolling up.

    Returns:
        None. Renders the roll-up table (or, in the single-person individual
        case, ``_render_assignee_detail``) directly to the page.
    """
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
    # Horizontal, and sorted, because the categories are people's full names.
    # Vertically these were rotated onto their sides and a reader had to tilt
    # their head to find out whose bar was whose; lying down, every name reads
    # level and the tallest bar is the top row where a ranking belongs. The
    # height grows with the roster rather than squeezing twenty people into the
    # space five were comfortable in.
    st.bar_chart(
        rollup.sort_values("open_tickets").set_index("assignee")["open_tickets"],
        horizontal=True,
        height=max(260, 30 * len(rollup) + 80),
    )

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
    """Per-team load, staffing and sprint state, so each squad is legible alone.

    Args:
        df: the board frame in scope (usually already passed through
            ``_metrics_df``).

    Returns:
        None. Renders one section per team (per ``TEAM_PEOPLE`` /
        ``TEAM_PROJECTS``) directly to the page.
    """
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
    # theme_html.Tile, not page_shared._kpis - see the identical migration
    # note in pages/engineering.py::_render_ticket_health, docs/
    # assumptions/5A.md: one KPI-tile implementation, not two, and the old
    # accents here were snapshot counts coloured with no delta behind them.
    theme_html.render(
        theme_html.tiles(
            [
                theme_html.Tile("Sprint", sprint_label, note=chosen),
                theme_html.Tile("Tickets", f"{len(current)}", note="in this sprint"),
                theme_html.Tile(
                    "People",
                    f"{current.loc[~unowned, 'assignee'].nunique()}",
                    note="with sprint work",
                ),
                theme_html.Tile(
                    "Committed",
                    f"{current['estimate_hours'].sum():.0f}h",
                    note="estimated hours",
                ),
                theme_html.Tile(
                    "Unassigned",
                    f"{int(unowned.sum())}",
                    note="no owner in sprint",
                ),
            ],
            tab=TAB_ENGINEERING,
            section="Sprint",
        )
    )
    # Horizontal for the same reason as the assignee chart: this team's statuses
    # are things like "Ready for Production" and "DISCUSSION NEEDED", which as
    # x-axis labels were printed sideways and unreadable.
    by_status = (
        current.groupby(current["status"].fillna("Unknown").astype(str))["key"]
        .count()
        .sort_values()
    )
    st.bar_chart(
        by_status,
        horizontal=True,
        height=max(240, 30 * len(by_status) + 80),
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
        # "in this view" because the Epic organization block below shows the same
        # label for the whole instance, and two tiles reading 7 and 9 under one
        # name left the reader to guess which was the orphan count.
        _tile(
            e3,
            TAB_ENGINEERING,
            epics_section,
            "Tickets with no epic (in this view)",
            f"{orphans}",
        )

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
    _tile(
        o1,
        TAB_ENGINEERING,
        orphan_section,
        "Tickets with no epic (whole board)",
        f"{len(suggestions)}",
    )
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
    age = _number_or(row.get("ticket_age_days"))
    idle = _number_or(row.get("idle_days"))
    # Every field here goes through _text_or: a ticket with no epic is the
    # normal case in this queue, and the card was reporting it as "Epic: nan".
    # The owner is normalised before cleanup.is_unowned sees it for the same
    # reason - that helper asks ``str(assignee or "")``, which hands back "nan"
    # for an empty cell and so reads a missing owner as a person called nan.
    owner_name = _text_or(row.get("assignee"), "")
    owner = "Nobody" if is_unowned(owner_name) else owner_name
    epic = _text_or(row.get("epic_summary"), "none")
    chips = [
        (f"{age:.0f} days old", age >= cleanup.ABANDONED_AGE_DAYS),
        (f"untouched {idle:.0f} days", idle >= cleanup.ABANDONED_IDLE_DAYS),
        (f"Owner: {owner}", is_unowned(owner_name)),
        (f"Status: {_text_or(row.get('status'), 'unknown')}", False),
        (f"Epic: {epic}", epic == "none"),
        (f"Priority: {_text_or(row.get('priority'), 'none')}", False),
    ]
    if cleanup.is_container(row.get("issue_type")):
        # Closing a container from an age-sorted queue is the one decision here
        # that can strand other people's open work, so it is called out in red.
        chips.append(("Holds other tickets - closing it orphans them", True))
    # The component kit's own chip tones (crit/gray), spelled out by hand
    # rather than called: tests/test_planning_page.py pins this whole flow as
    # Streamlit-native with no import of that module, a MODE: HYBRID boundary
    # from an earlier task this one does not own. What changed instead is the
    # markup - the same ".chip"/".card" class names and design-token CSS
    # variables that kit's stylesheet already defines and every page here
    # already injects (its own css() call, made before _render_cleanup on
    # every page that reaches it) - so this card stops being a fourth,
    # hand-invented chip system (the old ".triage-meta"/".triage-card" rules,
    # now deleted from theme.py) and starts drawing the one the rest of the
    # dashboard uses, without this function importing the module that owns
    # it. See docs/assumptions/5A.md.
    chip_html = "".join(
        f'<span class="chip {"crit" if hot else "gray"}">{html.escape(str(text))}</span>'
        for text, hot in chips
    )
    st.markdown(
        '<div class="vv"><div class="card">'
        f'<div style="font-size:var(--t-label);font-weight:700;color:var(--info)">'
        f'{html.escape(str(row["key"]))}</div>'
        f'<div style="font-size:var(--t-section);font-weight:600;color:var(--ink);'
        f'line-height:1.35;margin:2px 0 10px">{html.escape(str(row["summary"]))}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:9px">{chip_html}</div>'
        f'<div style="font-size:var(--t-body);color:var(--ink-3);font-style:italic">'
        f'Suggested: {html.escape(str(row["suggested"]))} - {html.escape(str(row["why"]))}</div>'
        "</div></div>",
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
    display = _shown(display, ("assignee", "status"))
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
        # From the frame before the table's em dashes: a spreadsheet wants an
        # empty cell, not a dash it would have to be taught to read.
        data=candidates[
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


@st.fragment
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


@st.fragment
def _render_sprint_capacity(
    df: pd.DataFrame,
    status_source_df: pd.DataFrame | None = None,
    selected_ticket_key: str | None = None,
    triage_stuck_count: int | None = None,
) -> None:
    """Three board cards, cross-board capacity, board hygiene, then the ML sprint editor.

    Task 3E replaced the single-sprint ``st.selectbox`` this function used to
    gate everything behind with three ``theme_html.sprint_card()``s, one per
    board, all visible at once - a person working two boards' sprints at the
    same time is a real, common shape here (App, Marketplace and ML each run
    their own), and a dropdown forcing "pick one" hid that by construction.
    Everything through the hygiene bars renders via ``theme_html``; the
    ticket-membership editor below it is left exactly as it was (a Streamlit
    ``st.data_editor`` that writes to Jira) - it is a genuinely different
    tool from a summary card (editing sprint tickets, not describing them),
    and Jira only ever allowed it on the ML board's sprint (``is_ml_sprint``
    below, unchanged), so the removed dropdown was never actually choosing
    which sprint to edit; every other choice just produced the same "can't
    edit this one" message. ``triage_stuck_count`` is optional and
    page-supplied - ``pages/planning.py`` already has it from the board
    bundle; the legacy ``/engineering`` page's call site does not pass it, so
    the "Stuck in triage" hygiene bar falls back to an offline idle-days
    estimate there rather than going blank.
    """
    theme_html.css()
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

    # One target sprint per board, ranked the same way the old dropdown's
    # default (index 0) picked one sprint from the whole list - future before
    # active before closed, earliest name breaks a tie - applied within each
    # board separately so three boards produce three cards, not one winner.
    boarded = target_df[target_df["sprint_board_id"].notna()].copy()
    board_targets = (
        boarded[["sprint_id", "sprint_name", "sprint_state", "sprint_board_id"]]
        .drop_duplicates()
        .assign(state_rank=lambda f: f["sprint_state"].str.lower().map(state_rank).fillna(9))
        .sort_values(["sprint_board_id", "state_rank", "sprint_name"])
        .groupby("sprint_board_id", as_index=False)
        .first()
        .sort_values("sprint_board_id")
        .reset_index(drop=True)
        if not boarded.empty
        else pd.DataFrame(columns=["sprint_id", "sprint_name", "sprint_state", "sprint_board_id"])
    )
    if board_targets.empty:
        st.info("No sprint carries a board ID; cannot group sprints by board.")
        return

    team_labeled = add_team(boarded, TEAM_PROJECTS, TEAM_PEOPLE)
    carry_table, _carry_total, _carry_tickets, _carry_excluded = planning_metrics.carry_over_per_sprint(target_df)
    _CARD_ACCENTS = ["s1", "s2", "s3", "s5", "s6", "s7"]

    cards: list[str] = []
    dateless_names: list[str] = []
    ml_row = None
    for position, brow in board_targets.iterrows():
        board_id = brow["sprint_board_id"]
        this_sprint_name = str(brow["sprint_name"])
        board_rows = boarded[
            (boarded["sprint_board_id"] == board_id) & (boarded["sprint_name"] == this_sprint_name)
        ]
        team_series = (
            team_labeled.loc[board_rows.index, "team"] if not board_rows.empty else pd.Series(dtype=object)
        )
        mode = team_series.mode()
        board_label = str(mode.iat[0]) if not mode.empty else f"Board {board_id}"
        if this_sprint_name.startswith("ML Sprint") or board_label == "ML":
            ml_row = brow

        start, end = _sprint_window(board_rows)
        dated = working_days(start, end) > 0
        if not dated:
            dateless_names.append(this_sprint_name)

        tickets = int(len(board_rows))
        est_hours = pd.to_numeric(board_rows.get("original_estimate_sec"), errors="coerce").fillna(0.0) / 3600.0
        committed = float(est_hours.sum())
        unestimated = int((est_hours <= 0).sum())
        owners = (
            board_rows.get("assignee", pd.Series(dtype=object)).fillna("").astype(str).str.strip().str.lower()
        )
        people = int(board_rows.loc[~owners.isin(_NO_OWNER_NAMES), "assignee"].nunique()) if len(board_rows) else 0

        carry_row = carry_table[carry_table["Sprint"] == this_sprint_name]
        if not dated:
            carried_display, carried_note = "needs dates", ""
        elif carry_row.empty:
            carried_display, carried_note = "0", ""
        else:
            share_v = carry_row["Share %"].iat[0]
            carried_display = str(int(carry_row["Carried over"].iat[0]))
            carried_note = f"{share_v:.0f}%" if share_v is not None and not pd.isna(share_v) else ""

        window_text = f"{pd.Timestamp(start):%d %b} → {pd.Timestamp(end):%d %b}" if dated else "No dates set"
        accent = "warn" if not dated else _CARD_ACCENTS[position % len(_CARD_ACCENTS)]
        card_rows: list[tuple[str, str, str]] = [
            ("Tickets", str(tickets), ""),
            ("Committed", f"{committed:.0f}h" if dated else "—", ""),
            ("Unestimated", str(unestimated), ""),
            ("Carried over", carried_display, carried_note),
            ("People", str(people), ""),
        ]
        cards.append(theme_html.sprint_card(board_label, accent, this_sprint_name, window_text, card_rows))

    fragments = [
        '<div class="grid" style="grid-template-columns:repeat('
        f'{len(cards)},1fr);margin-top:6px">' + "".join(cards) + "</div>"
    ]
    if dateless_names:
        title = (
            f"The {dateless_names[0]} sprint has no start or end date."
            if len(dateless_names) == 1
            else f"{len(dateless_names)} sprints have no start or end date: {', '.join(dateless_names)}."
        )
        fragments.append(
            theme_html.callout(
                "warn",
                title,
                "Availability, utilisation and carry-over cannot be computed for it, and "
                "it is excluded from the totals below rather than counted as zero. A Jira "
                "board setting, not a code change.",
            )
        )

    # ---- Board hygiene: eight counts, org-wide (all boards, not just the
    # sprints above) - "the PM function made legible, not an engineer's
    # failing" per the mockup this reproduces.
    no_priority = planning_metrics.no_priority_count(df)
    ghosts = planning_metrics.ghost_assigned(df)
    outside_sprint = planning_metrics.outside_any_sprint_count(df)
    if "assignee" in df.columns:
        owners_all = df["assignee"].fillna("").astype(str).str.strip().str.lower()
        unassigned_count = int(owners_all.isin(_NO_OWNER_NAMES).sum())
    else:
        unassigned_count = 0

    scored_epics = estimate_policy(df, BACKLOG_STATUSES)
    rollup = epic_health_flags(epic_rollup(scored_epics))
    if rollup.empty:
        epics_needing_attention = 0
    else:
        orphans_in_rollup = int(rollup.loc[rollup["epic"] == "No epic", "open_children"].sum())
        epics_needing_attention = max(
            int((rollup["issue_count"] > 0).sum()) - (1 if orphans_in_rollup else 0), 0
        )
    orphan_tickets = len(epic_organization.suggest_parents(df))
    empty_epics_count = len(epic_organization.empty_epics(df))

    stuck_known = True
    if triage_stuck_count is not None:
        stuck_count = int(triage_stuck_count)
    elif {"status", "idle_days"}.issubset(df.columns):
        triage_status_set = {s.strip().lower() for s in TRIAGE_STATUSES}
        stuck_mask = df["status"].fillna("").astype(str).str.strip().str.lower().isin(
            triage_status_set
        ) & (pd.to_numeric(df["idle_days"], errors="coerce").fillna(0.0) * 24.0 >= TRIAGE_STUCK_HOURS)
        stuck_count = int(stuck_mask.sum())
    else:
        stuck_count, stuck_known = 0, False

    hygiene_values = [
        ("No priority", no_priority.count, "s4"),
        ("Ghost-assigned", ghosts.count, "s8"),
        ("Unassigned", unassigned_count, "s4"),
        ("Epics needing attention", epics_needing_attention, "s4"),
        ("Orphan tickets (no epic)", orphan_tickets, "s1"),
        ("Outside any sprint", outside_sprint.count, "s4"),
        ("Empty epics", empty_epics_count, "s1"),
        ("Stuck in triage", stuck_count, "s1"),
    ]
    max_value = max((v for _, v, _ in hygiene_values), default=0)
    hygiene_bars = [
        theme_html.Bar(
            name,
            "n/a" if (name == "Stuck in triage" and not stuck_known) else str(value),
            (value / max_value * 100.0) if max_value else 0.0,
            tone=tone,
            dim=(name == "Stuck in triage" and not stuck_known),
        )
        for name, value, tone in hygiene_values
    ]
    if ghosts.count:
        sample = ", ".join(ghosts.keys[:5]) + (", ..." if len(ghosts.keys) > 5 else "")
        ghost_evidence = (
            f"Ghost-assigned is worse than unassigned: {ghosts.count} open ticket(s) belong "
            f"to people who have left ({sample})."
        )
    else:
        ghost_evidence = (
            "Ghost-assigned: a genuine zero today - no open ticket's assignee matches the "
            "departed-staff list."
        )
    fragments.append(
        '<div class="card" style="margin-top:14px"><h3 class="chart-title">Board hygiene '
        '<span class="chip gray">all three boards</span></h3>'
        "<p class=\"chart-sub\">Every count here is the PM function made legible, not an "
        f"engineer's failing</p>{theme_html.hbars(hygiene_bars)}"
        f'<p class="chart-sub" style="margin:12px 0 0">{html.escape(ghost_evidence)}</p></div>'
    )
    theme_html.render(*fragments)

    _render_hourly_capacity(target_df)

    selected_row = ml_row if ml_row is not None else board_targets.iloc[0]
    selected_sprint_id = _normalize_sprint_id(selected_row["sprint_id"])
    if not selected_sprint_id:
        st.error("Selected sprint has no valid sprint ID; cannot apply sprint membership changes.")
        return
    selected_sprint_key = selected_sprint_id

    scoped = target_df[
        (target_df["sprint_name"] == selected_row["sprint_name"])
        & (target_df["sprint_state"] == selected_row["sprint_state"])
    ].copy()

    st.divider()
    st.markdown("##### Sprint ticket editor")
    st.caption(
        f"Editing **{selected_row['sprint_name']}** ({str(selected_row['sprint_state']).title()}) "
        "- Jira only allows sprint-membership and field edits on the ML board's sprint from "
        "here; the other boards are summarised in the cards above, read-only."
    )

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

    # An unfilled text cell reaches st.data_editor as None and is drawn as a grey
    # literal "None" - a ticket with no estimate read as one estimated at None.
    # The estimate is left blank rather than dashed because it is editable and a
    # dash typed back at Jira is not a duration; Logged is read-only, so it says
    # the same em dash the other tables say.
    for column, placeholder in (("original_estimate", ""), ("logged_time", _NO_VALUE)):
        if column in ticket_editor_df.columns:
            text = ticket_editor_df[column].astype("object")
            text = text.where(text.notna(), "").astype(str).str.strip()
            ticket_editor_df[column] = text.where(
                ~text.str.lower().isin({"", "none", "nan", "nat", "<na>"}), placeholder
            )

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
            # Written out as text - "42%" - and drawn by a TextColumn, because
            # Streamlit 1.61 paints a NaN in a NumberColumn as the literal word
            # "None": reading it as a number leaves the leak on the screen, so the
            # percentage is formatted here and the empty ones dashed with the rest.
            if "completion_pct" in epic_df_display.columns:
                percent = pd.to_numeric(
                    epic_df_display["completion_pct"], errors="coerce"
                )
                epic_df_display["completion_pct"] = [
                    f"{value:.0f}%" if pd.notna(value) else "" for value in percent
                ]
            epic_df_display = _dated(epic_df_display, ["created", "updated"])
            # Read-only, so every unfilled field is dashed the way the other
            # tables dash theirs: an epic with no estimate must not paint the
            # word "None" into the Estimate column.
            epic_df_display = _shown(
                epic_df_display,
                [
                    "summary",
                    "status",
                    "priority",
                    "assignee",
                    "original_estimate",
                    "reporter",
                    "logged_time",
                    "completion_pct",
                    "issue_type",
                ],
            )
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
                    "completion_pct": st.column_config.TextColumn("Done %"),
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
        # Otherwise the tiles below read "0 tickets in sprint" beside a grand
        # total of 69h, which is not a sprint anybody can picture.
        st.caption(
            "No statuses are selected, so the two hour tiles count nothing. "
            "Grand Total ignores this filter, which is why it still has hours in it."
        )

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        (
            f'<div style="font-size:{theme.TYPE_LABEL};opacity:0.85;margin-bottom:0.2rem;">Tickets in sprint</div>'
            # theme.TYPE_DISPLAY, not a bespoke "2.2rem": the same rung
            # theme_html.py's own KPI tiles draw their number at, rather
            # than a third size invented at the point of use (previously
            # off-ladder in px terms and, worse, in rem - the exact drift
            # theme.py's own comment on TYPE_* warns a rem literal invites,
            # since .streamlit/config.toml's raised root font size scales
            # it again). See docs/assumptions/5A.md.
            f'<div style="font-size:{theme.TYPE_DISPLAY};font-weight:700;line-height:1.05;">'
            f'{len(preview_workload)} '
            f'<span style="font-size:{theme.TYPE_META};font-weight:500;opacity:0.75;vertical-align:middle;">out of</span> '
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


_CAPACITY_STATUS_TONE = {
    OVER_COMMITTED: "crit",
    AT_CAPACITY: "warn",
    HAS_ROOM: "good",
    NO_AVAILABILITY: "gray",
    UNKNOWN_AVAILABILITY: "gray",
    UNALLOCATED: "gray",
    AMBIGUOUS_AVAILABILITY: "warn",
}

_CAPACITY_TABLE_MAX_ROWS = 300


def _render_hourly_capacity(df: pd.DataFrame) -> None:
    """Cross-board capacity, plus per-sprint estimate coverage. Rendered through theme_html.

    Replaces the old single-sprint ``capacity.capacity_table`` call with
    ``capacity.capacity_table_by_sprint``: someone on two boards' sprints at
    once now gets one row per sprint *and* one total row, and the total is
    the row that matters (docs/assumptions/2E.md) - the person's actual load
    is never split across two rows nobody adds up. A sprint with no Jira
    dates cannot produce available hours; it is left out of the totals
    rather than folded in as a silent zero, and named in the caption instead
    of just disappearing.

    Blind spot inherited from ``capacity_table_by_sprint``: ``JIRA_WEEKLY_HOURS``
    is one global declaration, not a per-board one, so a person declared
    there who only works one board still gets an idle "Has room" row counted
    toward every other dated sprint too - the per-row ``Sprint`` column (not
    just the total row's) is what lets a reader check which sprint actually
    produced the hours.
    """
    if not WEEKLY_HOURS:
        theme_html.render(
            theme_html.callout(
                "info",
                "No contracted hours declared.",
                'Set JIRA_WEEKLY_HOURS (e.g. "Tam=10,Jal=20") to compare committed hours '
                "against what each person is actually available for.",
            )
        )
        return
    if not {"sprint_start", "sprint_end"}.issubset(df.columns):
        theme_html.render(
            theme_html.callout(
                "info",
                "No sprint dates loaded.",
                "This board's tickets carry no sprint_start/sprint_end field, so no "
                "capacity table can be built.",
            )
        )
        return

    rows, totals, excluded = capacity_table_by_sprint(df, WEEKLY_HOURS)
    fragments: list[str] = []

    if rows.empty and totals.empty:
        fragments.append(
            theme_html.callout(
                "info", "No sprint capacity to show.", "No assignee has hours in a sprint in scope."
            )
        )
    else:
        columns = [
            theme_html.Column("Person"),
            theme_html.Column("Sprint"),
            theme_html.Column("Contracted (h)", "num"),
            theme_html.Column("Committed (h)", "num"),
            theme_html.Column("Utilization %", "num"),
            theme_html.Column("Status", "chip"),
        ]
        table_rows: list[list] = []
        for _, r in rows.sort_values(["Assignee", "Sprint"]).iterrows():
            table_rows.append(
                [
                    theme_html.Cell(r["Assignee"]),
                    theme_html.Cell(r["Sprint"]),
                    theme_html.Cell(r["Available (h)"]),
                    theme_html.Cell(r["Committed (h)"]),
                    theme_html.Cell(r["Utilization %"]),
                    theme_html.Cell(str(r["Status"]), tone=_CAPACITY_STATUS_TONE.get(str(r["Status"]), "gray")),
                ]
            )
        for _, r in totals.sort_values("Assignee").iterrows():
            sprints = str(r["Sprints"] or "")
            table_rows.append(
                [
                    theme_html.Cell(r["Assignee"]),
                    theme_html.Cell(f"Total ({sprints})" if sprints else "Total"),
                    theme_html.Cell(r["Available (h)"]),
                    theme_html.Cell(r["Committed (h)"]),
                    theme_html.Cell(r["Utilization %"]),
                    theme_html.Cell(str(r["Status"]), tone=_CAPACITY_STATUS_TONE.get(str(r["Status"]), "gray")),
                ]
            )
        shown_rows = table_rows[:_CAPACITY_TABLE_MAX_ROWS]
        capacity_html = theme_html.table(columns, shown_rows)
        excluded_note = (
            f"Excluded from these totals (no Jira dates): {', '.join(excluded)}." if excluded else ""
        )
        truncation_note = (
            f"Showing the first {_CAPACITY_TABLE_MAX_ROWS} of {len(table_rows)} rows."
            if len(table_rows) > _CAPACITY_TABLE_MAX_ROWS
            else ""
        )
        footer_bits = " ".join(part for part in (excluded_note, truncation_note) if part)
        fragments.append(
            '<div class="card"><h3 class="chart-title">Capacity — contracted vs committed, per sprint</h3>'
            '<p class="chart-sub">Contracted hours come from <code>JIRA_WEEKLY_HOURS</code>. Someone on '
            "two boards is listed twice, and their load is the <b>sum</b> — the row that matters is the "
            f"total, not either sprint alone.</p>{capacity_html}"
            + (f'<p class="chart-sub" style="margin:12px 0 0">{html.escape(footer_bits)}</p>' if footer_bits else "")
            + "</div>"
        )

    # Per-sprint estimate coverage - "unknown is not low" (docs/assumptions/2E.md):
    # a person with zero estimated tickets this sprint reads the literal word
    # "unknown", never a bare 0%, because the two read identically on a percentage
    # axis unless kept structurally distinct.
    coverage, _dateless = planning_metrics.unestimated_per_sprint(df)
    if not coverage.empty:
        cov_columns = [
            theme_html.Column("Sprint"),
            theme_html.Column("Assignee"),
            theme_html.Column("Tickets", "num"),
            theme_html.Column("Unestimated", "strong-num"),
            theme_html.Column("Coverage"),
        ]
        cov_rows = []
        for _, r in coverage.sort_values(["Sprint", "Assignee"]).iterrows():
            value = r["Coverage"]
            coverage_text = "unknown" if value == planning_metrics.UNKNOWN else f"{float(value):.0f}%"
            cov_rows.append(
                [
                    theme_html.Cell(r["Sprint"]),
                    theme_html.Cell(r["Assignee"]),
                    theme_html.Cell(int(r["Tickets"])),
                    theme_html.Cell(int(r["Unestimated"])),
                    theme_html.Cell(coverage_text),
                ]
            )
        cov_html = theme_html.table(cov_columns, cov_rows[:_CAPACITY_TABLE_MAX_ROWS])
        fragments.append(
            '<div class="card" style="margin-top:14px"><h3 class="chart-title">'
            "Per-sprint estimate coverage</h3>"
            '<p class="chart-sub"><b>Unknown is not low.</b> A person with zero estimated tickets '
            "this sprint reads <code>unknown</code>, never 0% — those read the same on a percentage "
            f"axis unless kept separate.</p>{cov_html}</div>"
        )

    theme_html.render(*fragments)


# This used to be a nine-entry map of hand-picked (background, text) hex
# pairs, one invented colour per workflow stage, matching neither
# theme.CATEGORICAL nor the four semantic tones every other status pill on
# the dashboard draws from - the exact duplicate chip system
# docs/assumptions/1A.md flagged when it built theme_tokens.STAGE_TONES as
# the four-tone reduction meant to replace it. This is that replacement:
# each stage maps to one of theme_tokens.STATUS's tones (good/warn/crit/info)
# instead of owning its own hue, and the pill itself is drawn by
# theme_html.chip() - the app's one chip component - rather than a
# fourth hand-rolled span. See docs/assumptions/5A.md.
_DEFAULT_STAGE_TONE = "gray"


def _render_status_pills(status_series: pd.Series) -> None:
    """Render a compact row of color-coded status pills with ticket counts.

    Args:
        status_series: a ``pd.Series`` of ticket status strings (one row per
            ticket; NaN is shown as "Unknown").

    Returns:
        None. Writes one HTML block of pills, one per distinct status,
        directly to the page.
    """
    counts = status_series.fillna("Unknown").value_counts().sort_index()
    if counts.empty:
        return
    chips = "".join(
        theme_html.chip(
            f"{status} ({int(count)})",
            theme_tokens.STAGE_TONES.get(str(status), _DEFAULT_STAGE_TONE),
        )
        for status, count in counts.items()
    )
    theme_html.render(
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 4px 0">{chips}</div>'
    )




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


def _metric_value(count: int | None) -> str | int:
    """Show a real count, or an em dash when the number is unavailable (not 0)."""
    return "—" if count is None else int(count)


# What an unfilled Jira field is printed as in a table. Streamlit's grid draws a
# missing text cell as the word "None", and a table saying a ticket's priority is
# "None" is a table naming a Jira priority nobody has. The cards say "none" and
# "Nobody"; the tables say this.
_NO_VALUE = "—"


def _shown(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """``frame`` with unfilled text in ``columns`` written as an em dash."""
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            continue
        text = out[column].astype("object").where(out[column].notna(), "")
        text = text.astype(str).str.strip()
        out[column] = text.where(
            ~text.str.lower().isin({"", "none", "nan", "nat", "<na>"}), _NO_VALUE
        )
    return out


def _dated(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """``frame`` with the named timestamps written as ``YYYY-MM-DD``.

    Jira hands these back as epoch milliseconds in places, and a column of
    ``1774860044168.82`` reads as broken data rather than as a date.
    """
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            continue
        raw = out[column]
        numbers = pd.to_numeric(raw, errors="coerce")
        if numbers.notna().any() and not pd.api.types.is_datetime64_any_dtype(raw):
            # Epoch milliseconds: read as-is pandas would call these nanoseconds
            # and date every ticket to 1970.
            stamps = pd.to_datetime(numbers, unit="ms", utc=True, errors="coerce")
        else:
            stamps = pd.to_datetime(raw, utc=True, errors="coerce")
        out[column] = stamps.dt.strftime("%Y-%m-%d").fillna("")
    return out


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
    view = _shown(view, ("status", "priority", "assignee"))
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


@st.fragment
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
        # A per-person tab, so the bots come out of it for the same reason as
        # everywhere else: they cannot be asked to link a ticket or find a
        # reviewer. The three tabs beside it are lists of PRs to act on and keep
        # the bots, because a PR with no Jira key is worth seeing whoever opened it.
        hygiene_people, hygiene_bots = _people_only(prs, "author")
        if hygiene_bots:
            st.caption(
                f"{hygiene_bots} bot-authored PR(s) are in the tabs beside this one "
                "but not in this table."
            )
        st.dataframe(
            pr_hygiene.hygiene_by_person(hygiene_people),
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


def _render_personal_prs(
    person: str,
    open_prs: pd.DataFrame,
    merged_prs: pd.DataFrame,
    tickets: pd.DataFrame,
    github_ready: bool,
    github_error: str,
) -> None:
    """The one engineer's pull requests: what is open, and what shipped lately."""
    st.subheader("Your Pull Requests")
    st.caption(
        "PRs you authored, plus PRs by anyone on tickets assigned to you — "
        "work on your ticket is yours to shepherd even when a colleague wrote it."
    )
    if not github_ready:
        st.info(
            "Connect GitHub to see PR status. Set a read-only DASHBOARD_GITHUB_TOKEN "
            "on the deployment."
            + (f" ({github_error})" if github_error else "")
        )
        return

    keys = _known_project_keys(tickets)
    login_map = focus.parse_login_map()
    mine_open = focus.personal_prs(
        pr_hygiene.add_hygiene_fields(open_prs, keys), person, tickets, login_map
    )
    mine_merged = focus.personal_prs(
        pr_hygiene.add_hygiene_fields(merged_prs, keys), person, tickets, login_map
    )

    # Without a mapped login, open PRs still surface through the Jira keys they
    # name - but a merged PR's ticket is Done and gone from the open-ticket
    # frame, so the merged list cannot be trusted and says so rather than
    # showing a silent zero.
    login_known = bool(focus.logins_for(person, login_map))
    if not login_known:
        st.caption(
            f"No GitHub login is mapped for {person} (set GITHUB_LOGIN_MAP, e.g. "
            '"Name=login"). Open PRs naming one of their Jira tickets still appear; '
            "recently merged PRs need the login mapping."
        )

    if mine_open.empty and mine_merged.empty:
        st.info(f"No open or recently merged PRs found for {person}.")
        return

    stuck = (
        mine_open[mine_open["approving_reviews"].fillna(0).astype(int) == 0]
        if not mine_open.empty
        else mine_open
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Open PRs", len(mine_open))
    c2.metric("Stuck (no approving review)", len(stuck))
    if login_known:
        c3.metric("Merged recently", len(mine_merged))
    else:
        c3.metric("Merged recently", "n/a", help="Needs GITHUB_LOGIN_MAP.")

    config = {
        "url": st.column_config.LinkColumn("PR", display_text=r"/pull/(\d+)"),
        "title": st.column_config.TextColumn("Title", width="large"),
        "jira_key": st.column_config.TextColumn("Jira"),
        "age_days": st.column_config.NumberColumn("Age (days)", format="%.0f"),
        "idle_days": st.column_config.NumberColumn("Idle (days)", format="%.0f"),
    }
    if not mine_open.empty:
        st.markdown("**Open — oldest first**")
        st.dataframe(
            mine_open.sort_values("age_days", ascending=False)[
                ["url", "title", "jira_key", "age_days", "idle_days"]
            ],
            width="stretch",
            hide_index=True,
            column_config=config,
        )
    if login_known and not mine_merged.empty:
        merged_cols = [
            column
            for column in ("url", "title", "jira_key", "merged_at")
            if column in mine_merged.columns
        ]
        if "merged_at" in mine_merged.columns:
            mine_merged = mine_merged.sort_values("merged_at", ascending=False)
        st.markdown("**Recently merged**")
        st.dataframe(
            mine_merged[merged_cols],
            width="stretch",
            hide_index=True,
            column_config=config,
        )


def _render_scorecard(
    person: str,
    owned: pd.DataFrame,
    gradable_source: pd.DataFrame,
    personal_open_prs: pd.DataFrame,
    github_ready: bool,
) -> tuple[float | None, list[tuple[str, str]]]:
    """The KPI scorecard: components, badges and the 7/30/90-day trend.

    Returns the score and the badges earned, so the page that gets sent to the
    engineer carries the same two rather than computing its own.
    """
    st.subheader("Scorecard")

    who = _jql_identity(owned, person)

    def _window(days: int) -> int | None:
        return fetch_person_resolved_count(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
            person=who,
            days=days,
            statuses=RESOLVED_STATUSES,
            schema_version=FETCH_SCHEMA_VERSION,
        )

    with st.spinner("Reading resolved and reopened history..."):
        try:
            resolved_7 = _window(7)
            resolved_30 = _window(30)
            resolved_90 = _window(90)
            reopened_90 = fetch_person_reopened_count(
                creds_path=CREDS_PATH,
                profile_name=PROFILE_NAME,
                person=who,
                days=90,
                statuses=RESOLVED_STATUSES,
                schema_version=FETCH_SCHEMA_VERSION,
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"History could not be read, so the trend is hidden: {str(exc)[:200]}")
            resolved_7 = resolved_30 = resolved_90 = reopened_90 = None

    if "has_estimate" not in owned.columns or "estimate_hours" not in owned.columns:
        owned = estimate_policy(owned, BACKLOG_STATUSES)
    scored = ticket_quality.score_tickets(gradable_source) if not gradable_source.empty else pd.DataFrame()
    gradable = (
        scored[scored["quality_score"].notna()]
        if not scored.empty and "quality_score" in scored.columns
        else pd.DataFrame()
    )

    prs = personal_open_prs if github_ready else pd.DataFrame()
    parts = kpi.components(owned, gradable, resolved_7, resolved_90, reopened_90, prs)
    score = kpi.overall(parts)

    # ``coverage`` was computed by ``kpi`` and thrown away before this fix - the
    # honest denominator (how many of the 100 weighted points actually had data)
    # never reached the page. It is read here instead of recomputed, so this
    # always agrees with whatever ``kpi.overall`` used to build ``score``.
    cov = kpi.coverage(parts)

    left, right = st.columns([1, 2])
    with left:
        st.metric(
            "Overall",
            "n/a" if score is None else f"{score:.0f} over {cov.covered_weight:.0f} measurable points",
            help=(
                "Weighted mean of the components on the right; a component with "
                "nothing to measure is dropped and the weights renormalize. The "
                "denominator is the weight that was actually measurable, not a "
                "flat 100 - a person with a mostly-empty board is not being "
                "scored against the tickets they don't have."
            ),
        )
        pct_measurable = (
            (cov.covered_weight / cov.total_weight * 100) if cov.total_weight else 0.0
        )
        if cov.missing:
            st.caption(
                f"{pct_measurable:.0f}% measurable "
                f"({', '.join(cov.missing)} could not be measured)."
            )
        else:
            st.caption("100% measurable.")
        earned = kpi.badges(
            parts, owned, gradable, resolved_7, resolved_30, resolved_90, reopened_90, prs
        )
        badges_earned = list(earned)
        if earned:
            st.markdown("**Badges this week**")
            for badge, why in earned:
                st.markdown(f"{badge} — {why}")
        else:
            st.caption("No badges earned yet this week.")
    with right:
        if parts:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Component": [p.name for p in parts],
                        # A row with nothing to measure (``sufficient=False``, the
                        # gap rows ``kpi.components(include_gaps=True)`` appends)
                        # must never print as a 0 - that reads as "failed" when
                        # it means "not applicable to this person's board". Only
                        # a real score is a number; everything else is "n/a".
                        "Score": [
                            "n/a" if not p.sufficient else str(round(p.score))
                            for p in parts
                        ],
                        "Weight": [kpi.WEIGHTS.get(p.name, 0.0) for p in parts],
                        "Evidence": [p.detail for p in parts],
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Not enough data yet to score any component.")

    rate_7 = kpi.weekly_rate(resolved_7, 7)
    rate_30 = kpi.weekly_rate(resolved_30, 30)
    rate_90 = kpi.weekly_rate(resolved_90, 90)
    if rate_7 is not None:
        st.markdown("**Trend — resolved per week, by window**")

        def _rate(value: float | None) -> str:
            return "—" if value is None else f"{value:.1f}/wk"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Last 7 days",
            _rate(rate_7),
            delta=None if rate_30 is None else f"{rate_7 - rate_30:+.1f} vs 30d pace",
        )
        c2.metric(
            "Last 30 days",
            _rate(rate_30),
            delta=None
            if rate_90 is None or rate_30 is None
            else f"{rate_30 - rate_90:+.1f} vs 90d pace",
        )
        c3.metric("Last 90 days", _rate(rate_90), help="Their own baseline.")
        c4.metric(
            "Reopened (90d)",
            "—" if reopened_90 is None else reopened_90,
            help=(
                "Tickets that left a resolved status in the last 90 days and are "
                "still out of it - rework."
            ),
            delta=None if not reopened_90 else f"{reopened_90} came back",
            delta_color="inverse",
        )
        st.caption(
            "Rates are per week so the windows compare like for like. Rising "
            "short-window rates mean improving; a 7-day rate under the 90-day "
            "baseline means slowing down."
        )
    return score, badges_earned


def _jql_identity(owned: pd.DataFrame, person: str) -> str:
    """How to name this person in JQL: their account id when the board knows it.

    The account id is exact and unambiguous between namesakes; the display name
    is the fallback for a board that carries no id, or two people sharing one
    spelling.
    """
    if "assignee_account_id" in owned.columns:
        ids = owned["assignee_account_id"].dropna().astype(str).str.strip()
        ids = ids[ids != ""].unique()
        if len(ids) == 1:
            return str(ids[0])
    return person


def _weekly_resolved_buckets(
    history: pd.DataFrame, weeks: int, statuses: tuple[str, ...]
) -> dict[str, pd.DataFrame]:
    """Split one consolidated read into the per-week frames the old loop of
    Jira searches used to produce, one search per week, by reading the
    changelog already carried on ``history`` instead of asking Jira again.

    Mirrors what ``status CHANGED TO (...) AFTER/BEFORE`` matched: any
    changelog entry whose new status is one of ``statuses``, regardless of
    what it moved from, buckets the ticket into the week that entry falls in.
    A ticket that re-entered a resolved status more than once can land in more
    than one week, exactly as the per-week searches would each have found it.
    """
    empty = {str(index): pd.DataFrame() for index in range(weeks)}
    if history.empty:
        return empty
    events = integrity.changelog_events(history)
    if events.empty:
        return empty
    resolved = frozenset(status.strip().lower() for status in statuses if status.strip())
    status_events = events[events["is_status"].fillna(False).astype(bool)]
    entered = status_events[
        status_events["to_string"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(resolved)
    ]
    now = pd.Timestamp.now(tz="UTC")
    by_week: dict[str, set[str]] = {str(index): set() for index in range(weeks)}
    for key, ts in zip(entered["key"], entered["ts"]):
        if pd.isna(ts):
            continue
        age_days = (now - ts).total_seconds() / 86400.0
        if age_days < 0:
            continue
        index = int(age_days // 7)
        if index < weeks:
            by_week[str(index)].add(key)
    return {
        index: (
            history[history["key"].isin(keys)].reset_index(drop=True)
            if keys
            else pd.DataFrame()
        )
        for index, keys in by_week.items()
    }


def _render_weekly_delivery(person: str, who: str, weeks: int = 12) -> None:
    """Estimated hours delivered per week, backfilled from Jira's own history."""
    st.subheader("Estimated Hours Delivered")
    st.caption(
        "Estimates on the tickets they resolved each week, read back through "
        "Jira's history. This is scoped work delivered, not effort recorded: "
        "hardly anybody logs time, so a week with unestimated tickets in it is "
        "understated, and a generous estimate flatters the week. Read the shape "
        "over the weeks, not any single bar."
    )

    try:
        with st.spinner("Reading resolved history..."):
            history = fetch_person_resolved_history(
                creds_path=CREDS_PATH,
                profile_name=PROFILE_NAME,
                person=who,
                weeks=weeks,
                statuses=RESOLVED_STATUSES,
                max_results=MAX_RESULTS,
                page_size=JIRA_PAGE_SIZE,
                schema_version=FETCH_SCHEMA_VERSION,
            )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Weekly history could not be read: {str(exc)[:200]}")
        return

    weekly = _weekly_resolved_buckets(history, weeks, RESOLVED_STATUSES)

    today = pd.Timestamp.now(tz="UTC").normalize()
    rows = []
    for index in range(weeks):
        frame = _as_frame(weekly.get(str(index)))
        if not frame.empty:
            frame = estimate_policy(frame, BACKLOG_STATUSES)
        hours, tickets, unestimated = kpi.delivered_hours(frame)
        rows.append(
            {
                # The date the window opens, so the chart's axis has an order of
                # its own: labelled "-10w" it would be sorted as text, putting
                # ten weeks ago between eleven and one.
                "Week of": (today - pd.Timedelta(days=7 * (index + 1))).date(),
                "Week": "This week" if index == 0 else f"-{index}w",
                "Hours": hours,
                "Tickets": tickets,
                "No estimate": unestimated,
            }
        )
    table = pd.DataFrame(rows[::-1])

    if not table["Tickets"].sum():
        st.info(f"{person} has no tickets resolved in the last {weeks} weeks.")
        return

    st.bar_chart(table.set_index("Week of")["Hours"], height=240)
    recent = table.tail(4)
    c1, c2, c3 = st.columns(3)
    c1.metric("Last 4 weeks", f"{recent['Hours'].sum():.0f}h")
    c2.metric(
        "Weekly average",
        f"{table['Hours'].mean():.1f}h",
        help=f"Across all {weeks} weeks, including weeks with nothing resolved.",
    )
    blind = int(table["No estimate"].sum())
    c3.metric(
        "Resolved with no estimate",
        blind,
        help="Each of these delivered work the hours above cannot see.",
        delta=None if not blind else "hours understated",
        delta_color="inverse",
    )
    st.dataframe(table[::-1], width="stretch", hide_index=True)


def _render_individual_page(
    person: str,
    filtered: pd.DataFrame,
    organization: pd.DataFrame,
    open_prs: pd.DataFrame,
    merged_prs: pd.DataFrame,
    github_ready: bool,
    github_error: str,
    include_backlogs: bool,
) -> None:
    """One engineer's page: nothing on it belongs to anyone else.

    The point is focus - when someone opens their own view, every ticket, PR
    and number is theirs, so the page says where to put the next hour rather
    than what the rest of the organization is doing."""
    st.caption(f"Focused view — everything on this page is {person}'s own work.")

    _render_scope_breakdown(filtered, scope=SCOPE_INDIVIDUAL, include_backlogs=include_backlogs)

    # Their documentation habit, not just their assignments: tickets they wrote
    # count too, because the writer is who can make a ticket Devin-ready.
    owners = organization["assignee"].fillna("").astype(str).str.strip()
    reporters = (
        organization["reporter"].fillna("").astype(str).str.strip()
        if "reporter" in organization.columns
        else pd.Series("", index=organization.index)
    )
    name = str(person).strip()
    theirs = organization[(owners == name) | (reporters == name)]

    # The scorecard sees open AND recently merged PRs together (bounced merged
    # work still counts as rework), and the person's whole open board rather
    # than the sidebar-filtered slice - a headline score must not move because
    # a filter widget did.
    personal_prs = pd.DataFrame()
    if github_ready:
        keys = _known_project_keys(organization)
        # The merged-PR fetch carries no branch or body, so a Jira key can
        # only be read from a merged PR's title; mark those rows so the
        # clean-PR badge is not withheld for a key nobody could see.
        frames = []
        for frame, detectable in ((open_prs, True), (merged_prs, False)):
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                mine = focus.personal_prs(
                    pr_hygiene.add_hygiene_fields(frame, keys), person, organization
                ).copy()
                # A merged PR that names its ticket in the title has shown
                # its key despite the lighter fetch, so it is judgeable too.
                mine["key_detectable"] = detectable | mine.get(
                    "has_jira_key", pd.Series(False, index=mine.index)
                ).fillna(False).astype(bool)
                frames.append(mine)
        if frames:
            personal_prs = pd.concat(frames, ignore_index=True)
            if "url" in personal_prs.columns:
                personal_prs = personal_prs.drop_duplicates(subset="url")

    # Backlog tickets are parked on purpose - the page's other metrics exempt
    # them, and so does the score, regardless of the sidebar toggle.
    whole_board = _metrics_df(organization[owners == name], include_backlogs=False)

    st.divider()
    score, badges_earned = _render_scorecard(
        person, whole_board.copy(), theirs, personal_prs, github_ready
    )

    st.divider()
    _render_engineer_handout(
        person, annotated_board(whole_board, person), score, badges_earned
    )

    st.divider()
    _render_weekly_delivery(person, _jql_identity(whole_board, person))

    st.divider()
    _render_personal_prs(person, open_prs, merged_prs, organization, github_ready, github_error)

    st.divider()
    _render_ticket_quality(theirs)

    st.divider()
    _render_priority_queue(filtered, include_backlogs=include_backlogs)

    st.divider()
    _render_estimate_policy(filtered)

    st.divider()
    _render_stale_cleanup(filtered)

    st.divider()
    st.subheader("Sprint Capacity")
    _render_sprint_capacity(filtered, status_source_df=filtered, selected_ticket_key=None)


# How old an unreviewed PR has to be before nobody-was-asked counts as a
# decision rather than a PR opened an hour ago. Two days is the number Angel's
# own review-latency complaint is about.
TODAY_NO_REVIEWER_DAYS = 2.0

# Where "stalled" starts. Measured on status age, never on edit age: a label
# edit resets idle_days and would empty this tile in five minutes.
TODAY_STALLED_DAYS = 30.0


def _open_pr_signals(open_prs: pd.DataFrame, open_count_exact: object) -> dict[str, Any]:
    """The three PR facts the Today page opens with.

    Draft PRs are excluded from every count here. A draft says "not ready for
    you yet", so counting it as unreviewed would blame reviewers for work the
    author has not handed over.
    """
    try:
        total = _number_or(open_count_exact, float("nan"))
        if open_prs is None or (isinstance(open_prs, pd.DataFrame) and open_prs.empty):
            return {
                "total": int(total) if total == total else 0,
                "unapproved": 0,
                "never_reviewed": 0,
                "oldest_unreviewed_days": None,
                "no_reviewer_asked": 0,
            }

        live = open_prs.copy()
        if "is_draft" in live.columns:
            live = live[~live["is_draft"].fillna(False).astype(bool)]
        
        # After filtering drafts, check if we have any PRs left
        if live.empty:
            return {
                "total": int(total) if total == total else 0,
                "unapproved": 0,
                "never_reviewed": 0,
                "oldest_unreviewed_days": None,
                "no_reviewer_asked": 0,
            }

        approvals = live.get("approving_reviews", pd.Series(0, index=live.index))
        reviews = live.get("total_reviews", pd.Series(0, index=live.index))
        requests = live.get("review_requests", pd.Series(0, index=live.index))
        age = live.get("age_days", pd.Series(0.0, index=live.index)).fillna(0.0)

        unapproved = pd.Series(approvals, index=live.index).fillna(0).astype(int) == 0
        never = pd.Series(reviews, index=live.index).fillna(0).astype(int) == 0
        unasked = never & (pd.Series(requests, index=live.index).fillna(0).astype(int) == 0)

        oldest = age[never].max() if bool(never.any()) else None
        # Handle NaN values in oldest
        if oldest is not None and pd.isna(oldest):
            oldest = None
            
        return {
            "total": int(total) if total == total else int(len(live)),
            "unapproved": int(unapproved.sum()),
            "never_reviewed": int(never.sum()),
            "oldest_unreviewed_days": None if oldest is None else float(oldest),
            "no_reviewer_asked": int((unasked & (age > TODAY_NO_REVIEWER_DAYS)).sum()),
        }
    except Exception as e:
        logger.exception("Error in _open_pr_signals: %s", str(e))
        # Return safe defaults on any error
        return {
            "total": 0,
            "unapproved": 0,
            "never_reviewed": 0,
            "oldest_unreviewed_days": None,
            "no_reviewer_asked": 0,
        }


def _stalled_rows(
    df: pd.DataFrame, events: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, str]:
    """The open tickets that have not *moved* in TODAY_STALLED_DAYS, and the clock.

    A fallback to edit age has to say so - it is the gameable one.

    ``events`` is the board's changelog already flattened by the derivation
    layer (``_engineering_data`` hands it the current bundle's events) - when
    it is given, no changelog is re-parsed here at all. Callers that still
    pass a raw ``df`` carrying its own ``changelog`` column (direct tests,
    chiefly) get the old behaviour: it is parsed from ``df`` itself.
    """
    if df.empty:
        return df.iloc[0:0], "status age"
    try:
        with _log_stage("changelog:stalled_rows"):
            resolved_events = events if events is not None else integrity.changelog_events(df)
            ages = integrity.status_age_days(df, resolved_events)
        column = ages["status_age_days"]
        if not column.dropna().empty:
            # By position, not by label: status_age_days hands back a fresh
            # RangeIndex, and this frame's index has gaps in it wherever a
            # Backlog row was dropped. Asked by label, pandas refuses the mask
            # and the whole measurement silently fell through to edit age.
            keep = (column >= TODAY_STALLED_DAYS).fillna(False).to_numpy()
            rows = df[keep].copy()
            # The age these rows were *chosen* on travels with them, so anything
            # reporting the wait quotes the same clock the tile measured.
            rows[next_actions.STALLED_AGE_COLUMN] = column.to_numpy()[keep]
            return rows, "status age"
    except Exception:  # noqa: BLE001 - a changelog we cannot parse must not blank the page
        logger.exception("status_age_days failed; falling back to edit age")
    idle = df.get("idle_days", pd.Series(0.0, index=df.index)).fillna(0.0)
    rows = df[idle >= TODAY_STALLED_DAYS].copy()
    rows[next_actions.STALLED_AGE_COLUMN] = idle[idle >= TODAY_STALLED_DAYS]
    return rows, "edit age (no changelog)"


def _one_person_instead(
    bundle: "_EngineeringData", view: "_EngineeringView", slot: Any
) -> bool:
    """Draw the single-engineer page when the reader has narrowed to one name.

    Returns True when it did, so the caller stops. Org-wide sections under one
    person's name are somebody else's work wearing their heading, whichever way
    the reader got there - the Individual scope, or a Team multiselect whittled
    down to one.
    """
    people = view.selected_assignees
    if people is None or len(people) != 1:
        return False
    _render_individual_page(
        person=str(people[0]),
        filtered=view.filtered,
        organization=bundle.df,
        open_prs=bundle.open_prs,
        merged_prs=bundle.merged_prs,
        github_ready=bundle.github_ready,
        github_error=bundle.github_error,
        include_backlogs=view.include_backlogs,
    )
    _download_report(slot, TAB_ENGINEERING)
    return True


def _exclude_repos() -> tuple[frozenset[str], str]:
    """The repos left out of every PR read, and the caption naming them.

    Angel's scratch repos are not team output, but dropping them silently would
    let a filtered figure read as a complete one — the disabled-merchant mistake.
    The caption is the contract: every page carrying a PR figure prints it.

    The exclusion itself is applied by :func:`github_client.excluded_repos` in
    the search queries, so the counts that cannot be filtered afterwards (open
    total, merged in 7/30 days) obey it too and no two pages disagree. Names are
    read back from there so the caption cannot drift from what was excluded, and
    only the ones this org owns: ``-repo:someone-else/scratch`` is vacuous against
    an ``org:`` search, so treating it as a name to drop would hide this org's own
    ``scratch`` from the rows while every count kept it.

    A malformed ``GITHUB_ORG`` is not this function's error to raise: the page
    reads the caption before it reaches the check that reports config failures
    legibly, so raising here would replace the page with a stack trace.
    """
    try:
        env = github_client.load_github_env()
    except github_client.GitHubConfigError:
        env = None
    _, org = env or ("", github_client.DEFAULT_ORG)
    owned = f"{org.lower()}/"
    names = frozenset(
        name.split("/", 1)[1]
        for name in github_client.excluded_repos(org)
        if name.lower().startswith(owned)
    )
    caption = (
        "Excludes " + ", ".join(f"`{n}`" for n in sorted(names)) + " — "
        "founder scratch repos, left out of every PR figure on every page, "
        "stated here so a filtered number is never read as a complete one."
        if names
        else ""
    )
    return names, caption


def _truncation_note(total: int, shown: int) -> str:
    """The footer a cut table needs, or "" when nothing was cut."""
    if total <= shown:
        return ""
    return (
        f"Showing the {shown} oldest of {total}. The other {total - shown} are "
        "newer and still waiting — the queue is longer than this card."
    )


def _lazy_import_business():
    """The ``pages.business`` module, imported on first use.

    A local copy of app.py's own ``_lazy_import_business`` (kept separate to
    avoid render_shared.py importing app.py, which would be circular): both
    exist only so the 4,300-line business module is not loaded until a
    Business-page cache actually needs clearing.
    """
    from pages import business
    return business


def _clear_page_caches(page_title: str) -> None:
    """Drop only the reads the page in front of the reader depends on.

    Args:
        page_title: the ``*_PAGE_TITLE`` of the page the reader pressed
            Refresh on. ``BUSINESS_PAGE_TITLE`` clears the Business page's
            caches; any other title clears the engineering ones (Today,
            People, Delivery, Code, Planning and the legacy Engineering page
            all read the same fetches, so they share one cache set).

    Returns:
        None. Clears ``st.cache_data``-decorated fetchers in place and drops
        the session-held engineering bundle / board snapshot so the next
        read is live rather than replayed.

    ``st.cache_data.clear()`` emptied every cache in the process, so refreshing
    a ticket count also threw away the year of orders, the funnel, the project
    list and the user directory - and the next visitor to those pages paid for
    it. Clearing per page keeps the button honest about what it refreshes.
    """
    if page_title == BUSINESS_PAGE_TITLE:
        # Lazy import business module only when clearing business caches
        business = _lazy_import_business()
        business_caches = (
            business._order_book,
            business.fetch_store_prefixes_cached,
            business._funnel_cached,
            business._breakdown_cached,
            business._event_users_cached,
            # Both ads entries, or Refresh would drop the spend and go straight back
            # to a day-old list of accounts to read it for.
            business._ads_cached,
            business._ads_accounts,
            # Burn holds the three bills, and a quarter of an hour is long enough
            # that a reader who presses Refresh expects them to move too.
            business._openai_costs_cached,
            business._cloud_costs_cached,
            business._stripe_ledger_cached,
            business._stripe_disputes_cached,
            # Benchmarks move once a day, but a reader who has just changed a price
            # and pressed Refresh is asking about that price.
            business._price_benchmark_cached,
            # And whose listing each of those prices is: a merchant renamed or a
            # product re-slugged would otherwise keep its old name for a day.
            business._offer_merchants_cached,
            # And what those prices sold: the evidence beside them is only worth
            # refreshing if the bottles move with it.
            business._offer_sales_cached,
            # And what Vivino charges for the same wines: the whole point of a
            # Refresh after a merchant says they have fixed a price there.
            business._vivino_comparison_cached,
        )
        for cached in business_caches:
            cached.clear()
        # With the saved Vivino reads gone, the minutes-long pull must wait
        # for its button again rather than restart on the next screen draw.
        st.session_state.pop("vivino_reads", None)
    else:
        # Engineering page caches
        engineering = (
            fetch_tickets,
            fetch_resolved_count,
            fetch_resolved_tickets,
            fetch_person_resolved_count,
            fetch_person_reopened_count,
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
        for cached in engineering:
            cached.clear()
        # The session-held engineering bundle (Phase 2) would otherwise still
        # match its old fingerprint and hand back the reads just cleared.
        st.session_state.pop(_ENGINEERING_BUNDLE_KEY, None)
        st.session_state.pop(_ENGINEERING_DATA_AS_OF_KEY, None)
        # Refresh means a genuinely live read, rewritten to disk once it
        # finishes (_engineering_data does that itself) - not the file this
        # button was just asked to make stale.
        _delete_board_snapshot()
    logger.info("Cleared cached reads for the %s page", page_title)


