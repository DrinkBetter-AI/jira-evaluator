from __future__ import annotations

import html
import os
import re
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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
    JiraClient,
    JiraConfigError,
    normalize_base_url,
    load_jira_env,
    load_jira_profile,
)
from access_gate import require_password
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
from hygiene import (
    DEFAULT_STALE_DAYS,
    estimate_policy,
    policy_compliance_by_owner,
    stale_candidates,
)
from prioritization import add_priority_score, assignee_rollup
import ticket_quality
from transformations import add_ticket_health_fields
import write_access


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

FETCH_SCHEMA_VERSION = 7
JIRA_KEY_DISPLAY_PATTERN = r".*/browse/([^/?#]+)$"

# One Jira request per key, so bound how many the sprint editor asks about.
TRANSITION_LOOKUP_LIMIT = 50
# Upper bound on how many tickets a bulk write-back pre-selects.
BULK_ACTION_DEFAULT_LIMIT = 25
# Slices a composition pie shows before the tail collapses into "Other".
MIX_SLICE_LIMIT = 10
# Ceiling on tickets fetched per run; org-wide JQL can exceed the old fixed 1000.
MAX_RESULTS = _positive_int(os.getenv("JIRA_MAX_RESULTS"), default=1000)
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


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
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
        fields=DEFAULT_FIELDS,
        max_results=max_results,
        page_size=page_size,
    )


@st.cache_data(ttl=300, show_spinner=False)
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
        fields=DEFAULT_FIELDS,
        max_results=max_results,
        page_size=page_size,
    )


@st.cache_data(ttl=300, show_spinner=False)
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
        fields=DEFAULT_FIELDS,
        max_results=max_results,
        page_size=page_size,
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_open_prs_cached(token: str, org: str, schema_version: int) -> pd.DataFrame:
    _ = schema_version
    return github_client.fetch_open_prs(token, org)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_merged_prs_cached(token: str, org: str, days: int, schema_version: int) -> pd.DataFrame:
    _ = schema_version
    return github_client.fetch_merged_prs(token, org, days)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_merged_pr_count_cached(token: str, org: str, days: int, schema_version: int) -> int:
    _ = schema_version
    return github_client.merged_pr_count(token, org, days)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_open_pr_count_cached(token: str, org: str, schema_version: int) -> int:
    _ = schema_version
    return github_client.open_pr_count(token, org)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_project_keys(creds_path: str, profile_name: str) -> list[str]:
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.get_project_keys()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_all_priorities(creds_path: str, profile_name: str) -> list[str]:
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.get_all_priorities()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_all_users(creds_path: str, profile_name: str) -> list[dict[str, str]]:
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    return client.get_all_users()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_available_transition_statuses(
    creds_path: str,
    profile_name: str,
    issue_keys: tuple[str, ...],
) -> list[str]:
    client = JiraClient.resolve(creds_path=creds_path, profile_name=profile_name)
    available: set[str] = set()
    for key in issue_keys:
        try:
            transitions = client.get_issue_transitions(key)
        except Exception:  # noqa: BLE001
            continue
        for transition in transitions:
            to_status = str(transition.get("to_status", "")).strip()
            if to_status:
                available.add(to_status)
    return sorted(available)


def _metrics_df(df: pd.DataFrame, include_backlogs: bool) -> pd.DataFrame:
    if include_backlogs or "status" not in df.columns:
        return df
    statuses = df["status"].fillna("").astype(str).str.strip().str.lower()
    return df[~statuses.isin(BACKLOG_STATUSES)]


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

    kpi_strip(
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
    kpi_strip(
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


def _render_epics(df: pd.DataFrame) -> None:
    """Group open work by epic and name what is wrong with each one."""
    st.subheader("Epics")
    st.caption(
        "Open children only - the dashboard does not load Done tickets, so this is "
        "remaining work per epic, not completion."
    )

    scored = estimate_policy(df, BACKLOG_STATUSES)
    rollup = epic_health_flags(epic_rollup(scored))
    if rollup.empty:
        st.info("No tickets in the current scope.")
        return

    orphans = int(rollup.loc[rollup["epic"] == "No epic", "open_children"].sum())
    drifting = int((rollup["issue_count"] > 0).sum() - (1 if orphans else 0))
    e1, e2, e3 = st.columns(3)
    e1.metric("Epics with open work", int((rollup["epic"] != "No epic").sum()))
    e2.metric("Epics needing attention", max(drifting, 0))
    e3.metric("Tickets with no epic", orphans)

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

    _render_epic_organization(df)


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
    o1.metric("Tickets with no epic", int(len(suggestions)))
    o2.metric("With a suggested parent", int(len(matched)))
    o3.metric("Epics with nothing open", int(len(empty)))
    st.caption(
        "Suggestions come from the words a ticket shares with an epic and with the "
        "tickets already in it, scored so that a word common to every epic counts "
        "for nothing. Only epics in the ticket's own project are considered. "
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
        st.cache_data.clear()
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
    c1.metric("Policy Compliance", f"{compliance_pct:.0f}%")
    c2.metric("Missing Estimate", len(violations))
    c3.metric("Estimated Work", f"{in_scope['estimate_hours'].sum():.0f}h")

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
    c1.metric(f"Idle {threshold}d+", len(candidates))
    c2.metric("Unassigned", int(unassigned))
    c3.metric("Never Started", int(never_started))

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

    visible_keys = _transition_sample_keys(display_editor_df)
    current_statuses = sorted(display_editor_df["status"].dropna().astype(str).str.strip().unique().tolist())
    try:
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
                help="Change status — applied to Jira on Apply sprint selection",
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
                st.cache_data.clear()
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
    c2.metric(
        "Total estimated (sprint)",
        _fmt_seconds(preview_workload["estimate_seconds_live"].fillna(0).sum()),
        help="Sum of estimates for In Sprint ✅ tickets matching the selected statuses and assignee filter.",
    )
    c3.metric(
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
    c1.metric("New tickets (24h)", _metric_value(new_24h))
    c2.metric("New tickets (7d)", _metric_value(new_7d))
    c3.metric(f"Stuck in triage (> {triage_hours}h)", _metric_value(triage_stuck))

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
    c1.metric("Tickets resolved (7d)", _metric_value(ticket_count_7))
    c2.metric("Tickets resolved (30d)", _metric_value(ticket_count_30))
    c3.metric("PRs merged (7d)", _metric_value(pr_count_7))
    c4.metric("PRs merged (30d)", _metric_value(pr_count_30))

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
    c1.metric("Open PRs", open_count)
    c2.metric("Stuck (no approving review)", int(len(stuck)))
    c3.metric("Never reviewed", int(len(no_review)))
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


def _render_pr_hygiene(
    open_prs: pd.DataFrame,
    github_ready: bool,
    github_error: str = "",
    project_keys: list[str] | None = None,
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

    c1, c2, c3 = st.columns(3)
    c1.metric("No Jira key", int(len(no_key)))
    c2.metric(
        f"Stale (>{pr_hygiene.STALE_AGE_DAYS:.0f}d old or "
        f">{pr_hygiene.STALE_IDLE_DAYS:.0f}d idle)",
        int(len(stale)),
    )
    c3.metric("No reviewer", int(len(unowned)))
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
        "jira_key": st.column_config.TextColumn("Jira"),
    }

    key_tab, stale_tab, owner_tab, person_tab = st.tabs(
        ["No Jira key", "Stale", "No reviewer", "By person"]
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
    c1.metric("Ready for Devin", int(len(ready)))
    c2.metric("Nearly ready", int(len(maybe)))
    c3.metric("Unclear (score ≤2)", int(len(unclear)))
    c4.metric("Average score", f"{gradable['quality_score'].mean():.1f} / 5")
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


def main() -> None:
    st.set_page_config(page_title="Jira Ticket Health Dashboard", layout="wide")
    require_password()
    inject_styles()
    st.title("Jira Ticket Health Dashboard")
    st.caption("Visual monitoring for stale, idle, and high-risk tickets.")

    refresh_clicked = st.button("Refresh Data")

    if refresh_clicked:
        st.cache_data.clear()

    jql = JQL
    max_results = MAX_RESULTS
    page_size = 100

    try:
        raw_df = fetch_tickets(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
            jql=jql,
            max_results=max_results,
            page_size=page_size,
            schema_version=FETCH_SCHEMA_VERSION,
        )
    except JiraConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to fetch Jira issues: {exc}")
        st.stop()

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

    # Recently-resolved work lives outside the main (non-Done) fetch, so pull it
    # separately for the top-of-page snapshot. Two windows keep the 7d/30d split
    # exact without parsing each ticket's changelog. A failure here must not take
    # the whole dashboard down.
    def _resolved_count(days: int) -> int | None:
        try:
            return fetch_resolved_count(
                creds_path=CREDS_PATH,
                profile_name=PROFILE_NAME,
                days=days,
                statuses=RESOLVED_STATUSES,
                schema_version=FETCH_SCHEMA_VERSION,
            )
        except Exception:  # noqa: BLE001
            return None

    def _resolved(days: int) -> pd.DataFrame | None:
        # None (not an empty frame) signals a failed fetch, so the pie can say
        # "could not load" instead of an authoritative "nobody resolved anything".
        try:
            return fetch_resolved_tickets(
                creds_path=CREDS_PATH,
                profile_name=PROFILE_NAME,
                days=days,
                statuses=RESOLVED_STATUSES,
                max_results=max_results,
                page_size=page_size,
                schema_version=FETCH_SCHEMA_VERSION,
            )
        except Exception:  # noqa: BLE001
            return None

    # GitHub PR data is optional: without a token the PR views degrade to a hint
    # rather than an error, so the Jira dashboard still works standalone.
    github_error = ""
    open_prs = pd.DataFrame()
    merged_prs = pd.DataFrame()
    pr_count_7: int | None = None
    pr_count_30: int | None = None
    open_count_exact: int | None = None
    try:
        github_env = github_client.load_github_env()
    except Exception as exc:  # noqa: BLE001
        github_env = None
        github_error = str(exc)[:200]
    github_ready = github_env is not None
    if github_ready:
        token, org = github_env
        try:
            open_prs = fetch_open_prs_cached(token, org, FETCH_SCHEMA_VERSION)
            open_count_exact = fetch_open_pr_count_cached(token, org, FETCH_SCHEMA_VERSION)
            merged_prs = fetch_merged_prs_cached(token, org, 30, FETCH_SCHEMA_VERSION)
            pr_count_7 = fetch_merged_pr_count_cached(token, org, 7, FETCH_SCHEMA_VERSION)
            pr_count_30 = fetch_merged_pr_count_cached(token, org, 30, FETCH_SCHEMA_VERSION)
        except Exception as exc:  # noqa: BLE001
            github_ready = False
            github_error = str(exc)[:200]

    # Intake snapshot: brand-new tickets and anything stuck in triage. Each call
    # is independent and outage-safe so one failing query can't blank the page.
    def _created_count(days: int) -> int | None:
        try:
            return fetch_created_count(
                creds_path=CREDS_PATH,
                profile_name=PROFILE_NAME,
                days=days,
                schema_version=FETCH_SCHEMA_VERSION,
            )
        except Exception:  # noqa: BLE001
            return None

    def _triage_stuck_count() -> int | None:
        try:
            return fetch_triage_stuck_count(
                creds_path=CREDS_PATH,
                profile_name=PROFILE_NAME,
                statuses=TRIAGE_STATUSES,
                hours=TRIAGE_STUCK_HOURS,
                schema_version=FETCH_SCHEMA_VERSION,
            )
        except Exception:  # noqa: BLE001
            return None

    def _triage_stuck_list() -> pd.DataFrame | None:
        try:
            return fetch_triage_stuck_tickets(
                creds_path=CREDS_PATH,
                profile_name=PROFILE_NAME,
                statuses=TRIAGE_STATUSES,
                hours=TRIAGE_STUCK_HOURS,
                max_results=max_results,
                page_size=page_size,
                schema_version=FETCH_SCHEMA_VERSION,
            )
        except Exception:  # noqa: BLE001
            return None

    def _created_list(days: int) -> pd.DataFrame | None:
        try:
            return fetch_created_tickets(
                creds_path=CREDS_PATH,
                profile_name=PROFILE_NAME,
                days=days,
                max_results=max_results,
                page_size=page_size,
                schema_version=FETCH_SCHEMA_VERSION,
            )
        except Exception:  # noqa: BLE001
            return None

    _render_resolved_summary(
        _resolved_count(7),
        _resolved_count(30),
        _resolved(30),
        pr_count_7,
        pr_count_30,
        merged_prs,
        github_ready,
        github_error,
    )
    st.divider()

    _render_new_and_triage(
        _created_count(1),
        _created_count(7),
        _triage_stuck_count(),
        _created_list(7),
        _triage_stuck_list(),
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

        st.header("Filters")
        selected_statuses = st.multiselect("Status", options=statuses, default=[])
        selected_priorities = st.multiselect("Priority", options=priorities, default=[])
        min_idle = st.slider("Min idle days", min_value=0, max_value=180, value=0)
        min_age = st.slider("Min ticket age", min_value=0, max_value=365, value=0)
        include_backlogs = st.checkbox("Include Backlogs", value=False)
        color_by = st.radio("Bubble color", options=["priority", "assignee"], horizontal=True)

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

    st.divider()
    _render_mix(_metrics_df(filtered, include_backlogs))

    st.divider()
    _render_team_overview(_metrics_df(filtered, include_backlogs))

    st.divider()
    _render_epics(_metrics_df(filtered, include_backlogs))

    st.divider()
    # Backlog-inclusive on purpose: the backlog is what this section clears out.
    _render_cleanup(filtered, unassigned_source=unscoped)

    st.divider()
    _render_scope_breakdown(filtered, scope=scope, include_backlogs=include_backlogs)

    st.divider()
    _render_pr_section(open_prs, github_ready, github_error, open_count_exact)

    st.divider()
    _render_pr_hygiene(open_prs, github_ready, github_error, _known_project_keys(df))

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
    action_df = _metrics_df(filtered, include_backlogs)
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

        st.cache_data.clear()
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

                st.cache_data.clear()
                st.rerun()

    st.caption(
        "Team member filter uses Jira assignee display names from fetched data. "
        "For stricter JQL filtering, use assignee account IDs in JQL."
    )


if __name__ == "__main__":
    main()
