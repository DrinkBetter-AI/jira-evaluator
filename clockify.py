"""Clockify: what was billed, next to what shipped. WP11, the true cross-check.

Jira estimates and Jira worklogs are both self-reported by the person being
measured (see ``estimate_accuracy.py``'s own docstring on that). Clockify is a
third, independent number: what was actually invoiced. It is *also*
self-reported in the ordinary case - anyone can open a timer and let it run, or
type in a number by hand - so this module does not claim to catch padding
directly. What it can do is put three numbers for the same person-week next to
each other and let them disagree:

    hours billed (Clockify) · delivered (size-weighted merged PRs +
    changelog-credited resolutions) · hours logged in Jira.

No one of the three, alone, is evidence of anything. A week with real hours
billed and real delivered work and no Jira worklog just means Jira worklogs are
optional here (they mostly are). A week with hours billed and no delivered work
and no board movement is the thing worth a conversation - and even then, it has
an innocent reading: research, meetings, incident response, a long refactor
that hasn't merged yet. Say so on whatever page renders this.

**Enforcement over detection.** Clockify has a workspace setting called "Force
timer" (Pro/Enterprise plans only, confirmed against Clockify's own help
documentation: clockify.me/help/track-time-and-expenses/force-timer) that
disables manual time entry outright and only accepts a running timer. Turning
that on solves the reconstructed-timesheet problem at the source, for every
person, with no metric required and nothing to dispute. The detectors below
exist for the gap while that setting is off (or the plan doesn't support it) -
they are the fallback, not the first choice. Whoever owns the Clockify account
should be told this before this module's flags are.

**Credentials.** There is no ``CLOCKIFY_API_KEY`` / ``CLOCKIFY_WORKSPACE_ID``
in this environment and none is expected during this task. Every function here
is built so that absence is a normal, well-typed outcome: a result with
``available=False`` and a ``reason`` string, never an exception and never a
frame of zeros standing in for "we don't know." Real Clockify calls are
injected through a ``fetcher`` callable specifically so the whole module can be
exercised against hand-written fixture payloads without a key, a workspace, or
a network.

**Person mapping.** The roster (``roles.py``) knows Jira names and GitHub
logins; it does not know Clockify user ids or emails, because nobody has
confirmed yet whether Clockify emails match the vinovoss.com convention
(``DEVIN_PLAN.md`` WP11 lists this as an open prerequisite). So this module
keeps its own mapping, ``CLOCKIFY_USER_MAP``, and treats every roster name
absent from it as ``"unmapped"`` - a distinct, visible state, never coerced to
zero hours, because zero reads as "billed nothing" and a broken mapping is not
the same fact as billing nothing. Praveen Rai is the one name ``ROSTER.md``
already confirms is not on Clockify at all; that is a different, expected
state (``"not_on_clockify"``), not a broken mapping and not an error.

**The created-at tell is not implemented, on purpose.** Reconstructed
timesheets - filled in from memory at the end of the week - tend to look
different from timer-driven ones in four ways: perfectly round daily blocks,
one block per day instead of timer-grained entries, overlapping entries, and
entries created well after the work they claim. The fourth one needs the API
to expose *when the entry record itself was written*, separate from the
work's claimed start/end. Checked against Clockify's published API reference
(docs.clockify.me) and a second independent source describing the same
``TimeEntry`` schema: the detailed-report time entry object carries ``id``,
``description``, ``timeInterval`` (``start``/``end``/``duration``/timezone
fields), ``project``, ``task``, ``tags``, ``billable``, ``hourlyRate``,
``costRate``, ``customFieldValues``, ``type``, ``approvalRequestId`` - no
``createdAt``, no ``modifiedAt``, nothing that records when the row was
written versus when the work supposedly happened. Faking that from a proxy
(entry id ordering, page position, anything else that correlates with
insertion order but isn't documented as insertion order) is exactly the kind
of thing that gets a metric correctly disbelieved the first time someone
checks it against the raw export. So: three tells implemented
(:func:`detect_round_block_weeks`, :func:`detect_one_block_days`,
:func:`detect_overlaps`), the fourth explicitly stubbed as unavailable
(:func:`late_created_tell`) with the reason stated in the return value itself,
not just in this docstring.

**The blind spot, named once for the whole module:** nothing here proves
padding. A perfectly round 8.00h day can be an honest day someone rounded
when they forgot to start the timer at 9:00 sharp. A single daily block can be
one uninterrupted afternoon of real, hard, unbroken work. Overlapping entries
can be a timezone mis-set on a phone app. Every flag is a question to ask, in
the form "here is the evidence, what happened here" - never a verdict.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple

import pandas as pd
import requests

import estimate_accuracy

REPORTS_BASE_URL = "https://reports.api.clockify.me/v1"
_API_KEY_ENV_VAR = "CLOCKIFY_API_KEY"
_WORKSPACE_ENV_VAR = "CLOCKIFY_WORKSPACE_ID"
# "Name=clockifyEmailOrUserId;Name=...". No baked default: unlike the Jira
# roster (roles.py), nobody has confirmed a single Clockify identifier yet, so
# the honest default is empty, and every active person renders "unmapped"
# until this is populated - never a silent, wrong guess at an email address.
_USER_MAP_ENV_VAR = "CLOCKIFY_USER_MAP"

_PAGE_SIZE = 1000
# Detailed reports can run long; this bounds one read the same way
# github_client bounds its retries - a caller that hangs forever on a
# misbehaving API is worse than one that gives up and says so.
_MAX_PAGES = 200
_TIMEOUT_SECONDS = 30

# ROSTER.md, confirmed by Angel: the one person genuinely not on Clockify.
# Case-insensitive, matched the same way roles.Person.key is.
NOT_ON_CLOCKIFY: frozenset[str] = frozenset({"praveen rai"})

# Same threshold estimate_accuracy.py uses for its own modified z-score, and
# imported rather than copied so the two files can never quietly disagree on
# where "far enough to flag" sits.
OUTLIER_Z = estimate_accuracy.OUTLIER_Z

# Below this, a robust "same value every day" comparison folds:  one or two
# data points cannot tell a reconstructed week from a lucky one.
MIN_DAYS_FOR_BLOCK_TELL = 3

# A daily total within this many hours of a whole number counts as "round".
# 0.02h is 72 seconds - loose enough to catch 7:59:xx-rounding entries typed
# in as "8", tight enough that ordinary timer noise (a forgotten pause, a
# minute spent finding the stop button) does not trip it.
ROUND_BLOCK_TOLERANCE_HOURS = 0.02


# ---------------------------------------------------------------------------
# Configuration and credentials
# ---------------------------------------------------------------------------


class ClockifyConfig(NamedTuple):
    """Just enough to call the reports API: the key and which workspace."""

    api_key: str
    workspace_id: str


def _env_value(env: Mapping[str, str] | None, name: str) -> str:
    source = env if env is not None else os.environ
    return str(source.get(name, "") or "").strip()


def load_clockify_env(env: Mapping[str, str] | None = None) -> ClockifyConfig | None:
    """The API key and workspace id, or ``None`` when either is missing.

    Both are required - a key with no workspace cannot pick a report, and a
    workspace id with no key cannot authenticate - so this is deliberately
    all-or-nothing rather than partially configured. Every caller in this
    module treats ``None`` as "go build the unavailable result", never as
    "guess" or "raise".
    """
    api_key = _env_value(env, _API_KEY_ENV_VAR)
    workspace_id = _env_value(env, _WORKSPACE_ENV_VAR)
    if not api_key or not workspace_id:
        return None
    return ClockifyConfig(api_key=api_key, workspace_id=workspace_id)


def missing_env_reason(env: Mapping[str, str] | None = None) -> str:
    """Which of the two required env vars is missing, stated plainly."""
    has_key = bool(_env_value(env, _API_KEY_ENV_VAR))
    has_workspace = bool(_env_value(env, _WORKSPACE_ENV_VAR))
    if not has_key and not has_workspace:
        missing = f"{_API_KEY_ENV_VAR} and {_WORKSPACE_ENV_VAR} are both unset"
    elif not has_key:
        missing = f"{_API_KEY_ENV_VAR} is unset"
    else:
        missing = f"{_WORKSPACE_ENV_VAR} is unset"
    return f"Clockify unavailable: {missing}."


def load_clockify_user_map(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Roster name (lower-cased) -> Clockify user id or email, from env.

    Format matches the other roster env vars in this codebase
    (``roles_template.env``'s ``;``-joined ``key=value`` pairs), for the same
    reason: one parsing shape across every roster-adjacent env var. A
    malformed pair costs that pair, not the whole map - same rule
    ``roles.load_roster`` follows.
    """
    raw = _env_value(env, _USER_MAP_ENV_VAR)
    out: dict[str, str] = {}
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, _, identifier = pair.partition("=")
        name = name.strip()
        identifier = identifier.strip()
        if not name or not identifier:
            continue
        out[name.lower()] = identifier
    return out


def resolve_person_status(name: str, user_map: Mapping[str, str]) -> str:
    """``"mapped"`` / ``"unmapped"`` / ``"not_on_clockify"`` for one roster name.

    ``"not_on_clockify"`` beats an absent mapping so Praveen Rai never reads as
    a broken mapping - he is not a data problem, everyone else missing is.
    """
    key = str(name).strip().lower()
    if key in NOT_ON_CLOCKIFY:
        return "not_on_clockify"
    if key in user_map:
        return "mapped"
    return "unmapped"


# ---------------------------------------------------------------------------
# Fetching (real HTTP, injectable for tests)
# ---------------------------------------------------------------------------

Fetcher = Callable[[ClockifyConfig, dt.date, dt.date, int], dict[str, Any]]


def _default_fetcher(
    config: ClockifyConfig, start: dt.date, end: dt.date, page: int
) -> dict[str, Any]:
    """One page of the workspace's detailed report. Never called in tests -
    every test injects a fake ``fetcher`` instead, per the credential
    situation this module is built around."""
    url = f"{REPORTS_BASE_URL}/workspaces/{config.workspace_id}/reports/detailed"
    body = {
        "dateRangeStart": f"{start.isoformat()}T00:00:00Z",
        "dateRangeEnd": f"{end.isoformat()}T23:59:59Z",
        "detailedFilter": {"page": page, "pageSize": _PAGE_SIZE},
    }
    response = requests.post(
        url,
        json=body,
        headers={"X-Api-Key": config.api_key, "Content-Type": "application/json"},
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _entries_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse one detailed-report page into flat entry dicts.

    Tolerant of the two id spellings Clockify's endpoints mix
    (``id``/``_id``) and of a missing ``userEmail`` (some workspaces only
    expose ``userName``) - a partially-populated entry is kept with whatever
    identity fields it has rather than dropped, so mapping can still try the
    ones present.
    """
    rows: list[dict[str, Any]] = []
    for item in payload.get("timeentries", []) or []:
        interval = item.get("timeInterval") or {}
        rows.append(
            {
                "entry_id": item.get("id") or item.get("_id") or "",
                "user_id": str(item.get("userId") or "").strip(),
                "user_email": str(item.get("userEmail") or "").strip(),
                "user_name": str(item.get("userName") or "").strip(),
                "start": interval.get("start"),
                "end": interval.get("end"),
            }
        )
    return rows


_ENTRY_COLUMNS = [
    "entry_id",
    "user_id",
    "user_email",
    "user_name",
    "start",
    "end",
    "duration_hours",
    "day",
    "week_start",
    "person",
]


def _empty_entries() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in _ENTRY_COLUMNS})


def _finish_entries(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty_entries()
    frame = pd.DataFrame(rows)
    frame["start"] = pd.to_datetime(frame["start"], utc=True, errors="coerce")
    frame["end"] = pd.to_datetime(frame["end"], utc=True, errors="coerce")
    frame = frame[frame["start"].notna() & frame["end"].notna()]
    if frame.empty:
        return _empty_entries()
    frame["duration_hours"] = (
        (frame["end"] - frame["start"]).dt.total_seconds() / 3600.0
    )
    frame["day"] = frame["start"].dt.date
    frame["week_start"] = (
        frame["start"] - pd.to_timedelta(frame["start"].dt.weekday, unit="D")
    ).dt.date
    frame["person"] = pd.NA
    return frame.reset_index(drop=True)


class EntriesFetch(NamedTuple):
    """Raw, unaggregated Clockify entries, plus whether the fetch worked."""

    frame: pd.DataFrame
    available: bool
    reason: str = ""


def fetch_time_entries(
    config: ClockifyConfig,
    start: dt.date,
    end: dt.date,
    fetcher: Fetcher | None = None,
) -> EntriesFetch:
    """Every detailed-report entry in ``[start, end]``, paginated.

    Never raises: a request exception (bad key, workspace not found, network
    down, a 5xx) comes back as ``available=False`` with the exception text in
    ``reason``, the same degrade-not-crash shape ``github_client.PRFetch``
    uses for the same reason - one section of a multi-section dashboard
    failing should not take the rest of the page down with it.
    """
    fetch = fetcher or _default_fetcher
    rows: list[dict[str, Any]] = []
    try:
        for page in range(1, _MAX_PAGES + 1):
            payload = fetch(config, start, end, page)
            page_rows = _entries_from_payload(payload)
            if not page_rows:
                break
            rows.extend(page_rows)
            if len(page_rows) < _PAGE_SIZE:
                break
    except Exception as exc:  # noqa: BLE001 - degrade, never crash a dashboard read
        return EntriesFetch(_empty_entries(), False, f"Clockify unavailable: {exc}")
    return EntriesFetch(_finish_entries(rows), True, "")


# ---------------------------------------------------------------------------
# Person mapping over a fetched entries frame
# ---------------------------------------------------------------------------


def _reverse_user_map(user_map: Mapping[str, str]) -> dict[str, str]:
    """Clockify identifier (lower-cased) -> roster name."""
    return {identifier.strip().lower(): name for name, identifier in user_map.items()}


def attribute_entries(
    entries: pd.DataFrame, user_map: Mapping[str, str], roster_names: Mapping[str, str]
) -> pd.DataFrame:
    """Fill ``entries["person"]`` from ``user_id``/``user_email`` via ``user_map``.

    Matches on user id first, then email, both lower-cased. An entry that
    matches neither belongs to a Clockify user this dashboard has no roster
    identity for - kept in the returned frame with ``person`` left null
    rather than dropped, so a caller auditing the raw pull can still see it;
    :func:`weekly_billed_hours` excludes null-person rows from the per-person
    rollup, since there is no roster row to credit them to.

    ``roster_names`` maps the roster's own lower-cased key back to its
    display-cased name, so the ``person`` column carries the name as
    ``roles.py`` spells it, not as Clockify happens to.
    """
    if entries.empty:
        return entries
    reverse = _reverse_user_map(user_map)
    out = entries.copy()

    def _lookup(row: pd.Series) -> Any:
        for candidate in (row.get("user_id", ""), row.get("user_email", "")):
            key = str(candidate).strip().lower()
            if key and key in reverse:
                roster_key = reverse[key]
                return roster_names.get(roster_key, roster_key)
        return pd.NA

    out["person"] = out.apply(_lookup, axis=1)
    return out


# ---------------------------------------------------------------------------
# Weekly rollup: the triplet's first leg
# ---------------------------------------------------------------------------

_WEEKLY_COLUMNS = ["person", "week_start", "hours_billed", "status"]


def _week_starts(start: dt.date, end: dt.date) -> list[dt.date]:
    first = start - dt.timedelta(days=start.weekday())
    weeks = []
    cursor = first
    while cursor <= end:
        weeks.append(cursor)
        cursor += dt.timedelta(days=7)
    return weeks


def _empty_weekly() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in _WEEKLY_COLUMNS})


class WeeklyBilledHours(NamedTuple):
    """The Clockify leg of the triplet: hours billed, per person, per week.

    ``weekly``: one row per (roster name, week_start) with ``hours_billed``
    and ``status``. ``status == "mapped"`` rows carry a real number, which is
    ``0.0`` when Clockify genuinely has no entries that week for a mapped
    person - a true zero, earned by having an identity to check against.
    ``"unmapped"`` and ``"not_on_clockify"`` rows carry ``pd.NA``, never
    ``0.0``, by construction (see :func:`resolve_person_status`).

    ``entries``: the raw per-entry frame the rollup was built from, kept
    around because the reconstruction-tell detectors need entry-level detail
    the weekly total throws away.
    """

    weekly: pd.DataFrame
    entries: pd.DataFrame
    available: bool
    reason: str = ""


def weekly_billed_hours(
    people: Iterable[str],
    start: dt.date,
    end: dt.date,
    *,
    env: Mapping[str, str] | None = None,
    fetcher: Fetcher | None = None,
) -> WeeklyBilledHours:
    """The full WP11 read: for each name in ``people``, hours billed per week.

    ``people`` is whatever roster slice the caller wants reported on (the
    active engineering roster, say) - this module does not import
    ``roles.load_roster`` itself and pick a slice, because who gets reported
    on here is a page-wiring decision, not a data-fetching one.

    No key or no workspace id: returns immediately with ``available=False``
    and a stated reason, and an empty ``weekly``/``entries`` - not a frame of
    per-person NAs, since without credentials there is nothing to report per
    person at all, mapped or not. This never raises.
    """
    names = list(dict.fromkeys(str(p) for p in people if str(p).strip()))
    user_map = load_clockify_user_map(env)
    weeks = _week_starts(start, end)

    config = load_clockify_env(env)
    if config is None:
        return WeeklyBilledHours(_empty_weekly(), _empty_entries(), False, missing_env_reason(env))

    fetch = fetch_time_entries(config, start, end, fetcher)
    if not fetch.available:
        return WeeklyBilledHours(_empty_weekly(), _empty_entries(), False, fetch.reason)

    roster_names = {name.strip().lower(): name for name in names}
    attributed = attribute_entries(fetch.frame, user_map, roster_names)
    matched = attributed[attributed["person"].notna()] if not attributed.empty else attributed
    totals = (
        matched.groupby(["person", "week_start"])["duration_hours"].sum()
        if not matched.empty
        else pd.Series(dtype="float64")
    )

    rows: list[dict[str, Any]] = []
    for name in names:
        status = resolve_person_status(name, user_map)
        for week in weeks:
            if status == "mapped":
                hours = float(totals.get((name, week), 0.0))
            else:
                hours = pd.NA
            rows.append(
                {"person": name, "week_start": week, "hours_billed": hours, "status": status}
            )
    weekly = pd.DataFrame(rows, columns=_WEEKLY_COLUMNS) if rows else _empty_weekly()
    return WeeklyBilledHours(weekly, attributed, True, "")


def billed_hours_by_person(weekly: pd.DataFrame) -> pd.Series:
    """Total mapped, billed hours per person across a :class:`WeeklyBilledHours`.

    This is the seam WP11 asks for: ``estimate_accuracy.hours_per_delivered_line``
    (that module, line ~352) currently builds its ``logged_hours`` column as
    ``_numeric(frame, "time_spent_sec") / 3600.0`` - Jira worklogs. Swapping
    the source to Clockify, once a workspace key exists, is meant to be the
    one-line change of building ``weekly`` here via :func:`weekly_billed_hours`
    and reindexing that column from this series by ``assignee`` instead. Not
    made here - ``estimate_accuracy.py`` is out of this task's file list -
    this function only makes that swap a reindex instead of a rewrite.

    Unmapped and not-on-Clockify rows are excluded, not zero-filled: a person
    with no verified Clockify identity contributes nothing to this series
    rather than contributing a false zero that would silently deflate
    whatever ratio consumes it.
    """
    if weekly.empty or "status" not in weekly.columns:
        return pd.Series(dtype="Float64")
    mapped = weekly[weekly["status"] == "mapped"]
    if mapped.empty:
        return pd.Series(dtype="Float64")
    return mapped.groupby("person")["hours_billed"].sum().astype("Float64")


# ---------------------------------------------------------------------------
# MAD-based outliers - no fixed thresholds, per KPI_SPEC.md and this task
# ---------------------------------------------------------------------------


def _modified_z(values: pd.Series) -> pd.Series:
    """Median-absolute-deviation z-score. Same algorithm and same fallback as
    ``estimate_accuracy._modified_z`` (that function is private to its
    module, so this is a deliberate, documented hand-copy rather than an
    import reaching into another task's internals - both are exercised
    against the same ``OUTLIER_Z`` threshold, imported above, so the two
    cannot silently disagree on where the bar sits even though the code is
    duplicated).

    A fixed "more than N hours is suspicious" rule is both wrong (some roles
    and some weeks genuinely run long) and trivially gamed (log 7.9). This
    compares each value only to the distribution it came from - a person's
    own history, or their cohort's - so the bar moves with the data instead
    of sitting still for someone to learn where it is.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    clean = numeric.dropna()
    unknown = pd.Series(pd.NA, index=values.index, dtype="Float64")
    if len(clean) < 3:
        return unknown
    median = clean.median()
    deviations = (clean - median).abs()
    mad = deviations.median()
    if mad and not pd.isna(mad):
        return (0.6745 * (numeric - median) / mad).astype("Float64")
    mean_ad = deviations.mean()
    if mean_ad and not pd.isna(mean_ad):
        return ((numeric - median) / (1.253314 * mean_ad)).astype("Float64")
    return unknown


def self_history_outliers(weekly: pd.DataFrame, outlier_z: float = OUTLIER_Z) -> pd.DataFrame:
    """Per person, which weeks sit far from *that person's own* median week.

    This is the first of the two MAD comparisons KPI_SPEC/this task ask for:
    a week is flagged only against the same person's other weeks, so a role
    that always runs long (infrastructure, platform) does not read as
    permanently anomalous against people it was never fair to compare it to.
    Needs at least three of a person's own weeks to say anything; fewer than
    that returns ``modified_z`` as missing for every one of their rows, same
    "unanswerable, not clean" convention ``estimate_accuracy`` uses.
    """
    columns = ["person", "week_start", "hours_billed", "modified_z", "is_outlier"]
    if weekly.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    mapped = weekly[weekly["status"] == "mapped"].copy()
    if mapped.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    mapped["hours_billed"] = pd.to_numeric(mapped["hours_billed"], errors="coerce")
    mapped["modified_z"] = mapped.groupby("person")["hours_billed"].transform(_modified_z)
    mapped["is_outlier"] = (mapped["modified_z"].abs() > float(outlier_z)).fillna(False).astype(bool)
    return mapped[columns].sort_values(
        ["person", "week_start"], ignore_index=True
    )


def cohort_outliers(
    weekly: pd.DataFrame,
    role_of: Mapping[str, str] | Callable[[str], str | None],
    outlier_z: float = OUTLIER_Z,
) -> pd.DataFrame:
    """Per (role cohort, week), which person-weeks sit far from that cohort's median.

    The second MAD comparison: a week compared against the *other people in
    the same role that same week*, not the individual's own history. This is
    the one that catches "everyone on the team billed a normal week except
    this one person", which :func:`self_history_outliers` cannot see if that
    person has always run that high.

    Deliberately robust rather than mean-based: someone sitting on the high
    side of a genuinely spread-out cohort (a wide but ordinary range of
    hours across a role that week) does not flag just for being above the
    average - only a person-week whose *distance from the cohort's own
    typical spread* clears ``outlier_z`` does. Cohorts of fewer than three
    mapped people that week return missing rather than a score, matching
    ``roles.MIN_PEERS`` - two people cannot tell an outlier from a pair.
    """
    columns = ["person", "week_start", "role", "hours_billed", "modified_z", "is_outlier"]
    if weekly.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    mapped = weekly[weekly["status"] == "mapped"].copy()
    if mapped.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    lookup = role_of if callable(role_of) else (lambda name: role_of.get(name))
    mapped["role"] = mapped["person"].map(lookup)
    mapped = mapped[mapped["role"].notna()]
    if mapped.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    mapped["hours_billed"] = pd.to_numeric(mapped["hours_billed"], errors="coerce")
    mapped["modified_z"] = mapped.groupby(["role", "week_start"])["hours_billed"].transform(_modified_z)
    mapped["is_outlier"] = (mapped["modified_z"].abs() > float(outlier_z)).fillna(False).astype(bool)
    return mapped[columns].sort_values(["role", "week_start", "person"], ignore_index=True)


# ---------------------------------------------------------------------------
# Reconstruction tells
# ---------------------------------------------------------------------------


class TellResult(NamedTuple):
    """One reconstruction-tell detector's output.

    ``evidence``: the flagged rows only, per KPI_SPEC.md rule 4 - a page
    renders this directly rather than a bare count. ``available=False`` means
    the detector could not run at all (right now, only :func:`late_created_tell`
    ever sets this - the other three always run, and an empty ``evidence``
    frame from them means "ran, found nothing", not "could not check").
    """

    evidence: pd.DataFrame
    available: bool = True
    reason: str = ""


def _entries_with_person(entries: pd.DataFrame) -> pd.DataFrame:
    if entries.empty or "person" not in entries.columns:
        return entries.iloc[0:0]
    return entries[entries["person"].notna()].copy()


def detect_round_block_weeks(
    entries: pd.DataFrame,
    *,
    tolerance_hours: float = ROUND_BLOCK_TOLERANCE_HOURS,
    min_days: int = MIN_DAYS_FOR_BLOCK_TELL,
) -> TellResult:
    """Weeks where every worked day's total lands suspiciously on a whole hour.

    A timer stopped by hand almost never lands on an exact hour - a nine-to-
    five day includes a bathroom break, a slow Slack reply, a meeting that ran
    four minutes over. A week where *every* day comes out to exactly a round
    number, to the second, looks like it was typed in at the end of the week
    from memory ("I worked about 8 hours a day") rather than timed as it
    happened.

    Flags the whole week, not the day, because one round day is an
    unremarkable coincidence and several in the same week is the pattern.
    Requires at least ``min_days`` worked days in the week, and *every* one of
    them must land within ``tolerance_hours`` of an integer - a week with two
    round days and one at 7.9h does not flag (see the module tests: an
    honestly varied week like 7.9/8.1/8.0 does not).

    *Innocent reading, stated:* someone who genuinely works a fixed 8-to-5
    with a fixed lunch, every single day, produces exactly this pattern by
    living an unusually disciplined schedule. A flag is a reason to ask, not
    a verdict.
    """
    columns = ["person", "week_start", "days", "daily_hours"]
    scoped = _entries_with_person(entries)
    if scoped.empty:
        return TellResult(pd.DataFrame({c: pd.Series(dtype="object") for c in columns}))
    daily = (
        scoped.groupby(["person", "week_start", "day"])["duration_hours"]
        .sum()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for (person, week), group in daily.groupby(["person", "week_start"]):
        totals = group.sort_values("day")["duration_hours"].tolist()
        if len(totals) < min_days:
            continue
        if all(abs(total - round(total)) <= tolerance_hours for total in totals):
            rows.append(
                {
                    "person": person,
                    "week_start": week,
                    "days": len(totals),
                    "daily_hours": ", ".join(f"{t:.2f}" for t in totals),
                }
            )
    evidence = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(
        {c: pd.Series(dtype="object") for c in columns}
    )
    return TellResult(evidence)


def detect_one_block_days(
    entries: pd.DataFrame, *, min_days: int = MIN_DAYS_FOR_BLOCK_TELL
) -> TellResult:
    """Weeks where every worked day has exactly one entry, not timer-grained ones.

    A person actually using a timer through the day accumulates several
    entries per day - one per task switch, one per meeting break, one per
    "stepped away and forgot to note why". A single block covering the whole
    day, repeated across the week, is what a once-a-week timesheet fill-in
    produces instead.

    Flags the week when every worked day (at least ``min_days`` of them) has
    exactly one entry. A week with several entries most days does not flag,
    regardless of their total duration.

    *Innocent reading, stated:* one long uninterrupted task, worked start to
    finish with the timer simply left running, produces this pattern too -
    and is a perfectly normal way to work.
    """
    columns = ["person", "week_start", "days", "entries_per_day"]
    scoped = _entries_with_person(entries)
    if scoped.empty:
        return TellResult(pd.DataFrame({c: pd.Series(dtype="object") for c in columns}))
    counts = (
        scoped.groupby(["person", "week_start", "day"])["entry_id"]
        .count()
        .reset_index(name="count")
    )
    rows: list[dict[str, Any]] = []
    for (person, week), group in counts.groupby(["person", "week_start"]):
        per_day = group.sort_values("day")["count"].tolist()
        if len(per_day) < min_days:
            continue
        if all(c == 1 for c in per_day):
            rows.append(
                {
                    "person": person,
                    "week_start": week,
                    "days": len(per_day),
                    "entries_per_day": ", ".join(str(c) for c in per_day),
                }
            )
    evidence = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(
        {c: pd.Series(dtype="object") for c in columns}
    )
    return TellResult(evidence)


def detect_overlaps(entries: pd.DataFrame) -> TellResult:
    """Entries for the same person whose time ranges overlap.

    Two entries claiming overlapping wall-clock time cannot both have been
    timed live by the same person - at least one is wrong, whether from
    editing a stopped timer's end time by hand, a timezone mis-set on a
    second device, or two devices both left running. Detected with a running
    maximum of ``end`` over each person's entries sorted by ``start``, which
    catches a new entry overlapping *any* earlier one still open, not only
    the immediately preceding one (a nested pair, not just an adjacent one,
    is still caught).

    Returns the overlapping pair itself as evidence - the two entry ids and
    their start/end times - never a bare count, per KPI_SPEC.md rule 4.

    *Innocent reading, stated:* a phone app and a desktop app both running
    briefly after a laptop woke from sleep produces a few minutes of
    overlap with nobody double-billing anything.
    """
    columns = [
        "person",
        "entry_id_a",
        "start_a",
        "end_a",
        "entry_id_b",
        "start_b",
        "end_b",
        "overlap_hours",
    ]
    scoped = _entries_with_person(entries)
    if scoped.empty:
        return TellResult(pd.DataFrame({c: pd.Series(dtype="object") for c in columns}))
    rows: list[dict[str, Any]] = []
    for person, group in scoped.groupby("person"):
        ordered = group.sort_values("start").reset_index(drop=True)
        running_end = None
        running_row = None
        for _, row in ordered.iterrows():
            if running_end is not None and row["start"] < running_end:
                overlap = (min(row["end"], running_end) - row["start"]).total_seconds() / 3600.0
                rows.append(
                    {
                        "person": person,
                        "entry_id_a": running_row["entry_id"],
                        "start_a": running_row["start"],
                        "end_a": running_row["end"],
                        "entry_id_b": row["entry_id"],
                        "start_b": row["start"],
                        "end_b": row["end"],
                        "overlap_hours": round(overlap, 2),
                    }
                )
            if running_end is None or row["end"] > running_end:
                running_end = row["end"]
                running_row = row
    evidence = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(
        {c: pd.Series(dtype="object") for c in columns}
    )
    return TellResult(evidence)


_CREATED_AT_REASON = (
    "Not implemented: Clockify's detailed-report time entry object "
    "(docs.clockify.me) exposes id, description, timeInterval "
    "(start/end/duration/timezone), project, task, tags, billable, "
    "hourlyRate, costRate, customFieldValues, type, approvalRequestId - no "
    "createdAt or equivalent. There is no documented field recording when "
    "the entry record was written, separate from the work's claimed "
    "start/end, so a 'logged days after the work' tell cannot be built "
    "without inventing a proxy for a fact the API does not report."
)


def late_created_tell(entries: pd.DataFrame) -> TellResult:
    """The fourth reconstruction tell. Deliberately not implemented - see
    :data:`_CREATED_AT_REASON`. Kept as a function, same shape as the other
    three detectors, so a caller iterating "all four tells" gets a uniform,
    self-describing ``available=False`` result instead of a missing attribute
    or a silently-empty frame that would read as "checked, found nothing"
    when the true state is "cannot check this at all".
    """
    columns = ["person", "entry_id", "start", "reason"]
    return TellResult(
        pd.DataFrame({c: pd.Series(dtype="object") for c in columns}),
        available=False,
        reason=_CREATED_AT_REASON,
    )
