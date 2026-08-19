"""#team-snippets (Slack channel C098TCRRV2Q) compliance.

Replaces the "Weekly updates" scorecard component (``idle_days <= 7``), which
DEVIN_PLAN.md calls the most gameable signal in the system: ``idle_days``
resets on any changelog edit at all, including a label change nobody would
call work. Posting a #team-snippets update is at least a thing a person did
on purpose - it does not prove they did the work, only that they showed up
and wrote a sentence. That is a weaker claim than it sounds, and section
"Blind spot" below says so explicitly.

Identity join
-------------
Three separate identity namespaces meet here: a Jira display name (the
roster key in ``roles.py``), a GitHub login (``GITHUB_LOGIN_MAP``), and now
a Slack user ID, which appears nowhere else in this codebase. There is no
way to infer a Slack ID from a Jira name or a GitHub login, so the join has
to be explicit: ``SLACK_USER_MAP`` (env var, same "Name=value;..." shape as
``GITHUB_LOGIN_MAP``) maps a Slack user ID to the Jira display name that
resolves against ``roles.load_roster()``. Unlike ``roles.py``, this module
ships **no baked-in default map** - see ``load_slack_user_map`` for why.

The credential situation
-------------------------
There is no Slack bot token in this environment, and there will not be one
during this task. ``load_slack_env`` returns ``None`` rather than raising
when ``SLACK_BOT_TOKEN`` is unset, and every function that needs the
network checks for that ``None`` before it tries to use it. Nothing in this
module has ever made a real HTTP request against Slack; every code path is
exercised in ``tests/test_snippets.py`` against recorded fixture payloads
shaped like real ``conversations.history`` / ``conversations.replies``
responses.

The measured baseline (worked example)
---------------------------------------
Top-level posts only, ISO weeks W25-W32 (2026-06-15 through 2026-08-09),
the eight full weeks DEVIN_PLAN.md measured by hand before this module
existed::

    Farid 8/8 · Tam 7 · Mohsen 7 · Mehdi 7 · Mihai 6 · David 4 · Igor 4 ·
    Anouar 3 · Santi 2 · Alesya 2 · Ali 1 · Shawn 0 · Jal 0 · Gaston 0
    (never, in channel history) · Dina 0 · Robert 0

``tests/test_snippets.py`` builds a fixture payload shaped like a real
Slack API response and reproduces these exact ``weeks_posted`` counts via
:func:`build_report`.

Blind spot
----------
This is a presence signal, not a truth signal, and it will be gamed the
moment it carries weight - the same Goodhart caution DEVIN_PLAN.md states
for attendance and proactivity counts. A snippet is cheap to write: it
proves someone typed a sentence into Slack, not that the sentence is
accurate, not that the work described happened, and not that work happened
at all in a week with no snippet. Treat ``weeks_posted`` / ``current_streak``
as one input to a conversation, never as the whole verdict on a week.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, field
from typing import Mapping

import requests

import roles as _roles

SNIPPETS_CHANNEL_ID = "C098TCRRV2Q"

_TOKEN_ENV_VAR = "SLACK_BOT_TOKEN"
_SLACK_MAP_ENV_VAR = "SLACK_USER_MAP"

SLACK_API_BASE = "https://slack.com/api"
_HISTORY_PATH = "/conversations.history"
_REPLIES_PATH = "/conversations.replies"

_PAGE_LIMIT = 200
_MAX_PAGES = 20
_DEFAULT_WEEKS = 8


class SlackConfigError(RuntimeError):
    """Raised only for a programming error in how this module is called - never for a network/API failure, which degrades instead."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_slack_env(env: Mapping[str, str] | None = None) -> str | None:
    """Return the Slack bot token, or ``None`` when ``SLACK_BOT_TOKEN`` is unset.

    Never raises. There is no token in this environment; every caller in
    this module treats ``None`` as "Slack unavailable" and reports that as
    data, not as an exception.
    """
    source = env if env is not None else os.environ
    token = str(source.get(_TOKEN_ENV_VAR) or "").strip()
    return token or None


def load_slack_user_map(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Parse ``SLACK_USER_MAP`` ("Jira Name=SlackUserId;...") into ``{slack_user_id: jira_name}``.

    No baked-in default, unlike ``roles.load_roster``'s four env vars. This
    repo has never had a Slack bot token, so nobody has ever confirmed a
    real Slack user ID for anyone on the roster. A guessed default would
    risk crediting one person's snippet to another the day a real token is
    finally set - wrong attribution is worse than no attribution. Every
    Slack author is reported as unmapped until whoever holds Slack admin
    access populates this var with IDs read off ``users.info`` or a
    profile's "Copy member ID".

    A malformed pair costs that pair, not the whole parse, matching
    ``roles.load_roster``'s handling of its own env vars.
    """
    source = env if env is not None else os.environ
    raw = str(source.get(_SLACK_MAP_ENV_VAR) or "").strip()
    mapping: dict[str, str] = {}
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, _, user_id = pair.partition("=")
        name = name.strip()
        user_id = user_id.strip()
        if not name or not user_id:
            continue
        mapping[user_id] = name
    return mapping


# ---------------------------------------------------------------------------
# Week bucketing
# ---------------------------------------------------------------------------


def _iso_week_key(day: _dt.date) -> str:
    """``"2026-W25"`` for the ISO week (Monday-Sunday) containing ``day``."""
    year, week, _weekday = day.isocalendar()
    return f"{year}-W{week:02d}"


def _week_start(day: _dt.date) -> _dt.date:
    """The Monday of the ISO week containing ``day``."""
    return day - _dt.timedelta(days=day.weekday())


def _week_start_date(week_key: str) -> _dt.date:
    """The Monday :class:`~datetime.date` for a ``"YYYY-Www"`` key."""
    year_str, _, week_str = week_key.partition("-W")
    return _dt.date.fromisocalendar(int(year_str), int(week_str), 1)


def _ts_to_date(ts: str) -> _dt.date:
    """The UTC calendar date of a Slack message timestamp.

    Slack timestamps are UTC epoch seconds. The team spans Vietnam, Uruguay,
    Tunisia and the EU; UTC is the one neutral reference among them, and it
    is what every other client in this codebase (``cost_client.py``) already
    buckets by.
    """
    return _dt.datetime.fromtimestamp(float(ts), tz=_dt.timezone.utc).date()


def recent_full_weeks(now: _dt.date, count: int = _DEFAULT_WEEKS) -> tuple[str, ...]:
    """The ``count`` ISO week keys immediately before the week containing ``now``, oldest first.

    Never includes the week ``now`` falls in - a week in progress reads as
    a false "0" for everyone who simply hasn't posted yet this week, and
    DEVIN_PLAN.md's own baseline is stated as "the last 8 *full* weeks" for
    exactly this reason.
    """
    if count <= 0:
        return ()
    last_full_start = _week_start(now) - _dt.timedelta(weeks=1)
    starts = [last_full_start - _dt.timedelta(weeks=i) for i in range(count - 1, -1, -1)]
    return tuple(_iso_week_key(s) for s in starts)


def _oldest_ts_for_window(weeks_window: tuple[str, ...]) -> str | None:
    """The Slack ``oldest`` cursor value (epoch seconds) for the start of ``weeks_window``."""
    if not weeks_window:
        return None
    start = _week_start_date(weeks_window[0])
    stamp = _dt.datetime.combine(start, _dt.time.min, tzinfo=_dt.timezone.utc).timestamp()
    return f"{stamp:.6f}"


# ---------------------------------------------------------------------------
# Low-level fetch: pagination, throttling, degradation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PageFetch:
    """One paginated read: every message collected, and whether it's the whole thing."""

    messages: tuple[dict, ...]
    complete: bool
    reason: str | None


def _paginate(
    http: object,
    token: str,
    path: str,
    params: dict[str, str],
    max_pages: int = _MAX_PAGES,
) -> _PageFetch:
    """Page through a Slack conversations.* endpoint via ``cursor``/``has_more``.

    Stops and returns whatever it already collected - flagged incomplete,
    never silently truncated - on a rate limit (HTTP 429), any other HTTP
    error, a malformed/``ok: false`` payload, a transport failure, or
    running past ``max_pages``. This never raises: a degraded read is a
    result, not an exception, matching every other client in this codebase
    (``github_client.py``, ``cost_client.py``).

    Unlike ``github_client._graphql``, this does not retry/back off inside
    the call - Slack's own ``Retry-After`` on a 429 can run to a minute, and
    unlike GitHub's dashboard reads this module has no in-progress page
    render depending on the wait completing. A caller who wants a full read
    after a 429 just calls again; the partial result already collected is
    still valid data for whichever weeks it did reach.
    """
    messages: list[dict] = []
    cursor: str | None = None
    for page_num in range(max_pages):
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor
        try:
            response = http.get(
                f"{SLACK_API_BASE}{path}",
                params=page_params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
        except requests.RequestException as exc:
            return _PageFetch(
                messages=tuple(messages),
                complete=False,
                reason=f"Slack unavailable ({path}, page {page_num + 1}): {exc}",
            )
        if response.status_code == 429:
            return _PageFetch(
                messages=tuple(messages),
                complete=False,
                reason=(
                    f"rate limited by Slack (HTTP 429) on {path} after "
                    f"{page_num} page(s); {len(messages)} message(s) collected before the limit"
                ),
            )
        if response.status_code >= 400:
            return _PageFetch(
                messages=tuple(messages),
                complete=False,
                reason=f"Slack refused {path}: HTTP {response.status_code}",
            )
        payload = response.json()
        if not isinstance(payload, dict) or not payload.get("ok", False):
            error = payload.get("error", "unexpected response shape") if isinstance(payload, dict) else "unexpected response shape"
            return _PageFetch(
                messages=tuple(messages),
                complete=False,
                reason=f"Slack API error on {path}: {error}",
            )
        messages.extend(payload.get("messages", []) or [])
        meta = payload.get("response_metadata") or {}
        cursor = str(meta.get("next_cursor") or "").strip() or None
        if not payload.get("has_more") or not cursor:
            return _PageFetch(messages=tuple(messages), complete=True, reason=None)
    return _PageFetch(
        messages=tuple(messages),
        complete=False,
        reason=f"stopped after {max_pages} page(s) of {path} (safety cap); more may remain",
    )


def _countable(message: dict) -> bool:
    """A real person's message: has a ``user``, and isn't a join/bot/system subtype."""
    return bool(message.get("user")) and message.get("subtype") is None


@dataclass(frozen=True)
class RawFetch:
    """Raw messages read from the channel, before any roster join or week bucketing.

    ``top_level`` is every countable message from ``conversations.history``
    (thread parents included). ``replies`` is every countable message from
    a ``conversations.replies`` call that is *not* the thread's parent -
    the parent already appears in ``top_level`` and would double-count if
    kept here too.
    """

    available: bool
    partial: bool
    reason: str
    top_level: tuple[dict, ...]
    replies: tuple[dict, ...]


def fetch_channel_posts(
    token: str | None,
    channel: str = SNIPPETS_CHANNEL_ID,
    session: object | None = None,
    oldest: str | None = None,
    max_pages: int = _MAX_PAGES,
) -> RawFetch:
    """Read ``channel``'s top-level posts and thread replies.

    ``token=None`` (or empty) returns an unavailable :class:`RawFetch`
    immediately, without attempting any request - the no-credential path
    this whole module exists to make safe.

    Thread replies are only fetched for a top-level message that reports
    its own ``reply_count`` - Slack's own signal that a thread has
    children, and the only way to know one exists without asking. A 429 or
    any other failure encountered while paginating either endpoint marks
    the whole fetch ``partial``; messages already collected are still
    returned, never discarded.
    """
    if not token:
        return RawFetch(
            available=False,
            partial=False,
            reason=f"{_TOKEN_ENV_VAR} is not set; #team-snippets cannot be read",
            top_level=(),
            replies=(),
        )

    http = session if session is not None else requests

    params: dict[str, str] = {"channel": channel, "limit": str(_PAGE_LIMIT)}
    if oldest:
        params["oldest"] = oldest
    history = _paginate(http, token, _HISTORY_PATH, params, max_pages=max_pages)
    top_level = tuple(m for m in history.messages if _countable(m))

    partial = not history.complete
    reason = history.reason or ""

    replies: list[dict] = []
    for parent in top_level:
        if not parent.get("reply_count"):
            continue
        thread_ts = str(parent.get("thread_ts") or parent.get("ts") or "")
        if not thread_ts:
            continue
        thread_params = {"channel": channel, "ts": thread_ts, "limit": str(_PAGE_LIMIT)}
        thread = _paginate(http, token, _REPLIES_PATH, thread_params, max_pages=max_pages)
        if not thread.complete:
            partial = True
            reason = thread.reason or reason
        for message in thread.messages:
            if str(message.get("ts") or "") == thread_ts:
                continue  # the parent, re-sent by conversations.replies - already in top_level
            if _countable(message):
                replies.append(message)

    return RawFetch(
        available=True,
        partial=partial,
        reason=reason,
        top_level=top_level,
        replies=tuple(replies),
    )


# ---------------------------------------------------------------------------
# Roster join and weekly report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonSnippetStats:
    """One roster person's #team-snippets record over the reported weeks.

    ``top_level_by_week`` / ``reply_by_week`` are keyed by every week in
    the report's window, zero-filled for weeks with no activity - a
    consumer can always assume every reported week is present. Multiple
    posts inside one week collapse to that week counting once toward
    ``weeks_posted``; the raw per-week count is still there for whoever
    wants it, but the streak itself does not reward posting more than once
    a week.
    """

    name: str
    top_level_by_week: dict[str, int]
    reply_by_week: dict[str, int]
    weeks_posted: int
    weeks_in_window: int
    current_streak: int
    top_level_total: int
    reply_total: int


@dataclass(frozen=True)
class SnippetReport:
    """The joined, week-bucketed #team-snippets read - what People/Today consume.

    ``people`` only holds roster members with a confirmed ``SLACK_USER_MAP``
    entry. Anyone who posted under a Slack ID with no roster mapping is in
    ``unmapped`` instead - keyed by Slack user ID, message count as the
    value - never dropped and never folded into a real person's total.

    Blind spot: ``roster_size`` counts every active roster person, whether
    or not they have a Slack ID mapped yet. An active person with no
    mapping who *does* post shows up only in ``unmapped``, and the
    denominator does not shrink to account for that - so an unmapped active
    person always reads as a non-poster this week even on a week they
    posted. That gap is an admin task (finish populating ``SLACK_USER_MAP``),
    not a reason to lower the denominator and let it disappear quietly.
    """

    available: bool
    partial: bool
    reason: str
    weeks: tuple[str, ...]
    people: dict[str, PersonSnippetStats]
    unmapped: dict[str, int]
    roster_size: int


def _consecutive_from_end(weeks: tuple[str, ...], top_by_week: dict[str, int]) -> int:
    streak = 0
    for week in reversed(weeks):
        if top_by_week.get(week, 0) > 0:
            streak += 1
        else:
            break
    return streak


def build_report(
    fetch: RawFetch,
    roster: _roles.Roster,
    slack_user_map: Mapping[str, str],
    weeks_window: tuple[str, ...],
) -> SnippetReport:
    """Join ``fetch``'s raw messages to ``roster`` via ``slack_user_map``, bucketed by ``weeks_window``.

    Every active roster person named in ``slack_user_map`` is present in
    the result, including a zero-post person (weeks_posted=0) - the report
    is never restricted to people who posted. A Slack user ID in the
    messages but absent from ``slack_user_map``, or mapped to a name the
    roster does not recognise or has marked former, lands in ``unmapped``.
    """
    top_counts: dict[tuple[str, str], int] = {}
    reply_counts: dict[tuple[str, str], int] = {}

    for message in fetch.top_level:
        user_id = str(message.get("user") or "")
        if not user_id:
            continue
        week = _iso_week_key(_ts_to_date(message["ts"]))
        key = (user_id, week)
        top_counts[key] = top_counts.get(key, 0) + 1

    for message in fetch.replies:
        user_id = str(message.get("user") or "")
        if not user_id:
            continue
        week = _iso_week_key(_ts_to_date(message["ts"]))
        key = (user_id, week)
        reply_counts[key] = reply_counts.get(key, 0) + 1

    def _mapped_active_name(user_id: str) -> str | None:
        name = slack_user_map.get(user_id)
        if name is None:
            return None
        person = roster.person(name)
        if person is None or not person.active:
            return None
        return person.name

    all_user_ids = {uid for uid, _week in top_counts} | {uid for uid, _week in reply_counts}
    # Every mapped-active roster person is seeded in, even with zero messages
    # anywhere in the fetch, so a total non-poster still gets weeks_posted=0
    # rather than being absent from the report entirely.
    all_user_ids |= {uid for uid in slack_user_map if _mapped_active_name(uid) is not None}

    unmapped: dict[str, int] = {}
    per_name_top: dict[str, dict[str, int]] = {}
    per_name_reply: dict[str, dict[str, int]] = {}

    for user_id in all_user_ids:
        name = _mapped_active_name(user_id)
        if name is None:
            total = sum(c for (uid, _w), c in top_counts.items() if uid == user_id)
            total += sum(c for (uid, _w), c in reply_counts.items() if uid == user_id)
            if total:
                unmapped[user_id] = unmapped.get(user_id, 0) + total
            continue
        top_by_week = per_name_top.setdefault(name, {})
        reply_by_week = per_name_reply.setdefault(name, {})
        for week in weeks_window:
            t = top_counts.get((user_id, week), 0)
            r = reply_counts.get((user_id, week), 0)
            if t:
                top_by_week[week] = top_by_week.get(week, 0) + t
            if r:
                reply_by_week[week] = reply_by_week.get(week, 0) + r

    people: dict[str, PersonSnippetStats] = {}
    for name, top_by_week in per_name_top.items():
        reply_by_week = per_name_reply.get(name, {})
        full_top = {week: top_by_week.get(week, 0) for week in weeks_window}
        full_reply = {week: reply_by_week.get(week, 0) for week in weeks_window}
        weeks_posted = sum(1 for week in weeks_window if full_top[week] > 0)
        people[name] = PersonSnippetStats(
            name=name,
            top_level_by_week=full_top,
            reply_by_week=full_reply,
            weeks_posted=weeks_posted,
            weeks_in_window=len(weeks_window),
            current_streak=_consecutive_from_end(weeks_window, full_top),
            top_level_total=sum(full_top.values()),
            reply_total=sum(full_reply.values()),
        )

    roster_size = sum(1 for p in roster.people.values() if p.active)

    return SnippetReport(
        available=fetch.available,
        partial=fetch.partial,
        reason=fetch.reason,
        weeks=weeks_window,
        people=people,
        unmapped=unmapped,
        roster_size=roster_size,
    )


def weekly_snippet_report(
    roster: _roles.Roster | None = None,
    env: Mapping[str, str] | None = None,
    session: object | None = None,
    weeks: int = _DEFAULT_WEEKS,
    now: _dt.date | None = None,
    channel: str = SNIPPETS_CHANNEL_ID,
) -> SnippetReport:
    """The end-to-end read: env config, fetch, roster join - what a page should call.

    With no ``SLACK_BOT_TOKEN`` set, returns immediately with
    ``available=False`` and a reason - no request is attempted, this never
    raises, and the result never carries a ``0`` that could be mistaken for
    "measured zero snippets" instead of "not measured at all".
    """
    ros = roster if roster is not None else _roles.load_roster()
    now_date = now or _dt.datetime.now(_dt.timezone.utc).date()
    weeks_window = recent_full_weeks(now_date, weeks)
    roster_size = sum(1 for p in ros.people.values() if p.active)

    token = load_slack_env(env)
    if token is None:
        return SnippetReport(
            available=False,
            partial=False,
            reason=f"{_TOKEN_ENV_VAR} is not set; #team-snippets compliance cannot be read",
            weeks=weeks_window,
            people={},
            unmapped={},
            roster_size=roster_size,
        )

    slack_user_map = load_slack_user_map(env)
    oldest = _oldest_ts_for_window(weeks_window)
    fetch = fetch_channel_posts(token, channel=channel, session=session, oldest=oldest)
    return build_report(fetch, ros, slack_user_map, weeks_window)


def this_week_poster_count(report: SnippetReport) -> tuple[int, int]:
    """``(posters, roster_size)`` for the most recent week in ``report.weeks`` - what Today shows.

    A poster is counted on top-level posts only, matching DEVIN_PLAN.md's
    "poster count" baseline, which was measured the same way. ``0, 0`` when
    the report has no weeks (e.g. ``weeks=0`` was requested) or is
    unavailable with an empty window - callers should check
    ``report.available`` before treating the count as measured.
    """
    if not report.weeks:
        return 0, report.roster_size
    latest = report.weeks[-1]
    posters = sum(1 for stats in report.people.values() if stats.top_level_by_week.get(latest, 0) > 0)
    return posters, report.roster_size
