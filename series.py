"""Weekly time-series buckets and period-over-period deltas for the Today page.

Nothing in the app currently trends anything over more than one number. The
per-person "Estimated Hours Delivered" bar chart in ``render_shared.py``
(``_render_weekly_delivery``) is the only multi-week chart that exists, it is
scoped to one person, and it re-derives its own week boundaries from
``pd.Timestamp.now()`` on every call rather than from a shared, tested
primitive. PRs merged is never trended at all, and no tile on Today shows a
prior-period comparison of any kind. This module is that shared primitive:
one function to bucket any dated frame into a 12-point weekly series, one
function to turn a pair of numbers into a scored delta, and two thin
convenience wrappers that build the org-wide series Today needs (tickets
resolved, PRs merged) with bots set aside.

Week anchor - decided here, not left to the caller: buckets are calendar
weeks, **Monday 00:00:00 UTC** to the following Monday. Every timestamp is
converted to UTC before it is bucketed, so the same instant lands in the same
bucket no matter what timezone it was recorded or read back in - the team
spans Vietnam (UTC+7), Uruguay (UTC-3), Tunisia (UTC+1) and the EU (UTC+2),
and a sparkline that bucketed in "local" time would put the same Friday
merge in different weeks depending on who was looking at it. See
``docs/assumptions/2F.md`` for the rest of the calls made building this.

Blind spot shared by every function here, stated once: none of them know
whether the date column they were handed is an honest signal. ``idle_days``
resets on a label edit (KPI_SPEC.md exploit #1); if a caller feeds
``weekly_buckets`` a resettable timestamp instead of a machine-recorded one,
the sparkline built on top inherits that exploit silently. This module
buckets whatever it is given - it does not audit the column's meaning.
"""

from __future__ import annotations

import os
from typing import NamedTuple, Sequence

import pandas as pd

# --------------------------------------------------------------------------- #
# Bot exclusion
# --------------------------------------------------------------------------- #

# Duplicated from render_shared.BOT_LOGINS rather than imported. render_shared
# is the module that will eventually call into this one to paint the Today
# sparklines, so importing it from here risks a circular import the moment
# that wiring lands - possibly this same day, since other tasks on this
# branch are touching render_shared concurrently. Both lists read the same
# GITHUB_BOT_LOGINS env var, so the two stay in sync at runtime without a
# Python-level import; keep the literal defaults below identical to
# render_shared._DEFAULT_BOT_LOGINS if that list ever changes.
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

DEFAULT_BOT_LOGINS = frozenset(
    login.strip().lower()
    for login in os.getenv("GITHUB_BOT_LOGINS", ",".join(_DEFAULT_BOT_LOGINS)).split(",")
    if login.strip()
)


def _is_bot(login: object, bot_logins: frozenset[str]) -> bool:
    name = str(login or "").strip().lower()
    if name.endswith("[bot]"):
        name = name[: -len("[bot]")]
    return name in bot_logins


def _exclude_bots(
    frame: pd.DataFrame, login_col: str, bot_logins: Sequence[str] | None
) -> tuple[pd.DataFrame, int]:
    """The frame with bot-authored rows set aside, and how many were removed.

    Mirrors ``render_shared._people_only`` in behaviour (same fallbacks: an
    empty frame or a missing column passes through untouched, 0 excluded) so
    the two report the same bot count on the same data, without one module
    importing the other.
    """
    logins = frozenset(
        str(login).strip().lower() for login in (bot_logins or DEFAULT_BOT_LOGINS)
    )
    if frame.empty or login_col not in frame.columns:
        return frame, 0
    is_bot = frame[login_col].map(lambda login: _is_bot(login, logins))
    return frame[~is_bot], int(is_bot.sum())


# --------------------------------------------------------------------------- #
# Weekly buckets
# --------------------------------------------------------------------------- #


class WeekBucket(NamedTuple):
    """One point on a 12-week sparkline."""

    week_start: pd.Timestamp  # Monday 00:00:00 UTC - the week this bucket covers.
    label: str  # short axis label; "This week (partial)" for the last bucket.
    value: float  # row count for that week.
    is_partial: bool  # True only for the current, still-open week (the last bucket).


def _week_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Monday 00:00:00 UTC of the week containing ``ts``.

    ``ts`` may carry any tz (or none, treated as already UTC) - it is
    converted to UTC before the week boundary is found, which is the whole
    timezone-stability guarantee this module makes: two different local
    clock readings of the same instant produce the same ``_week_start``.
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    ts = ts.normalize()
    return ts - pd.Timedelta(days=ts.dayofweek)


def weekly_buckets(frame: pd.DataFrame, date_col: str, weeks: int = 12) -> list[WeekBucket]:
    """Row counts per calendar week, oldest first, over the trailing ``weeks`` weeks.

    Args:
        frame: any frame with a parseable date column. Unparseable values
            (``NaT`` after ``pd.to_datetime(..., errors="coerce")``) are
            dropped rather than counted, the same way a missing PR size is
            dropped rather than treated as zero elsewhere in this codebase.
        date_col: the column to bucket on. Its meaning is the caller's
            responsibility - see the module docstring's blind spot.
        weeks: how many trailing weeks to return. Defaults to 12.

    Returns:
        Exactly ``weeks`` buckets in chronological order (oldest first), even
        when ``frame`` is empty or lacks ``date_col`` - a sparkline with no
        points is a broken sparkline, so an empty result always means "12
        weeks of zero", never "no data to show". The last bucket is the
        current week, which is still in progress, and is the only one with
        ``is_partial=True``: a raw count for it is not comparable to a full
        week's count, and a caller that charts it without that flag invites
        exactly the "week 12 crashed" misread a partial bucket produces.

    Blind to: rows that never made it into ``frame`` at all (a throttled
    fetch, a truncated ``max_results`` page) look identical to a genuinely
    quiet week. This function only counts what it was handed.
    """
    if weeks < 1:
        raise ValueError("weeks must be at least 1")

    now = pd.Timestamp.now(tz="UTC")
    current_start = _week_start(now)
    starts = [current_start - pd.Timedelta(weeks=offset) for offset in range(weeks - 1, -1, -1)]
    counts = [0] * weeks

    if not frame.empty and date_col in frame.columns:
        parsed = pd.to_datetime(frame[date_col], utc=True, errors="coerce").dropna()
        for ts in parsed:
            week = _week_start(ts)
            offset_weeks = (current_start - week).days // 7
            index = weeks - 1 - offset_weeks
            if 0 <= index < weeks:
                counts[index] += 1

    buckets: list[WeekBucket] = []
    for index, start in enumerate(starts):
        is_partial = index == weeks - 1
        label = "This week (partial)" if is_partial else start.strftime("%b %d")
        buckets.append(WeekBucket(start, label, float(counts[index]), is_partial))
    return buckets


class BotExcludedSeries(NamedTuple):
    """A weekly series plus how many bot-authored rows were set aside to build it.

    Carried together so a page can print "excl. bots (23 bot merges)" next to
    the sparkline rather than a bare count nobody can audit - the same reason
    ``render_shared._people_only`` returns its exclusion count instead of
    just dropping the rows.
    """

    buckets: list[WeekBucket]
    bots_excluded: int


def tickets_resolved_series(
    tickets: pd.DataFrame,
    weeks: int = 12,
    date_col: str = "status_category_changed_date",
    login_col: str = "assignee",
    bot_logins: Sequence[str] | None = None,
) -> BotExcludedSeries:
    """Org-wide weekly count of tickets resolved, bots excluded.

    ``tickets`` is expected to already be scoped to resolved tickets - the
    shape :func:`data_layer.fetch_resolved_tickets` returns. There is no
    ``resolutiondate`` column on that frame, so this buckets on
    ``status_category_changed_date``, the closest machine-recorded stand-in
    for "when this left the board" the frame carries. ``RESOLVED_FIELDS``
    asks Jira for that field by name: the row builder emits the column either
    way, so omitting it yielded not a missing column but a null one, and
    twelve weeks of silent zeros. That is a deliberate
    approximation, recorded in ``docs/assumptions/2F.md``: a ticket that
    re-entered a resolved status more than once (KPI_SPEC exploit #2) only
    shows its most recent category change here, not every re-entry -
    ``integrity.reresolve_events`` is the tool for that, not this one.

    Blind to: a ticket resolved by a bot account under ``login_col`` is
    excluded from the human total but still real work someone should know
    happened - that is why the exclusion count travels with the series
    instead of the rows just vanishing.
    """
    people, excluded = _exclude_bots(tickets, login_col, bot_logins)
    return BotExcludedSeries(weekly_buckets(people, date_col, weeks), excluded)


def prs_merged_series(
    prs: pd.DataFrame,
    weeks: int = 12,
    date_col: str = "merged_at",
    login_col: str = "author",
    bot_logins: Sequence[str] | None = None,
) -> BotExcludedSeries:
    """Org-wide weekly count of PRs merged, bots excluded.

    ``prs`` is expected to be a merged-PR frame shaped like
    :func:`github_client.fetch_merged_prs` returns (``merged_at``, ``author``
    present). PR count alone is exploit #3 from KPI_SPEC.md - five trivial
    PRs beat one real one - so this is volume telemetry for the org, not a
    per-person performance signal; pair it with ``pr_quality.size_bands``
    before drawing any conclusion about an individual.

    Blind to: a merge commit landed by ``renovate`` or ``dependabot`` under a
    login not in ``bot_logins`` inflates the human count silently. The
    default list is a snapshot, not a guarantee - see the module docstring.
    """
    people, excluded = _exclude_bots(prs, login_col, bot_logins)
    return BotExcludedSeries(weekly_buckets(people, date_col, weeks), excluded)


# --------------------------------------------------------------------------- #
# Delta
# --------------------------------------------------------------------------- #


class Delta(NamedTuple):
    """A period-over-period change, with direction and goodness scored separately.

    ``direction`` says which way the number moved. ``is_good`` says whether
    that move was welcome - and depends entirely on the caller's
    ``higher_is_better``, since the same "up" is good for tickets resolved
    and bad for stalled tickets. Colouring a tile by direction alone, without
    asking which direction is good, is the classic dashboard lie this type
    exists to prevent.
    """

    magnitude: float | None  # abs(current - prior); None when either side is unknown.
    direction: str  # "up" | "down" | "flat" - "flat" also covers "unknown".
    is_good: bool | None  # None = no verdict. Never True when data is missing.


def delta(current: float | None, prior: float | None, *, higher_is_better: bool) -> Delta:
    """Score a period-over-period change. ``higher_is_better`` has no default on purpose.

    Args:
        current: this period's value.
        prior: the prior period's value. ``None`` or ``NaN`` means the prior
            period had no data to compare against - a cold-start tile, a
            metric added this week, a person with no tickets last month.
        higher_is_better: whether an increase is the good outcome for this
            metric. Required, not defaulted: a caller that forgets to
            specify it should get a ``TypeError``, not a silently wrong
            colour on a tile someone is judged by.

    Returns:
        A :class:`Delta`. When either value is missing, ``direction`` is
        reported as ``"flat"`` (there is nothing to point an arrow at) but
        ``is_good`` is ``None`` - explicitly not the same thing as a genuine
        tie, and never ``True``. A caller that maps ``is_good is not False``
        to "show green" would be wrong for exactly this reason; the correct
        check is ``is_good is True``.

        On a genuine tie (``current == prior``, both known) ``direction`` is
        also ``"flat"`` and ``is_good`` is also ``None`` - nothing changed,
        so there is nothing to call good or bad.

        ``magnitude`` is the raw absolute difference, not a percentage:
        percentage change is undefined at ``prior == 0`` (a metric going
        from zero to any nonzero value on an hourly team - new hire's first
        PR, first ticket in a new category - is common, not an edge case to
        special-case around), and a raw difference makes callers who want
        "N more than last week" a plain read at the cost of not saying
        whether that N is 5-on-a-base-of-5 or 5-on-a-base-of-500. Blind spot
        worth stating plainly: this function has no sense of scale. A
        caller that needs "up 300%" rather than "up 3" has to divide these
        two numbers itself, and needs its own zero-guard when it does.
    """
    if current is None or prior is None or pd.isna(current) or pd.isna(prior):
        return Delta(magnitude=None, direction="flat", is_good=None)

    current = float(current)
    prior = float(prior)
    diff = current - prior

    if diff > 0:
        direction = "up"
    elif diff < 0:
        direction = "down"
    else:
        direction = "flat"

    if direction == "flat":
        is_good = None
    else:
        is_good = (direction == "up") == higher_is_better

    return Delta(magnitude=abs(diff), direction=direction, is_good=is_good)
