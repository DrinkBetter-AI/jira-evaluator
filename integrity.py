"""Integrity checks: what the changelog says that the current board hides.

Every number the scorecard reads today is either self-declared (estimate,
priority, status) or resettable with a keystroke (``idle_days`` moves to zero
when anybody edits any field on the ticket). On a team of remote hourly
contractors that is an invitation: the cheapest way to look busy is to groom
the board rather than move the work.

The changelog is the one record on the board nobody edits. ``app.py`` already
fetches it - every ticket is read with ``expand="changelog"``, so each issue
arrives with its full ``histories[]``: who changed which field, from what, to
what, at what second. Today ``jira_client._extract_last_meaningful_activity``
reduces all of that to a single ``max()`` timestamp and the rest is dropped on
the floor. This module picks it back up.

What this module is for, in one line each:

- ``status_age_days``       - the staleness clock that a label edit cannot reset.
- ``cosmetic_touches``      - grooming the board instead of working it.
- ``estimate_churn``        - estimates raised after the work started.
- ``reresolve_events``      - work that bounced, including the bounces that healed.
- ``status_pingpong``       - resolution credit minted by moving a ticket in circles.
- ``cycle_time``            - the one metric you can only improve by finishing work.
- ``integrity_flags``       - the four above rolled into named flags, with evidence.
- ``credited_resolutions``  - resolution credit to the changelog author, not the
  current assignee, with former-staff credit flagged instead of counted.
- ``unattributed_resolutions`` - resolutions the changelog names nobody for.
- ``org_reopen_rate``       - organization-wide reopen count and share-of-resolved.
- ``end_to_end_cycle``      - the one headline cycle-time number, current vs baseline.

Three rules the whole module obeys:

1. Nothing here is a verdict. Every flag is a question to ask in a one-on-one,
   and every flag carries the ticket keys and timestamps that produced it,
   because an accusation without evidence is worse than no accusation.
2. Activity is attributed to ``history["author"]`` - the person who actually
   made the edit - never to the current assignee. Reassignment is itself one of
   the moves being measured, so scoring by assignee would let a ticket's history
   follow whoever holds it last.
3. Absence of a signal is not innocence. Everything below sees only what passed
   through Jira. Work done and never recorded, or padding that happens in the
   hour log rather than on the board, is invisible to all of it.

Shared blind spot, stated once: Jira returns at most ~100 changelog entries
inline with a search, and only entries a ticket actually generated. A very long
history is truncated (the oldest entries drop off), and a board migrated from
another tracker starts its history at the migration. Both make every "count of
events" here a floor, never a ceiling. Where a function needs a timestamp that
truncation may have removed, it says so and falls back to something honest.

Pure pandas and stdlib: no Streamlit, no network, no imports from ``app``. Every
function takes frames and returns frames so it can be tested with a dozen rows.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

import pandas as pd


# --------------------------------------------------------------------------
# Workflow vocabulary
# --------------------------------------------------------------------------

# The team's pipeline, ranked. Rank order is what makes "backward" meaningful:
# a transition whose destination ranks below its origin is work moving away from
# done. Names are matched normalised (lowercased, whitespace collapsed) because
# Jira spells the same status differently between company- and team-managed
# projects, and this board mixes "IN DEV ENV" with "Review in Staging".
#
# A status not in this map has rank None and is skipped by the direction-aware
# checks rather than guessed at. That is deliberate: inventing a rank for an
# unknown status would manufacture backward transitions out of a workflow this
# module has never seen.
STATUS_STAGES: dict[str, int] = {
    "backlog": 0,
    "discussion needed": 1,
    "triage": 1,
    "to do": 1,
    "todo": 1,
    "open": 1,
    "selected for development": 1,
    "ready for development": 1,
    "in progress": 2,
    "in development": 2,
    "blocked": 2,
    "in dev env": 3,
    "code review": 4,
    "in review": 4,
    "review": 4,
    "pr review": 4,
    "review in staging": 5,
    "in staging": 5,
    "qa": 5,
    "ready for production": 6,
    "released": 7,
    "released to production": 7,
    "done": 7,
    "closed": 7,
    "resolved": 7,
    "completed": 7,
    "cancelled": 7,
    "canceled": 7,
    "won't do": 7,
}

# At or past this rank the ticket is being worked, not planned. Used as the
# "mid-flight" line for estimate churn: an estimate written before work starts
# is planning, the same edit after work starts is a revision of the bill.
STARTED_RANK = 2

# Mirrors ``app._DEFAULT_RESOLVED_STATUSES``. The team counts more than Jira's
# Done category as resolved - notably "Review in Staging", which is emphatically
# not done - so a ticket can be counted resolved, dragged back, and counted
# resolved again. Pass the deployment's own ``JIRA_RESOLVED_STATUSES`` here if
# it has been overridden; the default is duplicated rather than imported so this
# module stays free of ``app``.
DEFAULT_RESOLVED_STATUSES: frozenset[str] = frozenset(
    {
        "done",
        "released",
        "released to production",
        "ready for production",
        "review in staging",
        "closed",
        "resolved",
        "completed",
    }
)

# Fields whose edit, on its own, moves no work. Editing a description can be
# real refinement and often is - see the blind spot note on ``cosmetic_touches``
# - so this list is not an accusation by itself; it only becomes interesting
# when the same person's status transitions are near zero over the same window.
COSMETIC_FIELDS: frozenset[str] = frozenset(
    {
        "labels",
        "description",
        "summary",
        "priority",
        "assignee",
        "duedate",
        "due date",
        "components",
        "component",
        "fixversions",
        "fix version",
        "rank",
        "environment",
        "issuetype",
        "issue type",
    }
)

# Time-tracking fields. Story points are deliberately absent: they are unitless,
# so "raised from 3 to 8" cannot be converted into hours on an hourly contract,
# and mixing the two would produce a padding number nobody can defend.
ESTIMATE_FIELDS: frozenset[str] = frozenset(
    {
        "timeoriginalestimate",
        "original estimate",
        "originalestimate",
        "timeestimate",
        "remaining estimate",
        "timetracking",
    }
)

_STATUS_FIELDS: frozenset[str] = frozenset({"status"})

# Jira duration units in seconds, for parsing "1w 2d 3h 30m".
_DURATION_UNITS = {"w": 144000.0, "d": 28800.0, "h": 3600.0, "m": 60.0, "s": 1.0}

# How many (key, timestamp) pairs a flag's evidence string shows before it says
# "+N more". Enough to start a conversation, short enough to fit a table cell.
EVIDENCE_LIMIT = 5


# --------------------------------------------------------------------------
# Flag thresholds
# --------------------------------------------------------------------------
# Deliberately few, deliberately blunt, deliberately named constants: a manager
# has to be able to say out loud what tripped a flag. Tune them against one
# quarter of real data before trusting the flag column in a one-on-one.

# Board grooming: this many field-only edits in the window, with status
# transitions at least this many times rarer, before it looks like grooming.
COSMETIC_BURST_TOUCHES = 8
COSMETIC_TO_TRANSITION_RATIO = 3.0

# Estimate inflation: either repeated mid-flight raises, or one big one.
ESTIMATE_INFLATION_MIN_EVENTS = 2
ESTIMATE_INFLATION_MIN_ADDED_HOURS = 4.0

# Ping-pong: backward moves authored, or repeat entries into a status that the
# board counts as resolved.
PINGPONG_MIN_BACKWARD = 3
PINGPONG_MIN_REPEAT_ENTRIES = 2

# Hidden rework: one ticket resolved twice and currently sitting resolved is
# already invisible to the reopened-count JQL, so one is enough to ask about.
HIDDEN_REWORK_MIN_TICKETS = 1


EVENT_COLUMNS = [
    "key",
    "entry_id",
    "ts",
    "author",
    "author_id",
    "field",
    "field_id",
    "from_string",
    "to_string",
    "from_id",
    "to_id",
    "is_status",
    "is_cosmetic",
    "is_estimate",
    "is_sprint_rollover",
    "from_stage",
    "to_stage",
]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _norm(value: object) -> str:
    """Normalised text for matching: lowercased, trimmed, whitespace collapsed."""
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _norm_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def stage_rank(status: object) -> int | None:
    """Where a status sits in the pipeline, or None when the status is unknown."""
    return STATUS_STAGES.get(_norm(status))


def _now(now: object | None) -> pd.Timestamp:
    if now is None:
        return pd.Timestamp.now(tz="UTC")
    stamp = pd.Timestamp(now)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _resolved_set(resolved_statuses: Iterable[str] | None) -> frozenset[str]:
    if resolved_statuses is None:
        return DEFAULT_RESOLVED_STATUSES
    return frozenset(_norm(s) for s in resolved_statuses if _norm(s))


def parse_duration_hours(value: object) -> float:
    """Hours out of whatever Jira put in a time-tracking changelog value.

    Jira is inconsistent here between tenants and between fields: the same edit
    arrives as ``"7200"`` (seconds, as a string) on one board and ``"2h"`` on
    another, and ``"1w 2d 3h"`` when somebody typed it that way. A bare number is
    read as seconds because that is what the API documents; anything with unit
    letters is read as a Jira duration with Jira's own week and day lengths
    (5 days, 8 hours). Returns NaN when the value is empty or unparseable, so a
    field this cannot read drops out of the arithmetic rather than reading zero.
    """
    text = str(value if value is not None else "").strip().lower()
    if not text or text in {"none", "nan", "null"}:
        return float("nan")
    try:
        return float(text) / 3600.0
    except ValueError:
        pass

    total = 0.0
    seen = False
    number = ""
    for char in text:
        if char.isdigit() or char in ".,":
            number += "." if char == "," else char
        elif char in _DURATION_UNITS:
            if number:
                try:
                    total += float(number) * _DURATION_UNITS[char]
                except ValueError:
                    return float("nan")
                seen = True
            number = ""
        elif char.isspace():
            continue
        else:
            return float("nan")
    if not seen:
        return float("nan")
    return total / 3600.0


def _is_sprint_rollover(field: str, from_string: object, to_string: object) -> bool:
    """A sprint field moving between two named sprints, i.e. automation.

    Mirrors ``jira_client._is_ignored_sprint_rollover_item`` in intent but stays
    local: this module must not import the network client, and the rule here is
    broader on purpose - any sprint-to-sprint move is board mechanics, whoever
    triggered it, and counting it as somebody's "activity" would credit the
    person who ran the sprint rollover with everyone's tickets.
    """
    if field not in {"sprint", "customfield_10020"}:
        return False
    return bool(_norm(from_string)) and bool(_norm(to_string))


def _histories_of(source: Any) -> list[dict[str, Any]]:
    """The ``histories`` list out of an issue, a changelog dict, or a bare list."""
    if source is None:
        return []
    if isinstance(source, list):
        return [h for h in source if isinstance(h, dict)]
    if not isinstance(source, dict):
        return []
    if "histories" in source:
        histories = source.get("histories") or []
        return [h for h in histories if isinstance(h, dict)]
    changelog = source.get("changelog") or {}
    if isinstance(changelog, dict):
        histories = changelog.get("histories") or []
        return [h for h in histories if isinstance(h, dict)]
    return []


def empty_events() -> pd.DataFrame:
    """A correctly typed, empty event frame - what every reader gets on no data."""
    frame = pd.DataFrame({column: pd.Series(dtype="object") for column in EVENT_COLUMNS})
    frame["ts"] = pd.Series(dtype="datetime64[ns, UTC]")
    for column in ("is_status", "is_cosmetic", "is_estimate", "is_sprint_rollover"):
        frame[column] = pd.Series(dtype="bool")
    for column in ("from_stage", "to_stage"):
        frame[column] = pd.Series(dtype="float64")
    return frame


# --------------------------------------------------------------------------
# The one parser everything else reads from
# --------------------------------------------------------------------------


def changelog_events(source: Any) -> pd.DataFrame:
    """Flatten Jira changelogs into one row per changed field.

    Accepts whatever shape the caller happens to be holding:

    - the raw ``issues`` list from ``JiraClient.search_issues`` (each dict with
      ``key`` and ``changelog.histories``),
    - a DataFrame carrying a ``key`` column and a ``changelog`` column (dict or
      histories list per row),
    - a mapping of ticket key to changelog or histories.

    The flexibility is not decoration. ``jira_client._issues_to_dataframe`` now
    carries the changelog through as a ``changelog`` column, but older callers
    still hold raw issue dicts and a snapshot loaded from disk may hold either;
    the same call works on all of them.

    One row per ``items[]`` entry, carrying the history entry's id, timestamp and
    author so a single save that touched five fields can be recognised as one
    action rather than five. Rows are sorted by ticket then time, which every
    downstream function relies on.
    """
    rows: list[dict[str, Any]] = []

    def _emit(key: object, histories: list[dict[str, Any]]) -> None:
        for index, history in enumerate(histories):
            author = history.get("author") or {}
            if not isinstance(author, dict):
                author = {}
            name = (
                author.get("displayName")
                or author.get("name")
                or author.get("emailAddress")
                or author.get("accountId")
                or "Unknown"
            )
            entry_id = history.get("id") or f"{key}:{index}"
            created = history.get("created")
            items = history.get("items") or []
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                field = _norm(item.get("field") or item.get("fieldId"))
                from_string = item.get("fromString")
                to_string = item.get("toString")
                rows.append(
                    {
                        "key": key,
                        "entry_id": str(entry_id),
                        "ts": created,
                        "author": str(name),
                        "author_id": author.get("accountId"),
                        "field": field,
                        "field_id": _norm(item.get("fieldId")) or field,
                        "from_string": from_string,
                        "to_string": to_string,
                        "from_id": item.get("from"),
                        "to_id": item.get("to"),
                        "is_status": field in _STATUS_FIELDS,
                        "is_cosmetic": field in COSMETIC_FIELDS,
                        "is_estimate": field in ESTIMATE_FIELDS,
                        "is_sprint_rollover": _is_sprint_rollover(
                            field, from_string, to_string
                        ),
                    }
                )

    if isinstance(source, pd.DataFrame):
        if source.empty or "changelog" not in source.columns:
            return empty_events()
        keys = source["key"] if "key" in source.columns else source.index
        for key, changelog in zip(keys, source["changelog"]):
            _emit(key, _histories_of(changelog))
    elif isinstance(source, Mapping):
        for key, changelog in source.items():
            _emit(key, _histories_of(changelog))
    elif isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
        for issue in source:
            if not isinstance(issue, dict):
                continue
            _emit(issue.get("key"), _histories_of(issue))
    else:
        return empty_events()

    if not rows:
        return empty_events()

    events = pd.DataFrame(rows)
    events["ts"] = pd.to_datetime(events["ts"], utc=True, errors="coerce", format="mixed")
    status_rows = events["is_status"]
    events["from_stage"] = pd.Series(
        [
            STATUS_STAGES.get(_norm(v)) if s else None
            for v, s in zip(events["from_string"], status_rows)
        ],
        index=events.index,
        dtype="float64",
    )
    events["to_stage"] = pd.Series(
        [
            STATUS_STAGES.get(_norm(v)) if s else None
            for v, s in zip(events["to_string"], status_rows)
        ],
        index=events.index,
        dtype="float64",
    )
    events = events.dropna(subset=["ts"])
    return events.sort_values(["key", "ts"]).reset_index(drop=True)[EVENT_COLUMNS]


def _window(events: pd.DataFrame, window_days: float | None, now: pd.Timestamp) -> pd.DataFrame:
    if events.empty or window_days is None:
        return events
    cutoff = now - pd.Timedelta(days=float(window_days))
    return events[events["ts"] >= cutoff]


def _join_unique(values: Iterable[Any]) -> str:
    """Names in first-seen order, deduplicated - ``dict`` keys, not a ``set``.

    Order matters in evidence: the first person to resolve a ticket and the
    fifth are different claims.
    """
    return ", ".join(dict.fromkeys(str(v) for v in values))


def _iso_list(values: Iterable[Any]) -> list[str]:
    return [pd.Timestamp(v).isoformat() for v in values]


def _evidence(pairs: Sequence[tuple[Any, Any]], limit: int = EVIDENCE_LIMIT) -> str:
    """``"VV-1 2026-08-01 09:12; VV-2 ... (+3 more)"`` - a flag's paper trail."""
    if not len(pairs):
        return ""
    shown = []
    for key, stamp in list(pairs)[:limit]:
        when = ""
        if stamp is not None and not (isinstance(stamp, float) and pd.isna(stamp)):
            stamp = pd.Timestamp(stamp)
            if not pd.isna(stamp):
                when = f" {stamp.strftime('%Y-%m-%d %H:%M')}"
        shown.append(f"{key}{when}")
    trail = "; ".join(shown)
    extra = len(pairs) - limit
    return f"{trail} (+{extra} more)" if extra > 0 else trail


# --------------------------------------------------------------------------
# 1. The honest staleness clock
# --------------------------------------------------------------------------


def status_age_days(
    tickets: pd.DataFrame,
    events: pd.DataFrame,
    *,
    now: object | None = None,
) -> pd.DataFrame:
    """Days since the ticket last actually moved, next to days since anyone touched it.

    ``transformations.add_ticket_health_fields`` computes ``idle_days`` from
    ``last_meaningful_activity``, which is the newest changelog entry of any
    kind. Adding a label to twenty tickets therefore resets twenty staleness
    clocks in five minutes - and 25 points of the KPI score (Weekly updates 15,
    Staleness 10) plus the whole stale-ticket queue hang off that number.

    ``status_age_days`` is the same clock driven only by status transitions.
    ``masked_days`` is the difference: how many days of apparent freshness came
    from edits that moved no work. A row with ``idle_days`` of 1 and
    ``status_age_days`` of 60 is a ticket that has not moved in two months and
    looks brand new on the board.

    Returns one row per ticket in ``tickets``: ``key``, ``assignee``, ``status``,
    ``last_status_change``, ``status_age_days``, ``idle_days``, ``masked_days``,
    ``status_changes`` (transitions seen in the changelog) and ``age_source``.

    What it cannot catch: a ticket whose changelog was truncated shows its oldest
    *surviving* transition, so its age is understated - ``age_source`` says
    ``"created"`` when no transition survived at all and the creation date had to
    stand in. And a status transition is only evidence that a status changed:
    bouncing a ticket In Progress -> To Do -> In Progress resets this clock too,
    which is why ``status_pingpong`` exists.
    """
    columns = [
        "key",
        "assignee",
        "status",
        "last_status_change",
        "status_age_days",
        "idle_days",
        "masked_days",
        "status_changes",
        "age_source",
    ]
    if tickets is None or tickets.empty or "key" not in tickets.columns:
        return pd.DataFrame(columns=columns)

    moment = _now(now)
    out = pd.DataFrame({"key": tickets["key"].values})
    out["assignee"] = (
        tickets["assignee"].fillna("Unassigned").astype(str).values
        if "assignee" in tickets.columns
        else "Unassigned"
    )
    out["status"] = (
        tickets["status"].astype(str).values if "status" in tickets.columns else ""
    )

    if events is None or events.empty:
        last_change = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
        changes = pd.Series(0, index=out.index, dtype="int64")
    else:
        status_events = events[events["is_status"].fillna(False).astype(bool)]
        grouped = status_events.groupby("key")["ts"]
        keys = out["key"].to_numpy()
        # ``reindex`` rather than ``map``: a ticket with no status history has to
        # come back as NaT, and mapping through an empty datetime series does not.
        last_change = pd.Series(
            grouped.max().reindex(keys).to_numpy(), index=out.index
        )
        last_change = pd.to_datetime(last_change, utc=True, errors="coerce")
        changes = (
            pd.Series(grouped.count().reindex(keys).to_numpy(), index=out.index)
            .fillna(0)
            .astype("int64")
        )

    created = (
        pd.to_datetime(tickets["created"], utc=True, errors="coerce").reset_index(drop=True)
        if "created" in tickets.columns
        else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    )

    anchor = last_change.fillna(created)
    out["age_source"] = [
        "status change" if pd.notna(l) else ("created" if pd.notna(c) else "unknown")
        for l, c in zip(last_change, created)
    ]
    out["last_status_change"] = last_change
    out["status_age_days"] = (
        (moment - anchor).dt.total_seconds().div(86400.0).clip(lower=0).round(1)
    )
    out["status_changes"] = changes

    if "idle_days" in tickets.columns:
        out["idle_days"] = (
            pd.to_numeric(tickets["idle_days"], errors="coerce").reset_index(drop=True).round(1)
        )
    else:
        out["idle_days"] = float("nan")
    out["masked_days"] = (out["status_age_days"] - out["idle_days"]).round(1)
    return out[columns]


# --------------------------------------------------------------------------
# 2. Grooming the board instead of working it
# --------------------------------------------------------------------------


def _entry_level(events: pd.DataFrame) -> pd.DataFrame:
    """One row per changelog save, with what that save actually did.

    A single Jira save can carry many ``items`` - status plus resolution plus
    assignee is one click, not three edits - so counting items would triple-count
    a real transition and make the cosmetic-to-transition ratio meaningless.
    """
    if events.empty:
        return pd.DataFrame(
            columns=[
                "key",
                "entry_id",
                "ts",
                "author",
                "has_status",
                "has_cosmetic",
                "has_estimate",
                "only_rollover",
                "fields",
            ]
        )
    grouped = events.groupby(["key", "entry_id", "ts", "author"], dropna=False, sort=False)
    out = grouped.agg(
        has_status=("is_status", "any"),
        has_cosmetic=("is_cosmetic", "any"),
        has_estimate=("is_estimate", "any"),
        only_rollover=("is_sprint_rollover", "all"),
        fields=("field", lambda values: ", ".join(sorted(set(values)))),
    ).reset_index()
    return out.sort_values(["ts", "key"]).reset_index(drop=True)


def cosmetic_touches(
    events: pd.DataFrame,
    *,
    window_days: float | None = 14.0,
    now: object | None = None,
) -> pd.DataFrame:
    """Per person, the saves in the window that changed fields but moved no work.

    A cosmetic touch is a changelog entry that carried at least one of
    ``COSMETIC_FIELDS`` and no status transition: a label added, a description
    reworded, a priority nudged, a ticket assigned to yourself and back. Each one
    resets ``idle_days`` on that ticket to zero and makes the board look tended.

    The signature to look for is not the raw count - a lead genuinely does groom
    the backlog - it is a high cosmetic count next to a low
    ``status_transitions`` count over the same window, which is why both are
    returned side by side along with ``cosmetic_per_transition``. The
    ``busiest_day`` columns exist so a burst is visible: forty touches spread
    over two weeks is curation, forty in one afternoon the day before a review is
    something else.

    Attribution is by ``history["author"]``, never by current assignee: this
    measures who did the editing.

    Returns one row per person: ``person``, ``cosmetic_touches``,
    ``cosmetic_tickets``, ``status_transitions``, ``status_tickets``,
    ``cosmetic_per_transition``, ``assignee_roundtrips``, ``busiest_day``,
    ``busiest_day_touches``, ``first_touch``, ``last_touch``, ``keys``,
    ``timestamps`` (list) and ``evidence``.

    What it cannot catch: it cannot read intent. Rewriting a vague ticket into
    something Devin can act on is exactly the work Angel has been asking for, and
    it lands here as a cosmetic touch. It also cannot see edits made outside
    Jira, and it says nothing about whether the person did real work elsewhere -
    a low count is not a compliment, only the absence of one signal. Automated
    sprint rollovers are excluded, but any other automation running under a human
    account will be credited to that human.
    """
    columns = [
        "person",
        "cosmetic_touches",
        "cosmetic_tickets",
        "status_transitions",
        "status_tickets",
        "cosmetic_per_transition",
        "assignee_roundtrips",
        "busiest_day",
        "busiest_day_touches",
        "first_touch",
        "last_touch",
        "keys",
        "timestamps",
        "evidence",
    ]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    moment = _now(now)
    scoped = _window(events, window_days, moment)
    entries = _entry_level(scoped)
    if entries.empty:
        return pd.DataFrame(columns=columns)

    # A save that only rolled a sprint over is board mechanics, not a touch.
    entries = entries[~entries["only_rollover"].fillna(False).astype(bool)]
    cosmetic = entries[
        entries["has_cosmetic"].fillna(False).astype(bool)
        & ~entries["has_status"].fillna(False).astype(bool)
    ]
    transitions = entries[entries["has_status"].fillna(False).astype(bool)]

    roundtrips = _assignee_roundtrips(scoped)
    people = sorted(
        set(entries["author"].dropna().astype(str))
        | set(transitions["author"].dropna().astype(str))
    )

    rows: list[dict[str, Any]] = []
    for person in people:
        mine = cosmetic[cosmetic["author"] == person].sort_values("ts")
        moved = transitions[transitions["author"] == person]
        touches = int(len(mine))
        moves = int(len(moved))
        if touches == 0 and moves == 0:
            continue
        by_day = (
            mine["ts"].dt.tz_convert("UTC").dt.date.value_counts()
            if touches
            else pd.Series(dtype="int64")
        )
        pairs = list(zip(mine["key"], mine["ts"]))
        rows.append(
            {
                "person": person,
                "cosmetic_touches": touches,
                "cosmetic_tickets": int(mine["key"].nunique()),
                "status_transitions": moves,
                "status_tickets": int(moved["key"].nunique()),
                "cosmetic_per_transition": (
                    round(touches / moves, 2) if moves else float("nan")
                ),
                "assignee_roundtrips": int(
                    (roundtrips["author"] == person).sum() if not roundtrips.empty else 0
                ),
                "busiest_day": str(by_day.index[0]) if touches else "",
                "busiest_day_touches": int(by_day.iloc[0]) if touches else 0,
                "first_touch": mine["ts"].min() if touches else pd.NaT,
                "last_touch": mine["ts"].max() if touches else pd.NaT,
                "keys": ", ".join(sorted(set(str(k) for k in mine["key"]))),
                "timestamps": [pd.Timestamp(t).isoformat() for t in mine["ts"]],
                "evidence": _evidence(pairs),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)[columns]
    return out.sort_values(
        ["cosmetic_touches", "status_transitions"], ascending=[False, True]
    ).reset_index(drop=True)


def _assignee_roundtrips(events: pd.DataFrame) -> pd.DataFrame:
    """Assignee changes that put a ticket back on someone it had already left.

    Taking a ticket, then handing it back, is the cheapest way to generate two
    changelog entries and reset the idle clock twice without touching the code.
    Returns ``key``, ``ts``, ``author`` for the closing half of each round trip.
    """
    if events.empty:
        return pd.DataFrame(columns=["key", "ts", "author"])
    moves = events[events["field"] == "assignee"].sort_values(["key", "ts"])
    rows: list[dict[str, Any]] = []
    for key, group in moves.groupby("key", sort=False):
        seen_from: set[str] = set()
        for _, row in group.iterrows():
            destination = _norm(row["to_string"])
            if destination and destination in seen_from:
                rows.append({"key": key, "ts": row["ts"], "author": row["author"]})
            origin = _norm(row["from_string"])
            if origin:
                seen_from.add(origin)
    return pd.DataFrame(rows, columns=["key", "ts", "author"])


# --------------------------------------------------------------------------
# 3. Estimates raised after the work started
# --------------------------------------------------------------------------


def _first_started(events: pd.DataFrame) -> pd.Series:
    """Per ticket, the first moment it entered a status at or past In Progress."""
    if events.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    started = events[
        events["is_status"].fillna(False).astype(bool)
        & (events["to_stage"] >= STARTED_RANK)
    ]
    if started.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return started.groupby("key")["ts"].min()


def _status_at(edits: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """For each row of ``edits``, the status its ticket sat in at that moment.

    An as-of join rather than a scan per edit: a quarter of one board is tens of
    thousands of changelog rows, and the naive version is quadratic on it.
    """
    if edits.empty:
        return pd.Series(dtype="object")
    status_events = events[events["is_status"].fillna(False).astype(bool)]
    if status_events.empty:
        return pd.Series("", index=edits.index, dtype="object")
    left = edits[["key", "ts"]].reset_index().sort_values("ts")
    right = (
        status_events[["key", "ts", "to_string"]]
        .assign(to_string=lambda frame: frame["to_string"].fillna("").astype(str))
        .sort_values("ts")
    )
    merged = pd.merge_asof(left, right, on="ts", by="key", direction="backward")
    return (
        merged.set_index("index")["to_string"].reindex(edits.index).fillna("").astype(str)
    )


def estimate_churn(
    events: pd.DataFrame,
    *,
    window_days: float | None = 90.0,
    now: object | None = None,
    include_pre_start: bool = False,
) -> pd.DataFrame:
    """Estimate edits made after the ticket was already being worked.

    On an hourly contract this is the most direct padding signal the board can
    produce. An estimate written while the ticket is still in Backlog or To Do is
    planning and is excluded by default. The same edit made after the ticket
    first entered In Progress is a revision of the bill, made by someone who by
    then knows how many hours they have already booked.

    Returns one row per edit: ``key``, ``ts``, ``author``, ``field``,
    ``old_value`` / ``new_value`` (exactly as Jira recorded them),
    ``old_hours`` / ``new_hours`` / ``delta_hours``, ``direction``
    (``raised`` / ``lowered`` / ``set`` / ``cleared`` / ``unchanged``),
    ``started_at``, ``days_after_start`` and ``status_at_change`` - so the row
    reads as a sentence: "raised VV-42 from 4h to 12h nine days after starting it,
    while it sat in Code Review".

    Pass ``include_pre_start=True`` to see planning-time edits too; they carry
    ``days_after_start`` as NaN.

    What it cannot catch: story points (unitless, deliberately not converted to
    hours), an estimate that was simply wrong and honestly corrected upward -
    which is a legitimate and common thing - and the much simpler dodge of
    setting a padded estimate once, at the start, and never touching it again.
    That last one is not visible in the changelog at all; it needs
    ``cycle_time`` and a human comparing estimate against delivered work. A
    ticket first estimated after work began shows as ``set``, not ``raised``,
    because there is no prior number to compare against.
    """
    columns = [
        "key",
        "ts",
        "author",
        "field",
        "old_value",
        "new_value",
        "old_hours",
        "new_hours",
        "delta_hours",
        "direction",
        "started_at",
        "days_after_start",
        "status_at_change",
    ]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    moment = _now(now)
    scoped = _window(events, window_days, moment)
    edits = scoped[scoped["is_estimate"].fillna(False).astype(bool)]
    if edits.empty:
        return pd.DataFrame(columns=columns)

    # Start times come from the full history, not the window: a ticket started
    # four months ago and re-estimated yesterday is exactly the case in question.
    started = _first_started(events)

    out = edits[["key", "ts", "author", "field"]].copy()
    out["old_value"] = edits["from_string"].values
    out["new_value"] = edits["to_string"].values
    out["started_at"] = pd.to_datetime(
        pd.Series(started.reindex(out["key"].to_numpy()).to_numpy(), index=out.index),
        utc=True,
        errors="coerce",
    )
    after_start = out["started_at"].notna() & (out["ts"] >= out["started_at"])
    if not include_pre_start:
        out = out[after_start].copy()
        after_start = after_start.loc[out.index]
    if out.empty:
        return pd.DataFrame(columns=columns)

    out["old_hours"] = [parse_duration_hours(v) for v in out["old_value"]]
    out["new_hours"] = [parse_duration_hours(v) for v in out["new_value"]]
    out["delta_hours"] = (out["new_hours"] - out["old_hours"]).round(2)
    # "set" and "cleared" are kept apart from "raised": a number that appeared
    # out of nothing has no prior to be a raise against, and calling it one
    # would be the first exaggeration in a table meant to survive an argument.
    direction = pd.Series("unchanged", index=out.index, dtype="object")
    direction = direction.mask(out["old_hours"].isna() & out["new_hours"].notna(), "set")
    direction = direction.mask(out["old_hours"].notna() & out["new_hours"].isna(), "cleared")
    direction = direction.mask(out["delta_hours"] > 0, "raised")
    direction = direction.mask(out["delta_hours"] < 0, "lowered")
    out["direction"] = direction
    out["old_hours"] = out["old_hours"].round(2)
    out["new_hours"] = out["new_hours"].round(2)
    out["days_after_start"] = (
        (out["ts"] - out["started_at"]).dt.total_seconds().div(86400.0).round(1)
    ).where(after_start)
    out["status_at_change"] = _status_at(out, events)
    return out[columns].sort_values("ts").reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Rework that healed, and therefore vanished
# --------------------------------------------------------------------------


def reresolve_events(
    events: pd.DataFrame,
    tickets: pd.DataFrame | None = None,
    *,
    window_days: float | None = 90.0,
    now: object | None = None,
    resolved_statuses: Iterable[str] | None = None,
) -> pd.DataFrame:
    """How many times each ticket was declared done, counted from the changelog.

    The dashboard's rework number comes from ``app._reopened_jql``:
    ``status CHANGED FROM (resolved) AFTER -Nd AND status NOT IN (resolved)``.
    That asks a question about the present - which tickets are broken *right now*
    - and calls the answer rework. A ticket resolved in March, reopened in April
    and re-resolved in May is not in that answer at all. The bounce cost real
    hours and left no trace, and Rework is 20% of the KPI score, the largest
    quality signal in it.

    Counting entries into resolved from the changelog instead makes every bounce
    permanent. An entry counts only when the ticket came from a status the board
    does *not* consider resolved, so a normal walk up the tail of the pipeline
    (Review in Staging -> Ready for Production -> Released, all three "resolved"
    by the team's list) counts once, not three times.

    Returns one row per ticket that entered a resolved status in the window:
    ``key``, ``resolutions``, ``reopens``, ``first_resolved``, ``last_resolved``,
    ``resolvers``, ``reopeners``, ``currently_resolved``, ``hidden_rework``
    (resolved more than once and sitting resolved now, i.e. invisible to the JQL)
    and ``timestamps``.

    Pass ``tickets`` to read ``currently_resolved`` from the ticket's live status;
    without it the last transition in the changelog stands in, which is wrong for
    any ticket whose history was truncated.

    What it cannot catch: a bounce that never touched the status field - a ticket
    called done, quietly fixed, and never moved back - and any bounce older than
    the changelog Jira returned. It also cannot tell an honest reopen (the tester
    found a real bug) from a process artefact (the ticket was resolved by mistake
    and un-resolved a minute later); ``timestamps`` is there so a two-minute
    round trip can be recognised as noise by eye.
    """
    columns = [
        "key",
        "resolutions",
        "reopens",
        "first_resolved",
        "last_resolved",
        "resolvers",
        "reopeners",
        "currently_resolved",
        "hidden_rework",
        "timestamps",
    ]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    moment = _now(now)
    resolved = _resolved_set(resolved_statuses)
    scoped = _window(events, window_days, moment)
    status_events = scoped[scoped["is_status"].fillna(False).astype(bool)].copy()
    if status_events.empty:
        return pd.DataFrame(columns=columns)

    status_events["_from_resolved"] = _norm_series(status_events["from_string"]).isin(resolved)
    status_events["_to_resolved"] = _norm_series(status_events["to_string"]).isin(resolved)
    entries = status_events[status_events["_to_resolved"] & ~status_events["_from_resolved"]]
    exits = status_events[status_events["_from_resolved"] & ~status_events["_to_resolved"]]
    if entries.empty:
        return pd.DataFrame(columns=columns)

    entries = entries.sort_values("ts")
    exits = exits.sort_values("ts")
    out = (
        entries.groupby("key", sort=False)
        .agg(
            resolutions=("ts", "count"),
            first_resolved=("ts", "min"),
            last_resolved=("ts", "max"),
            resolvers=("author", _join_unique),
            timestamps=("ts", _iso_list),
        )
        .reset_index()
    )
    reopened = (
        exits.groupby("key", sort=False)
        .agg(reopens=("ts", "count"), reopeners=("author", _join_unique))
        .reset_index()
    )
    out = out.merge(reopened, on="key", how="left")
    out["reopens"] = out["reopens"].fillna(0).astype("int64")
    out["reopeners"] = out["reopeners"].fillna("")

    # Where the ticket sits today. The live status is the truth when the caller
    # passes the board; the last transition in the changelog is the fallback, and
    # it is wrong for any ticket whose history Jira truncated.
    last_state = (
        status_events.sort_values("ts").groupby("key", sort=False)["_to_resolved"].last()
    )
    now_resolved = pd.Series(
        last_state.reindex(out["key"].to_numpy()).to_numpy(), index=out.index
    ).fillna(False)
    if tickets is not None and not tickets.empty and {"key", "status"} <= set(tickets.columns):
        live = _norm_series(tickets["status"]).isin(resolved)
        live.index = tickets["key"].to_numpy()
        live = live[~live.index.duplicated()]
        from_board = pd.Series(
            live.reindex(out["key"].to_numpy()).to_numpy(), index=out.index
        )
        now_resolved = from_board.where(from_board.notna(), now_resolved)
    out["currently_resolved"] = now_resolved.fillna(False).astype(bool)
    out["hidden_rework"] = (out["resolutions"] > 1) & out["currently_resolved"]

    return (
        out[columns]
        .sort_values(["resolutions", "reopens"], ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# 5. Circles that mint resolution credit
# --------------------------------------------------------------------------


def _backward_moves(events: pd.DataFrame) -> pd.DataFrame:
    """Every transition whose destination ranks below its origin, with its author."""
    if events.empty:
        return pd.DataFrame(columns=["key", "ts", "author", "from_string", "to_string"])
    status_events = events[events["is_status"].fillna(False).astype(bool)]
    backward = status_events[
        status_events["from_stage"].notna()
        & status_events["to_stage"].notna()
        & (status_events["to_stage"] < status_events["from_stage"])
    ]
    return backward[["key", "ts", "author", "from_string", "to_string"]].reset_index(drop=True)


def status_pingpong(
    events: pd.DataFrame,
    *,
    window_days: float | None = 90.0,
    now: object | None = None,
    resolved_statuses: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Tickets that went backward, or entered the same status more than once.

    "Review in Staging" counts as resolved on this board. A ticket moved into
    staging, back to In Progress, and into staging again has been counted
    resolved twice, and nothing anywhere decrements the first count. The same
    trick works on any status the resolved list contains that is not actually
    terminal - Ready for Production is the other one.

    Two counts, deliberately separate. ``backward_transitions`` is direction:
    moves down the pipeline, using ``STATUS_STAGES``. ``repeat_entries`` is
    repetition: for each status, every entry after the first. A ticket can score
    on the second without the first if the workflow loops sideways through
    statuses this module does not rank.

    ``staging_entries`` narrows repetition to statuses the board counts as
    resolved but that are not terminal - the ones that mint credit.

    Returns one row per ticket with a loop in it: ``key``,
    ``backward_transitions``, ``repeat_entries``, ``staging_entries``,
    ``most_repeated_status``, ``movers`` (authors of the backward moves),
    ``first_backward``, ``last_backward`` and ``timestamps``.

    What it cannot catch: statuses missing from ``STATUS_STAGES`` have no rank,
    so their moves are never "backward" - a workflow this module has not been
    taught is invisible to the direction half of the check, though the repetition
    half still sees it. It also cannot distinguish a ticket that legitimately
    failed staging three times (a hard bug) from one bounced deliberately; that
    difference lives in the PR and the comments, not the changelog.
    """
    columns = [
        "key",
        "backward_transitions",
        "repeat_entries",
        "staging_entries",
        "most_repeated_status",
        "movers",
        "first_backward",
        "last_backward",
        "timestamps",
    ]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    moment = _now(now)
    resolved = _resolved_set(resolved_statuses)
    scoped = _window(events, window_days, moment)
    status_events = scoped[scoped["is_status"].fillna(False).astype(bool)]
    if status_events.empty:
        return pd.DataFrame(columns=columns)

    backward = _backward_moves(status_events)

    # Entries into each status, per ticket. Every entry after the first is a
    # repeat; the ones that land on a status the board reads as resolved, and
    # that is not terminal, are the ones that mint credit twice.
    landings = status_events.assign(_to=_norm_series(status_events["to_string"]))
    counted = (
        landings.groupby(["key", "_to"], sort=False).size().reset_index(name="entries")
    )
    counted["repeats"] = (counted["entries"] - 1).clip(lower=0)
    counted["staging_repeats"] = counted["repeats"].where(
        counted["_to"].isin(_credit_statuses(resolved)), 0
    )
    per_key = counted.groupby("key", sort=False).agg(
        repeat_entries=("repeats", "sum"),
        staging_entries=("staging_repeats", "sum"),
    )
    ranked = counted.sort_values("entries", ascending=False).drop_duplicates("key")
    most_repeated = ranked.set_index("key").apply(
        lambda row: str(row["_to"]) if row["entries"] > 1 else "", axis=1
    )

    per_key_backward = (
        backward.groupby("key", sort=False).agg(
            backward_transitions=("ts", "count"),
            movers=("author", _join_unique),
            first_backward=("ts", "min"),
            last_backward=("ts", "max"),
            timestamps=("ts", _iso_list),
        )
        if len(backward)
        else pd.DataFrame(
            columns=[
                "backward_transitions",
                "movers",
                "first_backward",
                "last_backward",
                "timestamps",
            ]
        )
    )

    out = per_key.join(per_key_backward, how="outer").reset_index()
    out["backward_transitions"] = out["backward_transitions"].fillna(0).astype("int64")
    out["repeat_entries"] = out["repeat_entries"].fillna(0).astype("int64")
    out["staging_entries"] = out["staging_entries"].fillna(0).astype("int64")
    out["movers"] = out["movers"].fillna("")
    out["timestamps"] = out["timestamps"].apply(lambda v: v if isinstance(v, list) else [])
    out["most_repeated_status"] = (
        pd.Series(
            most_repeated.reindex(out["key"].to_numpy()).to_numpy(), index=out.index
        )
        .fillna("")
        .astype(str)
    )
    # A ticket that walked forward once has nothing to answer for.
    out = out[(out["backward_transitions"] > 0) | (out["repeat_entries"] > 0)]
    if out.empty:
        return pd.DataFrame(columns=columns)
    return (
        out[columns]
        .sort_values(
            ["staging_entries", "backward_transitions", "repeat_entries"], ascending=False
        )
        .reset_index(drop=True)
    )


def _credit_statuses(resolved: frozenset[str]) -> frozenset[str]:
    """Resolved statuses that are not the end of the line.

    "Review in Staging" and "Ready for Production" both count as resolved on this
    board and neither means the work is finished, so re-entering them is how
    resolution credit gets minted more than once for the same ticket.
    """
    return frozenset(
        status for status in resolved if (STATUS_STAGES.get(status) or 0) < 7
    )


# --------------------------------------------------------------------------
# 6. The metric you can only improve by finishing work
# --------------------------------------------------------------------------


class CycleTime(NamedTuple):
    """``detail``: one row per ticket and status. ``by_person``: the medians."""

    detail: pd.DataFrame
    by_person: pd.DataFrame


def cycle_time(
    events: pd.DataFrame,
    tickets: pd.DataFrame | None = None,
    *,
    now: object | None = None,
    resolved_statuses: Iterable[str] | None = None,
) -> CycleTime:
    """Time each ticket spent in each status, and the medians per person.

    Derived from consecutive status transitions: the ticket entered a status at
    one timestamp and left it at the next. The interval before the first recorded
    transition is included when ``tickets`` supplies ``created``; the interval
    after the last one runs to ``now`` and is marked ``is_open``.

    This is the metric that cannot be gamed by editing fields. Labels, estimates,
    priority, descriptions - none of them move it. It only improves when work
    actually finishes, and it exposes the two failure modes the board hides:
    tickets that sit in Code Review for three weeks, and tickets whose lead time
    bears no relation to the hours billed against them.

    ``detail`` columns: ``key``, ``assignee``, ``status``, ``entered``,
    ``exited``, ``days``, ``is_open``.
    ``by_person`` columns: ``person``, ``tickets``, ``lead_time_tickets``,
    ``median_lead_time_days`` (first start to first resolve),
    ``median_in_progress_days``, ``median_review_days``, ``open_tickets``,
    ``median_open_status_days``.

    Attribution: the ticket's current assignee when ``tickets`` provides one,
    otherwise whoever first moved it into progress. Both are approximations - a
    ticket handed over halfway attributes its whole history to one person - and
    that is the honest limit of a board that records assignment, not effort.

    What it cannot catch: waiting that happens off the board (a ticket "In
    Progress" while its owner is blocked on someone else's review shows as work),
    and it says nothing about hours. A two-day cycle time billed at thirty hours
    is a question this function raises and cannot answer.
    """
    detail_columns = ["key", "assignee", "status", "entered", "exited", "days", "is_open"]
    person_columns = [
        "person",
        "tickets",
        "lead_time_tickets",
        "median_lead_time_days",
        "median_in_progress_days",
        "median_review_days",
        "open_tickets",
        "median_open_status_days",
    ]
    if events is None or events.empty:
        return CycleTime(pd.DataFrame(columns=detail_columns), pd.DataFrame(columns=person_columns))

    moment = _now(now)
    resolved = _resolved_set(resolved_statuses)
    status_events = events[events["is_status"].fillna(False).astype(bool)].sort_values(
        ["key", "ts"]
    )
    if status_events.empty:
        return CycleTime(pd.DataFrame(columns=detail_columns), pd.DataFrame(columns=person_columns))

    created: dict[Any, pd.Timestamp] = {}
    assignees: dict[Any, str] = {}
    if tickets is not None and not tickets.empty and "key" in tickets.columns:
        if "created" in tickets.columns:
            created = dict(
                zip(tickets["key"], pd.to_datetime(tickets["created"], utc=True, errors="coerce"))
            )
        if "assignee" in tickets.columns:
            assignees = dict(
                zip(tickets["key"], tickets["assignee"].fillna("Unassigned").astype(str))
            )

    starters = (
        status_events[status_events["to_stage"] >= STARTED_RANK]
        .groupby("key")["author"]
        .first()
        .to_dict()
    )

    def _owner(key: object) -> str:
        return assignees.get(key) or starters.get(key) or "Unassigned"

    # A ticket left each status when it entered the next one, so the exit time
    # is the following transition of the same ticket - one shift, not a loop.
    spans = status_events[["key", "ts", "from_string", "to_string"]].copy()
    spans["entered"] = spans["ts"]
    spans["exited"] = spans.groupby("key", sort=False)["ts"].shift(-1)
    spans["is_open"] = spans["exited"].isna()
    spans["exited"] = spans["exited"].fillna(moment)
    spans["status"] = spans["to_string"].fillna("").astype(str)

    # The interval before the first recorded transition: only knowable when the
    # board says when the ticket was created.
    first_moves = spans.drop_duplicates("key", keep="first").copy()
    first_moves["entered"] = pd.to_datetime(
        pd.Series(
            [created.get(key) for key in first_moves["key"]], index=first_moves.index
        ),
        utc=True,
        errors="coerce",
    )
    first_moves["exited"] = first_moves["ts"]
    first_moves["status"] = first_moves["from_string"].fillna("").astype(str)
    first_moves["is_open"] = False
    first_moves = first_moves[
        first_moves["entered"].notna() & first_moves["status"].ne("")
    ]

    detail = pd.concat([first_moves, spans], ignore_index=True)
    detail["assignee"] = [_owner(key) for key in detail["key"]]
    detail["days"] = (
        (detail["exited"] - detail["entered"])
        .dt.total_seconds()
        .div(86400.0)
        .clip(lower=0)
        .round(2)
    )
    detail = detail.sort_values(["key", "entered"])[detail_columns]

    stage = detail["status"].map(stage_rank)
    detail_stage = detail.assign(_stage=stage)

    # Lead time: first start to the first resolution that followed it.
    starts = status_events[status_events["to_stage"] >= STARTED_RANK].groupby("key")["ts"].min()
    finishes = status_events[
        _norm_series(status_events["to_string"]).isin(resolved)
    ][["key", "ts"]]
    lead: dict[Any, float] = {}
    if len(starts) and not finishes.empty:
        paired = finishes.assign(
            _start=pd.Series(
                starts.reindex(finishes["key"].to_numpy()).to_numpy(),
                index=finishes.index,
            )
        )
        paired = paired[paired["_start"].notna() & (paired["ts"] >= paired["_start"])]
        if not paired.empty:
            first_finish = paired.groupby("key", sort=False).agg(
                _finish=("ts", "min"), _start=("_start", "first")
            )
            lead = (
                (first_finish["_finish"] - first_finish["_start"])
                .dt.total_seconds()
                .div(86400.0)
                .to_dict()
            )

    # Coding + dev-env time and review time are kept apart because they fail
    # differently: a long In Progress is work or a stuck engineer, a long Code
    # Review is usually the team not reviewing each other.
    detail_stage = detail_stage.assign(
        _in_progress=detail_stage["days"].where(
            detail_stage["_stage"].between(STARTED_RANK, 3), 0.0
        ),
        _review=detail_stage["days"].where(detail_stage["_stage"].between(4, 5), 0.0),
        _open=detail_stage["days"].where(detail_stage["is_open"], 0.0),
    )
    per_ticket = (
        detail_stage.groupby(["key", "assignee"], sort=False)
        .agg(
            in_progress_days=("_in_progress", "sum"),
            review_days=("_review", "sum"),
            open_status_days=("_open", "sum"),
            is_open=("is_open", "any"),
        )
        .reset_index()
    )
    per_ticket["lead_time_days"] = per_ticket["key"].map(lead)

    grouped = per_ticket.groupby("assignee", sort=False)
    by_person = (
        grouped.agg(
            tickets=("key", "nunique"),
            lead_time_tickets=("lead_time_days", "count"),
            median_lead_time_days=("lead_time_days", "median"),
            median_in_progress_days=("in_progress_days", "median"),
            median_review_days=("review_days", "median"),
            open_tickets=("is_open", "sum"),
        )
        .reset_index()
        .rename(columns={"assignee": "person"})
    )
    still_open = per_ticket[per_ticket["is_open"].fillna(False).astype(bool)]
    by_person["median_open_status_days"] = by_person["person"].map(
        still_open.groupby("assignee")["open_status_days"].median()
    )
    by_person["open_tickets"] = by_person["open_tickets"].astype("int64")
    for column in (
        "median_lead_time_days",
        "median_in_progress_days",
        "median_review_days",
        "median_open_status_days",
    ):
        by_person[column] = by_person[column].round(2)

    return CycleTime(detail.reset_index(drop=True), by_person[person_columns])


# --------------------------------------------------------------------------
# 7. The four questions, with their evidence
# --------------------------------------------------------------------------


FLAG_NAMES = ("board_grooming", "estimate_inflation", "staging_pingpong", "rework_hidden")

FLAG_MEANINGS = {
    "board_grooming": (
        "Many field-only edits, few status transitions: the board was tended, "
        "not worked. Ask what shipped."
    ),
    "estimate_inflation": (
        "Estimates raised after the work had already started. On an hourly "
        "contract, ask what changed about the scope."
    ),
    "staging_pingpong": (
        "Tickets moved backward or re-entered a status the board counts as "
        "resolved. Each re-entry mints resolution credit that is never removed."
    ),
    "rework_hidden": (
        "Tickets resolved more than once that are sitting resolved now, so the "
        "reopened-count JQL cannot see the bounce at all."
    ),
}


def integrity_flags(
    tickets: pd.DataFrame,
    events: pd.DataFrame,
    *,
    window_days: float = 30.0,
    now: object | None = None,
    resolved_statuses: Iterable[str] | None = None,
) -> pd.DataFrame:
    """One row per person: the named flags they tripped, and the proof.

    Four flags, no score. A single number would be argued with and could not be
    acted on; a flag with ticket keys and timestamps behind it is a question with
    a first sentence already written. Thresholds are the module constants above,
    and every flag column is paired with an ``_evidence`` column that names the
    tickets and the times.

    - ``board_grooming``: at least ``COSMETIC_BURST_TOUCHES`` field-only edits in
      the window, with cosmetic edits outnumbering status transitions by
      ``COSMETIC_TO_TRANSITION_RATIO`` (or no transitions at all).
    - ``estimate_inflation``: ``ESTIMATE_INFLATION_MIN_EVENTS`` mid-flight raises,
      or ``ESTIMATE_INFLATION_MIN_ADDED_HOURS`` hours added after work started.
    - ``staging_pingpong``: ``PINGPONG_MIN_BACKWARD`` backward transitions
      authored, or ``PINGPONG_MIN_REPEAT_ENTRIES`` re-entries into a status the
      board counts as resolved.
    - ``rework_hidden``: they resolved at least
      ``HIDDEN_REWORK_MIN_TICKETS`` ticket more than once where the ticket is
      resolved today - invisible to the rework metric on the scorecard.

    Attribution is by changelog author throughout, so the rows cover everybody
    who edited the board in the window, including people who hold no tickets.

    What it cannot catch, said plainly: none of these prove padding. Each one has
    an innocent reading - a lead grooming the backlog, an estimate honestly
    corrected, a hard bug that failed staging twice, a bounce nobody hid on
    purpose. They are conversation starters, and a flag that survives the
    conversation is the finding, not the flag itself. Equally, an empty row is
    not a clean bill of health: someone who does nothing at all trips nothing at
    all, which is what ``cycle_time`` and the delivery components are for.
    """
    columns = ["person", "flags", "flag_count"]
    for flag in FLAG_NAMES:
        columns += [flag, f"{flag}_evidence"]
    columns += [
        "cosmetic_touches",
        "status_transitions",
        "estimate_raises",
        "hours_added",
        "backward_moves",
        "reresolved_tickets",
        "window_days",
    ]

    moment = _now(now)
    resolved = _resolved_set(resolved_statuses)
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)
    scoped = _window(events, window_days, moment)
    if scoped.empty:
        return pd.DataFrame(columns=columns)

    touches = cosmetic_touches(scoped, window_days=None, now=moment)
    # The full history goes to ``estimate_churn`` even though only edits inside
    # the window are wanted: a ticket started four months ago and re-estimated
    # last week has to know when it started before it can call that a raise.
    churn = estimate_churn(events, window_days=window_days, now=moment)
    bounces = reresolve_events(
        scoped, tickets, window_days=None, now=moment, resolved_statuses=resolved
    )
    backward = _backward_moves(scoped[scoped["is_status"].fillna(False).astype(bool)])

    # Re-entries into a resolved-but-not-terminal status, attributed to whoever
    # made the move rather than to the ticket's owner.
    status_events = scoped[scoped["is_status"].fillna(False).astype(bool)].sort_values(
        ["key", "ts"]
    )
    if status_events.empty:
        staging = pd.DataFrame(columns=["key", "ts", "author"])
    else:
        landings = status_events.assign(_to=_norm_series(status_events["to_string"]))
        # cumcount over (ticket, destination): 0 is the first arrival, anything
        # above it is a return trip.
        landings["_visit"] = landings.groupby(["key", "_to"], sort=False).cumcount()
        staging = landings[
            (landings["_visit"] > 0) & landings["_to"].isin(_credit_statuses(resolved))
        ][["key", "ts", "author"]].reset_index(drop=True)

    people = sorted(
        set(scoped["author"].dropna().astype(str))
        | (set(touches["person"].astype(str)) if not touches.empty else set())
    )

    rows: list[dict[str, Any]] = []
    for person in people:
        mine_touches = touches[touches["person"] == person] if not touches.empty else touches
        cosmetic_count = int(mine_touches["cosmetic_touches"].iloc[0]) if len(mine_touches) else 0
        transitions = int(mine_touches["status_transitions"].iloc[0]) if len(mine_touches) else 0
        touch_evidence = str(mine_touches["evidence"].iloc[0]) if len(mine_touches) else ""

        mine_churn = (
            churn[(churn["author"] == person) & (churn["direction"] == "raised")]
            if not churn.empty
            else churn
        )
        raises = int(len(mine_churn))
        hours_added = (
            float(pd.to_numeric(mine_churn["delta_hours"], errors="coerce").fillna(0.0).sum())
            if raises
            else 0.0
        )

        mine_backward = backward[backward["author"] == person] if len(backward) else backward
        mine_staging = staging[staging["author"] == person] if len(staging) else staging

        # Exact name match against the resolver list, not a substring search:
        # "Ana" must not inherit "Anastasia"'s bounced tickets.
        hidden = (
            bounces[
                bounces["hidden_rework"].fillna(False).astype(bool)
                & bounces["resolvers"].map(
                    lambda names: person in str(names).split(", ")
                )
            ]
            if not bounces.empty
            else bounces
        )

        flag_values = {
            "board_grooming": cosmetic_count >= COSMETIC_BURST_TOUCHES
            and (
                transitions == 0
                or cosmetic_count / max(transitions, 1) >= COSMETIC_TO_TRANSITION_RATIO
            ),
            "estimate_inflation": raises >= ESTIMATE_INFLATION_MIN_EVENTS
            or hours_added >= ESTIMATE_INFLATION_MIN_ADDED_HOURS,
            "staging_pingpong": len(mine_backward) >= PINGPONG_MIN_BACKWARD
            or len(mine_staging) >= PINGPONG_MIN_REPEAT_ENTRIES,
            "rework_hidden": len(hidden) >= HIDDEN_REWORK_MIN_TICKETS,
        }
        evidence = {
            "board_grooming": (
                f"{cosmetic_count} field-only edits vs {transitions} status moves: "
                f"{touch_evidence}"
                if flag_values["board_grooming"]
                else ""
            ),
            "estimate_inflation": (
                f"+{hours_added:.1f}h over {raises} mid-flight raise(s): "
                + _evidence(list(zip(mine_churn["key"], mine_churn["ts"])))
                if flag_values["estimate_inflation"]
                else ""
            ),
            "staging_pingpong": (
                f"{len(mine_backward)} backward move(s), {len(mine_staging)} re-entry "
                "into a resolved status: "
                + _evidence(
                    list(zip(mine_backward["key"], mine_backward["ts"]))
                    + list(zip(mine_staging["key"], mine_staging["ts"]))
                )
                if flag_values["staging_pingpong"]
                else ""
            ),
            "rework_hidden": (
                f"{len(hidden)} ticket(s) resolved more than once and resolved now: "
                + _evidence(list(zip(hidden["key"], hidden["last_resolved"])))
                if flag_values["rework_hidden"]
                else ""
            ),
        }

        tripped = [flag for flag in FLAG_NAMES if flag_values[flag]]
        row: dict[str, Any] = {
            "person": person,
            "flags": ", ".join(tripped),
            "flag_count": len(tripped),
            "cosmetic_touches": cosmetic_count,
            "status_transitions": transitions,
            "estimate_raises": raises,
            "hours_added": round(hours_added, 1),
            "backward_moves": int(len(mine_backward)),
            "reresolved_tickets": int(len(hidden)),
            "window_days": float(window_days),
        }
        for flag in FLAG_NAMES:
            row[flag] = bool(flag_values[flag])
            row[f"{flag}_evidence"] = evidence[flag]
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)[columns]
    return out.sort_values(
        ["flag_count", "cosmetic_touches"], ascending=[False, False]
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
# 8. Resolution credit: who actually did it, and who did not
# --------------------------------------------------------------------------
#
# Every function below reads ``JIRA_FORMER_STAFF`` from the environment
# (semicolon-separated names, matching ``roles_template.env``): people the
# company no longer employs whose Jira account still sits on old tickets.
# Resolutions the changelog attributes to one of them are real changelog
# entries - somebody's account made the transition - but crediting a former
# employee's headcount is never the right answer, so those rows are kept and
# flagged rather than counted.

# Fallback when ``JIRA_FORMER_STAFF`` is unset - the list ``ROSTER.md`` names
# under "Former / inactive", copied here so a deployment or test environment
# that has not yet set the variable still excludes them by default rather than
# crediting them by accident. The env var is the source of truth when present.
_DEFAULT_FORMER_STAFF: frozenset[str] = frozenset(
    {
        "Sai Shankar",
        "Sarju",
        "Yantao He",
        "Dan O'Sullivan",
        "Shivanand",
        "Jon Wang",
        "Kevin Cai",
        "Ramin Shahid",
        "Amir",
        "Christina Lo",
        "Aleksei Pinchuk",
        "Saji",
        "Mark",
        "Courtney McNeil",
        "Lotte Karolina",
        "Jennifer",
        "Eva van Wielink",
        "Stanislav",
        "Saeid Parsa",
        "Armine Aproyan",
        "Haichen Song",
        "Dat",
    }
)


def _former_staff_from_env(env: Mapping[str, str] | None = None) -> frozenset[str]:
    """The former-staff roster: ``JIRA_FORMER_STAFF``, or ``ROSTER.md``'s list.

    A small private parser rather than an import of the roster object another
    agent is building in this same iteration window (``roles.py``): importing
    it here would tie this module's tests to whether that module has landed
    yet, and ``integrity.py`` is documented above as pure pandas and stdlib
    with no imports from the rest of the app. Consolidate into ``roles.py``
    once that module ships - tracked in ``docs/assumptions/2A.md``.

    Names are matched normalised (``_norm``: lowercased, whitespace collapsed),
    the same way every other name comparison in this module works, so
    "Sai Shankar" in the env value matches "sai  shankar" in a changelog author
    field without either side having to agree on capitalisation first.
    """
    source = env if env is not None else os.environ
    raw = source.get("JIRA_FORMER_STAFF")
    if not raw:
        return frozenset(_norm(name) for name in _DEFAULT_FORMER_STAFF)
    names = frozenset(_norm(part) for part in raw.split(";") if _norm(part))
    return names or frozenset(_norm(name) for name in _DEFAULT_FORMER_STAFF)


class CreditedResolutions(NamedTuple):
    """``detail``: one row per resolving transition. ``by_person``: the ledger."""

    detail: pd.DataFrame
    by_person: pd.DataFrame


_CREDIT_DETAIL_COLUMNS = [
    "key",
    "ts",
    "author",
    "from_string",
    "to_string",
    "is_former_staff",
    "credited",
    "reason",
]
_CREDIT_PERSON_COLUMNS = [
    "person",
    "credited_resolutions",
    "flagged_resolutions",
    "tickets",
    "keys",
    "evidence",
]


def credited_resolutions(
    events: pd.DataFrame,
    tickets: pd.DataFrame | None = None,
    *,
    window_days: float | None = 30.0,
    now: object | None = None,
    resolved_statuses: Iterable[str] | None = None,
    former_staff: Iterable[str] | None = None,
) -> CreditedResolutions:
    """Who actually resolved each ticket, from the changelog - not who holds it now.

    The board's resolved-ticket read has only ever asked Jira for a resolved
    ticket's *current* assignee. Every resolution a ticket has ever earned is
    therefore credited to whoever holds it today, which is how a departed
    tester still assigned to old work collects a resolution every time somebody
    else finishes a ticket nobody reassigned - 194 "resolved in 30d" for a
    person who no longer works here, second highest in the company.

    Credit instead goes to ``history["author"]`` on the specific changelog
    entry that made the resolving transition - the same rule ``reresolve_events``
    already applies when counting bounces, applied here per person rather than
    aggregated per ticket. The tail-of-pipeline walk (Review in Staging -> Ready
    for Production -> Released, all "resolved" by this team's list) still counts
    as one resolution: the filter is the same one ``reresolve_events`` uses, so
    only an entry from a status the board does not already consider resolved
    counts as a resolving transition.

    A resolution whose author is on the former-staff list (``JIRA_FORMER_STAFF``,
    see ``_former_staff_from_env``) comes back with ``is_former_staff=True`` and
    ``credited=False`` - visible in ``detail`` for the conversation ("this
    account is still authoring transitions"), excluded from ``by_person``'s
    ``credited_resolutions`` count so it cannot inflate anyone's total.
    ``flagged_resolutions`` on the same person's row says how many were held
    back, so the exclusion is never silent. A resolution with no author at all
    (see ``unattributed_resolutions``) is likewise ``credited=False`` and never
    appears in ``by_person``, because there is no person to credit it to.

    Returns a ``CreditedResolutions(detail, by_person)``:

    - ``detail``: one row per resolving transition (a ticket resolved twice has
      two rows) - ``key``, ``ts``, ``author``, ``from_string``, ``to_string``,
      ``is_former_staff``, ``credited``, ``reason`` (``""`` when credited,
      otherwise ``"former_staff"`` or ``"unattributed"``).
    - ``by_person``: one row per changelog author who resolved anything and is
      not on the former-staff list - ``person``, ``credited_resolutions``,
      ``flagged_resolutions`` (resolutions of theirs held back elsewhere in the
      window, e.g. duplicate authorship under a different display name is not
      tracked here), ``tickets`` (distinct keys credited), ``keys``, ``evidence``.

    What it cannot catch: an account still logging in and moving tickets under a
    former employee's old Jira identity looks identical to the exploit this
    closes - the fix removes the *involuntary* version (crediting someone for
    work they did not touch), not a deliberate one wearing someone else's
    account. It also inherits every blind spot of ``changelog_events`` - the
    ~100-entry truncation and a board migrated from another tracker - so a
    resolution whose transition fell off the truncated end of a very long
    history is invisible here, not wrongly credited.
    """
    empty = CreditedResolutions(
        pd.DataFrame(columns=_CREDIT_DETAIL_COLUMNS),
        pd.DataFrame(columns=_CREDIT_PERSON_COLUMNS),
    )
    if events is None or events.empty:
        return empty

    moment = _now(now)
    resolved = _resolved_set(resolved_statuses)
    former = (
        frozenset(_norm(name) for name in former_staff)
        if former_staff is not None
        else _former_staff_from_env()
    )

    scoped = _window(events, window_days, moment)
    status_events = scoped[scoped["is_status"].fillna(False).astype(bool)].copy()
    if status_events.empty:
        return empty

    status_events["_from_resolved"] = _norm_series(status_events["from_string"]).isin(resolved)
    status_events["_to_resolved"] = _norm_series(status_events["to_string"]).isin(resolved)
    entries = status_events[
        status_events["_to_resolved"] & ~status_events["_from_resolved"]
    ].sort_values("ts")
    if entries.empty:
        return empty

    detail = entries[["key", "ts", "author", "from_string", "to_string"]].copy()
    detail["author"] = detail["author"].fillna("").astype(str).str.strip()
    missing_author = detail["author"].eq("") | detail["author"].eq("Unknown")
    is_former = (~missing_author) & detail["author"].map(_norm).isin(former)
    detail["is_former_staff"] = is_former
    detail["credited"] = ~missing_author & ~is_former
    detail["reason"] = ""
    detail.loc[missing_author, "reason"] = "unattributed"
    detail.loc[is_former, "reason"] = "former_staff"
    detail = detail.reset_index(drop=True)[_CREDIT_DETAIL_COLUMNS]

    people = sorted(set(detail.loc[~missing_author.reset_index(drop=True), "author"]))
    rows: list[dict[str, Any]] = []
    for person in people:
        mine = detail[detail["author"] == person]
        credited = mine[mine["credited"]]
        flagged = mine[~mine["credited"]]
        rows.append(
            {
                "person": person,
                "credited_resolutions": int(len(credited)),
                "flagged_resolutions": int(len(flagged)),
                "tickets": int(credited["key"].nunique()),
                "keys": ", ".join(sorted(set(str(k) for k in credited["key"]))),
                "evidence": _evidence(list(zip(mine["key"], mine["ts"]))),
            }
        )

    by_person = pd.DataFrame(rows, columns=_CREDIT_PERSON_COLUMNS) if rows else empty.by_person
    by_person = by_person.sort_values(
        ["credited_resolutions", "flagged_resolutions"], ascending=False
    ).reset_index(drop=True)

    return CreditedResolutions(detail, by_person)


def unattributed_resolutions(
    events: pd.DataFrame,
    *,
    window_days: float | None = None,
    now: object | None = None,
    resolved_statuses: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Resolutions whose changelog entry names no author at all.

    ``changelog_events`` writes ``"Unknown"`` when a history entry's ``author``
    object is missing or empty - Jira does this for some automation, and for
    entries recorded before a user was deprovisioned. A resolution with no
    author is invisible to ``credited_resolutions`` (there is nobody to credit
    it to) and would otherwise vanish from every per-person view with no record
    that it happened. This function is that record.

    An empty ``author`` string is treated the same as the ``"Unknown"``
    sentinel, so a caller building an events frame directly - a test, or a
    changelog read from something other than ``changelog_events`` - does not
    have to know the sentinel's exact spelling to be counted correctly.

    Never raises on an empty, missing, or entirely history-less changelog; each
    returns an empty frame with the right columns rather than a stack trace, the
    same contract ``changelog_events`` itself gives an empty source.

    Returns one row per unattributed resolving transition: ``key``, ``ts``,
    ``from_string``, ``to_string``. Author is deliberately absent - the row
    exists precisely because there is none.

    What it cannot catch: a resolution attributed to a *wrong* but present name
    (a shared service account, a name that collides with a real person's) looks
    identical to a correctly attributed one here. This only catches entries with
    nobody named at all.
    """
    columns = ["key", "ts", "from_string", "to_string"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    moment = _now(now)
    resolved = _resolved_set(resolved_statuses)
    scoped = _window(events, window_days, moment)
    status_events = scoped[scoped["is_status"].fillna(False).astype(bool)].copy()
    if status_events.empty:
        return pd.DataFrame(columns=columns)

    status_events["_from_resolved"] = _norm_series(status_events["from_string"]).isin(resolved)
    status_events["_to_resolved"] = _norm_series(status_events["to_string"]).isin(resolved)
    entries = status_events[status_events["_to_resolved"] & ~status_events["_from_resolved"]]
    if entries.empty:
        return pd.DataFrame(columns=columns)

    author = entries["author"].fillna("").astype(str).str.strip()
    missing = author.eq("") | author.eq("Unknown")
    return entries[missing][columns].sort_values("ts").reset_index(drop=True)


class OrgReopenRate(NamedTuple):
    """Org-wide reopen count and share-of-resolved, with the evidence behind both."""

    count: int
    resolved_count: int
    share: float | None
    reopened_keys: list[str]
    resolved_keys: list[str]
    evidence: str
    window_days: float


def org_reopen_rate(
    events: pd.DataFrame,
    tickets: pd.DataFrame | None = None,
    *,
    window_days: float = 30.0,
    now: object | None = None,
    resolved_statuses: Iterable[str] | None = None,
) -> OrgReopenRate:
    """How much of what the org resolved in the window came back.

    The only reopen number the board has ever had is per-person and 90 days
    (one JQL query per engineer). There is no organization-wide figure, and a
    raw count on its own answers the wrong question - six reopens out of eight
    resolutions is a different sentence than six out of four hundred. This
    pairs the count with the same window's resolved count, computed the same
    way ``reresolve_events`` computes it, so the rate is never read without its
    denominator.

    ``share`` is ``None``, not ``0.0``, when nothing resolved in the window -
    zero reopens over zero resolutions is "no data", and rendering it the same
    way as "a perfect team" would be the more dangerous of the two lies.

    Returns an ``OrgReopenRate``: ``count`` (distinct tickets that left a
    resolved status and did not return to one within the window),
    ``resolved_count`` (distinct tickets that entered a resolved status in the
    window), ``share`` (``count / resolved_count``, or ``None``),
    ``reopened_keys``, ``resolved_keys``, ``evidence`` and ``window_days``.

    What it cannot catch: a ticket resolved just before the window opened and
    reopened just after it is invisible to both counts - a window boundary
    effect any fixed-window metric has. A ticket that bounced more than once
    inside the window is one entry in ``reopened_keys``, not one per bounce;
    see ``reresolve_events`` for the bounce-level detail this aggregates.
    """
    empty = OrgReopenRate(0, 0, None, [], [], "", float(window_days))
    if events is None or events.empty:
        return empty

    moment = _now(now)
    bounced = reresolve_events(
        events,
        tickets,
        window_days=window_days,
        now=moment,
        resolved_statuses=resolved_statuses,
    )
    if bounced.empty:
        return empty

    resolved_keys = sorted(set(bounced["key"]))
    reopened = bounced[bounced["reopens"] > 0]
    reopened_keys = sorted(set(reopened["key"]))
    resolved_count = len(resolved_keys)
    count = len(reopened_keys)
    share = (count / resolved_count) if resolved_count > 0 else None
    evidence = _evidence(list(zip(reopened["key"], reopened["last_resolved"])))
    return OrgReopenRate(
        count, resolved_count, share, reopened_keys, resolved_keys, evidence, float(window_days)
    )


# Below this many completed tickets in a window, a median is one or two
# tickets deciding the headline number - whichever one happened to be simple.
# Reporting that as a trend is exactly the misleading precision this module
# exists to prevent, so the function refuses instead.
MIN_COMPLETED_FOR_CYCLE_MEDIAN = 3


class EndToEndCycle(NamedTuple):
    """End-to-end lead time (first in-progress -> first resolved), current vs baseline."""

    median_days: float | None
    n: int
    baseline_median_days: float | None
    baseline_n: int
    reason: str | None
    baseline_reason: str | None
    detail: pd.DataFrame
    window_days: float


def end_to_end_cycle(
    events: pd.DataFrame,
    tickets: pd.DataFrame | None = None,
    *,
    baseline_days: float = 90.0,
    now: object | None = None,
    resolved_statuses: Iterable[str] | None = None,
    min_completed: int = MIN_COMPLETED_FOR_CYCLE_MEDIAN,
) -> EndToEndCycle:
    """The one cycle-time number a scorecard headline actually needs.

    ``cycle_time`` reports a median per status per person - the right tool for
    finding where a ticket got stuck, and the wrong one for a headline: nobody
    says "median Code Review 2.1 days" in a standup and means anything by it.
    This collapses the same transitions into one number per period: the median
    days from a ticket's first entry into an in-progress-or-later status to its
    first entry into a resolved one, for tickets that finished in the last
    ``baseline_days`` days (``median_days``), next to the same median for the
    ``baseline_days`` before that (``baseline_median_days``) - so the headline
    never ships without the number it is being compared against.

    Returns ``None`` for either median, with a stated ``reason``, when fewer
    than ``min_completed`` tickets finished in that period. A median of one or
    two tickets is not a trend, it is whichever ticket happened to be simple.

    Returns an ``EndToEndCycle``: ``median_days``, ``n`` (tickets counted in the
    current period), ``baseline_median_days``, ``baseline_n``, ``reason`` (set
    only when ``median_days`` is ``None``), ``baseline_reason`` (likewise for
    the baseline), ``detail`` (one row per finished ticket that fell in either
    period: ``key``, ``started``, ``resolved``, ``days``, ``period`` - "current"
    or "baseline") and ``window_days`` (``baseline_days``, named on the return
    so a caller does not have to remember what it passed in).

    What it cannot catch: a ticket that never recorded an in-progress transition
    - assigned and resolved with no changelog trail through the middle, or
    imported pre-resolved from another tracker - contributes to neither median;
    it is invisible, not counted as zero. Time spent blocked on someone else's
    review still counts as cycle time, because the board records status, not
    who was waiting on whom.
    """
    empty_detail = pd.DataFrame(columns=["key", "started", "resolved", "days", "period"])

    def _empty(reason: str) -> EndToEndCycle:
        return EndToEndCycle(None, 0, None, 0, reason, reason, empty_detail, float(baseline_days))

    if events is None or events.empty:
        return _empty("no changelog data")

    moment = _now(now)
    resolved = _resolved_set(resolved_statuses)
    status_events = events[events["is_status"].fillna(False).astype(bool)]
    if status_events.empty:
        return _empty("no status transitions in the changelog")

    starts = status_events[status_events["to_stage"] >= STARTED_RANK].groupby("key")["ts"].min()
    finishes = status_events[_norm_series(status_events["to_string"]).isin(resolved)][["key", "ts"]]
    if starts.empty or finishes.empty:
        return _empty("no ticket has both a start and a resolution recorded")

    paired = finishes.assign(
        _start=pd.Series(
            starts.reindex(finishes["key"].to_numpy()).to_numpy(), index=finishes.index
        )
    )
    paired = paired[paired["_start"].notna() & (paired["ts"] >= paired["_start"])]
    if paired.empty:
        return _empty("no ticket finished after it started")

    first_finish = paired.groupby("key", sort=False).agg(
        resolved=("ts", "min"), started=("_start", "first")
    )
    first_finish["days"] = (
        (first_finish["resolved"] - first_finish["started"])
        .dt.total_seconds()
        .div(86400.0)
        .round(2)
    )
    detail_all = first_finish.reset_index()

    current_cut = moment - pd.Timedelta(days=float(baseline_days))
    baseline_cut = moment - pd.Timedelta(days=float(baseline_days) * 2)
    current = detail_all[detail_all["resolved"] >= current_cut].copy()
    current["period"] = "current"
    baseline = detail_all[
        (detail_all["resolved"] >= baseline_cut) & (detail_all["resolved"] < current_cut)
    ].copy()
    baseline["period"] = "baseline"
    detail = pd.concat([current, baseline], ignore_index=True)[
        ["key", "started", "resolved", "days", "period"]
    ].reset_index(drop=True)

    def _summarise(frame: pd.DataFrame) -> tuple[float | None, int, str | None]:
        n = int(len(frame))
        if n < min_completed:
            return None, n, f"fewer than {min_completed} completed tickets in the window ({n} found)"
        return float(frame["days"].median()), n, None

    median_days, n, reason = _summarise(current)
    baseline_median_days, baseline_n, baseline_reason = _summarise(baseline)

    return EndToEndCycle(
        median_days,
        n,
        baseline_median_days,
        baseline_n,
        reason,
        baseline_reason,
        detail,
        float(baseline_days),
    )
