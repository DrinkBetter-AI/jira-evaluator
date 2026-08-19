"""Standup truth-check: what people said they'd do, against what the board shows.

Fireflies records the daily standups. This module turns a transcript into
per-person action items (what was said, by whom, at what second) and joins
those against the Jira changelog (``integrity.changelog_events``) over the
following days, to answer one question: did the thing show up on the board?

This feeds the **Integrity page only** - admin-gated, per ``ROSTER.md``
("Integrity-page access: decided - Angel only"). It is not a leaderboard
input and nothing here computes a score.

What this module is for, in one line each:

- ``load_fireflies_key``    - the one place ``FIREFLIES_API_KEY`` is read.
- ``resolve_speaker``       - a transcribed name against the alias table and
  the roster, reporting an unrecognised name rather than guessing at it.
- ``attendance_from_speakers`` - who was actually in the room, from who
  actually spoke. Never reads ``participants``/``meeting_attendees``.
- ``no_notice_absences``    - active roster members who didn't speak at a
  meeting, split into "someone flagged it in the room" and "nobody did".
- ``extract_action_items``  - sentences Fireflies tagged as a commitment,
  with a deep link to the moment.
- ``match_action_items``    - each action item against the Jira changelog in
  a trailing window: followed through, or still open. Never "lied about".
- ``truth_check``           - the four above, run over a batch of transcripts,
  pure and offline-testable.
- ``fetch_standup_truth_check`` - the live entry point: reads the env key,
  calls Fireflies, degrades to ``truth_check`` output that says why.

Why attendance is never read from ``participants``: that field is the
calendar invite list, not who showed up. It is verified - not suspected - to
be wrong on this team: Dat / Đào Nguyễn Anh was terminated and his account
removed (``ROSTER.md``), and he remained in standup invites and was still
giving live updates as recently as 13 Aug 2026 per Fireflies. A departed
person sitting in ``participants`` forever would read as "present" on every
subsequent standup if this module trusted that field. It doesn't: attendance
here is built exclusively from who is recorded speaking (``sentences`` /
``speakers``), so a departed person who never speaks again is correctly
invisible, and a departed person who *does* speak is correctly flagged
present - which is itself the alert worth raising, not a bug to hide.

Why the alias table exists and is kept separate from guessing: this team is
in Vietnam, Uruguay, Tunisia and the EU, and Fireflies' speech-to-text
mangles names across that spread constantly. "Tina" for "Dina" is the
documented case (Angel, 19 Aug). Silently fuzzy-matching an unrecognised name
to "the closest roster name" would read as an accusation the day it matches
the wrong person - to a departed name, or to nobody. So the table is a fixed,
explicit, easy-to-extend mapping, and anything not in it or in the roster is
reported through ``unresolved_speakers`` for a human to add, never guessed.

The blind spot, stated once because every function below inherits it: a
transcript records what was **said**, not what was true, and it can only
record it if the mic caught it. A person who says nothing on the call is
invisible to this module, not "absent" - they may have been in the room,
muted, unrecorded, or the transcript may simply have missed them, and
`no_notice_absences` cannot tell those apart from a genuine no-show. Every
function here reports evidence for a human conversation; none of them
concludes wrongdoing.

Pure pandas, stdlib and ``requests``: no Streamlit. ``truth_check`` and
everything it calls take plain dicts/frames and return frames, so the whole
join logic is unit-testable against fixtures without ever touching the
Fireflies network.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

import pandas as pd
import requests

import integrity

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

_KEY_ENV_VAR = "FIREFLIES_API_KEY"
FIREFLIES_GRAPHQL_URL = "https://api.fireflies.ai/graphql"
FIREFLIES_VIEW_URL = "https://app.fireflies.ai/view/{id}?t={t}"

# How many days of Fireflies history to ask for and how far forward to look
# for a matching Jira changelog entry, when a caller doesn't say otherwise.
# Matches the standup cadence: a commitment made today should show up on the
# board inside a work week, not a quarter.
DEFAULT_LOOKBACK_DAYS = 14
DEFAULT_FOLLOWTHROUGH_WINDOW_DAYS = 7


class FirefliesUnavailable(NamedTuple):
    """Why a live Fireflies read didn't happen. Never raised - always returned."""

    reason: str


def load_fireflies_key() -> str | None:
    """``FIREFLIES_API_KEY`` from the environment, or ``None`` when unset.

    The one place this module reads the credential. There is no fallback env
    var and no default: an unset key means "unavailable", not "try anyway".
    """
    key = os.getenv(_KEY_ENV_VAR, "").strip()
    return key or None


# ---------------------------------------------------------------------------
# Alias table
# ---------------------------------------------------------------------------

# Transcribed-name -> canonical roster name. Keys are matched lowercased and
# whitespace-collapsed (see ``_norm``), so "Tina", "tina " and "TINA" all hit
# the same entry. Values must be a name the caller's roster actually contains
# - this module doesn't validate that at import time (it has no roster of its
# own), so a stale alias just fails to resolve and is reported alongside a
# genuinely unrecognised name; there's no silent mismatch.
#
# Data-driven and meant to grow: every new entry is one line, sourced from a
# human confirming "yes, that's who Fireflies meant" - never from this module
# guessing a nearest match. Add entries as they're found; never delete one
# without confirming the mishearing has stopped (Fireflies' model can regress
# on a name from one call to the next).
SPEAKER_ALIASES: dict[str, str] = {
    # Confirmed by Angel, 19 Aug 2026: Fireflies mishears "Dina" as "Tina"
    # often enough that it isn't noise. Canonical form matches the Jira
    # display name in ROSTER.md / roles_template.env ("Dina QA").
    "tina": "Dina QA",
    "gina": "Dina QA",
    # "Dat" and common renderings of "Đào Nguyễn Anh" resolve straight
    # through the roster match below (ASCII "Dat" already matches his roster
    # entry case-insensitively); no alias entry needed for the common case,
    # only for genuine mishearings. Kept here as a documented non-decision:
    # do not add a "dao"/"anh" alias without a confirmed transcript to point
    # at, or it becomes exactly the guess this table exists to avoid.
}


def _norm(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


class SpeakerResolution(NamedTuple):
    """One speaker name, resolved or not. Never a guess - only a match or a miss."""

    raw: str
    canonical: str | None
    via_alias: bool


def resolve_speaker(
    raw_name: str,
    roster_names: Iterable[str],
    alias_table: Mapping[str, str] | None = None,
) -> SpeakerResolution:
    """Match a transcribed speaker name to the roster, or report it unresolved.

    Order: exact case-insensitive roster match first (the common case needs no
    alias at all), then the alias table, then unresolved. Checking the roster
    first means a real, correctly-transcribed name is never routed through an
    alias by accident.

    Returns the roster's own spelling in ``canonical`` on a match - not the
    alias table's value verbatim - except when the alias table's target isn't
    in the roster passed in, in which case that target is returned as-is
    (the caller's roster may be a subset; a raw name and an alias name both
    look up canonically, so this stays behaviour-identical to a full roster).
    """
    table = alias_table if alias_table is not None else SPEAKER_ALIASES
    norm = _norm(raw_name)
    if not norm:
        return SpeakerResolution(raw=raw_name, canonical=None, via_alias=False)

    roster_index = {_norm(name): name for name in roster_names}
    if norm in roster_index:
        return SpeakerResolution(raw=raw_name, canonical=roster_index[norm], via_alias=False)

    alias_target = table.get(norm)
    if alias_target is not None:
        canonical = roster_index.get(_norm(alias_target), alias_target)
        return SpeakerResolution(raw=raw_name, canonical=canonical, via_alias=True)

    return SpeakerResolution(raw=raw_name, canonical=None, via_alias=False)


# ---------------------------------------------------------------------------
# Transcript shape helpers
# ---------------------------------------------------------------------------


def _sentences(transcript: Mapping[str, Any]) -> list[dict[str, Any]]:
    sentences = transcript.get("sentences") or []
    return [s for s in sentences if isinstance(s, dict)]


def _meeting_id(transcript: Mapping[str, Any]) -> str:
    return str(transcript.get("id") or "")


def _meeting_date(transcript: Mapping[str, Any]) -> pd.Timestamp | None:
    """The meeting's timestamp. Fireflies' ``date`` is epoch milliseconds."""
    raw = transcript.get("date")
    if raw is None:
        return None
    try:
        millis = float(raw)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(raw, utc=True, errors="coerce")
        return None if pd.isna(parsed) else parsed
    return pd.to_datetime(millis, unit="ms", utc=True)


def deep_link(transcript_id: str, seconds: float) -> str:
    """``app.fireflies.ai/view/{id}?t={seconds}`` - every claim links to the moment."""
    return FIREFLIES_VIEW_URL.format(id=transcript_id, t=int(round(max(0.0, seconds))))


# ---------------------------------------------------------------------------
# Attendance - from speakers, never from participants
# ---------------------------------------------------------------------------

ATTENDANCE_COLUMNS = [
    "meeting_id",
    "date",
    "speaker_raw",
    "speaker_canonical",
    "resolved",
    "is_former_staff",
    "sentence_count",
    "first_ts",
]


def attendance_from_speakers(
    transcript: Mapping[str, Any],
    *,
    roster_names: Iterable[str],
    former_staff: Iterable[str] = (),
    alias_table: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Who spoke in this transcript - deliberately reads ``sentences`` only.

    This function never looks at ``transcript["participants"]`` or
    ``transcript["meeting_attendees"]``. Those are the calendar invite list:
    on this team it is known-stale (departed staff, per ``ROSTER.md``), and
    using it for attendance would mark a person present forever after they
    left. A person is "present" here iff at least one sentence is attributed
    to them, resolved or not.

    A departed name (``former_staff``) that *does* speak is still returned,
    with ``is_former_staff=True`` - that combination is the actual alert
    (Dat, 13 Aug), not a row to suppress.

    Blind spot: a muted attendee who never speaks is indistinguishable here
    from someone who wasn't on the call at all.
    """
    former_norm = {_norm(n) for n in former_staff}
    rows: dict[str, dict[str, Any]] = {}
    meeting_id = _meeting_id(transcript)
    date = _meeting_date(transcript)

    for sentence in _sentences(transcript):
        raw_name = sentence.get("speaker_name")
        if not raw_name or not _norm(raw_name):
            continue
        resolution = resolve_speaker(raw_name, roster_names, alias_table)
        identity_key = _norm(resolution.canonical) if resolution.canonical else f"unresolved:{_norm(raw_name)}"
        try:
            start = float(sentence.get("start_time"))
        except (TypeError, ValueError):
            start = None

        row = rows.setdefault(
            identity_key,
            {
                "meeting_id": meeting_id,
                "date": date,
                "speaker_raw": raw_name,
                "speaker_canonical": resolution.canonical,
                "resolved": resolution.canonical is not None,
                "is_former_staff": _norm(resolution.canonical) in former_norm
                if resolution.canonical
                else _norm(raw_name) in former_norm,
                "sentence_count": 0,
                "first_ts": start,
            },
        )
        row["sentence_count"] += 1
        if start is not None and (row["first_ts"] is None or start < row["first_ts"]):
            row["first_ts"] = start

    if not rows:
        return pd.DataFrame(columns=ATTENDANCE_COLUMNS)
    return pd.DataFrame(rows.values(), columns=ATTENDANCE_COLUMNS)


def attendance_summary(
    transcripts: Sequence[Mapping[str, Any]],
    *,
    roster_names: Iterable[str],
    former_staff: Iterable[str] = (),
    alias_table: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Per person: meetings spoken in, out of meetings held, plus ghost-speaker flags.

    ``attendance_rate`` is out of the number of transcripts passed in, so it
    is only meaningful when ``transcripts`` is one recurring meeting series
    over a consistent window - mixing standups with sprint reviews would
    understate everyone's rate against a denominator they were never invited
    to.
    """
    frames = [
        attendance_from_speakers(
            t, roster_names=roster_names, former_staff=former_staff, alias_table=alias_table
        )
        for t in transcripts
    ]
    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ATTENDANCE_COLUMNS)
    total_meetings = len(transcripts)
    if all_rows.empty or total_meetings == 0:
        return pd.DataFrame(
            columns=["person", "meetings_spoken", "meetings_total", "attendance_rate", "is_former_staff"]
        )

    key = all_rows["speaker_canonical"].fillna("unresolved: " + all_rows["speaker_raw"])
    grouped = (
        all_rows.assign(_person=key)
        .groupby("_person", dropna=False)
        .agg(
            meetings_spoken=("meeting_id", "nunique"),
            is_former_staff=("is_former_staff", "any"),
        )
        .reset_index()
        .rename(columns={"_person": "person"})
    )
    grouped["meetings_total"] = total_meetings
    grouped["attendance_rate"] = grouped["meetings_spoken"] / grouped["meetings_total"]
    return grouped[["person", "meetings_spoken", "meetings_total", "attendance_rate", "is_former_staff"]]


# ---------------------------------------------------------------------------
# No-notice absences
# ---------------------------------------------------------------------------

# Phrases that, spoken by anyone in the meeting near a roster member's name,
# read as that member's absence being flagged in the room. Deliberately a
# fixed, literal list rather than an NLP guess: a phrase this doesn't cover
# is a miss, and the function's docstring says so, rather than pretending the
# list is exhaustive.
ABSENCE_NOTICE_PHRASES: tuple[str, ...] = (
    "is out",
    "is off",
    "won't be able to make it",
    "wont be able to make it",
    "can't make it",
    "cant make it",
    "out sick",
    "out today",
    "on pto",
    "on vacation",
    "taking the day",
    "not joining",
    "couldn't join",
    "sent his regrets",
    "sent her regrets",
    "sent their regrets",
)

NO_NOTICE_COLUMNS = ["meeting_id", "date", "person", "notice", "evidence"]


def no_notice_absences(
    transcripts: Sequence[Mapping[str, Any]],
    *,
    active_roster_names: Iterable[str],
    alias_table: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Active roster members who didn't speak at a meeting, split by notice.

    ``active_roster_names`` must be the caller's live, active roster - not
    anything derived from ``participants`` - precisely so a departed person
    still sitting in a calendar invite (Dat) never resurfaces here as a daily
    "absence". The expected-attendee pool is a fact the caller already has
    (``roles.py``'s roster), not something this module infers from the
    transcript.

    "Notice" is detected only from what's actually in the transcript: another
    speaker's sentence mentioning the absent person's (resolved) name
    alongside one of ``ABSENCE_NOTICE_PHRASES``. This is a floor, not a
    ceiling - a notice given over Slack, or worded differently on the call,
    won't be caught, and will show up here as "no notice" even though notice
    was in fact given. Read a no-notice row as "not evidenced in this
    transcript", not as a confirmed silent no-show.
    """
    active_names = list(dict.fromkeys(str(n) for n in active_roster_names if _norm(n)))
    rows: list[dict[str, Any]] = []

    for transcript in transcripts:
        meeting_id = _meeting_id(transcript)
        date = _meeting_date(transcript)
        sentences = _sentences(transcript)
        present = set()
        mention_text = " ".join(_norm(s.get("text")) for s in sentences)
        for sentence in sentences:
            resolution = resolve_speaker(sentence.get("speaker_name", ""), active_names, alias_table)
            if resolution.canonical:
                present.add(_norm(resolution.canonical))

        for person in active_names:
            if _norm(person) in present:
                continue
            first_name = _norm(person).split(" ")[0]
            has_notice = False
            if first_name and first_name in mention_text:
                for phrase in ABSENCE_NOTICE_PHRASES:
                    # Loose proximity check: the phrase and the first name both
                    # appear somewhere in the meeting's combined text. Exact
                    # adjacency isn't required - standups say "heads up, Dina's
                    # out today" as often as "Dina is out today" - but this can
                    # false-positive if the name and an unrelated absence
                    # mention both happen to occur in the same meeting.
                    if phrase in mention_text:
                        has_notice = True
                        break
            rows.append(
                {
                    "meeting_id": meeting_id,
                    "date": date,
                    "person": person,
                    "notice": has_notice,
                    "evidence": "transcript mention of an absence phrase" if has_notice else "",
                }
            )

    if not rows:
        return pd.DataFrame(columns=NO_NOTICE_COLUMNS)
    return pd.DataFrame(rows, columns=NO_NOTICE_COLUMNS)


def no_notice_absence_count(no_notice_df: pd.DataFrame) -> pd.DataFrame:
    """Per person, the count of absences with no notice found in-transcript.

    Absences *with* notice are excluded here on purpose - an absence someone
    flagged in the room is not a signal worth a KPI line, only an unexplained
    one is.
    """
    if no_notice_df.empty:
        return pd.DataFrame(columns=["person", "no_notice_absences"])
    flagged = no_notice_df.loc[~no_notice_df["notice"]]
    if flagged.empty:
        return pd.DataFrame(columns=["person", "no_notice_absences"])
    return (
        flagged.groupby("person")
        .size()
        .reset_index(name="no_notice_absences")
        .sort_values("no_notice_absences", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Action items
# ---------------------------------------------------------------------------

ACTION_ITEM_COLUMNS = [
    "meeting_id",
    "date",
    "sentence_index",
    "speaker_raw",
    "speaker_canonical",
    "resolved",
    "ts_seconds",
    "text",
    "ticket_key",
    "deep_link",
]

# Standard Jira issue key: one or more uppercase letters/digits (starting with
# a letter), a hyphen, and a number - e.g. "VINO-231". Matches what
# ``jira_client``/``prioritization`` treat as a key elsewhere in this app.
TICKET_KEY_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9}-\d{1,6})\b")


def _mentioned_ticket_key(text: str) -> str | None:
    match = TICKET_KEY_RE.search(text or "")
    return match.group(1) if match else None


def extract_action_items(
    transcript: Mapping[str, Any],
    *,
    roster_names: Iterable[str],
    alias_table: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Sentences Fireflies tagged as a commitment (``ai_filters.task``), with a deep link.

    Only sentences carrying an explicit ``ai_filters.task`` truthy flag count
    - this module does not guess at commitment language ("I'll", "going to")
    itself. If a transcript carries no ``ai_filters`` at all (an older export,
    or a plan without that add-on), this returns empty for that transcript
    rather than falling back to a keyword heuristic that would read as this
    module inventing quotes.

    Blind spot: Fireflies' task classifier is itself a model, with its own
    false negatives (a real commitment phrased in a way it doesn't tag) and
    false positives (a hypothetical read as a promise). This is a starting
    list to click through, not a certified inventory of commitments.
    """
    meeting_id = _meeting_id(transcript)
    date = _meeting_date(transcript)
    rows: list[dict[str, Any]] = []

    for sentence in _sentences(transcript):
        ai_filters = sentence.get("ai_filters")
        if not isinstance(ai_filters, dict) or not ai_filters.get("task"):
            continue
        raw_name = sentence.get("speaker_name", "")
        resolution = resolve_speaker(raw_name, roster_names, alias_table)
        try:
            start = float(sentence.get("start_time"))
        except (TypeError, ValueError):
            start = 0.0
        text = str(sentence.get("text") or "")
        rows.append(
            {
                "meeting_id": meeting_id,
                "date": date,
                "sentence_index": sentence.get("index"),
                "speaker_raw": raw_name,
                "speaker_canonical": resolution.canonical,
                "resolved": resolution.canonical is not None,
                "ts_seconds": start,
                "text": text,
                "ticket_key": _mentioned_ticket_key(text),
                "deep_link": deep_link(meeting_id, start),
            }
        )

    if not rows:
        return pd.DataFrame(columns=ACTION_ITEM_COLUMNS)
    return pd.DataFrame(rows, columns=ACTION_ITEM_COLUMNS)


# ---------------------------------------------------------------------------
# Joining action items against the Jira changelog
# ---------------------------------------------------------------------------

MATCH_COLUMNS = [
    "meeting_id",
    "date",
    "speaker_canonical",
    "text",
    "ticket_key",
    "deep_link",
    "outcome",
    "matched_by",
    "matched_key",
    "matched_ts",
    "matched_author",
]


def match_action_items(
    action_items_df: pd.DataFrame,
    changelog_events: pd.DataFrame,
    *,
    window_days: float = DEFAULT_FOLLOWTHROUGH_WINDOW_DAYS,
) -> pd.DataFrame:
    """Each action item against the Jira changelog in the trailing window.

    Two ways to match, tried in order:

    1. **By ticket key** - the action item names a key (``ticket_key``) and
       that key has any changelog event, by anyone, in the window. Naming the
       ticket is the strongest signal someone can give about what they meant.
    2. **By speaker activity** - no key was named, but the resolved speaker
       authored a status-changing event on *some* ticket in the window. Weaker
       (it doesn't confirm this specific commitment), so it's labelled
       ``matched_by="author_activity"`` rather than ``"ticket_key"``, and a
       consumer that only wants the strong signal can filter on that column.

    An item that matches either way is ``outcome="followed_through"``. An item
    that matches neither is ``outcome="open"`` - not ``"broken"`` or
    ``"missed"``. This module reports what it can see in two systems; it does
    not have visibility into work done outside Jira (a fix landed with no
    ticket, a conversation that resolved it, a change of plan agreed later in
    the same standup) and calling an unmatched item a lie would claim
    certainty this join cannot have.

    An action item with no resolved speaker cannot be matched by author
    activity (there's nothing to look up) and can still match by ticket key.
    """
    if action_items_df.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    events = changelog_events if changelog_events is not None else integrity.empty_events()
    window = pd.Timedelta(days=float(window_days))
    rows: list[dict[str, Any]] = []

    for _, item in action_items_df.iterrows():
        date = item.get("date")
        outcome = "open"
        matched_by = None
        matched_key = None
        matched_ts = None
        matched_author = None

        window_start = date if pd.notna(date) else None
        window_end = (date + window) if pd.notna(date) else None

        ticket_key = item.get("ticket_key")
        if ticket_key and not events.empty and window_start is not None:
            candidates = events[
                (events["key"] == ticket_key)
                & (events["ts"] >= window_start)
                & (events["ts"] <= window_end)
            ]
            if not candidates.empty:
                first = candidates.sort_values("ts").iloc[0]
                outcome = "followed_through"
                matched_by = "ticket_key"
                matched_key = ticket_key
                matched_ts = first["ts"]
                matched_author = first["author"]

        speaker = item.get("speaker_canonical")
        if outcome == "open" and speaker and not events.empty and window_start is not None:
            candidates = events[
                (events["author"] == speaker)
                & events["is_status"]
                & (events["ts"] >= window_start)
                & (events["ts"] <= window_end)
            ]
            if not candidates.empty:
                first = candidates.sort_values("ts").iloc[0]
                outcome = "followed_through"
                matched_by = "author_activity"
                matched_key = first["key"]
                matched_ts = first["ts"]
                matched_author = first["author"]

        rows.append(
            {
                "meeting_id": item.get("meeting_id"),
                "date": date,
                "speaker_canonical": speaker,
                "text": item.get("text"),
                "ticket_key": ticket_key,
                "deep_link": item.get("deep_link"),
                "outcome": outcome,
                "matched_by": matched_by,
                "matched_key": matched_key,
                "matched_ts": matched_ts,
                "matched_author": matched_author,
            }
        )

    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class StandupResult(NamedTuple):
    """Everything the Integrity page needs, or a reason nothing is here.

    ``available=False`` is not an error state for a caller to special-case
    hard: every frame is still present, just empty, so a page can render its
    normal "no data" table state instead of branching on a sentinel.
    """

    available: bool
    reason: str
    action_items: pd.DataFrame
    matched_action_items: pd.DataFrame
    attendance: pd.DataFrame
    no_notice_absences: pd.DataFrame
    unresolved_speakers: tuple[str, ...]


def _empty_result(reason: str) -> StandupResult:
    return StandupResult(
        available=False,
        reason=reason,
        action_items=pd.DataFrame(columns=ACTION_ITEM_COLUMNS),
        matched_action_items=pd.DataFrame(columns=MATCH_COLUMNS),
        attendance=pd.DataFrame(columns=ATTENDANCE_COLUMNS),
        no_notice_absences=pd.DataFrame(columns=NO_NOTICE_COLUMNS),
        unresolved_speakers=(),
    )


def truth_check(
    transcripts: Sequence[Mapping[str, Any]],
    *,
    roster_names: Iterable[str],
    active_roster_names: Iterable[str] | None = None,
    former_staff: Iterable[str] = (),
    changelog_events: pd.DataFrame | None = None,
    alias_table: Mapping[str, str] | None = None,
    window_days: float = DEFAULT_FOLLOWTHROUGH_WINDOW_DAYS,
) -> StandupResult:
    """Run the full truth-check over already-fetched transcripts. Pure, offline.

    This is the function every test in ``tests/test_standup.py`` calls
    directly - it takes plain dicts (fixture-shaped Fireflies transcripts) and
    a changelog frame (``integrity.changelog_events`` output), so the whole
    join can be pinned without a network call or an API key.
    ``fetch_standup_truth_check`` is the thin live wrapper around this.
    """
    roster = list(roster_names)
    active_roster = list(active_roster_names) if active_roster_names is not None else roster
    events = changelog_events if changelog_events is not None else integrity.empty_events()

    action_frames = [
        extract_action_items(t, roster_names=roster, alias_table=alias_table) for t in transcripts
    ]
    action_items_df = (
        pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame(columns=ACTION_ITEM_COLUMNS)
    )
    matched = match_action_items(action_items_df, events, window_days=window_days)

    attendance_frames = [
        attendance_from_speakers(
            t, roster_names=roster, former_staff=former_staff, alias_table=alias_table
        )
        for t in transcripts
    ]
    attendance_df = (
        pd.concat(attendance_frames, ignore_index=True)
        if attendance_frames
        else pd.DataFrame(columns=ATTENDANCE_COLUMNS)
    )

    absences_df = no_notice_absences(
        transcripts, active_roster_names=active_roster, alias_table=alias_table
    )

    unresolved = set()
    for df in (action_items_df, attendance_df):
        if df.empty:
            continue
        unresolved_rows = df.loc[~df["resolved"], "speaker_raw"]
        unresolved.update(_norm(v) and v for v in unresolved_rows if v)
    # Preserve original casing, sorted for a stable, diffable test/UI output.
    unresolved_names = tuple(sorted({v for v in unresolved if v}))

    reason = "" if transcripts else "no transcripts in the requested window"
    return StandupResult(
        available=True,
        reason=reason,
        action_items=action_items_df,
        matched_action_items=matched,
        attendance=attendance_df,
        no_notice_absences=absences_df,
        unresolved_speakers=unresolved_names,
    )


def _fetch_transcripts(token: str, days: int) -> list[dict[str, Any]]:
    """One Fireflies GraphQL page of recent transcripts, sentences included.

    Kept intentionally small (title/date/participants/sentences with
    ai_filters) - this module only needs what it reads above; a caller who
    wants Fireflies' own summary/action-item text can fetch that separately.
    """
    query = """
    query RecentTranscripts($fromDate: DateTime, $limit: Int) {
      transcripts(fromDate: $fromDate, limit: $limit) {
        id
        title
        date
        participants
        meeting_attendees { displayName email }
        speakers { id name }
        sentences {
          index
          speaker_name
          speaker_id
          text
          start_time
          end_time
          ai_filters { task }
        }
      }
    }
    """
    from_date = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).isoformat()
    response = requests.post(
        FIREFLIES_GRAPHQL_URL,
        json={"query": query, "variables": {"fromDate": from_date, "limit": 50}},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(f"Fireflies GraphQL error: {payload['errors']}")
    return (payload.get("data") or {}).get("transcripts") or []


def fetch_standup_truth_check(
    *,
    roster_names: Iterable[str],
    active_roster_names: Iterable[str] | None = None,
    former_staff: Iterable[str] = (),
    changelog_events: pd.DataFrame | None = None,
    alias_table: Mapping[str, str] | None = None,
    days: int = DEFAULT_LOOKBACK_DAYS,
    window_days: float = DEFAULT_FOLLOWTHROUGH_WINDOW_DAYS,
) -> StandupResult:
    """The live entry point. Never raises - an unreachable Fireflies degrades
    to the same shape :func:`truth_check` returns, with a reason set.

    With no ``FIREFLIES_API_KEY`` set, this returns immediately, unavailable,
    with a reason - it does not attempt a request, and it does not return
    zeros dressed as real data (every frame is empty, and ``available`` says
    so explicitly, so a page can't mistake "no key" for "checked, all clean").
    """
    key = load_fireflies_key()
    if key is None:
        return _empty_result(f"{_KEY_ENV_VAR} not set - standup truth-check unavailable")

    try:
        transcripts = _fetch_transcripts(key, days)
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        return _empty_result(f"Fireflies unavailable: {exc}")

    result = truth_check(
        transcripts,
        roster_names=roster_names,
        active_roster_names=active_roster_names,
        former_staff=former_staff,
        changelog_events=changelog_events,
        alias_table=alias_table,
        window_days=window_days,
    )
    if not transcripts:
        return result._replace(reason="Fireflies reachable, no transcripts in the requested window")
    return result
