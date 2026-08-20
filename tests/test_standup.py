"""Tests for the standup truth-check (standup.py, WP10 / Task 6B).

There is no live Fireflies key in this environment, so every test here runs
against fixture payloads shaped like the real Fireflies GraphQL transcript
response (``id``/``date``/``participants``/``meeting_attendees``/``speakers``/
``sentences`` with ``ai_filters``), joined against a Jira changelog frame
built with real Jira changelog shapes so it flows through
``integrity.changelog_events`` exactly as production data would.

Each test is aimed at one thing the task spec calls out by name: no key,
speakers-not-participants (pinned to Dat's real case), the alias table, the
deep link, the changelog join's three-way outcome (followed through / open /
never "lied"), and the notice/no-notice split on absences.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import integrity  # noqa: E402
import standup  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _sentence(index, speaker_name, text, start_time, *, task=False):
    return {
        "index": index,
        "speaker_name": speaker_name,
        "speaker_id": speaker_name,
        "text": text,
        "start_time": start_time,
        "end_time": start_time + 4.0,
        "ai_filters": {"task": task},
    }


def _transcript(transcript_id, date_ms, participants, meeting_attendees, sentences):
    return {
        "id": transcript_id,
        "title": "Engineering Standup",
        "date": date_ms,
        "participants": participants,
        "meeting_attendees": meeting_attendees,
        "speakers": [{"id": s["speaker_id"], "name": s["speaker_name"]} for s in sentences],
        "sentences": sentences,
    }


# 13 Aug 2026 standup, epoch ms, matches the date ROSTER.md pins for Dat.
_AUG_13_MS = int(pd.Timestamp("2026-08-13T14:00:00Z").timestamp() * 1000)
_AUG_06_MS = int(pd.Timestamp("2026-08-06T14:00:00Z").timestamp() * 1000)

ROSTER_NAMES = ["Tam", "Shawn", "Dina QA", "Dat"]
ACTIVE_ROSTER_NAMES = ["Tam", "Shawn", "Dina QA"]
FORMER_STAFF = ["Dat"]


def _changelog_issue(key, author, from_status, to_status, created):
    return {
        "key": key,
        "changelog": {
            "histories": [
                {
                    "id": f"{key}-1",
                    "author": {"displayName": author},
                    "created": created,
                    "items": [
                        {
                            "field": "status",
                            "fieldId": "status",
                            "fromString": from_status,
                            "toString": to_status,
                        }
                    ],
                }
            ]
        },
    }


# ---------------------------------------------------------------------------
# No API key -> unavailable, never raises
# ---------------------------------------------------------------------------


def test_no_api_key_returns_unavailable_with_a_reason_and_does_not_raise(monkeypatch):
    monkeypatch.delenv("FIREFLIES_API_KEY", raising=False)
    result = standup.fetch_standup_truth_check(
        roster_names=ROSTER_NAMES, active_roster_names=ACTIVE_ROSTER_NAMES
    )
    assert result.available is False
    assert "FIREFLIES_API_KEY" in result.reason
    assert result.action_items.empty
    assert result.matched_action_items.empty
    assert result.attendance.empty
    assert result.no_notice_absences.empty
    assert result.unresolved_speakers == ()


def test_no_api_key_never_returns_zeros_that_read_as_real_data(monkeypatch):
    """Unavailable must be an explicit flag, not empty-but-silently-checked frames."""
    monkeypatch.delenv("FIREFLIES_API_KEY", raising=False)
    result = standup.fetch_standup_truth_check(roster_names=ROSTER_NAMES)
    assert result.available is False
    # Distinguishable from "checked, found nothing" (available=True, empty reason).
    assert result.reason != ""


def test_unreachable_fireflies_degrades_without_raising(monkeypatch):
    monkeypatch.setenv("FIREFLIES_API_KEY", "test-key")

    def _boom(*args, **kwargs):
        raise standup.requests.ConnectionError("no route to host")

    monkeypatch.setattr(standup.requests, "post", _boom)
    result = standup.fetch_standup_truth_check(roster_names=ROSTER_NAMES)
    assert result.available is False
    assert "Fireflies unavailable" in result.reason


# ---------------------------------------------------------------------------
# Attendance from speakers, never from participants - Dat's actual case
# ---------------------------------------------------------------------------


def test_departed_person_in_participants_but_never_speaking_is_not_marked_present():
    transcript = _transcript(
        "tr-ghost",
        _AUG_13_MS,
        participants=["dat@vinovoss.com", "tam@vinovoss.com"],
        meeting_attendees=[{"displayName": "Dat", "email": "dat@vinovoss.com"}],
        sentences=[
            _sentence(0, "Tam", "I'll pick up VINO-100 today.", 10.0, task=True),
        ],
    )
    attendance = standup.attendance_from_speakers(
        transcript, roster_names=ROSTER_NAMES, former_staff=FORMER_STAFF
    )
    assert "Dat" not in set(attendance["speaker_canonical"].dropna())
    # Confirms the read genuinely came from sentences, not from the (much
    # larger) participants/meeting_attendees lists carrying Dat's name.
    assert set(attendance["speaker_canonical"]) == {"Tam"}


def test_departed_person_who_does_speak_is_marked_present_and_flagged_former_staff():
    """Pinned to the verified case: Dat spoke as recently as 13 Aug 2026."""
    transcript = _transcript(
        "tr-dat-speaks",
        _AUG_13_MS,
        participants=["dat@vinovoss.com"],
        meeting_attendees=[{"displayName": "Dat", "email": "dat@vinovoss.com"}],
        sentences=[
            _sentence(0, "Dat", "I tested the checkout flow yesterday.", 5.0),
        ],
    )
    attendance = standup.attendance_from_speakers(
        transcript, roster_names=ROSTER_NAMES, former_staff=FORMER_STAFF
    )
    row = attendance.loc[attendance["speaker_canonical"] == "Dat"]
    assert len(row) == 1
    assert bool(row.iloc[0]["is_former_staff"]) is True


def test_attendance_ignores_participants_field_entirely_even_when_malformed():
    """A broken/huge participants list must not change who counts as present."""
    transcript = _transcript(
        "tr-malformed",
        _AUG_13_MS,
        participants="not-even-a-list",  # deliberately wrong shape
        meeting_attendees=None,
        sentences=[_sentence(0, "Shawn", "Shipping VINO-5.", 1.0)],
    )
    attendance = standup.attendance_from_speakers(transcript, roster_names=ROSTER_NAMES)
    assert set(attendance["speaker_canonical"]) == {"Shawn"}


# ---------------------------------------------------------------------------
# Alias table
# ---------------------------------------------------------------------------


def test_alias_table_maps_tina_to_dina():
    assert standup.SPEAKER_ALIASES["tina"] == "Dina QA"
    resolution = standup.resolve_speaker("Tina", ROSTER_NAMES)
    assert resolution.canonical == "Dina QA"
    assert resolution.via_alias is True


def test_alias_resolution_is_case_and_whitespace_insensitive():
    resolution = standup.resolve_speaker("  TINA  ", ROSTER_NAMES)
    assert resolution.canonical == "Dina QA"


def test_unrecognised_speaker_is_reported_unresolved_not_guessed_or_dropped():
    transcript = _transcript(
        "tr-unknown",
        _AUG_13_MS,
        participants=[],
        meeting_attendees=[],
        sentences=[
            _sentence(0, "Some New Contractor", "I'll look at the API today.", 3.0, task=True),
        ],
    )
    result = standup.truth_check(
        [transcript], roster_names=ROSTER_NAMES, active_roster_names=ACTIVE_ROSTER_NAMES
    )
    # Not dropped: the row is still present in the output frames...
    assert len(result.action_items) == 1
    assert bool(result.action_items.iloc[0]["resolved"]) is False
    assert result.action_items.iloc[0]["speaker_canonical"] is None
    # ...and not guessed: it's surfaced as unresolved rather than silently
    # matched to the nearest roster name.
    assert "Some New Contractor" in result.unresolved_speakers


def test_alias_table_is_data_driven_and_overridable_by_caller():
    """A caller can extend/override the table without editing standup.py."""
    custom = {"bobby": "Shawn"}
    resolution = standup.resolve_speaker("Bobby", ROSTER_NAMES, alias_table=custom)
    assert resolution.canonical == "Shawn"
    assert resolution.via_alias is True
    # The built-in Tina->Dina mapping does not leak into a caller-supplied table.
    assert standup.resolve_speaker("Tina", ROSTER_NAMES, alias_table=custom).canonical is None


# ---------------------------------------------------------------------------
# Action items: deep links
# ---------------------------------------------------------------------------


def test_action_item_produces_a_deep_link_with_correct_id_and_seconds():
    transcript = _transcript(
        "01ABCXYZ",
        _AUG_13_MS,
        participants=[],
        meeting_attendees=[],
        sentences=[
            _sentence(0, "Tam", "Not a commitment.", 1.0, task=False),
            _sentence(1, "Tam", "I'll ship VINO-42 by Friday.", 137.8, task=True),
        ],
    )
    items = standup.extract_action_items(transcript, roster_names=ROSTER_NAMES)
    assert len(items) == 1  # the non-task sentence is excluded
    row = items.iloc[0]
    assert row["ticket_key"] == "VINO-42"
    assert row["deep_link"] == "https://app.fireflies.ai/view/01ABCXYZ?t=138"


def test_deep_link_helper_rounds_and_floors_at_zero():
    assert standup.deep_link("abc", 12.4) == "https://app.fireflies.ai/view/abc?t=12"
    assert standup.deep_link("abc", -5.0) == "https://app.fireflies.ai/view/abc?t=0"


# ---------------------------------------------------------------------------
# Joining action items to the Jira changelog
# ---------------------------------------------------------------------------


def test_action_item_followed_by_matching_changelog_entry_counts_as_followed_through():
    transcript = _transcript(
        "tr-1",
        _AUG_06_MS,
        participants=[],
        meeting_attendees=[],
        sentences=[_sentence(0, "Tam", "I'll finish VINO-100 today.", 20.0, task=True)],
    )
    issues = [
        _changelog_issue(
            "VINO-100", "Tam", "In Progress", "Done", "2026-08-08T09:00:00.000+0000"
        )
    ]
    events = integrity.changelog_events(issues)
    matched = standup.match_action_items(
        standup.extract_action_items(transcript, roster_names=ROSTER_NAMES), events
    )
    row = matched.iloc[0]
    assert row["outcome"] == "followed_through"
    assert row["matched_by"] == "ticket_key"
    assert row["matched_key"] == "VINO-100"


def test_action_item_with_no_matching_entry_is_reported_open_not_a_lie():
    transcript = _transcript(
        "tr-2",
        _AUG_06_MS,
        participants=[],
        meeting_attendees=[],
        sentences=[_sentence(0, "Tam", "I'll finish VINO-999 today.", 20.0, task=True)],
    )
    events = integrity.empty_events()
    matched = standup.match_action_items(
        standup.extract_action_items(transcript, roster_names=ROSTER_NAMES), events
    )
    row = matched.iloc[0]
    assert row["outcome"] == "open"
    assert row["outcome"] not in {"lie", "broken", "missed", "false"}
    assert row["matched_key"] is None


def test_changelog_match_outside_the_followthrough_window_stays_open():
    transcript = _transcript(
        "tr-3",
        _AUG_06_MS,
        participants=[],
        meeting_attendees=[],
        sentences=[_sentence(0, "Tam", "I'll finish VINO-100 today.", 20.0, task=True)],
    )
    # A month later, well outside the default 7-day follow-through window.
    issues = [
        _changelog_issue(
            "VINO-100", "Tam", "In Progress", "Done", "2026-09-10T09:00:00.000+0000"
        )
    ]
    events = integrity.changelog_events(issues)
    matched = standup.match_action_items(
        standup.extract_action_items(transcript, roster_names=ROSTER_NAMES), events
    )
    assert matched.iloc[0]["outcome"] == "open"


def test_action_item_without_a_named_ticket_can_still_match_by_author_activity():
    transcript = _transcript(
        "tr-4",
        _AUG_06_MS,
        participants=[],
        meeting_attendees=[],
        sentences=[_sentence(0, "Shawn", "I'll get the search fix out today.", 20.0, task=True)],
    )
    issues = [
        _changelog_issue(
            "VINO-7", "Shawn", "In Progress", "Done", "2026-08-07T09:00:00.000+0000"
        )
    ]
    events = integrity.changelog_events(issues)
    matched = standup.match_action_items(
        standup.extract_action_items(transcript, roster_names=ROSTER_NAMES), events
    )
    row = matched.iloc[0]
    assert row["outcome"] == "followed_through"
    assert row["matched_by"] == "author_activity"
    assert row["matched_key"] == "VINO-7"


# ---------------------------------------------------------------------------
# No-notice absences
# ---------------------------------------------------------------------------


def test_absence_with_notice_is_excluded_from_the_no_notice_count():
    flagged_absence = _transcript(
        "tr-standup-2",
        _AUG_13_MS,
        participants=[],
        meeting_attendees=[],
        sentences=[
            _sentence(0, "Tam", "Heads up, Shawn is out sick today.", 1.0),
            _sentence(1, "Dina QA", "Got it, I'll cover his review.", 5.0),
        ],
    )
    absences = standup.no_notice_absences([flagged_absence], active_roster_names=ACTIVE_ROSTER_NAMES)
    shawn_rows = absences.loc[absences["person"] == "Shawn"]
    assert len(shawn_rows) == 1
    assert bool(shawn_rows.iloc[0]["notice"]) is True

    counts = standup.no_notice_absence_count(absences)
    assert "Shawn" not in set(counts.get("person", []))


def test_absence_with_no_notice_is_counted():
    transcript = _transcript(
        "tr-standup-3",
        _AUG_06_MS,
        participants=[],
        meeting_attendees=[],
        sentences=[_sentence(0, "Tam", "Just a quick sync today.", 1.0)],
    )
    absences = standup.no_notice_absences([transcript], active_roster_names=ACTIVE_ROSTER_NAMES)
    counts = standup.no_notice_absence_count(absences)
    row = counts.loc[counts["person"] == "Shawn"]
    assert len(row) == 1
    assert int(row.iloc[0]["no_notice_absences"]) == 1


def test_a_person_who_spoke_is_never_counted_as_absent():
    transcript = _transcript(
        "tr-standup-4",
        _AUG_06_MS,
        participants=[],
        meeting_attendees=[],
        sentences=[
            _sentence(0, "Tam", "Update from me.", 1.0),
            _sentence(1, "Shawn", "Update from me too.", 5.0),
            _sentence(2, "Dina QA", "QA update.", 9.0),
        ],
    )
    absences = standup.no_notice_absences([transcript], active_roster_names=ACTIVE_ROSTER_NAMES)
    assert absences.empty


# ---------------------------------------------------------------------------
# End-to-end truth_check wiring
# ---------------------------------------------------------------------------


def test_truth_check_end_to_end_over_a_small_batch():
    transcripts = [
        _transcript(
            "tr-e2e",
            _AUG_06_MS,
            participants=["dat@vinovoss.com"],
            meeting_attendees=[{"displayName": "Dat", "email": "dat@vinovoss.com"}],
            sentences=[
                _sentence(0, "Tam", "I'll fix VINO-100.", 12.0, task=True),
                _sentence(1, "Tina", "I'll retest VINO-200.", 40.0, task=True),
            ],
        )
    ]
    issues = [
        _changelog_issue("VINO-100", "Tam", "In Progress", "Done", "2026-08-07T09:00:00.000+0000")
    ]
    events = integrity.changelog_events(issues)
    result = standup.truth_check(
        transcripts,
        roster_names=ROSTER_NAMES,
        active_roster_names=ACTIVE_ROSTER_NAMES,
        former_staff=FORMER_STAFF,
        changelog_events=events,
    )
    assert result.available is True
    assert len(result.action_items) == 2
    assert set(result.matched_action_items["outcome"]) == {"followed_through", "open"}
    # "Tina" resolved through the alias table, not left unresolved.
    assert "Tina" not in result.unresolved_speakers
    assert "Dina QA" in set(result.action_items["speaker_canonical"])
    # Dat never spoke in this transcript, despite being in participants/invite.
    assert "Dat" not in set(result.attendance["speaker_canonical"].dropna())
