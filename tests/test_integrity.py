"""Offline checks for the changelog integrity module.

The module exists because every other number on the board can be reset with a
keystroke, so these tests are written as the adversary: each one builds the
smallest changelog that fakes a good week, and asserts the module sees through
it. The four named personas are the label-edit-only toucher, the mid-flight
estimate raiser, the staging ping-ponger and the reopen-then-re-resolve hider.

Fixtures are hand-built Jira changelog payloads rather than recorded ones, so a
reader can see exactly which field, timestamp and author produced each finding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import integrity  # noqa: E402
import prioritization  # noqa: E402


# Anchored to the running clock, not to a date typed into the file. The code
# under test measures staleness against the real now, so a fixed anchor here
# does not freeze the fixture - it only fixes one end of the interval and lets
# the other keep moving. A ticket written as "moved 40 days ago" was 40 days
# stale on the day the anchor was typed and 58 days stale a fortnight later,
# which is how a passing suite turns red having tested nothing new.
#
# Floored to the minute so durations between two fixture timestamps come out
# exact rather than carrying whatever microsecond the suite happened to start on.
NOW = pd.Timestamp.now(tz="UTC").floor("min")


def when(days_ago: float, hour: int = 9, minute: int = 0) -> str:
    """A Jira-shaped timestamp ``days_ago`` days before now."""
    stamp = (NOW - pd.Timedelta(days=days_ago)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.000+0000")


def item(field: str, from_string: object, to_string: object) -> dict:
    return {
        "field": field,
        "fieldId": field,
        "fieldtype": "jira",
        "fromString": from_string,
        "toString": to_string,
    }


def status(from_status: str, to_status: str) -> dict:
    return item("status", from_status, to_status)


def history(created: str, author: str, *items: dict, entry_id: str | None = None) -> dict:
    return {
        "id": entry_id or f"{author}-{created}",
        "created": created,
        "author": {"displayName": author, "accountId": author.lower()},
        "items": list(items),
    }


def issue(key: str, *histories: dict) -> dict:
    return {"key": key, "changelog": {"histories": list(histories)}}


def tickets(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# --------------------------------------------------------------------------
# The parser
# --------------------------------------------------------------------------


def test_a_changelog_becomes_one_row_per_changed_field():
    events = integrity.changelog_events(
        [
            issue(
                "VV-1",
                history(when(3), "Ana", status("To Do", "In Progress"), item("labels", "", "wip")),
            )
        ]
    )
    assert list(events["key"]) == ["VV-1", "VV-1"]
    assert set(events["field"]) == {"status", "labels"}
    # One save, so one entry id: a five-field save is one action, not five.
    assert events["entry_id"].nunique() == 1
    assert events.loc[events["field"] == "status", "is_status"].all()
    assert events.loc[events["field"] == "labels", "is_cosmetic"].all()
    assert events.loc[events["field"] == "status", "from_stage"].iloc[0] == 1.0
    assert events.loc[events["field"] == "status", "to_stage"].iloc[0] == 2.0


def test_the_parser_takes_issues_a_frame_or_a_mapping():
    raw = issue("VV-1", history(when(1), "Ana", status("To Do", "In Progress")))
    from_list = integrity.changelog_events([raw])
    from_frame = integrity.changelog_events(
        pd.DataFrame([{"key": "VV-1", "changelog": raw["changelog"]}])
    )
    from_mapping = integrity.changelog_events({"VV-1": raw["changelog"]["histories"]})
    for frame in (from_frame, from_mapping):
        assert list(frame["key"]) == list(from_list["key"])
        assert list(frame["field"]) == list(from_list["field"])


def test_no_changelog_at_all_is_an_empty_frame_not_a_crash():
    for source in (None, [], {}, pd.DataFrame(), [{"key": "VV-1"}]):
        events = integrity.changelog_events(source)
        assert events.empty
        assert set(integrity.EVENT_COLUMNS) <= set(events.columns)


def test_a_ticket_with_no_changelog_in_the_frame_is_skipped_not_fatal():
    # A ticket created and never edited has no histories, and Jira sometimes
    # returns the column empty; neither is an error.
    raw = issue("VV-1", history(when(1), "Ana", status("To Do", "In Progress")))
    frame = pd.DataFrame(
        [
            {"key": "VV-1", "changelog": raw["changelog"]},
            {"key": "VV-2", "changelog": None},
            {"key": "VV-3", "changelog": {"histories": []}},
        ]
    )
    events = integrity.changelog_events(frame)
    assert list(events["key"]) == ["VV-1"]


def test_events_are_sorted_by_ticket_then_time():
    events = integrity.changelog_events(
        [
            issue(
                "VV-2",
                history(when(1), "Ana", status("In Progress", "Done")),
                history(when(9), "Ana", status("To Do", "In Progress")),
            )
        ]
    )
    assert list(events["ts"]) == sorted(events["ts"])


@pytest.mark.parametrize(
    "value,hours",
    [
        ("7200", 2.0),
        ("2h", 2.0),
        ("1d", 8.0),
        ("1w", 40.0),
        ("3h 30m", 3.5),
        ("1d 4h", 12.0),
    ],
)
def test_jira_writes_durations_three_different_ways(value, hours):
    assert integrity.parse_duration_hours(value) == pytest.approx(hours)


@pytest.mark.parametrize("value", ["", None, "later", "nan"])
def test_an_unreadable_duration_is_nan_not_zero(value):
    # Zero would silently read as "they removed the estimate", which is a
    # different and much more damning claim than "this could not be parsed".
    assert pd.isna(integrity.parse_duration_hours(value))


# --------------------------------------------------------------------------
# 1. The honest staleness clock
# --------------------------------------------------------------------------


def test_a_label_edit_resets_idle_days_but_not_the_status_clock():
    events = integrity.changelog_events(
        [
            issue(
                "VV-1",
                history(when(40), "Ana", status("To Do", "In Progress")),
                # Yesterday's five-second label pass: the board now reads fresh.
                history(when(1), "Ana", item("labels", "", "reviewed")),
            )
        ]
    )
    board = tickets(
        {"key": "VV-1", "assignee": "Ana", "status": "In Progress", "idle_days": 1.0}
    )
    aged = integrity.status_age_days(board, events, now=NOW)
    row = aged.iloc[0]
    assert row["status_age_days"] == pytest.approx(40.125, abs=0.2)
    assert row["idle_days"] == 1.0
    assert row["masked_days"] == pytest.approx(39.1, abs=0.3)
    assert row["age_source"] == "status change"
    assert row["status_changes"] == 1


def test_a_ticket_that_never_moved_ages_from_its_creation_date():
    events = integrity.changelog_events(
        [issue("VV-2", history(when(2), "Ana", item("description", "old", "new")))]
    )
    board = tickets(
        {
            "key": "VV-2",
            "assignee": "Ana",
            "status": "Backlog",
            "idle_days": 2.0,
            "created": (NOW - pd.Timedelta(days=60)).isoformat(),
        }
    )
    row = integrity.status_age_days(board, events, now=NOW).iloc[0]
    assert row["age_source"] == "created"
    assert row["status_age_days"] == pytest.approx(60.0, abs=0.1)


def test_status_age_of_an_empty_board_is_an_empty_frame():
    assert integrity.status_age_days(pd.DataFrame(), integrity.empty_events()).empty


# --------------------------------------------------------------------------
# 2. The label-edit-only toucher
# --------------------------------------------------------------------------


def groomer_events() -> pd.DataFrame:
    """Twelve label edits in one afternoon, no status transition anywhere."""
    return integrity.changelog_events(
        [
            issue(
                f"VV-{n}",
                history(when(2, hour=14, minute=n), "Groomer", item("labels", "", "triaged")),
            )
            for n in range(1, 13)
        ]
    )


def test_a_label_pass_across_twenty_tickets_shows_as_touches_with_no_movement():
    touched = integrity.cosmetic_touches(groomer_events(), window_days=14, now=NOW)
    row = touched[touched["person"] == "Groomer"].iloc[0]
    assert row["cosmetic_touches"] == 12
    assert row["cosmetic_tickets"] == 12
    assert row["status_transitions"] == 0
    # No transitions at all: a ratio would be a division by zero dressed up as
    # a number, so the count stands on its own.
    assert pd.isna(row["cosmetic_per_transition"])
    assert row["busiest_day_touches"] == 12
    assert "VV-1" in row["evidence"]
    assert len(row["timestamps"]) == 12


def test_a_save_that_moved_the_status_is_a_transition_not_a_touch():
    events = integrity.changelog_events(
        [
            issue(
                "VV-1",
                # Status and labels in one click. Counting items instead of saves
                # would score this person a cosmetic touch for doing real work.
                history(when(1), "Ana", status("To Do", "In Progress"), item("labels", "", "wip")),
            )
        ]
    )
    row = integrity.cosmetic_touches(events, window_days=14, now=NOW).iloc[0]
    assert row["cosmetic_touches"] == 0
    assert row["status_transitions"] == 1


def test_touches_belong_to_whoever_typed_them_not_to_the_assignee():
    events = integrity.changelog_events(
        [issue("VV-1", history(when(1), "Bob", item("priority", "Low", "High")))]
    )
    touched = integrity.cosmetic_touches(events, window_days=14, now=NOW)
    # The ticket is Ana's; the edit is Bob's.
    assert list(touched["person"]) == ["Bob"]


def test_a_sprint_rollover_is_the_boards_doing_not_a_persons():
    events = integrity.changelog_events(
        [
            issue(
                "VV-1",
                history(when(1), "Admin", item("Sprint", "ML Sprint 41", "ML Sprint 42")),
            )
        ]
    )
    assert integrity.cosmetic_touches(events, window_days=14, now=NOW).empty


def test_taking_a_ticket_and_handing_it_straight_back_is_a_round_trip():
    events = integrity.changelog_events(
        [
            issue(
                "VV-1",
                history(when(3), "Ana", item("assignee", "Bob", "Ana")),
                history(when(2), "Ana", item("assignee", "Ana", "Bob")),
                history(when(1), "Ana", item("assignee", "Bob", "Ana")),
            )
        ]
    )
    row = integrity.cosmetic_touches(events, window_days=14, now=NOW).iloc[0]
    assert row["cosmetic_touches"] == 3
    assert row["assignee_roundtrips"] >= 1


def test_touches_outside_the_window_do_not_count():
    events = groomer_events()
    assert integrity.cosmetic_touches(events, window_days=1, now=NOW).empty


# --------------------------------------------------------------------------
# 3. The mid-flight estimate raiser
# --------------------------------------------------------------------------


def raiser_events() -> pd.DataFrame:
    return integrity.changelog_events(
        [
            issue(
                "VV-7",
                # Estimated honestly while planning.
                history(when(30), "Raiser", item("timeoriginalestimate", None, "4h")),
                history(when(20), "Raiser", status("To Do", "In Progress")),
                # Ten days into the work, the bill doubles. Twice.
                history(when(10), "Raiser", item("timeoriginalestimate", "4h", "12h")),
                history(when(5), "Raiser", status("In Progress", "Code Review")),
                history(when(4), "Raiser", item("timeoriginalestimate", "12h", "20h")),
            )
        ]
    )


def test_an_estimate_raised_after_the_work_started_is_reported_with_its_size():
    churn = integrity.estimate_churn(raiser_events(), window_days=90, now=NOW)
    assert list(churn["direction"]) == ["raised", "raised"]
    assert list(churn["delta_hours"]) == [8.0, 8.0]
    assert churn["days_after_start"].iloc[0] == pytest.approx(10.0, abs=0.1)
    # The status the ticket sat in when the number changed is the context that
    # makes the row worth reading.
    assert churn["status_at_change"].iloc[0] == "In Progress"
    assert churn["status_at_change"].iloc[1] == "Code Review"


def test_an_estimate_written_before_work_started_is_planning_not_padding():
    churn = integrity.estimate_churn(raiser_events(), window_days=90, now=NOW)
    assert len(churn) == 2  # the 30-days-ago original is excluded
    with_planning = integrity.estimate_churn(
        raiser_events(), window_days=90, now=NOW, include_pre_start=True
    )
    assert len(with_planning) == 3
    assert pd.isna(with_planning["days_after_start"].iloc[0])


def test_an_estimate_recorded_in_seconds_reads_the_same_as_one_in_hours():
    events = integrity.changelog_events(
        [
            issue(
                "VV-8",
                history(when(9), "Raiser", status("To Do", "In Progress")),
                history(when(3), "Raiser", item("timeoriginalestimate", "3600", "18000")),
            )
        ]
    )
    row = integrity.estimate_churn(events, window_days=90, now=NOW).iloc[0]
    assert row["old_hours"] == 1.0
    assert row["new_hours"] == 5.0
    assert row["direction"] == "raised"


def test_a_first_estimate_added_mid_flight_is_set_not_raised():
    events = integrity.changelog_events(
        [
            issue(
                "VV-9",
                history(when(9), "Raiser", status("To Do", "In Progress")),
                history(when(3), "Raiser", item("timeoriginalestimate", None, "16h")),
            )
        ]
    )
    row = integrity.estimate_churn(events, window_days=90, now=NOW).iloc[0]
    assert row["direction"] == "set"
    assert pd.isna(row["delta_hours"])


def test_a_lowered_estimate_is_recorded_too():
    events = integrity.changelog_events(
        [
            issue(
                "VV-10",
                history(when(9), "Ana", status("To Do", "In Progress")),
                history(when(3), "Ana", item("timeoriginalestimate", "8h", "4h")),
            )
        ]
    )
    row = integrity.estimate_churn(events, window_days=90, now=NOW).iloc[0]
    assert row["direction"] == "lowered"
    assert row["delta_hours"] == -4.0


# --------------------------------------------------------------------------
# 4. The reopen-then-re-resolve hider
# --------------------------------------------------------------------------


def hider_events() -> pd.DataFrame:
    return integrity.changelog_events(
        [
            issue(
                "VV-20",
                history(when(60), "Hider", status("To Do", "In Progress")),
                history(when(50), "Hider", status("In Progress", "Done")),
                # It came back, was fixed, and was closed again. The reopened
                # JQL asks only whether it is out of Done today: it is not.
                history(when(30), "Tester", status("Done", "In Progress")),
                history(when(20), "Hider", status("In Progress", "Done")),
            )
        ]
    )


def test_a_ticket_reopened_and_re_resolved_still_counts_as_rework():
    bounced = integrity.reresolve_events(
        hider_events(),
        tickets({"key": "VV-20", "status": "Done"}),
        window_days=90,
        now=NOW,
    )
    row = bounced.iloc[0]
    assert row["resolutions"] == 2
    assert row["reopens"] == 1
    assert bool(row["currently_resolved"])
    # The whole point: the scorecard's reopened count reports 0 for this ticket.
    assert bool(row["hidden_rework"])
    assert row["resolvers"] == "Hider"
    assert row["reopeners"] == "Tester"
    assert len(row["timestamps"]) == 2


def test_walking_up_the_tail_of_the_pipeline_is_one_resolution_not_three():
    # Review in Staging, Ready for Production and Released are all "resolved" by
    # the team's list. Counting every entry would flag every shipped ticket.
    events = integrity.changelog_events(
        [
            issue(
                "VV-21",
                history(when(10), "Ana", status("In Progress", "Review in Staging")),
                history(when(9), "Ana", status("Review in Staging", "Ready for Production")),
                history(when(8), "Ana", status("Ready for Production", "Released")),
            )
        ]
    )
    row = integrity.reresolve_events(events, window_days=90, now=NOW).iloc[0]
    assert row["resolutions"] == 1
    assert row["reopens"] == 0
    assert not row["hidden_rework"]


def test_a_ticket_still_out_of_done_is_rework_the_old_metric_already_saw():
    events = integrity.changelog_events(
        [
            issue(
                "VV-22",
                history(when(20), "Ana", status("In Progress", "Done")),
                history(when(5), "Tester", status("Done", "In Progress")),
            )
        ]
    )
    row = integrity.reresolve_events(
        events, tickets({"key": "VV-22", "status": "In Progress"}), window_days=90, now=NOW
    ).iloc[0]
    assert row["resolutions"] == 1
    assert not row["currently_resolved"]
    assert not row["hidden_rework"]


def test_bounces_older_than_the_window_are_out_of_the_count():
    assert integrity.reresolve_events(hider_events(), window_days=10, now=NOW).empty


# --------------------------------------------------------------------------
# 5. The staging ping-ponger
# --------------------------------------------------------------------------


def pingpong_events() -> pd.DataFrame:
    return integrity.changelog_events(
        [
            issue(
                "VV-30",
                history(when(20), "Ponger", status("In Progress", "Review in Staging")),
                history(when(18), "Ponger", status("Review in Staging", "In Progress")),
                history(when(16), "Ponger", status("In Progress", "Review in Staging")),
                history(when(14), "Ponger", status("Review in Staging", "In Progress")),
                history(when(12), "Ponger", status("In Progress", "Review in Staging")),
            )
        ]
    )


def test_a_ticket_bounced_through_staging_shows_every_lap():
    loops = integrity.status_pingpong(pingpong_events(), window_days=90, now=NOW)
    row = loops.iloc[0]
    assert row["key"] == "VV-30"
    assert row["backward_transitions"] == 2  # staging -> in progress, twice
    # Three entries into staging, two into In Progress: four repeats in total.
    assert row["repeat_entries"] == 3
    # Two of those re-entries minted resolution credit a second and third time.
    assert row["staging_entries"] == 2
    assert row["movers"] == "Ponger"


def test_a_ticket_that_walked_forward_once_is_not_in_the_pingpong_list():
    events = integrity.changelog_events(
        [
            issue(
                "VV-31",
                history(when(10), "Ana", status("To Do", "In Progress")),
                history(when(9), "Ana", status("In Progress", "Code Review")),
                history(when(8), "Ana", status("Code Review", "Done")),
            )
        ]
    )
    assert integrity.status_pingpong(events, window_days=90, now=NOW).empty


def test_a_status_the_module_has_never_seen_is_not_guessed_at():
    # No rank, so no direction: the repetition half of the check still fires,
    # the direction half stays silent rather than inventing a workflow.
    events = integrity.changelog_events(
        [
            issue(
                "VV-32",
                history(when(9), "Ana", status("Bespoke Stage", "Another Stage")),
                history(when(8), "Ana", status("Another Stage", "Bespoke Stage")),
                history(when(7), "Ana", status("Bespoke Stage", "Another Stage")),
            )
        ]
    )
    row = integrity.status_pingpong(events, window_days=90, now=NOW).iloc[0]
    assert row["backward_transitions"] == 0
    # "Another Stage" was entered twice; one entry after the first is one repeat.
    assert row["repeat_entries"] == 1


# --------------------------------------------------------------------------
# 6. Cycle time
# --------------------------------------------------------------------------


def test_time_in_each_status_comes_from_consecutive_transitions():
    events = integrity.changelog_events(
        [
            issue(
                "VV-40",
                history(when(10), "Ana", status("To Do", "In Progress")),
                history(when(6), "Ana", status("In Progress", "Code Review")),
                history(when(4), "Ana", status("Code Review", "Done")),
            )
        ]
    )
    detail, by_person = integrity.cycle_time(
        events, tickets({"key": "VV-40", "assignee": "Ana"}), now=NOW
    )
    spans = dict(zip(detail["status"], detail["days"]))
    assert spans["In Progress"] == pytest.approx(4.0, abs=0.01)
    assert spans["Code Review"] == pytest.approx(2.0, abs=0.01)
    # The ticket is still sitting in Done, and that interval runs to now.
    assert detail[detail["status"] == "Done"]["is_open"].all()
    row = by_person.iloc[0]
    assert row["person"] == "Ana"
    assert row["median_lead_time_days"] == pytest.approx(6.0, abs=0.01)
    assert row["median_in_progress_days"] == pytest.approx(4.0, abs=0.01)
    assert row["median_review_days"] == pytest.approx(2.0, abs=0.01)


def test_the_wait_before_the_first_transition_counts_when_the_board_says_when_it_was_created():
    events = integrity.changelog_events(
        [
            issue(
                "VV-41",
                # Taken at the anchor's own time of day, so the gap below is a
                # whole number of days. The old expectation of 29.875 was that
                # whole month minus the three hours between this fixture's 09:00
                # default and an anchor that happened to be written as 12:00 -
                # an artifact of the constant, never a property worth asserting.
                history(
                    when(5, hour=NOW.hour, minute=NOW.minute),
                    "Ana",
                    status("Backlog", "In Progress"),
                )
            )
        ]
    )
    detail, _ = integrity.cycle_time(
        events,
        tickets(
            {
                "key": "VV-41",
                "assignee": "Ana",
                "created": (NOW - pd.Timedelta(days=35)).isoformat(),
            }
        ),
        now=NOW,
    )
    backlog = detail[detail["status"] == "Backlog"].iloc[0]
    # Created 35 days ago, first moved 5 days ago: a month in Backlog that no
    # other metric on the board reports.
    assert backlog["days"] == pytest.approx(30.0, abs=0.01)


def test_without_a_ticket_frame_the_cycle_belongs_to_whoever_started_it():
    events = integrity.changelog_events(
        [
            issue(
                "VV-42",
                history(when(8), "Ana", status("To Do", "In Progress")),
                history(when(2), "Bob", status("In Progress", "Done")),
            )
        ]
    )
    _, by_person = integrity.cycle_time(events, now=NOW)
    assert list(by_person["person"]) == ["Ana"]
    assert by_person["median_lead_time_days"].iloc[0] == pytest.approx(6.0, abs=0.01)


def test_cycle_time_of_nothing_is_two_empty_frames():
    detail, by_person = integrity.cycle_time(integrity.empty_events())
    assert detail.empty and by_person.empty
    assert "median_lead_time_days" in by_person.columns


# --------------------------------------------------------------------------
# 7. The flags, and the evidence behind them
# --------------------------------------------------------------------------


def test_the_groomer_trips_board_grooming_and_nothing_else():
    flags = integrity.integrity_flags(pd.DataFrame(), groomer_events(), window_days=30, now=NOW)
    row = flags[flags["person"] == "Groomer"].iloc[0]
    assert bool(row["board_grooming"])
    assert not row["estimate_inflation"]
    assert not row["staging_pingpong"]
    assert not row["rework_hidden"]
    assert row["flags"] == "board_grooming"
    # An accusation with no trail is useless in a one-on-one.
    assert "VV-1" in row["board_grooming_evidence"]
    assert "12 field-only edits vs 0 status moves" in row["board_grooming_evidence"]


def test_the_estimate_raiser_trips_estimate_inflation_with_the_hours_named():
    flags = integrity.integrity_flags(pd.DataFrame(), raiser_events(), window_days=30, now=NOW)
    row = flags[flags["person"] == "Raiser"].iloc[0]
    assert bool(row["estimate_inflation"])
    assert row["estimate_raises"] == 2
    assert row["hours_added"] == 16.0
    assert "VV-7" in row["estimate_inflation_evidence"]
    assert "+16.0h" in row["estimate_inflation_evidence"]


def test_the_ponger_trips_staging_pingpong_with_the_laps_named():
    flags = integrity.integrity_flags(pd.DataFrame(), pingpong_events(), window_days=30, now=NOW)
    row = flags[flags["person"] == "Ponger"].iloc[0]
    assert bool(row["staging_pingpong"])
    assert row["backward_moves"] == 2
    assert "VV-30" in row["staging_pingpong_evidence"]


def test_the_hider_trips_rework_hidden_and_the_tester_does_not():
    flags = integrity.integrity_flags(
        tickets({"key": "VV-20", "status": "Done"}),
        hider_events(),
        window_days=90,
        now=NOW,
    )
    hider = flags[flags["person"] == "Hider"].iloc[0]
    tester = flags[flags["person"] == "Tester"].iloc[0]
    assert bool(hider["rework_hidden"])
    assert hider["reresolved_tickets"] == 1
    assert "VV-20" in hider["rework_hidden_evidence"]
    # The person who found the bug is not the person who hid the rework.
    assert not tester["rework_hidden"]


def test_a_lead_who_grooms_and_also_ships_is_not_flagged_for_grooming():
    # Same twelve label edits, but this person moved twelve tickets as well.
    # Volume alone is not the signal; volume without movement is.
    working = integrity.changelog_events(
        [
            issue(
                f"VV-{n}",
                history(when(2, hour=14, minute=n), "Groomer", item("labels", "", "triaged")),
                history(when(1, hour=10, minute=n), "Groomer", status("To Do", "In Progress")),
            )
            for n in range(1, 13)
        ]
    )
    row = integrity.integrity_flags(pd.DataFrame(), working, window_days=30, now=NOW).iloc[0]
    assert row["cosmetic_touches"] == 12
    assert row["status_transitions"] == 12
    assert not row["board_grooming"]


def test_an_estimate_raise_is_charged_to_whoever_typed_it():
    events = integrity.changelog_events(
        [
            issue(
                "VV-60",
                history(when(20), "Ana", status("To Do", "In Progress")),
                # The manager raised it, not the assignee.
                history(when(3), "Lead", item("timeoriginalestimate", "2h", "20h")),
            )
        ]
    )
    flags = integrity.integrity_flags(
        tickets({"key": "VV-60", "assignee": "Ana", "status": "In Progress"}),
        events,
        window_days=30,
        now=NOW,
    )
    lead = flags[flags["person"] == "Lead"].iloc[0]
    assert bool(lead["estimate_inflation"])
    assert lead["hours_added"] == 18.0
    assert not flags[flags["person"] == "Ana"]["estimate_inflation"].any()


def test_activity_older_than_the_window_does_not_reach_the_flags():
    old = integrity.changelog_events(
        [
            issue(
                f"VV-{n}",
                history(when(200, hour=14, minute=n), "Groomer", item("labels", "", "old")),
            )
            for n in range(1, 13)
        ]
    )
    assert integrity.integrity_flags(pd.DataFrame(), old, window_days=30, now=NOW).empty


def test_an_honest_week_trips_nothing():
    events = integrity.changelog_events(
        [
            issue(
                "VV-50",
                history(when(9), "Ana", status("To Do", "In Progress")),
                history(when(6), "Ana", item("description", "thin", "acceptance criteria")),
                history(when(5), "Ana", status("In Progress", "Code Review")),
                history(when(3), "Ana", status("Code Review", "Done")),
            )
        ]
    )
    row = integrity.integrity_flags(pd.DataFrame(), events, window_days=30, now=NOW).iloc[0]
    assert row["flag_count"] == 0
    assert row["flags"] == ""
    assert all(not row[flag] for flag in integrity.FLAG_NAMES)
    assert all(row[f"{flag}_evidence"] == "" for flag in integrity.FLAG_NAMES)


def test_the_most_flagged_person_sorts_to_the_top():
    events = pd.concat(
        [groomer_events(), raiser_events(), pingpong_events()], ignore_index=True
    )
    flags = integrity.integrity_flags(pd.DataFrame(), events, window_days=30, now=NOW)
    assert list(flags["flag_count"]) == sorted(flags["flag_count"], reverse=True)
    assert set(flags["person"]) == {"Groomer", "Raiser", "Ponger"}


def test_every_flag_has_a_sentence_explaining_what_it_means():
    # The column names go in front of a CEO; each one needs a plain reading.
    assert set(integrity.FLAG_MEANINGS) == set(integrity.FLAG_NAMES)
    assert all(len(text) > 40 for text in integrity.FLAG_MEANINGS.values())


def test_flags_of_an_empty_changelog_are_an_empty_frame():
    flags = integrity.integrity_flags(pd.DataFrame(), integrity.empty_events())
    assert flags.empty
    assert "board_grooming_evidence" in flags.columns


# --------------------------------------------------------------------------
# 8. Staleness wiring: idle_days can no longer buy a ticket out of a queue
# --------------------------------------------------------------------------


def test_a_label_only_edit_does_not_drop_a_ticket_out_of_the_stale_queue():
    # Same fixture as the honest-staleness-clock test above: a ticket that
    # moved 40 days ago and had one label touched yesterday. idle_days reads
    # 1 - freshly touched - but the ticket has not moved in 40 days.
    events = integrity.changelog_events(
        [
            issue(
                "VV-1",
                history(when(40), "Ana", status("To Do", "In Progress")),
                history(when(1), "Ana", item("labels", "", "reviewed")),
            )
        ]
    )
    board = tickets(
        {
            "key": "VV-1",
            "assignee": "Ana",
            "status": "In Progress",
            "idle_days": 1.0,
            "ticket_age_days": 41.0,
            "priority": "Medium",
            "carry_over_count": 0,
        }
    )
    scored = prioritization.add_priority_score(board, events)
    # The score used the honest clock, not the gamed one.
    assert scored["staleness_days"].iloc[0] > 40.0
    assert scored["masked_days"].iloc[0] > 38.0
    # "stale 40d", not the "stale 4" prefix this used to look for: that prefix
    # matched anything from 4 to 49 days, so the drift below it had to reach a
    # fortnight before the assertion noticed.
    assert "stale 40d" in scored["priority_reasons"].iloc[0]

    rollup = prioritization.assignee_rollup(scored, events)
    row = rollup[rollup["assignee"] == "Ana"].iloc[0]
    # The queue a label edit was supposed to buy this ticket out of.
    assert row["stale_15d_plus"] == 1
    assert row["avg_status_age_days"] > 40.0

    # Without events (the pre-2A call sites), the same board is still read the
    # old way - idle_days alone - so the fix is opt-in, not a silent change of
    # every caller's numbers underneath them.
    scored_no_events = prioritization.add_priority_score(board)
    assert scored_no_events["staleness_days"].iloc[0] == 1.0
    rollup_no_events = prioritization.assignee_rollup(scored_no_events)
    assert rollup_no_events[rollup_no_events["assignee"] == "Ana"].iloc[0]["stale_15d_plus"] == 0
    assert "masked_days" not in scored_no_events.columns
    assert "avg_status_age_days" not in rollup_no_events.columns


def test_a_rollup_that_fell_back_to_idle_days_does_not_label_them_status_age(monkeypatch):
    """Events present is not the same as status ages derivable.

    ``_staleness_days`` falls back to ``idle_days`` whenever it cannot line a
    ``status_age_days`` frame up with the board, and signals that by returning
    ``masked_days`` as None. The rollup used to gate its status-age columns on
    "were events passed in" instead, so on the fallback it returned
    ``avg_status_age_days`` filled with idle days and ``avg_masked_days``
    all-NaN beside it - idle days under a column named for status age, which
    is the exact substitution this module exists to make impossible.

    Reaching the fallback on a keyed board takes a stub: ``status_age_days``
    returns one row per ticket for any board carrying ``key``, so today the
    mismatch only happens on a keyless frame, which ``assignee_rollup``'s own
    ``("key", "count")`` aggregation cannot process anyway. The gate is
    therefore defensive rather than a live bug being closed - but it is gated
    on the signal that means what it says.
    """
    board = pd.DataFrame(
        [
            {
                "key": "VV-80",
                "assignee": "Ana",
                "status": "In Progress",
                "priority": "High",
                "idle_days": 42.0,
                "ticket_age_days": 60.0,
                "carry_over_count": 0.0,
            }
        ]
    )
    events = integrity.changelog_events(
        [issue("VV-80", history(when(3), "Ana", status("To Do", "In Progress")))]
    )
    scored = prioritization.add_priority_score(board, events)

    # The mismatch _staleness_days guards against, forced.
    monkeypatch.setattr(prioritization.integrity, "status_age_days", lambda *a, **k: pd.DataFrame())
    rollup = prioritization.assignee_rollup(scored, events)

    assert not rollup.empty
    assert "avg_status_age_days" not in rollup.columns
    assert "max_status_age_days" not in rollup.columns
    assert "avg_masked_days" not in rollup.columns
    # The idle-day columns are still there, still honest about what they are.
    assert rollup[rollup["assignee"] == "Ana"].iloc[0]["avg_idle_days"] == 42.0


# --------------------------------------------------------------------------
# 9. Resolution credit: the changelog author, not the current assignee
# --------------------------------------------------------------------------


def test_resolution_credit_goes_to_the_changelog_author_not_the_assignee():
    events = integrity.changelog_events(
        [
            issue(
                "VV-70",
                history(when(20), "Ana", status("To Do", "In Progress")),
                history(when(5), "Ana", status("In Progress", "Done")),
            )
        ]
    )
    # The ticket sits assigned to someone else entirely today - Ana did the work.
    credit = integrity.credited_resolutions(
        events, tickets({"key": "VV-70", "assignee": "SomeoneElse", "status": "Done"}),
        window_days=90, now=NOW,
    )
    row = credit.by_person.iloc[0]
    assert row["person"] == "Ana"
    assert row["credited_resolutions"] == 1
    assert "VV-70" in row["evidence"]


def test_the_tail_of_the_pipeline_credits_one_resolution_not_three():
    events = integrity.changelog_events(
        [
            issue(
                "VV-71",
                history(when(10), "Ana", status("In Progress", "Review in Staging")),
                history(when(9), "Ana", status("Review in Staging", "Ready for Production")),
                history(when(8), "Ana", status("Ready for Production", "Released")),
            )
        ]
    )
    credit = integrity.credited_resolutions(events, window_days=90, now=NOW)
    assert len(credit.detail) == 1
    assert credit.by_person.iloc[0]["credited_resolutions"] == 1


def test_a_ticket_resolved_twice_credits_two_resolutions_not_one():
    # Genuinely different from the tail-walk: this ticket left a resolved
    # status and came back into one, so it earned the credit twice.
    credit = integrity.credited_resolutions(hider_events(), window_days=90, now=NOW)
    row = credit.by_person[credit.by_person["person"] == "Hider"].iloc[0]
    assert row["credited_resolutions"] == 2
    assert row["tickets"] == 1


def test_a_former_staff_resolution_is_flagged_not_counted():
    events = integrity.changelog_events(
        [
            issue(
                "VV-72",
                # A departed tester's account is still authoring transitions on
                # a ticket nobody reassigned - exploit #4 from KPI_SPEC.md.
                history(when(3), "Sai Shankar", status("In Progress", "Done")),
            )
        ]
    )
    credit = integrity.credited_resolutions(events, window_days=90, now=NOW)
    # Visible in the detail, and marked why.
    detail_row = credit.detail.iloc[0]
    assert bool(detail_row["is_former_staff"])
    assert not bool(detail_row["credited"])
    assert detail_row["reason"] == "former_staff"
    # Not in the credited ledger at all - "second highest in the company" for
    # someone who no longer works here is exactly what this closes.
    assert credit.by_person.empty or not (
        (credit.by_person["person"] == "Sai Shankar")
        & (credit.by_person["credited_resolutions"] > 0)
    ).any()


def test_a_custom_former_staff_list_overrides_the_default():
    events = integrity.changelog_events(
        [issue("VV-73", history(when(3), "Ana", status("In Progress", "Done")))]
    )
    credit = integrity.credited_resolutions(
        events, window_days=90, now=NOW, former_staff=["Ana"]
    )
    row = credit.detail.iloc[0]
    assert bool(row["is_former_staff"])
    assert not bool(row["credited"])


def test_credited_resolutions_of_an_empty_changelog_is_empty_not_a_crash():
    credit = integrity.credited_resolutions(integrity.empty_events())
    assert credit.detail.empty
    assert credit.by_person.empty
    assert "credited_resolutions" in credit.by_person.columns


def test_unattributed_resolutions_finds_a_resolution_with_no_named_author():
    # A resolving transition Jira recorded with no author object at all - the
    # raw shape ``changelog_events`` reduces to the "Unknown" sentinel.
    no_author_history = {
        "id": "no-author-1",
        "created": when(2),
        "author": {},
        "items": [status("In Progress", "Done")],
    }
    events = integrity.changelog_events(
        [
            {
                "key": "VV-74",
                "changelog": {
                    "histories": [
                        {
                            "id": "start-1",
                            "created": when(20),
                            "author": {"displayName": "Ana"},
                            "items": [status("To Do", "In Progress")],
                        },
                        no_author_history,
                    ]
                },
            }
        ]
    )
    unattributed = integrity.unattributed_resolutions(events, window_days=90, now=NOW)
    assert list(unattributed["key"]) == ["VV-74"]
    assert "author" not in unattributed.columns

    # And it must not crash on a changelog with no history in it at all.
    assert integrity.unattributed_resolutions(integrity.empty_events()).empty
    assert integrity.unattributed_resolutions(
        integrity.changelog_events([{"key": "VV-75"}])
    ).empty


def test_org_reopen_rate_returns_the_count_and_its_denominator():
    result = integrity.org_reopen_rate(hider_events(), window_days=90, now=NOW)
    assert result.resolved_count == 1  # VV-20
    assert result.count == 1  # it came back once
    assert result.share == pytest.approx(1.0)
    assert "VV-20" in result.reopened_keys


def test_org_reopen_rate_share_is_none_not_zero_with_nothing_resolved():
    result = integrity.org_reopen_rate(integrity.empty_events(), window_days=30, now=NOW)
    assert result.count == 0
    assert result.resolved_count == 0
    assert result.share is None


def test_org_reopen_rate_of_a_clean_resolution_is_a_real_zero():
    events = integrity.changelog_events(
        [issue("VV-76", history(when(5), "Ana", status("In Progress", "Done")))]
    )
    result = integrity.org_reopen_rate(events, window_days=90, now=NOW)
    assert result.resolved_count == 1
    assert result.count == 0
    assert result.share == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 10. The one cycle-time headline
# --------------------------------------------------------------------------


def _finished_ticket(key: str, started_days_ago: float, resolved_days_ago: float) -> dict:
    return issue(
        key,
        history(when(started_days_ago), "Ana", status("To Do", "In Progress")),
        history(when(resolved_days_ago), "Ana", status("In Progress", "Done")),
    )


def test_end_to_end_cycle_reports_none_with_a_reason_under_three_tickets():
    events = integrity.changelog_events(
        [_finished_ticket("VV-80", 10, 2), _finished_ticket("VV-81", 12, 4)]
    )
    result = integrity.end_to_end_cycle(events, now=NOW)
    assert result.median_days is None
    assert result.n == 2
    assert "fewer than 3" in result.reason


def test_end_to_end_cycle_medians_current_against_baseline():
    current = [
        _finished_ticket("VV-90", 20, 10),
        _finished_ticket("VV-91", 24, 12),
        _finished_ticket("VV-92", 28, 14),
    ]
    # Longer cycles a hundred-odd days ago, so the comparison actually differs.
    baseline = [
        _finished_ticket("VV-93", 150, 120),
        _finished_ticket("VV-94", 160, 130),
        _finished_ticket("VV-95", 170, 140),
    ]
    events = integrity.changelog_events(current + baseline)
    result = integrity.end_to_end_cycle(events, baseline_days=90, now=NOW)
    assert result.n == 3
    assert result.reason is None
    assert result.median_days == pytest.approx(12.0, abs=0.1)
    assert result.baseline_n == 3
    assert result.baseline_median_days == pytest.approx(30.0, abs=0.1)
    assert set(result.detail["period"]) == {"current", "baseline"}


def test_end_to_end_cycle_of_an_empty_changelog_states_why():
    result = integrity.end_to_end_cycle(integrity.empty_events())
    assert result.median_days is None
    assert result.reason
    assert result.detail.empty
