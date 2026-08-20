"""Offline checks for clockify.py: no key, no workspace, no network, ever.

Every test below either supplies no credentials at all (checking the
unavailable path) or injects a fake ``fetcher`` shaped like a real Clockify
detailed-report response (checking everything downstream of a fetch). Nothing
here calls ``requests`` for real - there is no key in this environment and
there will not be one during this task, which is the whole design constraint
the module is built around.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clockify  # noqa: E402

WEEK1 = dt.date(2026, 8, 3)  # Monday
WEEK2 = dt.date(2026, 8, 10)  # Monday
RANGE_START = dt.date(2026, 8, 3)
RANGE_END = dt.date(2026, 8, 16)  # Sunday - exactly two Mon-Sun weeks


def _entry(
    entry_id: str,
    *,
    user_id: str = "u1",
    user_email: str = "tam@vinovoss.com",
    user_name: str = "Tam",
    start: str,
    end: str,
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "userId": user_id,
        "userEmail": user_email,
        "userName": user_name,
        "timeInterval": {"start": start, "end": end},
    }


def _fetcher_for(entries: list[dict[str, Any]]):
    """A fake fetcher that pages through ``entries`` like a real detailed report."""

    def fetch(config, start, end, page):
        page_size = clockify._PAGE_SIZE
        chunk = entries[(page - 1) * page_size : page * page_size]
        return {"timeentries": chunk}

    return fetch


ENV = {"CLOCKIFY_API_KEY": "k", "CLOCKIFY_WORKSPACE_ID": "w"}


# ---------------------------------------------------------------------------
# Credentials: absent is a normal, well-typed outcome
# ---------------------------------------------------------------------------


def test_no_key_and_no_workspace_returns_unavailable_with_reason():
    result = clockify.weekly_billed_hours(
        ["Tam"], RANGE_START, RANGE_END, env={}, fetcher=_fetcher_for([])
    )
    assert result.available is False
    assert result.reason
    assert "CLOCKIFY_API_KEY" in result.reason
    assert "CLOCKIFY_WORKSPACE_ID" in result.reason
    assert result.weekly.empty
    assert result.entries.empty


def test_key_without_workspace_is_still_unavailable():
    env = {"CLOCKIFY_API_KEY": "k"}
    assert clockify.load_clockify_env(env) is None
    result = clockify.weekly_billed_hours(["Tam"], RANGE_START, RANGE_END, env=env)
    assert result.available is False
    assert "CLOCKIFY_WORKSPACE_ID" in result.reason


def test_workspace_without_key_is_still_unavailable():
    env = {"CLOCKIFY_WORKSPACE_ID": "w"}
    assert clockify.load_clockify_env(env) is None
    result = clockify.weekly_billed_hours(["Tam"], RANGE_START, RANGE_END, env=env)
    assert result.available is False
    assert "CLOCKIFY_API_KEY" in result.reason


def test_a_failing_fetch_degrades_instead_of_raising():
    def broken_fetcher(config, start, end, page):
        raise RuntimeError("workspace not found")

    result = clockify.weekly_billed_hours(
        ["Tam"], RANGE_START, RANGE_END, env=ENV, fetcher=broken_fetcher
    )
    assert result.available is False
    assert "workspace not found" in result.reason


def test_pagination_stops_on_a_short_page():
    full_page = clockify._PAGE_SIZE
    entries = [
        _entry(
            f"e{i}",
            start="2026-08-03T09:00:00Z",
            end="2026-08-03T09:05:00Z",
        )
        for i in range(full_page + 3)
    ]
    calls = []

    def counting_fetcher(config, start, end, page):
        calls.append(page)
        chunk = entries[(page - 1) * full_page : page * full_page]
        return {"timeentries": chunk}

    env = dict(ENV)
    env["CLOCKIFY_USER_MAP"] = "Tam=tam@vinovoss.com"
    result = clockify.weekly_billed_hours(
        ["Tam"], RANGE_START, RANGE_END, env=env, fetcher=counting_fetcher
    )
    assert result.available is True
    assert calls == [1, 2]  # first page full, second short - stop after two calls
    assert len(result.entries) == full_page + 3


# ---------------------------------------------------------------------------
# Person mapping: unmapped is never zero, Praveen Rai is its own state
# ---------------------------------------------------------------------------


def test_unmapped_person_renders_unmapped_never_zero():
    env = dict(ENV)  # no CLOCKIFY_USER_MAP entry for this name at all
    result = clockify.weekly_billed_hours(
        ["Some New Contractor"], RANGE_START, RANGE_END, env=env, fetcher=_fetcher_for([])
    )
    assert result.available is True
    rows = result.weekly
    assert (rows["status"] == "unmapped").all()
    assert rows["hours_billed"].apply(lambda v: pd.isna(v)).all()
    # Explicitly: nothing anywhere turns this into a numeric zero.
    assert not (rows["hours_billed"] == 0).any()


def test_praveen_rai_is_not_on_clockify_a_distinct_state():
    env = dict(ENV)
    result = clockify.weekly_billed_hours(
        ["Praveen Rai"], RANGE_START, RANGE_END, env=env, fetcher=_fetcher_for([])
    )
    assert result.available is True
    rows = result.weekly
    assert (rows["status"] == "not_on_clockify").all()
    assert not (rows["status"] == "unmapped").any()
    assert rows["hours_billed"].apply(lambda v: pd.isna(v)).all()


def test_resolve_person_status_three_states():
    user_map = {"tam": "tam@vinovoss.com"}
    assert clockify.resolve_person_status("Tam", user_map) == "mapped"
    assert clockify.resolve_person_status("Praveen Rai", user_map) == "not_on_clockify"
    assert clockify.resolve_person_status("Nobody Confirmed", user_map) == "unmapped"
    # Even if somehow mapped, Praveen Rai still reads as not_on_clockify - the
    # roster fact wins over an accidental mapping entry.
    user_map_with_praveen = {"praveen rai": "praveen@vinovoss.com"}
    assert clockify.resolve_person_status("Praveen Rai", user_map_with_praveen) == "not_on_clockify"


def test_mapped_person_gets_real_hours_including_a_true_zero_week():
    entries = [
        _entry("e1", start="2026-08-03T09:00:00Z", end="2026-08-03T17:00:00Z"),  # 8h
        _entry("e2", start="2026-08-04T09:00:00Z", end="2026-08-04T17:00:00Z"),  # 8h
        # nothing at all in WEEK2
    ]
    env = dict(ENV)
    env["CLOCKIFY_USER_MAP"] = "Tam=tam@vinovoss.com"
    result = clockify.weekly_billed_hours(
        ["Tam"], RANGE_START, RANGE_END, env=env, fetcher=_fetcher_for(entries)
    )
    weekly = result.weekly.set_index("week_start")
    assert weekly.loc[WEEK1, "status"] == "mapped"
    assert weekly.loc[WEEK1, "hours_billed"] == pytest.approx(16.0)
    assert weekly.loc[WEEK2, "status"] == "mapped"
    assert weekly.loc[WEEK2, "hours_billed"] == 0.0  # a real, earned zero
    assert not pd.isna(weekly.loc[WEEK2, "hours_billed"])


def test_billed_hours_by_person_excludes_unmapped_and_not_on_clockify():
    weekly = pd.DataFrame(
        [
            {"person": "Tam", "week_start": WEEK1, "hours_billed": 20.0, "status": "mapped"},
            {"person": "Tam", "week_start": WEEK2, "hours_billed": 10.0, "status": "mapped"},
            {"person": "Ghost", "week_start": WEEK1, "hours_billed": pd.NA, "status": "unmapped"},
            {"person": "Praveen Rai", "week_start": WEEK1, "hours_billed": pd.NA, "status": "not_on_clockify"},
        ]
    )
    totals = clockify.billed_hours_by_person(weekly)
    assert totals["Tam"] == pytest.approx(30.0)
    assert "Ghost" not in totals.index
    assert "Praveen Rai" not in totals.index


# ---------------------------------------------------------------------------
# MAD-based outliers, no fixed thresholds
# ---------------------------------------------------------------------------


def _weekly_row(person: str, week: dt.date, hours: float) -> dict[str, Any]:
    return {"person": person, "week_start": week, "hours_billed": hours, "status": "mapped"}


def test_self_history_outlier_flags_a_week_far_from_own_median():
    weeks = [WEEK1 + dt.timedelta(weeks=i) for i in range(7)]
    hours = [8, 8, 8, 8, 8, 8, 40]  # one wildly long week among six ordinary ones
    weekly = pd.DataFrame([_weekly_row("Gaston", w, h) for w, h in zip(weeks, hours)])
    out = clockify.self_history_outliers(weekly).set_index("week_start")
    assert bool(out.loc[weeks[-1], "is_outlier"]) is True
    assert bool(out.loc[weeks[0], "is_outlier"]) is False


def test_cohort_above_mean_but_not_a_mad_outlier_does_not_flag():
    # A cohort with real spread: 5, 6, 7, 8, 9, 10. The person at 10 is above
    # both the mean and the median, but the cohort's own typical deviation
    # (MAD) is wide enough that 10 is not far in the units that matter.
    week = WEEK1
    values = {"a": 5, "b": 6, "c": 7, "d": 8, "e": 9, "high": 10}
    weekly = pd.DataFrame([_weekly_row(p, week, h) for p, h in values.items()])
    role_of = {p: "backend" for p in values}
    out = clockify.cohort_outliers(weekly, role_of).set_index("person")
    assert bool(out.loc["high", "is_outlier"]) is False
    assert out.loc["high", "hours_billed"] > sum(values.values()) / len(values)  # genuinely above mean


def test_cohort_outlier_flags_someone_far_from_a_tight_cohort():
    week = WEEK1
    values = {"a": 8, "b": 8, "c": 8, "d": 8, "outlier": 60}
    weekly = pd.DataFrame([_weekly_row(p, week, h) for p, h in values.items()])
    role_of = {p: "backend" for p in values}
    out = clockify.cohort_outliers(weekly, role_of).set_index("person")
    assert bool(out.loc["outlier", "is_outlier"]) is True
    assert bool(out.loc["a", "is_outlier"]) is False


def test_cohort_outliers_needs_a_role_lookup_hit_to_score():
    weekly = pd.DataFrame([_weekly_row("Nobody's Role", WEEK1, 8.0)])
    out = clockify.cohort_outliers(weekly, {})
    assert out.empty


# ---------------------------------------------------------------------------
# Reconstruction tells
# ---------------------------------------------------------------------------


def _entries_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build an already-attributed entries frame the way weekly_billed_hours would."""
    raw = clockify._finish_entries(rows)
    raw["person"] = "Santi"
    return raw


def test_round_block_week_flags_exactly_eight_hour_days():
    rows = [
        {
            "entry_id": f"e{i}",
            "user_id": "u1",
            "user_email": "santi@vinovoss.com",
            "user_name": "Santi",
            "start": f"2026-08-{3+i:02d}T09:00:00Z",
            "end": f"2026-08-{3+i:02d}T17:00:00Z",  # exactly 8.00h, every day
        }
        for i in range(5)
    ]
    entries = _entries_frame(rows)
    result = clockify.detect_round_block_weeks(entries)
    assert result.available is True
    assert len(result.evidence) == 1
    assert result.evidence.iloc[0]["person"] == "Santi"
    assert result.evidence.iloc[0]["days"] == 5


def test_varied_daily_hours_do_not_flag_as_round_blocks():
    starts_ends = [
        ("2026-08-03T09:00:00Z", "2026-08-03T16:54:00Z"),  # 7.9h
        ("2026-08-04T09:00:00Z", "2026-08-04T17:06:00Z"),  # 8.1h
        ("2026-08-05T09:00:00Z", "2026-08-05T17:00:00Z"),  # 8.0h
    ]
    rows = [
        {
            "entry_id": f"e{i}",
            "user_id": "u1",
            "user_email": "santi@vinovoss.com",
            "user_name": "Santi",
            "start": s,
            "end": e,
        }
        for i, (s, e) in enumerate(starts_ends)
    ]
    entries = _entries_frame(rows)
    result = clockify.detect_round_block_weeks(entries)
    assert result.available is True
    assert result.evidence.empty


def test_one_block_per_day_flags_across_the_week():
    rows = [
        {
            "entry_id": f"e{i}",
            "user_id": "u1",
            "user_email": "santi@vinovoss.com",
            "user_name": "Santi",
            "start": f"2026-08-{3+i:02d}T09:00:00Z",
            "end": f"2026-08-{3+i:02d}T13:00:00Z",
        }
        for i in range(4)
    ]
    entries = _entries_frame(rows)
    result = clockify.detect_one_block_days(entries)
    assert len(result.evidence) == 1
    assert result.evidence.iloc[0]["days"] == 4


def test_many_timer_grained_entries_per_day_do_not_flag_one_block():
    rows = []
    for day in range(3, 7):
        for slot in range(4):
            hour = 9 + slot * 2
            rows.append(
                {
                    "entry_id": f"e{day}_{slot}",
                    "user_id": "u1",
                    "user_email": "santi@vinovoss.com",
                    "user_name": "Santi",
                    "start": f"2026-08-{day:02d}T{hour:02d}:00:00Z",
                    "end": f"2026-08-{day:02d}T{hour+1:02d}:00:00Z",
                }
            )
    entries = _entries_frame(rows)
    result = clockify.detect_one_block_days(entries)
    assert result.evidence.empty


def test_overlapping_entries_are_detected_with_the_pair_as_evidence():
    rows = [
        {
            "entry_id": "e1",
            "user_id": "u1",
            "user_email": "santi@vinovoss.com",
            "user_name": "Santi",
            "start": "2026-08-03T09:00:00Z",
            "end": "2026-08-03T13:00:00Z",
        },
        {
            "entry_id": "e2",
            "user_id": "u1",
            "user_email": "santi@vinovoss.com",
            "user_name": "Santi",
            "start": "2026-08-03T12:00:00Z",  # starts an hour before e1 ends
            "end": "2026-08-03T15:00:00Z",
        },
    ]
    entries = _entries_frame(rows)
    result = clockify.detect_overlaps(entries)
    assert result.available is True
    assert len(result.evidence) == 1
    row = result.evidence.iloc[0]
    assert {row["entry_id_a"], row["entry_id_b"]} == {"e1", "e2"}
    assert row["overlap_hours"] == pytest.approx(1.0)


def test_non_overlapping_entries_are_not_flagged():
    rows = [
        {
            "entry_id": "e1",
            "user_id": "u1",
            "user_email": "santi@vinovoss.com",
            "user_name": "Santi",
            "start": "2026-08-03T09:00:00Z",
            "end": "2026-08-03T12:00:00Z",
        },
        {
            "entry_id": "e2",
            "user_id": "u1",
            "user_email": "santi@vinovoss.com",
            "user_name": "Santi",
            "start": "2026-08-03T13:00:00Z",
            "end": "2026-08-03T17:00:00Z",
        },
    ]
    entries = _entries_frame(rows)
    result = clockify.detect_overlaps(entries)
    assert result.evidence.empty


def test_created_at_tell_is_explicitly_absent_not_faked():
    entries = _entries_frame(
        [
            {
                "entry_id": "e1",
                "user_id": "u1",
                "user_email": "santi@vinovoss.com",
                "user_name": "Santi",
                "start": "2026-08-03T09:00:00Z",
                "end": "2026-08-03T17:00:00Z",
            }
        ]
    )
    result = clockify.late_created_tell(entries)
    assert result.available is False
    assert result.reason
    assert "createdAt" in result.reason or "created" in result.reason.lower()
    assert result.evidence.empty
    # Pin the absence at the source too: our own entry frame - built straight
    # from Clockify's documented response fields - never carries a created-at
    # column for this tell to (mis)use.
    assert "created_at" not in entries.columns
    assert "createdAt" not in entries.columns


# ---------------------------------------------------------------------------
# Empty / degenerate inputs never raise
# ---------------------------------------------------------------------------


def test_empty_entries_frame_is_safe_for_every_detector():
    empty = clockify._empty_entries()
    assert clockify.detect_round_block_weeks(empty).evidence.empty
    assert clockify.detect_one_block_days(empty).evidence.empty
    assert clockify.detect_overlaps(empty).evidence.empty
    assert clockify.late_created_tell(empty).available is False


def test_empty_weekly_frame_is_safe_for_outlier_functions():
    empty = clockify._empty_weekly()
    assert clockify.self_history_outliers(empty).empty
    assert clockify.cohort_outliers(empty, {}).empty
