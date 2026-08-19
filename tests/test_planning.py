"""Offline checks for Task 2E's planning metrics.

Every case here is one of the acceptance bullets made concrete: a person
holding two boards' sprints at once and getting one true total, a sprint
missing its Jira dates staying out of that total instead of reading as
zero, a departed assignee turning up as evidence (and the common,
currently-real case of nobody turning up at all), carry-over staying a
per-sprint regroup of the existing field rather than a second copy of it,
and a person with no estimates rendering the ``unknown`` sentinel in a way
nothing downstream can quietly turn into ``0``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import capacity  # noqa: E402
import planning_metrics  # noqa: E402


# ---------------------------------------------------------------------------
# capacity.capacity_table_by_sprint
# ---------------------------------------------------------------------------


def _ticket(key, assignee, sprint_name, hours, sprint_start=None, sprint_end=None):
    return {
        "key": key,
        "assignee": assignee,
        "sprint_name": sprint_name,
        "sprint_start": sprint_start,
        "sprint_end": sprint_end,
        "original_estimate_sec": hours * 3600.0,
    }


def test_a_person_on_two_boards_gets_two_rows_and_one_correct_total():
    df = pd.DataFrame(
        [
            _ticket("APP-1", "Farid Shahidi", "App Sprint 12", 10, "2026-08-03", "2026-08-14"),
            _ticket("MKT-1", "Farid Shahidi", "Marketplace Sprint 7", 6, "2026-08-03", "2026-08-14"),
        ]
    )
    weekly_hours = {"Farid Shahidi": 20.0}

    rows, totals, excluded = capacity.capacity_table_by_sprint(df, weekly_hours)

    assert excluded == []
    person_rows = rows[rows["Assignee"] == "Farid Shahidi"]
    assert len(person_rows) == 2
    assert set(person_rows["Sprint"]) == {"App Sprint 12", "Marketplace Sprint 7"}

    total_row = totals[totals["Assignee"] == "Farid Shahidi"].iloc[0]
    # 10 working days at 20h/wk = 40h available per sprint; two sprints, one person.
    assert total_row["Committed (h)"] == pytest.approx(16.0)
    assert total_row["Available (h)"] == pytest.approx(80.0)
    assert set(total_row["Sprints"].split(", ")) == {"App Sprint 12", "Marketplace Sprint 7"}


def test_a_dateless_sprint_is_excluded_from_the_capacity_total_not_counted_as_zero():
    df = pd.DataFrame(
        [
            _ticket("APP-1", "Gaston", "App Sprint 12", 10, "2026-08-03", "2026-08-14"),
            # ML board: real ticket, real hours, but Jira has no sprint dates set.
            _ticket("ML-1", "Gaston", "ML Sprint 4", 15, None, None),
        ]
    )
    weekly_hours = {"Gaston": 20.0}

    rows, totals, excluded = capacity.capacity_table_by_sprint(df, weekly_hours)

    assert excluded == ["ML Sprint 4"]
    # Both sprints still produce their own row - the ML one just can't price hours.
    assert set(rows[rows["Assignee"] == "Gaston"]["Sprint"]) == {"App Sprint 12", "ML Sprint 4"}

    total_row = totals[totals["Assignee"] == "Gaston"].iloc[0]
    # Only the dated sprint's 10 committed hours count; the ML sprint's 15h
    # contribute nothing here - not a 0 folded into the sum, an omission.
    assert total_row["Committed (h)"] == pytest.approx(10.0)
    assert total_row["Sprints"] == "App Sprint 12"


def test_capacity_table_still_works_standalone_for_one_sprint():
    committed = pd.Series({"Tam": 30.0})
    table = capacity.capacity_table(committed, {"Tam": 40.0}, "2026-08-03", "2026-08-14")
    assert list(table["Assignee"]) == ["Tam"]
    assert table.iloc[0]["Available (h)"] == pytest.approx(80.0)


def test_capacity_table_by_sprint_is_empty_but_valid_with_no_data():
    rows, totals, excluded = capacity.capacity_table_by_sprint(pd.DataFrame(), {})
    assert rows.empty
    assert totals.empty
    assert excluded == []


# ---------------------------------------------------------------------------
# planning_metrics.ghost_assigned
# ---------------------------------------------------------------------------


def _board(rows):
    return pd.DataFrame(rows)


def test_ghost_assigned_reconciles_against_the_former_staff_list():
    df = _board(
        [
            {"key": "VIN-1", "assignee": "Sai Shankar", "status_category": "To Do"},
            {"key": "VIN-2", "assignee": "Farid Shahidi", "status_category": "In Progress"},
            {"key": "VIN-3", "assignee": "Shivanand", "status_category": "In Progress"},
            # Resolved work by a former staffer is not an open ghost ticket.
            {"key": "VIN-4", "assignee": "Sai Shankar", "status_category": "Done"},
        ]
    )
    result = planning_metrics.ghost_assigned(df, former_staff=("Sai Shankar", "Shivanand"))
    assert result.count == 2
    assert set(result.keys) == {"VIN-1", "VIN-3"}
    assert result.by_person == {"Sai Shankar": 1, "Shivanand": 1}


def test_ghost_assigned_uses_the_baked_fallback_when_the_env_var_is_unset():
    df = _board([{"key": "VIN-9", "assignee": "Dat", "status_category": "In Progress"}])
    result = planning_metrics.ghost_assigned(df, env={})
    assert result.count == 1
    assert result.keys == ("VIN-9",)


def test_ghost_assigned_returns_an_empty_but_valid_result_with_no_ghosts():
    # The actual, measured 19 Aug 2026 state of the board: zero open tickets
    # held by former staff. This must render legibly, not as None or a crash.
    df = _board(
        [
            {"key": "VIN-1", "assignee": "Farid Shahidi", "status_category": "In Progress"},
            {"key": "VIN-2", "assignee": "Tam", "status_category": "To Do"},
        ]
    )
    result = planning_metrics.ghost_assigned(df, former_staff=("Sai Shankar",))
    assert result is not None
    assert result.count == 0
    assert result.keys == ()
    assert result.by_person == {}


def test_ghost_assigned_handles_an_empty_board():
    result = planning_metrics.ghost_assigned(pd.DataFrame())
    assert result.count == 0
    assert result.keys == ()
    assert result.by_person == {}


# ---------------------------------------------------------------------------
# planning_metrics.no_priority_count / outside_any_sprint_count
# ---------------------------------------------------------------------------


def test_no_priority_count_only_counts_open_tickets_with_a_blank_priority():
    df = _board(
        [
            {"key": "VIN-1", "priority": "", "status_category": "To Do"},
            {"key": "VIN-2", "priority": None, "status_category": "In Progress"},
            {"key": "VIN-3", "priority": "High", "status_category": "To Do"},
            {"key": "VIN-4", "priority": "", "status_category": "Done"},
        ]
    )
    result = planning_metrics.no_priority_count(df)
    assert result.count == 2
    assert set(result.keys) == {"VIN-1", "VIN-2"}


def test_outside_any_sprint_count_only_counts_open_tickets_with_no_sprint():
    df = _board(
        [
            {"key": "VIN-1", "sprint_name": None, "status_category": "To Do"},
            {"key": "VIN-2", "sprint_name": "App Sprint 12", "status_category": "In Progress"},
            {"key": "VIN-3", "sprint_name": "", "status_category": "To Do"},
            {"key": "VIN-4", "sprint_name": None, "status_category": "Done"},
        ]
    )
    result = planning_metrics.outside_any_sprint_count(df)
    assert result.count == 2
    assert set(result.keys) == {"VIN-1", "VIN-3"}


# ---------------------------------------------------------------------------
# planning_metrics.carry_over_per_sprint
# ---------------------------------------------------------------------------


def test_carry_over_per_sprint_counts_tickets_not_the_raw_lifetime_number():
    df = _board(
        [
            # carry_over_count=3 (three closed sprints already) but counts
            # once toward this sprint's tally, not three times.
            {
                "key": "VIN-1",
                "sprint_name": "App Sprint 12",
                "sprint_start": "2026-08-03",
                "sprint_end": "2026-08-14",
                "carry_over_count": 3,
            },
            {
                "key": "VIN-2",
                "sprint_name": "App Sprint 12",
                "sprint_start": "2026-08-03",
                "sprint_end": "2026-08-14",
                "carry_over_count": 0,
            },
        ]
    )
    result = planning_metrics.carry_over_per_sprint(df)
    row = result.by_sprint[result.by_sprint["Sprint"] == "App Sprint 12"].iloc[0]
    assert row["Tickets"] == 2
    assert row["Carried over"] == 1  # not 3 - one ticket, once, is what "carried over" counts.
    assert row["Share %"] == pytest.approx(50.0)
    assert result.total_carried == 1
    assert result.total_tickets == 2


def test_carry_over_per_sprint_excludes_a_dateless_sprint_from_the_total():
    df = _board(
        [
            {
                "key": "VIN-1",
                "sprint_name": "App Sprint 12",
                "sprint_start": "2026-08-03",
                "sprint_end": "2026-08-14",
                "carry_over_count": 1,
            },
            {
                "key": "ML-1",
                "sprint_name": "ML Sprint 4",
                "sprint_start": None,
                "sprint_end": None,
                "carry_over_count": 2,
            },
        ]
    )
    result = planning_metrics.carry_over_per_sprint(df)
    assert result.excluded_sprints == ("ML Sprint 4",)
    # The ML sprint's real, non-zero carry-over is left out of the total - it
    # is not summed in as 2, and the total is not silently zeroed either.
    assert result.total_carried == 1
    assert result.total_tickets == 1
    assert "ML Sprint 4" in set(result.by_sprint["Sprint"])


def test_carry_over_per_sprint_total_is_none_not_zero_when_every_sprint_is_dateless():
    df = _board(
        [
            {
                "key": "ML-1",
                "sprint_name": "ML Sprint 4",
                "sprint_start": None,
                "sprint_end": None,
                "carry_over_count": 2,
            }
        ]
    )
    result = planning_metrics.carry_over_per_sprint(df)
    assert result.total_carried is None
    assert result.total_tickets is None
    assert result.excluded_sprints == ("ML Sprint 4",)


# ---------------------------------------------------------------------------
# planning_metrics.unestimated_per_sprint
# ---------------------------------------------------------------------------


def test_a_person_with_no_estimates_yields_the_unknown_sentinel_not_zero():
    df = _board(
        [
            {
                "key": "ML-1",
                "assignee": "Mehdi Ordikhani",
                "sprint_name": "ML Sprint 4",
                "sprint_start": None,
                "sprint_end": None,
                "original_estimate_sec": 0,
            },
            {
                "key": "ML-2",
                "assignee": "Mehdi Ordikhani",
                "sprint_name": "ML Sprint 4",
                "sprint_start": None,
                "sprint_end": None,
                "original_estimate_sec": None,
            },
        ]
    )
    table, excluded = planning_metrics.unestimated_per_sprint(df)
    row = table[table["Assignee"] == "Mehdi Ordikhani"].iloc[0]
    assert row["Tickets"] == 2
    assert row["Unestimated"] == 2
    assert row["Coverage"] == planning_metrics.UNKNOWN
    assert row["Coverage"] != 0
    assert row["Coverage"] != 0.0
    assert excluded == ("ML Sprint 4",)

    # No downstream coercion turns the sentinel back into a number.
    coerced = pd.to_numeric(table["Coverage"], errors="coerce")
    assert pd.isna(coerced.iloc[table.index[table["Assignee"] == "Mehdi Ordikhani"][0]])
    filled = coerced.fillna(0)
    # fillna(0) on the *coerced* copy would zero it - proving exactly why
    # unestimated_per_sprint itself never runs a column through to_numeric.
    assert filled.iloc[0] == 0  # demonstrates the trap this module avoids, not this module's output
    assert table["Coverage"].iloc[0] == planning_metrics.UNKNOWN  # the real output is untouched


def test_unestimated_per_sprint_reports_a_real_partial_coverage_number():
    df = _board(
        [
            {
                "key": "APP-1",
                "assignee": "Shawn",
                "sprint_name": "App Sprint 12",
                "sprint_start": "2026-08-03",
                "sprint_end": "2026-08-14",
                "original_estimate_sec": 3600 * 4,
            },
            {
                "key": "APP-2",
                "assignee": "Shawn",
                "sprint_name": "App Sprint 12",
                "sprint_start": "2026-08-03",
                "sprint_end": "2026-08-14",
                "original_estimate_sec": 0,
            },
        ]
    )
    table, excluded = planning_metrics.unestimated_per_sprint(df)
    row = table[table["Assignee"] == "Shawn"].iloc[0]
    assert row["Tickets"] == 2
    assert row["Unestimated"] == 1
    assert row["Coverage"] == pytest.approx(50.0)
    assert excluded == ()


def test_unestimated_per_sprint_empty_board_is_empty_but_valid():
    table, excluded = planning_metrics.unestimated_per_sprint(pd.DataFrame())
    assert table.empty
    assert list(table.columns) == planning_metrics.UNESTIMATED_COLUMNS
    assert excluded == ()
