"""Offline checks for the one-on-one workload tiles and the Sprint column.

The tiles answer a manager's questions before a one-on-one - how many hours
of estimated work someone holds, how much of it sits in the active sprint,
and how much is urgent - so what is worth testing is that the arithmetic
counts the right rows and that a ticket without a sprint is named honestly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as dashboard  # noqa: E402


def tickets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # 8h, active sprint, urgent - counts everywhere.
            {"estimate_hours": 8.0, "sprint_state": "active", "priority": "Highest"},
            # 4h, future sprint, normal - future load only.
            {"estimate_hours": 4.0, "sprint_state": "future", "priority": "Normal"},
            # 2h, no sprint, High - urgent but not in the sprint.
            {"estimate_hours": 2.0, "sprint_state": None, "priority": "High"},
            # No estimate at all - counted as unestimated, adds no hours.
            {"estimate_hours": 0.0, "sprint_state": "active", "priority": "Highest"},
        ]
    )


def test_the_hour_tiles_count_the_right_rows():
    load = dashboard._workload_hours(tickets())
    assert load["total"] == 14.0
    assert load["sprint"] == 8.0
    assert load["urgent"] == 10.0
    assert load["unestimated"] == 1


def test_a_text_only_estimate_is_not_called_unestimated():
    # estimate_policy marks a ticket estimated only in words ("2h" with no
    # numeric seconds) as having an estimate; the tile must say the same.
    board = dashboard.estimate_policy(
        pd.DataFrame(
            [
                {"original_estimate": "2h", "original_estimate_sec": 0, "status": "To Do"},
                {"original_estimate": None, "original_estimate_sec": 0, "status": "To Do"},
            ]
        ),
        {"backlog"},
    )
    load = dashboard._workload_hours(board)
    assert load["unestimated"] == 1


def test_an_unestimated_epic_is_not_called_missing_an_estimate():
    # Containers hold their children's work; the estimate policy exempts them,
    # so the No Estimate tile must not count them either.
    board = dashboard.estimate_policy(
        pd.DataFrame(
            [
                {"original_estimate_sec": 0, "issue_type": "Epic", "status": "To Do"},
                {"original_estimate_sec": 0, "issue_type": "Task", "status": "To Do"},
            ]
        ),
        {"backlog"},
    )
    load = dashboard._workload_hours(board)
    assert load["unestimated"] == 1


def test_an_empty_board_reports_zero_everywhere():
    load = dashboard._workload_hours(pd.DataFrame())
    assert load["total"] == 0.0
    assert load["sprint"] == 0.0
    assert load["urgent"] == 0.0
    assert load["unestimated"] == 0


def test_a_planned_ticket_names_its_sprint_and_state():
    row = pd.Series(
        {"sprint_name": "ML Sprint 42", "sprint_state": "Active", "status": "In Progress"}
    )
    assert dashboard._sprint_label(row) == "ML Sprint 42 (active)"


def test_a_backlog_ticket_is_called_backlog_not_no_sprint():
    row = pd.Series({"sprint_name": None, "sprint_state": None, "status": "Backlog"})
    assert dashboard._sprint_label(row) == "Backlog"


def test_an_unplanned_ticket_shows_its_status_instead():
    row = pd.Series({"sprint_name": None, "sprint_state": None, "status": "To Do"})
    assert dashboard._sprint_label(row) == "No sprint (To Do)"


def test_a_ticket_only_in_closed_sprints_is_not_called_planned():
    row = pd.Series(
        {"sprint_name": "ML Sprint 12", "sprint_state": "closed", "status": "To Do"}
    )
    assert dashboard._sprint_label(row) == "No sprint (To Do)"
