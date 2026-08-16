"""Offline checks for the estimate-accuracy arithmetic.

The whole module compares numbers people supplied about themselves, so the tests
are mostly about restraint: no ratio from an unfinished ticket, no verdict from
four tickets, no outlier flag when the team is too small to have a shape, and no
zeros invented where Jira or GitHub simply did not answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import estimate_accuracy  # noqa: E402

HOUR = 3600


def ticket(
    key: str,
    assignee: str,
    estimate_h: float | None,
    logged_h: float | None,
    *,
    done: bool = True,
) -> dict:
    """One Jira row shaped like ``jira_client._issues_to_dataframe`` builds them."""
    return {
        "key": key,
        "assignee": assignee,
        "summary": f"{key} something",
        "status": "Done" if done else "In Progress",
        "status_category": "Done" if done else "In Progress",
        "resolution": "Done" if done else None,
        "original_estimate_sec": int((estimate_h or 0) * HOUR),
        "time_spent_sec": int((logged_h or 0) * HOUR),
    }


def padder(n: int = 6) -> pd.DataFrame:
    """Six finished tickets, each logging 40% of a comfortable estimate."""
    return pd.DataFrame(
        [ticket(f"MB-{i}", "Pat Padder", 10, 4) for i in range(n)]
        + [ticket(f"MB-9{i}", "Sam Straight", 10, 9.5) for i in range(n)]
    )


# --------------------------------------------------------------------------- #
# The ratio itself
# --------------------------------------------------------------------------- #


def test_the_ratio_is_logged_over_estimated_per_ticket():
    frame = pd.DataFrame([ticket("MB-1", "Pat", 8, 12), ticket("MB-2", "Pat", 8, 2)])
    out = estimate_accuracy.accuracy_ratio(frame).set_index("key")
    assert out.loc["MB-1", "ratio"] == 1.5
    assert bool(out.loc["MB-1", "over_ran"]) is True
    assert out.loc["MB-2", "ratio"] == 0.25
    assert bool(out.loc["MB-2", "under_ran"]) is True


def test_an_unfinished_ticket_has_no_accuracy_to_report():
    # Logged 1h of an 8h estimate because it started this morning - counting it
    # would mark every fresh ticket as a padded estimate.
    frame = pd.DataFrame([ticket("MB-1", "Pat", 8, 1, done=False)])
    assert estimate_accuracy.accuracy_ratio(frame).empty
    assert not estimate_accuracy.accuracy_ratio(frame, completed_only=False).empty


def test_tickets_missing_either_number_are_left_out_rather_than_scored_zero():
    frame = pd.DataFrame(
        [
            ticket("MB-1", "Pat", None, 6),  # no estimate: a coverage problem, not accuracy
            ticket("MB-2", "Pat", 6, None),  # nothing logged: says nothing either
            ticket("MB-3", "Pat", 6, 6),
        ]
    )
    out = estimate_accuracy.accuracy_ratio(frame)
    assert list(out["key"]) == ["MB-3"]


def test_the_distribution_shows_spread_and_not_just_the_median():
    # Two people with the same median and completely different reliability.
    steady = [ticket(f"S-{i}", "Steady", 10, h) for i, h in enumerate([9, 10, 10, 11, 10])]
    wild = [ticket(f"W-{i}", "Wild", 10, h) for i, h in enumerate([2, 3, 10, 25, 30])]
    out = estimate_accuracy.accuracy_by_person(pd.DataFrame(steady + wild)).set_index("assignee")
    assert out.loc["Steady", "median_ratio"] == out.loc["Wild", "median_ratio"] == 1.0
    assert out.loc["Steady", "iqr"] < 0.3
    assert out.loc["Wild", "iqr"] > 2.0
    assert out.loc["Wild", "tickets"] == 5
    assert bool(out.loc["Wild", "enough_data"]) is True


# --------------------------------------------------------------------------- #
# Padding
# --------------------------------------------------------------------------- #


def test_the_padder_is_separated_from_someone_who_estimates_well():
    out = estimate_accuracy.padding_index(padder()).set_index("assignee")
    assert out.loc["Pat Padder", "median_ratio"] == 0.4
    assert out.loc["Pat Padder", "under_run_share"] == 1.0
    assert out.loc["Pat Padder", "verdict"] == "estimates consistently generous"
    assert out.loc["Sam Straight", "under_run_share"] == 0.0
    assert out.loc["Sam Straight", "verdict"] == "estimates broadly hold"


def test_consistent_underestimation_reads_as_its_own_finding():
    frame = pd.DataFrame([ticket(f"MB-{i}", "Opti Mist", 4, 10) for i in range(5)])
    out = estimate_accuracy.padding_index(frame).set_index("assignee")
    assert out.loc["Opti Mist", "median_ratio"] == 2.5
    assert out.loc["Opti Mist", "verdict"] == "estimates consistently low"


def test_no_verdict_below_the_minimum_sample():
    frame = pd.DataFrame([ticket(f"MB-{i}", "New Hire", 10, 3) for i in range(3)])
    out = estimate_accuracy.padding_index(frame).set_index("assignee")
    assert out.loc["New Hire", "tickets"] == 3
    assert bool(out.loc["New Hire", "enough_data"]) is False
    assert out.loc["New Hire", "verdict"] == ""
    # The same ratios, once there are enough of them, do get a verdict.
    enough = pd.DataFrame([ticket(f"MB-{i}", "New Hire", 10, 3) for i in range(5)])
    assert estimate_accuracy.padding_index(enough).loc[0, "verdict"] != ""
    # And a caller can move the bar.
    assert estimate_accuracy.padding_index(frame, min_tickets=2).loc[0, "verdict"] != ""


def test_a_board_with_no_estimates_at_all_produces_an_empty_frame():
    frame = pd.DataFrame([ticket("MB-1", "Pat", None, 6), ticket("MB-2", "Pat", None, 3)])
    assert estimate_accuracy.padding_index(frame).empty
    assert estimate_accuracy.accuracy_by_person(frame).empty
    assert estimate_accuracy.accuracy_ratio(pd.DataFrame()).empty


def test_missing_time_tracking_columns_do_not_raise():
    # An older or partial Jira fetch that never carried the time-tracking fields.
    frame = pd.DataFrame([{"key": "MB-1", "assignee": "Pat", "status_category": "Done"}])
    assert estimate_accuracy.accuracy_ratio(frame).empty
    assert estimate_accuracy.padding_index(frame).empty


# --------------------------------------------------------------------------- #
# Hours against delivered change
# --------------------------------------------------------------------------- #


def prs_for(pairs: list[tuple[str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "number": i,
                "title": f"{key} work",
                "branch": "",
                "body": "",
                "author": "someone",
                "changed_lines": lines,
                "state": "MERGED",
            }
            for i, (key, lines) in enumerate(pairs)
        ]
    )


def test_hours_are_joined_to_delivered_lines_through_the_jira_key():
    tickets = pd.DataFrame(
        [ticket("MB-1", "Pat", 8, 8), ticket("MB-2", "Sam", 4, 2)]
    )
    prs = prs_for([("MB-1", 100), ("MB-1", 100), ("MB-2", 50)])
    out = estimate_accuracy.hours_per_delivered_line(
        tickets, prs, project_keys=["MB"]
    ).set_index("key")
    assert out.loc["MB-1", "prs"] == 2
    assert out.loc["MB-1", "changed_lines"] == 200
    assert out.loc["MB-1", "hours_per_100_lines"] == 4.0
    assert out.loc["MB-2", "hours_per_100_lines"] == 4.0


def test_the_outlier_is_judged_against_the_team_not_a_fixed_threshold():
    rows = [ticket(f"MB-{i}", "Peer", 4, 4) for i in range(8)]
    rows.append(ticket("MB-99", "Slow Hand", 40, 40))
    prs = prs_for([(f"MB-{i}", 400) for i in range(8)] + [("MB-99", 20)])
    out = estimate_accuracy.hours_per_delivered_line(
        pd.DataFrame(rows), prs, project_keys=["MB"]
    ).set_index("key")
    assert bool(out.loc["MB-99", "is_outlier"]) is True
    assert not out.drop(index="MB-99")["is_outlier"].any()
    assert out.loc["MB-99", "modified_z"] > estimate_accuracy.OUTLIER_Z


def test_a_team_too_small_or_too_uniform_to_have_a_shape_flags_nobody():
    tickets = pd.DataFrame([ticket("MB-1", "Pat", 8, 8), ticket("MB-2", "Sam", 8, 8)])
    prs = prs_for([("MB-1", 100), ("MB-2", 900)])
    out = estimate_accuracy.hours_per_delivered_line(tickets, prs, project_keys=["MB"])
    assert not out["is_outlier"].any()
    assert out["modified_z"].isna().all()

    identical = pd.DataFrame([ticket(f"MB-{i}", "Pat", 4, 4) for i in range(6)])
    same = prs_for([(f"MB-{i}", 100) for i in range(6)])
    out = estimate_accuracy.hours_per_delivered_line(identical, same, project_keys=["MB"])
    assert not out["is_outlier"].any()  # a zero deviation is not a divide-by-zero


def test_a_throttled_pr_fetch_yields_no_rate_rather_than_a_wrong_one():
    tickets = pd.DataFrame([ticket("MB-1", "Pat", 8, 8)])
    lean = prs_for([("MB-1", 100)]).drop(columns=["changed_lines"])
    assert estimate_accuracy.hours_per_delivered_line(tickets, lean, project_keys=["MB"]).empty

    partial = prs_for([("MB-1", None)])
    assert estimate_accuracy.hours_per_delivered_line(tickets, partial, project_keys=["MB"]).empty


def test_a_ticket_whose_prs_changed_nothing_has_no_rate_instead_of_an_infinite_one():
    tickets = pd.DataFrame([ticket("MB-1", "Pat", 8, 8)])
    prs = prs_for([("MB-1", 0)])
    out = estimate_accuracy.hours_per_delivered_line(tickets, prs, project_keys=["MB"])
    assert len(out) == 1
    assert pd.isna(out.loc[0, "hours_per_100_lines"])
    assert not bool(out.loc[0, "is_outlier"])


def test_tickets_and_prs_that_never_meet_produce_an_empty_join():
    tickets = pd.DataFrame([ticket("MB-1", "Pat", 8, 8)])
    assert estimate_accuracy.hours_per_delivered_line(
        tickets, prs_for([("XX-9", 100)]), project_keys=["MB"]
    ).empty
    assert estimate_accuracy.hours_per_delivered_line(tickets, pd.DataFrame()).empty
    assert estimate_accuracy.hours_per_delivered_line(pd.DataFrame(), prs_for([("MB-1", 1)])).empty
