"""Offline checks for the People page's per-person join.

``people_table`` does not compute anything new - it wires together five
modules that already exist and were already tested in isolation
(``kpi``, ``estimate_accuracy``, ``pr_quality``, ``integrity``, ``roles``).
These tests are therefore aimed at the wiring itself: does every metric
carry a correct ``n``, does an unscored person come back ``None`` and never
``0``, does a former employee never surface as a row, and is
``estimate_accuracy`` - a module nothing on this dashboard imported before
this task - actually on the call path.

Fixtures build raw Jira changelog payloads and run them through
``integrity.changelog_events``, the same pattern ``tests/test_integrity.py``
uses, so the "events" frame this module reads is exactly the shape the real
pipeline produces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import estimate_accuracy  # noqa: E402
import integrity  # noqa: E402
import people_table  # noqa: E402
import roles  # noqa: E402

NOW = pd.Timestamp("2026-08-16T12:00:00Z")


# --------------------------------------------------------------------------
# Changelog fixture helpers, matching tests/test_integrity.py exactly so the
# "events" frame built here is the real integrity.changelog_events shape.
# --------------------------------------------------------------------------


def when(days_ago: float, hour: int = 9, minute: int = 0) -> str:
    stamp = (NOW - pd.Timedelta(days=days_ago)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.000+0000")


def item(field: str, from_string: object, to_string: object) -> dict:
    return {"field": field, "fieldId": field, "fieldtype": "jira", "fromString": from_string, "toString": to_string}


def status(from_status: str, to_status: str) -> dict:
    return item("status", from_status, to_status)


def history(created: str, author: str, *items: dict, entry_id: str | None = None) -> dict:
    return {
        "id": entry_id or f"{author}-{created}-{items[0]['field'] if items else ''}",
        "created": created,
        "author": {"displayName": author, "accountId": author.lower()},
        "items": list(items),
    }


def issue(key: str, *histories: dict) -> dict:
    return {"key": key, "changelog": {"histories": list(histories)}}


def events_of(*issues: dict) -> pd.DataFrame:
    return integrity.changelog_events(list(issues))


def _empty() -> pd.DataFrame:
    return pd.DataFrame()


def small_roster() -> roles.Roster:
    """A four-person roster, independent of the real deployed default, so
    these tests do not silently break when someone edits ``roles_template.env``.
    """
    env = {
        "JIRA_ROLES": "backend=Alice,Carol;seo=Sam;exec=Eve",
        "GITHUB_LOGIN_MAP": "Alice=alice-gh;Carol=carol-gh",
        "JIRA_FORMER_STAFF": "Frank",
        "GITHUB_UNMAPPED_AUTHORS": "",
    }
    return roles.load_roster(env)


# --------------------------------------------------------------------------
# Totality and schema
# --------------------------------------------------------------------------


def test_the_function_is_total_empty_input_returns_the_full_empty_schema():
    out = people_table.people_table(
        _empty(), _empty(), _empty(), _empty(), _empty(), roster=small_roster(), now=NOW
    )
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == people_table.COLUMNS
    assert out.empty


def test_every_metric_column_has_a_matching_n_column_in_the_schema():
    metric_columns = [
        "delivered_points",
        "trivial_share",
        "cycle_median",
        "reviews_given",
        "ttfr_hours",
        "estimate_ratio",
        "estimate_iqr",
        "flag_count",
        "flag_severity",
    ]
    for col in metric_columns:
        assert f"n_{col}" in people_table.COLUMNS, f"missing n_{col}"
    # score's own sample size is spelled "n", not "n_score".
    assert "n" in people_table.COLUMNS


# --------------------------------------------------------------------------
# Scoring: never a bare zero, always a coverage percentage
# --------------------------------------------------------------------------


def test_a_person_under_the_measurable_weight_floor_is_none_not_zero():
    # Alice: open tickets only, no resolved history, no PRs, no gradable
    # tickets, no urgent tickets, and alone in her role (no peer cohort) -
    # Delivery + Weekly updates + Staleness + Carry-over + Estimates is 55 of
    # kpi.WEIGHTS' 100 points, under kpi.MIN_SCORABLE_WEIGHT (60).
    open_tickets = pd.DataFrame(
        [
            {
                "assignee": "Alice",
                "idle_days": 1.0,
                "carry_over_count": 0,
                "has_estimate": True,
                "issue_type": "Task",
                "priority": "Low",
            }
        ]
    )
    ev = events_of(issue("VV-1", history(when(2), "Alice", status("To Do", "In Progress"))))
    out = people_table.people_table(
        open_tickets, _empty(), _empty(), _empty(), ev, roster=small_roster(), now=NOW
    )
    row = out[out["person"] == "Alice"].iloc[0]
    assert pd.isna(row["score"])
    assert row["no_score_reason"] != ""
    assert "55" in row["no_score_reason"] and "60" in row["no_score_reason"]
    assert not pd.isna(row["measurable_pct"])
    assert row["measurable_pct"] == pytest.approx(55.0)


def test_measurable_pct_is_never_null_even_when_unscored():
    open_tickets = pd.DataFrame(
        [
            {"assignee": "Eve", "idle_days": 1.0, "carry_over_count": 0, "has_estimate": True, "issue_type": "Task", "priority": "Low"},
            {"assignee": "Sam", "idle_days": 1.0, "carry_over_count": 0, "has_estimate": True, "issue_type": "Task", "priority": "Low"},
            {"assignee": "Ghost", "idle_days": 1.0, "carry_over_count": 0, "has_estimate": True, "issue_type": "Task", "priority": "Low"},
        ]
    )
    out = people_table.people_table(
        open_tickets, _empty(), _empty(), _empty(), _empty(), roster=small_roster(), now=NOW
    )
    assert out["measurable_pct"].notna().all()
    assert len(out) == 3


def test_an_exec_and_a_no_rubric_person_both_appear_with_null_score_and_the_right_reason():
    open_tickets = pd.DataFrame(
        [
            {"assignee": "Eve", "idle_days": 1.0, "carry_over_count": 0, "has_estimate": True, "issue_type": "Task", "priority": "Low"},
            {"assignee": "Sam", "idle_days": 1.0, "carry_over_count": 0, "has_estimate": True, "issue_type": "Task", "priority": "Low"},
        ]
    )
    out = people_table.people_table(
        open_tickets, _empty(), _empty(), _empty(), _empty(), roster=small_roster(), now=NOW
    )
    by_person = out.set_index("person")

    exec_row = by_person.loc["Eve"]
    assert pd.isna(exec_row["score"])
    assert exec_row["no_score_reason"] == roles.EXEC_REASON
    assert exec_row["role"] == "exec"

    seo_row = by_person.loc["Sam"]
    assert pd.isna(seo_row["score"])
    assert "no rubric defined for role 'seo'" in seo_row["no_score_reason"]
    assert seo_row["role"] == "seo"


def test_a_role_unknown_person_still_gets_a_row_never_scored_wrongly():
    open_tickets = pd.DataFrame(
        [{"assignee": "Ghost", "idle_days": 1.0, "carry_over_count": 0, "has_estimate": True, "issue_type": "Task", "priority": "Low"}]
    )
    out = people_table.people_table(
        open_tickets, _empty(), _empty(), _empty(), _empty(), roster=small_roster(), now=NOW
    )
    row = out[out["person"] == "Ghost"].iloc[0]
    assert row["role"] is None
    assert pd.isna(row["score"])
    assert row["no_score_reason"] == "role unknown"


# --------------------------------------------------------------------------
# Former staff: never a row, no matter which source surfaced the name
# --------------------------------------------------------------------------


def test_former_staff_never_appear_as_rows():
    resolved_tickets = pd.DataFrame(
        [
            {
                "key": "VV-9",
                "assignee": "Frank",
                "resolution": "Done",
                "status_category": "Done",
                "original_estimate_sec": 3600,
                "time_spent_sec": 3600,
            }
        ]
    )
    prs = pd.DataFrame(
        [{"author": "Frank", "number": 1, "url": "https://x/1", "changed_lines": 5, "state": "MERGED", "merged_at": "2026-08-01T00:00:00Z"}]
    )
    out = people_table.people_table(
        _empty(), resolved_tickets, _empty(), prs, _empty(), roster=small_roster(), now=NOW
    )
    assert "Frank" not in set(out["person"])


# --------------------------------------------------------------------------
# n_* correctness, per metric - hand-computed expected values
# --------------------------------------------------------------------------


def test_estimate_accuracy_is_actually_called_and_its_numbers_land_on_the_row(monkeypatch):
    calls = {"n": 0}
    real = estimate_accuracy.accuracy_by_person

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(estimate_accuracy, "accuracy_by_person", spy)

    resolved_tickets = pd.DataFrame(
        [
            {
                "key": "VV-10",
                "assignee": "Carol",
                "resolution": "Done",
                "status_category": "Done",
                "original_estimate_sec": 10 * 3600,
                "time_spent_sec": 5 * 3600,  # ratio 0.5
            },
            {
                "key": "VV-11",
                "assignee": "Carol",
                "resolution": "Done",
                "status_category": "Done",
                "original_estimate_sec": 4 * 3600,
                "time_spent_sec": 6 * 3600,  # ratio 1.5
            },
        ]
    )
    out = people_table.people_table(
        _empty(), resolved_tickets, _empty(), _empty(), _empty(), roster=small_roster(), now=NOW
    )
    assert calls["n"] >= 1

    row = out[out["person"] == "Carol"].iloc[0]
    assert row["estimate_ratio"] == pytest.approx(1.0)
    assert row["estimate_iqr"] == pytest.approx(0.5)
    assert row["n_estimate_ratio"] == 2
    assert row["n_estimate_iqr"] == 2


def test_delivered_points_and_trivial_share_carry_the_right_n():
    prs = pd.DataFrame(
        [
            {
                "author": "alice-gh",
                "number": 1,
                "url": "https://x/1",
                "changed_lines": 5,
                "state": "MERGED",
                "merged_at": "2026-08-01T00:00:00Z",
            },
            {
                "author": "alice-gh",
                "number": 2,
                "url": "https://x/2",
                "changed_lines": 8,
                "state": "MERGED",
                "merged_at": "2026-08-02T00:00:00Z",
            },
        ]
    )
    out = people_table.people_table(
        _empty(), _empty(), _empty(), prs, _empty(), roster=small_roster(), now=NOW
    )
    row = out[out["person"] == "Alice"].iloc[0]
    # Both PRs are trivial (< 10 changed lines): 2 * SIZE_POINTS["trivial"] (1) = 2.0
    assert row["delivered_points"] == pytest.approx(2.0)
    assert row["n_delivered_points"] == 2
    assert row["trivial_share"] == pytest.approx(1.0)
    assert row["n_trivial_share"] == 2


def test_a_pr_authored_by_an_unmapped_login_still_gets_its_own_row():
    prs = pd.DataFrame(
        [
            {
                "author": "some-random-login",
                "number": 5,
                "url": "https://x/5",
                "changed_lines": 3,
                "state": "MERGED",
                "merged_at": "2026-08-01T00:00:00Z",
            }
        ]
    )
    out = people_table.people_table(
        _empty(), _empty(), _empty(), prs, _empty(), roster=small_roster(), now=NOW
    )
    row = out[out["person"] == "some-random-login"].iloc[0]
    assert row["role"] is None
    assert row["no_score_reason"] == "role unknown"
    assert row["delivered_points"] == pytest.approx(1.0)


def test_flag_count_and_flag_severity_share_a_correct_n():
    # The mid-flight estimate raiser: two raises after work started, 16h
    # added - trips estimate_inflation (weight 2.0) and nothing else.
    ev = events_of(
        issue(
            "VV-7",
            history(when(30), "Carol", item("timeoriginalestimate", None, "4h")),
            history(when(20), "Carol", status("To Do", "In Progress")),
            history(when(10), "Carol", item("timeoriginalestimate", "4h", "12h")),
            history(when(5), "Carol", status("In Progress", "Code Review")),
            history(when(4), "Carol", item("timeoriginalestimate", "12h", "20h")),
        )
    )
    out = people_table.people_table(
        _empty(), _empty(), _empty(), _empty(), ev, roster=small_roster(), now=NOW
    )
    row = out[out["person"] == "Carol"].iloc[0]
    assert row["flag_count"] == 1
    assert row["flag_severity"] == pytest.approx(2.0)
    assert row["n_flag_count"] == row["n_flag_severity"]
    # n is the underlying signal-event count (status transitions here), not
    # the flag count itself.
    assert row["n_flag_count"] > row["flag_count"]


def test_a_quiet_person_trips_no_flags_a_real_zero_not_missing_data():
    # Alice never touches the changelog at all - only Carol does, so the
    # events frame is non-empty (the pipeline ran) but has nothing to say
    # about Alice specifically. That is a known zero, not missing data.
    open_tickets = pd.DataFrame(
        [{"assignee": "Alice", "idle_days": 1.0, "carry_over_count": 0, "has_estimate": True, "issue_type": "Task", "priority": "Low"}]
    )
    ev = events_of(issue("VV-1", history(when(2), "Carol", status("To Do", "In Progress"))))
    out = people_table.people_table(
        open_tickets, _empty(), _empty(), _empty(), ev, roster=small_roster(), now=NOW
    )
    row = out[out["person"] == "Alice"].iloc[0]
    assert row["flag_count"] == 0
    assert not pd.isna(row["flag_count"])


def test_cycle_median_and_its_n_come_from_a_finished_ticket():
    ev = events_of(
        issue(
            "VV-20",
            history(when(10), "Carol", status("To Do", "In Progress")),
            history(when(4), "Carol", status("In Progress", "Resolved")),
        )
    )
    out = people_table.people_table(
        _empty(), _empty(), _empty(), _empty(), ev, roster=small_roster(), now=NOW
    )
    row = out[out["person"] == "Carol"].iloc[0]
    assert row["n_cycle_median"] == 1
    assert row["cycle_median"] == pytest.approx(6.0, abs=0.01)


# --------------------------------------------------------------------------
# Unmeasured stays unmeasured: NA, not an invented zero
# --------------------------------------------------------------------------


def test_a_person_with_no_pr_data_at_all_is_na_not_zero_on_pr_columns():
    open_tickets = pd.DataFrame(
        [{"assignee": "Alice", "idle_days": 1.0, "carry_over_count": 0, "has_estimate": True, "issue_type": "Task", "priority": "Low"}]
    )
    out = people_table.people_table(
        open_tickets, _empty(), _empty(), _empty(), _empty(), roster=small_roster(), now=NOW
    )
    row = out[out["person"] == "Alice"].iloc[0]
    assert pd.isna(row["delivered_points"])
    assert pd.isna(row["trivial_share"])
    assert pd.isna(row["reviews_given"])
    assert pd.isna(row["ttfr_hours"])
