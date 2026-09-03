"""Offline checks for role_kpis: the non-code rubrics' component computations.

``docs/assumptions/2C.md`` recorded that the QA/PM/designer/infrastructure
rubrics existed with "no component-computation function behind [them] yet".
``role_kpis`` is that computation; these tests aim at its own claims:

- every rubric component always comes back, measured or as a named
  ``sufficient=False`` gap - never silently dropped, never a zero;
- below MIN_N observations a component refuses to score;
- staleness and the stale queue read ``status_age_days`` (the clock cosmetic
  edits cannot reset), not ``idle_days``;
- triage latency counts an untouched ticket at its current age, so ignoring
  the queue can never outscore triaging it late;
- ``score_from_parts`` withholds a headline below 60% of rubric weight.

Fixtures build raw Jira changelog payloads through
``integrity.changelog_events``, the same pattern ``tests/test_integrity.py``
and ``tests/test_people_table.py`` use, so the events frame is exactly the
shape the real pipeline produces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import integrity  # noqa: E402
import kpi  # noqa: E402
import role_kpis  # noqa: E402
import roles  # noqa: E402

# Anchored to the running clock: see the note in tests/test_integrity.py. The
# staleness thresholds these fixtures straddle are measured against the real
# now, so a fixed anchor drifts every fixture across them given enough days.
NOW = pd.Timestamp.now(tz="UTC")


def when(days_ago: float, hour: int = 9, minute: int = 0) -> str:
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
        "id": entry_id or f"{author}-{created}-{items[0]['field'] if items else ''}",
        "created": created,
        "author": {"displayName": author, "accountId": author.lower()},
        "items": list(items),
    }


def issue(key: str, *histories: dict) -> dict:
    return {"key": key, "changelog": {"histories": list(histories)}}


def events_of(*issues: dict) -> pd.DataFrame:
    return integrity.changelog_events(list(issues))


def ticket_row(key: str, **overrides) -> dict:
    row = {
        "key": key,
        "assignee": "Dina QA",
        "status": "Done",
        "status_category": "Done",
        "resolution": "Done",
        "created": when(80.0),
        "priority": "Medium",
        "original_estimate_sec": 8 * 3600,
        "time_spent_sec": 8 * 3600,
        "issue_type": "Task",
    }
    row.update(overrides)
    return row


def build(open_tickets=None, all_tickets=None, events=None, prs=None, resolved=None):
    return role_kpis.build_inputs(
        open_tickets if open_tickets is not None else pd.DataFrame(),
        all_tickets if all_tickets is not None else pd.DataFrame(),
        events if events is not None else integrity.empty_events(),
        prs if prs is not None else pd.DataFrame(),
        resolved if resolved is not None else pd.DataFrame(),
        now=NOW,
    )


# --------------------------------------------------------------------------
# Shape guarantees
# --------------------------------------------------------------------------


def test_every_rubric_component_always_comes_back_measured_or_as_a_named_gap():
    inputs = build()
    for role in ("qa-automated", "qa-manual", "pm", "designer", "infrastructure"):
        rubric = roles.ROLE_RUBRIC[role]
        parts = role_kpis.components_for(inputs, "Nobody", role, pd.DataFrame(), pd.DataFrame())
        assert parts is not None
        assert [p.name for p in parts] == [c.name for c in rubric.components]
        for part in parts:
            assert not part.sufficient
            assert "insufficient data" in part.detail
            assert "needs" in part.detail  # the gap names its missing input


def test_code_and_sentinel_roles_are_not_this_modules_business():
    inputs = build()
    assert role_kpis.components_for(inputs, "Tam", "platform", pd.DataFrame(), pd.DataFrame()) is None
    assert role_kpis.components_for(inputs, "zoe", "advisor", pd.DataFrame(), pd.DataFrame()) is None
    assert role_kpis.components_for(inputs, "Angel Vossough", "exec", pd.DataFrame(), pd.DataFrame()) is None
    assert role_kpis.components_for(inputs, "X", None, pd.DataFrame(), pd.DataFrame()) is None


def test_score_from_parts_withholds_below_sixty_percent_of_weight():
    rubric = roles.QA_MANUAL_RUBRIC
    # Only "Verification cycle time" (20) measured: 20 < 60 -> no headline.
    parts = [
        kpi.Component("Verification cycle time", 80.0, "x", n=5),
    ] + [
        kpi.Component(c.name, 0.0, "insufficient data - needs x", n=0, sufficient=False)
        for c in rubric.components
        if c.name != "Verification cycle time"
    ]
    score, covered, note = role_kpis.score_from_parts(parts, rubric)
    assert score is None
    assert covered == 20.0
    assert "Not scored" in note


def test_score_from_parts_scores_a_weighted_mean_over_measured_weight():
    rubric = roles.QA_MANUAL_RUBRIC  # escape 35, validity 25, verify 20, rework 10, estimates 10
    parts = [
        kpi.Component("Defect escape rate", 100.0, "x", n=5),
        kpi.Component("Verification cycle time", 50.0, "x", n=5),
        kpi.Component("Rework after verification", 0.0, "x", n=5),
        kpi.Component("Bug validity rate", 0.0, "gap", n=0, sufficient=False),
        kpi.Component("Estimate accuracy", 0.0, "gap", n=0, sufficient=False),
    ]
    score, covered, note = role_kpis.score_from_parts(parts, rubric)
    assert covered == 65.0
    assert score == pytest.approx((35 * 100 + 20 * 50 + 10 * 0) / 65.0)
    assert "Bug validity rate" in note  # missing components are named, not passed


# --------------------------------------------------------------------------
# The accuracy ramp
# --------------------------------------------------------------------------


def test_accuracy_score_is_symmetric_in_log_space():
    assert role_kpis.accuracy_score(1.0) == 100.0
    assert role_kpis.accuracy_score(2.0) == 0.0
    assert role_kpis.accuracy_score(0.5) == 0.0
    assert role_kpis.accuracy_score(2.0 ** 0.5) == pytest.approx(50.0)
    assert role_kpis.accuracy_score(0.5 ** 0.5) == pytest.approx(50.0)
    assert role_kpis.accuracy_score(0.0) is None
    assert role_kpis.accuracy_score(-1.0) is None


# --------------------------------------------------------------------------
# QA: verification-anchored components
# --------------------------------------------------------------------------


def _qa_events() -> pd.DataFrame:
    """Dina verifies four tickets; QA-1 is reopened later; QA-2 is resolved twice."""
    return events_of(
        issue(
            "QA-1",
            history(when(30.0), "Shawn", status("To Do", "In Progress")),
            history(when(28.0), "Shawn", status("In Progress", "In Review")),
            history(when(27.0), "Dina QA", status("In Review", "Done")),
            history(when(20.0), "Shawn", status("Done", "In Progress")),  # reopened
        ),
        issue(
            "QA-2",
            history(when(25.0), "Shawn", status("To Do", "In Progress")),
            history(when(24.0), "Dina QA", status("In Progress", "Done")),
            history(when(23.0), "Shawn", status("Done", "In Progress")),
            history(when(22.0), "Dina QA", status("In Progress", "Done")),  # re-resolved
        ),
        issue(
            "QA-3",
            history(when(15.0), "Shawn", status("To Do", "In Review")),
            history(when(13.0), "Dina QA", status("In Review", "Done")),
        ),
        issue(
            "QA-4",
            history(when(10.0), "Shawn", status("To Do", "In Review")),
            history(when(9.0), "Dina QA", status("In Review", "Done")),
        ),
    )


def test_defect_escape_counts_reopens_on_tickets_this_person_verified():
    inputs = build(events=_qa_events())
    parts = role_kpis.components_for(
        inputs, "Dina QA", "qa-manual", pd.DataFrame(), pd.DataFrame()
    )
    escape = next(p for p in parts if p.name == "Defect escape rate")
    assert escape.sufficient
    assert escape.n == 4
    # QA-1 and QA-2 both saw a reopen -> 2 of 4 escaped -> 50.
    assert escape.score == pytest.approx(50.0)
    assert "QA-1" in escape.detail or "2 of 4" in escape.detail


def test_rework_after_verification_counts_double_resolutions():
    inputs = build(events=_qa_events())
    parts = role_kpis.components_for(
        inputs, "Dina QA", "qa-manual", pd.DataFrame(), pd.DataFrame()
    )
    rework = next(p for p in parts if p.name == "Rework after verification")
    assert rework.sufficient
    # Only QA-2 was declared done twice -> 1 of 4 -> 75.
    assert rework.score == pytest.approx(75.0)


def test_verification_cycle_scores_the_gap_before_the_resolving_move():
    inputs = build(events=_qa_events())
    parts = role_kpis.components_for(
        inputs, "Dina QA", "qa-manual", pd.DataFrame(), pd.DataFrame()
    )
    verify = next(p for p in parts if p.name == "Verification cycle time")
    assert verify.sufficient
    # Gaps: QA-1 1d, QA-2 1d then 1d, QA-3 2d, QA-4 1d -> median 1d -> 100.
    assert verify.score == pytest.approx(100.0)
    assert verify.n == 5


def test_below_min_n_verifications_the_qa_components_refuse_to_score():
    events = events_of(
        issue(
            "QA-9",
            history(when(5.0), "Shawn", status("To Do", "In Review")),
            history(when(4.0), "Dina QA", status("In Review", "Done")),
        ),
    )
    inputs = build(events=events)
    parts = role_kpis.components_for(
        inputs, "Dina QA", "qa-manual", pd.DataFrame(), pd.DataFrame()
    )
    escape = next(p for p in parts if p.name == "Defect escape rate")
    assert not escape.sufficient  # one verification is an anecdote, not a rate


def test_bug_validity_rate_stays_an_honest_gap():
    inputs = build(events=_qa_events())
    parts = role_kpis.components_for(
        inputs, "Dina QA", "qa-manual", pd.DataFrame(), pd.DataFrame()
    )
    validity = next(p for p in parts if p.name == "Bug validity rate")
    assert not validity.sufficient
    assert "bug-resolution outcomes" in validity.detail


# --------------------------------------------------------------------------
# Estimate accuracy feeds QA, designer and infrastructure alike
# --------------------------------------------------------------------------


def test_estimate_accuracy_component_scores_the_median_ratio():
    tickets = pd.DataFrame(
        [
            ticket_row("QA-1"),
            ticket_row("QA-2"),
            ticket_row("QA-3"),
            ticket_row("QA-4", time_spent_sec=16 * 3600),  # one 2x over-run
        ]
    )
    inputs = build(all_tickets=tickets)
    parts = role_kpis.components_for(
        inputs, "Dina QA", "qa-manual", pd.DataFrame(), pd.DataFrame()
    )
    acc = next(p for p in parts if p.name == "Estimate accuracy")
    assert acc.sufficient
    assert acc.n == 4
    assert acc.score == pytest.approx(100.0)  # median ratio is 1.0


def test_estimate_accuracy_needs_min_n_finished_tickets():
    tickets = pd.DataFrame([ticket_row("QA-1"), ticket_row("QA-2")])
    inputs = build(all_tickets=tickets)
    parts = role_kpis.components_for(
        inputs, "Dina QA", "qa-manual", pd.DataFrame(), pd.DataFrame()
    )
    acc = next(p for p in parts if p.name == "Estimate accuracy")
    assert not acc.sufficient


# --------------------------------------------------------------------------
# PM components
# --------------------------------------------------------------------------


def _pm_frames():
    """Three fresh tickets; Mihai triages two fast, ignores the third."""
    events = events_of(
        issue("PM-1", history(when(9.5), "Mihai Manea", item("priority", None, "High"))),
        issue("PM-2", history(when(7.0), "Mihai Manea", item("assignee", None, "Shawn"))),
        issue("PM-3"),  # created, never touched
    )
    tickets = pd.DataFrame(
        [
            ticket_row("PM-1", assignee="Shawn", created=when(10.0), status="To Do",
                       status_category="To Do", resolution=None),
            ticket_row("PM-2", assignee="Shawn", created=when(8.0), status="To Do",
                       status_category="To Do", resolution=None),
            ticket_row("PM-3", assignee="Unassigned", created=when(30.0), status="To Do",
                       status_category="To Do", resolution=None),
        ]
    )
    return tickets, events


def test_triage_latency_counts_an_untouched_ticket_at_its_current_age():
    tickets, events = _pm_frames()
    inputs = build(all_tickets=tickets, events=events)
    parts = role_kpis.components_for(inputs, "Mihai Manea", "pm", pd.DataFrame(), pd.DataFrame())
    triage = next(p for p in parts if p.name == "Triage latency")
    assert triage.sufficient
    assert triage.n == 3
    # Latencies: 0.5d, 1d, and PM-3 untouched at 30d -> median 1d -> 100.
    assert triage.score == pytest.approx(100.0)
    assert "untouched" in triage.detail


def test_ignoring_the_whole_queue_scores_zero_not_insufficient():
    tickets, _ = _pm_frames()
    events = events_of(issue("PM-9", history(when(1.0), "Somebody Else", status("To Do", "Done"))))
    inputs = build(all_tickets=tickets, events=events)
    parts = role_kpis.components_for(inputs, "Mihai Manea", "pm", pd.DataFrame(), pd.DataFrame())
    triage = next(p for p in parts if p.name == "Triage latency")
    assert triage.sufficient
    # Every ticket untouched: ages 10, 8, 30 -> median 10d -> ramp floor 0.
    assert triage.score == pytest.approx(0.0)


def test_board_hygiene_and_estimate_coverage_read_the_org_wide_board():
    open_tickets = pd.DataFrame(
        [
            {
                "key": "T-1", "assignee": "Shawn", "priority": "High",
                "policy_applies": True, "has_estimate": True,
            },
            {
                "key": "T-2", "assignee": "Unassigned", "priority": None,
                "policy_applies": True, "has_estimate": False,
            },
        ]
    )
    inputs = build(open_tickets=open_tickets)
    parts = role_kpis.components_for(inputs, "Mihai Manea", "pm", pd.DataFrame(), pd.DataFrame())
    hygiene = next(p for p in parts if p.name == "Board hygiene")
    assert hygiene.sufficient
    assert hygiene.n == 2
    assert hygiene.score == pytest.approx(50.0)  # 1/2 assigned, 1/2 prioritised, 1/2 estimated
    coverage = next(p for p in parts if p.name == "Estimate coverage")
    assert coverage.sufficient
    assert coverage.score == pytest.approx(50.0)


def test_stale_queue_reads_the_status_change_clock_not_idle_days():
    # T-1's only status move was 40 days ago, but it was cosmetically edited
    # yesterday. idle_days would call it fresh; status_age_days must not.
    events = events_of(
        issue(
            "T-1",
            history(when(40.0), "Shawn", status("To Do", "In Progress")),
            history(when(1.0), "Shawn", item("labels", None, "groomed")),
        ),
        issue("T-2", history(when(2.0), "Shawn", status("To Do", "In Progress"))),
    )
    open_tickets = pd.DataFrame(
        [
            {"key": "T-1", "assignee": "Shawn", "status": "In Progress", "idle_days": 1.0},
            {"key": "T-2", "assignee": "Shawn", "status": "In Progress", "idle_days": 2.0},
        ]
    )
    inputs = build(open_tickets=open_tickets, events=events)
    parts = role_kpis.components_for(inputs, "Mihai Manea", "pm", pd.DataFrame(), pd.DataFrame())
    stale = next(p for p in parts if p.name == "Stale queue rate")
    assert stale.sufficient
    assert stale.score == pytest.approx(50.0)  # T-1 is stale despite yesterday's grooming


# --------------------------------------------------------------------------
# Designer components
# --------------------------------------------------------------------------


def test_designer_staleness_is_immune_to_cosmetic_edits():
    events = events_of(
        issue(
            "D-1",
            history(when(40.0), "Robert Surpateanu", status("To Do", "In Progress")),
            history(when(0.5), "Robert Surpateanu", item("description", "a", "b")),
        ),
    )
    owned = pd.DataFrame(
        [{"key": "D-1", "assignee": "Robert Surpateanu", "status": "In Progress", "idle_days": 0.5}]
    )
    inputs = build(open_tickets=owned, events=events)
    parts = role_kpis.components_for(inputs, "Robert Surpateanu", "designer", owned, pd.DataFrame())
    staleness = next(p for p in parts if p.name == "Staleness")
    assert staleness.sufficient
    assert staleness.score == pytest.approx(0.0)  # 40d without a status move IS stale
    assert "cosmetic" in staleness.detail


def test_designer_cycle_time_scores_from_lead_time_median():
    events = events_of(
        *[
            issue(
                f"D-{i}",
                history(when(10.0 + i), "Robert Surpateanu", status("To Do", "In Progress")),
                history(when(8.0 + i), "Robert Surpateanu", status("In Progress", "Done")),
            )
            for i in range(3)
        ]
    )
    tickets = pd.DataFrame(
        [
            ticket_row(f"D-{i}", assignee="Robert Surpateanu", status="Done")
            for i in range(3)
        ]
    )
    inputs = build(all_tickets=tickets, events=events)
    parts = role_kpis.components_for(inputs, "Robert Surpateanu", "designer", pd.DataFrame(), pd.DataFrame())
    cycle = next(p for p in parts if p.name == "Cycle time")
    assert cycle.sufficient
    assert cycle.n == 3
    # 2-day lead times -> at/under the 3d fast bound -> 100.
    assert cycle.score == pytest.approx(100.0)


def test_designer_handoff_reads_the_declared_quality_score():
    gradable = pd.DataFrame(
        [
            {"key": "D-1", "assignee": "Robert Surpateanu", "quality_score": 4.0},
            {"key": "D-2", "assignee": "Robert Surpateanu", "quality_score": 2.0},
        ]
    )
    inputs = build()
    parts = role_kpis.components_for(
        inputs, "Robert Surpateanu", "designer", pd.DataFrame(), gradable
    )
    handoff = next(p for p in parts if p.name == "Handoff completeness")
    assert handoff.sufficient
    assert handoff.score == pytest.approx(60.0)  # mean 3.0 of 5
    assert "declared" in handoff.detail


# --------------------------------------------------------------------------
# Infrastructure components
# --------------------------------------------------------------------------


def test_infrastructure_estimate_churn_normalises_raises_by_tickets_moved():
    events = events_of(
        issue(
            "I-1",
            history(when(20.0), "Gaston", status("To Do", "In Progress")),
            history(when(18.0), "Gaston", item("timeoriginalestimate", "28800", "57600")),
            history(when(15.0), "Gaston", status("In Progress", "Done")),
        ),
        issue("I-2", history(when(10.0), "Gaston", status("To Do", "In Progress"))),
        issue("I-3", history(when(8.0), "Gaston", status("To Do", "In Progress"))),
    )
    inputs = build(events=events)
    parts = role_kpis.components_for(inputs, "Gaston", "infrastructure", pd.DataFrame(), pd.DataFrame())
    churn = next(p for p in parts if p.name == "Estimate churn")
    assert churn.sufficient
    assert churn.n == 3  # tickets they moved
    # 1 raise across 3 tickets moved -> 1 - 1/3.
    assert churn.score == pytest.approx(100.0 * (1 - 1 / 3))
    assert "+8h" in churn.detail


def test_hours_vs_output_stays_a_gap_without_pr_diffs():
    inputs = build()
    parts = role_kpis.components_for(inputs, "Gaston", "infrastructure", pd.DataFrame(), pd.DataFrame())
    hours = next(p for p in parts if p.name == "Hours vs delivered output")
    assert not hours.sufficient
    assert "linked PR diff" in hours.detail


# --------------------------------------------------------------------------
# people_table integration: the wired rubric reaches the table's score column
# --------------------------------------------------------------------------


def test_people_table_scores_a_qa_person_when_enough_components_have_data():
    import people_table

    events = _qa_events()
    tickets = pd.DataFrame(
        [
            ticket_row("QA-1"),
            ticket_row("QA-2"),
            ticket_row("QA-3"),
            ticket_row("QA-4"),
        ]
    )
    table = people_table.people_table(
        pd.DataFrame(), tickets, pd.DataFrame(), pd.DataFrame(), events,
        roster=roles.load_roster(), now=NOW,
    )
    row = table[table["person"] == "Dina QA"]
    assert not row.empty
    # escape 35 + verify 20 + rework 10 + estimates 10 = 75 of 100 >= 60:
    # a real score, from a rubric that had never been computed before.
    assert row["score"].iloc[0] == row["score"].iloc[0]  # not NaN
    assert float(row["measurable_pct"].iloc[0]) == pytest.approx(75.0)
