"""Offline checks for the engineer scorecard arithmetic.

The scorecard is a management aid, so what matters is that each component
scores from the inputs it names, that a missing input drops the component
rather than zeroing the person, and that badges appear only when earned.

It is also a scorecard that people are paid against, so a second class of test
matters just as much: the ones that pin down what happens when the inputs are
absent. An engineer who can raise their score by holding fewer tickets has been
handed an incentive nobody meant to give them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kpi  # noqa: E402


def board() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Fresh, estimated, urgent and moving.
            {"idle_days": 1.0, "priority": "High", "has_estimate": True,
             "carry_over_count": 0, "issue_type": "Task"},
            # Stale, unestimated, carried over twice.
            {"idle_days": 20.0, "priority": "Low", "has_estimate": False,
             "carry_over_count": 2, "issue_type": "Task"},
        ]
    )


def test_every_component_scores_from_its_own_inputs():
    parts = kpi.components(
        board(),
        pd.DataFrame([{"quality_score": 4.0, "devinable": "Yes"}]),
        resolved_7=2,
        resolved_90=18,
        reopened_90=3,
        prs=pd.DataFrame([{"changes_reviews": 1}, {"changes_reviews": 0}]),
    )
    by_name = {p.name: p.score for p in parts}
    assert by_name["Weekly updates"] == 50.0  # 1 of 2 touched this week
    assert by_name["Staleness"] == 50.0  # 1 of 2 idle 15+
    assert by_name["Estimates"] == 50.0
    assert abs(by_name["Carry-over"] - 100.0 * (1 - 1.0 / 3.0)) < 0.01  # avg 1.0
    assert by_name["Urgent response"] == 100.0  # the one High ticket moved
    assert by_name["Devin-ready docs"] == 80.0  # 4.0 / 5
    # Delivery: 2/wk now vs 1.4/wk baseline, capped ratio scaling.
    assert by_name["Delivery"] > 50.0
    # Rework: mean of (1 - 3/18) and 1/2 PRs clean.
    assert abs(by_name["Rework"] - ((100 * 15 / 18) + 50.0) / 2) < 0.1


def test_a_missing_input_drops_the_component_not_the_person():
    parts = kpi.components(
        pd.DataFrame(), pd.DataFrame(), None, None, None, pd.DataFrame()
    )
    assert parts == []
    assert kpi.overall(parts) is None


def test_a_fragment_of_the_scorecard_does_not_get_a_headline():
    # Delivery (10) and Rework (20) is 30 of 100 points. Publishing a number
    # from that is how an empty board used to outscore a full one.
    parts = [kpi.Component("Delivery", 100.0, ""), kpi.Component("Rework", 0.0, "")]
    assert kpi.overall(parts) is None
    report = kpi.coverage(parts)
    assert report.scorable is False
    assert report.covered_weight == 30.0
    assert "Weekly updates" in report.note


def test_enough_of_the_scorecard_scores_over_the_weight_that_had_data():
    parts = [
        kpi.Component("Rework", 100.0, ""),  # 20
        kpi.Component("Weekly updates", 100.0, ""),  # 15
        kpi.Component("Staleness", 0.0, ""),  # 10
        kpi.Component("Carry-over", 100.0, ""),  # 10
        kpi.Component("Estimates", 0.0, ""),  # 10
    ]
    # 45 of 65 points scored full, and the note names what was missing.
    assert kpi.overall(parts) == pytest.approx(100.0 * 45 / 65)
    report = kpi.coverage(parts)
    assert report.scorable is True
    assert "Delivery" in report.missing


def test_pushing_every_ticket_to_backlog_stops_the_score_instead_of_raising_it():
    # The gaming move the old renormalisation rewarded: hold no open non-backlog
    # tickets, lose 45 points of denominator, get scored on what is left.
    empty_board = kpi.components(
        pd.DataFrame(),
        pd.DataFrame([{"quality_score": 5.0}]),
        resolved_7=1,
        resolved_90=1,
        reopened_90=0,
        prs=pd.DataFrame([{"changes_reviews": 0}]),
    )
    assert {p.name for p in empty_board} == {"Delivery", "Rework", "Devin-ready docs"}
    assert kpi.overall(empty_board) is None
    assert "Weekly updates" in kpi.coverage(empty_board).missing


def test_a_component_with_no_data_can_be_shown_as_insufficient_rather_than_hidden():
    parts = kpi.components(
        board(),
        pd.DataFrame(),
        resolved_7=None,
        resolved_90=None,
        reopened_90=None,
        prs=pd.DataFrame(),
        include_gaps=True,
    )
    gaps = {p.name: p for p in parts if not p.sufficient}
    assert "Delivery" in gaps and "Devin-ready docs" in gaps
    assert gaps["Delivery"].detail.startswith("insufficient data")
    assert gaps["Delivery"].n == 0
    # A placeholder must never drag the score down as if it were a zero.
    assert kpi.overall(parts) == kpi.overall([p for p in parts if p.sufficient])


def test_every_component_says_how_many_things_it_counted():
    parts = kpi.components(
        board(),
        pd.DataFrame([{"quality_score": 4.0}]),
        resolved_7=2,
        resolved_90=18,
        reopened_90=1,
        prs=pd.DataFrame([{"changes_reviews": 0}]),
        peer_resolved_7=[0, 1, 2, 9],
    )
    by_name = {p.name: p for p in parts}
    # 100% on two tickets and 100% on forty are different claims; n is how a
    # reader tells them apart.
    assert by_name["Weekly updates"].n == 2
    assert by_name["Estimates"].n == 2
    assert by_name["Devin-ready docs"].n == 1
    assert by_name["Delivery"].n == 18
    assert by_name["Delivery vs team"].n == 4
    assert all(p.n is not None for p in parts)


def test_the_delivery_baseline_stops_counting_the_week_it_is_judging():
    # 12 resolved in 90 days, 12 of them this week: the prior 83 days were
    # empty. The old arithmetic divided by a baseline the burst had inflated.
    assert kpi.baseline_rate(12, 12, 7, 90) == 0.0
    # 4 this week out of 20 in the quarter leaves 16 over 83 days.
    assert kpi.baseline_rate(4, 20, 7, 90) == pytest.approx(7.0 * 16 / 83)
    # The counts come from separate queries and can disagree; never go negative.
    assert kpi.baseline_rate(9, 5, 7, 90) == 0.0
    assert kpi.baseline_rate(None, 20) is None


def test_a_quiet_quarter_no_longer_builds_a_baseline_a_burst_can_beat():
    # Someone who resolved nothing for 83 days and then closed 5 trivial
    # tickets scores 100 on the self-relative reading. It is not wrong - there
    # is nothing to compare against - and it is exactly why the peer reading
    # sits next to it carrying the same weight.
    parts = kpi.components(
        pd.DataFrame(), pd.DataFrame(), 5, 5, None, pd.DataFrame(),
        peer_resolved_7=[5, 9, 11, 14],
    )
    by_name = {p.name: p for p in parts}
    assert by_name["Delivery"].score == 100.0
    assert "no prior-83-day pace" in by_name["Delivery"].detail
    assert by_name["Delivery vs team"].score == pytest.approx(12.5)  # bottom of the team


def test_the_peer_reading_needs_a_real_cohort_before_it_says_anything():
    assert kpi.peer_percentile(3.0, [1.0, 5.0]) is None  # two people is not a team
    assert kpi.peer_percentile(3.0, [1.0, 2.0, 3.0, 9.0]) == pytest.approx(62.5)
    assert kpi.peer_percentile(None, [1.0, 2.0, 3.0]) is None
    # Everybody identical: nobody is above or below anybody.
    assert kpi.peer_percentile(4.0, [4.0, 4.0, 4.0]) == 50.0


def test_the_peer_reading_takes_a_mapping_of_person_to_count():
    parts = kpi.components(
        pd.DataFrame(), pd.DataFrame(), 6, 30, None, pd.DataFrame(),
        peer_resolved_7={"Ana": 6, "Bo": 1, "Cy": 2, "Di": 3},
    )
    by_name = {p.name: p.score for p in parts}
    assert by_name["Delivery vs team"] == pytest.approx(87.5)


def test_without_peer_counts_the_peer_component_is_absent_not_invented():
    parts = kpi.components(board(), pd.DataFrame(), 2, 18, 1, pd.DataFrame())
    assert "Delivery vs team" not in {p.name for p in parts}
    assert "Delivery vs team" in kpi.coverage(parts).missing


def test_an_epic_does_not_count_against_estimate_discipline():
    owned = pd.DataFrame(
        [
            {"idle_days": 1.0, "has_estimate": False, "issue_type": "Epic"},
            {"idle_days": 1.0, "has_estimate": True, "issue_type": "Task"},
        ]
    )
    parts = kpi.components(owned, pd.DataFrame(), None, None, None, pd.DataFrame())
    by_name = {p.name: p.score for p in parts}
    assert by_name["Estimates"] == 100.0


def test_a_merged_pr_that_cannot_show_its_key_does_not_cost_the_clean_badge():
    prs = pd.DataFrame(
        [
            {"has_jira_key": True, "is_unowned": False, "key_detectable": True},
            # Merged fetch carries no branch/body, so no key is visible.
            {"has_jira_key": False, "is_unowned": False, "key_detectable": False},
        ]
    )
    earned = dict(kpi.badges([], pd.DataFrame(), pd.DataFrame(), None, None, None, None, prs))
    assert "🔍 Clean PRs" in earned


def test_delivered_hours_sums_estimates_and_says_what_it_cannot_see():
    resolved = pd.DataFrame(
        [
            {"estimate_hours": 4.0, "issue_type": "Task"},
            {"estimate_hours": 2.5, "issue_type": "Bug"},
            {"estimate_hours": 0.0, "issue_type": "Bug"},
            # A closing epic is its children's work, not a week of its own.
            {"estimate_hours": 40.0, "issue_type": "Epic"},
        ]
    )
    assert kpi.delivered_hours(resolved) == (6.5, 3, 1)


def test_delivered_hours_of_an_empty_week_is_zero():
    assert kpi.delivered_hours(pd.DataFrame()) == (0.0, 0, 0)


def test_no_judgeable_pr_means_no_clean_badge():
    prs = pd.DataFrame(
        [
            # Only merged PRs, none of which can show a key: no evidence,
            # so no badge.
            {"has_jira_key": False, "is_unowned": False, "key_detectable": False},
        ]
    )
    earned = dict(kpi.badges([], pd.DataFrame(), pd.DataFrame(), None, None, None, None, prs))
    assert "🔍 Clean PRs" not in earned


def test_the_estimate_rule_follows_the_written_policy_when_present():
    owned = pd.DataFrame(
        [
            # Backlog ticket exempt by policy despite missing estimate.
            {"idle_days": 1.0, "has_estimate": False, "policy_applies": False},
            {"idle_days": 1.0, "has_estimate": True, "policy_applies": True},
        ]
    )
    parts = kpi.components(owned, pd.DataFrame(), None, None, None, pd.DataFrame())
    by_name = {p.name: p.score for p in parts}
    assert by_name["Estimates"] == 100.0


def test_badges_appear_only_when_earned():
    fresh = pd.DataFrame([{"idle_days": 2.0, "has_estimate": True, "issue_type": "Task"}])
    parts = kpi.components(fresh, pd.DataFrame(), 4, 12, 0, pd.DataFrame())
    earned = dict(
        kpi.badges(parts, fresh, pd.DataFrame(), 4, 6, 12, 0, pd.DataFrame())
    )
    assert "🚢 Shipper" in earned  # 4 resolved this week
    assert "📈 Trending up" in earned  # 4/wk > 1.4/wk >= 0.93/wk
    assert "⭐ Fresh board" in earned
    assert "📝 Estimator" in earned
    assert "🧪 No boomerangs" in earned  # 12 resolved, 0 back
    assert "🤖 AI teammate" not in earned


def test_no_badges_for_an_idle_board():
    stale = pd.DataFrame([{"idle_days": 30.0, "has_estimate": False, "issue_type": "Task"}])
    parts = kpi.components(stale, pd.DataFrame(), 0, 5, 2, pd.DataFrame())
    assert kpi.badges(parts, stale, pd.DataFrame(), 0, 1, 5, 2, pd.DataFrame()) == []


def test_clean_prs_badge_needs_keys_and_reviewers():
    prs = pd.DataFrame(
        [
            {"has_jira_key": True, "is_unowned": False, "changes_reviews": 0},
            {"has_jira_key": False, "is_unowned": False, "changes_reviews": 0},
        ]
    )
    earned = dict(kpi.badges([], pd.DataFrame(), pd.DataFrame(), None, None, None, None, prs))
    assert "🔍 Clean PRs" not in earned
    keyed = prs.assign(has_jira_key=True)
    earned = dict(
        kpi.badges([], pd.DataFrame(), pd.DataFrame(), None, None, None, None, keyed)
    )
    assert "🔍 Clean PRs" in earned
