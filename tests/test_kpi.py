"""Offline checks for the engineer scorecard arithmetic.

The scorecard is a management aid, so what matters is that each component
scores from the inputs it names, that a missing input drops the component
rather than zeroing the person, and that badges appear only when earned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

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


def test_the_overall_score_renormalizes_over_present_components():
    parts = [kpi.Component("Delivery", 100.0, ""), kpi.Component("Rework", 0.0, "")]
    assert kpi.overall(parts) == 50.0  # equal weights renormalized


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
