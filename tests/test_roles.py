"""Tests for the roster and rubric table (roles.py).

Two failure modes matter more than the arithmetic here: a role silently
falling through to no rubric (the stale-template bug this task exists to
fix), and a peer comparison quietly widening past its documented cohort or
comparing someone to themselves. Every test below is aimed at one of those
two, or at the roster round trip they both depend on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kpi  # noqa: E402
import roles  # noqa: E402

# Parsed at collection time from the live (default) roster - not from a
# separate hardcoded list - so a role key added to JIRA_ROLES without a
# rubric decision changes this parametrization and fails the build, per the
# task's explicit acceptance rule.
_ROSTER = roles.load_roster()
_PARSED_ROLE_KEYS = _ROSTER.roles()


def test_the_default_roster_parses_all_16_documented_keys():
    assert set(_PARSED_ROLE_KEYS) == set(roles.ROLE_ORDER)
    assert len(roles.ROLE_ORDER) == 16


@pytest.mark.parametrize("role", _PARSED_ROLE_KEYS)
def test_every_parsed_role_resolves_to_a_rubric_or_an_explicit_sentinel(role):
    lookup = roles.rubric_for_role(role)
    # "unclassified" and "role_unknown" are exactly the silent-drop failure
    # this task exists to close - a role actually present in JIRA_ROLES must
    # never land there.
    assert lookup.status in ("scored", "no_rubric", "exec")
    if lookup.status == "scored":
        assert lookup.rubric is not None
    else:
        assert lookup.rubric is None


@pytest.mark.parametrize("role", sorted(roles.NO_RUBRIC_ROLES))
def test_no_rubric_roles_return_the_sentinel_not_a_zero(role):
    lookup = roles.rubric_for_role(role)
    assert lookup.status == "no_rubric"
    assert lookup.rubric is None
    assert "no rubric defined" in lookup.reason


def test_seo_wine_and_advisor_are_exactly_the_no_rubric_roles():
    # Pins the three roles named in ROSTER.md/the task text; if a future
    # rubric decision is made for one of these, this test forces the change
    # to be explicit rather than accidental.
    assert roles.NO_RUBRIC_ROLES == frozenset({"seo", "wine", "advisor"})


def test_exec_is_a_distinct_status_not_folded_into_no_rubric():
    lookup = roles.rubric_for_role("exec")
    assert lookup.status == "exec"
    assert lookup.rubric is None
    assert lookup.reason == "exec — not scored"


class TestRubricWeightsSumExactly:
    """A rubric whose weights don't sum to its documented total must raise,
    never silently round or renormalise."""

    def test_every_scored_rubric_sums_to_its_documented_total(self):
        seen = set()
        for role, rubric in roles.ROLE_RUBRIC.items():
            if rubric.rubric_id in seen:
                continue
            seen.add(rubric.rubric_id)
            total = sum(w.points for w in rubric.components)
            assert total == pytest.approx(rubric.total)

    def test_a_mismatched_rubric_raises_at_construction(self):
        with pytest.raises(ValueError, match="do not round"):
            roles.Rubric(
                rubric_id="broken",
                label="broken",
                components=(
                    roles.Weight("A", 40.0, "some blind spot"),
                    roles.Weight("B", 40.0, "another blind spot"),
                ),
                total=100.0,
            )

    def test_a_component_with_no_blind_spot_raises(self):
        with pytest.raises(ValueError, match="blind spot"):
            roles.Weight("A", 10.0, "   ")

    def test_a_component_with_zero_or_negative_weight_raises(self):
        with pytest.raises(ValueError, match="positive weight"):
            roles.Weight("A", 0.0, "a blind spot")


def test_code_rubric_weights_match_kpi_weights_exactly():
    """roles.CODE_RUBRIC is a hand-kept copy of kpi.WEIGHTS (no import cycle
    is possible since kpi.py imports roles.py); this is the guard against the
    two silently drifting apart."""
    code_weights = roles.CODE_RUBRIC.weights()
    assert code_weights == kpi.WEIGHTS
    assert roles.CODE_RUBRIC.total == pytest.approx(kpi.TOTAL_WEIGHT)


def test_every_component_of_every_rubric_names_a_blind_spot():
    for rubric in {r.rubric_id: r for r in roles.ROLE_RUBRIC.values()}.values():
        for component in rubric.components:
            assert component.blind_spot.strip()


# ---------------------------------------------------------------------------
# overall()
# ---------------------------------------------------------------------------


def test_overall_for_an_exec_is_none_with_the_exact_reason():
    result = roles.overall("Angel Vossough", _ROSTER)
    assert result.score is None
    assert result.reason == "exec — not scored"

    result2 = roles.overall("Arsalan", _ROSTER)
    assert result2.score is None
    assert result2.reason == "exec — not scored"


def test_overall_for_a_no_rubric_role_is_none_and_says_so():
    result = roles.overall("Igor Taborsak", _ROSTER)
    assert result.score is None
    assert "no rubric defined" in result.reason


def test_overall_for_an_unknown_person_is_role_unknown():
    result = roles.overall("Nobody Here", _ROSTER)
    assert result.score is None
    assert result.reason == "role unknown"


def test_overall_withholds_a_headline_below_the_scorable_fraction():
    # Tam is platform -> code rubric, total 100, needs 60.
    result = roles.overall("Tam", _ROSTER, {"Delivery": 90.0})  # only 10 of 100
    assert result.score is None
    assert "not scored" in result.reason


def test_overall_scores_once_enough_weight_is_covered():
    scored = {
        "Delivery": 100.0,
        "Delivery vs team": 100.0,
        "Rework": 100.0,
        "Weekly updates": 100.0,
        "Staleness": 100.0,
        "Carry-over": 100.0,
        "Estimates": 0.0,
    }
    result = roles.overall("Tam", _ROSTER, scored)
    weights = roles.CODE_RUBRIC.weights()
    covered = sum(weights[name] for name in scored)
    expected = sum(weights[name] * value for name, value in scored.items()) / covered
    assert covered >= roles.CODE_RUBRIC.total * roles.MIN_SCORABLE_FRACTION
    assert result.score is not None
    assert result.score == pytest.approx(expected)


def test_overall_ignores_component_names_the_rubric_does_not_recognise():
    result = roles.overall("Tam", _ROSTER, {"Not a real component": 100.0})
    assert result.score is None
    assert "not scored" in result.reason


# ---------------------------------------------------------------------------
# Cohorts
# ---------------------------------------------------------------------------


def test_frontend_family_folds_into_one_cohort():
    assert roles.cohort_key("frontend") == roles.cohort_key("frontend-mobile")
    assert roles.cohort_key("frontend-mobile") == roles.cohort_key("mobile")


def test_platform_and_backend_do_not_fold_into_the_frontend_cohort():
    assert roles.cohort_key("platform") != roles.cohort_key("frontend")
    assert roles.cohort_key("backend") != roles.cohort_key("frontend")
    assert roles.cohort_key("platform") != roles.cohort_key("mobile")


def test_frontend_family_cohort_has_three_sufficient_peers():
    for person in ("Ali", "Farid Shahidi", "Mohsen Davoudi", "David"):
        result = roles.peer_cohort(_ROSTER, person)
        assert result.sufficient is True
        assert result.peer_count == 3
        assert person not in result.peers


def test_a_lone_role_reports_insufficient_peers_with_the_real_count():
    # Tam is the only platform person in the default roster.
    result = roles.peer_cohort(_ROSTER, "Tam")
    assert result.sufficient is False
    assert result.peer_count == 0
    assert "insufficient peers" in result.reason
    assert "0" in result.reason


def test_cohort_never_includes_the_person_themselves():
    for person in _ROSTER.people_in_role("frontend") + _ROSTER.people_in_role("mobile"):
        result = roles.peer_cohort(_ROSTER, person.name)
        assert person.name not in result.peers


def test_unknown_person_gets_role_unknown_cohort_sentinel():
    result = roles.peer_cohort(_ROSTER, "Nobody Here")
    assert result.sufficient is False
    assert result.peer_count == 0
    assert result.reason == "role unknown"


# ---------------------------------------------------------------------------
# GitHub login round trip
# ---------------------------------------------------------------------------


def _default_login_pairs():
    for pair in roles._DEFAULT_GITHUB_LOGIN_MAP.split(";"):
        name, _, login = pair.partition("=")
        yield name.strip(), login.strip()


@pytest.mark.parametrize("name,login", list(_default_login_pairs()))
def test_every_login_in_the_map_belongs_to_a_person_with_a_jira_role(name, login):
    assert _ROSTER.role_of(name) is not None, f"{name} ({login}) has no JIRA_ROLES entry"
    assert _ROSTER.name_for_login(login) == name


def test_lawrnsfeng_and_vossbackend_are_reported_as_unmapped_not_dropped():
    assert "lawrnsfeng" in _ROSTER.unmapped_logins
    assert "VossBackend" in _ROSTER.unmapped_logins
    assert _ROSTER.name_for_login("lawrnsfeng") is None
    assert _ROSTER.name_for_login("VossBackend") is None


# ---------------------------------------------------------------------------
# Former staff
# ---------------------------------------------------------------------------


def _default_former_staff():
    return [n.strip() for n in roles._DEFAULT_JIRA_FORMER_STAFF.split(";") if n.strip()]


@pytest.mark.parametrize("name", _default_former_staff())
def test_every_former_staff_name_is_inactive(name):
    assert _ROSTER.is_active(name) is False


@pytest.mark.parametrize("name", _default_former_staff())
def test_former_staff_never_land_in_a_cohort(name):
    who = _ROSTER.person(name)
    assert who is not None
    assert who.role is None
    result = roles.peer_cohort(_ROSTER, name)
    assert result.sufficient is False
    assert result.peers == ()
    # And former staff must never appear as *someone else's* peer either.
    for role in roles.ROLE_ORDER:
        for candidate in _ROSTER.people_in_role(role):
            peer_result = roles.peer_cohort(_ROSTER, candidate.name)
            assert who.name not in peer_result.peers


# ---------------------------------------------------------------------------
# kpi.py's routing
# ---------------------------------------------------------------------------


def test_kpi_rubric_for_routes_through_roles():
    assert kpi.rubric_for("Tam", _ROSTER) == roles.rubric_for_person(_ROSTER, "Tam")
    assert kpi.rubric_for("Angel Vossough", _ROSTER).status == "exec"
    assert kpi.rubric_for("Igor Taborsak", _ROSTER).status == "no_rubric"


def test_kpi_rubric_for_defaults_unlisted_people_to_role_unknown_not_scored():
    lookup = kpi.rubric_for("Someone Not On Any Roster", _ROSTER)
    assert lookup.status == "role_unknown"
    assert lookup.rubric is None


# ---------------------------------------------------------------------------
# Env override
# ---------------------------------------------------------------------------


def test_env_none_falls_back_to_the_baked_defaults_not_a_blank_roster():
    ros = roles.load_roster(env={})
    assert set(ros.roles()) == set(roles.ROLE_ORDER)
    assert ros.role_of("Tam") == "platform"


def test_env_override_wins_per_variable_over_the_baked_default():
    ros = roles.load_roster(env={"JIRA_ROLES": "seo=Somebody New"})
    assert ros.role_of("Somebody New") == "seo"
    assert ros.role_of("Tam") is None  # the override replaced JIRA_ROLES entirely
    # GITHUB_LOGIN_MAP was not overridden, so it still falls back to its own
    # baked default rather than being blanked out by the JIRA_ROLES override.
    assert ros.name_for_login("Phelan164") is None  # Tam no longer has a role...
    assert "phelan164" in ros.login_index  # ...but the login map itself still parsed
