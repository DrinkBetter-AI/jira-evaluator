"""Offline checks for the focused individual view's PR matching.

A personal page must show a person their own pull requests and nobody
else's, so what matters is that the two matching routes - the login map and
the Jira-key fallback - each find the right PRs and nothing more.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import focus  # noqa: E402


def prs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"number": 1, "author": "Phelan164", "jira_key": ""},
            {"number": 2, "author": "someone-else", "jira_key": "MB-10"},
            {"number": 3, "author": "someone-else", "jira_key": "MB-20"},
            {"number": 4, "author": "third-person", "jira_key": ""},
        ]
    )


def tickets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"key": "MB-10", "assignee": "Tam Nguyen"},
            {"key": "MB-20", "assignee": "Mehdi Ordikhani"},
        ]
    )


def test_the_login_map_reads_names_logins_and_alternates():
    mapping = focus.parse_login_map("Tam=Phelan164, Mehdi Ordikhani=mordikh|mehdi-o")
    assert mapping == {
        "tam": {"phelan164"},
        "mehdi ordikhani": {"mordikh", "mehdi-o"},
    }


def test_a_malformed_pair_costs_the_pair_not_the_map():
    mapping = focus.parse_login_map("garbage,Tam=Phelan164,=nobody,NoLogin=")
    assert mapping == {"tam": {"phelan164"}}


def test_a_first_name_key_matches_the_full_display_name():
    mapping = focus.parse_login_map("Tam=Phelan164")
    assert focus.logins_for("Tam Nguyen", mapping) == {"phelan164"}
    assert focus.logins_for("Mehdi Ordikhani", mapping) == set()


def test_a_two_word_key_matches_a_longer_display_name():
    mapping = focus.parse_login_map("Mehdi Ordikhani=mordikh")
    assert focus.logins_for("Mehdi Ordikhani Fard", mapping) == {"mordikh"}


def test_their_prs_are_theirs_by_login_or_by_ticket():
    mapping = focus.parse_login_map("Tam=Phelan164")
    mine = focus.personal_prs(prs(), "Tam Nguyen", tickets(), mapping)
    assert sorted(mine["number"].tolist()) == [1, 2]


def test_no_login_map_still_finds_prs_on_their_tickets():
    mine = focus.personal_prs(prs(), "Mehdi Ordikhani", tickets(), {})
    assert mine["number"].tolist() == [3]


def test_an_empty_frame_stays_empty():
    empty = pd.DataFrame()
    assert focus.personal_prs(empty, "Tam", tickets(), {}).empty
