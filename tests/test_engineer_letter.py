"""Offline checks for the page an engineer is sent about their own tickets.

The page leaves the dashboard - it is mailed, opened on a phone, printed - so
what is worth testing is that it says the same thing the screen says: the same
tier colours, the same hour totals, and a key that is a link into Jira rather
than a string of text. And that the link an engineer is sent lands on them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as dashboard  # noqa: E402
import engineer_letter  # noqa: E402


def board() -> pd.DataFrame:
    """Two tickets one behind the other, and one that is fine."""
    return pd.DataFrame(
        [
            {
                "key": "MB-8834",
                "assignee": "Santi Caamaño",
                "summary": "Search Experience Redesign",
                "status": "Review in Staging",
                "priority": "Highest",
                "priority_score": 66.2,
                "idle_days": 22.3,
                "ticket_age_days": 24.0,
                "estimate_hours": 0.0,
                "original_estimate_sec": 0,
                "issue_type": "Story",
                "sprint_state": None,
                "sprint_name": None,
                "created": "2026-07-22",
                "updated": "2026-07-24",
            },
            {
                "key": "MB-8594",
                "assignee": "Santi Caamaño",
                "summary": "Display strike-through original price",
                "status": "Review in Staging",
                "priority": "Highest",
                "priority_score": 73.7,
                "idle_days": 24.9,
                "ticket_age_days": 38.0,
                "estimate_hours": 1.0,
                "original_estimate_sec": 3600,
                "issue_type": "Bug",
                "sprint_state": "active",
                "sprint_name": "Marketplace 48",
                "created": "2026-07-09",
                "updated": "2026-07-27",
            },
            {
                "key": "MB-9001",
                "assignee": "Someone Else",
                "summary": "Not this person's ticket",
                "status": "To Do",
                "priority": "Low",
                "priority_score": 4.0,
                "idle_days": 1.0,
                "ticket_age_days": 2.0,
                "estimate_hours": 3.0,
                "original_estimate_sec": 10800,
                "issue_type": "Task",
                "sprint_state": "active",
                "sprint_name": "Marketplace 48",
                "created": "2026-08-14",
                "updated": "2026-08-15",
            },
        ]
    )


def page_for(person: str = "Santi Caamaño") -> engineer_letter.Page:
    owned = dashboard.annotated_board(board(), person)
    return dashboard.engineer_page(person, owned, score=60.0, badges=[("🚢", "Shipper")])


def test_the_page_holds_only_this_person_s_tickets():
    page = page_for()
    assert [t.key for t in page.tickets] == ["MB-8594", "MB-8834"]  # worst tier first
    assert "MB-9001" not in engineer_letter.one_pager(page)


def test_the_hours_on_the_page_are_the_dashboard_s_own_totals():
    owned = dashboard.annotated_board(board(), "Santi Caamaño")
    load = dashboard._workload_hours(owned)
    values = {tile.label: tile.value for tile in page_for().tiles}
    assert values["Estimated hours"] == f"{load['total']:.1f} h"
    assert values["Current sprint"] == f"{load['sprint']:.1f} h"
    assert values["Urgent (High+)"] == f"{load['urgent']:.1f} h"
    assert values["No estimate"] == str(load["unestimated"])
    assert values["Open tickets"] == "2"


def test_every_key_is_a_link_into_jira():
    html = engineer_letter.one_pager(page_for())
    for key in ("MB-8834", "MB-8594"):
        assert f'>{key}</a>' in html
        assert dashboard._jira_ticket_url(key) in html


def test_the_tiers_are_the_colours_the_dashboard_draws():
    """A tier recoloured on screen and not here would send a lying page."""
    assert engineer_letter.TIER_COLOURS == dashboard._TIER_BG
    html = engineer_letter.one_pager(page_for())
    assert engineer_letter.TIER_COLOURS["Needs attention"] in html


def test_the_stale_tickets_are_asked_about_by_name():
    page = page_for()
    asked = engineer_letter.needs_updating(page.tickets)
    # Longest silence first, and both are well past a week.
    assert [t.key for t in asked] == ["MB-8594", "MB-8834"]
    html = engineer_letter.one_pager(page)
    assert "Please leave a comment" in html
    assert "25 days without an update" in html


def test_a_board_that_moved_this_week_is_not_nagged():
    fresh = engineer_letter.Page(
        person="Fresh",
        tickets=[
            engineer_letter.Ticket(
                key="MB-1",
                url="https://example.atlassian.net/browse/MB-1",
                summary="Moved yesterday",
                status="In Progress",
                priority="High",
                tier="Healthy",
                sprint="Marketplace 48",
                idle_days=1.0,
            )
        ],
    )
    assert engineer_letter.needs_updating(fresh.tickets) == []
    assert "Keep it that way" in engineer_letter.one_pager(fresh)


def test_a_ticket_with_no_estimate_shows_blank_rather_than_a_zero():
    """A zero would read as an estimate of no work, which is a different claim."""
    row = [t for t in page_for().tickets if t.key == "MB-8834"][0]
    assert row.estimate_hours is None
    unestimated = engineer_letter.Ticket(
        key="MB-2",
        url="",
        summary="No estimate",
        status="To Do",
        priority="Low",
        tier="Low priority",
        sprint="Backlog",
        idle_days=None,
        estimate_hours=None,
    )
    html = engineer_letter.one_pager(
        engineer_letter.Page(person="Nobody", tickets=[unestimated])
    )
    assert "<td class=\"num\"></td>" in html
    # No link to hang on a ticket with no URL, and no broken anchor either.
    assert "<td class=\"key\">MB-2</td>" in html


def test_a_summary_that_contains_markup_is_escaped():
    nasty = engineer_letter.Ticket(
        key="MB-3",
        url="https://example.atlassian.net/browse/MB-3",
        summary="<script>alert(1)</script>",
        status="To Do",
        priority="Low",
        tier="Healthy",
        sprint="Backlog",
    )
    html = engineer_letter.one_pager(engineer_letter.Page(person="X", tickets=[nasty]))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_the_file_name_carries_the_person_and_the_day():
    assert engineer_letter.filename("Santi Caamaño").startswith("santi-caama")
    assert engineer_letter.filename("Santi Caamaño").endswith(".html")


def test_a_link_that_names_an_engineer_is_matched_to_them(monkeypatch):
    people = ["Mehdi Ordikhani Fard", "Santi Caamaño"]

    monkeypatch.setattr(dashboard.st, "query_params", {"person": "santi caamaño"})
    assert dashboard.requested_person(people) == "Santi Caamaño"

    # A roster name written short still lands on the Jira account it means.
    monkeypatch.setattr(dashboard.st, "query_params", {"person": "Mehdi Ordikhani"})
    assert dashboard.requested_person(people) == "Mehdi Ordikhani Fard"

    monkeypatch.setattr(dashboard.st, "query_params", {})
    assert dashboard.requested_person(people) is None

    # A name nobody here has selects nobody rather than the first person.
    monkeypatch.setattr(dashboard.st, "query_params", {"person": "A Stranger"})
    assert dashboard.requested_person(people) is None


def test_the_shared_link_keeps_the_address_the_reader_reached_us_on(monkeypatch):
    monkeypatch.setattr(
        dashboard.st,
        "context",
        type("context", (), {"url": "https://dash.example.app/engineering?person=Old"}),
    )
    link = dashboard.person_link("Santi Caamaño")
    assert link.startswith("https://dash.example.app/engineering?person=")
    assert link.count("?") == 1
    assert "Caama%C3%B1o" in link
