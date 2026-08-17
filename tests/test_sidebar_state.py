"""What the sidebar carries from one page to the next.

The six pages are one script, and Streamlit forgets a widget that a page does not
draw - so a status filter set on Engineering was gone on Delivery, and gone again
on coming back. Each choice is kept under a session key of its own instead, and
these are the ways that goes wrong: a carried choice the data no longer offers, a
link that names a person, an empty session on the first visit.
"""

from __future__ import annotations

import streamlit as st

import app


def _clear() -> None:
    for key in [k for k in st.session_state if k.startswith(app._CARRIED_PREFIX)]:
        del st.session_state[key]


def test_nothing_carried_yet_reads_as_the_default():
    _clear()
    assert app._carried("statuses", [], ["To Do"]) == []
    assert app._carried("scope", app.SCOPE_ORG, [app.SCOPE_ORG]) == app.SCOPE_ORG


def test_a_choice_survives_being_asked_for_again():
    """This is the page switch: the widget is gone, the choice is not."""
    _clear()
    app._carry("statuses", ["In Progress"])
    assert app._carried("statuses", [], ["To Do", "In Progress"]) == ["In Progress"]


def test_a_carried_status_the_board_no_longer_has_is_dropped():
    """Otherwise the view narrows to nothing on a filter nobody can see."""
    _clear()
    app._carry("statuses", ["Retired status"])
    assert app._carried("statuses", [], ["To Do", "In Progress"]) == []


def test_a_carried_scope_that_is_not_on_offer_falls_back():
    _clear()
    app._carry("scope", "Squad")
    scopes = [app.SCOPE_ORG, app.SCOPE_TEAM, app.SCOPE_INDIVIDUAL]
    assert app._carried("scope", app.SCOPE_ORG, scopes) == app.SCOPE_ORG


def test_a_number_carries_as_itself():
    _clear()
    app._carry("min_idle", 30)
    assert app._carried("min_idle", 0) == 30
