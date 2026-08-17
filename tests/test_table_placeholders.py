"""What a table prints where Jira holds nothing.

Streamlit's grid draws a missing text cell as the word "None", so the stuck-triage
and stale tables named a Jira priority nobody has. The cards on the same pages say
"none" and "Nobody"; the tables say an em dash.
"""

from __future__ import annotations

import pandas as pd

import app


def test_every_spelling_of_nothing_prints_as_a_dash():
    frame = pd.DataFrame(
        {
            "priority": ["High", None, "", "  ", "None", float("nan")],
            "assignee": ["Tam", None, "none", "NaN", "<NA>", "NaT"],
        }
    )
    shown = app._shown(frame, ("priority", "assignee"))
    assert shown["priority"].tolist() == ["High"] + [app._NO_VALUE] * 5
    assert shown["assignee"].tolist() == ["Tam"] + [app._NO_VALUE] * 5


def test_a_column_that_is_not_there_is_not_invented():
    frame = pd.DataFrame({"priority": ["High"]})
    shown = app._shown(frame, ("priority", "assignee"))
    assert list(shown.columns) == ["priority"]


def test_the_frame_it_was_given_is_left_alone():
    """The page draws this table again after; it must not inherit the dashes."""
    frame = pd.DataFrame({"priority": [None]})
    app._shown(frame, ("priority",))
    assert frame["priority"].isna().all()
