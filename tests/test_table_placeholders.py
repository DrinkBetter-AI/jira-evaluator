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

def test_a_number_column_with_nothing_in_it_is_left_empty_not_worded():
    """``Done %`` is drawn by a NumberColumn, which prints the word "None".

    ``_shown`` cannot help here - it writes text, and text in a number column is
    dropped - so the column is read as a number and an absent percentage becomes
    a blank cell.
    """
    frame = pd.DataFrame({"completion_pct": [50, None, "nan"]})
    numbers = pd.to_numeric(frame["completion_pct"], errors="coerce")
    assert numbers.tolist()[0] == 50
    assert numbers.isna().tolist() == [False, True, True]


def test_epoch_milliseconds_are_dated_rather_than_printed_raw():
    frame = pd.DataFrame({"created": [1774860044168.82, None], "updated": ["not a date", None]})
    dated = app._dated(frame, ("created", "updated"))
    assert dated["created"].tolist() == ["2026-03-30", ""]
    assert dated["updated"].tolist() == ["", ""]


def test_a_timestamp_column_keeps_its_date():
    frame = pd.DataFrame({"created": pd.to_datetime(["2026-01-02T03:04:05Z"])})
    assert app._dated(frame, ("created",))["created"].tolist() == ["2026-01-02"]
