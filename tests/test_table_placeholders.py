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


def test_a_percentage_is_written_out_as_text_so_the_empty_ones_can_be_dashed():
    """Streamlit 1.61 paints a NaN in a NumberColumn as the word "None".

    So ``Done %`` is formatted here and drawn by a TextColumn, which lets
    ``_shown`` dash the unfilled ones like every other column in the table.
    """
    percent = pd.to_numeric(pd.Series([50, None, "nan"]), errors="coerce")
    written = [f"{value:.0f}%" if pd.notna(value) else "" for value in percent]
    frame = pd.DataFrame({"completion_pct": written})
    shown = app._shown(frame, ("completion_pct",))
    assert shown["completion_pct"].tolist() == ["50%", app._NO_VALUE, app._NO_VALUE]


def test_epoch_milliseconds_are_dated_rather_than_printed_raw():
    frame = pd.DataFrame({"created": [1774860044168.82, None], "updated": ["not a date", None]})
    dated = app._dated(frame, ("created", "updated"))
    assert dated["created"].tolist() == ["2026-03-30", ""]
    assert dated["updated"].tolist() == ["", ""]


def test_a_timestamp_column_keeps_its_date():
    frame = pd.DataFrame({"created": pd.to_datetime(["2026-01-02T03:04:05Z"])})
    assert app._dated(frame, ("created",))["created"].tolist() == ["2026-01-02"]
