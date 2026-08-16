"""Offline checks on the visual bugs a reader can see from across the room.

The dashboard cannot be rendered here - it needs Jira and GitHub - so what is
checked is the shape of what the page would draw: that a chart with no title
does not sprout the word "undefined" where its title belongs, that a ranking of
twenty-three people is drawn as ten bars that still add up to twenty-three, and
that a ticket with no epic says so rather than saying "nan".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import theme  # noqa: E402


class _Sink:
    """Somewhere for theme.plot to draw, since there is no browser here."""

    def __init__(self) -> None:
        self.figures: list[object] = []

    def plotly_chart(self, figure, **_kwargs) -> None:
        self.figures.append(figure)


def _drawn(figure):
    """Put ``figure`` through theme.plot and hand back what Streamlit would send.

    The assertions are made against ``to_plotly_json`` rather than against the
    figure's attributes, because the bug lives in the JSON: Streamlit's frontend
    asks whether the layout it was given has a ``title`` key at all, and a
    ``layout.title`` object is created by writing to any field inside it.
    """
    sink = _Sink()
    theme.plot(figure, into=sink)
    assert len(sink.figures) == 1
    return sink.figures[0].to_plotly_json()


def test_an_untitled_chart_is_not_given_a_title_to_leave_empty():
    """The "undefined" heading, at its source.

    ``title_font=`` is plotly's magic underscore for ``title.font.size``, and
    setting it brings ``layout.title`` into existence carrying a font and no
    text. Streamlit then does the equivalent of
    ``if ("title" in layout) layout.title.text = "<b>" + String(layout.title.text) + "</b>"``,
    and ``String(undefined)`` is the six letters readers were seeing above the
    ticket composition chart in production. The fix is for the key not to be
    there, so that is what is asserted - not merely that the text is empty.
    """
    layout = _drawn(go.Figure())["layout"]
    assert "title" not in layout, layout.get("title")
    assert layout["font"]["size"] == theme.CHART_FONT


def test_a_titled_chart_still_gets_the_bigger_title_text():
    """The other half: the size is not being dropped, only withheld.

    A title that exists is still drawn at the size a shared screen can read,
    which is why the font was being set unconditionally in the first place.
    """
    figure = go.Figure()
    figure.update_layout(title=dict(text="Who resolved tickets (30 days)"))
    layout = _drawn(figure)["layout"]
    assert layout["title"]["text"] == "Who resolved tickets (30 days)"
    assert layout["title"]["font"]["size"] == theme.CHART_TITLE_FONT


def test_whitespace_is_not_a_title():
    """A title of spaces would print as "<b> </b>" and reserve room for nothing."""
    figure = go.Figure()
    figure.update_layout(title=dict(text="   "))
    assert "font" not in _drawn(figure)["layout"].get("title", {})


def test_a_long_tail_becomes_one_other_bar_and_nothing_is_lost():
    """Ten rows out of twenty-three, still summing to every ticket counted.

    The collapse is the point of the chart - it is what a pie was refusing to do -
    but a chart that quietly drops eleven people's work would be worse than the
    pie it replaced, so the total is what is asserted alongside the shape.
    """
    counts = pd.Series(
        {f"person-{index:02d}": 23 - index for index in range(23)}
    )
    figure = theme.rank_bar(counts, title="Who merged PRs", value_label="PRs")
    bar = figure.data[0]

    categories = list(bar.y)
    assert len(categories) == theme.RANK_ROWS
    # Drawn largest-first, which for a horizontal bar means last in the list.
    assert categories[-1] == "person-00"
    other = [name for name in categories if name.startswith("Other (")]
    assert other == ["Other (14)"], categories
    assert sum(bar.x) == counts.sum()
    assert dict(zip(categories, bar.x))["Other (14)"] == counts.iloc[9:].sum()


def test_a_short_list_is_left_alone_and_collapsing_twice_changes_nothing():
    """The figure and the table beside it are built from the same call."""
    counts = pd.Series({"a": 3, "b": 2, "c": 1})
    assert list(theme.ranked(counts, top_n=10).index) == ["a", "b", "c"]

    long = pd.Series({f"p{index}": 100 - index for index in range(30)})
    once = theme.ranked(long, top_n=10)
    twice = theme.ranked(once, top_n=10)
    assert list(once.index) == list(twice.index)
    assert once.sum() == twice.sum() == long.sum()


def test_the_ranking_never_rotates_a_name_and_grows_with_its_rows():
    """The complaint these bars answer was sideways labels, not colour."""
    small = theme.rank_bar(
        pd.Series({"a": 2, "b": 1}), title="t", value_label="things"
    )
    big = theme.rank_bar(
        pd.Series({f"p{index}": index for index in range(10)}),
        title="t",
        value_label="things",
    )
    assert small.layout.yaxis.tickangle == 0
    assert big.layout.yaxis.tickangle == 0
    assert big.layout.height > small.layout.height
    # Horizontal bars, so the names sit on the category axis rather than being
    # turned on their side under a vertical one.
    assert small.data[0].orientation == "h"


def test_the_collapsed_tail_is_drawn_as_leftovers_rather_than_as_a_person():
    colors = list(
        theme.rank_bar(
            pd.Series({f"p{index}": 20 - index for index in range(15)}),
            title="t",
            value_label="things",
        ).data[0].marker.color
    )
    # Reversed for drawing, so the "Other" row is first in the colour list.
    assert colors[0] == theme.CATEGORICAL[-1]
    assert set(colors[1:]) == {theme.CATEGORICAL[0]}


def test_the_type_scale_is_a_ladder_rather_than_a_pile_of_sizes():
    """Every size in the stylesheet comes from the tokens, not from the keyboard."""
    ladder = [
        theme.TYPE_META,
        theme.TYPE_LABEL,
        theme.TYPE_BODY,
        theme.TYPE_LEAD,
        theme.TYPE_SECTION,
        theme.TYPE_DISPLAY,
    ]
    sizes = [int(rung.removesuffix("px")) for rung in ladder]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    assert sizes[0] >= 13, "13px is the floor the readability complaint set"

    for rung in ladder:
        assert rung in theme._STYLE
    # The sizes the complaint named, gone from the stylesheet for good.
    for banned in ("0.76rem", "0.78rem", "0.85rem"):
        assert banned not in theme._STYLE, banned


def test_the_main_column_has_a_width_and_the_sidebar_is_left_alone():
    assert theme.CONTENT_MAX_WIDTH in theme._STYLE
    assert "stSidebar" not in theme._STYLE


def test_there_is_one_categorical_palette_and_it_is_not_the_semantic_one():
    assert len(theme.CATEGORICAL) == len(set(theme.CATEGORICAL)) == 8
    assert theme.CATEGORICAL[0] == "#2563eb", "the app's own primaryColor leads"
    # The accents say good/bad; a chart borrowing them would make "green" mean
    # "third series" somewhere on the same page.
    assert not set(theme.CATEGORICAL) & set(theme.ACCENTS.values())
    assert theme.categorical(3) == list(theme.CATEGORICAL[:3])
    assert theme.categorical(9)[8] == theme.CATEGORICAL[0], "wraps rather than runs out"
    assert theme.categorical(0) == []

    figure = theme.apply_palette(go.Figure())
    assert list(figure.layout.colorway) == list(theme.CATEGORICAL)


def test_the_config_file_and_the_palette_have_not_drifted_apart():
    """Streamlit recolours plotly from config.toml, so the two lists must agree.

    Nothing enforces this at runtime: a colour changed in one place and not the
    other would show up as an `st.bar_chart` disagreeing with the chart beside it.
    """
    config = (
        Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    ).read_text()
    for color in theme.CATEGORICAL:
        assert color in config, color


def test_a_missing_field_reads_as_missing_rather_than_as_nan():
    """The other thing readers were seeing: "Epic: nan" on the triage card.

    An empty cell in a pandas frame is a float NaN, and NaN is truthy, so
    ``row["epic_summary"] or "none"`` hands back the NaN and the card prints the
    three letters pandas prints. Checked on the helper the card now goes through
    rather than by rendering the card, because rendering it needs a Jira frame.
    """
    import app as dashboard  # noqa: PLC0415 - importing the app is not free

    for missing in (float("nan"), None, "", "   ", pd.NaT):
        assert dashboard._text_or(missing, "none") == "none", repr(missing)
    assert dashboard._text_or("Checkout rewrite", "none") == "Checkout rewrite"
    assert dashboard._text_or("  padded  ", "none") == "padded"

    assert dashboard._number_or(float("nan")) == 0.0
    assert dashboard._number_or(None, 3.0) == 3.0
    assert dashboard._number_or("12.5") == 12.5
    assert dashboard._number_or("not a number", 1.0) == 1.0


def test_bots_leave_the_person_tables_and_stay_in_the_totals():
    """A ranking of engineers that puts Devin on top is not a ranking of engineers."""
    import app as dashboard  # noqa: PLC0415

    assert dashboard._is_bot("devin-ai-integration")
    assert dashboard._is_bot("devin-ai-integration[bot]"), "GitHub spells it both ways"
    assert dashboard._is_bot("Dependabot"), "and in either case"
    assert not dashboard._is_bot("alice")
    assert not dashboard._is_bot(None)

    prs = pd.DataFrame(
        {"author": ["alice", "github-actions", "bob", "renovate[bot]"], "n": [1, 2, 3, 4]}
    )
    people, bots = dashboard._people_only(prs, "author")
    assert list(people["author"]) == ["alice", "bob"]
    # The count comes back so the page can say why its table is shorter than the
    # tile above it, instead of the two silently disagreeing.
    assert bots == 2

    empty, none = dashboard._people_only(pd.DataFrame(), "author")
    assert empty.empty and none == 0


def test_the_helpers_still_draw_where_they_are_told():
    """theme.plot's own contract, unchanged by the title fix."""
    drawn = {}
    theme.st = type(
        "capture", (), {"plotly_chart": staticmethod(lambda fig, **kw: drawn.update(kw))}
    )
    try:
        theme.plot(go.Figure(), width="stretch", key="anything")
    finally:
        theme.st = st_module
    assert drawn == {"width": "stretch", "key": "anything"}
