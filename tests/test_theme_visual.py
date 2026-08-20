"""Offline checks on the visual bugs a reader can see from across the room.

The dashboard cannot be rendered here - it needs Jira and GitHub - so what is
checked is the shape of what the page would draw: that a chart with no title
does not sprout the word "undefined" where its title belongs, that a ranking of
twenty-three people is drawn as ten bars that still add up to twenty-three, and
that a ticket with no epic says so rather than saying "nan".
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import page_shared  # noqa: E402
import render_shared  # noqa: E402
import theme  # noqa: E402
import theme_html  # noqa: E402
import theme_tokens  # noqa: E402
from pages import (  # noqa: E402
    business,
    code,
    delivery,
    engineering,
    integrity,
    people,
    planning,
    today,
)


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
    """Every size in the stylesheet comes from the tokens, not from the keyboard.

    Each rung has to be *used*, but not necessarily inside ``theme._STYLE``
    itself: task 5A moved the triage card's headline (the ``TYPE_SECTION``/
    20px rung) off of a page-private CSS class (``.triage-card
    .triage-summary``, since deleted - it duplicated theme_html.py's own
    ``.card``) and onto an inline style built from the same ``theme.
    TYPE_SECTION`` constant in ``pages/today.py``'s hero. A rung is "used"
    if it shows up literally in ``theme._STYLE`` or as ``theme.<NAME>`` in
    any page/render module - either way, nobody typed the pixel value by
    hand at the point of use. See docs/assumptions/5A.md.
    """
    ladder = {
        "TYPE_META": theme.TYPE_META,
        "TYPE_LABEL": theme.TYPE_LABEL,
        "TYPE_BODY": theme.TYPE_BODY,
        "TYPE_LEAD": theme.TYPE_LEAD,
        "TYPE_SECTION": theme.TYPE_SECTION,
        "TYPE_DISPLAY": theme.TYPE_DISPLAY,
    }
    sizes = [int(rung.removesuffix("px")) for rung in ladder.values()]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    assert sizes[0] >= 13, "13px is the floor the readability complaint set"

    repo_root = Path(__file__).resolve().parents[1]
    consumer_source = "".join(
        path.read_text()
        for path in (repo_root / "pages").glob("*.py")
    ) + (repo_root / "render_shared.py").read_text()
    for name, rung in ladder.items():
        assert rung in theme._STYLE or f"theme.{name}" in consumer_source, name
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


# ============================================================================
# Task 5A: the design conformance sweep.
#
# Seven agents built seven pages against one contract (theme_tokens.py's
# tokens, theme_html.py's components); this section is what makes them read
# as one product rather than seven. Every rule below scans real source across
# all eight pages this dashboard serves (pages/today.py, people.py,
# delivery.py, code.py, planning.py, engineering.py, business.py,
# integrity.py - the exact list app.py::_page_specs() registers) - not a
# spot check on one. Two of the eight (business.py, engineering.py) already
# had a narrower version of the hex/font-ladder pair in
# tests/test_price_charts.py and tests/test_engineering_page.py; those stay
# (they are not wrong), and this section is the one place that checks every
# page, all eight rules, in one pass.
#
# What cannot be scanned from source is said so at the point it is skipped:
# ``st.dataframe`` is drawn on a canvas no stylesheet or static scan can see
# into (theme.py's own CSS comment explains this), and a component library's
# own internal call-site count (theme_html.legend()'s two call sites) is
# small enough to read directly rather than parse generically.
# ============================================================================

_PAGES = {
    "today": today,
    "people": people,
    "delivery": delivery,
    "code": code,
    "planning": planning,
    "engineering": engineering,
    "business": business,
    "integrity": integrity,
}

# theme.py/theme_html.py/render_shared.py/page_shared.py are the shared
# rendering layer every page draws through - a literal hex or an off-ladder
# size hiding in one of these reaches every page at once, so they are swept
# alongside the eight page modules rather than left out because they are not
# themselves a "page".
_SHARED_RENDERING_MODULES = {
    "theme": theme,
    "theme_html": theme_html,
    "render_shared": render_shared,
    "page_shared": page_shared,
}

_ALL_SWEPT_MODULES = {**_PAGES, **_SHARED_RENDERING_MODULES}


def _source(module) -> str:
    return Path(module.__file__).read_text()


# theme_tokens.py's own six-rung ladder, plus the one rung above it
# (theme_tokens.HERO_SIZE) that a page's one hero number is allowed to draw
# at - see that constant's own docstring for why it is a rung of its own
# rather than a seventh entry in TYPE.
_ON_LADDER_SIZES = set(theme_tokens.TYPE.values()) | {theme_tokens.HERO_SIZE}


def test_no_page_or_shared_module_has_a_literal_hex_colour():
    """Every colour traces back to theme_tokens - across all eight pages.

    theme_html.py itself is exempt from the *first* assertion below (not the
    loop) for the same reason docs/assumptions/1A.md and this file's own
    ``_CHROME``/``_CSS_TOKENS`` give: a handful of chrome greys and the one
    brand-logo gradient the mockup itself never promoted to a ``--variable``.
    Those are pre-existing, documented, and checked by
    tests/test_theme_html.py already; what this test guards is that no *new*
    literal joins them anywhere else in the rendering stack.
    """
    hex_pattern = re.compile(r"#[0-9a-fA-F]{6}\b")
    for name, module in _ALL_SWEPT_MODULES.items():
        if module is theme_html:
            continue
        found = hex_pattern.findall(_source(module))
        assert found == [], f"{name}: {found}"


def test_no_page_or_shared_module_uses_an_off_ladder_font_size():
    """Every explicit font size is one of the seven sanctioned rungs.

    13 / 14 / 15 / 17 / 20 / 32 from theme_tokens.TYPE, plus 48
    (theme_tokens.HERO_SIZE) for the one hero number a page is allowed. A
    size typed as a plain number (``font=dict(size=11)``, an inline
    ``font-size: 11px``) rather than read from a token is exactly how the
    dashboard grew five different "small print" sizes before theme_tokens.py
    existed.
    """
    # theme_html.py is excluded here, not from the hex sweep alone: its 11px/
    # 12px rules (".chip", ".dim", ".av", ".rolechip", ".nnote", the SVG axis
    # labels, ...) are copied verbatim from the mockup's own stylesheet for
    # genuinely decorative micro-text - badge initials, footnote weights -
    # that was never part of the readability complaint theme_tokens.TYPE's
    # 13px floor answers (that complaint was about body/table/caption text).
    # No existing test in tests/test_theme_html.py enforces a ladder-only
    # rule on this file either; this one does not add one.
    pattern = re.compile(r"font[_-]size[\"' :=]+(\d+)", re.IGNORECASE)
    for name, module in _ALL_SWEPT_MODULES.items():
        if module is theme_html:
            continue
        for match in pattern.finditer(_source(module)):
            assert int(match.group(1)) in _ON_LADDER_SIZES, f"{name}: {match.group(0)}"


def test_no_axis_label_is_ever_rotated():
    """Every ``tickangle``/``textangle`` this dashboard sets is 0, on every page.

    "Turn the chart on its side" (theme.rank_bar's own docstring, and
    pages/code.py's ``_share_rank_bar``) is the answer to a label that does
    not fit - never a rotated tick. Checked as "every value this codebase
    sets is 0" rather than "no page sets a value" so a chart that states
    ``tickangle=0`` on purpose still passes; what fails is a future
    ``tickangle=45``.
    """
    pattern = re.compile(r"(?:tick|text)angle\s*=\s*(-?\d+)")
    for name, module in _ALL_SWEPT_MODULES.items():
        for match in pattern.finditer(_source(module)):
            assert int(match.group(1)) == 0, f"{name}: {match.group(0)}"


def test_every_plotly_line_is_2px():
    """Every ``line=dict(..., width=N)`` this codebase draws states 2px.

    ``theme_html.py``'s own SVG lines (``spark()``, ``linechart()``) are
    checked separately below, by source, since they are not built through
    plotly's ``line=dict(...)`` shape at all.
    """
    pattern = re.compile(r"line\s*=\s*dict\([^)]*?width\s*=\s*(\d+)", re.DOTALL)
    for name, module in _ALL_SWEPT_MODULES.items():
        for match in pattern.finditer(_source(module)):
            assert int(match.group(1)) == 2, f"{name}: {match.group(0)}"


def test_theme_html_svg_lines_are_2px_and_gridlines_are_hairline():
    """``spark()``/``linechart()`` draw at 2px; ``linechart()``'s gridlines at 1px.

    These are hand-built SVG strings, not plotly figures, so there is no
    figure spec to inspect - the source literal *is* the spec. Scanned
    directly rather than by rendering, per this file's own module docstring
    (no browser here).
    """
    source = _source(theme_html)
    polyline_widths = {
        int(w) for w in re.findall(r'<polyline[^>]*stroke-width="(\d+)"', source)
    }
    assert polyline_widths == {2}, polyline_widths
    gridline_widths = {
        int(w) for w in re.findall(r'<line [^>]*stroke-width="(\d+)"', source)
    }
    assert gridline_widths == {1}, gridline_widths


def _bar_thickness_px(figure) -> float:
    """The rendered pixel height of one bar in a horizontal ``go.Bar`` figure.

    Read from the figure's own layout (height, margin, bargap) and its own
    row count, rather than re-deriving it from theme.BAR_ROW_HEIGHT/BAR_GAP -
    the point is to check what the figure actually specifies, not that two
    constants agree with themselves.
    """
    layout = figure.layout
    n = len(figure.data[0].y)
    margin = layout.margin
    plot_height = layout.height - (margin.t or 0) - (margin.b or 0)
    bargap = layout.bargap if layout.bargap is not None else 0.0
    return (plot_height / n) * (1 - bargap)


def test_rank_bar_never_exceeds_the_24px_ceiling_and_has_4px_corners():
    """theme.rank_bar: 1, 10 and 23 rows all draw bars at or under 24px."""
    for count in (1, 10, 23):
        series = pd.Series({f"person-{i:02d}": count - i for i in range(count)})
        figure = theme.rank_bar(series, title="t", value_label="things")
        thickness = _bar_thickness_px(figure)
        assert thickness <= theme.BAR_MAX_HEIGHT + 0.01, (count, thickness)
        assert figure.data[0].marker.cornerradius == theme.BAR_CORNER_RADIUS


def test_code_pages_share_rank_bar_never_exceeds_the_24px_ceiling():
    """pages/code.py's own ``go.Bar`` (review-coverage severity ramp) matches."""
    for count in (1, 6, 15):
        shares = pd.Series(
            {f"repo-{i}": float(20 + i * 3) for i in range(count)}
        )
        figure = code._share_rank_bar(shares)
        thickness = _bar_thickness_px(figure)
        assert thickness <= theme.BAR_MAX_HEIGHT + 0.01, (count, thickness)
        assert figure.data[0].marker.cornerradius == theme.BAR_CORNER_RADIUS


def test_theme_html_hbar_css_matches_the_same_24px_4px_ceiling():
    """The HTML ``.hbar`` track (theme_html.py, not plotly) is under the ceiling.

    This is CSS, not a figure spec, so it is read directly: ``.hbar .track``'s
    stated height must be <= theme.BAR_MAX_HEIGHT, and its border-radius must
    use the 4px rung on the rounded end.
    """
    match = re.search(r"\.vv \.hbar \.track \{([^}]*)\}", _source(theme_html))
    assert match, "no .hbar .track rule found"
    body = match.group(1)
    height = int(re.search(r"height:\s*(\d+)px", body).group(1))
    assert height <= theme.BAR_MAX_HEIGHT, height
    assert "4px" in re.search(r"border-radius:\s*([^;]+);", body).group(1)


def _legend_call_sizes(module) -> list[int]:
    """How many ``(label, hue)`` pairs each ``theme_html.legend(...)`` call
    in ``module`` passes, for every call site whose argument is a literal
    list - both real call sites in this codebase are, and an AST walk
    catches this precisely where a regex would have to guess at bracket
    nesting.
    """
    import ast

    tree = ast.parse(_source(module))
    sizes = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "legend"
            and isinstance(getattr(node.func, "value", None), ast.Name)
            and node.func.value.id == "theme_html"
            and node.args
            and isinstance(node.args[0], ast.List)
        ):
            sizes.append(len(node.args[0].elts))
    return sizes


def test_a_legend_is_only_drawn_for_two_or_more_series():
    """Every ``theme_html.legend(...)`` call site passes at least two entries.

    "A legend for one series is furniture" - the rule this checks is a floor
    of 2, not a fixed count, so a legend that later grows a third series
    still passes.
    """
    found_any = False
    for name, module in _ALL_SWEPT_MODULES.items():
        for size in _legend_call_sizes(module):
            found_any = True
            assert size >= 2, f"{name}: legend() called with {size} entries"
    assert found_any, "expected at least one theme_html.legend() call site to check"


def test_ranked_bar_charts_never_show_a_legend_for_their_one_series():
    """theme.rank_bar and pages/code.py's _share_rank_bar are single-series
    rankings (one colour per severity band is still one *series*, not
    several) - both must set ``showlegend=False`` explicitly rather than
    leave plotly's own default (True) to draw a one-item legend nobody
    asked for."""
    figure = theme.rank_bar(pd.Series({"a": 3, "b": 2, "c": 1}), title="t", value_label="things")
    assert figure.layout.showlegend is False

    figure = code._share_rank_bar(pd.Series({"repo-a": 40.0, "repo-b": 96.0}))
    assert figure.layout.showlegend is False


# Every CSS rule in theme_html.py's component kit that draws a number - a
# hero, a KPI tile's value and delta, a decide card's count, a ranked bar's
# printed value, a table's numeric column, a rubric score, an evidence line -
# named by its selector. This is the positive list task 5A's sweep was built
# from (three of these - ".big", ".decide .n", ".tile .val"/".tile .delta" -
# were missing tabular-nums before this task; see docs/assumptions/5A.md).
_NUMERIC_HTML_SELECTORS = (
    ".vv .big",
    ".vv .decide .n",
    ".vv .tile .val",
    ".vv .tile .delta",
    ".vv .hbar .v",
    ".vv table.t td.num, .vv table.t th.num",
    ".vv .comp .cv",
    ".vv .evrow",
)


def test_tabular_nums_on_every_numeric_selector_theme_html_draws():
    """``font-variant-numeric: tabular-nums`` on every number-bearing rule.

    ``st.dataframe`` cannot be checked this way - it is drawn on a canvas no
    stylesheet or static scan can see into. theme.py's own CSS comment next
    to its (real-DOM) table rule explains this and says what the practical
    mitigation is (``st.column_config.NumberColumn``); this test covers the
    half of "HTML tables and Streamlit dataframes alike" that is actually
    HTML.
    """
    body = _source(theme_html)
    for selector in _NUMERIC_HTML_SELECTORS:
        escaped = re.escape(selector)
        match = re.search(escaped + r"\s*\{([^}]*)\}", body)
        assert match, f"no rule found for {selector!r}"
        assert "tabular-nums" in match.group(1), selector


def test_tabular_nums_on_theme_pys_real_dom_table_rule():
    """theme.py's ``[data-testid="stTable"]``/markdown-table rule, the one
    real-DOM table this dashboard's own stylesheet (rather than
    theme_html.py's) reaches."""
    match = re.search(r'\[data-testid="stTable"\] td[^{]*\{([^}]*)\}', theme._STYLE)
    assert match
    assert "tabular-nums" in match.group(1)


def _call_count(source: str, attr: str, *, on: str) -> int:
    """How many times ``on.attr(...)`` is actually *called* in ``source``.

    AST-based rather than a substring count: a docstring that mentions
    ``theme_html.hero()`` in backticks (pages/today.py's own
    ``_hero_fragment`` docstring does exactly this) contains the same
    characters a real call does, and a plain ``.count()`` cannot tell them
    apart.
    """
    import ast

    tree = ast.parse(source)
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == on
    )


def test_today_page_draws_exactly_one_live_hero():
    """pages/today.py's only *live* hero-sized number is _hero_fragment's.

    ``_decision_card``/``_render_attention_band`` also build hero-sized
    markup but are dead code: pages/today.py's own module docstring says
    tests/test_today.py pins their exact structure and the live page no
    longer calls either. Counting hero draws across the whole file would
    double-count a hero nobody sees; excluding those two functions' source
    is what pages/today.py's docstring itself says makes them unreachable.
    """
    live_source = inspect.getsource(today).replace(
        inspect.getsource(today._decision_card), ""
    ).replace(inspect.getsource(today._render_attention_band), "")
    assert _call_count(live_source, "hero", on="theme_html") == 1


def test_no_other_page_draws_a_hero():
    """Every page but Today draws zero heroes - "at most one" is satisfied
    trivially by having none, which is not a redesign mandate to add one."""
    for name, module in _PAGES.items():
        if name == "today":
            continue
        assert _call_count(_source(module), "hero", on="theme_html") == 0, name


def test_content_max_width_is_1280px_everywhere_it_is_declared():
    """theme_tokens.MAX_WIDTH, theme.CONTENT_MAX_WIDTH and theme_html's own
    ``--max-width`` all agree, and no page or shared module states a
    conflicting ``max-width`` for the main content column."""
    assert theme_tokens.MAX_WIDTH == "1280px"
    assert theme.CONTENT_MAX_WIDTH == "1280px"
    assert "--max-width: 1280px;" in theme_tokens.css_root()
    # Excludes: theme.py's own rule is an f-string reading CONTENT_MAX_WIDTH
    # (asserted equal to 1280px just above) rather than the literal text;
    # "@media (max-width: 960px)" is a responsive breakpoint, not a column
    # width; "70ch" is a prose line-length, not a column width either - the
    # only three other "max-width"s this codebase writes.
    conflicting = re.compile(r"max-width:(?!\s*(?:1280px|70ch|\{CONTENT_MAX_WIDTH\}))")
    for name, module in _ALL_SWEPT_MODULES.items():
        for match in conflicting.finditer(_source(module)):
            line_start = _source(module).rfind("\n", 0, match.start())
            line = _source(module)[line_start:match.start() + 20]
            assert "@media" in line, f"{name}: {match.group(0)}"


def test_app_injects_the_shared_styles_once_before_dispatching_to_any_page():
    """``app.py::main()`` calls ``inject_styles()`` once, before
    ``st.navigation`` picks a page - the mechanism that puts the 1280px
    ceiling (and everything else theme.py's stylesheet does) on every page
    without each of the eight page modules injecting it themselves."""
    import app as dashboard  # noqa: PLC0415 - importing the app is not free

    main_source = inspect.getsource(dashboard.main)
    assert main_source.count("inject_styles()") == 1
    assert main_source.index("inject_styles()") < main_source.index("st.navigation")


# ---------------------------------------------------------------------------
# Delta arrows: coloured by direction x goodness, never direction alone.
#
# series.delta() itself (magnitude/direction/is_good scored separately) has
# its own tests in tests/test_series.py; every Tile(delta_dir=, delta_good=)
# call site built from it is swept below for the failure mode named in the
# task brief - a green arrow on a rising stall count - plus
# pages/business.py's ``_delta_arrow`` (a different mechanism, st.metric's
# own delta_color, not a Tile), which is where that exact bug was found: two
# cost tiles (Google Cloud, OpenAI) coloured a rising cost green, and the
# "Biggest drop-off" tile coloured a loss green through "inverse" on an
# already-negative number. See docs/assumptions/5A.md.
# ---------------------------------------------------------------------------


def test_delta_arrow_colours_by_goodness_not_by_sign():
    """pages/business.py's ``_delta_arrow``: the ``higher_is_better`` flag
    decides the colour mapping, the sign of the change decides nothing on
    its own."""
    rising = "+$50.00"
    falling = "-$50.00"

    assert business._delta_arrow(None) == {}
    assert business._delta_arrow("flat") == {"delta": "flat", "delta_color": "off"}

    # A rise is good when higher_is_better - a growth tile, unchanged from
    # before this task.
    assert business._delta_arrow(rising, higher_is_better=True) == {
        "delta": rising,
        "delta_color": "normal",
    }
    # A rise is *not* good for a cost tile - the fix: the same "+$50" that
    # coloured green above must colour red here, never both from the sign
    # alone.
    assert business._delta_arrow(rising, higher_is_better=False) == {
        "delta": rising,
        "delta_color": "inverse",
    }
    assert business._delta_arrow(falling, higher_is_better=True) == {
        "delta": falling,
        "delta_color": "normal",
    }
    assert business._delta_arrow(falling, higher_is_better=False) == {
        "delta": falling,
        "delta_color": "inverse",
    }
    # The default (no higher_is_better passed) is "normal" - every existing
    # call site that relied on the old, unconditional "normal" behaviour
    # (the funnel's people/rate deltas, where a rise really is good) keeps
    # working unchanged.
    assert business._delta_arrow(rising) == {"delta": rising, "delta_color": "normal"}


def test_the_three_cost_tiles_colour_a_rising_cost_red_not_green():
    """Ads spend, Google Cloud and OpenAI: all three are cost panels, and a
    rising cost must never draw the growth-tile green.

    Regression test for the literal bug: before this task all three passed
    their raw ``_money_delta(...)`` straight to ``_delta_arrow`` with no
    ``higher_is_better``, so ``_delta_arrow``'s default ("normal") coloured
    a cost increase green. Scanned by source because building the live
    figures needs Stripe/BigQuery/OpenAI reads this suite does not have.
    """
    source = _source(business)
    for label, needle in (
        ("Ads spend", "_money_delta(spend.cost_change, money)"),
        ("Google Cloud", "_money_delta(burn.cost_change, money)\n            if burn.prev_cost and burn.comparable"),
        ("OpenAI", "_money_delta(burn.cost_change, money) if burn.prev_cost else None"),
    ):
        index = source.index(needle)
        window = source[index : index + 400]
        assert "higher_is_better=False" in window, label


def test_the_drop_off_tile_colours_a_loss_red_not_green():
    """"Biggest drop-off"'s delta is a literal negative magnitude, not a
    period-over-period change - ``delta_color="inverse"`` would flip that
    negative number to green (inverse flips red/negative to green), which
    is the same green-on-bad-news shape as the two cost tiles above."""
    source = _source(business)
    index = source.index('"Biggest drop-off"')
    window = source[index : index + 600]
    assert 'delta_color="normal"' in window
    assert 'delta_color="inverse"' not in window


def _tile_delta_calls(module):
    """Every ``theme_html.Tile(...)`` / ``Tile(...)`` call in ``module`` that
    passes a ``delta_dir`` keyword, as ``(delta_dir_expr, delta_good_expr)``
    source snippets (``delta_good_expr`` is ``None`` if the keyword is
    missing entirely)."""
    import ast

    tree = ast.parse(_source(module))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_tile = (isinstance(func, ast.Name) and func.id == "Tile") or (
            isinstance(func, ast.Attribute) and func.attr == "Tile"
        )
        if not is_tile:
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        if "delta_dir" not in kwargs:
            continue
        calls.append((ast.dump(kwargs["delta_dir"]), kwargs.get("delta_good")))
    return calls


def test_every_tile_with_a_delta_direction_also_carries_a_separate_goodness():
    """Every ``Tile(delta_dir=..., ...)`` call site also sets ``delta_good``
    from a *different* expression than ``delta_dir`` - never the same field
    read twice, which would make the arrow's colour just direction again
    under a different keyword."""
    found_any = False
    for name, module in {**_PAGES, "render_shared": render_shared}.items():
        for delta_dir_dump, delta_good_node in _tile_delta_calls(module):
            found_any = True
            assert delta_good_node is not None, f"{name}: {delta_dir_dump} has no delta_good"
            assert ast.dump(delta_good_node) != delta_dir_dump, name
    assert found_any, "expected at least one Tile(delta_dir=...) call site to check"
