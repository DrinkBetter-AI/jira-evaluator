"""Presentation helpers: the type scale, the palette, the KPI strip, the charts.

Everything a reader complains about that is not a number lives here. The
complaints this file answers are, in order: the text is too small to read on a
shared screen, the colours mean nothing because there are five palettes, and a
pie of twenty-three slices is a colour wheel rather than a chart.
"""

from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import theme_tokens


# Chart and caption text. Plotly draws at 12px and Streamlit captions at about
# 0.8rem, both of which sit beside metric numbers three times their size: the
# readers who complained were reading a screen share and a printed page, not a
# laptop a foot from their face.
CHART_FONT = 15
CHART_TITLE_FONT = 19
_TEMPLATE = "vinovoss"


# --- The type scale ----------------------------------------------------------
#
# One ladder, six rungs, rather than the 0.76 / 0.78 / 0.85 / 0.95 / 1 / 1.25 /
# 2rem that grew here a rule at a time. The rungs are named for the job a piece
# of text does, so a new rule picks a role instead of inventing a size, and
# nothing on this dashboard ends up half a pixel away from something else.
#
# They are declared in pixels on purpose. `.streamlit/config.toml` raises the
# app's root font size to 17px so that Streamlit's own dataframe - which is
# drawn on a canvas and cannot be reached by any stylesheet - lands at 15px
# instead of 14px. Sizes written in rem would have been scaled by that same
# 17/16 and quietly drifted off the ladder; these do not move.
#
# The numbers themselves live in theme_tokens.TYPE - this is that same ladder
# with "px" back on, since every caller here interpolates these straight into
# a CSS string or an f-string style attribute.
TYPE_META = f"{theme_tokens.TYPE['meta']}px"  # chips, notes, the small print under a number
TYPE_LABEL = f"{theme_tokens.TYPE['label']}px"  # what a number is called
TYPE_BODY = f"{theme_tokens.TYPE['body']}px"  # tables, captions, widget labels - most of this dashboard
TYPE_LEAD = f"{theme_tokens.TYPE['lead']}px"  # the prose that explains the numbers
TYPE_SECTION = f"{theme_tokens.TYPE['section']}px"  # a card's headline
TYPE_DISPLAY = f"{theme_tokens.TYPE['display']}px"  # the KPI number itself
TYPE_HERO = f"{theme_tokens.HERO_SIZE}px"  # the one number a page leads with - see theme_tokens.HERO_SIZE

# The widest the main column is allowed to get. `layout="wide"` with no ceiling
# turns a six-card KPI strip into a thin ribbon on a 34-inch monitor and drags
# the eye across two feet of table, so the content is centred in a column about
# as wide as a reader can track a row across.
#
# This used to be a separate, larger number (1560px) than the mockup's own
# `.shell{max-width:1280px}`. The mockup is the design spec; 1280px wins. See
# docs/assumptions/1A.md.
CONTENT_MAX_WIDTH = theme_tokens.MAX_WIDTH


# --- Colour ------------------------------------------------------------------

# Semantic, not categorical: these five say whether a number is good or bad
# and are used by the Streamlit-native tiles alone (pages/today.py's decision
# cards, page_shared._tile's st.metric calls) - never by a chart, or "green"
# stops meaning "healthy" the moment a series happens to be third in a legend.
#
# Four of the five are theme_tokens.STATUS's text colours; "neutral" has no
# status tone of its own (there is no such thing as a neutral status pill) and
# is set to the mockup's own primary ink instead - a neutral accent is just
# text with no meaning attached. See docs/assumptions/1A.md.
ACCENTS = {
    "neutral": theme_tokens.INK["1"],
    "danger": theme_tokens.STATUS["crit"][0],
    "warning": theme_tokens.STATUS["warn"][0],
    "good": theme_tokens.STATUS["good"][0],
    "info": theme_tokens.STATUS["info"][0],
}

# The one categorical sequence. Charts that need to tell series apart take these
# in order rather than each inventing their own set - the dashboard had five
# palettes, so the same status was three colours depending on which chart a
# reader was looking at.
#
# This is theme_tokens.SERIES - the mockup's own "series (validated: dataviz
# six-checks, light surface)" sequence - rather than a palette invented for
# charts alone, so a plotly bar and an HTML bar drawn from the same data are
# the same hex. It replaces an earlier Okabe-Ito-derived set that predated the
# mockup and did not match it. A list, not a tuple, so it compares equal to
# theme_tokens.SERIES; copied rather than aliased so mutating one does not
# mutate the other.
#
# Eight is the ceiling and it is already generous. No eight-colour set is fully
# separable in greyscale, which is the second reason `rank_bar` collapses a long
# tail rather than drawing it.
CATEGORICAL = list(theme_tokens.SERIES)


def categorical(count: int) -> list[str]:
    """The first ``count`` palette colours, repeating only if forced to.

    Repeating is a chart telling on itself: two series the same colour means the
    chart has more categories than a reader can hold, and the fix is to collapse
    the tail rather than to add a ninth hue nobody can name.
    """
    if count <= 0:
        return []
    return [CATEGORICAL[index % len(CATEGORICAL)] for index in range(count)]


def apply_palette(figure):
    """Draw this figure's series from the app's one categorical sequence.

    Opt-in rather than automatic: a chart whose colours carry their own meaning -
    a red/amber/green priority bucket, a status pill - would be made worse by
    having that meaning overwritten with "first colour, second colour".
    """
    figure.update_layout(colorway=list(CATEGORICAL))
    return figure


_STYLE = f"""
<style>
/* The readable column. `layout="wide"` is still right - the tables need the
   room - but "as wide as the window" is not a width, it is the absence of one. */
[data-testid="stMainBlockContainer"], [data-testid="stMain"] .block-container {{
  max-width: {CONTENT_MAX_WIDTH};
  margin-left: auto;
  margin-right: auto;
}}
/* The sidebar is deliberately untouched: it is a column of controls, not prose,
   and it is already as narrow as it should be. */

/* ".kpi-strip"/".kpi-card" used to live here: theme.kpi_strip()'s own CSS,
   a second KPI-tile system next to theme_html.py's ".kpis"/".tile". Deleted
   along with the function - see its own comment and docs/assumptions/5A.md. */

/* The triage card (render_shared._render_triage_card) used to have its own
   card shell and its own two-tone chip set here (".triage-card"/
   ".triage-meta"): a third invented chip vocabulary and a second card system
   next to theme_html.py's ".card"/".chip". Both are gone - that function now
   draws its card and its chips from theme_html's own CSS classes and design
   tokens directly (still without importing the module - see that function's
   own comment and docs/assumptions/5A.md for why it stays that way). */

/* The prose the numbers are explained in. Streamlit sizes captions, help text
   and widget labels for a dense form; beside a 32px metric they read as a
   footnote, and every honest qualification on this dashboard lives in one. */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
  font-size: {TYPE_BODY}; line-height: 1.5; color: {theme_tokens.INK['2']};
}}
[data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] p {{ font-size: {TYPE_BODY}; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
  font-size: {TYPE_LEAD};
}}

/* Tables. This dashboard is mostly tables, and they were the smallest text on
   it: Streamlit sizes a dataframe cell at 0.875rem while body copy sat at 1rem,
   so the numbers a reader came for were the hardest thing on the page to read.
   `st.dataframe` is drawn on a canvas and takes its size from the root font
   size rather than from any stylesheet, which is why the ceiling is raised in
   .streamlit/config.toml instead of here; these rules cover the tables that are
   real DOM - `st.table` and the ones written in markdown - so all three kinds
   agree, and give every header row the weight that separates a column name from
   the data under it. */
[data-testid="stTable"] td, [data-testid="stMarkdownContainer"] table td {{
  font-size: {TYPE_BODY}; font-variant-numeric: tabular-nums;
}}
[data-testid="stTable"] th, [data-testid="stMarkdownContainer"] table th {{
  font-size: {TYPE_BODY}; font-weight: 600;
}}
</style>
"""
# Every numeric column reads in tabular figures, "everywhere - HTML tables and
# Streamlit dataframes alike" (docs/assumptions/5A.md's conformance rule).
# ``st.dataframe`` cannot be reached from here - the comment above explains
# why: it is drawn on a canvas, and CSS (this file's whole mechanism) cannot
# touch canvas text, `font-variant-numeric` included. What a page *can* do for
# a canvas-drawn table is give every numeric column an explicit
# ``st.column_config.NumberColumn`` - the app's dataframes overwhelmingly do
# (grepped, not asserted here: this file draws no dataframes of its own to
# check). The CSS rule above is therefore the whole of what theme.py can
# enforce; tests/test_theme_visual.py tests this stylesheet, not a canvas.


def inject_styles() -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)
    chart_fonts()


def _titled(figure) -> bool:
    """Whether this figure has a title a reader would actually see.

    Read rather than written: touching ``figure.layout.title`` does not create
    the key, so asking the question is free where answering it wrongly is not.
    """
    title = getattr(figure.layout, "title", None)
    return bool(title is not None and str(title.text or "").strip())


def plot(figure, into=None, **kwargs):
    """Draw a plotly figure at this dashboard's text size.

    The size has to be set on the figure's own layout rather than left to the
    template: Streamlit's chart component merges its own font sizes into the
    template it finds, so a template alone is overwritten in the browser and
    only a chart that states its size keeps it.

    The title size is only stated when there is a title to size, and that is not
    tidiness. Plotly's magic underscore expands ``title_font=`` into
    ``layout.title.font.size``, which brings a ``layout.title`` object into
    existence with no ``text`` in it; Streamlit's own theming then does
    ``if ("title" in layout) layout.title.text = "<b>" + String(layout.title.text) + "</b>"``,
    and ``String(undefined)`` is the word "undefined". That is the literal
    "undefined" heading readers have been seeing above two charts in production.

    ``into`` is a column or other container to draw in, since a chart written as
    ``left.plotly_chart(...)`` would otherwise have to skip this and keep the
    small text.
    """
    figure.update_layout(font=dict(size=CHART_FONT))
    if _titled(figure):
        figure.update_layout(title_font=dict(size=CHART_TITLE_FONT))
    return (into or st).plotly_chart(figure, **kwargs)


def _current_template():
    """The template plotly is drawing with, whatever form the default is in.

    ``pio.templates.default`` is a name, a ``Template``, or names joined by
    ``+``; only the first form can be looked up, so the others are handled here
    rather than falling back to plotly's plain template and losing the app's
    own chart colours.
    """
    default = pio.templates.default
    if default is None:
        return pio.templates["plotly"]
    if not isinstance(default, str):
        return default
    for name in default.split("+"):
        if name in pio.templates:
            return pio.templates[name]
    return pio.templates["plotly"]


def chart_fonts() -> None:
    """Make every chart in the app draw its text at a size a room can read.

    Set as a plotly template rather than per figure, so a chart added later is
    legible without anyone remembering this.
    """
    if _TEMPLATE not in pio.templates:
        # Built on whatever template is in force rather than plotly's plain one:
        # Streamlit installs its own and makes it the default, and its frontend
        # recolours a figure that carries it. Copying "plotly" instead would
        # change every chart's colours to buy a larger font.
        larger = _current_template().to_plotly_json()
        layout = larger.setdefault("layout", {})
        layout["font"] = {**layout.get("font", {}), "size": CHART_FONT}
        layout["title"] = {
            **layout.get("title", {}),
            "font": {"size": CHART_TITLE_FONT},
        }
        pio.templates[_TEMPLATE] = larger
    pio.templates.default = _TEMPLATE


# --- Ranked bars, which is what a pie should have been ------------------------

# How many rows a ranked bar draws before the rest become one "Other" bar. Ten
# is about as far down a list as anyone reads, and the row that matters is
# always at the top.
RANK_ROWS = 10

# The conformance ceiling every bar on the dashboard is held to (docs/
# assumptions/5A.md): no bar taller than 24px, 4px rounded data-ends rather
# than a squared-off box. Public, not private - any page building its own
# `go.Bar` (`pages/code.py`'s `_share_rank_bar` is the other one) draws from
# these same two numbers instead of typing its own bargap/margin arithmetic
# and drifting off the ceiling, which is exactly what that page's own bar
# chart had done before this task (34px slot, 0.28 bargap - a 24.5px bar).
BAR_MAX_HEIGHT = 24
BAR_CORNER_RADIUS = 4
BAR_GAP = 0.25  # the fraction of each row's plot-area slot left empty between bars
# `BAR_ROW_HEIGHT` is the plot-area slot given to each category; chosen so
# the bar itself - `BAR_ROW_HEIGHT * (1 - BAR_GAP)` - lands exactly on
# `BAR_MAX_HEIGHT` regardless of row count, since plotly's `bargap` is a
# fraction of the slot, not an absolute pixel gap. A caller's own figure
# height is then `BAR_ROW_HEIGHT * n_rows + <that figure's own t+b margin>` -
# see `rank_bar` below for the reference computation, and
# `pages/code.py::_share_rank_bar` for a second figure using the same two
# numbers with its own (smaller) margin.
BAR_ROW_HEIGHT = BAR_MAX_HEIGHT / (1 - BAR_GAP)
_MARGIN_V = 96  # this function's own top+bottom margin (t=56, b=40 below)


def bar_gap_for(n_rows: int, plot_height: float) -> float:
    """The ``bargap`` that keeps every one of ``n_rows`` bars at or under
    ``BAR_MAX_HEIGHT``, given the plot area's actual pixel height.

    ``BAR_GAP`` alone only lands on the ceiling when the figure's own height
    was built from exactly ``BAR_ROW_HEIGHT * n_rows`` - true for a chart
    with enough rows, false the moment a chart's height floor (every
    ranked-bar chart here has one, so a two-row chart is not drawn as a
    sliver) makes each row's slot bigger than that. plotly's ``bargap`` is a
    *fraction* of the slot, not a pixel value, so the fraction has to grow to
    compensate - solved for here rather than left at a fixed constant, which
    is what let a one-row chart draw a 108px bar before this existed. Never
    returns less than ``BAR_GAP``, so a chart with enough rows still gets the
    normal, deliberate spacing rather than bars touching edge to edge.
    """
    if n_rows <= 0 or plot_height <= 0:
        return BAR_GAP
    row_slot = plot_height / n_rows
    return max(BAR_GAP, 1 - (BAR_MAX_HEIGHT / row_slot))

_OTHER = "Other"
# The label a collapsed tail carries, recognised again on the way back in so
# that ranking an already-ranked series is not a second collapse.
_OTHER_PATTERN = re.compile(rf"^{_OTHER} \((\d+)\)$")


def _other_size(label: object) -> int | None:
    """How many things a row is standing in for, if it is a collapsed tail."""
    match = _OTHER_PATTERN.match(str(label))
    return int(match.group(1)) if match else None


def ranked(series: pd.Series, *, top_n: int = RANK_ROWS) -> pd.Series:
    """The largest ``top_n`` entries, with everything below them added up.

    The result is never longer than ``top_n``, and the collapsed tail is always
    the last row rather than wherever its size would put it: a tail that outgrows
    the top named entry - which happens the moment a long tail is long enough to
    be worth collapsing - would otherwise rank first and read as the busiest
    person on the team.

    It sums to exactly what it was given, so a chart and the table beside it
    cannot disagree, and running it on its own output changes nothing, which is
    what lets the figure and its table be built from one call.
    """
    # ``fillna(0)`` rather than ``fillna(0.0)``, and the sums left in whatever
    # type they arrive as: these are usually ticket counts, and a table beside
    # the chart showing "9.0000" tickets is a table that looks broken.
    given = pd.Series(series)
    counted = pd.to_numeric(given, errors="coerce")
    # A ranking is drawn from counts indexed by category. Handed the categories
    # themselves - a per-ticket ``status`` column rather than its
    # ``value_counts()`` - every value coerces to nothing, and the chart drew ten
    # zero-length bars labelled 0, 1, 2 ... off the row numbers. Silently reading
    # that as "everything is zero" is the failure; say so instead.
    if len(given) and counted.isna().all():
        raise TypeError(
            "ranked() wants values indexed by category, e.g. series.value_counts(); "
            "it was given labels it cannot count"
        )
    counted = counted.fillna(0)
    carried_labels = [label for label in counted.index if _other_size(label) is not None]
    collapsed = sum(_other_size(label) or 0 for label in carried_labels)
    collapsed_value = counted[carried_labels].sum() if carried_labels else 0
    named = counted.drop(labels=carried_labels).sort_values(ascending=False)

    room = top_n - (1 if carried_labels else 0) if top_n > 0 else len(named)
    if top_n > 0 and len(named) > room:
        tail = named.iloc[top_n - 1 :]
        collapsed += len(tail)
        collapsed_value = collapsed_value + tail.sum()
        named = named.iloc[: top_n - 1]

    if not collapsed:
        return named
    return pd.concat(
        [named, pd.Series({f"{_OTHER} ({collapsed})": collapsed_value})]
    )


def rank_bar(
    series: pd.Series,
    *,
    title: str,
    value_label: str,
    top_n: int = RANK_ROWS,
):
    """A "who did how much" ranking, drawn as horizontal bars rather than a pie.

    A pie stops communicating at about six categories. Past that the slices are
    slivers, the labels are drawn on top of one another or dropped, the legend
    turns into a scrolling list, and a reader is left comparing angles - which is
    the one comparison people are measurably bad at. These charts had
    twenty-three slices each. A bar chart sorted by size answers the question a
    pie was being asked to answer ("who is at the top, and by how much") in the
    order the eye already reads.

    Horizontal, because the categories are people's names and status labels:
    vertical bars would have to rotate them, and a rotated label is one a reader
    tilts their head to read. Every name here stays level. The value is printed
    at the end of its own bar, so nobody has to measure anything against an axis.

    The height grows with the number of rows rather than being fixed, so ten
    people are not squeezed into the space three were comfortable in.
    """
    rows = ranked(series, top_n=top_n)
    frame = rows.rename_axis("category").reset_index(name="value")
    frame["category"] = frame["category"].astype(str)
    n = len(frame)

    # Plotly stacks the first category at the bottom of a horizontal bar chart,
    # so the order is reversed to put the largest at the top where a ranking
    # belongs.
    frame = frame.iloc[::-1]

    # One colour for the named rows, because they are one dimension and colouring
    # them individually would imply a distinction that is not there. The
    # collapsed tail is drawn in the palette's last colour so it is visually
    # set apart from the named rows rather than reading as somebody called
    # "Other". Under the old Okabe-Ito-derived CATEGORICAL that last colour was
    # a neutral slate; under theme_tokens.SERIES it is s8, a red - which now
    # reads as a warning rather than as leftovers. Flagged, not fixed here: the
    # test pinning this to CATEGORICAL[-1] (tests/test_theme_visual.py) is not
    # a file this task owns. See docs/assumptions/1A.md.
    colors = [
        CATEGORICAL[-1] if _other_size(name) is not None else CATEGORICAL[0]
        for name in frame["category"]
    ]

    figure = go.Figure(
        go.Bar(
            x=frame["value"],
            y=frame["category"],
            orientation="h",
            marker=dict(color=colors, cornerradius=BAR_CORNER_RADIUS),
            text=[f"{value:,.0f}" for value in frame["value"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:,.0f} " + value_label + "<extra></extra>",
        )
    )
    height = max(240, BAR_ROW_HEIGHT * n + _MARGIN_V)
    figure.update_layout(
        title=dict(text=title),
        height=height,
        margin=dict(t=56, b=40, l=8, r=48),
        showlegend=False,
        bargap=bar_gap_for(n, height - _MARGIN_V),
    )
    figure.update_xaxes(title_text=value_label, rangemode="tozero")
    # `tickangle=0` is the whole point of the chart: never rotated, whatever the
    # length of the name. `automargin` buys the labels the room they need
    # instead of truncating them.
    figure.update_yaxes(title_text="", tickangle=0, automargin=True, type="category")
    return figure


# `kpi_strip()` used to live here: a second KPI-tile implementation next to
# theme_html.tiles() - same card, same grid, told apart from it only by
# which helper a page happened to call. Both of its call sites
# (pages/engineering.py, render_shared.py's sprint tiles) now draw through
# theme_html.tiles()/Tile instead; see docs/assumptions/5A.md. `ACCENTS`
# above is kept - it is still how theme.py's other Streamlit-native tiles
# (pages/today.py's decision cards, page_shared._tile's st.metric calls)
# colour a value, and deleting it would have deleted those along with it.
