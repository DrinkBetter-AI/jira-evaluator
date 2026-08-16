"""Offline checks for the merchant-facing price charts and their text size.

These charts are made to be sent to a merchant, so what is worth testing is
that no wine is quietly drawn on top of another one, and that making the text
bigger did not silently repaint every chart in the dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as dashboard  # noqa: E402
import theme  # noqa: E402


def test_two_vintages_of_one_wine_get_two_rows():
    """A row is a plotly category, and two identical ones are drawn as one."""
    long_name = "Caymus Vineyards Cabernet Sauvignon Napa Valley"
    labels = dashboard._distinct_labels(
        pd.Series([f"{long_name} 2019", f"{long_name} 2020", "Le Jade Picpoul", None])
    )
    assert len(set(labels)) == 4, list(labels)
    assert list(labels)[:2] == [long_name[:46], f"{long_name[:46]} (2)"]


def test_a_line_is_only_fitted_where_there_are_dots_to_fit_it_through():
    assert dashboard._least_squares(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])) is None
    # Every wine priced the same has no slope, only a vertical stack.
    assert (
        dashboard._least_squares(pd.Series([1.0] * 5), pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        is None
    )
    ends, values = dashboard._least_squares(
        pd.Series([0.0, 10.0, 20.0]), pd.Series([10.0, 5.0, 0.0])
    )
    assert ends == [0.0, 20.0]
    assert values == [10.0, 0.0]


def test_every_chart_states_its_own_text_size():
    """A template alone is merged over by Streamlit's own font sizes.

    So the size has to be on the figure, and every chart in the app has to go
    through the one helper that puts it there - the readability complaint was
    about the whole dashboard, not the two charts added beside the complaint.
    """
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    # Any receiver, not only ``st``: a chart written as ``left.plotly_chart``
    # into a column would slip past a check for the one spelling.
    assert ".plotly_chart(" not in source, "a chart drawn around theme.plot()"
    assert source.count("theme.plot(") >= 11

    drawn = {}
    figure = go.Figure()
    theme.st = type(
        "capture", (), {"plotly_chart": staticmethod(lambda fig, **kw: drawn.update(kw))}
    )
    try:
        theme.plot(figure, width="stretch", key="anything")
    finally:
        theme.st = st_module
    assert figure.layout.font.size == theme.CHART_FONT
    # A title size is set on a chart that has a title, and only then. Setting it
    # unconditionally used to leave a ``title`` object carrying a font and no
    # text, which Streamlit's frontend rewrote into the literal word
    # "undefined" above the Ticket Composition chart in production.
    assert figure.layout.title.text is None
    titled = go.Figure()
    titled.update_layout(title=dict(text="Open tickets by status"))
    theme.st = type(
        "capture", (), {"plotly_chart": staticmethod(lambda fig, **kw: None)}
    )
    try:
        theme.plot(titled, width="stretch")
    finally:
        theme.st = st_module
    assert titled.layout.title.font.size == theme.CHART_TITLE_FONT
    # And the caller's own arguments still reach Streamlit untouched.
    assert drawn == {"width": "stretch", "key": "anything"}

    # A named container is drawn into rather than passed on as a chart argument.
    column = []
    theme.plot(
        go.Figure(),
        into=type(
            "column",
            (),
            {"plotly_chart": staticmethod(lambda fig, **kw: column.append(fig))},
        ),
        width="stretch",
    )
    assert len(column) == 1


def test_bigger_text_keeps_the_colours_the_app_already_drew_with():
    """The font is the change; a repainted dashboard would be a side effect."""
    original = pio.templates.default
    try:
        pio.templates["a-host-app"] = {
            "layout": {"paper_bgcolor": "#123456", "font": {"size": 11}}
        }
        pio.templates.default = "a-host-app"
        if theme._TEMPLATE in pio.templates:
            del pio.templates[theme._TEMPLATE]
        theme.chart_fonts()
        built = pio.templates[theme._TEMPLATE]
        assert built.layout.paper_bgcolor == "#123456"
        assert built.layout.font.size == theme.CHART_FONT
        assert built.layout.title.font.size == theme.CHART_TITLE_FONT
        # A default of several templates joined by "+" is still a lookup that
        # has to work, rather than a silent fall back to plotly's own colours.
        # The default moves off the template before it is removed, which some
        # plotly versions refuse.
        pio.templates.default = "a-host-app+plotly"
        del pio.templates[theme._TEMPLATE]
        theme.chart_fonts()
        assert pio.templates[theme._TEMPLATE].layout.paper_bgcolor == "#123456"
    finally:
        pio.templates.default = original
        for name in ("a-host-app", theme._TEMPLATE):
            if name in pio.templates:
                del pio.templates[name]
        pio.templates.default = original
