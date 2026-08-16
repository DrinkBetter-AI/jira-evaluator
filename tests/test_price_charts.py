"""Offline checks for the merchant-facing price charts and their text size.

These charts are made to be sent to a merchant, so what is worth testing is
that no wine is quietly drawn on top of another one, and that making the text
bigger did not silently repaint every chart in the dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.io as pio

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
        del pio.templates[theme._TEMPLATE]
        pio.templates.default = "a-host-app+plotly"
        theme.chart_fonts()
        assert pio.templates[theme._TEMPLATE].layout.paper_bgcolor == "#123456"
    finally:
        for name in ("a-host-app", theme._TEMPLATE):
            if name in pio.templates:
                del pio.templates[name]
        pio.templates.default = original
