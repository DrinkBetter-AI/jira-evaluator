"""Offline checks for the merchant-facing price charts and their text size.

These charts are made to be sent to a merchant, so what is worth testing is
that no wine is quietly drawn on top of another one, and that making the text
bigger did not silently repaint every chart in the dashboard.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as dashboard  # noqa: E402
import theme  # noqa: E402
import theme_tokens  # noqa: E402
import pages.business as business  # noqa: E402


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
    # Phase 8 split the dashboard across these modules, and Task 1C split it
    # further into one module per page - every one of them can draw a chart,
    # so all of them are in scope for this check.
    repo = Path(__file__).resolve().parents[1]
    source = "".join(
        (repo / name).read_text()
        for name in (
            "app.py",
            "data_layer.py",
            "page_shared.py",
            "render_shared.py",
            "pages/business.py",
            "pages/today.py",
            "pages/people.py",
            "pages/delivery.py",
            "pages/code.py",
            "pages/planning.py",
            "pages/engineering.py",
        )
    )
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
    # theme._TEMPLATE ("vinovoss") is process-global and other tests in this
    # suite call theme.chart_fonts()/inject_styles() too (any page render
    # test that draws a chart), so by the time this test runs it may already
    # be registered - and if it is, ``original`` above is that very name.
    # The old cleanup unconditionally deleted theme._TEMPLATE in `finally`
    # while restoring `pio.templates.default = original`, which - when
    # ``original`` was "vinovoss" - left the default pointing at a template
    # that had just been deleted, and every chart built by a test that ran
    # afterwards failed with "invalid value ... 'vinovoss'". Snapshotting
    # whether it existed before, and only deleting it here if this test is
    # the one that created it, is what makes this test order-independent
    # rather than merely order-independent-so-far. See docs/assumptions/5A.md.
    had_template_before = theme._TEMPLATE in pio.templates
    saved_template = pio.templates[theme._TEMPLATE] if had_template_before else None
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
        if "a-host-app" in pio.templates:
            del pio.templates["a-host-app"]
        if theme._TEMPLATE in pio.templates:
            del pio.templates[theme._TEMPLATE]
        if had_template_before:
            pio.templates[theme._TEMPLATE] = saved_template
        pio.templates.default = original


# --- Task 3F: tokens, and the two remaining pies -----------------------------


def test_business_page_has_no_literal_hex():
    """Every colour in pages/business.py traces back to theme_tokens.

    Same check as tests/test_theme_tokens.py's for theme.py itself: grepping
    the source, not a string built at import time, is the point - a literal
    hex typed into a new chart would slip past any test that only inspects
    the figures this file's own fixtures happen to build.
    """
    source = Path(business.__file__).read_text()
    literal_hexes = re.findall(r"#[0-9a-fA-F]{6}\b", source)
    assert literal_hexes == []


def test_business_page_has_no_off_ladder_font_size():
    """Any explicit font size in pages/business.py is one of the six rungs.

    13 / 14 / 15 / 17 / 20 / 32, per theme_tokens.TYPE - a size typed as a
    plain number (``font=dict(size=11)``, an inline ``font-size: 11px``)
    rather than a token would drift the first time someone "just nudges it a
    bit".
    """
    source = Path(business.__file__).read_text()
    ladder = set(theme_tokens.TYPE.values())
    for match in re.finditer(r"font[_-]size[\"' :=]+(\d+)", source, re.IGNORECASE):
        assert int(match.group(1)) in ladder, match.group(0)
    for match in re.finditer(r"font=dict\(size=(\d+)", source):
        assert int(match.group(1)) in ladder, match.group(0)


def test_business_page_draws_no_pie_charts():
    """DEVIN_PLAN section 7: both remaining pies are now ranked bars.

    A pie stops communicating past about six slices and cannot be compared
    across periods; a bar sorted by size can. Checked by source rather than
    by rendering every scenario, since a pie added back later would fail this
    the moment it is typed, not only when a particular fixture happens to hit
    that code path.
    """
    source = Path(business.__file__).read_text()
    assert "px.pie(" not in source
    assert "go.Pie(" not in source


def test_business_page_charts_take_the_token_colorway():
    """A single-series bar without its own semantic colour uses theme_tokens.

    The two charts that had no ``color=``/``color_discrete_map`` of their own
    (orders per day, funnel steps) are opted into ``theme.apply_palette``,
    which draws from ``theme_tokens.colorway()`` - checked directly here
    rather than by rendering, since ``theme.CATEGORICAL`` is itself pinned to
    ``theme_tokens.SERIES`` by tests/test_theme_tokens.py.
    """
    figure = go.Figure()
    theme.apply_palette(figure)
    assert list(figure.layout.colorway) == theme_tokens.colorway()
    source = Path(business.__file__).read_text()
    assert source.count("theme.apply_palette(") >= 2


# --- Task 3F: the Vivino tab's explicit unavailable state --------------------


def test_vivino_blocked_recognises_the_403():
    """Only a 403 reads as Vivino's known, permanent block."""
    assert business._vivino_blocked(Exception("403 Client Error: Forbidden")) is True
    assert (
        business._vivino_blocked(
            RuntimeError("Vivino refused the x listings: 403 Client Error")
        )
        is True
    )
    assert business._vivino_blocked(Exception("Connection timed out")) is False
    assert business._vivino_blocked(Exception("500 Internal Server Error")) is False


# --- Task 3F: ACTIVE_MERCHANTS resolves from env, falls back to the roster --


def test_active_merchants_env_var_wins_when_set():
    old = os.environ.get("ACTIVE_MERCHANTS")
    try:
        os.environ["ACTIVE_MERCHANTS"] = "Only Shop;Another Shop"
        assert business._active_merchant_names() == frozenset(
            {"Only Shop", "Another Shop"}
        )
    finally:
        if old is None:
            os.environ.pop("ACTIVE_MERCHANTS", None)
        else:
            os.environ["ACTIVE_MERCHANTS"] = old


def test_active_merchants_falls_back_to_the_five_merchant_default_when_unset():
    """Cloud Run carries no ACTIVE_MERCHANTS today - this is the live default."""
    old = os.environ.pop("ACTIVE_MERCHANTS", None)
    try:
        assert business._active_merchant_names() == frozenset(
            {
                "Yiannis",
                "Black Bear",
                "Capital Fine Wine",
                "TheWinesGood",
                "World of wine",
            }
        )
    finally:
        if old is not None:
            os.environ["ACTIVE_MERCHANTS"] = old


def test_little_international_is_not_active_in_either_path():
    """The vendor panel is newer than DEVIN_PLAN prereq 3, and wins."""
    old = os.environ.get("ACTIVE_MERCHANTS")
    try:
        os.environ.pop("ACTIVE_MERCHANTS", None)
        assert "Little International" not in business._active_merchant_names()
        assert "Little International Wine" not in business._active_merchant_names()
        # And explicitly setting the env var is respected even when it names
        # Little International - the code does not silently override an
        # operator's own choice, it only supplies a default.
        os.environ["ACTIVE_MERCHANTS"] = "Little International;Yiannis"
        assert business._active_merchant_names() == frozenset(
            {"Little International", "Yiannis"}
        )
    finally:
        if old is None:
            os.environ.pop("ACTIVE_MERCHANTS", None)
        else:
            os.environ["ACTIVE_MERCHANTS"] = old


# --- Task 3F: the store-metadata key fallback, wired into the page's own note


def test_store_metadata_note_names_the_one_key_when_every_row_agrees():
    rows = [
        ("Yiannis", {"status": "active"}),
        ("Black Bear", {"status": "inactive"}),
    ]
    note = business._store_metadata_note(rows)
    assert "'status'" in note
    assert "trim" in note


def test_store_metadata_note_names_each_store_when_keys_disagree():
    rows = [
        ("Yiannis", {"status": "active"}),
        ("Black Bear", {"is_active": False}),
    ]
    note = business._store_metadata_note(rows)
    assert "Yiannis=status" in note
    assert "Black Bear=is_active" in note


def test_store_metadata_note_does_not_crash_when_no_row_answers():
    rows = [("Yiannis", {"unrelated": "x"}), ("Black Bear", None)]
    note = business._store_metadata_note(rows)
    assert "none of" in note
    assert "ACTIVE_MERCHANTS" in note
    # And an empty read is exactly the same shape, not a special case that
    # raises where a populated-but-unreadable one does not.
    assert "none of" in business._store_metadata_note([])
