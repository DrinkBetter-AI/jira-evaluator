"""Checks that theme_tokens is the one place the design's numbers live.

Two things this guards against, both already happened once: a chart palette
that drifted from the page's own HTML because they were two literal lists in
two files, and a `config.toml` that drifted from `theme.py` because nothing
compared them. Both are string/list equality checks against the mockup's own
values, not against each other's guesses.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import theme
import theme_tokens

REPO = Path(__file__).resolve().parents[1]

# The mockup's own :root values, typed in here once more so a change to
# theme_tokens.py that quietly drifts from the design file (rather than one
# that intentionally updates it) is caught by a diff against the source of
# truth, not just against theme_tokens' own idea of itself.
_MOCKUP_INK = {"1": "#111827", "2": "#475569", "3": "#64748b", "4": "#94a3b8"}
_MOCKUP_PLANE = {
    "page": "#f8fafc",
    "card": "#ffffff",
    "line": "#e5e7eb",
    "line_soft": "#eef2f7",
}
_MOCKUP_SERIES = [
    "#2563eb",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
_MOCKUP_STATUS = {
    "good": ("#15803d", "#ecfdf3"),
    "warn": ("#b45309", "#fef7ec"),
    "crit": ("#b91c1c", "#fdecec"),
    "info": ("#1d4ed8", "#eff4ff"),
}
_MOCKUP_TYPE = {"meta": 13, "label": 14, "body": 15, "lead": 17, "section": 20, "display": 32}


def test_every_token_matches_the_mockups_root_block_verbatim():
    assert theme_tokens.INK == _MOCKUP_INK
    assert theme_tokens.PLANE == _MOCKUP_PLANE
    assert theme_tokens.SERIES == _MOCKUP_SERIES
    assert theme_tokens.STATUS == _MOCKUP_STATUS
    assert theme_tokens.TYPE == _MOCKUP_TYPE
    assert theme_tokens.MAX_WIDTH == "1280px"


def test_status_bg_is_the_second_half_of_each_status_tuple():
    assert theme_tokens.STATUS_BG == {
        tone: bg for tone, (_fg, bg) in theme_tokens.STATUS.items()
    }
    assert set(theme_tokens.STATUS_BG) == {"good", "warn", "crit", "info"}


def test_a_plotly_bar_and_an_html_bar_use_the_same_hex():
    """theme.CATEGORICAL (plotly) and config.toml's chartCategoricalColors
    (Streamlit's own Plotly/Altair/Vega-Lite recolouring) must be the same
    list as theme_tokens.SERIES - not "the same colours in some order", the
    same list, since a caller may index into either by position.
    """
    assert theme.CATEGORICAL == theme_tokens.SERIES

    config_path = REPO / ".streamlit" / "config.toml"
    config = tomllib.loads(config_path.read_text())
    assert config["theme"]["chartCategoricalColors"] == theme_tokens.SERIES


def test_theme_py_has_no_literal_hex_of_its_own():
    """Every colour in theme.py traces back to theme_tokens - grepping the
    source (not the strings it builds at import time) is the point: a literal
    hex typed into a docstring or an f-string is exactly the drift this
    guards against.
    """
    source = Path(theme.__file__).read_text()
    literal_hexes = re.findall(r"#[0-9a-fA-F]{6}\b", source)
    assert literal_hexes == []


def test_css_root_emits_every_series_and_status_and_plane_token():
    css = theme_tokens.css_root()
    for index in range(1, 9):
        assert f"--s{index}: {theme_tokens.SERIES[index - 1]};" in css
    for tone, (fg, bg) in theme_tokens.STATUS.items():
        assert f"--{tone}: {fg};" in css
        assert f"--{tone}-bg: {bg};" in css
    assert f"--ink: {theme_tokens.INK['1']};" in css
    for rung in ("2", "3", "4"):
        assert f"--ink-{rung}: {theme_tokens.INK[rung]};" in css
    for name, value in theme_tokens.PLANE.items():
        variable = "line-soft" if name == "line_soft" else name
        assert f"--{variable}: {value};" in css
    assert "--max-width: 1280px;" in css


def test_the_type_ladder_is_exactly_the_six_named_sizes():
    assert theme_tokens.TYPE == {
        "meta": 13,
        "label": 14,
        "body": 15,
        "lead": 17,
        "section": 20,
        "display": 32,
    }
    sizes = list(theme_tokens.TYPE.values())
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    assert 12 not in theme_tokens.TYPE.values()
    assert "12px" not in theme_tokens.css_root()


def test_colorway_hands_back_a_fresh_list_every_call():
    first = theme_tokens.colorway()
    first.append("#000000")
    second = theme_tokens.colorway()
    assert second == theme_tokens.SERIES
    assert "#000000" not in second
    assert first is not second


def test_stage_tones_only_uses_the_four_status_names():
    assert theme_tokens.STAGE_TONES
    assert set(theme_tokens.STAGE_TONES.values()) <= set(theme_tokens.STATUS)
