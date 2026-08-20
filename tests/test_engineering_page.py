"""Task 3G: the legacy /engineering page repainted with tokens, not touched.

Every section this page draws still has to work - this file is not a redesign
test - so what is pinned here is narrow: no colour or font size on this page
is typed in by hand instead of read from theme_tokens, the old per-bucket
hex map is gone in favour of theme.ACCENTS, and the new banner names and
links every one of the five pages that now carry this page's old detail.
"""

from __future__ import annotations

import re
from pathlib import Path

import theme
import theme_tokens
from pages import engineering

_SOURCE = Path(engineering.__file__).read_text()


def test_no_literal_hex_colour_in_the_page():
    """A colour typed straight into pages/engineering.py is exactly the drift
    theme_tokens.py exists to stop - see the identical check in
    tests/test_theme_tokens.py for theme.py itself."""
    assert re.findall(r"#[0-9a-fA-F]{6}\b", _SOURCE) == []


def test_no_off_ladder_font_size_in_the_page():
    """Every ``font-size`` this page writes must resolve to one of the six
    named rungs (13/14/15/17/20/32), through a token or a token var - never a
    literal pixel size invented at the point of use."""
    on_ladder = {f"{value}px" for value in theme_tokens.TYPE.values()}
    on_ladder_vars = {f"var(--t-{name})" for name in theme_tokens.TYPE}
    for match in re.finditer(r"font-size\s*:\s*([^;\"'}]+)", _SOURCE):
        value = match.group(1).strip()
        assert value in on_ladder or value in on_ladder_vars, (
            f"off-ladder font-size: {value!r}"
        )


def test_bucket_colors_are_the_shared_accent_tones_not_invented_hexes():
    """The old _BUCKET_COLORS was its own green/amber/red; it must now be
    exactly theme.ACCENTS's good/warning/danger, the same tones the KPI tiles
    use for the same meaning."""
    assert engineering._BUCKET_COLORS == {
        "Normal": theme.ACCENTS["good"],
        "High": theme.ACCENTS["warning"],
        "Urgent": theme.ACCENTS["danger"],
    }


def test_bubble_chart_dynamic_color_uses_the_shared_colorway(monkeypatch):
    """The non-aggregated bubble chart colours by an arbitrary column
    (priority, status, assignee, ...); it must draw from theme_tokens'
    colorway rather than plotly's own default qualitative sequence."""
    import pandas as pd
    import plotly.express as real_px

    captured: dict = {}
    original_scatter = real_px.scatter

    def spying_scatter(*args, **kwargs):
        captured["color_discrete_sequence"] = kwargs.get("color_discrete_sequence")
        return original_scatter(*args, **kwargs)

    monkeypatch.setattr(real_px, "scatter", spying_scatter)
    monkeypatch.setattr(theme, "plot", lambda *a, **k: None)

    df = pd.DataFrame(
        {
            "status": ["To Do", "In Progress"],
            "priority": ["High", "Low"],
            "issue_type": ["Story", "Bug"],
            "ticket_age_days": [3.0, 10.0],
            "idle_days": [1.0, 4.0],
            "key": ["AB-1", "AB-2"],
            "summary": ["a", "b"],
            "assignee": ["x", "y"],
        }
    )
    engineering._render_bubble_chart(df, color_by="priority", agg_priority=False)
    assert captured["color_discrete_sequence"] == theme_tokens.colorway()


def test_legacy_banner_names_and_links_every_new_page(monkeypatch):
    """The banner is the whole point of this task: a reader who followed an
    old /engineering link has to be told where the detail moved."""
    rendered: dict = {}
    monkeypatch.setattr(engineering.theme_html, "css", lambda: None)
    monkeypatch.setattr(
        engineering.theme_html,
        "render",
        lambda *fragments: rendered.setdefault("fragments", fragments),
    )

    engineering._render_legacy_banner()

    combined = "".join(rendered["fragments"])
    for label, path in engineering._NEW_PAGES:
        assert label in combined
        assert f'href="{path}"' in combined
    assert {label for label, _ in engineering._NEW_PAGES} == {
        "Today",
        "People",
        "Delivery",
        "Code",
        "Planning",
    }
