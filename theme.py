"""Presentation helpers: the KPI header strip and its stylesheet."""

from __future__ import annotations

import html

import plotly.io as pio
import streamlit as st


# Chart and caption text. Plotly draws at 12px and Streamlit captions at about
# 0.8rem, both of which sit beside metric numbers three times their size: the
# readers who complained were reading a screen share and a printed page, not a
# laptop a foot from their face.
CHART_FONT = 15
CHART_TITLE_FONT = 19
_TEMPLATE = "vinovoss"


ACCENTS = {
    "neutral": "#1f2937",
    "danger": "#b91c1c",
    "warning": "#b45309",
    "good": "#15803d",
    "info": "#1d4ed8",
}

_STYLE = """
<style>
.kpi-strip { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 0.25rem 0 1rem; }
.kpi-card {
  flex: 1 1 170px; padding: 0.9rem 1.1rem; border: 1px solid #e5e7eb;
  border-radius: 12px; background: #ffffff;
}
.kpi-card .kpi-label {
  font-size: 0.78rem; color: #6b7280; text-transform: none; letter-spacing: 0.01em;
}
.kpi-card .kpi-value { font-size: 2rem; font-weight: 700; line-height: 1.2; }
.kpi-card .kpi-note { font-size: 0.78rem; color: #9ca3af; }

/* Triage card: one ticket, sized to be judged at a glance. */
.triage-card {
  border: 1px solid #e5e7eb; border-radius: 14px; background: #ffffff;
  padding: 1.1rem 1.3rem; margin: 0.4rem 0 0.9rem;
}
.triage-card .triage-key { font-size: 0.85rem; font-weight: 700; color: #1d4ed8; }
.triage-card .triage-summary {
  font-size: 1.25rem; font-weight: 600; color: #111827; line-height: 1.35;
  margin: 0.15rem 0 0.7rem;
}
.triage-meta { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.6rem; }
.triage-meta span {
  background: #f3f4f6; color: #4b5563; border-radius: 6px;
  padding: 3px 9px; font-size: 0.76rem; white-space: nowrap;
}
/* The signals that argue for closing, so the eye finds them first. */
.triage-meta span.hot { background: #fdecec; color: #b91c1c; font-weight: 600; }
.triage-why { font-size: 0.85rem; color: #6b7280; font-style: italic; }

/* The prose the numbers are explained in. Streamlit sizes captions, help text
   and widget labels for a dense form; beside a 2rem metric they read as a
   footnote, and every honest qualification on this dashboard lives in one. */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
  font-size: 0.95rem; line-height: 1.5; color: #4b5563;
}
[data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] p { font-size: 0.95rem; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
  font-size: 1rem;
}
</style>
"""


def inject_styles() -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)
    chart_fonts()


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


def kpi_strip(cards: list[tuple[str, str, str, str]]) -> None:
    """Render ``(label, value, note, accent)`` cards as one horizontal strip."""
    blocks = []
    for label, value, note, accent in cards:
        color = ACCENTS.get(accent, ACCENTS["neutral"])
        blocks.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value" style="color:{color}">{html.escape(str(value))}</div>'
            f'<div class="kpi-note">{html.escape(note)}</div>'
            f"</div>"
        )
    st.markdown(f'<div class="kpi-strip">{"".join(blocks)}</div>', unsafe_allow_html=True)
