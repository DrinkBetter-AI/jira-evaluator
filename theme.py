"""Presentation helpers: the KPI header strip and its stylesheet."""

from __future__ import annotations

import html

import streamlit as st


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
</style>
"""


def inject_styles() -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)


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
