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
