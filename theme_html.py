"""The mockup's visual components, rendered from real data.

The design spec is ``vinovoss-dashboard-design.html``. Streamlit's stock
widgets could not match it - ``st.metric`` and ``st.dataframe`` look like a
data tool, and the mockup looks like a product - so the pages that follow it
emit the mockup's own HTML: same tokens, same cards, same bars, same tables.

The trade is stated once, here: these blocks are display. They do not sort,
filter or page. Where a reader needs interaction (bulk edits, the sprint
planner, downloads), the page keeps Streamlit widgets beside them.

Everything user-derived is escaped. A ticket summary is data, never markup.
"""

from __future__ import annotations

import html as _html
from typing import Any, Sequence

import pandas as pd
import streamlit as st

# The mockup's tokens, verbatim. theme.py owns the chart palette; this owns
# the page chrome. Both trace to the same design file.
_CSS = """
<style>
.vv * { box-sizing: border-box; }
.vv { --ink:#111827; --ink-2:#475569; --ink-3:#64748b; --ink-4:#94a3b8;
  --card:#ffffff; --line:#e5e7eb; --line-soft:#eef2f7;
  --s1:#2563eb; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s8:#e34948;
  --good:#15803d; --good-bg:#ecfdf3; --warn:#b45309; --warn-bg:#fef7ec;
  --crit:#b91c1c; --crit-bg:#fdecec; --info:#1d4ed8; --info-bg:#eff4ff;
  color:var(--ink); font-size:15px; line-height:1.45; }
.vv a { color: var(--info); text-decoration: none; }
.vv a:hover { text-decoration: underline; }
.vv .card { background:var(--card); border:1px solid var(--line);
  border-radius:12px; padding:18px 20px; margin:0 0 14px; }
.vv .kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:14px; margin:0 0 14px; }
.vv .tile { background:var(--card); border:1px solid var(--line);
  border-radius:12px; padding:14px 16px 12px; }
.vv .tile .lbl { font-size:14px; color:var(--ink-3); }
.vv .tile .val { font-size:32px; font-weight:700; line-height:1.15; margin:2px 0 0; }
.vv .tile .note { font-size:13px; color:var(--ink-4); margin-top:2px; }
.vv .tile .delta { font-size:13px; font-weight:600; }
.vv .chart-title { font-size:15px; font-weight:650; margin:0 0 2px; }
.vv .chart-sub { font-size:13px; color:var(--ink-3); margin:0 0 10px; }
.vv .hbars { display:grid; gap:7px; margin-top:6px; }
.vv .hbar { display:grid; grid-template-columns:190px 1fr 52px; align-items:center;
  gap:10px; font-size:14px; }
.vv .hbar .name { color:var(--ink-2); text-align:right; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }
.vv .hbar .track { height:16px; border-radius:0 4px 4px 0; background:var(--s1);
  min-width:2px; }
.vv .hbar .v { font-variant-numeric:tabular-nums; font-weight:600; color:var(--ink-2); }
.vv table.t { width:100%; border-collapse:collapse; font-size:14px; }
.vv table.t th { font-size:13px; font-weight:600; color:var(--ink-3); text-align:left;
  padding:8px 10px; border-bottom:1px solid var(--line); }
.vv table.t td { padding:9px 10px; border-bottom:1px solid var(--line-soft);
  vertical-align:middle; }
.vv table.t td.num, .vv table.t th.num { text-align:right;
  font-variant-numeric:tabular-nums; }
.vv .chip { display:inline-flex; align-items:center; gap:5px; border-radius:999px;
  padding:2px 9px; font-size:12px; font-weight:600; }
.vv .chip.warn { background:var(--warn-bg); color:var(--warn); }
.vv .chip.crit { background:var(--crit-bg); color:var(--crit); }
.vv .chip.info { background:var(--info-bg); color:var(--info); }
.vv .chip.good { background:var(--good-bg); color:var(--good); }
.vv .chip.gray { background:#f1f5f9; color:var(--ink-2); }
.vv .crit-t { color:var(--crit); } .vv .warn-t { color:var(--warn); }
.vv .good-t { color:var(--good); } .vv .dim { color:var(--ink-4); font-size:12px; }
</style>
"""

_ACCENT_TEXT = {"danger": "crit-t", "warning": "warn-t", "good": "good-t",
                "info": "", "neutral": ""}


def _esc(value: Any) -> str:
    return _html.escape(str(value))


def css() -> None:
    """Inject the component styles once per page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def tiles(cards: Sequence[tuple[str, str, str, str]]) -> None:
    """The mockup's KPI row: (label, value, note, accent) per tile."""
    blocks = []
    for label, value, note, accent in cards:
        tone = _ACCENT_TEXT.get(accent, "")
        blocks.append(
            f'<div class="tile"><div class="lbl">{_esc(label)}</div>'
            f'<div class="val {tone}">{_esc(value)}</div>'
            f'<div class="note">{_esc(note)}</div></div>'
        )
    st.markdown(f'<div class="vv"><div class="kpis">{"".join(blocks)}</div></div>',
                unsafe_allow_html=True)


def _bar_color(value: float, severity: bool) -> str:
    if not severity:
        return "var(--s1)"
    if value >= 95:
        return "var(--s8)"
    if value >= 70:
        return "var(--s2)"
    if value >= 40:
        return "var(--s4)"
    return "var(--s3)"


def hbars(
    rows: Sequence[tuple[str, float, str]],
    *,
    title: str,
    subtitle: str = "",
    footer: str = "",
    severity: bool = False,
    suffix: str = "",
) -> None:
    """The mockup's ranked bars: (name, value, display) rows, largest first.

    ``severity=True`` colors by magnitude the way the review-coverage chart
    does; otherwise every bar is the primary series color. Never a pie here -
    but a pie of the same rows is a separate, deliberate view, not a fallback.
    """
    top = max((value for _, value, _ in rows), default=0.0) or 1.0
    bars = []
    for name, value, display in rows:
        width = max(2.0, 100.0 * value / top)
        bars.append(
            f'<div class="hbar"><div class="name">{_esc(name)}</div>'
            f'<div class="track" style="width:{width:.1f}%;'
            f'background:{_bar_color(value, severity)}"></div>'
            f'<div class="v">{_esc(display)}{suffix}</div></div>'
        )
    body = (
        f'<div class="card"><h3 class="chart-title">{_esc(title)}</h3>'
        + (f'<p class="chart-sub">{_esc(subtitle)}</p>' if subtitle else "")
        + f'<div class="hbars">{"".join(bars)}</div>'
        + (f'<p class="chart-sub" style="margin:12px 0 0">{_esc(footer)}</p>' if footer else "")
        + "</div>"
    )
    st.markdown(f'<div class="vv">{body}</div>', unsafe_allow_html=True)


def table(
    frame: pd.DataFrame,
    columns: Sequence[tuple[str, str, str]],
    *,
    title: str,
    subtitle: str = "",
    footer: str = "",
    max_rows: int = 25,
) -> None:
    """The mockup's table inside a card.

    ``columns`` is (source column, heading, kind) where kind is ``text``,
    ``num``, ``link`` (the cell holds a URL; the last path segment is the
    label), or ``strong-num``. Cells are escaped; only the link's href is
    emitted as markup, and only when it starts with http.
    """
    heads = "".join(
        f'<th class="{ "num" if kind.endswith("num") else "" }">{_esc(head)}</th>'
        for _, head, kind in columns
    )
    body_rows = []
    for _, row in frame.head(max_rows).iterrows():
        cells = []
        for source, _, kind in columns:
            raw = row.get(source, "")
            if kind == "link":
                url = str(raw or "")
                if url.startswith("http"):
                    label = _esc(url.rstrip("/").rsplit("/", 1)[-1])
                    cells.append(f'<td><a href="{_esc(url)}">#{label}</a></td>')
                else:
                    cells.append('<td><span class="dim">none</span></td>')
            elif kind == "num":
                text = "" if pd.isna(raw) else (f"{raw:.0f}" if isinstance(raw, float) else _esc(raw))
                cells.append(f'<td class="num">{text}</td>')
            elif kind == "strong-num":
                text = "" if pd.isna(raw) else (f"{raw:.0f}" if isinstance(raw, float) else _esc(raw))
                tone = ' class="num crit-t"' if str(text) == "0" else ' class="num"'
                cells.append(f"<td{tone}><b>{text}</b></td>")
            else:
                cells.append(f"<td>{_esc('' if pd.isna(raw) else raw)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    body = (
        f'<div class="card"><h3 class="chart-title">{_esc(title)}</h3>'
        + (f'<p class="chart-sub">{_esc(subtitle)}</p>' if subtitle else "")
        + f'<table class="t"><thead><tr>{heads}</tr></thead>'
        + f'<tbody>{"".join(body_rows)}</tbody></table>'
        + (f'<p class="chart-sub" style="margin:12px 0 0">{_esc(footer)}</p>' if footer else "")
        + "</div>"
    )
    st.markdown(f'<div class="vv">{body}</div>', unsafe_allow_html=True)
