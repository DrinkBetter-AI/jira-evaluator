"""The mockup's visual components, rendered from real data.

The design spec is ``vinovoss-dashboard-design.html``. Streamlit's stock
widgets could not match it - ``st.metric`` and ``st.dataframe`` look like a
data tool, and the mockup looks like a product - so the pages that follow it
emit the mockup's own HTML: same tokens, same cards, same bars, same tables.

The trade is stated once, here: these blocks are display. They do not sort,
filter or page. Where a reader needs interaction (bulk edits, the sprint
planner, downloads), the page keeps Streamlit widgets beside them.

Everything user-derived is escaped. A ticket summary is data, never markup.

THE FROZEN CONTRACT. Every function below returns an HTML string built for
one wrapper: ``render()`` joins whatever fragments a page hands it, wraps
them in one ``<div class="vv">`` and writes that once. A component never
calls ``st.markdown`` itself in its new form - the sole exception is the
handful of legacy call shapes described below, kept only so the pages
written before this file existed keep working unmodified.

BACKWARD COMPATIBILITY. ``tiles``, ``hbars`` and ``table`` used to write
straight to Streamlit and return ``None``; ``pages/code.py``,
``pages/delivery.py`` and ``tests/test_theme_html.py`` call them that way
today, passing plain tuples instead of the ``Tile``/``Bar``/``Column``+``Cell``
types this file now defines. Rather than break those call sites, each of the
three functions inspects what it was handed:

  * ``tiles(cards)`` - if ``cards`` is empty or its first item is not a
    ``Tile`` (i.e. it is the old ``(label, value, note, accent)`` tuple),
    it runs the original 2024-shape rendering and writes it directly, same
    as before. A list of ``Tile`` runs the new rendering and returns a
    string, unwritten.
  * ``hbars(bars, *, title=..., ...)`` - the old signature required
    ``title`` as a keyword; if a caller passes it (or ``bars`` isn't a list
    of ``Bar``), the old card-with-a-title rendering runs and writes
    directly. A bare list of ``Bar`` returns just the ``.hbars`` rows,
    unwritten - the surrounding card and title are the caller's to build,
    same as every other bare-fragment function here.
  * ``table(columns, rows, ...)`` - if the first argument is a
    ``pandas.DataFrame`` (the old call shape was ``table(frame, columns,
    *, title=...)``), the old rendering runs and writes directly. A
    ``Sequence[Column]`` first argument runs the new rendering and returns
    a bare ``<table>``, unwritten.

Every one of the three also takes an explicit ``write: bool | None = None``
keyword that overrides the auto-detection in either direction, and the new
shape can be forced to write immediately with ``write=True`` if a page ever
wants that. Phase 3 pages should use the new, returning form - build
fragments, hand them all to ``render()`` once per section. See
``docs/assumptions/1B.md`` for the full reasoning and for the report-recording
mechanism folded into ``tiles``/``hbars`` at the same time.
"""

from __future__ import annotations

import html as _html
from typing import Any, NamedTuple, Sequence

import pandas as pd
import streamlit as st

import report as reporting
import theme_tokens

# ---------------------------------------------------------------------------
# Design tokens, imported rather than re-declared. theme_tokens.py is the
# single source of truth for every hex in this app; nothing below types one
# out fresh except the four chrome greys the mockup itself never promoted to
# a ``--variable`` (see the comment on ``_CHROME``, which mirrors the
# precedent theme_tokens.py sets for ``MUTED_BG`` in docs/assumptions/1A.md).
# ---------------------------------------------------------------------------

_STATUS_TONES = set(theme_tokens.STATUS)  # {"good", "warn", "crit", "info"}

# Greys the mockup's stylesheet writes as bare literals, never as a declared
# ``--variable`` - real, reused (each appears more than once in the design
# file), but not a token theme_tokens.py owns. Defined here, once, rather
# than typed again at every call site, exactly the treatment theme_tokens.py
# itself gives ``MUTED_BG``.
_CHROME = {
    "track_bg": "#e2e8f0",  # .scorebar .tr / .comp .tr base track
    "dim_track": "#cbd5e1",  # .hbar.dim .track
    "head_wash": "#f6f8fb",  # table header background
}


def _rgba(hex_color: str, alpha: float) -> str:
    """``hex_color`` as an ``rgba(...)`` string, computed rather than typed.

    The sticky header band needs a translucent wash of the page background
    (``--page`` at 92% opacity in the mockup); reading the channels out of
    the token's own hex keeps that wash tied to ``theme_tokens.PLANE`` if
    the page colour ever moves, instead of a second hex drifting from it.
    """
    value = hex_color.lstrip("#")
    r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _esc(value: Any) -> str:
    return _html.escape(str(value))


# The one value emitted as markup rather than text is a link's href, so the
# scheme is checked exactly, not by prefix: ``http`` also opens
# ``httpfoo:``-shaped values, and the rest of the codebase already spells the
# contract this way.
_LINK_SCHEMES = ("https://", "http://")


def _hue_var(hue: str) -> str:
    """A CSS colour reference for ``hue``: a token key, a raw literal, or "".

    Every colour a component draws is one of these three inputs, resolved
    the same way everywhere: ``"s3"``/``"crit"``/``"--s3"`` become
    ``var(--s3)``/``var(--crit)``; a literal already shaped like ``#rrggbb``
    or ``var(...)`` passes through; ``"gray"`` (no ``--gray`` token exists)
    falls back to ``--ink-4``, the muted ink shade the mockup itself reaches
    for whenever it wants "no particular colour". Nothing here ever reads or
    returns a bare hex - the whole point is that a chart's colour is always
    the live CSS variable, so it repaints if the token file ever does.
    """
    if not hue:
        return "var(--ink-4)"
    if hue.startswith("var(") or hue.startswith("#"):
        return hue
    key = hue[2:] if hue.startswith("--") else hue
    if key == "gray":
        return "var(--ink-4)"
    return f"var(--{key})"


_ARROW = {"up": "▲", "down": "▼", "flat": "—"}
_TONE_ICON = {"crit": "⚠", "warn": "⚠", "good": "✓", "info": "ⓘ", "gray": "•"}


def _delta_class(delta_good: bool | None) -> str:
    """Colour for a tile's delta line - goodness, never direction.

    A tile's arrow (``▲``/``▼``/``—``) always says which way the number
    moved; this says, independently, whether that move was good. The two
    are set from two different fields (``delta_dir`` and ``delta_good``) so
    a caller cannot accidentally wire them together and reproduce the
    up-is-always-green bug the mockup's own CSS comment warns against.
    """
    if delta_good is True:
        return "good-t"
    if delta_good is False:
        return "crit-t"
    return ""


# ---------------------------------------------------------------------------
# CSS. Built from theme_tokens at import time; ``.replace()`` on unique
# ``__TOKEN__`` markers rather than ``str.format``/f-string interpolation,
# because the stylesheet below is mostly literal ``{`` and ``%`` characters
# (grid templates, ``border-radius:50%``) that either substitution style
# would otherwise force escaping on every line.
# ---------------------------------------------------------------------------

_CSS_TOKENS = {
    "__RADIUS_SM__": theme_tokens.RADIUS["sm"],
    "__RADIUS_MD__": theme_tokens.RADIUS["md"],
    "__RADIUS_LG__": theme_tokens.RADIUS["lg"],
    "__RADIUS_PILL__": theme_tokens.RADIUS["pill"],
    "__SPACE_SM__": theme_tokens.SPACE["sm"],
    "__SPACE_LG__": theme_tokens.SPACE["lg"],
    "__MUTED_BG__": theme_tokens.MUTED_BG,
    "__TRACK_BG__": _CHROME["track_bg"],
    "__DIM_TRACK__": _CHROME["dim_track"],
    "__HEAD_WASH__": _CHROME["head_wash"],
    "__HEADER_WASH__": _rgba(theme_tokens.PLANE["page"], 0.92),
}

_CSS_BODY = """
.vv * { box-sizing: border-box; }
.vv { color: var(--ink); font-size: var(--t-body); line-height: 1.45; }
.vv a { color: var(--info); text-decoration: none; }
.vv a:hover { text-decoration: underline; }
.vv code { background: __MUTED_BG__; border-radius: __RADIUS_SM__; padding: 0 5px; font-size: 12px; }

.vv h2.sec { font-size: var(--t-section); font-weight: 650; margin: 30px 0 4px; }
.vv p.secnote { margin: 0 0 14px; font-size: var(--t-label); color: var(--ink-3); max-width: 70ch; }

.vv .card { background: var(--card); border: 1px solid var(--line); border-radius: __RADIUS_LG__; padding: 18px 20px; margin: 0 0 14px; }
.vv .grid { display: grid; gap: __SPACE_LG__; }

.vv header.top { position: sticky; top: 0; z-index: 9; background: __HEADER_WASH__; backdrop-filter: blur(6px); border-bottom: 1px solid var(--line); padding: 0 4px; }
.vv .brandrow { display: flex; align-items: center; gap: 14px; padding: 14px 0 10px; }
.vv .logo { width: 30px; height: 30px; border-radius: __RADIUS_MD__; background: linear-gradient(135deg, #7c1d3f, var(--s1)); display: grid; place-items: center; color: #fff; font-weight: 700; font-size: 14px; }
.vv .brand { font-size: var(--t-lead); font-weight: 650; }
.vv .fresh { margin-left: auto; display: flex; align-items: center; gap: 8px; font-size: var(--t-meta); color: var(--ink-3); }
.vv .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--good); }

.vv .hero { display: grid; grid-template-columns: minmax(300px, 1.15fr) 2fr; gap: __SPACE_LG__; margin-top: 6px; }
.vv .kicker { font-size: var(--t-meta); font-weight: 650; letter-spacing: .04em; text-transform: uppercase; display: flex; gap: 6px; align-items: center; }
.vv .big { font-size: 48px; font-weight: 700; line-height: 1.05; margin: 6px 0 2px; }
.vv .big small { font-size: var(--t-lead); font-weight: 500; color: var(--ink-3); }
.vv .meter { height: 8px; border-radius: 4px; margin: 12px 0 8px; overflow: hidden; }
.vv .meter i { display: block; height: 100%; border-radius: 4px 0 0 4px; }
.vv .sub { font-size: var(--t-label); color: var(--ink-2); }
.vv .decide { display: grid; grid-template-columns: repeat(3, 1fr); gap: __SPACE_LG__; }
.vv .decide .card { display: flex; flex-direction: column; gap: 6px; padding: 16px 18px; }
.vv .decide .n { font-size: 26px; font-weight: 700; }
.vv .decide .what { font-size: var(--t-label); font-weight: 600; }
.vv .decide .why { font-size: var(--t-meta); color: var(--ink-3); flex: 1; }
.vv .decide .act { font-size: var(--t-meta); font-weight: 600; }

.vv .chip { display: inline-flex; align-items: center; gap: 5px; border-radius: __RADIUS_PILL__; padding: 2px 9px; font-size: 12px; font-weight: 600; }
.vv .chip.warn { background: var(--warn-bg); color: var(--warn); }
.vv .chip.crit { background: var(--crit-bg); color: var(--crit); }
.vv .chip.info { background: var(--info-bg); color: var(--info); }
.vv .chip.good { background: var(--good-bg); color: var(--good); }
.vv .chip.gray { background: __MUTED_BG__; color: var(--ink-2); }
.vv .crit-t { color: var(--crit); }
.vv .warn-t { color: var(--warn); }
.vv .good-t { color: var(--good); }
.vv .dim { color: var(--ink-4); font-size: 12px; }

.vv .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: __SPACE_LG__; margin: 0 0 14px; }
.vv .tile { background: var(--card); border: 1px solid var(--line); border-radius: __RADIUS_LG__; padding: 14px 16px 12px; }
.vv .tile .lbl { font-size: var(--t-label); color: var(--ink-3); }
.vv .tile .val { font-size: var(--t-display); font-weight: 700; line-height: 1.15; margin: 2px 0 0; }
.vv .tile .row { display: flex; align-items: flex-end; justify-content: space-between; gap: 8px; }
.vv .tile .delta { font-size: var(--t-meta); font-weight: 600; }
.vv .tile .note { font-size: var(--t-meta); color: var(--ink-4); margin-top: 2px; }
.vv .spark { width: 96px; height: 30px; flex: none; }

.vv .charts2 { display: grid; grid-template-columns: 1.25fr 1fr; gap: __SPACE_LG__; }
.vv .chart-title { font-size: var(--t-body); font-weight: 650; margin: 0 0 2px; }
.vv .chart-sub { font-size: var(--t-meta); color: var(--ink-3); margin: 0 0 10px; }
.vv .legend { display: flex; gap: 16px; font-size: var(--t-meta); color: var(--ink-2); margin: 8px 0 0; }
.vv .legend .key { display: inline-flex; align-items: center; gap: 6px; }
.vv .legend .key i { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.vv svg text { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink-3); }
.vv svg .lab { font-weight: 600; fill: var(--ink-2); }

.vv .hbars { display: grid; gap: 7px; margin-top: 6px; }
.vv .hbar { display: grid; grid-template-columns: 190px 1fr 52px; align-items: center; gap: 10px; font-size: var(--t-label); }
.vv .hbar .name { color: var(--ink-2); text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.vv .hbar .track { height: 16px; border-radius: 0 4px 4px 0; background: var(--s1); min-width: 2px; }
.vv .hbar .v { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--ink-2); }
.vv .hbar.dim .track { background: __DIM_TRACK__; }
.vv .flagic { color: var(--warn); font-size: 12px; margin-left: 4px; }

.vv table.t { width: 100%; border-collapse: collapse; font-size: var(--t-label); }
.vv table.t th { font-size: var(--t-meta); font-weight: 600; color: var(--ink-3); text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); background: __HEAD_WASH__; white-space: nowrap; }
.vv table.t td { padding: 9px 10px; border-bottom: 1px solid var(--line-soft); vertical-align: middle; }
.vv table.t td.num, .vv table.t th.num { text-align: right; font-variant-numeric: tabular-nums; }
.vv table.t tbody tr:hover td { background: var(--page); }

.vv .who { display: flex; align-items: center; gap: 9px; }
.vv .av { width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center; font-size: 11px; font-weight: 700; color: #fff; flex: none; }
.vv .rolechip { font-size: 11px; font-weight: 600; color: var(--ink-3); background: __MUTED_BG__; border-radius: 5px; padding: 1px 6px; white-space: nowrap; }
.vv .scorebar { display: inline-flex; align-items: center; gap: 8px; }
.vv .scorebar .tr { width: 74px; height: 7px; border-radius: 4px; background: __TRACK_BG__; overflow: hidden; flex: none; }
.vv .scorebar .tr i { display: block; height: 100%; border-radius: 4px 0 0 4px; }
.vv .nnote { font-size: 11px; color: var(--ink-4); }

.vv .filters { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 14px; }
.vv .fchip { border: 1px solid var(--line); background: var(--card); border-radius: __RADIUS_PILL__; padding: 5px 13px; font-size: var(--t-meta); font-weight: 550; color: var(--ink-2); }
.vv .fchip.on { background: var(--s1); border-color: var(--s1); color: #fff; }

.vv .score2 { display: grid; grid-template-columns: 1.4fr 1fr; gap: __SPACE_LG__; }
.vv .comp { display: grid; grid-template-columns: 170px 1fr 52px 70px; gap: 10px; align-items: center; font-size: var(--t-label); padding: 6px 0; }
.vv .comp .cn { color: var(--ink-2); }
.vv .comp .cw { font-size: 11px; color: var(--ink-4); }
.vv .comp .tr { height: 9px; border-radius: 4px; background: __TRACK_BG__; overflow: hidden; }
.vv .comp .tr i { display: block; height: 100%; border-radius: 4px 0 0 4px; }
.vv .comp .cv { font-weight: 650; text-align: right; font-variant-numeric: tabular-nums; }
.vv .comp .cnote { font-size: 11px; color: var(--ink-4); text-align: right; }
.vv .comp.na .tr { background: repeating-linear-gradient(45deg, var(--line-soft) 0 6px, __TRACK_BG__ 6px 12px); }
.vv .comp.na .cv { color: var(--ink-4); font-weight: 500; }

.vv .intro-int { border-left: 4px solid var(--warn); }
.vv .evrow { font-size: var(--t-meta); color: var(--ink-2); font-variant-numeric: tabular-nums; padding: 3px 0; }
.vv .evrow code { background: __MUTED_BG__; border-radius: 4px; padding: 0 5px; font-size: 12px; }
.vv .innocent { font-size: var(--t-meta); color: var(--ink-4); border-top: 1px dashed var(--line); margin-top: 12px; padding-top: 9px; font-style: italic; }
.vv .intgrid { display: grid; grid-template-columns: 1fr 1fr; gap: __SPACE_LG__; }
.vv .maskbar { display: inline-block; height: 9px; border-radius: 4px; background: var(--warn); vertical-align: middle; }

.vv .stub { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: __SPACE_LG__; }
.vv .stub .card h3 { margin: 0 0 4px; font-size: var(--t-body); }
.vv .stub .card p { margin: 0; font-size: var(--t-meta); color: var(--ink-3); }

.vv .foot { margin-top: 44px; border-top: 1px solid var(--line); padding-top: 14px; font-size: var(--t-meta); color: var(--ink-4); }

@media (max-width: 960px) {
  .vv .hero, .vv .charts2, .vv .score2, .vv .intgrid { grid-template-columns: 1fr; }
  .vv .decide { grid-template-columns: 1fr; }
}
"""


def _build_css() -> str:
    vars_block = theme_tokens.css_root().replace(":root ", ".vv ", 1)
    body = _CSS_BODY
    for token, value in _CSS_TOKENS.items():
        body = body.replace(token, value)
    return f"<style>\n{vars_block}\n{body}\n</style>"


_CSS = _build_css()


def css() -> None:
    """Inject the component styles once per page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render(*fragments: str) -> None:
    """Write one or more component fragments inside a single ``.vv`` scope.

    Every function below except the legacy call shapes of ``tiles``/
    ``hbars``/``table`` (and their explicit ``write=True``) only returns a
    string; nothing draws until it passes through here. Composing several
    fragments into one ``render()`` call, rather than one call each, is why
    the CSS scope only has to be opened once per section.
    """
    st.markdown('<div class="vv">' + "".join(fragments) + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Report recording. page_shared.py's ``_tile``/``_kpis`` helpers register
# every metric they draw with the printable "Download report" - but
# pages/code.py and pages/delivery.py never call them; they call
# theme_html.tiles/hbars directly, so today Code's and Delivery's numbers
# are silently missing from that report. theme_html.py cannot import
# page_shared to fix the call sites itself (page_shared already imports
# theme_html - that would be a cycle) and pages/code.py, pages/delivery.py
# are out of this file's ownership. What is here instead: an optional
# ``tab=``/``section=`` pair on ``tiles``/``hbars`` that, when a caller
# supplies them, records into the exact ``st.session_state["tab_reports"]``
# structure page_shared._report() builds, through the standalone ``report``
# module both already depend on. "tab_reports" is the same string as
# page_shared.REPORTS_KEY, duplicated rather than imported - see
# docs/assumptions/1B.md for why, and for what breaks if that key is ever
# renamed without this literal moving with it.
# ---------------------------------------------------------------------------

_REPORTS_KEY = "tab_reports"


def _record(tab: str | None, section: str | None, entries: Sequence[tuple[str, str, str]]) -> None:
    if not tab or not section:
        return
    try:
        reports = st.session_state.setdefault(_REPORTS_KEY, {})
    except Exception:  # noqa: BLE001 - no live session state (a script, a test) costs nothing
        return
    built = reports.setdefault(tab, reporting.Report(tab))
    for label, value, note in entries:
        built.figure(section, label, str(value), note)


# ---------------------------------------------------------------------------
# NamedTuples. Every one has defaults sensible enough to build positionally
# or by keyword, per the contract.
# ---------------------------------------------------------------------------


class Tile(NamedTuple):
    label: str
    value: str
    unit: str = ""
    delta: str | None = None
    delta_dir: str | None = None  # "up" | "down" | "flat"
    delta_good: bool | None = None  # True/False/None - colour, kept apart from delta_dir's arrow
    note: str = ""
    spark: str | None = None  # a fragment already built by spark(), or None
    help: str | None = None


class Bar(NamedTuple):
    name: str
    value: str  # the text shown at the end of the bar, e.g. "104" or "88%"
    pct: float  # 0-100 track width, given explicitly - never inferred here
    tone: str = "s1"  # a theme_tokens hue key: s1..s8, good, warn, crit, info
    dim: bool = False
    flag: bool = False


class Column(NamedTuple):
    label: str
    kind: str = "text"  # text|num|link|strong-num|html|chip|avatar|scorebar
    help: str | None = None


class Cell(NamedTuple):
    """One table cell. Which fields matter depends on the column's ``kind``.

    ``value`` is the primary payload for every kind: plain text/number, a
    URL for ``"link"``, a pre-built fragment for ``"html"`` (unescaped -
    server-built fragments only, never raw Jira/GitHub text), a chip's
    label, an avatar's initials, or a scorebar's display text. ``tone``,
    ``note``, ``hue`` and ``pct`` are read only by the kinds that need them
    (``"chip"``/``"scorebar"`` read ``tone``; ``"scorebar"`` also reads
    ``note`` and ``pct``; ``"avatar"`` reads ``hue``) and are escaped the
    same as ``value`` before they reach the page, through
    ``chip()``/``avatar()``/``scorebar()`` - so a hostile string in any of
    these fields comes out as text, the same guarantee ``"text"`` cells
    give, even though the kind renders a colour and a shape rather than a
    plain cell.
    """

    value: Any = ""
    tone: str = ""
    note: str = ""
    hue: str = ""
    pct: float = 0.0


class DecideCard(NamedTuple):
    chip: str
    tone: str
    n: str
    what: str
    why: str
    action: tuple[str, str] | None = None


class Component(NamedTuple):
    """One rubric line of a scorecard.

    ``score`` is only read when ``sufficient`` is true - scorecard() never
    looks at it otherwise, so there is no argument combination that prints
    an insufficient component as a number (see scorecard()'s own docstring).
    """

    name: str
    weight: float
    score: float | None
    note: str = ""
    sufficient: bool = True


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------


def page_header(brand: str, freshness: str, sources: dict[str, bool]) -> str:
    """The sticky brand row: logo, freshness note, and a check per data source."""
    badges = " &nbsp; ".join(
        f'<span class="{"good-t" if ok else "crit-t"}">{_esc(name)} {"✓" if ok else "✗"}</span>'
        for name, ok in sources.items()
    )
    return (
        '<header class="top"><div class="brandrow">'
        '<div class="logo">V</div>'
        f'<div class="brand">{_esc(brand)}</div>'
        f'<div class="fresh"><span class="dot"></span> {_esc(freshness)}'
        + (f" &nbsp;·&nbsp; {badges}" if badges else "")
        + "</div></div></header>"
    )


def section(title: str, note: str = "") -> str:
    """A section heading: ``h2.sec`` plus an optional ``p.secnote``."""
    body = f'<h2 class="sec">{_esc(title)}</h2>'
    if note:
        body += f'<p class="secnote">{_esc(note)}</p>'
    return body


def foot(text: str) -> str:
    return f'<div class="foot">{_esc(text)}</div>'


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def hero(
    kicker: str,
    tone: str,
    big: str,
    unit: str,
    headline: str,
    meter_pct: float | None,
    sub: str,
    link: tuple[str, str] | None,
) -> str:
    """The lead attention card: one number, a meter, and what to do about it."""
    color = _hue_var(tone)
    icon = _TONE_ICON.get(tone, "•")
    meter_html = ""
    if meter_pct is not None:
        pct = max(0.0, min(100.0, float(meter_pct)))
        bg = f"var(--{tone}-bg)" if tone in _STATUS_TONES else theme_tokens.MUTED_BG
        meter_html = f'<div class="meter" style="background:{bg}"><i style="width:{pct:.0f}%;background:{color}"></i></div>'
    link_html = f' <a href="{_esc(link[0])}">{_esc(link[1])} →</a>' if link else ""
    return (
        f'<div class="card lead" style="border-left:4px solid {color}">'
        f'<div class="kicker" style="color:{color}">{icon} {_esc(kicker)}</div>'
        f'<div class="big">{_esc(big)}<small> {_esc(unit)}</small></div>'
        f'<div style="font-size:var(--t-lead);font-weight:600;margin-top:2px">{_esc(headline)}</div>'
        f"{meter_html}"
        f'<div class="sub">{_esc(sub)}{link_html}</div>'
        "</div>"
    )


def decide_cards(cards: Sequence[DecideCard]) -> str:
    """The three-up "what needs a decision today" row."""
    parts = []
    for card in cards:
        action_html = (
            f'<div class="act"><a href="{_esc(card.action[0])}">{_esc(card.action[1])} →</a></div>'
            if card.action
            else ""
        )
        parts.append(
            "<div class=\"card\">"
            f"{chip(card.chip, card.tone)}"
            f'<div class="n">{_esc(card.n)}</div>'
            f'<div class="what">{_esc(card.what)}</div>'
            f'<div class="why">{_esc(card.why)}</div>'
            f"{action_html}"
            "</div>"
        )
    return f'<div class="decide">{"".join(parts)}</div>'


def callout(tone: str, title: str, body: str) -> str:
    """A one-off banner, like the mockup's "Blocked on config" card."""
    color = _hue_var(tone if tone in _STATUS_TONES else "warn")
    icon = _TONE_ICON.get(tone, "⚠")
    label = tone.capitalize() if tone else "Note"
    return (
        f'<div class="card" style="border-left:4px solid {color}">'
        f'<div style="color:{color};font-weight:650;font-size:var(--t-meta);'
        f'text-transform:uppercase;letter-spacing:.04em">{icon} {_esc(label)}</div>'
        f'<p style="margin:8px 0 0;font-size:var(--t-lead)"><b>{_esc(title)}</b></p>'
        f'<p class="chart-sub" style="margin:4px 0 0">{_esc(body)}</p>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Stats: tiles, sparklines, the line chart, the legend
# ---------------------------------------------------------------------------

_ACCENT_TEXT = {"danger": "crit-t", "warning": "warn-t", "good": "good-t", "info": "", "neutral": ""}


def _tile_entry(card: "Tile | tuple") -> tuple[str, str, str]:
    if isinstance(card, Tile):
        return card.label, card.value, card.note
    label, value, note, _accent = card
    return label, value, note


def _legacy_tile_html(card: tuple) -> str:
    """The pre-1B rendering of one ``(label, value, note, accent)`` tile."""
    label, value, note, accent = card
    tone = _ACCENT_TEXT.get(accent, "")
    return (
        f'<div class="tile"><div class="lbl">{_esc(label)}</div>'
        f'<div class="val {tone}">{_esc(value)}</div>'
        f'<div class="note">{_esc(note)}</div></div>'
    )


def _tile_html(tile: Tile) -> str:
    val = f'<div class="val">{_esc(tile.value)}'
    if tile.unit:
        val += f'<small style="font-size:var(--t-label);font-weight:500;color:var(--ink-3)"> {_esc(tile.unit)}</small>'
    val += "</div>"
    row = f'<div class="row">{val}{tile.spark}</div>' if tile.spark else val
    delta_html = ""
    if tile.delta:
        arrow = _ARROW.get(tile.delta_dir or "flat", "—")
        cls = _delta_class(tile.delta_good)
        delta_html = f'<div class="delta {cls}">{arrow} {_esc(tile.delta)}</div>'
    note_html = f'<div class="note">{_esc(tile.note)}</div>' if tile.note else ""
    help_attr = f' title="{_esc(tile.help)}"' if tile.help else ""
    return f'<div class="tile"{help_attr}><div class="lbl">{_esc(tile.label)}</div>{row}{delta_html}{note_html}</div>'


def tiles(
    cards: Sequence[Tile] | Sequence[tuple[str, str, str, str]],
    *,
    write: bool | None = None,
    tab: str | None = None,
    section: str | None = None,
) -> str | None:
    """The KPI row. New form: ``Sequence[Tile]``, returns the ``.kpis`` markup.

    Legacy form: ``Sequence[(label, value, note, accent)]``, writes directly
    and returns ``None`` - see the module docstring for the detection rule.
    """
    cards = list(cards)
    _record(tab, section, [_tile_entry(c) for c in cards])
    is_legacy = write if write is not None else (not cards or not isinstance(cards[0], Tile))
    if is_legacy:
        body = f'<div class="kpis">{"".join(_legacy_tile_html(c) for c in cards)}</div>'
        st.markdown(f'<div class="vv">{body}</div>', unsafe_allow_html=True)
        return None
    html_out = f'<div class="kpis">{"".join(_tile_html(t) for t in cards)}</div>'
    if write:
        st.markdown(f'<div class="vv">{html_out}</div>', unsafe_allow_html=True)
        return None
    return html_out


def spark(series: Sequence[float], hue: str, fill: bool = False, w: int = 96, h: int = 30) -> str:
    """A standalone sparkline SVG - no JS, computed the same way the mockup's was.

    ``st.markdown`` strips ``<script>``, so the polyline this used to draw in
    the browser is drawn here in Python instead: same padding, same min/max
    scaling, same trailing dot. A caller embeds the return value inside a
    ``Tile(spark=...)`` or wherever else a chart belongs.
    """
    pad = 3
    color = _hue_var(hue)
    values = [float(v) for v in series]
    n = len(values)
    if n == 0:
        return f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}"></svg>'
    mn, mx = min(values), max(values)
    span = (mx - mn) or 1.0

    def x(i: int) -> float:
        return pad + ((i * (w - 2 * pad) / (n - 1)) if n > 1 else (w - 2 * pad) / 2)

    def y(v: float) -> float:
        return h - pad - ((v - mn) / span) * (h - 2 * pad)

    pts = " ".join(f"{x(i):.2f},{y(v):.2f}" for i, v in enumerate(values))
    inner = ""
    if fill:
        inner += f'<polygon points="{pad},{h - pad} {pts} {w - pad},{h - pad}" fill="{color}" opacity="0.1"/>'
    inner += f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
    lx, ly = x(n - 1), y(values[-1])
    inner += f'<circle cx="{lx:.2f}" cy="{ly:.2f}" r="3.5" fill="{color}" stroke="#fff" stroke-width="2"/>'
    return f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}">{inner}</svg>'


def _nice_ceiling(value: float) -> float:
    """The smallest "round" number at or above ``value``, for gridlines.

    ``value`` is always >= 0 here (chart data is counts/rates). 0 stays 0 -
    the caller decides what an all-zero chart's axis means, not this.
    """
    if value <= 0:
        return 0.0
    import math

    magnitude = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 2.5, 5, 10):
        candidate = step * magnitude
        if candidate >= value:
            return float(candidate)
    return float(10 * magnitude)


def linechart(
    series: dict[str, Sequence[float]],
    x_labels: Sequence[str],
    hues: Sequence[str],
    w: int = 620,
    h: int = 210,
    aria: str = "",
) -> str:
    """A standalone multi-series line chart SVG - no JS.

    Handles a single series, series of differing length (each series is
    plotted against its own point count, so nothing has to line up), an
    all-zero series (the axis collapses to 0 and the line stays flat rather
    than dividing by zero), and an empty series dict or an empty list for one
    series (that series, or the whole chart, draws as an empty labelled
    frame). Point tooltips are a native ``<title>`` child on each dot - the
    mockup's hover layer needed JS this file cannot ship.
    """
    left, right, top, bottom = 40, 20, 14, 28
    plot_w = max(w - left - right, 1)
    plot_h = max(h - top - bottom, 1)
    all_values = [float(v) for values in series.values() for v in values]
    raw_max = max(all_values) if all_values else 0.0
    domain_max = _nice_ceiling(raw_max)

    def y(v: float) -> float:
        if domain_max <= 0:
            return top + plot_h
        return top + (1 - v / domain_max) * plot_h

    parts: list[str] = []
    grid_steps = 4
    grid_vals = [domain_max * i / grid_steps for i in range(grid_steps + 1)] if domain_max > 0 else [0.0]
    for gv in grid_vals:
        gy = y(gv)
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{w - right}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left - 8}" y="{gy + 4:.1f}" text-anchor="end">{gv:.0f}</text>')

    n_labels = len(x_labels)
    if n_labels:
        span = max(n_labels - 1, 1)
        tick_idx = sorted({0, n_labels // 4, n_labels // 2, (3 * n_labels) // 4, n_labels - 1})
        for i in tick_idx:
            tx = left + i * plot_w / span
            parts.append(f'<text x="{tx:.1f}" y="{h - 8}" text-anchor="middle">{_esc(x_labels[i])}</text>')

    for i, (name, values) in enumerate(series.items()):
        if not values:
            continue
        hue = hues[i % len(hues)] if hues else "s1"
        color = _hue_var(hue)
        n = len(values)

        def xf(idx: int, n: int = n) -> float:
            return left + ((idx * plot_w / (n - 1)) if n > 1 else plot_w / 2)

        pts = " ".join(f"{xf(idx):.2f},{y(float(v)):.2f}" for idx, v in enumerate(values))
        if n > 1:
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )
        for idx, v in enumerate(values):
            label_x = _esc(x_labels[idx]) if idx < n_labels else str(idx)
            title = f"{label_x}: {v:g} {_esc(name)}"
            parts.append(
                f'<circle cx="{xf(idx):.2f}" cy="{y(float(v)):.2f}" r="3" fill="{color}" '
                f'stroke="#fff" stroke-width="2"><title>{title}</title></circle>'
            )

    body = "".join(parts)
    aria_attr = f' aria-label="{_esc(aria)}"' if aria else ""
    return f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img"{aria_attr}>{body}</svg>'


def legend(keys: Sequence[tuple[str, str]]) -> str:
    """A row of colour-swatch labels: ``(label, hue)`` pairs, in order."""
    items = "".join(
        f'<span class="key"><i style="background:{_hue_var(hue)}"></i> {_esc(label)}</span>' for label, hue in keys
    )
    return f'<div class="legend">{items}</div>'


# ---------------------------------------------------------------------------
# Ranked bars
# ---------------------------------------------------------------------------


def _bar_color(value: float, severity: bool) -> str:
    """The legacy severity ramp, unchanged: only used by the old call shape."""
    if not severity:
        return "var(--s1)"
    if value >= 95:
        return "var(--s8)"
    if value >= 70:
        return "var(--s2)"
    if value >= 40:
        return "var(--s4)"
    return "var(--s3)"


def _bar_entry(bar: "Bar | tuple") -> tuple[str, str, str]:
    if isinstance(bar, Bar):
        return bar.name, bar.value, ""
    name, _value, display = bar
    return name, display, ""


def _bars_html(bars: Sequence[Bar]) -> str:
    parts = []
    for bar in bars:
        cls = "hbar dim" if bar.dim else "hbar"
        style = f"width:{bar.pct:.1f}%"
        if not bar.dim:
            style += f";background:{_hue_var(bar.tone)}"
        flag_html = ' <span class="flagic" title="attention status"> ⚠</span>' if bar.flag else ""
        parts.append(
            f'<div class="{cls}"><div class="name">{_esc(bar.name)}{flag_html}</div>'
            f'<div class="track" style="{style}"></div>'
            f'<div class="v">{_esc(bar.value)}</div></div>'
        )
    return f'<div class="hbars">{"".join(parts)}</div>'


def _legacy_hbars_html(
    rows: Sequence[tuple[str, float, str]],
    *,
    title: str,
    subtitle: str,
    footer: str,
    severity: bool,
    suffix: str,
) -> str:
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
    return (
        f'<div class="card"><h3 class="chart-title">{_esc(title)}</h3>'
        + (f'<p class="chart-sub">{_esc(subtitle)}</p>' if subtitle else "")
        + f'<div class="hbars">{"".join(bars)}</div>'
        + (f'<p class="chart-sub" style="margin:12px 0 0">{_esc(footer)}</p>' if footer else "")
        + "</div>"
    )


def hbars(
    bars: Sequence[Bar] | Sequence[tuple[str, float, str]],
    *,
    title: str | None = None,
    subtitle: str = "",
    footer: str = "",
    severity: bool = False,
    suffix: str = "",
    write: bool | None = None,
    tab: str | None = None,
    section: str | None = None,
) -> str | None:
    """The ranked-bars grid. New form: ``Sequence[Bar]``, returns ``.hbars``.

    Legacy form: ``(name, value, display)`` rows plus a required ``title``
    keyword, writes a whole card directly and returns ``None`` - see the
    module docstring for the detection rule. Never a pie here - but a pie of
    the same rows is a separate, deliberate view, not a fallback.
    """
    bars = list(bars)
    _record(tab, section, [_bar_entry(b) for b in bars])
    is_legacy = write if write is not None else (title is not None or not bars or not isinstance(bars[0], Bar))
    if is_legacy:
        html_out = _legacy_hbars_html(
            bars, title=title or "", subtitle=subtitle, footer=footer, severity=severity, suffix=suffix
        )
        st.markdown(f'<div class="vv">{html_out}</div>', unsafe_allow_html=True)
        return None
    html_out = _bars_html(bars)
    if write:
        st.markdown(f'<div class="vv">{html_out}</div>', unsafe_allow_html=True)
        return None
    return html_out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _fmt_num(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.0f}"
    return _esc(value)


def _render_cell(kind: str, cell: Cell) -> str:
    if kind == "link":
        url = str(cell.value or "")
        if url.startswith(_LINK_SCHEMES):
            label = _esc(url.rstrip("/").rsplit("/", 1)[-1])
            return f'<td><a href="{_esc(url)}">#{label}</a></td>'
        return '<td><span class="dim">none</span></td>'
    if kind == "num":
        return f'<td class="num">{_fmt_num(cell.value)}</td>'
    if kind == "strong-num":
        text = _fmt_num(cell.value)
        tone = ' class="num crit-t"' if text == "0" else ' class="num"'
        return f"<td{tone}><b>{text}</b></td>"
    if kind == "html":
        return f"<td>{cell.value}</td>"
    if kind == "chip":
        return f"<td>{chip(str(cell.value), cell.tone)}</td>"
    if kind == "avatar":
        return f"<td>{avatar(str(cell.value), cell.hue)}</td>"
    if kind == "scorebar":
        return f"<td>{scorebar(cell.pct, cell.tone, str(cell.value), cell.note)}</td>"
    # "text" and any other kind fall back to plain, escaped text.
    return f"<td>{_fmt_text(cell.value)}</td>"


def _fmt_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return _esc(value)


def _table_html(columns: Sequence[Column], rows: Sequence[Sequence[Cell]]) -> str:
    heads = []
    for column in columns:
        cls = ' class="num"' if column.kind.endswith("num") else ""
        title_attr = f' title="{_esc(column.help)}"' if column.help else ""
        heads.append(f"<th{cls}{title_attr}>{_esc(column.label)}</th>")
    body_rows = []
    for row in rows:
        cells = [_render_cell(column.kind, cell) for column, cell in zip(columns, row)]
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="t"><thead><tr>{"".join(heads)}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def _legacy_table_html(
    frame: pd.DataFrame,
    columns: Sequence[tuple[str, str, str]],
    *,
    title: str,
    subtitle: str,
    footer: str,
    max_rows: int,
) -> str:
    heads = "".join(
        f'<th class="{"num" if kind.endswith("num") else ""}">{_esc(head)}</th>' for _, head, kind in columns
    )
    body_rows = []
    for _, row in frame.head(max_rows).iterrows():
        cells = []
        for source, _, kind in columns:
            raw = row.get(source, "")
            if kind == "link":
                url = str(raw or "")
                if url.startswith(_LINK_SCHEMES):
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
    return (
        f'<div class="card"><h3 class="chart-title">{_esc(title)}</h3>'
        + (f'<p class="chart-sub">{_esc(subtitle)}</p>' if subtitle else "")
        + f'<table class="t"><thead><tr>{heads}</tr></thead>'
        + f'<tbody>{"".join(body_rows)}</tbody></table>'
        + (f'<p class="chart-sub" style="margin:12px 0 0">{_esc(footer)}</p>' if footer else "")
        + "</div>"
    )


def table(
    columns: Sequence[Column] | pd.DataFrame,
    rows: Sequence[Sequence[Cell]] | Sequence[tuple[str, str, str]] | None = None,
    *,
    title: str | None = None,
    subtitle: str = "",
    footer: str = "",
    max_rows: int = 25,
    write: bool | None = None,
) -> str | None:
    """The card-free ``<table class="t">``. New form: ``(columns, rows)``.

    ``kind`` on a ``Column`` is one of ``text``, ``num``, ``link``,
    ``strong-num``, ``html``, ``chip``, ``avatar`` or ``scorebar``. Every
    kind except ``html`` escapes what it is given; ``html`` is the one
    place a cell's ``value`` is emitted verbatim, so it exists only for
    fragments this file itself built (``chip()``, ``avatar()``, another
    ``table()`` cell, ...) - never for a Jira summary or a GitHub title.

    Legacy form: ``table(frame, columns, *, title=...)`` where ``frame`` is
    a ``pandas.DataFrame`` and ``columns`` is ``(source, heading, kind)``
    triples - writes a whole card directly and returns ``None``. Rows past
    ``max_rows`` are not drawn in that form either way - these blocks
    display, they do not page - so a caller with a longer frame owes the
    reader a ``footer`` saying how much was cut.
    """
    if isinstance(columns, pd.DataFrame):
        legacy_columns = rows if rows is not None else []
        html_out = _legacy_table_html(
            columns, legacy_columns, title=title or "", subtitle=subtitle, footer=footer, max_rows=max_rows
        )
        if write is False:
            return html_out
        st.markdown(f'<div class="vv">{html_out}</div>', unsafe_allow_html=True)
        return None
    html_out = _table_html(columns, rows or [])
    if write:
        st.markdown(f'<div class="vv">{html_out}</div>', unsafe_allow_html=True)
        return None
    return html_out


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------


def avatar(initials: str, hue: str) -> str:
    return f'<span class="av" style="background:{_hue_var(hue)}">{_esc(initials)}</span>'


def rolechip(label: str) -> str:
    return f'<span class="rolechip">{_esc(label)}</span>'


def scorebar(pct: float, tone: str, value: str, note: str = "") -> str:
    out = (
        '<span class="scorebar"><span class="tr">'
        f'<i style="width:{max(0.0, min(100.0, float(pct))):.0f}%;background:{_hue_var(tone)}"></i>'
        f"</span><b>{_esc(value)}</b></span>"
    )
    if note:
        out += f' <span class="nnote">{_esc(note)}</span>'
    return out


def chip(label: str, tone: str) -> str:
    """A status pill: ``tone`` in warn|crit|info|good|gray.

    The label is always shown beside the colour - that pairing is the
    accessibility contract every tone in this file honours: colour never
    carries meaning alone.
    """
    return f'<span class="chip {_esc(tone)}">{_esc(label)}</span>'


def scorecard(components: Sequence[Component], overall: str, measurable: str, note: str) -> str:
    """The per-person rubric card.

    Every ``Component`` with ``sufficient=False`` renders the diagonal-hatch
    track and the literal string ``"n/a"`` - its ``score`` field is never
    read in that branch, so there is no argument combination, including a
    numeric ``score`` on an insufficient component, that prints a number for
    that row. This is enforced here, once, rather than left to callers to
    remember.
    """
    rows = []
    for component in components:
        if component.sufficient:
            pct = max(0.0, min(100.0, float(component.score if component.score is not None else 0.0)))
            rows.append(
                '<div class="comp"><div>'
                f'<div class="cn">{_esc(component.name)}</div>'
                f'<div class="cw">weight {_esc(component.weight)}</div></div>'
                f'<div class="tr"><i style="width:{pct:.0f}%;background:var(--s1)"></i></div>'
                f'<div class="cv">{pct:.0f}</div>'
                f'<div class="cnote">{_esc(component.note)}</div></div>'
            )
        else:
            rows.append(
                '<div class="comp na"><div>'
                f'<div class="cn">{_esc(component.name)}</div>'
                f'<div class="cw">weight {_esc(component.weight)}</div></div>'
                '<div class="tr"></div>'
                '<div class="cv">n/a</div>'
                f'<div class="cnote">{_esc(component.note)}</div></div>'
            )
    summary = (
        f'<p class="secnote" style="margin-top:12px">Overall <b>{_esc(overall)}</b> over '
        f"{_esc(measurable)} measurable points. {_esc(note)}</p>"
    )
    return f'<div class="card">{"".join(rows)}{summary}</div>'


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def intro_band(text: str) -> str:
    return (
        '<div class="card intro-int"><b>Visible to you only.</b> '
        f'<span style="color:var(--ink-2)">{_esc(text)}</span></div>'
    )


def evrow(html_fragment: str) -> str:
    """One evidence line. ``html_fragment`` is emitted verbatim.

    Build it from escaped pieces plus literal ``<code>``/``<b>`` tags, the
    same way this file's own helpers do - never hand this a raw Jira/GitHub
    string.
    """
    return f'<div class="evrow">{html_fragment}</div>'


def innocent(text: str) -> str:
    return f'<div class="innocent">{_esc(text)}</div>'


def maskbar(days: int, scale: float) -> str:
    width = max(0.0, float(days) * float(scale))
    return f'<span class="maskbar" style="width:{width:.1f}px"></span>'


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def sprint_card(board: str, accent: str, name: str, window: str, rows: Sequence[tuple[str, str, str]]) -> str:
    """One board's sprint summary card. ``rows`` is ``(label, value, note)``.

    The last row is drawn without its bottom border, matching the mockup's
    "People" row. A board with no dates set (the mockup's ML sprint) is a
    separate ``callout()`` beside this card, not a special case inside it -
    ``window`` here is always plain text.
    """
    color = _hue_var(accent)
    rows = list(rows)
    trs = []
    for index, (label, value, note) in enumerate(rows):
        border = "border:0" if index == len(rows) - 1 else ""
        note_html = f' <span class="nnote">{_esc(note)}</span>' if note else ""
        trs.append(
            f'<tr><td style="padding-left:0;{border}">{_esc(label)}</td>'
            f'<td style="text-align:right;padding-right:0;{border}"><b>{_esc(value)}</b>{note_html}</td></tr>'
        )
    table_html = f'<table style="margin-top:10px">{"".join(trs)}</table>'
    return (
        f'<div class="card" style="border-left:4px solid {color}">'
        '<div class="kicker" style="font-size:var(--t-meta);font-weight:650;'
        f'letter-spacing:.04em;color:{color};text-transform:uppercase">{_esc(board)}</div>'
        f'<p style="margin:6px 0 2px;font-size:var(--t-lead);font-weight:650">{_esc(name)}</p>'
        f'<p class="chart-sub" style="margin:0">{_esc(window)}</p>'
        f"{table_html}</div>"
    )


def stub_cards(cards: Sequence[tuple[str, str]]) -> str:
    """The Business tab's "unchanged this phase" placeholder grid."""
    items = "".join(f'<div class="card"><h3>{_esc(title)}</h3><p>{_esc(body)}</p></div>' for title, body in cards)
    return f'<div class="stub">{items}</div>'
