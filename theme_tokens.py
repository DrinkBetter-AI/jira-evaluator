"""Design tokens for the VinoVoss dashboard: the single source of truth.

``vinovoss-dashboard-design.html``'s ``<style>`` ``:root`` block is the design
spec. Every value below is copied from it verbatim - no rounding, no "close
enough" substitute. ``theme.py`` (the plotly/Streamlit side) and
``theme_html.py`` (the mockup-HTML side) both need to draw a bar from the same
hex; before this file they read from two different palettes and drifted, which
is why a plotly chart and an HTML bar on the same page used to disagree.

Nothing here renders anything. ``css_root()`` and ``colorway()`` are the only
two functions, because a token file that also knows how to draw is a token
file a second designer will edit around instead of through.
"""

from __future__ import annotations

# --- Ink: text colour, darkest to lightest -----------------------------------
# Mirrors the mockup's own variable names once you drop the leading "--ink" and
# read "1" for the bare one: --ink is INK["1"], --ink-2 is INK["2"], etc.
INK = {
    "1": "#111827",
    "2": "#475569",
    "3": "#64748b",
    "4": "#94a3b8",
}

# --- Planes: backgrounds and hairlines ---------------------------------------
PLANE = {
    "page": "#f8fafc",
    "card": "#ffffff",
    "line": "#e5e7eb",
    "line_soft": "#eef2f7",
}

# One more background the mockup reaches for outside the four above: the
# muted pill/tag fill used by ``.chip.gray``, ``.rolechip`` and ``.evrow
# code``. It has no ``--variable`` of its own in the mockup's ``:root`` - it's
# a literal repeated three times in the stylesheet - but it is real and reused
# enough to be a token rather than a fourth invented grey. See
# docs/assumptions/1A.md.
MUTED_BG = "#f1f5f9"

# --- Series: the one categorical sequence, fixed order, never cycled --------
# "validated: dataviz six-checks, light surface" per the mockup's own comment.
# Index 0 doubles as the app's primaryColor (.streamlit/config.toml), so a
# single-series chart and the brand mark are the same blue.
SERIES = [
    "#2563eb",  # s1 - blue, the app's primary
    "#eb6834",  # s2 - orange
    "#1baf7a",  # s3 - green
    "#eda100",  # s4 - amber
    "#e87ba4",  # s5 - pink
    "#008300",  # s6 - deep green
    "#4a3aa7",  # s7 - violet
    "#e34948",  # s8 - red
]

# SERIES, addressable by the same "s1".."s8" keys css_root() writes as
# CSS variables - for a caller that needs the real hex (a plotly marker
# colour, say) rather than a `var(--sN)` reference. Exists so a Plotly bar
# and an HTML bar reading the same severity ramp (theme_html.severity_hue)
# resolve to one dict instead of each module parsing the other's "s8" key by
# hand. See docs/assumptions/5A.md.
SERIES_BY_KEY = {f"s{index}": hex_value for index, hex_value in enumerate(SERIES, start=1)}

# --- Status: (text colour, background) per tone -------------------------------
# "icon + label always, never color alone" per the mockup's own comment - these
# are read alongside a word or an icon, never as the only signal.
STATUS = {
    "good": ("#15803d", "#ecfdf3"),
    "warn": ("#b45309", "#fef7ec"),
    "crit": ("#b91c1c", "#fdecec"),
    "info": ("#1d4ed8", "#eff4ff"),
}

# The second element of each STATUS tuple, addressable on its own: a caller
# that wants "warn-bg" without unpacking a tuple and remembering which index
# is which.
STATUS_BG = {tone: bg for tone, (_fg, bg) in STATUS.items()}

# --- Type: 13 / 14 / 15 / 17 / 20 / 32, named for the job the text does -----
TYPE = {
    "meta": 13,  # chips, notes, the small print under a number
    "label": 14,  # what a number is called
    "body": 15,  # tables, captions, widget labels
    "lead": 17,  # the prose that explains the numbers
    "section": 20,  # a card's headline
    "display": 32,  # the KPI number itself
}

# One rung above the ladder, on purpose: the mockup's ``.big`` hero number
# (``vinovoss-dashboard-design.html``) is deliberately larger than any KPI
# tile so a page has exactly one number a reader's eye lands on first. Per
# the conformance sweep (docs/assumptions/5A.md), every page draws at most
# one number at this size or larger - a second one competing for the same
# glance is no hero at all.
HERO_SIZE = 48

# The mockup's ``.shell{max-width:1280px}``. The app's own code said 1560px;
# the mockup wins (docs/assumptions/1A.md).
MAX_WIDTH = "1280px"

# --- Radius / space: inferred, not declared -----------------------------------
# The mockup has no ``--radius-*``/``--space-*`` custom properties - every
# corner and gap is a literal on the rule that uses it. These two ladders are
# built from the values that actually recur (a `grep -oE` count across the
# mockup's stylesheet), named for later callers so a new card doesn't invent
# its own 11px or 13px. Not wired into ``css_root()``: inventing new
# ``--variable`` names the mockup itself never declared is a bigger claim than
# "here are the recurring numbers", so these stay plain Python constants.
# See docs/assumptions/1A.md.
RADIUS = {
    "sm": "6px",  # small chips, code tags
    "md": "8px",  # buttons, controls
    "lg": "12px",  # cards, tiles - the mockup's most common radius
    "pill": "999px",  # status chips, filter pills
}

SPACE = {
    "xs": "6px",
    "sm": "8px",
    "md": "10px",
    "lg": "14px",  # the mockup's most common gap - card grids, KPI strips
    "xl": "20px",  # card padding
}

# --- Stage tones: what app.py's invented _STAGE_COLORS should become --------
# app.py:3515 hand-picks a distinct (background, text) pair per workflow
# stage - nine invented hexes with no counterpart here or in the mockup. This
# is not that table; it is the four-tone reduction a later agent can use to
# replace it, once app.py is in scope: each stage maps to one of STATUS's four
# tones (``STATUS[STAGE_TONES[stage]]`` gives the (fg, bg) pair), rather than
# each stage owning its own hue. The mapping is a judgment call, recorded in
# docs/assumptions/1A.md: "info" for ordinary in-flight work, "warn" for the
# review/staging gates, "crit" for the one stage that is a stalled request for
# a human decision, "good" for the stage that is about to ship.
STAGE_TONES = {
    "Backlog": "info",
    "DISCUSSION NEEDED": "crit",
    "To Do": "info",
    "In Progress": "info",
    "IN DEV ENV": "info",
    "Code Review": "warn",
    "Review in Staging": "warn",
    "Review": "warn",
    "Ready for Production": "good",
}


def css_root() -> str:
    """The ``:root{...}`` block these tokens compile to, injected once.

    Variable names match the mockup's own ``:root`` where the mockup declares
    one (``--ink``, ``--s1``..``--s8``, ``--good``/``--good-bg``, ...), plus
    ``--max-width``, which the mockup only ever wrote as a literal inside
    ``.shell`` and never promoted to a variable.
    """
    lines = [
        f"--ink: {INK['1']};",
        f"--ink-2: {INK['2']};",
        f"--ink-3: {INK['3']};",
        f"--ink-4: {INK['4']};",
        f"--page: {PLANE['page']};",
        f"--card: {PLANE['card']};",
        f"--line: {PLANE['line']};",
        f"--line-soft: {PLANE['line_soft']};",
    ]
    for index, hex_value in enumerate(SERIES, start=1):
        lines.append(f"--s{index}: {hex_value};")
    for tone, (fg, bg) in STATUS.items():
        lines.append(f"--{tone}: {fg};")
        lines.append(f"--{tone}-bg: {bg};")
    lines.append(f"--t-meta: {TYPE['meta']}px;")
    lines.append(f"--t-label: {TYPE['label']}px;")
    lines.append(f"--t-body: {TYPE['body']}px;")
    lines.append(f"--t-lead: {TYPE['lead']}px;")
    lines.append(f"--t-section: {TYPE['section']}px;")
    lines.append(f"--t-display: {TYPE['display']}px;")
    lines.append(f"--max-width: {MAX_WIDTH};")
    body = "\n  ".join(lines)
    return f":root {{\n  {body}\n}}"


def colorway() -> list[str]:
    """``SERIES``, as a fresh list every call.

    Plotly's ``colorway`` is handed to ``figure.update_layout`` and some
    callers mutate the list they got back (append a colour, pop the last
    one for a legend trick) rather than copying it first. Returning the
    module-level ``SERIES`` object directly would let one chart's mutation
    change every chart drawn after it.
    """
    return list(SERIES)
