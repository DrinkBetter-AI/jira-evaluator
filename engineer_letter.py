"""One engineer's page, for the engineer rather than for the manager.

The dashboard is behind a password and reads live from Jira, so it cannot be
handed to the person it is about: they get told their numbers second-hand, in a
one-on-one, a week later. This is the same page as a file - their hour totals,
their score, their badges, and their board coloured by how much attention each
ticket needs, with every key a link into Jira so the reply to "update your
tickets" is one click rather than a search.

It carries its own stylesheet and fetches nothing, so it can be sent in Slack,
opened on a phone, and printed to a PDF without anything being installed.
"""

from __future__ import annotations

import datetime as _dt
import html
import re
from dataclasses import dataclass, field

# The board's four attention tiers, in the colours the dashboard draws them in -
# the whole point of sending the page is that the red rows are visible before a
# word is read. Checked against the dashboard's own map by the tests, so a tier
# recoloured there cannot quietly go grey here.
#
# These are theme_tokens.STATUS_BG's crit/warn/good backgrounds and
# theme_tokens.MUTED_BG, copied as literals rather than imported: this module
# is deliberately dependency-free (see the module docstring - it has to
# survive being pasted into Slack with nothing installed), so the values
# travel by hand and the test named above is what keeps them from drifting
# apart, the same trade render_shared.py already makes for its own
# duplicated bot-login list. Before task 5A these four hexes were a fifth,
# unrelated palette (Bootstrap's default alert colours) that matched none of
# theme_tokens' own tones; see docs/assumptions/5A.md.
TIER_COLOURS = {
    "Needs attention": "#fdecec",  # theme_tokens.STATUS_BG["crit"]
    "Watch": "#fef7ec",  # theme_tokens.STATUS_BG["warn"]
    "Healthy": "#ecfdf3",  # theme_tokens.STATUS_BG["good"]
    "Low priority": "#f1f5f9",  # theme_tokens.MUTED_BG
}

# A ticket untouched for longer than this is asked about by name at the top of
# the page. A week, because the ask is a weekly update, so anything older has
# already missed one.
STALE_UPDATE_DAYS = 7

# Enough of the board to be useful without becoming a spreadsheet nobody reads.
_MAX_ASKS = 8


@dataclass(frozen=True)
class Ticket:
    """One row of the board, as the page needs it."""

    key: str
    url: str
    summary: str
    status: str
    priority: str
    tier: str
    sprint: str
    idle_days: float | None = None
    estimate_hours: float | None = None
    devin: str = ""


@dataclass(frozen=True)
class Tile:
    """One number at the top of the page."""

    label: str
    value: str
    note: str = ""


@dataclass
class Page:
    """Everything the page shows about one person."""

    person: str
    tiles: list[Tile] = field(default_factory=list)
    tickets: list[Ticket] = field(default_factory=list)
    score: str = ""
    score_note: str = ""
    badges: list[str] = field(default_factory=list)


def _hours(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}"


def _days(value: float | None) -> str:
    return "" if value is None else f"{value:.0f}"


def needs_updating(tickets: list[Ticket], days: int = STALE_UPDATE_DAYS) -> list[Ticket]:
    """The tickets to ask about by name, oldest silence first.

    Named rather than counted: "you have four stale tickets" is a statistic and
    "MB-8834 has not moved in 22 days" is a thing to do this afternoon.
    """
    stale = [
        ticket
        for ticket in tickets
        if ticket.idle_days is not None and ticket.idle_days >= days
    ]
    stale.sort(key=lambda ticket: -(ticket.idle_days or 0.0))
    return stale[:_MAX_ASKS]


def _link(ticket: Ticket) -> str:
    key = html.escape(ticket.key)
    if not ticket.url:
        return key
    return f'<a href="{html.escape(ticket.url, quote=True)}">{key}</a>'


def _rows(tickets: list[Ticket]) -> str:
    cells = []
    for ticket in tickets:
        colour = TIER_COLOURS.get(ticket.tier, "#ffffff")
        cells.append(
            f'<tr style="background:{colour}">'
            f"<td class=\"key\">{_link(ticket)}</td>"
            f"<td>{html.escape(ticket.summary)}</td>"
            f"<td>{html.escape(ticket.status)}</td>"
            f"<td>{html.escape(ticket.priority)}</td>"
            f'<td class="num">{_days(ticket.idle_days)}</td>'
            f'<td class="num">{_hours(ticket.estimate_hours)}</td>'
            f"<td>{html.escape(ticket.sprint)}</td>"
            f"<td>{html.escape(ticket.devin)}</td>"
            "</tr>"
        )
    return "".join(cells)


def _legend() -> str:
    swatches = "".join(
        f'<span class="swatch"><i style="background:{colour}"></i>{html.escape(name)}</span>'
        for name, colour in TIER_COLOURS.items()
    )
    return f'<div class="legend">{swatches}</div>'


def _asks(page: Page) -> str:
    stale = needs_updating(page.tickets)
    if not stale:
        return (
            '<section><h2>This week</h2><p class="good">Every ticket here has '
            f"been touched in the last {STALE_UPDATE_DAYS} days. Keep it that "
            "way.</p></section>"
        )
    items = "".join(
        f"<li>{_link(ticket)} — {html.escape(ticket.summary)} "
        f'<span class="quiet">({_days(ticket.idle_days)} days without an '
        f"update, {html.escape(ticket.status)})</span></li>"
        for ticket in stale
    )
    return (
        "<section><h2>This week</h2><p>Please leave a comment on each of these, "
        "or move it: where it stands, what is blocking it, and when you expect "
        f"it done. Nothing has changed on them in {STALE_UPDATE_DAYS} days or "
        f"more.</p><ol>{items}</ol></section>"
    )


def _tiles(page: Page) -> str:
    if not page.tiles:
        return ""
    cells = "".join(
        f'<div class="tile"><div class="label">{html.escape(tile.label)}</div>'
        f'<div class="value">{html.escape(tile.value)}</div>'
        + (f'<div class="note">{html.escape(tile.note)}</div>' if tile.note else "")
        + "</div>"
        for tile in page.tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _score(page: Page) -> str:
    if not page.score and not page.badges:
        return ""
    badges = "".join(f'<span class="badge">{html.escape(b)}</span>' for b in page.badges)
    scored = (
        f'<div class="tile"><div class="label">Scorecard</div>'
        f'<div class="value">{html.escape(page.score)}</div>'
        + (
            f'<div class="note">{html.escape(page.score_note)}</div>'
            if page.score_note
            else ""
        )
        + "</div>"
        if page.score
        else ""
    )
    return (
        "<section><h2>Score and badges</h2>"
        + (f'<div class="tiles">{scored}</div>' if scored else "")
        + (f'<div class="badges">{badges}</div>' if badges else "")
        + "</section>"
    )


def filename(person: str, *, now: _dt.date | None = None) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", str(person).lower()).strip("-") or "engineer"
    return f"{stem}-{(now or _dt.date.today()).isoformat()}.html"


def one_pager(page: Page, *, now: _dt.date | None = None) -> str:
    """The whole page as one file, ready to send."""
    when = (now or _dt.date.today()).isoformat()
    board = (
        "<section><h2>Your open tickets</h2>"
        + _legend()
        + '<table><thead><tr><th>Key</th><th>Summary</th><th>Status</th>'
        "<th>Priority</th><th>Idle</th><th>Est. h</th><th>Sprint</th>"
        "<th>Devin-able?</th></tr></thead>"
        f"<tbody>{_rows(page.tickets)}</tbody></table></section>"
        if page.tickets
        else '<section><h2>Your open tickets</h2><p class="quiet">No open '
        "tickets.</p></section>"
    )
    return _PAGE.format(
        person=html.escape(str(page.person)),
        when=when,
        tiles=_tiles(page),
        asks=_asks(page),
        score=_score(page),
        board=board,
        count=len(page.tickets),
    )


# One file, no external anything: it has to open the same from a phone on a
# train, which is where a message from a manager gets read.
_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{person} - your tickets, {when}</title>
<style>
  @page {{ size: A4 landscape; margin: 12mm; }}
  body {{
    font: 11pt/1.45 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #111827; margin: 0 auto; padding: 24px; max-width: 1100px;
  }}
  header {{ border-bottom: 2px solid #111827; padding-bottom: 8px; margin-bottom: 18px; }}
  h1 {{ font-size: 21pt; margin: 0; }}
  header p {{ color: #6b7280; margin: 4px 0 0; font-size: 10pt; }}
  section {{ margin-bottom: 20px; break-inside: avoid; }}
  h2 {{ font-size: 13pt; margin: 0 0 10px; color: #1f2937; }}
  .tiles {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }}
  .tile {{
    border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px 14px; min-width: 150px;
  }}
  .label {{ font-size: 9pt; text-transform: uppercase; letter-spacing: .04em; color: #6b7280; }}
  .value {{ font-size: 17pt; font-weight: 650; }}
  .note, .quiet {{ font-size: 9pt; color: #6b7280; }}
  .good {{ color: #15803d; font-weight: 600; }}
  .badges {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .badge {{
    background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 999px;
    padding: 4px 12px; font-size: 10pt;
  }}
  table {{ border-collapse: collapse; width: 100%; font-size: 9.5pt; }}
  th, td {{ border: 1px solid #d1d5db; padding: 5px 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f3f4f6; font-size: 9pt; text-transform: uppercase; letter-spacing: .03em; }}
  td.num {{ text-align: right; white-space: nowrap; }}
  td.key {{ white-space: nowrap; font-weight: 600; }}
  a {{ color: #1d4ed8; }}
  .legend {{ margin-bottom: 8px; font-size: 9pt; color: #4b5563; }}
  .swatch {{ display: inline-block; margin-right: 14px; }}
  .swatch i {{
    display: inline-block; width: 11px; height: 11px; border: 1px solid #9ca3af;
    margin-right: 5px; vertical-align: middle;
  }}
  ol {{ margin: 0; padding-left: 20px; }}
  li {{ margin-bottom: 5px; }}
  @media print {{ body {{ padding: 0; }} a {{ color: #111827; }} }}
</style></head>
<body>
<header>
  <h1>{person}</h1>
  <p>VinoVoss &middot; {count} open ticket(s) &middot; {when}</p>
</header>
{tiles}
{asks}
{score}
{board}
</body></html>
"""

__all__ = ["Page", "Ticket", "Tile", "TIER_COLOURS", "filename", "needs_updating", "one_pager"]
