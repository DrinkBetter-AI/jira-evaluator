"""A one-page picture of a shop's prices, for the shop rather than for us.

The panel argues in tables, which is the wrong shape for the argument: the
merchants being asked to drop a price are wine people, not analysts, and a
column called ``per_100_clicks`` reads as a demand rather than a case. The same
two figures as a coloured pie and a row of bars make it in one look - this much
of your range costs more than everyone else's, and here is what each slice of it
actually sold.

The page carries its own SVG and its own stylesheet and fetches nothing, so it
can be mailed to a merchant as an attachment, opened on a phone with no signal,
and printed to a PDF without anything being installed.
"""

from __future__ import annotations

import datetime as _dt
import html
import math
import re
from dataclasses import dataclass

# Green through to red, in the order the bands are given. Colour is the whole
# point of the page: a merchant who reads nothing else sees how much of the
# ring is red.
BAND_COLOURS = ("#15803d", "#65a30d", "#ea580c", "#b91c1c")
_GREY = "#9ca3af"

_RADIUS = 108
_CENTRE = 120


@dataclass(frozen=True)
class Band:
    """One price band as the merchant's page needs it."""

    name: str
    listings: int
    clicks: int
    bottles: int
    # Bottles per hundred clicks, or None where nobody clicked: a band Google
    # never showed has no rate, and a zero would read as one shoppers refused.
    per_100_clicks: float | None = None


def _colour(index: int) -> str:
    return BAND_COLOURS[index] if index < len(BAND_COLOURS) else _GREY


def _arc(start: float, sweep: float) -> str:
    """One pie slice as a path, from a start angle and a sweep, in radians."""
    if sweep >= 2 * math.pi - 1e-9:
        # A single band holding everything is a circle; an arc of exactly 360
        # degrees draws as nothing at all.
        return (
            f"M {_CENTRE} {_CENTRE - _RADIUS} "
            f"a {_RADIUS} {_RADIUS} 0 1 1 -0.1 0 Z"
        )
    x1 = _CENTRE + _RADIUS * math.sin(start)
    y1 = _CENTRE - _RADIUS * math.cos(start)
    x2 = _CENTRE + _RADIUS * math.sin(start + sweep)
    y2 = _CENTRE - _RADIUS * math.cos(start + sweep)
    large = 1 if sweep > math.pi else 0
    return (
        f"M {_CENTRE} {_CENTRE} L {x1:.2f} {y1:.2f} "
        f"A {_RADIUS} {_RADIUS} 0 {large} 1 {x2:.2f} {y2:.2f} Z"
    )


def pie_svg(bands: list[Band]) -> str:
    """The compared wines split by how they are priced, as a coloured ring.

    Not the whole catalogue: Google publishes a benchmark only where enough
    other shops sell the same bottle, and the rest never reach this page. The
    footer says so, because a merchant reading percentages of their range will
    otherwise take them for percentages of all of it.
    """
    total = sum(band.listings for band in bands)
    if total <= 0:
        return ""
    slices = []
    start = 0.0
    for index, band in enumerate(bands):
        if band.listings <= 0:
            continue
        sweep = 2 * math.pi * band.listings / total
        slices.append(
            f'<path d="{_arc(start, sweep)}" fill="{_colour(index)}" '
            f'stroke="#ffffff" stroke-width="2"><title>'
            f"{html.escape(band.name)}: {band.listings:,} wines</title></path>"
        )
        start += sweep
    return (
        f'<svg viewBox="0 0 {2 * _CENTRE} {2 * _CENTRE}" width="240" height="240" '
        'role="img" aria-label="Your wines by price against the market">'
        + "".join(slices)
        # The hole makes it the ring the dashboard draws, and keeps four wedges
        # from meeting in a point that reads as a fifth colour.
        + f'<circle cx="{_CENTRE}" cy="{_CENTRE}" r="{_RADIUS * 0.4:.0f}" '
        'fill="#ffffff"></circle>'
        + "</svg>"
    )


def bars_svg(bands: list[Band]) -> str:
    """What each band sold per hundred shoppers, as bars of the same colours."""
    rated = [
        (index, band)
        for index, band in enumerate(bands)
        if band.per_100_clicks is not None
    ]
    if not rated:
        return ""
    top = max(band.per_100_clicks or 0.0 for _, band in rated) or 1.0
    height, row, pad = 30, 44, 8
    width = 420
    label_width = 190
    rows = []
    for slot, (index, band) in enumerate(rated):
        rate = band.per_100_clicks or 0.0
        length = max(2.0, (width - label_width - 46) * rate / top)
        y = slot * row
        rows.append(
            f'<text x="0" y="{y + 20}" class="bar-label">'
            f"{html.escape(band.name)}</text>"
            f'<rect x="{label_width}" y="{y + pad - 2}" width="{length:.1f}" '
            f'height="{height - 10}" rx="3" fill="{_colour(index)}"></rect>'
            f'<text x="{label_width + length + 8:.1f}" y="{y + 20}" '
            f'class="bar-value">{rate:.0f}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {len(rated) * row}" width="100%" '
        f'height="{len(rated) * row}" role="img" '
        'aria-label="Bottles sold per 100 shoppers, by price band">'
        + "".join(rows)
        + "</svg>"
    )


def _legend(bands: list[Band]) -> str:
    total = sum(band.listings for band in bands) or 1
    items = []
    for index, band in enumerate(bands):
        share = band.listings / total
        items.append(
            f'<li><span class="swatch" style="background:{_colour(index)}"></span>'
            f"<strong>{band.listings:,}</strong> wines "
            f"({share:.0%}) {html.escape(band.name.lower())}</li>"
        )
    return f'<ul class="legend">{"".join(items)}</ul>'


def headline(bands: list[Band]) -> str:
    """The sentence the page is for, or an empty string if it cannot be made."""
    if len(bands) < 2:
        return ""
    cheap, dear = bands[0], bands[-1]
    if not cheap.per_100_clicks or not dear.per_100_clicks:
        return ""
    return (
        f"Your wines priced below the market sold {cheap.per_100_clicks:.0f} "
        f"bottles for every 100 shoppers who looked at them. Your wines priced "
        f"well above it sold {dear.per_100_clicks:.0f}."
    )


def filename(merchant: str, *, now: _dt.date | None = None) -> str:
    stamp = (now or _dt.date.today()).strftime("%Y-%m-%d")
    slug = re.sub(r"[^a-z0-9]+", "-", merchant.lower()).strip("-") or "merchant"
    return f"{slug}-your-prices-{stamp}.html"


def one_pager(
    merchant: str,
    bands: list[Band],
    *,
    sales_days: int,
    demand_days: int,
    now: _dt.date | None = None,
) -> str:
    """The whole page: a ring, some bars, one sentence, and the small print."""
    when = (now or _dt.date.today()).strftime("%d %B %Y")
    lead = headline(bands)
    bars = bars_svg(bands)
    return _PAGE.format(
        merchant=html.escape(merchant),
        when=html.escape(when),
        pie=pie_svg(bands),
        legend=_legend(bands),
        headline=f'<p class="headline">{html.escape(lead)}</p>' if lead else "",
        bars=(
            '<section><h2>What each of those sold</h2>'
            '<p class="sub">Bottles sold for every 100 shoppers who looked - '
            "which is the fair comparison, because a bigger group of wines "
            "does not sell more per shopper for being bigger.</p>"
            f"{bars}</section>"
            if bars
            else ""
        ),
        sales_days=sales_days,
        demand_days=demand_days,
    )


# One file, no network: it is going to be opened from an email attachment on
# somebody's phone in a shop, and printed from a laptop that may be offline.
_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{merchant} &middot; your prices against the market</title>
<style>
  @page {{ size: A4; margin: 16mm; }}
  body {{
    font: 12pt/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #111827; margin: 0 auto; padding: 28px; max-width: 820px;
  }}
  header {{ border-bottom: 2px solid #111827; padding-bottom: 10px; margin-bottom: 20px; }}
  h1 {{ font-size: 21pt; margin: 0; }}
  header p {{ color: #6b7280; margin: 4px 0 0; font-size: 10pt; }}
  section {{ margin-bottom: 26px; break-inside: avoid; }}
  h2 {{ font-size: 13pt; margin: 0 0 8px; }}
  .sub {{ color: #4b5563; font-size: 10.5pt; margin: 0 0 12px; }}
  .split {{ display: flex; gap: 28px; align-items: center; flex-wrap: wrap; }}
  .legend {{ list-style: none; margin: 0; padding: 0; font-size: 11pt; }}
  .legend li {{ margin-bottom: 8px; }}
  .swatch {{
    display: inline-block; width: 13px; height: 13px; border-radius: 3px;
    margin-right: 8px; vertical-align: -1px;
  }}
  .headline {{
    font-size: 14pt; font-weight: 600; line-height: 1.4; margin: 0 0 18px;
    padding: 14px 16px; background: #f0fdf4; border-left: 5px solid #15803d;
  }}
  .bar-label {{ font-size: 11px; fill: #374151; }}
  .bar-value {{ font-size: 12px; font-weight: 700; fill: #111827; }}
  footer {{ color: #6b7280; font-size: 9.5pt; border-top: 1px solid #e5e7eb; padding-top: 10px; }}
  @media print {{ body {{ padding: 0; }} }}
</style></head>
<body>
<header>
  <h1>{merchant}</h1>
  <p>Your prices against everyone else selling the same wine &middot; {when}</p>
</header>
{headline}
<section>
  <h2>Your compared wines, by how they are priced</h2>
  <div class="split">{pie}{legend}</div>
</section>
{bars}
<footer>
  These are the wines Google could compare: for each of them it reports what
  other shops charge for the same bottle. Wines no other shop sells have
  nothing to be compared with, and are not in the picture above.
  Shoppers are the last {demand_days} days of Google Shopping visits to your
  wines; bottles are what you sold in the last {sales_days} days.
  This compares your own wines with each other rather than being a trial, so a
  keenly priced wine may also be a wine people happen to want - but the bottles
  are your own.
</footer>
</body></html>
"""

__all__ = ["BAND_COLOURS", "Band", "bars_svg", "filename", "headline", "one_pager", "pie_svg"]
