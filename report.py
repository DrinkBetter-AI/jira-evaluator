"""What a tab said, gathered as it draws, so it can be printed.

Printing the dashboard itself gives a page of widgets: the sidebar, the radio
buttons, whatever happened to be scrolled into view, and none of the tables that
sit behind a tab. A leadership report is the figures and the sentences that
explain them, on one page, in the order they were read.

Each section hands its numbers here as it renders them, and the button at the
top of the tab offers the result as a self-contained HTML file. It carries its
own print stylesheet, so opening it and pressing Cmd-P gives a PDF without the
container needing a PDF library in it.
"""

from __future__ import annotations

import datetime as _dt
import html
import re
from dataclasses import dataclass, field

# Bold in a verdict line marks the figure the sentence is about, and it is the
# only markup those lines use.
_BOLD = re.compile(r"\*\*(.+?)\*\*")


@dataclass(frozen=True)
class Figure:
    """One tile: what it counts, what it read, and the aside beneath it."""

    label: str
    value: str
    note: str = ""


@dataclass
class Section:
    """One heading's worth of a tab, in the order it was drawn."""

    title: str
    figures: list[Figure] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Report:
    """The figures of one tab, collected while the tab renders."""

    def __init__(self, title: str) -> None:
        self.title = title
        self._sections: dict[str, Section] = {}

    def _section(self, title: str) -> Section:
        return self._sections.setdefault(title, Section(title))

    def figure(self, section: str, label: str, value: str, note: str = "") -> None:
        """Record a tile. A tile showing nothing is left out of the report."""
        if value in ("", "\u2014", None):
            return
        self._section(section).figures.append(Figure(label, str(value), note))

    def note(self, section: str, text: str) -> None:
        """Record a sentence: the part that says what the figures mean."""
        if text and text.strip():
            self._section(section).notes.append(text.strip())

    @property
    def empty(self) -> bool:
        return not any(
            section.figures or section.notes for section in self._sections.values()
        )

    def html(self, *, now: _dt.datetime | None = None) -> str:
        """The whole report as one printable page."""
        when = (now or _dt.datetime.now(_dt.timezone.utc)).strftime("%d %B %Y, %H:%M UTC")
        body = "\n".join(_section_html(s) for s in self._sections.values() if s.figures or s.notes)
        return _PAGE.format(
            title=html.escape(self.title),
            when=html.escape(when),
            body=body or "<p class='empty'>Nothing had loaded when this was taken.</p>",
        )

    def filename(self, *, now: _dt.datetime | None = None) -> str:
        stamp = (now or _dt.datetime.now(_dt.timezone.utc)).strftime("%Y-%m-%d")
        slug = re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-")
        return f"{slug}-{stamp}.html"


def _rich(text: str) -> str:
    """A verdict line as HTML: escaped, with its bold figure kept bold."""
    return _BOLD.sub(r"<strong>\1</strong>", html.escape(text))


def _section_html(section: Section) -> str:
    tiles = "".join(
        f'<div class="figure"><div class="label">{html.escape(f.label)}</div>'
        f'<div class="value">{html.escape(f.value)}</div>'
        + (f'<div class="note">{html.escape(f.note)}</div>' if f.note else "")
        + "</div>"
        for f in section.figures
    )
    notes = "".join(f"<li>{_rich(note)}</li>" for note in section.notes)
    return (
        f"<section><h2>{html.escape(section.title)}</h2>"
        + (f'<div class="figures">{tiles}</div>' if tiles else "")
        + (f"<ul>{notes}</ul>" if notes else "")
        + "</section>"
    )


# Deliberately one file with no external anything: it has to open and print the
# same from a laptop with no network, which is where a board pack gets read.
_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
  @page {{ size: A4; margin: 16mm; }}
  body {{
    font: 11pt/1.45 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #111827; margin: 0 auto; padding: 24px; max-width: 900px;
  }}
  header {{ border-bottom: 2px solid #111827; padding-bottom: 8px; margin-bottom: 18px; }}
  h1 {{ font-size: 20pt; margin: 0; }}
  header p {{ color: #6b7280; margin: 4px 0 0; font-size: 9.5pt; }}
  section {{ margin-bottom: 22px; break-inside: avoid; }}
  h2 {{ font-size: 12.5pt; margin: 0 0 10px; color: #1f2937; }}
  .figures {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }}
  .figure {{
    border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px 12px;
    min-width: 130px; flex: 0 1 auto;
  }}
  .label {{ font-size: 8.5pt; text-transform: uppercase; letter-spacing: .04em; color: #6b7280; }}
  .value {{ font-size: 15pt; font-weight: 650; }}
  .note {{ font-size: 8.5pt; color: #6b7280; }}
  ul {{ margin: 0; padding-left: 18px; }}
  li {{ margin-bottom: 5px; }}
  .empty {{ color: #6b7280; }}
  @media print {{ body {{ padding: 0; }} section {{ break-inside: avoid; }} }}
</style></head>
<body>
<header><h1>{title}</h1><p>VinoVoss &middot; {when}</p></header>
{body}
</body></html>
"""

__all__ = ["Figure", "Report", "Section"]
