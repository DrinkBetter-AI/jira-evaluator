"""The whole page as it was drawn, kept as it draws, printed as a PDF.

The tab report next door is a summary: the tiles and the verdict sentences, on
one page, for a reader who wants the figures. This is the other thing people
ask for - the board itself, all of it, in the order it appears on screen, with
its tables coloured the way the screen colours them and its charts drawn as the
charts they are. It exists because a design has to be looked at whole: font
against figure, chart against table, section against section. A summary cannot
be judged that way, and neither can a screenshot of whatever was scrolled into
view.

Nothing here asks the page to do anything differently. A recorder listens to
the Streamlit calls the page is already making - headings, captions, metrics,
tables, charts - and keeps them. The page renders exactly as it did before;
when the reader asks for the file, the recording is laid out as a printable
page and handed to WeasyPrint.

Two things are optional at runtime and the file is produced without them: the
PDF library itself (without it the same page is offered as HTML), and the chart
image renderer (without it a chart leaves a labelled gap rather than losing the
page).
"""

from __future__ import annotations

import base64
import contextlib
import datetime as _dt
import functools
import html
import logging
import re
import threading
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# A long table is the board's own content, but a thousand rows of it is a
# hundred pages of paper and nothing about the design that the first screenful
# does not already say.
MAX_ROWS = 120

# Wide enough to be the landscape page it is printed on, and tall enough that a
# chart keeps the shape it has on screen rather than becoming a strip.
CHART_WIDTH = 1400
CHART_HEIGHT = 620

# The calls a page makes to draw itself. Widgets are deliberately absent: a
# button or a selectbox is how the reader asked for what is on the page, not
# part of what the page said. ``write`` is absent too, and for the opposite
# reason: it draws by calling one of these itself, so listening to both would
# print every line of it twice.
DRAWING_CALLS = (
    "title",
    "header",
    "subheader",
    "markdown",
    "caption",
    "text",
    "metric",
    "dataframe",
    "table",
    "plotly_chart",
    "bar_chart",
    "line_chart",
    "area_chart",
    "info",
    "warning",
    "error",
    "success",
    "divider",
    "code",
    # Not drawings but structure: what a section is called when it is folded
    # away behind a label, and what the tabs beside it offer.
    "expander",
    "tabs",
)

_HEADINGS = {"title": 1, "header": 2, "subheader": 3}
_CALLOUTS = {"info": "info", "warning": "warning", "error": "error", "success": "success"}
_CHARTS = {"bar_chart": "bar", "line_chart": "line", "area_chart": "area"}


@dataclass(frozen=True)
class Heading:
    level: int
    text: str


@dataclass(frozen=True)
class Text:
    text: str
    caption: bool = False


@dataclass(frozen=True)
class Callout:
    """One of the page's tinted notes: info, warning, error, success."""

    kind: str
    text: str


@dataclass(frozen=True)
class Raw:
    """Markup the page wrote itself, kept as markup so it looks like itself."""

    html: str


@dataclass(frozen=True)
class Metric:
    label: str
    value: str
    delta: str = ""


@dataclass(frozen=True)
class Table:
    """The frame the page drew, kept as it is and turned to HTML on export.

    Held rather than rendered because every rerun of the page records afresh,
    and turning fifty tables into markup on every keystroke would be paid for
    by every reader to serve the one who eventually asks for the file.
    """

    data: object
    hide_index: bool = False
    # What the screen calls each column, and which it does not show at all.
    columns: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chart:
    figure: object


@dataclass(frozen=True)
class SimpleChart:
    """One of Streamlit's own charts, built into a figure only on export."""

    kind: str
    data: object
    x: object = None
    y: object = None
    title: str = ""


@dataclass(frozen=True)
class Rule:
    pass


@dataclass
class Snapshot:
    """One page's worth of drawing, in the order the page drew it."""

    page: str
    blocks: list[object] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not any(not isinstance(block, Rule) for block in self.blocks)

    def observe(self, call: str, args: tuple, kwargs: dict) -> None:
        """Keep what one drawing call put on the page, if it can be printed."""
        try:
            block = _block(call, args, kwargs)
        except Exception:  # noqa: BLE001 - a snapshot never breaks the page
            logger.debug("Snapshot skipped a %s call", call, exc_info=True)
            return
        if block is not None:
            self.blocks.append(block)

    def html(self, *, now: _dt.datetime | None = None) -> str:
        return _page_html(self, now=now)

    def pdf(self, *, now: _dt.datetime | None = None) -> bytes | None:
        """The page as a PDF, or None where no PDF library is installed."""
        return to_pdf(self.html(now=now))

    def filename(self, suffix: str, *, now: _dt.datetime | None = None) -> str:
        stamp = (now or _dt.datetime.now(_dt.timezone.utc)).strftime("%Y-%m-%d")
        slug = re.sub(r"[^a-z0-9]+", "-", self.page.lower()).strip("-")
        return f"{slug}-board-{stamp}.{suffix}"


def _block(call: str, args: tuple, kwargs: dict):
    """The recorded form of one drawing call, or None if there is nothing to keep."""
    if call == "divider":
        return Rule()
    if call in _HEADINGS:
        return Heading(_HEADINGS[call], _plain(args[0] if args else kwargs.get("body", "")))
    if call in _CALLOUTS:
        text = _plain(args[0] if args else kwargs.get("body", ""))
        return Callout(_CALLOUTS[call], text) if text else None
    if call == "metric":
        label = _plain(_arg(args, kwargs, 0, "label"))
        value = _arg(args, kwargs, 1, "value")
        delta = kwargs.get("delta") or (args[2] if len(args) > 2 else "")
        return Metric(label, "" if value is None else str(value), "" if delta is None else str(delta))
    if call in ("dataframe", "table"):
        return _table_block(
            _arg(args, kwargs, 0, "data"),
            bool(kwargs.get("hide_index")),
            kwargs.get("column_config") or {},
        )
    if call == "plotly_chart":
        figure = _arg(args, kwargs, 0, "figure_or_data")
        return Chart(figure) if figure is not None else None
    if call in _CHARTS:
        return _simple_chart(_CHARTS[call], _arg(args, kwargs, 0, "data"), kwargs)
    if call == "code":
        body = _plain(args[0] if args else kwargs.get("body", ""))
        return Text(body) if body else None
    if call == "expander":
        label = _plain(_arg(args, kwargs, 0, "label"))
        return Heading(4, label) if label else None
    if call == "tabs":
        labels = [_plain(one) for one in (_arg(args, kwargs, 0, "tabs") or [])]
        return Text("Tabs: " + " | ".join(label for label in labels if label)) if labels else None

    # markdown, caption, text, write.
    body = _arg(args, kwargs, 0, "body")
    if isinstance(body, (pd.DataFrame, pd.Series)):
        return _table_block(body)
    text = _plain(body)
    if not text:
        return None
    if kwargs.get("unsafe_allow_html") and "<" in text:
        # The KPI strip and the triage cards are written as markup by the page
        # itself; kept as markup they print as the cards they are on screen.
        return Raw(text)
    return Text(text, caption=call == "caption")


def _arg(args: tuple, kwargs: dict, index: int, name: str):
    if len(args) > index:
        return args[index]
    return kwargs.get(name)


def _plain(value) -> str:
    return "" if value is None else str(value).strip()


def _table_block(data, hide_index: bool = False, columns: dict | None = None) -> Table | None:
    """A table the page drew, if what it drew was a table at all."""
    if _frame_of(data) is None:
        return None
    return Table(data, hide_index, dict(columns or {}))


def _frame_of(data) -> pd.DataFrame | None:
    """The rows behind whatever was handed to ``dataframe``, or None."""
    behind = getattr(data, "data", None)
    if hasattr(data, "to_html") and isinstance(behind, (pd.DataFrame, pd.Series)):
        data = behind  # a Styler: the colours are its own, the rows are these
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if not isinstance(data, pd.DataFrame):
        return None
    return None if data.empty else data


def _table_html(block: Table) -> str:
    """The page's table as markup, with the colours it was styled with kept."""
    frame = _frame_of(block.data)
    if frame is None:
        return ""
    rows = len(frame)
    styler = block.data if hasattr(block.data, "to_html") and hasattr(block.data, "data") else None
    hidden, named = _column_plan(block, frame)
    # A styler writes its cells as they are, so a ticket summary that happens to
    # contain a tag would be markup on the page rather than the text the screen
    # shows. That table is printed plain instead: one table's tint is a smaller
    # loss than a board rearranged by somebody's ticket title.
    if rows <= MAX_ROWS and styler is not None and not _has_markup(frame):
        # Hiding and relabelling are the styler's own instructions to itself,
        # given after the screen has already drawn from it, so nothing on screen
        # is redrawn by them.
        with contextlib.suppress(Exception):
            if block.hide_index:
                styler = styler.hide(axis="index")
            if hidden:
                styler = styler.hide(subset=hidden, axis="columns")
            shown = [column for column in frame.columns if column not in hidden]
            if any(named.get(column, column) != column for column in shown):
                styler = styler.relabel_index(
                    [named.get(column, column) for column in shown], axis="columns"
                )
            return styler.to_html()
    printed = frame.drop(columns=hidden, errors="ignore").rename(columns=named)
    note = f'<p class="caption">First {MAX_ROWS} of {rows} rows.</p>' if rows > MAX_ROWS else ""
    return printed.head(MAX_ROWS).to_html(index=not block.hide_index, border=0, escape=True) + note


def _column_plan(block: Table, frame: pd.DataFrame) -> tuple[list, dict]:
    """Which columns the screen hides, and what it calls the rest.

    ``st.dataframe`` is given a `column_config` naming its columns for the
    reader - "Idle (days)" rather than `idle_days` - and dropping the ones it
    keeps only to work with. A printed board that showed `key_url` beside
    Summary would not be the board on screen.
    """
    hidden = [name for name, spec in block.columns.items() if spec is None and name in frame.columns]
    named = {
        name: str(spec["label"])
        for name, spec in block.columns.items()
        if isinstance(spec, dict) and spec.get("label") and name in frame.columns
    }
    return hidden, named


def _has_markup(frame: pd.DataFrame) -> bool:
    """Whether anything printed verbatim carries what a browser reads as a tag.

    The cells, but the row labels and the column names too: a styler writes all
    three as they are, and any of the three can be a ticket summary or a name
    somebody chose.
    """
    labels = pd.Series(list(frame.columns) + list(frame.index)).astype(str)
    if labels.str.contains("<", regex=False).any():
        return True
    text = frame.select_dtypes(include=["object", "string"])
    if text.empty:
        return False
    return bool(text.astype(str).apply(lambda column: column.str.contains("<", regex=False)).any().any())


def _simple_chart(kind: str, data, kwargs: dict) -> SimpleChart | None:
    """Streamlit's own bar/line/area charts, kept for export to draw."""
    if _frame_of(data) is None:
        return None
    return SimpleChart(
        kind,
        data,
        kwargs.get("x"),
        kwargs.get("y"),
        str(kwargs.get("title") or ""),
    )


def _simple_figure(block: SimpleChart):
    """The figure behind one of Streamlit's own charts.

    ``st.bar_chart(frame)`` names no axes: it plots the index against every
    numeric column, and plotly refuses a frame of mixed types outright, so the
    axes Streamlit would have chosen are chosen here.
    """
    import plotly.express as px  # local: only a printed page needs it

    frame = _frame_of(block.data)
    x = block.x
    if x is None:
        frame = frame.reset_index()
        x = frame.columns[0]
    y = block.y or [
        column
        for column in frame.columns
        if column != x and pd.api.types.is_numeric_dtype(frame[column])
    ]
    builder = {"bar": px.bar, "line": px.line, "area": px.area}[block.kind]
    return builder(frame, x=x, y=y or None, title=block.title or None)


def _in_sidebar(container) -> bool:
    """Whether a call is drawing into the sidebar rather than the page.

    ``with st.sidebar:`` does not change which container a bare ``st.caption``
    is called on - it changes the one Streamlit is currently drawing into - so
    the destination has to be read from the container the call will actually
    reach, not from the one it was called on.
    """
    active = getattr(container, "_active_dg", container)
    return getattr(active, "_root_container", 0) == 1


# Where the page being recorded on this thread is kept. Streamlit gives each
# browser session its own script thread, so one reader's board is the thread's
# own business and never another's.
_listening = threading.local()

# The wrappers go on once and stay on. Putting them on and taking them off
# around each page would be two readers' business at once: the second to start
# would capture the first's wrapper as the original, the first to finish would
# take the second's off mid-page, and whatever was left installed would hold a
# finished recording - with its frames and figures - for the life of the
# process. Installed once, they hold nothing: with no page being recorded on
# this thread they are a function call and a None check.
_installed = threading.Lock()
_ready = False


def _listen() -> None:
    """Put the recorder in the path of every drawing call, once, for good."""
    global _ready
    with _installed:
        if _ready:
            return
        from streamlit.delta_generator import DeltaGenerator
        import streamlit as st

        for name in DRAWING_CALLS:
            original = getattr(DeltaGenerator, name, None)
            if original is None:
                continue

            def wrapper(self, *args, _name=name, _original=original, **kwargs):
                snapshot = getattr(_listening, "snapshot", None)
                if snapshot is not None and not _in_sidebar(self):
                    snapshot.observe(_name, args, kwargs)
                return _original(self, *args, **kwargs)

            functools.update_wrapper(wrapper, original)
            module_level = getattr(st, name, None)
            setattr(DeltaGenerator, name, wrapper)
            # ``st.dataframe`` and friends were bound to the main container when
            # Streamlit was imported, so patching the class alone leaves them.
            owner = getattr(module_level, "__self__", None)
            if owner is not None:
                setattr(st, name, wrapper.__get__(owner, DeltaGenerator))
        _ready = True


@contextlib.contextmanager
def recording(page: str):
    """Keep every drawing call the page makes on this thread, then let go.

    The sidebar is skipped: the filters are how the page was asked for, not
    part of the board.
    """
    _listen()
    snapshot = Snapshot(page)
    before = getattr(_listening, "snapshot", None)
    _listening.snapshot = snapshot
    try:
        yield snapshot
    finally:
        _listening.snapshot = before


def _weasyprint():
    """WeasyPrint's HTML class, or None where it is not installed."""
    try:
        from weasyprint import HTML  # noqa: PLC0415 - optional at runtime

        return HTML
    except Exception:  # noqa: BLE001 - any import trouble is the same answer
        logger.info("No PDF library installed; the board is offered as HTML")
        return None


def _inline_only(url: str):
    """Fetch nothing: the board carries everything it needs inside itself.

    A typesetter given an address will go and get it, from the internet or from
    the disk it is running on, and the board is assembled partly from markup the
    page wrote. The charts are already inline, the styling is already inline, so
    there is nothing left to fetch and no reason to allow it.
    """
    from weasyprint.urls import default_url_fetcher  # noqa: PLC0415 - optional

    if url.startswith("data:"):
        return default_url_fetcher(url)
    raise ValueError("the printed board fetches nothing from outside itself")


def to_pdf(page: str) -> bytes | None:
    """The page as PDF bytes, or None where it could not be made into one.

    A font that cannot be found, an image that cannot be read: the reader asked
    for their board and is given the printable page instead of a traceback.
    """
    render = _weasyprint()
    if render is None:
        return None
    try:
        return render(string=page, url_fetcher=_inline_only).write_pdf()
    except Exception:  # noqa: BLE001 - the board is offered as HTML instead
        logger.warning("The board could not be laid out as a PDF", exc_info=True)
        return None


# Streamlit's chart template does not name its colours: it fills them with
# placeholders (#000001, #000002...) and its own frontend swaps them for the
# theme's palette as it draws. An image renderer has no frontend, so a chart
# exported as it stands comes out in shades of black - these are the colours
# the browser would have put there.
STREAMLIT_COLOURS = (
    "#0068c9",
    "#83c9ff",
    "#ff2b2b",
    "#ffabab",
    "#29b09d",
    "#7defa1",
    "#ff8700",
    "#ffd16a",
    "#6d3fc0",
    "#d5dae5",
)
_PLACEHOLDER = re.compile(r"^#0{5}([0-9a-fA-F])$")


def _coloured(value):
    """The same figure fragment with Streamlit's placeholder colours filled in."""
    if isinstance(value, dict):
        return {key: _coloured(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coloured(item) for item in value]
    if isinstance(value, str):
        match = _PLACEHOLDER.match(value)
        if match:
            return STREAMLIT_COLOURS[(int(match.group(1), 16) - 1) % len(STREAMLIT_COLOURS)]
    return value


def _printable(figure):
    """The figure as the browser draws it: its colours, and its text size."""
    import plotly.graph_objects as go  # local: only a printed page needs it

    import theme

    printable = go.Figure(_coloured(figure.to_plotly_json()))
    printable.update_layout(
        template="plotly_white",
        font=dict(size=theme.CHART_FONT),
        title_font=dict(size=theme.CHART_TITLE_FONT),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    return printable


def _chart_image(block) -> str:
    """A chart as an inline image, or "" when it could not be drawn.

    A chart image needs a renderer that is not always installed, so one that
    cannot be drawn leaves a labelled gap: a board of forty sections is not
    worth losing over one of them.
    """
    try:
        figure = block.figure if isinstance(block, Chart) else _simple_figure(block)
        raw = _printable(figure).to_image(
            format="png", width=CHART_WIDTH, height=CHART_HEIGHT, scale=2
        )
    except Exception:  # noqa: BLE001 - a missing renderer loses one chart, not the page
        logger.info("Chart could not be drawn into the snapshot", exc_info=True)
        return ""
    return base64.b64encode(raw).decode("ascii")


_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])")
_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# A dollar sign is escaped on the way to Streamlit, whose markdown would
# otherwise read a pair of them as maths. Nothing here reads maths, so the
# backslash is the page's business and not the reader's.
_ESCAPED = re.compile(r"\\([$*_`\[\]])")


def _rich(text: str) -> str:
    """One line of the page's markdown as HTML: bold, code, links, italics.

    The links are read off the line as the page wrote it, before the line is
    escaped: a Jira search address carries ``&`` between its terms, and escaping
    it once as part of the line and again as an address would send the reader
    somewhere that does not exist.
    """
    pieces: list[str] = []
    written = 0
    for link in _LINK.finditer(text):
        pieces.append(html.escape(text[written : link.start()]))
        pieces.append(
            f'<a href="{html.escape(link.group(2), quote=True)}">{html.escape(link.group(1))}</a>'
        )
        written = link.end()
    pieces.append(html.escape(text[written:]))
    marked = "".join(pieces)
    marked = _CODE.sub(r"<code>\1</code>", marked)
    marked = _BOLD.sub(r"<strong>\1</strong>", marked)
    marked = _ITALIC.sub(r"<em>\1</em>", marked)
    return _ESCAPED.sub(r"\1", marked)


_HASH_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _text_html(block: Text) -> str:
    lines = [line for line in block.text.splitlines() if line.strip()]
    bullets = [line for line in lines if line.lstrip().startswith(("- ", "* "))]
    css = "caption" if block.caption else "body"
    if bullets and len(bullets) == len(lines):
        items = "".join(f"<li>{_rich(line.lstrip()[2:])}</li>" for line in lines)
        return f'<ul class="{css}">{items}</ul>'
    out = []
    for line in lines:
        # Sections written as markdown rather than as st.subheader: the hashes
        # are a heading on screen and would be five literal hashes on paper.
        heading = _HASH_HEADING.match(line.strip())
        if heading:
            level = min(len(heading.group(1)) + 1, 6)
            out.append(f"<h{level}>{_rich(heading.group(2))}</h{level}>")
        else:
            out.append(f'<p class="{css}">{_rich(line)}</p>')
    return "".join(out)


def _blocks_html(snapshot: Snapshot) -> str:
    out: list[str] = []
    metrics: list[Metric] = []

    def flush() -> None:
        # Metrics drawn one after another are a row of tiles on screen, and the
        # row is the thing being judged, so they are printed as one.
        if not metrics:
            return
        tiles = "".join(
            f'<div class="tile"><div class="tile-label">{html.escape(m.label)}</div>'
            f'<div class="tile-value">{html.escape(m.value)}</div>'
            + (f'<div class="tile-delta">{html.escape(m.delta)}</div>' if m.delta else "")
            + "</div>"
            for m in metrics
        )
        out.append(f'<div class="tiles">{tiles}</div>')
        metrics.clear()

    for block in snapshot.blocks:
        if isinstance(block, Rule) and not metrics and out and out[-1] == "<hr>":
            # Rules with only widgets between them are one line on paper. Tiles
            # waiting to be written are content, so the line before them stays.
            continue
        if isinstance(block, Metric):
            metrics.append(block)
            continue
        flush()
        if isinstance(block, Heading):
            out.append(f"<h{block.level}>{html.escape(block.text)}</h{block.level}>")
        elif isinstance(block, Text):
            out.append(_text_html(block))
        elif isinstance(block, Callout):
            out.append(f'<div class="note {block.kind}">{_rich(block.text)}</div>')
        elif isinstance(block, Raw):
            out.append(block.html)
        elif isinstance(block, Table):
            markup = _table_html(block)
            if markup:
                out.append(f'<div class="table">{markup}</div>')
        elif isinstance(block, (Chart, SimpleChart)):
            image = _chart_image(block)
            out.append(
                f'<div class="chart"><img src="data:image/png;base64,{image}"></div>'
                if image
                else '<div class="chart missing">A chart was drawn here.</div>'
            )
        elif isinstance(block, Rule):
            out.append("<hr>")
    flush()
    return "\n".join(out)


def _page_html(snapshot: Snapshot, *, now: _dt.datetime | None = None) -> str:
    when = (now or _dt.datetime.now(_dt.timezone.utc)).strftime("%d %B %Y, %H:%M UTC")
    return _PAGE.format(
        title=html.escape(f"{snapshot.page} board"),
        when=html.escape(when),
        body=_blocks_html(snapshot) or '<p class="caption">Nothing had drawn when this was taken.</p>',
    )


# The dashboard's own colours and type, so the file is the board rather than a
# document about it: the screen's text colour, its card borders and radii, its
# muted captions, and the same sans the browser renders it in. Landscape,
# because the board is wide and a portrait page would reflow every table into
# something nobody designed.
_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
  @page {{ size: A4 landscape; margin: 12mm; }}
  body {{
    font: 11pt/1.5 "Source Sans Pro", -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #111827; background: #ffffff; margin: 0;
  }}
  header {{ border-bottom: 2px solid #111827; padding-bottom: 8px; margin-bottom: 16px; }}
  header h1 {{ font-size: 22pt; margin: 0; font-weight: 700; }}
  header p {{ color: #6b7280; margin: 4px 0 0; font-size: 10pt; }}
  h1 {{ font-size: 19pt; margin: 18px 0 8px; }}
  h2 {{ font-size: 15pt; margin: 16px 0 8px; }}
  h3 {{ font-size: 13pt; margin: 14px 0 6px; }}
  p.body {{ margin: 4px 0; }}
  p.caption, ul.caption {{ color: #4b5563; font-size: 9.5pt; margin: 4px 0; }}
  ul {{ margin: 4px 0 4px 18px; padding: 0; }}
  li {{ margin: 2px 0; }}
  hr {{ border: none; border-top: 1px solid #e5e7eb; margin: 14px 0; }}
  .tiles {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 8px 0 12px; }}
  .tile {{
    border: 1px solid #e5e7eb; border-radius: 12px; padding: 9px 14px;
    min-width: 150px; background: #ffffff;
  }}
  .tile-label {{ font-size: 9pt; color: #6b7280; }}
  .tile-value {{ font-size: 19pt; font-weight: 700; line-height: 1.2; }}
  .tile-delta {{ font-size: 9pt; color: #6b7280; }}
  .table {{ margin: 8px 0 12px; break-inside: avoid; }}
  /* A board's tables are wider than any page. Fixed layout wraps a twenty
     column table into the margins instead of running off the right edge and
     losing whatever was on the end of it. */
  table {{
    border-collapse: collapse; width: 100%; font-size: 8pt;
    table-layout: fixed; word-wrap: break-word; overflow-wrap: anywhere;
  }}
  th, td {{
    border-bottom: 1px solid #e5e7eb; padding: 4px 7px; text-align: left;
    vertical-align: top;
  }}
  th {{ background: #f8fafc; font-weight: 600; color: #374151; }}
  .chart {{ margin: 8px 0 14px; break-inside: avoid; }}
  .chart img {{ width: 100%; height: auto; }}
  .chart.missing {{ color: #6b7280; font-size: 9.5pt; font-style: italic; }}
  /* The page's tinted notes, in the colours Streamlit tints them. */
  .note {{
    border-radius: 8px; padding: 7px 11px; margin: 6px 0; font-size: 10pt;
    break-inside: avoid;
  }}
  .note.info {{ background: #e7f3fe; color: #0b4a6f; }}
  .note.warning {{ background: #fff6e0; color: #7a4b00; }}
  .note.error {{ background: #fdecec; color: #8b1a1a; }}
  .note.success {{ background: #e6f6ec; color: #14532d; }}
  /* The page's own cards, drawn from the dashboard's stylesheet. */
  .kpi-strip {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 8px 0 12px; }}
  .kpi-card {{
    flex: 1 1 170px; padding: 9px 14px; border: 1px solid #e5e7eb;
    border-radius: 12px; background: #ffffff;
  }}
  .kpi-card .kpi-label {{ font-size: 9pt; color: #6b7280; }}
  .kpi-card .kpi-value {{ font-size: 19pt; font-weight: 700; line-height: 1.2; }}
  .kpi-card .kpi-note {{ font-size: 9pt; color: #9ca3af; }}
  .triage-card {{
    border: 1px solid #e5e7eb; border-radius: 14px; padding: 10px 14px;
    margin: 6px 0 10px; break-inside: avoid;
  }}
  .triage-card .triage-key {{ font-size: 9pt; font-weight: 700; color: #1d4ed8; }}
  .triage-card .triage-summary {{ font-size: 13pt; font-weight: 600; margin: 2px 0 6px; }}
  .triage-meta {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 5px; }}
  .triage-meta span {{
    background: #f3f4f6; color: #4b5563; border-radius: 6px; padding: 2px 8px; font-size: 8.5pt;
  }}
  .triage-meta span.hot {{ background: #fdecec; color: #b91c1c; font-weight: 600; }}
  .triage-why {{ font-size: 9pt; color: #6b7280; font-style: italic; }}
  a {{ color: #1d4ed8; }}
  code {{
    font-family: "DejaVu Sans Mono", monospace; font-size: 9pt;
    background: #f3f4f6; border-radius: 4px; padding: 1px 4px; color: #b91c1c;
  }}
</style></head>
<body>
<header><h1>{title}</h1><p>VinoVoss &middot; {when}</p></header>
{body}
</body></html>
"""

__all__ = ["Snapshot", "recording", "to_pdf"]
