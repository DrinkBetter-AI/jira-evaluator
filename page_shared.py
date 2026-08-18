"""Phase 8: the small reporting / tile / KPI / board-delivery framework
shared by every page - the printable-report bookkeeping (`_report`,
`_tile`, `_kpis`, `_said`), the whole-board PDF/HTML offer
(`_download_report`, `_offer_board_snapshot`, `_build_board_file`,
`_deliver_board_snapshot`), and a handful of tiny formatting helpers used
across pages. Split out of app.py so data_layer.py and pages/*.py can use
it without importing app.py itself.
"""

from __future__ import annotations

import collections
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import datetime as _dt
import hashlib
import html
import logging
import os
import pickle
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable, NamedTuple
from urllib.parse import quote, unquote, urlencode

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from dotenv import load_dotenv

# Local settings come from a .env file when one is present, so `streamlit run
# app.py` needs no exported variables. This runs before the local imports below
# because some of them read the environment at import time (change_audit picks
# its log path there). Existing variables win, so a real deployment's injected
# environment is never overridden by a file that happens to be lying around.
load_dotenv(override=False)

import read_log
from change_audit import (
    append_operation,
    finalize_operation,
    load_operations,
    new_operation_record,
    summarize_operations,
)
from jira_client import (
    DEFAULT_CREDS_PATH,
    DEFAULT_FIELDS,
    DEFAULT_PROFILE_NAME,
    MAX_PARALLEL_REQUESTS,
    JiraClient,
    JiraConfigError,
    normalize_base_url,
    load_jira_env,
    load_jira_profile,
)
from access_gate import render_sign_out, require_password
import focus
import github_client
import kpi
import integrity
import next_actions
import pr_hygiene
import pr_quality
import theme_html
from capacity import (
    capacity_table,
    same_person,
    match_weekly_hours,
    parse_weekly_hours,
    working_days,
)
import cleanup
from cleanup import is_unowned
import engineer_letter
import epic_organization
from epics import epic_health_flags, epic_rollup
from teams import (
    NO_OWNER_TEAM,
    DEFAULT_TEAM_PEOPLE,
    add_team,
    parse_team_people,
    parse_team_projects,
    team_summary,
)
import theme
from theme import inject_styles, kpi_strip
import report as reporting
import snapshot as board
from hygiene import (
    CONTAINER_ISSUE_TYPES,
    DEFAULT_STALE_DAYS,
    estimate_policy,
    policy_compliance_by_owner,
    stale_candidates,
)
import ads_client
import ads_evidence
import cost_client
import amplitude_client
import merchant_client
import merchant_letter
import vivino_client
import orders
import orders_client
from prioritization import add_priority_score, assignee_rollup
import sprint_planner
import ticket_quality
from transformations import add_ticket_health_fields
import write_access
from contextlib import contextmanager



logger = logging.getLogger(__name__)




@contextmanager
def _log_stage(name: str):
    """Time one derivation step at INFO level.

    A slow rerun has to be traceable to the one changelog walk or reshape that
    caused it, not guessed at from the page's total. Every stage worth timing
    - a changelog parse, a bundle derivation - wraps its work in this so the
    same log line shape names it and its cost.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        logger.info("stage %s: %.3fs", name, time.perf_counter() - started)


def _text_or(value: object, default: str) -> str:
    """``value`` as text, or ``default`` when there is nothing in it.

    ``value or default`` reads as though it does this and does not: an empty
    cell in a pandas frame is a float NaN, NaN is truthy, and ``NaN or "none"``
    is therefore NaN - which is how the triage card came to tell a reviewer
    "Epic: nan" about a ticket that simply has no epic. Whitespace counts as
    nothing too, because a Jira field cleared by hand often keeps a space.
    """
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        # A list or an array has no single truthiness; it is also not missing.
        pass
    return str(value).strip() or default


def _number_or(value: object, default: float = 0.0) -> float:
    """``value`` as a float, or ``default`` when it is missing or not a number.

    The same NaN trap as ``_text_or``: ``float(row["idle_days"] or 0)`` keeps
    the NaN and prints "nan days old" on the card.
    """
    number = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(number) else float(number)


def _positive_int(value: str | None, *, default: int) -> int:
    """Read a positive integer setting; a typo costs the setting, not the app."""
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


TAB_ENGINEERING = "Engineering"
TAB_BUSINESS = "Business"
REPORTS_KEY = "tab_reports"
# Which page was asked for as a file, and where on it the button that asked
# sits: the file can only be built once the page has finished drawing, so the
# ask is recorded here and answered into the same corner afterwards.
BOARD_ASKED_KEY = "board_snapshot_asked"
BOARD_SLOT_KEY = "board_snapshot_slot"
# The file once it has been made, kept so that taking it does not withdraw it.
BOARD_FILE_KEY = "board_snapshot_file_held"


def _report(tab: str) -> reporting.Report:
    """The report being gathered for one tab on this run.

    Streamlit redraws a tab from scratch on every interaction, so the figures
    are collected afresh each run and the file offered for download is always
    the page as it currently reads.
    """
    reports = st.session_state.setdefault(REPORTS_KEY, {})
    return reports.setdefault(tab, reporting.Report(tab))


def _reset_reports() -> None:
    st.session_state[REPORTS_KEY] = {}


def _tile(column, tab: str, section: str, label: str, value: str, **kwargs) -> None:
    """One metric, drawn on the page and kept for the printable report."""
    _report(tab).figure(section, label, value, str(kwargs.get("delta") or ""))
    column.metric(label, value, **kwargs)


def _kpis(tab: str, section: str, cards: list[tuple[str, str, str, str]]) -> None:
    """A KPI strip, drawn on the page and kept for the printable report."""
    for label, value, note, _accent in cards:
        _report(tab).figure(section, label, value, note)
    kpi_strip(cards)


def _said(tab: str, section: str, lines: list[str]) -> None:
    """Verdict lines, drawn as bullets and kept for the printable report."""
    for line in lines:
        _report(tab).note(section, line)
    # Escaped where it is drawn, plain in the report: these lines are money
    # sentences, and Streamlit reads a pair of dollar signs on one line as inline
    # maths - "$3,669 earned, $39 refunded" rendered as italics with both symbols
    # eaten. The report is HTML and wants the dollars as written.
    st.markdown("\n".join(f"- {_unmathed(line)}" for line in lines))


def _download_report(slot, tab: str) -> None:
    """Offer this tab's figures as a page that prints to a PDF.

    Drawn into a slot reserved before the sections run, because the file can
    only be built once they have: the button sits at the top of the tab where a
    reader looks for it, and holds what the whole tab ended up saying.
    """
    built = _report(tab)
    if not built.empty:
        slot.download_button(
            "Download report",
            data=built.html(),
            file_name=built.filename(),
            mime="text/html",
            key=f"download_{tab.lower()}",
            help="A one-page summary of this tab. Open it and print to PDF.",
        )
    # Offered whatever the summary came to. A tab whose figures failed to read is
    # still a tab full of sections, and its own emptiness is the design somebody
    # wants to look at.
    _offer_board_snapshot(slot, tab)


def _offer_board_snapshot(slot, tab: str) -> None:
    """Offer the page itself - all of it - as a PDF someone else can read.

    The report above is a summary; this is the board: every section in the
    order it is drawn, tables in the colours they are tinted, charts as the
    charts they are. A design can only be judged whole, and a reader outside
    the dashboard - an advisor, a model being asked what to change - cannot be
    handed a screen.
    """
    st.session_state[BOARD_SLOT_KEY] = (slot, tab)
    if slot.button(
        "Whole board as PDF",
        key=f"snapshot_{tab.lower()}",
        help="Every section of this page, drawn as it looks, in one PDF.",
    ):
        st.session_state[BOARD_ASKED_KEY] = tab


def _page_name(page) -> str:
    """What the file calls the page, whatever Streamlit has named it."""
    return str(getattr(page, "title", "") or "Dashboard")


def _build_board_file(recorded: board.Snapshot, tab: str) -> dict | None:
    """The board as a file to hand over: a PDF, or the page where none can be made.

    Nothing about wanting a file is worth the page the reader is looking at, so a
    board that cannot be laid out says so where the button is and leaves the
    dashboard standing.
    """
    drawn = _dt.datetime.now().strftime("%H:%M")
    try:
        with st.spinner("Drawing the board into a PDF..."):
            # Drawn once: every chart on it is an image somebody has to render,
            # and the fallback below is the same page, not another one.
            page = recorded.html()
            printed = board.to_pdf(page)
    except Exception:  # noqa: BLE001 - a board nobody can print is not a broken page
        logger.warning("The board could not be drawn into a file", exc_info=True)
        st.warning("This board could not be drawn into a file.")
        return None
    if printed:
        return {
            "tab": tab,
            "board": recorded.fingerprint(),
            "label": "Download PDF",
            "data": printed,
            "name": recorded.filename("pdf"),
            "mime": "application/pdf",
            "drawn": drawn,
            "help": "Press Whole board as PDF again for a board drawn now.",
        }
    # No PDF made here - no library where this is deployed, or a board that would
    # not lay out: the same page, as the page, for a browser to print, rather
    # than nothing at all.
    return {
        "tab": tab,
        "board": recorded.fingerprint(),
        "label": "Download board",
        "data": page.encode("utf-8"),
        "name": recorded.filename("html"),
        "mime": "text/html",
        "drawn": drawn,
        "help": "This board could not be made into a PDF here. Open it and print to PDF.",
    }


def _deliver_board_snapshot(recorded: board.Snapshot) -> None:
    """Hand over the recording of the page that has just finished drawing.

    The file stays on offer once it has been made. Taking a download is itself a
    rerun of the page, so an offer withdrawn as soon as it is accepted is an
    offer that can be pulled out from under the reader mid-download - and asking
    again would mean drawing every chart on the board a second time.

    It stays on offer only while it is still this page, though: change a filter
    or the scope and the board beside the button is not the board inside it, so
    the offer is withdrawn rather than handing over last time's figures.
    """
    asked = st.session_state.pop(BOARD_ASKED_KEY, None)
    registered = st.session_state.pop(BOARD_SLOT_KEY, None)
    if registered is None:
        return
    slot, tab = registered
    if asked and not recorded.empty:
        made = _build_board_file(recorded, asked)
        if made is not None:
            st.session_state[BOARD_FILE_KEY] = made
    held = st.session_state.get(BOARD_FILE_KEY)
    if held and (held["tab"] != tab or held.get("board") != recorded.fingerprint()):
        st.session_state.pop(BOARD_FILE_KEY, None)
        held = None
    if not held:
        return
    # Named by the minute it was drawn, because a section that reruns on its own -
    # a plan re-ticked, a chart's own control - redraws under the button without
    # the page being drawn again, and nothing here would know. The reader can see
    # whether the file is older than what they are looking at.
    stamped = held.get("drawn")
    slot.download_button(
        f"{held['label']} (drawn {stamped})" if stamped else held["label"],
        data=held["data"],
        file_name=held["name"],
        mime=held["mime"],
        key="board_snapshot_file",
        help=held["help"],
    )


def _unmathed(text: str) -> str:
    """A sentence with money in it, safe to hand to ``st.markdown``.

    Streamlit reads a pair of dollar signs on one line as inline LaTeX, so a
    sentence saying ``$399 of $1,426`` renders as maths with both symbols eaten
    and the figures between them in italics. Escaped only where it is drawn: the
    same sentence goes into the printable report, which wants the plain text.
    """
    return text.replace("$", "\\$")


def _as_frame(value: Any) -> pd.DataFrame:
    """A frame from a read that may have failed; a failure reads as no rows."""
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()
