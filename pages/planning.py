"""The Planning page: commitments, capacity and board hygiene.

Split out of app.py in Task 1C, restyled in Task 3E. The sprint-cards,
capacity/coverage tables and board-hygiene bars render as HTML through
``theme_html`` (inside ``render_shared._render_sprint_capacity`` and
``_render_hourly_capacity``, the two functions 3E rewrote); the triage
one-decision-per-card flow (``_render_cleanup``) is untouched and stays
Streamlit-native - it writes to Jira and is not this task's to rebuild.
"""

from __future__ import annotations

import html

import streamlit as st

import theme_html
from data_layer import TRIAGE_STUCK_HOURS, _engineering_context
from page_shared import TAB_ENGINEERING, _download_report
from render_shared import (
    _metrics_df,
    _one_person_instead,
    _render_cleanup,
    _render_epics,
    _render_estimate_policy,
    _render_new_and_triage,
    _render_sprint_capacity,
    _render_sprint_plan,
)


def _card(body_html: str, *, title: str, subtitle: str = "", footer: str = "") -> str:
    """Wrap a bare new-form fragment in the mockup's card chrome.

    ``theme_html``'s new-form ``table()``/``hbars()`` return only the bare
    fragment on purpose (1B.md: "no generic card() function in this file...
    composition is the caller's"). Duplicated here rather than imported -
    every page that draws its own card chrome (``pages/delivery.py``,
    ``pages/today.py``) keeps its own copy for the same reason: it is not
    part of ``render_shared``'s frozen API, and it is cheap enough that a
    shared import would only add a coupling nobody needs.
    """
    parts = [f'<div class="card"><h3 class="chart-title">{html.escape(title)}</h3>']
    if subtitle:
        parts.append(f'<p class="chart-sub">{html.escape(subtitle)}</p>')
    parts.append(body_html)
    if footer:
        parts.append(f'<p class="chart-sub" style="margin:12px 0 0">{html.escape(footer)}</p>')
    parts.append("</div>")
    return "".join(parts)


def _freshness_caption() -> str:
    """A short "how fresh is this" line for ``page_header``, without importing ``app``."""
    import time

    from data_layer import _ENGINEERING_DATA_AS_OF_KEY

    as_of = st.session_state.get(_ENGINEERING_DATA_AS_OF_KEY)
    if as_of is None:
        return "Data freshness unknown"
    age_seconds = max(0.0, time.time() - as_of)
    if age_seconds < 90:
        return "Data as of moments ago"
    minutes = int(age_seconds // 60)
    if minutes < 60:
        return f"Data as of {minutes}m ago"
    return f"Data as of {minutes // 60}h ago"


def _render_planning_page() -> None:
    """Commitments, capacity and board hygiene — the PM function, made legible."""
    theme_html.css()
    bundle, view, slot = _engineering_context()
    if _one_person_instead(bundle, view, slot):
        return

    theme_html.render(
        theme_html.page_header(
            "VinoVoss · Planning",
            _freshness_caption(),
            {"Jira": True, "GitHub": bool(bundle.github_ready)},
        ),
        theme_html.section(
            "Planning",
            "Sprints run in parallel, one board each. Where a sprint has no dates, the "
            "numbers that need them stay blank or excluded rather than reading zero.",
        ),
    )

    data = bundle.data
    df = bundle.df
    _render_new_and_triage(
        data.get("created_count_1"),
        data.get("created_count_7"),
        data.get("triage_stuck_count"),
        data.get("created_7"),
        data.get("triage_stuck"),
        TRIAGE_STUCK_HOURS,
    )
    metrics_view = _metrics_df(view.filtered, view.include_backlogs)
    st.divider()
    _render_epics(metrics_view, organization_source=df)
    st.divider()
    # Backlog-inclusive on purpose: the backlog is what this section clears out.
    _render_cleanup(view.filtered, unassigned_source=view.unscoped)
    st.divider()
    _render_estimate_policy(view.filtered)
    st.divider()
    st.subheader("Sprint Planner")
    _render_sprint_plan(df)
    st.divider()
    st.subheader("Sprint Capacity")
    _render_sprint_capacity(
        view.filtered,
        status_source_df=view.filtered,
        selected_ticket_key=None,
        triage_stuck_count=data.get("triage_stuck_count"),
    )

    theme_html.render(
        theme_html.foot(
            "Board hygiene counts are the PM function made legible, not an engineer's "
            "failing. Capacity totals exclude any sprint Jira has no dates for."
        )
    )
    _download_report(slot, TAB_ENGINEERING)
