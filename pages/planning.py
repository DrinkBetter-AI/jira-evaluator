"""The Planning page: commitments, capacity and board hygiene.

Split out of app.py in Task 1C. This page has no private helpers of its own
- everything it draws is also drawn by at least one other page (mostly the
legacy Engineering page), so those helpers live in ``render_shared``.
"""

from __future__ import annotations

import streamlit as st

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


def _render_planning_page() -> None:
    """Commitments, capacity and board hygiene — the PM function, made legible."""
    st.caption(
        "Sprints run in parallel, one board each. Where a sprint has no dates, the "
        "numbers that need them stay blank rather than reading zero."
    )
    bundle, view, slot = _engineering_context()
    if _one_person_instead(bundle, view, slot):
        return

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
        view.filtered, status_source_df=view.filtered, selected_ticket_key=None
    )
    _download_report(slot, TAB_ENGINEERING)
