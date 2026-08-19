"""The People page: who is doing what, ranked within their own role.

Split out of app.py in Task 1C. Like Planning, this page has no private
helpers - both sections it draws (``_render_team_overview`` and
``_render_scope_breakdown``) are also drawn by the legacy Engineering page,
so they live in ``render_shared``.
"""

from __future__ import annotations

import streamlit as st

from data_layer import _engineering_context
from page_shared import TAB_ENGINEERING, _download_report
from render_shared import (
    _metrics_df,
    _one_person_instead,
    _render_scope_breakdown,
    _render_team_overview,
)


def _render_people_page() -> None:
    """Who is doing what, compared within their own role.

    Kept apart from Delivery on purpose: Delivery counts work, this page ranks
    people, and a reader should have to choose which question they are asking.
    """
    st.caption(
        "Scores compare within a role only. A component with too little data says so "
        "instead of scoring, and every figure shows its n."
    )
    bundle, view, slot = _engineering_context()
    if _one_person_instead(bundle, view, slot):
        return

    metrics_view = _metrics_df(view.filtered, view.include_backlogs)
    _render_team_overview(metrics_view)
    st.divider()
    _render_scope_breakdown(
        view.filtered, scope=view.scope, include_backlogs=view.include_backlogs
    )
    _download_report(slot, TAB_ENGINEERING)
