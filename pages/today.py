"""The Today page: the landing page, what needs a decision right now.

Split out of app.py in Task 1C. Every helper below is private to this page
(nothing else calls them) except ``_open_pr_signals`` and ``_stalled_rows``,
which Delivery / Code also use and which therefore live in ``render_shared``.
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

import theme
from data_layer import TRIAGE_STUCK_HOURS, _NO_OWNER_NAMES, _engineering_data
from page_shared import _as_frame, _text_or
from render_shared import (
    TODAY_NO_REVIEWER_DAYS,
    TODAY_STALLED_DAYS,
    BACKLOG_STATUSES,
    _jira_ticket_url,
    _metrics_df,
    _open_pr_signals,
    _stalled_rows,
    logger,
)


def _ownerless_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The open tickets belonging to nobody, not just how many there are.

    Ownerless is worse than badly owned: no scorecard can carry it, so it is
    invisible in every per-person view on the dashboard. The rows are returned
    because a reader who is told 91 tickets have no owner needs the 91.
    """
    if df.empty or "assignee" not in df.columns:
        return df.iloc[0:0]
    names = df["assignee"].fillna("").astype(str).str.strip().str.lower()
    return df[names.isin(_NO_OWNER_NAMES)]


def _ownerless(df: pd.DataFrame) -> int:
    """How many open tickets belong to nobody."""
    return int(len(_ownerless_rows(df)))


def _estimate_coverage(df: pd.DataFrame) -> tuple[int, int]:
    """Tickets carrying an original estimate, out of those the policy asks.

    Read through ``estimate_policy`` rather than by hand, because that is where
    Delivery and Planning read the same number from: a frame arriving here has
    no ``has_estimate`` column of its own, and counting rows without one as
    estimated made this page say 100% while the other two said 77% of the very
    same tickets. The policy also exempts epics and initiatives, which hold
    other tickets' hours rather than their own.
    """
    from hygiene import estimate_policy

    if df.empty:
        return 0, 0
    scored = estimate_policy(df, BACKLOG_STATUSES)
    in_policy = scored[scored["policy_applies"].fillna(False).astype(bool)]
    if in_policy.empty:
        return 0, 0
    estimated = int(in_policy["has_estimate"].fillna(False).astype(bool).sum())
    return estimated, int(len(in_policy))


def _decision_card(column, *, chip: str, accent: str, value: str, headline: str, note: str) -> None:
    """One "somebody has to decide this" card: chip, number, what it means."""
    color = theme.ACCENTS.get(accent, theme.ACCENTS["neutral"])
    with column:
        with st.container(border=True):
            st.markdown(
                f'<div style="font-size:{theme.TYPE_META};font-weight:600;color:{color};'
                f'text-transform:uppercase;letter-spacing:.04em">{html.escape(chip)}</div>'
                f'<div style="font-size:{theme.TYPE_DISPLAY};font-weight:700;line-height:1.1;'
                f'margin:.15rem 0">{html.escape(value)}</div>'
                f'<div style="font-size:{theme.TYPE_LABEL};font-weight:600">{html.escape(headline)}</div>'
                f'<div style="font-size:{theme.TYPE_META};color:#64748b;margin-top:.35rem">'
                f"{html.escape(note)}</div>",
                unsafe_allow_html=True,
            )


def _render_attention_band(
    prs: dict[str, Any],
    *,
    github_ready: bool,
    github_error: str,
    triage_stuck: object,
    ownerless: int,
    open_total: int,
) -> None:
    """The one number the page exists to put in front of a reader, then three decisions.

    Nineteen sections at one visual level meant the finding that mattered - most
    open PRs carry no approving review - sat below twenty tiles and two pies. It
    is the hero here, and nothing else on the page is drawn at its size.
    """
    try:
        hero, a, b, c = st.columns([2.1, 1, 1, 1])

        with hero:
            with st.container(border=True):
                if not github_ready:
                    st.markdown(
                        f'<div style="font-size:{theme.TYPE_META};font-weight:600;'
                        f'color:{theme.ACCENTS["warning"]};text-transform:uppercase">GitHub unavailable</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("**PR review health cannot be read**")
                    st.caption(
                        f"({github_error})" if github_error else "Set DASHBOARD_GITHUB_TOKEN."
                    )
                else:
                    total = prs.get("total", 0) or 0
                    unapproved = prs.get("unapproved", 0) or 0
                    share = (unapproved / total) if total else 0.0
                    oldest = prs.get("oldest_unreviewed_days")
                    never_reviewed = prs.get("never_reviewed", 0) or 0

                    st.markdown(
                        f'<div style="font-size:{theme.TYPE_META};font-weight:600;'
                        f'color:{theme.ACCENTS["danger"]};text-transform:uppercase;'
                        f'letter-spacing:.04em">⚠ Needs a decision</div>'
                        f'<div style="margin:.2rem 0"><span style="font-size:44px;font-weight:700;'
                        f'line-height:1">{unapproved}</span>'
                        f'<span style="font-size:{theme.TYPE_LEAD};color:#64748b"> of {total}</span></div>'
                        f'<div style="font-size:{theme.TYPE_SECTION};font-weight:600">'
                        f"open PRs have no approving review</div>",
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(share, 0.0), 1.0))
                    oldest_text = (
                        f" Oldest unreviewed PR: **{oldest:.0f} days**." if oldest and not pd.isna(oldest) else ""
                    )
                    st.markdown(
                        f"{never_reviewed} have never been reviewed at all."
                        f"{oldest_text}"
                    )
    except Exception as e:
        logger.exception("Error rendering attention band PR section: %s", str(e))
        st.error(f"Error displaying PR metrics: {str(e)}")

        _decision_card(
            a,
            chip="Triage",
            accent="warning",
            value=_text_or(triage_stuck, "—"),
            headline=f"tickets stuck in triage > {TRIAGE_STUCK_HOURS:.0f}h",
            note="Untriaged bugs age silently — nobody has decided they matter.",
        )
        _decision_card(
            b,
            chip="Review",
            accent="danger",
            value=str(prs.get("no_reviewer_asked", 0)) if github_ready else "—",
            headline=f"PRs > {TODAY_NO_REVIEWER_DAYS:.0f} days with no reviewer asked",
            note="Nobody was requested and nobody has looked. These stall by default.",
        )
        share_note = (
            f"{ownerless / open_total:.0%} of open work belongs to nobody, so nobody's score carries it."
            if open_total
            else "No open tickets in scope."
        )
        _decision_card(
            c,
            chip="Ownership",
            accent="info",
            value=str(ownerless),
            headline="open tickets with no owner",
            note=share_note,
        )
    except Exception as e:
        logger.exception("Error rendering decision cards: %s", str(e))
        st.error(f"Error displaying decision metrics: {str(e)}")


_ACTION_QUEUE_NAMES = {
    "review": "the open PRs",
    "triage": "the triage queue",
    "ownership": "the unowned tickets",
    "stalled": "the stalled tickets",
}


def _action_queues(
    bundle: "_EngineeringData",
    board: pd.DataFrame,
    *,
    stalled: pd.DataFrame | None = None,
) -> tuple[dict[str, list[next_actions.Action]], set[str]]:
    """Every tile's number turned into the named work behind it, and what could not be read.

    A failed read leaves no key in ``data`` at all, which makes an empty queue
    and "there is nothing to do" indistinguishable. The second value keeps them
    apart: an outage announced as an all-clear is the one failure of this section
    a reader has no way of detecting.
    """
    import next_actions
    from transformations import add_ticket_health_fields

    # Taken from the caller where it has already been measured: selecting the
    # stalled rows walks every ticket's changelog, and the tile needs the same
    # rows, so computing them here again doubled the cost of drawing the page.
    if stalled is None:
        stalled, _ = _stalled_rows(board, events=getattr(bundle, "events", None))
    # The triage read is the raw Jira frame - it has never been through
    # add_ticket_health_fields, so it carries a created date and no age. Enriched
    # here rather than defaulted to zero, because "0d in triage" about a ticket
    # that has been sitting for nine days is worse than no row at all.
    triage = _as_frame(bundle.data.get("triage_stuck"))
    if not triage.empty:
        triage = add_ticket_health_fields(triage)
    unknown = set()
    if not bundle.github_ready:
        unknown.add("review")
    if bundle.data.get("triage_stuck") is None:
        unknown.add("triage")
    return {
        "review": (
            next_actions.review_actions(bundle.open_prs) if bundle.github_ready else []
        ),
        "triage": next_actions.triage_actions(triage, url_for=_jira_ticket_url),
        "ownership": next_actions.ownership_actions(
            _ownerless_rows(board), url_for=_jira_ticket_url
        ),
        "stalled": next_actions.stalled_actions(stalled, url_for=_jira_ticket_url),
    }, unknown


def _render_action_list(actions: list[next_actions.Action]) -> None:
    """The actions as numbered lines, each naming its item and linking to it."""
    lines = []
    for position, action in enumerate(actions, start=1):
        item = (
            f"[{html.escape(action.subject)}]({action.url})"
            if action.url.lower().startswith(("http://", "https://"))
            else f"`{action.subject}`"
        )
        lines.append(
            f"{position}. **{action.verb}** {item} — {html.escape(action.detail)}"
        )
    st.markdown("\n".join(lines))


def _render_action_queue(
    label: str,
    actions: list[next_actions.Action],
    *,
    empty: str,
    unknown: bool = False,
) -> None:
    """One tile's work, in an expander, with every row clickable.

    ``unknown`` is the difference between "there is none" and "we could not
    look": a queue that is empty because its source failed must not be reported
    as clear.
    """
    import next_actions

    with st.expander(f"{label} ({'unknown' if unknown else len(actions)})"):
        if unknown:
            st.warning(
                "This could not be read, so it is unknown rather than clear — "
                "try Refresh Data."
            )
            return
        if not actions:
            st.success(empty)
            return
        st.dataframe(
            next_actions.as_frame(actions),
            width="stretch",
            hide_index=True,
            column_config={
                "Open": st.column_config.LinkColumn("Open", display_text="open ↗"),
                "Why": st.column_config.TextColumn("Why", width="large"),
            },
        )


def _render_next_actions(
    queues: dict[str, list[next_actions.Action]],
    *,
    unknown: set[str] | None = None,
) -> None:
    """What to do, before any number saying why.

    The tiles above are counts, and a count is not a move: a reader was told 75
    pull requests carry no approving review and left to work out which 75 and
    whose. This says the moves, longest wait first, one problem at a time so a
    five-line list still spans review, triage, ownership and stalled work - and
    every item is the link to the thing itself.
    """
    import next_actions
    st.subheader("Do these next")
    unknown = set(unknown or ())
    top = next_actions.rank(queues, limit=5)
    if not top:
        if unknown:
            # An unreadable source is not an empty queue: presenting an outage as
            # an all-clear is the one thing here a reader cannot check.
            missing = " and ".join(
                sorted(_ACTION_QUEUE_NAMES[kind] for kind in unknown)
            )
            st.warning(
                f"Nothing found that needs a decision — but {missing} could not "
                "be read, so this list is incomplete rather than empty."
            )
        else:
            st.success("Nothing is waiting on a decision: no unreviewed PR, no untriaged ticket, nothing ownerless or stalled.")
        return
    st.caption("Ranked by how long each has been waiting, one per problem before a second from any.")
    _render_action_list(top)

    _render_action_queue(
        "Open PRs with no approving review",
        queues["review"],
        empty="Every open PR has an approving review.",
        unknown="review" in unknown,
    )
    _render_action_queue(
        f"Tickets stuck in triage > {TRIAGE_STUCK_HOURS:.0f}h",
        queues["triage"],
        empty="Nothing has been sitting in triage.",
        unknown="triage" in unknown,
    )
    _render_action_queue(
        "Open tickets with no owner",
        queues["ownership"],
        empty="Every open ticket has an owner.",
    )
    _render_action_queue(
        f"Tickets that have not moved in {TODAY_STALLED_DAYS:.0f}d",
        queues["stalled"],
        empty="Everything open has moved this month.",
    )


def _render_today_page() -> None:
    """The landing page: what needs a decision, then this week in six numbers.

    Deliberately org-wide and filter-free. A reader who lands here is asking
    "what is wrong right now", not "what is wrong within these four statuses" -
    the scoped views are the pages behind it.
    """
    st.caption(
        "What needs a decision today, then the week in numbers. "
        "Counts are telemetry, not performance — people are scored on the People page."
    )
    bundle = _engineering_data()
    # Backlog rows are left out here as they are on Delivery and Engineering by
    # default. Counting them made this page open with "16 open tickets" above a
    # Delivery page reading 14 from the same gather, with nothing on either
    # screen to reconcile them - and a landing page that disagrees with the page
    # behind it is worse than one that counts less.
    df = _metrics_df(bundle.df, include_backlogs=False)
    parked = int(len(bundle.df)) - int(len(df))

    prs = _open_pr_signals(bundle.open_prs, bundle.open_count_exact)
    ownerless = _ownerless(df)
    _render_attention_band(
        prs,
        github_ready=bundle.github_ready,
        github_error=bundle.github_error,
        triage_stuck=bundle.data.get("triage_stuck_count"),
        ownerless=ownerless,
        open_total=int(len(df)),
    )

    st.divider()
    stalled_rows, stalled_clock = _stalled_rows(df, events=bundle.events)
    queues, unreadable = _action_queues(bundle, df, stalled=stalled_rows)
    _render_next_actions(queues, unknown=unreadable)

    st.divider()
    st.subheader("This week")

    stalled = int(len(stalled_rows))
    estimated, estimable = _estimate_coverage(df)
    coverage_note = (
        f"{estimated} of {estimable} past Backlog" if estimable else "nothing to estimate"
    )
    resolved_7 = bundle.data.get("resolved_count_7")
    merged_7 = bundle.pr_count_7

    theme.kpi_strip(
        [
            ("Tickets resolved · 7d", _text_or(resolved_7, "—"), "changelog-credited", "info"),
            (
                "PRs merged · 7d",
                _text_or(merged_7, "—") if bundle.github_ready else "—",
                "bots excluded" if bundle.github_ready else "GitHub unavailable",
                "info",
            ),
            (
                "Open tickets",
                str(len(df)),
                (
                    f"Backlog excluded ({parked} parked)"
                    if parked
                    else "current JQL scope"
                ),
                "neutral",
            ),
            (
                f"Stalled {TODAY_STALLED_DAYS:.0f}d+",
                str(stalled),
                f"by {stalled_clock}, not edit age",
                "danger" if stalled else "good",
            ),
            (
                "Estimate coverage",
                f"{estimated / estimable:.0%}" if estimable else "—",
                coverage_note,
                "warning" if estimable and estimated / estimable < 0.8 else "good",
            ),
            ("Ownerless", str(ownerless), "no assignee on the board", "warning" if ownerless else "good"),
        ]
    )

    st.divider()
    st.subheader("Where open work sits")
    st.caption(
        f"{len(df)} open tickets by status — ranked, not sliced."
        + (f" {parked} Backlog ticket(s) are not counted here." if parked else "")
    )
    if df.empty:
        st.info("No open tickets returned for the current JQL.")
    else:
        theme.plot(
            theme.rank_bar(
                df["status"].fillna("(none)").astype(str).value_counts(),
                title="",
                value_label="tickets",
            )
        )
