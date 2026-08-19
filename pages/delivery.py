"""The Delivery page: throughput and queues, counted rather than scored.

Split out of app.py in Task 1C. ``_cycle_by_status``, ``_stale_with_masked``
and ``_stalled_count`` are private to this page (nothing else calls them);
everything else this page draws is shared with at least one other page and
lives in ``render_shared``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import theme_html
from data_layer import _engineering_context
from page_shared import TAB_ENGINEERING, _download_report, _log_stage, _text_or
from render_shared import (
    TODAY_STALLED_DAYS,
    _exclude_repos,
    _jira_ticket_url,
    _metrics_df,
    _one_person_instead,
    _render_priority_queue,
    _stalled_rows,
    _truncation_note,
    logger,
)


def _stalled_count(
    df: pd.DataFrame, events: pd.DataFrame | None = None
) -> tuple[int, str]:
    """How many open tickets have not *moved* in TODAY_STALLED_DAYS."""
    rows, clock = _stalled_rows(df, events)
    return int(len(rows)), clock


def _cycle_by_status(
    df: pd.DataFrame, events: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Median days a ticket sits in each status, from real transitions.

    This is the chart that reframes the bottleneck: when the slowest statuses
    are review stages, the constraint is attention, not build capacity - and no
    amount of pressure on authors moves it.

    ``events`` behaves as in ``_stalled_rows``: pass the bundle's already-flat
    changelog to skip a re-parse, or omit it to parse ``df["changelog"]``
    directly (what direct callers, chiefly tests, still do).
    """
    import integrity

    try:
        with _log_stage("changelog:cycle_by_status"):
            resolved_events = events if events is not None else integrity.changelog_events(df)
            cycle = integrity.cycle_time(resolved_events, df)
        detail = cycle.detail
    except Exception:  # noqa: BLE001 - an unparseable changelog must not blank the page
        logger.exception("cycle_time failed; the by-status chart is omitted")
        return pd.DataFrame(columns=["status", "median_days", "n"])
    if detail.empty:
        return pd.DataFrame(columns=["status", "median_days", "n"])
    closed = detail[~detail["is_open"].fillna(False).astype(bool)]
    if closed.empty:
        return pd.DataFrame(columns=["status", "median_days", "n"])
    out = (
        closed.groupby("status")
        .agg(median_days=("days", "median"), n=("days", "size"))
        .reset_index()
        .sort_values("median_days", ascending=False)
    )
    # Below five stays out: a median of two intervals is an anecdote wearing math.
    return out[out["n"] >= 5].reset_index(drop=True)


def _stale_with_masked(
    df: pd.DataFrame,
    top_n: int = 12,
    min_status_age: float = TODAY_STALLED_DAYS,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The stale queue with the gaming made visible per row.

    ``masked_days`` is apparent freshness from edits that moved no work. A row
    with status age 186 and last-touched 2 is a ticket that has not moved in six
    months and looks alive - the exact shape a label-edit sweep produces.

    Only tickets past the clock the stalled tile uses qualify: ranking by status
    age alone lists a healthy board's newest tickets under a heading that says
    abandoned.

    ``events`` behaves as in ``_stalled_rows``: the bundle's already-flat
    changelog when the caller has one, otherwise parsed from ``df`` itself.
    """
    import integrity

    if df.empty:
        return pd.DataFrame()
    try:
        with _log_stage("changelog:stale_with_masked"):
            resolved_events = events if events is not None else integrity.changelog_events(df)
            ages = integrity.status_age_days(df, resolved_events)
    except Exception:  # noqa: BLE001
        logger.exception("status_age_days failed; the stale table is omitted")
        # An unreadable history is not a clean board, and the caller has to be
        # able to tell the two apart before it congratulates anybody.
        failed = pd.DataFrame()
        failed.attrs["stale_unreadable"] = True
        return failed
    if ages.empty:
        return pd.DataFrame()
    ages = ages[(ages["status_age_days"] >= min_status_age).fillna(False)]
    if ages.empty:
        return pd.DataFrame()
    total = int(len(ages))
    ages = ages.sort_values("status_age_days", ascending=False).head(top_n)
    # What was cut travels with the frame, so the card can tell the reader.
    ages.attrs["stale_total"] = total
    summaries = df.set_index("key").get("summary", pd.Series(dtype=object))
    ages["summary"] = ages["key"].map(summaries).fillna("")
    ages["ticket"] = ages["key"] + "  " + ages["summary"].astype(str).str.slice(0, 60)
    ages["url"] = ages["key"].map(_jira_ticket_url)
    return ages


def _render_delivery_page() -> None:
    """What finished and how long it took, in the mockup's shape.

    Counts, never verdicts: people are scored on the People page, and the
    caption says so because the distinction is the whole design. The repo
    exclusion caption rides along wherever PR-derived figures appear.
    """
    from teams import add_team, team_summary

    from render_shared import TEAM_PEOPLE, TEAM_PROJECTS

    theme_html.css()
    _, exclusion_caption = _exclude_repos()
    st.caption(
        "Org-wide throughput and the queues behind it. Counts here are telemetry — "
        "people are scored on the People page."
        + (f" {exclusion_caption}" if exclusion_caption else "")
    )
    bundle, view, slot = _engineering_context()
    if _one_person_instead(bundle, view, slot):
        return

    # One scope for the whole page: tiles and charts reading the org-wide frame
    # while the tables below read the sidebar's selection left the top half
    # ignoring every filter while labelling itself "current scope". The backlog
    # choice belongs in that scope too, or Delivery counts a board Today does not.
    df = _metrics_df(view.filtered, view.include_backlogs)
    data = bundle.data
    stalled, stalled_clock = _stalled_count(df, events=bundle.events)
    cycle = _cycle_by_status(df, events=bundle.events)
    overall_median = None
    if not cycle.empty:
        in_progress = cycle[cycle["status"].str.lower().eq("in progress")]
        overall_median = float(in_progress["median_days"].iloc[0]) if not in_progress.empty else None

    resolved_7 = data.get("resolved_count_7")
    oldest = float(df["ticket_age_days"].max()) if "ticket_age_days" in df.columns and len(df) else None

    theme_html.tiles(
        [
            (
                "Resolved · 7d",
                _text_or(resolved_7, "—"),
                "org-wide — the one tile the sidebar does not narrow",
                "info",
            ),
            (
                "Median In-Progress",
                f"{overall_median:.1f}d" if overall_median is not None else "—",
                "days in In Progress, from real transitions",
                "info",
            ),
            (
                f"Stalled {TODAY_STALLED_DAYS:.0f}d+",
                str(stalled),
                # The boilerplate promise only holds when the clock kept it:
                # "by edit age, never edit age" is what saying both produced.
                f"by {stalled_clock}"
                + (", never edit age" if stalled_clock == "status age" else ""),
                "danger" if stalled else "good",
            ),
            ("Open tickets", str(len(df)), "current scope", "neutral"),
            (
                "Oldest open",
                f"{oldest:.0f}d" if oldest else "—",
                "age of the oldest ticket in scope",
                "warning" if oldest and oldest > 180 else "neutral",
            ),
        ]
    )

    left, right = st.columns(2)
    with left:
        team_view = add_team(df, TEAM_PROJECTS, TEAM_PEOPLE)
        summary = team_summary(team_view)
        if not summary.empty:
            theme_html.hbars(
                [(row.team, float(row.open), str(int(row.open))) for row in summary.itertuples()],
                title="Where open work sits, by team",
                subtitle=f"{len(df)} open tickets — ranked, never sliced",
            )
    with right:
        if cycle.empty:
            st.info("Not enough closed status intervals to draw cycle time yet.")
        else:
            slow_review = cycle.head(3)["status"].str.contains("review", case=False).sum()
            footer = (
                "The slowest statuses are review, not build. The bottleneck is "
                "attention, not capacity."
                if slow_review >= 2
                else ""
            )
            theme_html.hbars(
                [
                    (row.status, float(row.median_days), f"{row.median_days:.1f}")
                    for row in cycle.head(8).itertuples()
                ],
                title="Cycle time by status",
                subtitle="Median days a ticket sits before it moves on (n ≥ 5)",
                footer=footer,
                severity=True,
            )

    stale = _stale_with_masked(df, events=bundle.events)
    if stale.attrs.get("stale_unreadable"):
        st.warning(
            "Status history could not be read, so the stale queue is omitted — "
            "the stalled count above fell back to edit age."
        )
    elif stale.empty and df.empty:
        st.info("No open tickets in this scope.")
    elif stale.empty:
        st.success(
            "Nothing stale in scope: every open ticket here has changed status "
            f"within {TODAY_STALLED_DAYS:.0f} days."
        )
    else:
        theme_html.table(
            stale,
            [
                ("url", "Ticket", "link"),
                ("summary", "Summary", "text"),
                ("assignee", "Owner", "text"),
                ("status", "Status", "text"),
                ("status_age_days", "Status age", "num"),
                ("idle_days", "Last touched", "num"),
                ("masked_days", "Masked", "num"),
            ],
            title="Stale & abandoned — by status age",
            subtitle=(
                "Days since the ticket actually moved. Masked is apparent freshness "
                "from edits that moved no work — a label edit resets last-touched, never this."
            ),
            footer=" ".join(
                part
                for part in (
                    _truncation_note(int(stale.attrs.get("stale_total", len(stale))), len(stale)),
                    "The innocent reading: a comment or a linked ticket is a real edit. The "
                    "pattern worth asking about is a large masked figure across many tickets "
                    "in the same week.",
                )
                if part
            ),
        )

    st.divider()
    _render_priority_queue(df, include_backlogs=view.include_backlogs)
    _download_report(slot, TAB_ENGINEERING)
