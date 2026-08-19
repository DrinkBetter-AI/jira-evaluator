"""The Delivery page: throughput and queues, counted rather than scored.

Split out of app.py in Task 1C. ``_cycle_by_status``, ``_stale_with_masked``
and ``_stalled_count`` are private to this page (nothing else calls them);
everything else this page draws is shared with at least one other page and
lives in ``render_shared``.

Task 3C replaced the ``Stalled Nd+`` / ``Open tickets`` / ``Oldest open``
tiles with the mockup's ``Staging round-trips`` / ``Reopened · 30d`` /
``Unattributed`` (2A's ``integrity.reresolve_events``, ``org_reopen_rate``
and ``unattributed_resolutions``) and ported the stale table to
``theme_html``'s new ``table()`` form. ``_stalled_count`` stays defined
(unused by the render function now) only because ``app.py``'s
``__getattr__`` re-export list names it - see ``docs/assumptions/1C.md``.
See ``docs/assumptions/3C.md`` for every judgment call made doing this.
"""

from __future__ import annotations

import html

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


def _card(body_html: str, *, title: str, subtitle: str = "", footer: str = "") -> str:
    """Wrap a bare new-form fragment in the mockup's card chrome.

    ``theme_html.table()``'s new form (``Column``/``Cell``) returns only the
    bare ``<table>`` on purpose - 1B.md: "no generic card() function in this
    file... composition is the caller's". This matches
    ``theme_html._legacy_table_html``'s markup exactly (same class names, same
    title/subtitle/footer placement) so porting the stale table to the new
    form does not change what the page shows.
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
    """A short "how fresh is this" line for ``page_header``, without importing ``app``.

    ``app.py`` keeps ``_board_age_caption`` and the session-state key it reads
    private to itself; importing from ``app`` here would be exactly the
    circular import Task 1C's split exists to prevent. This is a page-private
    read of the same ``data_layer`` session-state key.
    """
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


def _combined_org_events(
    bundle_events: pd.DataFrame, resolved_30: pd.DataFrame | None
) -> pd.DataFrame:
    """Changelog for tickets open now, plus tickets resolved in the trailing 30 days.

    ``bundle_events`` alone only carries history for tickets the board's JQL
    still returns (``statusCategory != Done``) - a ticket that fully closed
    drops out of it entirely, and with it every entry/exit it ever made
    through a resolved-class status. ``resolved_30`` (``data.get("resolved_30")``,
    2A's ``fetch_resolved_tickets`` with ``expand=changelog``) is the other
    half: every ticket that entered a resolved status in the trailing 30 days,
    whether it is still open or fully closed now. The two are concatenated and
    deduped on ``(key, entry_id)`` so a ticket that is both (still open, but
    dipped into a resolved-class status recently) is not double-counted.

    This is what lets Reopened/Unattributed count against a real resolved
    population instead of "whatever the sidebar's open-ticket JQL still
    returns". Blind to: a ticket that fully resolved and closed more than 30
    days ago is in neither frame - invisible here the same way it is
    invisible to every other org-wide figure on this page that leans on
    ``resolved_30``.
    """
    import integrity

    parts = []
    if bundle_events is not None and not bundle_events.empty:
        parts.append(bundle_events)
    if isinstance(resolved_30, pd.DataFrame) and not resolved_30.empty:
        resolved_events = integrity.changelog_events(resolved_30)
        if not resolved_events.empty:
            parts.append(resolved_events)
    if not parts:
        return pd.DataFrame(columns=integrity.EVENT_COLUMNS)
    combined = pd.concat(parts, ignore_index=True)
    if {"key", "entry_id"} <= set(combined.columns):
        combined = combined.drop_duplicates(subset=["key", "entry_id"])
    return combined.reset_index(drop=True)


def _delta_fields(delta: "series.Delta", *, window_label: str, fmt=None) -> tuple[str, str, bool | None]:
    """A tile's ``(delta, delta_dir, delta_good)`` from a ``series.Delta``.

    Always returns a non-empty ``delta`` string with an explicit
    ``delta_dir`` - a missing prior period (``delta.magnitude is None``) says
    so in words rather than omitting the line, and ``delta_good`` stays
    whatever ``series.delta`` scored it (``None`` for missing data or a
    genuine tie), which is what keeps the tile neutral instead of green.
    """
    fmt = fmt or (lambda magnitude: f"{magnitude:.0f}")
    if delta.magnitude is None:
        return f"no prior-{window_label} data", delta.direction, delta.is_good
    sign = {"up": "+", "down": "-"}.get(delta.direction, "")
    return f"{sign}{fmt(delta.magnitude)} vs prior {window_label}", delta.direction, delta.is_good


def _staging_round_trips(
    events: pd.DataFrame, tickets: pd.DataFrame, now: pd.Timestamp, window_days: float = 30.0
) -> tuple[int, "series.Delta"]:
    """Org-wide count of "entered a resolved status, left, came back" - and its trend.

    Scoped to ``bundle.events`` alone (currently-open tickets), not the wider
    ``_combined_org_events`` the Reopened/Unattributed tiles use, on purpose:
    that frame carries each open ticket's *entire* history, unbounded by any
    fetch window, so both the current and the prior 30-day slice are read off
    the same population and a real delta can be computed - mixing in
    ``resolved_30`` (which only ever covers the trailing 30 days) would make
    the prior slice systematically thinner than the current one and turn
    every delta into a fake "improvement". The trade: a ticket that bounced
    through staging and has since fully closed is invisible here - see
    ``docs/assumptions/3C.md``.
    """
    import integrity
    import series

    def _count(evts: pd.DataFrame, moment: pd.Timestamp) -> int:
        if evts is None or evts.empty:
            return 0
        bounced = integrity.reresolve_events(evts, tickets, window_days=window_days, now=moment)
        return int(bounced["reopens"].sum()) if not bounced.empty else 0

    current = _count(events, now)
    prior_cutoff = now - pd.Timedelta(days=window_days)
    prior_events = (
        events[events["ts"] < prior_cutoff] if events is not None and not events.empty else events
    )
    prior = _count(prior_events, prior_cutoff)
    return current, series.delta(float(current), float(prior), higher_is_better=False)


def _reopen_note(rate: "integrity.OrgReopenRate") -> str:
    """The denominator, always - "—" (never "0%") when nothing resolved in the window."""
    if rate.share is None:
        return f"— of {rate.resolved_count} resolved"
    return f"{rate.share:.1%} of {rate.resolved_count} resolved"


def _resolved_7d_delta(
    resolved_7: object, resolved_30: pd.DataFrame | None, now: pd.Timestamp
) -> "series.Delta":
    """Trailing-7d resolved count vs. the 7 days before that, approximated from ``resolved_30``.

    ``resolved_7`` itself (``data.get("resolved_count_7")``) is the exact,
    uncapped org-wide count and is what the tile's headline value shows. There
    is no equivalent exact fetch for "resolved 7-14 days ago", so the prior
    side is read from ``resolved_30`` (already fetched for this page, capped
    at ``max_results`` the same as every other figure derived from it) - an
    approximation, not a second exact count. Documented in
    ``docs/assumptions/3C.md`` rather than left silent.
    """
    import series

    prior = None
    if (
        isinstance(resolved_30, pd.DataFrame)
        and not resolved_30.empty
        and "status_category_changed_date" in resolved_30.columns
    ):
        changed = pd.to_datetime(resolved_30["status_category_changed_date"], utc=True, errors="coerce")
        start = now - pd.Timedelta(days=14)
        end = now - pd.Timedelta(days=7)
        mask = changed.notna() & (changed >= start) & (changed < end)
        prior = float(mask.sum())
    current = None
    if resolved_7 is not None:
        try:
            current = float(resolved_7) if not pd.isna(resolved_7) else None
        except (TypeError, ValueError):
            current = None
    return series.delta(current, prior, higher_is_better=True)


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
    bundle, view, slot = _engineering_context()
    if _one_person_instead(bundle, view, slot):
        return

    theme_html.render(
        theme_html.page_header(
            "VinoVoss · Delivery",
            _freshness_caption(),
            {"Jira": True, "GitHub": bool(bundle.github_ready)},
        ),
        theme_html.section(
            "Delivery",
            "Org-wide throughput and the queues behind it. Counts here are telemetry — "
            "people are scored on the People page."
            + (f" {exclusion_caption}" if exclusion_caption else ""),
        ),
    )

    # One scope for the whole page: tiles and charts reading the org-wide frame
    # while the tables below read the sidebar's selection left the top half
    # ignoring every filter while labelling itself "current scope". The backlog
    # choice belongs in that scope too, or Delivery counts a board Today does not.
    df = _metrics_df(view.filtered, view.include_backlogs)
    data = bundle.data
    cycle = _cycle_by_status(df, events=bundle.events)
    overall_median = None
    if not cycle.empty:
        in_progress = cycle[cycle["status"].str.lower().eq("in progress")]
        overall_median = float(in_progress["median_days"].iloc[0]) if not in_progress.empty else None

    resolved_7 = data.get("resolved_count_7")
    resolved_30 = data.get("resolved_30")
    now = pd.Timestamp.now(tz="UTC")

    resolved_text, resolved_dir, resolved_good = _delta_fields(
        _resolved_7d_delta(resolved_7, resolved_30, now), window_label="week"
    )

    # No changelog anywhere - a genuinely empty board or a cold-start read -
    # is a different fact than "zero staging round-trips this org has ever
    # had", and the tiles below have to say so rather than print a 0 that
    # reads as a clean bill of health.
    no_org_history = (bundle.events is None or bundle.events.empty) and (
        resolved_30 is None or (isinstance(resolved_30, pd.DataFrame) and resolved_30.empty)
    )
    if no_org_history:
        round_trips_tile = theme_html.Tile(
            "Staging round-trips", "—", note="no changelog data read"
        )
        reopened_tile = theme_html.Tile("Reopened · 30d", "—", note="no changelog data read")
        unattributed_tile = theme_html.Tile("Unattributed", "—", note="no changelog data read")
    else:
        import integrity
        import series

        round_trips, round_trips_delta = _staging_round_trips(bundle.events, bundle.df, now)
        rt_text, rt_dir, rt_good = _delta_fields(round_trips_delta, window_label="30d")
        round_trips_tile = theme_html.Tile(
            "Staging round-trips",
            str(round_trips),
            delta=rt_text,
            delta_dir=rt_dir,
            delta_good=rt_good,
            note="entered a resolved status twice+",
        )

        combined_events = _combined_org_events(bundle.events, resolved_30)
        reopen_rate = integrity.org_reopen_rate(combined_events, bundle.df, window_days=30.0, now=now)
        # There is no reliable prior-30d slice for this: an accurate "resolved
        # in the window" denominator needs resolved_30 (the last 30 days
        # only), so a matching prior window would need the 30 days before
        # that - data this page has no fetch for. Reporting a number built
        # from a systematically thinner prior population would read as a
        # trend that is really just a coverage gap, so the delta is honestly
        # "no prior data" (renders neutral, never green) rather than invented.
        reopen_delta = series.delta(reopen_rate.share, None, higher_is_better=False)
        reopen_text, reopen_dir, reopen_good = _delta_fields(
            reopen_delta, window_label="30d", fmt=lambda m: f"{m * 100:.1f}pp"
        )
        reopened_tile = theme_html.Tile(
            "Reopened · 30d",
            str(reopen_rate.count),
            delta=reopen_text,
            delta_dir=reopen_dir,
            delta_good=reopen_good,
            note=_reopen_note(reopen_rate),
        )

        unattributed_rows = integrity.unattributed_resolutions(
            combined_events, window_days=30.0, now=now
        )
        unattributed_count = int(len(unattributed_rows))
        unattributed_delta = series.delta(float(unattributed_count), None, higher_is_better=False)
        unattr_text, unattr_dir, unattr_good = _delta_fields(unattributed_delta, window_label="30d")
        unattributed_tile = theme_html.Tile(
            "Unattributed",
            str(unattributed_count),
            delta=unattr_text,
            delta_dir=unattr_dir,
            delta_good=unattr_good,
            note="no changelog — credited to nobody",
        )

    # No ``write=True`` here, though task 5A fixed the crash that used to
    # follow it (docs/assumptions/3C.md, docs/assumptions/5A.md): the pattern
    # 1B.md recommends still holds regardless - build the fragment, hand it
    # to render() explicitly, so this section's tiles land in the same
    # render() call as anything else the page draws around them.
    tiles_html = theme_html.tiles(
        [
            theme_html.Tile(
                "Resolved · 7d",
                _text_or(resolved_7, "—"),
                delta=resolved_text if resolved_7 is not None else None,
                delta_dir=resolved_dir if resolved_7 is not None else None,
                delta_good=resolved_good if resolved_7 is not None else None,
                note="org-wide — the one tile the sidebar does not narrow",
            ),
            theme_html.Tile(
                "Median In-Progress",
                f"{overall_median:.1f}" if overall_median is not None else "—",
                unit="d" if overall_median is not None else "",
                note="days in In Progress, from real transitions",
            ),
            round_trips_tile,
            reopened_tile,
            unattributed_tile,
        ],
        tab=TAB_ENGINEERING,
        section="Delivery",
    )
    theme_html.render(tiles_html)

    left, right = st.columns(2)
    with left:
        team_view = add_team(df, TEAM_PROJECTS, TEAM_PEOPLE)
        summary = team_summary(team_view)
        if not summary.empty:
            theme_html.hbars(
                [(row.team, float(row.open), str(int(row.open))) for row in summary.itertuples()],
                title="Where open work sits, by team",
                subtitle=f"{len(df)} open tickets — ranked, never sliced",
                tab=TAB_ENGINEERING,
                section="Where open work sits, by team",
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
                tab=TAB_ENGINEERING,
                section="Cycle time by status",
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
        # New table() form: Column/Cell, ranked on status_age_days (never
        # idle_days) with idle_days still visible as "last touched" and
        # masked_days as the gap between them - see 2A's render-call-site
        # note and docs/assumptions/3C.md. What this shows is unchanged from
        # the legacy call above; only the construction is new.
        columns = [
            theme_html.Column("Ticket", "link"),
            theme_html.Column("Summary", "text"),
            theme_html.Column("Owner", "text"),
            theme_html.Column("Status", "text"),
            theme_html.Column("Status age", "num"),
            theme_html.Column("Last touched", "num"),
            theme_html.Column("Masked", "num"),
        ]
        rows = [
            [
                theme_html.Cell(row.url),
                theme_html.Cell(row.summary),
                theme_html.Cell(row.assignee),
                theme_html.Cell(row.status),
                theme_html.Cell(row.status_age_days),
                theme_html.Cell(row.idle_days),
                theme_html.Cell(row.masked_days),
            ]
            for row in stale.itertuples()
        ]
        table_html = theme_html.table(
            columns, rows, tab=TAB_ENGINEERING, section="Stale & abandoned"
        )
        theme_html.render(
            _card(
                table_html,
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
        )

    theme_html.render(
        theme_html.foot(
            "Counts here are telemetry, not a verdict — pair a spike with the evidence "
            "before drawing a conclusion. People are scored on the People page."
        )
    )
    st.divider()
    _render_priority_queue(df, include_backlogs=view.include_backlogs)
    _download_report(slot, TAB_ENGINEERING)
