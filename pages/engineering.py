"""The legacy combined Engineering page (``/engineering``), kept whole at the
address people already have (Slack links, bookmarks).

Split out of app.py in Task 1C. Everything this page draws that another page
also draws lives in ``render_shared``; ``_render_metrics``, ``_render_mix``,
``_render_bubble_chart``, ``_contribution_ranking``, ``_render_resolved_summary``,
``_pr_review_label`` and ``_render_pr_section`` are private to this page - no
other page reaches them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import theme
from data_layer import (
    CREDS_PATH,
    PROFILE_NAME,
    TRIAGE_STUCK_HOURS,
    _engineering_data,
    _engineering_filters,
)
from jira_client import JiraClient
from page_shared import TAB_ENGINEERING, _download_report, _kpis, _tile
from render_shared import (
    BACKLOG_STATUSES,
    BULK_ACTION_DEFAULT_LIMIT,
    ENGINEERING_PAGE_TITLE,
    MIX_SLICE_LIMIT,
    TEAM_PEOPLE,
    TEAM_PROJECTS,
    _apply_action_with_audit,
    _clear_page_caches,
    _known_project_keys,
    _metric_value,
    _metrics_df,
    _people_only,
    _render_cleanup,
    _render_epics,
    _render_estimate_policy,
    _render_individual_page,
    _render_new_and_triage,
    _render_pr_hygiene,
    _render_priority_queue,
    _render_scope_breakdown,
    _render_sprint_capacity,
    _render_sprint_plan,
    _render_stale_cleanup,
    _render_status_pills,
    _render_team_overview,
    _render_ticket_quality,
)

_PRIORITY_BUCKET_MAP = {
    "low": "Normal",
    "lowest": "Normal",
    "normal": "Normal",
    "medium": "Normal",
    "high": "High",
    "highest": "Urgent",
    "urgent": "Urgent",
}
_BUCKET_COLORS = {"Normal": "#2ECC71", "High": "#F5A623", "Urgent": "#E74C3C"}

@st.fragment
def _render_metrics(
    df: pd.DataFrame,
    include_backlogs: bool = False,
    *,
    unassigned_source: pd.DataFrame | None = None,
) -> None:
    """The headline numbers for the current scope.

    ``unassigned_source`` is the same data before the assignee scope filter. No
    unowned ticket can match a selected assignee, so within Team or Individual
    scope the unassigned count is structurally zero - a green zero reading as
    "nothing is ownerless" when the truth is "ownerless work is out of view".
    Counting it from the pre-scope frame, and saying so on the card, keeps the
    number honest.
    """
    from hygiene import estimate_policy

    metrics_df = _metrics_df(df, include_backlogs)
    total_open = int(len(metrics_df))
    avg_idle = float(metrics_df["idle_days"].mean()) if total_open else 0.0
    max_idle = float(metrics_df["idle_days"].max()) if total_open else 0.0
    oldest = float(metrics_df["ticket_age_days"].max()) if total_open else 0.0

    # Estimate coverage is scored by hygiene.estimate_policy, the same rule the
    # Policy Compliance section two sections down uses, rather than by a second
    # count that happens to live here.
    #
    # It used to be its own thing and it was wrong twice over. It looked first
    # for an "estimate_seconds" column, which no code path puts on a raw ticket
    # frame - jira_client emits "original_estimate_sec" - so the branch was dead
    # and every run fell through to "is there any text in original_estimate",
    # counted over every open ticket. That denominator includes epics and
    # initiatives, which hold other tickets' hours rather than their own and
    # which the policy deliberately exempts, and Backlog tickets, which have not
    # been asked for an estimate yet. The headline said one number and Policy
    # Compliance said another with nothing on the page to explain the gap.
    scored = estimate_policy(metrics_df, BACKLOG_STATUSES)
    in_policy = (
        scored[scored["policy_applies"].fillna(False).astype(bool)]
        if not scored.empty
        else scored
    )
    estimate_scope = int(len(in_policy))
    estimated_tickets = (
        int(in_policy["has_estimate"].fillna(False).astype(bool).sum())
        if estimate_scope
        else 0
    )
    estimate_coverage_pct = (
        (estimated_tickets / estimate_scope * 100.0) if estimate_scope else 0.0
    )

    _LATE_STAGE_STATUSES = {"IN DEV ENV", "Review in Staging", "Ready for Production"}
    _STALE_THRESHOLD_DAYS = 6
    stale_late_stage = 0
    if "status" in metrics_df.columns and "idle_days" in metrics_df.columns and total_open:
        stale_late_stage = int(
            (
                metrics_df["status"].fillna("").astype(str).isin(_LATE_STAGE_STATUSES)
                & (metrics_df["idle_days"] > _STALE_THRESHOLD_DAYS)
            ).sum()
        )

    idle_30 = 0
    if total_open:
        idle_30 = int(pd.to_numeric(metrics_df["idle_days"], errors="coerce").fillna(0).ge(30).sum())

    out_of_scope = unassigned_source is not None
    owner_df = _metrics_df(unassigned_source, include_backlogs) if out_of_scope else metrics_df
    unassigned = 0
    if len(owner_df):
        owners = owner_df["assignee"].fillna("").astype(str).str.strip().str.lower()
        unassigned = int(owners.isin({"", "unassigned", "none"}).sum())

    _kpis(
        TAB_ENGINEERING,
        "Ticket health",
        [
            ("Open tickets", f"{total_open}", "current scope", "neutral"),
            (
                "Stalled 30d+",
                f"{idle_30}",
                f"{idle_30 / total_open * 100:.0f}% of open" if total_open else "—",
                "danger" if idle_30 else "good",
            ),
            (
                "Unassigned",
                f"{unassigned}",
                "no owner set, so outside this scope" if out_of_scope else "no owner set",
                "warning" if unassigned else "good",
            ),
            (
                "Estimate coverage",
                f"{estimate_coverage_pct:.0f}%",
                f"of {estimate_scope} past Backlog; epics exempt",
                "good" if estimate_coverage_pct >= 80 else "warning",
            ),
            (
                "Stale late stage",
                f"{stale_late_stage}",
                "in dev/staging/prod, idle >6d",
                "danger" if stale_late_stage else "good",
            ),
            ("Oldest ticket", f"{oldest:.0f}d", f"avg idle {avg_idle:.0f}d · max {max_idle:.0f}d", "info"),
        ]
    )

    if "status" in metrics_df.columns and total_open:
        _render_status_pills(metrics_df["status"])


@st.fragment
def _render_mix(df: pd.DataFrame) -> None:
    """Composition of the tickets currently in view, as a share rather than a count.

    The other charts answer "how much" and "who"; this answers "of what" - which
    part of the work a reader is looking at before they read any table. It shows
    the filtered scope, Backlog statuses included only when the sidebar is, so it
    always agrees with the headline tiles above it.
    """
    from teams import add_team

    st.subheader("Ticket Composition")
    if df.empty:
        st.info("No tickets in the current scope.")
        return

    # Team is derived, not a Jira field, so it has to be attached before it can
    # be offered as a slice.
    df = add_team(df, TEAM_PROJECTS, TEAM_PEOPLE)
    dimensions = {
        "Status": "status",
        "Team": "team",
        "Priority": "priority",
        "Issue type": "issue_type",
        "Assignee": "assignee",
        "Project": "project_key",
    }
    available = {
        label: column for label, column in dimensions.items() if column in df.columns
    }
    if not available:
        st.info("No breakdown fields available in the current data.")
        return

    label = st.radio(
        "Break down by",
        options=list(available.keys()),
        horizontal=True,
        key="mix_dimension",
    )
    st.caption(
        "Follows the sidebar scope and filters, including whether Backlog "
        "statuses are shown."
    )
    counts = (
        df[available[label]]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace({"": "Unknown"})
        .value_counts()
    )
    # Ranked bars, not a pie. Broken down by Assignee or Status this routinely
    # ran past twenty categories, at which point the slices were slivers, the
    # labels were drawn over each other and the legend scrolled - and even at
    # six a reader is left comparing angles, which is the comparison people are
    # worst at. The tail still collapses into one honest "Other" row, and the
    # table beside it is built from the same collapsed series so the two cannot
    # disagree.
    top = theme.ranked(counts, top_n=MIX_SLICE_LIMIT)

    mix = top.rename_axis(label).reset_index(name="tickets")
    figure = theme.rank_bar(
        top,
        title=f"Tickets by {label.lower()}",
        value_label="tickets",
        top_n=MIX_SLICE_LIMIT,
    )
    left, right = st.columns([3, 2])
    theme.plot(figure, into=left, width="stretch")
    right.dataframe(
        mix.assign(share=(mix["tickets"] / mix["tickets"].sum() * 100).round(1)),
        width="stretch",
        hide_index=True,
        column_config={
            "tickets": st.column_config.NumberColumn("Tickets"),
            "share": st.column_config.NumberColumn("Share %", format="%.1f"),
        },
    )


def _render_bubble_chart(
    df: pd.DataFrame,
    color_by: str = "priority",
    agg_priority: bool = False,
    chart_key: str = "bubble_chart",
) -> str | None:
    import plotly.express as px

    if df.empty:
        st.info("No data available for staleness bubble chart.")
        return None

    plot_df = df.copy()

    STATUS_ORDER = [
        "Backlog",
        "DISCUSSION NEEDED",
        "To Do",
        "In Progress",
        "IN DEV ENV",
        "Review in Staging",
        "Code Review",
        "Ready for Production",
    ]
    statuses = plot_df["status"].fillna("Unknown")
    # Any status not in the fixed list gets appended at the top.
    extra = [s for s in statuses.unique() if s not in STATUS_ORDER]
    full_order = STATUS_ORDER + extra
    status_to_y = {s: i for i, s in enumerate(full_order)}
    rng = np.random.default_rng(seed=42)
    plot_df["y_jitter"] = (
        statuses.map(status_to_y).astype(float)
        + rng.uniform(-0.35, 0.35, size=len(plot_df))
    )
    plot_df["status_label"] = statuses

    age = plot_df["ticket_age_days"].clip(lower=1)
    plot_df["bubble_size"] = ((age - age.min()) / (age.max() - age.min() + 1e-9) * 31 + 3).round(1)

    plot_df["marker_symbol"] = (
        plot_df["issue_type"].fillna("").astype(str).str.strip().str.lower()
        .map(lambda t: "triangle-up" if t == "epic" else "circle")
    )

    if agg_priority and color_by == "priority":
        plot_df["priority_bucket"] = (
            plot_df["priority"].fillna("none").astype(str).str.strip().str.lower()
            .map(_PRIORITY_BUCKET_MAP)
            .fillna("Normal")
        )
        fig = px.scatter(
            plot_df,
            x="idle_days",
            y="y_jitter",
            size="bubble_size",
            color="priority_bucket",
            color_discrete_map=_BUCKET_COLORS,
            category_orders={"priority_bucket": ["Normal", "High", "Urgent"]},
            custom_data=["key", "summary", "assignee", "status_label", "priority", "ticket_age_days", "idle_days", "issue_type"],
            # Commented out, this chart drew the word "undefined" where its title
            # belongs: theme.plot set a title font, that brought a title object
            # into being with no text in it, and Streamlit printed the missing
            # text. theme.plot no longer does that, and the chart says what it is.
            title="Staleness vs workflow status (priorities grouped)",
            labels={"idle_days": "Idle Days", "y_jitter": "Status", "priority_bucket": "Priority"},
            size_max=34,
            opacity=0.3,
        )
    else:
        fig = px.scatter(
            plot_df,
            x="idle_days",
            y="y_jitter",
            size="bubble_size",
            color=color_by,
            custom_data=["key", "summary", "assignee", "status_label", "priority", "ticket_age_days", "idle_days", "issue_type"],
            title="Staleness vs workflow status",
            labels={"idle_days": "Idle Days", "y_jitter": "Status"},
            size_max=34,
            opacity=0.3,
        )

    for trace in fig.data:
        custom_rows = getattr(trace, "customdata", None)
        if custom_rows is None:
            continue
        trace.marker.symbol = [
            "triangle-up" if str(row[7]).strip().lower() == "epic" else "circle"
            for row in custom_rows
        ]

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br>"
            "Assignee: %{customdata[2]}<br>"
            "Status: %{customdata[3]}<br>"
            "Priority: %{customdata[4]}<br>"
            "Age: %{customdata[5]:.1f} days<br>"
            "Idle: %{customdata[6]:.1f} days<br>"
            "Type: %{customdata[7]}"
            "<extra></extra>"
        )
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(status_to_y.values()),
        ticktext=list(status_to_y.keys()),
        title="Status",
    )
    fig.update_layout(height=560)
    event = theme.plot(fig, width="stretch", on_select="rerun", key=chart_key)
    points = (event or {}).get("selection", {}).get("points", [])
    if points:
        return str(points[0].get("customdata", [None])[0])
    return None

def _contribution_ranking(
    labels: pd.Series, value_name: str, title: str, unavailable: bool = False
) -> None:
    """Render a 'who did how much' ranking from a series of names, or a note if empty.

    This was a pie, and with twenty-three people in the window it was a ring of
    slivers over a legend nobody could read - which is the shape "who is doing
    the work" was being asked in. theme.rank_bar answers it as a sorted list
    with the numbers written on the bars.

    ``unavailable`` distinguishes a failed lookup from a genuinely empty window so
    an errored fetch doesn't masquerade as an authoritative "nobody did anything".
    """
    counts = labels.value_counts()
    if counts.empty:
        if unavailable:
            st.caption(f"Could not load {value_name} \u2014 try Refresh Data.")
        else:
            st.caption(f"No {value_name} in the last 30 days.")
        return
    theme.plot(
        theme.rank_bar(counts, title=title, value_label=value_name),
        width="stretch",
    )


@st.fragment
def _render_resolved_summary(
    ticket_count_7: int | None,
    ticket_count_30: int | None,
    resolved_30: pd.DataFrame | None,
    pr_count_7: int | None,
    pr_count_30: int | None,
    merged_prs: pd.DataFrame,
    github_ready: bool,
    github_error: str = "",
) -> None:
    """Login landing snapshot: tickets and PRs resolved in the last 7 / 30 days,
    with a ranking of who resolved tickets and who merged PRs. Tile counts come
    from server-side counts (Jira's approximate count for tickets, GitHub's exact
    issueCount for PRs) and are org-wide, bots included; the dataframes drive only
    the rankings, which are people and therefore are not. A ``None`` count means
    the lookup failed and renders as "—", distinct from a genuine 0."""
    st.subheader("Resolved in the Last 7 / 30 Days")

    c1, c2, c3, c4 = st.columns(4)
    shipped = "Resolved in the last 7 / 30 days"
    _tile(
        c1, TAB_ENGINEERING, shipped, "Tickets resolved (7d)", _metric_value(ticket_count_7)
    )
    _tile(
        c2,
        TAB_ENGINEERING,
        shipped,
        "Tickets resolved (30d)",
        _metric_value(ticket_count_30),
    )
    _tile(c3, TAB_ENGINEERING, shipped, "PRs merged (7d)", _metric_value(pr_count_7))
    _tile(c4, TAB_ENGINEERING, shipped, "PRs merged (30d)", _metric_value(pr_count_30))

    # None means the ticket fetch failed (distinct from an empty 30-day window),
    # so the chart can say "could not load" instead of asserting nobody resolved any.
    tickets_unavailable = resolved_30 is None
    resolved_df = pd.DataFrame() if resolved_30 is None else resolved_30

    left, right = st.columns(2)
    with left:
        ticket_people = (
            resolved_df["assignee"].fillna("Unassigned").astype(str).str.strip().replace("", "Unassigned")
            if "assignee" in resolved_df.columns and not resolved_df.empty
            else pd.Series(dtype=str)
        )
        _contribution_ranking(
            ticket_people, "tickets", "Who resolved tickets (30 days)",
            unavailable=tickets_unavailable,
        )
        if (
            not tickets_unavailable
            and ticket_count_30 is not None
            and len(ticket_people) < int(ticket_count_30)
        ):
            st.caption(
                f"Chart shows a {len(ticket_people)}-ticket sample of ~{int(ticket_count_30)} "
                "resolved (fetch limit); ticket tiles are Jira's approximate counts."
            )
    with right:
        if not github_ready:
            st.caption(
                "PR charts need a GitHub token. "
                + (f"({github_error})" if github_error else "Set DASHBOARD_GITHUB_TOKEN.")
            )
        else:
            # The tiles above are org-wide and count everything that merged; this
            # chart is a ranking of people, and Devin and the Actions runner are
            # not people. Filtered here rather than upstream so the two numbers
            # stay honest, and said out loud below so a reader who adds the bars
            # up and finds they fall short knows why.
            merged_people, merged_bots = _people_only(merged_prs, "author")
            pr_people = (
                merged_people["author"].fillna("unknown").astype(str)
                if "author" in merged_people.columns and not merged_people.empty
                else pd.Series(dtype=str)
            )
            _contribution_ranking(pr_people, "PRs", "Who merged PRs (30 days)")
            if merged_bots:
                st.caption(
                    f"{merged_bots} bot-authored PR(s) are in the tiles above but not "
                    "in this chart (Devin, GitHub Actions, dependabot, renovate)."
                )

    st.caption(
        "Ticket resolved = transitioned into Done / Released / Ready for Production / Review in "
        "Staging in the window (credited to current assignee). PR merged = merged anywhere in the "
        "org in the window (credited to the PR author, by GitHub username)."
    )


# --- Pull requests -----------------------------------------------------------

# A stuck PR is open with no approving review. We classify from the actual
# review counts, not reviewDecision: GitHub only populates reviewDecision when
# the base branch *requires* review (branch protection / CODEOWNERS), so in
# repos without that rule it stays null even when approving reviews exist -
# which would otherwise mark reviewed/approved PRs as stuck.
def _pr_review_label(row: pd.Series) -> str:
    if int(row.get("approving_reviews", 0) or 0) > 0:
        return "Approved"
    if int(row.get("changes_reviews", 0) or 0) > 0:
        return "Changes requested"
    if int(row.get("total_reviews", 0) or 0) > 0:
        return "Reviewed, not approved"
    return "No review yet"


@st.fragment
def _render_pr_section(
    open_prs: pd.DataFrame,
    github_ready: bool,
    github_error: str = "",
    open_count_exact: int | None = None,
) -> None:
    """Org-wide PR health: per-person open/stuck counts and the stuck PR queue."""
    st.subheader("Pull Requests")
    if not github_ready:
        st.info(
            "Connect GitHub to see PR status. Set a read-only DASHBOARD_GITHUB_TOKEN "
            "on the deployment."
            + (f" ({github_error})" if github_error else "")
        )
        return
    if open_prs.empty:
        st.success("No open PRs across the organization.")
        return

    prs = open_prs.copy()
    prs["review"] = prs.apply(_pr_review_label, axis=1)
    # Stuck = open, non-draft, with no approving review (the user's definition).
    # The draft half of that was written in the comment and the caption but never
    # in the code, so the stuck table listed drafts directly beneath a footnote
    # saying drafts were excluded, and read four where Today read three from the
    # same PRs. A draft says "not handed over yet"; it is not a reviewer's fault.
    if "is_draft" in prs.columns:
        drafts = prs["is_draft"].fillna(False).astype(bool)
    else:
        drafts = pd.Series(False, index=prs.index)
    draft_count = int(drafts.sum())
    prs = prs[~drafts]
    if prs.empty:
        st.success(
            f"No open PRs waiting on a review across the organization"
            f"{f' ({draft_count} draft(s) excluded)' if draft_count else ''}."
        )
        return
    prs["stuck"] = prs["approving_reviews"].fillna(0).astype(int) == 0

    # Counted before the drafts came out: this is the paging question ("did we
    # read every open PR?"), not the review question.
    fetched = int(len(open_prs))
    # Exact org-wide open count isn't paging-capped; the frame is (max_prs), so
    # fall back to the fetched size only if the exact count is unavailable.
    open_count = fetched if open_count_exact is None else int(open_count_exact)
    stuck = prs[prs["stuck"]]
    no_review = prs[prs["total_reviews"].fillna(0).astype(int) == 0]
    c1, c2, c3 = st.columns(3)
    review = "Pull requests"
    _tile(c1, TAB_ENGINEERING, review, "Open PRs", str(open_count))
    _tile(
        c2,
        TAB_ENGINEERING,
        review,
        "Stuck (no approving review)",
        f"{len(stuck)}",
    )
    _tile(c3, TAB_ENGINEERING, review, "Never reviewed", f"{len(no_review)}")
    if draft_count:
        st.caption(
            f"{draft_count} draft PR(s) are in the Open PRs tile but in neither review "
            "count, nor in the lists below — a draft has not been handed over yet."
        )
    if open_count_exact is not None and fetched < open_count_exact:
        st.caption(
            f"Per-person and stuck lists cover the {fetched} oldest of {open_count_exact} "
            "open PRs (fetch limit); the Open PRs tile is exact."
        )

    # Per-person PR status: who is holding open and stuck work, and their oldest.
    # The three tiles above are org-wide and include the bots, because a PR Devin
    # opened and nobody reviewed is still a stuck PR somebody has to deal with.
    # This table is a list of people to talk to, and no conversation is going to
    # be had with dependabot, so the bots come out of it here and the difference
    # is stated rather than left for a reader to discover by adding the column up.
    people, bot_prs = _people_only(prs, "author")
    by_person = (
        people.groupby("author")
        .agg(
            open_prs=("number", "size"),
            stuck_prs=("stuck", "sum"),
            oldest_days=("age_days", "max"),
            idle_days=("idle_days", "max"),
        )
        .reset_index()
        .sort_values(["stuck_prs", "oldest_days"], ascending=[False, False])
    )
    by_person["stuck_prs"] = by_person["stuck_prs"].astype(int)
    st.markdown(
        "**PR status by person** (GitHub username)"
        + (
            f" — {bot_prs} bot PR(s) excluded here and counted in the tiles above"
            if bot_prs
            else ""
        )
    )
    st.dataframe(
        by_person,
        width="stretch",
        hide_index=True,
        column_config={
            "author": st.column_config.TextColumn("Person"),
            "open_prs": st.column_config.NumberColumn("Open"),
            "stuck_prs": st.column_config.NumberColumn("Stuck"),
            "oldest_days": st.column_config.NumberColumn("Oldest (days)", format="%.0f"),
            "idle_days": st.column_config.NumberColumn("Most idle (days)", format="%.0f"),
        },
    )

    # The stuck queue itself, oldest first, so nobody has to hunt for the PRs
    # that have been sitting unreviewed.
    st.markdown("**Stuck PRs — open with no approving review, oldest first**")
    stuck_display = stuck.sort_values("age_days", ascending=False)[
        ["url", "title", "author", "review", "age_days", "idle_days"]
    ]
    st.dataframe(
        stuck_display,
        width="stretch",
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("PR", display_text=r"/pull/(\d+)"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "author": st.column_config.TextColumn("Author"),
            "review": st.column_config.TextColumn("Review"),
            "age_days": st.column_config.NumberColumn("Age (days)", format="%.0f"),
            "idle_days": st.column_config.NumberColumn("Idle (days)", format="%.0f"),
        },
    )
    st.caption(
        "Stuck = open PR without an approving review (no review yet, review pending, or changes "
        "requested). Drafts are excluded. Counts are org-wide."
    )

    # Waiting on a reviewer: open PRs with nobody assigned to review AND no review
    # yet — these fall through the cracks because no one is on the hook for them.
    # "review_requests" may be absent if the client predates the field; treat
    # missing as 0 so an older cache degrades to "no reviewer" rather than crashing.
    requests_series = (
        prs["review_requests"] if "review_requests" in prs.columns else 0
    )
    no_reviewer = prs[
        (pd.Series(requests_series, index=prs.index).fillna(0).astype(int) == 0)
        & (prs["total_reviews"].fillna(0).astype(int) == 0)
    ]
    st.markdown("**Waiting on a reviewer — nobody assigned, no review yet**")
    min_age = st.slider(
        "Only show PRs older than (days)",
        min_value=0,
        max_value=30,
        value=2,
        key="no_reviewer_min_age",
    )
    waiting = no_reviewer[no_reviewer["age_days"] > float(min_age)].sort_values(
        "age_days", ascending=False
    )
    if waiting.empty:
        st.success(
            f"No unassigned, unreviewed PRs older than {min_age} day(s)."
        )
    else:
        st.dataframe(
            waiting[["url", "title", "author", "age_days", "idle_days"]],
            width="stretch",
            hide_index=True,
            column_config={
                "url": st.column_config.LinkColumn("PR", display_text=r"/pull/(\d+)"),
                "title": st.column_config.TextColumn("Title", width="large"),
                "author": st.column_config.TextColumn("Author"),
                "age_days": st.column_config.NumberColumn("Age (days)", format="%.0f"),
                "idle_days": st.column_config.NumberColumn("Idle (days)", format="%.0f"),
            },
        )
        st.caption(
            f"{len(waiting)} open PR(s) older than {min_age} day(s) with no reviewer "
            "requested and no review yet — assign someone so they don't stall."
        )


def _render_engineering_page() -> None:
    import write_access
    from change_audit import load_operations, summarize_operations
    st.caption("Visual monitoring for stale, idle, and high-risk tickets.")
    # Reserved before the sections run: the download button can only be built
    # once they have, but the slot has to sit at the top where a reader looks
    # for it.
    engineering_slot = st.columns([5, 1])[1]

    bundle = _engineering_data()
    data = bundle.data
    errors = bundle.errors
    raw_df = bundle.raw_df
    df = bundle.df
    github_ready = bundle.github_ready
    github_error = bundle.github_error
    open_prs = bundle.open_prs
    merged_prs = bundle.merged_prs
    pr_count_7 = bundle.pr_count_7
    pr_count_30 = bundle.pr_count_30
    open_count_exact = bundle.open_count_exact
    assignees = bundle.assignees
    statuses = bundle.statuses
    priorities = bundle.priorities
    max_results = bundle.max_results
    page_size = bundle.page_size

    view = _engineering_filters(bundle)
    scope = view.scope
    selected_assignees = view.selected_assignees
    selected_statuses = view.selected_statuses
    selected_priorities = view.selected_priorities
    min_idle = view.min_idle
    min_age = view.min_age
    include_backlogs = view.include_backlogs
    color_by = view.color_by
    allow_writes = view.allow_writes
    filtered = view.filtered
    unscoped = view.unscoped

    # One engineer means one page about them alone: the org-wide sections would
    # only be somebody else's work wearing their name at the top. That holds
    # however the reader narrowed to them - the Individual scope, or the Team
    # multiselect whittled down to a single name.
    if selected_assignees is not None and len(selected_assignees) == 1:
        _render_individual_page(
            person=str(selected_assignees[0]),
            filtered=filtered,
            organization=df,
            open_prs=open_prs,
            merged_prs=merged_prs,
            github_ready=github_ready,
            github_error=github_error,
            include_backlogs=include_backlogs,
        )
        _download_report(engineering_slot, TAB_ENGINEERING)
        return

    # None (not an empty frame) marks a read that failed, so a section can say
    # "could not load" instead of an authoritative "there is nothing here".
    _render_resolved_summary(
        data.get("resolved_count_7"),
        data.get("resolved_count_30"),
        data.get("resolved_30"),
        pr_count_7,
        pr_count_30,
        merged_prs,
        github_ready,
        github_error,
    )
    st.divider()

    _render_new_and_triage(
        data.get("created_count_1"),
        data.get("created_count_7"),
        data.get("triage_stuck_count"),
        data.get("created_7"),
        data.get("triage_stuck"),
        TRIAGE_STUCK_HOURS,
    )
    st.divider()

    _render_metrics(
        filtered,
        include_backlogs=include_backlogs,
        unassigned_source=unscoped if selected_assignees is not None else None,
    )

    # One backlog-filtered view, shared: four sections asked for the same frame
    # with the same arguments and each rebuilt it.
    metrics_view = _metrics_df(filtered, include_backlogs)

    st.divider()
    _render_mix(metrics_view)

    st.divider()
    _render_team_overview(metrics_view)

    st.divider()
    _render_epics(metrics_view, organization_source=df)

    st.divider()
    # Backlog-inclusive on purpose: the backlog is what this section clears out.
    _render_cleanup(filtered, unassigned_source=unscoped)

    st.divider()
    _render_scope_breakdown(filtered, scope=scope, include_backlogs=include_backlogs)

    st.divider()
    _render_pr_section(open_prs, github_ready, github_error, open_count_exact)

    st.divider()
    # Every ticket, not the scoped slice: a PR belongs to the org whichever team
    # or person the dashboard is currently looking at.
    _render_pr_hygiene(
        open_prs, github_ready, github_error, _known_project_keys(df), tickets=df
    )

    st.divider()
    # Backlog-inclusive on purpose: a backlog ticket is the best kind to hand off,
    # and it is where badly written tickets accumulate unseen.
    _render_ticket_quality(filtered)

    st.divider()
    _render_priority_queue(filtered, include_backlogs=include_backlogs)

    st.divider()
    _render_estimate_policy(filtered)

    st.divider()
    _render_stale_cleanup(filtered)

    restore_requested = bool(st.session_state.pop("restore_sprint_ticket_table", False))
    bubble_chart_version = int(st.session_state.get("bubble_chart_version", 0))
    if restore_requested:
        bubble_chart_version += 1
        st.session_state["bubble_chart_version"] = bubble_chart_version

    agg_priority = st.checkbox(
        "Aggregate Priorities (Normal / High / Urgent)",
        value=False,
        help="Buckets: Normal = None/Low/Normal · High = High · Urgent = Highest/Urgent",
    )
    selected_key = _render_bubble_chart(
        filtered,
        color_by=color_by,
        agg_priority=agg_priority,
        chart_key=f"bubble_chart_{bubble_chart_version}",
    )

    if restore_requested:
        active_sprint_ticket_key = None
    else:
        active_sprint_ticket_key = selected_key if selected_key and selected_key in filtered["key"].values else None

    st.divider()
    st.subheader("Sprint Planner")
    _render_sprint_plan(df)

    st.divider()
    st.subheader("Sprint Capacity")
    _render_sprint_capacity(
        filtered,
        status_source_df=filtered,
        selected_ticket_key=active_sprint_ticket_key,
    )

    st.divider()
    st.subheader("Suggested First Action")

    PRIORITY_OPTIONS = ["Highest", "High", "Normal", "Low", "Lowest"]
    action_type = st.selectbox(
        "Action",
        options=["Set None-priority tickets", "Change status"],
        index=0,
        help="Default action keeps the first cleanup flow: None priority -> Normal.",
    )

    # Bulk writes must not reach tickets the user has hidden with Include Backlogs.
    action_df = metrics_view
    status_options = sorted(action_df["status"].dropna().astype(str).unique().tolist())
    normalized_priority = action_df["priority"].fillna("").astype(str).str.strip().str.lower()
    none_priority_keys = sorted(action_df[normalized_priority.isin(["", "none"])]["key"].tolist())

    with st.container(border=True):
        if action_type == "Set None-priority tickets":
            st.markdown("**Detected tickets without priority**")
            st.caption(
                f"{len(none_priority_keys)} ticket(s) in the current view have no priority set."
            )
            if none_priority_keys:
                preview = ", ".join(none_priority_keys[:15])
                suffix = " ..." if len(none_priority_keys) > 15 else ""
                st.caption(f"Sample: {preview}{suffix}")

            default_keys = none_priority_keys[:BULK_ACTION_DEFAULT_LIMIT]
            if len(none_priority_keys) > len(default_keys):
                st.caption(
                    f"Only the first {BULK_ACTION_DEFAULT_LIMIT} are pre-selected; "
                    "add more explicitly if you mean to update them."
                )
            selected_keys = st.multiselect(
                "Tickets to update",
                options=none_priority_keys,
                default=default_keys,
                help="Remove any tickets you do not want to update.",
            )

            target_priority = st.selectbox(
                "Suggested priority",
                options=PRIORITY_OPTIONS,
                index=2,
                help="Normal is selected by default as the first cleanup action.",
            )
            target_label = f"priority '{target_priority}'"
        else:
            st.markdown("**Change ticket status**")
            if not status_options:
                st.info("No statuses available in the current filtered view.")
                source_status = None
                target_status = None
                selected_keys = []
            else:
                source_status = st.selectbox("From status", options=status_options, index=0)
                to_options = [s for s in status_options if s != source_status] or status_options
                target_status = st.selectbox("To status", options=to_options, index=0)

                source_keys = sorted(action_df[action_df["status"] == source_status]["key"].tolist())
                default_source_keys = source_keys[:BULK_ACTION_DEFAULT_LIMIT]
                if len(source_keys) > len(default_source_keys):
                    st.caption(
                        f"Only the first {BULK_ACTION_DEFAULT_LIMIT} are pre-selected; "
                        "add more explicitly if you mean to update them."
                    )
                selected_keys = st.multiselect(
                    "Tickets to update",
                    options=source_keys,
                    default=default_source_keys,
                    help="Only tickets currently in the selected source status are listed.",
                )
                target_label = f"status '{source_status}' -> '{target_status}'"

        apply_suggestion = st.button(
            f"Apply to {len(selected_keys)} ticket(s)",
            disabled=(not selected_keys) or (not write_access.writes_enabled()),
            type="primary",
        )

    if apply_suggestion and selected_keys:
        client = JiraClient.resolve(
            creds_path=CREDS_PATH,
            profile_name=PROFILE_NAME,
        )
        with st.spinner(f"Updating {len(selected_keys)} tickets..."):
            if action_type == "Set None-priority tickets":
                succeeded, failed, operation = _apply_action_with_audit(
                    client=client,
                    action_type="priority",
                    selected_keys=selected_keys,
                    target=target_priority,
                )
            else:
                succeeded, failed, operation = _apply_action_with_audit(
                    client=client,
                    action_type="status",
                    selected_keys=selected_keys,
                    target=target_status,
                    source_status=source_status,
                )

        if succeeded:
            st.success(
                f"Updated {len(succeeded)} ticket(s) to {target_label}. Operation ID: {operation['operation_id']}"
            )
        if failed:
            for key, err in failed.items():
                st.error(f"{key}: {err}")

        _clear_page_caches(ENGINEERING_PAGE_TITLE)
        st.rerun()

    st.divider()
    st.subheader("Change History and Revert")
    operations = load_operations(limit=30)
    if not operations:
        st.info("No write operations have been logged yet.")
    else:
        st.dataframe(pd.DataFrame(summarize_operations(operations)), width="stretch")

        op_options = {
            (
                f"{op.get('created_at', '')} | {op.get('action_type', '')} | "
                f"{op.get('target', '')} | success={op.get('success_count', 0)} | "
                f"id={str(op.get('operation_id', ''))[:8]}"
            ): op
            for op in operations
            if op.get("success_count", 0) > 0
        }

        if not op_options:
            st.caption("No successful operation available for revert.")
        else:
            selected_label = st.selectbox(
                "Select operation to revert",
                options=list(op_options.keys()),
            )
            selected_operation = op_options[selected_label]
            confirm_revert = st.checkbox("I understand revert may partially fail due to Jira workflow rules.")

            revert_clicked = st.button(
                "Revert selected operation",
                disabled=(not confirm_revert) or (not write_access.writes_enabled()),
            )

            if revert_clicked:
                client = JiraClient.resolve(
                    creds_path=CREDS_PATH,
                    profile_name=PROFILE_NAME,
                )

                revert_succeeded: list[str] = []
                revert_failed: dict[str, str] = {}
                parent_id = selected_operation.get("operation_id")
                successful_items = [it for it in selected_operation.get("items", []) if it.get("success")]

                with st.spinner(f"Reverting {len(successful_items)} ticket(s)..."):
                    for item in successful_items:
                        key = str(item.get("key", ""))
                        before = item.get("before") or {}
                        try:
                            if selected_operation.get("action_type") == "priority":
                                original_priority_id = before.get("priority_id")
                                if not original_priority_id:
                                    raise RuntimeError("Original priority id missing in audit record.")

                                rev_succeeded, rev_failed, rev_op = _apply_action_with_audit(
                                    client=client,
                                    action_type="revert_priority",
                                    selected_keys=[key],
                                    target=str(original_priority_id),
                                    parent_operation_id=str(parent_id),
                                )
                            elif selected_operation.get("action_type") == "status":
                                original_status = before.get("status")
                                if not original_status:
                                    raise RuntimeError("Original status missing in audit record.")

                                rev_succeeded, rev_failed, rev_op = _apply_action_with_audit(
                                    client=client,
                                    action_type="revert_status",
                                    selected_keys=[key],
                                    target=str(original_status),
                                    parent_operation_id=str(parent_id),
                                )
                            else:
                                raise RuntimeError("Selected operation type is not revertible by this tool.")

                            revert_succeeded.extend(rev_succeeded)
                            revert_failed.update(rev_failed)
                        except Exception as exc:  # noqa: BLE001
                            revert_failed[key] = str(exc)

                if revert_succeeded:
                    st.success(f"Reverted {len(revert_succeeded)} ticket(s).")
                if revert_failed:
                    for key, err in revert_failed.items():
                        st.error(f"Revert failed for {key}: {err}")

                _clear_page_caches(ENGINEERING_PAGE_TITLE)
                st.rerun()

    st.caption(
        "Team member filter uses Jira assignee display names from fetched data. "
        "For stricter JQL filtering, use assignee account IDs in JQL."
    )
    _download_report(engineering_slot, TAB_ENGINEERING)


