"""The Code page: pull request health across every repo.

Split out of app.py in Task 1C. ``_team_prs``, ``_repo_review_coverage``,
``_render_code_kpis``, ``_render_repo_coverage``, ``_share_rank_bar``,
``_render_stuck_queue`` and ``_render_findings_and_citizenship`` are private
to this page. ``_render_pr_hygiene`` (used by this page and the legacy
Engineering page) lives in ``render_shared``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import theme_html
from data_layer import _engineering_context
from page_shared import TAB_ENGINEERING, _download_report
from render_shared import (
    TODAY_NO_REVIEWER_DAYS,
    _exclude_repos,
    _known_project_keys,
    _one_person_instead,
    _open_pr_signals,
    _render_pr_hygiene,
    _truncation_note,
)

# The rendered tables are display, not paging widgets, so a long queue is cut.
_STUCK_QUEUE_ROWS = 25


def _team_prs(prs: pd.DataFrame) -> pd.DataFrame:
    """PRs that count, for a frame that may predate the query-level exclusion."""
    if prs.empty:
        return prs
    excluded, _ = _exclude_repos()
    if excluded and "repo" in prs.columns:
        prs = prs[~prs["repo"].astype(str).isin(excluded)]
    return prs


def _repo_review_coverage(open_prs: pd.DataFrame) -> pd.DataFrame:
    """Per repo: how much open work carries no approving review.

    The org-wide 92% says the review culture is missing; this says where. A repo
    at 100% is one with no assigned reviewer, which is a different fix from
    chasing individuals — and the fix is per-repo, so the cut must be too.
    """
    if open_prs.empty or "repo" not in open_prs.columns:
        return pd.DataFrame(
            columns=["repo", "open_prs", "unreviewed", "unreviewed_share",
                     "never_reviewed", "median_age_days"]
        )
    live = open_prs
    if "is_draft" in live.columns:
        live = live[~live["is_draft"].fillna(False).astype(bool)]
    if live.empty:
        return pd.DataFrame(
            columns=["repo", "open_prs", "unreviewed", "unreviewed_share",
                     "never_reviewed", "median_age_days"]
        )
    approvals = live.get("approving_reviews", pd.Series(0, index=live.index)).fillna(0).astype(int)
    reviews = live.get("total_reviews", pd.Series(0, index=live.index)).fillna(0).astype(int)
    age = live.get("age_days", pd.Series(0.0, index=live.index)).fillna(0.0)
    frame = pd.DataFrame(
        {
            "repo": live["repo"].astype(str),
            "unreviewed": (approvals == 0).astype(int),
            "never_reviewed": (reviews == 0).astype(int),
            "age_days": age,
        }
    )
    out = (
        frame.groupby("repo")
        .agg(
            open_prs=("unreviewed", "size"),
            unreviewed=("unreviewed", "sum"),
            never_reviewed=("never_reviewed", "sum"),
            median_age_days=("age_days", "median"),
        )
        .reset_index()
    )
    out["unreviewed_share"] = out["unreviewed"] / out["open_prs"]
    # Worst first: the repo with the least review is the point of the chart.
    return out.sort_values(
        ["unreviewed_share", "open_prs"], ascending=[False, False]
    ).reset_index(drop=True)


def _render_code_kpis(open_prs: pd.DataFrame, merged_prs: pd.DataFrame) -> None:
    """The five numbers the mockup opens with, from the frames already fetched."""
    import pr_quality

    signals = _open_pr_signals(open_prs, None)
    share = (signals["unapproved"] / signals["total"]) if signals["total"] else 0.0
    oldest = signals["oldest_unreviewed_days"]

    oldest_repo = ""
    if not open_prs.empty and "repo" in open_prs.columns:
        live = open_prs
        if "is_draft" in live.columns:
            live = live[~live["is_draft"].fillna(False).astype(bool)]
        reviews = live.get("total_reviews", pd.Series(0, index=live.index)).fillna(0).astype(int)
        never = live[reviews == 0]
        if not never.empty:
            oldest_repo = str(never.loc[never["age_days"].idxmax()].get("repo", ""))

    # The tile that carries an accusation reports the accusation's own column.
    # ``self_merged`` is merely ``merged_by == author``, which is normal once a
    # colleague has approved; ``merged_without_outside_approval`` is the review
    # process not happening, and GitHub does not let an author approve their own
    # PR at all, so "approved own work" described something impossible.
    selfm = pr_quality.self_merge(merged_prs)
    unapproved_merges = (
        int(selfm["merged_without_outside_approval"].sum()) if not selfm.empty else 0
    )
    self_merged = int(selfm["self_merged"].sum()) if not selfm.empty else 0
    merged_total = int(selfm["merged_prs"].sum()) if not selfm.empty else 0
    unapproved_share = (unapproved_merges / merged_total) if merged_total else 0.0

    theme_html.tiles(
        [
            (
                "Open PRs",
                str(signals["total"]),
                "open and handed to a reviewer · the search excludes drafts",
                "neutral",
            ),
            (
                "No approving review",
                str(signals["unapproved"]),
                f"{share:.0%} of open · {signals['never_reviewed']} never reviewed",
                "danger" if share > 0.5 else "warning",
            ),
            (
                "Nobody asked",
                str(signals["no_reviewer_asked"]),
                f"> {TODAY_NO_REVIEWER_DAYS:.0f} days old · no request, no review",
                "danger" if signals["no_reviewer_asked"] else "good",
            ),
            (
                "Oldest unreviewed",
                f"{oldest:.0f}d" if oldest else "—",
                oldest_repo or "no unreviewed PRs",
                "warning" if oldest else "good",
            ),
            (
                "Merged unapproved · 30d",
                str(unapproved_merges),
                (
                    f"{unapproved_share:.0%} of merges · nobody else approved first"
                    f" · {self_merged} pressed merge on their own PR"
                    if merged_total
                    else "no merges in window"
                ),
                "danger" if unapproved_share > 0.25 else "neutral",
            ),
        ]
    )


@st.fragment
def _render_repo_coverage(open_prs: pd.DataFrame) -> None:
    st.subheader("Review coverage by repo")
    st.caption(
        "Share of open PRs with no approving review — worst first. This is where "
        "review culture is missing, not who is slow."
    )
    coverage = _repo_review_coverage(open_prs)
    if coverage.empty:
        st.info("No open pull requests to measure.")
        return
    never = coverage[coverage["unreviewed_share"] >= 1.0]
    footer = (
        f"{len(never)} repo(s) have no approving review on any open PR: "
        + ", ".join(never["repo"].tolist())
        + ". Those are the ones with no assigned reviewer, not the ones with bad code."
        if len(never)
        else ""
    )
    theme_html.hbars(
        [
            (row.repo, float(row.unreviewed_share * 100), f"{row.unreviewed_share:.0%}")
            for row in coverage.itertuples()
        ],
        title="",
        footer=footer,
        severity=True,
    )


def _share_rank_bar(shares: pd.Series):
    """A rank bar over precomputed percentages, colored by severity."""
    import plotly.graph_objects as go

    values = shares.sort_values()
    colors = [
        "#e34948" if v >= 95 else "#eb6834" if v >= 70 else "#eda100" if v >= 40 else "#1baf7a"
        for v in values
    ]
    figure = go.Figure(
        go.Bar(
            x=values.values,
            y=[str(i) for i in values.index],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}%" for v in values.values],
            textposition="outside",
        )
    )
    figure.update_layout(
        height=max(240, 34 * len(values) + 90),
        margin=dict(t=16, b=40, l=8, r=56),
        showlegend=False,
        bargap=0.28,
    )
    figure.update_xaxes(title_text="% of open PRs with no approving review", range=[0, 112])
    figure.update_yaxes(title_text="", tickangle=0, automargin=True, type="category")
    return figure


@st.fragment
def _render_stuck_queue(open_prs: pd.DataFrame, tickets: pd.DataFrame) -> None:
    import pr_hygiene

    st.subheader("Stuck queue — open, no approving review, oldest first")
    st.caption(
        'Drafts excluded. A rejecting review still counts as attention, so '
        '"reviews" of 0 is the harsher column.'
    )
    if open_prs.empty:
        st.info("No open pull requests.")
        return
    live = open_prs
    if "is_draft" in live.columns:
        live = live[~live["is_draft"].fillna(False).astype(bool)]
    approvals = live.get("approving_reviews", pd.Series(0, index=live.index)).fillna(0).astype(int)
    stuck = live[approvals == 0].copy()
    if stuck.empty:
        st.success("Every open PR has an approving review.")
        return
    stuck = pr_hygiene.add_hygiene_fields(stuck, _known_project_keys(tickets))
    stuck["asked"] = (
        stuck.get("review_requests", pd.Series(0, index=stuck.index)).fillna(0).astype(int) > 0
    ).map({True: "yes", False: "no"})
    stuck = stuck.sort_values("age_days", ascending=False)
    # This org has run 75+ unapproved open PRs, so the hidden tail is the normal
    # case rather than the edge one: the footer names the cut, or 25 rows read as
    # the whole queue.
    theme_html.table(
        stuck,
        [
            ("url", "PR", "link"),
            ("title", "Title", "text"),
            ("author", "Author", "text"),
            ("repo", "Repo", "text"),
            ("age_days", "Age (d)", "num"),
            ("total_reviews", "Reviews", "strong-num"),
            ("asked", "Asked", "text"),
            ("jira_key", "Ticket", "text"),
        ],
        title="",
        footer=_truncation_note(len(stuck), _STUCK_QUEUE_ROWS),
        max_rows=_STUCK_QUEUE_ROWS,
    )


@st.fragment
def _render_findings_and_citizenship(merged_prs: pd.DataFrame, open_prs: pd.DataFrame) -> None:
    import pr_quality

    left, right = st.columns(2)
    judged_pool = pd.concat([merged_prs, open_prs], ignore_index=True) if not merged_prs.empty or not open_prs.empty else pd.DataFrame()

    with left:
        st.subheader("Devin findings per author")
        st.caption(
            'Share of **judged** PRs where Devin requested changes. Judged is shown '
            'so that "no findings" is never confused with "not reviewed".'
        )
        findings = pr_quality.devin_findings_by_author(judged_pool)
        if findings.empty:
            st.info("No AI-review data in the fetched PRs — needs the extended PR payload.")
        else:
            findings = findings.sort_values("prs_judged", ascending=False)
            low_n = findings["prs_judged"] < 5
            findings = findings.assign(
                changes_requested_share=(findings["changes_requested_share"] * 100).round(0)
            )
            findings.loc[low_n, "changes_requested_share"] = float("nan")
            st.dataframe(
                findings[["author", "prs_judged", "prs_changes_requested", "changes_requested_share"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "author": st.column_config.TextColumn("Author"),
                    "prs_judged": st.column_config.NumberColumn("Judged"),
                    "prs_changes_requested": st.column_config.NumberColumn("Changes asked"),
                    "changes_requested_share": st.column_config.NumberColumn(
                        "Share", format="%d%%", help="Blank below 5 judged PRs — insufficient data."
                    ),
                },
            )

    with right:
        st.subheader("Review citizenship")
        st.caption(
            "Reviews **given**. Two people carrying the review load for twelve "
            "engineers is the 92% in one sentence."
        )
        citizens = pr_quality.review_citizenship(judged_pool)
        if citizens.empty:
            st.info("No review events in the fetched PRs — needs the extended PR payload.")
        else:
            citizens = citizens.sort_values("reviews_given", ascending=False)
            st.dataframe(
                citizens[
                    ["reviewer", "reviews_given", "distinct_authors_reviewed",
                     "median_hours_to_first_review"]
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "reviewer": st.column_config.TextColumn("Person"),
                    "reviews_given": st.column_config.NumberColumn("Given"),
                    "distinct_authors_reviewed": st.column_config.NumberColumn("Authors"),
                    "median_hours_to_first_review": st.column_config.NumberColumn(
                        "TTFR (h)", format="%.0f"
                    ),
                },
            )


def _render_code_page() -> None:
    """PR health across the team's repos, in the mockup's order.

    Five numbers, then where review is missing, then the queue of what is stuck,
    then how the work was written. The old PR sections remain reachable on
    /engineering; this page is the designed view of the same data.
    """
    theme_html.css()
    _, exclusion_caption = _exclude_repos()
    st.caption(
        "PR health across all team repos. Drafts are excluded from review counts — "
        "a draft has not been handed to a reviewer yet."
        + (f" {exclusion_caption}" if exclusion_caption else "")
    )
    bundle, view, slot = _engineering_context()
    if _one_person_instead(bundle, view, slot):
        return

    if not bundle.github_ready:
        st.error(
            "GitHub is unavailable, so this page cannot be drawn. "
            + (f"({bundle.github_error})" if bundle.github_error else "Set DASHBOARD_GITHUB_TOKEN.")
        )
        _download_report(slot, TAB_ENGINEERING)
        return

    open_prs = _team_prs(bundle.open_prs)
    merged_prs = _team_prs(bundle.merged_prs)

    _render_code_kpis(open_prs, merged_prs)
    st.divider()
    _render_repo_coverage(open_prs)
    st.divider()
    _render_stuck_queue(open_prs, bundle.df)
    st.divider()
    _render_findings_and_citizenship(merged_prs, open_prs)
    st.divider()
    # Traceability of tickets to PRs stays: it is the bridge to the Jira side.
    _render_pr_hygiene(
        open_prs,
        bundle.github_ready,
        bundle.github_error,
        _known_project_keys(bundle.df),
        tickets=bundle.df,
    )
    _download_report(slot, TAB_ENGINEERING)
