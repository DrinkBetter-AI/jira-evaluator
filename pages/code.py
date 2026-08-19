"""The Code page: pull request health across every repo.

Split out of app.py in Task 1C. ``_team_prs``, ``_repo_review_coverage``,
``_render_code_kpis``, ``_render_repo_coverage``, ``_share_rank_bar``,
``_render_stuck_queue`` and ``_render_findings_and_citizenship`` are private
to this page. ``_render_pr_hygiene`` (used by this page and the legacy
Engineering page) lives in ``render_shared``.

Task 3D adds five things, all additive, none of them touching the nine
sections above: the open-PR tile's honest "including drafts" total (the
headline number excludes drafts by construction — ``_open_query`` carries
``draft:false`` — so the exclusion is silent unless said out loud), a
"Merged 30d" column and an "exempt" role chip on the Devin-findings table,
an "Unprompted reviews" column (2D's ``pr_quality.unprompted_reviews`` — the
proactivity signal DEVIN_PLAN §6 says the rest of review citizenship is only
compliance), a "went to draft after a review was requested" row (2D's
``pr_quality.draft_transitions`` — KPI_SPEC exploit #6, made visible instead
of silently vanishing from every ``draft:false`` queue on the dashboard),
and an honest callout when the extended GraphQL payload that all three of
those need degrades to the lean one. See ``docs/assumptions/3D.md`` for the
call shape and ownership decisions behind each.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

import theme
import theme_html
import theme_tokens
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


def _render_code_kpis(
    open_prs: pd.DataFrame,
    merged_prs: pd.DataFrame,
    *,
    open_including_drafts: int | None = None,
) -> None:
    """The five numbers the mockup opens with, from the frames already fetched.

    ``open_including_drafts`` is optional and defaults to ``None`` — the
    caller (``_render_code_page``) passes the real total only when it could
    actually read it from GitHub (see ``_open_pr_total_including_drafts``).
    ``None`` keeps the note exactly as it always read ("the search excludes
    drafts"): an unreachable draft count must never be silently shown as
    "0 including drafts", so absence is absence, not zero.
    """
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

    # "of N including drafts": the headline is draft:false by construction
    # (``_open_query``), so the exclusion is silent unless said out loud here.
    open_note = "open and handed to a reviewer · the search excludes drafts"
    if open_including_drafts is not None:
        open_note = f"open and handed to a reviewer · of {open_including_drafts} including drafts"

    theme_html.tiles(
        [
            (
                "Open PRs",
                str(signals["total"]),
                open_note,
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
    """A rank bar over precomputed percentages, colored by severity.

    The colour ramp and the bar geometry both read from shared tokens rather
    than typing their own: ``theme_html.severity_hue`` is the same threshold
    logic the HTML review-coverage bars use (``_render_repo_coverage`` above,
    ``severity=True``), resolved to a real hex through
    ``theme_tokens.SERIES_BY_KEY`` since a plotly marker cannot take a CSS
    ``var(...)`` reference; ``theme.BAR_MAX_HEIGHT``/``BAR_CORNER_RADIUS`` are
    the same 24px/4px ceiling ``theme.rank_bar`` draws every other ranked bar
    on the dashboard to. See docs/assumptions/5A.md.
    """
    import plotly.graph_objects as go

    values = shares.sort_values()
    colors = [
        theme_tokens.SERIES_BY_KEY[theme_html.severity_hue(v)] for v in values
    ]
    margin = dict(t=16, b=40, l=8, r=56)
    figure = go.Figure(
        go.Bar(
            x=values.values,
            y=[str(i) for i in values.index],
            orientation="h",
            marker=dict(color=colors, cornerradius=theme.BAR_CORNER_RADIUS),
            text=[f"{v:.0f}%" for v in values.values],
            textposition="outside",
        )
    )
    margin_v = margin["t"] + margin["b"]
    height = max(240, theme.BAR_ROW_HEIGHT * len(values) + margin_v)
    figure.update_layout(
        height=height,
        margin=margin,
        showlegend=False,
        # theme.bar_gap_for(), not the fixed theme.BAR_GAP: the 240px floor
        # above inflates each row's slot past the ceiling for a short list
        # (few repos), and only a bargap solved for that actual slot keeps
        # the bar itself at or under theme.BAR_MAX_HEIGHT regardless. See
        # that function's own docstring and docs/assumptions/5A.md.
        bargap=theme.bar_gap_for(len(values), height - margin_v),
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


# --------------------------------------------------------------------------- #
# Task 3D additions: the "of N including drafts" total, the extended-payload
# pool the proactivity/draft-hiding signals need, and the three new sections
# built on top of them. See docs/assumptions/3D.md.
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=300, show_spinner=False)
def _open_pr_total_including_drafts_cached(token: str, org: str) -> int | None:
    """GitHub's own count of every open PR, draft or not, cached 5 minutes.

    ``github_client._open_query`` hardcodes ``draft:false`` for every other
    open-PR read on this page, so the headline "Open PRs" tile is silently
    missing every draft. This reuses the same exclusion helper
    (``_without_excluded``) and count primitive (``_search_count``) with the
    ``draft:false`` term dropped, so the two totals are directly comparable —
    same org, same excluded repos, only the draft filter differs. ``None`` on
    any failure: a wrong number here would misstate the real total, and
    absence is not the same claim as zero.
    """
    import github_client

    try:
        query = github_client._without_excluded(org, f"org:{org} is:pr is:open")
        return github_client._search_count(token, query)
    except Exception:
        return None


def _open_pr_total_including_drafts(token: str | None, org: str | None) -> int | None:
    """The cached total, or ``None`` when there is no token/org to ask with."""
    if not token or not org:
        return None
    return _open_pr_total_including_drafts_cached(token, org)


@st.cache_data(ttl=300, show_spinner=False)
def _extended_pr_pool_cached(token: str, org: str) -> tuple[pd.DataFrame, bool, str]:
    """Open + merged PRs via the extended GraphQL payload, cached 5 minutes.

    ``fetch_open_prs_extended``/``fetch_merged_prs_extended`` already floor-cache
    the real GraphQL read at >=1h inside ``github_client`` itself (see
    docs/assumptions/2D.md) — this 5-minute wrapper only saves the two Python
    calls and the concat on every rerun in between, the same TTL
    ``data_layer.py`` uses for the plain (non-extended) open/merged reads this
    page already draws from ``bundle``. ``data_layer.py`` has no
    extended-aware wrapper yet (out of this task's ownership — see 2D's
    "Render call sites for Phase 3"), so this calls ``github_client`` directly
    rather than going through it.
    """
    import github_client

    open_fetch = github_client.fetch_open_prs_extended(token, org)
    merged_fetch = github_client.fetch_merged_prs_extended(token, org, 30)
    frames = [f for f in (open_fetch.frame, merged_fetch.frame) if not f.empty]
    pool = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    degraded = bool(open_fetch.degraded or merged_fetch.degraded)
    reason = open_fetch.reason or merged_fetch.reason
    return pool, degraded, reason


def _extended_pr_pool(token: str | None, org: str | None) -> tuple[pd.DataFrame, bool, str]:
    """The extended pool, or an honest "no token" degradation with nothing fetched."""
    if not token or not org:
        return (
            pd.DataFrame(),
            True,
            "No GitHub token configured — the proactivity and draft-hiding signals need it.",
        )
    return _extended_pr_pool_cached(token, org)


def _pr_url_lookup(pool: pd.DataFrame) -> dict[int, str]:
    """First known URL per PR number in ``pool``.

    ``pr_quality.unprompted_reviews`` hands back PR numbers as evidence, not
    URLs — it has no reason to carry a URL column itself. This is the join
    back to one, built from the same pool the numbers came from. A PR number
    can repeat across repos; the first URL seen for a number wins, so a
    reader should take a linked number as "here is *a* matching PR", not
    proof there is only one.
    """
    if pool.empty or "number" not in pool.columns or "url" not in pool.columns:
        return {}
    lookup: dict[int, str] = {}
    for number, url in zip(pool["number"], pool["url"]):
        if pd.isna(number) or not url:
            continue
        try:
            n = int(number)
        except (TypeError, ValueError):
            continue
        lookup.setdefault(n, str(url))
    return lookup


def _pr_evidence_html(numbers: tuple, url_lookup: dict[int, str]) -> str:
    """Linked PR numbers for one reviewer's unprompted-review evidence."""
    if not numbers:
        return '<span class="dim">none</span>'
    parts = []
    for n in numbers:
        n_int = int(n)
        url = url_lookup.get(n_int)
        if url:
            parts.append(f'<a href="{html.escape(url)}">#{n_int}</a>')
        else:
            parts.append(f"#{n_int}")
    return ", ".join(parts)


def _unprompted_reviews_rows(pool: pd.DataFrame) -> pd.DataFrame:
    """Thin wrapper so the page's own tests can call this without a fetch."""
    import pr_quality

    return pr_quality.unprompted_reviews(pool)


def _render_unprompted_reviews(pool: pd.DataFrame) -> None:
    """DEVIN_PLAN §6's proactivity signal, its own UI column, evidence linked.

    "the rest is compliance" — ``review_citizenship`` (next to this) counts
    every review given, whether the reviewer was asked or went looking for
    work to unblock; this isolates the second kind and cites the PRs.
    """
    st.subheader("Unprompted reviews")
    st.caption(
        "Reviews given before anyone asked for them — DEVIN_PLAN §6's "
        'proactivity signal: "the rest is compliance."'
    )
    out = _unprompted_reviews_rows(pool)
    if out.empty:
        st.info(
            "No unprompted-review evidence in the fetched PRs — needs the extended "
            "PR payload (GITHUB_EXTENDED_PR_DATA=1)."
        )
        return
    url_lookup = _pr_url_lookup(pool)
    columns = [
        theme_html.Column("Reviewer"),
        theme_html.Column("Unprompted", kind="strong-num"),
        theme_html.Column(
            "Evidence", kind="html", help="PRs reviewed before anyone asked this person"
        ),
        theme_html.Column(
            "Top partner",
            help="Reused from reciprocity() — a closed approval loop, not proactivity",
        ),
    ]
    rows = []
    for row in out.itertuples():
        partner = (
            f"{row.top_partner} ({row.top_partner_share:.0%})"
            if row.top_partner and pd.notna(row.top_partner_share)
            else "—"
        )
        rows.append(
            [
                theme_html.Cell(row.reviewer),
                theme_html.Cell(row.unprompted_reviews),
                theme_html.Cell(_pr_evidence_html(row.prs, url_lookup)),
                theme_html.Cell(partner),
            ]
        )
    theme_html.render(theme_html.table(columns, rows))


def _devin_findings_rows(judged_pool: pd.DataFrame, merged_prs: pd.DataFrame) -> pd.DataFrame:
    """Devin findings per author, plus 30d merge volume and the exec exemption.

    The findings rate alone invites "how much are they actually shipping" —
    ``merged_30d`` answers that in the same row. ``exempt`` is the roster's
    own exec role (``roles.EXEC_ROLE``, the same one ``roles.rubric_for_role``
    already refuses to score) — an exec author showing up on this table with
    no exemption marked would read as "this person is being graded like an
    engineer," which is not the roster's decision.
    """
    import pr_quality
    import roles

    findings = pr_quality.devin_findings_by_author(judged_pool)
    if findings.empty:
        return findings
    findings = findings.sort_values("prs_judged", ascending=False).reset_index(drop=True)
    low_n = findings["prs_judged"] < 5
    findings = findings.assign(
        changes_requested_share=(findings["changes_requested_share"] * 100).round(0)
    )
    findings.loc[low_n, "changes_requested_share"] = float("nan")

    if not merged_prs.empty and "author" in merged_prs.columns:
        merged_30d = merged_prs.groupby(merged_prs["author"].astype(str)).size()
    else:
        merged_30d = pd.Series(dtype="int64")
    findings["merged_30d"] = (
        findings["author"].astype(str).map(merged_30d).fillna(0).astype(int)
    )

    roster = roles.load_roster()

    def _is_exempt(login: str) -> bool:
        name = roster.name_for_login(login) or login
        return roster.role_of(name) == roles.EXEC_ROLE

    findings["exempt"] = findings["author"].astype(str).apply(_is_exempt)
    return findings


def _findings_table_html(findings: pd.DataFrame) -> str:
    """Render the enriched Devin-findings frame through the new table kit.

    Switched from ``st.dataframe`` to ``theme_html.table`` for this one
    table because the exempt chip needs a real fragment
    (``theme_html.rolechip``), which ``st.column_config`` has no way to draw.
    """
    columns = [
        theme_html.Column("Author"),
        theme_html.Column("Judged", kind="num"),
        theme_html.Column("Changes asked", kind="num"),
        theme_html.Column(
            "Share", kind="num", help="Blank below 5 judged PRs — insufficient data."
        ),
        theme_html.Column("Merged 30d", kind="strong-num"),
        theme_html.Column("Role", kind="html"),
    ]
    rows = []
    for row in findings.itertuples():
        share = (
            "" if pd.isna(row.changes_requested_share) else f"{row.changes_requested_share:.0f}%"
        )
        role_html = theme_html.rolechip("Exempt") if row.exempt else ""
        rows.append(
            [
                theme_html.Cell(row.author),
                theme_html.Cell(row.prs_judged),
                theme_html.Cell(row.prs_changes_requested),
                theme_html.Cell(share),
                theme_html.Cell(row.merged_30d),
                theme_html.Cell(role_html),
            ]
        )
    return theme_html.table(columns, rows)


def _draft_hiding_rows(pool: pd.DataFrame) -> pd.DataFrame:
    """PRs from ``pool`` that went to draft *after* a review was requested.

    KPI_SPEC exploit #6: this is exactly the subset ``_open_query``'s
    ``draft:false`` makes silently invisible everywhere else on the
    dashboard. A PR opened as a draft, or one that went to draft before ever
    being asked for review, is not in this frame — see
    ``pr_quality.draft_transitions``'s own docstring for why ``any``/``any``
    ordering is the rule, not "the first draft after the first request."
    """
    import pr_quality

    detail = pr_quality.draft_transitions(pool).detail
    if detail.empty:
        return detail
    return detail[detail["after_review_request"].fillna(False).astype(bool)].reset_index(drop=True)


def _render_draft_hiding_row(pool: pd.DataFrame) -> None:
    st.subheader("Went to draft after a review was requested")
    st.caption(
        "`_open_query` carries `draft:false`, so a PR that goes back to draft "
        "vanishes from Open, Stuck and every hygiene queue with no flag anywhere "
        "else on this page. This is the subset where that happened *after* someone "
        "was already asked to review it — a queue to open, not a verdict: a "
        "reviewer finding a real problem and the author pulling back to fix it "
        "looks identical here."
    )
    flagged = _draft_hiding_rows(pool)
    if flagged.empty:
        st.success(
            "No PR went to draft after a review was requested. A PR opened as a "
            "draft, or marked ready once and never drafted again, is not evidence "
            "of anything — that's just how it was written."
        )
        return
    theme_html.render(
        theme_html.table(
            [
                theme_html.Column("Author"),
                theme_html.Column("PR", kind="link"),
                theme_html.Column("Round trips", kind="num"),
            ],
            [
                [
                    theme_html.Cell(row.author),
                    theme_html.Cell(row.url),
                    theme_html.Cell(row.draft_round_trips),
                ]
                for row in flagged.itertuples()
            ],
        )
    )


@st.fragment
def _render_findings_and_citizenship(
    merged_prs: pd.DataFrame, open_prs: pd.DataFrame, extended_pool: pd.DataFrame
) -> None:
    import pr_quality

    left, mid, right = st.columns(3)
    judged_pool = pd.concat([merged_prs, open_prs], ignore_index=True) if not merged_prs.empty or not open_prs.empty else pd.DataFrame()

    with left:
        st.subheader("Devin findings per author")
        st.caption(
            'Share of **judged** PRs where Devin requested changes. Judged is shown '
            'so that "no findings" is never confused with "not reviewed".'
        )
        findings = _devin_findings_rows(judged_pool, merged_prs)
        if findings.empty:
            st.info("No AI-review data in the fetched PRs — needs the extended PR payload.")
        else:
            theme_html.render(_findings_table_html(findings))

    with mid:
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

    with right:
        _render_unprompted_reviews(extended_pool)


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

    import github_client

    open_prs = _team_prs(bundle.open_prs)
    merged_prs = _team_prs(bundle.merged_prs)

    env = github_client.load_github_env()
    token, org = env if env else (None, None)

    _render_code_kpis(
        open_prs,
        merged_prs,
        open_including_drafts=_open_pr_total_including_drafts(token, org),
    )
    st.divider()
    _render_repo_coverage(open_prs)
    st.divider()
    _render_stuck_queue(open_prs, bundle.df)
    st.divider()

    # The proactivity (unprompted reviews) and draft-hiding sections both need
    # the extended timeline payload, fetched once here and shared between them
    # rather than fetched twice. Degraded (switch off, throttled, unavailable)
    # is said out loud instead of quietly showing lean-query numbers as complete.
    extended_pool, extended_degraded, extended_reason = _extended_pr_pool(token, org)
    if extended_degraded and extended_reason:
        theme_html.render(
            theme_html.callout("warn", "PR quality data unavailable — lean mode", extended_reason)
        )

    _render_findings_and_citizenship(merged_prs, open_prs, extended_pool)
    st.divider()
    _render_draft_hiding_row(extended_pool)
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
