"""The People page: one row per active person, ranked within their own role.

Task 1C's extraction left this page rendering a *teams* summary plus an
assignee workload rollup and calling it "People" - there was no per-person
table at all. Task 3B replaces that with the real thing: the full roster
table (``people_table.people_table``, built in 2B) rendered through the
HTML kit, a role filter and a person selector as the hybrid mode's
Streamlit widgets (chips and a selector cannot post back from injected
HTML - ``docs/assumptions/1C.md``/the task brief calls this the documented
cost of the hybrid), and an auditable per-person scorecard with evidence
links, built fresh from the same bulk frames the table uses rather than by
reusing ``render_shared._render_scorecard``'s Streamlit-native rendering
(that function stays exactly as it was; see ``docs/assumptions/3B.md`` for
why touching it was not necessary to make the within-role claim true).

Card composition (``.card`` wrappers, the ``.score2`` two-up grid) is the
caller's job per ``docs/assumptions/1B.md`` - ``_card()`` below is this
page's own private copy of the same wrapper ``pages/delivery.py`` already
carries, not a shared import.
"""

from __future__ import annotations

import html
from urllib.parse import quote

import pandas as pd
import streamlit as st

import integrity
import kpi
import people_table as people_table_mod
import roles
import series
import theme_html
from data_layer import RESOLVED_STATUSES, _engineering_context
from hygiene import estimate_policy
from page_shared import TAB_ENGINEERING, _download_report
from render_shared import BACKLOG_STATUSES, _metrics_df, _one_person_instead

WEEKS = 12

# Deterministic, non-role-coded avatar colours - the mockup cycles s1..s8 by
# eye; this cycles the same eight tokens by a stable hash of the name so the
# same person gets the same colour on every render without a hand-kept map.
_AVATAR_HUES = ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8")

# Roles the mockup renders muted (exec, no-rubric, unmapped) get the neutral
# ink shade instead of a series colour - matches ``var(--ink-4)`` used for
# Igor/Angel/Arsalan's avatars in the design.
_MUTED_ROLE_STATUSES = frozenset({"exec", "no_rubric", "role_unknown", "unclassified"})


def _freshness_caption() -> str:
    """A short "how fresh is this" line for ``page_header``, without importing ``app``.

    Page-private by convention - see ``pages/delivery.py``'s identical
    docstring. Importing from ``app`` here would be the circular import
    Task 1C's split exists to prevent.
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


def _card(body_html: str, *, title: str = "", subtitle: str = "") -> str:
    """Wrap a bare new-form fragment in the mockup's card chrome.

    Same markup ``pages/delivery.py``'s ``_card()`` produces - duplicated
    here rather than imported because neither page owns the other.
    """
    parts = ['<div class="card">']
    if title:
        parts.append(f'<h3 class="chart-title">{html.escape(title)}</h3>')
    if subtitle:
        parts.append(f'<p class="chart-sub">{html.escape(subtitle)}</p>')
    parts.append(body_html)
    parts.append("</div>")
    return "".join(parts)


def _combined_org_events(
    bundle_events: pd.DataFrame, resolved_tickets: pd.DataFrame | None
) -> pd.DataFrame:
    """Changelog for open tickets, plus tickets resolved recently.

    ``bundle_events`` alone only carries history for tickets the board's JQL
    still returns (open, non-Done) - a ticket that fully closed drops out of
    it, and with it every resolving transition it ever made.
    ``resolved_tickets`` (``bundle.data["resolved_30"]``, fetched with
    ``expand=changelog``) is the other half. Deduped on ``(key, entry_id)``
    so a ticket in both frames is not double-counted. Same construction as
    ``pages/delivery.py``'s ``_combined_org_events`` - duplicated rather than
    imported, since that function is private to a page this task does not
    own.

    Blind to: a ticket that fully resolved and closed more than 30 days ago
    (the window ``resolved_30`` was fetched over) is in neither frame -
    invisible here the same way it is invisible to Delivery's org-wide
    figures built the same way.
    """
    parts = []
    if bundle_events is not None and not bundle_events.empty:
        parts.append(bundle_events)
    if isinstance(resolved_tickets, pd.DataFrame) and not resolved_tickets.empty:
        resolved_events = integrity.changelog_events(resolved_tickets)
        if not resolved_events.empty:
            parts.append(resolved_events)
    if not parts:
        return pd.DataFrame(columns=integrity.EVENT_COLUMNS)
    combined = pd.concat(parts, ignore_index=True)
    if {"key", "entry_id"} <= set(combined.columns):
        combined = combined.drop_duplicates(subset=["key", "entry_id"])
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Small formatting helpers - every one distinguishes "not measured" (NA) from
# a real, informative zero, per KPI_SPEC.md's "never a substituted zero"
# rule. None of these ever coalesce a missing value to "0".
# ---------------------------------------------------------------------------


def _is_na(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def _avatar_hue(person: str, role_status: str) -> str:
    if role_status in _MUTED_ROLE_STATUSES:
        return "gray"
    return _AVATAR_HUES[abs(hash(person)) % len(_AVATAR_HUES)]


def _initials(name: str) -> str:
    parts = [p for p in str(name).replace("-", " ").split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _role_status(reason: str, role: str | None) -> str:
    """Which muted state a ``no_score_reason`` string represents, for styling."""
    if reason == roles.EXEC_REASON:
        return "exec"
    if reason.startswith(roles.NO_RUBRIC_REASON):
        return "no_rubric"
    if reason == "role unknown" or role is None:
        return "role_unknown"
    return "insufficient"


def _person_cell_html(person: str, role: str | None, role_status: str) -> str:
    label = roles.ROLE_LABELS.get(role, role or "unmapped")
    av = theme_html.avatar(_initials(person), _avatar_hue(person, role_status))
    return f'<div class="who">{av}<b>{html.escape(str(person))}</b> {theme_html.rolechip(label)}</div>'


def _score_tone(pct: float) -> str:
    if pct >= 70.0:
        return "good"
    if pct >= 40.0:
        return "s4"
    return "crit"


def _score_cell_html(row: pd.Series) -> str:
    score = row["score"]
    if not _is_na(score):
        pct = float(score)
        n = row["n"]
        note = f"n={int(n)}" if not _is_na(n) else ""
        return theme_html.scorebar(pct, _score_tone(pct), f"{pct:.0f}", note)
    reason = str(row["no_score_reason"] or "")
    status = _role_status(reason, row["role"])
    if status == "exec":
        label, note = "exec — not scored", "output shown, never ranked"
    elif status == "no_rubric":
        label, note = "no rubric defined", "not board-shaped work"
    elif status == "role_unknown":
        label, note = "role unknown", "not on the roster"
    else:
        measurable = row["measurable_pct"]
        pct_text = f"{float(measurable):.0f}%" if not _is_na(measurable) else "0%"
        label = f"no score — {pct_text} measurable"
        note = reason[:90] if reason else "insufficient data"
    chip = theme_html.chip(label, "gray")
    note_html = f' <span class="nnote">{html.escape(note)}</span>' if note else ""
    return chip + note_html


def _delivered_cell_html(row: pd.Series) -> str:
    points = row["delivered_points"]
    if _is_na(points):
        return "—"
    body = f"{float(points):.0f} pts"
    trivial = row["n_trivial_share"]
    share = row["trivial_share"]
    if not _is_na(share) and not _is_na(trivial) and int(trivial) > 0:
        pct = float(share) * 100.0
        flag = " ⚑" if pct >= 35.0 else ""
        body += f' <span class="nnote">{pct:.0f}% trivial{flag}</span>'
    return body


def _cycle_cell_text(row: pd.Series) -> str:
    median = row["cycle_median"]
    return "—" if _is_na(median) else f"{float(median):.1f}d"


def _reviews_cell_text(row: pd.Series) -> str:
    given = row["reviews_given"]
    ttfr = row["ttfr_hours"]
    given_text = "—" if _is_na(given) else str(int(given))
    ttfr_text = "—" if _is_na(ttfr) else f"{float(ttfr):.0f}h"
    return f"{given_text} · {ttfr_text}"


def _estimate_cell_html(row: pd.Series) -> str:
    ratio = row["estimate_ratio"]
    iqr = row["estimate_iqr"]
    if _is_na(ratio):
        return "—"
    body = f"×{float(ratio):.1f}"
    if not _is_na(iqr):
        flag = " ⚑" if float(iqr) >= 1.5 else ""
        body += f' <span class="nnote">IQR {float(iqr):.1f}{flag}</span>'
    return body


def _flags_cell_html(row: pd.Series) -> str:
    count = row["flag_count"]
    if _is_na(count):
        return theme_html.chip("—", "gray")
    count = int(count)
    if count == 0:
        return theme_html.chip("0", "gray")
    severity = row["flag_severity"]
    severity = 0.0 if _is_na(severity) else float(severity)
    tone = "crit" if severity >= 3.0 else "warn"
    return theme_html.chip(f"⚑ {count}", tone)


_TABLE_COLUMNS = [
    theme_html.Column("Person", "html"),
    theme_html.Column("Score", "html"),
    theme_html.Column(
        "Delivered",
        "html",
        help="Resolved tickets weighted by PR size band; trivial-heavy mixes are flagged.",
    ),
    theme_html.Column("Cycle (med)", "text"),
    theme_html.Column("Reviews given · TTFR", "text"),
    theme_html.Column("Estimate accuracy", "html"),
    theme_html.Column("Flags", "html"),
]


def _table_row(row: pd.Series) -> list[theme_html.Cell]:
    reason = str(row["no_score_reason"] or "")
    status = _role_status(reason, row["role"]) if _is_na(row["score"]) else "scored"
    return [
        theme_html.Cell(_person_cell_html(row["person"], row["role"], status)),
        theme_html.Cell(_score_cell_html(row)),
        theme_html.Cell(_delivered_cell_html(row)),
        theme_html.Cell(_cycle_cell_text(row)),
        theme_html.Cell(_reviews_cell_text(row)),
        theme_html.Cell(_estimate_cell_html(row)),
        theme_html.Cell(_flags_cell_html(row)),
    ]


def _sorted_table(table_df: pd.DataFrame) -> pd.DataFrame:
    if table_df.empty:
        return table_df
    out = table_df.copy()
    out["_role_sort"] = out["role"].fillna("~unmapped")
    out["_score_sort"] = pd.to_numeric(out["score"], errors="coerce").fillna(-1.0)
    out = out.sort_values(
        ["_role_sort", "_score_sort", "person"], ascending=[True, False, True]
    )
    return out.drop(columns=["_role_sort", "_score_sort"])


def _role_filter_options(table_df: pd.DataFrame) -> tuple[list[str], dict[str, str | None]]:
    """(pill labels, label -> role key) - "All roles" plus every role present."""
    present = sorted({r for r in table_df.get("role", pd.Series(dtype=object)) if pd.notna(r) and r})
    labels = ["All roles"] + [roles.ROLE_LABELS.get(r, r) for r in present]
    label_to_role: dict[str, str | None] = {"All roles": None}
    for r in present:
        label_to_role[roles.ROLE_LABELS.get(r, r)] = r
    return labels, label_to_role


def _jira_search_url(browse_base: str, jql: str) -> str:
    site = browse_base.rsplit("/browse", 1)[0] if "/browse" in browse_base else browse_base
    return f"{site}/issues/?jql={quote(jql)}"


def _github_search_url(org: str | None, query: str) -> str:
    scoped = f"org:{org} {query}" if org else query
    return f"https://github.com/search?q={quote(scoped)}&type=pullrequests"


# ---------------------------------------------------------------------------
# The per-person scorecard: built fresh from the same bulk frames the table
# uses, so a code-role person's headline number agrees with their row (both
# go through ``kpi.components``/``kpi.overall`` on the same inputs). This is
# also where the cohort peer-percentile is wired for the component-level
# breakdown - see the module docstring and ``docs/assumptions/3B.md``.
# ---------------------------------------------------------------------------


def _credited_map(
    events: pd.DataFrame, resolved_tickets: pd.DataFrame, window_days: float, now=None
) -> dict[str, int]:
    result = integrity.credited_resolutions(
        events,
        tickets=resolved_tickets,
        window_days=window_days,
        now=now,
        resolved_statuses=RESOLVED_STATUSES,
    )
    if result.by_person is None or result.by_person.empty:
        return {}
    return dict(zip(result.by_person["person"], result.by_person["credited_resolutions"].astype(int)))


def _reopened_map(
    all_tickets: pd.DataFrame, events: pd.DataFrame, window_days: float = 90.0, now=None
) -> dict[str, int]:
    frame = integrity.integrity_flags(
        all_tickets, events, window_days=window_days, now=now, resolved_statuses=RESOLVED_STATUSES
    )
    if frame is None or frame.empty:
        return {}
    return dict(zip(frame["person"], frame["reresolved_tickets"].astype(int)))


def _components_for_person(
    person: str,
    roster: roles.Roster,
    open_tickets: pd.DataFrame,
    gradable_tickets: pd.DataFrame,
    prs: pd.DataFrame,
    combined_events: pd.DataFrame,
    resolved_tickets: pd.DataFrame,
    now=None,
) -> tuple[list[kpi.Component] | None, roles.RubricLookup, roles.CohortResult | None]:
    """The component breakdown for one person, or ``None`` when their rubric
    has no component-computation wired (see ``docs/assumptions/2C.md``:
    QA/PM/design/infra rubrics exist but nothing feeds them yet - future
    work, not this page's).
    """
    lookup = roles.rubric_for_person(roster, person)
    if lookup.status != "scored" or lookup.role not in roles.CODE_ROLES:
        return None, lookup, None

    events_present = combined_events is not None and not combined_events.empty
    resolved_7_map = _credited_map(combined_events, resolved_tickets, 7.0, now) if events_present else {}
    resolved_90_map = _credited_map(combined_events, resolved_tickets, 90.0, now) if events_present else {}
    reopened_map = _reopened_map(open_tickets, combined_events, 90.0, now) if events_present else {}

    login = roster.login_for(person)
    owned = (
        open_tickets[open_tickets.get("assignee", pd.Series(dtype=object)) == person]
        if not open_tickets.empty
        else pd.DataFrame()
    )
    gradable = (
        gradable_tickets[gradable_tickets.get("assignee", pd.Series(dtype=object)) == person]
        if not gradable_tickets.empty
        else pd.DataFrame()
    )
    prs_mine = (
        prs[prs.get("author", pd.Series(dtype=object)) == login]
        if login and not prs.empty
        else pd.DataFrame()
    )

    resolved_7 = resolved_7_map.get(person, 0) if events_present else None
    resolved_90 = resolved_90_map.get(person, 0) if events_present else None
    reopened_90 = reopened_map.get(person, 0) if events_present else None

    cohort = roles.peer_cohort(roster, person)
    peer_pool = None
    if cohort.sufficient and events_present:
        peer_pool = {person: resolved_7}
        for peer in cohort.peers:
            peer_pool[peer] = resolved_7_map.get(peer, 0)

    parts = kpi.components(
        owned,
        gradable,
        resolved_7,
        resolved_90,
        reopened_90,
        prs_mine,
        peer_resolved_7=peer_pool,
        include_gaps=True,
    )

    # A cohort below MIN_PEERS makes ``kpi.components`` skip "Delivery vs
    # team" entirely, and ``include_gaps`` then fills it with the generic
    # "insufficient data - needs the same-week resolved counts of at least 3
    # teammates" placeholder. That is true but not the specific number the
    # task requires: the reader should see how many peers this person's own
    # cohort actually has. Overwritten here, once, rather than left generic -
    # this is the wiring that makes the page's within-role claim honest.
    if not cohort.sufficient:
        for part in parts:
            if part.name == "Delivery vs team":
                part.detail = cohort.reason
                part.n = cohort.peer_count

    order = list(kpi.WEIGHTS)
    parts.sort(key=lambda p: order.index(p.name) if p.name in order else len(order))
    return parts, lookup, cohort


def _delivered_spark(prs: pd.DataFrame, login: str | None) -> str:
    mine = (
        prs[prs.get("author", pd.Series(dtype=object)) == login]
        if login and not prs.empty
        else pd.DataFrame()
    )
    buckets = series.weekly_buckets(mine, "merged_at", weeks=WEEKS)
    values = [b.value for b in buckets]
    return theme_html.spark(values, "s1", fill=True, w=280, h=56)


def _evidence_rows(
    person: str,
    roster: roles.Roster,
    prs: pd.DataFrame,
    combined_events: pd.DataFrame,
    resolved_tickets: pd.DataFrame,
    browse_base: str,
    github_org: str | None,
    now=None,
) -> list[tuple[str, str, str]]:
    """(label, count text, url) for the four evidence rows - every count links
    to the query it came from, per the task brief: "a score nobody can audit
    is a score nobody will believe."
    """
    login = roster.login_for(person)
    events_present = combined_events is not None and not combined_events.empty

    merged_count = 0
    if login and isinstance(prs, pd.DataFrame) and not prs.empty and "author" in prs.columns:
        mine = prs[prs["author"] == login]
        if "merged_at" in mine.columns:
            merged_count = int(mine["merged_at"].notna().sum())
        else:
            merged_count = int(len(mine))

    resolved_90 = _credited_map(combined_events, resolved_tickets, 90.0, now).get(person, 0) if events_present else 0

    reviews_count = 0
    if login and isinstance(prs, pd.DataFrame) and not prs.empty:
        import pr_quality

        citizenship = pr_quality.review_citizenship(prs)
        if not citizenship.empty and "reviewer" in citizenship.columns:
            row = citizenship[citizenship["reviewer"] == login]
            if not row.empty:
                reviews_count = int(row["reviews_given"].iloc[0])

    churn_count = 0
    if events_present:
        churn = integrity.estimate_churn(combined_events, window_days=90.0, now=now)
        if not churn.empty and "author" in churn.columns:
            mine = churn[
                (churn["author"] == person) & churn["direction"].isin(["raised", "lowered"])
            ]
            churn_count = int(len(mine))

    jql_resolved = f'assignee = "{person}" AND resolutiondate >= -90d ORDER BY resolutiondate DESC'
    jql_estimates = f'assignee = "{person}" AND updated >= -90d ORDER BY updated DESC'

    return [
        (
            "PRs merged this period",
            str(merged_count),
            _github_search_url(github_org, f"is:pr is:merged author:{login}" if login else "is:pr is:merged"),
        ),
        (
            "Tickets resolved (changelog-credited)",
            str(resolved_90),
            _jira_search_url(browse_base, jql_resolved),
        ),
        (
            "Reviews given",
            str(reviews_count),
            _github_search_url(github_org, f"is:pr reviewed-by:{login}" if login else "is:pr reviewed"),
        ),
        (
            "Estimate revisions mid-flight",
            str(churn_count),
            _jira_search_url(browse_base, jql_estimates),
        ),
    ]


def _scorecard_fragment(
    person: str,
    row: pd.Series,
    roster: roles.Roster,
    open_tickets: pd.DataFrame,
    gradable_tickets: pd.DataFrame,
    prs: pd.DataFrame,
    combined_events: pd.DataFrame,
    resolved_tickets: pd.DataFrame,
    browse_base: str,
    github_org: str | None,
) -> str:
    role = row["role"]
    role_label = roles.ROLE_LABELS.get(role, role or "unmapped")

    parts, lookup, cohort = _components_for_person(
        person, roster, open_tickets, gradable_tickets, prs, combined_events, resolved_tickets
    )

    header = theme_html.section(
        f"Scorecard — {person}",
        f"{role_label} rubric" if lookup.status == "scored" else lookup.reason,
    )

    if parts is None:
        # Exec, no-rubric, unknown role, or a rubric with no component-
        # computation wired yet (QA/PM/design/infra - docs/assumptions/2C.md
        # names this future work, not this task's). Never a fabricated
        # component table for these - the honest state is the sentence
        # itself.
        body = theme_html.callout(
            "info",
            f"{person} — {lookup.reason}",
            "This role has no per-component breakdown wired into the scorecard yet."
            if lookup.status == "scored"
            else "Shown for board attribution only; never ranked.",
        )
        return header + _card(body)

    overall_score = kpi.overall(parts)
    cov = kpi.coverage(parts)
    components = [
        theme_html.Component(
            name=p.name,
            weight=kpi.WEIGHTS.get(p.name, 0.0),
            score=(p.score if p.sufficient else None),
            note=p.detail,
            sufficient=p.sufficient,
        )
        for p in parts
    ]
    scorecard_html = theme_html.scorecard(
        components,
        "n/a" if overall_score is None else f"{overall_score:.0f}",
        f"{cov.covered_weight:.0f}",
        cov.note,
    )

    login = roster.login_for(person)
    spark_html = _delivered_spark(prs, login)
    evidence = _evidence_rows(
        person, roster, prs, combined_events, resolved_tickets, browse_base, github_org
    )
    evidence_columns = [
        theme_html.Column("Evidence", "text"),
        theme_html.Column("links", "link"),
    ]
    # ``table()``'s link cells render the URL, not the count - the count is
    # what a reader wants to see next to the label, so it is folded into the
    # label text itself rather than added as a third column the kit's link
    # cell would swallow.
    evidence_rows = [
        [theme_html.Cell(f"{label} ({count})"), theme_html.Cell(url)]
        for label, count, url in evidence
    ]
    evidence_html = theme_html.table(evidence_columns, evidence_rows)

    right_card = _card(
        f'<p class="chart-title">Delivered, {WEEKS} weeks <span class="nnote">'
        "(PRs merged)</span></p>"
        f"{spark_html}"
        f'<div style="margin-top:12px">{evidence_html}</div>'
    )
    return header + f'<div class="score2">{scorecard_html}{right_card}</div>'


def _render_people_page() -> None:
    """One row per active person, ranked within their own role.

    Table and scorecard are HTML through the kit (Mode: Hybrid); the role
    filter and the person selector are ``st.pills``/``st.selectbox`` above
    them, since a filter chip cannot post back from injected HTML.
    """
    theme_html.css()
    bundle, view, slot = _engineering_context()
    if _one_person_instead(bundle, view, slot):
        return

    roster = roles.load_roster()

    resolved_tickets = bundle.data.get("resolved_30")
    resolved_tickets = resolved_tickets if isinstance(resolved_tickets, pd.DataFrame) else pd.DataFrame()
    combined_events = _combined_org_events(bundle.events, resolved_tickets)

    open_tickets = estimate_policy(_metrics_df(bundle.df, include_backlogs=False), BACKLOG_STATUSES)

    import ticket_quality

    gradable_tickets = (
        ticket_quality.score_tickets(bundle.df) if not bundle.df.empty else bundle.df
    )

    prs_frames = []
    if bundle.github_ready:
        for frame in (bundle.open_prs, bundle.merged_prs):
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                prs_frames.append(frame)
    prs = pd.concat(prs_frames, ignore_index=True) if prs_frames else pd.DataFrame()

    table_df = people_table_mod.people_table(
        open_tickets, resolved_tickets, gradable_tickets, prs, combined_events, roster=roster
    )

    theme_html.render(
        theme_html.page_header(
            "VinoVoss · People",
            _freshness_caption(),
            {"Jira": True, "GitHub": bool(bundle.github_ready)},
        ),
        theme_html.section(
            "People",
            "Scores compare within a role only — a backend engineer is never ranked "
            "against a manual tester. A component with too little data says so instead "
            "of scoring, and below 60 points of measurable weight there is no score at "
            "all. Peer comparison uses each person's own role cohort (frontend, "
            "frontend-mobile and mobile share one cohort); a cohort under 3 peers shows "
            "its actual peer count instead of a percentile.",
        ),
    )

    if table_df.empty:
        theme_html.render(_card("No active person has any recorded activity yet."))
        _download_report(slot, TAB_ENGINEERING)
        return

    role_labels, label_to_role = _role_filter_options(table_df)
    selected_label = st.pills("Role", options=role_labels, default=role_labels[0], key="people_role_pill")
    selected_role = label_to_role.get(selected_label or role_labels[0])

    shown = table_df if selected_role is None else table_df[table_df["role"] == selected_role]
    shown = _sorted_table(shown)

    rows = [_table_row(row) for _, row in shown.iterrows()]
    theme_html.render(_card(theme_html.table(_TABLE_COLUMNS, rows)))

    person_options = shown["person"].tolist() if not shown.empty else table_df["person"].tolist()
    if not person_options:
        _download_report(slot, TAB_ENGINEERING)
        return

    selected_person = st.selectbox("Person", options=person_options, key="people_person_select")
    selected_row = table_df[table_df["person"] == selected_person]
    if selected_row.empty:
        _download_report(slot, TAB_ENGINEERING)
        return
    row = selected_row.iloc[0]

    import github_client
    from render_shared import JIRA_BROWSE_BASE

    github_org = None
    if bundle.github_ready:
        try:
            env = github_client.load_github_env()
        except Exception:  # noqa: BLE001 - the scorecard degrades to an org-less search link
            env = None
        if env:
            _token, github_org = env

    theme_html.render(
        _scorecard_fragment(
            selected_person,
            row,
            roster,
            open_tickets,
            gradable_tickets,
            prs,
            combined_events,
            resolved_tickets,
            JIRA_BROWSE_BASE,
            github_org,
        )
    )

    _download_report(slot, TAB_ENGINEERING)
