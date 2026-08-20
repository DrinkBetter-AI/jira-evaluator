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

import access_gate
import estimate_accuracy
import integrity
import kpi
import people_table as people_table_mod
import pr_quality
import role_kpis
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


# The windows this page scores over are 90 days wide - ``_credited_map(..., 90.0)``,
# ``integrity_flags(..., 90.0)``, ``estimate_churn(window_days=90.0)`` - but the
# bundle's opening read only carries 30 (``resolved_30``, sized for the per-person
# pie elsewhere). Feeding a 30-day frame to a 90-day window does not fail, it
# understates: resolutions 31-90 days ago are in neither frame, so a person's
# prior-period pace reads as zero and a rate measured against it reads as
# perfect. This is the same dedicated wider read ``pages/today.py`` makes for its
# 12-week throughput line, for the same reason, kept as its own function so tests
# can replace it without a Jira credential.
_RESOLVED_WINDOW_DAYS = 90


def _fetch_resolved_window(days: int = _RESOLVED_WINDOW_DAYS) -> pd.DataFrame:
    """Resolved tickets over ``days``, with changelog, through the cached reader."""
    import data_layer

    return data_layer.fetch_resolved_tickets(
        creds_path=data_layer.CREDS_PATH,
        profile_name=data_layer.PROFILE_NAME,
        days=days,
        statuses=data_layer.RESOLVED_STATUSES,
        max_results=data_layer.MAX_RESULTS,
        page_size=data_layer.JIRA_PAGE_SIZE,
        schema_version=data_layer.FETCH_SCHEMA_VERSION,
    )


def _resolved_window_or_bundle(bundle) -> pd.DataFrame:
    """The 90-day resolved frame, falling back to the bundle's 30-day one.

    A failed wider read degrades to exactly what this page saw before - the
    bundle's ``resolved_30`` - rather than to an empty frame, because an empty
    frame here would zero every changelog-credited count on the page. The
    fallback is narrower than the window that reads it, which is the flaw this
    function exists to fix; it is the safe direction to fail in (an understated
    count, never an invented one) and it is what runs when Jira is unreachable.
    """
    fallback = bundle.data.get("resolved_30")
    fallback = fallback if isinstance(fallback, pd.DataFrame) else pd.DataFrame()
    try:
        wider = _fetch_resolved_window()
    except Exception:  # noqa: BLE001 - a narrower window, not a broken page
        return fallback
    if not isinstance(wider, pd.DataFrame) or wider.empty:
        return fallback
    return wider


def _combined_org_events(
    bundle_events: pd.DataFrame, resolved_tickets: pd.DataFrame | None
) -> pd.DataFrame:
    """Changelog for open tickets, plus tickets resolved recently.

    ``bundle_events`` alone only carries history for tickets the board's JQL
    still returns (open, non-Done) - a ticket that fully closed drops out of
    it, and with it every resolving transition it ever made.
    ``resolved_tickets`` (this page's own 90-day read, see
    ``_resolved_window_or_bundle``, fetched with ``expand=changelog``) is the
    other half. Deduped on ``(key, entry_id)`` so a ticket in both frames is
    not double-counted. Same construction as ``pages/delivery.py``'s
    ``_combined_org_events`` - duplicated rather than imported, since that
    function is private to a page this task does not own.

    Blind to: a ticket that fully resolved and closed before the window
    ``resolved_tickets`` was fetched over is in neither frame - invisible
    here the same way it is invisible to Delivery's org-wide figures built
    the same way. That window is what has to match the window the callers
    score over; a 30-day frame read by a 90-day metric is the bug
    ``_resolved_window_or_bundle`` exists to close.
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


def _credited_by_person(
    events: pd.DataFrame, resolved_tickets: pd.DataFrame, window_days: float, now=None
) -> pd.DataFrame:
    """``credited_resolutions``' per-person frame, or an empty one with its columns."""
    result = integrity.credited_resolutions(
        events,
        tickets=resolved_tickets,
        window_days=window_days,
        now=now,
        resolved_statuses=RESOLVED_STATUSES,
    )
    if result.by_person is None or result.by_person.empty:
        return pd.DataFrame(columns=["person", "credited_resolutions", "keys"])
    return result.by_person


def _credited_map(
    events: pd.DataFrame, resolved_tickets: pd.DataFrame, window_days: float, now=None
) -> dict[str, int]:
    by_person = _credited_by_person(events, resolved_tickets, window_days, now)
    if by_person.empty:
        return {}
    return dict(zip(by_person["person"], by_person["credited_resolutions"].astype(int)))


def _credited_keys(by_person: pd.DataFrame, person: str) -> list[str]:
    """The ticket keys behind one person's credited count, in the order stored."""
    if by_person.empty or "keys" not in by_person.columns:
        return []
    row = by_person[by_person["person"] == person]
    if row.empty:
        return []
    raw = row["keys"].iloc[0]
    return [key.strip() for key in str(raw or "").split(",") if key.strip()]


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
    role_inputs: role_kpis.RoleKpiInputs | None = None,
) -> tuple[list[kpi.Component] | None, roles.RubricLookup, roles.CohortResult | None]:
    """The component breakdown for one person, or ``None`` when their role
    has no rubric (exec, seo/wine/advisor, off-roster).

    Code roles go through :func:`kpi.components` exactly as before. Scored
    non-code roles (QA, PM, designer, infrastructure) go through
    :func:`role_kpis.components_for` - the wiring
    ``docs/assumptions/2C.md`` used to name as future work.
    """
    lookup = roles.rubric_for_person(roster, person)
    if lookup.status != "scored":
        return None, lookup, None
    if lookup.role not in roles.CODE_ROLES:
        if role_inputs is None:
            return None, lookup, None
        gradable = (
            gradable_tickets[gradable_tickets.get("assignee", pd.Series(dtype=object)) == person]
            if not gradable_tickets.empty
            else pd.DataFrame()
        )
        owned = (
            open_tickets[open_tickets.get("assignee", pd.Series(dtype=object)) == person]
            if not open_tickets.empty
            else pd.DataFrame()
        )
        parts = role_kpis.components_for(role_inputs, person, lookup.role, owned, gradable)
        return parts, lookup, None

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

    credited_90 = (
        _credited_by_person(combined_events, resolved_tickets, 90.0, now)
        if events_present
        else pd.DataFrame(columns=["person", "credited_resolutions", "keys"])
    )
    resolved_90 = 0
    if not credited_90.empty:
        mine_credit = credited_90[credited_90["person"] == person]
        if not mine_credit.empty:
            resolved_90 = int(mine_credit["credited_resolutions"].iloc[0])

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

    # The count beside this link is credited from the changelog - whoever
    # authored the resolving transition - so the link has to select the same
    # tickets. ``assignee = person AND resolutiondate >= -90d`` selects a
    # different population twice over: it reads the current assignee, which is
    # the field the credit fix exists to stop trusting, and it reads Jira's
    # ``resolutiondate``, which a move into "Review in Staging" does not set
    # even though this team counts that as resolved. Listing the credited keys
    # is the only query that returns exactly the tickets that were counted.
    credited_keys = _credited_keys(credited_90, person)
    if credited_keys:
        jql_resolved = "key IN (" + ", ".join(credited_keys) + ") ORDER BY resolutiondate DESC"
    else:
        # Nothing credited: a query that honestly returns nothing, rather than
        # an assignee search that would return somebody else's tickets.
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
    role_inputs: role_kpis.RoleKpiInputs | None = None,
) -> str:
    role = row["role"]
    role_label = roles.ROLE_LABELS.get(role, role or "unmapped")

    parts, lookup, cohort = _components_for_person(
        person,
        roster,
        open_tickets,
        gradable_tickets,
        prs,
        combined_events,
        resolved_tickets,
        role_inputs=role_inputs,
    )

    header = theme_html.section(
        f"Scorecard — {person}",
        f"{role_label} rubric" if lookup.status == "scored" else lookup.reason,
    )

    if parts is None:
        # Exec, no-rubric, or unknown role. Never a fabricated component
        # table for these - the honest state is the sentence itself.
        body = theme_html.callout(
            "info",
            f"{person} — {lookup.reason}",
            "This role's component breakdown had no data to compute from."
            if lookup.status == "scored"
            else "Shown for board attribution only; never ranked.",
        )
        return header + _card(body)

    if lookup.role in roles.CODE_ROLES:
        overall_score = kpi.overall(parts)
        cov = kpi.coverage(parts)
        weights: dict[str, float] = dict(kpi.WEIGHTS)
        covered_weight, note = cov.covered_weight, cov.note
    else:
        rubric = lookup.rubric
        assert rubric is not None  # "scored" status guarantees it
        overall_score, covered_weight, note = role_kpis.score_from_parts(parts, rubric)
        weights = rubric.weights()
    components = [
        theme_html.Component(
            name=p.name,
            weight=weights.get(p.name, 0.0),
            score=(p.score if p.sufficient else None),
            note=p.detail,
            sufficient=p.sufficient,
        )
        for p in parts
    ]
    scorecard_html = theme_html.scorecard(
        components,
        "n/a" if overall_score is None else f"{overall_score:.0f}",
        f"{covered_weight:.0f}",
        note,
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
    evidence_html = theme_html.table(
        evidence_columns, evidence_rows, tab=TAB_ENGINEERING, section="People"
    )

    right_card = _card(
        f'<p class="chart-title">Delivered, {WEEKS} weeks <span class="nnote">'
        "(PRs merged)</span></p>"
        f"{spark_html}"
        f'<div style="margin-top:12px">{evidence_html}</div>'
    )
    return header + f'<div class="score2">{scorecard_html}{right_card}</div>'


# ---------------------------------------------------------------------------
# The per-person KPI detail: every KPI_SPEC.md family that is safe to show to
# the whole team, for the selected person, with sample sizes and honest NA
# states. The integrity families (padding, grooming, reciprocity) are NOT
# here - they render only for a proven admin session, below.
# ---------------------------------------------------------------------------


def _kv_table(rows: list[tuple[str, str]]) -> str:
    """A two-column metric/value table through the kit."""
    columns = [theme_html.Column("Metric", "text"), theme_html.Column("Value", "html")]
    cells = [[theme_html.Cell(label), theme_html.Cell(value)] for label, value in rows]
    return theme_html.table(columns, cells, tab=TAB_ENGINEERING, section="People")


def _na(need: str) -> str:
    return f'— <span class="nnote">{html.escape(need)}</span>'


def _person_row(frame: pd.DataFrame, key_col: str, who: str) -> pd.Series | None:
    if frame is None or frame.empty or key_col not in frame.columns:
        return None
    mine = frame[frame[key_col] == who]
    return None if mine.empty else mine.iloc[0]


def _credited_in(detail: pd.DataFrame, person: str, days: float, now: pd.Timestamp) -> int | None:
    """Distinct tickets whose resolving transition ``person`` authored within ``days``."""
    if detail is None or detail.empty:
        return None
    mine = detail[detail["credited"].fillna(False).astype(bool) & (detail["author"] == person)]
    if mine.empty:
        return 0
    ts = pd.to_datetime(mine["ts"], utc=True, errors="coerce")
    recent = mine[ts >= now - pd.Timedelta(days=days)]
    return int(recent["key"].nunique())


def _kpi_detail_fragment(
    person: str,
    login: str | None,
    role_inputs: role_kpis.RoleKpiInputs,
    prs: pd.DataFrame,
    github_ready: bool,
) -> str:
    """Three cards: output & flow, estimates, code quality & reviews.

    Everything here is a KPI_SPEC.md §3 metric already computed by
    ``integrity``/``pr_quality``/``estimate_accuracy``; this fragment only
    selects one person's row and renders it with its ``n``. Counts are shown
    next to size and cycle context, never alone - a raw count on its own is
    exploit #3 (ticket splitting) waiting to be farmed.
    """
    now = role_inputs.now if role_inputs.now is not None else pd.Timestamp.now(tz="UTC")

    # --- Output & flow ---------------------------------------------------
    flow_rows: list[tuple[str, str]] = []
    r7 = _credited_in(role_inputs.credited_detail, person, 7.0, now)
    r90 = _credited_in(role_inputs.credited_detail, person, 90.0, now)
    if r90 is None:
        flow_rows.append(("Tickets resolved (changelog-credited)", _na("needs changelog history")))
    else:
        flow_rows.append(
            (
                "Tickets resolved (changelog-credited)",
                f"<b>{r7}</b> in 7d · <b>{r90}</b> in 90d "
                '<span class="nnote">credited to whoever moved the ticket, not who holds it</span>',
            )
        )
    cycle = _person_row(role_inputs.cycle_by_person, "person", person)
    if cycle is not None and pd.notna(cycle.get("median_lead_time_days")):
        n_cycle = int(cycle.get("lead_time_tickets", 0) or 0)
        extras = []
        for label, col in (("in progress", "median_in_progress_days"), ("in review", "median_review_days")):
            value = cycle.get(col)
            if pd.notna(value):
                extras.append(f"{label} {float(value):.1f}d")
        extra_text = f' <span class="nnote">{" · ".join(extras)}</span>' if extras else ""
        flow_rows.append(
            (
                "Cycle time (median, start → resolve)",
                f"<b>{float(cycle['median_lead_time_days']):.1f}d</b> over n={n_cycle}{extra_text}",
            )
        )
    else:
        flow_rows.append(("Cycle time (median, start → resolve)", _na("needs ≥1 completed, timed ticket")))

    if github_ready and login:
        bands = _person_row(pr_quality.size_bands(prs), "author", login)
        if bands is not None and int(bands.get("prs", 0) or 0) > 0:
            mix = " / ".join(
                f"{int(bands[b])} {b}" for b in ("trivial", "small", "medium", "large", "oversized")
            )
            median_lines = bands.get("median_changed_lines")
            median_text = (
                f' <span class="nnote">median {float(median_lines):.0f} changed lines</span>'
                if pd.notna(median_lines)
                else ""
            )
            flow_rows.append(("PRs by size band", f"{mix}{median_text}"))
            share = bands.get("trivial_share")
            if pd.notna(share):
                flag = " ⚑" if float(share) >= 0.35 else ""
                flow_rows.append(
                    (
                        "Trivial share",
                        f"{100.0 * float(share):.0f}%{flag} "
                        '<span class="nnote">a spike of trivial PRs against an unchanged median is ticket-splitting</span>',
                    )
                )
        else:
            flow_rows.append(("PRs by size band", _na("no PRs in the fetched window")))
    else:
        flow_rows.append(("PRs by size band", _na("GitHub data unavailable" if not github_ready else "no GitHub login mapped")))

    flow_card = _card(
        _kv_table(flow_rows),
        title="Output & flow",
        subtitle="Size-weighted and cycle-anchored — never a bare ticket count.",
    )

    # --- Estimates -------------------------------------------------------
    est_rows: list[tuple[str, str]] = []
    acc = _person_row(role_inputs.accuracy, "assignee", person)
    if acc is not None and pd.notna(acc.get("median_ratio")):
        n_acc = int(acc.get("tickets", 0) or 0)
        spread = ""
        if pd.notna(acc.get("p25_ratio")) and pd.notna(acc.get("p75_ratio")):
            spread = (
                f' <span class="nnote">p25–p75 ×{float(acc["p25_ratio"]):.1f}–×{float(acc["p75_ratio"]):.1f}</span>'
            )
        est_rows.append(
            (
                "Logged / estimated (median)",
                f"<b>×{float(acc['median_ratio']):.2f}</b> over n={n_acc}{spread}",
            )
        )
        iqr = acc.get("iqr")
        if pd.notna(iqr):
            flag = " ⚑ wide spread — not really estimating" if float(iqr) >= 1.5 else ""
            est_rows.append(("Ratio IQR", f"{float(iqr):.2f}{flag}"))
        est_rows.append(
            (
                "Hours, estimated vs logged",
                f"{float(acc.get('estimated_hours', 0) or 0):.0f}h estimated · "
                f"{float(acc.get('logged_hours', 0) or 0):.0f}h logged",
            )
        )
        if not bool(acc.get("enough_data", False)):
            est_rows.append(
                ("Verdict", _na(f"withheld below {role_kpis.MIN_N} tickets — this is noise, not signal"))
            )
    else:
        est_rows.append(
            ("Logged / estimated (median)", _na("needs finished tickets with both an estimate and logged time"))
        )
    est_card = _card(
        _kv_table(est_rows),
        title="Estimate integrity",
        subtitle="Both numbers are self-reported; this shows inconsistency, not ground truth.",
    )

    # --- Code quality & reviews -----------------------------------------
    quality_rows: list[tuple[str, str]] = []
    if github_ready and login and isinstance(prs, pd.DataFrame) and not prs.empty:
        devin = _person_row(pr_quality.devin_findings_by_author(prs), "author", login)
        if devin is not None and int(devin.get("prs_judged", 0) or 0) > 0:
            judged = int(devin["prs_judged"])
            cr = int(devin.get("prs_changes_requested", 0) or 0)
            share = devin.get("changes_requested_share")
            share_text = f"{100.0 * float(share):.0f}%" if pd.notna(share) else "—"
            quality_rows.append(
                (
                    "AI review: changes requested",
                    f"<b>{share_text}</b> ({cr} of {judged} judged PRs) "
                    '<span class="nnote">share of judged PRs — shipping more is not scored worse</span>',
                )
            )
            reviewed = devin.get("prs_ai_reviewed")
            if pd.notna(reviewed):
                quality_rows.append(
                    (
                        "AI review coverage",
                        f"{int(reviewed)} of their PRs carried an AI review "
                        '<span class="nnote">zero findings on an unreviewed PR means unreviewed, not clean</span>',
                    )
                )
        else:
            quality_rows.append(("AI review: changes requested", _na("no AI-judged PRs in the window")))
        abandoned = _person_row(pr_quality.abandoned_rate(prs), "author", login)
        if abandoned is not None and int(abandoned.get("closed_prs", 0) or 0) > 0:
            quality_rows.append(
                (
                    "Abandoned PRs",
                    f"{int(abandoned['abandoned'])} of {int(abandoned['closed_prs'])} decided PRs "
                    f"({100.0 * float(abandoned['abandoned_rate']):.0f}%) closed without merging",
                )
            )
        trace = _person_row(pr_quality.traceability(prs), "author", login)
        if trace is not None and int(trace.get("judgeable", 0) or 0) > 0:
            quality_rows.append(
                (
                    "Traceability (merged PRs naming a ticket)",
                    f"{int(trace['with_key'])} of {int(trace['judgeable'])} "
                    f"({100.0 * float(trace['traceability']):.0f}%)",
                )
            )
        citizenship = _person_row(pr_quality.review_citizenship(prs), "reviewer", login)
        if citizenship is not None and int(citizenship.get("reviews_given", 0) or 0) > 0:
            ttfr = citizenship.get("median_hours_to_first_review")
            ttfr_text = f" · median {float(ttfr):.0f}h to first review" if pd.notna(ttfr) else ""
            quality_rows.append(
                (
                    "Reviews given",
                    f"<b>{int(citizenship['reviews_given'])}</b> across "
                    f"{int(citizenship.get('distinct_authors_reviewed', 0) or 0)} colleague(s)"
                    f"{ttfr_text} · {int(citizenship.get('approvals_given', 0) or 0)} approvals, "
                    f"{int(citizenship.get('changes_requested_given', 0) or 0)} change requests",
                )
            )
        else:
            quality_rows.append(
                (
                    "Reviews given",
                    '<b>0</b> <span class="nnote">reviewing is scored work — never reviewing anyone is a real zero</span>',
                )
            )
    else:
        quality_rows.append(("PR quality", _na("GitHub data unavailable for this render")))
    quality_card = _card(
        _kv_table(quality_rows),
        title="Code quality & review citizenship",
        subtitle="From GitHub review records — written by the system, not by the person measured.",
    )

    return (
        theme_html.section(
            f"KPI detail — {person}",
            "Every number carries its sample size; a dash with a note means not measured, never zero.",
        )
        + f'<div class="score2">{flow_card}{est_card}</div>'
        + quality_card
    )


# ---------------------------------------------------------------------------
# The admin-only integrity section. Two rules, inherited from
# pages/integrity.py: computed ONLY for a proven admin session (a non-admin
# render never calls into integrity's flag functions from here), and every
# card states its innocent reading. Flags are prompts for a conversation,
# never verdicts.
# ---------------------------------------------------------------------------


def _integrity_admin_fragment(
    person: str,
    login: str | None,
    role_inputs: role_kpis.RoleKpiInputs,
    all_tickets: pd.DataFrame,
    combined_events: pd.DataFrame,
    prs: pd.DataFrame,
) -> str:
    rows: list[tuple[str, str]] = []

    flags_frame = integrity.integrity_flags(all_tickets, combined_events, window_days=30.0)
    flag_row = _person_row(flags_frame, "person", person)
    if flag_row is not None and int(flag_row.get("flag_count", 0) or 0) > 0:
        tripped = [name for name in integrity.FLAG_NAMES if bool(flag_row.get(name, False))]
        evidence_bits = []
        for name in tripped:
            evidence = str(flag_row.get(f"{name}_evidence", "") or "")[:160]
            evidence_bits.append(f"<b>{html.escape(name.replace('_', ' '))}</b>: {html.escape(evidence)}")
        rows.append(("Flags (30d)", "<br>".join(evidence_bits)))
    else:
        rows.append(("Flags (30d)", "none tripped — which is not a clean bill of health; doing nothing trips nothing"))

    touches = _person_row(integrity.cosmetic_touches(combined_events), "person", person)
    if touches is not None:
        rows.append(
            (
                "Cosmetic touches (14d)",
                f"{int(touches['cosmetic_touches'])} field-only edits across "
                f"{int(touches['cosmetic_tickets'])} tickets vs {int(touches['status_transitions'])} real "
                f"status moves · busiest day {html.escape(str(touches.get('busiest_day') or '—'))} "
                f"({int(touches.get('busiest_day_touches', 0) or 0)} touches)",
            )
        )

    churn = role_inputs.churn
    if churn is not None and not churn.empty and "author" in churn.columns:
        mine = churn[(churn["author"] == person) & churn["direction"].isin(["raised", "lowered"])]
        if not mine.empty:
            raised = mine[mine["direction"] == "raised"]
            added = float(pd.to_numeric(raised.get("delta_hours"), errors="coerce").fillna(0).sum())
            examples = ", ".join(
                f"{html.escape(str(r['key']))} ({'+' if r['direction'] == 'raised' else '−'}"
                f"{abs(float(r.get('delta_hours') or 0)):.0f}h, day {float(r.get('days_after_start') or 0):.0f})"
                for _, r in mine.head(4).iterrows()
            )
            rows.append(
                (
                    "Estimate churn (90d, mid-flight)",
                    f"{len(raised)} raise(s) adding {added:.0f}h, {len(mine) - len(raised)} lowering(s) — {examples}",
                )
            )
        else:
            rows.append(("Estimate churn (90d, mid-flight)", "no mid-flight estimate edits"))

    padding = _person_row(estimate_accuracy.padding_index(all_tickets), "assignee", person)
    if padding is not None:
        if bool(padding.get("enough_data", False)):
            rows.append(
                (
                    "Padding index",
                    f"median ×{float(padding['median_ratio']):.2f}, "
                    f"{100.0 * float(padding['under_run_share']):.0f}% of tickets finished under 60% of estimate "
                    f"over {int(padding['tickets'])} tickets — {html.escape(str(padding.get('verdict') or ''))}",
                )
            )
        else:
            rows.append(
                (
                    "Padding index",
                    f"no verdict — only {int(padding.get('tickets', 0) or 0)} tickets, "
                    f"below the {estimate_accuracy.MIN_TICKETS_FOR_VERDICT}-ticket floor (reading noise otherwise)",
                )
            )

    if login and isinstance(prs, pd.DataFrame) and not prs.empty:
        recip = _person_row(pr_quality.reciprocity(prs).by_person, "reviewer", login)
        if recip is not None and int(recip.get("reviews_given", 0) or 0) > 0:
            partner = html.escape(str(recip.get("top_partner") or "—"))
            share = recip.get("top_partner_share")
            share_text = f"{100.0 * float(share):.0f}%" if pd.notna(share) else "—"
            rows.append(
                (
                    "Review reciprocity",
                    f"top partner {partner} ({share_text} of reviews given) · "
                    f"{int(recip.get('rubber_stamp_approvals', 0) or 0)} empty-body, zero-thread approvals",
                )
            )
        selfm = _person_row(pr_quality.self_merge(prs), "author", login)
        if selfm is not None and int(selfm.get("merged_prs", 0) or 0) > 0:
            rows.append(
                (
                    "Self-merges",
                    f"{int(selfm.get('self_merged', 0) or 0)} of {int(selfm['merged_prs'])} own merges; "
                    f"<b>{int(selfm.get('merged_without_outside_approval', 0) or 0)}</b> merged with no outside approval",
                )
            )

    body = _kv_table(rows) + theme_html.innocent(
        "Every row above has an innocent reading — grooming can be curation, an estimate raise can be "
        "honest scope discovery, high reciprocity is arithmetic on a small team. These are prompts to "
        "open the tickets and PRs named, never verdicts."
    )
    return theme_html.section(
        f"Integrity signals — {person}",
        "Admin-only. Not computed, not rendered, for non-admin sessions.",
    ) + _card(body)


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

    resolved_tickets = _resolved_window_or_bundle(bundle)
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

    all_tickets = (
        pd.concat(
            [f for f in (open_tickets, resolved_tickets) if isinstance(f, pd.DataFrame) and not f.empty],
            ignore_index=True,
            sort=False,
        )
        if (not open_tickets.empty or not resolved_tickets.empty)
        else pd.DataFrame()
    )

    # The org-wide frames every non-code rubric and the KPI detail read -
    # built once per render and shared with people_table so the table's
    # scores and the scorecard's breakdown come from the same computation.
    role_inputs = role_kpis.build_inputs(
        open_tickets,
        all_tickets,
        combined_events,
        prs,
        resolved_tickets,
        resolved_statuses=RESOLVED_STATUSES,
    )

    table_df = people_table_mod.people_table(
        open_tickets,
        resolved_tickets,
        gradable_tickets,
        prs,
        combined_events,
        roster=roster,
        role_kpi_inputs=role_inputs,
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
    theme_html.render(
        _card(theme_html.table(_TABLE_COLUMNS, rows, tab=TAB_ENGINEERING, section="People"))
    )

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
            role_inputs=role_inputs,
        )
    )

    login = roster.login_for(selected_person)
    theme_html.render(
        _kpi_detail_fragment(
            selected_person, login, role_inputs, prs, bool(bundle.github_ready)
        )
    )

    # The integrity families are Angel-only, same two-gate rule as the
    # Integrity page: admin_access_granted() is the cheap, never-prompting
    # check, so a non-admin session neither renders NOR COMPUTES any of it -
    # the fragment function is simply never called.
    if access_gate.admin_access_granted():
        theme_html.render(
            _integrity_admin_fragment(
                selected_person, login, role_inputs, all_tickets, combined_events, prs
            )
        )
    elif access_gate.admin_password_configured():
        theme_html.render(
            theme_html.foot(
                "Integrity signals for this person are admin-only — unlock them once on the "
                "Integrity page and they will appear here too."
            )
        )

    _download_report(slot, TAB_ENGINEERING)
