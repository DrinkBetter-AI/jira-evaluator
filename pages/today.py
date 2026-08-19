"""The Today page: the landing page, what needs a decision right now.

Split out of app.py in Task 1C. Every helper below is private to this page
(nothing else calls them) except ``_open_pr_signals`` and ``_stalled_rows``,
which Delivery / Code also use and which therefore live in ``render_shared``.

Task 3A rebuilt the render path as full HTML through ``theme_html`` (the
component kit built for the mockup - see ``docs/assumptions/1B.md``), because
Today is read-only and the mockup's own look is the point. The data-shaping
helpers above this line (``_ownerless``, ``_estimate_coverage``,
``_action_queues`` and friends) are unchanged and still feed the new render
path. ``_decision_card`` and ``_render_attention_band`` - the pre-3A
Streamlit-widget hero and decide cards - are kept, unchanged, purely because
``tests/test_today.py`` pins their exact structure (the three
dedented-out-of-the-``except`` regression tests in the "Task 2G" block); the
live page no longer calls either of them. ``_hero_fragment``/``_decide_cards``
below are their HTML-form replacements. "Do these next" stays exactly as it
was: Streamlit widgets (``st.expander``, ``st.dataframe`` with a
``LinkColumn``), because it is the one part of this page that is interactive,
and the component kit is explicitly display-only (``theme_html.py``'s own
module docstring says so).

``series`` is imported at module top rather than function-local, unlike the
other analytics modules this file reaches for (``next_actions``, ``hygiene``,
``integrity``, ``github_client``) - it is used in nearly every Task 3A
function below, and it is as dependency-free as ``theme_html`` itself (no
Streamlit import, no project-internal import besides ``pandas``; see its own
module docstring), so importing it eagerly costs nothing the app does not
already pay for ``pandas`` alone.
"""

from __future__ import annotations

import html
from typing import Any, NamedTuple
from urllib.parse import quote

import pandas as pd
import streamlit as st

import data_layer
import series
import theme
import theme_html
from data_layer import TRIAGE_STUCK_HOURS, _NO_OWNER_NAMES, _engineering_data
from page_shared import _as_frame, _text_or
from render_shared import (
    JIRA_BROWSE_BASE,
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
    # Allocated in its own try, separate from the hero body below, so that
    # even a failure in st.columns itself (the literal pre-split bug: an
    # exception "at or before" the allocation line) leaves hero/a/b/c bound
    # to something rather than unbound - the three decide cards need them
    # regardless of whether the hero rendered at all (docs/assumptions/2G.md,
    # bug 1).
    try:
        hero, a, b, c = st.columns([2.1, 1, 1, 1])
    except Exception as e:
        logger.exception("Error allocating attention band columns: %s", str(e))
        st.error(f"Error displaying PR metrics: {str(e)}")
        hero = a = b = c = st.container()

    try:
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

    # Dedented out of the try/except above: these three cards are the page's
    # reason to exist and must render on the happy path, not only when the
    # hero throws (see docs/assumptions/2G.md, bug 1).
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


# --------------------------------------------------------------------------- #
# Task 3A: full-HTML render path (theme_html), and the data it needs that the
# old Streamlit-widget path above never had to compute: 12-week series, deltas,
# and the links a mockup card can carry that a bare KPI number cannot.
# --------------------------------------------------------------------------- #

_THROUGHPUT_WEEKS = 12
# Chosen so the current + baseline windows end_to_end_cycle compares exactly
# cover the same 12*7=84 days this page already fetches for the throughput
# series - not the function's own baseline_days=90 default, which would ask
# for 180 days of resolved-ticket history this page has no other use for and
# has not fetched.
_CYCLE_BASELINE_DAYS = _THROUGHPUT_WEEKS * 7 / 2.0


def _freshness_caption() -> str:
    """"Data as of today HH:MM" - the mockup's own wording, computed fresh per render."""
    return pd.Timestamp.now().strftime("Data as of today %H:%M")


def _github_review_search_url() -> str | None:
    """A real GitHub search link for open, non-draft PRs with zero reviews.

    GitHub's search syntax has no qualifier for "has reviews but none
    approving" (the hero's actual "unapproved" number), only ``review:none``
    ("nobody has reviewed this at all" - the harsher, ``never_reviewed``
    subset). Linking to the nearest real, correct query beats either
    inventing a filter GitHub does not support or linking to "#" the way the
    mockup does; the gap between "unapproved" and "never reviewed" is
    recorded in ``docs/assumptions/3A.md``. Returns ``None`` when no GitHub
    token/org is configured, so a caller can drop the link rather than point
    at a query nobody can run.
    """
    import github_client

    try:
        env = github_client.load_github_env()
    except Exception:  # noqa: BLE001 - a bad GITHUB_ORG here just means no link
        return None
    if env is None:
        return None
    _token, org = env
    query = f"org:{org} is:pr is:open draft:false review:none"
    return f"https://github.com/search?q={quote(query)}&type=pullrequests"


def _jql_search_url(jql: str) -> str | None:
    """A Jira "Issue navigator" link for ``jql``, or ``None`` if the browse base isn't http(s).

    Built from ``JIRA_BROWSE_BASE`` (already vetted to be a plain http(s) URL
    by ``render_shared._browse_base``) with its trailing ``/browse`` swapped
    for ``/issues/?jql=``, rather than a second, separately-configured site
    URL - one source of truth for "which Jira site" stays render_shared's.
    """
    base = JIRA_BROWSE_BASE
    if not base.lower().startswith(("http://", "https://")):
        return None
    site = base[: -len("/browse")] if base.endswith("/browse") else base
    return f"{site}/issues/?jql={quote(jql)}"


def _jql_quoted_list(values: "tuple[str, ...]") -> str:
    return ", ".join(f'"{value}"' for value in values)


def _hero_fragment(prs: dict[str, Any], *, github_ready: bool, github_error: str) -> str:
    """The lead card: ``theme_html.hero()`` on the happy path, an honest callout otherwise.

    Same numbers ``_render_attention_band``'s hero half computes, drawn
    through the component kit instead of raw ``st.markdown``+``st.progress``.
    """
    if not github_ready:
        body = (
            f"GitHub could not be read ({github_error})."
            if github_error
            else "GitHub could not be read. Set DASHBOARD_GITHUB_TOKEN."
        )
        return theme_html.callout("warn", "PR review health cannot be read", body)

    total = prs.get("total", 0) or 0
    unapproved = prs.get("unapproved", 0) or 0
    share = (unapproved / total * 100.0) if total else 0.0
    oldest = prs.get("oldest_unreviewed_days")
    never_reviewed = prs.get("never_reviewed", 0) or 0
    oldest_text = (
        f" Oldest unreviewed PR: {oldest:.0f} days." if oldest and not pd.isna(oldest) else ""
    )
    sub = f"{never_reviewed} have never been reviewed at all.{oldest_text}"
    link_url = _github_review_search_url()
    link = (link_url, "See PRs with no review") if link_url else None
    return theme_html.hero(
        "Needs a decision",
        "crit",
        str(unapproved),
        f"of {total}",
        "open PRs have no approving review",
        share,
        sub,
        link,
    )


def _decide_cards(
    prs: dict[str, Any],
    *,
    github_ready: bool,
    triage_stuck: object,
    ownerless: int,
    open_total: int,
) -> list["theme_html.DecideCard"]:
    """The three decide cards as ``theme_html.DecideCard``s, each with a real action link.

    Same three numbers ``_render_attention_band``'s decide cards compute -
    this is their HTML-kit form, not a new definition of any of them.
    """
    triage_jql = (
        f"({data_layer.JQL}) AND status in ({_jql_quoted_list(data_layer.TRIAGE_STATUSES)}) "
        "ORDER BY created ASC"
    )
    ownership_jql = f"({data_layer.JQL}) AND assignee is EMPTY ORDER BY created ASC"
    triage_url = _jql_search_url(triage_jql)
    ownership_url = _jql_search_url(ownership_jql)
    review_url = _github_review_search_url() if github_ready else None

    share_note = (
        f"{ownerless / open_total:.0%} of open work belongs to nobody, so nobody's score carries it."
        if open_total
        else "No open tickets in scope."
    )
    return [
        theme_html.DecideCard(
            chip="⏱ Triage",
            tone="warn",
            n=_text_or(triage_stuck, "—"),
            what=f"tickets stuck in triage > {TRIAGE_STUCK_HOURS:.0f}h",
            why="Untriaged bugs age silently — nobody has decided they matter.",
            action=(triage_url, "Assign owners") if triage_url else None,
        ),
        theme_html.DecideCard(
            chip="👀 Review",
            tone="crit",
            n=str(prs.get("no_reviewer_asked", 0)) if github_ready else "—",
            what=f"PRs > {TODAY_NO_REVIEWER_DAYS:.0f} days with no reviewer asked",
            why="Nobody was requested and nobody has looked. These stall by default.",
            action=(review_url, "Assign reviewers") if review_url else None,
        ),
        theme_html.DecideCard(
            chip="∅ Ownership",
            tone="info",
            n=str(ownerless),
            what="open tickets with no owner",
            why=share_note,
            action=(ownership_url, "Run owner sweep") if ownership_url else None,
        ),
    ]


def _fetch_resolved_window(days: int) -> pd.DataFrame:
    """The Jira half of the 12-week throughput read: a page-private wider window.

    The bundle's own opening reads only carry 30 days of resolved tickets
    (``resolved_30``, sized for the per-person pie on other pages) - not
    enough for a 12-week trend. This is a second, wider, ``st.cache_data``-
    cached read (``fetch_resolved_tickets`` itself is cached 5 minutes), the
    same pattern ``render_shared._render_weekly_delivery`` already uses to
    get one person's history: a dedicated read, not a rework of the opening
    gather. Kept as its own function so tests can replace it without a real
    Jira credential.
    """
    return data_layer.fetch_resolved_tickets(
        creds_path=data_layer.CREDS_PATH,
        profile_name=data_layer.PROFILE_NAME,
        days=days,
        statuses=data_layer.RESOLVED_STATUSES,
        max_results=data_layer.MAX_RESULTS,
        page_size=data_layer.JIRA_PAGE_SIZE,
        schema_version=data_layer.FETCH_SCHEMA_VERSION,
    )


def _fetch_merged_window(days: int) -> pd.DataFrame:
    """The GitHub half of the 12-week throughput read. Empty frame, not an error, with no token."""
    import github_client

    env = github_client.load_github_env()
    if env is None:
        return pd.DataFrame()
    token, org = env
    return data_layer.fetch_merged_prs_cached(token, org, days, data_layer.FETCH_SCHEMA_VERSION)


class _Throughput(NamedTuple):
    """12 weeks of tickets-resolved and PRs-merged, plus the cycle-time pair they feed.

    ``ok=False`` means the chart section renders a labelled callout instead of
    a chart drawn from nothing - never zeros standing in for "could not read
    this". ``resolved``/``prs`` may still be partially populated even when
    ``ok`` is ``False`` (the Jira half can succeed while the GitHub half
    fails, or vice versa) so the tiles that only need one side still get a
    real spark.
    """

    ok: bool
    error: str
    resolved: list  # list[series.WeekBucket]
    prs: list  # list[series.WeekBucket]
    bots_excluded: int
    cycle: object  # integrity.EndToEndCycle | None


def _throughput(bundle: "_EngineeringData") -> _Throughput:
    """Fetch and bucket the 12-week series this page did not have before Task 3A.

    One extra Jira read and, if GitHub is configured, one extra GitHub read -
    each already ``st.cache_data``-cached for 5 minutes by the function it
    calls, so a second render of the same page in that window costs nothing
    further. A failed read degrades to an honest error string; a missing
    GitHub token degrades the PR half the same way the rest of this page
    already does, while the Jira half (and the cycle-time pair it feeds) is
    still returned.
    """
    import integrity

    days = _THROUGHPUT_WEEKS * 7
    try:
        resolved_df = _fetch_resolved_window(days)
    except Exception as exc:  # noqa: BLE001
        return _Throughput(False, f"Jira: {str(exc)[:200]}", [], [], 0, None)

    resolved_series = series.tickets_resolved_series(resolved_df, weeks=_THROUGHPUT_WEEKS)
    events = integrity.changelog_events(resolved_df)
    cycle = integrity.end_to_end_cycle(events, resolved_df, baseline_days=_CYCLE_BASELINE_DAYS)

    if not bundle.github_ready:
        return _Throughput(
            False,
            bundle.github_error or "GitHub unavailable",
            resolved_series.buckets,
            [],
            resolved_series.bots_excluded,
            cycle,
        )
    try:
        merged_df = _fetch_merged_window(days)
    except Exception as exc:  # noqa: BLE001
        return _Throughput(
            False,
            f"GitHub: {str(exc)[:200]}",
            resolved_series.buckets,
            [],
            resolved_series.bots_excluded,
            cycle,
        )
    pr_series = series.prs_merged_series(merged_df, weeks=_THROUGHPUT_WEEKS)
    return _Throughput(
        True,
        "",
        resolved_series.buckets,
        pr_series.buckets,
        resolved_series.bots_excluded + pr_series.bots_excluded,
        cycle,
    )


def _week_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Monday 00:00 UTC of the week containing ``ts`` - duplicated in miniature from ``series._week_start``.

    ``series.weekly_buckets`` only counts rows; the one thing on this page
    that needs to bucket a *value* (cycle-time days) by week instead of
    counting them has no primitive in series.py to call, and that module's
    week-bucketing helper is private. Copied rather than promoted, for a
    single caller.
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    ts = ts.normalize()
    return ts - pd.Timedelta(days=ts.dayofweek)


def _weekly_median_days(detail: pd.DataFrame, weeks: int) -> list[float]:
    """Median end-to-end cycle days per calendar week, from ``end_to_end_cycle``'s own ``detail`` rows.

    Weeks with no ticket finishing in them carry the nearest known median
    forward, then back (``ffill``/``bfill``) rather than a fabricated zero,
    which would read as "resolved instantly" instead of "nothing finished
    this week". An entirely empty ``detail`` - no finished tickets anywhere
    in the window - carries all the way through to 0.0 for every week: the
    honest floor for a tile with truly nothing behind it, not a crash.
    """
    current_start = _week_start(pd.Timestamp.now(tz="UTC"))
    buckets: list[list[float]] = [[] for _ in range(weeks)]
    if detail is not None and not detail.empty:
        resolved = pd.to_datetime(detail["resolved"], utc=True, errors="coerce")
        day_counts = pd.to_numeric(detail["days"], errors="coerce")
        for ts, day_count in zip(resolved, day_counts):
            if pd.isna(ts) or pd.isna(day_count):
                continue
            week = _week_start(ts)
            offset = (current_start - week).days // 7
            index = weeks - 1 - offset
            if 0 <= index < weeks:
                buckets[index].append(float(day_count))
    medians = [float(pd.Series(vals).median()) if vals else None for vals in buckets]
    filled = pd.Series(medians, dtype="float64").ffill().bfill()
    return [0.0 if pd.isna(v) else float(v) for v in filled]


def _stalled_weekly(stalled_rows: pd.DataFrame, weeks: int) -> list:
    """Weekly count of tickets *newly* crossing the stalled threshold - not a snapshot history.

    No board-state history is kept anywhere in this codebase (confirmed
    against ``snapshot.py``, which records a page's own drawing for PDF
    export, not the board's state over time), so "how many tickets were
    stalled as of week N" cannot be reconstructed. This buckets each
    currently-stalled ticket's own age (``now - age`` = roughly when it
    crossed the threshold) instead: real, derived from today's data, and it
    puts a ticket stalled 90 days in the one week it went stale, not in all
    thirteen weeks since.
    """
    import next_actions

    column = next_actions.STALLED_AGE_COLUMN
    if stalled_rows.empty or column not in stalled_rows.columns:
        return series.weekly_buckets(pd.DataFrame(), "went_stale_at", weeks)
    now = pd.Timestamp.now(tz="UTC")
    age_days = pd.to_numeric(stalled_rows[column], errors="coerce").fillna(0.0)
    went_stale_at = now - pd.to_timedelta(age_days, unit="D")
    synthetic = pd.DataFrame({"went_stale_at": went_stale_at})
    return series.weekly_buckets(synthetic, "went_stale_at", weeks)


def _estimate_weekly_ratio(df: pd.DataFrame, weeks: int) -> tuple[list[float | None], list[float]]:
    """Weekly estimate-coverage percent among policy-applicable tickets, grouped by creation week.

    Not a reconstructed "coverage as of week N" (same missing-history reason
    as ``_stalled_weekly``) but a real, non-fabricated cohort measure: of the
    policy-applicable tickets *created* in week N, what share currently
    carry an estimate. Returns two parallel lists of ``weeks`` values - the
    raw ratio (``None`` for a week with no policy-applicable ticket created
    in it: a real gap, which is what lets the tile's delta render neutral
    instead of inventing a verdict) and the same series filled
    forward/back for the sparkline, which cannot draw a ``None``.
    """
    from hygiene import estimate_policy

    if df.empty:
        return [None] * weeks, [0.0] * weeks
    scored = estimate_policy(df, BACKLOG_STATUSES)
    in_policy = scored[scored["policy_applies"].fillna(False).astype(bool)]
    if in_policy.empty:
        return [None] * weeks, [0.0] * weeks
    numerator = in_policy[in_policy["has_estimate"].fillna(False).astype(bool)]
    num_buckets = series.weekly_buckets(numerator, "created", weeks)
    den_buckets = series.weekly_buckets(in_policy, "created", weeks)
    raw: list[float | None] = [
        (n.value / d.value * 100.0) if d.value else None for n, d in zip(num_buckets, den_buckets)
    ]
    filled_series = pd.Series(raw, dtype="float64").ffill().bfill()
    filled = [0.0 if pd.isna(v) else float(v) for v in filled_series]
    return raw, filled


def _delta_text(d: "series.Delta", fmt: str) -> str:
    """The delta line's text: a formatted magnitude, or an honest word when there is none.

    ``fmt`` is a ``str.format``-style template taking the magnitude, e.g.
    ``"{:.0f} vs prior wk"``. Distinguishes the two ``magnitude is None``-
    adjacent cases ``series.delta`` itself distinguishes: ``None`` (no prior
    data - a cold start) and ``0`` (a genuine tie) render different words,
    never the same "flat" that would blur "nothing to compare against" into
    "nothing changed".
    """
    if d.magnitude is None:
        return "no prior data"
    if d.magnitude == 0:
        return "flat vs prior period"
    return fmt.format(d.magnitude)


def _tiles(
    bundle: "_EngineeringData",
    df: pd.DataFrame,
    stalled_rows: pd.DataFrame,
    stalled_clock: str,
    parked: int,
    throughput: _Throughput,
) -> list["theme_html.Tile"]:
    """The six "This week" tiles, each with a real sparkline and a scored delta.

    The tile set matches the mockup exactly: Median cycle time is here where
    the pre-3A page showed Ownerless (the mockup's own choice - see
    ``docs/assumptions/3A.md``). Every delta's ``higher_is_better`` is set by
    hand per metric, not inferred from direction, per ``series.delta``'s own
    contract: a stalled-tickets increase and a cycle-time decrease are both
    "up" in one case and "down" in the other, and only one of each pair is
    good.
    """
    weeks = _THROUGHPUT_WEEKS

    # --- Tickets resolved -------------------------------------------------
    resolved_7 = bundle.data.get("resolved_count_7")
    resolved_buckets = throughput.resolved
    resolved_spark = [b.value for b in resolved_buckets] if resolved_buckets else None
    if len(resolved_buckets) >= 3:
        d_resolved = series.delta(
            resolved_buckets[-2].value, resolved_buckets[-3].value, higher_is_better=True
        )
    else:
        d_resolved = series.delta(None, None, higher_is_better=True)
    tile_resolved = theme_html.Tile(
        label="Tickets resolved · 7d",
        value=_text_or(resolved_7, "—"),
        delta=_delta_text(d_resolved, "{:.0f} vs prior wk"),
        delta_dir=d_resolved.direction,
        delta_good=d_resolved.is_good,
        note="changelog-credited",
        spark=theme_html.spark(resolved_spark, "s1") if resolved_spark else None,
    )

    # --- PRs merged ---------------------------------------------------
    merged_7 = bundle.pr_count_7 if bundle.github_ready else None
    pr_buckets = throughput.prs
    pr_spark = [b.value for b in pr_buckets] if pr_buckets else None
    if len(pr_buckets) >= 3:
        d_prs = series.delta(pr_buckets[-2].value, pr_buckets[-3].value, higher_is_better=True)
    else:
        d_prs = series.delta(None, None, higher_is_better=True)
    tile_prs = theme_html.Tile(
        label="PRs merged · 7d",
        value=(_text_or(merged_7, "—") if bundle.github_ready else "—"),
        delta=_delta_text(d_prs, "{:.0f} vs prior wk"),
        delta_dir=d_prs.direction,
        delta_good=d_prs.is_good,
        note="bots excluded" if bundle.github_ready else "GitHub unavailable",
        spark=theme_html.spark(pr_spark, "s2") if pr_spark else None,
    )

    # --- Median cycle time -------------------------------------------
    cycle = throughput.cycle
    median_days = cycle.median_days if cycle is not None else None
    baseline_days = cycle.baseline_median_days if cycle is not None else None
    d_cycle = series.delta(median_days, baseline_days, higher_is_better=False)
    detail = cycle.detail if cycle is not None else pd.DataFrame()
    cycle_spark = _weekly_median_days(detail, weeks)
    if median_days is not None:
        cycle_value, cycle_unit = f"{median_days:.1f}", "d"
        cycle_note = "In Progress → resolved, from changelog"
    else:
        cycle_value, cycle_unit = "—", ""
        cycle_note = (cycle.reason if cycle is not None else None) or "insufficient data"
    tile_cycle = theme_html.Tile(
        label="Median cycle time",
        value=cycle_value,
        unit=cycle_unit,
        delta=_delta_text(d_cycle, "{:.1f}d vs baseline"),
        delta_dir=d_cycle.direction,
        delta_good=d_cycle.is_good,
        note=cycle_note,
        spark=theme_html.spark(cycle_spark, "s1"),
    )

    # --- Open tickets ---------------------------------------------------
    created_buckets = series.weekly_buckets(df, "created", weeks)
    open_spark = [b.value for b in created_buckets]
    d_open = series.delta(
        created_buckets[-2].value, created_buckets[-3].value, higher_is_better=False
    )
    tile_open = theme_html.Tile(
        label="Open tickets",
        value=str(len(df)),
        delta=_delta_text(d_open, "{:.0f} new/wk vs prior wk"),
        delta_dir=d_open.direction,
        delta_good=d_open.is_good,
        note=(f"Backlog excluded ({parked} parked)" if parked else "current JQL scope"),
        spark=theme_html.spark(open_spark, "s1"),
    )

    # --- Stalled ----------------------------------------------------
    stalled_buckets = _stalled_weekly(stalled_rows, weeks)
    stalled_spark = [b.value for b in stalled_buckets]
    d_stalled = series.delta(
        stalled_buckets[-2].value, stalled_buckets[-3].value, higher_is_better=False
    )
    tile_stalled = theme_html.Tile(
        label=f"Stalled {TODAY_STALLED_DAYS:.0f}d+",
        value=str(len(stalled_rows)),
        delta=_delta_text(d_stalled, "{:.0f} newly stalled/wk"),
        delta_dir=d_stalled.direction,
        delta_good=d_stalled.is_good,
        note=f"by {stalled_clock}, not edit age",
        spark=theme_html.spark(stalled_spark, "s8"),
        help=f"No status transition in {TODAY_STALLED_DAYS:.0f} days — field edits don't reset this clock.",
    )

    # --- Estimate coverage -----------------------------------------
    estimated, estimable = _estimate_coverage(df)
    coverage_note = f"{estimated} of {estimable} past Backlog" if estimable else "nothing to estimate"
    raw_ratio, filled_ratio = _estimate_weekly_ratio(df, weeks)
    d_cov = series.delta(raw_ratio[-2], raw_ratio[-3], higher_is_better=True)
    tile_cov = theme_html.Tile(
        label="Estimate coverage",
        value=f"{estimated / estimable:.0%}" if estimable else "—",
        delta=_delta_text(d_cov, "{:.0f}pp vs prior wk"),
        delta_dir=d_cov.direction,
        delta_good=d_cov.is_good,
        note=coverage_note,
        spark=theme_html.spark(filled_ratio, "s3"),
    )

    return [tile_resolved, tile_prs, tile_cycle, tile_open, tile_stalled, tile_cov]


def _throughput_chart_fragment(throughput: _Throughput) -> str:
    """The "Throughput — 12 weeks" card: a real ``theme_html.linechart``, or a callout.

    Never a chart drawn from nothing: when neither read succeeded
    (``throughput.ok`` false and no buckets at all came back), this is a
    labelled ``callout()`` instead of an empty or zeroed chart.
    """
    if not throughput.resolved and not throughput.prs:
        return (
            '<div class="card"><p class="chart-title">Throughput — 12 weeks</p>'
            + theme_html.callout(
                "warn",
                "Throughput chart unavailable",
                throughput.error or "Could not be read.",
            )
            + "</div>"
        )
    resolved_vals = [b.value for b in throughput.resolved]
    pr_vals = [b.value for b in throughput.prs]
    labels = [b.label for b in throughput.resolved] or [b.label for b in throughput.prs]
    chart_svg = theme_html.linechart(
        {"Tickets resolved": resolved_vals, "PRs merged": pr_vals},
        labels,
        ["s1", "s2"],
        aria=f"Line chart of weekly tickets resolved and PRs merged over {_THROUGHPUT_WEEKS} weeks",
    )
    legend_html = theme_html.legend([("Tickets resolved", "s1"), ("PRs merged", "s2")])
    sub = "Tickets resolved and PRs merged per week"
    if not throughput.ok:
        sub += f" — {throughput.error}" if throughput.error else " — partial read"
    sub += f" — excl. bots ({throughput.bots_excluded} bot merges)"
    return (
        '<div class="card"><p class="chart-title">Throughput — 12 weeks</p>'
        f"<p class=\"chart-sub\">{html.escape(sub)}</p>"
        f"{chart_svg}{legend_html}</div>"
    )


def _status_hbars_fragment(df: pd.DataFrame, parked: int) -> str:
    """The "Where open work sits" card: ``theme_html.hbars()`` instead of the old plotly bar."""
    if df.empty:
        return (
            '<div class="card"><p class="chart-title">Where open work sits</p>'
            + theme_html.callout("info", "No open tickets", "Nothing returned for the current JQL scope.")
            + "</div>"
        )
    counts = df["status"].fillna("(none)").astype(str).value_counts()
    top = float(counts.max()) if len(counts) else 1.0
    bars = [
        theme_html.Bar(name=str(status), value=str(int(count)), pct=max(2.0, 100.0 * count / top), tone="s1")
        for status, count in counts.items()
    ]
    sub = f"{len(df)} open tickets by status — ranked, not sliced."
    if parked:
        sub += f" {parked} Backlog ticket(s) are not counted here."
    return (
        '<div class="card"><p class="chart-title">Where open work sits</p>'
        f'<p class="chart-sub">{html.escape(sub)}</p>'
        f"{theme_html.hbars(bars)}</div>"
    )


def _hero_and_decide_fragment(
    prs: dict[str, Any],
    *,
    github_ready: bool,
    github_error: str,
    triage_stuck: object,
    ownerless: int,
    open_total: int,
) -> str:
    """The ``.hero`` grid: the lead card plus the three decide cards, hero failure isolated.

    The three decide cards are the page's reason to exist (see the "Task 2G"
    regression block at the top of ``tests/test_today.py``, pinned against the
    pre-3A Streamlit path) - a broken hero must not take them down too. Hero
    and decide-card computation are two separate ``try`` blocks for exactly
    that reason: an exception in ``_hero_fragment`` degrades the lead card to
    a callout, never the whole ``.hero`` row.
    """
    try:
        hero_html = _hero_fragment(prs, github_ready=github_ready, github_error=github_error)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error rendering Today hero: %s", str(exc))
        hero_html = theme_html.callout(
            "crit", "PR review health could not be drawn", f"Error: {str(exc)[:200]}"
        )
    try:
        decide_html = theme_html.decide_cards(
            _decide_cards(
                prs,
                github_ready=github_ready,
                triage_stuck=triage_stuck,
                ownerless=ownerless,
                open_total=open_total,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error rendering Today decide cards: %s", str(exc))
        decide_html = theme_html.callout(
            "crit", "Decisions could not be drawn", f"Error: {str(exc)[:200]}"
        )
    return f'<div class="hero">{hero_html}{decide_html}</div>'


def _render_today_page() -> None:
    """The landing page: what needs a decision, then this week in six numbers.

    Deliberately org-wide and filter-free. A reader who lands here is asking
    "what is wrong right now", not "what is wrong within these four statuses" -
    the scoped views are the pages behind it. Rendered as full HTML through
    ``theme_html`` (Task 3A) - see the module docstring for why "Do these
    next" alone stays Streamlit widgets.
    """
    theme_html.css()
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

    theme_html.render(
        theme_html.page_header(
            "VinoVoss Engineering",
            _freshness_caption(),
            {"Jira": True, "GitHub": bundle.github_ready},
        ),
        _hero_and_decide_fragment(
            prs,
            github_ready=bundle.github_ready,
            github_error=bundle.github_error,
            triage_stuck=bundle.data.get("triage_stuck_count"),
            ownerless=ownerless,
            open_total=int(len(df)),
        ),
    )

    st.divider()
    stalled_rows, stalled_clock = _stalled_rows(df, events=bundle.events)
    queues, unreadable = _action_queues(bundle, df, stalled=stalled_rows)
    _render_next_actions(queues, unknown=unreadable)

    st.divider()
    throughput = _throughput(bundle)
    tiles = _tiles(bundle, df, stalled_rows, stalled_clock, parked, throughput)
    theme_html.render(
        theme_html.section(
            "This week",
            "Counts are telemetry, not performance — people are scored on the People page.",
        ),
        theme_html.tiles(tiles),
        '<div class="charts2" style="margin-top:14px">'
        + _throughput_chart_fragment(throughput)
        + _status_hbars_fragment(df, parked)
        + "</div>",
        theme_html.foot(
            "Today is org-wide and filter-free by design — scoped views live on the pages behind it."
        ),
    )
