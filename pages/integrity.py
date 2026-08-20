"""The Integrity page: CEO-only, and never a verdict.

``integrity.integrity_flags`` (the roll-up) and its four constituent
functions — ``cosmetic_touches``, ``estimate_churn``, ``reresolve_events``
and ``pr_quality.reciprocity``/``flag_self_merges`` — have been sitting in
the codebase, tested, with zero call sites. This page is the wiring, not new
math: every number below already exists in ``integrity.py``/``pr_quality.py``
and is only assembled and linked back to its evidence here.

Access is two gates, not one. ``access_gate.require_password()`` (checked in
``app.py::main`` before any page runs) is the shared password every hourly
contractor already knows — it is not, and must never become, what protects
this page. ``access_gate.require_admin_password()`` is a second, independent
credential (``DASHBOARD_ADMIN_PASSWORD``) that only Angel has. It is called
first, before a single other line in this module runs:

    def _render_integrity_page() -> None:
        if not access_gate.require_admin_password():
            return
        import integrity
        ...

That ordering is the whole security property this file exists to guarantee.
A non-admin session never gets past the ``if``, so it never reaches the
``import integrity`` line, let alone a call into
``integrity.cosmetic_touches``/``estimate_churn``/``reresolve_events``/
``integrity_flags`` or ``pr_quality.reciprocity``/``flag_self_merges``/
``self_merge``. Not hidden-but-computed — not computed at all. See
``docs/assumptions/4.md`` for why the page is registered unhidden-behind-a-
password rather than gated purely by nav visibility, and why
``require_admin_password`` is written as a return-value gate rather than
relying on ``st.stop()`` alone.

Three framing rules apply to every card below (DEVIN_PLAN WP6,
``KPI_SPEC.md``):

1. Each card states its innocent reading, via ``theme_html.innocent()``,
   *always* — including when the card has nothing to show. An empty card is
   not a clean bill of health; it is the absence of one signal.
2. Every number links to the ticket or PR behind it. A flag nobody can check
   is an accusation, not a metric.
3. No fixed thresholds. Every card shows the top three candidates by
   magnitude, with evidence (DEVIN_PLAN §4.4) — never "more than N", which a
   contractor two edits under the line reads as permission.
"""

from __future__ import annotations

import html as _html

import pandas as pd
import streamlit as st

import access_gate
import theme_html
from render_shared import _jira_ticket_url

INTEGRITY_PAGE_TITLE = "Integrity"

# The same window defaults integrity.py's own functions carry — named again
# here, rather than left implicit, so a reader of this page's source sees
# the window without opening integrity.py.
_COSMETIC_WINDOW_DAYS = 14.0
_CHURN_WINDOW_DAYS = 90.0
_RERESOLVE_WINDOW_DAYS = 90.0
_FLAGS_WINDOW_DAYS = 30.0

# DEVIN_PLAN §4.4: no fixed threshold, the top three by magnitude, always
# with evidence. One constant, used by every card, so "three" cannot drift
# to a different number on one card without it being a visible, deliberate
# edit here.
TOP_N = 3

# Minimum peers a role needs in the *current window's* cosmetic-touch data
# before its own median is trusted as a baseline; below that, a solo role
# (Infra, PM, ML — ROSTER.md) falls back to the org-wide median rather than
# comparing a person only to themselves, which would always read as zero
# excess no matter how large their count is.
_MIN_ROLE_PEERS_FOR_BASELINE = 2


# ---------------------------------------------------------------------------
# Small building blocks shared by every card
# ---------------------------------------------------------------------------


def _wrap_card(title: str, subtitle: str, body_html: str, innocent_text: str) -> str:
    """The mockup's card chrome plus the mandatory innocent-reading footer.

    Duplicated rather than shared with ``pages/planning.py``'s/``pages/
    today.py``'s own ``_card`` helpers, for the same reason those two don't
    share one either (docs/assumptions/1B.md: card composition is the
    caller's, not the kit's) — the one difference here is that
    ``theme_html.innocent()`` is appended unconditionally, every single
    call, which is this page's own rule, not the kit's.
    """
    parts = [f'<div class="card"><h3 class="chart-title">{_html.escape(title)}</h3>']
    if subtitle:
        parts.append(f'<p class="chart-sub">{_html.escape(subtitle)}</p>')
    parts.append(body_html)
    parts.append(theme_html.innocent(innocent_text))
    parts.append("</div>")
    return "".join(parts)


def _top_n(frame: pd.DataFrame, by: str, n: int = TOP_N) -> pd.DataFrame:
    """The ``n`` largest rows of ``frame`` by ``frame[by]`` — magnitude, not a cutoff.

    No row is ever dropped for falling "below a threshold"; only ever for
    not being in the top ``n``. Ties keep their original (already-computed)
    order, since ``sort_values`` is stable.
    """
    if frame is None or frame.empty:
        return frame
    return frame.sort_values(by, ascending=False, kind="mergesort").head(n)


def _linked_jira_keys(keys: list[str], limit: int = 6) -> str:
    """Every ``keys`` entry as a real link to its Jira ticket, comma-joined.

    Never handed raw Jira text — a ticket key is escaped the same as any
    other value that reaches ``theme_html.evrow``'s verbatim fragment.
    """
    shown = [str(k).strip() for k in keys if str(k).strip()][:limit]
    if not shown:
        return '<span class="dim">no tickets</span>'
    extra = len([k for k in keys if str(k).strip()]) - len(shown)
    links = [
        f'<a href="{_html.escape(_jira_ticket_url(key))}">{_html.escape(key)}</a>' for key in shown
    ]
    out = ", ".join(links)
    if extra > 0:
        out += f" (+{extra} more)"
    return out


def _linked_pr_urls(urls: list[str], limit: int = 6) -> str:
    """Every ``urls`` entry as ``#<number>`` linking to the real PR, comma-joined.

    Mirrors ``pages/code.py``'s own ``_pr_evidence_html`` convention (last
    path segment as the visible label) rather than inventing a second style
    for the same kind of link on a different page.
    """
    shown = [str(u).strip() for u in urls if str(u).strip()][:limit]
    if not shown:
        return '<span class="dim">no PRs</span>'
    extra = len([u for u in urls if str(u).strip()]) - len(shown)
    links = []
    for url in shown:
        label = url.rstrip("/").rsplit("/", 1)[-1]
        links.append(f'<a href="{_html.escape(url)}">#{_html.escape(label)}</a>')
    out = ", ".join(links)
    if extra > 0:
        out += f" (+{extra} more)"
    return out


def _empty_note(text: str) -> str:
    return f'<p class="chart-sub">{_html.escape(text)}</p>'


# ---------------------------------------------------------------------------
# Card 1 — Freshness that isn't: board grooming, baselined within role
# ---------------------------------------------------------------------------


def _baseline_within_role(touches: pd.DataFrame, roster) -> pd.DataFrame:
    """Attach each person's role and a role-relative cosmetic-touch baseline.

    Grooming is a PM's job and an engineer's tell — the same raw count means
    two different things depending who produced it, and a threshold that
    doesn't know the difference produces exactly the wrong flag (a PM who
    grooms the backlog all day outranking an engineer with a handful of
    field-only edits). The baseline is the *median* cosmetic-touch count
    among everyone else sharing that role in this same window's data; a role
    with fewer than :data:`_MIN_ROLE_PEERS_FOR_BASELINE` people represented
    falls back to the org-wide median rather than comparing a solo role only
    to itself, which would always read as zero excess.
    """
    out = touches.copy()
    out["role"] = out["person"].map(lambda p: roster.role_of(p))
    role_counts = out.groupby("role")["person"].transform("count")
    role_median = out.groupby("role")["cosmetic_touches"].transform("median")
    org_median = float(out["cosmetic_touches"].median()) if len(out) else 0.0
    out["role_baseline"] = role_median.where(role_counts >= _MIN_ROLE_PEERS_FOR_BASELINE, org_median)
    out["excess_vs_role"] = out["cosmetic_touches"] - out["role_baseline"]
    return out


def _freshness_card(integrity_mod, events: pd.DataFrame, roster) -> str:
    touches = integrity_mod.cosmetic_touches(events, window_days=_COSMETIC_WINDOW_DAYS)
    body = ""
    if touches is not None and not touches.empty:
        baselined = _baseline_within_role(touches, roster)
        top = _top_n(baselined, "excess_vs_role")
        rows = []
        for row in top.itertuples():
            keys = [k for k in str(row.keys).split(", ") if k]
            # ``pandas.map`` turns a ``None`` return into ``NaN`` on an object
            # column (verified: ``role`` is a plain string column, so a
            # missing role is not-a-string, and pandas' own coercion rule for
            # that mix is NaN, not None) - ``pd.isna`` catches both, ``or``
            # alone would not, since a NaN float is truthy in Python.
            role_label = "role unknown" if pd.isna(row.role) else row.role
            rows.append(
                theme_html.evrow(
                    f"<b>{_html.escape(row.person)}</b> ({_html.escape(role_label)}) — "
                    f"{int(row.cosmetic_touches)} field-only edit(s) vs "
                    f"{int(row.status_transitions)} status move(s) in "
                    f"{_COSMETIC_WINDOW_DAYS:.0f}d, role baseline "
                    f"{row.role_baseline:.1f}: {_linked_jira_keys(keys)}"
                )
            )
        body = "".join(rows)
    if not body:
        body = _empty_note(
            f"No field-only edits with no accompanying status move in the last "
            f"{_COSMETIC_WINDOW_DAYS:.0f} days."
        )
    return _wrap_card(
        "Freshness that isn't",
        "Board grooming vs status moves — top 3 above their role's own baseline",
        body,
        "A high count next to a low status-move count is a question, not a "
        "verdict: a lead genuinely does groom the backlog, and rewriting a "
        "vague ticket into something Devin can act on lands here too. That is "
        "exactly why this is baselined within role rather than against one "
        "fixed number for everybody.",
    )


# ---------------------------------------------------------------------------
# Card 2 — Estimates revised mid-flight
# ---------------------------------------------------------------------------


def _estimate_card(integrity_mod, events: pd.DataFrame) -> str:
    churn = integrity_mod.estimate_churn(events, window_days=_CHURN_WINDOW_DAYS)
    body = ""
    if churn is not None and not churn.empty:
        raised = churn[churn["direction"] == "raised"]
        if not raised.empty:
            per_person = (
                raised.groupby("author")
                .agg(
                    hours_added=("delta_hours", "sum"),
                    raises=("key", "count"),
                    keys=("key", lambda s: list(dict.fromkeys(s))),
                )
                .reset_index()
            )
            top = _top_n(per_person, "hours_added")
            rows = [
                theme_html.evrow(
                    f"<b>{_html.escape(row.author)}</b> — +{row.hours_added:.1f}h over "
                    f"{int(row.raises)} mid-flight raise(s) in {_CHURN_WINDOW_DAYS:.0f}d: "
                    f"{_linked_jira_keys(row.keys)}"
                )
                for row in top.itertuples()
            ]
            body = "".join(rows)
    if not body:
        body = _empty_note(
            f"No estimate was raised after work had already started in the last "
            f"{_CHURN_WINDOW_DAYS:.0f} days."
        )
    return _wrap_card(
        "Estimates revised mid-flight",
        "Top 3 by hours added after the ticket was already In Progress or later",
        body,
        "On an hourly contract this is the most direct padding signal the "
        "board can produce — and it is also what an honestly-corrected "
        "estimate looks like. It cannot tell the two apart; ask what changed "
        "about the scope before treating a raise as either one.",
    )


# ---------------------------------------------------------------------------
# Card 3 — Staging round-trips
# ---------------------------------------------------------------------------


def _staging_card(integrity_mod, events: pd.DataFrame, tickets: pd.DataFrame) -> str:
    bounces = integrity_mod.reresolve_events(
        events, tickets, window_days=_RERESOLVE_WINDOW_DAYS
    )
    body = ""
    if bounces is not None and not bounces.empty:
        repeated = bounces[bounces["resolutions"] > 1]
        if not repeated.empty:
            top = _top_n(repeated, "resolutions")
            rows = []
            for row in top.itertuples():
                hidden_note = (
                    " — currently resolved, invisible to the reopened-count metric"
                    if row.hidden_rework
                    else ""
                )
                rows.append(
                    theme_html.evrow(
                        f"{_linked_jira_keys([row.key])} — resolved {int(row.resolutions)}x, "
                        f"{int(row.reopens)} reopen(s), by "
                        f"{_html.escape(str(row.resolvers))}{hidden_note}"
                    )
                )
            body = "".join(rows)
    if not body:
        body = _empty_note(
            f"No ticket entered a resolved status more than once in the last "
            f"{_RERESOLVE_WINDOW_DAYS:.0f} days."
        )
    return _wrap_card(
        "Staging round-trips",
        "Top 3 tickets by number of times declared done",
        body,
        "A bounce is not proof anybody hid anything: a hard bug can fail "
        "staging twice honestly, and the reopener is often a different "
        "person than the resolver, doing their job. What it reliably does is "
        "mint resolution credit that the reopened-count metric never takes "
        "back — that's worth a look regardless of why it happened.",
    )


# ---------------------------------------------------------------------------
# Card 4 — Review pairs & self-merges
# ---------------------------------------------------------------------------


def _review_card(pr_quality_mod, merged_prs: pd.DataFrame, open_prs: pd.DataFrame) -> str:
    rows: list[str] = []

    self_roll = pr_quality_mod.self_merge(merged_prs) if merged_prs is not None else None
    if self_roll is not None and not self_roll.empty:
        candidates = self_roll[self_roll["merged_without_outside_approval"] > 0]
        if not candidates.empty:
            top = _top_n(candidates, "merged_without_outside_approval")
            detail = pr_quality_mod.flag_self_merges(merged_prs)
            for row in top.itertuples():
                unapproved = pd.DataFrame()
                if not detail.empty:
                    mine = detail[detail["author"] == row.author]
                    # ``is False`` rather than ``== False``: ``outside_approval``
                    # can hold ``pd.NA`` on the lean PR payload (pr_quality's own
                    # convention — see self_merge()'s _true() helper), and a
                    # boolean mask built from ``==`` against NA raises instead of
                    # excluding those rows.
                    unapproved = mine[mine["outside_approval"].apply(lambda v: v is False)]
                urls = unapproved["url"].tolist() if not unapproved.empty else []
                rows.append(
                    theme_html.evrow(
                        f"<b>{_html.escape(row.author)}</b> — {int(row.merged_without_outside_approval)} "
                        f"of {int(row.merged_prs)} merged PR(s) with no outside approval: "
                        f"{_linked_pr_urls(urls)}"
                    )
                )

    pools = [frame for frame in (merged_prs, open_prs) if frame is not None and not frame.empty]
    pool = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()
    if not pool.empty:
        _, by_person = pr_quality_mod.reciprocity(pool)
        if by_person is not None and not by_person.empty:
            loops = by_person[
                by_person["top_partner"].notna() & (by_person["reviews_given"] >= 3)
            ]
            if not loops.empty:
                top_loops = _top_n(loops, "concentration")
                for row in top_loops.itertuples():
                    share = row.top_partner_share if pd.notna(row.top_partner_share) else 0.0
                    rows.append(
                        theme_html.evrow(
                            f"<b>{_html.escape(row.reviewer)}</b> reviews almost nobody but "
                            f"<b>{_html.escape(str(row.top_partner))}</b> "
                            f"({share:.0%} of {int(row.reviews_given)} review(s) given, "
                            f"concentration {row.concentration:.2f})"
                        )
                    )

    body = "".join(rows) or _empty_note(
        "No unapproved self-merge and no concentrated review pair in scope."
    )
    return _wrap_card(
        "Review pairs & self-merges",
        "Top merges with no outside approval, plus the most concentrated reviewers",
        body,
        "Pressing merge on your own PR after a colleague approved it is "
        "normal; a small team also legitimately reviews the same few people "
        "over and over because they own the same subsystem. What is worth "
        "reading is whether the linked PRs and approvals have anything real "
        "in them — pull them up before concluding either way.",
    )


# ---------------------------------------------------------------------------
# The roll-up: integrity.integrity_flags, as designed
# ---------------------------------------------------------------------------


def _rollup_card(integrity_mod, tickets: pd.DataFrame, events: pd.DataFrame) -> str:
    flags = integrity_mod.integrity_flags(tickets, events, window_days=_FLAGS_WINDOW_DAYS)
    tripped = flags[flags["flag_count"] > 0] if flags is not None and not flags.empty else flags

    if tripped is not None and not tripped.empty:
        columns = [
            theme_html.Column("Person", "text"),
            theme_html.Column("Flags tripped", "text"),
            theme_html.Column("Count", "num"),
        ]
        table_rows = [
            [
                theme_html.Cell(r.person),
                theme_html.Cell(r.flags),
                theme_html.Cell(r.flag_count),
            ]
            for r in tripped.itertuples()
        ]
        # No tab=/section=: this page never offers a printable report - see
        # _render_integrity_page's own comment on why.
        body = theme_html.table(columns, table_rows)
    else:
        body = _empty_note(
            f"Nobody tripped a named flag in the last {_FLAGS_WINDOW_DAYS:.0f} days."
        )
    return _wrap_card(
        "Who tripped a flag, last 30 days",
        "integrity.integrity_flags — the four checks below, rolled up per person",
        body,
        "None of these four flags prove padding. Each has an innocent "
        "reading — see the card it comes from below for the evidence. Equally, "
        "appearing on no row here is not a clean bill of health: someone who "
        "does nothing at all trips nothing at all.",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _render_integrity_page() -> None:
    """CEO-only. The gate is the first line; everything integrity-shaped sits after it."""
    if not access_gate.require_admin_password():
        return

    # Imports and calls both sit behind the gate above — a non-admin session
    # never executes this line, so integrity.py and pr_quality.py are never
    # imported, let alone called, for that session.
    import integrity
    import pr_quality
    import roles
    from data_layer import _engineering_context

    theme_html.css()
    # ``_engineering_context()`` still hands back a reserved report-button
    # slot, same as every other Engineering page - deliberately unused here.
    # This page's own point is that a flag stays inside the gated screen,
    # next to the evidence and the innocent reading that make it checkable,
    # not printed to a standalone file someone can forward without either.
    # No call below passes ``tab=``/``section=`` to ``theme_html.table()``
    # for the same reason (see ``_rollup_card``'s own ``theme_html.table()``
    # call) - so this page never calls ``_download_report`` at all, and its
    # figures never reach ``st.session_state["tab_reports"]``. See
    # ``docs/assumptions/5B.md`` (the decision point raised) and
    # ``docs/assumptions/5C.md`` (the decision made) - do not "fix" this by
    # wiring the button back in.
    bundle, view, _slot = _engineering_context()
    tickets = view.filtered if view.filtered is not None and not view.filtered.empty else bundle.df
    events = bundle.events
    roster = roles.load_roster()

    theme_html.render(
        theme_html.page_header(
            "VinoVoss · Integrity",
            "Visible to you only",
            {"Jira": True, "GitHub": bool(bundle.github_ready)},
        ),
        theme_html.intro_band(
            "This page is never a verdict. Every flag below carries its own "
            "innocent reading, stated on the card — a lead grooming the "
            "backlog, an estimate honestly corrected, a hard bug that failed "
            "staging twice, two people who genuinely own the same code. An "
            "empty card is not a clean bill of health: someone who touches "
            "nothing at all trips nothing at all. Every number links to the "
            "ticket or PR behind it — read them before concluding anything, "
            "not instead of it."
        ),
        _rollup_card(integrity, tickets, events),
    )

    theme_html.render(
        f'<div class="intgrid">'
        f"{_freshness_card(integrity, events, roster)}"
        f"{_estimate_card(integrity, events)}"
        f"{_staging_card(integrity, events, tickets)}"
        f"{_review_card(pr_quality, bundle.merged_prs, bundle.open_prs)}"
        f"</div>"
    )

    theme_html.render(
        theme_html.foot(
            "Four cards, no score. Thresholds are magnitude, not a fixed cutoff: "
            "each card shows the top three candidates by size, always with the "
            "tickets or PRs behind them, never a pass/fail line a contractor "
            "could learn to stay just under."
        )
    )
