"""One row per person: every scored and machine-recorded fact this dashboard
can currently join together, in one place.

The People page has a teams summary and an assignee workload rollup and
nothing that answers "how is this one person doing" without opening five
tabs. Every number below already exists somewhere in this codebase -
``kpi.py`` scores code-producing engineers, ``estimate_accuracy.py`` has
never been imported by any page, ``pr_quality.py`` and ``integrity.py`` each
compute one piece of the picture, and ``roles.py`` says who is even supposed
to be measured on what, and how. :func:`people_table` is the join, not new
analysis: one call per source, one lookup built from each, one row per
person.

``KPI_SPEC.md`` §1's rules apply directly to every column here:

- Machine-recorded facts outrank declared ones. Resolved-ticket counts come
  from :func:`integrity.credited_resolutions` (the changelog author of the
  actual resolving transition), not a ticket's current-assignee field - the
  same fix that used to credit a departed QA tester with 194 resolutions.
- Every metric carries its own ``n`` (see the ``n_*`` columns below and
  ``score``'s own ``n``). A median or a ratio with one observation behind it
  and one with thirty are not the same claim.
- Below the measurable-weight floor, ``score`` is ``None`` and
  ``no_score_reason`` says why - never ``0``. A "0/100" published from an
  empty scorecard reads as the worst person on the team; the honest read is
  "not enough data to say."
- Every value this module cannot compute is ``None``/``NA``, never a
  substituted zero, *except* where the source module itself has already
  argued a zero is a true, informative fact (``flag_count``,
  ``reviews_given``, ``delivered_points``: see each column's note below).

Identity is the two-system problem this module exists to solve before
anything else can be joined. Jira tickets and the changelog carry a
person's Jira display name; GitHub PRs and reviews carry a GitHub login.
``roles.Roster`` is the only thing that knows both, so every raw key read
here - ticket assignee, changelog author, PR author, PR reviewer - goes
through :func:`_canonical_person` before it is grouped or looked up. A PR
author with no roster mapping keeps their raw login as their identity
rather than being merged into "Unknown" or silently dropped: that person is
real, they are just not on the roster yet, and the row says so through a
``None`` role and a ``"role unknown"`` ``no_score_reason``.

What this module does not do, on purpose: it does not fetch anything, does
not call Jira or GitHub, and does not compute the ``owned``/``gradable``
ticket frames ``kpi.components`` needs - those already carry ``idle_days``,
``carry_over_count``, ``has_estimate`` and ``policy_applies`` by the time
they reach here (``hygiene.estimate_policy`` / ``add_ticket_health_fields``,
built and wired by other tasks). Handing this module raw, unprocessed
ticket frames will not crash it - ``kpi.components`` degrades those
components to "insufficient data" the same way it always has - but it will
under-report what could have been measured.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

import estimate_accuracy
import integrity
import kpi
import pr_quality
import role_kpis
import roles

# Per-flag severity, used only for ``flag_severity`` - a single number that
# separates "tripped one weak flag" from "tripped the two flags that mint
# fake delivery/resolution credit." Weights are judgment calls, not derived:
# ``staging_pingpong`` and ``rework_hidden`` both manufacture resolution
# credit that nothing else on the scorecard can see, so they weigh the same
# and the most; ``estimate_inflation`` moves billed hours directly, so it
# outweighs ``board_grooming``, which has the most innocent readings of the
# four (a lead actually grooming the backlog looks identical to this flag).
FLAG_SEVERITY_WEIGHTS: dict[str, float] = {
    "board_grooming": 1.0,
    "estimate_inflation": 2.0,
    "staging_pingpong": 3.0,
    "rework_hidden": 3.0,
}

_UNRESOLVABLE = frozenset({"", "unassigned", "unknown", "none", "nan"})

COLUMNS: list[str] = [
    "person",
    "role",
    "score",
    "n",
    "delivered_points",
    "n_delivered_points",
    "trivial_share",
    "n_trivial_share",
    "cycle_median",
    "n_cycle_median",
    "reviews_given",
    "n_reviews_given",
    "ttfr_hours",
    "n_ttfr_hours",
    "estimate_ratio",
    "n_estimate_ratio",
    "estimate_iqr",
    "n_estimate_iqr",
    "flag_count",
    "n_flag_count",
    "flag_severity",
    "n_flag_severity",
    "measurable_pct",
    "no_score_reason",
]

# dtype per column, for :func:`_empty_table` and for casting the assembled
# rows. Nullable pandas extension dtypes throughout so a missing value is a
# real ``pd.NA``, not a float ``NaN`` masquerading as int, and not a silent
# ``0``.
_DTYPES: dict[str, str] = {
    "person": "object",
    "role": "object",
    "score": "Float64",
    "n": "Int64",
    "delivered_points": "Float64",
    "n_delivered_points": "Int64",
    "trivial_share": "Float64",
    "n_trivial_share": "Int64",
    "cycle_median": "Float64",
    "n_cycle_median": "Int64",
    "reviews_given": "Int64",
    "n_reviews_given": "Int64",
    "ttfr_hours": "Float64",
    "n_ttfr_hours": "Int64",
    "estimate_ratio": "Float64",
    "n_estimate_ratio": "Int64",
    "estimate_iqr": "Float64",
    "n_estimate_iqr": "Int64",
    "flag_count": "Int64",
    "n_flag_count": "Int64",
    "flag_severity": "Float64",
    "n_flag_severity": "Int64",
    "measurable_pct": "Float64",
    "no_score_reason": "object",
}


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=_DTYPES[col]) for col in COLUMNS})


def _clean(value: object) -> str:
    return str(value or "").strip()


def _canonical_person(raw: object, roster: roles.Roster) -> str | None:
    """The one identity a raw Jira name or GitHub login resolves to.

    Tries a Jira-name match first, then a GitHub-login match; a string that
    is neither (not on the roster, not a mapped login) comes back unchanged
    as its own identity, so an untracked contributor still gets a row
    instead of being folded into "Unknown". Returns ``None`` for a blank,
    "Unassigned" or "Unknown" value - there is no person there to report.
    """
    text = _clean(raw)
    if not text or text.lower() in _UNRESOLVABLE:
        return None
    who = roster.person(text)
    if who is not None:
        return who.name
    mapped = roster.name_for_login(text)
    if mapped is not None:
        return mapped
    return text


def _person_set(frame: pd.DataFrame | None, column: str, roster: roles.Roster) -> set[str]:
    if frame is None or frame.empty or column not in frame.columns:
        return set()
    out: set[str] = set()
    for raw in frame[column]:
        person = _canonical_person(raw, roster)
        if person is not None:
            out.add(person)
    return out


def _credited_counts(
    events: pd.DataFrame,
    tickets: pd.DataFrame | None,
    window_days: float,
    resolved_statuses: Iterable[str] | None,
    roster: roles.Roster,
    now: object | None,
) -> dict[str, int]:
    """Machine-recorded resolved-in-window count, per person.

    ``integrity.credited_resolutions`` already excludes former-staff and
    unattributed authorship; this only remaps its ``person`` key (a Jira
    display name) through the roster for consistent casing and accumulates
    two raw keys that canonicalize to the same person (rare, but a changelog
    can carry more than one spelling of the same name).
    """
    result = integrity.credited_resolutions(
        events,
        tickets=tickets,
        window_days=window_days,
        now=now,
        resolved_statuses=resolved_statuses,
    )
    out: dict[str, int] = {}
    if result.by_person is None or result.by_person.empty:
        return out
    for _, row in result.by_person.iterrows():
        person = _canonical_person(row["person"], roster)
        if person is None:
            continue
        out[person] = out.get(person, 0) + int(row["credited_resolutions"])
    return out


def _flags_by_person(
    tickets: pd.DataFrame,
    events: pd.DataFrame,
    window_days: float,
    resolved_statuses: Iterable[str] | None,
    roster: roles.Roster,
    now: object | None,
) -> dict[str, dict[str, object]]:
    """``flag_count``, ``flag_severity``, and their shared ``n``, per person.

    ``n`` is the total count of signal events (``cosmetic_touches``,
    ``status_transitions``, ``backward_moves``, ``reresolved_tickets``) the
    four flags were computed from - not the flag count itself, since two
    people can both trip 1 of 4 flags on wildly different amounts of board
    activity. ``reresolved_tickets`` is kept separately too: it doubles as
    the machine-recorded ``reopened_90`` this module feeds into
    :func:`kpi.components`'s Rework component - see
    ``docs/assumptions/2B.md``.
    """
    frame = integrity.integrity_flags(
        tickets, events, window_days=window_days, now=now, resolved_statuses=resolved_statuses
    )
    out: dict[str, dict[str, object]] = {}
    if frame is None or frame.empty:
        return out
    for _, row in frame.iterrows():
        person = _canonical_person(row["person"], roster)
        if person is None:
            continue
        severity = sum(
            FLAG_SEVERITY_WEIGHTS[flag]
            for flag in integrity.FLAG_NAMES
            if bool(row.get(flag, False))
        )
        signal_n = int(
            row.get("cosmetic_touches", 0) or 0
        ) + int(row.get("status_transitions", 0) or 0) + int(
            row.get("backward_moves", 0) or 0
        ) + int(row.get("reresolved_tickets", 0) or 0)
        reresolved = int(row.get("reresolved_tickets", 0) or 0)
        existing = out.get(person)
        if existing is None:
            out[person] = {
                "flag_count": int(row["flag_count"]),
                "flag_severity": float(severity),
                "n": signal_n,
                "reresolved_tickets": reresolved,
            }
        else:
            existing["flag_count"] += int(row["flag_count"])
            existing["flag_severity"] += float(severity)
            existing["n"] += signal_n
            existing["reresolved_tickets"] += reresolved
    return out


def _size_lookup(prs: pd.DataFrame, roster: roles.Roster) -> dict[str, dict[str, object]]:
    """Delivered points, trivial share and their sample sizes, per person.

    Built on :func:`pr_quality.delivered_points` (points, ``prs``,
    ``trivial_share``) and :func:`pr_quality.size_bands` (``unsized``, for
    the trivial-share denominator) so the two agree on how a PR was
    classified - see ``pr_quality.SIZE_POINTS`` for the band weights and
    their own documented blind spot (diff size measures typing, not
    difficulty).
    """
    out: dict[str, dict[str, object]] = {}
    if prs is None or prs.empty:
        return out
    bands = pr_quality.size_bands(prs)
    unsized_by_author: dict[str, int] = {}
    if not bands.empty:
        for _, row in bands.iterrows():
            person = _canonical_person(row["author"], roster)
            if person is None:
                continue
            unsized_by_author[person] = unsized_by_author.get(person, 0) + int(row["unsized"])
    for record in pr_quality.delivered_points(prs):
        person = _canonical_person(record.author, roster)
        if person is None:
            continue
        unsized = unsized_by_author.get(person, 0)
        sized = max(record.prs - unsized, 0)
        entry = out.setdefault(
            person,
            {
                "delivered_points": 0.0,
                "n_delivered_points": 0,
                "trivial_share": None,
                "n_trivial_share": 0,
            },
        )
        entry["delivered_points"] += record.points
        entry["n_delivered_points"] += record.prs
        entry["n_trivial_share"] += sized
        # ``trivial_share`` is already sized/trivial from ``size_bands``; a
        # second raw key for the same canonical person (rare) would need a
        # PR-weighted average to stay exact, which is more machinery than
        # the collision is worth - last value wins, and it is almost always
        # the only value.
        entry["trivial_share"] = (
            record.trivial_share if record.trivial_share == record.trivial_share else None
        )  # NaN != NaN
    return out


def _review_lookup(prs: pd.DataFrame, roster: roles.Roster) -> dict[str, dict[str, object]]:
    """Reviews given and time-to-first-review, per person.

    ``n_ttfr_hours`` is approximated as ``prs_reviewed`` (every PR this
    person reviewed at all), not the narrower count of PRs where they were
    specifically first - :func:`pr_quality.review_citizenship` computes that
    narrower count internally but does not expose it, and this module does
    not reach into its private ``_review_events``. Documented rather than
    worked around: treat ``n_ttfr_hours`` as an upper bound on the real
    sample size behind ``ttfr_hours``, per ``docs/assumptions/2B.md``.
    """
    out: dict[str, dict[str, object]] = {}
    if prs is None or prs.empty:
        return out
    citizenship = pr_quality.review_citizenship(prs)
    if citizenship.empty:
        return out
    for _, row in citizenship.iterrows():
        person = _canonical_person(row["reviewer"], roster)
        if person is None:
            continue
        entry = out.setdefault(
            person,
            {"reviews_given": 0, "n_reviews_given": 0, "ttfr_hours": None, "n_ttfr_hours": 0},
        )
        entry["reviews_given"] += int(row["reviews_given"])
        entry["n_reviews_given"] += int(row["prs_reviewed"])
        ttfr = row.get("median_hours_to_first_review")
        if pd.notna(ttfr):
            entry["ttfr_hours"] = float(ttfr)
            entry["n_ttfr_hours"] += int(row["prs_reviewed"])
    return out


def _cycle_lookup(
    events: pd.DataFrame,
    tickets: pd.DataFrame | None,
    roster: roles.Roster,
    now: object | None,
) -> dict[str, dict[str, object]]:
    """Median lead time (first start to first resolve) and its ticket count.

    Uses :func:`integrity.cycle_time`'s ``median_lead_time_days`` - the one
    metric in this table its own module docstring calls "the metric that
    cannot be gamed by editing fields." Its blind spot travels with it
    unchanged: waiting off-board (blocked on someone else's review) still
    reads as active cycle time.
    """
    out: dict[str, dict[str, object]] = {}
    if events is None or events.empty:
        return out
    by_person = integrity.cycle_time(events, tickets=tickets, now=now).by_person
    if by_person is None or by_person.empty:
        return out
    for _, row in by_person.iterrows():
        person = _canonical_person(row["person"], roster)
        if person is None:
            continue
        median = row.get("median_lead_time_days")
        n = int(row.get("lead_time_tickets", 0) or 0)
        entry = out.setdefault(person, {"cycle_median": None, "n_cycle_median": 0})
        if pd.notna(median):
            entry["cycle_median"] = float(median)
        entry["n_cycle_median"] += n
    return out


def _estimate_lookup(tickets: pd.DataFrame, roster: roles.Roster) -> dict[str, dict[str, object]]:
    """Median estimate-accuracy ratio and IQR, per person - see
    :func:`estimate_accuracy.accuracy_by_person`. This is the module the
    rest of the dashboard never imported; wiring it in is half the point of
    this file. Its own stated limit applies unchanged: both the estimate
    and the logged time are self-reported, so this detects *inconsistency*
    between two declared numbers, never ground-truth padding.
    """
    out: dict[str, dict[str, object]] = {}
    if tickets is None or tickets.empty:
        return out
    by_person = estimate_accuracy.accuracy_by_person(tickets)
    if by_person.empty:
        return out
    for _, row in by_person.iterrows():
        person = _canonical_person(row["assignee"], roster)
        if person is None:
            continue
        n = int(row.get("tickets", 0) or 0)
        entry = out.setdefault(
            person,
            {"estimate_ratio": None, "n_estimate_ratio": 0, "estimate_iqr": None, "n_estimate_iqr": 0},
        )
        entry["n_estimate_ratio"] += n
        entry["n_estimate_iqr"] += n
        if pd.notna(row.get("median_ratio")):
            entry["estimate_ratio"] = float(row["median_ratio"])
        if pd.notna(row.get("iqr")):
            entry["estimate_iqr"] = float(row["iqr"])
    return out


def _score_for(
    person: str,
    role_lookup: roles.RubricLookup,
    roster: roles.Roster,
    owned: pd.DataFrame,
    gradable: pd.DataFrame,
    prs_mine: pd.DataFrame,
    resolved_7: int | None,
    resolved_90: int | None,
    reopened_90: int | None,
    peer_resolved_7: Mapping[str, int] | None,
    role_kpi_inputs: "role_kpis.RoleKpiInputs | None" = None,
) -> tuple[float | None, int, float, str]:
    """``(score, n, measurable_pct, no_score_reason)`` for one person.

    Code-producing roles (``roles.CODE_ROLES``) are scored the way the
    single-person scorecard already does: :func:`kpi.components` ->
    :func:`kpi.overall` / :func:`kpi.coverage`. Scored non-code roles (QA,
    PM, designer, infrastructure) go through
    :func:`role_kpis.components_for` -> :func:`role_kpis.score_from_parts`
    against their own rubric - the wiring ``docs/assumptions/2C.md`` used to
    name as future work. Every other classification - no rubric at all
    (seo/wine/advisor), exec, or not on the roster - goes through
    :func:`roles.overall` with an empty component-score map, which is
    exactly what that function is for: it returns ``None`` and the correct
    named reason (``"exec — not scored"``, the no-rubric sentinel) without
    this module having to special-case any of those cases itself.
    """
    if role_lookup.status == "scored" and role_lookup.role in roles.CODE_ROLES:
        parts = kpi.components(
            owned,
            gradable,
            resolved_7,
            resolved_90,
            reopened_90,
            prs_mine,
            peer_resolved_7=peer_resolved_7,
        )
        score = kpi.overall(parts)
        cov = kpi.coverage(parts)
        measurable_pct = (
            100.0 * cov.covered_weight / cov.total_weight if cov.total_weight else 0.0
        )
        n = sum(int(p.n or 0) for p in parts if p.sufficient)
        reason = "" if score is not None else cov.note
        return score, n, measurable_pct, reason

    if (
        role_lookup.status == "scored"
        and role_lookup.rubric is not None
        and role_kpi_inputs is not None
    ):
        parts = role_kpis.components_for(
            role_kpi_inputs, person, role_lookup.role, owned, gradable
        )
        if parts is not None:
            score, covered, note = role_kpis.score_from_parts(parts, role_lookup.rubric)
            measurable_pct = (
                100.0 * covered / role_lookup.rubric.total if role_lookup.rubric.total else 0.0
            )
            n = sum(int(p.n or 0) for p in parts if p.sufficient)
            reason = "" if score is not None else note
            return score, n, measurable_pct, reason

    result = roles.overall(person, roster, component_scores={})
    return None, 0, 0.0, result.reason


def people_table(
    open_tickets: pd.DataFrame,
    resolved_tickets: pd.DataFrame,
    gradable_tickets: pd.DataFrame,
    prs: pd.DataFrame,
    events: pd.DataFrame,
    *,
    roster: roles.Roster | None = None,
    resolved_statuses: Iterable[str] | None = None,
    now: object | None = None,
    role_kpi_inputs: "role_kpis.RoleKpiInputs | None" = None,
) -> pd.DataFrame:
    """One row per person: the People page's missing table.

    Parameters are the bundles the rest of the dashboard already fetches,
    handed over unmodified - this function does not fetch, filter to a
    window, or add health fields itself:

    - ``open_tickets``: every open, non-backlog ticket, across everyone,
      with ``idle_days``/``carry_over_count``/``has_estimate`` already added
      (``hygiene``'s job, not this module's) - ``kpi.components``'s
      ``owned`` input, sliced per person here.
    - ``resolved_tickets``: tickets resolved recently, across everyone, for
      the estimate-accuracy join and (concatenated with ``open_tickets``)
      for cycle time and the integrity flags.
    - ``gradable_tickets``: quality-scored tickets (owned or reported),
      across everyone, already scored by ``ticket_quality`` - this module's
      "gradable" input, sliced per person.
    - ``prs``: open and recently-merged PRs, across everyone, ideally the
      extended-fetch shape (``timeline_events`` present) so
      ``pr_quality.review_citizenship``'s time-to-first-review has data to
      work with; degrades gracefully (fewer populated columns) on the lean
      shape.
    - ``events``: ``integrity.changelog_events(...)`` over the same tickets,
      across everyone.

    ``resolved_statuses`` and ``now`` pass straight through to every
    ``integrity``/``estimate_accuracy`` call that accepts them (``now``
    defaults to the real current time; tests pin it for a reproducible
    "days ago" fixture, same as ``tests/test_integrity.py`` does).

    Returns :data:`COLUMNS` always, in that order, even for an all-empty
    input - there is no code path that returns a bare, columnless
    ``DataFrame``.

    What the whole table cannot see, stated once here rather than per
    column: every PR-side number is only as good as the roster's GitHub
    login map (``roles.load_roster``'s ``GITHUB_LOGIN_MAP``) - an author
    whose login is missing from that map still gets a row, keyed by their
    raw login instead of a Jira name, with ``role=None`` and
    ``no_score_reason="role unknown"``.
    """
    ros = roster if roster is not None else roles.load_roster()

    people = (
        _person_set(open_tickets, "assignee", ros)
        | _person_set(resolved_tickets, "assignee", ros)
        | _person_set(gradable_tickets, "assignee", ros)
        | _person_set(events, "author", ros)
        | _person_set(prs, "author", ros)
    )
    # Former staff never appear as rows, regardless of which source surfaced
    # their name - a departed tester's account can still author changelog
    # entries or sit as a ticket's stale assignee, and this is where that
    # stops mattering.
    people = {p for p in people if ros.is_active(p) is not False}
    if not people:
        return _empty_table()

    all_tickets = pd.concat(
        [f for f in (open_tickets, resolved_tickets) if f is not None and not f.empty],
        ignore_index=True,
        sort=False,
    ) if (open_tickets is not None and not open_tickets.empty) or (
        resolved_tickets is not None and not resolved_tickets.empty
    ) else pd.DataFrame()

    resolved_7 = _credited_counts(events, resolved_tickets, 7.0, resolved_statuses, ros, now)
    resolved_90 = _credited_counts(events, resolved_tickets, 90.0, resolved_statuses, ros, now)
    events_present = events is not None and not events.empty
    flags = _flags_by_person(all_tickets, events, 90.0, resolved_statuses, ros, now)
    sizes = _size_lookup(prs, ros)
    reviews = _review_lookup(prs, ros)
    cycles = _cycle_lookup(events, all_tickets if not all_tickets.empty else None, ros, now)
    estimates = _estimate_lookup(all_tickets, ros)

    # The shared org-wide frames the non-code rubrics (QA, PM, designer,
    # infrastructure) score against - built once here, not once per person,
    # and only when somebody on the roster actually needs them.
    role_inputs = role_kpi_inputs
    if role_inputs is None:
        needs_role_kpis = any(
            (lk := roles.rubric_for_person(ros, p)).status == "scored"
            and lk.role not in roles.CODE_ROLES
            for p in people
        )
        role_inputs = (
            role_kpis.build_inputs(
                open_tickets,
                all_tickets,
                events,
                prs,
                resolved_tickets,
                resolved_statuses=resolved_statuses,
                now=now,
            )
            if needs_role_kpis
            else None
        )

    # Peer pool for the code rubric's "Delivery vs team" component: every
    # active member of the same role cohort (``roles.peer_cohort``), scored
    # against the same machine-recorded ``resolved_7`` this table already
    # built - a teammate absent from ``resolved_7`` resolved zero in the
    # window (a real, known fact once ``events`` is non-empty), not an
    # unknown one, so the lookup defaults to 0 rather than dropping them.
    def _peer_pool(person: str) -> Mapping[str, int] | None:
        cohort = roles.peer_cohort(ros, person)
        if not cohort.sufficient:
            return None
        pool = {person: resolved_7.get(person, 0) if events_present else None}
        for peer in cohort.peers:
            pool[peer] = resolved_7.get(peer, 0) if events_present else None
        return pool

    prs_person = pd.Series(dtype="object")
    if prs is not None and not prs.empty and "author" in prs.columns:
        prs_person = prs["author"].map(lambda v: _canonical_person(v, ros))

    open_person = pd.Series(dtype="object")
    if open_tickets is not None and not open_tickets.empty and "assignee" in open_tickets.columns:
        open_person = open_tickets["assignee"].map(lambda v: _canonical_person(v, ros))

    gradable_person = pd.Series(dtype="object")
    if (
        gradable_tickets is not None
        and not gradable_tickets.empty
        and "assignee" in gradable_tickets.columns
    ):
        gradable_person = gradable_tickets["assignee"].map(lambda v: _canonical_person(v, ros))

    rows: list[dict[str, object]] = []
    for person in sorted(people):
        role_lookup = roles.rubric_for_person(ros, person)

        owned = open_tickets[open_person == person] if not open_person.empty else pd.DataFrame()
        gradable = (
            gradable_tickets[gradable_person == person]
            if not gradable_person.empty
            else pd.DataFrame()
        )
        prs_mine = prs[prs_person == person] if not prs_person.empty else pd.DataFrame()

        r7 = resolved_7.get(person, 0) if events_present else None
        r90 = resolved_90.get(person, 0) if events_present else None
        flag_row = flags.get(person)
        reopened_90 = (
            (0 if flag_row is None else int(flag_row["reresolved_tickets"]))
            if events_present
            else None
        )

        score, n, measurable_pct, reason = _score_for(
            person,
            role_lookup,
            ros,
            owned,
            gradable,
            prs_mine,
            r7,
            r90,
            reopened_90,
            _peer_pool(person) if role_lookup.role in roles.CODE_ROLES else None,
            role_kpi_inputs=role_inputs,
        )

        size_row = sizes.get(person, {})
        review_row = reviews.get(person, {})
        cycle_row = cycles.get(person, {})
        estimate_row = estimates.get(person, {})

        if flag_row is None:
            # Non-null 0 when the pipeline actually ran and this person just
            # tripped nothing (``integrity_flags``'s own reading of a quiet
            # board); ``NA`` only when there was no changelog to read at all.
            flag_count = 0 if events_present else pd.NA
            flag_severity = 0.0 if events_present else pd.NA
            n_flag = 0
        else:
            flag_count = flag_row["flag_count"]
            flag_severity = flag_row["flag_severity"]
            n_flag = flag_row["n"]

        rows.append(
            {
                "person": person,
                "role": role_lookup.role,
                "score": score,
                "n": n,
                "delivered_points": size_row.get("delivered_points", 0.0 if prs is not None and not prs.empty else pd.NA),
                "n_delivered_points": size_row.get("n_delivered_points", 0),
                "trivial_share": size_row.get("trivial_share"),
                "n_trivial_share": size_row.get("n_trivial_share", 0),
                "cycle_median": cycle_row.get("cycle_median"),
                "n_cycle_median": cycle_row.get("n_cycle_median", 0),
                "reviews_given": review_row.get("reviews_given", 0 if prs is not None and not prs.empty else pd.NA),
                "n_reviews_given": review_row.get("n_reviews_given", 0),
                "ttfr_hours": review_row.get("ttfr_hours"),
                "n_ttfr_hours": review_row.get("n_ttfr_hours", 0),
                "estimate_ratio": estimate_row.get("estimate_ratio"),
                "n_estimate_ratio": estimate_row.get("n_estimate_ratio", 0),
                "estimate_iqr": estimate_row.get("estimate_iqr"),
                "n_estimate_iqr": estimate_row.get("n_estimate_iqr", 0),
                "flag_count": flag_count,
                "n_flag_count": n_flag,
                "flag_severity": flag_severity,
                "n_flag_severity": n_flag,
                "measurable_pct": measurable_pct,
                "no_score_reason": reason,
            }
        )

    out = pd.DataFrame(rows, columns=COLUMNS)
    for col, dtype in _DTYPES.items():
        out[col] = out[col].astype(dtype)
    return out
