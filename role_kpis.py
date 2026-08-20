"""Component computations for the non-code rubrics: QA, PM, designer, infrastructure.

``roles.py`` (WP5) defined five rubrics but only the code rubric ever had a
component-computation behind it (``kpi.components``). ``docs/assumptions/2C.md``
names the gap explicitly: "None of these five rubrics has a
component-computation function behind it yet ... that's future work". This
module is that work, for the four non-code rubrics.

Design rules, inherited from KPI_SPEC.md §1 and enforced the same way
``kpi.components`` enforces them:

1. Machine-recorded facts over declared ones wherever one exists. Defect
   escape and rework come from the changelog (``integrity.reresolve_events``
   + ``integrity.credited_resolutions``), staleness from
   ``integrity.status_age_days`` (the honest clock, not the groomable
   ``idle_days``), cycle time from status transitions.
2. A component that cannot be measured is reported as insufficient
   (``Component.sufficient=False``), never scored zero and never silently
   dropped - :func:`components_for` always returns one row per rubric
   component, exactly like ``kpi.components(include_gaps=True)``.
3. Every component carries its sample size ``n`` and refuses to score below
   :data:`MIN_N` observations.
4. The headline is ``roles.overall``'s job: below 60% of the rubric's weight
   with data, there is no score at all.

The scoring ramps below (fast day / slow day pairs) are working agreements,
not laws of nature - they are module constants precisely so a change to one
is a visible, reviewable edit rather than a buried magic number. Each is a
linear ramp: at or under the fast bound scores 100, at or over the slow
bound scores 0.

What this module reuses rather than reinvents: every underlying frame comes
from ``integrity.py`` / ``estimate_accuracy.py`` / ``kpi.py`` functions that
are already unit-tested; this module only joins them per person and maps
them onto rubric components. The expensive org-wide frames are computed once
in :func:`build_inputs` and shared across every person scored in a render,
so scoring N people costs one pass over the changelog, not N.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

import estimate_accuracy
import integrity
import kpi
import roles

# Minimum observations before a component says anything. Matches the spirit of
# kpi.MIN_PEERS / estimate_accuracy's no-verdict-below-5 rule; three is the
# floor at which a median stops being an anecdote.
MIN_N = 3

# Verification latency ramp (QA): days a ticket sat in its final pre-resolve
# status before this person's resolving transition. Same-day verification
# scores 100; a week in the verification queue scores 0.
VERIFY_FAST_DAYS = 1.0
VERIFY_SLOW_DAYS = 7.0

# Lead-time ramp (designer, infrastructure): median first-start-to-first-
# resolve days from integrity.cycle_time - the one input in this module its
# own source calls ungameable by field edits.
CYCLE_FAST_DAYS = 3.0
CYCLE_SLOW_DAYS = 21.0

# Triage ramp (PM): days from a ticket's creation to the PM's first touch of
# any kind on it. A ticket nobody has triaged yet counts at its current age -
# ignoring a ticket must never score better than triaging it late.
TRIAGE_FAST_DAYS = 1.0
TRIAGE_SLOW_DAYS = 7.0

# Estimate-accuracy scoring: 100 at a median ratio of exactly 1.0, falling
# linearly in log2 space to 0 at half or double (ratio 0.5 or 2.0). Log space
# so that under- and over-running by the same factor cost the same.
ACCURACY_ZERO_AT_LOG2 = 1.0

# The window every changelog-derived component below reads over, matching the
# People page's own 90-day resolved read (pages/people._RESOLVED_WINDOW_DAYS).
WINDOW_DAYS = 90.0

# What each component needs before it can say anything - the plain words its
# "insufficient data" placeholder uses, mirroring kpi.COMPONENT_INPUTS.
COMPONENT_INPUTS: dict[str, str] = {
    "Defect escape rate": f"at least {MIN_N} changelog-credited verifications in {WINDOW_DAYS:.0f}d",
    "Verification cycle time": f"at least {MIN_N} timed verifications in {WINDOW_DAYS:.0f}d",
    "Automation coverage added": "per-PR file lists, which this dashboard does not fetch yet",
    "Rework after verification": f"at least {MIN_N} changelog-credited verifications in {WINDOW_DAYS:.0f}d",
    "Estimate accuracy": f"at least {MIN_N} finished tickets with both an estimate and logged time",
    "Bug validity rate": "bug-resolution outcomes (duplicate/cannot-reproduce vs fixed) on bugs they reported",
    "Triage latency": f"tickets created in the last {WINDOW_DAYS:.0f}d, with their changelogs",
    "Board hygiene": "open non-backlog tickets across the org",
    "Estimate coverage": "open tickets the estimate policy applies to, across the org",
    "Stale queue rate": "open non-backlog tickets with changelog history",
    "Cycle time": f"at least {MIN_N} tickets moved from started to resolved in {WINDOW_DAYS:.0f}d",
    "Staleness": "open non-backlog tickets they own",
    "Handoff completeness": "quality-scored tickets they own or reported",
    "Hours vs delivered output": f"at least {MIN_N} finished tickets with logged hours and a linked PR diff",
    "Estimate churn": f"at least {MIN_N} tickets they moved in {WINDOW_DAYS:.0f}d, plus estimate-edit history",
}


@dataclass
class RoleKpiInputs:
    """Every org-wide frame the four non-code rubrics read, computed once.

    Build with :func:`build_inputs`. Attributes are plain DataFrames (empty,
    never ``None``) so per-person code never branches on presence, only on
    emptiness - the same convention the integrity module's own outputs keep.
    """

    open_tickets: pd.DataFrame = field(default_factory=pd.DataFrame)
    all_tickets: pd.DataFrame = field(default_factory=pd.DataFrame)
    credited_detail: pd.DataFrame = field(default_factory=pd.DataFrame)
    reresolve: pd.DataFrame = field(default_factory=pd.DataFrame)
    cycle_by_person: pd.DataFrame = field(default_factory=pd.DataFrame)
    status_age: pd.DataFrame = field(default_factory=pd.DataFrame)
    accuracy: pd.DataFrame = field(default_factory=pd.DataFrame)
    churn: pd.DataFrame = field(default_factory=pd.DataFrame)
    hours_per_line: pd.DataFrame = field(default_factory=pd.DataFrame)
    status_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    events_present: bool = False
    now: pd.Timestamp | None = None


def _frame(value: object) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def build_inputs(
    open_tickets: pd.DataFrame,
    all_tickets: pd.DataFrame,
    events: pd.DataFrame,
    prs: pd.DataFrame,
    resolved_tickets: pd.DataFrame,
    *,
    resolved_statuses=None,
    now: object | None = None,
) -> RoleKpiInputs:
    """One pass over the org-wide frames, shared by every person scored.

    ``open_tickets``: open non-backlog tickets after ``hygiene.estimate_policy``
    (the PM components read these org-wide, not per person).
    ``all_tickets``: open + recently-resolved concatenated, for estimate
    accuracy and the hours/diff join. ``events``:
    ``integrity.changelog_events`` over the same tickets. ``prs``: open and
    merged PRs together (may be empty - only Hours vs delivered output reads
    it here). ``resolved_tickets``: the resolved window backing credited
    resolutions.
    """
    open_tickets = _frame(open_tickets)
    all_tickets = _frame(all_tickets)
    events = _frame(events)
    prs = _frame(prs)
    resolved_tickets = _frame(resolved_tickets)
    events_present = not events.empty
    moment = integrity._now(now)

    if events_present:
        credited = integrity.credited_resolutions(
            events,
            tickets=resolved_tickets if not resolved_tickets.empty else None,
            window_days=WINDOW_DAYS,
            now=now,
            resolved_statuses=resolved_statuses,
        )
        credited_detail = credited.detail
        reresolve = integrity.reresolve_events(
            events,
            tickets=all_tickets if not all_tickets.empty else None,
            window_days=WINDOW_DAYS,
            now=now,
            resolved_statuses=resolved_statuses,
        )
        cycle_by_person = integrity.cycle_time(
            events,
            tickets=all_tickets if not all_tickets.empty else None,
            now=now,
        ).by_person
        status_age = integrity.status_age_days(open_tickets, events)
        churn = integrity.estimate_churn(events, window_days=WINDOW_DAYS, now=now)
        status_events = events[events["is_status"].fillna(False).astype(bool)].copy()
    else:
        credited_detail = pd.DataFrame()
        reresolve = pd.DataFrame()
        cycle_by_person = pd.DataFrame()
        status_age = pd.DataFrame()
        churn = pd.DataFrame()
        status_events = pd.DataFrame()

    accuracy = (
        estimate_accuracy.accuracy_by_person(all_tickets, min_tickets=MIN_N)
        if not all_tickets.empty
        else pd.DataFrame()
    )
    hours_per_line = (
        estimate_accuracy.hours_per_delivered_line(all_tickets, prs)
        if not all_tickets.empty and not prs.empty
        else pd.DataFrame()
    )

    return RoleKpiInputs(
        open_tickets=open_tickets,
        all_tickets=all_tickets,
        credited_detail=credited_detail,
        reresolve=reresolve,
        cycle_by_person=cycle_by_person,
        status_age=status_age,
        accuracy=accuracy,
        churn=churn,
        hours_per_line=hours_per_line,
        status_events=status_events,
        raw_events=events,
        events_present=events_present,
        now=moment,
    )


# ---------------------------------------------------------------------------
# Small scoring helpers
# ---------------------------------------------------------------------------


def _ramp(value: float, fast: float, slow: float) -> float:
    """100 at or under ``fast``, 0 at or over ``slow``, linear between."""
    if value <= fast:
        return 100.0
    if value >= slow:
        return 0.0
    return 100.0 * (slow - value) / (slow - fast)


def accuracy_score(median_ratio: float) -> float | None:
    """0-100 from a logged/estimated median ratio; ``None`` when unusable.

    100 at exactly 1.0, 0 at half or double, linear in log2 space so a
    consistent 2x under-run (padding's signature) costs exactly as much as a
    consistent 2x over-run (estimating blind).
    """
    if median_ratio is None or not math.isfinite(median_ratio) or median_ratio <= 0:
        return None
    off = abs(math.log2(median_ratio))
    return 100.0 * max(0.0, 1.0 - off / ACCURACY_ZERO_AT_LOG2)


def _gap(name: str) -> kpi.Component:
    return kpi.Component(
        name,
        0.0,
        f"insufficient data - needs {COMPONENT_INPUTS.get(name, 'more data')}",
        n=0,
        sufficient=False,
    )


def _verified_keys(inputs: RoleKpiInputs, person: str) -> pd.DataFrame:
    """Credited resolving transitions authored by ``person``: key, ts."""
    detail = inputs.credited_detail
    if detail.empty or "author" not in detail.columns:
        return pd.DataFrame(columns=["key", "ts"])
    mine = detail[
        detail["credited"].fillna(False).astype(bool) & (detail["author"] == person)
    ]
    return mine[["key", "ts"]].copy()


# ---------------------------------------------------------------------------
# Component computations. Each returns a Component or None (None = no data;
# components_for substitutes the insufficient placeholder).
# ---------------------------------------------------------------------------


def _defect_escape(inputs: RoleKpiInputs, person: str) -> kpi.Component | None:
    """Share of this person's verified tickets that later came back.

    Verification = a changelog-credited resolving transition they authored;
    escape = that ticket shows at least one reopen in the same window
    (``integrity.reresolve_events``). Blind to bugs nobody filed, and to
    reopens outside the window - the rubric's own stated blind spot.
    """
    verified = _verified_keys(inputs, person)
    keys = sorted(set(verified["key"].astype(str)))
    if len(keys) < MIN_N:
        return None
    rr = inputs.reresolve
    escaped: list[str] = []
    if not rr.empty and {"key", "reopens"} <= set(rr.columns):
        reopens = dict(zip(rr["key"].astype(str), pd.to_numeric(rr["reopens"], errors="coerce").fillna(0)))
        escaped = [k for k in keys if reopens.get(k, 0) > 0]
    share = len(escaped) / len(keys)
    detail = f"{len(escaped)} of {len(keys)} tickets they verified were later reopened"
    if escaped:
        detail += f" ({', '.join(escaped[:5])}{'…' if len(escaped) > 5 else ''})"
    return kpi.Component("Defect escape rate", 100.0 * (1.0 - share), detail, n=len(keys))


def _rework_after_verification(inputs: RoleKpiInputs, person: str) -> kpi.Component | None:
    """Share of their verified tickets that needed more than one resolution.

    Counts every entry into resolved from the changelog, so a reopen that was
    quietly re-resolved (invisible to the reopened JQL) still lands here.
    """
    verified = _verified_keys(inputs, person)
    keys = sorted(set(verified["key"].astype(str)))
    if len(keys) < MIN_N:
        return None
    rr = inputs.reresolve
    reworked: list[str] = []
    if not rr.empty and {"key", "resolutions"} <= set(rr.columns):
        resolutions = dict(
            zip(rr["key"].astype(str), pd.to_numeric(rr["resolutions"], errors="coerce").fillna(0))
        )
        reworked = [k for k in keys if resolutions.get(k, 0) > 1]
    share = len(reworked) / len(keys)
    detail = f"{len(reworked)} of {len(keys)} verified tickets were declared done more than once"
    return kpi.Component("Rework after verification", 100.0 * (1.0 - share), detail, n=len(keys))


def _verification_cycle(inputs: RoleKpiInputs, person: str) -> kpi.Component | None:
    """Median days between a ticket's last pre-resolve move and their resolving move.

    Read from status transitions only, so it cannot be reset by field edits.
    Blind to CI queue time and to work done before the final status hop.
    """
    verified = _verified_keys(inputs, person)
    if verified.empty:
        return None
    status_events = inputs.status_events
    if status_events.empty:
        return None
    ts_by_key: dict[str, list[pd.Timestamp]] = {}
    for key, group in status_events.groupby("key"):
        ts_by_key[str(key)] = sorted(pd.to_datetime(group["ts"], utc=True, errors="coerce").dropna())
    gaps: list[float] = []
    for _, row in verified.iterrows():
        key = str(row["key"])
        resolve_ts = pd.to_datetime(row["ts"], utc=True, errors="coerce")
        if pd.isna(resolve_ts):
            continue
        prior = [t for t in ts_by_key.get(key, []) if t < resolve_ts]
        if not prior:
            continue
        gaps.append((resolve_ts - max(prior)).total_seconds() / 86400.0)
    if len(gaps) < MIN_N:
        return None
    median = float(pd.Series(gaps).median())
    return kpi.Component(
        "Verification cycle time",
        _ramp(median, VERIFY_FAST_DAYS, VERIFY_SLOW_DAYS),
        f"median {median:.1f}d from the ticket's previous move to their resolving move, "
        f"over {len(gaps)} verifications (100 at ≤{VERIFY_FAST_DAYS:.0f}d, 0 at ≥{VERIFY_SLOW_DAYS:.0f}d)",
        n=len(gaps),
    )


def _estimate_accuracy_component(inputs: RoleKpiInputs, person: str) -> kpi.Component | None:
    acc = inputs.accuracy
    if acc.empty or "assignee" not in acc.columns:
        return None
    mine = acc[acc["assignee"] == person]
    if mine.empty:
        return None
    row = mine.iloc[0]
    n = int(row.get("tickets", 0) or 0)
    if n < MIN_N or not bool(row.get("enough_data", False)):
        return None
    score = accuracy_score(float(row["median_ratio"]) if pd.notna(row["median_ratio"]) else None)
    if score is None:
        return None
    iqr = row.get("iqr")
    iqr_text = f", IQR {float(iqr):.1f}" if pd.notna(iqr) else ""
    return kpi.Component(
        "Estimate accuracy",
        score,
        f"median logged/estimated ×{float(row['median_ratio']):.2f} over {n} finished tickets{iqr_text} "
        "(100 at ×1.0, 0 at half or double)",
        n=n,
    )


def _cycle_time_component(inputs: RoleKpiInputs, person: str) -> kpi.Component | None:
    by_person = inputs.cycle_by_person
    if by_person.empty or "person" not in by_person.columns:
        return None
    mine = by_person[by_person["person"] == person]
    if mine.empty:
        return None
    row = mine.iloc[0]
    n = int(row.get("lead_time_tickets", 0) or 0)
    median = row.get("median_lead_time_days")
    if n < MIN_N or pd.isna(median):
        return None
    median = float(median)
    return kpi.Component(
        "Cycle time",
        _ramp(median, CYCLE_FAST_DAYS, CYCLE_SLOW_DAYS),
        f"median {median:.1f}d first-start to first-resolve over {n} tickets "
        f"(100 at ≤{CYCLE_FAST_DAYS:.0f}d, 0 at ≥{CYCLE_SLOW_DAYS:.0f}d)",
        n=n,
    )


def _staleness_component(inputs: RoleKpiInputs, owned: pd.DataFrame) -> kpi.Component | None:
    """Share of their open tickets whose *status* moved within kpi.STALE_DAYS.

    Deliberately reads ``integrity.status_age_days`` rather than ``idle_days``:
    KPI_SPEC.md exploit #1 is that idle_days resets on cosmetic edits, and the
    designer rubric's blind-spot note exists precisely because the code rubric
    still carries that flaw. This component does not.
    """
    if owned is None or owned.empty or "key" not in owned.columns:
        return None
    age = inputs.status_age
    if age.empty or "key" not in age.columns:
        return None
    mine = age[age["key"].isin(set(owned["key"].astype(str)))]
    if mine.empty:
        return None
    ages = pd.to_numeric(mine["status_age_days"], errors="coerce").dropna()
    if ages.empty:
        return None
    fresh = int((ages < kpi.STALE_DAYS).sum())
    return kpi.Component(
        "Staleness",
        100.0 * fresh / len(ages),
        f"{len(ages) - fresh} of {len(ages)} open ticket(s) without a status move for "
        f"{kpi.STALE_DAYS:.0f}+ days (status-change clock, immune to cosmetic edits)",
        n=int(len(ages)),
    )


def _handoff_component(gradable: pd.DataFrame) -> kpi.Component | None:
    """Mean ticket-quality score, same declared-score basis as Devin-ready docs."""
    if gradable is None or gradable.empty or "quality_score" not in gradable.columns:
        return None
    quality = pd.to_numeric(gradable["quality_score"], errors="coerce").dropna()
    if quality.empty:
        return None
    return kpi.Component(
        "Handoff completeness",
        100.0 * float(quality.mean()) / 5.0,
        f"avg ticket score {quality.mean():.1f}/5 over {len(quality)} tickets - a declared "
        "score, graded by whoever scores it",
        n=int(len(quality)),
    )


def _triage_latency(inputs: RoleKpiInputs, person: str) -> kpi.Component | None:
    """Median days from a ticket's creation to this person's first touch of it.

    A ticket they have not touched yet counts at its current age, so ignoring
    the queue can never outscore triaging it late. Reads every ticket created
    in the window from the frames the page already holds; a ticket created and
    fully closed out of both frames is invisible here (stated, not hidden).
    """
    tickets = inputs.all_tickets
    if tickets.empty or "created" not in tickets.columns or inputs.now is None:
        return None
    if "key" not in tickets.columns:
        return None
    created_all = pd.to_datetime(tickets["created"], utc=True, errors="coerce")
    window_start = inputs.now - pd.Timedelta(days=WINDOW_DAYS)
    recent = tickets[(created_all >= window_start) & created_all.notna()].copy()
    if recent.empty:
        return None
    recent = recent.drop_duplicates(subset=["key"])
    events = inputs.raw_events
    if events is None or events.empty:
        return None
    # First touch = first changelog entry of ANY kind authored by the person -
    # for a PM, setting a priority, an assignee, a sprint or a label is all
    # real triage, so no field filter here.
    mine = events[events["author"] == person]
    first_touch: dict[str, pd.Timestamp] = {}
    if not mine.empty:
        ts = pd.to_datetime(mine["ts"], utc=True, errors="coerce")
        grouped = mine.assign(_ts=ts).dropna(subset=["_ts"]).groupby("key")["_ts"].min()
        first_touch = {str(k): v for k, v in grouped.items()}
    created = pd.to_datetime(recent["created"], utc=True, errors="coerce")
    latencies: list[float] = []
    untouched = 0
    for key, created_ts in zip(recent["key"].astype(str), created):
        if pd.isna(created_ts):
            continue
        touch = first_touch.get(key)
        if touch is not None and touch >= created_ts:
            latencies.append((touch - created_ts).total_seconds() / 86400.0)
        else:
            untouched += 1
            latencies.append((inputs.now - created_ts).total_seconds() / 86400.0)
    if len(latencies) < MIN_N:
        return None
    median = float(pd.Series(latencies).median())
    return kpi.Component(
        "Triage latency",
        _ramp(median, TRIAGE_FAST_DAYS, TRIAGE_SLOW_DAYS),
        f"median {median:.1f}d from creation to their first touch over {len(latencies)} new "
        f"tickets; {untouched} still untouched count at their current age "
        f"(100 at ≤{TRIAGE_FAST_DAYS:.0f}d, 0 at ≥{TRIAGE_SLOW_DAYS:.0f}d)",
        n=len(latencies),
    )


def _board_hygiene(inputs: RoleKpiInputs) -> kpi.Component | None:
    """Org-wide share of open non-backlog tickets that are assigned, prioritised
    and (where the policy applies) estimated. A PM is scored on the whole
    board; the blind spot - an uncooperative team reads the same as an absent
    PM - is the rubric's own.
    """
    open_tickets = inputs.open_tickets
    if open_tickets.empty:
        return None
    n = len(open_tickets)
    shares: list[float] = []
    notes: list[str] = []
    if "assignee" in open_tickets.columns:
        assigned = int((~open_tickets["assignee"].fillna("Unassigned").astype(str).str.strip().isin(["", "Unassigned"])).sum())
        shares.append(assigned / n)
        notes.append(f"{assigned}/{n} assigned")
    if "priority" in open_tickets.columns:
        prioritised = int(open_tickets["priority"].notna().sum())
        shares.append(prioritised / n)
        notes.append(f"{prioritised}/{n} prioritised")
    if {"policy_applies", "has_estimate"} <= set(open_tickets.columns):
        applies = open_tickets[open_tickets["policy_applies"].fillna(False).astype(bool)]
        if not applies.empty:
            estimated = int(applies["has_estimate"].fillna(False).astype(bool).sum())
            shares.append(estimated / len(applies))
            notes.append(f"{estimated}/{len(applies)} estimated where required")
    if not shares:
        return None
    return kpi.Component(
        "Board hygiene",
        100.0 * sum(shares) / len(shares),
        "org-wide: " + ", ".join(notes),
        n=n,
    )


def _estimate_coverage(inputs: RoleKpiInputs) -> kpi.Component | None:
    open_tickets = inputs.open_tickets
    if open_tickets.empty or not {"policy_applies", "has_estimate"} <= set(open_tickets.columns):
        return None
    applies = open_tickets[open_tickets["policy_applies"].fillna(False).astype(bool)]
    if applies.empty:
        return None
    estimated = int(applies["has_estimate"].fillna(False).astype(bool).sum())
    return kpi.Component(
        "Estimate coverage",
        100.0 * estimated / len(applies),
        f"{estimated} of {len(applies)} open tickets the estimate policy applies to carry one "
        "(presence only - accuracy is the estimate-accuracy metric's job)",
        n=int(len(applies)),
    )


def _stale_queue(inputs: RoleKpiInputs) -> kpi.Component | None:
    age = inputs.status_age
    if age.empty or "status_age_days" not in age.columns:
        return None
    ages = pd.to_numeric(age["status_age_days"], errors="coerce").dropna()
    if ages.empty:
        return None
    stale = int((ages >= kpi.STALE_DAYS).sum())
    return kpi.Component(
        "Stale queue rate",
        100.0 * (1.0 - stale / len(ages)),
        f"{stale} of {len(ages)} open tickets org-wide without a status move for "
        f"{kpi.STALE_DAYS:.0f}+ days",
        n=int(len(ages)),
    )


def _hours_vs_output(inputs: RoleKpiInputs, person: str) -> kpi.Component | None:
    hpl = inputs.hours_per_line
    if hpl.empty or "assignee" not in hpl.columns:
        return None
    mine = hpl[hpl["assignee"] == person]
    if len(mine) < MIN_N:
        return None
    outliers = int(mine["is_outlier"].fillna(False).astype(bool).sum())
    n = len(mine)
    return kpi.Component(
        "Hours vs delivered output",
        100.0 * (n - outliers) / n,
        f"{outliers} of {n} finished tickets landed far from the team's hours-per-changed-line "
        "median (MAD-based; a flag means open the PR, not a verdict)",
        n=n,
    )


def _estimate_churn_component(inputs: RoleKpiInputs, person: str) -> kpi.Component | None:
    """Mid-flight estimate raises they authored, per ticket they moved.

    An estimate revised upward after work started is KPI_SPEC's most direct
    padding signal; here it is normalised by how much they actually moved so
    volume alone neither hides nor inflates it. Blind to a padded estimate
    set once before start and never touched.
    """
    status_events = inputs.status_events
    if status_events.empty:
        return None
    moved = status_events[status_events["author"] == person]
    tickets_moved = int(moved["key"].nunique()) if not moved.empty else 0
    if tickets_moved < MIN_N:
        return None
    churn = inputs.churn
    raises = 0
    hours_added = 0.0
    if not churn.empty and "author" in churn.columns:
        mine = churn[(churn["author"] == person) & (churn["direction"] == "raised")]
        raises = int(len(mine))
        hours_added = float(pd.to_numeric(mine.get("delta_hours"), errors="coerce").fillna(0).sum())
    share = min(1.0, raises / tickets_moved)
    detail = (
        f"{raises} mid-flight estimate raise(s) (+{hours_added:.0f}h) across {tickets_moved} "
        "tickets they moved"
    )
    return kpi.Component("Estimate churn", 100.0 * (1.0 - share), detail, n=tickets_moved)


# ---------------------------------------------------------------------------
# The per-role assembly
# ---------------------------------------------------------------------------


def components_for(
    inputs: RoleKpiInputs,
    person: str,
    role: str | None,
    owned: pd.DataFrame,
    gradable: pd.DataFrame,
) -> list[kpi.Component] | None:
    """The component rows for one person on a non-code rubric, or ``None``.

    ``None`` means "not this module's rubric" - code roles keep going through
    ``kpi.components``, and exec/no-rubric/unknown roles keep their sentinels.
    A returned list always has exactly one row per rubric component, in rubric
    order; unmeasurable components come back ``sufficient=False`` with the
    named input they need, never as a zero.
    """
    if role is None or role in roles.CODE_ROLES:
        return None
    rubric = roles.ROLE_RUBRIC.get(role)
    if rubric is None:
        return None

    computed: dict[str, kpi.Component | None] = {}
    names = {c.name for c in rubric.components}

    if "Defect escape rate" in names:
        computed["Defect escape rate"] = _defect_escape(inputs, person)
    if "Verification cycle time" in names:
        computed["Verification cycle time"] = _verification_cycle(inputs, person)
    if "Rework after verification" in names:
        computed["Rework after verification"] = _rework_after_verification(inputs, person)
    if "Estimate accuracy" in names:
        computed["Estimate accuracy"] = _estimate_accuracy_component(inputs, person)
    if "Triage latency" in names:
        computed["Triage latency"] = _triage_latency(inputs, person)
    if "Board hygiene" in names:
        computed["Board hygiene"] = _board_hygiene(inputs)
    if "Estimate coverage" in names:
        computed["Estimate coverage"] = _estimate_coverage(inputs)
    if "Stale queue rate" in names:
        computed["Stale queue rate"] = _stale_queue(inputs)
    if "Cycle time" in names:
        computed["Cycle time"] = _cycle_time_component(inputs, person)
    if "Staleness" in names:
        computed["Staleness"] = _staleness_component(inputs, owned)
    if "Handoff completeness" in names:
        computed["Handoff completeness"] = _handoff_component(gradable)
    if "Hours vs delivered output" in names:
        computed["Hours vs delivered output"] = _hours_vs_output(inputs, person)
    if "Estimate churn" in names:
        computed["Estimate churn"] = _estimate_churn_component(inputs, person)
    # "Automation coverage added" and "Bug validity rate" need inputs this
    # dashboard does not fetch (per-PR file lists; bug-resolution outcomes on
    # linked reports). They stay honest gaps rather than proxies that would
    # score noise - COMPONENT_INPUTS names exactly what each is waiting on.

    out: list[kpi.Component] = []
    for weight in rubric.components:
        part = computed.get(weight.name)
        out.append(part if part is not None else _gap(weight.name))
    return out


def score_from_parts(
    parts: list[kpi.Component], rubric: roles.Rubric
) -> tuple[float | None, float, str]:
    """``(score, covered_weight, note)`` for a non-code rubric's parts.

    The same 60%-of-weight withholding rule ``kpi.overall``/``roles.overall``
    both enforce, computed here directly from the component rows so the page
    can show the covered weight and the honest note beside the headline.
    """
    weights = rubric.weights()
    scored = [(weights.get(p.name, 0.0), p.score) for p in parts if p.sufficient]
    covered = sum(w for w, _ in scored)
    missing = [p.name for p in parts if not p.sufficient]
    needed = rubric.total * roles.MIN_SCORABLE_FRACTION
    if covered < needed or covered <= 0:
        note = (
            f"Not scored: only {covered:.0f} of {rubric.total:.0f} points of the "
            f"{rubric.rubric_id} rubric's weight had data ({needed:.0f} needed)."
        )
        if missing:
            note += f" Missing: {', '.join(missing)}."
        return None, covered, note
    score = sum(w * s for w, s in scored) / covered
    if missing:
        note = (
            f"Scored on {covered:.0f} of {rubric.total:.0f} points of weight. No data for: "
            f"{', '.join(missing)} - those components are not scored, not passed."
        )
    else:
        note = f"Scored on all {rubric.total:.0f} points of weight."
    return score, covered, note
