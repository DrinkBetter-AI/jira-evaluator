"""One engineer's scorecard: components, trend and badges.

Everything here is pure arithmetic over frames the dashboard already fetches,
so the scorecard is auditable: every component names its inputs and a person
can recompute their own number from the page.

Design choices, deliberate:

- No raw ticket counts as performance. Output is trended two ways at once:
  against the person's own prior-quarter pace, and against the team in the same
  week. Neither alone is safe - the first rewards volatility, the second
  punishes whoever drew the hard tickets - so both are shown.
- Rework weighs as much as delivery: a resolved ticket that comes back, or a
  PR bounced with change requests, is the costliest kind of speed.
- A component with nothing to measure is reported as insufficient data and
  left out of the headline, with the missing weight named. It is never quietly
  renormalized away: an empty board used to *raise* a score by deleting the
  denominator, which is a hole an hourly contractor can drive through.
- Every component carries its sample size ``n``, because 100% on four tickets
  and 100% on forty are not the same claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd

from hygiene import CONTAINER_ISSUE_TYPES
import roles

# Component weights, summing to 100. Rework matches delivery on purpose, and
# delivery's twenty points are split between the self-relative reading and the
# peer-relative one so that neither can carry the headline alone.
WEIGHTS = {
    "Delivery": 10.0,
    "Delivery vs team": 10.0,
    "Rework": 20.0,
    "Weekly updates": 15.0,
    "Staleness": 10.0,
    "Carry-over": 10.0,
    "Estimates": 10.0,
    "Devin-ready docs": 10.0,
    "Urgent response": 5.0,
}

TOTAL_WEIGHT = sum(WEIGHTS.values())

# Below this much measured weight the headline is withheld rather than
# published from whatever happens to be left. Three components out of nine is
# not a score, it is a fragment, and a fragment that flatters whoever has the
# emptiest board.
MIN_SCORABLE_WEIGHT = 60.0

# A peer percentile computed against one or two colleagues is noise wearing a
# number, so the peer component needs a real cohort before it says anything.
MIN_PEERS = 3

# What each component needs before it can say anything, in the plain words the
# "insufficient data" note uses.
COMPONENT_INPUTS = {
    "Delivery": "this person's resolved counts for the last 7 and 90 days",
    "Delivery vs team": f"the same-week resolved counts of at least {MIN_PEERS} teammates",
    "Rework": "resolved and reopened history, or reviewed PRs",
    "Weekly updates": "open non-backlog tickets",
    "Staleness": "open non-backlog tickets",
    "Carry-over": "open tickets carrying sprint history",
    "Estimates": "tickets the estimate policy applies to",
    "Devin-ready docs": "quality-scored tickets they own or reported",
    "Urgent response": "open High+ priority tickets",
}

# A ticket untouched this long has, by the team's working agreement, missed
# its weekly update.
WEEKLY_UPDATE_DAYS = 7.0
STALE_DAYS = 15.0
URGENT_IDLE_DAYS = 3.0
_URGENT = {"highest", "high", "urgent", "critical", "blocker"}

# Carry-overs at which the component reads zero.
CARRY_OVER_CEILING = 3.0


@dataclass
class Component:
    """One scorecard row.

    ``n`` is how many observations the score stands on - tickets, PRs, resolved
    issues - and it is part of the finding, not decoration. ``sufficient`` is
    False on a placeholder row that exists only to say a component could not be
    measured; those rows are excluded from ``overall`` and must be rendered as
    "insufficient data" rather than as a zero.
    """

    name: str
    score: float  # 0-100
    detail: str
    n: int | None = None
    sufficient: bool = True


@dataclass
class Coverage:
    """How much of the scorecard actually had data behind it."""

    covered_weight: float
    total_weight: float
    measured: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    scorable: bool = True
    note: str = ""


def delivered_hours(resolved: pd.DataFrame) -> tuple[float, int, int]:
    """Estimated hours on a set of resolved tickets, and how blind that sum is.

    Returns ``(hours, tickets, unestimated)``. The hours are what the team
    *estimated* the work at, not effort anyone recorded: with almost nothing
    logged in Jira, this is the closest honest proxy, and the unestimated count
    says how far the sum understates the week.
    """
    if resolved.empty:
        return 0.0, 0, 0
    hours = pd.to_numeric(
        resolved.get("estimate_hours", pd.Series(dtype="float64")), errors="coerce"
    ).fillna(0.0)
    containers = (
        _norm(resolved.get("issue_type", pd.Series("", index=resolved.index)))
        .str.replace(r"[-_\s]", "", regex=True)
        .isin(CONTAINER_ISSUE_TYPES)
    )
    # An epic closing is not a week's work delivered; its children already are.
    counted = resolved[~containers]
    hours = hours[~containers]
    return (
        round(float(hours.sum()), 1),
        int(len(counted)),
        int((hours <= 0).sum()),
    )


def _norm(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _pct(hit: int, total: int) -> float:
    return 100.0 * hit / total if total else 0.0


def weekly_rate(count: int | None, days: int) -> float | None:
    """Events per week over a window; None when the read failed.

    The windows this is called with are cumulative and overlapping (the 90-day
    count contains the 7-day one), which is right for the trend strip on the
    page - each row answers "how fast over the last N days" - and wrong for
    comparing a week against a baseline. Use ``baseline_rate`` for that.
    """
    if count is None or days <= 0:
        return None
    return 7.0 * count / days


def baseline_rate(
    recent_count: int | None,
    window_count: int | None,
    recent_days: int = 7,
    window_days: int = 90,
) -> float | None:
    """Weekly pace over the window *excluding* the recent period.

    ``resolved_90`` is a count over the last ninety days, and those ninety days
    include the seven the current rate is measured over. Comparing one against
    the other therefore compares this week against a baseline this week is part
    of: a burst drags its own yardstick up, which flattens the ratio and hides
    the burst. Subtracting the recent count leaves the prior 83 days, which is
    the comparison the component claims to be making.

    Returns None when either count is missing, and clamps at zero because the
    two counts come from separate JQL queries and can disagree at the boundary.
    """
    if recent_count is None or window_count is None:
        return None
    days = window_days - recent_days
    if days <= 0:
        return None
    return 7.0 * max(window_count - recent_count, 0) / days


def peer_percentile(value: float | None, peers: Sequence[float] | None) -> float | None:
    """Where ``value`` sits in the team's distribution, 0-100, ties split evenly.

    The person's own number belongs in ``peers``: this is a rank within the
    team, and leaving themselves out would compare them against a team they are
    not on. Returns None below ``MIN_PEERS`` colleagues, because a percentile
    over two people is a coin toss with decimal places.

    What it cannot see: difficulty. Whoever takes the gnarly tickets resolves
    fewer of them and ranks lower for it, which is exactly why this is half of
    delivery and not all of it.
    """
    if value is None or peers is None:
        return None
    numbers = [float(p) for p in peers if p is not None and not pd.isna(p)]
    if len(numbers) < MIN_PEERS:
        return None
    below = sum(1 for p in numbers if p < value)
    ties = sum(1 for p in numbers if p == value)
    return 100.0 * (below + 0.5 * ties) / len(numbers)


def _peer_values(peers: Sequence[float] | Mapping[str, float] | None) -> list[float] | None:
    if peers is None:
        return None
    if isinstance(peers, Mapping):
        return [float(v) for v in peers.values() if v is not None and not pd.isna(v)]
    return [float(v) for v in peers if v is not None and not pd.isna(v)]


def components(
    owned: pd.DataFrame,
    gradable: pd.DataFrame,
    resolved_7: int | None,
    resolved_90: int | None,
    reopened_90: int | None,
    prs: pd.DataFrame,
    peer_resolved_7: Sequence[int] | Mapping[str, int] | None = None,
    include_gaps: bool = False,
) -> list[Component]:
    """The scorecard rows for one person.

    ``owned``: their open tickets, after ``estimate_policy`` and health fields.
    ``gradable``: their quality-scored tickets (owned or reported).
    ``prs``: their PRs, open and recently merged together.
    ``peer_resolved_7``: every teammate's 7-day resolved count, this person
    included, for the peer-relative delivery component. Omit it and that
    component is reported as insufficient data rather than guessed at.
    ``include_gaps``: also return a placeholder row for every component that
    could not be measured, so the page can show what is missing instead of
    silently shrinking the scorecard. Those rows carry ``sufficient=False`` and
    are ignored by ``overall``.
    """
    out: list[Component] = []

    rate_now = weekly_rate(resolved_7, 7)
    rate_base = baseline_rate(resolved_7, resolved_90, 7, 90)
    if rate_now is not None and rate_base is not None:
        if rate_base > 0:
            ratio = min(rate_now / rate_base, 2.0) / 2.0
            basis = f"{rate_now:.1f} resolved/week now vs {rate_base:.1f} over the prior 83 days"
        else:
            # Nothing resolved in the prior 83 days: there is no baseline to
            # beat, so anything at all reads as 100. That is the honest answer
            # to a meaningless ratio, and the reason the peer component exists.
            ratio = 1.0 if rate_now > 0 else 0.0
            basis = f"{rate_now:.1f} resolved/week now, no prior-83-day pace to compare against"
        out.append(
            Component(
                "Delivery",
                100.0 * ratio,
                basis,
                n=int(resolved_90 or 0),
            )
        )

    peers = _peer_values(peer_resolved_7)
    percentile = peer_percentile(
        None if resolved_7 is None else float(resolved_7), peers
    )
    if percentile is not None:
        out.append(
            Component(
                "Delivery vs team",
                percentile,
                (
                    f"{resolved_7} resolved this week: {percentile:.0f}th percentile "
                    f"of {len(peers or [])} people in the same 7 days"
                ),
                n=len(peers or []),
            )
        )

    rework_parts: list[float] = []
    rework_notes: list[str] = []
    if reopened_90 is not None and resolved_90:
        reopen_share = min(reopened_90 / resolved_90, 1.0)
        rework_parts.append(100.0 * (1.0 - reopen_share))
        rework_notes.append(f"{reopened_90} of {resolved_90} resolved came back")
    if not prs.empty and "changes_reviews" in prs.columns:
        bounced = int((prs["changes_reviews"].fillna(0).astype(int) > 0).sum())
        rework_parts.append(_pct(len(prs) - bounced, len(prs)))
        rework_notes.append(f"{bounced} of {len(prs)} PRs got change requests")
    if rework_parts:
        out.append(
            Component(
                "Rework",
                sum(rework_parts) / len(rework_parts),
                "; ".join(rework_notes),
                n=int(resolved_90 or 0) + int(len(prs)),
            )
        )

    if not owned.empty and "idle_days" in owned.columns:
        idle = pd.to_numeric(owned["idle_days"], errors="coerce").fillna(0.0)
        updated = int((idle <= WEEKLY_UPDATE_DAYS).sum())
        out.append(
            Component(
                "Weekly updates",
                _pct(updated, len(owned)),
                f"{updated} of {len(owned)} open tickets touched in the last 7 days",
                n=int(len(owned)),
            )
        )
        fresh = int((idle < STALE_DAYS).sum())
        out.append(
            Component(
                "Staleness",
                _pct(fresh, len(owned)),
                f"{len(owned) - fresh} of {len(owned)} ticket(s) idle {STALE_DAYS:.0f}+ days",
                n=int(len(owned)),
            )
        )

    if not owned.empty and "carry_over_count" in owned.columns:
        carry = pd.to_numeric(owned["carry_over_count"], errors="coerce").fillna(0.0)
        avg = float(carry.mean())
        out.append(
            Component(
                "Carry-over",
                100.0 * max(0.0, 1.0 - avg / CARRY_OVER_CEILING),
                f"{avg:.1f} sprint roll-overs per open ticket on average",
                n=int(len(owned)),
            )
        )

    if not owned.empty and "has_estimate" in owned.columns:
        # The team's written rule (hygiene.estimate_policy) exempts backlog
        # tickets and containers; score people against that rule, not a
        # stricter private one.
        if "policy_applies" in owned.columns:
            applies = owned[owned["policy_applies"].fillna(False).astype(bool)]
        else:
            applies = owned
            if "issue_type" in owned.columns:
                is_container = (
                    _norm(owned["issue_type"])
                    .str.replace(r"[-_\s]", "", regex=True)
                    .isin(CONTAINER_ISSUE_TYPES)
                )
                applies = owned[~is_container]
        if not applies.empty:
            estimated = int(applies["has_estimate"].fillna(False).astype(bool).sum())
            out.append(
                Component(
                    "Estimates",
                    _pct(estimated, len(applies)),
                    f"{estimated} of {len(applies)} tickets carry an estimate",
                    n=int(len(applies)),
                )
            )

    if not gradable.empty and "quality_score" in gradable.columns:
        quality = pd.to_numeric(gradable["quality_score"], errors="coerce").dropna()
        if not quality.empty:
            ready = 0
            if "devinable" in gradable.columns:
                ready = int((_norm(gradable["devinable"]) == "yes").sum())
            out.append(
                Component(
                    "Devin-ready docs",
                    100.0 * float(quality.mean()) / 5.0,
                    f"avg ticket score {quality.mean():.1f}/5 over {len(quality)} tickets, "
                    f"{ready} ready for Devin",
                    n=int(len(quality)),
                )
            )

    if not owned.empty and {"priority", "idle_days"} <= set(owned.columns):
        urgent = owned[_norm(owned["priority"]).isin(_URGENT)]
        if not urgent.empty:
            idle = pd.to_numeric(urgent["idle_days"], errors="coerce").fillna(0.0)
            moving = int((idle <= URGENT_IDLE_DAYS).sum())
            out.append(
                Component(
                    "Urgent response",
                    _pct(moving, len(urgent)),
                    f"{moving} of {len(urgent)} High+ tickets moved within 3 days",
                    n=int(len(urgent)),
                )
            )

    if include_gaps:
        out.extend(gap_components(out))
    return out


def gap_components(parts: list[Component]) -> list[Component]:
    """A placeholder row for every weighted component that had no data.

    The score is zero only because a dataclass needs a number there; the row
    exists to be rendered as "insufficient data - needs X", never as a zero.
    ``sufficient=False`` keeps it out of ``overall``, so these can be appended
    to a list of real components without changing anybody's number.
    """
    measured = {p.name for p in parts}
    return [
        Component(
            name,
            0.0,
            f"insufficient data - needs {COMPONENT_INPUTS.get(name, 'more data')}",
            n=0,
            sufficient=False,
        )
        for name in WEIGHTS
        if name not in measured
    ]


def coverage(parts: list[Component]) -> Coverage:
    """How much of the 100 points had data, and what to say about the rest.

    The old ``overall`` renormalised over whatever components existed, which
    meant an engineer holding no open non-backlog tickets quietly lost 45 points
    of denominator and was scored on the remainder. Moving every ticket to
    Backlog therefore *raised* the score. This is the note that makes that
    visible, and ``scorable`` is the switch that stops a headline being
    published from a fragment.
    """
    measured = [p.name for p in parts if p.sufficient]
    covered = sum(WEIGHTS.get(name, 0.0) for name in measured)
    missing = [name for name in WEIGHTS if name not in set(measured)]
    scorable = covered >= MIN_SCORABLE_WEIGHT
    if not missing:
        note = f"Scored on all {TOTAL_WEIGHT:.0f} points of weight."
    elif scorable:
        note = (
            f"Scored on {covered:.0f} of {TOTAL_WEIGHT:.0f} points of weight. "
            f"No data for: {', '.join(missing)} - those components are not scored, "
            "not passed."
        )
    else:
        note = (
            f"Not scored: only {covered:.0f} of {TOTAL_WEIGHT:.0f} points of weight "
            f"had any data ({MIN_SCORABLE_WEIGHT:.0f} needed). Missing: "
            f"{', '.join(missing)}. The components that do have data are shown "
            "individually; a headline built on the rest would say more about the "
            "empty board than about the person."
        )
    return Coverage(
        covered_weight=covered,
        total_weight=float(TOTAL_WEIGHT),
        measured=measured,
        missing=missing,
        scorable=scorable,
        note=note,
    )


def overall(parts: list[Component]) -> float | None:
    """Weighted mean of the components that had data - or None when too few did.

    Two things this deliberately does not do. It does not score an unmeasured
    component as zero: nobody should be punished for a component their work does
    not produce. And it does not renormalise away an unlimited amount of missing
    weight either, because that is the same hole from the other side - it pays a
    full-looking score for a fraction of the scorecard, and the fastest way to
    shrink the scorecard is to stop holding tickets. Below
    ``MIN_SCORABLE_WEIGHT`` the answer is None, and the caller is expected to
    show ``coverage(parts).note`` in its place.
    """
    scored = [p for p in parts if p.sufficient]
    weighted = [(WEIGHTS.get(p.name, 0.0), p.score) for p in scored]
    total_weight = sum(w for w, _ in weighted)
    if total_weight <= 0 or total_weight < MIN_SCORABLE_WEIGHT:
        return None
    return sum(w * s for w, s in weighted) / total_weight


def badges(
    parts: list[Component],
    owned: pd.DataFrame,
    gradable: pd.DataFrame,
    resolved_7: int | None,
    resolved_30: int | None,
    resolved_90: int | None,
    reopened_90: int | None,
    prs: pd.DataFrame,
) -> list[tuple[str, str]]:
    """(emoji + name, why) pairs the person has earned this week."""
    earned: list[tuple[str, str]] = []
    # Placeholder rows score zero by construction, so a badge must never read
    # one as a real result.
    by_name = {p.name: p for p in parts if p.sufficient}

    if resolved_7 is not None and resolved_7 >= 3:
        earned.append(("🚢 Shipper", f"{resolved_7} tickets resolved this week"))

    # These three windows overlap on purpose: they are the same cumulative rates
    # the trend strip shows, so the badge agrees with the page. The effect is
    # that a burst lifts all three at once and the badge under-fires rather than
    # over-fires - the safe direction for something that hands out praise. The
    # Delivery component uses the exclusive baseline instead, where the
    # comparison has to be exact.
    now = weekly_rate(resolved_7, 7)
    mid = weekly_rate(resolved_30, 30)
    base = weekly_rate(resolved_90, 90)
    if None not in (now, mid, base) and now > mid >= base and now > 0:
        earned.append(
            ("📈 Trending up", "resolving faster this week than the 30- and 90-day pace")
        )

    if not owned.empty and "idle_days" in owned.columns:
        idle = pd.to_numeric(owned["idle_days"], errors="coerce").fillna(0.0)
        if bool((idle <= WEEKLY_UPDATE_DAYS).all()):
            earned.append(("⭐ Fresh board", "every open ticket touched this week"))

    estimates = by_name.get("Estimates")
    if estimates is not None and estimates.score >= 100.0:
        earned.append(("📝 Estimator", "every ticket carries an estimate"))

    if not gradable.empty and "devinable" in gradable.columns:
        ready = int((_norm(gradable["devinable"]) == "yes").sum())
        if ready >= 3:
            earned.append(("🤖 AI teammate", f"{ready} tickets written ready for Devin"))

    if (
        reopened_90 is not None
        and resolved_90 is not None
        and resolved_90 >= 3
        and reopened_90 == 0
    ):
        earned.append(("🧪 No boomerangs", "nothing resolved in 90 days came back"))

    if not prs.empty and {"has_jira_key", "is_unowned"} <= set(prs.columns):
        # A merged PR's fetch carries no branch or body, so a missing key
        # there is invisible data, not a missing key; judge only rows whose
        # key could actually be seen.
        judgeable = (
            prs[prs["key_detectable"].fillna(False).astype(bool)]
            if "key_detectable" in prs.columns
            else prs
        )
        keyed = not judgeable.empty and bool(
            judgeable["has_jira_key"].fillna(False).astype(bool).all()
        )
        owned_prs = ~prs["is_unowned"].fillna(False).astype(bool)
        if keyed and bool(owned_prs.all()):
            earned.append(("🔍 Clean PRs", "every PR names its ticket and nobody left one unowned"))

    return earned


def rubric_for(person: str, roster: roles.Roster | None = None) -> roles.RubricLookup:
    """Which rubric applies to ``person`` - the WP5 fix, in one call.

    Rubric selection used to have nowhere to look role up at all: this module
    scored every person on the same nine components regardless of what they
    actually do. It now goes entirely through ``roles.py`` (the roster
    parsed from env, falling back to the baked ``roles_template.env``
    defaults when the env vars aren't set) rather than any local table here,
    so a role added to ``JIRA_ROLES`` without a rubric decision surfaces as
    ``"unclassified"`` - never a silent pass-through onto this module's own
    components. Anyone not on the roster at all comes back ``"role_unknown"``:
    not scored, and never scored wrongly by default.

    For code-producing roles (``roles.CODE_ROLES``) the returned rubric is
    ``roles.CODE_RUBRIC``, which is this module's own nine ``WEIGHTS`` - use
    ``components()``/``overall()`` above to actually score those people.
    Every other scored role's rubric is defined in ``roles.py``; the
    component-computation for those still needs building, which is future
    work, not this function's job.
    """
    ros = roster if roster is not None else roles.load_roster()
    return roles.rubric_for_person(ros, person)
