"""One engineer's scorecard: components, trend and badges.

Everything here is pure arithmetic over frames the dashboard already fetches,
so the scorecard is auditable: every component names its inputs and a person
can recompute their own number from the page.

Design choices, deliberate:

- No raw ticket counts as performance. Output is trended against the
  person's own 90-day baseline, not raced against colleagues.
- Rework weighs as much as delivery: a resolved ticket that comes back, or a
  PR bounced with change requests, is the costliest kind of speed.
- A component with nothing to measure is dropped and the weights renormalize,
  rather than scoring a person on an empty denominator.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hygiene import CONTAINER_ISSUE_TYPES

# Component weights. Rework matches delivery on purpose.
WEIGHTS = {
    "Delivery": 20.0,
    "Rework": 20.0,
    "Weekly updates": 15.0,
    "Staleness": 10.0,
    "Carry-over": 10.0,
    "Estimates": 10.0,
    "Devin-ready docs": 10.0,
    "Urgent response": 5.0,
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
    name: str
    score: float  # 0-100
    detail: str


def _norm(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _pct(hit: int, total: int) -> float:
    return 100.0 * hit / total if total else 0.0


def weekly_rate(count: int | None, days: int) -> float | None:
    """Events per week over a window; None when the read failed."""
    if count is None or days <= 0:
        return None
    return 7.0 * count / days


def components(
    owned: pd.DataFrame,
    gradable: pd.DataFrame,
    resolved_7: int | None,
    resolved_90: int | None,
    reopened_90: int | None,
    prs: pd.DataFrame,
) -> list[Component]:
    """The scorecard rows for one person.

    ``owned``: their open tickets, after ``estimate_policy`` and health fields.
    ``gradable``: their quality-scored tickets (owned or reported).
    ``prs``: their PRs, open and recently merged together.
    """
    out: list[Component] = []

    rate_now = weekly_rate(resolved_7, 7)
    rate_base = weekly_rate(resolved_90, 90)
    if rate_now is not None and rate_base is not None:
        if rate_base > 0:
            ratio = min(rate_now / rate_base, 2.0) / 2.0
        else:
            ratio = 1.0 if rate_now > 0 else 0.0
        out.append(
            Component(
                "Delivery",
                100.0 * ratio,
                f"{rate_now:.1f} resolved/week now vs {rate_base:.1f} their 90-day pace",
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
            )
        )
        fresh = int((idle < STALE_DAYS).sum())
        out.append(
            Component(
                "Staleness",
                _pct(fresh, len(owned)),
                f"{len(owned) - fresh} ticket(s) idle {STALE_DAYS:.0f}+ days",
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
                    f"avg ticket score {quality.mean():.1f}/5, {ready} ready for Devin",
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
                )
            )

    return out


def overall(parts: list[Component]) -> float | None:
    """Weighted mean of the components that exist; None when none do."""
    weighted = [(WEIGHTS.get(p.name, 0.0), p.score) for p in parts]
    total_weight = sum(w for w, _ in weighted)
    if total_weight <= 0:
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
    by_name = {p.name: p for p in parts}

    if resolved_7 is not None and resolved_7 >= 3:
        earned.append(("🚢 Shipper", f"{resolved_7} tickets resolved this week"))

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
