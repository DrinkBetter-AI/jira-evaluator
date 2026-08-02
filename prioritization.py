"""Prioritization scoring and roll-up aggregations for the Jira dashboard.

The score combines the signals the evaluator already computes (ticket age, idle
time, sprint carry-over) with Jira's own priority field and due dates, so the
same numbers that drive the health metrics also drive the ranking.
"""

from __future__ import annotations

import pandas as pd


PRIORITY_WEIGHTS: dict[str, float] = {
    "highest": 40.0,
    "urgent": 40.0,
    "high": 30.0,
    "medium": 18.0,
    "normal": 18.0,
    "low": 8.0,
    "lowest": 4.0,
    # This instance also defines these two; an idea is not work yet, and an
    # unset priority should not borrow weight from a real one.
    "idea": 2.0,
    "none": 0.0,
}
# Anything the instance adds later scores as barely-prioritised rather than
# silently mid-table, which is the safer direction for a name nobody mapped.
DEFAULT_PRIORITY_WEIGHT = 4.0

LATE_STAGE_STATUSES = {"IN DEV ENV", "Review in Staging", "Ready for Production"}
LATE_STAGE_STALE_DAYS = 6

MAX_IDLE_PRESSURE = 20.0
MAX_AGE_PRESSURE = 10.0
MAX_CARRY_OVER_PRESSURE = 15.0
LATE_STAGE_PRESSURE = 10.0

IDLE_SATURATION_DAYS = 30.0
AGE_SATURATION_DAYS = 180.0

SCORE_COLUMNS = ["priority_score", "priority_rank", "priority_reasons"]

# Jira writes this placeholder for issues nobody owns; it is work, not a person.
NO_OWNER_LABEL = "(no owner)"
_NO_OWNER_NAMES = {"", "unassigned", "none"}


def _priority_weight(value: object) -> float:
    # pd.isna over truthiness: a frame that has been through fillna or
    # convert_dtypes carries NaN or pd.NA here, and those are missing priorities
    # too - not the string "nan", and not a reason to raise.
    name = "" if value is None or pd.isna(value) else str(value).strip().lower()
    # A blank field and the priority literally named "None" are one bucket
    # everywhere else in the dashboard, so they have to score the same here.
    return PRIORITY_WEIGHTS.get(name or "none", DEFAULT_PRIORITY_WEIGHT)


def _due_pressure(due_date: object, now: pd.Timestamp) -> float:
    if due_date is None or pd.isna(due_date):
        return 0.0
    # Jira due dates are date-only, so compare whole days: due today is not yet late.
    # Coerced here rather than assumed, so the scorer works on a raw Jira frame.
    due = pd.to_datetime(due_date, utc=True, errors="coerce")
    if pd.isna(due):
        return 0.0
    days_left = (due.normalize() - now.normalize()).days
    if days_left < 0:
        return 20.0
    if days_left <= 3:
        return 15.0
    if days_left <= 7:
        return 10.0
    if days_left <= 14:
        return 5.0
    return 0.0


def _reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    priority = str(row.get("priority") or "").strip()
    if _priority_weight(priority) >= 30.0:
        reasons.append(f"{priority} priority")
    if float(row.get("idle_days") or 0) >= 14:
        reasons.append(f"idle {float(row['idle_days']):.0f}d")
    if float(row.get("ticket_age_days") or 0) >= 90:
        reasons.append(f"aged {float(row['ticket_age_days']):.0f}d")
    if float(row.get("due_pressure") or 0) >= 15:
        reasons.append("due date at risk")
    if float(row.get("carry_over_count") or 0) > 0:
        reasons.append(f"carried over {int(row['carry_over_count'])}x")
    if float(row.get("late_stage_pressure") or 0) > 0:
        reasons.append("stale in late stage")
    if not reasons:
        reasons.append("no escalation signals")
    return ", ".join(reasons)


def add_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``priority_score`` (0-100), ``priority_rank`` and ``priority_reasons``.

    Expects the health fields added by :func:`transformations.add_ticket_health_fields`.
    """
    out = df.copy()
    if out.empty:
        out["priority_score"] = pd.Series(dtype="float64")
        out["priority_rank"] = pd.Series(dtype="int64")
        out["priority_reasons"] = pd.Series(dtype="object")
        return out

    now = pd.Timestamp.now(tz="UTC")

    idle_days = pd.to_numeric(out.get("idle_days"), errors="coerce").fillna(0.0)
    age_days = pd.to_numeric(out.get("ticket_age_days"), errors="coerce").fillna(0.0)
    carry_over = pd.to_numeric(out.get("carry_over_count"), errors="coerce").fillna(0.0)

    priority_pressure = out.get("priority", pd.Series(index=out.index, dtype="object")).map(
        _priority_weight
    )
    idle_pressure = idle_days.clip(upper=IDLE_SATURATION_DAYS).div(IDLE_SATURATION_DAYS).mul(
        MAX_IDLE_PRESSURE
    )
    age_pressure = age_days.clip(upper=AGE_SATURATION_DAYS).div(AGE_SATURATION_DAYS).mul(
        MAX_AGE_PRESSURE
    )
    carry_over_pressure = carry_over.mul(5.0).clip(upper=MAX_CARRY_OVER_PRESSURE)

    due_series = out.get("due_date", pd.Series(index=out.index, dtype="object"))
    due_pressure = due_series.map(lambda value: _due_pressure(value, now))

    status = out.get("status", pd.Series(index=out.index, dtype="object")).fillna("").astype(str)
    late_stage_pressure = (
        status.isin(LATE_STAGE_STATUSES) & (idle_days > LATE_STAGE_STALE_DAYS)
    ).astype(float).mul(LATE_STAGE_PRESSURE)

    out["due_pressure"] = due_pressure
    out["late_stage_pressure"] = late_stage_pressure

    total = (
        priority_pressure
        + idle_pressure
        + age_pressure
        + carry_over_pressure
        + due_pressure
        + late_stage_pressure
    )
    out["priority_score"] = total.clip(lower=0.0, upper=100.0).round(1)
    out["priority_reasons"] = out.apply(_reasons, axis=1)
    out["priority_rank"] = (
        out["priority_score"].rank(method="first", ascending=False).astype("int64")
    )

    return out.drop(columns=["due_pressure", "late_stage_pressure"])


def assignee_rollup(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ticket health per assignee for the organization-wide view."""
    columns = [
        "assignee",
        "open_tickets",
        "avg_priority_score",
        "top_priority_score",
        "avg_idle_days",
        "max_idle_days",
        "stale_15d_plus",
        "unprioritized",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    owners = working["assignee"].fillna("").astype(str).str.strip()
    unowned = owners.str.lower().isin(_NO_OWNER_NAMES)
    working["assignee"] = owners.mask(unowned, NO_OWNER_LABEL)
    working["_unowned"] = unowned.astype(int)
    idle_days = pd.to_numeric(working.get("idle_days"), errors="coerce").fillna(0.0)
    working["_stale"] = (idle_days >= 15).astype(int)
    working["_unprioritized"] = (
        working.get("priority", pd.Series(index=working.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["", "none"])
        .astype(int)
    )

    grouped = working.groupby("assignee", dropna=False).agg(
        open_tickets=("key", "count"),
        avg_priority_score=("priority_score", "mean"),
        top_priority_score=("priority_score", "max"),
        avg_idle_days=("idle_days", "mean"),
        max_idle_days=("idle_days", "max"),
        stale_15d_plus=("_stale", "sum"),
        unprioritized=("_unprioritized", "sum"),
        _unowned=("_unowned", "max"),
    )

    rollup = grouped.reset_index()
    for column in ["avg_priority_score", "top_priority_score", "avg_idle_days", "max_idle_days"]:
        rollup[column] = pd.to_numeric(rollup[column], errors="coerce").round(1)
    # Ownerless work is kept visible but sorted last, so it cannot outrank a
    # real person in a table about who is carrying what.
    return rollup.sort_values(
        ["_unowned", "avg_priority_score", "open_tickets"],
        ascending=[True, False, False],
    ).reset_index(drop=True)[columns]
