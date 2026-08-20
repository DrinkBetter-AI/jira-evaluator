"""Prioritization scoring and roll-up aggregations for the Jira dashboard.

The score combines the signals the evaluator already computes (ticket age, idle
time, sprint carry-over) with Jira's own priority field and due dates, so the
same numbers that drive the health metrics also drive the ranking.

The staleness term used to be driven by ``idle_days``, which resets to zero on
any changelog edit at all - a label, a priority nudge, a description reword.
That let a groomed ticket outrank a genuinely stale one on both the score and
the "stale 15d+" queue count without a line of code moving. Both
:func:`add_priority_score` and :func:`assignee_rollup` now accept an optional
``events`` frame (``integrity.changelog_events`` output); when it is supplied
the staleness term is driven by :func:`integrity.status_age_days` instead - the
same days-since-a-real-status-transition clock a label edit cannot touch.
``idle_days`` stays on the output as "last touched", and ``masked_days`` (the
gap between the two, the tell that grooming happened) is added alongside it.

``events`` is optional and defaults to ``None`` on purpose: callers that have
not been wired to pass the board's changelog through yet (tracked in
``docs/assumptions/2A.md`` under "Render call sites for Phase 3") fall back to
the old ``idle_days``-driven behaviour exactly, rather than changing scores out
from under a caller that never opted in.
"""

from __future__ import annotations

import pandas as pd

import integrity


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


def _staleness_days(
    df: pd.DataFrame, events: pd.DataFrame | None
) -> tuple[pd.Series, pd.Series | None]:
    """The staleness clock the score uses, and ``masked_days`` when it is derivable.

    Falls back to ``idle_days`` when ``events`` is not supplied or carries no
    rows this board's tickets can be matched to - the same number the score used
    before this module knew about the changelog, not a zero that would read as
    "perfectly fresh".
    """
    idle_days = pd.to_numeric(df.get("idle_days"), errors="coerce").fillna(0.0)
    if events is None or events.empty or "key" not in df.columns:
        return idle_days, None

    aged = integrity.status_age_days(df, events)
    if aged.empty or len(aged) != len(df):
        return idle_days, None

    staleness = pd.to_numeric(aged["status_age_days"].reset_index(drop=True), errors="coerce")
    staleness.index = df.index
    staleness = staleness.fillna(idle_days)
    masked = pd.to_numeric(aged["masked_days"].reset_index(drop=True), errors="coerce")
    masked.index = df.index
    return staleness, masked


def _reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    priority = str(row.get("priority") or "").strip()
    if _priority_weight(priority) >= 30.0:
        reasons.append(f"{priority} priority")
    if float(row.get("staleness_days") or 0) >= 14:
        reasons.append(f"stale {float(row['staleness_days']):.0f}d")
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


def add_priority_score(df: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add ``priority_score`` (0-100), ``priority_rank`` and ``priority_reasons``.

    Expects the health fields added by :func:`transformations.add_ticket_health_fields`.

    Pass ``events`` (``integrity.changelog_events`` output, keyed the same way as
    ``df``) so the staleness term - 20 of the 100 points - is driven by
    :func:`integrity.status_age_days` rather than ``idle_days``. Without it the
    score falls back to ``idle_days`` exactly as before. See the module
    docstring for why this is opt-in rather than automatic.

    Adds ``staleness_days`` (the clock the score actually used) and, when
    ``events`` made it derivable, ``masked_days`` - the gap between
    ``status_age_days`` and ``idle_days``, i.e. how many days of apparent
    freshness came from edits that moved no work. ``idle_days`` itself is left
    untouched on the output: it is still worth showing, as "last touched".
    """
    out = df.copy()
    if out.empty:
        out["priority_score"] = pd.Series(dtype="float64")
        out["priority_rank"] = pd.Series(dtype="int64")
        out["priority_reasons"] = pd.Series(dtype="object")
        out["staleness_days"] = pd.Series(dtype="float64")
        return out

    now = pd.Timestamp.now(tz="UTC")

    staleness_days, masked_days = _staleness_days(out, events)
    age_days = pd.to_numeric(out.get("ticket_age_days"), errors="coerce").fillna(0.0)
    carry_over = pd.to_numeric(out.get("carry_over_count"), errors="coerce").fillna(0.0)

    priority_pressure = out.get("priority", pd.Series(index=out.index, dtype="object")).map(
        _priority_weight
    )
    idle_pressure = (
        staleness_days.clip(upper=IDLE_SATURATION_DAYS).div(IDLE_SATURATION_DAYS).mul(
            MAX_IDLE_PRESSURE
        )
    )
    age_pressure = age_days.clip(upper=AGE_SATURATION_DAYS).div(AGE_SATURATION_DAYS).mul(
        MAX_AGE_PRESSURE
    )
    carry_over_pressure = carry_over.mul(5.0).clip(upper=MAX_CARRY_OVER_PRESSURE)

    due_series = out.get("due_date", pd.Series(index=out.index, dtype="object"))
    due_pressure = due_series.map(lambda value: _due_pressure(value, now))

    status = out.get("status", pd.Series(index=out.index, dtype="object")).fillna("").astype(str)
    late_stage_pressure = (
        status.isin(LATE_STAGE_STATUSES) & (staleness_days > LATE_STAGE_STALE_DAYS)
    ).astype(float).mul(LATE_STAGE_PRESSURE)

    out["due_pressure"] = due_pressure
    out["late_stage_pressure"] = late_stage_pressure
    out["staleness_days"] = staleness_days.round(1)
    if masked_days is not None:
        out["masked_days"] = masked_days.round(1)

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


def assignee_rollup(df: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate ticket health per assignee for the organization-wide view.

    ``stale_15d_plus`` - the queue count - is driven by ``idle_days`` unless
    ``events`` is supplied, in which case it switches to
    :func:`integrity.status_age_days` for the same reason :func:`add_priority_score`
    does: ``idle_days`` resets on a label edit, so a groomed row could leave this
    queue without the ticket having moved. ``avg_idle_days``/``max_idle_days``
    stay as before ("last touched"); ``avg_status_age_days``/``max_status_age_days``
    and ``avg_masked_days`` are added alongside them only when ``events`` makes
    them derivable, so a caller that has not been wired to pass the changelog
    through yet (see ``docs/assumptions/2A.md``) gets the exact schema it always
    has.
    """
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
    events_columns = ["avg_status_age_days", "max_status_age_days", "avg_masked_days"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    owners = working["assignee"].fillna("").astype(str).str.strip()
    unowned = owners.str.lower().isin(_NO_OWNER_NAMES)
    working["assignee"] = owners.mask(unowned, NO_OWNER_LABEL)
    working["_unowned"] = unowned.astype(int)
    idle_days = pd.to_numeric(working.get("idle_days"), errors="coerce").fillna(0.0)
    staleness_days, masked_days = _staleness_days(working, events)
    # ``masked_days is not None`` is the signal that ``_staleness_days`` really
    # read status ages, not that events were merely passed in. When it falls
    # back to ``idle_days`` (no ``key`` column, or a ``status_age_days`` frame
    # that does not line up row-for-row) there is nothing to put in a column
    # named ``avg_status_age_days``, and emitting idle days under that name
    # would be the exact substitution this module exists to stop.
    has_events = masked_days is not None
    working["_status_age_days"] = staleness_days
    working["_masked_days"] = masked_days if masked_days is not None else float("nan")
    working["_stale"] = (staleness_days >= 15).astype(int)
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
        avg_status_age_days=("_status_age_days", "mean"),
        max_status_age_days=("_status_age_days", "max"),
        avg_masked_days=("_masked_days", "mean"),
        stale_15d_plus=("_stale", "sum"),
        unprioritized=("_unprioritized", "sum"),
        _unowned=("_unowned", "max"),
    )

    rollup = grouped.reset_index()
    numeric_columns = [
        "avg_priority_score",
        "top_priority_score",
        "avg_idle_days",
        "max_idle_days",
        "avg_status_age_days",
        "max_status_age_days",
        "avg_masked_days",
    ]
    for column in numeric_columns:
        rollup[column] = pd.to_numeric(rollup[column], errors="coerce").round(1)
    # Ownerless work is kept visible but sorted last, so it cannot outrank a
    # real person in a table about who is carrying what.
    rollup = rollup.sort_values(
        ["_unowned", "avg_priority_score", "open_tickets"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return rollup[columns + events_columns] if has_events else rollup[columns]
