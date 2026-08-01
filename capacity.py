"""Part-time capacity: what a person is holding versus the hours they actually have.

Committed hours mean nothing without each person's availability - 32h in a
sprint is half a full-timer's fortnight and double an 8h/week contractor's. The
weekly hours per person are declared in configuration (Jira does not carry
them) and spread over the sprint's own working days.
"""

from __future__ import annotations

import pandas as pd


WORKING_DAYS_PER_WEEK = 5.0

OVER_COMMITTED = "Over-committed"
AT_CAPACITY = "At capacity"
HAS_ROOM = "Has room"
UNKNOWN_AVAILABILITY = "Unknown"

# Utilization above this reads as over-committed rather than merely full.
OVER_COMMITTED_RATIO = 1.0
AT_CAPACITY_RATIO = 0.85


def parse_weekly_hours(spec: str) -> dict[str, float]:
    """Parse ``"Tam=10,Shivanand=20"`` into ``{"Tam": 10.0, "Shivanand": 20.0}``.

    Malformed entries are skipped rather than raising: a typo in an environment
    variable should cost one person's row, not the whole dashboard.
    """
    hours: dict[str, float] = {}
    for entry in str(spec or "").split(","):
        name, _, value = entry.partition("=")
        name = name.strip()
        if not name or not value.strip():
            continue
        try:
            hours[name] = float(value.strip())
        except ValueError:
            continue
    return hours


def working_days(start: object, end: object) -> float:
    """Weekdays in the inclusive [start, end] window; 0 when either is missing."""
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return 0.0
    first = pd.Timestamp(start).tz_localize(None).normalize()
    last = pd.Timestamp(end).tz_localize(None).normalize()
    if last < first:
        return 0.0
    return float(len(pd.bdate_range(first, last)))


def available_hours(weekly_hours: float, start: object, end: object) -> float:
    """Hours a person has inside the sprint window at their weekly rate."""
    days = working_days(start, end)
    if not days or weekly_hours <= 0:
        return 0.0
    return round(weekly_hours / WORKING_DAYS_PER_WEEK * days, 1)


def _status(committed: float, available: float) -> str:
    if available <= 0:
        return UNKNOWN_AVAILABILITY
    ratio = committed / available
    if ratio > OVER_COMMITTED_RATIO:
        return OVER_COMMITTED
    if ratio >= AT_CAPACITY_RATIO:
        return AT_CAPACITY
    return HAS_ROOM


def capacity_table(
    committed_hours: pd.Series,
    weekly_hours: dict[str, float],
    start: object,
    end: object,
) -> pd.DataFrame:
    """Committed vs available hours per assignee for one sprint window.

    ``committed_hours`` is indexed by assignee. People with declared hours but
    nothing assigned still appear, since idle capacity is the point of the view.
    """
    columns = [
        "Assignee",
        "Committed (h)",
        "Available (h)",
        "Utilization %",
        "Delta (h)",
        "Status",
    ]
    committed = pd.to_numeric(committed_hours, errors="coerce").fillna(0.0)
    names = sorted(set(committed.index.astype(str)) | set(weekly_hours))
    if not names:
        return pd.DataFrame(columns=columns)

    rows = []
    for name in names:
        held = float(committed.get(name, 0.0))
        capacity = available_hours(weekly_hours.get(name, 0.0), start, end)
        rows.append(
            {
                "Assignee": name,
                "Committed (h)": round(held, 1),
                "Available (h)": capacity,
                "Utilization %": round(held / capacity * 100.0, 0) if capacity > 0 else None,
                "Delta (h)": round(capacity - held, 1) if capacity > 0 else None,
                "Status": _status(held, capacity),
            }
        )

    table = pd.DataFrame(rows, columns=columns)
    order = {OVER_COMMITTED: 0, AT_CAPACITY: 1, HAS_ROOM: 2, UNKNOWN_AVAILABILITY: 3}
    return (
        table.assign(_order=table["Status"].map(order))
        .sort_values(["_order", "Committed (h)"], ascending=[True, False])
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )
