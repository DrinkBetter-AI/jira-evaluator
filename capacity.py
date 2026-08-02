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
AMBIGUOUS_AVAILABILITY = "Ambiguous roster name"
NO_AVAILABILITY = "No hours this sprint"
UNALLOCATED = "Unallocated"
UNASSIGNED = "Unassigned"

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


def _name_tokens(name: str) -> set[str]:
    return set(str(name).strip().lower().replace(".", " ").split())


def same_person(left: str, right: str) -> bool:
    """Whether two spellings of a name denote one person.

    The comparison is symmetric because the roster is hand-written and Jira is
    not: ``Farid`` must find ``Farid Shahidi``, and ``Mehdi Ordikhani`` must
    find ``Mehdi Ordikhani Fard``.
    """
    a, b = _name_tokens(left), _name_tokens(right)
    if not a or not b:
        return False
    return a <= b or b <= a


def match_weekly_hours(name: str, weekly_hours: dict[str, float]) -> float | None:
    """Declared hours for a Jira display name, tolerating shorter spellings."""
    if name in weekly_hours:
        return weekly_hours[name]
    for declared, hours in weekly_hours.items():
        if same_person(declared, name):
            return hours
    return None


def resolve_weekly_hours(
    names: list[str], weekly_hours: dict[str, float]
) -> tuple[dict[str, float], set[str]]:
    """Assign each declared allowance to at most one person.

    A roster written as bare first names is ambiguous the moment Jira holds two
    people who share one: crediting ``Dan=40`` to both ``Dan Smith`` and ``Dan
    Jones`` would invent a second contractor's worth of capacity. Such a
    declaration is withheld from everyone it could mean, and its claimants come
    back in the second element so the caller can say why.
    """
    resolved: dict[str, float] = {}
    spelled_out: set[str] = set()
    ambiguous: set[str] = set()
    for declared, hours in weekly_hours.items():
        exact = [name for name in names if name == declared]
        matches = exact or [name for name in names if same_person(declared, name)]
        if len(matches) == 1:
            # A later, looser entry must not overwrite hours declared against the
            # full Jira name, or spelling the name out would not settle a clash.
            if not exact and matches[0] in spelled_out:
                continue
            resolved[matches[0]] = hours
            if exact:
                spelled_out.add(matches[0])
        elif len(matches) > 1:
            ambiguous.update(matches)
    # An exact declaration outranks the ambiguity of a shorter one it collides
    # with: spelling a name in full is how the reviewer resolves the clash.
    ambiguous -= set(resolved)
    return resolved, ambiguous


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


def _status(
    name: str,
    committed: float,
    available: float,
    declared: bool,
    ambiguous: bool = False,
) -> str:
    if name == UNASSIGNED:
        return UNALLOCATED
    if ambiguous:
        return AMBIGUOUS_AVAILABILITY
    if available <= 0:
        return NO_AVAILABILITY if declared else UNKNOWN_AVAILABILITY
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
    known = set(committed.index.astype(str))
    # Someone already visible under their Jira display name must not appear a
    # second time under the short name the roster happens to use.
    unmatched = {
        declared
        for declared in weekly_hours
        if not any(same_person(declared, name) for name in known)
    }
    names = sorted(known | unmatched)
    if not names:
        return pd.DataFrame(columns=columns)

    declared_by_name, ambiguous = resolve_weekly_hours(names, weekly_hours)

    rows = []
    for name in names:
        held = float(committed.get(name, 0.0))
        declared_hours = declared_by_name.get(name)
        capacity = available_hours(declared_hours or 0.0, start, end)
        rows.append(
            {
                "Assignee": name,
                "Committed (h)": round(held, 1),
                "Available (h)": capacity,
                "Utilization %": round(held / capacity * 100.0, 0) if capacity > 0 else None,
                "Delta (h)": round(capacity - held, 1) if capacity > 0 else None,
                "Status": _status(
                    name,
                    held,
                    capacity,
                    declared_hours is not None,
                    ambiguous=name in ambiguous,
                ),
            }
        )

    table = pd.DataFrame(rows, columns=columns)
    # Blank cells are None, so a table where nobody has declared hours would carry
    # object-dtype columns into a numeric column config; coerce to float so the
    # dtype is the same whether or not anyone's availability is known.
    for column in ("Committed (h)", "Available (h)", "Utilization %", "Delta (h)"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    order = {
        OVER_COMMITTED: 0,
        AT_CAPACITY: 1,
        HAS_ROOM: 2,
        NO_AVAILABILITY: 3,
        UNALLOCATED: 4,
        AMBIGUOUS_AVAILABILITY: 5,
        UNKNOWN_AVAILABILITY: 6,
    }
    return (
        table.assign(_order=table["Status"].map(order))
        .sort_values(["_order", "Committed (h)"], ascending=[True, False])
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )
