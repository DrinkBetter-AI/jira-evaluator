"""Board-hygiene and per-sprint compute for the Planning page.

Five metrics with no compute path anywhere else in the dashboard, per
``IMPLEMENTATION_PLAN.md`` Task 2E: who holds tickets after leaving,
carry-over broken out by sprint instead of only as a lifetime average,
the two board-hygiene bars that today have a bar and no number, and a
per-sprint view of who has no estimate at all. Every function follows
``KPI_SPEC.md`` §1: prefer the machine-recorded field over a declared one
(``carry_over_count`` is reused, never recomputed), name the blind spot in
the docstring, and emit evidence (ticket keys, per-person breakdowns) rather
than a bare score.

The ML board carries no sprint start/end date in Jira, and nothing here can
set one - see ``capacity.py``'s ``capacity_table_by_sprint`` and the two
functions below that aggregate ``carry_over_count`` across sprints. A
sprint with no dates is never folded into a cross-sprint total as a silent
zero; it is left out of the total and named in an ``excluded_sprints``
return value instead, so the page can say so.
"""

from __future__ import annotations

import os
from typing import Mapping, NamedTuple, Sequence

import pandas as pd

from capacity import UNASSIGNED, same_person, working_days


# The string sentinel used wherever "no estimates at all" would otherwise
# have to be reported as a numeric 0. Kept as a string (not ``None``/``NaN``)
# on purpose: a stray ``pd.to_numeric(..., errors="coerce")`` or ``.fillna(0)``
# turns ``None``/``NaN`` into ``0`` silently, which is exactly the failure
# this exists to prevent. A string survives both untouched (it becomes
# ``NaN`` under ``to_numeric``, never ``0``), and nothing in this module
# calls either on the column that carries it.
UNKNOWN = "unknown"

# roles_template.env's JIRA_FORMER_STAFF, as of the 19 Aug 2026 roster
# refresh (ROSTER.md "Former / inactive - 21 names still present in board
# data"). Used only when the env var itself is unset, so a missing
# deployment config degrades to the last confirmed list instead of finding
# nobody. roles.py (built in parallel, Task 2D-adjacent) will eventually own
# roster parsing; this is a small private copy rather than an import of a
# module under active concurrent development, and can be consolidated once
# that lands - see docs/assumptions/2E.md.
_FORMER_STAFF_FALLBACK: tuple[str, ...] = (
    "Sai Shankar",
    "Sarju",
    "Yantao He",
    "Dan O'Sullivan",
    "Shivanand",
    "Jon Wang",
    "Kevin Cai",
    "Ramin Shahid",
    "Amir",
    "Christina Lo",
    "Aleksei Pinchuk",
    "Saji",
    "Mark",
    "Courtney McNeil",
    "Lotte Karolina",
    "Jennifer",
    "Eva van Wielink",
    "Stanislav",
    "Saeid Parsa",
    "Armine Aproyan",
    "Haichen Song",
    "Dat",
)


class BoardResidue(NamedTuple):
    """A board-hygiene count plus the ticket keys behind it.

    Never returned as a bare integer - KPI_SPEC.md §1 rule 4 ("emit
    evidence, never a bare score") applies to housekeeping counts as much as
    to person-level scores. ``count`` and ``len(keys)`` always agree.
    """

    count: int
    keys: tuple[str, ...]


class GhostAssigned(NamedTuple):
    """Open tickets held by a former staff member, plus a per-person tally.

    ``by_person`` is empty and ``count`` is 0 in the case that actually
    ships (ROSTER.md, measured 19 Aug 2026: former staff hold zero open
    tickets) - this is a normal, valid result, not a missing one, and the
    caller does not need to special-case it to render correctly.

    Blind spot: reconciles against the departed-staff list, not against an
    enumerated active roster, so it cannot see a ghost whose Jira display
    name is a former employee under a spelling ``same_person`` cannot
    resolve to anything on ``JIRA_FORMER_STAFF`` (a nickname change, a
    married name). It also cannot see the reverse failure mode ROSTER.md
    documents as the *real* current problem: former staff still holding the
    *current-assignee* credit for a resolution someone else did the work
    on. That is a changelog-attribution question, not an open-ticket one,
    and is out of scope here.
    """

    count: int
    keys: tuple[str, ...]
    by_person: dict[str, int]


class CarryOverBySprint(NamedTuple):
    """Per-sprint carry-over table, plus a total across the dated sprints.

    ``total_carried``/``total_tickets`` are ``None`` - not ``0`` - when no
    sprint in scope has dates, because summing zero dated sprints is "we
    don't know", not "nobody carried anything over". This mirrors
    ``jira_client.py``'s own carry-over field: it stays a lifetime count per
    ticket there; this only regroups that same field by the sprint each
    ticket is in right now, it does not recompute or duplicate it.
    """

    by_sprint: pd.DataFrame
    total_carried: int | None
    total_tickets: int | None
    excluded_sprints: tuple[str, ...]


def _former_staff(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Departed-staff names from ``JIRA_FORMER_STAFF``, or the baked fallback."""
    source_env = env if env is not None else os.environ
    raw = source_env.get("JIRA_FORMER_STAFF")
    if not raw:
        return _FORMER_STAFF_FALLBACK
    names = tuple(name.strip() for name in raw.split(";") if name.strip())
    return names or _FORMER_STAFF_FALLBACK


def _open_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows not in Jira's Done status category.

    Blind spot: a project that never sets ``statusCategory`` (custom
    workflows sometimes don't) makes every row read as open; there is no
    second signal here to fall back on.
    """
    if "status_category" not in df.columns:
        return pd.Series(True, index=df.index)
    return ~df["status_category"].fillna("").astype(str).str.strip().str.lower().eq("done")


def _sprint_has_dates(group: pd.DataFrame) -> bool:
    """Whether this sprint's rows carry a usable start/end (the ML-board problem)."""
    if "sprint_start" not in group.columns or "sprint_end" not in group.columns:
        return False
    starts = group["sprint_start"].dropna()
    ends = group["sprint_end"].dropna()
    if starts.empty or ends.empty:
        return False
    return working_days(starts.iloc[0], ends.iloc[0]) > 0


def ghost_assigned(
    df: pd.DataFrame,
    *,
    former_staff: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> GhostAssigned:
    """Open tickets whose assignee is a known departed staff member.

    "Not on the active roster" is approximated here as "assigned to someone
    on the confirmed departed list" rather than by enumerating who currently
    *is* active - the active roster is ``roles.py``'s job, being built in
    parallel; this needs only the much smaller, already-agreed former-staff
    list, so it is read directly rather than imported. See the class
    docstring above for the reconciliation blind spot.
    """
    names = tuple(former_staff) if former_staff is not None else _former_staff(env)
    if df is None or df.empty or "assignee" not in df.columns or not names:
        return GhostAssigned(count=0, keys=(), by_person={})

    open_df = df[_open_mask(df)]
    if open_df.empty:
        return GhostAssigned(count=0, keys=(), by_person={})

    assignees = open_df["assignee"].fillna("").astype(str).str.strip()

    def _is_former(name: str) -> bool:
        return bool(name) and any(same_person(name, former) for former in names)

    is_ghost = assignees.apply(_is_former)
    matched = open_df.loc[is_ghost]
    if matched.empty:
        return GhostAssigned(count=0, keys=(), by_person={})

    keys = tuple(matched["key"].astype(str)) if "key" in matched.columns else tuple(matched.index.astype(str))
    by_person = matched["assignee"].fillna("").astype(str).value_counts().to_dict()
    return GhostAssigned(count=int(len(matched)), keys=keys, by_person=by_person)


def no_priority_count(df: pd.DataFrame) -> BoardResidue:
    """Open tickets with Jira's priority field empty.

    Blind spot: a Jira instance that adds a literal priority option named
    "None" reads as set, not missing, here - this only catches an empty
    field, not a declared "nobody has decided" value.
    """
    if df is None or df.empty or "priority" not in df.columns:
        return BoardResidue(count=0, keys=())
    open_df = df[_open_mask(df)]
    if open_df.empty:
        return BoardResidue(count=0, keys=())
    mask = open_df["priority"].fillna("").astype(str).str.strip().eq("")
    matched = open_df.loc[mask]
    keys = tuple(matched["key"].astype(str)) if "key" in matched.columns else tuple()
    return BoardResidue(count=int(mask.sum()), keys=keys)


def outside_any_sprint_count(df: pd.DataFrame) -> BoardResidue:
    """Open tickets carrying no sprint at all.

    Blind spot: a ticket whose sprint field points at a long-closed sprint
    (rather than being blank) reads as "in a sprint" here even though it is,
    in practice, exactly as unplanned - this only catches the blank case.
    """
    if df is None or df.empty or "sprint_name" not in df.columns:
        return BoardResidue(count=0, keys=())
    open_df = df[_open_mask(df)]
    if open_df.empty:
        return BoardResidue(count=0, keys=())
    sprint = open_df["sprint_name"]
    mask = sprint.isna() | sprint.astype(str).str.strip().eq("")
    matched = open_df.loc[mask]
    keys = tuple(matched["key"].astype(str)) if "key" in matched.columns else tuple()
    return BoardResidue(count=int(mask.sum()), keys=keys)


def carry_over_per_sprint(df: pd.DataFrame) -> CarryOverBySprint:
    """Per-sprint carry-over count and share, from the existing field.

    ``carry_over_count`` (``jira_client.py:873``, emitted at ``:924``) is a
    lifetime count of closed sprints a still-open ticket has passed through,
    already scored as a mean across every open ticket in ``hygiene.py`` and
    ``kpi.py``. This does not touch, recompute, or duplicate that number -
    it groups the same field by the sprint each ticket sits in *now*, and a
    ticket only ever counts once per sprint it is currently in (a ticket
    with ``carry_over_count == 3`` contributes 1 to that sprint's "Carried
    over" tally, not 3 - the 3 already lives in the lifetime metric this
    does not re-derive).

    Blind spot: a ticket that carried over five sprints and then dropped out
    of planning entirely (no sprint at all right now) is invisible to every
    per-sprint view here, exactly as it already is to the lifetime one.
    """
    columns = ["Sprint", "Tickets", "Carried over", "Share %", "Dated"]
    if (
        df is None
        or df.empty
        or "sprint_name" not in df.columns
        or "carry_over_count" not in df.columns
    ):
        return CarryOverBySprint(pd.DataFrame(columns=columns), None, None, ())

    scoped = df[df["sprint_name"].notna()].copy()
    if scoped.empty:
        return CarryOverBySprint(pd.DataFrame(columns=columns), None, None, ())

    scoped["_carried"] = pd.to_numeric(scoped["carry_over_count"], errors="coerce").fillna(0.0).gt(0)

    rows = []
    excluded: list[str] = []
    for sprint_name, group in scoped.groupby(scoped["sprint_name"].astype(str), sort=True):
        dated = _sprint_has_dates(group)
        if not dated:
            excluded.append(sprint_name)
        tickets = int(len(group))
        carried = int(group["_carried"].sum())
        share = round(carried / tickets * 100.0, 1) if tickets else None
        rows.append(
            {
                "Sprint": sprint_name,
                "Tickets": tickets,
                "Carried over": carried,
                "Share %": share,
                "Dated": dated,
            }
        )

    table = pd.DataFrame(rows, columns=columns)
    dated_rows = table[table["Dated"]]
    if dated_rows.empty:
        total_carried, total_tickets = None, None
    else:
        total_tickets = int(dated_rows["Tickets"].sum())
        total_carried = int(dated_rows["Carried over"].sum())
    return CarryOverBySprint(table, total_carried, total_tickets, tuple(excluded))


UNESTIMATED_COLUMNS = ["Sprint", "Assignee", "Tickets", "Unestimated", "Coverage"]


def unestimated_per_sprint(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Per sprint, per assignee: how many open tickets carry no numeric estimate.

    ``Coverage`` is the percentage of that person's tickets in that sprint
    that DO carry a numeric estimate. When none of them do, the cell holds
    the ``UNKNOWN`` sentinel rather than ``0.0`` - a person with zero
    estimated tickets this sprint is a person nobody has gotten an estimate
    from (or who is declining to give one), which is a different fact from
    "estimated everything at 0% accuracy", and the two read identically on
    a percentage axis unless kept structurally distinct. Nothing in this
    function calls ``fillna`` or ``pd.to_numeric`` on the ``Coverage``
    column, so there is no path here that turns the sentinel into a number.

    The second return value names sprints with no start/end date in Jira -
    informational only. Unlike hours (``capacity.capacity_table_by_sprint``)
    or a lifetime average (``carry_over_per_sprint``), a plain ticket count
    doesn't depend on the sprint's calendar window to be correct, so
    dateless sprints are not excluded from this table or from any total a
    caller derives from it; they are only flagged so the page can still say
    which sprints are missing dates.

    Blind spot: only ``original_estimate_sec`` counts as an estimate, unlike
    ``hygiene.estimate_policy``, which also credits a text-only original
    estimate ("2h" with no numeric seconds parsed). A per-sprint hours view
    is about numeric committed hours, so a words-only estimate reads the
    same as no estimate at all here.
    """
    if df is None or df.empty or "sprint_name" not in df.columns:
        return pd.DataFrame(columns=UNESTIMATED_COLUMNS), ()

    scoped = df[df["sprint_name"].notna()].copy()
    if scoped.empty:
        return pd.DataFrame(columns=UNESTIMATED_COLUMNS), ()

    assignee = scoped["assignee"].fillna(UNASSIGNED).astype(str).str.strip()
    scoped["_assignee"] = assignee.mask(assignee.eq(""), UNASSIGNED)
    estimate_sec = pd.to_numeric(scoped.get("original_estimate_sec"), errors="coerce").fillna(0.0)
    scoped["_estimated"] = estimate_sec.gt(0)

    rows = []
    excluded: list[str] = []
    for sprint_name, group in scoped.groupby(scoped["sprint_name"].astype(str), sort=True):
        if not _sprint_has_dates(group):
            excluded.append(sprint_name)
        for person, sub in group.groupby("_assignee"):
            tickets = int(len(sub))
            estimated = int(sub["_estimated"].sum())
            unestimated = tickets - estimated
            coverage = UNKNOWN if estimated == 0 else round(estimated / tickets * 100.0, 0)
            rows.append(
                {
                    "Sprint": sprint_name,
                    "Assignee": person,
                    "Tickets": tickets,
                    "Unestimated": unestimated,
                    "Coverage": coverage,
                }
            )

    table = pd.DataFrame(rows, columns=UNESTIMATED_COLUMNS)
    return table, tuple(excluded)


__all__ = [
    "UNKNOWN",
    "BoardResidue",
    "GhostAssigned",
    "CarryOverBySprint",
    "UNESTIMATED_COLUMNS",
    "ghost_assigned",
    "no_priority_count",
    "outside_any_sprint_count",
    "carry_over_per_sprint",
    "unestimated_per_sprint",
]
