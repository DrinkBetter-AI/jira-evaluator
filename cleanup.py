"""Backlog triage: which old tickets to look at, and what to do with them.

Deciding the fate of hundreds of tickets is only bearable one ticket at a time,
so this module supplies the queue and a default recommendation per ticket; the
decisions themselves live in the UI until the reviewer applies them.
"""

from __future__ import annotations

import pandas as pd


KEEP = "Keep"
CLOSE = "Close"
NEEDS_OWNER = "Needs owner"
SKIP = "Skip"
DECISIONS = (CLOSE, NEEDS_OWNER, KEEP, SKIP)

# A ticket nobody has touched for half a year is not "in progress" in any sense
# a person would recognise, whatever its status column says.
ABANDONED_IDLE_DAYS = 180
ABANDONED_AGE_DAYS = 365
NEVER_TAKEN_IDLE_DAYS = 90

_NO_OWNER = {"", "unassigned", "none"}


def is_unowned(assignee: object) -> bool:
    return str(assignee or "").strip().lower() in _NO_OWNER


def suggest_decision(row: pd.Series) -> tuple[str, str]:
    """A default decision and the one-line reason behind it.

    Deliberately conservative: everything it cannot argue for closing comes back
    as Keep, because the reviewer said some old tickets still matter.
    """
    age = float(row.get("ticket_age_days") or 0)
    idle = float(row.get("idle_days") or 0)
    unowned = is_unowned(row.get("assignee"))

    if age >= ABANDONED_AGE_DAYS and idle >= ABANDONED_IDLE_DAYS:
        return CLOSE, f"{age:.0f}d old, untouched for {idle:.0f}d"
    if unowned and idle >= NEVER_TAKEN_IDLE_DAYS:
        return CLOSE, f"never assigned, idle {idle:.0f}d"
    if idle >= ABANDONED_IDLE_DAYS:
        return CLOSE, f"stalled {idle:.0f}d with an owner"
    if unowned:
        return NEEDS_OWNER, f"no owner, idle {idle:.0f}d"
    return KEEP, f"owned and touched {idle:.0f}d ago"


def build_queue(
    df: pd.DataFrame,
    *,
    unassigned_only: bool = False,
    limit: int = 100,
) -> pd.DataFrame:
    """The oldest open tickets, worst first, with a suggested decision each."""
    columns = [
        "key",
        "summary",
        "status",
        "assignee",
        "reporter",
        "priority",
        "issue_type",
        "epic_summary",
        "ticket_age_days",
        "idle_days",
        "suggested",
        "why",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    frame = df.copy()
    if unassigned_only:
        frame = frame[frame["assignee"].map(is_unowned)]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame = frame.sort_values("ticket_age_days", ascending=False).head(limit).copy()
    suggestions = frame.apply(suggest_decision, axis=1)
    frame["suggested"] = [s[0] for s in suggestions]
    frame["why"] = [s[1] for s in suggestions]

    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns].reset_index(drop=True)


# Projects here run different workflows, and few of them offer "Done" from a
# Backlog status, so closing is expressed as a preference and resolved per
# ticket. Order matters: a cleanup closure should stay distinguishable from a
# ticket that was genuinely finished, which is why Done comes last.
CLOSING_STATUS_PREFERENCE = (
    "Archived",
    "Won't Do",
    "Wont Do",
    "Not needed",
    "Not Needed",
    "Cancelled",
    "Canceled",
    "Closed",
    "Rejected",
    "Done",
)


def closing_status(available: list[str]) -> str | None:
    """The best closing transition a ticket actually offers, or None."""
    lowered = {str(name).strip().lower(): str(name).strip() for name in available}
    for preferred in CLOSING_STATUS_PREFERENCE:
        match = lowered.get(preferred.lower())
        if match:
            return match
    return None


def decision_summary(decisions: dict[str, str]) -> dict[str, int]:
    """How many tickets sit in each decision bucket."""
    counts = {name: 0 for name in DECISIONS}
    for value in decisions.values():
        if value in counts:
            counts[value] += 1
    return counts


def pending_closures(queue: pd.DataFrame, decisions: dict[str, str]) -> list[str]:
    """Keys marked for closing, in queue order so the reviewer recognises them."""
    if queue.empty:
        return []
    return [key for key in queue["key"] if decisions.get(key) == CLOSE]
