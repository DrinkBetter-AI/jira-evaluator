"""A first draft of a sprint: a few named goals, filled to each person's hours.

A sprint is decided as "finish onboarding, finalise the quiz, ship checkout",
not as the top forty rows of a priority list - highest priority does not mean
this sprint. So the goals come first and the ranking only orders the work inside
them, while each person's hours are capped at what they actually have, after
taking out the standing overhead (review, Slack, the meetings everyone attends)
that no plan budgets for and every sprint spends.

The output is a proposal. Every row carries why it landed where it did, and the
caller is expected to overrule it.
"""

from __future__ import annotations

import re

import pandas as pd

from capacity import (
    WORKING_DAYS_PER_WEEK,
    available_hours,
    resolve_weekly_hours,
    working_days,
)


# Meetings, code review, Slack: hours that are spent every week and appear on no
# ticket. Budgeting a sprint without them is how a plan is full before it starts.
DEFAULT_OVERHEAD_HOURS_PER_WEEK = 4.0

# What an unestimated ticket is assumed to cost. It has to be something - leaving
# it at zero would let the sprint absorb unlimited unestimated work - and the row
# says so, because the honest fix is to estimate the ticket.
DEFAULT_TICKET_HOURS = 4.0

PLANNED = "This sprint"
NEXT_UP = "Next sprint"
NO_CAPACITY = "No room"
UNASSIGNED = "Unassigned"

_IN_FLIGHT = ("in progress", "in review", "review in staging", "in development")

NO_GOAL = "No goal"

_GOAL_STOPWORDS = frozenset(
    "and the for with into from all any our new finalize finalise finish"
    " complete ship launch".split()
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def parse_goals(spec: str) -> list[str]:
    """Goal names from one comma-separated line, in the order they were written.

    Order is the whole point: the first goal is the one the sprint is for, and
    gets the hours before the second one does.
    """
    seen: list[str] = []
    for part in str(spec or "").split(","):
        goal = part.strip()
        if goal and goal.lower() not in {existing.lower() for existing in seen}:
            seen.append(goal)
    return seen


def _goal_words(goal: str) -> set[str]:
    """The words in a goal that identify work, minus the verb it starts with.

    "Finalize checkout" and "Ship checkout" are the same goal; matching on
    *finalize* would file every ticket that happens to say "finalize".
    """
    words = {
        word
        for word in _WORD_RE.findall(str(goal or "").lower())
        if len(word) > 2 and word not in _GOAL_STOPWORDS
    }
    return words


def match_goals(df: pd.DataFrame, goals: list[str]) -> pd.Series:
    """Which goal each ticket serves, by the words it shares with the goal.

    A ticket is read through its summary, its epic's name and its labels, since
    "checkout" is as likely to be the epic as the ticket. Earlier goals win ties:
    a ticket that could serve either belongs to the one the sprint is named for.
    """
    if df.empty or not goals:
        return pd.Series([NO_GOAL] * len(df), index=df.index, dtype="object")

    def _column(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series("", index=df.index, dtype="object")
        values = df[name]
        if values.dtype == object and values.map(lambda item: isinstance(item, list)).any():
            return values.map(
                lambda item: " ".join(map(str, item)) if isinstance(item, list) else ""
            )
        return values.fillna("").astype(str)

    text = (
        _column("summary") + " " + _column("epic_summary") + " " + _column("labels")
    ).str.lower()
    words = text.map(lambda value: set(_WORD_RE.findall(value)))
    wanted = [(goal, _goal_words(goal)) for goal in goals]

    def _match(ticket_words: set[str]) -> str:
        for goal, goal_words in wanted:
            if goal_words and goal_words & ticket_words:
                return goal
        return NO_GOAL

    return words.map(_match)


def person_capacity(
    names: list[str],
    weekly_hours: dict[str, float],
    start: object,
    end: object,
    overhead_per_week: float = DEFAULT_OVERHEAD_HOURS_PER_WEEK,
) -> pd.DataFrame:
    """Hours each person can actually spend on tickets in this sprint window.

    Overhead is deducted per week and pro-rated over the sprint's own working
    days, so a two-week sprint takes out twice what a one-week sprint does. A
    part-timer's overhead is not scaled down: someone on ten hours a week still
    sits in the same standup.
    """
    columns = ["assignee", "weekly_hours", "gross_hours", "overhead_hours", "planning_hours"]
    days = working_days(start, end)
    declared, _ambiguous = resolve_weekly_hours(sorted(set(names)), weekly_hours)
    rows = []
    for name in sorted(set(names)):
        weekly = declared.get(name)
        if weekly is None:
            continue
        gross = available_hours(weekly, start, end)
        overhead = round(max(float(overhead_per_week), 0.0) / WORKING_DAYS_PER_WEEK * days, 1)
        rows.append(
            {
                "assignee": name,
                "weekly_hours": float(weekly),
                "gross_hours": gross,
                "overhead_hours": overhead,
                # Overhead can exceed a very part-time week; the floor is no
                # capacity, never negative capacity that later work borrows from.
                "planning_hours": round(max(gross - overhead, 0.0), 1),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _ticket_hours(df: pd.DataFrame, default_hours: float) -> pd.Series:
    """Estimated hours per ticket, falling back to an assumption."""
    for column in ("estimate_hours", "original_estimate_sec", "estimate_seconds"):
        if column in df.columns:
            hours = pd.to_numeric(df[column], errors="coerce")
            if column.endswith(("_sec", "_seconds")):
                hours = hours / 3600.0
            return hours.where(hours > 0)
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")


def plan_sprint(
    df: pd.DataFrame,
    capacity: pd.DataFrame,
    goals: list[str] | None = None,
    default_hours: float = DEFAULT_TICKET_HOURS,
) -> pd.DataFrame:
    """Fill each person's sprint, goal by goal and highest priority within each.

    Work already under way is placed before anything new regardless of goal or
    priority: a sprint that starts three half-finished tickets and finishes none
    is the failure this is meant to prevent. Work serving no goal is planned last
    rather than dropped, so the hours a goal is losing to it stay visible.
    """
    columns = [
        "key",
        "goal",
        "summary",
        "assignee",
        "status",
        "priority_score",
        "hours",
        "estimated",
        "plan",
        "why",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    frame = df.copy()
    hours = _ticket_hours(frame, default_hours)
    frame["estimated"] = hours.notna()
    frame["hours"] = hours.fillna(float(default_hours)).astype(float).round(1)
    frame["assignee"] = (
        frame.get("assignee", pd.Series(UNASSIGNED, index=frame.index))
        .fillna(UNASSIGNED)
        .astype(str)
        .str.strip()
        .replace("", UNASSIGNED)
    )
    frame["priority_score"] = pd.to_numeric(
        frame.get("priority_score", pd.Series(0.0, index=frame.index)), errors="coerce"
    ).fillna(0.0)
    statuses = frame.get("status", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["_in_flight"] = statuses.str.strip().str.lower().isin(_IN_FLIGHT)

    named = list(goals or [])
    if "goal" not in frame.columns:
        frame["goal"] = match_goals(frame, named)
    rank = {goal: index for index, goal in enumerate(named)}
    frame["_goal_rank"] = frame["goal"].map(lambda goal: rank.get(goal, len(rank)))

    budget = {
        str(row["assignee"]): float(row["planning_hours"])
        for _, row in capacity.iterrows()
    }

    ordered = frame.sort_values(
        ["_in_flight", "_goal_rank", "priority_score"], ascending=[False, True, False]
    )
    rows = []
    for _, ticket in ordered.iterrows():
        owner = str(ticket["assignee"])
        cost = float(ticket["hours"])
        left = budget.get(owner)
        if owner == UNASSIGNED:
            plan, why = NO_CAPACITY, "nobody owns it - give it an owner first"
        elif left is None:
            plan, why = NEXT_UP, "no declared hours for this person"
        elif cost <= left:
            budget[owner] = round(left - cost, 1)
            plan = PLANNED
            why = (
                "already in flight"
                if ticket["_in_flight"]
                else f"fits in {left:.1f}h left"
            )
        else:
            plan = NEXT_UP
            why = f"needs {cost:.1f}h, only {left:.1f}h left"
        if not ticket["estimated"]:
            why = f"{why} (assumed {cost:.0f}h, unestimated)"
        rows.append(
            {
                "key": ticket.get("key"),
                "goal": ticket.get("goal", NO_GOAL),
                "summary": ticket.get("summary"),
                "assignee": owner,
                "status": ticket.get("status"),
                "priority_score": round(float(ticket["priority_score"]), 1),
                "hours": cost,
                "estimated": bool(ticket["estimated"]),
                "plan": plan,
                "why": why,
            }
        )

    order = {PLANNED: 0, NEXT_UP: 1, NO_CAPACITY: 2}
    plan_df = pd.DataFrame(rows, columns=columns)
    return (
        plan_df.assign(
            _order=plan_df["plan"].map(order),
            _goal=plan_df["goal"].map(lambda goal: rank.get(goal, len(rank))),
        )
        .sort_values(
            ["_order", "_goal", "assignee", "priority_score"],
            ascending=[True, True, True, False],
        )
        .drop(columns=["_order", "_goal"])
        .reset_index(drop=True)
    )


def goal_load(plan: pd.DataFrame, goals: list[str] | None = None) -> pd.DataFrame:
    """Per goal: how much of it fits in this sprint and how much is left over.

    The question a goal-led sprint has to answer is whether all three goals fit,
    and if not which one is being half-done.
    """
    columns = ["goal", "tickets", "hours", "planned_tickets", "planned_hours", "left_out"]
    if plan.empty:
        return pd.DataFrame(columns=columns)
    chosen = plan[plan["plan"].eq(PLANNED)]
    table = (
        plan.groupby("goal", dropna=False)
        .agg(tickets=("key", "count"), hours=("hours", "sum"))
        .join(
            chosen.groupby("goal", dropna=False).agg(
                planned_tickets=("key", "count"), planned_hours=("hours", "sum")
            )
        )
        .fillna(0)
        .reset_index()
    )
    table["planned_tickets"] = table["planned_tickets"].astype(int)
    table["left_out"] = table["tickets"] - table["planned_tickets"]
    for column in ("hours", "planned_hours"):
        table[column] = table[column].round(1)
    rank = {goal: index for index, goal in enumerate(goals or [])}
    return (
        table.assign(_rank=table["goal"].map(lambda goal: rank.get(goal, len(rank))))
        .sort_values(["_rank", "hours"], ascending=[True, False])
        .drop(columns=["_rank"])
        .reset_index(drop=True)[columns]
    )


def plan_load(plan: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    """Per person: hours available, hours planned, hours left, and what is waiting.

    Driven by the plan it is handed rather than recomputed, so it keeps telling
    the truth after a human has overruled rows in the table above it.
    """
    columns = [
        "assignee",
        "planning_hours",
        "planned_hours",
        "left_hours",
        "planned_tickets",
        "waiting_tickets",
    ]
    if capacity.empty and plan.empty:
        return pd.DataFrame(columns=columns)

    chosen = plan[plan["plan"].eq(PLANNED)] if not plan.empty else plan
    planned = chosen.groupby("assignee")["hours"].sum() if not chosen.empty else pd.Series(dtype=float)
    counts = chosen.groupby("assignee")["key"].count() if not chosen.empty else pd.Series(dtype=int)
    waiting = (
        plan[plan["plan"].ne(PLANNED)].groupby("assignee")["key"].count()
        if not plan.empty
        else pd.Series(dtype=int)
    )

    # Someone with no declared hours and nothing planned is not under-loaded, they
    # are simply not being planned for; a row of blanks per such person would bury
    # the handful of people the sprint is actually about.
    names = sorted(
        set(capacity["assignee"].astype(str) if not capacity.empty else [])
        | set(chosen["assignee"].astype(str) if not chosen.empty else [])
    )
    budget = (
        capacity.set_index("assignee")["planning_hours"].to_dict()
        if not capacity.empty
        else {}
    )
    rows = []
    for name in names:
        available = budget.get(name)
        used = float(planned.get(name, 0.0))
        rows.append(
            {
                "assignee": name,
                "planning_hours": float(available) if available is not None else None,
                "planned_hours": round(used, 1),
                "left_hours": round(float(available) - used, 1) if available is not None else None,
                "planned_tickets": int(counts.get(name, 0)),
                "waiting_tickets": int(waiting.get(name, 0)),
            }
        )
    table = pd.DataFrame(rows, columns=columns)
    for column in ("planning_hours", "planned_hours", "left_hours"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table.sort_values("planned_hours", ascending=False).reset_index(drop=True)


__all__ = [
    "DEFAULT_OVERHEAD_HOURS_PER_WEEK",
    "DEFAULT_TICKET_HOURS",
    "NEXT_UP",
    "NO_CAPACITY",
    "NO_GOAL",
    "PLANNED",
    "goal_load",
    "match_goals",
    "parse_goals",
    "person_capacity",
    "plan_load",
    "plan_sprint",
]
