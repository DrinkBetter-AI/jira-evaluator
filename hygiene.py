"""Backlog hygiene: the estimate policy and the stale-ticket cleanup queue.

Two questions the health metrics cannot answer on their own: who is ignoring the
"estimate before it leaves Backlog" rule, and which old tickets are dead rather
than merely slow.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_STALE_DAYS = 90

# A ticket that never left these statuses was never really started.
NOT_STARTED_STATUSES = {"backlog", "to do", "todo", "discussion needed", "open"}


def _estimate_seconds(df: pd.DataFrame) -> pd.Series:
    """Estimate in seconds per row, from whichever column the fetch produced.

    The numeric column is authoritative, but a row can carry only the human
    string (``"2h"``); a marker second keeps such a row out of the violation
    list without inventing a duration.
    """
    seconds = pd.Series(0.0, index=df.index)
    for column in ("estimate_seconds", "original_estimate_sec"):
        if column in df.columns:
            seconds = seconds.where(
                seconds > 0, pd.to_numeric(df[column], errors="coerce").fillna(0.0)
            )
    if "original_estimate" in df.columns:
        has_text = df["original_estimate"].fillna("").astype(str).str.strip().ne("")
        seconds = seconds.where(~(seconds.le(0) & has_text), 1.0)
    return seconds


def _normalized(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index)
    return df[column].fillna("").astype(str).str.strip().str.lower()


def estimate_policy(df: pd.DataFrame, backlog_statuses: set[str]) -> pd.DataFrame:
    """Flag tickets that left Backlog without an estimate.

    The rule Angel set with the team: a ticket moving out of Backlog must carry
    an estimate, so anything past Backlog with no estimate is a violation.
    """
    out = df.copy()
    if out.empty:
        out["estimate_hours"] = pd.Series(dtype="float64")
        out["policy_applies"] = pd.Series(dtype="bool")
        out["policy_violation"] = pd.Series(dtype="bool")
        return out

    statuses = _normalized(out, "status")
    estimate = _estimate_seconds(out)
    out["estimate_hours"] = (estimate / 3600.0).round(2)
    out["policy_applies"] = ~statuses.isin({s.lower() for s in backlog_statuses})
    out["policy_violation"] = out["policy_applies"] & estimate.le(0)
    return out


def policy_compliance_by_owner(df: pd.DataFrame) -> pd.DataFrame:
    """Per-assignee compliance with the estimate policy, worst offenders first."""
    columns = ["assignee", "past_backlog", "missing_estimate", "compliance_pct", "estimated_hours"]
    if df.empty or "policy_applies" not in df.columns:
        return pd.DataFrame(columns=columns)

    scored = df[df["policy_applies"].fillna(False).astype(bool)].copy()
    if scored.empty:
        return pd.DataFrame(columns=columns)

    scored["assignee"] = scored["assignee"].fillna("Unassigned").astype(str)
    scored["_missing"] = scored["policy_violation"].astype(int)

    grouped = scored.groupby("assignee", dropna=False).agg(
        past_backlog=("key", "count"),
        missing_estimate=("_missing", "sum"),
        estimated_hours=("estimate_hours", "sum"),
    )
    rollup = grouped.reset_index()
    rollup["compliance_pct"] = (
        (rollup["past_backlog"] - rollup["missing_estimate"]) / rollup["past_backlog"] * 100.0
    ).round(0)
    rollup["estimated_hours"] = rollup["estimated_hours"].round(1)
    return rollup.sort_values(
        ["compliance_pct", "missing_estimate"], ascending=[True, False]
    ).reset_index(drop=True)[columns]


def stale_reasons(row: pd.Series) -> str:
    """Why this ticket reads as abandoned rather than merely slow."""
    reasons: list[str] = []
    if str(row.get("assignee") or "").strip().lower() in {"", "unassigned", "none"}:
        reasons.append("unassigned")
    if float(row.get("estimate_hours") or 0) <= 0:
        reasons.append("no estimate")
    if row.get("due_date") is None or pd.isna(row.get("due_date")):
        reasons.append("no due date")
    if str(row.get("status") or "").strip().lower() in NOT_STARTED_STATUSES:
        reasons.append("never started")
    if str(row.get("priority") or "").strip().lower() in {"", "none"}:
        reasons.append("no priority")
    carry_over = float(row.get("carry_over_count") or 0)
    if carry_over >= 3:
        reasons.append(f"carried over {int(carry_over)}x")
    return ", ".join(reasons) if reasons else "aged but tended"


def stale_candidates(
    df: pd.DataFrame,
    idle_days: int = DEFAULT_STALE_DAYS,
    backlog_statuses: set[str] | None = None,
) -> pd.DataFrame:
    """Tickets idle past the threshold, annotated with why they look abandoned.

    Sorted by neglect (most reasons first, then longest idle) so the top of the
    list is the safest to close in bulk.
    """
    scored = estimate_policy(df, backlog_statuses or set())
    if scored.empty:
        scored["stale_reasons"] = pd.Series(dtype="object")
        scored["neglect_signals"] = pd.Series(dtype="int64")
        return scored

    idle = pd.to_numeric(scored.get("idle_days"), errors="coerce").fillna(0.0)
    candidates = scored[idle >= float(idle_days)].copy()
    if candidates.empty:
        candidates["stale_reasons"] = pd.Series(dtype="object")
        candidates["neglect_signals"] = pd.Series(dtype="int64")
        return candidates

    candidates["stale_reasons"] = candidates.apply(stale_reasons, axis=1)
    candidates["neglect_signals"] = (
        candidates["stale_reasons"]
        .where(candidates["stale_reasons"] != "aged but tended", "")
        .map(lambda text: len([part for part in str(text).split(",") if part.strip()]))
    )
    return candidates.sort_values(
        ["neglect_signals", "idle_days"], ascending=[False, False]
    ).reset_index(drop=True)
