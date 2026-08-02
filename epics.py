"""Epic rollups over the tickets the dashboard already fetched.

The JQL loads open work only, so these numbers describe *remaining* work per
epic - never completion percentage, which would need the Done tickets too.
"""

from __future__ import annotations

import pandas as pd


NO_EPIC = "No epic"


def epic_rollup(df: pd.DataFrame) -> pd.DataFrame:
    """Open children per epic, with the signals that mark an epic as drifting."""
    columns = [
        "epic",
        "epic_key",
        "open_children",
        "owners",
        "avg_idle",
        "max_idle",
        "unassigned",
        "no_estimate",
        "estimated_hours",
        "sprints",
    ]
    if df.empty or "epic_key" not in df.columns:
        return pd.DataFrame(columns=columns)

    frame = df.copy()
    # A sub-task belongs to a story that is itself counted here; counting it too
    # would double the epic's open work and swell the orphan pile.
    if "issue_type" in frame.columns:
        frame = frame[
            frame["issue_type"].fillna("").astype(str).str.strip().str.lower().ne("sub-task")
        ]
        if frame.empty:
            return pd.DataFrame(columns=columns)
    frame["epic_key"] = frame["epic_key"].fillna("").astype(str).str.strip()
    frame["epic"] = (
        frame.get("epic_summary", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    frame.loc[frame["epic_key"].eq(""), ["epic_key", "epic"]] = ["", NO_EPIC]
    frame["epic"] = frame["epic"].where(frame["epic"].ne(""), frame["epic_key"])

    frame["_idle"] = pd.to_numeric(frame.get("idle_days"), errors="coerce").fillna(0.0)
    owners = frame.get("assignee", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["_unassigned"] = owners.str.strip().str.lower().isin({"", "unassigned", "none"}).astype(int)
    if "policy_violation" in frame.columns:
        frame["_no_estimate"] = frame["policy_violation"].fillna(False).astype(int)
    else:
        frame["_no_estimate"] = 0
    if "estimate_hours" in frame.columns:
        frame["_hours"] = pd.to_numeric(frame["estimate_hours"], errors="coerce").fillna(0.0)
    else:
        frame["_hours"] = 0.0
    # Tickets outside any sprint must not count as a sprint of their own.
    frame["_sprint"] = (
        frame.get("sprint_name", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
    )

    grouped = frame.groupby(["epic", "epic_key"], dropna=False).agg(
        open_children=("key", "count"),
        owners=("assignee", "nunique"),
        avg_idle=("_idle", "mean"),
        max_idle=("_idle", "max"),
        unassigned=("_unassigned", "sum"),
        no_estimate=("_no_estimate", "sum"),
        estimated_hours=("_hours", "sum"),
        sprints=("_sprint", "nunique"),
    )
    rollup = grouped.reset_index()
    rollup["avg_idle"] = rollup["avg_idle"].round(1)
    rollup["estimated_hours"] = rollup["estimated_hours"].round(1)
    return rollup.sort_values(
        ["open_children", "avg_idle"], ascending=[False, False]
    ).reset_index(drop=True)[columns]


def epic_health_flags(rollup: pd.DataFrame, stale_days: float = 60.0) -> pd.DataFrame:
    """Annotate each epic with what is wrong with it, worst first."""
    if rollup.empty:
        out = rollup.copy()
        out["issues"] = pd.Series(dtype="object")
        out["issue_count"] = pd.Series(dtype="int64")
        return out

    def _issues(row: pd.Series) -> str:
        found: list[str] = []
        if row["epic"] == NO_EPIC:
            return "tickets with no epic"
        if row["avg_idle"] >= stale_days:
            found.append(f"idle {row['avg_idle']:.0f}d on average")
        if row["unassigned"]:
            found.append(f"{int(row['unassigned'])} unassigned")
        if row["no_estimate"]:
            found.append(f"{int(row['no_estimate'])} without estimate")
        if row["sprints"] > 2:
            found.append(f"spread over {int(row['sprints'])} sprints")
        if row["owners"] > 3:
            found.append(f"{int(row['owners'])} owners")
        return ", ".join(found) if found else "healthy"

    out = rollup.copy()
    out["issues"] = out.apply(_issues, axis=1)
    out["issue_count"] = out["issues"].map(
        lambda text: 0 if text == "healthy" else len(str(text).split(","))
    )
    return out.sort_values(
        ["issue_count", "open_children"], ascending=[False, False]
    ).reset_index(drop=True)
