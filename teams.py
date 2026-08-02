"""Team membership derived from Jira projects.

Jira has no notion of "the Marketplace team"; the closest durable signal is the
project a ticket lives in. ``JIRA_TEAM_PROJECTS`` maps one onto the other, e.g.
``"Marketplace=MB,SI;App=AS,OA;Design=MAR"``. Without it every project is its
own team, which is still useful and needs no configuration.
"""

from __future__ import annotations

import pandas as pd


UNASSIGNED_TEAM = "Other"


def parse_team_projects(spec: str) -> dict[str, str]:
    """Project key -> team name, from the ``Team=KEY,KEY;Team=KEY`` spec."""
    mapping: dict[str, str] = {}
    for group in str(spec or "").split(";"):
        team, _, keys = group.partition("=")
        team = team.strip()
        if not team:
            continue
        for key in keys.split(","):
            key = key.strip().upper()
            if key:
                mapping[key] = team
    return mapping


def add_team(df: pd.DataFrame, project_teams: dict[str, str]) -> pd.DataFrame:
    """Attach a ``team`` column; unmapped projects keep their key as the team."""
    out = df.copy()
    if "project_key" not in out.columns:
        out["team"] = UNASSIGNED_TEAM
        return out

    keys = out["project_key"].fillna("").astype(str).str.strip().str.upper()
    if project_teams:
        out["team"] = keys.map(project_teams).fillna(
            keys.where(keys.ne(""), UNASSIGNED_TEAM)
        )
    else:
        out["team"] = keys.where(keys.ne(""), UNASSIGNED_TEAM)
    return out


def team_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-team open counts, idle pressure and estimate coverage."""
    columns = ["team", "open", "people", "avg_idle", "idle_30d", "unassigned", "no_estimate"]
    if df.empty or "team" not in df.columns:
        return pd.DataFrame(columns=columns)

    frame = df.copy()
    frame["_idle"] = pd.to_numeric(frame.get("idle_days"), errors="coerce").fillna(0.0)
    owners = frame.get("assignee", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["_unassigned"] = owners.str.strip().str.lower().isin({"", "unassigned", "none"}).astype(int)
    frame["_idle30"] = (frame["_idle"] >= 30).astype(int)
    if "policy_violation" in frame.columns:
        frame["_no_estimate"] = frame["policy_violation"].fillna(False).astype(int)
    else:
        frame["_no_estimate"] = 0

    grouped = frame.groupby("team", dropna=False).agg(
        open=("key", "count"),
        people=("assignee", "nunique"),
        avg_idle=("_idle", "mean"),
        idle_30d=("_idle30", "sum"),
        unassigned=("_unassigned", "sum"),
        no_estimate=("_no_estimate", "sum"),
    )
    rollup = grouped.reset_index()
    rollup["avg_idle"] = rollup["avg_idle"].round(1)
    return rollup.sort_values("open", ascending=False).reset_index(drop=True)[columns]
