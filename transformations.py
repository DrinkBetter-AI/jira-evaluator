from __future__ import annotations

from typing import Iterable

import pandas as pd


DEFAULT_ACTIVE_STATUSES = {
    "in progress",
    "code review",
    "review in staging",
    "discussion needed",
}

PRIORITY_WEIGHTS = {
    "highest": 10,
    "urgent": 10,
    "high": 7,
    "normal": 4,
    "medium": 4,
    "low": 1,
    "lowest": 1,
    "none": 0,
    "": 0,
}


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_lower_set(values: Iterable[str]) -> set[str]:
    return {v.strip().lower() for v in values if str(v).strip()}


def _workflow_stage(row: pd.Series) -> str:
    status_category = _normalize_text(row.get("status_category"))
    if status_category:
        return status_category

    status = _normalize_text(row.get("status")).lower()
    if status in {"done", "closed", "resolved"}:
        return "Done"
    if status in {"in progress", "code review", "review in staging", "discussion needed", "review"}:
        return "In Progress"
    return "To Do"


def add_ticket_health_fields(
    df: pd.DataFrame,
    zombie_idle_threshold: int = 3,
    active_statuses: Iterable[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["ticket_age_days"] = pd.Series(dtype="int64")
        out["idle_days"] = pd.Series(dtype="int64")
        out["is_zombie"] = pd.Series(dtype="bool")
        out["idle_bucket"] = pd.Series(dtype="object")
        out["age_bucket"] = pd.Series(dtype="object")
        out["workflow_stage"] = pd.Series(dtype="object")
        out["priority_weight"] = pd.Series(dtype="float64")
        out["risk_score"] = pd.Series(dtype="float64")
        return out

    out = df.copy()

    out["created"] = _to_utc(out.get("created"))
    out["updated"] = _to_utc(out.get("updated"))
    out["status_category_changed_date"] = _to_utc(out.get("status_category_changed_date"))
    out["due_date"] = _to_utc(out.get("due_date"))

    now = pd.Timestamp.now(tz="UTC")
    out["ticket_age_days"] = (now - out["created"]).dt.total_seconds().div(86400).clip(lower=0)
    out["idle_days"] = (now - out["updated"]).dt.total_seconds().div(86400).clip(lower=0)

    out["ticket_age_days"] = out["ticket_age_days"].fillna(0).round(1)
    out["idle_days"] = out["idle_days"].fillna(0).round(1)

    out["idle_bucket"] = pd.cut(
        out["idle_days"],
        bins=[-0.001, 2, 7, 14, float("inf")],
        labels=["0-2 days", "3-7 days", "8-14 days", "15+ days"],
    ).astype("object")

    out["age_bucket"] = pd.cut(
        out["ticket_age_days"],
        bins=[-0.001, 7, 30, 90, 180, float("inf")],
        labels=["0-7 days", "8-30 days", "31-90 days", "91-180 days", "180+ days"],
    ).astype("object")

    out["workflow_stage"] = out.apply(_workflow_stage, axis=1)

    out["priority_normalized"] = out["priority"].fillna("none").astype(str).str.strip().str.lower()
    out["priority_weight"] = out["priority_normalized"].map(PRIORITY_WEIGHTS).fillna(0)

    active_status_set = _to_lower_set(active_statuses or DEFAULT_ACTIVE_STATUSES)
    status_lower = out["status"].fillna("").astype(str).str.strip().str.lower()

    out["is_zombie"] = (status_lower.isin(active_status_set)) & (
        out["idle_days"] >= float(zombie_idle_threshold)
    )

    active_status_penalty = status_lower.isin(active_status_set).astype(float) * 10.0
    out["risk_score"] = (
        out["idle_days"] * 2.0
        + out["ticket_age_days"] * 0.5
        + out["priority_weight"]
        + active_status_penalty
    ).round(2)

    return out
