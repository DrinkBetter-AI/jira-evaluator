from __future__ import annotations

import pandas as pd


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


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
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["ticket_age_days"] = pd.Series(dtype="int64")
        out["idle_days"] = pd.Series(dtype="int64")
        out["idle_bucket"] = pd.Series(dtype="object")
        out["age_bucket"] = pd.Series(dtype="object")
        out["workflow_stage"] = pd.Series(dtype="object")
        return out

    out = df.copy()

    out["created"] = _to_utc(out.get("created"))
    out["updated"] = _to_utc(out.get("updated"))
    if "last_meaningful_activity" in out.columns:
        out["last_meaningful_activity"] = _to_utc(out.get("last_meaningful_activity"))
    else:
        out["last_meaningful_activity"] = pd.NaT
    out["status_category_changed_date"] = _to_utc(out.get("status_category_changed_date"))
    out["due_date"] = _to_utc(out.get("due_date"))

    activity_anchor = out["last_meaningful_activity"].fillna(out["updated"])

    now = pd.Timestamp.now(tz="UTC")
    out["ticket_age_days"] = (now - out["created"]).dt.total_seconds().div(86400).clip(lower=0)
    out["idle_days"] = (now - activity_anchor).dt.total_seconds().div(86400).clip(lower=0)

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

    return out
