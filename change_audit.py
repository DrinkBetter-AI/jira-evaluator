from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Containers have an ephemeral filesystem, so point this at a mounted volume to
# keep the revert history across restarts.
LOG_FILE = Path(
    os.getenv(
        "JIRA_AUDIT_LOG_PATH",
        str(Path(__file__).resolve().parent / "logs" / "jira_ticket_changes.jsonl"),
    )
).expanduser()
LOG_DIR = LOG_FILE.parent


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def new_operation_record(
    action_type: str,
    target: str,
    selected_keys: list[str],
    source_status: str | None = None,
    parent_operation_id: str | None = None,
) -> dict[str, Any]:
    return {
        "operation_id": str(uuid.uuid4()),
        "parent_operation_id": parent_operation_id,
        "created_at": _now_iso(),
        "action_type": action_type,
        "target": target,
        "source_status": source_status,
        "selected_keys": selected_keys,
        "items": [],
        "success_count": 0,
        "failure_count": 0,
    }


def append_operation(record: dict[str, Any]) -> None:
    _ensure_log_dir()
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_operations(limit: int = 50) -> list[dict[str, Any]]:
    if not LOG_FILE.exists():
        return []

    ops: list[dict[str, Any]] = []
    with LOG_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ops.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if limit <= 0:
        return list(reversed(ops))
    return list(reversed(ops[-limit:]))


def finalize_operation(record: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    success_count = sum(1 for item in items if item.get("success"))
    failure_count = len(items) - success_count
    record["items"] = items
    record["success_count"] = success_count
    record["failure_count"] = failure_count
    record["completed_at"] = _now_iso()
    return record


def summarize_operations(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for op in operations:
        rows.append(
            {
                "operation_id": op.get("operation_id"),
                "created_at": op.get("created_at"),
                "action_type": op.get("action_type"),
                "target": op.get("target"),
                "source_status": op.get("source_status"),
                "tickets": len(op.get("selected_keys", [])),
                "success": op.get("success_count", 0),
                "failed": op.get("failure_count", 0),
                "parent_operation_id": op.get("parent_operation_id"),
            }
        )
    return rows
