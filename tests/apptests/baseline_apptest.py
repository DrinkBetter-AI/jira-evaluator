"""AppTest smoke: baseline wall time for each Engineering-side page.

Phase 0 of the latency/cost plan: before touching the derivation or caching
paths, record how long a cold render of each page actually takes against a
board the size the plan is aimed at (1000 tickets, realistic changelogs). Every
later phase's "faster" claim is checked against the numbers this prints, not
against belief.

    python3 tests/apptests/baseline_apptest.py
"""

from __future__ import annotations

import os
import sys
import time

from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
os.environ["DASHBOARD_REPO"] = REPO

HARNESS = str(Path(__file__).resolve().parent / "_baseline_harness.py")

HARNESS_SOURCE = '''
import datetime as dt
import os
import sys

import pandas as pd

sys.path.insert(0, os.environ["DASHBOARD_REPO"])

import app as dashboard

NOW = pd.Timestamp.now(tz="UTC")

STATUSES = ["To Do", "In Progress", "In Review", "Blocked", "Done"]
PRIORITIES = ["Highest", "High", "Normal", "Low"]
ASSIGNEES = [f"Engineer {n}" for n in range(15)]
N = int(os.environ.get("BASELINE_TICKETS", "1000"))


def _changelog(key: str, created: pd.Timestamp, updated: pd.Timestamp, status: str) -> list[dict]:
    """A handful of real-looking status moves between created and updated.

    Three to five hops through the board, the shape every genuinely worked
    ticket has - not the single hop a minimal fixture would use, which would
    understate exactly the cost (repeated changelog flattening) this baseline
    exists to measure.
    """
    span = max((updated - created).total_seconds(), 3600.0)
    hops = STATUSES[: STATUSES.index(status) + 1] if status in STATUSES else STATUSES[:2]
    events = []
    for index in range(1, len(hops)):
        ts = created + pd.Timedelta(seconds=span * index / max(len(hops), 1))
        events.append(
            {
                "id": f"{key}-h{index}",
                "created": ts.isoformat(),
                "author": {"displayName": ASSIGNEES[index % len(ASSIGNEES)]},
                "items": [
                    {"field": "status", "fromString": hops[index - 1], "toString": hops[index]}
                ],
            }
        )
    return events


def _tickets(*a, **k):
    rows = []
    for i in range(N):
        status = STATUSES[i % len(STATUSES)]
        created = NOW - pd.Timedelta(days=180 - (i % 180))
        updated = NOW - pd.Timedelta(hours=(i % 240))
        key = f"MB-{i}"
        rows.append(
            {
                "key": key,
                "summary": f"Synthetic ticket {i}",
                "description": "x" * 200,
                "status": status,
                "status_category": "Done" if status == "Done" else "In Progress",
                "priority": PRIORITIES[i % len(PRIORITIES)],
                "assignee": ASSIGNEES[i % len(ASSIGNEES)],
                "assignee_account_id": f"a{i % len(ASSIGNEES)}",
                "reporter": ASSIGNEES[(i + 1) % len(ASSIGNEES)],
                "created": created,
                "updated": updated,
                "last_meaningful_activity": updated,
                "due_date": pd.NaT,
                "issue_type": "Task" if i % 3 else "Bug",
                "project_key": "MB",
                "project_name": "Marketplace",
                "parent_key": None,
                "parent_type": None,
                "epic_key": None,
                "epic_summary": None,
                "epic_status": None,
                "labels": "",
                "resolution": None,
                "status_category_changed_date": updated,
                "original_estimate": 3600 * (1 + i % 5),
                "sprint_id": None,
                "sprint_name": None,
                "sprint_state": None,
                "carry_over_count": i % 3,
                "changelog": _changelog(key, created, updated, status),
            }
        )
    return pd.DataFrame(rows)


dashboard.fetch_tickets = _tickets
dashboard.fetch_all_priorities = lambda *a, **k: PRIORITIES
dashboard.fetch_all_users = lambda *a, **k: {}
dashboard.fetch_available_transition_statuses = lambda *a, **k: ["Done"]
# GitHub, Amplitude and the metered clients are not part of what this baseline
# measures - refused rather than left to whatever keys the machine has, same
# as the other smoke tests.
dashboard.github_client.load_github_env = lambda: None
dashboard.amplitude_client.load_amplitude_env = lambda: None
dashboard.cost_client.load_billing_env = lambda: None
dashboard.merchant_client.load_merchant_env = lambda: None

dashboard.inject_styles()
dashboard._reset_reports()

PAGE = os.environ.get("HARNESS_PAGE", "today")
RENDERERS = {
    "today": dashboard._render_today_page,
    "delivery": dashboard._render_delivery_page,
    "code": dashboard._render_code_page,
    "planning": dashboard._render_planning_page,
    "engineering": dashboard._render_engineering_page,
}
RENDERERS[PAGE]()
'''

with open(HARNESS, "w") as handle:
    handle.write(HARNESS_SOURCE)

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ("today", "delivery", "code", "planning", "engineering")

print(f"baseline board: {os.environ.get('BASELINE_TICKETS', '1000')} tickets")
timings: dict[str, float] = {}
for page in PAGES:
    os.environ["HARNESS_PAGE"] = page
    test = AppTest.from_file(HARNESS, default_timeout=300)
    started = time.perf_counter()
    test.run()
    elapsed = time.perf_counter() - started
    assert not test.exception, (page, [e.value for e in test.exception])
    timings[page] = elapsed
    print(f"{page}: {elapsed:.2f}s (cold)")

# The floor later phases are measured against: a synthetic 1000-ticket render
# finishing at all, in a bounded time, is the regression signal - the exact
# seconds will move release to release, and the plan's own logging (Phase 0's
# `_log_stage` lines) is where the per-stage breakdown lives.
for page, elapsed in timings.items():
    assert elapsed < 120, (page, elapsed)

print("baseline: ok")
