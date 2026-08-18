"""Synthetic Jira board: a deterministic 1000-ticket fixture with real changelogs.

Built for the latency/cost optimization track (see baseline_apptest.py) so the
derivation work in later phases - the cached board bundle, fragment-wrapped
sections, the on-disk snapshot - has a fixture big enough to show its cost and
stable enough to diff against. Every column a downstream reader touches
(status, changelog, sprint window, estimate) is populated; nothing here is a
placeholder that only looks like a ticket.

Kept import-free of ``app``/Streamlit on purpose: pure pandas and stdlib, so
both the AppTest harness (which imports ``app``) and plain unit tests (which
diff derivation functions directly) can build the same board without paying
for a Streamlit runtime just to get a DataFrame.
"""

from __future__ import annotations

import datetime as dt
import random

import pandas as pd

# (status, statusCategory) - a mix of open and closed stages, weighted so most
# tickets are open (the common case for a live board's org-wide JQL).
_STATUS_CHOICES: list[tuple[str, str]] = [
    ("Backlog", "To Do"),
    ("Backlog", "To Do"),
    ("To Do", "To Do"),
    ("To Do", "To Do"),
    ("In Progress", "In Progress"),
    ("In Progress", "In Progress"),
    ("Code Review", "In Progress"),
    ("Review in Staging", "In Progress"),
    ("Discussion Needed", "In Progress"),
    ("Done", "Done"),
    ("Closed", "Done"),
]
_ASSIGNEES = [f"Engineer {i}" for i in range(1, 13)]
_PRIORITIES = ["Highest", "High", "Medium", "Low"]
_PROJECTS = ["ENG", "PLAT", "DATA"]
_ISSUE_TYPES = ["Story", "Bug", "Task"]

DEFAULT_SEED = 20260818
DEFAULT_SIZE = 1000


def _iso(moment: dt.datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000+0000")


def build_synthetic_board(
    n: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
    *,
    now: dt.datetime | None = None,
) -> pd.DataFrame:
    """A DataFrame shaped like ``JiraClient._issues_to_dataframe``'s output.

    Deterministic given the same ``n``/``seed``/``now``: two calls produce
    byte-identical frames, which is what makes it usable as a before/after
    fixture for the derivation-layer equivalence test.
    """
    rng = random.Random(seed)
    moment = now or dt.datetime(2026, 8, 18, 12, 0, 0, tzinfo=dt.timezone.utc)
    rows: list[dict[str, object]] = []

    for i in range(1, n + 1):
        key = f"ENG-{i}"
        project = rng.choice(_PROJECTS)
        status, category = rng.choice(_STATUS_CHOICES)
        assignee = rng.choice(_ASSIGNEES) if rng.random() > 0.04 else "Unassigned"
        priority = rng.choice(_PRIORITIES)
        age_days = rng.uniform(1, 400)
        created = moment - dt.timedelta(days=age_days)

        # A handful of real status transitions, each a separate changelog
        # entry - the shape ``changelog_events`` and ``status_age_days`` read.
        histories: list[dict[str, object]] = []
        cursor = created
        prior_status = "To Do"
        n_transitions = rng.randint(0, 5)
        step = age_days / max(n_transitions, 1)
        for t in range(n_transitions):
            cursor = cursor + dt.timedelta(days=rng.uniform(0.2, step + 1))
            if cursor >= moment:
                break
            next_status, _next_category = rng.choice(_STATUS_CHOICES)
            histories.append(
                {
                    "id": f"{i}-{t}",
                    "created": _iso(cursor),
                    "author": {
                        "displayName": assignee if assignee != "Unassigned" else "Bot"
                    },
                    "items": [
                        {
                            "field": "status",
                            "fieldId": "status",
                            "fromString": prior_status,
                            "toString": next_status,
                        }
                    ],
                }
            )
            prior_status = next_status

        # 15% of tickets that moved at least once get a trailing cosmetic edit
        # after their last real transition - the label-sweep shape that
        # ``masked_days`` exists to catch (apparent freshness, no real work).
        if histories and rng.random() < 0.15:
            cosmetic_at = cursor + dt.timedelta(days=rng.uniform(1, 5))
            if cosmetic_at >= moment:
                cosmetic_at = moment - dt.timedelta(hours=1)
            histories.append(
                {
                    "id": f"{i}-cosmetic",
                    "created": _iso(cosmetic_at),
                    "author": {"displayName": "Bot"},
                    "items": [
                        {
                            "field": "labels",
                            "fieldId": "labels",
                            "fromString": "",
                            "toString": "triaged",
                        }
                    ],
                }
            )

        # 5% carry no changelog at all - the edit-age fallback path every
        # reader downstream (a KPI tile, a stale queue) has to survive.
        changelog = None if rng.random() < 0.05 else {"histories": histories}

        last_meaningful = histories[-1]["created"] if histories else None
        updated = last_meaningful or _iso(created)

        sprint_active = rng.random() < 0.5
        sprint_start = moment - dt.timedelta(days=7)
        sprint_end = moment + dt.timedelta(days=7)

        rows.append(
            {
                "key": key,
                "summary": f"Synthetic ticket {i}",
                "description": "",
                "status": status,
                "status_category": category,
                "priority": priority,
                "assignee": assignee,
                "assignee_account_id": None if assignee == "Unassigned" else f"acc-{assignee}",
                "reporter": "Reporter Bot",
                "created": _iso(created),
                "updated": updated,
                "last_meaningful_activity": last_meaningful,
                "changelog": changelog,
                "due_date": None,
                "issue_type": rng.choice(_ISSUE_TYPES),
                "project_key": project,
                "project_name": project,
                "parent_key": None,
                "parent_type": None,
                "epic_key": f"{project}-{(i % 20) + 1}" if rng.random() < 0.6 else None,
                "epic_summary": "Epic work" if rng.random() < 0.6 else None,
                "epic_status": None,
                "labels": "",
                "resolution": "Done" if category == "Done" else None,
                "status_category_changed_date": updated,
                "original_estimate": None,
                "logged_time": None,
                "completion_pct": None,
                "original_estimate_sec": rng.choice([0, 3600, 7200, 14400])
                if rng.random() < 0.7
                else 0,
                "time_spent_sec": 0,
                "sprint_id": (i % 30) if sprint_active else None,
                "sprint_name": f"Sprint {i % 30}" if sprint_active else None,
                "sprint_state": "active" if sprint_active else None,
                "sprint_board_id": 1 if sprint_active else None,
                "sprint_start": _iso(sprint_start) if sprint_active else None,
                "sprint_end": _iso(sprint_end) if sprint_active else None,
                "carry_over_count": 0,
                "ticket_url": f"https://example.atlassian.net/browse/{key}",
            }
        )

    return pd.DataFrame(rows)
