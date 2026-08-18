"""The weekly-delivery panel's read: one search instead of twelve.

The panel used to ask Jira once per week for twelve weeks, each search
carrying its own date window. It now asks once for the whole span and buckets
the answer into weeks locally from the changelog already on it
(``expand=changelog``). ``_weekly_resolved_buckets`` is the split; this pins it
against a changelog rather than trusting it by inspection, and
``fetch_person_resolved_history`` is pinned to prove it is one Jira call, not
twelve.
"""

from __future__ import annotations

import pandas as pd
import pytest

import app


def _ticket(key: str, transitions: list[tuple[pd.Timestamp, str, str]], **fields) -> dict:
    """A ticket whose changelog moves through the given (ts, from, to) states."""
    return {
        "key": key,
        "assignee": fields.get("assignee", "Tam"),
        "status": fields.get("status", "Done"),
        "issue_type": fields.get("issue_type", "Task"),
        "changelog": [
            {
                "created": ts.isoformat(),
                "items": [
                    {
                        "field": fields.get("field", "status"),
                        "fromString": src,
                        "toString": dst,
                    }
                ],
                "author": {"displayName": "Tam"},
            }
            for ts, src, dst in transitions
        ],
    }


def test_empty_history_returns_an_empty_frame_per_week():
    buckets = app._weekly_resolved_buckets(pd.DataFrame(), weeks=12, statuses=("Done",))
    assert set(buckets) == {str(index) for index in range(12)}
    assert all(frame.empty for frame in buckets.values())


def test_a_ticket_resolved_this_week_lands_in_bucket_zero():
    now = pd.Timestamp.now(tz="UTC")
    history = pd.DataFrame(
        [_ticket("MB-1", [(now - pd.Timedelta(days=2), "In Progress", "Done")])]
    )
    buckets = app._weekly_resolved_buckets(history, weeks=12, statuses=("Done",))
    assert list(buckets["0"]["key"]) == ["MB-1"]
    assert buckets["1"].empty


def test_a_ticket_resolved_ten_days_ago_lands_in_bucket_one():
    now = pd.Timestamp.now(tz="UTC")
    history = pd.DataFrame(
        [_ticket("MB-2", [(now - pd.Timedelta(days=10), "In Progress", "Done")])]
    )
    buckets = app._weekly_resolved_buckets(history, weeks=12, statuses=("Done",))
    assert list(buckets["1"]["key"]) == ["MB-2"]
    assert buckets["0"].empty


def test_re_entering_a_resolved_status_in_two_weeks_appears_in_both():
    """A ticket the old per-week searches would have found twice, once each week."""
    now = pd.Timestamp.now(tz="UTC")
    history = pd.DataFrame(
        [
            _ticket(
                "MB-3",
                [
                    (now - pd.Timedelta(days=1), "In Progress", "Ready for Production"),
                    (now - pd.Timedelta(days=22), "In Progress", "Ready for Production"),
                ],
            )
        ]
    )
    buckets = app._weekly_resolved_buckets(
        history, weeks=12, statuses=("Ready for Production",)
    )
    assert "MB-3" in set(buckets["0"]["key"])
    assert "MB-3" in set(buckets["3"]["key"])


def test_a_move_between_two_resolved_statuses_still_counts_as_an_entry():
    """Mirrors the JQL: CHANGED TO matches on the new status alone."""
    now = pd.Timestamp.now(tz="UTC")
    history = pd.DataFrame(
        [_ticket("MB-4", [(now - pd.Timedelta(days=3), "Ready for Production", "Done")])]
    )
    buckets = app._weekly_resolved_buckets(
        history, weeks=12, statuses=("Ready for Production", "Done")
    )
    assert list(buckets["0"]["key"]) == ["MB-4"]


def test_a_non_status_field_change_is_not_a_resolution():
    now = pd.Timestamp.now(tz="UTC")
    history = pd.DataFrame(
        [
            _ticket(
                "MB-5",
                [(now - pd.Timedelta(days=1), "Low", "High")],
                field="priority",
            )
        ]
    )
    buckets = app._weekly_resolved_buckets(history, weeks=12, statuses=("Done",))
    assert all(frame.empty for frame in buckets.values())


def test_a_transition_older_than_the_window_is_dropped():
    now = pd.Timestamp.now(tz="UTC")
    history = pd.DataFrame(
        [_ticket("MB-6", [(now - pd.Timedelta(days=200), "In Progress", "Done")])]
    )
    buckets = app._weekly_resolved_buckets(history, weeks=12, statuses=("Done",))
    assert all(frame.empty for frame in buckets.values())


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search_issues(self, **kwargs):
        self.calls.append(kwargs)
        return pd.DataFrame()


def test_the_weekly_history_read_is_one_jira_search_not_twelve(monkeypatch):
    """The old panel looped ``fetch_person_resolved_week`` once per week; the
    replacement makes exactly one call to Jira for the whole span."""
    fake = _FakeClient()
    monkeypatch.setattr(app.JiraClient, "resolve", classmethod(lambda cls, **kw: fake))
    app.fetch_person_resolved_history.clear()

    app.fetch_person_resolved_history(
        creds_path="creds.yml",
        profile_name="profile",
        person="Tam",
        weeks=12,
        statuses=("Done", "Released"),
        max_results=100,
        page_size=50,
        schema_version=1,
    )

    assert len(fake.calls) == 1, fake.calls
    call = fake.calls[0]
    assert call["expand"] == "changelog"
    assert "AFTER -84d" in call["jql"], call["jql"]
    assert "description" not in call["fields"], call["fields"]


def test_an_unknown_person_or_status_never_reaches_jira(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(app.JiraClient, "resolve", classmethod(lambda cls, **kw: fake))
    app.fetch_person_resolved_history.clear()

    out = app.fetch_person_resolved_history(
        creds_path="creds.yml",
        profile_name="profile",
        person="",
        weeks=12,
        statuses=(),
        max_results=100,
        page_size=50,
        schema_version=1,
    )
    assert out.empty
    assert fake.calls == []
