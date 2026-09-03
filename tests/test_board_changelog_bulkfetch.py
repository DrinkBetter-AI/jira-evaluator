"""The board's changelog is read in bulk, not expanded on the search.

``fetch_tickets`` stopped asking for ``expand="changelog"`` because the new
``/search/jql`` endpoint serves it slowly and in small pages. The history now
comes from ``JiraClient.get_changelogs_bulk`` (POST /changelog/bulkfetch) and
is stitched back onto the board frame by ``data_layer._attach_board_changelogs``
before the board is derived - so everything downstream still reads a
``changelog`` column and never knows the difference.
"""

from __future__ import annotations

import contextlib

import pandas as pd
import pytest

import data_layer
import jira_client


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, pages: list[dict]) -> None:
        self._pages = list(pages)
        self.calls: list[dict] = []

    def post(self, url, json=None, timeout=None):  # noqa: A002 - mirrors requests
        self.calls.append(json)
        return _FakeResponse(self._pages.pop(0))


def _client_with(session: _FakeSession) -> jira_client.JiraClient:
    client = jira_client.JiraClient(
        base_url="https://example.atlassian.net", email="e@x", api_token="t"
    )

    @contextlib.contextmanager
    def _lend():
        yield session

    client._session = _lend  # type: ignore[method-assign]
    return client


def test_get_changelogs_bulk_keys_by_issue_id_and_follows_pages() -> None:
    session = _FakeSession(
        [
            {
                "issueChangeLogs": [
                    {"issueId": "1001", "changeHistories": [{"id": "a"}]},
                    {"issueId": "1002", "changeHistories": [{"id": "b"}]},
                ],
                "nextPageToken": "PAGE2",
            },
            {
                "issueChangeLogs": [
                    {"issueId": "1001", "changeHistories": [{"id": "c"}]},
                ],
            },
        ]
    )
    client = _client_with(session)

    out = client.get_changelogs_bulk(["1001", "1002", " ", "1001"])

    assert out == {
        "1001": [{"id": "a"}, {"id": "c"}],
        "1002": [{"id": "b"}],
    }
    # De-duplicated and sorted before the request, second call carried the token.
    assert session.calls[0]["issueIdsOrKeys"] == ["1001", "1002"]
    assert session.calls[1]["nextPageToken"] == "PAGE2"


def test_get_changelogs_bulk_is_empty_for_no_ids() -> None:
    session = _FakeSession([])
    assert _client_with(session).get_changelogs_bulk([]) == {}
    assert session.calls == []


def test_get_changelogs_bulk_raises_on_an_http_error() -> None:
    session = _FakeSession([])
    session.post = lambda *a, **k: _FakeResponse({}, status_code=500)  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="bulkfetch failed"):
        _client_with(session).get_changelogs_bulk(["1"])


def test_attach_leaves_a_frame_that_already_carries_a_changelog_alone() -> None:
    df = pd.DataFrame(
        {"key": ["ENG-1"], "changelog": [{"histories": [{"id": "x"}]}]}
    )
    out = data_layer._attach_board_changelogs(df)
    assert out.loc[0, "changelog"] == {"histories": [{"id": "x"}]}


def test_attach_treats_an_all_null_changelog_column_as_absent(monkeypatch) -> None:
    """_issues_to_dataframe emits an all-null changelog column when nothing was
    expanded - it must not be mistaken for a changelog already carried."""
    df = pd.DataFrame(
        {"key": ["ENG-1"], "id": ["1001"], "changelog": [None], "updated": ["2026-01-01"]}
    )
    monkeypatch.setattr(
        data_layer,
        "fetch_ticket_changelogs",
        lambda *a, **k: {"1001": [{"id": "h1"}]},
    )
    out = data_layer._attach_board_changelogs(df)
    assert out.loc[0, "changelog"] == {"histories": [{"id": "h1"}]}


def test_attach_gives_a_frame_with_no_id_an_empty_changelog_column() -> None:
    df = pd.DataFrame({"key": ["ENG-1"], "updated": ["2026-01-01"]})
    out = data_layer._attach_board_changelogs(df)
    assert "changelog" in out.columns
    assert out.loc[0, "changelog"] is None


def test_attach_reads_histories_in_bulk_and_rebuilds_last_meaningful_activity(
    monkeypatch,
) -> None:
    df = pd.DataFrame(
        {
            "key": ["ENG-1", "ENG-2"],
            "id": ["1001", "1002"],
            "updated": ["2026-01-01", "2026-01-01"],
            "last_meaningful_activity": [None, None],
        }
    )
    histories = {
        "1001": [
            {
                "created": "2026-02-01T00:00:00.000+0000",
                "items": [{"field": "status", "toString": "In Progress"}],
            }
        ],
    }
    seen: list[tuple] = []

    def _fake(creds_path, profile_name, issue_ids, schema_version):
        seen.append(issue_ids)
        return histories

    monkeypatch.setattr(data_layer, "fetch_ticket_changelogs", _fake)

    out = data_layer._attach_board_changelogs(df)

    assert seen == [("1001", "1002")]  # sorted tuple, the cache key
    assert out.loc[0, "changelog"] == {"histories": histories["1001"]}
    assert out.loc[1, "changelog"] == {"histories": []}
    # Recomputed from the freshly attached history, not left at None.
    assert str(out.loc[0, "last_meaningful_activity"]).startswith("2026-02-01")
    assert out.loc[1, "last_meaningful_activity"] is None


def test_attach_falls_back_to_no_history_when_the_bulk_read_fails(monkeypatch) -> None:
    df = pd.DataFrame({"key": ["ENG-1"], "id": ["1001"], "updated": ["2026-01-01"]})

    def _boom(*a, **k):
        raise RuntimeError("Jira down")

    monkeypatch.setattr(data_layer, "fetch_ticket_changelogs", _boom)

    out = data_layer._attach_board_changelogs(df)
    assert out.loc[0, "changelog"] == {"histories": []}
