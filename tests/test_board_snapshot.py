"""Phase 3: the on-disk board snapshot fails open.

_read_board_snapshot() is the one thing a cold start's fast path depends on
being trustworthy - not "correct", trustworthy: a bad snapshot must never be
the reason a page raises. A missing file, a snapshot older than the
staleness limit, and a corrupt or otherwise unreadable file all have to take
the same route back to the caller (None, meaning "read live") with nothing
thrown - table-driven here so the three cases stay visibly the same shape.
"""

from __future__ import annotations

import pickle
import time

import pandas as pd
import pytest

import app
import data_layer


@pytest.fixture(autouse=True)
def snapshot_path(tmp_path, monkeypatch):
    """Every test gets its own snapshot file, never the real container path."""
    path = tmp_path / "board_snapshot.pkl"
    monkeypatch.setattr(data_layer, "_SNAPSHOT_PATH", path)
    return path


def _write_valid_snapshot(path, written_at: float) -> None:
    persisted = app._PersistedBoard(
        written_at=written_at,
        df=pd.DataFrame({"key": ["ENG-1"], "status": ["In Progress"]}),
        events=pd.DataFrame(),
        data={},
        github_ready=False,
        github_error="",
        open_prs=pd.DataFrame(),
        merged_prs=pd.DataFrame(),
        pr_count_7=None,
        pr_count_30=None,
        open_count_exact=None,
        assignees=[],
        statuses=[],
        priorities=[],
        max_results=1000,
        page_size=250,
    )
    with open(path, "wb") as fh:
        pickle.dump(persisted, fh)


def _leave_missing(path) -> None:
    pass  # the fixture never created the file - this is the "missing" case


def _write_stale(path) -> None:
    _write_valid_snapshot(
        path, time.time() - app._SNAPSHOT_STALE_LIMIT_SECONDS - 60.0
    )


def _write_corrupt(path) -> None:
    path.write_bytes(b"not a pickle at all, just noise\x00\x01\x02")


@pytest.mark.parametrize(
    "break_it",
    [_leave_missing, _write_stale, _write_corrupt],
    ids=["missing_file", "stale_beyond_limit", "corrupt_file"],
)
def test_a_bad_snapshot_falls_back_to_the_live_path_without_raising(
    snapshot_path, break_it
) -> None:
    break_it(snapshot_path)
    # The whole point: this must not raise, regardless of which of the three
    # ways the snapshot is bad. A caller that gets None takes the live path -
    # exactly what it would do with no snapshot mechanism at all.
    result = app._read_board_snapshot()
    assert result is None


def test_a_fresh_readable_snapshot_is_used(snapshot_path) -> None:
    """The fail-open path must not also swallow a perfectly good snapshot."""
    written_at = time.time() - 120.0  # two minutes old: well inside the limit
    _write_valid_snapshot(snapshot_path, written_at)
    result = app._read_board_snapshot()
    assert result is not None
    bundle, returned_written_at = result
    assert returned_written_at == written_at
    assert list(bundle.df["key"]) == ["ENG-1"]
    # The one column deliberately never persisted.
    assert "changelog" not in bundle.df.columns


def test_a_snapshot_exactly_at_the_boundary_is_still_used(snapshot_path) -> None:
    written_at = time.time() - app._SNAPSHOT_STALE_LIMIT_SECONDS + 5.0
    _write_valid_snapshot(snapshot_path, written_at)
    assert app._read_board_snapshot() is not None


def test_write_then_read_round_trips(snapshot_path) -> None:
    """A snapshot this process just wrote is exactly what it reads back."""
    bundle = app._EngineeringData(
        data={"resolved_count_7": 3},
        errors={},
        raw_df=pd.DataFrame({"key": ["ENG-9"]}),
        df=pd.DataFrame({"key": ["ENG-9"], "assignee": ["Tam"], "status": ["Done"]}),
        events=pd.DataFrame(),
        github_ready=True,
        github_error="",
        open_prs=pd.DataFrame(),
        merged_prs=pd.DataFrame(),
        pr_count_7=5,
        pr_count_30=20,
        open_count_exact=7,
        assignees=["Tam"],
        statuses=["Done"],
        priorities=[],
        max_results=1000,
        page_size=250,
    )
    app._write_board_snapshot(bundle)
    result = app._read_board_snapshot()
    assert result is not None
    read_back, _written_at = result
    pd.testing.assert_frame_equal(
        read_back.df.reset_index(drop=True), bundle.df.reset_index(drop=True)
    )
    assert read_back.assignees == ["Tam"]
    assert read_back.pr_count_7 == 5


def test_a_cold_session_serves_the_snapshot_without_reading_tickets_first(
    snapshot_path, monkeypatch
) -> None:
    """The snapshot's whole purpose is skipping the wait, so it must skip it.

    ``_engineering_data`` also probes ``fetch_tickets`` to decide whether a
    bundle the session is already holding is still current. That probe once ran
    ahead of everything, including this branch - so the one reader the snapshot
    exists to spare (a cold session, holding nothing) paid a full paginated
    ticket fetch before the snapshot was even opened. The probe is only
    meaningful when there is a held bundle to compare against; with nothing
    held there is nothing to compare, and reading tickets here buys nothing but
    seconds of blank page.
    """
    _write_valid_snapshot(snapshot_path, time.time() - 120.0)
    monkeypatch.setattr(data_layer.st, "session_state", {})  # cold: nothing held

    # Recorded rather than raised from: the probe's caller catches Exception,
    # so an AssertionError thrown here would be swallowed and the test would
    # pass against the very ordering it exists to forbid.
    probes: list[int] = []

    def _probe(**_kwargs):
        probes.append(1)
        return _write_valid_snapshot  # never reached in a passing run

    monkeypatch.setattr(data_layer, "fetch_tickets", _probe)
    # The refresh is fire-and-forget and would otherwise read live behind us.
    refreshes: list[tuple[int, int]] = []
    monkeypatch.setattr(
        data_layer,
        "_start_background_board_refresh",
        lambda max_results, page_size: refreshes.append((max_results, page_size)),
    )

    bundle = data_layer._engineering_data()

    assert probes == []
    assert list(bundle.df["key"]) == ["ENG-1"]
    # Served stale on purpose - so the refresh behind the reader is the other
    # half of the bargain, not an optional extra.
    assert len(refreshes) == 1


def test_a_session_holding_a_bundle_still_probes_before_trusting_it(
    snapshot_path, monkeypatch
) -> None:
    """The probe is confined to its case, not deleted from it.

    A held bundle is only reusable if the board has not moved underneath it,
    and the probe's fingerprint is the only thing that knows. This is the path
    where ``fetch_tickets`` is warm and answers in milliseconds.
    """
    held_bundle = app._EngineeringData(
        data={},
        errors={},
        raw_df=pd.DataFrame({"key": ["ENG-1"], "status": ["In Progress"]}),
        df=pd.DataFrame({"key": ["ENG-1"], "status": ["In Progress"]}),
        events=pd.DataFrame(),
        github_ready=False,
        github_error="",
        open_prs=pd.DataFrame(),
        merged_prs=pd.DataFrame(),
        pr_count_7=None,
        pr_count_30=None,
        open_count_exact=None,
        assignees=[],
        statuses=[],
        priorities=[],
        max_results=1000,
        page_size=250,
    )
    fingerprint = data_layer._board_fingerprint(
        held_bundle.raw_df, data_layer.JQL, data_layer.FETCH_SCHEMA_VERSION
    )
    monkeypatch.setattr(
        data_layer.st,
        "session_state",
        {data_layer._ENGINEERING_BUNDLE_KEY: (fingerprint, time.time(), held_bundle)},
    )

    calls: list[int] = []

    def _probe(**_kwargs):
        calls.append(1)
        return held_bundle.raw_df

    monkeypatch.setattr(data_layer, "fetch_tickets", _probe)

    assert data_layer._engineering_data() is held_bundle
    assert calls == [1]


def test_the_store_is_local_by_default_and_gcs_when_a_bucket_is_set(monkeypatch) -> None:
    monkeypatch.delenv("JIRA_DASHBOARD_SNAPSHOT_BUCKET", raising=False)
    assert isinstance(data_layer._snapshot_store(), data_layer._LocalSnapshotStore)

    monkeypatch.setenv("JIRA_DASHBOARD_SNAPSHOT_BUCKET", "vinovoss-board-cache")
    monkeypatch.setenv("JIRA_DASHBOARD_SNAPSHOT_OBJECT", "custom/name.pkl")
    store = data_layer._snapshot_store()
    assert isinstance(store, data_layer._GcsSnapshotStore)
    assert store._bucket_name == "vinovoss-board-cache"
    assert store._blob_name == "custom/name.pkl"


def test_gcs_store_round_trips_through_a_fake_bucket(monkeypatch) -> None:
    """The GCS path writes whole objects and reads a missing one back as None."""

    class _FakeBlob:
        def __init__(self, store: dict, name: str) -> None:
            self._store = store
            self._name = name

        def upload_from_string(self, data: bytes, content_type: str = "") -> None:
            self._store[self._name] = data

        def download_as_bytes(self) -> bytes:
            from google.cloud.exceptions import NotFound

            if self._name not in self._store:
                raise NotFound(self._name)
            return self._store[self._name]

        def delete(self) -> None:
            from google.cloud.exceptions import NotFound

            if self._name not in self._store:
                raise NotFound(self._name)
            del self._store[self._name]

    objects: dict[str, bytes] = {}

    class _FakeBucket:
        def blob(self, name: str) -> _FakeBlob:
            return _FakeBlob(objects, name)

    class _FakeClient:
        def bucket(self, name: str) -> _FakeBucket:
            return _FakeBucket()

    monkeypatch.setattr(data_layer, "_gcs_client", lambda: _FakeClient())
    store = data_layer._GcsSnapshotStore("bucket", "board.pkl")

    assert store.read() is None  # nothing written yet
    store.write(b"payload-bytes")
    assert store.read() == b"payload-bytes"
    store.delete()
    assert store.read() is None
    store.delete()  # deleting a missing object is a no-op, not an error


def test_deleting_a_snapshot_that_does_not_exist_does_not_raise(snapshot_path) -> None:
    app._delete_board_snapshot()  # nothing to delete - must be a no-op, not an error


def test_delete_board_snapshot_removes_the_file(snapshot_path) -> None:
    _write_valid_snapshot(snapshot_path, time.time())
    assert snapshot_path.exists()
    app._delete_board_snapshot()
    assert not snapshot_path.exists()
