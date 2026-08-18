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


def test_deleting_a_snapshot_that_does_not_exist_does_not_raise(snapshot_path) -> None:
    app._delete_board_snapshot()  # nothing to delete - must be a no-op, not an error


def test_delete_board_snapshot_removes_the_file(snapshot_path) -> None:
    _write_valid_snapshot(snapshot_path, time.time())
    assert snapshot_path.exists()
    app._delete_board_snapshot()
    assert not snapshot_path.exists()
