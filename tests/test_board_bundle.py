"""Phase 1 equivalence: the cached board-derivation layer changes nothing.

``_stalled_rows``, ``_stalled_count``, ``_cycle_by_status`` and
``_stale_with_masked`` used to reparse ``df["changelog"]`` on every call. They
now take an optional ``events`` argument and read the board's changelog,
already flattened once by ``_derive_board``, from there instead - but when
``events`` is omitted they still parse ``df["changelog"]`` directly, exactly
as before. That fallback branch *is* the old, uncached path, still live in
this file, which means the new cached-derivation path can be diffed against
it directly in the same test run - no need to resurrect a deleted code path
or freeze a fixture that would drift out from under a wall-clock-sensitive
computation (status age is measured against "now").

Uses the Phase 0 synthetic 1000-ticket board (``tests/apptests/_synthetic_board``)
so the comparison exercises a realistically sized, realistically varied board:
open and closed tickets, tickets with no changelog at all, and tickets with a
trailing cosmetic edit after their last real transition (the ``masked_days``
case).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "apptests"))
from _synthetic_board import build_synthetic_board  # noqa: E402

import app
import integrity


@pytest.fixture(scope="module")
def raw_board() -> pd.DataFrame:
    return build_synthetic_board()


@pytest.fixture(scope="module")
def old_style_df(raw_board: pd.DataFrame) -> pd.DataFrame:
    """What every page's frame looked like before Phase 1: reshaped, changelog intact."""
    return app.add_priority_score(app.add_ticket_health_fields(raw_board))


@pytest.fixture(scope="module")
def bundle(raw_board: pd.DataFrame) -> "app._BoardDerivation":
    fingerprint = app._board_fingerprint(raw_board, "synthetic-jql", 1)
    return app._derive_board(raw_board, fingerprint)


def test_the_bundle_drops_the_changelog_column(bundle) -> None:
    """The whole point of the layer: pages stop carrying raw history."""
    assert "changelog" not in bundle.df.columns


def test_the_bundle_reshapes_the_board_the_same_way_as_before(
    old_style_df: pd.DataFrame, bundle
) -> None:
    """Health fields and priority score must be unaffected by the new layer."""
    shared = [c for c in old_style_df.columns if c != "changelog"]
    pd.testing.assert_frame_equal(
        bundle.df[shared].reset_index(drop=True),
        old_style_df[shared].reset_index(drop=True),
    )


def test_the_bundles_events_match_a_direct_parse_of_the_raw_changelog(
    raw_board: pd.DataFrame, bundle
) -> None:
    direct = integrity.changelog_events(raw_board)
    pd.testing.assert_frame_equal(
        bundle.events.reset_index(drop=True), direct.reset_index(drop=True)
    )


def test_stalled_rows_are_unchanged_by_the_cached_derivation_layer(
    old_style_df: pd.DataFrame, bundle
) -> None:
    old_rows, old_clock = app._stalled_rows(old_style_df)
    new_rows, new_clock = app._stalled_rows(bundle.df, events=bundle.events)
    assert new_clock == old_clock
    shared = [c for c in old_rows.columns if c != "changelog"]
    pd.testing.assert_frame_equal(
        new_rows[shared].reset_index(drop=True),
        old_rows[shared].reset_index(drop=True),
    )


def test_stalled_count_is_unchanged_by_the_cached_derivation_layer(
    old_style_df: pd.DataFrame, bundle
) -> None:
    old_count, old_clock = app._stalled_count(old_style_df)
    new_count, new_clock = app._stalled_count(bundle.df, events=bundle.events)
    assert (new_count, new_clock) == (old_count, old_clock)


def test_cycle_by_status_is_unchanged_by_the_cached_derivation_layer(
    old_style_df: pd.DataFrame, bundle
) -> None:
    old_cycle = app._cycle_by_status(old_style_df)
    new_cycle = app._cycle_by_status(bundle.df, events=bundle.events)
    pd.testing.assert_frame_equal(
        new_cycle.reset_index(drop=True), old_cycle.reset_index(drop=True)
    )


def test_stale_with_masked_is_unchanged_by_the_cached_derivation_layer(
    old_style_df: pd.DataFrame, bundle
) -> None:
    old_stale = app._stale_with_masked(old_style_df)
    new_stale = app._stale_with_masked(bundle.df, events=bundle.events)
    assert new_stale.attrs.get("stale_total") == old_stale.attrs.get("stale_total")
    shared = [c for c in old_stale.columns if c != "changelog"]
    pd.testing.assert_frame_equal(
        new_stale[shared].reset_index(drop=True),
        old_stale[shared].reset_index(drop=True),
    )


def test_the_derivation_is_cached_by_fingerprint_not_by_frame_identity(
    raw_board: pd.DataFrame,
) -> None:
    """A cache hit must not re-walk the changelog.

    Real reruns hand back a different DataFrame object carrying identical
    data (``fetch_tickets`` is itself cached and returns a copy each time);
    the fingerprint - not the frame's identity or content - is what
    ``_derive_board`` is keyed on.
    """
    fingerprint = app._board_fingerprint(raw_board, "synthetic-jql", 1)
    first = app._derive_board(raw_board, fingerprint)
    second_raw = raw_board.copy(deep=True)
    second = app._derive_board(second_raw, fingerprint)
    pd.testing.assert_frame_equal(first.df, second.df)
    pd.testing.assert_frame_equal(first.events, second.events)


def test_the_fingerprint_changes_when_the_board_does(raw_board: pd.DataFrame) -> None:
    """A read that actually changed must not be served a stale bundle."""
    grown = pd.concat([raw_board, raw_board.iloc[[0]]], ignore_index=True)
    same_jql_fp = app._board_fingerprint(raw_board, "synthetic-jql", 1)
    grown_fp = app._board_fingerprint(grown, "synthetic-jql", 1)
    assert same_jql_fp != grown_fp
