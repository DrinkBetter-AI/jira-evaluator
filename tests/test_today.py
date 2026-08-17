"""The Today page's signals - the numbers the landing page opens with.

Each one is a count a reader will act on, so each one is pinned here against the
way it could quietly go wrong: drafts counted as unreviewed, a label edit
emptying the stalled queue, backlog tickets dragging estimate coverage down.
"""

from __future__ import annotations

import types
import pandas as pd
import pytest

import app
import hygiene
import theme
import next_actions


def _prs(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_a_draft_is_not_an_unreviewed_pr():
    """A draft says "not ready for you yet", so nobody is late reviewing it."""
    prs = _prs(
        is_draft=[True, False],
        approving_reviews=[0, 0],
        total_reviews=[0, 0],
        review_requests=[0, 0],
        age_days=[40.0, 40.0],
    )
    signals = app._open_pr_signals(prs, None)
    assert signals["unapproved"] == 1
    assert signals["never_reviewed"] == 1


def test_an_approved_pr_is_not_counted_as_needing_a_decision():
    prs = _prs(
        is_draft=[False, False],
        approving_reviews=[1, 0],
        total_reviews=[2, 0],
        review_requests=[1, 0],
        age_days=[3.0, 9.0],
    )
    signals = app._open_pr_signals(prs, None)
    assert signals["unapproved"] == 1
    assert signals["oldest_unreviewed_days"] == 9.0


def test_a_reviewed_but_unapproved_pr_still_had_someone_look():
    """`never_reviewed` is the harsher count: a rejecting review is attention."""
    prs = _prs(
        is_draft=[False],
        approving_reviews=[0],
        total_reviews=[3],
        review_requests=[1],
        age_days=[5.0],
    )
    signals = app._open_pr_signals(prs, None)
    assert signals["unapproved"] == 1
    assert signals["never_reviewed"] == 0
    assert signals["no_reviewer_asked"] == 0


def test_nobody_asked_needs_both_no_request_and_no_review_and_some_age():
    prs = _prs(
        is_draft=[False, False, False],
        approving_reviews=[0, 0, 0],
        total_reviews=[0, 0, 0],
        review_requests=[0, 1, 0],
        # Under the two-day floor: opened this morning is not a review failure.
        age_days=[9.0, 9.0, 0.5],
    )
    assert app._open_pr_signals(prs, None)["no_reviewer_asked"] == 1


def test_the_exact_open_count_wins_over_the_page_of_prs_fetched():
    """The count query sees every open PR; the frame may be one page of them."""
    prs = _prs(
        is_draft=[False],
        approving_reviews=[0],
        total_reviews=[0],
        review_requests=[0],
        age_days=[1.0],
    )
    assert app._open_pr_signals(prs, 79)["total"] == 79


def test_no_prs_at_all_reads_as_zero_rather_than_an_error():
    signals = app._open_pr_signals(pd.DataFrame(), None)
    assert signals == {
        "total": 0,
        "unapproved": 0,
        "never_reviewed": 0,
        "oldest_unreviewed_days": None,
        "no_reviewer_asked": 0,
    }


def test_ownerless_counts_every_spelling_of_nobody():
    df = pd.DataFrame({"assignee": ["Tam", "Unassigned", None, "  none  ", ""]})
    assert app._ownerless(df) == 4


def test_estimate_coverage_excludes_the_backlog():
    """Backlog tickets are not expected to carry an estimate yet.

    Counting them would make the tile a measure of backlog size rather than of
    whether the team estimates the work it has actually picked up.
    """
    df = pd.DataFrame(
        {
            "status": ["In Progress", "In Progress", "Backlog"],
            "original_estimate_sec": [7200, 0, 0],
        }
    )
    estimated, estimable = app._estimate_coverage(df)
    assert (estimated, estimable) == (1, 2)


def test_estimate_coverage_reads_the_estimates_not_a_missing_flag():
    """The tile said 100% while Delivery and Planning said 77% of the same tickets.

    Today's frame carries no ``has_estimate`` column - that is added by
    ``estimate_policy`` further down the page - and a missing flag was read as
    "estimated", so every non-backlog ticket counted as covered.
    """
    df = pd.DataFrame(
        {
            "status": ["In Progress", "To Do", "Review in Staging"],
            "original_estimate_sec": [3600, 0, 0],
        }
    )
    assert app._estimate_coverage(df) == (1, 3)


def test_estimate_coverage_agrees_with_the_policy_the_other_pages_read():
    """One number, one definition: Today cannot disagree with Delivery."""
    df = pd.DataFrame(
        {
            "status": ["In Progress", "To Do", "Backlog", "In Progress"],
            "issue_type": ["Task", "Bug", "Task", "Epic"],
            "original_estimate_sec": [3600, 0, 0, 0],
        }
    )
    scored = hygiene.estimate_policy(df, app.BACKLOG_STATUSES)
    in_policy = scored[scored["policy_applies"]]
    expected = (
        int(in_policy["has_estimate"].sum()),
        int(len(in_policy)),
    )
    assert app._estimate_coverage(df) == expected
    # The epic is exempt (it holds other tickets' hours) and Backlog is not asked.
    assert expected == (1, 2)


def test_a_status_column_is_counted_before_it_is_ranked():
    """The Today chart drew ten zero-length bars labelled 0, 1, 2 ...

    ``ranked`` counts values indexed by category, so handing it the per-ticket
    status column coerced every status to 0 and labelled the bars off the row
    numbers. It now refuses rather than drawing an empty chart.
    """
    statuses = pd.Series(["To Do", "To Do", "In Progress"])
    with pytest.raises(TypeError):
        theme.ranked(statuses)
    ranked = theme.ranked(statuses.value_counts())
    assert ranked.loc["To Do"] == 2
    assert ranked.loc["In Progress"] == 1


def test_stalled_is_measured_on_status_age_not_edit_age():
    """The point of the tile: a label edit must not empty the stalled queue.

    `idle_days` of 1 with no status transition for months is the exact shape of
    a masked ticket, and it has to keep counting as stalled.
    """
    df = pd.DataFrame(
        {
            "key": ["ENG-1"],
            "assignee": ["Tam"],
            "status": ["In Progress"],
            "idle_days": [1.0],
            "created": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "changelog": [
                [
                    {
                        "created": "2026-01-02T00:00:00.000+0000",
                        "items": [
                            {
                                "field": "status",
                                "fromString": "To Do",
                                "toString": "In Progress",
                            }
                        ],
                        "author": {"displayName": "Tam"},
                    }
                ]
            ],
        }
    )
    count, clock = app._stalled_count(df)
    assert count == 1
    assert clock == "status age"


def test_a_ticket_with_no_changelog_says_which_clock_it_fell_back_to():
    """Silence about the fallback would present the gameable clock as the honest one."""
    df = pd.DataFrame(
        {
            "key": ["ENG-2"],
            "assignee": ["Tam"],
            "status": ["In Progress"],
            "idle_days": [45.0],
            "changelog": [None],
        }
    )
    count, clock = app._stalled_count(df)
    assert count == 1
    assert "edit age" in clock


def test_an_empty_board_stalls_nothing():
    assert app._stalled_count(pd.DataFrame()) == (0, "status age")
    assert app._estimate_coverage(pd.DataFrame()) == (0, 0)
    assert app._ownerless(pd.DataFrame()) == 0


def test_the_rows_behind_the_ownerless_tile_are_the_tickets_themselves():
    """A count with no rows behind it cannot be acted on, which was the complaint."""
    df = pd.DataFrame(
        {
            "key": ["ENG-1", "ENG-2", "ENG-3"],
            "assignee": ["Tam", None, "Unassigned"],
            "status": ["In Progress", "To Do", "To Do"],
            "ticket_age_days": [1.0, 30.0, 90.0],
        }
    )
    rows = app._ownerless_rows(df)
    assert rows["key"].tolist() == ["ENG-2", "ENG-3"]
    assert app._ownerless(df) == len(rows)


def test_the_rows_behind_the_stalled_tile_are_the_tickets_themselves():
    df = pd.DataFrame(
        {
            "key": ["ENG-1", "ENG-2"],
            "assignee": ["Tam", "Mehdi"],
            "status": ["In Progress", "In Progress"],
            "idle_days": [1.0, 45.0],
            "changelog": [None, None],
        }
    )
    rows, clock = app._stalled_rows(df)
    assert rows["key"].tolist() == ["ENG-2"]
    assert "edit age" in clock


def test_today_counts_the_board_delivery_counts_and_not_the_backlog():
    """The landing page said 16 open tickets above a Delivery page reading 14."""
    df = pd.DataFrame(
        {
            "key": ["ENG-1", "ENG-2", "ENG-3"],
            "status": ["In Progress", "Backlog", "To Do"],
            "assignee": ["Tam", "Tam", None],
        }
    )
    board = app._metrics_df(df, include_backlogs=False)
    assert board["key"].tolist() == ["ENG-1", "ENG-3"]
    assert app._ownerless(board) == 1


def test_stalled_is_still_status_age_after_the_backlog_rows_are_dropped():
    """A gappy index made the mask unalignable, and the honest clock was lost.

    The exception was swallowed, so the tile quietly reported edit age while its
    own caption said "not edit age" - a filtered board is the normal case here.
    """
    df = pd.DataFrame(
        {
            "key": ["ENG-1", "ENG-2"],
            "assignee": ["Tam", "Tam"],
            "status": ["Backlog", "In Progress"],
            "idle_days": [1.0, 1.0],
            "created": [pd.Timestamp("2026-01-01T00:00:00Z")] * 2,
            "changelog": [
                None,
                [
                    {
                        "created": "2026-01-02T00:00:00.000+0000",
                        "items": [
                            {
                                "field": "status",
                                "fromString": "To Do",
                                "toString": "In Progress",
                            }
                        ],
                        "author": {"displayName": "Tam"},
                    }
                ],
            ],
        }
    )
    board = app._metrics_df(df, include_backlogs=False)
    rows, clock = app._stalled_rows(board)
    assert clock == "status age"
    assert rows["key"].tolist() == ["ENG-2"]


def test_a_stalled_row_carries_the_clock_it_was_selected_on():
    """A ticket picked for 60 days of no movement must not claim one day.

    ``idle_days`` is reset by any field edit, which is exactly why the tile
    measures status age - so the age travels with the rows.
    """
    df = pd.DataFrame(
        {
            "key": ["ENG-1"],
            "assignee": ["Tam"],
            "status": ["In Progress"],
            "idle_days": [1.0],
            "created": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "changelog": [
                [
                    {
                        "created": "2026-01-02T00:00:00.000+0000",
                        "items": [
                            {
                                "field": "status",
                                "fromString": "To Do",
                                "toString": "In Progress",
                            }
                        ],
                        "author": {"displayName": "Tam"},
                    }
                ]
            ],
        }
    )
    rows, clock = app._stalled_rows(df)
    assert clock == "status age"
    assert rows[next_actions.STALLED_AGE_COLUMN].iloc[0] > 30
    action = next_actions.stalled_actions(rows, url_for=lambda key: f"https://j/{key}")[0]
    assert "no movement in 1d" not in action.detail


def test_a_failed_read_is_unknown_and_not_an_all_clear():
    """An outage must not be handed to a reader as "nothing to do".

    ``_gather`` leaves a failed read out of ``data`` entirely, so the empty
    queue it produces is indistinguishable from a clear one unless the page is
    told which sources it could not read.
    """
    bundle = types.SimpleNamespace(
        data={},  # every read failed, triage_stuck included
        github_ready=False,
        open_prs=pd.DataFrame(),
    )
    board = pd.DataFrame(
        {"key": ["ENG-1"], "assignee": ["Tam"], "status": ["In Progress"]}
    )
    queues, unknown = app._action_queues(bundle, board)
    assert unknown == {"review", "triage"}
    assert queues["review"] == [] and queues["triage"] == []
    assert set(app._ACTION_QUEUE_NAMES) == set(queues)


def test_a_triage_read_that_worked_and_found_nothing_is_not_unknown():
    bundle = types.SimpleNamespace(
        data={"triage_stuck": pd.DataFrame()},
        github_ready=True,
        open_prs=pd.DataFrame(),
    )
    _, unknown = app._action_queues(bundle, pd.DataFrame())
    assert unknown == set()


def test_the_action_list_reuses_the_stalled_rows_the_tile_measured(monkeypatch):
    """Selecting stalled rows walks every changelog, so it happens once a render."""

    def refuse(_df):  # pragma: no cover - called only if the rows are recomputed
        raise AssertionError("_stalled_rows called a second time")

    monkeypatch.setattr(app, "_stalled_rows", refuse)
    bundle = types.SimpleNamespace(
        data={"triage_stuck": pd.DataFrame()},
        github_ready=True,
        open_prs=pd.DataFrame(),
    )
    board = pd.DataFrame(
        {"key": ["ENG-1"], "assignee": ["Tam"], "status": ["In Progress"]}
    )
    stalled = board.assign(**{next_actions.STALLED_AGE_COLUMN: [44.0]})
    queues, _ = app._action_queues(bundle, board, stalled=stalled)
    assert queues["stalled"][0].days == 44.0
