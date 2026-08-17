"""The Today page's signals - the numbers the landing page opens with.

Each one is a count a reader will act on, so each one is pinned here against the
way it could quietly go wrong: drafts counted as unreviewed, a label edit
emptying the stalled queue, backlog tickets dragging estimate coverage down.
"""

from __future__ import annotations

import pandas as pd

import app


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
            "has_estimate": [True, False, False],
        }
    )
    estimated, estimable = app._estimate_coverage(df)
    assert (estimated, estimable) == (1, 2)


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
