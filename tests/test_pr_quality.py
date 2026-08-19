"""Offline checks for the PR-quality arithmetic.

Every case here is a behaviour somebody could be graded on, so the tests are
written as the situations they are meant to catch: the engineer who splits one
change into five trivial PRs, the one who merges their own work unreviewed, the
pair who only ever approve each other, and - as important as any of them - the
throttled fetch, where a field is missing and the metric has to say "unknown"
rather than "zero".
"""

from __future__ import annotations

import sys
from pathlib import Path

import time as time_module

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import github_client  # noqa: E402
import pr_quality  # noqa: E402

DEVIN = "devin-ai-integration[bot]"


def pr(
    number: int,
    author: str,
    *,
    changed_lines: int | None = 50,
    state: str = "MERGED",
    merged_by: str | None = None,
    base: str = "main",
    reviews: list[dict] | None = None,
    title: str = "Some change",
    branch: str = "feature",
    body: str = "",
    hygiene: bool = True,
    threads: int | None = 0,
    ready_at: str | None = None,
    created_at: str = "2026-08-01T09:00:00Z",
    timeline_events: list[dict] | None = "unset",  # type: ignore[assignment]
) -> dict:
    """One PR row shaped like :func:`github_client._to_frame` builds them.

    ``timeline_events`` defaults to the sentinel ``"unset"`` rather than
    ``None`` so a fixture that never mentions it drops the column entirely
    (an extended-payload-blind row, same as a real lean/detail fetch) instead
    of accidentally asserting "extended payload fetched, zero events" for
    every existing test in this file.
    """
    row = {
        "number": number,
        "url": f"https://github.com/acme/repo/pull/{number}",
        "title": title,
        "state": state,
        "author": author,
        "branch": branch,
        "body": body,
        "hygiene_fetched": hygiene,
        "created_at": pd.Timestamp(created_at),
        "merged_at": pd.Timestamp("2026-08-02T09:00:00Z") if state == "MERGED" else pd.NaT,
        "closed_at": pd.NaT if state == "OPEN" else pd.Timestamp("2026-08-02T09:00:00Z"),
        "review_ready_at": pd.Timestamp(ready_at) if ready_at else pd.NaT,
        "approving_reviews": sum(
            1 for r in reviews or [] if r.get("state") == "APPROVED"
        ),
        "changed_lines": changed_lines,
        "changed_files": 1 if changed_lines else None,
        "merged_by": merged_by,
        "base_branch": base,
        "review_threads": threads,
        "comments": threads,
        "reviews": reviews,
    }
    if timeline_events != "unset":
        row["timeline_events"] = timeline_events
    return row


def review(reviewer: str, state: str, at: str = "2026-08-01T12:00:00Z", body: str = "") -> dict:
    return {"reviewer": reviewer, "state": state, "submitted_at": at, "body": body}


def requested(reviewer: str, at: str) -> dict:
    """A ``ReviewRequestedEvent``, shaped like :func:`github_client._timeline_events`."""
    return {"type": "review_requested", "created_at": at, "requested_reviewer": reviewer}


def drafted(at: str) -> dict:
    """A ``ConvertToDraftEvent``."""
    return {"type": "converted_to_draft", "created_at": at, "requested_reviewer": None}


def ready(at: str) -> dict:
    """A ``ReadyForReviewEvent``."""
    return {"type": "ready_for_review", "created_at": at, "requested_reviewer": None}


# --------------------------------------------------------------------------- #
# Size
# --------------------------------------------------------------------------- #


def test_splitting_one_change_into_five_trivial_prs_is_visible():
    frame = pd.DataFrame(
        [pr(n, "splitter", changed_lines=4) for n in range(1, 6)]
        + [pr(9, "shipper", changed_lines=300)]
    )
    bands = pr_quality.size_bands(frame).set_index("author")
    assert bands.loc["splitter", "trivial"] == 5
    assert bands.loc["splitter", "trivial_share"] == 1.0
    assert bands.loc["splitter", "median_changed_lines"] == 4
    # Same PR count would rank these two the same; the bands do not.
    assert bands.loc["shipper", "medium"] == 1
    assert bands.loc["shipper", "trivial"] == 0


def test_the_bands_have_the_boundaries_they_claim():
    assert pr_quality.size_band(9) == "trivial"
    assert pr_quality.size_band(10) == "small"
    assert pr_quality.size_band(99) == "small"
    assert pr_quality.size_band(100) == "medium"
    assert pr_quality.size_band(400) == "large"
    assert pr_quality.size_band(1000) == "large"
    assert pr_quality.size_band(1001) == "oversized"
    assert pr_quality.size_band(None) == ""


def test_a_throttled_fetch_counts_as_unsized_not_as_trivial():
    frame = pd.DataFrame(
        [pr(1, "alice", changed_lines=None), pr(2, "alice", changed_lines=4)]
    )
    bands = pr_quality.size_bands(frame).set_index("author")
    assert bands.loc["alice", "unsized"] == 1
    assert bands.loc["alice", "trivial"] == 1
    # One of the two PRs had a size; the share is over that one, not over both.
    assert bands.loc["alice", "trivial_share"] == 1.0


# --------------------------------------------------------------------------- #
# The AI reviewer
# --------------------------------------------------------------------------- #


def test_devin_findings_are_counted_per_pr_and_split_by_verdict():
    frame = pd.DataFrame(
        [
            pr(
                1,
                "alice",
                threads=7,
                reviews=[
                    review(DEVIN, "CHANGES_REQUESTED"),
                    review(DEVIN, "CHANGES_REQUESTED"),
                    review(DEVIN, "COMMENTED"),
                    review("bob", "APPROVED"),
                ],
            ),
            pr(2, "alice", reviews=[review(DEVIN, "APPROVED")]),
        ]
    )
    detail = pr_quality.devin_findings(frame).set_index("number")
    assert detail.loc[1, "ai_reviews"] == 3
    assert detail.loc[1, "ai_changes_requested"] == 2
    assert detail.loc[1, "ai_commented"] == 1
    assert detail.loc[1, "review_threads"] == 7
    assert detail.loc[2, "ai_approved"] == 1
    assert detail.loc[2, "ai_changes_requested"] == 0

    by_author = pr_quality.devin_findings_by_author(frame).set_index("author")
    assert by_author.loc["alice", "prs_judged"] == 2
    assert by_author.loc["alice", "prs_changes_requested"] == 1
    assert by_author.loc["alice", "changes_requested_share"] == 0.5
    assert by_author.loc["alice", "ai_findings"] == 2


def test_a_configurable_reviewer_login_is_honoured():
    frame = pd.DataFrame([pr(1, "alice", reviews=[review("some-other-bot", "CHANGES_REQUESTED")])])
    assert pr_quality.devin_findings(frame).loc[0, "ai_changes_requested"] == 0
    matched = pr_quality.devin_findings(frame, reviewer_patterns=("some-other-bot",))
    assert matched.loc[0, "ai_changes_requested"] == 1


def test_reviews_that_were_never_fetched_read_as_unknown_not_as_clean():
    lean = pd.DataFrame([pr(1, "alice")]).drop(columns=["reviews"])
    detail = pr_quality.devin_findings(lean)
    assert bool(detail.loc[0, "reviews_fetched"]) is False
    assert pd.isna(detail.loc[0, "ai_changes_requested"])
    # Nobody is graded on a question the API was never asked.
    assert pr_quality.devin_findings_by_author(lean).empty

    # Same again when the column exists but this row's value is None.
    partial = pd.DataFrame([pr(1, "alice", reviews=None), pr(2, "bob", reviews=[])])
    findings = pr_quality.devin_findings(partial).set_index("number")
    assert pd.isna(findings.loc[1, "ai_reviews"])
    assert findings.loc[2, "ai_reviews"] == 0
    assert list(pr_quality.devin_findings_by_author(partial)["author"]) == ["bob"]


# --------------------------------------------------------------------------- #
# Reviewing as work
# --------------------------------------------------------------------------- #


def test_reviews_given_are_credited_and_the_ai_is_not_on_the_leaderboard():
    frame = pd.DataFrame(
        [
            pr(
                1,
                "alice",
                created_at="2026-08-01T09:00:00Z",
                reviews=[
                    review(DEVIN, "CHANGES_REQUESTED", "2026-08-01T09:05:00Z"),
                    review("bob", "APPROVED", "2026-08-01T11:00:00Z"),
                    review("carol", "COMMENTED", "2026-08-01T15:00:00Z"),
                    review("alice", "COMMENTED", "2026-08-01T10:00:00Z"),
                ],
            ),
            pr(
                2,
                "carol",
                created_at="2026-08-02T09:00:00Z",
                reviews=[review("bob", "CHANGES_REQUESTED", "2026-08-02T13:00:00Z")],
            ),
        ]
    )
    citizenship = pr_quality.review_citizenship(frame).set_index("reviewer")
    assert DEVIN not in citizenship.index  # excluded by default
    assert "alice" not in citizenship.index  # self-review is not citizenship
    assert citizenship.loc["bob", "reviews_given"] == 2
    assert citizenship.loc["bob", "distinct_authors_reviewed"] == 2
    assert citizenship.loc["bob", "approvals_given"] == 1
    assert citizenship.loc["bob", "changes_requested_given"] == 1
    # Bob was first on both: 2h after PR 1 opened, 4h after PR 2.
    assert citizenship.loc["bob", "median_hours_to_first_review"] == 3.0
    # Carol was only ever second, so she has no response time to show.
    assert pd.isna(citizenship.loc["carol", "median_hours_to_first_review"])


def test_the_review_clock_starts_when_a_reviewer_was_asked_for():
    frame = pd.DataFrame(
        [
            pr(
                1,
                "alice",
                created_at="2026-08-01T09:00:00Z",  # opened as a draft a day earlier
                ready_at="2026-08-02T09:00:00Z",
                reviews=[review("bob", "APPROVED", "2026-08-02T12:00:00Z")],
            )
        ]
    )
    citizenship = pr_quality.review_citizenship(frame).set_index("reviewer")
    assert citizenship.loc["bob", "median_hours_to_first_review"] == 3.0


def test_a_team_that_never_reviews_produces_an_empty_frame_not_an_error():
    frame = pd.DataFrame([pr(1, "alice", reviews=[]), pr(2, "bob", reviews=[])])
    assert pr_quality.review_citizenship(frame).empty
    assert pr_quality.reciprocity(frame).pairs.empty
    assert pr_quality.review_citizenship(pd.DataFrame()).empty


# --------------------------------------------------------------------------- #
# Reciprocity
# --------------------------------------------------------------------------- #


def test_a_mutual_approval_pair_shows_up_as_a_closed_loop():
    rows = []
    for n in range(1, 5):
        rows.append(pr(n, "alice", reviews=[review("bob", "APPROVED")]))
        rows.append(pr(100 + n, "bob", reviews=[review("alice", "APPROVED")]))
    # A third person who spreads their reviewing around.
    rows.append(pr(200, "alice", reviews=[review("carol", "COMMENTED")]))
    rows.append(pr(201, "bob", reviews=[review("carol", "COMMENTED")]))
    rows.append(pr(202, "dan", reviews=[review("carol", "COMMENTED")]))
    result = pr_quality.reciprocity(pd.DataFrame(rows))

    pairs = result.pairs.set_index(["reviewer", "author"])
    assert pairs.loc[("bob", "alice"), "reviews"] == 4
    assert bool(pairs.loc[("bob", "alice"), "mutual"]) is True
    assert bool(pairs.loc[("carol", "dan"), "mutual"]) is False

    people = result.by_person.set_index("reviewer")
    assert people.loc["bob", "top_partner"] == "alice"
    assert people.loc["bob", "concentration"] == 1.0  # bob reviews nobody else
    assert people.loc["carol", "concentration"] < 0.4  # three people, evenly
    # Every one of those approvals was silent, on a PR with no threads.
    assert people.loc["bob", "rubber_stamp_approvals"] == 4
    assert people.loc["bob", "rubber_stamp_share"] == 1.0


def test_an_approval_with_something_written_on_it_is_not_a_rubber_stamp():
    frame = pd.DataFrame(
        [
            pr(1, "alice", reviews=[review("bob", "APPROVED", body="checked the migration")]),
            pr(2, "alice", threads=3, reviews=[review("bob", "APPROVED")]),
        ]
    )
    people = pr_quality.reciprocity(frame).by_person.set_index("reviewer")
    assert people.loc["bob", "approvals_given"] == 2
    assert people.loc["bob", "rubber_stamp_approvals"] == 0


# --------------------------------------------------------------------------- #
# How work got merged
# --------------------------------------------------------------------------- #


def test_the_self_merger_is_separated_from_the_normal_case():
    frame = pd.DataFrame(
        [
            # Merged their own PR that nobody approved: the review never happened.
            pr(1, "solo", merged_by="solo", reviews=[]),
            # Merged their own PR after a colleague approved: ordinary.
            pr(2, "solo", merged_by="solo", reviews=[review("bob", "APPROVED")]),
            # Only the AI approved it.
            pr(3, "solo", merged_by="solo", reviews=[review(DEVIN, "APPROVED")]),
            # Someone else merged it, with an approval.
            pr(4, "team", merged_by="bob", reviews=[review("bob", "APPROVED")]),
            # Merged into their own feature branch: nothing shipped.
            pr(5, "solo", merged_by="solo", base="solo/stack", reviews=[review("bob", "APPROVED")]),
        ]
    )
    out = pr_quality.self_merge(frame).set_index("author")
    assert out.loc["solo", "merged_prs"] == 4
    assert out.loc["solo", "self_merged"] == 4
    assert out.loc["solo", "merged_without_outside_approval"] == 2  # PRs 1 and 3
    assert out.loc["solo", "ai_only_approval"] == 1
    assert out.loc["solo", "merged_off_trunk"] == 1
    assert out.loc["team", "self_merged"] == 0
    assert out.loc["team", "merged_without_outside_approval"] == 0


def test_an_unfetched_review_list_leaves_approval_unknown_rather_than_missing():
    lean = pd.DataFrame([pr(1, "solo", merged_by="solo")]).drop(columns=["reviews"])
    out = pr_quality.self_merge(lean).set_index("author")
    assert out.loc["solo", "self_merged"] == 1
    # approving_reviews was 0 on the lean row, which GitHub can answer: unapproved.
    assert out.loc["solo", "merged_without_outside_approval"] == 1
    assert out.loc["solo", "unknown_approval"] == 0

    unknown = lean.copy()
    unknown["approving_reviews"] = None
    out = pr_quality.self_merge(unknown).set_index("author")
    assert out.loc["solo", "unknown_approval"] == 1
    assert out.loc["solo", "merged_without_outside_approval"] == 0


def test_open_prs_are_not_counted_as_merged_work():
    frame = pd.DataFrame([pr(1, "alice", state="OPEN", merged_by=None)])
    assert pr_quality.self_merge(frame).empty
    assert pr_quality.traceability(frame).empty


# --------------------------------------------------------------------------- #
# Abandoned work
# --------------------------------------------------------------------------- #


def test_abandoned_prs_are_a_rate_over_what_was_actually_decided():
    frame = pd.DataFrame(
        [
            pr(1, "alice", state="MERGED"),
            pr(2, "alice", state="CLOSED"),
            pr(3, "alice", state="CLOSED"),
            pr(4, "alice", state="OPEN"),  # undecided: not abandoned
            pr(5, "bob", state="MERGED"),
        ]
    )
    out = pr_quality.abandoned_rate(frame).set_index("author")
    assert out.loc["alice", "closed_prs"] == 3
    assert out.loc["alice", "abandoned"] == 2
    assert round(float(out.loc["alice", "abandoned_rate"]), 3) == 0.667
    assert out.loc["bob", "abandoned_rate"] == 0.0


def test_abandonment_falls_back_to_timestamps_when_state_is_missing():
    frame = pd.DataFrame([pr(1, "alice", state="MERGED"), pr(2, "alice", state="CLOSED")])
    frame["state"] = ""
    out = pr_quality.abandoned_rate(frame).set_index("author")
    assert out.loc["alice", "closed_prs"] == 2
    assert out.loc["alice", "abandoned"] == 1


# --------------------------------------------------------------------------- #
# Traceability of shipped work
# --------------------------------------------------------------------------- #


def test_merged_work_is_measured_on_whether_it_names_a_ticket():
    frame = pd.DataFrame(
        [
            pr(1, "alice", title="MB-12 fix login"),
            pr(2, "alice", title="tidy up", branch="tidy", body=""),
            pr(3, "bob", title="drive-by", branch="MB-99-refactor"),
        ]
    )
    out = pr_quality.traceability(frame, project_keys=["MB"]).set_index("author")
    assert out.loc["alice", "judgeable"] == 2
    assert out.loc["alice", "with_key"] == 1
    assert out.loc["alice", "traceability"] == 0.5
    assert out.loc["bob", "traceability"] == 1.0  # key was in the branch


def test_a_pr_whose_key_was_never_looked_for_is_not_counted_as_missing_one():
    # The merged fetch without hygiene fields: no branch, no body, and a title
    # that happens not to carry the key.
    frame = pd.DataFrame(
        [
            pr(1, "alice", title="tidy up", branch="", body="", hygiene=False),
            pr(2, "alice", title="MB-3 real fix", branch="", body="", hygiene=False),
        ]
    )
    out = pr_quality.traceability(frame, project_keys=["MB"]).set_index("author")
    assert out.loc["alice", "merged_prs"] == 2
    assert out.loc["alice", "not_judgeable"] == 1
    # The one PR whose key was visible in the title is judged, and passes.
    assert out.loc["alice", "judgeable"] == 1
    assert out.loc["alice", "traceability"] == 1.0


# --------------------------------------------------------------------------- #
# Reviews nobody asked for
# --------------------------------------------------------------------------- #


def test_a_review_by_someone_who_was_requested_is_not_unprompted():
    frame = pd.DataFrame(
        [
            pr(
                1,
                "alice",
                reviews=[review("bob", "APPROVED", "2026-08-01T12:00:00Z")],
                timeline_events=[requested("bob", "2026-08-01T10:00:00Z")],
            )
        ]
    )
    out = pr_quality.unprompted_reviews(frame)
    assert out.empty


def test_a_review_by_someone_never_requested_is_unprompted():
    frame = pd.DataFrame(
        [
            pr(
                1,
                "alice",
                reviews=[review("carol", "APPROVED", "2026-08-01T12:00:00Z")],
                timeline_events=[requested("bob", "2026-08-01T10:00:00Z")],
            )
        ]
    )
    out = pr_quality.unprompted_reviews(frame).set_index("reviewer")
    assert out.loc["carol", "unprompted_reviews"] == 1
    assert out.loc["carol", "prs"] == (1,)
    assert "bob" not in out.index


def test_a_review_request_arriving_after_the_review_still_counts_it_unprompted():
    # Ordering pinned: the request timestamp is *after* the review's.
    frame = pd.DataFrame(
        [
            pr(
                1,
                "alice",
                reviews=[review("bob", "APPROVED", "2026-08-01T09:00:00Z")],
                timeline_events=[requested("bob", "2026-08-01T15:00:00Z")],
            )
        ]
    )
    out = pr_quality.unprompted_reviews(frame).set_index("reviewer")
    assert out.loc["bob", "unprompted_reviews"] == 1


def test_unprompted_reviews_is_read_alongside_reciprocity():
    # bob reviews alice unprompted four times and reviews nobody else: the
    # concentration/top_partner columns must say so, straight from reciprocity.
    rows = [
        pr(n, "alice", reviews=[review("bob", "APPROVED")], timeline_events=[])
        for n in range(1, 5)
    ]
    out = pr_quality.unprompted_reviews(pd.DataFrame(rows)).set_index("reviewer")
    assert out.loc["bob", "unprompted_reviews"] == 4
    assert out.loc["bob", "top_partner"] == "alice"
    assert out.loc["bob", "concentration"] == 1.0


def test_self_reviews_and_the_ai_are_never_unprompted_evidence():
    frame = pd.DataFrame(
        [
            pr(
                1,
                "alice",
                reviews=[
                    review("alice", "COMMENTED"),
                    review(DEVIN, "CHANGES_REQUESTED"),
                ],
                timeline_events=[],
            )
        ]
    )
    assert pr_quality.unprompted_reviews(frame).empty


def test_unprompted_reviews_without_the_extended_payload_is_empty_not_wrong():
    lean = pd.DataFrame([pr(1, "alice", reviews=[review("bob", "APPROVED")])])
    assert "timeline_events" not in lean.columns
    assert pr_quality.unprompted_reviews(lean).empty


# --------------------------------------------------------------------------- #
# Hiding in draft
# --------------------------------------------------------------------------- #


def test_going_draft_after_a_review_request_is_flagged():
    frame = pd.DataFrame(
        [
            pr(
                1,
                "alice",
                state="CLOSED",
                timeline_events=[
                    requested("bob", "2026-08-01T09:00:00Z"),
                    drafted("2026-08-01T10:00:00Z"),
                ],
            )
        ]
    )
    detail = pr_quality.draft_transitions(frame).detail.set_index("number")
    assert detail.loc[1, "draft_round_trips"] == 1
    assert bool(detail.loc[1, "after_review_request"]) is True


def test_a_pr_opened_as_a_draft_and_marked_ready_once_is_not_flagged():
    frame = pd.DataFrame(
        [pr(1, "alice", timeline_events=[ready("2026-08-01T09:00:00Z")])]
    )
    detail = pr_quality.draft_transitions(frame).detail.set_index("number")
    assert detail.loc[1, "draft_round_trips"] == 0
    assert bool(detail.loc[1, "after_review_request"]) is False


def test_going_draft_before_any_request_is_not_flagged():
    # The draft conversion precedes the request: an author's own choice to
    # keep working, not a reaction to being asked for review.
    frame = pd.DataFrame(
        [
            pr(
                1,
                "alice",
                timeline_events=[
                    drafted("2026-08-01T08:00:00Z"),
                    requested("bob", "2026-08-01T09:00:00Z"),
                ],
            )
        ]
    )
    detail = pr_quality.draft_transitions(frame).detail.set_index("number")
    assert detail.loc[1, "draft_round_trips"] == 1
    assert bool(detail.loc[1, "after_review_request"]) is False


def test_the_flagged_subset_gets_a_real_abandoned_rate_not_a_second_metric():
    frame = pd.DataFrame(
        [
            pr(
                1,
                "alice",
                state="CLOSED",
                timeline_events=[requested("bob", "2026-08-01T09:00:00Z"), drafted("2026-08-01T10:00:00Z")],
            ),
            pr(
                2,
                "alice",
                state="MERGED",
                timeline_events=[requested("bob", "2026-08-01T09:00:00Z"), drafted("2026-08-01T10:00:00Z")],
            ),
            # Not flagged: opened as a draft, never asked for review first.
            pr(3, "alice", state="CLOSED", timeline_events=[ready("2026-08-01T09:00:00Z")]),
        ]
    )
    outcome = pr_quality.draft_transitions(frame).outcome.set_index("author")
    assert outcome.loc["alice", "closed_prs"] == 2  # only the two flagged PRs
    assert outcome.loc["alice", "abandoned"] == 1
    assert outcome.loc["alice", "merged"] == 1


def test_draft_transitions_without_the_extended_payload_is_empty_not_wrong():
    lean = pd.DataFrame([pr(1, "alice")])
    result = pr_quality.draft_transitions(lean)
    assert result.detail.empty
    assert result.outcome.empty


# --------------------------------------------------------------------------- #
# Size in points
# --------------------------------------------------------------------------- #


def test_delivered_points_binds_count_to_median_size_structurally():
    with pytest.raises(TypeError):
        pr_quality.DeliveredPoints(author="x", prs=5, points=10.0, trivial_share=0.0)  # type: ignore[call-arg]


def test_five_trivial_prs_score_below_one_medium_pr():
    splitter = pd.DataFrame([pr(n, "splitter", changed_lines=4) for n in range(1, 6)])
    shipper = pd.DataFrame([pr(9, "shipper", changed_lines=300)])
    split_points = {d.author: d for d in pr_quality.delivered_points(splitter)}
    ship_points = {d.author: d for d in pr_quality.delivered_points(shipper)}
    assert split_points["splitter"].prs == 5
    assert split_points["splitter"].points < ship_points["shipper"].points
    # The binding itself: every element carries both together.
    assert split_points["splitter"].median_changed_lines == 4


def test_delivered_points_returns_a_list_not_a_frame_with_a_bare_count_column():
    frame = pd.DataFrame([pr(1, "alice", changed_lines=50)])
    out = pr_quality.delivered_points(frame)
    assert isinstance(out, list)
    assert all(isinstance(item, pr_quality.DeliveredPoints) for item in out)
    assert not isinstance(out, pd.DataFrame)


def test_delivered_points_survives_an_empty_frame():
    assert pr_quality.delivered_points(pd.DataFrame()) == []


# --------------------------------------------------------------------------- #
# The payload underneath all of it
# --------------------------------------------------------------------------- #


def test_the_detail_payload_lands_in_the_frame():
    node = {
        "number": 7,
        "title": "MB-1 fix",
        "state": "MERGED",
        "author": {"login": "alice"},
        "additions": 30,
        "deletions": 12,
        "changedFiles": 3,
        "commits": {"totalCount": 4},
        "mergedBy": {"login": "alice"},
        "baseRefName": "main",
        "reviewNodes": {
            "nodes": [
                {
                    "author": {"login": "bob"},
                    "state": "APPROVED",
                    "submittedAt": "2026-08-01T12:00:00Z",
                    "body": "fine",
                }
            ]
        },
        "reviewThreads": {"totalCount": 5},
        "comments": {"totalCount": 9},
        "timelineItems": {
            "nodes": [
                {"__typename": "ReviewRequestedEvent", "createdAt": "2026-08-01T11:00:00Z"},
                {"__typename": "ReadyForReviewEvent", "createdAt": "2026-08-01T10:00:00Z"},
            ]
        },
    }
    row = github_client._to_frame([node]).iloc[0]
    assert row["changed_lines"] == 42
    assert row["changed_files"] == 3
    assert row["commits"] == 4
    assert row["merged_by"] == "alice"
    assert row["base_branch"] == "main"
    assert row["review_threads"] == 5
    assert row["comments"] == 9
    assert row["reviews"] == [
        {
            "reviewer": "bob",
            "state": "APPROVED",
            "submitted_at": "2026-08-01T12:00:00Z",
            "body": "fine",
        }
    ]
    # The earliest of the two events: when the PR first asked for a reviewer.
    assert row["review_ready_at"] == pd.Timestamp("2026-08-01T10:00:00Z")
    assert bool(row["detail_fetched"]) is True


def test_a_lean_response_leaves_the_detail_columns_unknown_not_zero():
    row = github_client._to_frame(
        [{"number": 7, "author": {"login": "alice"}, "allReviews": {"totalCount": 2}}]
    ).iloc[0]
    for column in (
        "additions",
        "deletions",
        "changed_lines",
        "changed_files",
        "commits",
        "merged_by",
        "base_branch",
        "review_threads",
        "comments",
        "reviews",
    ):
        assert row[column] is None, column
    assert bool(row["detail_fetched"]) is False
    assert bool(row["hygiene_fetched"]) is False
    # The columns the lean payload does answer keep answering.
    assert row["total_reviews"] == 2


def test_a_review_body_is_truncated_rather_than_carried_whole():
    node = {
        "number": 1,
        "author": {"login": "alice"},
        "additions": 1,
        "deletions": 0,
        "reviewNodes": {
            "nodes": [{"author": {"login": DEVIN}, "state": "COMMENTED", "body": "x" * 5000}]
        },
    }
    reviews = github_client._to_frame([node]).iloc[0]["reviews"]
    assert len(reviews[0]["body"]) == github_client.MAX_REVIEW_BODY_CHARS


def test_the_rich_query_degrades_to_the_lean_one_when_the_token_cannot_afford_it(monkeypatch):
    seen: list[str] = []

    def fake_graphql(token, query, variables):
        seen.append(query)
        if query is github_client._DETAIL_SEARCH_QUERY:
            raise github_client.GitHubConfigError("API rate limit exceeded")
        return {
            "search": {
                "issueCount": 1,
                "pageInfo": {"hasNextPage": False, "endCursor": None},
                "nodes": [{"number": 1, "author": {"login": "alice"}}],
            }
        }

    monkeypatch.setattr(github_client, "_graphql", fake_graphql)
    nodes = github_client._search_prs(
        "token", "org:acme is:pr", 100, github_client._DETAIL_SEARCH_QUERY, github_client._SEARCH_QUERY
    )
    assert [n["number"] for n in nodes] == [1]
    assert seen == [github_client._DETAIL_SEARCH_QUERY, github_client._SEARCH_QUERY]


def test_a_failure_of_the_lean_query_is_still_an_error(monkeypatch):
    def always_fails(token, query, variables):
        raise github_client.GitHubConfigError("Bad credentials")

    monkeypatch.setattr(github_client, "_graphql", always_fails)
    try:
        github_client._search_prs(
            "token", "org:acme is:pr", 100, github_client._SEARCH_QUERY, github_client._SEARCH_QUERY
        )
    except github_client.GitHubConfigError:
        pass
    else:  # pragma: no cover - the assertion is the raise
        raise AssertionError("a broken token must not look like an empty org")


# --------------------------------------------------------------------------- #
# The extended payload's switch and its degrade behaviour
# --------------------------------------------------------------------------- #


def test_the_switch_is_off_by_default_and_says_so(monkeypatch):
    monkeypatch.delenv("GITHUB_EXTENDED_PR_DATA", raising=False)
    assert github_client.extended_pr_data_enabled() is False

    monkeypatch.setattr(
        github_client,
        "fetch_open_prs",
        lambda token, org, max_prs, detail=True: pd.DataFrame([pr(1, "alice")]),
    )
    result = github_client.fetch_open_prs_extended("token", "acme")
    assert isinstance(result, github_client.PRFetch)
    assert result.degraded is True
    assert "switched off" in result.reason
    assert not result.frame.empty


def test_the_extended_query_degrades_to_the_lean_one_and_never_raises(monkeypatch):
    monkeypatch.setenv("GITHUB_EXTENDED_PR_DATA", "1")
    seen: list[str] = []

    def fake_graphql(token, query, variables):
        seen.append(query)
        if query is github_client._EXTENDED_HYGIENE_SEARCH_QUERY:
            raise github_client.GitHubConfigError("API rate limit exceeded")
        return {
            "search": {
                "issueCount": 1,
                "pageInfo": {"hasNextPage": False, "endCursor": None},
                "nodes": [{"number": 1, "author": {"login": "alice"}}],
            }
        }

    monkeypatch.setattr(github_client, "_graphql", fake_graphql)
    result = github_client.fetch_open_prs_extended("token", "acme")
    assert isinstance(result, github_client.PRFetch)
    assert result.degraded is True
    assert result.reason  # a non-empty, human reason
    assert list(result.frame["number"]) == [1]
    assert seen == [
        github_client._EXTENDED_HYGIENE_SEARCH_QUERY,
        github_client._HYGIENE_SEARCH_QUERY,
    ]


def test_an_outright_outage_still_returns_a_value_not_an_exception(monkeypatch):
    monkeypatch.setenv("GITHUB_EXTENDED_PR_DATA", "1")

    def always_fails(token, query, variables):
        raise github_client.GitHubConfigError("Bad credentials")

    monkeypatch.setattr(github_client, "_graphql", always_fails)
    result = github_client.fetch_open_prs_extended("token", "acme")
    assert isinstance(result, github_client.PRFetch)
    assert result.degraded is True
    assert "unavailable" in result.reason.lower()
    assert result.frame.empty


def test_the_extended_read_is_cached_for_at_least_an_hour(monkeypatch):
    monkeypatch.setenv("GITHUB_EXTENDED_PR_DATA", "1")
    calls = {"n": 0}

    def fake_graphql(token, query, variables):
        calls["n"] += 1
        return {
            "search": {
                "issueCount": 1,
                "pageInfo": {"hasNextPage": False, "endCursor": None},
                "nodes": [{"number": 1, "author": {"login": "alice"}}],
            }
        }

    monkeypatch.setattr(github_client, "_graphql", fake_graphql)
    first = github_client.fetch_open_prs_extended("token", "acme")
    second = github_client.fetch_open_prs_extended("token", "acme")
    assert first.degraded is False
    assert calls["n"] == 1  # the second call was served from cache
    assert second.frame.equals(first.frame)
    assert github_client._EXTENDED_CACHE_TTL_SECONDS >= 3600.0


def test_a_successful_extended_read_carries_the_new_columns(monkeypatch):
    monkeypatch.setenv("GITHUB_EXTENDED_PR_DATA", "1")

    def fake_graphql(token, query, variables):
        return {
            "search": {
                "issueCount": 1,
                "pageInfo": {"hasNextPage": False, "endCursor": None},
                "nodes": [
                    {
                        "number": 1,
                        "author": {"login": "alice"},
                        "createdAt": "2026-08-01T09:00:00Z",
                        "updatedAt": "2026-08-01T09:00:00Z",
                        "draftRequestTimeline": {
                            "nodes": [
                                {
                                    "__typename": "ReviewRequestedEvent",
                                    "createdAt": "2026-08-01T10:00:00Z",
                                    "requestedReviewer": {"__typename": "User", "login": "bob"},
                                },
                                {
                                    "__typename": "ConvertToDraftEvent",
                                    "createdAt": "2026-08-01T11:00:00Z",
                                },
                            ]
                        },
                    }
                ],
            }
        }

    monkeypatch.setattr(github_client, "_graphql", fake_graphql)
    result = github_client.fetch_open_prs_extended("token", "acme")
    assert result.degraded is False
    row = result.frame.iloc[0]
    assert bool(row["extended_fetched"]) is True
    assert row["timeline_events"] == [
        {"type": "review_requested", "created_at": "2026-08-01T10:00:00Z", "requested_reviewer": "bob"},
        {"type": "converted_to_draft", "created_at": "2026-08-01T11:00:00Z", "requested_reviewer": None},
    ]


def test_every_rollup_survives_an_empty_frame():
    empty = pd.DataFrame()
    for call in (
        pr_quality.size_bands,
        pr_quality.classify_sizes,
        pr_quality.devin_findings,
        pr_quality.devin_findings_by_author,
        pr_quality.review_citizenship,
        pr_quality.self_merge,
        pr_quality.flag_self_merges,
        pr_quality.abandoned_rate,
        pr_quality.traceability,
        pr_quality.unprompted_reviews,
    ):
        out = call(empty)
        assert isinstance(out, pd.DataFrame) and out.empty
    pairs, people = pr_quality.reciprocity(empty)
    assert pairs.empty and people.empty
    detail, outcome = pr_quality.draft_transitions(empty)
    assert detail.empty and outcome.empty
    assert pr_quality.delivered_points(empty) == []


def test_a_refused_read_repeats_what_github_said(monkeypatch):
    """``403 Client Error`` alone cannot tell a permission from an IP allow list."""

    class Refusal:
        status_code = 403
        text = (
            '{"message":"Although you appear to have the correct authorization '
            'credentials, the organization has an IP allow list enabled"}'
        )

    monkeypatch.setattr(github_client.requests, "post", lambda *a, **k: Refusal())
    with pytest.raises(github_client.GitHubConfigError) as raised:
        github_client._graphql("t", "{viewer{login}}", {})
    assert "403" in str(raised.value)
    assert "IP allow list" in str(raised.value)


@pytest.fixture(autouse=True)
def _open_door():
    """The shared throttle door is module state; no test inherits another's."""
    github_client._NOT_BEFORE = 0.0
    github_client._BUDGET.deadline = None
    github_client._clear_extended_pr_cache()
    yield
    github_client._NOT_BEFORE = 0.0
    github_client._BUDGET.deadline = None
    github_client._clear_extended_pr_cache()


@pytest.fixture
def clock(monkeypatch):
    """A sleep that costs no wall clock but still moves the clock it is measured by."""
    slept: list[float] = []
    now = [1000.0]

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(github_client.time, "sleep", sleep)
    monkeypatch.setattr(github_client.time, "monotonic", lambda: now[0])
    return slept


class _Response:
    def __init__(self, status_code: int, text: str = "", headers: dict | None = None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def json(self) -> dict:
        return {"data": {"ok": True}}


def test_a_throttled_read_waits_and_asks_again(monkeypatch, clock):
    """A secondary limit is a wait, not an outage: the page must not go blank."""
    replies = [
        _Response(403, '{"message":"You have exceeded a secondary rate limit"}',
                  {"retry-after": "1"}),
        _Response(200),
    ]
    monkeypatch.setattr(
        github_client.requests, "post", lambda *a, **k: replies.pop(0)
    )
    assert github_client._graphql("t", "{}", {}) == {"ok": True}
    assert sum(clock) == pytest.approx(1.0)


def test_a_refusal_that_waiting_cannot_fix_is_not_retried(monkeypatch):
    """A missing permission repeated four times is four wasted requests."""
    calls: list[int] = []

    def one(*a, **k):
        calls.append(1)
        return _Response(403, '{"message":"Resource not accessible by personal '
                              'access token"}')

    monkeypatch.setattr(github_client.requests, "post", one)
    monkeypatch.setattr(github_client.time, "sleep", lambda s: None)
    with pytest.raises(github_client.GitHubConfigError) as raised:
        github_client._graphql("t", "{}", {})
    assert len(calls) == 1
    assert "not accessible" in str(raised.value)


def test_reads_are_serialised_so_a_burst_cannot_trip_the_limit(monkeypatch):
    """Concurrency on one token is the thing GitHub refuses."""
    import threading as _t
    import time as time_module

    overlapping = []
    live = {"n": 0}
    guard = _t.Lock()

    def post(*a, **k):
        with guard:
            live["n"] += 1
            overlapping.append(live["n"])
        time_module.sleep(0.01)
        with guard:
            live["n"] -= 1
        return _Response(200)

    monkeypatch.setattr(github_client.requests, "post", post)
    threads = [
        _t.Thread(target=lambda: github_client._graphql("t", "{}", {}))
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max(overlapping) == 1


def test_a_minute_long_wait_is_waited_out_not_discarded(monkeypatch, clock):
    """GitHub asks for 60s; retrying at 2s spends every attempt inside the ban."""
    throttle = _Response(
        403, '{"message":"secondary rate limit"}', {"retry-after": "60"}
    )
    replies = [throttle, _Response(200)]
    monkeypatch.setattr(
        github_client.requests, "post", lambda *a, **k: replies.pop(0)
    )
    github_client._graphql("t", "{}", {})
    assert sum(clock) == pytest.approx(github_client._RETRY_CEILING_SECONDS)


def test_waiting_stops_at_the_budget_rather_than_hanging_the_page(monkeypatch, clock):
    """A throttled org gets a message, not a spinner."""
    monkeypatch.setattr(
        github_client.requests,
        "post",
        lambda *a, **k: _Response(
            403, '{"message":"secondary rate limit"}', {"retry-after": "30"}
        ),
    )
    with pytest.raises(github_client.GitHubConfigError):
        github_client._graphql("t", "{}", {})
    assert sum(clock) <= github_client._RETRY_BUDGET_SECONDS


def test_one_reader_being_throttled_stops_the_others_asking(monkeypatch, clock):
    """The limit belongs to the token, so the backoff has to as well."""
    throttle = _Response(
        403, '{"message":"secondary rate limit"}', {"retry-after": "5"}
    )
    monkeypatch.setattr(
        github_client.requests,
        "post",
        lambda *a, **k: throttle if not clock else _Response(200),
    )
    github_client._graphql("t", "{}", {})
    assert clock == [pytest.approx(5.0)]

    # A second reader, arriving while the door the first one shut is still closed,
    # waits for it rather than spending a request finding out it is closed.
    github_client._hold_off(7.0)
    clock.clear()
    github_client._graphql("t", "{}", {})
    assert clock == [pytest.approx(7.0)]


def test_an_hourly_quota_that_cannot_refill_in_time_is_not_waited_on(
    monkeypatch, clock
):
    """Sleeping 30s against a 40-minute reset wastes the budget and still fails."""
    monkeypatch.setattr(
        github_client.requests,
        "post",
        lambda *a, **k: _Response(
            403,
            '{"message":"API rate limit exceeded for user ID 1"}',
            {
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": str(int(time_module.time()) + 2400),
            },
        ),
    )
    with pytest.raises(github_client.GitHubConfigError):
        github_client._graphql("t", "{}", {})
    assert clock == []


def test_a_reader_already_queued_does_not_ask_through_a_shut_door(monkeypatch, clock):
    """The door is checked with the lock held, so the queue waits with the rest."""
    github_client._hold_off(11.0)
    monkeypatch.setattr(
        github_client.requests, "post", lambda *a, **k: _Response(200)
    )
    github_client._graphql("t", "{}", {})
    assert clock == [pytest.approx(11.0)]


def test_giving_up_still_shuts_the_door_behind_it(monkeypatch, clock):
    """The window outlives the reader that found it, so the others must see it."""
    monkeypatch.setattr(
        github_client.requests,
        "post",
        lambda *a, **k: _Response(
            403, '{"message":"secondary rate limit"}', {"retry-after": "60"}
        ),
    )
    with pytest.raises(github_client.GitHubConfigError):
        github_client._graphql("t", "{}", {})
    with github_client._STATE_LOCK:
        assert github_client._NOT_BEFORE > github_client.time.monotonic()


def test_a_reader_who_cannot_afford_the_wait_does_not_ask_anyway(monkeypatch, clock):
    """Half a wait then a request is a request made during the ban."""
    asked: list[int] = []
    monkeypatch.setattr(
        github_client.requests,
        "post",
        lambda *a, **k: asked.append(1) or _Response(200),
    )
    github_client._hold_off(github_client._RETRY_BUDGET_SECONDS + 30)
    with pytest.raises(github_client.GitHubConfigError, match="throttling"):
        github_client._graphql("t", "{}", {})
    assert asked == []
    assert clock == []


def test_the_pages_of_one_read_share_one_budget(monkeypatch, clock):
    """Ten pages with an allowance each is minutes; one read has one allowance."""
    replies = [
        _Response(403, '{"message":"secondary rate limit"}', {"retry-after": "60"}),
        _Response(200),
    ]
    monkeypatch.setattr(
        github_client.requests, "post", lambda *a, **k: replies.pop(0)
    )
    with github_client._read_budget():
        github_client._graphql("t", "{}", {})  # spends 60 of the 75 waiting
        github_client._hold_off(60.0)
        # The next page of the same read cannot afford another minute, so it is
        # abandoned rather than starting its own budget over.
        with pytest.raises(github_client.GitHubConfigError, match="throttling"):
            github_client._graphql("t", "{}", {})
    assert clock == [pytest.approx(60.0)]


def test_a_gateway_error_is_asked_again_not_reported_as_an_outage(monkeypatch, clock):
    """GitHub's edge answers a burst with nginx's 502 as readily as with a 403."""
    replies = [_Response(502, "<html>502 Bad Gateway</html>"), _Response(200)]
    monkeypatch.setattr(
        github_client.requests, "post", lambda *a, **k: replies.pop(0)
    )
    assert github_client._graphql("t", "{}", {}) == {"ok": True}
    assert clock == [pytest.approx(github_client._RETRY_BACKOFF_SECONDS)]


def test_the_door_is_shut_before_the_next_reader_is_let_through(monkeypatch, clock):
    """Shut after the lock, the reader queued behind asks inside the window."""
    doors: list[float] = []
    original = github_client._hold_off

    def hold_off(seconds: float) -> None:
        assert github_client._REQUEST_LOCK.locked(), "door shut after the lock"
        doors.append(seconds)
        original(seconds)

    monkeypatch.setattr(github_client, "_hold_off", hold_off)
    monkeypatch.setattr(
        github_client.requests,
        "post",
        lambda *a, **k: _Response(
            403, '{"message":"secondary rate limit"}', {"retry-after": "30"}
        ),
    )
    with pytest.raises(github_client.GitHubConfigError):
        github_client._graphql("t", "{}", {})
    assert doors == [pytest.approx(30.0), pytest.approx(30.0), pytest.approx(30.0)]
