"""What the Code page claims, against what its data can support.

Every case here is a sentence an executive would act on: that merges went in
unapproved, that a queue is this long, that a figure covers the whole
organisation. A tile whose wording outruns its column is the failure being
pinned, not a crash.
"""

from __future__ import annotations

import pandas as pd

import app
import github_client


def _merged(authors, mergers, approvals) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "author": authors,
            "number": list(range(1, len(authors) + 1)),
            "url": [f"https://github.com/o/r/pull/{i}" for i in range(len(authors))],
            "merged_by": mergers,
            "merged_at": pd.to_datetime(["2026-08-01T00:00:00Z"] * len(authors)),
            "state": ["MERGED"] * len(authors),
            "base_branch": ["main"] * len(authors),
            "approving_reviews": approvals,
        }
    )


def test_the_merge_tile_counts_unapproved_merges_not_self_presses(monkeypatch):
    """Pressing merge on work a colleague approved is not a review failure.

    GitHub does not let an author approve their own PR, so the old "author
    approved own work" note described something impossible while colouring a
    normal merge as danger.
    """
    captured: dict = {}
    monkeypatch.setattr(app.theme_html, "tiles", lambda cards: captured.setdefault("cards", cards))
    monkeypatch.setattr(app.theme_html, "hbars", lambda *a, **k: None)

    # Two self-presses; only one of them went in with nobody else's approval.
    merged = _merged(["tam", "tam"], ["tam", "tam"], [1, 0])
    app._render_code_kpis(pd.DataFrame(), merged)

    label, value, note, accent = captured["cards"][4]
    assert "unapproved" in label.lower()
    assert value == "1"
    assert "nobody else approved" in note
    assert "2 pressed merge on their own PR" in note
    assert "approved own work" not in note
    assert accent == "danger"  # 1 of 2 merges is over the threshold


def test_the_open_pr_tile_does_not_promise_a_draft_count_it_cannot_have(monkeypatch):
    """``_open_query`` searches ``draft:false``, so "of N including drafts" repeated N."""
    captured: dict = {}
    monkeypatch.setattr(app.theme_html, "tiles", lambda cards: captured.setdefault("cards", cards))
    open_prs = pd.DataFrame(
        {
            "is_draft": [False, False],
            "approving_reviews": [0, 1],
            "total_reviews": [0, 1],
            "review_requests": [1, 1],
            "age_days": [3.0, 4.0],
        }
    )
    app._render_code_kpis(open_prs, pd.DataFrame())

    _, value, note, _ = captured["cards"][0]
    assert value == "2"
    assert "including drafts" not in note
    assert "excludes drafts" in note


def test_a_cut_queue_says_how_much_it_cut():
    """The hidden tail is the normal case here, so 25 rows must not read as all."""
    assert app._truncation_note(78, 25) == (
        "Showing the 25 oldest of 78. The other 53 are newer and still waiting "
        "— the queue is longer than this card."
    )


def test_a_queue_that_fits_says_nothing():
    assert app._truncation_note(25, 25) == ""


def test_excluded_repos_leave_the_search_queries_not_just_the_frames(monkeypatch):
    """A count from search cannot be filtered afterwards, so the query must exclude.

    Otherwise the Code page drops a repo's rows while the org-wide open and
    merged counts still include them, and the same population reads two ways.
    """
    monkeypatch.setenv("GITHUB_EXCLUDE_REPOS", "scratch, jira-evaluator")
    for query in (
        github_client._open_query("DrinkBetter-AI"),
        github_client._merged_query("DrinkBetter-AI", 30),
        github_client._closed_unmerged_query("DrinkBetter-AI", 30),
    ):
        assert "-repo:DrinkBetter-AI/scratch" in query
        assert "-repo:DrinkBetter-AI/jira-evaluator" in query


def test_an_owner_qualified_exclusion_is_kept_as_given(monkeypatch):
    monkeypatch.setenv("GITHUB_EXCLUDE_REPOS", "avosmod8/scratch")
    assert github_client.excluded_repos("DrinkBetter-AI") == ("avosmod8/scratch",)


def test_a_name_that_could_smuggle_a_qualifier_is_dropped(monkeypatch):
    """The value is interpolated into the search query, so it is validated."""
    monkeypatch.setenv("GITHUB_EXCLUDE_REPOS", "ok-repo, is:draft org:evil")
    assert github_client.excluded_repos("DrinkBetter-AI") == ("DrinkBetter-AI/ok-repo",)


def test_the_open_query_still_sorts_oldest_first_with_exclusions(monkeypatch):
    """The sort qualifier is not a repo filter and must survive the rewrite."""
    monkeypatch.setenv("GITHUB_EXCLUDE_REPOS", "scratch")
    assert github_client._open_query("DrinkBetter-AI").endswith("sort:created-asc")


def test_the_caption_names_what_was_excluded(monkeypatch):
    monkeypatch.setenv("GITHUB_EXCLUDE_REPOS", "scratch")
    monkeypatch.setenv("DASHBOARD_GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_ORG", "DrinkBetter-AI")
    names, caption = app._exclude_repos()
    assert names == frozenset({"scratch"})
    assert "`scratch`" in caption
    assert "every page" in caption


def test_no_exclusion_configured_is_a_silent_page(monkeypatch):
    monkeypatch.delenv("GITHUB_EXCLUDE_REPOS", raising=False)
    monkeypatch.setenv("DASHBOARD_GITHUB_TOKEN", "t")
    names, caption = app._exclude_repos()
    assert not names
    assert caption == ""
