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
from pages import code as code_page

DEVIN = "devin-ai-integration[bot]"


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
    monkeypatch.setattr(app.theme_html, "tiles", lambda cards, **_: captured.setdefault("cards", cards))
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
    monkeypatch.setattr(app.theme_html, "tiles", lambda cards, **_: captured.setdefault("cards", cards))
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


def test_another_owners_repo_is_not_treated_as_this_orgs(monkeypatch):
    """``-repo:someone-else/scratch`` never matches an ``org:`` search.

    So it must not drop this org's own ``scratch`` rows either, or the page's
    table and the org-wide counts describe different populations again.
    """
    monkeypatch.setenv("GITHUB_EXCLUDE_REPOS", "avosmod8/scratch")
    monkeypatch.setenv("DASHBOARD_GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_ORG", "DrinkBetter-AI")
    names, caption = app._exclude_repos()
    assert names == frozenset()
    assert caption == ""
    kept = app._team_prs(pd.DataFrame({"repo": ["scratch"], "author": ["tam"]}))
    assert list(kept["repo"]) == ["scratch"]


def test_a_malformed_org_does_not_replace_the_page_with_a_stack_trace(monkeypatch):
    """The page prints this caption before the check that reports config errors."""
    monkeypatch.setenv("GITHUB_EXCLUDE_REPOS", "scratch")
    monkeypatch.setenv("DASHBOARD_GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_ORG", "not a valid org")
    names, caption = app._exclude_repos()
    assert names == frozenset({"scratch"})  # falls back to the default org
    assert "`scratch`" in caption


def test_no_exclusion_configured_is_a_silent_page(monkeypatch):
    monkeypatch.delenv("GITHUB_EXCLUDE_REPOS", raising=False)
    monkeypatch.setenv("DASHBOARD_GITHUB_TOKEN", "t")
    names, caption = app._exclude_repos()
    assert not names
    assert caption == ""


# --------------------------------------------------------------------------- #
# Task 3D: the "of N including drafts" secondary count
# --------------------------------------------------------------------------- #


def test_the_open_pr_tile_says_the_real_total_when_it_is_known(monkeypatch):
    """The counterpart to the existing "does not promise a draft count" test."""
    captured: dict = {}
    monkeypatch.setattr(app.theme_html, "tiles", lambda cards, **_: captured.setdefault("cards", cards))
    open_prs = pd.DataFrame(
        {
            "is_draft": [False, False],
            "approving_reviews": [0, 1],
            "total_reviews": [0, 1],
            "review_requests": [1, 1],
            "age_days": [3.0, 4.0],
        }
    )
    code_page._render_code_kpis(open_prs, pd.DataFrame(), open_including_drafts=96)

    _, value, note, _ = captured["cards"][0]
    assert value == "2"
    assert "of 96 including drafts" in note


def test_no_token_means_the_drafts_total_is_absent_not_a_lying_zero():
    """Unreachable is unreachable — never rendered as "0 including drafts"."""
    assert code_page._open_pr_total_including_drafts(None, None) is None
    assert code_page._open_pr_total_including_drafts("", "DrinkBetter-AI") is None
    assert code_page._open_pr_total_including_drafts("t", "") is None


# --------------------------------------------------------------------------- #
# Task 3D: the extended-payload pool and its honest degradation
# --------------------------------------------------------------------------- #


def test_no_token_means_the_extended_pool_is_an_honest_degradation():
    pool, degraded, reason = code_page._extended_pr_pool(None, None)
    assert pool.empty
    assert degraded is True
    assert "token" in reason.lower()


# --------------------------------------------------------------------------- #
# Task 3D: unprompted reviews — the proactivity signal, its own column
# --------------------------------------------------------------------------- #


def test_unprompted_reviews_has_per_person_counts_and_linked_pr_evidence():
    pool = pd.DataFrame(
        [
            {
                "number": 11,
                "url": "https://github.com/o/r/pull/11",
                "author": "alice",
                "reviews": [
                    {
                        "reviewer": "bob",
                        "state": "APPROVED",
                        "submitted_at": "2026-08-01T09:00:00Z",
                        "body": "",
                    }
                ],
                # Nothing in the timeline asked bob for this review.
                "timeline_events": [],
            }
        ]
    )
    out = code_page._unprompted_reviews_rows(pool)
    assert list(out["reviewer"]) == ["bob"]
    assert int(out.loc[0, "unprompted_reviews"]) == 1
    assert out.loc[0, "prs"] == (11,)
    assert out.loc[0, "pr_refs"] == ((11, "https://github.com/o/r/pull/11"),)

    evidence = code_page._pr_evidence_html(out.loc[0, "pr_refs"])
    assert '<a href="https://github.com/o/r/pull/11">#11</a>' in evidence


def test_the_same_pr_number_in_two_repos_links_to_each_repos_own_pr():
    """A PR number is unique inside a repo, not across an org.

    Both PRs below are #42, in different repositories, both reviewed
    unprompted by the same person. The evidence has to carry two links to
    two different URLs. Resolving a bare number to "the first #42 we saw"
    - which is what this column used to do - sends a reader to the wrong
    repository's PR and shows one row where there are two.
    """
    def unprompted_pr(repo: str, author: str) -> dict:
        return {
            "number": 42,
            "url": f"https://github.com/o/{repo}/pull/42",
            "author": author,
            "reviews": [
                {
                    "reviewer": "bob",
                    "state": "APPROVED",
                    "submitted_at": "2026-08-01T09:00:00Z",
                    "body": "",
                }
            ],
            "timeline_events": [],
        }

    pool = pd.DataFrame([unprompted_pr("api", "alice"), unprompted_pr("web", "carol")])
    out = code_page._unprompted_reviews_rows(pool)
    assert int(out.loc[0, "unprompted_reviews"]) == 2
    assert out.loc[0, "pr_refs"] == (
        (42, "https://github.com/o/api/pull/42"),
        (42, "https://github.com/o/web/pull/42"),
    )

    evidence = code_page._pr_evidence_html(out.loc[0, "pr_refs"])
    assert '<a href="https://github.com/o/api/pull/42">#42</a>' in evidence
    assert '<a href="https://github.com/o/web/pull/42">#42</a>' in evidence


def test_evidence_falls_back_to_a_bare_number_when_the_url_is_unknown():
    assert code_page._pr_evidence_html(((5, ""),)) == "#5"
    assert "none" in code_page._pr_evidence_html(())


def test_unprompted_reviews_column_renders_an_honest_empty_state(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(code_page.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(code_page.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(code_page.st, "info", lambda msg: captured.setdefault("info", msg))
    code_page._render_unprompted_reviews(pd.DataFrame())
    assert "extended" in captured["info"].lower()


# --------------------------------------------------------------------------- #
# Task 3D: Merged 30d and the exempt role chip on the Devin findings table
# --------------------------------------------------------------------------- #


def _judged_row(number: int, author: str) -> dict:
    return {
        "author": author,
        "number": number,
        "url": f"https://github.com/o/r/pull/{number}",
        "reviews": [
            {
                "reviewer": DEVIN,
                "state": "CHANGES_REQUESTED",
                "submitted_at": "2026-08-01T12:00:00Z",
                "body": "",
            }
        ],
    }


def test_devin_findings_rows_adds_merged_30d_and_marks_the_exec_exemption():
    """``avosmod8`` is the roster's baked-in login for the exec role (Angel
    Vossough); an unmapped login has no role at all, so it is never exempt.
    """
    judged_pool = pd.DataFrame(
        [_judged_row(n, "avosmod8") for n in range(1, 6)]
        + [_judged_row(6, "regular_dev")]
    )
    merged_prs = pd.DataFrame({"author": ["avosmod8", "avosmod8", "regular_dev"]})

    findings = code_page._devin_findings_rows(judged_pool, merged_prs)
    by_author = findings.set_index("author")

    assert bool(by_author.loc["avosmod8", "exempt"]) is True
    assert bool(by_author.loc["regular_dev", "exempt"]) is False
    assert int(by_author.loc["avosmod8", "merged_30d"]) == 2
    assert int(by_author.loc["regular_dev", "merged_30d"]) == 1


def test_findings_table_renders_the_exempt_chip_only_for_the_flagged_row():
    findings = pd.DataFrame(
        {
            "author": ["avosmod8", "regular_dev"],
            "prs_judged": [6, 6],
            "prs_changes_requested": [1, 4],
            "changes_requested_share": [16.0, 66.0],
            "merged_30d": [3, 9],
            "exempt": [True, False],
        }
    )
    html_out = code_page._findings_table_html(findings)
    assert html_out.count("rolechip") == 1
    assert "Exempt" in html_out
    assert ">3<" in html_out and ">9<" in html_out  # both Merged 30d values present


# --------------------------------------------------------------------------- #
# Task 3D: the draft-hiding row (KPI_SPEC exploit #6)
# --------------------------------------------------------------------------- #


def _timeline_pr(number: int, events: list) -> dict:
    return {
        "number": number,
        "url": f"https://github.com/o/r/pull/{number}",
        "author": "alice",
        "timeline_events": events,
    }


def test_draft_hiding_row_flags_only_the_after_request_case():
    """One PR hid in draft after a request; one was simply opened as a draft
    and marked ready once — the second is not this at all."""
    pool = pd.DataFrame(
        [
            _timeline_pr(
                1,
                [
                    {
                        "type": "review_requested",
                        "created_at": "2026-08-01T09:00:00Z",
                        "requested_reviewer": "bob",
                    },
                    {
                        "type": "converted_to_draft",
                        "created_at": "2026-08-01T10:00:00Z",
                        "requested_reviewer": None,
                    },
                ],
            ),
            _timeline_pr(
                2,
                [
                    {
                        "type": "ready_for_review",
                        "created_at": "2026-08-01T09:00:00Z",
                        "requested_reviewer": None,
                    }
                ],
            ),
        ]
    )
    flagged = code_page._draft_hiding_rows(pool)
    assert list(flagged["number"]) == [1]
    assert int(flagged.loc[0, "draft_round_trips"]) == 1


def test_draft_hiding_row_renders_the_innocent_empty_state_when_nothing_is_flagged(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(code_page.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(code_page.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(code_page.st, "success", lambda msg: captured.setdefault("success", msg))
    code_page._render_draft_hiding_row(pd.DataFrame())
    assert "not evidence" in captured["success"]


# --------------------------------------------------------------------------- #
# Task 3D: renders end to end with no GitHub data at all
# --------------------------------------------------------------------------- #


def test_the_page_renders_end_to_end_with_no_github_data_and_says_so_honestly(monkeypatch):
    """``github_ready`` true, every PR frame empty, no token reachable for the
    two new direct GitHub reads. Nothing here may raise, and the degraded
    extended payload must be said out loud through a callout rather than
    silently showing lean numbers as complete.
    """

    class _Bundle:
        df = pd.DataFrame()
        github_ready = True
        github_error = ""
        open_prs = pd.DataFrame()
        merged_prs = pd.DataFrame()

    class _View:
        selected_assignees = None

    monkeypatch.setattr(code_page, "_engineering_context", lambda: (_Bundle(), _View(), object()))
    monkeypatch.setattr(code_page, "_download_report", lambda *a, **k: None)
    monkeypatch.setattr(code_page, "_render_pr_hygiene", lambda *a, **k: None)
    monkeypatch.setattr(github_client, "load_github_env", lambda: None)

    rendered: list = []
    monkeypatch.setattr(
        code_page.theme_html, "render", lambda *frags: rendered.extend(frags)
    )

    code_page._render_code_page()  # must not raise

    assert any("token" in frag.lower() for frag in rendered)
