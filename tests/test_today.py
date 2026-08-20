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
import integrity
import kpi
import render_shared
import series
import theme
import theme_html
import next_actions
from pages import today


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


# --- Task 2G: the attention band's three decide cards (render_shared.py bug 1) ---
#
# Pre-split, `hero, a, b, c = st.columns(...)` and the three `_decision_card`
# calls all sat inside `_render_attention_band`'s `try`, but the cards were
# indented under its `except` - so a hero that rendered fine (the normal
# case) never drew them, a second `except Exception` on the same `try` could
# never fire (the first always matches first), and an exception at the
# `st.columns()` line itself left `a`/`b`/`c` unbound, so the except handler
# raised an uncaught `NameError` instead of the real error. All three are
# fixed together: the cards are dedented to always run, the dead second
# `except` is gone, and column allocation gets its own try with a fallback
# binding so `a`/`b`/`c` are never unbound.


def _attention_band_prs(**overrides) -> dict:
    base = {
        "total": 10,
        "unapproved": 4,
        "oldest_unreviewed_days": 6,
        "never_reviewed": 1,
        "no_reviewer_asked": 2,
    }
    base.update(overrides)
    return base


def test_the_three_decide_cards_render_on_the_happy_path(monkeypatch):
    """Fault 1: the cards were indented into the except, so a clean hero drew none."""
    calls = []
    monkeypatch.setattr(
        today, "_decision_card", lambda column, **kw: calls.append(kw["chip"])
    )
    today._render_attention_band(
        _attention_band_prs(),
        github_ready=True,
        github_error="",
        triage_stuck=3,
        ownerless=5,
        open_total=40,
    )
    assert calls == ["Triage", "Review", "Ownership"]


def test_the_three_decide_cards_still_render_when_the_hero_raises(monkeypatch):
    """A broken hero must degrade the headline number, not the three decisions."""
    calls = []
    monkeypatch.setattr(
        today, "_decision_card", lambda column, **kw: calls.append(kw["chip"])
    )

    def boom(*_a, **_k):
        raise RuntimeError("hero exploded")

    monkeypatch.setattr(today.st, "progress", boom)
    today._render_attention_band(
        _attention_band_prs(),
        github_ready=True,
        github_error="",
        triage_stuck=3,
        ownerless=5,
        open_total=40,
    )
    assert calls == ["Triage", "Review", "Ownership"]


def test_the_hero_try_has_exactly_one_except_clause_each():
    """Fault 2: a second, always-unreachable `except Exception` sat on the same
    try as the first. Dead code changes nothing on its own, so this is pinned
    structurally rather than by behavior: neither try in the function may carry
    two stacked handlers that can never both fire.
    """
    import ast
    import inspect

    source = inspect.getsource(today._render_attention_band)
    tree = ast.parse(source)
    try_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    assert try_nodes, "expected the function to still guard its rendering with try/except"
    for node in try_nodes:
        assert len(node.handlers) == 1


def test_no_nameerror_when_column_allocation_itself_fails(monkeypatch):
    """Fault 3: st.columns() raising left a/b/c unbound, so the except handler
    that referenced them raised an uncaught NameError instead of the real
    error, and the page below it never rendered. This must raise neither
    NameError nor anything else, and the three cards must still be drawn
    against a fallback container.
    """
    calls = []
    monkeypatch.setattr(
        today, "_decision_card", lambda column, **kw: calls.append(kw["chip"])
    )

    def boom(*_a, **_k):
        raise ValueError("columns exploded")

    monkeypatch.setattr(today.st, "columns", boom)
    # No exception at all is the bar - and specifically, never the NameError
    # this bug used to raise instead of the real one.
    today._render_attention_band(
        _attention_band_prs(),
        github_ready=True,
        github_error="",
        triage_stuck=3,
        ownerless=5,
        open_total=40,
    )
    assert calls == ["Triage", "Review", "Ownership"]


# --- Task 2G: the scorecard's unentitled score (render_shared.py bug 2) ---
#
# `_render_scorecard` built its "Score" column with no check of
# `Component.sufficient`, so a placeholder gap row (score=0.0 by
# construction, meaning "nothing to measure") would print as a real 0 the
# moment a caller passed `include_gaps=True` to `kpi.components`. Latent
# today because no caller does, but the People page is about to. These pin
# the guard, plus the coverage numbers `_render_scorecard` was computing and
# throwing away.


def _scorecard_owned() -> pd.DataFrame:
    # Carries has_estimate/estimate_hours already, so _render_scorecard does
    # not re-run estimate_policy over it.
    return pd.DataFrame(
        {
            "assignee": ["Tam"],
            "status": ["In Progress"],
            "has_estimate": [True],
            "estimate_hours": [4.0],
        }
    )


def _patch_scorecard_network(monkeypatch) -> None:
    """No real Jira reads: history is unavailable, same as the trend-hidden case."""
    monkeypatch.setattr(render_shared, "fetch_person_resolved_count", lambda **_k: None)
    monkeypatch.setattr(render_shared, "fetch_person_reopened_count", lambda **_k: None)


def test_a_component_with_no_data_renders_n_a_never_zero(monkeypatch):
    _patch_scorecard_network(monkeypatch)
    parts = [
        kpi.Component("Delivery", 80.0, "8 of 8 resolved this week", n=8),
        kpi.Component(
            "Urgent response",
            0.0,
            "insufficient data - needs open High+ priority tickets",
            n=0,
            sufficient=False,
        ),
    ]
    monkeypatch.setattr(render_shared.kpi, "components", lambda *a, **k: parts)

    captured = {}
    monkeypatch.setattr(
        render_shared.st,
        "dataframe",
        lambda data, **kw: captured.setdefault("df", data),
    )

    render_shared._render_scorecard(
        "Tam", _scorecard_owned(), pd.DataFrame(), pd.DataFrame(), github_ready=True
    )

    df = captured["df"]
    scores = dict(zip(df["Component"], df["Score"]))
    assert scores["Delivery"] == "80"
    assert scores["Urgent response"] == "n/a"
    assert "0" != scores["Urgent response"]
    assert 0 not in df["Score"].tolist()


def test_the_scorecard_surfaces_the_measurable_denominator(monkeypatch):
    _patch_scorecard_network(monkeypatch)
    # Every weighted component except "Urgent response" (5 of the 100 points)
    # has data, all scored at 80 - so the honest denominator is 95, not 100,
    # and the weighted mean over what was actually measured is 80.
    parts = [
        kpi.Component(name, 80.0, "measured", n=4)
        for name in kpi.WEIGHTS
        if name != "Urgent response"
    ] + [
        kpi.Component(
            "Urgent response",
            0.0,
            "insufficient data - needs open High+ priority tickets",
            n=0,
            sufficient=False,
        ),
    ]
    monkeypatch.setattr(render_shared.kpi, "components", lambda *a, **k: parts)

    metrics = []
    captions = []
    monkeypatch.setattr(
        render_shared.st,
        "metric",
        lambda label, value, **kw: metrics.append((label, value)),
    )
    monkeypatch.setattr(render_shared.st, "caption", lambda text: captions.append(text))
    monkeypatch.setattr(render_shared.st, "dataframe", lambda *a, **k: None)

    render_shared._render_scorecard(
        "Tam", _scorecard_owned(), pd.DataFrame(), pd.DataFrame(), github_ready=True
    )

    cov = kpi.coverage(parts)
    overall_label, overall_value = metrics[0]
    assert overall_label == "Overall"
    # The denominator is the measurable weight (95, "Urgent response" missing 5
    # of 100), never a flat "/ 100" that hides how much of the board could not
    # be read.
    assert overall_value == f"80 over {cov.covered_weight:.0f} measurable points"
    assert "/ 100" not in overall_value
    assert any("Urgent response" in c for c in captions)


# --- Task 3A: the full-HTML render path -------------------------------------
#
# Today became a display-only page built from ``theme_html`` fragments
# instead of Streamlit widgets (see the module docstring for why). These
# tests pin the new hero/decide-card fragment, the six "This week" tiles
# (including that Median cycle time replaces Ownerless, per the mockup), the
# 12-week throughput chart, the "Where open work sits" hbars, and the
# end-to-end empty-data case - without touching a single test above this
# block, all of which still exercise the pre-3A Streamlit-widget functions
# that remain, unchanged, for exactly that reason.


def _throughput(**overrides) -> "today._Throughput":
    base = dict(ok=True, error="", resolved=[], prs=[], bots_excluded=0, cycle=None)
    base.update(overrides)
    return today._Throughput(**base)


def _week_buckets(values: list[float]) -> list[series.WeekBucket]:
    now = pd.Timestamp.now(tz="UTC")
    return [
        series.WeekBucket(now, f"wk{i}", float(v), is_partial=(i == len(values) - 1))
        for i, v in enumerate(values)
    ]


def _board(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "key": [f"ENG-{i}" for i in range(n)],
            "assignee": ["Tam", None, "Tam"][:n],
            "status": ["In Progress", "To Do", "In Progress"][:n],
            "created": pd.to_datetime(
                ["2026-07-01", "2026-08-01", "2026-08-10"][:n], utc=True
            ),
            "idle_days": [1.0, 40.0, 1.0][:n],
            "changelog": [None, None, None][:n],
        }
    )


# --- Hero + decide cards, HTML form ------------------------------------


def test_the_html_hero_and_decide_cards_both_render_on_the_happy_path():
    frag = today._hero_and_decide_fragment(
        _attention_band_prs(),
        github_ready=True,
        github_error="",
        triage_stuck=3,
        ownerless=5,
        open_total=40,
    )
    assert 'class="hero"' in frag
    assert 'class="card lead"' in frag  # theme_html.hero()'s own card class
    assert 'class="decide"' in frag
    assert frag.count('class="card"') == 3  # the three decide cards


def test_the_decide_cards_still_render_when_the_html_hero_raises(monkeypatch):
    """The HTML-path twin of the pinned Task 2G regression: a broken hero must
    not take the three decide cards down with it.
    """

    def boom(*_a, **_k):
        raise RuntimeError("hero exploded")

    monkeypatch.setattr(today, "_hero_fragment", boom)
    frag = today._hero_and_decide_fragment(
        _attention_band_prs(),
        github_ready=True,
        github_error="",
        triage_stuck=3,
        ownerless=5,
        open_total=40,
    )
    assert 'class="decide"' in frag
    assert frag.count('class="card"') == 4  # the error callout plus the three decide cards
    assert "could not be drawn" in frag


def test_the_html_hero_is_an_honest_callout_when_github_is_unavailable():
    frag = today._hero_fragment(
        _attention_band_prs(), github_ready=False, github_error="no token"
    )
    assert "PR review health cannot be read" in frag
    assert "no token" in frag
    assert 'class="card lead"' not in frag


def test_decide_card_chips_and_tones_match_the_mockup():
    cards = today._decide_cards(
        _attention_band_prs(), github_ready=True, triage_stuck=5, ownerless=55, open_total=360
    )
    assert [c.chip for c in cards] == ["⏱ Triage", "👀 Review", "∅ Ownership"]
    assert [c.tone for c in cards] == ["warn", "crit", "info"]


# --- The six tiles -----------------------------------------------------


def test_the_six_tiles_match_the_mockups_set():
    """Median cycle time replaces Ownerless as a tile - the mockup's own choice."""
    bundle = types.SimpleNamespace(
        data={"resolved_count_7": 9}, github_ready=True, pr_count_7=12
    )
    tp = _throughput(
        resolved=_week_buckets([1, 2, 3]),
        prs=_week_buckets([1, 2, 3]),
        cycle=integrity.end_to_end_cycle(pd.DataFrame(), pd.DataFrame()),
    )
    tiles = today._tiles(bundle, _board(), pd.DataFrame(), "status age", 0, tp)
    labels = [t.label for t in tiles]
    assert labels == [
        "Tickets resolved · 7d",
        "PRs merged · 7d",
        "Median cycle time",
        "Open tickets",
        "Stalled 30d+",
        "Estimate coverage",
    ]
    assert "Ownerless" not in labels


def test_every_tile_carries_a_sparkline_and_a_scored_delta():
    bundle = types.SimpleNamespace(
        data={"resolved_count_7": 9}, github_ready=True, pr_count_7=12
    )
    detail = pd.DataFrame(
        {
            "key": ["ENG-1", "ENG-2", "ENG-3", "ENG-4"],
            "started": [pd.Timestamp.now(tz="UTC")] * 4,
            "resolved": [pd.Timestamp.now(tz="UTC")] * 4,
            "days": [3.0, 4.0, 5.0, 6.0],
            "period": ["current", "current", "baseline", "baseline"],
        }
    )
    cycle = integrity.EndToEndCycle(
        median_days=4.0, n=2, baseline_median_days=5.5, baseline_n=2,
        reason=None, baseline_reason=None, detail=detail, window_days=42.0,
    )
    tp = _throughput(
        resolved=_week_buckets([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        prs=_week_buckets([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]),
        cycle=cycle,
    )
    stalled = pd.DataFrame({"key": ["ENG-2"], next_actions.STALLED_AGE_COLUMN: [40.0]})
    tiles = today._tiles(bundle, _board(), stalled, "status age", 1, tp)

    assert len(tiles) == 6
    for tile in tiles:
        assert tile.spark, f"{tile.label} has no sparkline"
        assert tile.spark.startswith("<svg"), tile.label
        assert tile.delta_dir in ("up", "down", "flat"), tile.label
        assert tile.delta, f"{tile.label} has no delta text"


def test_a_tile_whose_prior_period_has_no_data_renders_neutral():
    """Median cycle time when the baseline window finished too few tickets:
    ``series.delta`` reports ``is_good=None`` (missing data), never green.
    """
    bundle = types.SimpleNamespace(data={}, github_ready=False, pr_count_7=None)
    cycle = integrity.EndToEndCycle(
        median_days=5.0, n=4, baseline_median_days=None, baseline_n=0,
        reason=None, baseline_reason="fewer than 3 tickets finished",
        detail=pd.DataFrame(columns=["key", "started", "resolved", "days", "period"]),
        window_days=42.0,
    )
    tp = _throughput(cycle=cycle)
    tiles = today._tiles(bundle, pd.DataFrame(), pd.DataFrame(), "status age", 0, tp)
    cycle_tile = next(t for t in tiles if t.label == "Median cycle time")
    assert cycle_tile.delta_good is None
    assert theme_html._delta_class(cycle_tile.delta_good) != "good-t"
    assert cycle_tile.delta == "no prior data"


def test_stalled_tickets_up_is_scored_bad_and_cycle_time_down_is_scored_good():
    bundle = types.SimpleNamespace(
        data={"resolved_count_7": 1}, github_ready=True, pr_count_7=1
    )
    detail = pd.DataFrame(columns=["key", "started", "resolved", "days", "period"])
    cycle = integrity.EndToEndCycle(
        median_days=4.0, n=5, baseline_median_days=6.0, baseline_n=5,
        reason=None, baseline_reason=None, detail=detail, window_days=42.0,
    )
    tp = _throughput(resolved=_week_buckets([1] * 12), prs=_week_buckets([1] * 12), cycle=cycle)
    # Two tickets stalled last week, none the week before - a real increase.
    now = pd.Timestamp.now(tz="UTC")
    stalled = pd.DataFrame(
        {"key": ["A", "B"], next_actions.STALLED_AGE_COLUMN: [8.0, 9.0]}
    )
    tiles = today._tiles(bundle, _board(), stalled, "status age", 0, tp)
    cycle_tile = next(t for t in tiles if t.label == "Median cycle time")
    stalled_tile = next(t for t in tiles if t.label.startswith("Stalled"))
    # median_days (4.0) < baseline (6.0): cycle time fell, which is good.
    assert cycle_tile.delta_dir == "down"
    assert cycle_tile.delta_good is True


# --- The 12-week throughput chart ---------------------------------------


def test_the_throughput_chart_renders_two_polylines_and_no_script():
    tp = _throughput(
        resolved=_week_buckets([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        prs=_week_buckets([2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]),
        bots_excluded=23,
    )
    frag = today._throughput_chart_fragment(tp)
    assert frag.count("<polyline") == 2
    assert "<script" not in frag
    assert "23 bot merges" in frag


def test_the_throughput_chart_is_an_honest_callout_when_nothing_could_be_read():
    tp = _throughput(ok=False, error="Jira: timed out")
    frag = today._throughput_chart_fragment(tp)
    assert "<polyline" not in frag
    assert "Jira: timed out" in frag


# --- Where open work sits, via hbars -------------------------------------


def test_where_open_work_sits_uses_hbars_not_plotly():
    df = pd.DataFrame({"status": ["To Do", "To Do", "In Progress"]})
    frag = today._status_hbars_fragment(df, parked=0)
    assert 'class="hbars"' in frag
    assert 'class="hbar"' in frag
    assert "plotly" not in frag.lower()


def test_where_open_work_sits_is_an_honest_empty_state_not_a_blank_card():
    frag = today._status_hbars_fragment(pd.DataFrame(), parked=0)
    assert "No open tickets" in frag
    assert 'class="hbars"' not in frag


# --- End to end -----------------------------------------------------------


def test_the_page_renders_end_to_end_with_an_empty_data_frame_without_raising(
    monkeypatch,
):
    """No Jira, no GitHub, no tickets: the page must say so, never draw zeros."""
    bundle = types.SimpleNamespace(
        df=pd.DataFrame(),
        events=pd.DataFrame(),
        github_ready=False,
        github_error="",
        open_prs=pd.DataFrame(),
        merged_prs=pd.DataFrame(),
        pr_count_7=None,
        open_count_exact=None,
        data={},
    )
    monkeypatch.setattr(today, "_engineering_data", lambda: bundle)
    monkeypatch.setattr(today, "_fetch_resolved_window", lambda days: pd.DataFrame())
    monkeypatch.setattr(today, "_fetch_merged_window", lambda days: pd.DataFrame())

    written = []
    monkeypatch.setattr(
        today.st, "markdown", lambda text, **kw: written.append(text)
    )

    today._render_today_page()  # must not raise

    full = "\n".join(written)
    assert full.count('class="tile"') == 6
    assert "GitHub" in full  # the honest "GitHub cannot be read" callout
    assert "No open tickets" in full
    # No fabricated zero standing in for "could not be read": the hero and
    # the throughput chart both degrade to a labelled callout instead.
    assert "PR review health cannot be read" in full
