"""The People page: the roster table, the role filter, and the per-person
scorecard.

Two layers of test here, matching how ``pages/people.py`` is built: small
pure functions (cell formatting, the cohort wiring) are tested directly with
crafted ``pandas`` objects, and a handful of full-page tests exercise
``_render_people_page`` against a stub ``_EngineeringData``/``_EngineeringView``
bundle the same way ``tests/test_delivery_page.py`` does - there is no
Streamlit test harness in this repo, so every ``st.markdown`` fragment the
page writes is captured and asserted on as HTML.
"""

from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import unquote

import pandas as pd

import data_layer
import integrity
import roles
import theme_html
from pages import people as people_page

# ---------------------------------------------------------------------------
# Small builders
# ---------------------------------------------------------------------------


def _hist(ts: str, author: str, field: str, frm: str, to: str) -> dict:
    return {
        "created": ts,
        "author": {"displayName": author},
        "items": [{"field": field, "fromString": frm, "toString": to}],
    }


def _ago(now: pd.Timestamp, days: float) -> str:
    return (now - pd.Timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.000+0000")


def _ticket(
    key: str,
    assignee: str,
    *histories: dict,
    status: str = "In Progress",
    priority: str = "Medium",
    idle_days: float = 1.0,
    carry_over_count: int = 0,
    issue_type: str = "Story",
    summary: str = "Fix the thing that broke in production last week",
    description: str = "A description long enough to clear the quality bar. " * 3,
) -> dict:
    return {
        "key": key,
        "assignee": assignee,
        "status": status,
        "priority": priority,
        "idle_days": idle_days,
        "carry_over_count": carry_over_count,
        "issue_type": issue_type,
        "summary": summary,
        "description": description,
        "created": "2026-01-01T00:00:00.000+0000",
        "changelog": {"histories": list(histories)},
    }


def _row(**overrides) -> pd.Series:
    base = {
        "person": "Someone",
        "role": "backend",
        "score": pd.NA,
        "n": pd.NA,
        "delivered_points": pd.NA,
        "n_delivered_points": 0,
        "trivial_share": pd.NA,
        "n_trivial_share": 0,
        "cycle_median": pd.NA,
        "n_cycle_median": 0,
        "reviews_given": pd.NA,
        "n_reviews_given": 0,
        "ttfr_hours": pd.NA,
        "n_ttfr_hours": 0,
        "estimate_ratio": pd.NA,
        "n_estimate_ratio": 0,
        "estimate_iqr": pd.NA,
        "n_estimate_iqr": 0,
        "flag_count": pd.NA,
        "n_flag_count": 0,
        "flag_severity": pd.NA,
        "n_flag_severity": 0,
        "measurable_pct": 0.0,
        "no_score_reason": "",
    }
    base.update(overrides)
    return pd.Series(base)


# ---------------------------------------------------------------------------
# Cell formatting - NA vs a real zero, never coalesced
# ---------------------------------------------------------------------------


def test_score_cell_renders_a_scorebar_for_a_real_score():
    out = people_page._score_cell_html(_row(score=81.0, n=31))
    assert 'class="scorebar"' in out
    assert "<b>81</b>" in out
    assert "n=31" in out


def test_score_cell_never_shows_zero_for_an_insufficient_person():
    out = people_page._score_cell_html(
        _row(
            score=pd.NA,
            role="backend",
            measurable_pct=10.0,
            no_score_reason=(
                "not scored: only 10 of 100 points of weight had any data "
                "(60 needed). Missing: Rework, Estimates."
            ),
        )
    )
    assert "no score" in out
    assert "10% measurable" in out
    assert ">0<" not in out
    assert "<b>0</b>" not in out


def test_score_cell_renders_exec_sentinel():
    out = people_page._score_cell_html(
        _row(score=pd.NA, role="exec", no_score_reason=roles.EXEC_REASON)
    )
    assert "exec — not scored" in out


def test_score_cell_renders_no_rubric_sentinel():
    out = people_page._score_cell_html(
        _row(
            score=pd.NA,
            role="seo",
            no_score_reason="no rubric defined for role 'seo'",
        )
    )
    assert "no rubric defined" in out


def test_score_cell_renders_role_unknown_sentinel():
    out = people_page._score_cell_html(
        _row(score=pd.NA, role=None, no_score_reason="role unknown")
    )
    assert "role unknown" in out


def test_delivered_cell_distinguishes_missing_from_a_real_zero():
    missing = people_page._delivered_cell_html(_row(delivered_points=pd.NA))
    assert missing == "—"
    real_zero = people_page._delivered_cell_html(
        _row(delivered_points=0.0, n_trivial_share=0, trivial_share=pd.NA)
    )
    assert "0 pts" in real_zero
    assert real_zero != "—"


def test_flags_cell_gray_zero_vs_dash_for_no_changelog():
    zero = people_page._flags_cell_html(_row(flag_count=0, flag_severity=0.0))
    assert 'class="chip gray">0<' in zero
    dash = people_page._flags_cell_html(_row(flag_count=pd.NA, flag_severity=pd.NA))
    assert 'class="chip gray">—<' in dash


def test_flags_cell_tone_escalates_with_severity():
    warn = people_page._flags_cell_html(_row(flag_count=1, flag_severity=1.0))
    assert 'class="chip warn"' in warn
    crit = people_page._flags_cell_html(_row(flag_count=2, flag_severity=3.0))
    assert 'class="chip crit"' in crit


def test_cycle_and_reviews_cells_render_a_dash_for_missing_data():
    assert people_page._cycle_cell_text(_row(cycle_median=pd.NA)) == "—"
    assert people_page._cycle_cell_text(_row(cycle_median=6.2)) == "6.2d"
    assert people_page._reviews_cell_text(_row(reviews_given=pd.NA, ttfr_hours=pd.NA)) == "— · —"
    assert people_page._reviews_cell_text(_row(reviews_given=0, ttfr_hours=pd.NA)) == "0 · —"


def test_estimate_cell_renders_a_dash_for_missing_ratio():
    assert people_page._estimate_cell_html(_row(estimate_ratio=pd.NA)) == "—"
    out = people_page._estimate_cell_html(_row(estimate_ratio=1.4, estimate_iqr=0.9))
    assert "×1.4" in out and "IQR 0.9" in out


# ---------------------------------------------------------------------------
# Role filter and sort
# ---------------------------------------------------------------------------


def test_role_filter_options_include_all_roles_plus_every_role_present():
    table = pd.DataFrame({"role": ["backend", "backend", "frontend", None]})
    labels, mapping = people_page._role_filter_options(table)
    assert labels[0] == "All roles"
    assert "Backend" in labels and "Frontend" in labels
    assert mapping["All roles"] is None
    assert mapping["Backend"] == "backend"


def test_sorted_table_orders_by_role_then_score_descending():
    table = pd.DataFrame(
        {
            "person": ["Low", "High", "Mid"],
            "role": ["backend", "backend", "backend"],
            "score": [40.0, 90.0, 60.0],
        }
    )
    out = people_page._sorted_table(table)
    assert out["person"].tolist() == ["High", "Mid", "Low"]


def test_sorted_table_puts_unscored_people_last_within_their_role():
    table = pd.DataFrame(
        {
            "person": ["Scored", "Unscored"],
            "role": ["backend", "backend"],
            "score": [70.0, pd.NA],
        }
    )
    out = people_page._sorted_table(table)
    assert out["person"].tolist() == ["Scored", "Unscored"]


# ---------------------------------------------------------------------------
# The cohort wiring - the caption's claim has to be true in the code.
# ---------------------------------------------------------------------------


def _roster():
    return roles.load_roster()


def test_a_role_with_fewer_than_min_peers_reports_the_actual_peer_count():
    """Tam is the only 'platform' person on the default roster: 0 peers."""
    roster = _roster()
    cohort = roles.peer_cohort(roster, "Tam")
    assert cohort.sufficient is False
    assert cohort.peer_count < roles.MIN_PEERS

    now = pd.Timestamp("2026-08-19T00:00:00Z")
    tickets = [
        _ticket(
            "ENG-1",
            "Tam",
            _hist(_ago(now, 5), "Tam", "status", "In Progress", "Done"),
        )
    ]
    raw = pd.DataFrame(tickets)
    events = integrity.changelog_events(raw)

    parts, lookup, cohort_result = people_page._components_for_person(
        "Tam", roster, raw, raw, pd.DataFrame(), events, pd.DataFrame(), now=now
    )
    assert lookup.status == "scored"
    assert parts is not None
    by_name = {p.name: p for p in parts}
    delivery_vs_team = by_name["Delivery vs team"]
    assert delivery_vs_team.sufficient is False
    assert delivery_vs_team.n == cohort_result.peer_count
    assert "insufficient peers" in delivery_vs_team.detail
    assert str(cohort_result.peer_count) in delivery_vs_team.detail


def test_a_role_with_enough_peers_gets_a_real_cohort_percentile():
    """frontend + frontend-mobile + mobile fold into one cohort of 4 (David,
    Mohsen, Farid, Ali) - David's 3 peers clear MIN_PEERS.
    """
    roster = _roster()
    cohort = roles.peer_cohort(roster, "David")
    assert cohort.sufficient is True
    assert cohort.peer_count >= roles.MIN_PEERS

    now = pd.Timestamp("2026-08-19T00:00:00Z")
    tickets = []
    # David resolves one, each peer resolves a different number, so the
    # percentile is not a tie across the board.
    for person, count in (("David", 1), ("Mohsen Davoudi", 0), ("Farid Shahidi", 2), ("Ali", 1)):
        for i in range(count):
            tickets.append(
                _ticket(
                    f"ENG-{person}-{i}",
                    person,
                    _hist(_ago(now, 2), person, "status", "In Progress", "Done"),
                )
            )
    # David also needs at least one open ticket to appear as "owned".
    tickets.append(_ticket("ENG-DAVID-OPEN", "David"))
    raw = pd.DataFrame(tickets)
    events = integrity.changelog_events(raw)

    parts, lookup, cohort_result = people_page._components_for_person(
        "David", roster, raw, raw, pd.DataFrame(), events, pd.DataFrame(), now=now
    )
    assert lookup.status == "scored"
    by_name = {p.name: p for p in parts}
    delivery_vs_team = by_name["Delivery vs team"]
    assert delivery_vs_team.sufficient is True
    assert delivery_vs_team.n == 4
    assert "percentile" in delivery_vs_team.detail


def test_components_for_person_returns_none_for_exec_no_rubric_and_role_unknown():
    roster = _roster()
    empty = pd.DataFrame()
    for person, expected_status in (
        ("Angel Vossough", "exec"),
        ("Igor Taborsak", "no_rubric"),
        ("Nobody On The Roster", "role_unknown"),
    ):
        parts, lookup, cohort = people_page._components_for_person(
            person, roster, empty, empty, empty, empty, empty
        )
        assert parts is None
        assert lookup.status == expected_status
        assert cohort is None


# ---------------------------------------------------------------------------
# The scorecard: hatch track / literal n/a, and the honest denominator.
# ---------------------------------------------------------------------------


def test_scorecard_never_prints_a_number_for_an_insufficient_component():
    """theme_html's own invariant, exercised the way this page calls it: a
    numeric score on a sufficient=False component is still unreachable.
    """
    components = [
        theme_html.Component("Delivery", 10.0, 84.0, "n=18 PRs", sufficient=True),
        theme_html.Component(
            "Urgent response", 5.0, 999.0, "insufficient data - needs High+ tickets", sufficient=False
        ),
    ]
    out = theme_html.scorecard(components, "84", "95", "note")
    assert 'class="comp na"' in out
    assert ">n/a<" in out
    assert "999" not in out


def test_scorecard_fragment_carries_the_hatch_row_and_the_measurable_denominator():
    roster = _roster()
    now = pd.Timestamp("2026-08-19T00:00:00Z")
    tickets = [
        _ticket(
            "ENG-1",
            "Tam",
            _hist(_ago(now, 5), "Tam", "status", "In Progress", "Done"),
        )
    ]
    raw = pd.DataFrame(tickets)
    events = integrity.changelog_events(raw)
    row = _row(person="Tam", role="platform", score=pd.NA)

    fragment = people_page._scorecard_fragment(
        "Tam",
        row,
        roster,
        raw,
        raw,
        pd.DataFrame(),
        events,
        pd.DataFrame(),
        "https://vinovoss.atlassian.net/browse",
        None,
    )
    # No urgent-priority ticket exists in the fixture, so "Urgent response"
    # is always an insufficient gap row - the hatch track must show for it.
    assert 'class="comp na"' in fragment
    assert ">n/a<" in fragment
    assert "measurable points" in fragment
    assert "/100" not in fragment


def test_scorecard_fragment_degrades_honestly_for_exec_and_no_rubric_people():
    roster = _roster()
    empty = pd.DataFrame()
    exec_row = _row(person="Angel Vossough", role="exec", score=pd.NA, no_score_reason=roles.EXEC_REASON)
    fragment = people_page._scorecard_fragment(
        "Angel Vossough", exec_row, roster, empty, empty, empty, empty, empty,
        "https://vinovoss.atlassian.net/browse", None,
    )
    assert roles.EXEC_REASON in fragment
    assert "never ranked" in fragment


# ---------------------------------------------------------------------------
# Evidence rows: every count is a resolvable URL.
# ---------------------------------------------------------------------------


def test_every_evidence_row_carries_a_resolvable_url():
    roster = _roster()
    now = pd.Timestamp("2026-08-19T00:00:00Z")
    tickets = [
        _ticket("ENG-1", "David", _hist(_ago(now, 3), "David", "status", "In Progress", "Done"))
    ]
    raw = pd.DataFrame(tickets)
    events = integrity.changelog_events(raw)
    prs = pd.DataFrame(
        [
            {
                "number": 1,
                "url": "https://github.com/DrinkBetter-AI/vinovoss-frontend/pull/1",
                "author": "ahref13",
                "merged_at": pd.Timestamp("2026-08-10T00:00:00Z"),
                "created_at": pd.Timestamp("2026-08-01T00:00:00Z"),
                "reviews": [],
            }
        ]
    )
    rows = people_page._evidence_rows(
        "David", roster, prs, events, raw, "https://vinovoss.atlassian.net/browse", "DrinkBetter-AI", now=now
    )
    assert len(rows) == 4
    for label, count, url in rows:
        assert label
        assert int(count) >= 0
        assert url.startswith("https://")


# ---------------------------------------------------------------------------
# Full-page rendering.
# ---------------------------------------------------------------------------


def _bundle(raw_df: pd.DataFrame, *, github_ready: bool = False, data: dict | None = None):
    events = integrity.changelog_events(raw_df)
    return data_layer._EngineeringData(
        data=data or {},
        errors={},
        raw_df=raw_df,
        df=raw_df,
        events=events,
        github_ready=github_ready,
        github_error="",
        open_prs=pd.DataFrame(),
        merged_prs=pd.DataFrame(),
        pr_count_7=0,
        pr_count_30=0,
        open_count_exact=0,
        assignees=[],
        statuses=[],
        priorities=[],
        max_results=100,
        page_size=100,
    )


def _view(raw_df: pd.DataFrame):
    return data_layer._EngineeringView(
        scope="Organization",
        selected_assignees=[],
        selected_statuses=[],
        selected_priorities=[],
        min_idle=0,
        min_age=0,
        include_backlogs=False,
        color_by=None,
        allow_writes=False,
        filtered=raw_df,
        unscoped=raw_df,
    )


def _render_page(
    monkeypatch, raw_df: pd.DataFrame, *, data: dict | None = None, role_choice: str | None = None
) -> str:
    """Render the People page against a stub bundle; return every markdown fragment."""
    bundle = _bundle(raw_df, data=data)
    view = _view(raw_df)
    slot = SimpleNamespace(download_button=lambda *a, **k: None)

    captured: list[str] = []
    monkeypatch.setattr(people_page, "_engineering_context", lambda: (bundle, view, slot))
    monkeypatch.setattr(people_page, "_download_report", lambda *a, **k: None)
    monkeypatch.setattr(people_page.st, "markdown", lambda body, **k: captured.append(body))
    monkeypatch.setattr(people_page.st, "pills", lambda *a, **k: role_choice or "All roles")
    monkeypatch.setattr(people_page.st, "selectbox", lambda *a, **k: (k.get("options") or a[1])[0] if (k.get("options") or (a[1] if len(a) > 1 else None)) else None)
    people_page._render_people_page()
    return "\n".join(captured)


def test_the_table_renders_one_row_per_active_person_with_avatar_rolechip_and_scorebar(monkeypatch):
    now = pd.Timestamp("2026-08-19T00:00:00Z")
    raw_df = pd.DataFrame(
        [
            _ticket(
                "ENG-1",
                "Tam",
                _hist(_ago(now, 3), "Tam", "status", "In Progress", "Done"),
            )
        ]
    )
    html_out = _render_page(monkeypatch, raw_df, data={"resolved_30": pd.DataFrame()})
    assert "Tam" in html_out
    assert 'class="av"' in html_out
    assert 'class="rolechip"' in html_out
    assert 'class="scorebar"' in html_out or "no score" in html_out


def test_former_staff_never_appear_in_any_row(monkeypatch):
    now = pd.Timestamp("2026-08-19T00:00:00Z")
    raw_df = pd.DataFrame(
        [
            _ticket(
                "ENG-1",
                "Sai Shankar",
                _hist(_ago(now, 3), "Sai Shankar", "status", "In Progress", "Done"),
            ),
            _ticket(
                "ENG-2",
                "Tam",
                _hist(_ago(now, 3), "Tam", "status", "In Progress", "Done"),
            ),
        ]
    )
    html_out = _render_page(monkeypatch, raw_df, data={"resolved_30": pd.DataFrame()})
    assert "Sai Shankar" not in html_out
    assert "Tam" in html_out


def test_exec_and_no_rubric_rows_render_their_sentinels(monkeypatch):
    now = pd.Timestamp("2026-08-19T00:00:00Z")
    raw_df = pd.DataFrame(
        [
            _ticket("ENG-1", "Angel Vossough", _hist(_ago(now, 3), "Angel Vossough", "priority", "Low", "High")),
            _ticket("ENG-2", "Igor Taborsak", _hist(_ago(now, 3), "Igor Taborsak", "priority", "Low", "High")),
        ]
    )
    html_out = _render_page(monkeypatch, raw_df, data={"resolved_30": pd.DataFrame()})
    assert "exec — not scored" in html_out
    assert "no rubric defined" in html_out


def test_role_filter_narrows_the_table_and_all_roles_restores_it(monkeypatch):
    now = pd.Timestamp("2026-08-19T00:00:00Z")
    raw_df = pd.DataFrame(
        [
            _ticket("ENG-1", "Tam", _hist(_ago(now, 3), "Tam", "status", "In Progress", "Done")),
            _ticket("ENG-2", "Shawn", _hist(_ago(now, 3), "Shawn", "status", "In Progress", "Done")),
        ]
    )
    all_roles = _render_page(monkeypatch, raw_df, data={"resolved_30": pd.DataFrame()})
    assert "Tam" in all_roles and "Shawn" in all_roles

    platform_only = _render_page(
        monkeypatch, raw_df, data={"resolved_30": pd.DataFrame()}, role_choice="Platform"
    )
    assert "Tam" in platform_only
    assert "Shawn" not in platform_only


def test_the_page_renders_end_to_end_on_an_empty_frame_without_raising(monkeypatch):
    html_out = _render_page(monkeypatch, pd.DataFrame(), data={})
    assert "People" in html_out


def _one_event_frame() -> pd.DataFrame:
    """A single, structurally complete changelog event.

    ``_evidence_rows`` only needs ``combined_events`` to be non-empty and
    well-shaped - the credited count itself is stubbed in these tests - but
    ``estimate_churn`` reads ``ts`` on the way past, so the row carries every
    column ``integrity.EVENT_COLUMNS`` declares rather than just a key.
    """
    return integrity.changelog_events(
        [
            {
                "key": "VV-100",
                "changelog": {
                    "histories": [
                        {
                            "id": "1",
                            "created": "2026-08-01T09:00:00.000+0000",
                            "author": {"accountId": "acc-priya", "displayName": "Priya Shah"},
                            "items": [
                                {
                                    "field": "status",
                                    "fieldId": "status",
                                    "fromString": "In Progress",
                                    "toString": "Review in Staging",
                                }
                            ],
                        }
                    ]
                },
            }
        ]
    )


# ---------------------------------------------------------------------------
# The window the page reads has to be the window the page scores over.
# ---------------------------------------------------------------------------


def test_the_page_reads_ninety_days_not_the_bundles_thirty(monkeypatch):
    """``_credited_map(..., 90.0)`` needs 90 days of changelog to read.

    The bundle's opening read carries ``resolved_30``. Handing it to a 90-day
    window does not raise - it silently omits every resolution 31 to 90 days
    old, so a prior-period pace reads as zero and a rate measured against
    zero reads as perfect. The page takes its own wider read instead.
    """
    wide = pd.DataFrame([{"key": "VV-9", "assignee": "alice"}])
    calls: dict = {}

    def _fake_fetch(days: int = 90) -> pd.DataFrame:
        calls["days"] = days
        return wide

    monkeypatch.setattr(people_page, "_fetch_resolved_window", _fake_fetch)
    bundle = SimpleNamespace(data={"resolved_30": pd.DataFrame([{"key": "VV-1"}])})

    out = people_page._resolved_window_or_bundle(bundle)
    assert calls["days"] == 90
    assert list(out["key"]) == ["VV-9"]


def test_an_unreachable_wider_read_falls_back_to_the_bundle_not_to_empty(monkeypatch):
    """Failing to an empty frame would zero every credited count on the page.

    The fallback is narrower than the window that reads it - which is the
    flaw being fixed - but it is the safe direction to fail in: an
    understated count, never an invented one.
    """
    narrow = pd.DataFrame([{"key": "VV-1", "assignee": "alice"}])

    def _boom(days: int = 90) -> pd.DataFrame:
        raise RuntimeError("Jira unreachable")

    monkeypatch.setattr(people_page, "_fetch_resolved_window", _boom)
    bundle = SimpleNamespace(data={"resolved_30": narrow})

    out = people_page._resolved_window_or_bundle(bundle)
    assert list(out["key"]) == ["VV-1"]


def test_an_empty_wider_read_also_falls_back(monkeypatch):
    narrow = pd.DataFrame([{"key": "VV-1", "assignee": "alice"}])
    monkeypatch.setattr(people_page, "_fetch_resolved_window", lambda days=90: pd.DataFrame())
    bundle = SimpleNamespace(data={"resolved_30": narrow})
    assert list(people_page._resolved_window_or_bundle(bundle)["key"]) == ["VV-1"]


# ---------------------------------------------------------------------------
# The evidence link has to select the tickets the count counted.
# ---------------------------------------------------------------------------


def test_the_resolved_evidence_link_lists_the_credited_keys(monkeypatch):
    """The count is changelog-credited; the link has to be too.

    ``assignee = person AND resolutiondate >= -90d`` selects a different
    population twice over: it reads the current assignee - the field the
    credit fix exists to stop trusting - and it reads Jira's
    ``resolutiondate``, which a move into "Review in Staging" never sets even
    though this team counts that as resolved. Listing the credited keys is
    the only query that returns exactly what was counted.
    """
    by_person = pd.DataFrame(
        [{"person": "Priya Shah", "credited_resolutions": 2, "keys": "VV-100, VV-101"}]
    )
    monkeypatch.setattr(people_page, "_credited_by_person", lambda *a, **k: by_person)

    roster = roles.load_roster()
    rows = people_page._evidence_rows(
        person="Priya Shah",
        roster=roster,
        prs=pd.DataFrame(),
        combined_events=_one_event_frame(),
        resolved_tickets=pd.DataFrame(),
        browse_base="https://example.atlassian.net",
        github_org=None,
    )
    label, value, url = next(r for r in rows if "resolved" in r[0].lower())
    assert value == "2"
    decoded = unquote(url)
    assert "key IN (VV-100, VV-101)" in decoded
    assert "assignee" not in decoded


def test_the_resolved_evidence_link_stays_an_honest_query_when_nothing_is_credited(monkeypatch):
    """No credited keys means no key list to build - not an empty IN () clause."""
    monkeypatch.setattr(
        people_page,
        "_credited_by_person",
        lambda *a, **k: pd.DataFrame(columns=["person", "credited_resolutions", "keys"]),
    )
    rows = people_page._evidence_rows(
        person="Priya Shah",
        roster=roles.load_roster(),
        prs=pd.DataFrame(),
        combined_events=_one_event_frame(),
        resolved_tickets=pd.DataFrame(),
        browse_base="https://example.atlassian.net",
        github_org=None,
    )
    label, value, url = next(r for r in rows if "resolved" in r[0].lower())
    assert value == "0"
    assert "IN ()" not in unquote(url)
