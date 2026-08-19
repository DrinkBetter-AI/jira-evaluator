"""Phase 4: the Integrity page and its access control.

The house rule the first test in this file exists to enforce, verbatim from
the task brief: "the page must not compute at all for a non-admin session.
Not 'compute and hide': the integrity functions must not be called."
``test_a_non_admin_session_never_calls_into_integrity_or_pr_quality`` proves
that with spies, not by reading the source and trusting it.

Fixtures below build the smallest real changelog Jira could produce (the
same ``issue()``/``history()``/``item()``/``status()`` shape
``tests/test_integrity.py`` uses) rather than hand-shaping DataFrames, so a
reader can see exactly which field, timestamp and author produced each
finding.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import access_gate  # noqa: E402
import app  # noqa: E402
import integrity  # noqa: E402
import pr_quality  # noqa: E402
import roles  # noqa: E402
import theme_html  # noqa: E402
from pages import integrity as integrity_page  # noqa: E402


NOW = pd.Timestamp("2026-08-19T12:00:00Z")


# ---------------------------------------------------------------------------
# Changelog fixture builders — mirrors tests/test_integrity.py exactly, kept
# local rather than imported so this file's ownership stays self-contained.
# ---------------------------------------------------------------------------


def when(days_ago: float, hour: int = 9, minute: int = 0) -> str:
    stamp = (NOW - pd.Timedelta(days=days_ago)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.000+0000")


def item(field: str, from_string: object, to_string: object) -> dict:
    return {"field": field, "fieldId": field, "fieldtype": "jira", "fromString": from_string, "toString": to_string}


def status(from_status: str, to_status: str) -> dict:
    return item("status", from_status, to_status)


def history(created: str, author: str, *items: dict, entry_id: str | None = None) -> dict:
    return {
        "id": entry_id or f"{author}-{created}",
        "created": created,
        "author": {"displayName": author, "accountId": author.lower()},
        "items": list(items),
    }


def issue(key: str, *histories: dict) -> dict:
    return {"key": key, "changelog": {"histories": list(histories)}}


@pytest.fixture(autouse=True)
def _clean_admin_state(monkeypatch):
    """Every test starts with a fresh, unset admin gate."""
    monkeypatch.delenv(access_gate.ADMIN_PASSWORD_ENV, raising=False)
    monkeypatch.delenv(access_gate.PASSWORD_ENV, raising=False)
    import streamlit as st

    st.session_state.pop(access_gate._ADMIN_SESSION_KEY, None)
    st.session_state.pop(access_gate._SESSION_KEY, None)
    yield
    st.session_state.pop(access_gate._ADMIN_SESSION_KEY, None)
    st.session_state.pop(access_gate._SESSION_KEY, None)


# ---------------------------------------------------------------------------
# The test that matters most: zero calls into integrity.py / pr_quality.py
# for a non-admin session.
# ---------------------------------------------------------------------------


class _Spy:
    def __init__(self):
        self.calls = 0

    def __call__(self, *a, **k):
        self.calls += 1
        return pd.DataFrame()


def test_a_non_admin_session_never_calls_into_integrity_or_pr_quality(monkeypatch):
    monkeypatch.setattr(access_gate, "require_admin_password", lambda: False)

    spies = {
        name: _Spy()
        for name in (
            "cosmetic_touches",
            "estimate_churn",
            "reresolve_events",
            "integrity_flags",
        )
    }
    for name, spy in spies.items():
        monkeypatch.setattr(integrity, name, spy)
    pr_spies = {name: _Spy() for name in ("reciprocity", "flag_self_merges", "self_merge")}
    for name, spy in pr_spies.items():
        monkeypatch.setattr(pr_quality, name, spy)

    # A data layer that would blow up if it were ever reached — the gate must
    # return before this module is even imported inside the render function.
    def _boom():
        raise AssertionError("_engineering_context must not be reached for a non-admin session")

    monkeypatch.setattr("data_layer._engineering_context", _boom, raising=False)

    integrity_page._render_integrity_page()

    assert all(spy.calls == 0 for spy in spies.values()), spies
    assert all(spy.calls == 0 for spy in pr_spies.values()), pr_spies


def test_the_gate_is_the_functions_first_real_statement():
    """A structural guard against someone moving the gate check later.

    Parsed with ``ast`` rather than string-matching the source, so
    reformatting the docstring or adding a blank line can't fool it: the
    function's first statement after its docstring must be exactly
    ``if not access_gate.require_admin_password(): return``.
    """
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(integrity_page._render_integrity_page))
    func = ast.parse(source).body[0]
    assert isinstance(func, ast.FunctionDef)
    body = func.body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # the docstring
    first = body[0]
    assert isinstance(first, ast.If)
    assert isinstance(first.test, ast.UnaryOp) and isinstance(first.test.op, ast.Not)
    assert ast.unparse(first.test.operand) == "access_gate.require_admin_password()"
    assert any(isinstance(stmt, ast.Return) for stmt in first.body)


# ---------------------------------------------------------------------------
# Admin session: the four cards actually compute and render.
# ---------------------------------------------------------------------------


def _synthetic_bundle():
    """A tiny board: one cosmetic-groomer, one estimate-raiser, one bouncer, one self-merger."""
    events = integrity.changelog_events(
        [
            # Tam: five label-only edits, one status move — board grooming.
            issue(
                "VV-1",
                history(when(10), "Tam", item("labels", "", "a")),
                history(when(9), "Tam", item("labels", "a", "b")),
                history(when(8), "Tam", item("labels", "b", "c")),
                history(when(7), "Tam", item("labels", "c", "d")),
                history(when(6), "Tam", item("labels", "d", "e")),
                history(when(5), "Tam", status("To Do", "In Progress")),
            ),
            # Shawn: raises the estimate nine days after starting.
            issue(
                "VV-2",
                history(when(20), "Shawn", status("To Do", "In Progress")),
                history(when(11), "Shawn", item("timeoriginalestimate", "4h", "20h")),
            ),
            # Farid: resolves, reopens, resolves again — a staging round-trip.
            issue(
                "VV-3",
                history(when(15), "Farid", status("In Progress", "Review in Staging")),
                history(when(12), "Farid", status("Review in Staging", "In Progress")),
                history(when(9), "Farid", status("In Progress", "Review in Staging")),
            ),
        ]
    )
    df = pd.DataFrame(
        [
            {"key": "VV-1", "assignee": "Tam", "status": "In Progress", "created": when(30)},
            {"key": "VV-2", "assignee": "Shawn", "status": "In Progress", "created": when(30)},
            {"key": "VV-3", "assignee": "Farid", "status": "Review in Staging", "created": when(30)},
        ]
    )
    merged_prs = pd.DataFrame(
        [
            {
                "number": 101,
                "url": "https://github.com/DrinkBetter-AI/vinovoss/pull/101",
                "author": "David",
                "merged_by": "David",
                "base_branch": "main",
                "state": "MERGED",
                "reviews": [],
            }
        ]
    )
    open_prs = pd.DataFrame(columns=["number", "url", "author", "reviews"])
    view = SimpleNamespace(filtered=df, unscoped=df, include_backlogs=True)
    bundle = SimpleNamespace(
        df=df,
        events=events,
        github_ready=True,
        merged_prs=merged_prs,
        open_prs=open_prs,
    )
    return bundle, view


def test_an_admin_session_renders_cards_and_calls_the_flag_functions(monkeypatch):
    monkeypatch.setattr(access_gate, "require_admin_password", lambda: True)
    bundle, view = _synthetic_bundle()
    monkeypatch.setattr("data_layer._engineering_context", lambda: (bundle, view, None))
    # roles.load_roster() with no env override reads os.environ directly and
    # falls back to its own baked-in defaults (which include "Tam"/"Shawn") -
    # exercising the real function is more honest here than stubbing it, and
    # avoids the classic monkeypatch trap of a replacement that calls the
    # module attribute it just replaced.

    rendered = []
    monkeypatch.setattr(theme_html, "render", lambda *frags: rendered.append("".join(frags)))
    monkeypatch.setattr(theme_html, "css", lambda: None)

    calls = {"cosmetic_touches": 0, "estimate_churn": 0, "reresolve_events": 0, "integrity_flags": 0}
    for name in calls:
        real = getattr(integrity, name)

        def _wrap(*a, _name=name, _real=real, **k):
            calls[_name] += 1
            return _real(*a, **k)

        monkeypatch.setattr(integrity, name, _wrap)

    integrity_page._render_integrity_page()

    assert all(n > 0 for n in calls.values()), calls
    html = "".join(rendered)
    assert "Tam" in html  # the board-grooming card's top offender
    assert "Shawn" in html  # the estimate-raise card
    assert "VV-3" in html  # the staging round-trip card
    assert 'class="innocent"' in html


# ---------------------------------------------------------------------------
# Nav registration: absent when the credential isn't configured, present
# (and hidden) once it is.
# ---------------------------------------------------------------------------


def test_integrity_is_absent_from_the_registry_when_the_admin_password_is_not_configured(monkeypatch):
    monkeypatch.setattr(app, "_integrity_readable", lambda: False)
    titles = [spec.title for spec in app._page_specs()]
    assert app.INTEGRITY_PAGE_TITLE not in titles


def test_integrity_is_present_and_hidden_once_the_admin_password_is_configured(monkeypatch):
    monkeypatch.setattr(app, "_integrity_readable", lambda: True)
    specs = app._page_specs()
    mine = next(s for s in specs if s.title == app.INTEGRITY_PAGE_TITLE)
    assert mine.hidden is True
    assert mine.url_path == "integrity"
    # Still unique once it joins, same invariant test_pages.py holds business to.
    paths = [spec.url_path for spec in specs]
    assert len(paths) == len(set(paths))


def test_pages_emits_visibility_hidden_for_the_integrity_spec(monkeypatch):
    monkeypatch.setattr(app, "_integrity_readable", lambda: True)
    monkeypatch.setattr(app, "_business_readable", lambda: False)
    import streamlit as st

    captured = []
    real_page = st.Page

    def _capture(*a, **k):
        captured.append(k)
        return real_page(*a, **k)

    monkeypatch.setattr(st, "Page", _capture)
    app._pages()
    integrity_kwargs = [k for k in captured if k.get("url_path") == "integrity"]
    assert integrity_kwargs, "Integrity page was not built"
    assert integrity_kwargs[0]["visibility"] == "hidden"


# ---------------------------------------------------------------------------
# Access control: the admin credential is independent and fails closed.
# ---------------------------------------------------------------------------


def test_the_shared_password_alone_does_not_grant_admin_access(monkeypatch):
    import streamlit as st

    monkeypatch.setenv(access_gate.PASSWORD_ENV, "shared-secret")
    st.session_state[access_gate._SESSION_KEY] = True  # main gate: signed in
    monkeypatch.delenv(access_gate.ADMIN_PASSWORD_ENV, raising=False)

    assert access_gate.admin_access_granted() is False


def test_an_unset_admin_password_fails_closed_even_with_a_stale_session_flag(monkeypatch):
    import streamlit as st

    monkeypatch.delenv(access_gate.ADMIN_PASSWORD_ENV, raising=False)
    # Simulates a tampered / stale session — the check must not trust it.
    st.session_state[access_gate._ADMIN_SESSION_KEY] = True

    assert access_gate.admin_access_granted() is False
    assert access_gate.admin_password_configured() is False


def test_a_correctly_authenticated_session_is_recognised(monkeypatch):
    import streamlit as st

    monkeypatch.setenv(access_gate.ADMIN_PASSWORD_ENV, "letmein")
    st.session_state[access_gate._ADMIN_SESSION_KEY] = True

    assert access_gate.admin_access_granted() is True


def test_require_admin_password_fails_closed_when_unset(monkeypatch):
    monkeypatch.delenv(access_gate.ADMIN_PASSWORD_ENV, raising=False)
    assert access_gate.require_admin_password() is False


def test_require_admin_password_returns_true_for_an_already_granted_session(monkeypatch):
    import streamlit as st

    monkeypatch.setenv(access_gate.ADMIN_PASSWORD_ENV, "letmein")
    st.session_state[access_gate._ADMIN_SESSION_KEY] = True
    assert access_gate.require_admin_password() is True


# ---------------------------------------------------------------------------
# Evidence rows resolve to working URLs.
# ---------------------------------------------------------------------------


def test_linked_jira_keys_use_the_real_browse_base_and_the_exact_key():
    html = integrity_page._linked_jira_keys(["VV-42"])
    from render_shared import _jira_ticket_url

    expected = _jira_ticket_url("VV-42")
    assert f'href="{expected}"' in html
    assert expected.startswith(("http://", "https://"))
    assert expected.endswith("/VV-42")
    assert ">VV-42<" in html


def test_linked_pr_urls_keep_the_full_github_url_and_label_the_number():
    url = "https://github.com/DrinkBetter-AI/vinovoss/pull/205"
    html = integrity_page._linked_pr_urls([url])
    assert f'href="{url}"' in html
    assert ">#205<" in html


def test_empty_key_list_renders_a_dim_placeholder_not_a_broken_link():
    html = integrity_page._linked_jira_keys([])
    assert "href=" not in html
    assert "no tickets" in html


# ---------------------------------------------------------------------------
# Every card renders its innocent reading, even when empty.
# ---------------------------------------------------------------------------


def test_freshness_card_states_its_innocent_reading_when_there_is_nothing_to_show():
    empty_events = integrity.changelog_events([])
    html = integrity_page._freshness_card(integrity, empty_events, roles.load_roster(env={}))
    assert 'class="innocent"' in html
    assert "No field-only edits" in html


def test_estimate_card_states_its_innocent_reading_when_there_is_nothing_to_show():
    empty_events = integrity.changelog_events([])
    html = integrity_page._estimate_card(integrity, empty_events)
    assert 'class="innocent"' in html
    assert "No estimate was raised" in html


def test_staging_card_states_its_innocent_reading_when_there_is_nothing_to_show():
    empty_events = integrity.changelog_events([])
    empty_tickets = pd.DataFrame(columns=["key"])
    html = integrity_page._staging_card(integrity, empty_events, empty_tickets)
    assert 'class="innocent"' in html
    assert "No ticket entered a resolved status" in html


def test_review_card_states_its_innocent_reading_when_there_is_nothing_to_show():
    empty_prs = pd.DataFrame(columns=["number", "url", "author", "reviews"])
    html = integrity_page._review_card(pr_quality, empty_prs, empty_prs)
    assert 'class="innocent"' in html
    assert "No unapproved self-merge" in html


def test_rollup_card_states_its_innocent_reading_when_nobody_trips_a_flag():
    empty_events = integrity.changelog_events([])
    empty_tickets = pd.DataFrame(columns=["key"])
    html = integrity_page._rollup_card(integrity, empty_tickets, empty_events)
    assert 'class="innocent"' in html
    assert "Nobody tripped" in html


def test_every_card_renders_the_innocent_reading_even_when_populated():
    """Not just the empty branch — the footer is unconditional, every call."""
    bundle, _view = _synthetic_bundle()
    roster = roles.load_roster(env={})
    freshness = integrity_page._freshness_card(integrity, bundle.events, roster)
    estimate = integrity_page._estimate_card(integrity, bundle.events)
    staging = integrity_page._staging_card(integrity, bundle.events, bundle.df)
    review = integrity_page._review_card(pr_quality, bundle.merged_prs, bundle.open_prs)
    for html in (freshness, estimate, staging, review):
        assert html.count('class="innocent"') == 1


# ---------------------------------------------------------------------------
# Top-3-by-magnitude, no fixed threshold.
# ---------------------------------------------------------------------------


def test_top_n_keeps_exactly_three_of_five_and_they_are_the_largest():
    frame = pd.DataFrame(
        {
            "person": ["A", "B", "C", "D", "E"],
            "magnitude": [5.0, 40.0, 12.0, 3.0, 25.0],
        }
    )
    top = integrity_page._top_n(frame, "magnitude")
    assert len(top) == 3
    assert list(top["person"]) == ["B", "E", "C"]
    assert list(top["magnitude"]) == [40.0, 25.0, 12.0]


def test_top_n_on_a_shorter_frame_returns_everything():
    frame = pd.DataFrame({"person": ["A", "B"], "magnitude": [1.0, 2.0]})
    top = integrity_page._top_n(frame, "magnitude")
    assert len(top) == 2


def test_top_n_never_selects_by_a_fixed_cutoff():
    """Every value here is 'small' by any plausible fixed threshold; top_n
    still returns exactly three, because it ranks, it does not filter."""
    frame = pd.DataFrame({"person": list("ABCDE"), "magnitude": [1, 1, 1, 2, 2]})
    top = integrity_page._top_n(frame, "magnitude")
    assert len(top) == 3


# ---------------------------------------------------------------------------
# Cosmetic touches are baselined within role.
# ---------------------------------------------------------------------------


def _touches_frame(rows: list[tuple[str, int, int]]) -> pd.DataFrame:
    """``rows`` of (person, cosmetic_touches, status_transitions)."""
    return pd.DataFrame(
        [
            {
                "person": person,
                "cosmetic_touches": touches,
                "cosmetic_tickets": touches,
                "status_transitions": transitions,
                "status_tickets": transitions,
                "cosmetic_per_transition": float(touches) / max(transitions, 1),
                "assignee_roundtrips": 0,
                "busiest_day": "",
                "busiest_day_touches": 0,
                "first_touch": pd.NaT,
                "last_touch": pd.NaT,
                "keys": "VV-1, VV-2",
                "timestamps": [],
                "evidence": "",
            }
            for person, touches, transitions in rows
        ]
    )


def _roster_for(mapping: dict[str, str]) -> roles.Roster:
    """A tiny roster: {person: role}, all active, no logins."""
    people = {
        name.strip().lower(): roles.Person(name=name, role=role, active=True)
        for name, role in mapping.items()
    }
    return roles.Roster(people=people, unmapped_logins=(), login_index={})


def test_a_pm_with_many_grooming_touches_does_not_outrank_an_engineer_with_few():
    # Mihai (PM) grooms the board constantly — that's the job. Two more PMs
    # in the same bucket establish a high role baseline. David (frontend), a
    # peer engineer bucket, touches far less in absolute terms but is well
    # above his own much lower role baseline.
    touches = _touches_frame(
        [
            ("Mihai", 40, 2),
            ("OtherPM1", 38, 3),
            ("OtherPM2", 42, 1),
            ("David", 6, 3),
            ("OtherFE1", 1, 5),
            ("OtherFE2", 0, 4),
        ]
    )
    roster = _roster_for(
        {
            "Mihai": "pm",
            "OtherPM1": "pm",
            "OtherPM2": "pm",
            "David": "frontend",
            "OtherFE1": "frontend",
            "OtherFE2": "frontend",
        }
    )
    baselined = integrity_page._baseline_within_role(touches, roster)
    mihai_excess = baselined.loc[baselined["person"] == "Mihai", "excess_vs_role"].iloc[0]
    david_excess = baselined.loc[baselined["person"] == "David", "excess_vs_role"].iloc[0]
    assert david_excess > mihai_excess


def test_a_role_with_too_few_peers_falls_back_to_the_org_median():
    # Gaston (infra) is alone in his role bucket in this window's data — his
    # own count cannot be its own baseline (that would always read as zero
    # excess), so it must fall back to the org-wide median.
    touches = _touches_frame(
        [
            ("Gaston", 10, 1),
            ("David", 2, 4),
            ("Mohsen", 2, 4),
            ("Farid", 2, 4),
        ]
    )
    roster = _roster_for(
        {"Gaston": "infrastructure", "David": "frontend", "Mohsen": "frontend", "Farid": "frontend"}
    )
    baselined = integrity_page._baseline_within_role(touches, roster)
    gaston_baseline = baselined.loc[baselined["person"] == "Gaston", "role_baseline"].iloc[0]
    org_median = touches["cosmetic_touches"].median()
    assert gaston_baseline == org_median
    assert gaston_baseline != 10  # not "compared only to himself"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
