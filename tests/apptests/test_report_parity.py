"""Task 5B: does "Download report" actually carry what the page shows?

Agent 1B found that ``theme_html.tiles()``/``theme_html.hbars()`` never
recorded into the printable report and added an optional ``tab=``/
``section=`` keyword pair that, when a caller supplies it, records the same
way ``page_shared._tile``/``_kpis`` always have. This file is the follow-up
the task brief asked for: render each page for real, build its report, and
check - per page, per component - whether what is on screen is actually in
the file the "Download report" button hands over.

The answer is "partially". Where a call site was updated to pass
``tab=``/``section=`` (``pages/delivery.py``'s ``theme_html.tiles(...)``
call), the fix holds and is asserted plainly below. Where a call site was
not - ``pages/code.py``'s ``theme_html.tiles()``/``theme_html.hbars()``
calls, both pages' ``theme_html.hbars()``/``theme_html.table()`` calls that
draw ranked bars and tables, and every page that only ever calls
``theme_html.table()`` (``pages/people.py``) - the gap is real, end to end,
with real rendered data, not just by reading the source. Those are pinned
below as ``xfail`` cases: each asserts the parity that *should* hold, with a
reason pointing at which file the fix belongs to (none of them are pages/
theme_html.py/render_shared.py, which are out of this task's ownership -
see ``docs/assumptions/5B.md``). An ``xfail`` that starts passing
(``XPASS``) is the signal that a fix landed and the marker should come off.

Today and Integrity are not here at all: neither page calls
``_download_report`` in the first place (confirmed by
``grep -n _download_report pages/today.py pages/integrity.py`` returning
nothing), so there is no report to check parity against - see
``docs/assumptions/5B.md``.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

REPO = str(Path(__file__).resolve().parents[2])
APPTESTS_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, REPO)
sys.path.insert(0, APPTESTS_DIR)
os.environ["DASHBOARD_REPO"] = REPO
os.environ["APPTESTS_DIR"] = APPTESTS_DIR

from streamlit.testing.v1 import AppTest  # noqa: E402

import plotly.io as pio  # noqa: E402

import access_gate  # noqa: E402
import ads_client  # noqa: E402
import amplitude_client  # noqa: E402
import cost_client  # noqa: E402
import data_layer  # noqa: E402
import github_client  # noqa: E402
import merchant_client  # noqa: E402
import orders_client  # noqa: E402
import page_shared  # noqa: E402
import theme  # noqa: E402
from pages import code as code_page  # noqa: E402

def _write_baseline_harness() -> str:
    """(Re)generate ``_baseline_harness.py`` from ``baseline_apptest.py``'s own copy.

    ``tests/apptests/_*_harness.py`` is gitignored (".gitignore":
    "the smoke tests generate their stubbed dashboard beside themselves") -
    it is a build artifact, not a checked-in file, and the only thing that
    normally regenerates it is ``baseline_apptest.py`` running as its own
    ``*_apptest.py`` subprocess (``tests/test_apptests.py``, marked
    ``@pytest.mark.slow``). This file's tests are not slow and must not
    depend on that subprocess having already run in the same pytest
    session - on a fresh checkout, or when only ``pytest -m "not slow"`` is
    selected, it may never run at all, and this file would raise
    ``FileNotFoundError`` reaching for a harness nobody built. Pulling
    ``HARNESS_SOURCE`` back out of ``baseline_apptest.py`` via ``ast`` (the
    same technique ``report_apptest.py`` already uses to reuse
    ``ads_apptest.py``'s harness) keeps one source of truth for the harness
    body while making this file self-sufficient.
    """
    tree = ast.parse((Path(__file__).resolve().parent / "baseline_apptest.py").read_text())
    source = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "HARNESS_SOURCE"
    )
    path = Path(__file__).resolve().parent / "_baseline_harness.py"
    path.write_text(source)
    return str(path)


BASELINE_HARNESS = _write_baseline_harness()
CODE_HARNESS = str(Path(__file__).resolve().parent / "_code_report_data.py")


@pytest.fixture(autouse=True)
def _isolate_dashboard_module_state(monkeypatch):
    """Undo whatever a harness run below mutated, once the test is done.

    ``AppTest.from_file(...).run()`` executes each harness in *this* process,
    unlike the ``*_apptest.py`` scripts ``tests/test_apptests.py`` runs one
    subprocess per script for exactly this reason (see that module's own
    docstring). ``_baseline_harness.py``/``_code_report_data.py`` both
    reassign module-level functions - ``ads_client.load_ads_env``,
    ``github_client.load_github_env``, ``data_layer.fetch_tickets`` and
    friends - as plain attribute assignments, which is fine inside a
    throwaway subprocess and is not fine here: left alone, the stub from the
    first test in this file would still be sitting on ``ads_client`` when
    ``tests/test_ads_client.py`` ran later in the same pytest session (this
    is exactly the failure that surfaced the first time this file was added
    to the suite - fixed here, not by making ``tests/test_ads_client.py``
    defensive against a foreign module's test fixture).

    ``monkeypatch.setattr(obj, name, getattr(obj, name))`` is the standard
    trick for registering a snapshot for pytest to restore on teardown, even
    though the actual mutation happens later - inside the exec'd harness
    script - rather than in this function body.

    ``pio.templates`` gets the same treatment for a different reason: it is
    plotly's own global, process-wide registry, and every harness run below
    calls ``dashboard.inject_styles()`` -> ``theme.chart_fonts()``, which
    registers a "vinovoss" template into it and points ``pio.templates.
    default`` at it - and never unregisters it, because in the real app that
    registration is meant to outlive the whole process. Left alone, that is
    what turned ``tests/test_price_charts.py::test_bigger_text_keeps_the_
    colours_the_app_already_drew_with`` red the first time this file ran
    ahead of it in the same session: that test snapshots ``pio.templates.
    default`` as "whatever it was before this test", and if this file has
    already left it at the string ``"vinovoss"`` by then, that test's own
    cleanup deletes the ``"vinovoss"`` template object but restores
    ``pio.templates.default`` back to the now-dangling string
    ``"vinovoss"`` - a latent assumption in a test file this task does not
    own, exposed only because this file is (as far as this run is
    concerned) the first thing in the suite to touch that global. Fixed
    here, the same way as the module-attribute leaks above: snapshot and
    restore both ``pio.templates.default`` and whether ``theme._TEMPLATE``
    was registered at all, so this file leaves that global exactly as it
    found it.

    ``st.cache_data.clear()`` runs before every test for a third, related
    reason: ``data_layer.py``'s fetch/derive functions are ``@st.cache_data``
    -decorated, and outside a real Streamlit server session (every "No
    runtime found, using MemoryCacheStorageManager" warning in this suite's
    output) that cache is process-wide rather than per-session - so a cache
    entry another test file populated earlier in the same pytest run can
    still answer a call this file makes, silently standing in for this
    file's own stubbed ``fetch_tickets``/``_engineering_context``. Clearing
    before (not just after) matters too: a stale entry left by whatever ran
    immediately before this fixture first takes effect would otherwise
    still be sitting there for this file's first test.

    A fourth, harder-to-see leak lives one level below ``st.cache_data``:
    ``data_layer._engineering_data()`` also persists its gathered board to
    ``_SNAPSHOT_PATH`` - a fixed path under ``tempfile.gettempdir()``, so it
    is shared by every process on the machine, not scoped to this one
    (``data_layer.py``'s own "warm snapshot" feature: a fast cold start reads
    that file before paying for a live gather). A fresh ``AppTest`` session
    has empty ``st.session_state``, so ``_engineering_data()`` always takes
    the "nothing held yet" branch and reads whatever is sitting at that path
    - which can be a bundle an *entirely different* test (this file's own
    Code-page harness, ``tests/test_board_snapshot.py``'s temp-path fixture
    aside, ``tests/apptests/test_apptests.py``'s slow subprocesses, or
    another process on this machine) wrote seconds earlier, board and
    ``events`` included, regardless of what this test just stubbed
    ``fetch_tickets`` to return. That is what made the delivery-tiles
    assertion below flip between "Median In-Progress" et al. holding real
    numbers and the page's own "no changelog data read" fallback tiles
    (``pages/delivery.py``'s ``no_org_history`` branch) - confirmed by
    reading ``test.session_state["tab_reports"]`` mid-failure and finding a
    ``Report`` object present but empty, and separately by inspecting
    ``/tmp/jira_dashboard_board_snapshot.pkl``'s mtime moving independently
    of anything in this file - not a collection-order effect (a suspected
    interaction with ``tests/apptests/test_changelog_fixture.py`` did not
    reproduce once isolated) and not fixed by ``st.cache_data.clear()``
    alone, since the snapshot lives on disk, outside that registry.
    ``data_layer._delete_board_snapshot()`` (the same helper the app's own
    "Refresh" affordance calls, per its docstring: "a stale file cannot
    outlive the reads it came from") is used here rather than reaching into
    ``_SNAPSHOT_PATH`` directly, so this stays correct if that path or its
    format ever changes. Running it before *and* after mirrors the
    ``st.cache_data.clear()`` pattern above for the same reason: a snapshot
    left by whatever ran immediately before this fixture first takes effect
    must not answer this file's first test, and this file must not leave a
    snapshot behind for whatever runs immediately after it either.
    """
    import streamlit as st

    st.cache_data.clear()
    data_layer._delete_board_snapshot()

    for module, name in (
        (github_client, "load_github_env"),
        (amplitude_client, "load_amplitude_env"),
        (ads_client, "load_ads_env"),
        (cost_client, "load_openai_env"),
        (cost_client, "load_stripe_env"),
        (cost_client, "load_billing_env"),
        (orders_client, "load_medusa_env"),
        (merchant_client, "load_merchant_env"),
        (data_layer, "fetch_tickets"),
        (data_layer, "_engineering_context"),
        (code_page, "_engineering_context"),
        (access_gate, "require_admin_password"),
    ):
        monkeypatch.setattr(module, name, getattr(module, name))

    template_default_before = pio.templates.default
    had_template_before = theme._TEMPLATE in pio.templates
    yield
    if not had_template_before and theme._TEMPLATE in pio.templates:
        del pio.templates[theme._TEMPLATE]
    pio.templates.default = template_default_before
    st.cache_data.clear()
    data_layer._delete_board_snapshot()


def _run_baseline(page: str) -> AppTest:
    os.environ["BASELINE_PAGE"] = page
    os.environ["BASELINE_ADMIN"] = "0"
    test = AppTest.from_file(BASELINE_HARNESS, default_timeout=300)
    test.run()
    assert not test.exception, [e.value for e in test.exception]
    return test


def _run_code_with_github() -> AppTest:
    test = AppTest.from_file(CODE_HARNESS, default_timeout=300)
    test.run()
    assert not test.exception, [e.value for e in test.exception]
    return test


def _rendered_text(test: AppTest) -> str:
    return "".join(m.value for m in test.markdown)


def _report_html(test: AppTest, tab: str) -> str:
    """The built report's HTML for ``tab``, or "" if the tab never even offered one."""
    try:
        reports = test.session_state[page_shared.REPORTS_KEY]
    except KeyError:
        return ""
    built = reports.get(tab)
    if built is None or built.empty:
        return ""
    return built.html()


# ---------------------------------------------------------------------------
# Delivery: the fixed call site (tiles) holds; the un-fixed ones (hbars,
# table) don't - all against the same render, same synthetic board.
# ---------------------------------------------------------------------------


def test_delivery_tiles_are_recorded_in_the_report():
    """1B's fix, proven end to end: the tiles shown on screen are in the report."""
    test = _run_baseline("delivery")
    rendered = _rendered_text(test)
    report = _report_html(test, page_shared.TAB_ENGINEERING)
    assert report, "Delivery offered no report at all"
    for label in ("Median In-Progress", "Staging round-trips", "Reopened", "Unattributed"):
        assert label in rendered, f"{label!r} is not even on screen - fixture drifted"
        assert label in report, f"{label!r} is on screen but missing from the report"


@pytest.mark.xfail(
    reason=(
        "pages/delivery.py's two theme_html.hbars() calls ('Where open work "
        "sits, by team' and 'Cycle time by status') don't pass tab=/section=, "
        "so 1B's recording fix never fires for them. Fix belongs to "
        "pages/delivery.py, a page module outside this task's ownership."
    ),
    strict=True,
)
def test_delivery_hbars_are_recorded_in_the_report():
    test = _run_baseline("delivery")
    rendered = _rendered_text(test)
    report = _report_html(test, page_shared.TAB_ENGINEERING)
    assert "Where open work sits, by team" in rendered
    assert "Where open work sits, by team" in report


@pytest.mark.xfail(
    reason=(
        "theme_html.table() has no tab=/section= parameters at all (only "
        "tiles()/hbars() got 1B's fix), so no call to it can ever record "
        "into the report regardless of the call site. Fix belongs to "
        "theme_html.py, outside this task's ownership."
    ),
    strict=True,
)
def test_delivery_stale_table_is_recorded_in_the_report():
    test = _run_baseline("delivery")
    rendered = _rendered_text(test)
    report = _report_html(test, page_shared.TAB_ENGINEERING)
    assert "Stale &amp; abandoned" in rendered
    assert "Stale" in report


# ---------------------------------------------------------------------------
# Code: nothing is recorded at all - the KPI tiles, the review-coverage
# bars and the stuck-code table are all on screen and all absent from the
# report. Needs GitHub "ready" (the credential-free baseline harness proves
# the page degrades safely without it, but never reaches these sections at
# all in that state - see _code_report_data.py's own docstring).
# ---------------------------------------------------------------------------


def test_code_page_renders_its_pr_sections_with_github_ready():
    """Fixture sanity: if this fails, the gaps below are unproven, not fixed."""
    test = _run_code_with_github()
    rendered = _rendered_text(test)
    subheaders = [s.value for s in test.subheader]
    assert "Open PRs" in rendered
    assert "Review coverage by repo" in subheaders
    assert any(s.startswith("Stuck queue") for s in subheaders)


@pytest.mark.xfail(
    reason=(
        "pages/code.py's _render_code_kpis() calls theme_html.tiles([...]) "
        "with no tab=/section= - the five headline PR numbers never reach "
        "1B's recording path. Fix belongs to pages/code.py, a page module "
        "outside this task's ownership."
    ),
    strict=True,
)
def test_code_kpi_tiles_are_recorded_in_the_report():
    test = _run_code_with_github()
    report = _report_html(test, page_shared.TAB_ENGINEERING)
    assert "Open PRs" in report
    assert "No approving review" in report


@pytest.mark.xfail(
    reason=(
        "pages/code.py's _render_repo_coverage() calls theme_html.hbars() "
        "with no tab=/section=. Fix belongs to pages/code.py."
    ),
    strict=True,
)
def test_code_repo_coverage_bars_are_recorded_in_the_report():
    test = _run_code_with_github()
    report = _report_html(test, page_shared.TAB_ENGINEERING)
    assert "vinovoss-api" in report  # the repo with no approving review


@pytest.mark.xfail(
    reason=(
        "pages/code.py's stuck-queue table is drawn through theme_html.table(), "
        "which has no tab=/section= parameters at all - see the delivery-table "
        "xfail above for the same root cause. Fix belongs to theme_html.py."
    ),
    strict=True,
)
def test_code_stuck_queue_table_is_recorded_in_the_report():
    test = _run_code_with_github()
    report = _report_html(test, page_shared.TAB_ENGINEERING)
    assert "ENG-11 fix checkout crash" in report


# ---------------------------------------------------------------------------
# People: the page renders real content (tables, via theme_html.table()) but
# never calls page_shared._tile/_kpis/_said or theme_html.tiles()/hbars()
# with tab=/section= at all - its report is unconditionally empty, so the
# "Download report" button never even appears.
# ---------------------------------------------------------------------------


def test_people_page_renders_real_content():
    """Fixture sanity for the xfail below: the page has something to report."""
    test = _run_baseline("people")
    rendered = _rendered_text(test)
    assert len(rendered) > 5000
    assert "<table" in rendered


@pytest.mark.xfail(
    reason=(
        "pages/people.py never calls page_shared._tile/_kpis/_said or "
        "theme_html.tiles()/hbars() with tab=/section= - it only calls the "
        "unrecordable theme_html.table() (see the delivery-table xfail "
        "above). Its report is unconditionally empty and the 'Download "
        "report' button never renders. Fix belongs to pages/people.py."
    ),
    strict=True,
)
def test_people_page_offers_a_download_report_button():
    test = _run_baseline("people")
    labels = [b.label for b in test.download_button]
    assert "Download report" in labels
