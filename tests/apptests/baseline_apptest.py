"""AppTest smoke: every dashboard page against the synthetic 1000-ticket board.

This is the fixture the latency/cost optimization track (cached board bundle,
per-run memoisation, warm snapshot, fragment-wrapped sections, on-demand board
recording) is measured and equivalence-tested against. It proves two things:
each page still renders without an exception against a realistic-sized board,
and (via ``_log_stage``'s INFO lines, read from ``caplog`` in the timed cases
below) that a second navigation to a page already visited this run is cheap.

Task 5B extended this from six pages to all eight: Business (credential-free,
same as every other source here) and Integrity, in both of the sessions that
matter - a default, non-admin session (the state a fresh Cloud Run deploy and
a new contributor's laptop are both in) that must render nothing Integrity-
shaped and must not call a single one of ``integrity.py``'s/``pr_quality.py``'s
functions, and an admin session that must call all of them and render all
four cards. ``HARNESS_SOURCE`` below carries the mechanics for both.

The Jira and GitHub reads are stubbed at the same boundary the other apptests
use: ``fetch_tickets`` is replaced outright (real credentials are never
reached), and every other integration is switched off so a page renders from
the synthetic board alone.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
APPTESTS_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, REPO)
sys.path.insert(0, APPTESTS_DIR)
# The harness is generated below and read back by AppTest, which runs it in
# this process; the env vars are how it finds the same checkout and this
# directory (for the synthetic-board module it imports).
os.environ["DASHBOARD_REPO"] = REPO
os.environ["APPTESTS_DIR"] = APPTESTS_DIR

HARNESS = str(Path(__file__).resolve().parent / "_baseline_harness.py")

HARNESS_SOURCE = '''
import os
import sys

sys.path.insert(0, os.environ["DASHBOARD_REPO"])
sys.path.insert(0, os.environ["APPTESTS_DIR"])

import streamlit as st

import app as dashboard
import data_layer
from _synthetic_board import build_synthetic_board

BOARD = build_synthetic_board()

# Every optional integration is switched off, the same way the other apptests
# do it, so a page renders from the synthetic board alone rather than reaching
# for real credentials.
dashboard.github_client.load_github_env = lambda: None
dashboard.amplitude_client.load_amplitude_env = lambda: None
dashboard.ads_client.load_ads_env = lambda: None
dashboard.cost_client.load_openai_env = lambda: None
dashboard.cost_client.load_stripe_env = lambda: None
dashboard.cost_client.load_billing_env = lambda: None
dashboard.orders_client.load_medusa_env = lambda: None
dashboard.merchant_client.load_merchant_env = lambda: None


def _fetch_tickets(*args, **kwargs):
    # A fresh copy every call: the real cached reader hands back a frame the
    # caller owns, and a shared mutable frame would let one page's reshape
    # bleed into the next page's read within the same process.
    return BOARD.copy()


data_layer.fetch_tickets = _fetch_tickets

dashboard.inject_styles()
dashboard._reset_reports()

PAGE = os.environ.get("BASELINE_PAGE", "today")
RENDERERS = {
    "today": dashboard._render_today_page,
    "people": dashboard._render_people_page,
    "delivery": dashboard._render_delivery_page,
    "code": dashboard._render_code_page,
    "planning": dashboard._render_planning_page,
    "engineering": dashboard._render_engineering_page,
    "business": dashboard._render_business,
    "integrity": dashboard._render_integrity_page,
}

if PAGE == "integrity":
    # Task 5B: the Integrity page is CEO-only (access_gate.require_admin_
    # password), and the two sessions worth proving render-safe are opposite
    # ends of that gate. A default session here has no DASHBOARD_ADMIN_
    # PASSWORD anywhere in this process - the same credential-free state
    # every other page above renders from - so running RENDERERS["integrity"]
    # unmodified already *is* the non-admin case: the real, un-mocked
    # ``access_gate.require_admin_password()`` sees no credential configured,
    # writes its own ``st.error(...)`` and calls ``st.stop()`` - which, under
    # a real AppTest run (unlike a bare unit test), actually halts the script
    # right there. Nothing after that point in this file executes at all for
    # a non-admin session: not the spies' wrapping (already installed below,
    # before the gate runs), not a second write to session_state, nothing.
    # BASELINE_ADMIN=1 flips the gate the way tests/test_integrity_page.py
    # does it - by replacing access_gate.require_admin_password() itself -
    # rather than by supplying a real admin password, so this harness never
    # needs a credential of its own to prove either side of the gate.
    import access_gate
    import integrity
    import pr_quality

    _INTEGRITY_SPIES = (
        (integrity, "cosmetic_touches"),
        (integrity, "estimate_churn"),
        (integrity, "reresolve_events"),
        (integrity, "integrity_flags"),
        (pr_quality, "reciprocity"),
        (pr_quality, "flag_self_merges"),
        (pr_quality, "self_merge"),
    )
    _calls = {name: 0 for _module, name in _INTEGRITY_SPIES}

    # Written *before* the gate runs. A non-admin run's ``st.stop()`` (inside
    # ``RENDERERS[PAGE]()`` below) means this is the only write that session
    # ever gets - so what the outer script reads back for that session is
    # "every function, zero calls" precisely because nothing ran, not
    # because a spy politely reported zero.
    st.session_state["_integrity_call_counts"] = dict(_calls)

    def _spy(module, name):
        real = getattr(module, name)

        def _wrapped(*a, _name=name, _real=real, **k):
            _calls[_name] += 1
            return _real(*a, **k)

        setattr(module, name, _wrapped)

    for _module, _name in _INTEGRITY_SPIES:
        _spy(_module, _name)

    if os.environ.get("BASELINE_ADMIN") == "1":
        access_gate.require_admin_password = lambda: True

    RENDERERS[PAGE]()
    # Only reached for an admin session. Read back by the outer script
    # through ``test.session_state``.
    st.session_state["_integrity_call_counts"] = dict(_calls)
else:
    RENDERERS[PAGE]()
'''

with open(HARNESS, "w") as handle:
    handle.write(HARNESS_SOURCE)

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = (
    "today",
    "people",
    "delivery",
    "code",
    "planning",
    "integrity",
    "engineering",
    "business",
)


def run(page: str, *, admin: bool = False, timeout: float = 300) -> AppTest:
    os.environ["BASELINE_PAGE"] = page
    os.environ["BASELINE_ADMIN"] = "1" if admin else "0"
    test = AppTest.from_file(HARNESS, default_timeout=timeout)
    started = time.perf_counter()
    test.run()
    elapsed = time.perf_counter() - started
    assert not test.exception, (page, admin, [e.value for e in test.exception])
    label = f"{page} (admin)" if admin else page
    print(f"{label}: rendered without exception in {elapsed:.2f}s")
    return test


def _rendered_text(test: AppTest) -> str:
    """Every markdown fragment the run wrote, concatenated.

    Every page here (Integrity included) is drawn through
    ``theme_html.render`` -> ``st.markdown(..., unsafe_allow_html=True)``, so
    this is where an Integrity-shaped heading or card would show up if the
    gate had let one through.
    """
    return "".join(m.value for m in test.markdown)


# Every one of the eight pages has to survive a realistically sized board,
# credential-free, before any later phase can claim to have sped one up
# without breaking it.
for page in PAGES:
    test = run(page)
    if page == "integrity":
        # The default run above has no DASHBOARD_ADMIN_PASSWORD anywhere in
        # this process - it *is* the non-admin session, not a separate mode.
        # access_gate.require_admin_password() is not mocked for this run,
        # so the real function's own st.error()+st.stop() fired, which is
        # why the count stayed at the pre-gate zero (see HARNESS_SOURCE above)
        # and why nothing Integrity-shaped is anywhere in the markdown.
        # "Not hidden-but-computed - not computed at all" (pages/integrity.py
        # module docstring), proved by the live gate halting the script, not
        # by a mock standing in for it.
        calls = test.session_state["_integrity_call_counts"]
        assert all(n == 0 for n in calls.values()), ("non-admin computed", calls)
        errors = [e.value for e in test.error]
        assert any("DASHBOARD_ADMIN_PASSWORD is not set" in e for e in errors), errors
        rendered = _rendered_text(test)
        assert "VinoVoss · Integrity" not in rendered, "non-admin session rendered the page header"
        assert "intgrid" not in rendered, "non-admin session rendered the card grid"

admin_test = run("integrity", admin=True)
admin_calls = admin_test.session_state["_integrity_call_counts"]
# pages/integrity.py's own body calls all four integrity.py functions and
# exactly one of pr_quality's (self_merge, from _review_card) - reciprocity
# and flag_self_merges are spied here only to prove they stay at zero on the
# non-admin run above; the page itself has no call site for either of them
# (grep pages/integrity.py - "pr_quality." appears only in the module
# docstring and this file's own comments).
_CALLED_ON_ADMIN = (
    "cosmetic_touches",
    "estimate_churn",
    "reresolve_events",
    "integrity_flags",
    "self_merge",
)
assert all(admin_calls[name] > 0 for name in _CALLED_ON_ADMIN), (
    "admin session skipped a function",
    admin_calls,
)
assert not admin_test.error, [e.value for e in admin_test.error]
admin_rendered = _rendered_text(admin_test)
assert "VinoVoss · Integrity" in admin_rendered, "admin session did not render the page header"
assert "intgrid" in admin_rendered, "admin session did not render the card grid"

print("all eight dashboard pages render against the synthetic 1000-ticket board")
print(
    "integrity: non-admin session renders nothing and calls nothing; "
    "admin session renders all four cards and calls every flag function"
)
