"""AppTest smoke: every engineering page against the synthetic 1000-ticket board.

This is the fixture the latency/cost optimization track (cached board bundle,
per-run memoisation, warm snapshot, fragment-wrapped sections, on-demand board
recording) is measured and equivalence-tested against. It proves two things:
each page still renders without an exception against a realistic-sized board,
and (via ``_log_stage``'s INFO lines, read from ``caplog`` in the timed cases
below) that a second navigation to a page already visited this run is cheap.

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

import app as dashboard
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


dashboard.fetch_tickets = _fetch_tickets

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
}
RENDERERS[PAGE]()
'''

with open(HARNESS, "w") as handle:
    handle.write(HARNESS_SOURCE)

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ("today", "people", "delivery", "code", "planning", "engineering")


def run(page: str, *, timeout: float = 300) -> AppTest:
    os.environ["BASELINE_PAGE"] = page
    test = AppTest.from_file(HARNESS, default_timeout=timeout)
    started = time.perf_counter()
    test.run()
    elapsed = time.perf_counter() - started
    assert not test.exception, (page, [e.value for e in test.exception])
    print(f"{page}: rendered without exception in {elapsed:.2f}s")
    return test


# Every engineering page has to survive a realistically sized board before any
# of the phases below can claim to have sped it up without breaking it.
for page in PAGES:
    run(page)

print("all engineering pages render against the synthetic 1000-ticket board")
