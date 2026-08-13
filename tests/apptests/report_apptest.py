"""AppTest smoke: each tab offers its own figures as a printable report.

Runs the same stubbed dashboard the ads smoke test builds - Jira, the CRM,
Amplitude, Google Ads and the cost APIs all answered locally - and reads the
file the download button would hand over, so the check is on the report's
contents rather than on the button existing.

    python3 tests/apptests/report_apptest.py
"""

from __future__ import annotations

import ast
import os
import sys

from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
# The harness is generated below and read back by AppTest, which runs it in
# this process; the env var is how it finds the same checkout.
os.environ["DASHBOARD_REPO"] = REPO

HARNESS = str(Path(__file__).resolve().parent / "_report_harness.py")

# The ads smoke test already builds a dashboard with every source stubbed;
# copying its harness keeps one description of that world rather than two.
tree = ast.parse(open(Path(__file__).resolve().parent / "ads_apptest.py").read())
source = next(
    node.value.value
    for node in tree.body
    if isinstance(node, ast.Assign)
    and getattr(node.targets[0], "id", "") == "HARNESS_SOURCE"
)
with open(HARNESS, "w") as handle:
    handle.write(source)

from streamlit.testing.v1 import AppTest  # noqa: E402

os.environ["ADS_MODE"] = "live"

# The file itself: read from the app's own session rather than over HTTP, which
# an AppTest has no server for.
import app as dashboard  # noqa: E402

# One page per run, as the app does: each builds its own tab's report, and each
# has to offer it without the other page having been opened.
pages: dict[str, str] = {}
for page in ("business", "engineering"):
    os.environ["HARNESS_PAGE"] = page
    test = AppTest.from_file(HARNESS, default_timeout=300)
    test.run()
    assert not test.exception, (page, [e.value for e in test.exception])
    labels = [b.label for b in test.download_button]
    assert labels.count("Download report") == 1, (page, labels)
    pages.update(
        {tab: built.html() for tab, built in test.session_state[dashboard.REPORTS_KEY].items()}
    )
assert set(pages) == {"Engineering", "Business"}, list(pages)

business = pages["Business"]
for expected in (
    "Orders, Revenue &amp; AOV",
    "Ads Spend &amp; Return",
    "Spend (30d)",
    "$2,402.10",
    "Commission per $1 spent",
    "AI costs",
    "Payments",
):
    assert expected in business, (expected, business[:400])
# The sentence that explains the headline, with its figure still bold.
assert "<strong>" in business
assert "**" not in business

engineering = pages["Engineering"]
for expected in ("Ticket health", "Open tickets", "Ticket clarity"):
    assert expected in engineering, (expected, engineering[:400])
# Each tab reports itself: the shop's figures are not in the engineering pack.
assert "Ads Spend" not in engineering
# Jira answers this harness but GitHub does not, so the resolved and PR tiles
# read as unavailable - and a report says nothing rather than printing a dash
# that would be read as a zero.
assert "PRs merged" not in engineering, engineering
assert "\u2014" not in engineering, engineering

import re  # noqa: E402

print("both tabs offer a report: ok")
print("business:", re.findall(r"<h2>([^<]+)</h2>", business))
print("engineering:", re.findall(r"<h2>([^<]+)</h2>", engineering))
