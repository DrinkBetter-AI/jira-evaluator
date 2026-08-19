"""Report-parity harness for the Code page, with GitHub answering.

Named without the family's usual ``_..._harness.py`` suffix
(``.gitignore``'s ``tests/apptests/_*_harness.py`` line treats that shape as
a build artifact the matching ``*_apptest.py`` driver regenerates on every
run from an embedded string - see ``baseline_apptest.py``/
``_baseline_harness.py`` for that pattern). This file is not generated: it
is hand-authored and checked in directly, the same way ``_synthetic_
board.py`` is, so it needs a name the gitignore rule does not match, or a
fresh checkout (and CI) would silently never have it.

``_baseline_harness.py`` disables GitHub entirely (the credential-free
state task 5B's smoke test targets), which is enough to prove Code renders
safely with nothing to show - but ``_render_code_page`` returns right after
its "GitHub is unavailable" banner in that state, before a single one of
its tiles/hbars/tables ever draws. Proving or disproving report-export
parity for those needs GitHub *ready*, with a few realistic open and merged
PRs - this file builds that, monkeypatching ``pages.code._engineering_
context`` the same way ``tests/test_code_page.py``'s own
"renders end to end with no GitHub data" case does, rather than reaching a
real GitHub token.
"""

import os
import sys

sys.path.insert(0, os.environ["DASHBOARD_REPO"])
sys.path.insert(0, os.environ["APPTESTS_DIR"])

import pandas as pd
import streamlit as st

import app as dashboard
import data_layer
import github_client
from pages import code as code_page
from _synthetic_board import build_synthetic_board

dashboard.github_client.load_github_env = lambda: ("gh-stub-token", "DrinkBetter-AI")


def _open_prs() -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    return pd.DataFrame(
        [
            {
                "number": 501,
                "url": "https://github.com/DrinkBetter-AI/vinovoss/pull/501",
                "title": "ENG-11 fix checkout crash",
                "author": "priya",
                "repo": "vinovoss",
                "state": "OPEN",
                "is_draft": False,
                "approving_reviews": 0,
                "total_reviews": 0,
                "review_requests": 0,
                "age_days": 12.0,
                "idle_days": 5.0,
                "reviews": [],
                "created_at": now - pd.Timedelta(days=12),
            },
            {
                "number": 502,
                "url": "https://github.com/DrinkBetter-AI/vinovoss/pull/502",
                "title": "ENG-22 add retry to webhook",
                "author": "marco",
                "repo": "vinovoss-api",
                "state": "OPEN",
                "is_draft": False,
                "approving_reviews": 1,
                "total_reviews": 2,
                "review_requests": 1,
                "age_days": 3.0,
                "idle_days": 1.0,
                "reviews": [],
                "created_at": now - pd.Timedelta(days=3),
            },
        ]
    )


def _merged_prs() -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    return pd.DataFrame(
        [
            {
                "number": 490,
                "url": "https://github.com/DrinkBetter-AI/vinovoss/pull/490",
                "author": "priya",
                "merged_by": "priya",
                "base_branch": "main",
                "state": "MERGED",
                "approving_reviews": 0,
                "merged_at": now - pd.Timedelta(days=2),
                "reviews": [],
            },
            {
                "number": 491,
                "url": "https://github.com/DrinkBetter-AI/vinovoss/pull/491",
                "author": "marco",
                "merged_by": "priya",
                "base_branch": "main",
                "state": "MERGED",
                "approving_reviews": 1,
                "merged_at": now - pd.Timedelta(days=4),
                "reviews": [],
            },
        ]
    )


class _Bundle:
    df = build_synthetic_board(n=20)
    github_ready = True
    github_error = ""
    open_prs = _open_prs()
    merged_prs = _merged_prs()


class _View:
    selected_assignees = None
    filtered = _Bundle.df


_SLOT = st.empty()
code_page._engineering_context = lambda: (_Bundle(), _View(), _SLOT)
# The extended (GraphQL) pool feeds the proactivity/draft-hiding sections
# only - not the KPI tiles, review-coverage bars or stuck queue this harness
# is about - so it is left genuinely tokenless (the honest "no token"
# degradation _extended_pr_pool already has) rather than built out too.
github_client.load_github_env = lambda: None

dashboard.inject_styles()
dashboard._reset_reports()

code_page._render_code_page()
