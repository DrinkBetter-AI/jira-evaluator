"""Minimal GitHub client for the dashboard's PR views.

Only what the dashboard needs: org-wide open PRs (with their review decision) and
recently-merged PRs (for the resolved counts and the per-author pie). Everything
goes through a single GraphQL search endpoint so one token and a handful of
requests cover the whole organisation, and results stay repo-agnostic -
the dashboard reports overall numbers, never a per-repo breakdown.
"""

from __future__ import annotations

import datetime as _dt
import os
import re

import pandas as pd
import requests

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
DEFAULT_ORG = "DrinkBetter-AI"
# GitHub logins/orgs: alphanumeric and single hyphens only. Enforced so the org,
# which is interpolated into the search query, cannot smuggle extra qualifiers.
_ORG_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")

# Env var names, checked in order. The dedicated dashboard token comes first so
# an ambient GITHUB_TOKEN/GH_TOKEN (Actions, Codespaces, gh CLI) - which often
# has a narrower, repo-scoped permission set - does not shadow the token an
# operator explicitly configured for org-wide PR search.
_TOKEN_ENV_VARS = ("DASHBOARD_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")


class GitHubConfigError(RuntimeError):
    """Raised when GitHub access is not configured."""


def load_github_env() -> tuple[str, str] | None:
    """Return ``(token, org)`` from the environment, or ``None`` when unset."""
    token = ""
    for name in _TOKEN_ENV_VARS:
        token = os.getenv(name, "").strip()
        if token:
            break
    if not token:
        return None
    org = os.getenv("GITHUB_ORG", DEFAULT_ORG).strip() or DEFAULT_ORG
    if not _ORG_RE.match(org):
        raise GitHubConfigError(f"Invalid GITHUB_ORG: {org!r}")
    return token, org


def _graphql(token: str, query: str, variables: dict) -> dict:
    response = requests.post(
        GITHUB_GRAPHQL_URL,
        json={"query": query, "variables": variables},
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        messages = "; ".join(e.get("message", "") for e in payload["errors"])
        raise GitHubConfigError(f"GitHub GraphQL error: {messages}")
    return payload["data"]


_PR_FIELDS = """
number
title
url
isDraft
reviewDecision
createdAt
updatedAt
mergedAt
author { login }
repository { name }
approvingReviews: reviews(states: APPROVED) { totalCount }
changesReviews: reviews(states: CHANGES_REQUESTED) { totalCount }
allReviews: reviews { totalCount }
"""

_SEARCH_QUERY = """
query($q: String!, $after: String) {
  search(query: $q, type: ISSUE, first: 100, after: $after) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes { ... on PullRequest { %s } }
  }
}
""" % _PR_FIELDS


_COUNT_QUERY = """
query($q: String!) { search(query: $q, type: ISSUE, first: 1) { issueCount } }
"""


def _search_count(token: str, query: str) -> int:
    """Total matches for a search query, from GitHub's own ``issueCount``.

    Not bounded by pagination, so it stays correct even past the point where
    the result nodes themselves would be capped.
    """
    return int(_graphql(token, _COUNT_QUERY, {"q": query})["search"]["issueCount"])


def _search_prs(token: str, query: str, max_prs: int) -> list[dict]:
    nodes: list[dict] = []
    after: str | None = None
    while len(nodes) < max_prs:
        data = _graphql(token, _SEARCH_QUERY, {"q": query, "after": after})
        search = data["search"]
        added = [n for n in search["nodes"] if n]
        nodes.extend(added)
        page = search["pageInfo"]
        cursor = page["endCursor"]
        # Stop if GitHub says there's no more, or defends against a pathological
        # response (a page that adds nothing or a cursor that never advances)
        # that would otherwise spin until the rate limit errors out.
        if not page["hasNextPage"] or not added or cursor == after or not cursor:
            break
        after = cursor
    return nodes[:max_prs]


def _to_frame(nodes: list[dict]) -> pd.DataFrame:
    rows = []
    for n in nodes:
        rows.append(
            {
                "number": n.get("number"),
                "title": n.get("title") or "",
                "url": n.get("url") or "",
                "is_draft": bool(n.get("isDraft")),
                "review_decision": n.get("reviewDecision"),
                "created_at": n.get("createdAt"),
                "updated_at": n.get("updatedAt"),
                "merged_at": n.get("mergedAt"),
                "author": (n.get("author") or {}).get("login") or "unknown",
                "repo": (n.get("repository") or {}).get("name") or "",
                "approving_reviews": (n.get("approvingReviews") or {}).get("totalCount", 0),
                "changes_reviews": (n.get("changesReviews") or {}).get("totalCount", 0),
                "total_reviews": (n.get("allReviews") or {}).get("totalCount", 0),
            }
        )
    frame = pd.DataFrame(rows)
    for col in ("created_at", "updated_at", "merged_at"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    return frame


def _open_query(org: str) -> str:
    return f"org:{org} is:pr is:open draft:false sort:created-asc"


def open_pr_count(token: str, org: str) -> int:
    """Exact count of open, non-draft PRs org-wide (never paging-capped)."""
    return _search_count(token, _open_query(org))


def fetch_open_prs(token: str, org: str, max_prs: int = 400) -> pd.DataFrame:
    """Open, non-draft PRs across the org, oldest first, with review counts."""
    frame = _to_frame(_search_prs(token, _open_query(org), max_prs))
    if frame.empty:
        return frame
    now = pd.Timestamp.now(tz="UTC")
    frame["age_days"] = (now - frame["created_at"]).dt.total_seconds() / 86400.0
    frame["idle_days"] = (now - frame["updated_at"]).dt.total_seconds() / 86400.0
    return frame


def _merged_query(org: str, days: int) -> str:
    # ISO timestamp (not a bare date) so the window is an exact rolling -Nd,
    # matching Jira's "AFTER -Nd" rather than spanning up to an extra day.
    since = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return f"org:{org} is:pr is:merged merged:>={since}"


def merged_pr_count(token: str, org: str, days: int) -> int:
    """Exact count of PRs merged org-wide in the window (never paging-capped)."""
    return _search_count(token, _merged_query(org, days))


def fetch_merged_prs(token: str, org: str, days: int, max_prs: int = 1000) -> pd.DataFrame:
    """PRs merged anywhere in the org within the last ``days`` (for the pie).

    Capped at ``max_prs`` (GitHub search only exposes the first 1,000 results);
    the headline tile uses :func:`merged_pr_count` so it is not affected.
    """
    return _to_frame(_search_prs(token, _merged_query(org, days), max_prs))
