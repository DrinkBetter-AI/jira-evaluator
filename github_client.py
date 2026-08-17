"""Minimal GitHub client for the dashboard's PR views.

Only what the dashboard needs: org-wide open PRs (with their review decision),
recently-merged PRs (for the resolved counts and the per-author pie) and the
PRs that were closed without ever merging. Everything goes through a single
GraphQL search endpoint so one token and a handful of requests cover the whole
organisation, and results stay repo-agnostic - the dashboard reports overall
numbers, never a per-repo breakdown.

Two payloads exist because they cost very different amounts:

- the lean payload (:data:`_PR_FIELDS`) is scalars and ``totalCount``s only, one
  search page for one rate-limit point;
- the detail payload adds diff size, the merger, and the review and timeline
  nodes that make review quality measurable at all. Its nested ``first:``
  arguments multiply against the 100 PRs a page returns, so they are named
  constants and deliberately small.

Every detail field is optional on the way into the frame. A throttled or older
response that carries only the lean fields still produces a usable frame, with
``None`` - not ``0`` - in the columns nobody answered, so a metric can tell "no
reviews" apart from "never asked". :func:`_search_prs` makes that concrete: if
the richer query fails outright (a 403 from a rate-limited token, a field the
server will not serve), it retries once with the lean query rather than leaving
the dashboard with nothing.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
import requests

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
DEFAULT_ORG = "DrinkBetter-AI"
# GitHub logins/orgs: alphanumeric and single hyphens only. Enforced so the org,
# which is interpolated into the search query, cannot smuggle extra qualifiers.
_ORG_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
# A repository name, optionally owner-qualified. Same reason as the org: these
# are interpolated into the search query, so a value that could carry a space
# or a second qualifier is dropped rather than sent.
_REPO_RE = re.compile(r"^(?:[A-Za-z0-9][A-Za-z0-9-]{0,38}/)?[A-Za-z0-9._-]{1,100}$")

# Env var names, checked in order. The dedicated dashboard token comes first so
# an ambient GITHUB_TOKEN/GH_TOKEN (Actions, Codespaces, gh CLI) - which often
# has a narrower, repo-scoped permission set - does not shadow the token an
# operator explicitly configured for org-wide PR search.
_TOKEN_ENV_VARS = ("DASHBOARD_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")

# GitHub asks for requests from one token to be made serially, and answers a
# burst with a secondary-limit 403 rather than a queue. The dashboard opens with
# a dozen reads at once, so without this lock every load raced itself into a 403
# that the pages then reported as "GitHub unavailable" - a throttle reading as an
# outage. The lock costs sequence, not time: the reads were never the slow part.
_REQUEST_LOCK = threading.Lock()
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0
# GitHub's secondary limit commonly asks for a minute, and asking again inside
# the window it named can extend the penalty, so the ceiling has to be able to
# hold that minute rather than truncating it into a request made during the ban.
_RETRY_CEILING_SECONDS = 60.0
# The instant before which nobody may ask again. A throttle applies to the token,
# not to the thread that happened to hit it: without this, the reader that was
# refused sleeps politely while the other dozen keep asking inside the very
# window GitHub closed, and they spend their own retries being refused too.
_NOT_BEFORE = 0.0
_STATE_LOCK = threading.Lock()
# The deadline of the read this thread is serving, shared by its every request.
_BUDGET = threading.local()
# Total seconds one read may spend waiting out throttles before it gives up: one
# full minute-long wait and a little more. A page that says "GitHub is throttling
# us" is more use than one that spins: the reader can come back, and the other
# sections still draw.
_RETRY_BUDGET_SECONDS = 75.0


def _retry_after(response: requests.Response, attempt: int) -> float | None:
    """How long to wait before asking again, or ``None`` when waiting is futile.

    A requested wait longer than the ceiling is cut down to the ceiling, never
    discarded: GitHub's secondary limit routinely asks for a minute, and ignoring
    that in favour of a two-second backoff spends every retry inside the window
    it was told to sit out, which is how a throttle became an error page.

    The hourly *primary* limit is the one exception. Its reset can be forty
    minutes away, and no retry this function could schedule will outlast it, so
    it returns ``None`` and the caller reports the exhausted quota at once rather
    than sleeping through a budget that cannot help.
    """
    reset_in = _seconds_until_reset(response)
    if (
        response.headers.get("x-ratelimit-remaining") == "0"
        and reset_in is not None
        and reset_in > _RETRY_CEILING_SECONDS
    ):
        return None
    raw = response.headers.get("retry-after", "")
    if raw.isdigit() and int(raw) > 0:
        return min(float(raw), _RETRY_CEILING_SECONDS)
    if reset_in is not None and reset_in > 0:
        return float(min(reset_in, _RETRY_CEILING_SECONDS))
    return min(_RETRY_BACKOFF_SECONDS * (2**attempt), _RETRY_CEILING_SECONDS)


def _seconds_until_reset(response: requests.Response) -> int | None:
    """Seconds until the quota named by ``x-ratelimit-reset`` refills."""
    raw = response.headers.get("x-ratelimit-reset", "")
    if not raw.isdigit():
        return None
    return int(int(raw) - time.time())


@contextmanager
def _read_budget() -> Iterator[None]:
    """Give one logical read a single deadline, however many requests it takes.

    A budget per request is not a budget: a merged-PR read is ten pages and then,
    if the rich payload fails, ten more, so a per-request allowance multiplies into
    minutes of a page that was promised a message instead of a spinner. The
    deadline lives on the thread, so the pages of one read - and the lean retry
    after them - share the allowance the first page opened, and a read that arrives
    with nothing left says so rather than waiting again.
    """
    if getattr(_BUDGET, "deadline", None) is not None:
        yield  # An inner read inherits the outer one's deadline, never a new one.
        return
    _BUDGET.deadline = time.monotonic() + _RETRY_BUDGET_SECONDS
    try:
        yield
    finally:
        _BUDGET.deadline = None


def _remaining() -> float:
    """Seconds this read may still spend waiting, counting time blocked on others."""
    deadline = getattr(_BUDGET, "deadline", None)
    if deadline is None:
        return _RETRY_BUDGET_SECONDS
    return deadline - time.monotonic()


def _hold_off(seconds: float) -> None:
    """Close the door on every reader until ``seconds`` from now."""
    global _NOT_BEFORE
    with _STATE_LOCK:
        _NOT_BEFORE = max(_NOT_BEFORE, time.monotonic() + seconds)


def _await_turn(allowance: float) -> float | None:
    """Sleep until the shared door opens, or ``None`` if ``allowance`` is too small.

    A reader that cannot afford the whole wait does not spend part of it and then
    ask anyway: that is a request made during the ban, which is what shut the door
    in the first place. It gives up its turn instead, and the page says so.
    """
    with _STATE_LOCK:
        delay = _NOT_BEFORE - time.monotonic()
    if delay <= 0:
        return 0.0
    if delay > allowance:
        return None
    time.sleep(delay)
    return delay


def _throttled(response: requests.Response) -> bool:
    """Whether a refusal is "too fast" or "not now", which waiting fixes.

    A gateway error belongs here as much as a rate limit does. GitHub's edge
    answers a burst with an nginx ``502`` as readily as with a documented 403, and
    the dashboard spent days reporting that as "GitHub unavailable" when asking
    again a moment later would have worked.
    """
    if response.status_code in (429, 502, 503, 504):
        return True
    if response.status_code != 403:
        return False
    body = response.text.lower()
    return "rate limit" in body or "secondary" in body or "try again" in body


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


def excluded_repos(org: str) -> tuple[str, ...]:
    """``owner/name`` for every repo ``GITHUB_EXCLUDE_REPOS`` takes out of scope.

    The exclusion belongs in the query rather than in the frame: a page that
    filters its own rows disagrees with the search counts, which cannot be
    filtered afterwards at all, and the same population then reads two ways
    depending on which page you are standing on.
    """
    raw = os.getenv("GITHUB_EXCLUDE_REPOS", "")
    names = []
    for entry in raw.split(","):
        name = entry.strip()
        if not name or not _REPO_RE.match(name):
            continue
        names.append(name if "/" in name else f"{org}/{name}")
    return tuple(dict.fromkeys(names))


def _without_excluded(org: str, query: str) -> str:
    """``query`` with a ``-repo:`` term per excluded repository."""
    terms = " ".join(f"-repo:{name}" for name in excluded_repos(org))
    return f"{query} {terms}" if terms else query


def _graphql(token: str, query: str, variables: dict) -> dict:
    """One GraphQL read: serialised, retried while throttled, quoted when refused.

    Three things a single request has to get right here.

    Serialised, because concurrent reads on one token are what GitHub's secondary
    limit exists to stop, and the dashboard's opening burst tripped it on every
    load. Retried with GitHub's own ``Retry-After`` up to a fixed budget, because
    a throttle is a wait, not a failure, but a reader would rather be told than
    watch a spinner. Quoted, because a bare
    ``403 Client Error`` cannot distinguish a missing permission, an org IP allow
    list and a secondary limit - the three days this bug survived were spent
    guessing between exactly those.

    The wait is shared rather than private: a refusal shuts the door for every
    reader until the instant GitHub named - including when this read is the one
    giving up, because the window outlives the reader that found it - and the door
    is checked while holding the lock, immediately before the request goes out.
    Checking it earlier would let the dozen readers already queued past a door
    that shut while they waited, and each would spend an attempt learning what the
    first one already knew.
    """
    response = None
    for attempt in range(_RETRY_ATTEMPTS):
        with _REQUEST_LOCK:
            # Asked for after the lock, so time spent blocked behind another
            # reader's wait is charged too: otherwise fourteen readers each
            # believe they have the whole budget and spend it one after another.
            if _await_turn(_remaining()) is None:
                break
            response = requests.post(
                GITHUB_GRAPHQL_URL,
                json={"query": query, "variables": variables},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=30,
            )
            # Shutting the door happens under the same lock that sent the
            # request. Released first, the reader queued behind would be woken,
            # read a door that had not yet been shut, and ask inside the window.
            refused = _throttled(response)
            pause = _retry_after(response, attempt) if refused else 0.0
            if refused:
                _hold_off(pause if pause is not None else _RETRY_CEILING_SECONDS)
        if not refused:
            break
        if (
            pause is None
            or attempt == _RETRY_ATTEMPTS - 1
            or pause > _remaining()
        ):
            break
    if response is None:
        raise GitHubConfigError(
            "GitHub is throttling this token; the reads were abandoned rather "
            "than sent during the wait it asked for. Reload in a minute."
        )
    if response.status_code >= 400:
        reason = " ".join(response.text.split())[:300]
        raise GitHubConfigError(
            f"GitHub refused the read: HTTP {response.status_code}"
            + (f" — {reason}" if reason else "")
        )
    payload = response.json()
    if payload.get("errors"):
        messages = "; ".join(e.get("message", "") for e in payload["errors"])
        raise GitHubConfigError(f"GitHub GraphQL error: {messages}")
    return payload["data"]


_PR_FIELDS = """
number
title
url
state
isDraft
reviewDecision
createdAt
updatedAt
mergedAt
closedAt
author { login }
repository { name }
approvingReviews: reviews(states: APPROVED) { totalCount }
changesReviews: reviews(states: CHANGES_REQUESTED) { totalCount }
allReviews: reviews { totalCount }
reviewRequests(first: 1) { totalCount }
"""

# Only the open-PR path looks for a Jira key, and a body can be thousands of
# words: asking for it on the merged query too would multiply that payload by
# the 1,000 PRs it pages through for a chart that never reads it.
_HYGIENE_ONLY_FIELDS = """
headRefName
bodyText
"""

_HYGIENE_FIELDS = _PR_FIELDS + _HYGIENE_ONLY_FIELDS

# Node budgets for the detail payload. GitHub charges a query by the product of
# the ``first:`` arguments down each branch, so each of these multiplies by the
# 100 PRs a search page returns: 50 reviews is 5,000 nodes a page, 20 timeline
# items another 2,000. They are constants so an operator who starts seeing 403s
# can cut the cost in one place without touching the query text.
_REVIEW_NODES = 50
_TIMELINE_NODES = 20
# Only ``totalCount`` is read off reviewThreads, and ``first:`` buys node budget
# for nodes we never select - so it is 1, not 100. Raise it only if this query
# ever starts selecting the threads themselves.
_REVIEW_THREAD_NODES = 1
# Same argument, taken to zero: the commit count is a scalar and no commit node
# is selected. If a server ever refuses a zero page size, this becomes 1 at a
# cost of 100 nodes a page - and until someone changes it, the whole detail
# query would fail and fall back to the lean one, which is loud rather than
# silent.
_COMMIT_NODES = 0

# What review quality is actually made of. ``additions``/``deletions``/
# ``changedFiles`` let output be weighted instead of counted (a README typo and
# a 900-line feature are both "1" without them); ``mergedBy`` and ``baseRefName``
# separate "merged by a colleague into trunk" from "merged by me into my own
# branch"; the review nodes carry who reviewed, when, and what the AI reviewer
# said; the counts say how many issues a PR actually drew; the timeline events
# give a start line for time-to-first-review, since a PR that sat in draft for a
# week was not waiting on a reviewer.
_DETAIL_ONLY_FIELDS = """
additions
deletions
changedFiles
commits(first: %d) { totalCount }
mergedBy { login }
baseRefName
reviewNodes: reviews(first: %d) {
  nodes { author { login } state submittedAt body }
}
reviewThreads(first: %d) { totalCount }
comments { totalCount }
timelineItems(itemTypes: [READY_FOR_REVIEW_EVENT, REVIEW_REQUESTED_EVENT], first: %d) {
  nodes {
    __typename
    ... on ReadyForReviewEvent { createdAt }
    ... on ReviewRequestedEvent { createdAt }
  }
}
""" % (_COMMIT_NODES, _REVIEW_NODES, _REVIEW_THREAD_NODES, _TIMELINE_NODES)

_DETAIL_FIELDS = _PR_FIELDS + _DETAIL_ONLY_FIELDS
_DETAIL_HYGIENE_FIELDS = _PR_FIELDS + _DETAIL_ONLY_FIELDS + _HYGIENE_ONLY_FIELDS

# A review body is prose written for a human; the AI reviewer's run to hundreds
# of lines. Only its presence and its first paragraph are ever read, and 1,000
# PRs times 50 reviews of unbounded text is a frame nobody can hold.
MAX_REVIEW_BODY_CHARS = 1000

_SEARCH_TEMPLATE = """
query($q: String!, $after: String) {
  search(query: $q, type: ISSUE, first: 100, after: $after) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes { ... on PullRequest { %s } }
  }
}
"""

_SEARCH_QUERY = _SEARCH_TEMPLATE % _PR_FIELDS
_HYGIENE_SEARCH_QUERY = _SEARCH_TEMPLATE % _HYGIENE_FIELDS
_DETAIL_SEARCH_QUERY = _SEARCH_TEMPLATE % _DETAIL_FIELDS
_DETAIL_HYGIENE_SEARCH_QUERY = _SEARCH_TEMPLATE % _DETAIL_HYGIENE_FIELDS


def _query_for(detail: bool, hygiene: bool) -> tuple[str, str]:
    """``(query, fallback)``: what to ask for, and what to retry with.

    The fallback is always the cheapest payload that still answers the caller's
    non-negotiable question - hygiene keeps its branch and body, because a
    traceability number computed off a payload that never carried them would be
    wrong rather than missing.
    """
    if detail and hygiene:
        return _DETAIL_HYGIENE_SEARCH_QUERY, _HYGIENE_SEARCH_QUERY
    if detail:
        return _DETAIL_SEARCH_QUERY, _SEARCH_QUERY
    if hygiene:
        return _HYGIENE_SEARCH_QUERY, _HYGIENE_SEARCH_QUERY
    return _SEARCH_QUERY, _SEARCH_QUERY


_COUNT_QUERY = """
query($q: String!) { search(query: $q, type: ISSUE, first: 1) { issueCount } }
"""


def _search_count(token: str, query: str) -> int:
    """Total matches for a search query, from GitHub's own ``issueCount``.

    Not bounded by pagination, so it stays correct even past the point where
    the result nodes themselves would be capped.
    """
    with _read_budget():
        data = _graphql(token, _COUNT_QUERY, {"q": query})
    return int(data["search"]["issueCount"])


def _page(token: str, gql: str, query: str, max_prs: int) -> list[dict]:
    nodes: list[dict] = []
    after: str | None = None
    while len(nodes) < max_prs:
        data = _graphql(token, gql, {"q": query, "after": after})
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


def _search_prs(
    token: str,
    query: str,
    max_prs: int,
    gql: str = _SEARCH_QUERY,
    fallback_gql: str | None = None,
) -> list[dict]:
    """Page a search, retrying once with a cheaper payload if the rich one fails.

    The expensive query is the one a rate limit kills first: it costs tens of
    times what the lean one does, and the token was already seen returning 403
    in production. So a failure of the detail payload degrades to the lean
    payload and starts the paging over, which loses columns but keeps the
    dashboard's PR counts alive. A failure of the lean payload is a real
    outage - no token, no network, no permission - and is raised as before.

    Paging restarts rather than resuming because the two payloads are different
    shapes; half a frame with detail columns and half without would read as
    "these people had no reviews" instead of "this page was never fetched".
    """
    with _read_budget():
        try:
            return _page(token, gql, query, max_prs)
        except (GitHubConfigError, requests.RequestException):
            if not fallback_gql or fallback_gql == gql:
                raise
        return _page(token, fallback_gql, query, max_prs)


def _int_or_none(value: object) -> int | None:
    """An int, or ``None`` when the field was not in the response.

    Deliberately not ``or 0``: a PR the API never described is not a PR of zero
    lines, and a metric that cannot tell those apart will read a throttled fetch
    as a team that shipped nothing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _count_or_none(node: dict, key: str) -> int | None:
    """``totalCount`` off a connection, or ``None`` when it was not requested."""
    connection = node.get(key)
    if not isinstance(connection, dict):
        return None
    return _int_or_none(connection.get("totalCount"))


def _review_rows(node: dict) -> list[dict] | None:
    """The PR's reviews as plain dicts, or ``None`` when they were not fetched.

    ``None`` and ``[]`` mean different things and both happen: ``[]`` is a PR
    nobody reviewed, which is the finding; ``None`` is the lean payload, which
    is no finding at all.
    """
    connection = node.get("reviewNodes")
    if not isinstance(connection, dict):
        return None
    rows: list[dict] = []
    for review in connection.get("nodes") or []:
        if not isinstance(review, dict):
            continue
        rows.append(
            {
                "reviewer": (review.get("author") or {}).get("login") or "unknown",
                "state": review.get("state") or "",
                "submitted_at": review.get("submittedAt"),
                "body": (review.get("body") or "")[:MAX_REVIEW_BODY_CHARS],
            }
        )
    return rows


def _review_ready_at(node: dict) -> str | None:
    """When the PR first asked for a reviewer, from the timeline events.

    The earliest "ready for review" or "review requested" event, because the
    clock a reviewer can be judged against starts when they were asked, not when
    a draft was opened. ``None`` when the timeline was not fetched or the PR
    never asked anyone - callers fall back to ``created_at`` and say so.
    """
    timeline = node.get("timelineItems")
    if not isinstance(timeline, dict):
        return None
    stamps = [
        item.get("createdAt")
        for item in timeline.get("nodes") or []
        if isinstance(item, dict) and item.get("createdAt")
    ]
    return min(stamps) if stamps else None


def _to_frame(nodes: list[dict]) -> pd.DataFrame:
    rows = []
    for n in nodes:
        additions = _int_or_none(n.get("additions"))
        deletions = _int_or_none(n.get("deletions"))
        rows.append(
            {
                "number": n.get("number"),
                "title": n.get("title") or "",
                "url": n.get("url") or "",
                "state": n.get("state") or "",
                "is_draft": bool(n.get("isDraft")),
                # Absent on the merged-PR query, which does not request them.
                "branch": n.get("headRefName") or "",
                # Truncated because only the first Jira key in it is ever read.
                "body": (n.get("bodyText") or "")[:2000],
                # Whether branch and body were asked for at all. Without this a
                # merged PR fetched on the lean payload looks like a PR that
                # failed to name its ticket, rather than one nobody asked about.
                "hygiene_fetched": ("headRefName" in n) or ("bodyText" in n),
                "review_decision": n.get("reviewDecision"),
                "created_at": n.get("createdAt"),
                "updated_at": n.get("updatedAt"),
                "merged_at": n.get("mergedAt"),
                "closed_at": n.get("closedAt"),
                "author": (n.get("author") or {}).get("login") or "unknown",
                "repo": (n.get("repository") or {}).get("name") or "",
                "approving_reviews": (n.get("approvingReviews") or {}).get("totalCount", 0),
                "changes_reviews": (n.get("changesReviews") or {}).get("totalCount", 0),
                "total_reviews": (n.get("allReviews") or {}).get("totalCount", 0),
                "review_requests": (n.get("reviewRequests") or {}).get("totalCount", 0),
                # Everything below is the detail payload: None when the lean
                # query answered, whether by choice or after a throttled retry.
                "additions": additions,
                "deletions": deletions,
                "changed_files": _int_or_none(n.get("changedFiles")),
                "changed_lines": (
                    additions + deletions
                    if additions is not None and deletions is not None
                    else None
                ),
                "commits": _count_or_none(n, "commits"),
                "merged_by": (n.get("mergedBy") or {}).get("login"),
                "base_branch": n.get("baseRefName"),
                "review_threads": _count_or_none(n, "reviewThreads"),
                "comments": _count_or_none(n, "comments"),
                "reviews": _review_rows(n),
                "review_ready_at": _review_ready_at(n),
                "detail_fetched": "additions" in n,
            }
        )
    frame = pd.DataFrame(rows)
    for col in ("created_at", "updated_at", "merged_at", "closed_at", "review_ready_at"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    return frame


def _open_query(org: str) -> str:
    return _without_excluded(org, f"org:{org} is:pr is:open draft:false") + " sort:created-asc"


def open_pr_count(token: str, org: str) -> int:
    """Exact count of open, non-draft PRs org-wide (never paging-capped)."""
    return _search_count(token, _open_query(org))


def fetch_open_prs(
    token: str, org: str, max_prs: int = 400, detail: bool = True
) -> pd.DataFrame:
    """Open, non-draft PRs across the org, oldest first, with review counts.

    ``detail`` adds the diff size, the reviews and the timeline events the
    review-quality metrics need. It is on by default here because the open set
    is the bounded one - a few hundred PRs, four pages - and because the
    questions it answers (who is waiting on a review, who reviews anyone) are
    only actionable while the PR is still open. It degrades to the lean payload
    on its own if the token cannot afford it.
    """
    gql, fallback = _query_for(detail=detail, hygiene=True)
    frame = _to_frame(_search_prs(token, _open_query(org), max_prs, gql, fallback))
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
    return _without_excluded(org, f"org:{org} is:pr is:merged merged:>={since}")


def merged_pr_count(token: str, org: str, days: int) -> int:
    """Exact count of PRs merged org-wide in the window (never paging-capped)."""
    return _search_count(token, _merged_query(org, days))


def fetch_merged_prs(
    token: str,
    org: str,
    days: int,
    max_prs: int = 1000,
    detail: bool = True,
    hygiene: bool = False,
) -> pd.DataFrame:
    """PRs merged anywhere in the org within the last ``days`` (for the pie).

    Capped at ``max_prs`` (GitHub search only exposes the first 1,000 results);
    the headline tile uses :func:`merged_pr_count` so it is not affected.

    ``detail`` is on by default because this is the only fetch that can answer
    the questions about shipped work: how large the merged change was, who
    pressed merge, whether anyone approved it first. ``hygiene`` (branch and
    body, for the Jira key) stays off by default: a body is thousands of words
    and this query pages through up to 1,000 PRs, so it is opt-in for the
    traceability view that actually reads it - preferably over a shorter window.
    """
    gql, fallback = _query_for(detail=detail, hygiene=hygiene)
    return _to_frame(_search_prs(token, _merged_query(org, days), max_prs, gql, fallback))


def _closed_unmerged_query(org: str, days: int) -> str:
    since = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return _without_excluded(org, f"org:{org} is:pr is:closed is:unmerged closed:>={since}")


def closed_unmerged_pr_count(token: str, org: str, days: int) -> int:
    """Exact count of PRs closed without merging in the window.

    The denominator half of this - PRs merged in the same window - already has
    :func:`merged_pr_count`, so the pair gives an abandoned-work rate that no
    amount of re-opening and re-closing a branch can inflate.
    """
    return _search_count(token, _closed_unmerged_query(org, days))


def fetch_closed_unmerged_prs(
    token: str, org: str, days: int, max_prs: int = 500, detail: bool = False
) -> pd.DataFrame:
    """PRs closed without merging in the last ``days``, per author.

    Abandoned work is invisible in every existing view: a PR that is opened,
    reviewed, argued over and then binned costs exactly as many hours as one
    that ships, and leaves no trace in the merged counts. ``detail`` is off by
    default because size and reviews on a discarded branch answer a much smaller
    question than the count itself does.

    A high rate is not automatically waste - superseded and duplicate PRs get
    closed for good reasons - which is why this returns the PRs and not a
    verdict.
    """
    gql, fallback = _query_for(detail=detail, hygiene=False)
    return _to_frame(
        _search_prs(token, _closed_unmerged_query(org, days), max_prs, gql, fallback)
    )
