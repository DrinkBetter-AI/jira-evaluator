"""Tests for snippets.py - #team-snippets compliance.

No real Slack call is made anywhere in this file - there is no Slack bot
token in this environment. Every test either exercises the no-token path
directly, or drives the module against a stub HTTP session returning
fixture payloads shaped like real ``conversations.history`` /
``conversations.replies`` responses.
"""

from __future__ import annotations

import datetime as _dt

import pytest

import roles
import snippets

# The "now" DEVIN_PLAN.md's baseline was measured against: a date inside
# ISO week 2026-W33, so the eight full weeks immediately before it are
# 2026-W25 through 2026-W32 - the exact window the baseline documents.
_BASELINE_NOW = _dt.date(2026, 8, 12)


def _ts(year: int, week: int, weekday: int, hour: int = 12, sub: str = "000000") -> str:
    """A Slack-shaped timestamp string for a given ISO (year, week, weekday)."""
    day = _dt.date.fromisocalendar(year, week, weekday)
    moment = _dt.datetime.combine(day, _dt.time(hour=hour), tzinfo=_dt.timezone.utc)
    return f"{int(moment.timestamp())}.{sub}"


def _msg(user: str, ts: str, reply_count: int | None = None, thread_ts: str | None = None) -> dict:
    message = {"type": "message", "user": user, "text": "snippet", "ts": ts}
    if reply_count is not None:
        message["reply_count"] = reply_count
        message["thread_ts"] = thread_ts or ts
    elif thread_ts is not None:
        # A reply carries thread_ts pointing at its parent even though it has
        # no reply_count of its own - only a thread's root message does.
        message["thread_ts"] = thread_ts
    return message


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(self._payload)

    def json(self) -> dict:
        return self._payload


class StubSlack:
    """A minimal stand-in for ``requests`` recording every call it serves."""

    def __init__(self, resolver):
        self._resolver = resolver
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, headers=None, timeout=None):
        assert url.startswith(snippets.SLACK_API_BASE)
        path = url[len(snippets.SLACK_API_BASE):]
        params = dict(params or {})
        self.calls.append((path, params))
        return self._resolver(path, params)


class RaisingSlack:
    """Fails the test if anything ever calls it - proves the no-token path never reaches the network."""

    def get(self, *args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("Slack was called despite no token being configured")


# ---------------------------------------------------------------------------
# Roster / Slack-ID fixture shared by several tests
# ---------------------------------------------------------------------------

# Slack user IDs are a third identity namespace with no relation to Jira
# names or GitHub logins - these are fixture-only IDs, invented for the
# test, and are not real Slack user IDs.
_UID = {
    "Farid Shahidi": "UFARID001",
    "Tam": "UTAM000001",
    "Mohsen Davoudi": "UMOHSEN001",
    "Mehdi Ordikhani": "UMEHDI0001",
    "Mihai Manea": "UMIHAI0001",
    "David": "UDAVID0001",
    "Igor Taborsak": "UIGOR00001",
    "Anouar Kacem": "UANOUAR001",
    "Santi Caamaño": "USANTI0001",
    "Alesya Kasovich": "UALESYA001",
    "Ali": "UALI000001",
    "Shawn": "USHAWN0001",
    "Jal Haidar": "UJAL000001",
    "Gaston": "UGASTON001",
    "Dina QA": "UDINA00001",
    "Robert Surpateanu": "UROBERT001",
}

_BASELINE_WEEKS_POSTED = {
    "Farid Shahidi": 8,
    "Tam": 7,
    "Mohsen Davoudi": 7,
    "Mehdi Ordikhani": 7,
    "Mihai Manea": 6,
    "David": 4,
    "Igor Taborsak": 4,
    "Anouar Kacem": 3,
    "Santi Caamaño": 2,
    "Alesya Kasovich": 2,
    "Ali": 1,
    "Shawn": 0,
    "Jal Haidar": 0,
    "Gaston": 0,
    "Dina QA": 0,
    "Robert Surpateanu": 0,
}

_SLACK_USER_MAP = {uid: name for name, uid in _UID.items()}


def _baseline_messages() -> tuple[list[dict], list[dict]]:
    """Top-level and reply message lists reproducing the W25-W32 baseline.

    Farid's first week carries three top-level messages (one week must
    still count once). One of them opens a thread David replies to twice
    in a week David has no top-level post of his own that week (replies
    must not inflate David's top-level weeks_posted). An unmapped Slack ID
    posts twice, and Jal has one message from outside the window entirely
    (2026-W07, "once ever, in Feb" per DEVIN_PLAN.md) that must not count.
    """
    top: list[dict] = []
    replies: list[dict] = []

    thread_root_ts = _ts(2026, 25, 1, hour=9)
    top.append(_msg(_UID["Farid Shahidi"], thread_root_ts, reply_count=2, thread_ts=thread_root_ts))
    top.append(_msg(_UID["Farid Shahidi"], _ts(2026, 25, 1, hour=10)))
    top.append(_msg(_UID["Farid Shahidi"], _ts(2026, 25, 1, hour=11)))

    for name, count in _BASELINE_WEEKS_POSTED.items():
        if name == "Farid Shahidi":
            # Week 0 already covered by the three messages above.
            weeks = range(26, 26 + (count - 1))
        else:
            weeks = range(25, 25 + count)
        for week in weeks:
            top.append(_msg(_UID[name], _ts(2026, week, 3)))

    # David replies in the thread Farid opened, in week index 5 (2026-W30),
    # a week David posted no top-level message of his own that week.
    replies.append(_msg(_UID["David"], _ts(2026, 30, 1, hour=9, sub="000100"), thread_ts=thread_root_ts))
    replies.append(_msg(_UID["David"], _ts(2026, 30, 1, hour=9, sub="000200"), thread_ts=thread_root_ts))

    # An unmapped author - posts, but SLACK_USER_MAP names nobody for this ID.
    top.append(_msg("UPRAVEEN001", _ts(2026, 25, 2)))
    top.append(_msg("UPRAVEEN001", _ts(2026, 26, 2)))

    # Jal's one-ever post, outside the W25-W32 window (2026-W07, "in Feb").
    top.append(_msg(_UID["Jal Haidar"], _ts(2026, 7, 2)))

    return top, replies


def _resolver_for(top: list[dict], replies: list[dict]):
    """A StubSlack resolver serving ``top`` as one history page and ``replies`` per thread."""

    def resolve(path, params):
        if path == snippets._HISTORY_PATH:
            return FakeResponse(200, {"ok": True, "messages": top, "has_more": False})
        if path == snippets._REPLIES_PATH:
            thread_ts = params.get("ts")
            parent = next((m for m in top if m["ts"] == thread_ts), None)
            thread_messages = ([parent] if parent else []) + [
                m for m in replies if m.get("thread_ts") == thread_ts
            ]
            return FakeResponse(200, {"ok": True, "messages": thread_messages, "has_more": False})
        raise AssertionError(f"unexpected path {path}")

    return resolve


# ---------------------------------------------------------------------------
# No-token path
# ---------------------------------------------------------------------------


def test_no_token_is_unavailable_not_an_exception():
    report = snippets.weekly_snippet_report(env={}, session=RaisingSlack(), now=_BASELINE_NOW)
    assert report.available is False
    assert report.partial is False
    assert "SLACK_BOT_TOKEN" in report.reason
    assert report.people == {}
    assert report.unmapped == {}
    # The window is still computed even when unavailable, so a page can say
    # *which* weeks it can't report on.
    assert report.weeks == snippets.recent_full_weeks(_BASELINE_NOW, 8)


def test_fetch_channel_posts_with_no_token_does_not_raise_or_call_out():
    fetch = snippets.fetch_channel_posts(None, session=RaisingSlack())
    assert fetch.available is False
    assert fetch.top_level == ()
    assert fetch.replies == ()
    assert "SLACK_BOT_TOKEN" in fetch.reason

    fetch_empty = snippets.fetch_channel_posts("", session=RaisingSlack())
    assert fetch_empty.available is False


def test_load_slack_env_returns_none_when_unset():
    assert snippets.load_slack_env(env={}) is None
    assert snippets.load_slack_env(env={"SLACK_BOT_TOKEN": "  "}) is None
    assert snippets.load_slack_env(env={"SLACK_BOT_TOKEN": "xoxb-real"}) == "xoxb-real"


def test_load_slack_user_map_has_no_baked_default():
    # Unlike roles.load_roster, an unset SLACK_USER_MAP maps nobody - there
    # is no confirmed Slack ID for anyone in this codebase to bake in.
    assert snippets.load_slack_user_map(env={}) == {}


def test_load_slack_user_map_parses_pairs_and_skips_malformed_entries():
    raw = "Farid Shahidi=U123;;garbage;Tam=U456"
    mapping = snippets.load_slack_user_map(env={"SLACK_USER_MAP": raw})
    assert mapping == {"U123": "Farid Shahidi", "U456": "Tam"}


# ---------------------------------------------------------------------------
# The W25-W32 baseline, fixture-driven
# ---------------------------------------------------------------------------


def test_baseline_weeks_posted_reproduces_exactly():
    top, replies = _baseline_messages()
    stub = StubSlack(_resolver_for(top, replies))
    report = snippets.weekly_snippet_report(
        env={"SLACK_BOT_TOKEN": "xoxb-fixture", "SLACK_USER_MAP": _raw_slack_user_map()},
        session=stub,
        now=_BASELINE_NOW,
    )
    assert report.available is True
    assert report.partial is False
    assert report.weeks == ("2026-W25", "2026-W26", "2026-W27", "2026-W28", "2026-W29", "2026-W30", "2026-W31", "2026-W32")

    for name, expected in _BASELINE_WEEKS_POSTED.items():
        stats = report.people[name]
        assert stats.weeks_posted == expected, f"{name}: expected {expected}, got {stats.weeks_posted}"
        assert stats.weeks_in_window == 8


def _raw_slack_user_map() -> str:
    return ";".join(f"{name}={uid}" for name, uid in _UID.items())


def test_baseline_farid_streak_is_full_and_unbroken():
    top, replies = _baseline_messages()
    stub = StubSlack(_resolver_for(top, replies))
    report = snippets.weekly_snippet_report(
        env={"SLACK_BOT_TOKEN": "xoxb-fixture", "SLACK_USER_MAP": _raw_slack_user_map()},
        session=stub,
        now=_BASELINE_NOW,
    )
    farid = report.people["Farid Shahidi"]
    assert farid.current_streak == 8
    assert farid.weeks_posted == 8


def test_baseline_jals_february_post_does_not_count_in_window():
    top, replies = _baseline_messages()
    stub = StubSlack(_resolver_for(top, replies))
    report = snippets.weekly_snippet_report(
        env={"SLACK_BOT_TOKEN": "xoxb-fixture", "SLACK_USER_MAP": _raw_slack_user_map()},
        session=stub,
        now=_BASELINE_NOW,
    )
    jal = report.people["Jal Haidar"]
    assert jal.weeks_posted == 0
    assert jal.top_level_total == 0


def test_baseline_gaston_never_posted_still_appears_as_a_real_zero():
    top, replies = _baseline_messages()
    stub = StubSlack(_resolver_for(top, replies))
    report = snippets.weekly_snippet_report(
        env={"SLACK_BOT_TOKEN": "xoxb-fixture", "SLACK_USER_MAP": _raw_slack_user_map()},
        session=stub,
        now=_BASELINE_NOW,
    )
    assert "Gaston" in report.people
    gaston = report.people["Gaston"]
    assert gaston.weeks_posted == 0
    assert gaston.top_level_total == 0
    assert gaston.reply_total == 0


# ---------------------------------------------------------------------------
# Individual acceptance behaviours
# ---------------------------------------------------------------------------


def test_three_posts_in_one_week_count_as_one_week_of_streak():
    uid = "USOLO0001"
    top = [
        _msg(uid, _ts(2026, 25, 1, hour=9)),
        _msg(uid, _ts(2026, 25, 1, hour=10)),
        _msg(uid, _ts(2026, 25, 1, hour=11)),
    ]
    stub = StubSlack(_resolver_for(top, []))
    ros = roles.load_roster(env={"JIRA_ROLES": "backend=Solo"})
    fetch = snippets.fetch_channel_posts("xoxb-fixture", session=stub)
    weeks_window = snippets.recent_full_weeks(_BASELINE_NOW, 8)
    report = snippets.build_report(fetch, ros, {uid: "Solo"}, weeks_window)

    solo = report.people["Solo"]
    assert solo.weeks_posted == 1
    assert solo.top_level_by_week["2026-W25"] == 3
    assert solo.top_level_total == 3


def test_thread_replies_are_counted_separately_and_never_inflate_top_level():
    top, replies = _baseline_messages()
    stub = StubSlack(_resolver_for(top, replies))
    report = snippets.weekly_snippet_report(
        env={"SLACK_BOT_TOKEN": "xoxb-fixture", "SLACK_USER_MAP": _raw_slack_user_map()},
        session=stub,
        now=_BASELINE_NOW,
    )
    david = report.people["David"]
    # David's own top-level activity: 4 weeks (2026-W25..W28), unaffected by
    # his two thread replies in 2026-W30.
    assert david.weeks_posted == 4
    assert david.top_level_by_week["2026-W30"] == 0
    assert david.reply_by_week["2026-W30"] == 2
    assert david.reply_total == 2
    assert david.top_level_total == 4  # one post per week, four weeks


def test_unmapped_slack_user_is_reported_not_dropped_or_misattributed():
    top, replies = _baseline_messages()
    stub = StubSlack(_resolver_for(top, replies))
    report = snippets.weekly_snippet_report(
        env={"SLACK_BOT_TOKEN": "xoxb-fixture", "SLACK_USER_MAP": _raw_slack_user_map()},
        session=stub,
        now=_BASELINE_NOW,
    )
    assert report.unmapped == {"UPRAVEEN001": 2}
    # Not folded into anybody's real total.
    for stats in report.people.values():
        assert "Praveen" not in stats.name


def test_pagination_follows_has_more_and_counts_the_second_page():
    uid_page1 = "UPAGE00001"
    uid_page2 = "UPAGE00002"
    page1 = {
        "ok": True,
        "messages": [_msg(uid_page1, _ts(2026, 25, 1))],
        "has_more": True,
        "response_metadata": {"next_cursor": "cursor-abc"},
    }
    page2 = {
        "ok": True,
        "messages": [_msg(uid_page2, _ts(2026, 26, 1))],
        "has_more": False,
    }

    def resolve(path, params):
        assert path == snippets._HISTORY_PATH
        if "cursor" in params:
            assert params["cursor"] == "cursor-abc"
            return FakeResponse(200, page2)
        return FakeResponse(200, page1)

    stub = StubSlack(resolve)
    fetch = snippets.fetch_channel_posts("xoxb-fixture", session=stub)

    assert fetch.available is True
    assert fetch.partial is False
    seen_users = {m["user"] for m in fetch.top_level}
    assert seen_users == {uid_page1, uid_page2}
    # Prove the second page was actually requested with the cursor.
    history_calls = [c for c in stub.calls if c[0] == snippets._HISTORY_PATH]
    assert len(history_calls) == 2
    assert history_calls[1][1]["cursor"] == "cursor-abc"


def test_rate_limit_degrades_to_flagged_partial_not_a_raise():
    uid = "URATE0001"
    page1 = {
        "ok": True,
        "messages": [_msg(uid, _ts(2026, 25, 1))],
        "has_more": True,
        "response_metadata": {"next_cursor": "cursor-xyz"},
    }

    def resolve(path, params):
        assert path == snippets._HISTORY_PATH
        if "cursor" in params:
            return FakeResponse(429, {})
        return FakeResponse(200, page1)

    stub = StubSlack(resolve)
    fetch = snippets.fetch_channel_posts("xoxb-fixture", session=stub)

    assert fetch.available is True
    assert fetch.partial is True
    assert "429" in fetch.reason or "rate limit" in fetch.reason.lower()
    # The first page's message survives - not a silent truncation to nothing.
    assert len(fetch.top_level) == 1
    assert fetch.top_level[0]["user"] == uid


def test_rate_limit_on_thread_replies_also_marks_partial():
    thread_root = _ts(2026, 25, 1)
    top = [_msg("UROOT0001", thread_root, reply_count=1, thread_ts=thread_root)]

    def resolve(path, params):
        if path == snippets._HISTORY_PATH:
            return FakeResponse(200, {"ok": True, "messages": top, "has_more": False})
        if path == snippets._REPLIES_PATH:
            return FakeResponse(429, {})
        raise AssertionError(path)

    stub = StubSlack(resolve)
    fetch = snippets.fetch_channel_posts("xoxb-fixture", session=stub)
    assert fetch.available is True
    assert fetch.partial is True
    assert fetch.top_level  # the top-level post itself is still there
    assert fetch.replies == ()


def test_slack_api_error_payload_degrades_without_raising():
    def resolve(path, params):
        return FakeResponse(200, {"ok": False, "error": "invalid_auth"})

    stub = StubSlack(resolve)
    fetch = snippets.fetch_channel_posts("xoxb-bad-token", session=stub)
    assert fetch.available is True  # the token existed; the request itself failed
    assert fetch.partial is True
    assert "invalid_auth" in fetch.reason
    assert fetch.top_level == ()


# ---------------------------------------------------------------------------
# this_week_poster_count
# ---------------------------------------------------------------------------


def test_this_week_poster_count_uses_the_latest_window_week_top_level_only():
    top, replies = _baseline_messages()
    stub = StubSlack(_resolver_for(top, replies))
    report = snippets.weekly_snippet_report(
        env={"SLACK_BOT_TOKEN": "xoxb-fixture", "SLACK_USER_MAP": _raw_slack_user_map()},
        session=stub,
        now=_BASELINE_NOW,
    )
    posters, roster_size = snippets.this_week_poster_count(report)
    latest_week = report.weeks[-1]
    expected = sum(1 for s in report.people.values() if s.top_level_by_week[latest_week] > 0)
    assert posters == expected
    assert roster_size == report.roster_size


def test_this_week_poster_count_handles_empty_window():
    empty_report = snippets.SnippetReport(
        available=True, partial=False, reason="", weeks=(), people={}, unmapped={}, roster_size=12,
    )
    assert snippets.this_week_poster_count(empty_report) == (0, 12)


# ---------------------------------------------------------------------------
# Week bucketing sanity
# ---------------------------------------------------------------------------


def test_recent_full_weeks_never_includes_the_in_progress_week():
    # 2026-08-12 falls in ISO week 2026-W33; the window must stop at W32.
    weeks = snippets.recent_full_weeks(_BASELINE_NOW, 8)
    assert weeks[-1] == "2026-W32"
    assert "2026-W33" not in weeks
    assert len(weeks) == 8


def test_recent_full_weeks_zero_count_is_empty():
    assert snippets.recent_full_weeks(_BASELINE_NOW, 0) == ()
