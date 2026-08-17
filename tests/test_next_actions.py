"""The landing page's counts, turned into named work somebody can do."""

import pandas as pd

import next_actions


def _url(key: str) -> str:
    return f"https://jira.example/browse/{key}"


def _prs(rows: list[dict]) -> pd.DataFrame:
    base = {
        "repo": "frontend",
        "number": 1,
        "url": "https://github.com/o/frontend/pull/1",
        "author": "tam",
        "is_draft": False,
        "approving_reviews": 0,
        "total_reviews": 0,
        "review_requests": 0,
        "age_days": 1.0,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_a_pr_nobody_was_asked_to_review_asks_for_a_reviewer():
    actions = next_actions.review_actions(_prs([{"number": 205, "age_days": 12.0}]))
    assert len(actions) == 1
    assert actions[0].verb == "Name a reviewer for"
    assert actions[0].subject == "frontend#205"
    assert actions[0].url.endswith("/pull/1")
    assert "asked nobody" in actions[0].detail
    assert actions[0].days == 12.0


def test_a_requested_but_unread_pr_asks_for_the_reviewer_to_be_chased():
    actions = next_actions.review_actions(_prs([{"review_requests": 1}]))
    assert actions[0].verb == "Chase the reviewer on"


def test_a_reviewed_but_unapproved_pr_asks_for_a_decision():
    actions = next_actions.review_actions(
        _prs([{"review_requests": 1, "total_reviews": 2}])
    )
    assert actions[0].verb == "Get a decision on"


def test_an_approved_pr_is_not_an_action():
    assert next_actions.review_actions(_prs([{"approving_reviews": 1}])) == []


def test_a_draft_is_nobodys_review_to_do():
    assert next_actions.review_actions(_prs([{"is_draft": True}])) == []


def test_review_actions_open_with_the_longest_wait():
    actions = next_actions.review_actions(
        _prs([{"number": 1, "age_days": 2.0}, {"number": 2, "age_days": 40.0}])
    )
    assert [action.subject for action in actions] == ["frontend#2", "frontend#1"]


def test_triage_actions_ask_for_a_decision_and_link_the_ticket():
    triage = pd.DataFrame(
        [{"key": "MB-1", "summary": "Checkout 500s", "ticket_age_days": 9.0}]
    )
    action = next_actions.triage_actions(triage, url_for=_url)[0]
    assert action.verb == "Accept or close"
    assert action.subject == "MB-1"
    assert action.url == "https://jira.example/browse/MB-1"
    # The age of the ticket, not a claim about how long it has been in triage:
    # the triage read carries no status history to measure that from.
    assert "9d old, still untriaged" in action.detail
    assert "in triage" not in action.detail


def test_an_empty_triage_queue_has_nothing_to_do():
    assert next_actions.triage_actions(pd.DataFrame(), url_for=_url) == []
    assert next_actions.triage_actions(None, url_for=_url) == []


def test_ownership_actions_name_the_ticket_that_belongs_to_nobody():
    ownerless = pd.DataFrame(
        [{"key": "MB-2", "status": "To Do", "ticket_age_days": 200.0}]
    )
    action = next_actions.ownership_actions(ownerless, url_for=_url)[0]
    assert action.verb == "Assign an owner to"
    assert "200d old" in action.detail
    assert "To Do" in action.detail


def test_stalled_actions_address_the_owner_by_name():
    stalled = pd.DataFrame([{"key": "MB-3", "assignee": "Mehdi", "idle_days": 45.0}])
    action = next_actions.stalled_actions(stalled, url_for=_url)[0]
    assert action.verb == "Ask Mehdi about"
    assert action.detail == "no movement in 45d"


def test_a_stalled_ticket_with_no_owner_is_still_asked_about():
    stalled = pd.DataFrame([{"key": "MB-4", "assignee": None, "idle_days": 45.0}])
    action = next_actions.stalled_actions(stalled, url_for=_url)[0]
    assert action.verb == "Find out what happened to"


def test_a_row_with_no_key_cannot_be_linked_so_is_not_offered():
    frame = pd.DataFrame([{"key": "", "status": "To Do", "ticket_age_days": 1.0}])
    assert next_actions.ownership_actions(frame, url_for=_url) == []


def _action(kind: str, days: float) -> next_actions.Action:
    return next_actions.Action(
        kind=kind,
        verb="Do",
        subject=f"{kind}-{days:.0f}",
        url="https://example.com",
        detail="",
        owner="",
        days=days,
    )


def test_the_opening_list_spans_the_problems_rather_than_one_long_queue():
    queues = {
        "review": [_action("review", days) for days in (1200.0, 1100.0, 1000.0)],
        "triage": [_action("triage", 3.0)],
        "ownership": [_action("ownership", 400.0)],
    }
    chosen = next_actions.rank(queues, limit=3)
    assert {action.kind for action in chosen} == {"review", "triage", "ownership"}


def test_within_a_round_the_longest_wait_goes_first():
    queues = {
        "review": [_action("review", 10.0)],
        "triage": [_action("triage", 90.0)],
    }
    assert [action.kind for action in next_actions.rank(queues, limit=2)] == [
        "triage",
        "review",
    ]


def test_a_deep_queue_supplies_the_rest_once_the_others_are_spent():
    queues = {
        "review": [_action("review", days) for days in (9.0, 8.0, 7.0)],
        "triage": [_action("triage", 100.0)],
    }
    chosen = next_actions.rank(queues, limit=4)
    assert [action.kind for action in chosen] == [
        "triage",
        "review",
        "review",
        "review",
    ]


def test_nothing_waiting_is_no_actions_rather_than_an_error():
    assert next_actions.rank({"review": [], "triage": []}) == []


def test_the_table_carries_the_link_in_its_own_column():
    frame = next_actions.as_frame([_action("review", 5.0)])
    assert list(frame.columns) == [
        "Do this",
        "Item",
        "Open",
        "Why",
        "Waiting (days)",
        "Owner",
    ]
    assert frame.loc[0, "Open"] == "https://example.com"
    assert frame.loc[0, "Owner"] == "—"


def test_an_action_reads_as_one_sentence():
    action = next_actions.Action(
        kind="review",
        verb="Name a reviewer for",
        subject="frontend#205",
        url="",
        detail="open 12d, tam asked nobody",
        owner="tam",
        days=12.0,
    )
    assert action.sentence == "Name a reviewer for frontend#205 — open 12d, tam asked nobody"


def test_a_triage_frame_that_carries_no_age_column_still_yields_actions():
    """The raw triage read has a created date and no age, and used to crash the page."""
    triage = pd.DataFrame([{"key": "MB-9", "summary": "500s on pay"}])
    actions = next_actions.triage_actions(triage, url_for=_url)
    assert [action.subject for action in actions] == ["MB-9"]
    assert actions[0].days == 0.0


def test_stalled_wording_prefers_the_selection_clock_over_the_edit_clock():
    stalled = pd.DataFrame(
        [
            {
                "key": "ENG-1",
                "assignee": "Tam",
                "idle_days": 1.0,
                next_actions.STALLED_AGE_COLUMN: 61.0,
            }
        ]
    )
    action = next_actions.stalled_actions(stalled, url_for=_url)[0]
    assert action.detail == "no movement in 61d"
    assert action.days == 61.0
