"""Pins ``integrity.changelog_events`` (and what reads it) against a hand-built payload.

``tests/fixtures/changelog_snapshot.json`` is the fixture: the shape
``JiraClient.search_issues(expand="changelog")`` actually returns for two
issues, seven changelog entries, every one of them chosen for a specific
reason (see the ``_comment`` key in that file, and the per-row comments
below). Nothing in it is random - a fuzzed or generated changelog would
happen to exercise these paths, but a reader could never tell which case a
given row was standing in for, and a parsing regression could hide inside
"probably fine" noise. Every field of every parsed row is asserted here
explicitly, so a drift in attribution, timing, or field classification shows
up as "this exact field, this exact row" rather than a vague frame-shape
diff.

``test_a_single_mutated_field_fails_exactly_one_row_comparison`` is the proof
that the pin actually pins something: it changes one character of the
fixture and shows that exactly one of the per-field checks below stops
matching - not zero (the fixture and the pin drifting together, silently)
and not many (the checks not being independent of each other).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import integrity  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "changelog_snapshot.json"


def _load_issues() -> list[dict]:
    payload = json.loads(FIXTURE.read_text())
    return payload["issues"]


# ---------------------------------------------------------------------------
# The pin: one dict per expected row, in the order changelog_events() sorts
# to (key, then ts). Every value is written out by hand against the fixture
# above and against changelog_events()'s own documented column meanings
# (integrity.py, "changelog_events" docstring and EVENT_COLUMNS) - not
# derived from running the function once and copying its answer.
# ---------------------------------------------------------------------------

EXPECTED_ROWS = [
    # VV-100/h1: a plain status transition, both ends unresolved statuses,
    # named author. The baseline case every other row is a variation on.
    {
        "key": "VV-100",
        "entry_id": "10001",
        "ts": pd.Timestamp("2026-01-05T09:00:00", tz="UTC"),
        "author": "Priya Shah",
        "author_id": "acc-priya",
        "field": "status",
        "field_id": "status",
        "from_string": "To Do",
        "to_string": "In Progress",
        "from_id": None,
        "to_id": None,
        "is_status": True,
        "is_cosmetic": False,
        "is_estimate": False,
        "is_sprint_rollover": False,
        "from_stage": 1.0,  # STATUS_STAGES["to do"]
        "to_stage": 2.0,  # STATUS_STAGES["in progress"]
    },
    # VV-100/h2: a label-only edit - COSMETIC_FIELDS, not a status move.
    # is_status False means from_stage/to_stage are never computed for it
    # (changelog_events only fills them in "if s", s being is_status).
    {
        "key": "VV-100",
        "entry_id": "10002",
        "ts": pd.Timestamp("2026-01-06T14:30:00", tz="UTC"),
        "author": "Priya Shah",
        "author_id": "acc-priya",
        "field": "labels",
        "field_id": "labels",
        "from_string": None,
        "to_string": "needs-design",
        "from_id": None,
        "to_id": None,
        "is_status": False,
        "is_cosmetic": True,
        "is_estimate": False,
        "is_sprint_rollover": False,
        "from_stage": None,
        "to_stage": None,
    },
    # VV-100/h3: the resolving transition - In Progress (not in
    # DEFAULT_RESOLVED_STATUSES) to Review in Staging (is). This is the
    # ticket's first "resolution" per reresolve_events() (asserted below).
    {
        "key": "VV-100",
        "entry_id": "10003",
        "ts": pd.Timestamp("2026-01-10T11:15:00", tz="UTC"),
        "author": "Priya Shah",
        "author_id": "acc-priya",
        "field": "status",
        "field_id": "status",
        "from_string": "In Progress",
        "to_string": "Review in Staging",
        "from_id": None,
        "to_id": None,
        "is_status": True,
        "is_cosmetic": False,
        "is_estimate": False,
        "is_sprint_rollover": False,
        "from_stage": 2.0,
        "to_stage": 5.0,  # STATUS_STAGES["review in staging"]
    },
    # VV-100/h4: the reopen - straight back out of the resolved status, by a
    # different author than resolved it. reresolve_events() reads this as
    # the ticket's one "reopen".
    {
        "key": "VV-100",
        "entry_id": "10004",
        "ts": pd.Timestamp("2026-01-11T08:45:00", tz="UTC"),
        "author": "Marco Diaz",
        "author_id": "acc-marco",
        "field": "status",
        "field_id": "status",
        "from_string": "Review in Staging",
        "to_string": "In Progress",
        "from_id": None,
        "to_id": None,
        "is_status": True,
        "is_cosmetic": False,
        "is_estimate": False,
        "is_sprint_rollover": False,
        "from_stage": 5.0,
        "to_stage": 2.0,
    },
    # VV-100/h5: the re-resolution - a second, distinct entry into a
    # resolved status on the same ticket. This is the row that makes
    # reresolve_events() count VV-100's "resolutions" as 2, not 1, and mark
    # it hidden_rework (resolved more than once, currently sitting
    # resolved) - exactly the case app._reopened_jql's present-tense query
    # cannot see (integrity.py's own module docstring and
    # reresolve_events()'s docstring both say why).
    {
        "key": "VV-100",
        "entry_id": "10005",
        "ts": pd.Timestamp("2026-01-12T16:00:00", tz="UTC"),
        "author": "Marco Diaz",
        "author_id": "acc-marco",
        "field": "status",
        "field_id": "status",
        "from_string": "In Progress",
        "to_string": "Review in Staging",
        "from_id": None,
        "to_id": None,
        "is_status": True,
        "is_cosmetic": False,
        "is_estimate": False,
        "is_sprint_rollover": False,
        "from_stage": 2.0,
        "to_stage": 5.0,
    },
    # VV-200/h1: a plain status transition on a second ticket, so VV-200's
    # own resolving move (next row) is not the ticket's first changelog
    # entry - the same shape a real board full of many-transition tickets
    # has, not a resolution-on-first-touch special case.
    {
        "key": "VV-200",
        "entry_id": "20001",
        "ts": pd.Timestamp("2026-02-10T09:00:00", tz="UTC"),
        "author": "Devin Bot",
        "author_id": "acc-devinbot",
        "field": "status",
        "field_id": "status",
        "from_string": "To Do",
        "to_string": "In Progress",
        "from_id": None,
        "to_id": None,
        "is_status": True,
        "is_cosmetic": False,
        "is_estimate": False,
        "is_sprint_rollover": False,
        "from_stage": 1.0,
        "to_stage": 2.0,
    },
    # VV-200/h2: a resolving transition with the "author" key removed from
    # the history entry entirely - the deprovisioned-user / some-automation
    # shape. changelog_events() falls back to the "Unknown" sentinel; author_id
    # is None because there is no author dict to read accountId from. This is
    # the row unattributed_resolutions() has to catch (asserted below) -
    # invisible to credited_resolutions() because there is nobody to credit.
    {
        "key": "VV-200",
        "entry_id": "20002",
        "ts": pd.Timestamp("2026-02-11T10:00:00", tz="UTC"),
        "author": "Unknown",
        "author_id": None,
        "field": "status",
        "field_id": "status",
        "from_string": "In Progress",
        "to_string": "Done",
        "from_id": None,
        "to_id": None,
        "is_status": True,
        "is_cosmetic": False,
        "is_estimate": False,
        "is_sprint_rollover": False,
        "from_stage": 2.0,
        "to_stage": 7.0,  # STATUS_STAGES["done"]
    },
]


def _row_dict(row: pd.Series) -> dict:
    """One parsed row as a plain dict, values normalised the way EXPECTED_ROWS writes them."""
    out = row.to_dict()
    # NaN (pandas' float null) reads as None in every EXPECTED_ROWS entry -
    # from_id/to_id/from_stage/to_stage/from_string are all "None" above,
    # never "nan", so the comparison doesn't need a NaN-aware equals.
    for field, value in list(out.items()):
        if isinstance(value, float) and pd.isna(value):
            out[field] = None
    return out


def _parse() -> pd.DataFrame:
    return integrity.changelog_events(_load_issues())


def test_seven_rows_in_key_then_timestamp_order():
    events = _parse()
    assert len(events) == len(EXPECTED_ROWS)
    assert list(events["key"]) == [row["key"] for row in EXPECTED_ROWS]
    assert list(events["entry_id"]) == [row["entry_id"] for row in EXPECTED_ROWS]


@pytest.mark.parametrize("index", range(len(EXPECTED_ROWS)))
def test_every_field_of_every_row_matches_the_pin(index):
    """One test per row: a drift shows up as "row N" plus the exact field diff."""
    events = _parse()
    actual = _row_dict(events.iloc[index])
    expected = EXPECTED_ROWS[index]
    assert actual == expected


def test_reresolve_events_reads_the_ticket_that_bounced_and_healed():
    """VV-100: resolved, reopened, resolved again - reresolve_events()'s reading of it."""
    events = _parse()
    out = integrity.reresolve_events(events, window_days=None)
    row = out[out["key"] == "VV-100"].iloc[0]
    assert row["resolutions"] == 2  # h3 and h5
    assert row["reopens"] == 1  # h4
    assert row["resolvers"] == "Priya Shah, Marco Diaz"
    assert row["reopeners"] == "Marco Diaz"
    assert row["currently_resolved"] is True or bool(row["currently_resolved"]) is True
    assert bool(row["hidden_rework"]) is True  # resolved twice, sitting resolved now


def test_reresolve_events_reads_the_bot_resolution_once():
    """VV-200: one resolving transition, one author, no bounce."""
    events = _parse()
    out = integrity.reresolve_events(events, window_days=None)
    row = out[out["key"] == "VV-200"].iloc[0]
    assert row["resolutions"] == 1
    assert row["reopens"] == 0
    assert bool(row["hidden_rework"]) is False


def test_unattributed_resolutions_catches_the_authorless_one():
    """The one row with no author is the one row this function exists to find."""
    events = _parse()
    out = integrity.unattributed_resolutions(events, window_days=None)
    assert list(out["key"]) == ["VV-200"]
    assert out.iloc[0]["from_string"] == "In Progress"
    assert out.iloc[0]["to_string"] == "Done"


def test_a_single_mutated_field_fails_exactly_one_row_comparison():
    """Flip one character of the fixture; exactly one of the per-row checks above should fail.

    Rebuilds the same comparison ``test_every_field_of_every_row_matches_the_pin``
    makes, against a payload with VV-100/h1's ``toString`` changed from
    "In Progress" to "In Review" - a realistic one-word attribution slip, not
    a structural change. If the pin were vague (comparing frame shape, or a
    hash of the whole table) this would either fail nothing or fail
    everything; comparing field by field means it fails exactly row 0's
    ``to_string`` (and, because ``to_string`` also feeds ``to_stage``,
    row 0's ``to_stage`` too - both from the one character changed) and
    nothing else.
    """
    issues = _load_issues()
    mutated = json.loads(json.dumps(issues))  # deep copy
    item = mutated[0]["changelog"]["histories"][0]["items"][0]
    assert item["toString"] == "In Progress"
    item["toString"] = "In Review"

    events = integrity.changelog_events(mutated)
    assert len(events) == len(EXPECTED_ROWS)

    mismatches: dict[int, list[str]] = {}
    for index, expected in enumerate(EXPECTED_ROWS):
        actual = _row_dict(events.iloc[index])
        diff = [field for field in expected if actual[field] != expected[field]]
        if diff:
            mismatches[index] = diff

    assert set(mismatches) == {0}, mismatches
    assert set(mismatches[0]) == {"to_string", "to_stage"}, mismatches
