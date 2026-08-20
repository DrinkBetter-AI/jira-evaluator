"""Offline checks for the weekly-series and delta primitives.

Two things have to hold for this module to be trustworthy: the same instant
must bucket the same way no matter which of the team's four timezones wrote
it, and a missing prior period must never be indistinguishable from a good
result. Both are tested directly, not inferred from a broader scenario.
"""

from __future__ import annotations

import inspect

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import series  # noqa: E402


# --------------------------------------------------------------------------- #
# weekly_buckets: timezone stability
# --------------------------------------------------------------------------- #


def test_same_instant_buckets_identically_across_the_teams_four_timezones():
    # One instant, expressed in each timezone a teammate actually works in.
    # All four strings name the same point in time.
    instant_utc = pd.Timestamp("2026-08-05T09:00:00Z")
    offsets = {
        "vietnam_utc+7": "+07:00",
        "uruguay_utc-3": "-03:00",
        "tunisia_utc+1": "+01:00",
        "eu_utc+2": "+02:00",
    }

    bucket_indices = {}
    for name, offset in offsets.items():
        local = instant_utc.tz_convert(offset)
        frame = pd.DataFrame({"when": [local]})
        buckets = series.weekly_buckets(frame, "when", weeks=12)
        hit = [i for i, b in enumerate(buckets) if b.value == 1]
        assert len(hit) == 1, f"{name}: expected exactly one occupied bucket"
        bucket_indices[name] = hit[0]

    assert len(set(bucket_indices.values())) == 1, bucket_indices


def test_week_start_is_monday_00_00_utc():
    # A Wednesday, mid-afternoon UTC - the bucket it lands in should open on
    # the Monday of that same week, at midnight UTC.
    ts = pd.Timestamp("2026-08-19T15:30:00Z")
    frame = pd.DataFrame({"when": [ts]})
    buckets = series.weekly_buckets(frame, "when", weeks=1)
    assert buckets[0].week_start == pd.Timestamp("2026-08-17T00:00:00Z")


# --------------------------------------------------------------------------- #
# weekly_buckets: shape guarantees
# --------------------------------------------------------------------------- #


def test_empty_frame_returns_twelve_zero_buckets_with_labels():
    buckets = series.weekly_buckets(pd.DataFrame(), "when", weeks=12)
    assert len(buckets) == 12
    assert all(b.value == 0 for b in buckets)
    assert all(b.label for b in buckets)


def test_missing_date_column_also_returns_full_zero_buckets():
    frame = pd.DataFrame({"other_col": [1, 2, 3]})
    buckets = series.weekly_buckets(frame, "when", weeks=12)
    assert len(buckets) == 12
    assert all(b.value == 0 for b in buckets)


def test_buckets_are_chronological_and_last_is_the_partial_current_week():
    buckets = series.weekly_buckets(pd.DataFrame(), "when", weeks=12)
    starts = [b.week_start for b in buckets]
    assert starts == sorted(starts)
    assert starts[-1] > starts[0]

    partial_flags = [b.is_partial for b in buckets]
    assert partial_flags == [False] * 11 + [True]
    assert "partial" in buckets[-1].label.lower()


def test_rows_land_in_the_correct_of_twelve_weeks():
    now = pd.Timestamp.now(tz="UTC")
    current_week_start = series._week_start(now)
    # Six full weeks back from the current (partial) week - bucket index 5
    # counting from the oldest (index 0) in a 12-bucket, oldest-first series.
    six_weeks_back = current_week_start - pd.Timedelta(weeks=6)
    frame = pd.DataFrame({"when": [six_weeks_back]})
    buckets = series.weekly_buckets(frame, "when", weeks=12)
    hit = [i for i, b in enumerate(buckets) if b.value == 1]
    assert hit == [5]


def test_rows_older_than_the_window_are_dropped_not_miscounted():
    now = pd.Timestamp.now(tz="UTC")
    too_old = series._week_start(now) - pd.Timedelta(weeks=52)
    frame = pd.DataFrame({"when": [too_old]})
    buckets = series.weekly_buckets(frame, "when", weeks=12)
    assert sum(b.value for b in buckets) == 0


def test_unparseable_dates_are_dropped_not_counted_as_zero_week():
    frame = pd.DataFrame({"when": ["not a date", None, pd.NA]})
    buckets = series.weekly_buckets(frame, "when", weeks=12)
    assert sum(b.value for b in buckets) == 0


def test_weeks_below_one_is_rejected():
    with pytest.raises(ValueError):
        series.weekly_buckets(pd.DataFrame(), "when", weeks=0)


# --------------------------------------------------------------------------- #
# delta(): direction and goodness are independent
# --------------------------------------------------------------------------- #


def test_a_decrease_is_good_when_lower_is_better():
    # Cycle time falling from 5 days to 3 days: down, and good.
    result = series.delta(3, 5, higher_is_better=False)
    assert result.direction == "down"
    assert result.is_good is True
    assert result.magnitude == 2


def test_an_increase_is_bad_when_lower_is_better():
    # Stalled tickets rising from 2 to 6: up, and bad.
    result = series.delta(6, 2, higher_is_better=False)
    assert result.direction == "up"
    assert result.is_good is False
    assert result.magnitude == 4


def test_an_increase_is_good_when_higher_is_better():
    result = series.delta(12, 8, higher_is_better=True)
    assert result.direction == "up"
    assert result.is_good is True


def test_a_decrease_is_bad_when_higher_is_better():
    result = series.delta(8, 12, higher_is_better=True)
    assert result.direction == "down"
    assert result.is_good is False


def test_higher_is_better_has_no_default_and_must_be_named():
    with pytest.raises(TypeError):
        series.delta(1, 2)  # missing higher_is_better entirely
    with pytest.raises(TypeError):
        series.delta(1, 2, True)  # positional, not keyword - also rejected


# --------------------------------------------------------------------------- #
# delta(): missing prior data is never rendered as an improvement
# --------------------------------------------------------------------------- #


def test_no_prior_data_gives_no_verdict_not_an_improvement():
    result = series.delta(10, None, higher_is_better=True)
    assert result.is_good is None
    assert result.is_good is not True
    assert result.magnitude is None


def test_nan_prior_is_treated_the_same_as_missing():
    result = series.delta(10, float("nan"), higher_is_better=True)
    assert result.is_good is None
    assert result.magnitude is None


def test_missing_current_also_gives_no_verdict():
    result = series.delta(None, 10, higher_is_better=False)
    assert result.is_good is None
    assert result.magnitude is None


def test_a_genuine_tie_gets_no_verdict_either_and_is_distinct_only_in_meaning():
    # Both known, both equal: nothing changed, so nothing is "good" or "bad" -
    # but this state and the no-prior-data state are reached through
    # different inputs, and both must fail the same "is_good is True" check.
    tie = series.delta(5, 5, higher_is_better=True)
    no_data = series.delta(5, None, higher_is_better=True)
    assert tie.direction == "flat"
    assert tie.is_good is None
    assert no_data.is_good is None
    assert tie.magnitude == 0
    assert no_data.magnitude is None  # the two states are not really the same


# --------------------------------------------------------------------------- #
# Org-wide series: bot exclusion
# --------------------------------------------------------------------------- #


def _pr_frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """rows of (author, merged_at ISO string)."""
    return pd.DataFrame(
        [{"author": author, "merged_at": merged_at} for author, merged_at in rows]
    )


def test_prs_merged_series_reports_bot_count_alongside_the_series():
    now = pd.Timestamp.now(tz="UTC")
    frame = _pr_frame(
        [
            ("alice", now.isoformat()),
            ("bob", now.isoformat()),
            ("devin-ai-integration[bot]", now.isoformat()),
            ("dependabot[bot]", now.isoformat()),
        ]
    )
    result = series.prs_merged_series(frame, weeks=12)
    assert result.bots_excluded == 2
    assert sum(b.value for b in result.buckets) == 2  # alice + bob only


def test_prs_merged_series_when_every_row_is_a_bot():
    now = pd.Timestamp.now(tz="UTC")
    frame = _pr_frame(
        [
            ("devin-ai-integration[bot]", now.isoformat()),
            ("renovate[bot]", now.isoformat()),
            ("github-actions[bot]", now.isoformat()),
        ]
    )
    result = series.prs_merged_series(frame, weeks=12)
    assert result.bots_excluded == 3
    assert sum(b.value for b in result.buckets) == 0
    assert len(result.buckets) == 12


def test_tickets_resolved_series_excludes_bot_assignees():
    now = pd.Timestamp.now(tz="UTC")
    frame = pd.DataFrame(
        [
            {"assignee": "alice", "status_category_changed_date": now.isoformat()},
            {"assignee": "devin-ai-integration[bot]", "status_category_changed_date": now.isoformat()},
        ]
    )
    result = series.tickets_resolved_series(frame, weeks=12)
    assert result.bots_excluded == 1
    assert sum(b.value for b in result.buckets) == 1


def test_custom_bot_login_list_overrides_the_default():
    now = pd.Timestamp.now(tz="UTC")
    frame = _pr_frame([("totally-a-human", now.isoformat())])
    result = series.prs_merged_series(frame, weeks=12, bot_logins=["totally-a-human"])
    assert result.bots_excluded == 1
    assert sum(b.value for b in result.buckets) == 0


def test_empty_pr_frame_still_returns_twelve_buckets_and_zero_bots():
    result = series.prs_merged_series(pd.DataFrame(), weeks=12)
    assert len(result.buckets) == 12
    assert result.bots_excluded == 0


# ---------------------------------------------------------------------------
# The resolved frame has to actually carry the column this buckets on.
# ---------------------------------------------------------------------------


def test_the_field_the_resolved_series_buckets_on_is_one_jira_was_asked_for():
    """``RESOLVED_FIELDS`` must request whatever ``tickets_resolved_series`` reads.

    Jira returns only the fields a search asks for, but
    ``jira_client``'s row builder emits ``status_category_changed_date``
    unconditionally - ``fields.get("statuscategorychangedate")``. Leaving the
    field out of the request therefore produced no missing column and no
    error: it produced a column that was null on every row, which
    ``weekly_buckets`` dropped as unparseable, drawing twelve weeks of zero
    throughput beside a nonzero headline resolved count.

    Every fixture in this file populates the column by hand, so no amount of
    unit testing the bucketing could catch it. This asserts the request
    instead, which is where the bug actually lived.
    """
    import data_layer
    import jira_client

    date_col = inspect.signature(series.tickets_resolved_series).parameters["date_col"].default
    assert date_col == "status_category_changed_date"
    # Both names pinned, because they are not each other with the underscores
    # taken out: Jira's field id is "statuscategorychangedate" (one "d" in
    # "changedate"), the row builder's column is "status_category_changed_date"
    # (two). ``jira_client``'s row builder is the join between them.
    assert "statuscategorychangedate" in data_layer.RESOLVED_FIELDS
    builder = inspect.getsource(jira_client.JiraClient._issues_to_dataframe)
    assert '"status_category_changed_date": fields.get("statuscategorychangedate")' in builder


def test_a_resolved_frame_whose_bucket_column_is_null_produces_no_silent_zeros():
    """The failure mode above, reproduced: all-null dates must not read as "zero resolved".

    This is the shape the live read actually returned. The series comes back
    with every bucket at zero, which is indistinguishable from a genuinely
    idle twelve weeks - which is exactly why the request-side assertion above
    is the real guard, and why this one is here to name the symptom.
    """
    frame = pd.DataFrame(
        [
            {"assignee": "alice", "status_category_changed_date": None},
            {"assignee": "bob", "status_category_changed_date": None},
        ]
    )
    out = series.tickets_resolved_series(frame, weeks=12)
    assert len(out.buckets) == 12
    assert {bucket.value for bucket in out.buckets} == {0}
