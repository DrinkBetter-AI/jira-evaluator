"""Dry-run bytes-scanned checks for the BigQuery reads Phase 6 bounded.

No live BigQuery credential is available in this environment - there is no
project or dataset to point a real dry run at - so these do not run one.
Instead a fake client stands in for a partitioned table and reports
``total_bytes_processed`` the way a real dry run would: for whichever
partitions the query's own predicate would keep, with the rest untouched. Same
contract as a real dry run - report bytes, touch no data - so what is actually
being checked is the thing that survives without live BigQuery: does each
read's own query construction carry a predicate that prunes this table, and
does honoring it come in well under what the same table would cost unfiltered.

``billing_coverage`` carries no such predicate - finding its own range is the
query's job, so nothing can be handed to it in advance - and is checked for the
bound the plan asks of it instead: the disk cache, which keeps a second call
the same day from paying for the scan again rather than shrinking any one
scan.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ads_client as ac  # noqa: E402
import cost_client as cc  # noqa: E402

TODAY = dt.date(2026, 8, 6)

# The number itself does not matter - only that it is the same for the bounded
# and unbounded halves of each comparison.
_BYTES_PER_DAY = 25 * 1024 * 1024  # 25 MiB


class _FakeJob:
    def __init__(self, total_bytes_processed: int, rows=None, frame=None):
        self.total_bytes_processed = total_bytes_processed
        self._rows = rows or []
        self._frame = frame

    def result(self):
        return self

    def __iter__(self):
        return iter(self._rows)

    def to_dataframe(self):
        return self._frame if self._frame is not None else pd.DataFrame()


class _FakePartitionedTable:
    """A BigQuery client standing in for one ingestion-time partitioned table.

    ``span`` is every day the table has ever written a partition for - reading
    it with no bound at all, as every one of these three queries did before
    Phase 6, scans the whole thing. ``query`` reports bytes only for the
    partitions a bound this module recognises would keep, so a query that adds
    no bound reports the same bytes as the pre-Phase-6 query did.
    """

    def __init__(self, span: tuple[dt.date, dt.date]):
        self.span = span
        self.calls: list[dict] = []

    def _clip(self, lower: dt.date, upper: dt.date) -> int:
        start, end = self.span
        lower, upper = max(lower, start), min(upper, end)
        if lower > upper:
            return 0
        return ((upper - lower).days + 1) * _BYTES_PER_DAY

    @property
    def full_scan_bytes(self) -> int:
        start, end = self.span
        return ((end - start).days + 1) * _BYTES_PER_DAY

    def query(self, sql, job_config=None):
        params = {
            p.name: p.value
            for p in (getattr(job_config, "query_parameters", None) or [])
        }
        start, end = self.span
        if "partition_last" in params and "first" in params:
            # cloud_costs: _PARTITIONTIME BETWEEN TIMESTAMP(@first) AND @partition_last
            upper = params["partition_last"]
            upper = upper.date() if hasattr(upper, "date") else upper
            total = self._clip(params["first"], upper)
        elif "floor" in params:
            # loaded_from: segments_date >= @floor
            total = self._clip(params["floor"], end)
        else:
            # No recognised bound - the pre-Phase-6 shape of all three queries.
            total = self._clip(start, end)
        self.calls.append({"sql": sql, "params": params, "bytes": total})
        if "first_day" in sql:
            return _FakeJob(total, rows=[{"first_day": start}])
        if "HAVING" in sql:  # the billing_coverage probe
            return _FakeJob(total, rows=[{"first": start, "last": end}])
        return _FakeJob(total, frame=None)


# Almost four years of partitions: enough that an unbounded scan is not
# accidentally close to a bounded one, however the windows below are sized.
_SPAN = (dt.date(2023, 1, 1), TODAY)


def test_cloud_costs_dry_run_prunes_partitions_an_unbounded_read_would_scan():
    """The plan: bound cloud_costs on the ingestion-time partition, not just
    ``DATE(usage_start_time)`` which prunes nothing on this table."""
    table = _FakePartitionedTable(_SPAN)
    cc.cloud_costs(table, "p.d.t", 30, now=TODAY)
    call = table.calls[-1]
    assert "_PARTITIONTIME" in call["sql"]
    assert {"first", "last", "partition_last"} <= call["params"].keys()
    assert 0 < call["bytes"] < table.full_scan_bytes * 0.2, (
        call["bytes"],
        table.full_scan_bytes,
    )


def test_loaded_from_dry_run_prunes_partitions_an_unbounded_read_would_scan():
    """The plan: floor loaded_from to as far back as any window can ask -
    which is merchant_client.SALES_DAYS (90), the longest of any caller."""
    table = _FakePartitionedTable(_SPAN)
    config = ac.AdsConfig("w266-project-329918", "google_ads", None, None)
    ac.loaded_from(table, config, "8876864797", now=TODAY)
    call = table.calls[-1]
    assert "segments_date >= @floor" in call["sql"]
    assert "floor" in call["params"]
    assert 0 < call["bytes"] < table.full_scan_bytes * 0.2, (
        call["bytes"],
        table.full_scan_bytes,
    )
    # The floor is a lower bound on correctness, not a guess: it is exactly the
    # longest window any caller (merchant_client.SALES_DAYS) ever asks of it.
    assert (TODAY - call["params"]["floor"]).days == ac._LOADED_FROM_FLOOR_DAYS
    assert ac._LOADED_FROM_FLOOR_DAYS >= 90


def test_billing_coverage_dry_run_is_unbounded_but_asked_once_a_day(tmp_path):
    """billing_coverage has no column to predicate on - it is finding the
    range no caller can hand it in advance - so its own scan is not pruned.
    What Phase 6 bounds instead is how often that scan happens: the plan's
    disk cache, keyed by day, so a second call the same day costs nothing."""
    table = _FakePartitionedTable(_SPAN)
    cc.billing_coverage(table, "p.d.t", now=TODAY, cache_dir=tmp_path)
    assert len(table.calls) == 1
    # No bound recognised: this one call is exactly what a full scan reports.
    assert table.calls[-1]["bytes"] == table.full_scan_bytes

    cc.billing_coverage(table, "p.d.t", now=TODAY, cache_dir=tmp_path)
    # Served from the disk cache: no second BigQuery call, so no second scan.
    assert len(table.calls) == 1

    # A new day rolls the cache key over, as the plan asks: "once a day".
    cc.billing_coverage(
        table, "p.d.t", now=TODAY + dt.timedelta(days=1), cache_dir=tmp_path
    )
    assert len(table.calls) == 2
