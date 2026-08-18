"""Phase 7: job.total_bytes_billed reaches the read log for every BigQuery
call site in cost_client.py and ads_client.py, exactly as app.py's cached
entry points (wrapped with @read_log.logged_read) would see it in production.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ads_client as ac  # noqa: E402
import cost_client as cc  # noqa: E402
import read_log  # noqa: E402

TODAY = dt.date(2026, 8, 6)


class _FakeJob:
    def __init__(self, total_bytes_billed, rows=None, frame=None):
        self.total_bytes_billed = total_bytes_billed
        self._rows = rows or []
        self._frame = frame if frame is not None else pd.DataFrame()

    def result(self):
        return self

    def __iter__(self):
        return iter(self._rows)

    def to_dataframe(self):
        return self._frame


class _FakeClient:
    """Reports a fixed total_bytes_billed on every job, whatever the query."""

    def __init__(self, total_bytes_billed: int):
        self.total_bytes_billed = total_bytes_billed

    def query(self, sql, job_config=None):
        if "first_day" in sql:
            return _FakeJob(self.total_bytes_billed, rows=[{"first_day": TODAY}])
        if "MIN(day)" in sql:
            return _FakeJob(
                self.total_bytes_billed, rows=[{"first": TODAY, "last": TODAY}]
            )
        return _FakeJob(self.total_bytes_billed, frame=pd.DataFrame())


def _read_bytes_billed(source: str, call):
    @read_log.logged_read(source)
    def _wrapped():
        return call()

    with read_log.track_page_reads() as reads:
        _wrapped()
    assert len(reads) == 1
    return reads[0]["bytes_billed"]


def test_ads_client_run_reports_bytes_billed():
    client = _FakeClient(total_bytes_billed=987_654)
    billed = _read_bytes_billed(
        "test.ads_run", lambda: ac._run(client, "SELECT 1")
    )
    assert billed == 987_654


def test_ads_client_loaded_from_reports_bytes_billed():
    client = _FakeClient(total_bytes_billed=222_000)
    config = ac.AdsConfig("proj", "dataset", None, None)
    billed = _read_bytes_billed(
        "test.loaded_from",
        lambda: ac.loaded_from(client, config, "8876864797", now=TODAY),
    )
    assert billed == 222_000


def test_cost_client_cloud_costs_reports_bytes_billed():
    client = _FakeClient(total_bytes_billed=555_111)
    billed = _read_bytes_billed(
        "test.cloud_costs", lambda: cc.cloud_costs(client, "p.d.t", 30, now=TODAY)
    )
    assert billed == 555_111


def test_cost_client_billing_coverage_reports_bytes_billed(tmp_path):
    client = _FakeClient(total_bytes_billed=333_222)
    billed = _read_bytes_billed(
        "test.billing_coverage",
        lambda: cc.billing_coverage(client, "p.d.t", now=TODAY, cache_dir=tmp_path),
    )
    assert billed == 333_222

    # The disk cache Phase 6 added means a second call the same day never
    # touches BigQuery at all, so there is no job and nothing to bill.
    billed_again = _read_bytes_billed(
        "test.billing_coverage",
        lambda: cc.billing_coverage(client, "p.d.t", now=TODAY, cache_dir=tmp_path),
    )
    assert billed_again is None
