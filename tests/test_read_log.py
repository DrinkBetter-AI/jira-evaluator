"""Phase 7: one INFO line per outbound read, cache hit/miss inferred from
whether the wrapped body actually ran, and the per-page read collector."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import read_log  # noqa: E402


def _fake_cached(calls):
    """Stands in for st.cache_data: only calls fn on the first invocation,
    with the given key - exactly what wraps every real read this module logs."""
    cache = {}

    def decorate(fn):
        def wrapper(key):
            if key in cache:
                return cache[key]
            result = fn(key)
            cache[key] = result
            return result

        return wrapper

    return decorate


def test_first_call_logs_as_a_miss_and_second_as_a_hit(caplog):
    cache = {}

    @read_log.logged_read("test.source")
    def cached_read(key):
        if key in cache:
            return cache[key]
        read_log.mark_executed()
        cache[key] = key * 2
        return cache[key]

    with caplog.at_level("INFO", logger="reads"):
        first = cached_read("a")
        second = cached_read("a")

    assert first == second == "aa"
    records = [r for r in caplog.records if r.name == "reads"]
    assert len(records) == 2
    assert "cache=miss" in records[0].message
    assert "cache=hit" in records[1].message
    assert "source=test.source" in records[0].message


def test_bill_bytes_only_reaches_the_read_currently_in_flight():
    cache = {}

    @read_log.logged_read("test.bq")
    def cached_bq_read(key):
        if key in cache:
            return cache[key]
        read_log.mark_executed()
        read_log.bill_bytes(12345)
        cache[key] = key
        return cache[key]

    with read_log.track_page_reads() as reads:
        cached_bq_read("x")
        cached_bq_read("x")  # cache hit: no job, so no bytes billed

    assert reads[0]["cache"] == "miss"
    assert reads[0]["bytes_billed"] == 12345
    assert reads[1]["cache"] == "hit"
    assert reads[1]["bytes_billed"] is None


def test_track_page_reads_collects_only_reads_made_inside_the_block():
    @read_log.logged_read("test.outside")
    def outside_read():
        read_log.mark_executed()
        return 1

    outside_read()  # before any tracker is open: logged, not collected

    with read_log.track_page_reads() as reads:
        outside_read()
        assert len(reads) == 1

    outside_read()  # after the tracker closed: not appended to the old list
    assert len(reads) == 1


def test_nested_page_reads_restore_the_outer_sink():
    @read_log.logged_read("test.nested")
    def a_read():
        read_log.mark_executed()
        return 1

    with read_log.track_page_reads() as outer:
        a_read()
        with read_log.track_page_reads() as inner:
            a_read()
        a_read()

    assert len(inner) == 1
    assert len(outer) == 2
