"""The opening gather's concurrency ceiling.

The gather pool is deliberately wider than Jira's own limit, because the reads
it runs are split across two unrelated hosts and the GitHub half has no reason
to queue behind the Jira half. That width is only safe while something else
holds the Jira reads to their ceiling, which is what these tests pin: lose the
gate and the pool quietly starts nine concurrent Jira calls against a client
that promises eight.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import data_layer
from jira_client import MAX_PARALLEL_REQUESTS


class _PeakCounter:
    """Records the most tasks that were ever inside the gate at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self._live += 1
            self.peak = max(self.peak, self._live)

    def leave(self) -> None:
        with self._lock:
            self._live -= 1


def test_no_more_than_the_ceiling_of_jira_reads_run_at_once() -> None:
    counter = _PeakCounter()
    started = threading.Barrier(MAX_PARALLEL_REQUESTS, timeout=5.0)

    def _task() -> None:
        counter.enter()
        try:
            # Hold until the gate is provably full, so a missing gate shows up
            # as a peak above the ceiling rather than as tasks that politely
            # took turns because each finished before the next began.
            try:
                started.wait()
            except threading.BrokenBarrierError:
                pass
        finally:
            counter.leave()

    tasks = [data_layer._jira_read(_task) for _ in range(MAX_PARALLEL_REQUESTS + 4)]
    with ThreadPoolExecutor(max_workers=data_layer._GATHER_WORKERS) as pool:
        for future in [pool.submit(task) for task in tasks]:
            future.result()

    assert counter.peak <= MAX_PARALLEL_REQUESTS


def test_the_engineering_reads_are_all_gated() -> None:
    """The ceiling is worth nothing if a new read is added outside it."""
    counter = _PeakCounter()

    def _task() -> None:
        counter.enter()
        counter.leave()

    reads = data_layer._engineering_reads(max_results=10, page_size=10)
    # Every read the page opens with is Jira's, so every one of them must come
    # back wrapped. Comparing against a freshly wrapped callable is the only
    # thing that distinguishes a gated task from a bare lambda.
    assert reads
    assert all(
        task.__qualname__ == data_layer._jira_read(_task).__qualname__
        for task in reads.values()
    )


def test_a_saturated_jira_gate_does_not_hold_up_a_github_read() -> None:
    """The whole point of the wide pool: one wave across two providers.

    If the ceiling were enforced by narrowing the pool instead of by gating the
    Jira tasks, this is the property that would be lost - the GitHub reads would
    sit in the queue behind Jira's slowest page fetch for no reason at all.
    """
    release = threading.Event()
    github_ran = threading.Event()

    def _blocked_jira() -> None:
        release.wait(timeout=5.0)

    def _github() -> None:
        github_ran.set()

    tasks = [data_layer._jira_read(_blocked_jira) for _ in range(MAX_PARALLEL_REQUESTS)]
    tasks.append(_github)

    with ThreadPoolExecutor(max_workers=data_layer._GATHER_WORKERS) as pool:
        futures = [pool.submit(task) for task in tasks]
        try:
            # The gate is saturated by the Jira tasks and stays that way until
            # released; the GitHub read must get through regardless.
            assert github_ran.wait(timeout=5.0)
        finally:
            release.set()
        for future in futures:
            future.result()
