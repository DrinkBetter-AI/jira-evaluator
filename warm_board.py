"""Refresh the shared board snapshot on a schedule, off the request path.

The engineering board opens with fourteen reads. Thirteen are counts and small
lists that answer in about a second; the fourteenth is the whole open board -
every description, every changelog, paginated - and it is the read a cold page
load waits on. ``data_layer._engineering_data`` already serves a warm snapshot
when one exists and only gathers live when it does not. This script is what
keeps one existing.

Run it on a schedule - Cloud Scheduler triggering a Cloud Run Job, a cron, a
GitHub Actions workflow - every few minutes, with the same environment the app
has (Jira creds, GitHub token, and ``JIRA_DASHBOARD_SNAPSHOT_BUCKET`` so the
snapshot lands where every app instance reads it). With the warmer running, no
reader ever triggers the live gather: they read the object this wrote.

Exit code is 0 only when a snapshot was written and read back, 1 otherwise, so
a scheduler can alert on a warmer that has stopped keeping up.

    python warm_board.py
"""

from __future__ import annotations

import logging
import sys
import time

from dotenv import load_dotenv

# Same as app.py: existing environment wins, a stray .env never overrides a
# real deployment's injected config. Must run before importing data_layer,
# which reads the environment at import time.
load_dotenv(override=False)

import data_layer

logger = logging.getLogger("warm_board")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    max_results = data_layer.MAX_RESULTS
    page_size = data_layer.JIRA_PAGE_SIZE

    started = time.perf_counter()
    outcome = data_layer._engineering_gather_and_shape(max_results, page_size)

    if outcome.bundle is None:
        logger.error(
            "Warmer gathered no usable board: %s",
            outcome.fatal_error or "empty board",
        )
        return 1

    data_layer._write_board_snapshot(outcome.bundle)

    # A warmer that logs success while having written nothing is worse than one
    # that fails loudly: confirm the snapshot is there and readable before
    # claiming the cycle worked.
    if data_layer._read_board_snapshot() is None:
        logger.error("Warmer wrote a snapshot that did not read back")
        return 1

    logger.info(
        "Board snapshot refreshed in %.1fs (%d tickets)",
        time.perf_counter() - started,
        len(outcome.bundle.raw_df),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
