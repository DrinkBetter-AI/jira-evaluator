"""Phase 7 guardrail: one INFO line per outbound read.

Every metered read (Jira, GitHub, Amplitude, BigQuery, Stripe, OpenAI) in
this app flows through a small number of ``st.cache_data``-wrapped entry
points. ``logged_read`` wraps those entry points *outside* ``st.cache_data``
so the wrapper still runs on a cache hit, but the wrapped body only runs on
a miss - which is what tells the two apart without touching Streamlit's
cache internals.

``mark_executed()`` must be the first thing the wrapped function's body
does: since ``st.cache_data`` skips the body entirely on a hit, the marker
only ever fires on a miss.

``bill_bytes()`` lets a BigQuery call site (``cost_client.py``,
``ads_client.py``) attach ``job.total_bytes_billed`` to whichever read is
currently in flight, so the same log line and the "reads this page made"
expander both see it.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger("reads")

_executed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_read_executed", default=False
)
_bytes_billed: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "_read_bytes_billed", default=None
)
_sink: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "_read_sink", default=None
)


def mark_executed() -> None:
    """Call as the first statement of a cache_data-wrapped read's body."""
    _executed.set(True)


def bill_bytes(total_bytes_billed: int | None) -> None:
    """Attach a BigQuery job's billed bytes to the read in flight, if any."""
    if total_bytes_billed is not None:
        _bytes_billed.set(total_bytes_billed)


class track_page_reads:
    """Context manager: collect the reads made while it is open.

    Nests safely - a fresh list is bound for the duration of the ``with``
    block and the previous one (if any) is restored on exit, so calling this
    once per page render never leaks entries across pages or sessions.
    """

    def __enter__(self) -> list[dict[str, Any]]:
        self._reads: list[dict[str, Any]] = []
        self._token = _sink.set(self._reads)
        return self._reads

    def __exit__(self, *exc_info: object) -> None:
        _sink.reset(self._token)


def _describe(args: tuple, kwargs: dict) -> str:
    parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
    text = ", ".join(parts)
    return text if len(text) <= 120 else text[:117] + "..."


def logged_read(source: str) -> Callable:
    """Decorator: log one INFO line per call, with cache hit/miss inferred
    from whether the wrapped body actually ran. Wrap OUTSIDE st.cache_data."""

    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            exec_token = _executed.set(False)
            bytes_token = _bytes_billed.set(None)
            start = time.monotonic()
            try:
                return fn(*args, **kwargs)
            finally:
                executed = _executed.get()
                total_bytes_billed = _bytes_billed.get()
                _executed.reset(exec_token)
                _bytes_billed.reset(bytes_token)
                duration_ms = (time.monotonic() - start) * 1000
                cache = "miss" if executed else "hit"
                window = _describe(args, kwargs)
                logger.info(
                    "read source=%s window=%s duration_ms=%.1f cache=%s bytes_billed=%s",
                    source,
                    window,
                    duration_ms,
                    cache,
                    total_bytes_billed,
                )
                sink = _sink.get()
                if sink is not None:
                    sink.append(
                        {
                            "source": source,
                            "window": window,
                            "duration_ms": round(duration_ms, 1),
                            "cache": cache,
                            "bytes_billed": total_bytes_billed,
                        }
                    )

        # st.cache_data hands back a callable with its own ``.clear()`` (used
        # throughout this app to invalidate one read on Refresh) - proxy it
        # through so wrapping here does not hide it from callers.
        if hasattr(fn, "clear"):
            wrapper.clear = fn.clear

        return wrapper

    return decorate
