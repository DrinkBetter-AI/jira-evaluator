"""Whether the dashboard may change Jira during this page run.

The dashboard is for looking at the backlog; changing it is a separate intent,
and someone reading a report should not be one misclick from closing a ticket.
So editing is off when the page loads and the reviewer turns it on deliberately.

The override is thread-local because Streamlit runs each browser session in its
own script thread: one viewer enabling edits must not enable them for another.
"""

from __future__ import annotations

import os
import threading

WRITES_ENV = "JIRA_ALLOW_WRITES"
_ENABLED_VALUES = {"1", "true", "yes", "on"}

_state = threading.local()

READ_ONLY_MESSAGE = (
    "Jira editing is off. Turn on 'Allow Jira edits' in the sidebar to change "
    "anything; until then the dashboard only reads."
)


def env_default() -> bool:
    """Deployment-level default for the sidebar switch."""
    return os.getenv(WRITES_ENV, "").strip().lower() in _ENABLED_VALUES


def set_writes_enabled(enabled: bool) -> None:
    """Record the reviewer's choice for the rest of this script run."""
    _state.enabled = bool(enabled)


def writes_enabled() -> bool:
    return bool(getattr(_state, "enabled", env_default()))


def require_writes_enabled() -> None:
    """Raise rather than reach Jira, so a missed UI guard still cannot write."""
    if not writes_enabled():
        raise RuntimeError(READ_ONLY_MESSAGE)
