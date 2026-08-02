"""Shared-password gate for hosted deployments.

When ``DASHBOARD_PASSWORD`` is unset the gate is a no-op, so local runs are
unaffected; hosted instances set it and every visitor types it once per session.
"""

from __future__ import annotations

import hmac
import os
import time

import streamlit as st

PASSWORD_ENV = "DASHBOARD_PASSWORD"
_SESSION_KEY = "_access_granted"
# Each wrong guess costs a second more than the last, capped so a locked-out
# visitor is not stuck forever; enough to make guessing pointless.
_MAX_BACKOFF_SECONDS = 30
# Counted per process rather than per session: a new websocket would otherwise
# reset the backoff, which is exactly what a script guessing passwords does.
_failed_attempts = 0


def require_password() -> None:
    """Stop the script until the visitor supplies the shared password."""
    # Stripped, so a whitespace-only value is a missing password rather than a
    # gate that anyone can walk through by typing a space.
    expected = os.getenv(PASSWORD_ENV, "").strip()
    if not expected:
        return
    if st.session_state.get(_SESSION_KEY):
        return

    st.title("Jira Ticket Health Dashboard")
    entered = st.text_input("Password", type="password")
    if not entered:
        st.stop()
    global _failed_attempts
    if hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8")):
        st.session_state[_SESSION_KEY] = True
        _failed_attempts = 0
        st.rerun()

    _failed_attempts += 1
    time.sleep(min(_failed_attempts, _MAX_BACKOFF_SECONDS))
    st.error("Incorrect password.")
    st.stop()
