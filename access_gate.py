"""Shared-password gate for hosted deployments.

Locally, where the only visitor is the person who started the process, an unset
``DASHBOARD_PASSWORD`` skips the gate. On Cloud Run it fails closed instead: the
service is reachable from the internet and can write to Jira, so a forgotten
environment variable must stop the app rather than open it.
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
# Cloud Run sets this for every revision; its presence is how the process knows
# it is serving the public rather than one developer's laptop.
_HOSTED_ENV = "K_SERVICE"


def require_password() -> None:
    """Stop the script until the visitor supplies the shared password."""
    # Stripped, so a whitespace-only value is a missing password rather than a
    # gate that anyone can walk through by typing a space.
    expected = os.getenv(PASSWORD_ENV, "").strip()
    if not expected:
        if os.getenv(_HOSTED_ENV, "").strip():
            st.error(
                f"{PASSWORD_ENV} is not set on this deployment. Refusing to serve "
                "a dashboard that can write to Jira without a gate in front of it."
            )
            st.stop()
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
