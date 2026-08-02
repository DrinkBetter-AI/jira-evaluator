"""Shared-password gate for hosted deployments.

When ``DASHBOARD_PASSWORD`` is unset the gate is a no-op, so local runs are
unaffected; hosted instances set it and every visitor types it once per session.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

PASSWORD_ENV = "DASHBOARD_PASSWORD"
_SESSION_KEY = "_access_granted"


def require_password() -> None:
    """Stop the script until the visitor supplies the shared password."""
    expected = os.getenv(PASSWORD_ENV, "")
    if not expected:
        return
    if st.session_state.get(_SESSION_KEY):
        return

    st.title("Jira Ticket Health Dashboard")
    entered = st.text_input("Password", type="password")
    if not entered:
        st.stop()
    if hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8")):
        st.session_state[_SESSION_KEY] = True
        st.rerun()
    st.error("Incorrect password.")
    st.stop()
