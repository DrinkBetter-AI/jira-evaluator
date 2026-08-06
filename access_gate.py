"""Shared-password gate for hosted deployments.

Locally, where the only visitor is the person who started the process, an unset
``DASHBOARD_PASSWORD`` skips the gate. On Cloud Run it fails closed instead: the
service is reachable from the internet and can write to Jira, so a forgotten
environment variable must stop the app rather than open it.

Once entered, the password is remembered in the browser rather than only in
``st.session_state``: session state dies with the websocket, so a refresh, a
second tab or a Cloud Run cold start used to re-prompt someone who had just
logged in. The browser instead keeps a signed, expiring cookie that carries no
secret of its own and is only accepted while it verifies against the current
``DASHBOARD_PASSWORD`` — rotating the password logs everyone out.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time

import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

PASSWORD_ENV = "DASHBOARD_PASSWORD"
_SESSION_KEY = "_access_granted"
# Set on the browser, read back through ``st.context.cookies`` on the next page
# load. Not HttpOnly, because only JavaScript can set it from inside Streamlit;
# it is a signed timestamp, so reading it reveals nothing the holder of it does
# not already have.
_COOKIE_NAME = "jira_dashboard_access"
_COOKIE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
# Writing the cookie is a rendered iframe, so it happens once per session rather
# than on every rerun; the expiry slides forward each time someone comes back.
_COOKIE_WRITTEN_KEY = "_access_cookie_written"
# Survives for the rest of the session: ``st.context.cookies`` keeps replaying
# the cookie the browser sent when the page loaded, so without this flag a
# sign-out would be undone by the very next rerun.
_SIGNED_OUT_KEY = "_access_signed_out"
# Each wrong guess costs a second more than the last, capped low: the sleep
# holds a Streamlit script thread, and the service runs on a single instance, so
# a long backoff would let a few wrong guesses stall the dashboard for everyone.
# Slowing a script down is all this can honestly claim; real lockout needs an
# identity layer in front of the app.
_MAX_BACKOFF_SECONDS = 3
# Counted per process rather than per session: a new websocket would otherwise
# reset the backoff, which is exactly what a script guessing passwords does.
_failed_attempts = 0
# Cloud Run sets this for every revision; its presence is how the process knows
# it is serving the public rather than one developer's laptop.
_HOSTED_ENV = "K_SERVICE"


def _signature(expected: str, expiry: int) -> str:
    """Sign an expiry with a key derived from the password, never the password."""
    key = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.new(key, str(expiry).encode("utf-8"), hashlib.sha256).hexdigest()


def _mint_token(expected: str) -> str:
    expiry = int(time.time()) + _COOKIE_MAX_AGE_SECONDS
    return f"{expiry}.{_signature(expected, expiry)}"


def _token_is_valid(token: str, expected: str) -> bool:
    """True only for a well-formed, unexpired token signed by this password."""
    raw_expiry, _, signature = token.partition(".")
    if not signature or not raw_expiry.isdigit():
        return False
    expiry = int(raw_expiry)
    if expiry <= time.time():
        return False
    return hmac.compare_digest(signature, _signature(expected, expiry))


def _cookie_grants_access(expected: str) -> bool:
    # ``st.context.cookies`` reflects the request that opened the websocket, so
    # it sees the cookie written during an earlier visit but not one written a
    # moment ago in this session. That is exactly the split we want: session
    # state covers now, the cookie covers next time.
    cookies = getattr(st.context, "cookies", None) or {}
    token = cookies.get(_COOKIE_NAME, "")
    if not token:
        return False
    if _token_is_valid(token, expected):
        logger.info("Access granted from a remembered browser cookie")
        return True
    logger.info("Ignoring an expired or unsigned access cookie")
    return False


def _write_cookie(value: str, max_age: int) -> None:
    """Set the cookie from inside the component iframe's parent document."""
    secure = "; Secure" if os.getenv(_HOSTED_ENV, "").strip() else ""
    payload = json.dumps(f"{_COOKIE_NAME}={value}; Path=/; Max-Age={max_age}; SameSite=Lax{secure}")
    components.html(
        f"""
        <script>
        try {{
            (window.parent || window).document.cookie = {payload};
        }} catch (err) {{
            document.cookie = {payload};
        }}
        </script>
        """,
        height=0,
    )


def _remember_browser(expected: str) -> None:
    if st.session_state.get(_COOKIE_WRITTEN_KEY):
        return
    _write_cookie(_mint_token(expected), _COOKIE_MAX_AGE_SECONDS)
    st.session_state[_COOKIE_WRITTEN_KEY] = True


def sign_out() -> None:
    """Forget this browser, so the next page load asks for the password again."""
    st.session_state.pop(_SESSION_KEY, None)
    st.session_state.pop(_COOKIE_WRITTEN_KEY, None)
    # The deletion cannot be written here: ``st.rerun`` throws the page away
    # before an iframe rendered in this run reaches the browser. The gate writes
    # it on the next run instead, where the password prompt keeps the page alive
    # long enough for the script to execute.
    st.session_state[_SIGNED_OUT_KEY] = True
    logger.info("Sign-out requested; the access cookie will be cleared on rerun")


def render_sign_out() -> None:
    """Offer a way out of a month-long cookie, but only where a gate exists."""
    if not os.getenv(PASSWORD_ENV, "").strip():
        return
    with st.sidebar:
        if st.button("Sign out", icon=":material/logout:", use_container_width=True):
            sign_out()
            st.rerun()


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
        _remember_browser(expected)
        return
    # Checked before the cookie so that signing out wins over the cookie the
    # browser is still sending for this page load.
    signed_out = st.session_state.get(_SIGNED_OUT_KEY, False)
    if not signed_out and _cookie_grants_access(expected):
        st.session_state[_SESSION_KEY] = True
        _remember_browser(expected)
        return

    st.title("Jira Ticket Health Dashboard")
    if signed_out:
        # Rendered here rather than in ``sign_out`` because this run ends at a
        # prompt instead of a rerun, so the browser actually executes it.
        _write_cookie("", 0)
    entered = st.text_input("Password", type="password")
    if not entered:
        st.stop()
    global _failed_attempts
    if hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8")):
        st.session_state[_SESSION_KEY] = True
        st.session_state.pop(_SIGNED_OUT_KEY, None)
        _failed_attempts = 0
        # The cookie is written on the next run: ``st.rerun`` discards anything
        # rendered from here on, iframes included.
        st.rerun()

    _failed_attempts += 1
    time.sleep(min(_failed_attempts, _MAX_BACKOFF_SECONDS))
    st.error("Incorrect password.")
    st.stop()
