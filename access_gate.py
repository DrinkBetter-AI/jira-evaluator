"""Shared-password gate for hosted deployments.

Locally, where the only visitor is the person who started the process, an unset
``DASHBOARD_PASSWORD`` skips the gate. On Cloud Run it fails closed instead: the
service is reachable from the internet and can write to Jira, so a forgotten
environment variable must stop the app rather than open it.

Once entered, the password is remembered in the browser rather than only in
``st.session_state``: session state dies with the websocket, so a refresh, a
second tab or a Cloud Run cold start used to re-prompt someone who had just
logged in. The browser instead keeps a signed, expiring cookie that carries no
secret of its own. When ``DASHBOARD_COOKIE_KEY`` is set, the cookie is signed
with that independent key and rotating the password does NOT invalidate existing
sessions; only rotating the cookie key does. When ``DASHBOARD_COOKIE_KEY`` is
unset, the cookie is derived from the password and rotating the password logs
everyone out.

What this is and is not. The cookie is a bearer token written from JavaScript,
so it cannot be ``HttpOnly`` and any script running in the page could read it;
it is signed over the browser it was issued to and sent ``Secure`` on every
https origin, which limits replay but does not prevent it, and the signing key
is stretched with scrypt so that holding a cookie is not a cheap oracle for
guessing the password. Setting ``DASHBOARD_COOKIE_KEY`` removes the password
from that derivation altogether and gives the one thing a shared password
cannot otherwise offer — revocation for everybody without changing what anybody
types. A shared password in front of an app is a doormat, not a lock: anything
needing real sessions, real revocation or per-person audit wants an identity
proxy (IAP) in front of the service instead.
"""

from __future__ import annotations

import functools
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
# An independent signing secret, when a deployment sets one. Worth setting: it
# takes the password out of the cookie's derivation entirely, and rotating it
# signs every browser out without making anyone learn a new password.
_COOKIE_KEY_ENV = "DASHBOARD_COOKIE_KEY"
# Prefixed to the cookie so a change of scheme invalidates rather than
# misreads. `1` was `<expiry>.<sha256-keyed signature>`.
_TOKEN_VERSION = "v2"
# Fixed rather than random: a salt kept in one process would not verify a cookie
# on the next instance or after a restart. It is doing the smaller of a salt's
# two jobs - no shared rainbow table across deployments - while scrypt's cost
# does the work that matters.
_KDF_SALT = b"jira-dashboard-access-cookie"
# ~100ms and 16MB per derivation here, once per process; per guess for anybody
# working backwards from a cookie.
_SCRYPT_COST = 2**14
_SCRYPT_BLOCK_SIZE = 8
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


@functools.lru_cache(maxsize=4)
def _signing_key(expected: str) -> bytes:
    """The HMAC key behind the cookie, which is not one hash of the password.

    The cookie is ``<expiry>.<signature>`` and the expiry is plaintext, so anyone
    holding a cookie - off a shared laptop, a proxy log, a screenshot of dev
    tools - has a verifier for guesses at the password. A single SHA-256 makes
    checking a guess as cheap as one hash, which against a short human-chosen
    shared secret is no obstacle at all, and the prize is Jira write access
    rather than one session.

    Two answers, in order of preference. ``DASHBOARD_COOKIE_KEY``, when set, is
    an independent secret: the cookie is then signed by something the password
    cannot be guessed from at any price, and rotating it signs everyone out
    *without* changing the password people type - the central revocation the gate
    otherwise lacks. Unset, the key is stretched out of the password with scrypt
    instead, which costs this process about a tenth of a second once and costs an
    attacker that much per guess.

    Cached because scrypt is deliberately slow and the answer only changes when
    the password does, which is a restart.
    """
    override = os.getenv(_COOKIE_KEY_ENV, "").strip()
    if override:
        return hashlib.sha256(override.encode("utf-8")).digest()
    return hashlib.scrypt(
        expected.encode("utf-8"),
        salt=_KDF_SALT,
        n=_SCRYPT_COST,
        r=_SCRYPT_BLOCK_SIZE,
        p=1,
        dklen=32,
    )


def _signature(expected: str, expiry: int) -> str:
    """Sign an expiry, and the browser it was issued to, never the password."""
    message = f"{expiry}|{_browser_fingerprint()}".encode("utf-8")
    return hmac.new(_signing_key(expected), message, hashlib.sha256).hexdigest()


def _browser_fingerprint() -> str:
    """A stable-enough mark of the browser the cookie was minted for.

    Not identity and not a defence against anyone who can also copy headers - it
    is one cheap step up from a bearer token that works from anywhere, so a
    cookie lifted out of one browser does not replay from another. The cost is
    that a browser which changes its user agent, as an update can, asks for the
    password once more; that is the right way round for a month-long token.
    """
    headers = getattr(st.context, "headers", None) or {}
    agent = headers.get("User-Agent") or headers.get("user-agent") or ""
    return hashlib.sha256(agent.encode("utf-8")).hexdigest()[:16]


def _mint_token(expected: str) -> str:
    expiry = int(time.time()) + _COOKIE_MAX_AGE_SECONDS
    return f"{_TOKEN_VERSION}.{expiry}.{_signature(expected, expiry)}"


def _token_is_valid(token: str, expected: str) -> bool:
    """True only for a well-formed, unexpired token signed by this password."""
    version, _, rest = token.partition(".")
    # Anything minted before the signature covered a browser, or by an older
    # derivation, is not upgraded in place - it is simply not accepted, and the
    # holder types the password once. A gate that honoured its own weaker tokens
    # would have gained nothing by strengthening them.
    if version != _TOKEN_VERSION:
        return False
    raw_expiry, _, signature = rest.partition(".")
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
    """Set the cookie from inside the component iframe's parent document.

    ``Secure`` is decided in the browser by the scheme actually in use, not by
    ``K_SERVICE``: keying it on Cloud Run meant any other hosted deployment -
    behind a company proxy, on a VM, anywhere with a certificate but no Google
    environment variable - sent a month-long token over plain HTTP. The one
    place this must not fire is ``http://localhost``, and that is precisely what
    the scheme says.
    """
    payload = json.dumps(f"{_COOKIE_NAME}={value}; Path=/; Max-Age={max_age}; SameSite=Lax")
    components.html(
        f"""
        <script>
        try {{
            const target = (window.parent || window);
            const secure = target.location.protocol === "https:" ? "; Secure" : "";
            target.document.cookie = {payload} + secure;
        }} catch (err) {{
            document.cookie = {payload};
        }}
        </script>
        """,
        height=0,
    )


def _remember_browser(expected: str) -> None:
    # Only write the cookie once per session. st.context.cookies reflects the
    # initial WebSocket request, so it won't show a cookie we just wrote until
    # the next full page load. We track the write attempt in session state to
    # avoid writing it repeatedly on every rerun.
    if st.session_state.get(_COOKIE_WRITTEN_KEY):
        return
    # If a valid cookie is already present from a previous session, mark it as written
    # so we don't try to overwrite it. We must check validity, not just presence,
    # otherwise an expired cookie would prevent writing a fresh one.
    existing_token = st.context.cookies.get(_COOKIE_NAME, "")
    if existing_token and _token_is_valid(existing_token, expected):
        logger.info("Valid access cookie already present in browser, skipping write")
        st.session_state[_COOKIE_WRITTEN_KEY] = True
        return
    # Write the cookie and immediately mark it as written to prevent duplicate writes
    if existing_token:
        logger.info("Replacing expired or invalid access cookie with fresh one")
    else:
        logger.info("Writing new access cookie to browser")
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


# ---------------------------------------------------------------------------
# Admin gate: a second, independent credential in front of the Integrity page.
#
# The shared DASHBOARD_PASSWORD above gets anyone in the door - every hourly
# contractor knows it, because it is the only thing standing between them and
# the dashboard they are measured on. The Integrity page names people and
# shows padding evidence; it needs a door the contractors do not have a key
# to. DASHBOARD_ADMIN_PASSWORD is that door: a second secret, checked nowhere
# near require_password() above, so knowing the first teaches nothing about
# the second.
#
# Unlike require_password(), there is no local-dev bypass for an unset
# credential. The main gate treats an unset DASHBOARD_PASSWORD as "nobody but
# the developer running this on their own laptop can reach it" and opens up;
# that reasoning does not apply here; a page that shows named padding
# evidence must fail closed on a missing credential in every environment, not
# just the hosted one - ROSTER.md's late confirmation ("Integrity-page
# access: decided - Angel only") does not carry an exception for localhost.
# ---------------------------------------------------------------------------

ADMIN_PASSWORD_ENV = "DASHBOARD_ADMIN_PASSWORD"
_ADMIN_SESSION_KEY = "_admin_access_granted"
# Kept apart from the main gate's _failed_attempts - a script hammering the
# admin password should not also, incidentally, be slowed down (or sped up)
# by whatever the shared-password gate's counter happens to be doing.
_admin_failed_attempts = 0


def admin_password_configured() -> bool:
    """Whether this deployment has ``DASHBOARD_ADMIN_PASSWORD`` set at all.

    Configuration, not identity - the same distinction ``_business_readable``
    draws for the Business page's own credentials. ``app.py``'s page registry
    calls this (via ``_integrity_readable``) to decide whether to advertise
    the Integrity page's ``_PageSpec`` at all: a deployment that never set
    this credential keeps the page out of the registry for literally
    everyone, the same as an unconfigured Business page. This is
    deliberately *not* a per-session identity check - Streamlit only routes
    a URL to a page that was already part of that session's very first
    ``st.navigation()`` call, so gating registration on "has this session
    already authenticated" would mean nobody could ever reach the page to
    type the password on in the first place. Identity is enforced
    separately, unconditionally and per-session by
    :func:`require_admin_password`, inside the page's own render function -
    see ``docs/assumptions/4.md``.
    """
    return bool(os.getenv(ADMIN_PASSWORD_ENV, "").strip())


def admin_access_granted() -> bool:
    """Cheap, side-effect-free: has *this* session already proven the admin password?

    Never prompts and never sleeps - safe to call from anywhere that only
    needs to know the answer, not force it. Fails closed when
    ``DASHBOARD_ADMIN_PASSWORD`` is unset, regardless of what session state
    might otherwise say (a stale flag from a credential that has since been
    unset must not keep granting access), matching
    :func:`require_admin_password`'s own fail-closed rule.
    """
    if not admin_password_configured():
        return False
    return bool(st.session_state.get(_ADMIN_SESSION_KEY))


def require_admin_password() -> bool:
    """Gate the Integrity page. Returns ``True`` only for a proven admin session.

    This is a return-value gate, not solely a ``st.stop()``-and-hope: this
    module's own doctest above (and ``docs/assumptions/4.md``) notes that
    ``st.stop()`` is a documented no-op outside a real Streamlit script run,
    so a caller that must also work under a bare unit test cannot rely on it
    alone to guarantee "never runs the rest of the page" - only a live
    Streamlit session gets that from ``st.stop()``'s halting side effect.
    ``pages/integrity.py`` therefore branches on this function's return
    value explicitly (``if not require_admin_password(): return``) *and*
    this function still calls ``st.stop()`` in every failing branch, so a
    live run halts here even if a future edit to the caller forgets to
    check the return value.

    No local-dev bypass, ever: an unset ``DASHBOARD_ADMIN_PASSWORD`` returns
    ``False`` unconditionally, before anything password-shaped is even read
    from the visitor.
    """
    expected = os.getenv(ADMIN_PASSWORD_ENV, "").strip()
    if not expected:
        st.error(
            f"{ADMIN_PASSWORD_ENV} is not set on this deployment. This page "
            "stays disabled until it is - there is no local-dev bypass here."
        )
        st.stop()
        return False
    if st.session_state.get(_ADMIN_SESSION_KEY):
        return True

    st.title("Integrity — admin sign-in required")
    st.caption(
        "Visible to Angel only. The shared dashboard password above does not "
        "unlock this page - it needs its own credential."
    )
    entered = st.text_input("Admin password", type="password", key="_integrity_admin_password")
    if not entered:
        st.stop()
        return False

    global _admin_failed_attempts
    if hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8")):
        st.session_state[_ADMIN_SESSION_KEY] = True
        _admin_failed_attempts = 0
        # The cookie-writing gate above reruns for the same reason: anything
        # rendered from here on in this run is thrown away by st.rerun().
        st.rerun()
        return True

    _admin_failed_attempts += 1
    time.sleep(min(_admin_failed_attempts, _MAX_BACKOFF_SECONDS))
    st.error("Incorrect admin password.")
    st.stop()
    return False
