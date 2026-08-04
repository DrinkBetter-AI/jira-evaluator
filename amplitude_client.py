"""Read-only client for Amplitude's Dashboard REST API: funnels and event counts.

The order book says how much the shop sold. It cannot say how many people tried
and failed, and that is the number a leadership meeting actually argues about.
This module answers it from the product's own event stream: how far people get
towards an order, and how many of them hit an error on the way.

Every call is a GET against ``/api/2/funnels`` or ``/api/2/events/segmentation``,
authenticated with the project's API key and secret key as HTTP Basic. Amplitude
issues no narrower credential than a project key, which is why this module has
no other verb.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import requests

# Amplitude serves EU projects from a different host, and a project's keys are
# refused by the wrong one, so the region is configurable rather than assumed.
DEFAULT_BASE_URL = "https://amplitude.com"

_KEY_ENV_VAR = "AMPLITUDE_API_KEY"
_SECRET_ENV_VAR = "AMPLITUDE_SECRET_KEY"

_TIMEOUT_SECONDS = 90

# Amplitude counts a funnel conversion only if the steps happen within this
# window. A wine is not a T-shirt: people read about it, leave, and come back,
# so a day-long window would report the shop as worse than it is.
DEFAULT_CONVERSION_DAYS = 7

# Ordered rather than "any order", because the question is whether people get
# through the shop, not whether they eventually touched all of these events.
_FUNNEL_MODE = "ordered"


@dataclass(frozen=True)
class Step:
    """One rung of the funnel: what to count, and what to call it."""

    label: str
    event: str


# Deliberately not home -> search -> product: on this shop 28k of 39k monthly
# visitors land straight on a product page from search engines and never see the
# home page, so a funnel starting there would describe a few hundred people.
DEFAULT_FUNNEL = (
    Step("Visited", "_active"),
    Step("Product page", "pdp_viewed"),
    Step("Added to cart", "cart_product_added"),
    Step("Started checkout", "checkout_started"),
    Step("Submitted payment", "checkout_payment_submitted"),
    Step("Order placed", "checkout_order_completed"),
)

# Things going wrong, counted in people rather than events: one person meeting
# the same error ten times is one person to apologise to, not ten.
FRICTION_EVENTS = (
    Step("Saw an app error", "app_error_occurred"),
    Step("Add to cart failed", "cart_product_add_failed"),
    Step("Checkout blocked", "cart_checkout_blocked"),
    Step("Payment failed", "checkout_payment_failed"),
    Step("Searched, found nothing", "search_zero_results"),
)

# Whether the thing the company is built on is being used at all.
AI_EVENTS = (
    Step("Opened Voss AI", "vossai_opened"),
    Step("Asked Voss AI something", "vossai_query_submitted"),
    Step("Voss AI found nothing", "vossai_zero_results"),
)


class AmplitudeConfigError(RuntimeError):
    """Raised when the Amplitude credential is missing or refused."""


def load_amplitude_env() -> tuple[str, str, str] | None:
    """Return ``(api_key, secret_key, base_url)``, or ``None`` when unset.

    Both keys or neither: a project API key on its own is the storefront-side
    credential and cannot read anything here, so accepting it alone would only
    produce a confusing 401 later.
    """
    key = os.getenv(_KEY_ENV_VAR, "").strip()
    secret = os.getenv(_SECRET_ENV_VAR, "").strip()
    if not key and not secret:
        return None
    if not key or not secret:
        raise AmplitudeConfigError(
            f"Both {_KEY_ENV_VAR} and {_SECRET_ENV_VAR} are needed; Amplitude's "
            "Dashboard API authenticates with the pair, not the API key alone."
        )
    base = os.getenv("AMPLITUDE_API_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    # The secret key travels as a credential, so the host it is sent to has to be
    # deliberate and the connection encrypted.
    if not base.lower().startswith("https://"):
        raise AmplitudeConfigError(
            "AMPLITUDE_API_URL must be an https:// address; the secret key is "
            "sent to it as a credential and will not be sent in the clear."
        )
    return key, secret, base.rstrip("/")


def parse_funnel(spec: str) -> tuple[Step, ...]:
    """Read a funnel out of ``"Label=event,Label=event"``, else the default one.

    Every shop's path to an order is its own, and the events behind these labels
    are this codebase's; a deployment that renames them should not need a patch.
    """
    steps: list[Step] = []
    for chunk in spec.split(","):
        label, _, event = chunk.partition("=")
        label, event = label.strip(), event.strip()
        if label and event:
            steps.append(Step(label, event))
    return tuple(steps) if len(steps) >= 2 else DEFAULT_FUNNEL


def _window(days: int, offset_days: int = 0) -> tuple[str, str]:
    """Amplitude's ``YYYYMMDD`` bounds for the ``days`` up to ``offset_days`` ago.

    Both bounds are inclusive whole days in the project's own timezone, which is
    why this cannot simply subtract from a timestamp and why the last day is
    yesterday: today is still being written and would read as a slump.
    """
    end = _dt.date.today() - _dt.timedelta(days=1 + offset_days)
    start = end - _dt.timedelta(days=max(days, 1) - 1)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _get(
    credentials: tuple[str, str, str], path: str, params: list[tuple[str, str]]
) -> dict:
    key, secret, base = credentials
    response = requests.get(
        f"{base}{path}",
        params=params,
        auth=(key, secret),
        timeout=_TIMEOUT_SECONDS,
        headers={"Accept": "application/json"},
    )
    if response.status_code in (401, 403):
        raise AmplitudeConfigError(
            "Amplitude refused the API key and secret key pair. Check they both "
            "belong to the same project, and that the project is in the region "
            "AMPLITUDE_API_URL points at."
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise AmplitudeConfigError("Amplitude returned an unexpected response shape.")
    return payload


def _event_param(event: str) -> str:
    return json.dumps({"event_type": event})


def funnel(
    credentials: tuple[str, str, str],
    steps: Sequence[Step],
    days: int,
) -> pd.DataFrame:
    """People reaching each step, with the drop-off between them.

    Columns: ``step``, ``event``, ``users``, ``from_previous``, ``from_start``,
    ``lost``. ``from_previous`` is the number leadership should look at - it
    names the one screen that is costing the most - and ``from_start`` is the
    figure people mean by "conversion rate".
    """
    if len(steps) < 2:
        raise ValueError("A funnel needs at least two steps.")
    start, end = _window(days)
    params: list[tuple[str, str]] = [("e", _event_param(s.event)) for s in steps]
    params += [
        ("start", start),
        ("end", end),
        ("mode", _FUNNEL_MODE),
        ("cs", str(DEFAULT_CONVERSION_DAYS * 24 * 60 * 60)),
    ]
    payload = _get(credentials, "/api/2/funnels", params)
    counts = _funnel_counts(payload, len(steps))
    frame = pd.DataFrame(
        {
            "step": [s.label for s in steps],
            "event": [s.event for s in steps],
            "users": counts,
        }
    )
    first = counts[0] if counts else 0
    previous = frame["users"].shift(1)
    frame["from_previous"] = _ratio(frame["users"], previous)
    frame["from_start"] = _ratio(frame["users"], pd.Series([first] * len(frame)))
    frame["lost"] = (previous - frame["users"]).fillna(0).astype(int)
    return frame


def _funnel_counts(payload: dict, expected: int) -> list[int]:
    """The per-step user counts out of a funnels response.

    Amplitude reports the same numbers under more than one key depending on the
    query, so this prefers the cumulative counts and falls back rather than
    failing: a funnel that renders one step short is still readable, an
    exception is not.
    """
    data = payload.get("data")
    block: dict = {}
    if isinstance(data, list) and data and isinstance(data[0], dict):
        block = data[0]
    elif isinstance(data, dict):
        block = data
    for name in ("cumulativeRaw", "cumulative", "stepByStep"):
        values = block.get(name)
        if isinstance(values, list) and values:
            counts = _as_ints(values)
            if len(counts) >= expected:
                return counts[:expected]
    series = block.get("series")
    if isinstance(series, list) and series and isinstance(series[0], list):
        counts = _as_ints(series[0])
        if len(counts) >= expected:
            return counts[:expected]
    raise AmplitudeConfigError(
        "Amplitude returned a funnel with no step counts in it. The events in "
        "AMPLITUDE_FUNNEL may not exist in this project."
    )


def _as_ints(values: list) -> list[int]:
    out: list[int] = []
    for value in values:
        number = pd.to_numeric(value, errors="coerce")
        out.append(0 if pd.isna(number) else int(number))
    return out


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """``numerator / denominator``, with an empty step reading as nothing lost."""
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = numerator.divide(denominator.where(denominator > 0))
    return result.fillna(0.0)


def event_users(
    credentials: tuple[str, str, str],
    events: Sequence[Step],
    days: int,
) -> pd.DataFrame:
    """How many distinct people fired each event in the window.

    One request per event: Amplitude's segmentation endpoint takes a second
    event only as a comparison series, and reading five friction counts as five
    small calls is simpler than reading them as two shapes.
    """
    rows = []
    for step in events:
        rows.append(
            {
                "label": step.label,
                "event": step.event,
                "users": _unique_users(credentials, step.event, days),
            }
        )
    return pd.DataFrame(rows, columns=["label", "event", "users"])


def _unique_users(credentials: tuple[str, str, str], event: str, days: int) -> int:
    """Distinct people firing ``event`` over the whole window, not per day.

    The interval is the window, so Amplitude dedupes across it itself. Summing
    daily uniques instead would count somebody who came back twice as two
    people, and for an error count that overstates the harm.
    """
    start, end = _window(days)
    payload = _get(
        credentials,
        "/api/2/events/segmentation",
        [
            ("e", _event_param(event)),
            ("start", start),
            ("end", end),
            ("m", "uniques"),
            ("i", str(max(days, 1))),
        ],
    )
    data = payload.get("data")
    if not isinstance(data, dict):
        return 0
    series = data.get("series")
    if not isinstance(series, list) or not series:
        return 0
    first = series[0]
    if not isinstance(first, list):
        return 0
    return sum(_as_ints(first))


__all__ = [
    "AI_EVENTS",
    "AmplitudeConfigError",
    "DEFAULT_CONVERSION_DAYS",
    "DEFAULT_FUNNEL",
    "FRICTION_EVENTS",
    "Step",
    "event_users",
    "funnel",
    "load_amplitude_env",
    "parse_funnel",
]
