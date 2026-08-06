"""Read-only client for Amplitude's Dashboard REST API: funnels and event counts.

The order book says how much the shop sold. It cannot say how many people tried
and failed, and that is the number a leadership meeting actually argues about.
This module answers it from the product's own event stream: how far people get
towards an order, and how many of them hit an error on the way.

Every call is a GET against ``/api/2/funnels``, authenticated with the project's
API key and secret key as HTTP Basic. Amplitude issues no narrower credential
than a project key, which is why this module has no other verb.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
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

# Amplitude's pseudo-event for "did anything at all".
_ANY_EVENT = "_active"


@dataclass(frozen=True)
class Step:
    """One rung of the funnel: what to count, and what to call it."""

    label: str
    event: str


# Deliberately not home -> search -> product: on this shop 28k of 39k monthly
# visitors land straight on a product page from search engines and never see the
# home page, so a funnel starting there would describe a few hundred people.
DEFAULT_FUNNEL = (
    Step("Visited", _ANY_EVENT),
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

# Where an error happened, and what it said. Grouped by event property because
# "4% of visitors saw an error" is not a ticket anybody can pick up, whereas
# "1,557 of them on the product page, all chunk-load failures" is one.
ERROR_EVENT = "app_error_occurred"
ERROR_BREAKDOWNS = (
    ("Where it happened", "page_type"),
    ("What it said", "message"),
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
    offset_days: int = 0,
) -> pd.DataFrame:
    """People reaching each step, with the drop-off between them.

    Columns: ``step``, ``event``, ``users``, ``from_previous``, ``from_start``,
    ``lost``. ``from_previous`` is the number leadership should look at - it
    names the one screen that is costing the most - and ``from_start`` is the
    figure people mean by "conversion rate".

    ``offset_days`` shifts the whole window back, so passing ``days`` again as
    the offset reads the period immediately before this one: a rate is only
    news next to the rate before it.
    """
    if len(steps) < 2:
        raise ValueError("A funnel needs at least two steps.")
    start, end = _window(days, offset_days)
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
    query, so this tries each in turn. Only keys carrying *people* are read:
    ``stepByStep`` and the ``stepTrans`` family are deliberately not among them,
    being ratios and transitions that would truncate to a funnel of zeros - which
    reads as a shop nobody visited rather than as a response this code failed to
    understand. An unrecognised shape raises instead.
    """
    data = payload.get("data")
    block: dict = {}
    if isinstance(data, list) and data and isinstance(data[0], dict):
        block = data[0]
    elif isinstance(data, dict):
        block = data
    for name in ("cumulativeRaw", "cumulative"):
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
    offset_days: int = 0,
) -> pd.DataFrame:
    """How many distinct people fired each event in the window.

    One request per event, because each count has to be deduplicated over the
    whole window independently of the others.
    """
    rows = []
    for step in events:
        rows.append(
            {
                "label": step.label,
                "event": step.event,
                "users": _unique_users(
                    credentials, step.event, days, offset_days
                ),
            }
        )
    return pd.DataFrame(rows, columns=["label", "event", "users"])


def _unique_users(
    credentials: tuple[str, str, str],
    event: str,
    days: int,
    offset_days: int = 0,
) -> int:
    """Distinct people firing ``event`` over the whole window, not per day.

    Asked as a two-step funnel rather than of the segmentation endpoint, whose
    interval only comes in days, weeks or months: no interval spans an arbitrary
    window, so any count from it is a sum of buckets, and adding buckets counts
    somebody who came back next week as two people - which for an error count
    overstates how many customers were let down. A funnel's first step is
    unconditional and deduplicated over the whole interval, so it is the count
    wanted, and it is the same arithmetic as the funnel above, which matters when
    the two are read side by side.

    The second step is ``_active`` (any event at all) purely because Amplitude
    will not accept a one-step funnel; nothing reads it.
    """
    start, end = _window(days, offset_days)
    payload = _get(
        credentials,
        "/api/2/funnels",
        [
            ("e", _event_param(event)),
            ("e", _event_param(_ANY_EVENT)),
            ("start", start),
            ("end", end),
            ("mode", _FUNNEL_MODE),
            ("cs", str(DEFAULT_CONVERSION_DAYS * 24 * 60 * 60)),
        ],
    )
    try:
        return _funnel_counts(payload, 1)[0]
    except AmplitudeConfigError:
        # An event nobody has ever fired - a friction event on a healthy week,
        # or one this deployment does not send - is a zero, not a broken panel.
        # A refused credential has already raised from _get by now.
        return 0


# How many rows of a breakdown are worth reading out. Past this the tail is
# noise, and the caption says what was left off rather than the table implying
# these are all of them.
BREAKDOWN_ROWS = 8

# Amplitude will return every distinct value of a property, and a message field
# carrying a build hash has thousands of them. This ceiling is on what it sends,
# not on what is shown: the values are collapsed into families first, so the
# families are only right if the whole tail arrived.
_BREAKDOWN_LIMIT = 1000


def event_breakdown(
    credentials: tuple[str, str, str],
    event: str,
    prop: str,
    days: int,
) -> pd.DataFrame:
    """Occurrences of ``event`` split by one of its event properties.

    Counted in events rather than people, deliberately: this is the only figure
    here that can be added up across days without counting somebody twice, and
    the question a breakdown answers - which of these is the big one - is about
    volume. The people count next to it stays the deduplicated one.

    Columns: ``value``, ``events``, ordered by ``events``, with values collapsed
    into families first (see ``collapse_value``) so a hundred build hashes read
    as one problem.
    """
    start, end = _window(days)
    payload = _get(
        credentials,
        "/api/2/events/segmentation",
        [
            (
                "e",
                json.dumps(
                    {
                        "event_type": event,
                        "group_by": [{"type": "event", "value": prop}],
                    }
                ),
            ),
            ("m", "totals"),
            ("start", start),
            ("end", end),
            ("i", "1"),
            ("limit", str(_BREAKDOWN_LIMIT)),
        ],
    )
    rows = _segmentation_totals(payload, event)
    if not rows:
        return pd.DataFrame(columns=["value", "events"])
    frame = pd.DataFrame(rows, columns=["value", "events"])
    frame["value"] = frame["value"].map(collapse_value)
    frame = frame.groupby("value", as_index=False)["events"].sum()
    return frame.sort_values("events", ascending=False).reset_index(drop=True)


def _segmentation_totals(payload: dict, event: str) -> list[tuple[str, int]]:
    """``(label, total)`` per group in an event segmentation response.

    Amplitude returns one series of per-interval numbers per group. Summing them
    is only valid because the metric asked for is ``totals``; the same sum over
    unique users would count a person who came back on Tuesday twice.

    A grouping Amplitude did not apply - an event that never carries the
    property - comes back as the ungrouped shape: one series labelled with the
    event itself. That is dropped, because a table whose only row is
    ``app_error_occurred`` claims to be a breakdown and is not one.
    """
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    series = data.get("series")
    labels = data.get("seriesLabels")
    if not isinstance(series, list) or not isinstance(labels, list):
        return []
    rows: list[tuple[str, int]] = []
    for index, values in enumerate(series):
        if index >= len(labels) or not isinstance(values, list):
            continue
        label = _series_label(labels[index])
        if len(series) == 1 and label == event:
            return []
        rows.append((label, sum(_as_ints(values))))
    return rows


def _series_label(label) -> str:
    """A group's name, however Amplitude wrapped it.

    Grouped responses label a series either with the property value itself or
    with an ``[event index, value]`` pair, depending on the endpoint.
    """
    if isinstance(label, list):
        parts = [str(part) for part in label if isinstance(part, str)]
        return parts[-1] if parts else str(label)
    return str(label)


# A hashed build asset, an id, or a number, anywhere in a property value. The
# hex alternative is deliberately long: a four-character word should stay a word.
# No boundary at the end, so a unit stuck to a number (30000ms) collapses too.
_VARIABLE_PART = re.compile(r"\b(?:[0-9a-f]{8,}|\d+)", re.IGNORECASE)

# An address, and the bracketed aside built around one: "(error: https://...)"
# left behind as "(error:" is worse than not having said it.
_BRACKETED_URL = re.compile(r"[([{][^()\[\]{}]*https?://[^()\[\]{}]*[)\]}]")
_URL = re.compile(r"https?://\S+")


def collapse_value(value: str) -> str:
    """One error, however many builds it happened on.

    ``Loading chunk 36187 failed. (error: /_next/static/chunks/36187-2d6f...js)``
    is the same bug as the same line with a different hash in it, and there are
    hundreds of those hashes. Grouped raw, the top of the table is ten rows of
    one problem and the real second-biggest problem is off the bottom.
    """
    text = str(value).strip()
    if not text:
        return "(not set)"
    text = text.replace("\n", " ")
    text = _BRACKETED_URL.sub("", text)
    text = _URL.sub("", text)
    text = _VARIABLE_PART.sub("#", text)
    text = " ".join(text.split()).strip(" .:-()")
    return text or "(not set)"


__all__ = [
    "AI_EVENTS",
    "AmplitudeConfigError",
    "BREAKDOWN_ROWS",
    "DEFAULT_CONVERSION_DAYS",
    "DEFAULT_FUNNEL",
    "ERROR_BREAKDOWNS",
    "ERROR_EVENT",
    "FRICTION_EVENTS",
    "Step",
    "collapse_value",
    "event_breakdown",
    "event_users",
    "funnel",
    "load_amplitude_env",
    "parse_funnel",
]
