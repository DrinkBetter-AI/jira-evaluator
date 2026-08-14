"""What a merchant charges on Vivino against what they charge on VinoVoss.

Some merchants list the same cellar in both shops, and the suspicion this
answers is simple: are they quoting Vivino a keener price for the same bottle?
Vivino's public explore feed is read per merchant, single 0.75l bottles only,
and matched to the shop's own catalogue by wine name and vintage - the only
identifiers both sides publish.

Read-only throughout: nothing here writes to Vivino, and the requests are the
same ones a browser makes opening the shop's public page.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass

import pandas as pd
import requests

# The merchants known to keep a Vivino shop, by the name the CRM's store list
# gives them. A merchant missing here is not compared rather than guessed at:
# Vivino has no search that ties a shop to a name reliably enough to trust.
VIVINO_SHOPS: dict[str, str] = {
    "Yiannis Wine Shop": "yiannis-wine-shop",
    "Capital Fine Wine": "capital-fine-wine",
}

_SITE = "https://www.vivino.com"
_EXPLORE = f"{_SITE}/api/explore/explore"
_HEADERS = {
    # Vivino serves the public API to browsers; a bare client gets a 403.
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
_TIMEOUT = 30
# A page holds 24 listings, and the feed stops turning pages long before a
# 13,000-wine shop runs out, so the read walks the price axis instead: when
# the pages stop, start again from the highest price seen. The request budget
# is the honest ceiling - a shop bigger than it is reported as partly read.
_PAGE_SIZE = 24
_MAX_REQUESTS = 900
_PAUSE_SECONDS = 0.15
# The wall-clock ceiling, held under the deployment's 1800s request limit: a
# slow feed ends the read with what it has rather than outliving the server's
# patience and losing everything to a dropped connection.
_TIME_BUDGET_SECONDS = 1500
_PRICE_CEILING = 100000
# Within this fraction of each other, the two prices are the same price: both
# shops round differently and Vivino's feed lags a reprice by a day.
SAME_PRICE = 0.02

_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_LETTERS = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


class VivinoError(RuntimeError):
    """Raised when Vivino cannot be read at all, as distinct from reading nothing."""


@dataclass(frozen=True)
class Shop:
    """One merchant's Vivino listings, and how completely they were read."""

    slug: str
    listings: pd.DataFrame  # name, year, price, key
    listed: int  # what Vivino says the shop holds, matched or not
    complete: bool  # False when the request budget ran out first
    packs: int = 0  # wines offered only per bottle of a 3/6/12 pack


@dataclass(frozen=True)
class Comparison:
    """The same bottles priced in both shops, one row per matched wine."""

    rows: pd.DataFrame  # wine, year, ours, theirs, gap
    listed: int
    complete: bool
    unmatched_ours: int = 0
    ours_counted: int = 0  # our own comparable bottles, matched or not
    packs: int = 0  # Vivino wines left out because their price needs a pack

    @property
    def matched(self) -> int:
        return len(self.rows)

    @property
    def cheaper_there(self) -> pd.DataFrame:
        """Matched wines Vivino sells for meaningfully less, worst first."""
        if self.rows.empty:
            return self.rows
        below = self.rows[self.rows["gap"] < -SAME_PRICE]
        return below.sort_values("gap").reset_index(drop=True)

    @property
    def dearer_there(self) -> int:
        return int((self.rows["gap"] > SAME_PRICE).sum()) if self.matched else 0

    @property
    def same(self) -> int:
        return self.matched - len(self.cheaper_there) - self.dearer_there


def wine_key(name: str, year: int) -> str:
    """One string both catalogues agree on for the same bottle.

    Name and vintage are all either side publishes, so the key is the name
    with its accents, punctuation and any embedded year stripped, plus the
    vintage. A non-vintage bottle keeps year 0 and only matches non-vintage.
    """
    flat = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore")
    text = _LETTERS.sub(" ", flat.decode().lower())
    text = _YEAR.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    return f"{text}|{int(year)}"


def vintage_year(value: object) -> int:
    """A vintage as an integer year, with Vivino's "N.V." reading as none."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def year_of(name: str) -> int:
    """The vintage a listing's name carries, or 0 where it carries none."""
    found = _YEAR.findall(str(name))
    match = _YEAR.search(str(name))
    return int(match.group(0)) if found and match else 0


def merchant_id(slug: str, session: requests.Session | None = None) -> int | None:
    """The numeric id behind a Vivino shop page, or None for a page without one."""
    http = session or requests.Session()
    try:
        page = http.get(
            f"{_SITE}/en/merchants/{slug}", headers=_HEADERS, timeout=_TIMEOUT
        )
        page.raise_for_status()
    except requests.RequestException as exc:
        raise VivinoError(f"Vivino's shop page for {slug} could not be read: {exc}")
    match = re.search(r"merchant_id[^0-9]{0,10}(\d+)", page.text)
    return int(match.group(1)) if match else None


def _page(
    http: requests.Session, merchant: int, price_from: float, page: int
) -> dict:
    reply = http.get(
        _EXPLORE,
        headers=_HEADERS,
        timeout=_TIMEOUT,
        params={
            "country_code": "US",
            "currency_code": "USD",
            "merchant_id": merchant,
            "order_by": "price",
            "order": "asc",
            "page": page,
            "price_range_min": price_from,
            "price_range_max": _PRICE_CEILING,
        },
    )
    reply.raise_for_status()
    return reply.json().get("explore_vintage", {})


def fetch_shop(slug: str, session: requests.Session | None = None) -> Shop:
    """Every single-bottle listing the shop has on Vivino, or as many as the budget reads.

    The explore feed sorts by price and stops turning pages well short of a
    big shop, so when a walk stalls the read resumes from the highest price it
    reached. Bottles other than 0.75l are dropped - a magnum is not the same
    product at a different price.
    """
    http = session or requests.Session()
    merchant = merchant_id(slug, http)
    if merchant is None:
        # A page without a readable merchant id is a page that could not be
        # read - a redesign, a bot wall - not a merchant with an empty shop.
        raise VivinoError(
            f"Vivino's page for {slug} no longer carries a merchant id, so "
            "their listings could not be read."
        )

    rows: dict[int, tuple[str, int, float]] = {}
    packed: set[int] = set()
    listed = 0
    price_from = 0.0
    requests_made = 0
    started = time.monotonic()

    def in_budget() -> bool:
        return (
            requests_made < _MAX_REQUESTS
            and time.monotonic() - started < _TIME_BUDGET_SECONDS
        )
    # Complete only when a walk serves nothing at or above the last price
    # seen, or nothing that was not already seen; a failure or an exhausted
    # budget is an admission, not a completion.
    complete = False
    while in_budget():
        page_number = 1
        new_rows = 0
        drained = False
        top_reached = False
        failed = False
        # Held for the whole walk: raising the floor between pages would
        # shift the result set under the page number and skip listings.
        walk_from = price_from
        while in_budget():
            try:
                found = _page(http, merchant, walk_from, page_number)
            except requests.RequestException as exc:
                if not rows:
                    raise VivinoError(f"Vivino refused the {slug} listings: {exc}")
                failed = True
                break
            requests_made += 1
            listed = max(listed, int(found.get("records_matched") or 0))
            matches = found.get("matches") or []
            if not matches:
                drained = True
                top_reached = page_number == 1
                break
            for match in matches:
                price = match.get("price") or {}
                bottle = price.get("bottle_type") or {}
                vintage = match.get("vintage") or {}
                amount = price.get("amount")
                if amount:
                    price_from = max(price_from, float(amount))
                if not amount or bottle.get("volume_ml") != 750:
                    continue
                # A case price is quoted per bottle but earned by buying the
                # case; only a price a shopper pays for one bottle compares
                # with a single-bottle price here.
                if (
                    (price.get("bottle_quantity") or 1) > 1
                    or (price.get("minimum_unit_quantity") or 1) > 1
                ):
                    packed.add(int(vintage.get("id") or 0) or hash(str(vintage.get("name"))))
                    continue
                name = str(vintage.get("name") or "").strip()
                if not name:
                    continue
                key = int(vintage.get("id") or 0) or hash((name, amount))
                if key not in rows:
                    # Some listings carry the vintage only in the wine's name,
                    # leaving the year field blank; it is the same bottle.
                    year = vintage_year(vintage.get("year")) or year_of(name)
                    rows[key] = (name, year, float(amount))
                    new_rows += 1
            page_number += 1
            time.sleep(_PAUSE_SECONDS)
        if failed:
            break
        if top_reached:
            complete = True
            break
        if not new_rows:
            if price_from > walk_from:
                # The floor advanced on bottles that are not compared - other
                # sizes, unpriced rows - so the shop is not finished, only
                # this walk; the next one starts above them.
                continue
            # The walk ran its pages out serving only listings already seen at
            # an unmoved floor: everything at or above it is known, the top.
            complete = drained
            break

    frame = pd.DataFrame(
        [(name, year, price) for name, year, price in rows.values()],
        columns=["name", "year", "price"],
    )
    if not frame.empty:
        frame["key"] = [
            wine_key(name, year) for name, year in zip(frame["name"], frame["year"])
        ]
        # The same wine can appear at two prices while a reprice propagates;
        # the cheapest is the one a shopper is shown first.
        frame = (
            frame.sort_values("price").drop_duplicates("key").reset_index(drop=True)
        )
    return Shop(
        slug=slug,
        listings=frame,
        listed=listed,
        complete=complete,
        packs=len(packed),
    )


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["name", "year", "price", "key"])


def compare(ours: pd.DataFrame, shop: Shop) -> Comparison:
    """Our single-bottle prices beside Vivino's, matched by name and vintage.

    ``ours`` needs ``title``, ``year`` and ``price`` columns; the gap is
    Vivino's price against ours, so a negative gap is a bottle they sell
    cheaper on Vivino than they sell it to this shop's customers.
    """
    if ours.empty or shop.listings.empty:
        return Comparison(
            rows=pd.DataFrame(columns=["wine", "year", "ours", "theirs", "gap"]),
            listed=shop.listed,
            complete=shop.complete,
            unmatched_ours=len(ours),
            ours_counted=len(ours),
            packs=shop.packs,
        )
    mine = ours.copy()
    mine["key"] = [
        wine_key(title, year) for title, year in zip(mine["title"], mine["year"])
    ]
    # Two of our listings with one key is the same wine twice (a relist, a
    # duplicate row); the cheapest is what a shopper pays, so it stands.
    mine = mine.sort_values("price").drop_duplicates("key")
    joined = mine.merge(
        shop.listings[["key", "price"]].rename(columns={"price": "theirs"}),
        on="key",
        how="left",
    )
    matched = joined[joined["theirs"].notna()].copy()
    matched["gap"] = (matched["theirs"] - matched["price"]) / matched["price"]
    rows = (
        matched.rename(columns={"title": "wine", "price": "ours"})[
            ["wine", "year", "ours", "theirs", "gap"]
        ]
        .sort_values("gap")
        .reset_index(drop=True)
    )
    return Comparison(
        rows=rows,
        listed=shop.listed,
        complete=shop.complete,
        unmatched_ours=int(joined["theirs"].isna().sum()),
        ours_counted=len(joined),
        packs=shop.packs,
    )


def verdicts(name: str, result: Comparison) -> list[str]:
    """The comparison in sentences somebody can put to the merchant."""
    if not result.listed:
        return [
            f"{name} has a Vivino shop page but no live listings today, so "
            "there is nothing of theirs to compare."
        ]
    if not result.ours_counted:
        return [
            f"{name} lists {result.listed:,} wines on Vivino, but none of "
            "their own single-bottle prices could be read here, so there is "
            "nothing on our side to compare - a catalogue problem, not a "
            "pricing finding."
        ]
    if not result.matched:
        return [
            f"None of {name}'s {result.listed:,} Vivino listings matched a "
            "wine of theirs here by name and vintage, which is worth a look "
            "in itself: the same cellar should overlap."
        ]
    lines = []
    cheaper = result.cheaper_there
    if len(cheaper):
        worst = cheaper.iloc[0]
        lines.append(
            f"**{name} sells {len(cheaper):,} of the matched wines cheaper on "
            f"Vivino than here** - worst is {worst['wine']} at "
            f"{worst['gap']:+.0%}. That is the list to put in front of them."
        )
    else:
        lines.append(
            f"**{name} gives Vivino no better price on any matched wine** - "
            f"of {result.matched:,} wines in both shops, none is cheaper "
            "there beyond rounding."
        )
    lines.append(
        f"{result.matched:,} wines matched by name and vintage out of "
        f"{result.listed:,} they list on Vivino; {result.same:,} are priced "
        f"the same and {result.dearer_there:,} are more expensive on Vivino."
    )
    if not result.complete:
        lines.append(
            "Vivino's feed stopped before the whole shop was read, so these "
            "counts are of the listings that were - more overlap may exist."
        )
    return lines


__all__ = [
    "SAME_PRICE",
    "VIVINO_SHOPS",
    "Comparison",
    "Shop",
    "VivinoError",
    "compare",
    "fetch_shop",
    "merchant_id",
    "vintage_year",
    "verdicts",
    "wine_key",
    "year_of",
]
