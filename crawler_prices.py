"""Read the daily Vivino crawl as a fast, read-only alternative to Vivino's feed.

The crawler stores every merchant offer in one row per VinoVoss vintage. This
reader keeps the comparison's existing rules in Python, where the offer shape
is explicit and a merchant's pack prices cannot accidentally be compared with
one-bottle prices.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Iterable, Sequence
from contextlib import closing

import pandas as pd
import psycopg2

import orders_client
import vivino_client

_CRAWLER_SQL = """select c.wine_id,
       c.vintage_id,
       c.vintage_year,
       c.prices,
       c.updated_at,
       w.title
  from crawler.vivino_prices c
  left join public.wines_denormalized w on w.wine_id = c.wine_id
"""

_MERCHANT_PREFIXES: dict[str, tuple[str, ...]] = {
    "yiannis-wine-shop": ("yiannis",),
    "capital-fine-wine": ("capital",),
}
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _normalise_merchant(value: object) -> str:
    """Make merchant names comparable despite punctuation and casing."""
    return _NON_ALPHANUMERIC.sub("", str(value or "").lower())


def _is_merchant_offer(offer: object, prefixes: tuple[str, ...]) -> bool:
    """Return whether an offer belongs to the requested crawler merchant."""
    if not isinstance(offer, dict):
        return False
    merchant = _normalise_merchant(offer.get("merchant_name"))
    return any(merchant.startswith(prefix) for prefix in prefixes)


def _quantity(offer: dict) -> int:
    """Read a bottle quantity, treating absent and malformed values as one."""
    try:
        return int(offer.get("bottle_quantity") or 1)
    except (TypeError, ValueError):
        return 1


def _volume(offer: dict) -> int | None:
    """Read an offer's volume without letting malformed data abort the read."""
    try:
        return int(offer.get("volume_ml"))
    except (TypeError, ValueError):
        return None


def _amount(offer: dict) -> float | None:
    """Read a price that can be compared as a number."""
    try:
        amount = float(offer.get("amount"))
    except (TypeError, ValueError):
        return None
    return amount if amount > 0 else None


def _empty_listings() -> pd.DataFrame:
    """Return the same listing shape as the live Vivino reader."""
    return pd.DataFrame(columns=["name", "year", "price", "key"])


def shop_from_rows(
    slug: str,
    rows: Iterable[Sequence[object]],
) -> vivino_client.Shop | None:
    """Turn crawler rows into one merchant's comparable Vivino shop."""
    prefixes = _MERCHANT_PREFIXES.get(slug, ())
    if not prefixes:
        return None

    merchant_vintages: set[int] = set()
    packed_vintages: set[int] = set()
    single_vintages: set[int] = set()
    crawled: list[_dt.datetime] = []
    candidates: list[tuple[str, int, float]] = []

    for row in rows:
        if len(row) != 6:
            raise ValueError("Crawler rows must contain six columns.")
        _, vintage_id, vintage_year, prices, updated_at, title = row
        try:
            vintage = int(vintage_id)
        except (TypeError, ValueError):
            continue
        offers = prices if isinstance(prices, list) else []
        matching = [
            offer for offer in offers if _is_merchant_offer(offer, prefixes)
        ]
        if not matching:
            continue
        if updated_at is not None:
            crawled.append(updated_at)
        merchant_vintages.add(vintage)
        for offer in matching:
            if not isinstance(offer, dict) or _volume(offer) != 750:
                continue
            if _quantity(offer) > 1:
                packed_vintages.add(vintage)
                continue
            single_vintages.add(vintage)
            if title is None or not str(title).strip():
                continue
            amount = _amount(offer)
            if amount is None:
                continue
            try:
                year = int(vintage_year) if vintage_year is not None else 0
            except (TypeError, ValueError):
                year = 0
            candidates.append((str(title).strip(), year, amount))

    if not merchant_vintages:
        return None

    frame = pd.DataFrame(candidates, columns=["name", "year", "price"])
    if frame.empty:
        frame = _empty_listings()
    else:
        frame["key"] = [
            vivino_client.wine_key(name, year)
            for name, year in zip(frame["name"], frame["year"])
        ]
        frame = (
            frame.sort_values("price")
            .drop_duplicates("key")
            .reset_index(drop=True)
        )
    crawled_at = max(crawled) if crawled else None
    return vivino_client.Shop(
        slug=slug,
        listings=frame,
        listed=len(merchant_vintages),
        complete=True,
        packs=len(packed_vintages - single_vintages),
        crawled_at=crawled_at,
    )


def fetch_crawled_shop(
    config: orders_client.DbConfig,
    slug: str,
) -> vivino_client.Shop | None:
    """Read one merchant from Postgres, or return None when it has no offers."""
    try:
        with closing(
            orders_client.connect_readonly(config, purpose="crawler read")
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(_CRAWLER_SQL)
                rows = cursor.fetchall()
    except psycopg2.Error as exc:
        raise orders_client.MedusaConfigError(
            f"The crawler read at {config.label} refused the read: "
            f"{str(exc).strip()}"
        ) from exc
    return shop_from_rows(slug, rows)


__all__ = ["fetch_crawled_shop", "shop_from_rows"]
