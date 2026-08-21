"""Unit coverage for the Postgres-backed Vivino crawler transformation."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import crawler_prices as cp  # noqa: E402


UTC = _dt.timezone.utc


def row(
    wine_id: int,
    vintage_id: int,
    year: int | None,
    offers: list[dict],
    updated_at: _dt.datetime,
    title: str | None,
):
    return (wine_id, vintage_id, year, offers, updated_at, title)


def offer(
    merchant_name: str,
    amount: float,
    *,
    volume_ml: int = 750,
    bottle_quantity: int = 1,
):
    return {
        "merchant_name": merchant_name,
        "amount": amount,
        "volume_ml": volume_ml,
        "bottle_quantity": bottle_quantity,
    }


def test_filters_merchant_and_volume_and_keeps_cheapest_single_offer():
    read = cp.shop_from_rows(
        "yiannis-wine-shop",
        [
            row(
                1,
                11,
                2020,
                [
                    offer("Capital Fine Wine", 4.0),
                    offer("Yianniswinery.com", 19.0),
                    offer("Yiannis Wine Shop", 17.0),
                    offer("Yianniswinery.com", 9.0, volume_ml=1500),
                ],
                _dt.datetime(2026, 8, 21, 17, tzinfo=UTC),
                "Wine A",
            ),
        ],
    )

    assert read is not None
    assert read.listed == 1
    assert read.packs == 0
    assert list(read.listings["name"]) == ["Wine A"]
    assert list(read.listings["price"]) == [17.0]


def test_pack_only_vintage_is_counted_and_mixed_vintage_is_listed():
    read = cp.shop_from_rows(
        "yiannis-wine-shop",
        [
            row(
                1,
                11,
                2020,
                [offer("Yianniswinery.com", 9.0, bottle_quantity=12)],
                _dt.datetime(2026, 8, 21, 17, tzinfo=UTC),
                "Pack Wine",
            ),
            row(
                2,
                12,
                2021,
                [
                    offer("Yianniswinery.com", 8.0, bottle_quantity=6),
                    offer("Yianniswinery.com", 18.0),
                ],
                _dt.datetime(2026, 8, 21, 18, tzinfo=UTC),
                "Single Wine",
            ),
        ],
    )

    assert read is not None
    assert read.listed == 2
    assert read.packs == 1
    assert list(read.listings["name"]) == ["Single Wine"]


def test_dedupes_same_key_at_cheapest_price_and_tracks_freshness():
    read = cp.shop_from_rows(
        "capital-fine-wine",
        [
            row(
                1,
                11,
                2020,
                [offer("Capital Fine Wine", 21.0)],
                _dt.datetime(2026, 8, 21, 17, tzinfo=UTC),
                "Wine A",
            ),
            row(
                2,
                12,
                2020,
                [offer("capital-fine-wine.com", 19.0)],
                _dt.datetime(2026, 8, 21, 19, tzinfo=UTC),
                "Wine A",
            ),
            row(
                3,
                13,
                2020,
                [offer("Yianniswinery.com", 11.0)],
                _dt.datetime(2026, 8, 21, 23, tzinfo=UTC),
                "Other Merchant",
            ),
        ],
    )

    assert read is not None
    assert len(read.listings) == 1
    assert read.listings.iloc[0]["price"] == 19.0
    assert read.crawled_at == _dt.datetime(2026, 8, 21, 19, tzinfo=UTC)


def test_skips_blank_titles_but_keeps_vintage_listing_count():
    read = cp.shop_from_rows(
        "yiannis-wine-shop",
        [
            row(
                1,
                11,
                None,
                [offer("Yianniswinery.com", 14.0)],
                _dt.datetime(2026, 8, 21, 17, tzinfo=UTC),
                " ",
            ),
        ],
    )

    assert read is not None
    assert read.listed == 1
    assert read.listings.empty
    assert read.packs == 0


def test_returns_none_when_no_offer_matches_the_merchant():
    read = cp.shop_from_rows(
        "capital-fine-wine",
        [
            row(
                1,
                11,
                2020,
                [offer("Yianniswinery.com", 14.0)],
                _dt.datetime(2026, 8, 21, 17, tzinfo=UTC),
                "Wine A",
            ),
        ],
    )

    assert read is None


def test_unknown_slug_has_no_crawler_merchant():
    assert cp.shop_from_rows("unknown-shop", []) is None
