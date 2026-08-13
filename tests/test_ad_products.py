"""Offline checks for how several ad accounts' product rows are put together.

Offline like the other check scripts: the BigQuery read either side of this is
verified against the live dataset, and what is worth a test here is the shaping,
because a column lost in it is a tab that will not draw.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ads_client  # noqa: E402
import app as dashboard  # noqa: E402


def rows(offers: list[tuple[str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        offers, columns=["offer", "spend", "clicks"]
    ).assign(impressions=100, ad_conversions=0.0)


def empty() -> pd.DataFrame:
    """What ``ads_client.product_stats`` returns for an account with no rows."""
    return pd.DataFrame(columns=list(ads_client.PRODUCT_COLUMNS))


def test_one_account_at_rest_does_not_cost_the_other_its_clicks():
    """An empty frame's columns are object dtype, and concatenating one used to
    promote clicks and impressions to object, where the numeric aggregation
    dropped them and the ledger raised on the columns it is built from."""
    together = dashboard._offers_together([rows([("a", 5.0, 7)]), empty()])
    assert list(together["offer"]) == ["a"]
    assert int(together["clicks"].iloc[0]) == 7
    assert int(together["impressions"].iloc[0]) == 100
    assert set(ads_client.PRODUCT_COLUMNS) <= set(together.columns)


def test_a_bottle_two_accounts_advertised_is_one_wine_with_both_their_spend():
    together = dashboard._offers_together(
        [rows([("a", 5.0, 7), ("b", 1.0, 1)]), rows([("a", 3.0, 2)])]
    )
    both = together[together["offer"] == "a"].iloc[0]
    assert float(both["spend"]) == 8.0 and int(both["clicks"]) == 9
    # Costliest first, as the panel reads it.
    assert list(together["offer"]) == ["a", "b"]


def test_every_account_at_rest_is_an_empty_ledger_rather_than_an_error():
    together = dashboard._offers_together([empty(), empty()])
    assert together.empty
    assert list(together.columns) == list(ads_client.PRODUCT_COLUMNS)
    assert dashboard._offers_together([]).empty
