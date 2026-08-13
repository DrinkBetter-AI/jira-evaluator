"""What each advertised bottle cost, and what it gave back.

The account argues about ad strategy in aggregate - spend, clicks, return - and
aggregates are where a disagreement stops being settleable: the person running
the campaign can always say the average hides the wines that work. This joins
the three records that between them name every wine, so any claim on the panel
can be opened and read line by line:

* Google Ads' own product table, one row per offer per day, for spend and clicks
* Merchant Center's benchmark, for what everybody else charges for that bottle
* the shop's order book, for the bottles that were actually paid for

The join key is the offer id all three carry. What it cannot do is attribute:
bottles sold are every sale of that wine in the window, not the sales an ad
caused, so a wine can show return it would have earned anyway. That is stated
wherever a return figure appears rather than corrected for, because correcting
for it would need conversion tracking the feed does not carry.
"""

from __future__ import annotations

import pandas as pd

import merchant_client

# Columns of the per-wine ledger, in the order the table and its file show them.
LEDGER_COLUMNS = (
    "offer",
    "title",
    "merchant",
    "spend",
    "clicks",
    "impressions",
    "bottles",
    "sold_revenue",
    "price",
    "benchmark",
    "gap",
)

# What a wine has to be to count as one worth the money: it sold at least a
# bottle. Deliberately not a return threshold - the return is not attributed, so
# a threshold on it would be arithmetic dressed as a rule.
SOLD = "Sold something"
NOTHING = "Sold nothing"

SPLIT_COLOURS = {SOLD: "#15803d", NOTHING: "#b91c1c"}


def ledger(
    ads: pd.DataFrame,
    prices: merchant_client.Prices,
    sales: merchant_client.Sales,
    named: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Every advertised offer, with its price, its market price and its sales.

    Left-joined from the ads side on purpose: a wine that took spend belongs in
    the ledger whether or not Google could benchmark it, and 4 offers in 5 have
    no benchmark, so an inner join would quietly drop most of the spend and
    every figure taken from it would be wrong about the account.
    """
    if ads.empty:
        return pd.DataFrame(columns=list(LEDGER_COLUMNS))
    wines = prices.offers[["offer", "title", "price", "benchmark", "gap"]]
    frame = sales.against(ads.merge(wines, on="offer", how="left"))
    if "sold_revenue" not in frame.columns:
        frame = frame.assign(bottles=0, sold_revenue=0.0)
    frame["merchant"] = frame["offer"].map(
        lambda offer: merchant_client.MERCHANT_SEPARATOR.join((named or {}).get(offer, ()))
    )
    return frame[list(LEDGER_COLUMNS)].sort_values("spend", ascending=False)


def spend_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Spend on the wines that sold, against spend on the wines that did not.

    The first picture the panel draws, because it is the one that does not
    depend on price at all: however the campaign is meant to work, this is how
    much of it went to bottles nobody bought.
    """
    if frame.empty:
        return pd.DataFrame(columns=["outcome", "wines", "spend", "clicks", "revenue"])
    sold = frame["bottles"] > 0
    rows = [
        {
            "outcome": name,
            "wines": int(part["offer"].size),
            "spend": round(float(part["spend"].sum()), 2),
            "clicks": int(part["clicks"].sum()),
            "revenue": round(float(part["sold_revenue"].sum()), 2),
        }
        for name, part in ((SOLD, frame[sold]), (NOTHING, frame[~sold]))
    ]
    return pd.DataFrame(rows)


def by_band(frame: pd.DataFrame) -> pd.DataFrame:
    """The same ledger folded into price bands, with what each gave back.

    Only the offers Google could benchmark: a wine with no market price cannot
    be in a band about market price, and putting it in one would make the band
    a statement about the join rather than about pricing.
    """
    columns = ["band", "wines", "spend", "clicks", "bottles", "revenue", "per_dollar"]
    priced = frame[frame["gap"].notna()] if "gap" in frame.columns else frame.iloc[0:0]
    if priced.empty:
        return pd.DataFrame(columns=columns)
    banded = priced.assign(band=merchant_client.band_of(priced["gap"]))
    grouped = (
        banded.groupby("band", observed=False)
        .agg(
            wines=("offer", "size"),
            spend=("spend", "sum"),
            clicks=("clicks", "sum"),
            bottles=("bottles", "sum"),
            revenue=("sold_revenue", "sum"),
        )
        .reset_index()
    )
    # Revenue for each dollar the band cost, blank where it cost nothing: a band
    # that took no spend has no return per dollar, and 0 would read as a band
    # that was advertised and sold nothing.
    grouped["per_dollar"] = (
        grouped["revenue"] / grouped["spend"].where(grouped["spend"] > 0)
    ).round(0)
    return grouped[columns]


def waste(frame: pd.DataFrame, gap: float = merchant_client.DEAR_GAP) -> pd.DataFrame:
    """Wines priced well above the market that took clicks and sold nothing.

    The list to act on in the campaign rather than in the shop: every one of
    them is a bottle Google charged for showing to somebody who then bought
    elsewhere, and none of them needs a merchant's agreement to stop.
    """
    if frame.empty or "gap" not in frame.columns:
        return frame
    return frame[
        (frame["gap"] > gap) & (frame["clicks"] > 0) & (frame["bottles"] <= 0)
    ].sort_values("spend", ascending=False)


def most_clicked(frame: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """The wines shoppers chose most, whatever their price did next.

    Clicks are the demand signal the shop does not own: a shopper who clicks
    has chosen this bottle from a row of competing ones, so a wine with clicks
    and no sales is a wine that was wanted and then not bought.
    """
    if frame.empty:
        return frame
    return frame.nlargest(limit, "clicks")


def sale_price_feed(
    frame: pd.DataFrame,
    gap: float = merchant_client.DEAR_GAP,
    limit: int | None = None,
) -> pd.DataFrame:
    """A supplemental feed that prices the worst offenders at the market.

    Merchant Center takes a ``sale_price`` per offer in a supplemental feed, so
    a price can be tested without the shop's own prices being rewritten and
    without a merchant having to agree first. The suggested price is the
    benchmark itself: the point of the test is to be at the market, not under
    it.

    Ordered by spend, so a shop that only wants to try twenty of them tries the
    twenty that are costing the most.
    """
    columns = ["id", "title", "price", "sale_price", "clicks", "spend"]
    if frame.empty or "gap" not in frame.columns:
        return pd.DataFrame(columns=columns)
    wanted = frame[(frame["gap"] > gap) & frame["benchmark"].notna()]
    if limit is not None:
        wanted = wanted.nlargest(limit, "spend")
    feed = pd.DataFrame(
        {
            "id": wanted["offer"],
            "title": wanted["title"],
            "price": wanted["price"].round(2),
            "sale_price": wanted["benchmark"].round(2),
            "clicks": wanted["clicks"],
            "spend": wanted["spend"].round(2),
        }
    )
    return feed.sort_values("spend", ascending=False).reset_index(drop=True)


def verdicts(frame: pd.DataFrame) -> list[str]:
    """What the ledger says, in the order an argument about it would go.

    Each line is a claim the panel puts a table under, so it can be checked
    rather than believed.
    """
    if frame.empty:
        return []
    total = float(frame["spend"].sum())
    if total <= 0:
        return []
    split = spend_split(frame)
    nothing = split[split["outcome"] == NOTHING]
    lines = []
    if not nothing.empty:
        wasted = float(nothing["spend"].iloc[0])
        wines = int(nothing["wines"].iloc[0])
        lines.append(
            f"**{wasted / total:.0%} of the ad spend went to {wines:,} "
            f"wine{'' if wines == 1 else 's'} that sold nothing** - "
            f"${wasted:,.0f} of ${total:,.0f}."
        )
    bands = by_band(frame)
    rated = bands[bands["per_dollar"].notna()]
    if len(rated) > 1:
        best = rated.loc[rated["per_dollar"].idxmax()]
        worst = rated.loc[rated["per_dollar"].idxmin()]
        if float(worst["per_dollar"]) > 0:
            lines.append(
                f"**A dollar spent on wines {str(best['band']).lower()} returned "
                f"${float(best['per_dollar']):,.0f}, against "
                f"${float(worst['per_dollar']):,.0f} on wines "
                f"{str(worst['band']).lower()}.** Revenue is every sale of those "
                "wines in the window, not sales the ads can be shown to have "
                "caused."
            )
    unpriced = frame[frame["gap"].isna()] if "gap" in frame.columns else frame.iloc[0:0]
    if not unpriced.empty:
        lines.append(
            f"{float(unpriced['spend'].sum()) / total:.0%} of the spend went to "
            f"{len(unpriced):,} offer{'' if len(unpriced) == 1 else 's'} Google "
            "publishes no benchmark for, so price says nothing about that part of "
            "the account either way."
        )
    return lines
