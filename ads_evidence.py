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

# What each claim is about, so the wines behind it are chosen by which claim it
# is rather than by where it happened to land in the list.
WASTED = "sold-nothing"
BY_PRICE = "by-price"
NO_BENCHMARK = "no-benchmark"

# Google bills an ad account in its own currency and the shop's feed prices in
# its own, so a sentence about both carries two symbols rather than one assumed.
# Written out here because these sentences are also the printable report, where
# there is no caption above them to fall back on.
_SYMBOLS = {"usd": "$", "eur": "\u20ac", "gbp": "\u00a3"}


def _sum(amount: float, currency: str) -> str:
    """A round money figure, labelled with the currency it is actually in."""
    symbol = _SYMBOLS.get(currency.lower(), "")
    return f"{symbol}{amount:,.0f}" + (f" {currency.upper()}" if not symbol else "")


def _rate(amount: float, currency: str) -> str:
    """A return per unit spent, to the cent while it is small enough to matter."""
    symbol = _SYMBOLS.get(currency.lower(), "")
    figure = f"{amount:,.0f}" if amount >= 10 else f"{amount:,.2f}"
    return f"{symbol}{figure}" + (f" {currency.upper()}" if not symbol else "")


def sold_known(frame: pd.DataFrame) -> bool:
    """Whether this ledger knows what each wine sold, or merely has no figure.

    An order book that could not be read and a wine nobody bought both arrive as
    an absence, and only one of them is a fact about the wine: every claim about
    what the money bought is withheld unless the sales were read.
    """
    if frame.empty or "bottles" not in frame.columns:
        return False
    return bool(frame["bottles"].notna().all())


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
    if not sales.read or not sales.measured_against(ads[["offer"]]):
        # Unread, or read and matched not one advertised offer: neither of those
        # is unsold. ``Sales.against`` fills a wine with no sale with a zero,
        # which is right for a bottle nobody bought and wrong for a database
        # that was down or a set of offer ids the order book has never heard of -
        # left as zeroes either would have this panel announce that every
        # advertised bottle was wasted money, on our own failed join.
        frame = frame.assign(bottles=pd.NA, sold_revenue=pd.NA)
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
    if not sold_known(frame):
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


def split_hovers(split: pd.DataFrame, spent: str = "usd") -> list[str]:
    """What each slice of the spend ring says when it is pointed at.

    Whole sentences, one per slice, rather than a template over ``customdata``:
    the money has to be written here anyway to carry the currency Google billed,
    and a two-column array of a number and a string read back through Plotly's
    own number format is how that tooltip came to say ``- across NaN wines``.
    """
    return [
        f"{row.outcome}: {_sum(float(row.spend), spent)} across "
        f"{int(row.wines):,} wine" + ("" if int(row.wines) == 1 else "s")
        for row in split.itertuples()
    ]


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
    # Kept to the cent rather than the unit: this column is also the test for
    # whether a band returned anything at all, and a band returning forty cents
    # a dollar rounded to the unit is a band reported as having sold nothing.
    if not sold_known(frame):
        # A band still has wines, spend and clicks in it without the order book.
        # What it gave back is the column that would be invented.
        grouped[["bottles", "revenue", "per_dollar"]] = pd.NA
    else:
        grouped["per_dollar"] = (
            grouped["revenue"] / grouped["spend"].where(grouped["spend"] > 0)
        ).round(2)
    return grouped[columns]


def waste(frame: pd.DataFrame, gap: float = merchant_client.DEAR_GAP) -> pd.DataFrame:
    """Wines priced well above the market that took clicks and sold nothing.

    The list to act on in the campaign rather than in the shop: every one of
    them is a bottle Google charged for showing to somebody who then bought
    elsewhere, and none of them needs a merchant's agreement to stop.
    """
    if frame.empty or "gap" not in frame.columns or not sold_known(frame):
        return frame.iloc[0:0]
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
    # A wine nobody clicked is not one of the most clicked, however few there
    # are: padding the table out with zeroes would have the caption above it
    # calling unwanted bottles the ones shoppers chose.
    return frame[frame["clicks"] > 0].nlargest(limit, "clicks")


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


def verdicts(
    frame: pd.DataFrame, spent: str = "usd", money: str = ""
) -> list[tuple[str, str]]:
    """What the ledger says, in the order an argument about it would go.

    Each claim is a line the panel puts a table under, so it can be checked
    rather than believed, and each carries the name of the wines it is about:
    any of them can be left out - a shut order book takes the first two, a feed
    Google benchmarks entirely takes the last - and a claim paired with the
    wines that happened to be in its position would be a claim about somebody
    else's wines.
    """
    if frame.empty:
        return []
    total = float(frame["spend"].sum())
    if total <= 0:
        return []
    money = money or spent
    split = spend_split(frame)
    nothing = split[split["outcome"] == NOTHING]
    claims: list[tuple[str, str]] = []
    # ``spend_split`` always draws both outcomes, so a window in which every
    # advertised wine sold arrives here as a row of zeroes: claiming 0% of the
    # spend went to 0 wines is the panel talking about nothing.
    if not nothing.empty and int(nothing["wines"].iloc[0]) > 0:
        wasted = float(nothing["spend"].iloc[0])
        wines = int(nothing["wines"].iloc[0])
        claims.append(
            (
                WASTED,
                f"**{wasted / total:.0%} of the ad spend went to {wines:,} "
                f"wine{'' if wines == 1 else 's'} that sold nothing** - "
                f"{_sum(wasted, spent)} of {_sum(total, spent)}.",
            )
        )
    bands = by_band(frame)
    rated = bands[bands["per_dollar"].notna()]
    if len(rated) > 1:
        best = rated.loc[rated["per_dollar"].idxmax()]
        worst = rated.loc[rated["per_dollar"].idxmin()]
        if float(worst["per_dollar"]) > 0:
            claims.append(
                (
                    BY_PRICE,
                    f"**{_sum(1, spent)} spent on wines "
                    f"{str(best['band']).lower()} returned "
                    f"{_rate(float(best['per_dollar']), money)}, against "
                    f"{_rate(float(worst['per_dollar']), money)} on wines "
                    f"{str(worst['band']).lower()}.** Revenue is every sale of "
                    "those wines in the window, not sales the ads can be shown "
                    "to have caused.",
                )
            )
    unpriced = frame[frame["gap"].isna()] if "gap" in frame.columns else frame.iloc[0:0]
    if not unpriced.empty:
        claims.append(
            (
                NO_BENCHMARK,
                f"{float(unpriced['spend'].sum()) / total:.0%} of the spend went "
                f"to {len(unpriced):,} offer{'' if len(unpriced) == 1 else 's'} "
                "Google publishes no benchmark for, so price says nothing about "
                "that part of the account either way.",
            )
        )
    return claims


def advice(frame: pd.DataFrame, spent: str = "usd", money: str = "") -> list[str]:
    """What to do about all of it, largest lever first.

    Evidence is not a decision, and the person who reads this runs the campaign:
    he will fairly ask what he is meant to change on Monday. Each line is sized
    from this ledger rather than from advertising lore, so a lever worth nothing
    on this account is left out instead of being recommended because it usually
    matters somewhere.
    """
    if frame.empty or float(frame["spend"].sum()) <= 0:
        return []
    total = float(frame["spend"].sum())
    money = money or spent
    said: list[str] = []
    split = spend_split(frame)
    sold = split[split["outcome"] == SOLD]
    nothing = split[split["outcome"] == NOTHING]
    if (
        not sold.empty
        and not nothing.empty
        and float(sold["spend"].iloc[0]) > 0
        and int(nothing["wines"].iloc[0]) > 0
    ):
        said.append(
            f"**Put the budget behind the {int(sold['wines'].iloc[0]):,} wines "
            f"that sold, and away from the {int(nothing['wines'].iloc[0]):,} "
            "that did not.** The sellers took "
            f"{_sum(float(sold['spend'].iloc[0]), spent)} of the "
            f"{_sum(total, spent)} and stood beside "
            f"{_sum(float(sold['revenue'].iloc[0]), money)} of revenue; the rest "
            f"took {_sum(float(nothing['spend'].iloc[0]), spent)} and stood "
            "beside nothing. "
            "This is the only lever here worth a project - bids and listing "
            "groups that favour proven bottles move more money than any pruning "
            "below."
        )
    stop = waste(frame)
    if not stop.empty:
        said.append(
            f"**Exclude the {len(stop):,} clicked, expensive, unsold wines from "
            f"the campaign - {_sum(float(stop['spend'].sum()), spent)} over "
            f"{merchant_client.SALES_DAYS} days.** In the campaign, not in "
            "Merchant Center: delisting an offer loses its free Shopping listing "
            "too, while Shopping charges per click, so an exclusion saves the "
            "money from the day it is made."
        )
    bands = by_band(frame)
    rated = bands[bands["per_dollar"].notna()]
    if len(rated) > 1:
        best = rated.loc[rated["per_dollar"].idxmax()]
        worst = rated.loc[rated["per_dollar"].idxmin()]
        if float(worst["per_dollar"]) > 0:
            said.append(
                f"**Reprice rather than only re-bid.** {_sum(1, spent)} on wines "
                f"{str(best['band']).lower()} came back "
                f"{float(best['per_dollar']) / float(worst['per_dollar']):.0f}x "
                f"what it did on wines {str(worst['band']).lower()}, so price is "
                "doing more to the return than bidding is. The sale-price tab "
                "makes a supplemental feed that tests exactly that without a "
                "merchant having to agree first."
            )
    unpriced = frame[frame["gap"].isna()] if "gap" in frame.columns else frame.iloc[0:0]
    # Gated on the order book like every other lever here: this one points the
    # reader at what those bottles sold, which is the one column an unread order
    # book has left blank, so ungated it advises reading a column it withheld.
    if (
        sold_known(frame)
        and not unpriced.empty
        and float(unpriced["spend"].sum()) / total > _MOSTLY_UNPRICED
    ):
        said.append(
            f"**Judge the {float(unpriced['spend'].sum()) / total:.0%} with no "
            "benchmark on what it sold, not on its price.** Nobody else lists "
            "those bottles, so there is no market price to be above: sales are "
            "the only test they can be put to, and they are in the ledger."
        )
    if not sold_known(frame):
        said.append(
            "**Read the shop's own orders before acting on any of this.** "
            "Google's conversion tracking records a fraction of the shop's "
            "orders, so a wine that looks unsold to the campaign may well have "
            "sold; the sales here are the shop's own, and today they could not "
            "be read."
        )
    return said


# How much of the spend has to be on unbenchmarked bottles before saying so is
# advice rather than a footnote.
_MOSTLY_UNPRICED = 0.2
