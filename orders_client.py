"""Read-only reader for the Medusa order book, straight out of Postgres.

The dashboard reports how the shop is doing beside how engineering is doing, so
it needs the order book - and only the order book. This used to page the CRM's
``/admin/orders`` endpoint, which costs ~16 sequential HTTP round trips and about
ten seconds for the year the dashboard looks back over, paid again on every cold
start. Medusa keeps that same data in the ``medusa`` schema of the shared
``private_dataset`` database, where the whole year is one query and ~250ms.

Every statement here is a SELECT, and the connection is opened read-only, because
the credential is a general application user rather than a reporting one: the
narrowness has to come from this module.

Two figures are derived rather than stored, because Medusa computes them at read
time rather than keeping a column:

* ``status`` is ``canceled`` once ``canceled_at`` is set, and the stored enum
  otherwise.
* ``payment_status`` is read from the order's payment collections and its
  summary. Neither alone is enough: a capture does not always reconcile into
  ``order_summary`` (leaving ``paid_total`` at 0 on an order whose payment row
  has a ``captured_at``), and a refund does not always land on the collection
  (leaving ``refunded_amount`` at 0 on an order whose summary knows about it).
  Each figure therefore takes whichever source reports more money moved.
"""

from __future__ import annotations

import datetime as _dt
import os
from contextlib import closing
from dataclasses import dataclass

import pandas as pd
import psycopg2

DEFAULT_HOST = "db.prod.vinovoss.private"
DEFAULT_DATABASE = "private_dataset"
DEFAULT_USER = "app__vinovoss_backend"
DEFAULT_PORT = 5432
# Medusa is a schema inside the shared database rather than a database of its own.
SCHEMA = "medusa"

_PASSWORD_ENV_VARS = ("MEDUSA_DB_PASSWORD", "POSTGRES_PASSWORD")

# Long enough to cover a cold read of a year of orders on a slow link, short
# enough that a wedged connection fails the section rather than hanging the tab.
_CONNECT_TIMEOUT_SECONDS = 10
_STATEMENT_TIMEOUT_MS = 30_000

# An order that was paid, including one since refunded: the refund is netted off
# the total rather than the order being struck out, so a wholly refunded sale
# contributes nothing and a half-refunded one contributes half.
PAID_PAYMENT_STATUSES = frozenset({"captured", "partially_refunded", "refunded"})

# Money is stored to more decimal places than it is charged in, so an amount
# captured and an order's total agree to the cent and not to the last digit.
_CENT = 0.01

COLUMNS = [
    "id",
    "display_id",
    "created_at",
    "updated_at",
    "status",
    "payment_status",
    "total",
    "refunded_total",
    "currency_code",
]

# Line-item columns: what was bought, how many, and for how much.
ITEM_COLUMNS = [
    "order_id",
    "item_id",
    "created_at",
    "status",
    "payment_status",
    "currency_code",
    "title",
    "product_handle",
    "quantity",
    "revenue",
    "refunded",
]

# Without any one of these the arithmetic is not merely incomplete, it is wrong:
# no total is $0 of revenue, no payment state is every order unpaid, no date is
# an empty window.
REQUIRED_FIELDS = ("created_at", "payment_status", "total")


@dataclass(frozen=True)
class OrderBook:
    """Orders and their line items, plus when they were last read.

    Two frames rather than one: the tiles count orders, the wine and merchant
    tables count what was in them, and a shared order may hold wine from more
    than one merchant.
    """

    orders: pd.DataFrame
    items: pd.DataFrame
    window_days: int
    synced_at: _dt.datetime


class MedusaConfigError(RuntimeError):
    """Raised when the order database is misconfigured or refuses the credential."""


@dataclass(frozen=True)
class DbConfig:
    """Where the order book is read from."""

    host: str
    database: str
    user: str
    password: str
    port: int

    @property
    def label(self) -> str:
        """Names the source without carrying the password into a cache key."""
        return f"{self.user}@{self.host}:{self.port}/{self.database}"


def load_medusa_env() -> DbConfig | None:
    """Return the order database's config from the environment, or ``None``.

    Only the password is required: the shop has one order book, and defaulting
    the rest to it means a deployment cannot half-configure its way into
    reporting a different shop's figures. The dev CRM keeps its orders in the
    same schema on a different host, so a wrong host reads plausible numbers
    for the wrong shop rather than failing.
    """
    password = ""
    for name in _PASSWORD_ENV_VARS:
        password = os.getenv(name, "").strip()
        if password:
            break
    if not password:
        return None
    port_text = os.getenv("MEDUSA_DB_PORT", "").strip()
    try:
        port = int(port_text) if port_text else DEFAULT_PORT
    except ValueError as exc:
        raise MedusaConfigError(
            f"MEDUSA_DB_PORT must be a port number, not {port_text!r}."
        ) from exc
    return DbConfig(
        host=os.getenv("MEDUSA_DB_HOST", "").strip()
        or os.getenv("POSTGRES_HOST", "").strip().strip('"')
        or DEFAULT_HOST,
        database=os.getenv("MEDUSA_DB_NAME", "").strip()
        or os.getenv("POSTGRES_DATABASE", "").strip().strip('"')
        or DEFAULT_DATABASE,
        user=os.getenv("MEDUSA_DB_USER", "").strip()
        or os.getenv("POSTGRES_USER", "").strip().strip('"')
        or DEFAULT_USER,
        password=password,
        port=port,
    )


def _connect(config: DbConfig):
    """A read-only connection, or a message naming what to fix."""
    try:
        connection = psycopg2.connect(
            host=config.host,
            port=config.port,
            dbname=config.database,
            user=config.user,
            password=config.password,
            connect_timeout=_CONNECT_TIMEOUT_SECONDS,
            options=f"-c statement_timeout={_STATEMENT_TIMEOUT_MS}",
        )
    except psycopg2.OperationalError as exc:
        raise MedusaConfigError(
            f"Could not reach the order database at {config.label}: "
            f"{str(exc).strip()} "
            "The host is private to the VPC, so a Cloud Run revision needs "
            "direct VPC egress (--network=default --subnet=default) to see it."
        ) from exc
    # Belt and braces over a credential that is allowed to write: a bug here
    # cannot become an UPDATE against the shop's order book.
    connection.set_session(readonly=True, autocommit=True)
    return connection


def _frame(connection, sql: str, params: dict) -> pd.DataFrame:
    """One query's rows as a frame, with the column names the cursor reports.

    Not ``pandas.read_sql``: it supports SQLAlchemy connectables and sqlite3
    only, and warns on every call when handed a psycopg2 connection. Going
    through the cursor is what it would do anyway, minus a dependency and the
    warning.
    """
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [column.name for column in cursor.description]
        rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=columns)


# The order-level figures, as one named query both reads below build on: the
# items query needs an order's date and payment state on every line, and
# restating that CASE would let the two drift apart.
_ORDER_BOOK_CTE = f"""
with collections as (
    select opc.order_id,
           count(*) filter (where c.status <> 'canceled')                          as live,
           sum(c.amount) filter (where c.status <> 'canceled')                      as amount,
           sum(coalesce(c.authorized_amount, 0))
               filter (where c.status <> 'canceled')                               as authorized,
           sum(coalesce(c.captured_amount, 0))
               filter (where c.status <> 'canceled')                               as captured,
           sum(coalesce(c.refunded_amount, 0))
               filter (where c.status <> 'canceled')                               as refunded,
           bool_or(c.status = 'awaiting')                                          as any_awaiting
    from {SCHEMA}.order_payment_collection opc
    join {SCHEMA}.payment_collection c
      on c.id = opc.payment_collection_id
     and c.deleted_at is null
    where opc.deleted_at is null
    group by opc.order_id
),
book as (
    select o.id,
           o.display_id,
           o.created_at,
           o.updated_at,
           o.currency_code,
           o.version,
           case
               when o.canceled_at is not null then 'canceled'
               else o.status::text
           end                                                as status,
           t.total,
           -- Whichever of the summary and the payment collections reports more
           -- money moved; either can lag the other.
           greatest(coalesce(t.paid, 0), coalesce(c.captured, 0))     as captured,
           greatest(coalesce(t.refunded, 0), coalesce(c.refunded, 0)) as refunded_total,
           c.order_id  as has_collections,
           c.live,
           c.amount    as collections_amount,
           c.authorized,
           c.any_awaiting
    from {SCHEMA}."order" o
    -- The summary is versioned and an edited order keeps every version; the
    -- newest is the order as it stands.
    join lateral (
        select (s.totals->>'current_order_total')::numeric as total,
               (s.totals->>'paid_total')::numeric          as paid,
               (s.totals->>'refunded_total')::numeric      as refunded
        from {SCHEMA}.order_summary s
        where s.order_id = o.id
          and s.deleted_at is null
        order by s.version desc
        limit 1
    ) t on true
    left join collections c on c.order_id = o.id
    where o.deleted_at is null
      -- A draft is a quote the shop is drawing up, not a sale.
      and o.is_draft_order = false
      and o.created_at >= %(since)s
),
priced as (
    select b.*,
           case
               when b.refunded_total > 0
                    and b.refunded_total >= b.captured - {_CENT} then 'refunded'
               when b.refunded_total > 0                         then 'partially_refunded'
               when b.captured > 0
                    and b.captured >= b.total - {_CENT}          then 'captured'
               when b.captured > 0                               then 'partially_captured'
               when b.has_collections is null                    then 'not_paid'
               when b.live = 0                                   then 'canceled'
               when b.authorized > 0
                    and b.authorized
                        >= b.collections_amount - {_CENT}        then 'authorized'
               when b.authorized > 0                             then 'partially_authorized'
               when b.any_awaiting                               then 'awaiting'
               else 'not_paid'
           end as payment_status
    from book b
)
"""

_ORDERS_SQL = (
    _ORDER_BOOK_CTE
    + """
select id,
       display_id,
       created_at,
       updated_at,
       status,
       payment_status,
       total,
       refunded_total,
       currency_code
from priced
order by created_at
"""
)

_ITEMS_SQL = (
    _ORDER_BOOK_CTE
    + f"""
select p.id                                                as order_id,
       oli.id                                              as item_id,
       p.created_at,
       p.status,
       p.payment_status,
       p.currency_code,
       coalesce(nullif(oli.product_title, ''), oli.title)   as title,
       coalesce(oli.product_handle, '')                     as product_handle,
       oi.quantity,
       -- The line item holds the list price; the order item holds what this
       -- order was actually charged, which a bulk discount makes lower.
       oi.quantity * coalesce(oi.unit_price, oli.unit_price) as revenue,
       p.refunded_total
from priced p
-- An edited order keeps the line items of every version it has had; the order's
-- own version says which set it is now.
join {SCHEMA}.order_item oi
  on oi.order_id = p.id
 and oi.version = p.version
 and oi.deleted_at is null
join {SCHEMA}.order_line_item oli
  on oli.id = oi.item_id
order by p.created_at
"""
)

_STORES_SQL = f"""
select name,
       metadata->>'store_prefix' as prefix
from {SCHEMA}.store
where deleted_at is null
"""


def _normalise_orders(frame: pd.DataFrame) -> pd.DataFrame:
    """Types and casing the metrics rely on, and a refusal if a figure is absent."""
    # Every figure is defined by these three. Filling a missing one in would not
    # give a smaller answer, it would give a wrong one - a confident $0 for a
    # shop that took money - so the section refuses instead.
    missing = [name for name in REQUIRED_FIELDS if name not in frame.columns]
    if missing:
        raise MedusaConfigError(
            f"The order database returned orders with no {', '.join(missing)}, so "
            "the order figures cannot be trusted. Check the Medusa schema version."
        )
    for column in ("created_at", "updated_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    for column in ("total", "refunded_total"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in ("status", "payment_status", "currency_code"):
        frame[column] = frame[column].fillna("").astype(str).str.strip().str.lower()
    return frame[COLUMNS].sort_values("created_at").reset_index(drop=True)


def _share_refunds(items: pd.DataFrame) -> pd.Series:
    """Spread each order's refund across its lines, in proportion to their value.

    A refund is recorded against the order, not against the bottle that went
    back, so the merchant who sold it cannot be known. Splitting it by value at
    least keeps a merchant's revenue from counting money the customer got back,
    and keeps the per-merchant total reconcilable with the revenue tile.
    """
    gross = items.groupby("order_id")["revenue"].transform("sum")
    refunded = items["refunded_total"]
    # A refund on an order whose lines are worth nothing has nowhere to go, and
    # dividing by that zero would put a NaN into every merchant's revenue.
    share = (refunded * items["revenue"] / gross).where(gross > 0, 0.0)
    return share.clip(lower=0.0, upper=items["revenue"]).round(2)


def _normalise_items(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per line item, carrying its order's date and payment state.

    Denormalised on purpose: every question asked of the items - what sold, whose
    wine it was, what it earned - is also filtered by when the order was placed
    and whether it stood, and a per-item frame answers those without a join.
    """
    if frame.empty:
        return pd.DataFrame(columns=ITEM_COLUMNS)
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    frame["quantity"] = (
        pd.to_numeric(frame["quantity"], errors="coerce").fillna(0).astype(int)
    )
    for column in ("revenue", "refunded_total"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["revenue"] = frame["revenue"].round(2)
    for column in ("status", "payment_status", "currency_code", "product_handle"):
        frame[column] = frame[column].fillna("").astype(str).str.strip().str.lower()
    frame["title"] = frame["title"].fillna("").astype(str).str.strip()
    frame["refunded"] = _share_refunds(frame)
    return frame[ITEM_COLUMNS].reset_index(drop=True)


def read_order_book(
    config: DbConfig,
    days: int,
    now: _dt.datetime | None = None,
) -> OrderBook:
    """The last ``days`` days of orders and their line items, in one read.

    Read whole rather than topped up incrementally: a year of orders is a single
    ~250ms query, so keeping a book and grafting changed orders onto it would be
    complexity bought with the risk of a stale row.
    """
    read_at = now or _dt.datetime.now(_dt.timezone.utc)
    since = read_at - _dt.timedelta(days=days)
    params = {"since": since}
    try:
        # `closing`, not a bare `with`: psycopg2's context manager ends the
        # transaction and leaves the socket open, so a bare `with` would leak a
        # connection on every refresh until the server ran out of them.
        with closing(_connect(config)) as connection:
            orders = _frame(connection, _ORDERS_SQL, params)
            items = _frame(connection, _ITEMS_SQL, params)
    except psycopg2.Error as exc:
        raise MedusaConfigError(
            f"The order database at {config.label} refused the read: "
            f"{str(exc).strip()}"
        ) from exc
    return OrderBook(
        orders=_normalise_orders(orders),
        items=_normalise_items(items),
        window_days=days,
        synced_at=read_at,
    )


# Prefixes the CRM no longer registers but which are still on wine it has sold.
# TheWinesGood's store record says `thewinesgood`, its products say
# `thewinegood` - a letter's difference that would otherwise leave a year of its
# sales credited to nobody. Confirmed as the same shop by Angel.
_DEFAULT_PREFIX_ALIASES = {"thewinegood": "TheWinesGood"}


def _prefix_aliases() -> dict[str, str]:
    """Retired handle prefixes mapped to the merchant they belonged to.

    A merchant that changed its prefix keeps the old one on everything it has
    already sold, and the store record only remembers the current one, so that
    history would otherwise be attributed to nobody. Add more with
    ``MEDUSA_STORE_PREFIX_ALIASES="oldprefix=Store Name,other=Other Store"``.
    """
    aliases: dict[str, str] = dict(_DEFAULT_PREFIX_ALIASES)
    for entry in os.getenv("MEDUSA_STORE_PREFIX_ALIASES", "").split(","):
        prefix, _, name = entry.partition("=")
        if prefix.strip() and name.strip():
            aliases[prefix.strip().lower()] = name.strip()
    return aliases


def fetch_stores(config: DbConfig) -> dict[str, str]:
    """Each merchant's handle prefix mapped to its name.

    Every product a merchant lists is slugged with that prefix, and the order
    line keeps the product handle, so the prefix is what ties a bottle sold back
    to the shop that sold it.
    """
    try:
        with closing(_connect(config)) as connection:
            with connection.cursor() as cursor:
                cursor.execute(_STORES_SQL)
                rows = cursor.fetchall()
    except psycopg2.Error as exc:
        raise MedusaConfigError(
            f"The order database at {config.label} refused the store list: "
            f"{str(exc).strip()}"
        ) from exc
    prefixes: dict[str, str] = {}
    for name, prefix in rows:
        prefix = str(prefix or "").strip().lower()
        name = str(name or "").strip()
        if prefix and name:
            prefixes[prefix] = name
    # Configured aliases lose to a live store prefix: a prefix the CRM is using
    # today belongs to whoever it says, not to a stale setting.
    return {**_prefix_aliases(), **prefixes}


__all__ = [
    "COLUMNS",
    "ITEM_COLUMNS",
    "REQUIRED_FIELDS",
    "SCHEMA",
    "DbConfig",
    "MedusaConfigError",
    "OrderBook",
    "PAID_PAYMENT_STATUSES",
    "fetch_stores",
    "load_medusa_env",
    "read_order_book",
]
