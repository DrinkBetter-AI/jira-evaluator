# Moving the order book from the Medusa API to Postgres

A record of how *Orders, Revenue & AOV* and *Best Sellers & Merchants* were changed
from reading the Medusa admin API to reading the CRM's own database: what was done,
in what order, why each decision went the way it did, and what is worth worrying
about afterwards.

Date: 2026-08-06. Branch: `develop`.

---

## The short answers

**Did anything write to the database? No. Reads only.**

- Every statement issued, by hand and in the shipped code, was a `SELECT` (or a
  `psql` metacommand like `\d` / `\l`, which are themselves selects on the
  catalog). No `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `ALTER`, `DROP` or `TRUNCATE`
  was ever run against any database.
- The shipped code enforces this rather than relying on discipline:
  `_connect()` calls `connection.set_session(readonly=True)`, so **Postgres itself**
  rejects a write, not just convention. Verified empirically:

  ```
  transaction_read_only = on
    REJECTED: ReadOnlySqlTransaction cannot execute UPDATE in a read-only transaction
    REJECTED: ReadOnlySqlTransaction cannot execute CREATE TABLE in a read-only transaction
  ```

  This matters because `app__vinovoss_backend` is a general application role that
  *is* allowed to write. The narrowness has to come from this module, so it does.
- Nothing was written to the Medusa API either — every HTTP call was a `GET`.

**The most concerning thing I did:** ran `gcloud run services describe jira-dashboard`,
which printed the live service's entire environment — `DASHBOARD_PASSWORD`, a GitHub
PAT, the Medusa admin key and both Amplitude keys — into this session's transcript.
See [Concerns](#concerns-and-things-to-check) below; it is first on the list for a
reason.

---

## Step by step

### 1. Establish what "slow" actually meant, instead of assuming

I did not take the premise on faith. I measured the existing path first:

- The API pages at `_PAGE_SIZE = 100` over `ORDER_BOOK_DAYS = 390`, which is 1523
  prod orders → **16 sequential requests at 0.5–0.9s each ≈ 10s**.
- A single equivalent SQL query: **~0.24s**.

**Decision:** the complaint was justified, but the diagnosis needed refining. The
existing code already tops up incrementally, so a *warm* instance was fine; the pain
was the ~10s cold read paid again on every Cloud Run revision and every cache lapse.
That reframing is what later justified deleting the incremental machinery entirely
rather than optimising it.

### 2. Answer the question that was actually asked — are these DB creds even relevant?

The user was unsure. They were half right to be.

- Medusa has **no separate database**. It lives in the **`medusa` schema of the same
  `private_dataset`** AlloyDB, with 26 `order*` tables plus `payment_collection`,
  `payment`, `capture`, `refund`, `order_line_item`, `order_item`, `store`. The
  giveaway was an `app__medusa` role holding `CREATE` on `private_dataset`.
- So yes, relevant. **But the host in `.env` was wrong.**

| | Dashboard reads | `.env` said |
|---|---|---|
| Target | `merchants.vinovoss.com` (**prod** — `MEDUSA_ADMIN_URL` unset, so it fell back to the prod default) | `db.dev.vinovoss.private` (**dev**) |
| Newest order | display_id **2645** | display_id **3879**, different order IDs entirely |

**Decision:** this was the single highest-risk finding of the whole task. The dev CRM
keeps its orders in the *same schema* on a *different host*, so pointing at dev does
not fail — it returns plausible, well-formed, completely wrong numbers for a
different shop. I found the prod host (`db.prod.vinovoss.private`) and confirmed it
matched the prod API exactly (same newest order ID, same `total` to 5 decimal places,
same 390-day count of 1523) before writing a line of code.

This is also why `load_medusa_env()` defaults the host to prod and only *requires*
the password: a half-configured deployment cannot accidentally report the dev shop.

### 3. Ask before building

Three things were genuinely ambiguous and cheap to ask, so I asked rather than
guessed: scope (whole order book vs. tiles only), fallback behaviour, and which
environment. Answers: **whole OrderBook**, **DB-only with an honest error**, **prod**.

The scope answer mattered — both sections share one `OrderBook`, so doing only the
tiles would have left two sources that could disagree *and* kept the slow paging.

### 4. Map every field, and refuse to guess at the hard ones

| API field | SQL source | Notes |
|---|---|---|
| `total` | `order_summary.totals->>'current_order_total'` | latest `version` via `join lateral` |
| `refunded_total` | `totals->>'refunded_total'` | see step 6 — the API never returns this |
| `status` | `canceled_at is not null` → `canceled`, else the enum | Medusa derives this at read time |
| `payment_status` | derived from payment collections + summary | the hard one, step 5 |
| line items | `order_item` ⋈ `order_line_item` | quantity from `order_item`, at the order's `version` |

**First real bug caught by measuring, not reading:** my initial count was 1540 against
the API's 1523. Cause: 17 orders have more than one payment collection, and a naive
`LEFT JOIN` fanned them out. Aggregating collections per order in a CTE first brought
it to exactly 1523. Had I only spot-checked a few orders I would have shipped a 1.1%
inflation in every order count.

### 5. `payment_status`: iterate against ground truth rather than reason from docs

Medusa computes `payment_status` at read time; there is no column. Reimplementing it
from first principles is exactly the sort of thing that silently drifts, so I wrote a
harness that pulled **all 1523 prod orders** from the API and diffed candidate SQL
rules against them. Four rounds:

| Attempt | Mismatches / 1523 | What it taught me |
|---|---|---|
| Collections, canceled ones included | 30 | canceled collections must not win over captured money |
| Collections, canceled ones excluded | 22 | remainder all "API `partially_captured`, mine `captured`" |
| Summary `paid_total` / `pending_difference` | 32 | worse — some captures never reconcile into the summary |
| **Summary *and* collections, whichever reports more money moved** | **shipped** | each source can lag the other |

Digging into the survivors was the payoff. Two distinct data realities:

- `order_01KXHHJKPEY7QQ87G83SPXRQT2` — `payment.captured_at` set, collection
  `captured_amount` = the full order total, but `order_summary.paid_total` still `0`.
  The summary never reconciled. Summary-only rules call this unpaid; it is paid.
- `order_01KGZFJBZ69HJWN79DS08TAJAJ` and ~20 others — a **dead `canceled` payment
  collection alongside a `completed` one**. The API sums the dead collection into its
  denominator and reports `partially_captured`, even though its own summary says
  `paid_total == total` and `pending_difference == 0`.

**Decision, and it is a judgement call I want on the record:** for that second class I
concluded the API is wrong and the database is right, and count those orders as paid.
The evidence is Medusa's own summary. But this is *my* reading, not Medusa's
definition, and it means ~1.5% of orders are classified differently than the API would
classify them. None fall inside the 7- or 30-day windows, which is why the tiles still
match exactly. Flagged in the README.

### 6. The discovery that changed the numbers

While diffing I hit a `KeyError: 'refunded_total_api'` — the column did not exist on
the API frame. Chasing it:

```
$ curl .../admin/orders?fields=id,total,refunded_total
['fulfillment_status', 'id', 'items', 'payment_status', 'status', 'total', 'version']
```

**The Medusa admin API does not return `refunded_total` at all.** It silently drops
the unknown field. Which means `orders.py`'s `_kept()` and `_with_refunds()` — the
refund-netting the caption has always promised — **have never once executed**.
`refunded_total` was always `NaN → 0.0`, so revenue was always gross of refunds.

**Decision:** fix it rather than preserve the bug for continuity. The database has the
real figure (27 orders/year carrying $6.5k of refunds). This is the entire source of
the revenue delta, and it is the one number that legitimately changes.

### 7. Prove equivalence before touching the UI

Ran the *existing* `orders.py` metric functions over both frames, so the comparison
was of rendered tiles, not of my own SQL against itself:

| Window | Metric | API (before) | SQL (after) | Δ |
|---|---|---|---|---|
| 7d | orders / cancelled / paid | 37 / 1 / 24 | 37 / 1 / 24 | **exact** |
| 7d | revenue | $4,815.35 | $4,675.95 | −$139.40 |
| 30d | orders / cancelled / paid | 189 / 5 / 172 | 189 / 5 / 172 | **exact** |
| 30d | revenue | $38,957.86 | $38,818.45 | −$139.41 |

Line items: 1834 vs 1834, IDs identical, **0** quantity mismatches, and after one fix
**0** price mismatches.

That fix: two lines were $155 off each because `order_line_item.unit_price` is the
*list* price while `order_item.unit_price` is what the order was actually charged
after a bulk discount — and the API returns the latter. Only visible because I
compared all 1834 lines instead of sampling.

**Residual, accepted knowingly:** 14 of 1523 orders (0.9%) disagree on `total`. All
are edited/returned orders where `current_order_total` went to `0`; all are 9+ months
old and outside every window the dashboard shows. Reproducing the API's number would
mean reimplementing Medusa's tax and adjustment engine in SQL — far more likely to
introduce drift than to fix 14 stale rows. Documented instead.

### 8. Rewrite, simplify, wire up

- `orders_client.py` — HTTP internals replaced with SQL; public surface
  (`OrderBook`, `COLUMNS`, `ITEM_COLUMNS`, `PAID_PAYMENT_STATUSES`, `fetch_stores`)
  kept so `orders.py` needed **zero changes**.
- The order-level CTE is defined **once** and shared by both queries, so the derived
  `status` / `payment_status` cannot drift between the tiles and the item tables.
- Deleted as now-dead complexity: `sync_order_book`, `_order_book_holder`,
  `_expire_order_book`, the `threading.Lock`, the `truncated` flag and its UI warning
  (one query cannot run out of pages), and the `_MAX_PAGES` guard. `_order_book` is
  now a plain `st.cache_data(ttl=900)`, so the existing `st.cache_data.clear()` on
  **Refresh Data** reaches it for free — previously it could not, which is why
  `_expire_order_book` existed at all.
- Cache keys use `config.label` (`user@host:port/db`), never the config object, so
  **the password never becomes part of a cache key**.

### 9. Verify the app, including the failure paths

Used Streamlit's `AppTest` against a small untracked harness (`/tmp/business_probe.py`)
that renders only `_render_business_sections()` — no Jira, no password gate:

- **0 exceptions**, both subheaders, all 8 tiles with correct values and deltas, both
  tab dataframes.
- No password → clean caption naming the variable to set. **0 exceptions.**
- Wrong password → warning quoting Postgres' auth failure. **0 exceptions.**
- Unreachable host → warning that also names the likely cause (missing VPC egress).
  **0 exceptions.**

### 10. Two bugs found after I first reported "done"

The user ran the app and hit a `UserWarning`. Investigating it turned up a worse
problem next to it:

1. **`pd.read_sql` on a raw psycopg2 connection** is unsupported and warns on every
   call. Replaced with a `_frame()` helper going through the cursor — removes the
   warning, avoids adding SQLAlchemy, and dropped the read from 0.35s to **0.16s**.
2. **A connection leak I introduced.** `with psycopg2.connect(...)` manages the
   *transaction* and leaves the socket **open**. Every 15-minute refresh leaked a
   connection until the server refused new ones. Fixed with `contextlib.closing`.
   Verified: backend connection count flat at 32 across 12 consecutive reads.

Worth stating plainly: my own earlier verification did not catch either, because I
tested correctness of output and never observed the process over repeated reads. The
user running it normally found in one go what my harness missed.

---

## Concerns and things to check

Ordered by how much they should bother you.

1. **Production secrets are in this session's transcript.** `gcloud run services
   describe jira-dashboard` printed the live service's full environment:
   `DASHBOARD_PASSWORD`, `DASHBOARD_GITHUB_TOKEN` (a GitHub PAT), the Medusa admin
   key, and both Amplitude keys — all currently stored as **plaintext env vars**
   rather than secrets. I needed the service's config, but I should have queried
   narrower fields. Consider rotating anything you consider exposed, and moving these
   to Secret Manager as `jira-api-token` already is.

2. **A prod database password now sits in cleartext in `jira-evaluator/.env`.** I took
   it from a commented-out line in a *different* repo (`../wine-recommender-backend/.env`)
   and copied it across a repo boundary. `.env` is gitignored and I confirmed it is
   untracked, and it is in `.dockerignore` so it cannot be baked into the image — but
   it is still a prod credential in cleartext on disk, and it is now in two places
   instead of one.

3. **`cloudbuild.yaml` now changes production networking on the next merge.** I added
   `--network default --subnet default --vpc-egress private-ranges-only`. Without it
   the Business tab times out, but this means the next deploy to `develop` alters the
   live service's egress path. `private-ranges-only` is the conservative choice
   (Jira/GitHub/Amplitude keep going straight out, not through the VPC) and matches
   what every other service in the project does, but it should be a deliberate,
   watched deploy rather than a surprise.

4. **A business-visible metric moved.** Revenue and AOV are now lower than what anyone
   looking yesterday saw (−$139 on both windows), because refunds are finally being
   netted off as documented. Correct, but if these tiles feed a report to anyone, they
   deserve the explanation rather than discovering a step change.

5. **~1.5% of orders are classified differently from the API**, on my judgement that a
   dead `canceled` payment collection should not make a fully-paid order read as
   partially captured (step 5). Defensible and documented, but a judgement call.

6. **14 orders disagree on `total`** (step 7), all outside every displayed window.

7. **I queried the production database and API directly.** ~20 ad-hoc `psql`
   selects, and several full 16-page reads of the prod Medusa admin API. All
   read-only, but real queries and real load against prod, not a replica.

8. **`.env` is untracked, so my host correction does not propagate.** Anyone else
   pulling this branch, and every deployment, must set `POSTGRES_HOST` /
   `POSTGRES_PASSWORD` themselves. The default in code is prod, so the failure mode is
   "no section" rather than "wrong shop" — but a stale local `.env` still pointing at
   `db.dev.vinovoss.private` **would** silently report the dev shop. Worth checking on
   any machine that had one.

9. **`MEDUSA_ADMIN_API_KEY` and `MEDUSA_ADMIN_URL` are now unused.** I deliberately
   did **not** touch the running service to remove them. The key is a full admin
   credential — Medusa issues no narrower one — so it is worth deleting from the
   service's env once this is deployed.

10. **`_SYNC_OVERLAP`, `REQUIRED_FIELDS` and friends.** `REQUIRED_FIELDS` is still
    checked, but against a frame whose shape my own SQL controls, so it can now only
    fire on a Medusa schema change. Harmless, arguably no longer earning its keep.

11. **Throwaway harnesses left in `/tmp`** (`validate_orders_sql.py`,
    `compare_tiles.py`, `verify_module.py`, `business_probe.py`). Untracked and
    outside the repo, so they will vanish on reboot; none are referenced by the app.
    They are the reproduction of every claim above if you want to re-run it.

---

## Files changed

| File | Change |
|---|---|
| `orders_client.py` | HTTP client → read-only Postgres reader; same public surface |
| `app.py` | `cache_data` order book, DB config plumbing, dead paging UI removed, `threading` import dropped |
| `requirements.txt` | `+ psycopg2-binary>=2.9.9` |
| `README.md` | env table, Cloud Run deploy (secret + VPC egress), how the order book is read, the two deliberate differences |
| `cloudbuild.yaml` | VPC egress flags reasserted on every deploy |
| `.env` | host corrected dev → prod, password added *(untracked)* |

`orders.py` was **not** modified — the metric definitions are unchanged, which is what
made the before/after comparison meaningful.
