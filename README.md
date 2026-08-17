# jira-evaluator

Jira ticket management and health evaluation for the ML team, plus a Streamlit
dashboard for organizing tickets, seeing status transparently, and prioritizing
work at both the organization and the individual level.

## What it does

- Pulls open tickets live from the Jira Cloud REST API (`jira_client.py`), including
  changelogs and sprint fields — there are no exported files or offline snapshots.
- Derives per-ticket health metrics (`transformations.py`).
- Ranks tickets with a composite priority score (`prioritization.py`).
- Renders it all, and writes changes back to Jira, in a Streamlit app (`app.py`),
  with every mutation recorded in an audit log (`change_audit.py`).

## Running the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Leave `DASHBOARD_PASSWORD` unset for local runs, and keep it out of `.env` in
particular: `app.py` loads that file at import, so a copy pasted there to mirror
the deployment turns the gate on locally and every `streamlit run` opens with a
password prompt. Local runs are visible only to the person who started them, so
there is nothing for the gate to protect. If the prompt does appear, entering the
password once is enough — the browser keeps the signed cookie described under
[Deploying to Cloud Run](#deploying-to-cloud-run) for thirty days, so restarting
the server does not ask again. **Sign out** in the sidebar clears it.

### Configuration

Credentials resolve from environment variables first, and fall back to the YAML
profile when they are not all set.

| Variable                      | Required                         | Default                                          | Purpose                                                                                                                                                                |
| -------------------------------| ----------------------------------| --------------------------------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `JIRA_BASE_URL`               | yes (env mode)                   | —                                                | Jira site, e.g. `https://vinovoss.atlassian.net`                                                                                                                       |
| `JIRA_EMAIL`                  | yes (env mode)                   | —                                                | Atlassian account email used for API auth                                                                                                                              |
| `JIRA_API_TOKEN`              | yes (env mode)                   | —                                                | Atlassian API token ([create one](https://id.atlassian.com/manage-profile/security/api-tokens))                                                                        |
| `JIRA_CREDS_PATH`             | no                               | `~/.creds/vinovoss.yml`                          | YAML fallback credentials file                                                                                                                                         |
| `JIRA_PROFILE`                | no                               | `ML-TEAM-MANAGEMENT`                             | Profile name inside the YAML file                                                                                                                                      |
| `JIRA_DASHBOARD_JQL`          | no                               | `statusCategory != Done ORDER BY updated ASC`    | Ticket scope the dashboard loads                                                                                                                                       |
| `JIRA_TEAM_MEMBERS`           | no                               | `Tam,Mehdi Ordikhani`                            | Comma-separated defaults for the Team scope                                                                                                                            |
| `JIRA_MAX_RESULTS`            | no                               | `1000`                                           | Ceiling on tickets fetched per run; the dashboard warns when the result set is truncated                                                                               |
| `JIRA_PAGE_SIZE`              | no                               | `250`                                            | Tickets per Jira page; every page is a round trip, so lower it only if a tenant rejects the larger page                                                                |
| `JIRA_BACKLOG_STATUSES`       | no                               | `Backlog`                                        | Comma-separated statuses hidden when *Include Backlogs* is off                                                                                                         |
| `JIRA_WEEKLY_HOURS`           | no                               | —                                                | Hours per week each person is available, e.g. `Tam=10,Jal=20,Mehdi Ordikhani=40`; drives *Availability vs Commitment*                                                  |
| `JIRA_AUDIT_LOG_PATH`         | no                               | `logs/jira_ticket_changes.jsonl`                 | Where write-back history is recorded; point at durable storage when containerized                                                                                      |
| `JIRA_BROWSE_BASE`            | no                               | `<resolved Jira site>/browse`                    | Base URL for ticket hyperlinks; defaults to the site the credentials resolve to                                                                                        |
| `JIRA_TEAM_PROJECTS`          | no                               | —                                                | Which Jira projects form each team, e.g. `Marketplace=MB;App=AS,OA;Design=MAR`; used only where the assignee roster has no answer                                      |
| `JIRA_TEAM_PEOPLE`            | no                               | the VinoVoss roster in `teams.py`                | Who sits on each team, e.g. `Design=Robert,Alesya;App=Ali,Farid`; first names match Jira display names, and a `Former staff` team surfaces work still owned by leavers |
| `JIRA_EXTRA_PROJECT_KEYS`     | no                               | —                                                | Extra project keys a PR may reference, e.g. `MDP,WT2`, for projects the account cannot see; used by *PR Hygiene*                                                       |
| `PR_STALE_AGE_DAYS`           | no                               | `14`                                             | A PR open longer than this counts as stale                                                                                                                             |
| `PR_STALE_IDLE_DAYS`          | no                               | `7`                                              | A PR untouched for longer than this counts as stale                                                                                                                    |
| `GITHUB_EXCLUDE_REPOS`        | no                               | —                                                | Repos left out of every PR figure, e.g. `scratch,spike-repo` (or `owner/name`); the exclusion goes into the GitHub search queries, so the counts that cannot be filtered afterwards obey it too, and the Code page names what was excluded |
| `VIVINO_PROXY`                | no locally, yes on Cloud Run     | —                                                | Forward proxy URL (`http://user:password@host:port`) Vivino requests go through; Vivino refuses Cloud Run's shared egress addresses with 403s, so the hosted app needs a proxy whose address Vivino serves - unset, requests go direct, which works from most other hosts. The URL carries a credential, so mount it from Secret Manager rather than a plain env var |
| `POSTGRES_PASSWORD`           | no                               | —                                                | Password for the order database (`MEDUSA_DB_PASSWORD` also accepted); without it the *Orders, Revenue & AOV* section says so and everything else works                 |
| `POSTGRES_HOST`               | no                               | `db.prod.vinovoss.private`                       | Host holding the `medusa` schema (`MEDUSA_DB_HOST` also accepted). **The dev CRM uses the same schema on `db.dev.vinovoss.private`, so a wrong host reports a different shop rather than failing** |
| `POSTGRES_DATABASE`           | no                               | `private_dataset`                                | Database the `medusa` schema lives in (`MEDUSA_DB_NAME` also accepted)                                                                                                  |
| `POSTGRES_USER`               | no                               | `app__vinovoss_backend`                          | Role the order book is read as (`MEDUSA_DB_USER` also accepted); needs only SELECT on the `medusa` schema                                                               |
| `MEDUSA_STORE_PREFIX_ALIASES` | no                               | —                                                | Retired product-handle prefixes mapped to a merchant, e.g. `oldprefix=Store Name`; without it that merchant's older sales show as *Unattributed*                       |
| `AMPLITUDE_API_KEY`           | no                               | —                                                | Amplitude project API key (Settings → Projects → your project); needed with the secret key for *Product Funnel & Friction*                                             |
| `AMPLITUDE_SECRET_KEY`        | no                               | —                                                | The same project's secret key; Amplitude's Dashboard API authenticates on the pair, and the API key alone is refused                                                   |
| `AMPLITUDE_API_URL`           | no                               | `https://amplitude.com`                          | Set to `https://analytics.eu.amplitude.com` for an EU-region project, whose keys the US host refuses                                                                   |
| `AMPLITUDE_FUNNEL`            | no                               | product page → cart → checkout → payment → order | The funnel's steps as `Label=event_name` pairs, in order, e.g. `Visited=_active,Bought=checkout_order_completed`                                                       |
| `GOOGLE_ADS_BQ_PROJECT` | no | the project the credentials resolve to | GCP project holding the Google Ads data transfer; on Cloud Run the service's own project is right, so this is only for reading another project's dataset |
| `GOOGLE_ADS_BQ_DATASET` | no | `google_ads` | Dataset the Ads transfer writes to; without a readable one the *Ads Spend & Return* section says so and everything else works |
| `MARKETPLACE_COMMISSION_RATE` | no | `12%` | The share of a sale the marketplace keeps, as `12`, `12%` or `0.12`. Only a fallback: with `STRIPE_READONLY_API_KEY` set, *commission per unit spent* uses the commission Stripe actually charged, which already reflects each merchant's own rate |
| `GOOGLE_ADS_CUSTOMER_ID` | no | every account in the dataset | Restrict the ads figures to one account, e.g. `887-686-4797`; by default every account the transfer writes is added up |
| `OPENAI_ADMIN_KEY` | no | — | OpenAI **organization admin** key (Organization settings → API keys → Admin keys); needed for the AI line of *Burn*. An ordinary `sk-proj-…` project key is refused by the cost endpoint and is named as such rather than shown as a 401 |
| `STRIPE_READONLY_API_KEY` | no | — | Stripe **restricted** key (`rk_…`) with read access to balance transactions, charges, disputes and payouts; needed for the payments line of *Burn*. A full `sk_…` secret key can move money and is refused |
| `GCP_BILLING_BQ_PROJECT` | no | the key's own project, else the ambient one | GCP project holding the Cloud billing export. Read independently of the Ads settings, so a bad Ads value cannot take the Cloud bill off the page; unset on Cloud Run, which runs in the project the export is in |
| `GCP_BILLING_BQ_DATASET` | no | `billing_export` | Dataset the *standard usage cost* billing export writes to; the table inside it is found by name, and until Google writes one the *Cloud costs* section says so |
| `GCP_BIGQUERY_READONLY_KEY` | no | — | A BigQuery service-account key as JSON, for reading the Ads and billing datasets from outside GCP; unset on Cloud Run, which authenticates as its own service account. Also the credential the Merchant Center read uses |
| `GOOGLE_MERCHANT_ID` | no | — | Merchant Center account id (top right of merchants.google.com), for the *Price competitiveness* section. The account reading it — the key above, or Cloud Run's own service account — must be a user of that Merchant Center account with read access, and its GCP project must be registered against the account as a Merchant API developer; without either the section says so and everything else works |
| `GOOGLE_MERCHANT_COUNTRY` | no | `US` | Two-letter country the feed targets. Benchmarks are published per country, so reading a country the feed does not target returns no rows rather than an error; the section names the country it read when it finds nothing |
| `DASHBOARD_PASSWORD`          | no locally, **yes on Cloud Run** | —                                                | Shared password visitors must enter; remembered per browser for 30 days in a signed cookie; leave it unset locally (a copy in `.env` prompts on every run), unset on Cloud Run (`K_SERVICE` present) refuses to serve at all                                     |
| `DASHBOARD_COOKIE_KEY` | no, but set it when hosted | derived from `DASHBOARD_PASSWORD` with scrypt | Independent secret signing the access cookie, so the cookie is not a verifier for guesses at the password; rotating it signs every browser out without changing the password anyone types |

All three of `JIRA_BASE_URL`, `JIRA_EMAIL` and `JIRA_API_TOKEN` must be present for
env mode; otherwise the YAML profile is used:

```yaml
Jira:
  ML-TEAM-MANAGEMENT:
    base_url: https://vinovoss.atlassian.net
    email: you@drinkbetter.ai
    api_token: <atlassian-api-token>
```

Widen `JIRA_DASHBOARD_JQL` to cover more than one project when you want a true
organization-wide view, e.g. `project in (MB, ML) AND statusCategory != Done`.

### Deploying to Cloud Run

The `Dockerfile` runs Streamlit on the port Cloud Run injects. Deploy it privately
and let Google sign-in gate access, so anyone on the Workspace domain can open the
URL and nobody else can:

> Cloud Run IAM authenticates callers, not browsers: a viewer without a signed
> request gets a bare 403 rather than a login page, and turning that into a real
> sign-in needs IAP, which in turn needs the project to sit inside a Google Cloud
> Organization. Where that is not the case, the fallback below trades the IAM
> binding for the app's own shared password — and then `DASHBOARD_PASSWORD` is the
> only thing between the public internet and Jira write access, so it is required,
> not optional.

```bash
PROJECT=<gcp-project-id>
REGION=us-central1

# Store the Jira token once.
printf '%s' '<atlassian-api-token>' | \
  gcloud secrets create jira-api-token --data-file=- --project "$PROJECT"

# And the order database's password, for the Business tab.
printf '%s' '<order-db-password>' | \
  gcloud secrets create orders-db-password --data-file=- --project "$PROJECT"

# The proxy Vivino reads go through; its URL carries a credential, so it
# lives in Secret Manager like the other credentials.
printf '%s' 'http://<user>:<password>@<proxy-host>:<port>' | \
  gcloud secrets create vivino-proxy --data-file=- --project "$PROJECT"

gcloud run deploy jira-dashboard \
  --source . \
  --project "$PROJECT" --region "$REGION" \
  --no-allow-unauthenticated \
  --session-affinity --max-instances 1 \
  --network default --subnet default --vpc-egress private-ranges-only \
  --set-env-vars "JIRA_BASE_URL=https://vinovoss.atlassian.net,JIRA_EMAIL=<service-account-email>" \
  --set-secrets "JIRA_API_TOKEN=jira-api-token:latest,POSTGRES_PASSWORD=orders-db-password:latest,VIVINO_PROXY=vivino-proxy:latest"

# Let the whole Workspace domain in (requires IAP or domain-restricted sharing).
gcloud run services add-iam-policy-binding jira-dashboard \
  --project "$PROJECT" --region "$REGION" \
  --member "domain:vinovoss.com" --role roles/run.invoker
```

If the project is outside an Organization, deploy publicly with the password gate
instead of the IAM binding:

```bash
# A signing key for the "remember this browser" cookie, independent of the
# password. Rotate this secret to sign every browser out at once.
python -c 'import secrets; print(secrets.token_urlsafe(32))' | tr -d '\n' | \
  gcloud secrets create dashboard-cookie-key --data-file=- --project "$PROJECT"

gcloud run deploy jira-dashboard \
  --source . \
  --project "$PROJECT" --region "$REGION" \
  --allow-unauthenticated \
  --session-affinity --max-instances 1 \
  --network default --subnet default --vpc-egress private-ranges-only \
  --set-env-vars "JIRA_BASE_URL=https://vinovoss.atlassian.net,JIRA_EMAIL=<service-account-email>,DASHBOARD_PASSWORD=<shared-password>" \
  --set-secrets "JIRA_API_TOKEN=jira-api-token:latest,POSTGRES_PASSWORD=orders-db-password:latest,DASHBOARD_COOKIE_KEY=dashboard-cookie-key:latest,VIVINO_PROXY=vivino-proxy:latest"
```

The order database answers on a private VPC address, so without
`--network default --subnet default` the *Business* tab reports a connection
timeout while every Jira section keeps working. `--vpc-egress private-ranges-only`
keeps Jira, GitHub and Amplitude going straight out to the internet rather than
through the VPC.

A shared password is weaker than Google sign-in: it does not identify who is
looking, cannot be revoked per person, and only throttles guessing. Treat it as a
stopgap until the service can live in an Organization behind IAP.

Entering it once is enough for thirty days per browser. The gate sets a
`jira_dashboard_access` cookie holding an expiry signed with a key derived from
`DASHBOARD_PASSWORD` — the password itself never reaches the browser, and a
tampered, expired or stale-password cookie simply fails its check and brings the
prompt back. This is what stops a refresh, a second tab or a Cloud Run cold start
from asking again, none of which survive Streamlit's per-websocket session state.
Two consequences worth knowing: changing `DASHBOARD_PASSWORD` signs everyone out,
and **Sign out** in the sidebar is the way to end a session early, since closing
the tab leaves the cookie in place.

Be clear about what that cookie is: a bearer token. Streamlit can only set it
from JavaScript, so it cannot be `HttpOnly` and any script in the page could read
it. Three things narrow the blast radius rather than close it — the signature
covers the browser the cookie was issued to, so it does not replay from a
different one; `Secure` is set on every https origin (decided by the page's own
scheme, not by whether the deployment happens to be Cloud Run); and the signing
key is stretched out of the password with scrypt, so a leaked cookie is not a
cheap oracle for guessing a short shared password.

**Set `DASHBOARD_COOKIE_KEY` on any real deployment.** It replaces the password
in that derivation with an independent secret, so the cookie says nothing about
the password at any price, and rotating it signs every browser out *without*
making anyone learn a new password — the central revocation a shared password
otherwise cannot offer. Rotate it if a laptop goes missing.

The command prints the service URL. Streamlit holds per-user state on a websocket,
hence `--session-affinity` and the single instance: they keep a viewer's reconnects
on the instance that served them. Each viewer still gets their own filters and
selections.

Cloud Run's filesystem is per-instance and in-memory, so the write-back audit log
(and with it the *Change History and Revert* section) is wiped on every restart.
Mount durable storage and set `JIRA_AUDIT_LOG_PATH` to a path inside it if reverting
past changes matters:

```bash
  --add-volume "name=audit,type=cloud-storage,bucket=<bucket>" \
  --add-volume-mount "volume=audit,mount-path=/audit" \
  --set-env-vars "JIRA_AUDIT_LOG_PATH=/audit/jira_ticket_changes.jsonl"
```

## Dashboard layout

The dashboard is two pages, chosen from the navigation at the top: **Engineering**
holds everything below, and **Business** holds the shop's numbers on their own.
They answer a different question from ticket and PR health, and the sidebar scope
and filters do not apply to them. Pages rather than tabs because Streamlit runs
the body of every tab on every rerun whichever one the browser is showing, so a
reader watching the shop's figures was paying to rebuild twenty engineering
sections; a page that is not open does not run at all. The Business page is
listed only when the order database, Google Ads or Amplitude is configured.

Each page carries a **Download report** button at its top right: the page's
headline figures and the sentences explaining them, as a self-contained HTML
page that prints to a PDF (open it and press Cmd/Ctrl-P). It holds what the
page currently reads, so a scope or window changed in the sidebar is reflected
in it, and a figure that could not be read is left out rather than printed as
a dash.

The Business page holds:

- *Orders, Revenue & AOV* for the last 7 and 30 days, each against the window
  before it.
- *Best Sellers & Merchants* over a window of 30, 90, 180 or 360 days: the wines
  selling most bottles, and revenue, orders and cancellations per merchant. A
  merchant is read from the handle prefix on each order line (`store_prefix` in
  the CRM), since an order carries at most one store link while its lines may come
  from several; a line whose prefix matches no current merchant is listed as
  *Unattributed* rather than credited to a guess, and
  `MEDUSA_STORE_PREFIX_ALIASES` maps prefixes a merchant has since retired.

The order book is read from the `medusa` schema of the CRM's own Postgres
database rather than its admin API. The API is not wrong, it is slow: a year of
orders is ~16 sequential pages and about ten seconds, paid again on every cold
start, where the same year is one query and about a third of a second. So the
whole year is re-read outright every 15 minutes and there is no incremental
top-up to go stale. It no longer waits behind a button: the button existed only
because tabs run whether or not they are being looked at, and now that this is
its own page the read happens when somebody asks for it.

Two figures Medusa computes at read time are derived here instead, and are worth
knowing when a number is queried:

- `status` is *canceled* once `canceled_at` is set, and the stored value otherwise.
- `payment_status` comes from the order's payment collections and its summary,
  taking whichever reports more money moved. Neither alone is enough: a capture
  does not always reconcile into `order_summary`, and a refund does not always
  land on the payment collection.

Against a live year of prod orders this matches the admin API exactly on order
count, cancellations, currency, paid-order count and every line item's quantity
and price. It deliberately differs in two places. Revenue is now **lower**,
because the API does not return `refunded_total` at all — it silently drops the
field — so the refund netting the tiles have always described was never actually
happening; roughly 27 orders a year carry a refund. And ~1.5% of orders that the
API calls *partially_captured* are counted here as fully paid: those orders have a
dead `canceled` payment collection alongside a `completed` one, and the API sums
the dead one into its denominator even though its own summary reports the order
fully paid with nothing pending.

1. **Scope** (sidebar) — `Organization` (every assignee returned by the JQL), `Team`
   (multi-select pre-filled from `JIRA_TEAM_MEMBERS`), or `Individual` (one assignee).
2. **Filters** (sidebar) — status, priority, minimum idle days, minimum ticket age,
   and an *Include Backlogs* toggle that the summary and analysis sections respect.
   The bubble chart and the sprint editor deliberately ignore it, since pulling a
   Backlog ticket into a sprint requires being able to see it.
3. **Headline strip** — open tickets, tickets stalled 30d+, unassigned, estimate
   coverage, stale late-stage work, and oldest ticket age, colored by severity.
4. **Teams** — open load, staffing, idle pressure and estimate gaps per team, plus
   the active (or next) sprint for the selected team: ticket count, people,
   committed hours and status mix. A ticket's team comes from its assignee
   (`JIRA_TEAM_PEOPLE`) first, since part-time people work across projects, then
   from `JIRA_TEAM_PROJECTS`, then from the raw project key. Unowned tickets are
   grouped as *Unassigned work* rather than credited to a team.
5. **Epics** — open children rolled up per epic with the signals that mark an epic
   as drifting (idle, unassigned children, missing estimates, spread across sprints,
   too many owners), plus a count of tickets with no epic at all. Because the JQL
   loads open work only, this is remaining work per epic, never completion.
   Under it, **Epic organization** takes the tickets with no epic and suggests a
   parent for each from the epics that already exist: the suggestion is the epic
   whose own name and whose existing children's summaries share the most
   distinctive words with the ticket, restricted to the ticket's own project, and
   the words that earned it are shown so the guess can be judged rather than
   trusted. A ticket whose best match explains less than a third of what it is
   about keeps its row but loses the guess, since a wrong parent offered
   confidently is worse than none. A companion tab lists epics with no open
   children left - finished or abandoned, closeable either way. Nothing is filed
   automatically; every suggestion is read-only.
6. **Backlog Cleanup** — the oldest open tickets presented one at a time, because a
   backlog is triaged in single decisions rather than in a table of 200 rows. Each
   card carries the ticket's age, idle time, owner, status, epic and priority, with
   the signals arguing for closure highlighted, and a suggested decision derived in
   `cleanup.suggest_decision`: a ticket a year old and untouched for six months, or
   one nobody ever took that has sat 90 days, is proposed for closing; anything else
   defaults to *Keep*. Two queues: oldest open, and oldest unassigned - the latter
   ignores the assignee scope filter, since a ticket with no owner belongs to no
   team and would otherwise be empty by construction outside Organization scope.
   Decisions live in the session and are exportable as CSV; only the explicit *Apply* button at the
   bottom writes to Jira, through the same audited path as every other write.
   Because each project runs its own workflow - few statuses here offer *Done*
   from a backlog state - the closing status is resolved per ticket from the
   transitions Jira actually offers it, preferring *Archived*, *Won't Do*, *Not
   needed* or *Cancelled* over *Done* so a cleanup closure stays distinguishable
   from a real completion (`cleanup.CLOSING_STATUS_PREFERENCE`). A ticket offering
   no closing transition is reported by name rather than forced. Backlog tickets are
   always included here regardless of *Include Backlogs* - they are the point.
7. **Assignee Breakdown** — per-assignee roll-up (open tickets, average and top
   priority score, average and max idle days, tickets idle 15d+, tickets with no
   priority). In `Individual` scope this collapses to that person's headline numbers.
8. **Prioritized Queue** — tickets ranked by priority score with the reasons behind
   the score, and a CSV export.
9. **Estimate Policy** — the team rule is that a ticket carries an estimate once it
   leaves Backlog. Anything past a status listed in `JIRA_BACKLOG_STATUSES` with no
   estimate is a violation; the panel shows overall compliance, a per-owner table,
   and the offending tickets (CSV export included). Backlog tickets are exempt and
   are excluded from the denominator.
10. **Stale & Abandoned** — tickets idle past a threshold (90 days by default,
   adjustable), ranked by how many neglect signals they carry: unassigned, no
   estimate, no due date, never started, no priority, carried over 3+ sprints. Read
   only — it recommends what to close or send back to Backlog, it never writes.
   Backlog tickets are always included here regardless of *Include Backlogs*.
11. **Orders, Revenue & AOV** (*Business* tab) — the shop beside the engineering,
   read live from the
   Medusa CRM: orders, captured revenue and average order value over the last 7 and
   30 days, each against the equivalent window before it, plus a per-day bar for the
   month. Revenue counts captured payments only and cancelled orders count towards
   neither revenue nor AOV, so the order tile and the revenue tile deliberately do
   not divide into each other - the fourth tile names the gap (cancelled / placed
   but not yet paid). AOV is captured revenue over the paid orders that produced it,
   not over every order placed, which would understate the basket whenever payment
   capture lags. Anything refunded is netted off. Read-only: every statement is a
   `SELECT` against the CRM's `medusa` schema on a connection opened read-only,
   refreshed at most every fifteen minutes. Needs
   `POSTGRES_PASSWORD`. Below it, **Best Sellers & Merchants** ranks wines by
   bottles sold - not by revenue, which would answer which bottle is dearest
   rather than which is wanted - and breaks revenue, orders and cancellations out
   per merchant, over 30, 90, 180 or 360 days. Bottles count what customers chose,
   so an order awaiting payment counts there while revenue still waits for
   capture; ice packs are add-ons and are kept out of the wine ranking.
12. **Price competitiveness** (*Business* tab) — what the shop charges against what
   everyone else selling the same bottle charges, read live from Merchant Center.
   Google works the benchmark out across every merchant in Shopping and it is the
   one number the order book cannot hold: it says what was charged, not what the
   shop next door charged for the same wine. Three tiles — the share of priced
   products above the benchmark, the typical (median, not mean: a handful sit near
   twice the market and would drag an average) gap, and how many products were
   compared — over three tables. **Ask the merchants** is the negotiation list:
   the hundred bottles ranked on clicks times the gap, which is the demand
   Shopping actually saw times how far over the market that demand was asked to
   pay, with what it would take to reach the market price on each, which
   merchants list that bottle (matched through the catalogue, since Google names
   the wine and not whose listing it is), Google's own suggested cut where it
   publishes one, and a slider asking what one percentage off the list would do
   — a merchant agrees to a percentage over a range, not to five thousand prices.
   Nothing there predicts extra orders: the feed carries no conversion tracking,
   so conversions read zero on every row and an order count would be invented.
   **Cheaper than the market** is the same read the other way up, the wines
   already under the benchmark and being clicked on, which need nobody's
   agreement and only the ad budget. **What price did to sales** is the evidence
   to send a merchant that is being asked to come down: the catalogue grouped
   into four bands against the market — cheaper, about the same, up to 25%
   dearer, more than that — and for each the wines in it, the clicks Shopping
   gave it, the bottles the shop's own order book sold over the last 90 days and
   the bottles per 100 clicks, which is the only one of those that survives the
   bands being different sizes. On the live feed that reads 29 bottles per 100
   clicks under the market against 8 above it. The bottles come from the CRM's
   paid orders joined to Google's offer id through the product's `external_id`,
   because Merchant Center reports no conversions on this feed at all — an
   absence of tracking, not of sales — and Google Ads' own product conversions
   are too thin to give any one wine a rate. It is a comparison and not an
   experiment, and it says so: a keenly priced wine may also be a wine people
   want. A wine nobody clicked has no rate rather than a nought, and a CRM that
   cannot be reached loses the tab and the column rather than reporting that
   nothing sold. Each of these tables downloads as the columns on the screen and
   no others. **Dearest bottles** is the plain ranking by gap, demand or no
   demand.
   Above all four, a **merchant** picker cuts the whole panel — tiles, verdicts,
   tables and the files they download — to the wines one merchant lists, which
   is what gets sent to that merchant; the wines are matched to merchants through
   the CRM's store prefixes, so a merchant with no benchmarked wine says so
   rather than reading as a merchant with nothing to fix.
   Products no other merchant sells have no benchmark and are left out rather
   than counted as competitive, and a price within two pence of the benchmark is
   the same price rather than a problem. Prices in a second currency are set
   aside and named, never compared with the main one. Where Google also publishes
   a suggested price, the verdicts say how many products it would cut and what it
   predicts that would do to clicks and conversions — counted over the offers
   that were compared, since that report carries no benchmark, country or
   currency filter and would otherwise claim a cut on more products than the
   panel compared. Not read from BigQuery,
   though a Merchant Center transfer once wrote these rows there: Google
   deprecated `export_price_benchmarks`, so benchmarks reach only the Merchant
   API now — its `v1` endpoints, `v1beta` having been switched off in February
   2026. Needs `GOOGLE_MERCHANT_ID`, a reader on the Merchant Center account,
   and the GCP project the credential belongs to registered against that
   account with a verified `API_DEVELOPER` user (`developerRegistration:registerGcp`);
   refreshed at most every six hours, which is as fresh as a benchmark
   recomputed daily can be.
13. **Ads Spend & Return** (*Business* tab) — what the orders cost to win, read from
   Google's own Ads-to-BigQuery transfer rather than the Ads API, which issues
   developer tokens to manager accounts only and this account has none above it.
   Spend, the CRM's orders over exactly the same days, ad spend per order, revenue
   per currency unit spent, and Google's own conversion count, over 7 or 30 days.
   Two figures deliberately sit side by side rather than being reconciled into one:
   *Google's own conversions* credits the day of the *click*, counts view-throughs
   and splits one sale across several ads, while the CRM counts money captured on
   the day it arrived — so the CRM is the figure to quote and a gap over a quarter
   between them is called out as tracking worth checking. Revenue per unit spent
   counts *every* order in the window, including the ones no ad won, so it is a
   ceiling rather than a return; below roughly 3x, wine's margin does not cover the
   ad that sold it. The campaign table is ordered dearest first and drops campaigns
   that spent nothing — an account holding twelve campaigns of which two run would
   otherwise bury the two that cost money — while a campaign that spent money and
   has no snapshot row is still listed, by id. Spend ends *yesterday* and is counted
   in the ad account's own timezone; a window that starts before the transfer's
   history does says how much of it has been loaded, because Google's transfer loads
   one day per run and backfills only when asked, so a new transfer would otherwise
   report one day of spend as a month of it. Where that history begins is asked of
   the whole table rather than inferred from the window's rows: the stats table has
   no row for a day on which nothing ran, so an account that paused would otherwise
   be reported as a broken feed instead of a quiet one. Money in two currencies is
   never added — the CRM side is taken in the shop's main currency, only the ad
   accounts billing in the commonest currency are summed (any others are named in a
   caption), and if the shop and the ad account disagree, return per unit spent is
   left blank rather than dividing one currency by another. Every figure, tiles and
   sentences alike, is quoted in the ad account's own currency. Costs arrive as
   micros and are divided out; campaign names come from the newest daily snapshot
   rather than joined per day, or a campaign renamed mid-window would appear twice
   with its spend split between the two names. The headline is **commission per
   unit spent** — the marketplace's commission over the window, divided by spend —
   because the revenue an ad wins is the merchant's and only the commission on it
   is income here; 1.00 is where an ad pays for itself. Merchants are on their own
   agreements (10%, 12%, 12.5%), so a single rate estimates a figure Stripe holds
   exactly: where a Stripe key is set and the account takes application fees in
   the ads' own currency, the numerator is the commission Stripe charged, net of
   refunds. Without one it falls back to revenue at
   `MARKETPLACE_COMMISSION_RATE` — 1,226 at 12% against 176 of spend is 0.84 — and
   the caption under the headline says which of the two is on screen. Gross
   return per unit spent is kept beside it as the ceiling it is. Needs a readable dataset
   (`GOOGLE_ADS_BQ_DATASET`, default `google_ads`); read-only, every statement is a
   `SELECT` under a credential holding `bigquery.dataViewer` and
   `bigquery.jobUser` and no access to Google Ads itself.
14. **Product Funnel & Friction** (*Business* tab) — how far visitors get towards
   being one of those orders, read from Amplitude. The default funnel runs product
   page → cart → checkout → payment → order, deliberately *not* home page → search
   → product: on this shop the overwhelming majority of visitors arrive on a product
   page from a search engine and never see the home page, so a funnel starting there
   would describe a few hundred people out of tens of thousands. *Used the site* —
   everybody who did anything at all — is a tile beside the funnel rather than its
   first step, for the same reason from the other direction: Amplitude counts a step
   only when it happened *after* the one before, and for somebody who lands straight
   on a product page that view is the first thing they ever did, so "did anything"
   then "viewed a product" cannot both be satisfied. Read against production, using
   it as step one reported 9,656 product-page viewers where 30,530 people saw one,
   and dragged every rate below it down in proportion. Counted in people rather than
   visits, steps must happen in that
   order within 7 days of each other (wine is read about and bought later, so a
   one-day window would report the shop as worse than it is), and the window ends
   *yesterday*, because today is still being recorded and always reads as a slump.
   *From previous step* is the column to act on — it names the single screen costing
   the most — while *from the start* is what people mean by "conversion rate". Each
   rate is shown against the same window immediately before it, in percentage points
   rather than percent (2% to 3% is a rise of one point, and calling it fifty percent
   is how a modest week gets reported as a triumph); a move under a tenth of a point
   reads as *flat*, and a project with no previous period simply has no column. Under
   the table, **what each step means** says the same thing in sentences — *2 of every
   100 people who got as far as the product page went on; 98 did not* — because the
   person this is for reads it between meetings and should not have to do the
   arithmetic to find the screen that is losing the shop its customers.
   Beside it, **What went wrong** counts the people who hit an app error, a failed
   add-to-cart, a blocked checkout, a failed payment or an empty search, as a share
   of everyone who visited; one person meeting the same error ten times is one
   person to apologise to, not ten. Below that, **where the errors are** breaks the
   app errors down by the page they happened on and the message they carried, so
   "4% of visitors saw an error" becomes something a ticket can be written about.
   That breakdown alone is counted in *times* rather than people — the one figure
   here that can honestly be added up across days — and message values are collapsed
   into families first (`Loading chunk 36187 failed` and the same line with another
   build hash are one problem, as is the same line whose `(error: …)` aside was
   truncated before its closing bracket), or a single broken deploy fills the table
   with near-identical rows and pushes the real second-biggest problem off the bottom.
   **Voss AI** is reach rather than engagement: how many people opened it, asked it
   something, and got nothing back. Needs `AMPLITUDE_API_KEY` and
   `AMPLITUDE_SECRET_KEY`; read-only, every call is a `GET /api/2/funnels` bar the
   breakdowns, which are `GET /api/2/events/segmentation`. The people counts are
   asked as two-step funnels rather than of the segmentation endpoint, whose interval
   only comes in days, weeks or months: no interval spans an arbitrary window, so a
   count from it is a sum of buckets, and adding buckets counts somebody who came
   back next week as two people.
15. **Burn** (*Business* tab) — what the business spends and what its own payment
   ledger says it kept, beside the revenue above it, on one window so a week of one
   bill is never read against a month of another. **AI spend** is OpenAI's own
   organization cost report, broken down by line item and project, because "$763 on
   OpenAI" is not actionable and "$227 of it re-sending cached context" is: the
   cached share is called out as its own tile, being the one line on an AI invoice
   that is usually a choice rather than a fact. A month's run rate divides by every
   day in the window rather than by the days that had charges — a quiet weekend is
   part of the bill — and today *counts*, unlike the ad figures, because a provider
   bills as it goes. Where the previous period came to under a twentieth of this one,
   the change is reported as *this spend is new* rather than as a percentage: a month
   that went from $3 to $763 is a true +25,000% and a useless sentence. The endpoint
   is organization-scoped, so it needs an admin key, which is named plainly when a
   project key is pasted instead. **Payments** is Stripe's balance transactions, and
   what it holds was not what was expected: this account is a *Connect platform*, so
   each sale's card fees are charged on the merchant's own connected account and what
   lands here is the marketplace's commission — reported as commission kept, not as a
   cost, with Stripe's own fees shown as the nil they are rather than omitted, since
   "what do the card fees cost us" deserves an answer. It is deliberately not
   reconciled with the CRM's captured revenue above: one is the platform's cut, the
   other the merchants' takings. Disputes are counted, never subtracted — a dispute
   is money at risk with an outcome still to come. **Cloud costs** is Google's own
   billing export, by service and net of credits, because a committed-use discount
   is money never charged: it is usually the largest of the three bills and the one
   nobody sees until the month ends. The export is not retroactive and backfills over
   hours, so a window reaching further back than it does is labelled as the shorter
   period it really is.
   Needs `OPENAI_ADMIN_KEY` and `STRIPE_READONLY_API_KEY`; each line appears on its
   own as the key arrives, and every call is a `GET` under a credential that cannot
   write.
16. **PR Hygiene** — open PRs across the organization that are untraceable, stalled
   or unowned: no Jira key anywhere in the title, branch name or description
   (matched against every project key Jira exposes, plus `JIRA_EXTRA_PROJECT_KEYS`,
   so a string like `UTF-8` does not read as a ticket); open past
   `PR_STALE_AGE_DAYS` or untouched past `PR_STALE_IDLE_DAYS`, with the reason
   named; and nobody requested to review with no review yet. A first tab answers a
   different question - which open PRs matter: those whose Jira ticket is High,
   Highest or Urgent *and* already in dev, code review or staging, longest-idle
   first, because that is work one review away from shipping. Tickets merely In
   Progress are excluded: the code is still being written. Includes a per-author
   table and a CSV of everything flagged. Needs `DASHBOARD_GITHUB_TOKEN`.
17. **Ticket Quality & Ready for Devin** — every ticket graded out of 5 on whether
   someone outside the original conversation could pick it up: a summary of at
   least four words, a description of at least 120 characters, explicit acceptance
   criteria (the words, or a checklist of three or more items), an estimate, and an
   epic. Each row names what is missing. *Devin-able* is stricter and separate from
   the score: **Yes** needs the goal and the finish line written down and work that
   does not hinge on being in the room, **Maybe** is one gap short or reads as a
   conversation (design, research, an outage), **No** is neither. Epics and
   initiatives are exempt and score *n/a*. The per-person table groups by reporter,
   since the person who wrote the ticket is the one who can say what done means.
   Backlog tickets are always included regardless of *Include Backlogs* — an unowned
   backlog ticket is the best kind to hand off, and the worst-written ones collect
   there unseen.
18. **Sprint Planner** — a first draft of one team's next sprint, built from goals
   rather than from the top of a priority list. Name two or three goals for the
   sprint ("Onboarding, Quiz, Checkout", most important first) and every ticket
   that shares a word with one — in its summary, its epic's name or its labels —
   is filled into its owner's hours, goal by goal and highest priority first
   inside each goal. Work already in progress is placed before anything new, on
   the grounds that a sprint which starts three tickets and finishes none is the
   failure worth preventing. Each person's hours come from `JIRA_WEEKLY_HOURS`
   over the sprint's working days, **minus an overhead** — code review, Slack,
   meetings — that defaults to 4h/week and is editable, because those hours are
   spent every sprint and appear on no ticket. An unestimated ticket is assumed
   to cost 4h (also editable) and the row says so, since counting it as zero
   would let the sprint absorb unlimited unmeasured work. *Goals* shows what each
   goal needs against what fits, which is how you find out that all three do not.
   Everything is a proposal: tick or untick any row, change any ticket's hours,
   and the per-person load follows the decision rather than the suggestion.
   Nothing reaches Jira until *Add to sprint* is pressed with the edit switch
   armed, and it only adds tickets to an active or future sprint. Jira moves an
   issue between sprints rather than copying it, so a ticket already on another
   open sprint leaves it; the section names those tickets before the button.
19. **Bubble chart, Sprint Capacity, Suggested First Action** — the existing
   age-vs-idle chart, sprint planning tables, and bulk Jira write-back actions.
20. **Availability vs Commitment** — inside Sprint Capacity. Committed estimate hours
   per person against what they are actually available for, which matters when most
   contributors are part-time: `JIRA_WEEKLY_HOURS` is spread over the weekdays
   between the sprint's start and end dates, so 20h/week across a 10-working-day
   sprint is 40h available. Status is *Over-committed* above 100% utilization, *At
   capacity* from 85%, otherwise *Has room*. In the Organization scope everyone with
   declared hours appears, including anyone carrying nothing at all, which is how
   spare capacity surfaces; the Team and Individual scopes narrow the table to the
   people selected, so that someone merely filtered out is not read as idle. People
   without declared hours show *Unknown*, and a sprint with no dates in Jira
   disables the table rather than guessing. A declaration written as a bare first
   name that matches two people in Jira is withheld from both rather than granted
   twice, which shows as *Ambiguous roster name*; spell it as the full display name.

## How the page loads

The dashboard is a live view over three remote systems, so almost all of what a
reader waits for is network latency rather than computation — deriving every
health field for a thousand tickets takes about a tenth of a second. A cold load
went from ~30s to ~12s by removing waiting, not work:

- **The opening reads go out together.** Nine Jira queries and five GitHub ones
  depend on none of each other, so `_gather` runs them on a thread pool and the
  wait is the slowest of them rather than their sum. Each worker is handed the
  script's run context so the `st.cache_data` wrappers still see the session's
  cache. A read that fails returns `None` and costs its own section, not the page.
- **One pooled, retrying HTTP session per credential set.** Every call used to
  open its own `requests.Session`, and so paid a fresh TLS handshake; the pool is
  shared across threads, and retries cover 429 and 5xx on idempotent methods only,
  so a write can never be replayed.
- **The Sprint Capacity status dropdown is asked for only when edits are armed.**
  Jira answers about legal transitions one issue at a time, and fifty of those in
  series was ~16s — the single most expensive thing on the page, spent widening a
  dropdown a read-only visitor cannot use. Armed, the same lookups run
  concurrently and take about two seconds.
- **Each query asks for the fields it renders.** The snapshot lists need five
  fields, and the resolved list feeds one pie and needs one, rather than all
  seventeen defaults including every ticket's full description.
- **Caches refresh in the background.** When a TTL lapses the reader gets the
  slightly stale answer immediately while the replacement is fetched behind them,
  instead of one unlucky visitor every five minutes paying the cold start for
  everybody. *Refresh data* clears the caches for the page being looked at — not
  every cache in the process, which used to throw away the year of orders to
  refresh a ticket count.
- **Each cache is kept for as long as its source actually moves.** *Ads Spend &
  Return* is the clearest case: the grain is a day, the newest day is yesterday,
  and Google's transfer writes it once a day, so re-reading every fifteen minutes
  bought no freshness and paid a round of BigQuery jobs for it. The spend is held
  six hours and the account list a day, both keyed on the date so they roll over
  when the transfer does rather than mid-morning. *Cloud costs* is the same case
  and held the same way: the export is written in arrears, its last day is
  yesterday, and each read costs two scans of the whole export — the coverage
  probe has no day to filter on, and `DATE(usage_start_time)` prunes no
  ingestion-time partition. The widest window is the only one read, so the
  radio's narrower options are cut from the frame in hand.
- **The ads panel reads one window, not the one selected.** `daily_stats` already
  fetches twice the days it is asked for so the previous period can be compared,
  so the widest option contains every narrower one and the 7/30 radio slices the
  frame in pandas. Moving it used to be a cold BigQuery read; it is now a redraw.
- **BigQuery jobs go out together too.** Per ad account the dashboard wants the
  account's name and currency, the transfer's first loaded day, the campaign names
  and the daily stats — four independent jobs with about a second of latency each,
  run in series. `_parallel` is `_gather`'s sibling for a single section: same
  thread pool and run-context handling, but it raises rather than returning
  failures, because the section reports its own errors. Against the live dataset
  a cold read of both windows went from ~8s to ~3s, and switching window from
  seconds to about a millisecond.
- **The ads read starts while the order book is still being read.** They are
  different systems drawn one above the other, so the page waited on the sum of
  two networks for no reason but layout; the read is now kicked off at the top of
  the Business page and lands in the cache entry the panel goes on to ask for.
- **The BigQuery client is built once per process.** Loading the credential and
  opening a session took ~2.7s and was paid again on every cache miss; it is an
  `st.cache_resource`, which is what that cache is for.

While the reads are outstanding the page shows a progress bar. It is paced by the
clock rather than by how many queries have answered: a dozen of the fourteen come
back inside the first second and the open-ticket query holds the page for several
more, so a bar driven by the count rushes to nine tenths and then looks broken for
most of the wait. The label alongside it carries the true count, the bar stops at
95% until the last answer lands, and no duration is promised — a slow Jira makes
the bar wait rather than lie. A warm page answers in milliseconds and shows no bar
at all.

## How fresh the numbers are

Everything is read live and cached briefly; nothing is precomputed or exported.
*Refresh data* forces the page being looked at to re-read now.

| Section | Read from | Behind by |
| --- | --- | --- |
| Tickets, resolved/created/triage counts, PRs | Jira and GitHub APIs | up to 5 minutes |
| Sprint statuses, priorities, user directory | Jira API | 10 minutes; the project list, an hour |
| *Orders, Revenue & AOV*, *Best Sellers & Merchants* | the CRM's own Postgres tables | up to 15 minutes |
| *Ads Spend & Return* | Google's Ads-to-BigQuery transfer | **a day**, by design |
| *Cloud costs* | Google Cloud's billing export | up to 6 hours; the export is written in arrears and ends yesterday |
| *Price competitiveness* | Merchant Center's Merchant API | up to 6 hours; Google recomputes benchmarks daily |
| *Product Funnel & Friction* | Amplitude Dashboard API | **a day**, by design |

The order book is the freshest thing here in kind, not just in minutes: it is a
direct read of the CRM's tables, so an order is visible the moment the shop
commits it, and the only delay is the 15-minute cache. The funnel is different —
its window deliberately ends *yesterday*, because today is still being written and
a half-finished day reads as a slump. So the funnel is never a report on today,
whatever the cache does, and Amplitude's own ingestion lag sits on top of that.
The ads figures end yesterday for the same reason and arrive once a day from
Google's transfer, so their cache is measured in hours rather than minutes: there
is no fresher answer to fetch, and *Refresh data* is there for anyone who wants to
prove it.

Caches refresh in the background, which trades a little more staleness for never
making a reader wait: once a TTL lapses the next visitor is served the old answer
straight away while the replacement is fetched behind them, so a figure can be
slightly older than the interval above until that refresh lands.

Interaction cost is handled separately, because Streamlit reruns the whole script
on every widget change:

- Sections that own their widgets (*Ticket Composition*, *Teams*, *Epics*,
  *Backlog Cleanup*, *Assignee Breakdown*, *Pull Requests*, *PR Hygiene*, *Ticket
  Quality*, *Prioritized Queue*, *Estimate Policy*, *Stale & Abandoned*) are
  `st.fragment`s, so clicking inside one rebuilds that section instead of all
  twenty.
- The sidebar filters sit behind *Apply filters*: narrowing by status, priority
  and two sliders is one rerun rather than four. Scope stays outside the form
  because it decides which widget appears beneath it.

## Metric definitions

Computed in `transformations.add_ticket_health_fields`:

| Field | Definition |
| --- | --- |
| `ticket_age_days` | Days since `created`. |
| `idle_days` | Days since the last *meaningful* activity — the newest changelog entry that is not a pure ML-sprint rollover, falling back to `updated`. |
| `idle_bucket` | `0-2`, `3-7`, `8-14`, `15+` days. |
| `age_bucket` | `0-7`, `8-30`, `31-90`, `91-180`, `180+` days. |
| `workflow_stage` | Jira status category, or a status-name fallback. |
| `carry_over_count` | Closed sprints a still-open ticket has already passed through. |

## Priority score

`prioritization.add_priority_score` adds `priority_score` (0–100, clipped),
`priority_rank` and `priority_reasons`. It reuses the metrics above so the ranking
never diverges from the health numbers:

| Component | Contribution |
| --- | --- |
| Jira priority | Highest/Urgent 40, High 30, Medium/Normal 18, Low 8, Lowest 4, Idea 2, and 0 for an unprioritised ticket — whether Jira holds the priority literally named `None` or no priority at all, which the rest of the dashboard also treats as one bucket; an unrecognised name scores 4 |
| Idle time | up to 20, linear over `idle_days` saturating at 30 days |
| Ticket age | up to 10, linear over `ticket_age_days` saturating at 180 days |
| Sprint carry-over | 5 per closed sprint carried, capped at 15 |
| Due date | 20 overdue, 15 within 3 days, 10 within 7, 5 within 14, else 0 |
| Late-stage staleness | 10 when in `IN DEV ENV`, `Review in Staging` or `Ready for Production` and idle > 6 days |

Tune the weights via the constants at the top of `prioritization.py`.

## Checks

    pip install -r requirements-dev.txt
    python -m pytest -q                 # everything
    python -m pytest -q -m "not slow"    # the readers and rules only

No credentials are needed and none are used: Jira, the order book, Merchant
Center, Google Ads, Amplitude, Stripe and OpenAI are each answered by a stub, so
a run tests the code rather than the live shop. `tests/` holds the readers and
their rules; `tests/apptests/` renders whole pages through Streamlit's `AppTest`
and asserts on what the tiles and sentences say, which is slower and marked
`slow`. Both run on every pull request (`.github/workflows/checks.yml`).

## Other scripts

- `get_jira_projects.py` — list the Jira projects visible to the configured account.
- `notebooks/eval_jiraprojects.ipynb` — exploratory ticket evaluation.
- `optional/` — experimental LLM-based ticket parsing helpers.
