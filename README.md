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

### Configuration

Credentials resolve from environment variables first, and fall back to the YAML
profile when they are not all set.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `JIRA_BASE_URL` | yes (env mode) | — | Jira site, e.g. `https://vinovoss.atlassian.net` |
| `JIRA_EMAIL` | yes (env mode) | — | Atlassian account email used for API auth |
| `JIRA_API_TOKEN` | yes (env mode) | — | Atlassian API token ([create one](https://id.atlassian.com/manage-profile/security/api-tokens)) |
| `JIRA_CREDS_PATH` | no | `~/.creds/vinovoss.yml` | YAML fallback credentials file |
| `JIRA_PROFILE` | no | `ML-TEAM-MANAGEMENT` | Profile name inside the YAML file |
| `JIRA_DASHBOARD_JQL` | no | `statusCategory != Done ORDER BY updated ASC` | Ticket scope the dashboard loads |
| `JIRA_TEAM_MEMBERS` | no | `Tam,Mehdi Ordikhani` | Comma-separated defaults for the Team scope |
| `JIRA_MAX_RESULTS` | no | `1000` | Ceiling on tickets fetched per run; the dashboard warns when the result set is truncated |
| `JIRA_BACKLOG_STATUSES` | no | `Backlog` | Comma-separated statuses hidden when *Include Backlogs* is off |
| `JIRA_WEEKLY_HOURS` | no | — | Hours per week each person is available, e.g. `Tam=10,Jal=20,Mehdi Ordikhani=40`; drives *Availability vs Commitment* |
| `JIRA_AUDIT_LOG_PATH` | no | `logs/jira_ticket_changes.jsonl` | Where write-back history is recorded; point at durable storage when containerized |
| `JIRA_BROWSE_BASE` | no | `<resolved Jira site>/browse` | Base URL for ticket hyperlinks; defaults to the site the credentials resolve to |
| `JIRA_TEAM_PROJECTS` | no | — | Which Jira projects form each team, e.g. `Marketplace=MB;App=AS,OA;Design=MAR`; used only where the assignee roster has no answer |
| `JIRA_TEAM_PEOPLE` | no | the VinoVoss roster in `teams.py` | Who sits on each team, e.g. `Design=Robert,Alesya;App=Ali,Farid`; first names match Jira display names, and a `Former staff` team surfaces work still owned by leavers |
| `JIRA_EXTRA_PROJECT_KEYS` | no | — | Extra project keys a PR may reference, e.g. `MDP,WT2`, for projects the account cannot see; used by *PR Hygiene* |
| `PR_STALE_AGE_DAYS` | no | `14` | A PR open longer than this counts as stale |
| `PR_STALE_IDLE_DAYS` | no | `7` | A PR untouched for longer than this counts as stale |
| `MEDUSA_ADMIN_API_KEY` | no | — | Medusa secret API key (Settings → Secret API Keys in the CRM); without it the *Orders, Revenue & AOV* section says so and everything else works |
| `MEDUSA_ADMIN_URL` | no | `https://merchants.vinovoss.com` | CRM admin API the order figures are read from |
| `MEDUSA_STORE_PREFIX_ALIASES` | no | — | Retired product-handle prefixes mapped to a merchant, e.g. `oldprefix=Store Name`; without it that merchant's older sales show as *Unattributed* |
| `AMPLITUDE_API_KEY` | no | — | Amplitude project API key (Settings → Projects → your project); needed with the secret key for *Product Funnel & Friction* |
| `AMPLITUDE_SECRET_KEY` | no | — | The same project's secret key; Amplitude's Dashboard API authenticates on the pair, and the API key alone is refused |
| `AMPLITUDE_API_URL` | no | `https://amplitude.com` | Set to `https://analytics.eu.amplitude.com` for an EU-region project, whose keys the US host refuses |
| `AMPLITUDE_FUNNEL` | no | visit → product page → cart → checkout → payment → order | The funnel's steps as `Label=event_name` pairs, in order, e.g. `Visited=_active,Bought=checkout_order_completed` |
| `DASHBOARD_PASSWORD` | no locally, **yes on Cloud Run** | — | Shared password visitors must enter; unset locally means no gate, unset on Cloud Run (`K_SERVICE` present) refuses to serve at all |

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

gcloud run deploy jira-dashboard \
  --source . \
  --project "$PROJECT" --region "$REGION" \
  --no-allow-unauthenticated \
  --session-affinity --max-instances 1 \
  --set-env-vars "JIRA_BASE_URL=https://vinovoss.atlassian.net,JIRA_EMAIL=<service-account-email>" \
  --set-secrets "JIRA_API_TOKEN=jira-api-token:latest"

# Let the whole Workspace domain in (requires IAP or domain-restricted sharing).
gcloud run services add-iam-policy-binding jira-dashboard \
  --project "$PROJECT" --region "$REGION" \
  --member "domain:vinovoss.com" --role roles/run.invoker
```

If the project is outside an Organization, deploy publicly with the password gate
instead of the IAM binding:

```bash
gcloud run deploy jira-dashboard \
  --source . \
  --project "$PROJECT" --region "$REGION" \
  --allow-unauthenticated \
  --session-affinity --max-instances 1 \
  --set-env-vars "JIRA_BASE_URL=https://vinovoss.atlassian.net,JIRA_EMAIL=<service-account-email>,DASHBOARD_PASSWORD=<shared-password>" \
  --set-secrets "JIRA_API_TOKEN=jira-api-token:latest"
```

A shared password is weaker than Google sign-in: it does not identify who is
looking, cannot be revoked per person, and only throttles guessing. Treat it as a
stopgap until the service can live in an Organization behind IAP.

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

The page splits in two at the top: an **Engineering** tab holding everything below,
and a **Business** tab holding the shop's numbers on their own. They answer a
different question from ticket and PR health, and the sidebar scope and filters do
not apply to them. The Business tab holds:

- *Orders, Revenue & AOV* for the last 7 and 30 days, each against the window
  before it.
- *Best Sellers & Merchants* over a window of 30, 90, 180 or 360 days: the wines
  selling most bottles, and revenue, orders and cancellations per merchant. A
  merchant is read from the handle prefix on each order line (`store_prefix` in
  the CRM), since the admin API exposes no order-to-store link to an API-key
  caller; a line whose prefix matches no current merchant is listed as
  *Unattributed* rather than credited to a guess, and
  `MEDUSA_STORE_PREFIX_ALIASES` maps prefixes a merchant has since retired.

The order book is read once for the whole year and then topped up: each refresh
asks the CRM only for orders placed or changed since the last read, so a year of
trading does not cost a year of rows every time the page loads. That first read
waits for a button, because Streamlit runs the body of every tab on every rerun
whichever one the browser is showing — otherwise a visitor who only wanted the
engineering numbers would still wait for the shop's.

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
   capture lags. Read-only: every call is a `GET /admin/orders`, refreshed at most
   every fifteen minutes and incrementally after the first read. Needs
   `MEDUSA_ADMIN_API_KEY`. Below it, **Best Sellers & Merchants** ranks wines by
   bottles sold - not by revenue, which would answer which bottle is dearest
   rather than which is wanted - and breaks revenue, orders and cancellations out
   per merchant, over 30, 90, 180 or 360 days. Bottles count what customers chose,
   so an order awaiting payment counts there while revenue still waits for
   capture; ice packs are add-ons and are kept out of the wine ranking.
12. **Product Funnel & Friction** (*Business* tab) — how far visitors get towards
   being one of those orders, read from Amplitude. The default funnel starts at any
   activity and runs product page → cart → checkout → payment → order, deliberately
   *not* home page → search → product: on this shop the overwhelming majority of
   visitors arrive on a product page from a search engine and never see the home
   page, so a funnel starting there would describe a few hundred people out of tens
   of thousands. Counted in people rather than visits, steps must happen in that
   order within 7 days of each other (wine is read about and bought later, so a
   one-day window would report the shop as worse than it is), and the window ends
   *yesterday*, because today is still being recorded and always reads as a slump.
   *From previous step* is the column to act on — it names the single screen costing
   the most — while *from the start* is what people mean by "conversion rate".
   Beside it, **What went wrong** counts the people who hit an app error, a failed
   add-to-cart, a blocked checkout, a failed payment or an empty search, as a share
   of everyone who visited; one person meeting the same error ten times is one
   person to apologise to, not ten. **Voss AI** is reach rather than engagement: how
   many people opened it, asked it something, and got nothing back. Needs
   `AMPLITUDE_API_KEY` and `AMPLITUDE_SECRET_KEY`; read-only, every call is a `GET`.
13. **PR Hygiene** — open PRs across the organization that are untraceable, stalled
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
14. **Ticket Quality & Ready for Devin** — every ticket graded out of 5 on whether
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
15. **Sprint Planner** — a first draft of one team's next sprint, built from goals
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
16. **Bubble chart, Sprint Capacity, Suggested First Action** — the existing
   age-vs-idle chart, sprint planning tables, and bulk Jira write-back actions.
17. **Availability vs Commitment** — inside Sprint Capacity. Committed estimate hours
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

## Other scripts

- `get_jira_projects.py` — list the Jira projects visible to the configured account.
- `notebooks/eval_jiraprojects.ipynb` — exploratory ticket evaluation.
- `optional/` — experimental LLM-based ticket parsing helpers.
