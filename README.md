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
| `JIRA_TEAM_MEMBERS` | no | `Tam,Shivanand,Mehdi Ordikhani` | Comma-separated defaults for the Team scope |
| `JIRA_MAX_RESULTS` | no | `1000` | Ceiling on tickets fetched per run; the dashboard warns when the result set is truncated |
| `JIRA_BACKLOG_STATUSES` | no | `Backlog` | Comma-separated statuses hidden when *Include Backlogs* is off |
| `JIRA_WEEKLY_HOURS` | no | — | Hours per week each person is available, e.g. `Tam=10,Shivanand=20,Mehdi Ordikhani=40`; drives *Availability vs Commitment* |
| `JIRA_AUDIT_LOG_PATH` | no | `logs/jira_ticket_changes.jsonl` | Where write-back history is recorded; point at durable storage when containerized |
| `JIRA_BROWSE_BASE` | no | `<resolved Jira site>/browse` | Base URL for ticket hyperlinks; defaults to the site the credentials resolve to |
| `JIRA_TEAM_PROJECTS` | no | — | Which Jira projects form each team, e.g. `Marketplace=MB;App=AS,OA;Design=MAR`; used only where the assignee roster has no answer |
| `JIRA_TEAM_PEOPLE` | no | the VinoVoss roster in `teams.py` | Who sits on each team, e.g. `Design=Robert,Alesya;App=Ali,Farid`; first names match Jira display names, and a `Former staff` team surfaces work still owned by leavers |
| `DASHBOARD_PASSWORD` | no locally, **yes when publicly reachable** | — | Shared password visitors must enter; unset or blank means no gate at all |

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
  --set-env-vars "JIRA_BASE_URL=https://vinovoss.atlassian.net,JIRA_EMAIL=<service-account-email>" \
  --set-secrets "JIRA_API_TOKEN=jira-api-token:latest" \
  --update-env-vars "DASHBOARD_PASSWORD=<shared-password>"
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
6. **Backlog Cleanup** — the oldest open tickets presented one at a time, because a
   backlog is triaged in single decisions rather than in a table of 200 rows. Each
   card carries the ticket's age, idle time, owner, status, epic and priority, with
   the signals arguing for closure highlighted, and a suggested decision derived in
   `cleanup.suggest_decision`: a ticket a year old and untouched for six months, or
   one nobody ever took that has sat 90 days, is proposed for closing; anything else
   defaults to *Keep*. Two queues: oldest open, and oldest unassigned. Decisions live
   in the session and are exportable as CSV; only the explicit *Apply* button at the
   bottom writes to Jira, transitioning the tickets marked *Close* to a status you
   name, through the same audited path as every other write.
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
11. **Bubble chart, Sprint Capacity, Suggested First Action** — the existing
   age-vs-idle chart, sprint planning tables, and bulk Jira write-back actions.
12. **Availability vs Commitment** — inside Sprint Capacity. Committed estimate hours
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
| Jira priority | Highest/Urgent 40, High 30, Medium/Normal 18, Low 8, Lowest/unset 4 |
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
