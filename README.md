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
| `JIRA_BROWSE_BASE` | no | `<resolved Jira site>/browse` | Base URL for ticket hyperlinks; defaults to the site the credentials resolve to |

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

## Dashboard layout

1. **Scope** — `Organization` (every assignee returned by the JQL), `Team`
   (multi-select pre-filled from `JIRA_TEAM_MEMBERS`), or `Individual` (one assignee).
2. **Filters** — status, priority, minimum idle days, minimum ticket age, and an
   *Include Backlogs* toggle that all downstream sections respect.
3. **Metrics** — open ticket count, average/max idle days, oldest ticket age,
   estimate coverage, and tickets stale in a late stage.
4. **Assignee Breakdown** — per-assignee roll-up (open tickets, average and top
   priority score, average and max idle days, tickets idle 15d+, tickets with no
   priority). In `Individual` scope this collapses to that person's headline numbers.
5. **Prioritized Queue** — tickets ranked by priority score with the reasons behind
   the score, and a CSV export.
6. **Bubble chart, Sprint Capacity, Suggested First Action** — the existing
   age-vs-idle chart, sprint planning tables, and bulk Jira write-back actions.

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
