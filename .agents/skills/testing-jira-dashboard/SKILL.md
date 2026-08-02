---
name: testing-jira-dashboard
description: How to run and test the Streamlit Jira ticket health dashboard end-to-end without Jira credentials, using a synthetic monkeypatch harness, plus how to validate scope filtering, the prioritized queue and rollups against independently computed expectations.
---

# Testing the Jira ticket health dashboard

## What the app is

`streamlit run app.py` renders a Jira ticket health dashboard. It normally pulls live data through
`jira_client.py`, which needs `JIRA_BASE_URL` / `JIRA_EMAIL` / `JIRA_API_TOKEN` (env first) or a
YAML credentials file (default `~/.creds/vinovoss.yml`). Without either, the page renders a clean
red `Configuration error: ...` banner instead of a traceback — that banner is itself a useful
assertion for "no creds" behaviour.

## Running without Jira credentials (the normal case for agents)

Do not try to reach a real Jira instance. Instead run an **untracked** harness next to `app.py`
that imports the dashboard module and monkeypatches its fetch functions before calling
`dashboard.main()`:

- patch `app.fetch_tickets` to return a synthetic `pandas.DataFrame`
- patch `fetch_all_priorities`, `fetch_all_users`, `fetch_available_transition_statuses`
- then call `dashboard.main()`

Run it with `streamlit run jira_demo_app.py --server.port <port>`. Keep the harness untracked and
never commit it. Cache the generated frame (e.g. a pickle in `/tmp`) so the UI and your expectation
script read *identical* data.

Requirements: streamlit must be `>=1.49` (older versions crash on `width="stretch"`).

## Make the synthetic data adversarial

A good default frame is ~60 rows containing: Backlog and non-Backlog statuses, several named
assignees **plus null assignees**, missing due dates, very large idle/age values, non-ASCII
summaries, sprint + carry-over fields, and rows with no priority. Also generate variant datasets
selected by an env var (e.g. `DEMO_DATASET=no_assignees` for an all-null-assignee frame, empty
frame, single assignee) — these expose the interesting edge cases.

## Start several instances at once, one per config

Config is read from the environment at startup, so testing env-var behaviour means one Streamlit
process per config, each on its own port, e.g.:

| Port | Config |
| --- | --- |
| 8501 | default `JIRA_TEAM_MEMBERS` |
| 8502 | `JIRA_TEAM_MEMBERS="<two names present in the data>"` |
| 8503 | `JIRA_TEAM_MEMBERS="Nobody,Ghost"` (names absent from data) |
| 8504 | adversarial dataset (all-null assignees) |
| 8505 | plain `streamlit run app.py`, no harness, no creds |
| 8506 | `DEMO_DATASET=bulk` (70 rows, all in one active sprint, all priority `None`) |
| 8507 | full env credential triple (`JIRA_BASE_URL`+`JIRA_EMAIL`+`JIRA_API_TOKEN`) + YAML profile |
| 8508 | `JIRA_BASE_URL` only (a deliberately wrong host) + YAML profile |
| 8509 | `DEMO_DATASET=mixed` (few sprint members, non-members holding extra statuses) |
| 8510 | `DEMO_DATASET=emptysprint` (sprint's only member is an Epic → zero editor rows included) |
| 8511 | `DEMO_DATASET=bulkstatus` (40 rows in one status, 6 in another) |
| 8512 | `JIRA_MAX_RESULTS` set below the frame size (truncation-warning smoke) |

Then just navigate between `localhost:850x` tabs in the browser. If a port is busy, find and kill
the stale process before restarting.

## Verify numbers, don't eyeball them

Write a small script that loads the *same* cached frame and calls the library functions directly
(`transformations.add_ticket_health_fields`, `prioritization.add_priority_score`,
`prioritization.assignee_rollup`) to produce expected totals, per-assignee rollups and the top-N
priority queue. Compare those to what the UI shows. Also hand-check one score against the weights
table in `README.md` (priority weight + idle + age + carry-over + due-date urgency + late-stage
staleness, clipped to 100) so you're validating the documented contract and not just the code
against itself.

## Scope semantics worth asserting

The scope resolver distinguishes "no filter" from "explicit empty selection". Regression-test both:

- Organization → no assignee filter (caption states the assignee count).
- Team with members selected → filtered counts.
- Team with the multiselect **cleared** → warning + **zero** tickets everywhere downstream
  (metrics, Assignee Breakdown, Prioritized Queue, bubble chart, Sprint Capacity). If org-wide
  numbers appear here, the sentinel/`None` handling has regressed.
- `JIRA_TEAM_MEMBERS` containing names absent from the data → empty + warned, not org-wide.
- Individual scope on a frame with no assignees → warning, no traceback.

## Recipes for the write-path / lookup behaviours

**Transition-status sampling.** Make the `fetch_available_transition_statuses` stub log the exact
key tuple it receives to `/tmp/transition_keys_<dataset>.json` and return a couple of values that
exist *only* in the transition response (e.g. `Blocked`, `Done`). Then the log file gives you an
exact assertion on which keys were sampled, and seeing those transition-only values in a Sprint
Tickets **Status** cell dropdown proves the lookup actually ran. Cases worth covering:

- all rows in-sprint → log is capped (`TRANSITION_LOOKUP_LIMIT`, currently 50) and contains only
  in-sprint keys;
- mixed frame → in-sprint keys **plus** the lowest key of each status no in-sprint row holds, and
  *not* extra keys for statuses already covered;
- sprint with zero included editor rows → the log must NOT be empty (one key per displayed status),
  otherwise the dropdown loses all reachable statuses;
- **sprint with ≥ `TRANSITION_LOOKUP_LIMIT` members** whose statuses do not cover some outside
  row's status → the outside key must still be in the log. This is the boundary that a
  "members first, then per-status" implementation silently fails: the members alone fill the
  budget and the per-status pass is never reached. Build a frame with e.g. 55 members using only
  two statuses plus two non-members carrying unique statuses and one non-member with an
  already-covered status (which must NOT be reserved). Compute the expected list offline by
  importing the real function against the frame, and compare old vs. new behaviour so the case is
  actually discriminating before you run the UI.

**Making the Status dropdown discriminating.** The dropdown is the union of *every displayed
status* and the transition response, so its contents alone never prove which keys were sampled.
Have the stub return a per-key marker option (`{"ML-7900": "Escalated (via ML-7900)", …}`) for the
keys it was asked about; then typing `via` in the Status cell dropdown filter shows exactly the
sampled keys. Include a marker for a key that *should* be over budget so its absence is evidence
too.

Note the sample is computed from the editable `include` column, so it follows what the user has
ticked, not immutable sprint membership; it is rewritten on every rerun, so read the log right
after loading the state you want to assert on.

**Bulk pre-selection caps.** Both Suggested First Action paths (`Set None-priority tickets` and
`Change status`) pre-select at most `BULK_ACTION_DEFAULT_LIMIT` (25) and show
"Only the first 25 are pre-selected; add more explicitly if you mean to update them." Assert both
the >25 case (25 chips + caption, and the non-selected keys still *offered* in the dropdown) and
the <25 case (all chips, no caption). Never click Apply.

**Browse-URL precedence.** `JIRA_BROWSE_BASE` wins; otherwise the env host is used only when the
full triple is present; otherwise the YAML profile `base_url`; otherwise a hard-coded default. Two
discriminating instances (full triple vs. `JIRA_BASE_URL` only, with a YAML profile pointing at a
*different*, obviously named host such as `yaml-fallback.atlassian.net`) is the only way to catch a
regression here. You will not be logged into Jira, so assert on the **address bar** after clicking a
queue Key link — an Atlassian login redirect still carries the target URL in its `continue=` param.

**Result-cap warning.** Launch one instance with `JIRA_MAX_RESULTS` below the synthetic frame size
and confirm the warning names the cap, the JQL ordering and both remedies. The actual truncation
happens in the real fetch path, so the harness only proves the warning branch.

## Asserting the priority-score formula

Scoring changes are easiest to prove with a tiny purpose-built frame where every input except the
one under test is identical — e.g. a `duedate` dataset of three `To Do` / `Normal` rows differing
only by a date-only due date (today / yesterday / +10d). Then the Prioritized Queue's **Score**
column isolates `_due_pressure` exactly (20 overdue / 15 <=3d / 10 <=7d / 5 <=14d), and a *delta*
between two rows is a far stronger assertion than an absolute number. Compute the expected values
offline first by importing `transformations.add_ticket_health_fields` + `prioritization.
add_priority_score` and running the builder directly — the `/tmp/demo_tickets_<dataset>.pkl` cache
only appears after a browser session has loaded that port.

Note the **Why** column is driven separately by `prioritization._reasons`, whose thresholds may not
match the score buckets (e.g. "due date at risk" fires at `due_pressure >= 15`, so a due-*today*
ticket still shows it). Assert score and reason independently rather than assuming they move
together.

## Gotchas

- The Streamlit page has no browser chrome if the window is in fullscreen; press `F11` before using
  the address bar, otherwise `ctrl+l` + typing lands inside whatever Streamlit widget has focus.
- Prefer a separate browser tab per port so widget state on the main instance is preserved.
- Suggested First Action should derive its candidate rows from the same frame as the headline
  metrics (`_metrics_df(filtered, include_backlogs)`), so both the "N ticket(s) ... have no priority
  set" list and the `Change status` *From status* options must shrink/grow with the `Include
  Backlogs` checkbox. This regressed once (the section read the raw filtered frame and let bulk
  writes reach hidden Backlog rows), so re-check both directions whenever that section changes:
  toggle the checkbox and assert the key list changes *and* that `Backlog` leaves/enters the
  *From status* dropdown.
- Restarting the instances: run the launcher in its own process
  (`setsid nohup ./launch_instances.sh >/tmp/launcher.log 2>&1 </dev/null & disown`). Chaining
  `pkill -f "streamlit run jira_demo_app.py"` with the relaunch in one shell command kills the shell
  itself and leaves nothing running.
- Avoid clicking any "Apply ..." button — those are the Jira write paths. "Change History and
  Revert" showing "No write operations have been logged yet." is good end-of-run evidence that
  nothing was written.

## Devin Secrets Needed

None for the synthetic path. For live-Jira testing you would need `JIRA_BASE_URL`, `JIRA_EMAIL`
and `JIRA_API_TOKEN` (or the YAML credentials file); these were not available in this environment.
