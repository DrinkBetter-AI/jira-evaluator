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

`main()` now builds an `st.navigation` over two callable pages (Engineering, Business) and runs
the selected one, so a harness that patches `app.fetch_tickets` and calls `dashboard.main()` still
works — but only the Engineering page renders unless you click through, and the Business page is
listed only when the order database or Amplitude is configured. To exercise one page directly,
call `app._render_engineering_page()` or `app._render_business()` from your own script instead.

**Faster than a browser: `streamlit.testing.v1.AppTest`.** `AppTest.from_file("app.py")` runs the
whole page headlessly against whatever credentials are in the environment, and `at.exception`,
`at.error`, `at.subheader` and `at.metric` give you assertions without clicking anything. Widgets
can be driven directly — `find(at.toggle, "Allow Jira edits").set_value(True).run()`,
`find(at.radio, "View").set_value("Team").run()`, `find(at.button, "Apply filters").click().run()`
— which is the cheapest regression net for the scope, filter and write-arming paths, and timing
`at.run()` measures cold load and rerun cost directly. Note form submit buttons appear in
`at.button`, not under a `form_submit_button` type.

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

**Transition-status sampling.** This lookup only runs when the sidebar's **Allow Jira edits**
toggle is armed — it costs one Jira request per key and read-only visitors cannot act on the
answer, so it is not on the default load path. Arm the toggle first or the log file below stays
empty and every case reads as a pass. When it does run, the keys go out concurrently through
`JiraClient.get_issue_transitions_bulk`, so the log's *order* means nothing; assert on the set.

Make the `fetch_available_transition_statuses` stub log the exact
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

## Live-data testing (when working credentials are available)

When features depend on real Jira/GitHub numbers (e.g. the resolved-snapshot tiles, the Pull
Requests section, or a Devin-able heuristic you want to check against real ticket summaries), test
against live data rather than the synthetic harness. Two credential-plumbing gotchas bite here:

- **A stale base-shell `JIRA_API_TOKEN` shadows your binding.** The environment may already export
  a `JIRA_API_TOKEN` that 401s. Binding the working secret to a var literally named
  `JIRA_API_TOKEN` via the tool `env` does not reliably override it. Instead bind the secret to a
  **non-colliding** name (e.g. `MYTOK`) and, inside the launch command, assign
  `JIRA_API_TOKEN="$MYTOK"` so it is set from the process's first moment. Verify with
  `GET /rest/api/3/myself` → 200 before launching. Pair the session token with its matching
  `JIRA_EMAIL` and the tenant's `JIRA_BASE_URL` (whatever the working credential belongs to — get
  these from the session secret store / deployed env, not hardcoded here).
- **GitHub token: don't blank it.** `github_client.load_github_env()` checks
  `DASHBOARD_GITHUB_TOKEN` → `GITHUB_TOKEN` → `GH_TOKEN`, taking the first **non-empty** value
  (blank/unset vars are skipped, so `GITHUB_TOKEN=""` falls through to `GH_TOKEN` rather than
  breaking anything). Get a working token with `GITHUB_TOKEN=$(gh auth token)` (the box's `gh` is
  authenticated to the org, read is enough). GraphQL calls 401 only if all three are blank/unset or
  the found token is invalid — so leave the base `GH_TOKEN` intact as a fallback.

Launch in its own process so a `pkill` in the same shell can't take out your session:
`setsid venv/bin/streamlit run app.py --server.port 8501 --server.headless true > /tmp/log 2>&1 < /dev/null &`.

**Pre-compute expected values with the real clients before the UI run** (small probe scripts using
`JiraClient.approximate_count` and the GitHub client), so the UI numbers are checked against known
values, not eyeballed. These numbers drift daily as tickets/PRs land, so **always re-derive them
with the probe scripts** — never assert against a frozen figure. As one dated sanity snapshot
(2026-08, order-of-magnitude only): Jira resolved 7d/30d ≈ 1900s/2100s (Jira *approximate* counts —
the UI says so; not capped at `JIRA_MAX_RESULTS`), PRs merged 7d/30d in the ~90/~300 range,
open/stuck/never-reviewed PRs ≈ 60s/50s/0.

**Resolved tiles vs. pie sample caption.** The ticket tiles come from Jira's `approximate-count`
endpoint and are independent of the fetched frame size. The "Pie shows a N-ticket sample of ~M
resolved (fetch limit); ticket tiles are Jira's approximate counts." caption only renders when the
fetched frame is **smaller** than the exact count. So at `JIRA_MAX_RESULTS=3000` (≥ the count) the
caption does not appear — to demonstrate it, run a second instance with a small cap (e.g.
`JIRA_MAX_RESULTS=150`) and confirm the tiles stay 1949/2134 while the caption shows a 150-sample.
This dual run is also the cleanest anti-regression proof that the tile is not the paging cap.

**Stuck-PR spot-check.** The browser is not signed into GitHub (private repos), so a clicked
`/pull/N` link renders a GitHub sign-in page — the address bar still proves the link target is
correct. To prove a stuck PR genuinely lacks an approving review, use the authenticated CLI:
`gh api repos/<org>/<repo>/pulls/<n>/reviews --jq 'group_by(.state)|map({state:.[0].state,count:length})'`
and confirm no `APPROVED` state (COMMENTED/CHANGES_REQUESTED still count as stuck).

**Devin-able? Story-type regression.** The NO-keyword scan must run on the **summary only**; the
Jira issue type ("Story", "Design") must not leak into it, or engineering Story tickets get
mislabeled. A good live discriminator is a Story like MB-5591 "Migrate Axios to Fetch and refactor
authService" which must read **Yes** (the `migrat` prefix on the summary wins; the "Story" type does
not force it to No/Maybe). Sort the board by the Devin-able? column (Sort-by selectbox or header
click) and confirm every value is one of Yes/No/Maybe. The board is wide — zoom the browser out
(ctrl+minus) so the Key and Devin-able? columns are visible together in one screenshot.

## Testing the PR Hygiene section (and any GitHub-backed section)

- **Recompute expectations at the same moment as the UI run.** Open-PR data drifts hourly: during
  one run a PR merged between the pre-computation and the UI load, moving open PRs 60 → 59 and the
  "No Jira key" tile 14 → 13. Re-run the probe script immediately after loading the page and diff
  the two sets (`set(zip(repo, number))`) to attribute any mismatch to drift rather than a bug,
  instead of declaring a failure.
- **Probe script shape** (module-level functions, there is no `GitHubClient` class):
  `from github_client import fetch_open_prs, load_github_env` → `token, org = load_github_env()` →
  `fetch_open_prs(token, org)` → `pr_hygiene.add_hygiene_fields(prs, project_keys)`. `fetch_tickets`
  needs all of `creds_path, profile_name, jql, max_results, page_size, schema_version` as keywords.
- **Find the discriminating rows offline first.** Diff known-key matching against
  `pr_hygiene._GENERIC_KEY_RE`: rows where they disagree are the false-positive evidence (e.g. a PR
  body containing `UTF-8`, or a branch containing `DEVIN-2747`) and must *stay* in the "No Jira key"
  tab. Also list PRs whose key comes only from `branch` and only from `body` — those must be absent
  from that tab. This is the only way to test key detection without hand-reading 60 PRs.
- **Config fixes are often tile-invisible.** Widening the project-key list (ticket keys → Jira's
  project list → including archived projects) changed the counts by zero on a real org, because no
  open PR referenced the newly-added projects. Assert such fixes via the section caption
  ("matched against N known project keys", app.py `_render_pr_hygiene`) and via targeted
  `find_jira_key` calls (`mb-1234-fix-login` → `MB-1234`) rather than chasing a count change.
- **One instance per env-var permutation** beats restarting: `8600` token + defaults, `8601` no
  token, `8602` `PR_STALE_AGE_DAYS=1`, `8603` `PR_STALE_AGE_DAYS=7d`. Threshold changes are read at
  import, so they cannot be exercised without a separate process. A garbage threshold should leave
  the tile label at the default (`Stale (>14d old or >7d idle)`); the label itself is the assertion.
- **No GitHub token at all? Run a hybrid harness: synthetic PRs + live Jira.** Any PR-hygiene
  behaviour that joins PRs to tickets (e.g. `pr_hygiene.critical_in_flight`) can still be driven
  end-to-end. Copy the ticket harness pattern but patch only the GitHub side, leaving the Jira
  fetch live and read-only:
  ```python
  import app as dashboard
  dashboard.github_client.load_github_env = lambda: ("synthetic-token", "OrgName")
  dashboard.fetch_open_prs_cached = lambda *a, **k: PRS.copy()
  dashboard.fetch_open_pr_count_cached = lambda *a, **k: len(PRS)
  dashboard.fetch_merged_prs_cached = lambda *a, **k: pd.DataFrame()   # merged charts tolerate empty
  dashboard.fetch_merged_pr_count_cached = lambda *a, **k: 0
  dashboard.main()
  ```
  The synthetic PR frame needs `number, title, url, is_draft, branch, body, review_decision,
  created_at, updated_at, author, repo, approving_reviews, changes_reviews, total_reviews,
  review_requests, age_days, idle_days` (mirror `github_client._to_frame`). Point the PRs at **real
  live ticket keys** picked from a probe of the Jira frame, so the join is exercised against real
  priorities/statuses. Guard the harness with `if os.getenv("DEMO_BUILD_ONLY") == "1": raise
  SystemExit(0)` before `main()` — Streamlit only executes the script when a browser session
  connects, so an expectation script cannot rely on the app process having written the frame cache.
- **Make excluded rows the *idlest/oldest* ones.** Tables like the critical tab sort by
  `idle_days` desc: if every deliberately-excluded PR (wrong status, wrong priority, no key,
  key with no matching ticket) has a bigger idle than every included PR, a broken filter cannot
  produce a passing-looking screenshot — the leak lands in row 1.
- **Link columns:** `st.column_config.LinkColumn(display_text=JIRA_KEY_DISPLAY_PATTERN)` shows the
  bare key. To prove the href without an Atlassian session, click it: Jira bounces to
  `/jira/get-started?continueUrl=<url-encoded browse URL>`, and the address bar is the assertion.
- **GitHub token:** `DASHBOARD_GITHUB_TOKEN=$(gh auth token)` works but expires in ~1 hour — keep a
  relaunch script (see `/home/ubuntu/launch_pr16.sh` pattern) and re-run it for long sessions.
- **Scrolling Streamlit pages full of dataframes:** the wheel is captured by whichever `st.dataframe`
  is under the cursor, silently scrolling table rows instead of the page. Put the cursor in the
  right-hand margin (e.g. x≈990) to scroll the page, and over the table only when you deliberately
  want more rows.

## Testing the Ticket Quality / scoring section (and any per-ticket derived column)

- The section is rendered from `ticket_quality.score_tickets(filtered)` and is passed **`filtered`**,
  not `_metrics_df(...)`, so Backlog tickets are always included regardless of *Include Backlogs* —
  don't expect the tiles to move with that checkbox.
- Build the expectation offline by importing the app's own modules against the live fetch:
  `PYTHONPATH=<repo> python expect.py` with
  `app.fetch_tickets(creds_path=..., profile_name=..., jql=app.JQL, max_results=app.MAX_RESULTS,
  page_size=100, schema_version=app.FETCH_SCHEMA_VERSION)` then `ticket_quality.score_tickets(df)`.
  Note the script's own directory (not the cwd) goes on `sys.path`, so `PYTHONPATH` is required if
  the script lives outside the repo.
- **The strongest single check for a description-dependent feature is the description coverage
  itself** (e.g. "638/699 tickets have a non-empty description"). If ADF conversion or the fetched
  field list silently broke, every ticket loses `has_description`/`has_acceptance`, the tiles
  collapse toward 0 and the average toward ~2 — so non-zero, plausible tiles are meaningful.
- Scoring invariants are cheaper to prove from the exported CSV than from the UI: download the CSV
  and assert row count == gradable, the value mix (Yes/Maybe/No), absence of exempt container keys,
  and structural relations such as *number of named gaps == 5 − score* across **all** rows. Use the
  UI for a handful of visible rows and the CSV for exhaustiveness.
- Exemption checks (Epics/Initiatives must not be graded) are best done with the `st.dataframe`
  **search** in the table toolbar: hover the table, click the magnifier (icons sit just above the
  table's top-right corner, ~x=910/924/937/950 at 1024px width), then click the *"Type to search"*
  text itself before typing — clicking elsewhere in the toolbar leaves focus on the tab and the
  keystrokes go nowhere. `0 results` for a known Epic key is direct evidence of the exemption.
- Long text columns (`Missing`) are truncated by column width and **cannot** be widened by dragging
  the header separator or by `ctrl+-`; the table's fullscreen button helps a little, but plan to
  confirm the full strings from the CSV.

## Testing the Sprint Planner (and any capacity/roster-driven section)

- **`JIRA_WEEKLY_HOURS` is mandatory** or the section short-circuits to "Nobody has declared
  hours ...". Run one instance with it and one without (the warning case is a real assertion).
  Exercise it in **short spellings** (`Farid=20`, `Mehdi=40`) rather than full Jira display
  names — short-name resolution against `JIRA_TEAM_PEOPLE` roster entries is the fragile path
  and has broken twice (ambiguity emptying the capacity frame; lowercase roster labels showing
  as a second person). A roster-only person with declared hours should appear as spare capacity
  labelled with the *declared* spelling.
- The planner is fed the **unfiltered** frame, so sidebar scope/filters do not move it; assert it
  still renders under each scope rather than expecting its numbers to change.
- **Recompute everything offline against the same snapshot.** Import `sprint_planner` and mirror
  the section: `match_goals` → `person_capacity(names, weekly, days, overhead_per_week)` →
  `plan_sprint` → `goal_load` / `plan_load`. Capacity arithmetic (`weekly/5*days` minus
  `overhead/5*days`) renders as a plausible number even when it is off by a whole day, so an
  independent figure is the only way to catch it. Drive the knobs from env vars in the script so
  each UI state has a matching expectation.
- Pick knob values that make effects unmistakable: overhead **4 → 20 h/week** zeroes a 20h/week
  person and forces rows out; assumed hours **4 → 10** separates estimated from unestimated rows.
- **Editor-key identity.** The `st.data_editor` key mixes team + goals + knobs + a SHA-1
  fingerprint of the plan's ordered ticket keys. Assert both halves: an edit must *survive*
  `Refresh Data` when the ticket set is unchanged, and must be *discarded* when team/goals/knobs
  change (tick → switch team → switch back → computed default, with no tick landing on a
  different ticket). Making live data change under fixed knobs is generally not possible
  read-only; say so rather than claiming that half.
- **In-flight work is special-cased**: it survives the "Only goal work" filter and, if it costs
  more than its owner's remaining hours, it is still planned with the budget floored at zero
  (*Why* reads `already in flight, Xh against Yh left`) and the person shows negative "Left (h)"
  plus an "Over their hours" warning. Combine goals + a high overhead to reach that state.
- Ordering claims are cheapest to prove from the exported CSV: `priority_score` is in the CSV but
  not in the on-screen editor, so "priority does not beat goal rank" is only checkable there.
- Warnings that live on the write path may still render **before** the edit-switch check (e.g.
  "these tickets would leave another open sprint"). Read the code to see whether a warning is
  gated on writes being enabled; if it is not, it can be verified without arming anything —
  typically by choosing a team whose tickets span two open sprints.
- Reruns after a knob change or a checkbox tick can take 10-20s on a ~700 ticket instance; wait
  for the greyed-out overlay to clear before screenshotting, or you will capture the old table.

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
