# DECISIONS.md — what 24 agents built, and what they decided in your absence

Written for Angel after the parallel implementation of `IMPLEMENTATION_PLAN.md`.
Twenty-four agents worked across Phases 1–6, each forbidden from asking a
question — when something was ambiguous, they picked the most reasonable
option, implemented it, and recorded the choice in `docs/assumptions/*.md`
(24 files, one per task). This is the consolidated version of those 24 files
plus a spot-check against the actual code (branch
`feat/dashboard-implementation`, full suite green: 1,044 passed, 1 skipped, 0 xfailed).

---

## 1. What shipped

`app.py` (6,847 lines, all seven pages in one file) is now `app.py` (502
lines) plus `render_shared.py` and eight page modules — a mechanical split
that unblocked everyone else and is verified line-for-line identical to the
pre-split behavior. Five read-only pages (Today, Delivery, Code, Planning,
Engineering) are rebuilt as server-rendered HTML off one shared component
kit (`theme_html.py`, 18 components, zero JavaScript, one token system —
`theme_tokens.py` — so a Plotly bar and an HTML bar are finally the same
color). A large amount of compute that existed but was never called got
wired in: resolution credit now follows the changelog author instead of the
current assignee (this is the fix for "Sai Shankar, 194 resolved, second
highest in the company" — a departed tester whose name was still sitting in
an `assignee` field), staleness scoring switched from `idle_days` to
`status_age_days` so a label edit can't reset a stale ticket's clock, and a
16-role rubric system (`roles.py`) replaced the old flat comparison so
nobody is scored against a discipline they don't practice. A new
admin-only Integrity page (`/integrity`, gated behind a second password
`DASHBOARD_ADMIN_PASSWORD` you have not set yet) surfaces four evidence
cards — cosmetic-touch grooming, mid-flight estimate inflation, staging
round-trips, and self-merge/review-reciprocity patterns — each with an
"innocent reading" stated up front and every number linked to the real
Jira ticket or PR. Three new integrations (Slack snippets, Fireflies
standup truth-check, Clockify billed-vs-delivered) are built and tested
against fixture data end to end, but none has a live credential yet, so
none is wired into a page.

---

## 2. Decisions you should look at

Ranked by consequence, not by which phase built them.

### 2.1 Resolution credit follows the changelog author, not the current assignee
**File:** `integrity.py`, `data_layer.py` (2A) · **Reversal cost: low, but reversing it un-fixes the bug it exists to fix.**
The whole reason this project started (a departed tester credited with 194
resolutions) is fixed by joining resolved tickets against `expand=changelog`
and crediting whoever's name is on the resolving transition, not whoever
currently sits in `assignee`. A resolution credited to someone on the
former-staff list is **flagged, not silently dropped** — it still shows up
with `is_former_staff=True` so you see "this account is still authoring
transitions" instead of a number that just quietly changed. This is wired
into the People table and the Delivery page's "who resolved tickets"
ranking. It is **not yet wired into** `data_layer.py:_derive_board`'s
`priority_score`/`priority_rank` (the board-wide sort everyone sees), into
sprint planning's ranking, or into `hygiene.py`'s abandoned-ticket queue,
`cleanup.py`'s triage sort, `kpi.py`'s "Weekly updates"/"Staleness"
components, or the epic/team rollups — all of these still read the old,
gameable `idle_days`/assignee-based numbers. `docs/assumptions/2A.md` lists
every one of these call sites by file and line. This is the single highest-
value follow-up: the credit fix exists and works, but half the dashboard
hasn't been pointed at it yet.

### 2.2 Peer comparison cohorts are almost all too small to compare
**File:** `roles.py`, `people_table.py`, `pages/people.py` (2C, 2B, 3B) · **Reversal cost: none — this is a fact about your headcount, not a tunable.**
With `MIN_PEERS = 3` and 12 active engineers spread across 16 role keys,
only one cohort clears the bar today: `frontend`/`frontend-mobile`/`mobile`
folded together (David, Mohsen, Farid, Ali — 4 people). Every other role
(platform: just Tam; backend: just Shawn; ai-recommendation: just Mehdi;
...) shows "insufficient peers" with the actual peer count stated, not a
percentile. This was verified against the live default roster, not assumed.
`IMPLEMENTATION_PLAN.md §13` itself says this is correct — "suppressing it
is correct" — and I'd agree: a percentile against 0–2 peers is noise
dressed as data. If you want more comparison to actually happen, the lever
is `roles.MIN_PEERS` (currently 3, a one-line change) or restructuring role
keys to fold more people together — not a code bug to fix.

### 2.3 Five of eleven new rubrics have weights but no scoring function behind them
**File:** `roles.py` (2C) · **Reversal cost: this isn't a decision to reverse, it's unfinished work to finish.**
QA-automated, QA-manual, PM, designer, and infrastructure all got a
`Rubric` (weights, blind spots, the whole table) designed from scratch —
there was no prior art anywhere in the codebase for any of them. But
`roles.overall()` is the generic scoring engine; nothing computes the actual
component values ("Triage latency," "Board hygiene," "Handoff completeness")
from real ticket data yet. Only the `code` rubric (7 of the 16 roles) is
fully wired end to end. Anyone in a non-code role sees their rubric's name
and structure but not a live score — this is stated honestly on the People
page as "no component-computation function yet," not faked. If you want
Mihai, Gaston, Robert, Alesya, Santi, or Dina scored, that's the next
compute task, and the weight tables are already there to build against.

### 2.4 Two independent access gates, and the Integrity page is discoverable by URL regardless of login
**File:** `access_gate.py`, `app.py` (Phase 4) · **Reversal cost: low, but read the mechanism before touching it.**
`DASHBOARD_ADMIN_PASSWORD` (a second credential, separate session-state key,
no local-dev bypass — unlike the main password, it fails closed even with a
stale session flag) gates the page's *content*. But the page's
*registration* in Streamlit's navigation is gated on whether the env var is
**set at all**, not on whether the current session is authenticated —
because Streamlit only routes a URL to a page that was part of that
session's very first navigation call; a page absent from that call is
unreachable by URL for the rest of the session, full stop. So once you set
`DASHBOARD_ADMIN_PASSWORD` in Cloud Run, `/integrity` becomes visible in the
route table for every session, admin or not, and every visitor without the
admin password sees a plain sign-in prompt. This was verified in tests: a
non-admin session makes **zero calls** into `integrity.py`/`pr_quality.py` —
pinned structurally (parsing the function's source with `ast` to confirm
the gate is the first statement), not just behaviorally. This is a
reasonable design, but it means "hidden" doesn't mean invisible to a URL —
it means gated at the content layer. Fine for a password-protected internal
tool; worth knowing if you ever think "hidden" means "nobody can even try."

### 2.5 `flag_severity` (the Integrity page's weighted score) is an invented number
**File:** `people_table.py` (2B) · **Reversal cost: low — one constant dict, `FLAG_SEVERITY_WEIGHTS`.**
`integrity.integrity_flags` deliberately returns four booleans and no
combined score — its own docstring says "a single number would be argued
with." The People table needed a `flag_severity` column anyway, so one agent
invented weights: `staging_pingpong` and `rework_hidden` (both mint
resolution credit) at 3.0, `estimate_inflation` (moves billed hours
directly) at 2.0, `board_grooming` (the flag with the most innocent reading
— a lead who genuinely grooms the backlog looks identical) at 1.0. These
are defensible but arbitrary, and they are the only place in the whole
system where a padding signal gets compressed into a single ranking number.
If you want to argue with a number, this is the one to look at first.

### 2.6 The admin credential decides whether a report can leave the screen — and Integrity deliberately has no "Download report" button
**File:** `report.py` call sites, `pages/integrity.py` (5C) · **Reversal cost: low to add a report; the decision to omit it was deliberate, not an oversight.**
Delivery, Code, People, Planning, Engineering, and now Today all populate
the printable "Download report" export. Integrity does not, on purpose: the
agent's reasoning (documented in the page's own source comment, per your
instruction that any such decision be written where the next person will
actually see it) is that an admin-gated page whose entire point is "never a
verdict, every flag ships with its innocent reading and linked evidence"
loses that framing the moment it becomes a forwardable PDF someone can
screenshot out of context. This is a real policy call about how integrity
data can leave the room, not a bug. If you want an exportable version, say
so explicitly — it's a few lines, but someone should decide it's wanted,
not just add it back because a checklist expects one.

### 2.7 The Today page's headline PR link doesn't match the number it labels
**File:** `pages/today.py` (3A) · **Reversal cost: low, but there is no exact fix — GitHub's search syntax doesn't have the qualifier this needs.**
The hero tile's "unapproved" count needs a GitHub search for "has reviews
but none approving." GitHub's search syntax has no such qualifier — only
`review:none` ("nobody has reviewed this at all," a stricter subset). The
agent linked to the nearest real, correct query rather than inventing a
fake one or leaving a dead `#` link, and documented the gap. This is a
real, small, permanent discrepancy between what the tile says and what
clicking it searches for — worth knowing before someone clicks through
expecting an exact match and gets confused about the count.

### 2.8 `unestimated_per_sprint` and `estimate_policy` disagree about what counts as "estimated"
**File:** `planning_metrics.py` (2E) · **Reversal cost: low — a documented, deliberate scope difference, not a bug, but worth knowing they don't agree.**
`hygiene.estimate_policy` (used elsewhere) credits a ticket with only a
human-readable estimate ("2h", no parseable seconds) as estimated, because
that function checks *policy compliance*. The new per-sprint hours metric
only counts `original_estimate_sec` — a text-only estimate contributes zero
hours either way, so it's treated the same as no estimate. Both are correct
for what they measure, but if you ever see the Planning page's "unestimated"
count and the estimate-policy compliance number disagree for the same
sprint, this is why — not a data bug.

### 2.9 Capacity math double-counts idle hours for anyone declared on multiple sprints
**File:** `capacity.py` (2E) · **Reversal cost: none currently possible — there's no board-to-person roster in the codebase to fix it against.**
A person in `JIRA_WEEKLY_HOURS` who genuinely works only one board still
gets an "idle, has room" row — and that sprint's worth of available hours
added to their cross-sprint total — for every *other* dated sprint present
in the frame, even ones they hold zero tickets on. There's no data anywhere
that says "this person only works this board," so this can't be fixed
without adding that roster. The per-person `Sprints` column is there
specifically so a reviewer can see which sprints actually contributed real
hours versus which just added phantom idle capacity — but the cross-board
total itself is inflated for anyone who splits time unevenly across boards.
Read the totals table's `Sprints` column, not just the total, until this
gets a real fix.

### 2.10 Comma-separated `ACTIVE_MERCHANTS` was a latent bug; it's now semicolon-separated to match `roles_template.env`
**File:** `pages/business.py` (3F) · **Reversal cost: none — this was already wrong, now it's fixed and matches the file that's the source of truth.**
The old code split the merchant list on `,`, but merchant names can
legitimately contain a comma ("Little International Wine, Inc"), and every
other env var in `roles_template.env` (`JIRA_ROLES`, `GITHUB_LOGIN_MAP`) is
semicolon-separated. Pasting the real `ACTIVE_MERCHANTS` value into Cloud
Run under the old code would have silently turned the whole string into one
merchant name. Fixed. Not a judgment call to revisit — just flagging that
the env var format changed, so if you have it saved anywhere with commas,
update it to semicolons when you set it in Cloud Run.

### 2.11 The store-active-or-not lookup was never verified against your live database
**File:** `merchant_client.py`, `pages/business.py` (3F) · **Reversal cost: none — this is Phase-0 item 0.6, still open, see §3.**
The code tries four possible metadata keys (`status`, `store_status`,
`is_active`, `active`) in a fixed order and stops at the first one present
— it does not fall through to the next key if the first key's value is
present but unrecognized, on purpose (falling through would silently prefer
whichever key happens to parse over the one the store is actually using).
Nobody has run `select name, metadata from medusa.store limit 12` against
the real DB to confirm which key is actually meaningful. Until that
happens, the business page's merchant-active detection is a best guess, not
a verified fact.

### 2.12 Report tables record as prose sentences, not structured data
**File:** `theme_html.py`, page call sites (5C) · **Reversal cost: medium — a real design constraint, not an oversight.**
`theme_html.table()`'s rows don't map cleanly onto the printable report's
figure model (a report figure is one label/value/note; a table row is
several cells of different kinds). The fix was to record each table row as
one prose sentence, cells joined by " · ", under the section that already
names the on-screen chart. This closed six previously-broken report gaps
(the stale table, the Code stuck-PR queue, People's table, etc. were all
silently missing from every printed report before this task). It's a
reasonable resolution, but it means anyone reading a printed report sees
tables rendered as run-on sentences rather than actual tables — worth
knowing if the printed report is ever handed to someone outside the
dashboard.

---

## 3. Still blocked on you

Every one of these degrades to an **explicit, stated "unavailable" state** —
never a silent zero standing in for missing data. That was checked, not
assumed: the whole project's design principle (`KPI_SPEC.md §1`, restated in
almost every assumptions file) is that a missing measurement must never
render as a flattering or damning zero.

| Item | Unblocks | Current behavior without it |
|---|---|---|
| `JIRA_ROLES`, `GITHUB_LOGIN_MAP` in Cloud Run | The entire 16-role rubric system, role-based cohort comparison, every People page score | `roles.py` ships with the confirmed roster baked in as a Python default (`roles_template.env`), so scoring already works today even with the env var unset — but the env var is what lets you correct or extend the roster without a code deploy. Right now Cloud Run is running on the baked-in default. |
| `ACTIVE_MERCHANTS` in Cloud Run | Correct filtering of the Business page's merchant list | Same as above — a five-merchant default is baked into `pages/business.py`, copied verbatim from `roles_template.env`. Correct today by luck of a good default, not because the env var is actually set. |
| `DASHBOARD_ADMIN_PASSWORD` in Cloud Run | The entire Integrity page | Unset: `/integrity` is completely absent from the app's navigation for every session — not hidden-but-reachable, genuinely unregistered. Nobody, including you, can currently reach it in production. |
| Sprint start/end dates on the ML board in Jira | Capacity, utilization, and carry-over math for the ML team | Every capacity/utilization number for ML-board sprints reads as an explicit "excluded from totals, dates not set" callout — never a zero, never silently dropped from the page. The Planning page names every dateless sprint by name, not just "the ML one." |
| `store` metadata key verification (`select name, metadata from medusa.store limit 12`) | Trimming the merchant-active lookup from 4 guessed keys to 1 verified one | The 4-key guess-in-order logic runs as-is; a merchant page caption states plainly when the DB is unreachable or misconfigured rather than showing a wrong number. |
| Slack bot token | Weekly snippet-posting compliance (`snippets.py`) replacing the current, most-gameable "Weekly updates" scorecard component | `snippets.py` is fully built and tested against fixture data but has zero live call sites — it isn't imported by any page. `SLACK_USER_MAP` (the Jira-name-to-Slack-ID join) ships with **no baked-in default**, deliberately: nobody has ever confirmed a real Slack user ID for anyone, and guessing risks crediting one person's post to another. |
| Clockify read-only API key + workspace ID | Billed-vs-delivered hours comparison (`clockify.py`) — the actual padding check, per your own framing | Same as Slack: fully built, fixture-tested, zero live call sites, zero live wiring into any page. `CLOCKIFY_USER_MAP` also ships with no default for the same reason. Note: Clockify's API does not expose when a time entry was *created* (only its claimed start/end) — the "entry created days after the claimed work" reconstruction tell was verified absent from the API, not implemented as an approximation, and is stubbed as permanently unavailable rather than faked from a proxy signal. |
| Fireflies API access | Standup truth-check (`standup.py`) — attendance and commitment follow-through cross-checked against the Jira changelog | Same as above: built, tested against a hand-written fixture transcript, zero live wiring. |
| Confirm or retire GitHub logins `lawrnsfeng` and `VossBackend` | Clean PR authorship attribution | Both are already excluded from every per-person view and explicitly labeled "unmapped authors" rather than silently attributed to the wrong person or dropped. Confirmed against the live roster: they resolve to `unmapped_logins`, exactly as designed. This is a low-urgency item — the system already handles the unknown case safely — but the underlying question (who are these two, are they still active) is still open. |

---

## 4. What this does not do

Carried forward from `IMPLEMENTATION_PLAN.md §13`, confirmed still true
against the shipped code, plus what the agents deferred along the way:

- **Hiding a PR in draft is detected, not prevented.** `draft_transitions()`
  flags a PR that went draft after a review was requested, and the Code page
  surfaces those rows in the same queue they'd have been in had they stayed
  open — but nothing stops someone from doing it. Prevention is a GitHub
  branch-protection setting, not a dashboard feature.
- **Slack responsiveness stays unscored, on purpose.** The plan calls
  surveillance-plus-four-timezone-noise not worth it. Snippet *posting* is
  scored (once the Slack token lands); response latency is not and isn't
  planned to be.
- **No metric proves padding.** Every integrity flag ships with its stated
  innocent reading, by construction — `theme_html.innocent()` is called
  unconditionally on every card, including the empty branch, and this is
  enforced by a test, not a convention someone could forget. An empty
  integrity row is explicitly **not** a clean bill of health: someone who
  does nothing at all trips nothing at all. That sentence is in
  `integrity.py`'s own docstring, not just this report.
- **Peer cohorts are thin: 12 active engineers across 16 role keys.**
  `MIN_PEERS=3` suppresses comparison for all but one cohort today (see
  §2.2). Confirmed correct, not a bug — see that section for the actual
  numbers.
- **Hover tooltips and clickable filter chips were lost to the
  no-JavaScript constraint.** `st.markdown(unsafe_allow_html=True)` strips
  `<script>` tags, so tooltips degrade to native `title=` attributes and
  filter chips became `st.pills` (a real Streamlit widget, not clickable
  HTML) wherever a page needs interaction. This is the stated cost of the
  hybrid rendering approach, paid knowingly, not an oversight.
- **Clockify's "entry created after the claimed work" tell is absent, not
  approximated.** Verified against Clockify's own API documentation (two
  independent sources) before writing anything: no field on a time entry
  records when the record was written, separate from its claimed start/end.
  The function exists with the same shape as the other three detectors and
  always returns "unavailable" — it was not fabricated from a proxy signal
  like entry-ID ordering or page position, which the task explicitly
  disallowed.
- **Report-export parity has one remaining real gap.** The 12-week
  throughput line chart on Today (`theme_html.linechart()`) has no recording
  mechanism at all — `tiles()`, `hbars()`, and `table()` all record into the
  printable report; `linechart()` was never given one, because its return
  value is a bare SVG with no (label, value, note) shape to record. Small,
  but real, and explicitly left open rather than quietly worked around.

---

## 5. Full decision log

Everything else, grouped by area — reference material, not the reading
section.

### Design system / tokens
- `theme_tokens.MAX_WIDTH` follows the mockup (1280px), overriding the
  old `theme.py` value (1560px), per explicit task instruction.
- Every invented grey in the old `theme.py`/`app.py` (`#6b7280`,
  `#4b5563`, `#9ca3af`, `#1f2937`, `#f3f4f6`) was mapped onto the mockup's
  real ink ladder or the mockup's own repeated literals — none were kept as
  independently-invented colors.
- `theme.CATEGORICAL` changed from a tuple to a list so it can equal
  `theme_tokens.SERIES` under Python's `==`; every caller checked for
  tuple-specific reliance first (none found).
- The collapsed "Other" bar in `rank_bar` changed color (grey → red) as a
  side effect of the palette swap; left as-is rather than fixed, because
  fixing which color it reads from would break a test outside that task's
  file ownership. Flagged for whoever next touches `rank_bar`.
- Four competing chip/KPI-tile systems were consolidated to one
  (`theme_html.chip()`/`tiles()`); `kpi_strip()`, `page_shared._kpis()`,
  and the invented per-stage color dict were deleted. `pages/business.py`'s
  `st.metric`-based tiles were deliberately **not** migrated — different
  rendering mechanism, out of scope for a conformance sweep, not a missed
  case.
- Migrating old static-accent KPI tiles onto the new kit dropped their
  red/green coloring, on purpose: a static count colored by an unearned
  judgment (no delta baseline) was itself a conformance bug of the same
  shape the delta-arrow rule targets.
- `pages/business.py`'s delta-arrow coloring bug (rising cost tiles colored
  green because `delta_color="normal"` colors any rise green regardless of
  whether rising is good) was found and fixed — this is the same "green
  arrow on a bad number" failure mode the whole redesign exists to close,
  found live in the one page that wasn't otherwise touched this phase.
- `theme_tokens.HERO_SIZE = 48` added as a size rung above the six-size
  type ladder for the one-hero-per-page rule.
- Bar geometry (24px max height, 4px corner radius) implemented as a
  function of row count and chart height floor, not a fixed pixel value —
  needed because the height floor inflates the per-row slot for small `n`.

### Compute layer
- `idle_days` stays visible everywhere as "last touched"; only what
  *scores* off it was changed to `status_age_days` where wired (see §2.1
  for what's still not wired).
- Former-staff resolution credits are flagged, never dropped from output.
- `org_reopen_rate`'s denominator is "distinct tickets that entered a
  resolved status in the window," matching `reresolve_events` exactly so
  the two numbers can't disagree about what "resolved" means.
- `SIZE_POINTS` PR-size weights (trivial=1, small=3, medium=8, large=20,
  oversized=20 — flat past large) are invented, not derived from any spec;
  documented as such.
- `unprompted_reviews` only resolves `User`/`Bot` review requesters, not
  `Team`/`Mannequin` — a team-routed review request is invisible to that
  reviewer's evidence, stated as a blind spot.
- `people_table.py`'s person universe is "who appears in the data," not
  "everyone on the roster" — a silent contractor with zero activity gets no
  row. Chosen to satisfy a literal "empty in, empty out" test; the
  tradeoff is documented.
- `n_ttfr_hours` (time-to-first-review) is an upper bound (`prs_reviewed`),
  not the exact "was first" count, because getting the exact count would
  mean depending on a private name in a file outside that task's ownership.
- Week bucketing across the whole system (series, snippets, cost) is
  calendar weeks, Monday 00:00 UTC — the one timezone-neutral anchor across
  the team's Vietnam/Uruguay/Tunisia/EU spread.
- `delta()` reports raw magnitude, not percentage (percentage is undefined
  at `prior == 0`, which is a common, meaningful case here — a new hire's
  first PR, not an edge case to divide-by-zero on).
- A genuine tie in `delta()` renders neutral (`is_good=None`), not "good"
  or "bad" — distinguished internally from "no prior data" but both render
  the same to the caller.
- No historical board-state snapshot exists anywhere in the codebase.
  Sparklines for stock-like metrics (open ticket count, stalled count) are
  built from real-but-different signals (creation-week inflow, age-at-onset
  distribution) rather than fabricated snapshots — documented per-tile in
  `docs/assumptions/3A.md`.

### Pages
- The old ML-board ticket editor (writes to Jira) was kept, not deleted,
  when the single-sprint dropdown was replaced with three side-by-side
  cards — it's real write-access accountability tooling nothing in the
  brief asked to remove.
- Planning's board-hygiene bars are computed org-wide, not scoped to the
  three highlighted sprints, matching the mockup's own "all three boards"
  framing.
- The Code page's "self-merged" label was deliberately kept as "Merged
  unapproved" — a GitHub author cannot approve their own PR, so the
  mockup's term is wrong and the code's term is right; the mockup gets
  fixed to match the code, not the reverse.
- `pages/engineering.py` (the old combined page) is kept alive at
  `/engineering` because links to it live in Slack and bookmarks; it now
  carries a banner pointing at the five new focused pages.
- `pages/delivery.py` and `app.py` both draw a page title today (a known,
  visible duplication — plain Streamlit `st.title()` plus the new HTML
  `page_header()`) until a future task that owns `app.py` removes one.

### Access / integrity
- `require_admin_password()` is a return-value gate, not solely
  `st.stop()`-and-hope — verified empirically that `st.stop()` no-ops
  outside a live Streamlit run, so the function both calls `st.stop()` for
  a live deployment and returns `False`/`True` so a bare test can prove the
  gate actually blocks execution.
- The Integrity page's four cards each rank by magnitude with no fixed
  threshold (`_top_n`), per the plan's explicit "no fixed threshold" rule —
  tested by constructing five candidates that would all read as "small"
  under any plausible cutoff and confirming exactly three still render.
- The Integrity roll-up card's cosmetic-touch baseline compares each person
  against the *median of their own role*, falling back to the org-wide
  median for roles with fewer than two people represented (Infra, PM, ML
  are effectively solo) — documented as an approximation, not a claim that
  two-person medians are trustworthy.

### Testing / regression net
- A hand-built, two-ticket, seven-entry changelog fixture
  (`tests/fixtures/changelog_snapshot.json`) pins exact per-field parsing
  output — including a departed-author case and a re-resolution case — so a
  future silent attribution regression is caught by an exact-value diff,
  not a shape check.
- Found and fixed a real cross-test pollution bug: `data_layer.py` persists
  a full board snapshot to a fixed, machine-wide temp-file path
  (`/tmp/jira_dashboard_board_snapshot.pkl`), shared across every process on
  the machine — a test suite run could silently read another process's
  stale snapshot instead of its own synthetic fixture. Fixed by deleting
  the snapshot before and after the affected test file's runs.
- CI already runs the full suite (split by a `slow` marker) on every PR and
  push to `develop`/`main`; no new workflow was needed for any of the new
  test files.
