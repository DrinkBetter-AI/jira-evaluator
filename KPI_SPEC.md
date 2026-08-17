# KPI specification — measuring hourly contractors without being gamed

Prepared for Angel Vossough, 16 Aug 2026.
Branch: `redesign/dashboard-and-kpis`. Companion document: `DASHBOARD_REDESIGN.md`.

---

## 1. The principle

Your engineers are hourly, remote, and contracted. Their incentive is to maximise billable hours and apparent output. A metric they can move without doing more work is not a metric — it's a target.

Right now **every input to the scorecard except PR age and reopen count is either self-declared or resettable with a keystroke.** That is the core problem, and it is worse than "no KPIs", because a gameable KPI actively rewards the behaviour you're trying to stop.

Four rules follow, and every metric below obeys them:

1. **Prefer machine-recorded facts over declared ones.** A merged diff, a review timestamp, a status transition in the changelog — these are written by the system. Estimates, priorities and logged hours are written by the person being measured.
2. **Where you must use a declared number, cross-check it against a recorded one.** An estimate alone is worthless; an estimate against actual cycle time and diff size is evidence.
3. **Every metric names its own blind spot.** A number that doesn't say what it can't see will be over-trusted, and then quietly gamed through the gap.
4. **Emit evidence, never bare scores.** "Sai has a grooming flag" is useless in a one-on-one. "Sai made 34 field-only edits across 19 tickets on Friday afternoon with 2 status transitions that week — here are the ticket keys" is a conversation.

---

## 2. The gaming surface as it stands

Found by reading the code, not by observing anyone. Each of these is currently exploitable.

| # | Exploit | Mechanism | Cost to the engineer |
|---|---|---|---|
| 1 | **Reset the idle clock** | `idle_days` derives from the newest changelog entry of *any* kind. A label edit, a one-character description change, a priority toggle all zero it. This drives 25% of the score plus the entire stale queue. | 5 minutes on a Friday |
| 2 | **Mint resolutions in staging** | "Review in Staging" counts as resolved. Moving in registers permanently in the 7/30/90d window; moving back out decrements nothing. | One drag |
| 3 | **Split tickets** | Nothing is size-weighted. Five trivial tickets beat one real one on the resolved tile, on Delivery, and on the Shipper badge. | Free, and looks diligent |
| 4 | **Claim someone else's finish** | Resolutions credit the *current assignee*, not whoever did the work. Assign a nearly-done ticket to yourself before it lands. | One dropdown |
| 5 | **Pad the estimate** | `has_estimate` checks presence, never accuracy. Padding raises "Estimated Hours Delivered", pushes utilisation over 100% ("Over-committed" — an argument against more work), and satisfies four separate metrics. `time_spent` is fetched and never compared to it. | Free, and rewarded twice |
| 6 | **Hide a PR in draft** | `_open_query` carries `draft:false`. Converting to draft removes a PR from Open, Stuck, Never reviewed, and every hygiene tab. | One click |
| 7 | **Trade approvals** | `stuck` is `approving_reviews == 0`. Nothing records *who* reviewed. A reciprocal approval ring is completely invisible; so is self-merging. | One favour |
| 8 | **Erase rework** | `_reopened_jql` requires `status NOT IN (resolved)` *now*, so re-resolving a reopened ticket removes it from the rework metric entirely. | Fix it again |
| 9 | **Drop a component** | `overall()` renormalises over whichever components have data. Holding zero open non-backlog tickets removes 45 of 100 points of denominator. **Pushing all your work to Backlog raises your score.** | Structural |
| 10 | **Avoid sprints** | Carry-over is `len(closed_sprints)`. Never add tickets to a sprint → zero carry-over → 100/100 on that component. | Structural |

Worth preserving, because they already resist gaming: PR `age_days` from `createdAt` (cannot be reset by pushing a commit), `carry_over_count` counting closed sprints even after a ticket leaves the sprint, server-side `issueCount` totals, and `critical_in_flight` (needs both a lie in Jira *and* a stalled PR).

---

## 3. The new metrics

Five families. Everything marked **built** is implemented and unit-tested on the branch.

### 3.1 Flow honesty — closes exploits 1, 2, 8

**`status_age_days`** *(built, `integrity.py`)*
Days since the last real **status transition**, from the changelog, as opposed to `idle_days` which resets on any edit. Returns `masked_days` — the gap between the two — which is itself the tell. A ticket showing `idle_days = 1, status_age_days = 60, masked_days = 59` has been groomed, not worked. **Replace `idle_days` with this everywhere it drives a score.**

**`cosmetic_touches`** *(built)*
Per person, changelog saves in a window that changed only non-status fields, with distinct tickets touched, the busiest single day, and assignee round-trips. Attributed by changelog **author**, not current assignee. Read it beside that person's status-transition count: many touches and few transitions is board grooming.
*Blind to:* intent. Rewriting a thin ticket into something Devin can act on lands here too. A low count is not a compliment.

**`reresolve_events`** *(built)*
Counts transitions *into* a resolved status per ticket from the changelog, so a reopen-then-re-resolve cycle is visible. Marks `hidden_rework=True` on exactly the tickets the current JQL structurally cannot see. Treats a normal Staging → Ready for Prod → Released walk as **one** resolution, not three.

**`status_pingpong`** *(built)*
Backward transitions and repeat entries into the same status, with `staging_entries` isolating re-entries into resolved-but-not-terminal statuses. This is exploit 2, made visible.

**`cycle_time`** *(built)*
Time in each status per ticket, and per-person medians for lead time, in-progress and review. **This is the one metric in the whole system that cannot be gamed by editing fields** — only by actually finishing work faster.
*Blind to:* hours. A two-day cycle billed at 30 hours is a question this raises and cannot answer.

### 3.2 Estimate integrity — closes exploit 5, and is your most direct answer to hour-padding

**`accuracy_ratio` / `accuracy_by_person`** *(built, `estimate_accuracy.py`)*
`time_spent_sec / original_estimate_sec` per finished ticket; median **and IQR** per person. The IQR matters: a median of 1.0 with an IQR of 2.5 means the person isn't estimating at all, they're guessing symmetrically.

**`padding_index`** *(built)*
Median ratio plus the share of tickets finishing under 60% of estimate. Returns **no verdict below 5 tickets** — with four tickets you are reading noise.

**`estimate_churn`** *(built, `integrity.py`)*
Estimates edited *after* the ticket entered an in-progress status, with old value, new value, direction, who changed it, and how many days into the work. **An estimate revised upward mid-flight on an hourly contract is the single most direct padding signal available**, and it is currently invisible.

**`hours_per_delivered_line`** *(built)*
Logged hours over PR changed lines, joined by Jira key. Outliers by modified z-score against the team median (MAD-based, not a hard threshold). This is the only place a self-reported number meets a machine-recorded one.
*Blind to:* honest hard thinking on a small diff. A flag means open the PR, not open a conversation.

> **Honesty note:** logged time is *also* self-reported. These metrics detect **inconsistency between two declared numbers and one recorded one**, not ground truth. The only true cross-check is invoiced hours against cycle time, and invoices aren't in this system.

### 3.3 Output that resists splitting — closes exploit 3

**`size_bands`** *(built, `pr_quality.py`)*
PRs classified trivial (<10 changed lines) / small / medium / large / oversized (>1000), per person, with the median. Splitting one change into five PRs shows up here as a spike of trivial PRs against an unchanged median — invisible in a raw count.
*Blind to:* difficulty. Lockfiles and generated code inflate it.

**Report count and median size together, always.** A count alone is exploit 3.

### 3.4 Code quality — your "how accurate are the PRs" question

**`devin_findings`** *(built)*
Reviews and comments authored by Devin, split by `CHANGES_REQUESTED` / `COMMENTED` / `APPROVED`, per PR and per author. **This is the closest thing to a direct measure of how many issues an engineer's PRs have, and it is currently not fetched at all** — Devin's reviews are ordinary GitHub review nodes sitting in an endpoint the app already calls.
Reported as a **share of judged PRs**, not a total, so shipping more work isn't automatically a worse score.
*Blind to:* whether Devin ran. Zero findings means clean *or* unreviewed; `reviews_fetched` distinguishes them. **See open decision #4.**

**`abandoned_rate`** *(built)* — closed-unmerged over decided PRs. Hours that landed in a branch that never shipped.

**`traceability`** *(built)* — share of **merged** PRs carrying a resolvable Jira key. Today hygiene is measured only on *open* PRs, so nobody is ever assessed on whether shipped work was traceable. 23 open PRs currently have no key.

### 3.5 Review citizenship — closes exploit 7

**`review_citizenship`** *(built)* — reviews **given** per person, distinct authors reviewed, median time-to-first-review (clock starts at ready-for-review, so draft time isn't held against the reviewer). Reviewing is work; today it is entirely unmeasured, which makes never reviewing anyone the rational choice.

**`reciprocity`** *(built)* — the pair matrix, mutual-review flags, top-partner share, and a concentration index, plus counts of approvals given with an empty body and zero review threads (rubber-stamping).
*Blind to:* team size. On a team of four, high reciprocity is arithmetic, not collusion. **This is a prompt to open the PRs, never proof of anything.**

**`self_merge`** *(built)* — separates "merged my own PR after a colleague approved" (fine) from "merged my own PR that nobody approved" (branch-protection material), plus `merged_off_trunk`, where the merge count rises and nothing ships.

---

## 4. Scorecard changes — closes exploits 9 and 10

**Component-dropping is fixed.** A component with no data is now reported as *"insufficient data — needs X"*, not silently removed from the denominator. `overall()` returns **`None`** below 60 points of measurable weight rather than a flattering score built on a third of the rubric. Pushing every ticket to Backlog now produces "n/a", not a higher number.

**Every component carries its sample size `n`.** 100% on 4 tickets and 100% on 40 are no longer the same number on screen.

**Delivery is fixed and supplemented.** The old formula scored you against your own trailing 90 days — so a quiet quarter built a low baseline that a burst of trivial tickets then scored 100 against. The metric literally rewarded volatility, and `resolved_90` included the last 7 days, double-counting them. The baseline is now the prior 83 days, and a **peer-relative** component sits alongside the self-relative one: percentile against the team in the same window, requiring at least 3 peers.

**What to kill:** the raw "Tickets resolved (7d/30d)" tiles as a *performance* signal. Keep them as volume telemetry, but they are exploits 2, 3 and 4 stacked, and they are the first thing on the page. Anything presented as performance should be size-weighted and cycle-time-anchored.

---

## 5. What ships when

**Phase 1 — no new API calls, works today.** Everything in `integrity.py` and `estimate_accuracy.py`. The full Jira changelog was already being downloaded on every load and reduced to a single `max()` timestamp; I've plumbed it through as a `changelog` column, so all of §3.1 and §3.2 run on data you already have. This is the highest value per unit of risk in the whole document.

**Phase 2 — needs the extended GraphQL.** Everything in `pr_quality.py`. The query additions are written; cost goes from ~29 to ~1,030 of your 5,000 points/hour, which is fine for hourly caching and not fine for per-keystroke refetch. It degrades to the lean query on failure and reports `unknown` rather than `0`. **Resolve the 403 first.**

**Phase 3 — needs you.** The integrity panel UI, and the open decisions in `DASHBOARD_REDESIGN.md` §0.

---

## 6. A caution worth stating plainly

None of these metrics prove anyone is padding hours. Each flag has an innocent reading, and I've written that into every docstring. Their value is that they make the *cheap* moves visible, which changes what's rational: when grooming the board and splitting tickets stop registering as output, the effort goes back into the work.

The failure mode to avoid is treating a flag as a verdict. An empty integrity row is not a clean bill of health either — someone doing nothing at all trips nothing.
