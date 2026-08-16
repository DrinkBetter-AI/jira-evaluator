# Implementation plan — dashboard redesign & role-aware KPIs

Prepared for Angel Vossough · 16 Aug 2026 · re-verified under Fable 5.
The design spec is the mockup: `vinovoss-dashboard-design.html` (open it in a browser — Today, People, and Integrity are fully designed; Delivery/Code/Planning/Business are scoped stubs).
Companions: `DASHBOARD_REDESIGN.md` (analysis), `KPI_SPEC.md` (metric definitions).

---

## 0. Verification statement

Every material claim in the two companion documents was re-checked directly against the code, not against the overnight summaries. The "undefined" title bug and its fix were both reproduced empirically; the four structural exploits (staging-counts-as-resolved, current-assignee attribution, re-resolve erasure, scorecard renormalisation) were confirmed at their exact source lines; `integrity.py`, `pr_quality.py` and `estimate_accuracy.py` were stress-tested with adversarial synthetic data and behaved correctly (masked-days arithmetic, mid-flight estimate raises, mutual-approval detection, Devin-findings attribution). The full suite passes: **446 passed, 1 skipped**. Two corrections from the re-check: one stale docstring in `integrity.py` (fixed), and one wrong suspicion of mine — the routing fix's `visibility="hidden"` parameter is real in Streamlit 1.61.1.

---

## 1. What Angel must provide before Devin starts

These block work packages below; nothing else does.

1. **GitHub token 403.** PR metrics are dead on the live dashboard right now, and WP4 roughly triples GraphQL cost (~29 → ~1,030 of 5,000 points/hour — fine cached hourly). Likely rate limiting or expiry; check the token's type and limits.
2. **Role roster.** Fill `roles_template.env` (every name seen in the data is pre-listed). Paste the ones you know; the PM can finish it. Format matches `JIRA_TEAM_PEOPLE`.
3. **`ACTIVE_MERCHANTS`.** Exact catalogue spellings observed in the app: `TheWinesGood, Yiannis Wine Shop, Capital Fine Wine, Little International Wine`. Confirm these four strings and set the env var — the filter is exact-match.
4. **Sprint dates in Jira.** Availability/utilisation stays empty until sprint start/end dates are set on the board. Config, not code.
5. **Does Devin review every PR or only some repos?** Determines whether "AI-review findings" can be a scorecard component (full coverage) or evidence-only (partial). Metric handles both; the *weight* decision needs the answer.
6. **GITHUB_LOGIN_MAP.** Without it, per-person PR joins quietly fall back to Jira-key matching only. One env var: `Tam=tungph;Farid=faridsh69;...` — worth 20 minutes of the PM's time.

---

## 2. Work packages for Devin

Already done on `redesign/dashboard-and-kpis` (do **not** re-implement): the undefined-title fix, /engineering routing, nan-epic fix, type scale + tokens, `rank_bar` replacing the three unreadable pies, single validated palette, bot filtering in person tables, honest estimate-coverage tile, `ACTIVE_MERCHANTS` plumbing, the `changelog` column, and the four metric modules with 446 passing tests.

### WP1 — Merge the branch and deploy `redesign/dashboard-and-kpis`
Small. Review the diff, merge, deploy, set `ACTIVE_MERCHANTS` and `GITHUB_LOGIN_MAP`.
*Accept:* no "undefined" titles in prod; /engineering resolves; Estimate coverage tile equals Policy Compliance; bots out of person tables; price section captions the merchant roster.

### WP2 — Information architecture: one page → six
Split `_render_engineering_page` into the six `st.Page`s in the mockup: **Today, People, Delivery, Code, Planning, Integrity** (Business unchanged). Pure reorganisation of existing render functions — no metric changes. Today page per the mockup: attention band (hero = stuck-PR share + three decision cards), six stat tiles with sparklines, throughput line chart, ranked status bars. Sidebar scope/filters persist across pages via session state.
*Accept:* every existing section reachable in ≤2 clicks; Today renders in <3s from cache; deep links per page work.

### WP3 — Flow-honesty wiring (Jira-only; no new API calls)
Replace `idle_days` with `integrity.status_age_days` everywhere a *score or queue* depends on it (KPI strip "Stalled", Stale & Abandoned ranking, priority-score staleness term). Keep `idle_days` visible beside it as "last touched". Add re-resolve/ping-pong counts to the resolved roll-up so a staging round-trip stops minting credits. Resolution credit moves from current assignee to **changelog author** of the resolving transition.
*Accept:* a label-only edit no longer removes a ticket from any stale queue (test exists); a ticket entering staging twice counts one resolution; the resolved ranking names whoever moved the ticket.

### WP4 — Extended PR data + Code page
Switch `fetch_open_prs`/`fetch_merged_prs` to the extended GraphQL (already written in `github_client.py`), cached ≥1h, degrading to the lean query on failure with an on-page notice ("PR quality data unavailable — lean mode"). Build the Code page: stuck/never-reviewed queues, Devin findings per author (share of judged PRs), size bands, merged-work traceability, review citizenship, abandoned rate.
*Accept:* graceful degradation proven by revoking the token in staging; per-author tables show `reviews_fetched` so "no findings" ≠ "not reviewed".

### WP5 — Role-aware scorecards + People page
Read `JIRA_ROLES`. Per-role rubrics (weights below are the brainstorm draft, not final):

| Component | Code roles¹ | QA-auto | QA-manual | PM |
|---|---|---|---|---|
| Delivery, size-weighted | 20 | 20 | — | — |
| Cycle time vs role median | 15 | 15 | 15 (time-to-verify) | — |
| Rework / re-resolves | 15 | 10 | — | — |
| Estimate accuracy | 15 | 15 | — | — |
| Review citizenship | 10 | 10 | — | — |
| AI-review findings² | 10 | 10 | — | — |
| Ticket hygiene | 10 | 10 | 20 (bug report quality) | 25 (tickets they write) |
| Urgent response | 5 | 5 | 10 | 15 (triage latency) |
| Bugs found that were fixed (validity) | — | 15 | 35 | — |
| Tickets verified | — | — | 20 | — |
| Epic & board hygiene (orphans, empty epics, estimate coverage of team) | — | — | — | 35 |
| Sprint discipline (dates set, carry-over rate of team) | — | — | — | 25 |

¹ backend, frontend, mobile, ML-ops/AI, recommendation, AI-recommendation. ² weight goes to 0 and redistributes if Devin coverage is partial (see prerequisite 5). Biz dev: no engineering scorecard.
Leaderboard ranks **within role only**. Unmeasured components render as "insufficient data — needs X"; below 60 measurable points, "no score" (already enforced in `kpi.py`). Every component shows `n`.
*Accept:* a person with role unset appears under "role unknown", unscored; QA-manual scoring runs with zero GitHub data; switching a person's role never changes another person's score.

### WP6 — Integrity page (CEO-only)
Gate: visible only when the signed-in session is Angel's (simplest: `DASHBOARD_ADMIN_COOKIE`-style second password or an env-listed viewer). Four cards per the mockup — masked freshness, mid-flight estimate revisions, staging round-trips, review pairs & self-merges — each with evidence rows linking to Jira/GitHub and the fixed "innocent reading" footnote. Integrity applies to **all hourly contractors**; cosmetic-touch counts are baselined *within role* so the PM's legitimate grooming doesn't false-positive.
*Accept:* zero integrity UI rendered for non-admin sessions (not just hidden — not computed); every flag row links to its raw evidence; role-baselined thresholds covered by tests.

### WP7 — Design-system conformance pass
Apply the mockup's remaining specs app-wide: bars ≤24px with 4px rounded data-ends, 2px lines, hairline solid gridlines, legends for ≥2 series, no rotated axis labels anywhere, delta arrows colored by direction-times-goodness, `tabular-nums` in table columns, the two remaining Business-page pies converted to ranked bars, Vivino tab replaced with an explicit "unavailable — Vivino blocks our requests" state.
*Accept:* screenshot review against the mockup; no chart with >8 colors; no vertical text.

### WP8 — Regression net
AppTest-based smoke render of every page with stubbed data (fragments already exist in `tests/test_theme_visual.py` — extend to all six pages); a fixture snapshot of one real Jira payload so changelog parsing is pinned; CI green required.

**Order:** WP1 → WP2 → WP3 → (WP4 ∥ WP5) → WP6 → WP7 → WP8 runs alongside everything.
**Do not** start WP4 before the 403 is diagnosed, or WP5 before the roster exists.

---

## 3. Design tokens (for Devin — the mockup is normative)

Type: 13/14/15/17/20/32px (meta/label/body/lead/section/display), base font 17px. Ink: `#111827` / `#475569` / `#64748b` / `#94a3b8`. Planes: page `#f8fafc`, card `#ffffff`, hairline `#e5e7eb`. Series (validated, fixed order, never cycled): `#2563eb #eb6834 #1baf7a #eda100 #e87ba4 #008300 #4a3aa7 #e34948` — fold to "Other" past 8; aqua/yellow/magenta require visible labels or a table (sub-3:1 contrast). Status (icon + label, never color alone): good `#15803d`, warn `#b45309`, critical `#b91c1c`, info `#1d4ed8`. Content max-width 1560px. One hero number per page, ≥48px.

---

## 4. Brainstorm agenda (you + me, before handing WP5/WP6 to Devin)

1. The rubric weights table above — per role, what's over/under-weighted?
2. AI-recommendation vs recommendation engineer — same rubric or split? (I drafted them identical.)
3. What the engineers see: their own scorecard only (current assumption) — do they also see their own *rank within role*?
4. Integrity thresholds — how many masked days / cosmetic-per-transition before a flag shows? (I propose: no fixed threshold; always show top 3 by magnitude with evidence.)
5. Whether invoiced hours can be imported monthly (CSV) — it is the only true cross-check for `hours_per_delivered_line`.

---

## 5. Communication signals (added 16 Aug after Slack/Fireflies verification)

Angel's context: #team-snippets weekly goals are mandatory and mostly ignored; daily standup (no Fri), ML meeting 2×/wk, marketplace meeting daily (no Thu); Fireflies and Coworker record meetings; some people are slow to respond on Slack.

**Measured baseline — #team-snippets (C098TCRRV2Q), last 8 full weeks (W25–W32):**
Farid 8/8 · Tam 7 · Mohsen 7 · Mehdi 7 · Mihai 6 · David 4 · Igor 4 · Anouar 3 · Santi 2 · Alesya 2 · Ali 1 · Shawn 0 · Jal 0 (once ever, in Feb) · **Gaston 0 — never posted in the channel's history** · Dina 0 · Robert 0. The channel died in March, was revived in June, and is decaying again (W31: 4 posters, W32: 2). Parse counts top-level posts only; thread replies would need including.

### WP9 — Snippet compliance (small; needs a Slack bot token)
Read one channel via `conversations.history` (+ thread replies), attribute by author, week-bucket, join to `JIRA_ROLES`. Scorecard: this **replaces** the old "Weekly updates" component (idle_days ≤ 7 — the most gameable signal in the system) with an observable act: weeks-posted / weeks-active. People page shows the streak; Today page shows this week's poster count vs roster.
*Accept:* thread replies counted; people on holiday excluded via a simple absence list; the metric names its blind spot (a posted snippet is presence, not truth).

### WP10 — Standup truth-check (Phase 3; the strongest padding evidence available)
Fireflies transcripts carry per-person action items with timestamps (verified on the 8/10–8/14 meetings). Join a person's stated commitments against the Jira changelog over the following days: "said X at standup three days running, ticket untouched" is the direct, quotable form of hour-padding evidence — far stronger than any proxy. Also derive *actual* attendance from transcript speakers, **never** from the participants field (verified: that field is the calendar invite list — departed staff appear on every standup).
*Accept:* per-person weekly table: commitments stated, matching board movement, unmatched commitments with meeting deep-links (`app.fireflies.ai/view/{id}?t={seconds}`). Integrity-page only.

### Slack responsiveness — deliberately NOT scored
Measuring reply latency org-wide is surveillance, and it's noisy across Vietnam/Uruguay/Tunisia/EU timezones. If wanted later: evidence-only, limited to direct @-mentions containing a question in work channels, business-hours-adjusted per person's timezone, shown as a median with links. Not a scorecard component.

### New names surfaced (Angel to confirm)
- **Praveen Rai** — posted snippets in June; not in Jira assignee data.
- **Dat (Đào Nguyễn Anh, dat@vinovoss.com)** — gives QA/staging updates at standup; not in the roster Angel sorted.
- Standup summary mentions "Tina" — probably Fireflies mishearing "Dina"; treat name-matching in WP10 with an alias table.

### WP11 — Clockify: billed hours vs delivered work (the true cross-check)
The team bills through Clockify (everyone but Praveen). Read-only API key +
workspace id → weekly detailed report per person. Join three numbers per person
per week on the Integrity page: **hours billed** (Clockify) · **delivered**
(size-weighted merged PRs + changelog-credited resolutions) · **hours logged in
Jira** (where present). The outlier metric `hours_per_delivered_line` in
`estimate_accuracy.py` switches its hours source from Jira worklogs to Clockify,
which makes it real. Flags: billed-hours weeks with near-zero board and PR
movement; billed hours diverging from Jira-logged hours on the same tickets.
*Accept:* per-person weekly triplet with links; MAD-based outliers only (no fixed
thresholds); Angel-only page; a missing Clockify mapping renders "unmapped", never zero.
*Blind spot stated on-page:* Clockify entries are still self-reported — this
catches inconsistency between what was billed and what the systems recorded,
not ground truth.

**Prerequisite from Angel:** Clockify API key (read-only) + workspace name, and
whether Clockify user emails match the vinovoss.com emails.

**Policy (Angel, 16 Aug): hours are tracked with the live timer at the moment
the work happens — not entered manually after the fact.** Two consequences:
1. *Enforcement beats detection where possible:* Clockify has a workspace
   setting that disallows manually-added entries ("force timer"). Turning it on
   makes the policy self-enforcing at the source and is worth doing regardless
   of the dashboard. (Check the plan tier supports it.)
2. *Detection where enforcement is off:* the dashboard flags entry patterns that
   look reconstructed rather than lived — perfectly round daily blocks (exactly
   8.00h), one single block per day instead of timer-grained entries, and
   overlapping entries. If the API exposes when an entry was *created* (vs when
   the work supposedly happened), entries logged days later get flagged too —
   verify this field exists before promising it; do not fake it from proxies.
   Same MAD-outlier discipline as the rest: patterns ranked by magnitude with
   evidence, no fixed thresholds, innocent reading stated (some legitimate work
   — meetings, offline reviews — gets back-filled honestly).

**Dat's offboarding:** fired, yet present in standup invites and giving updates
as recently as 13 Aug (per Fireflies). Confirm Jira/GitHub/Slack/Zoom access is
actually revoked — the ghost-roster problem is an access problem too, not just
a metrics one.

---

## 6. Decisions from Angel (16 Aug, evening) — these are final, not brainstorm

1. **Proactivity panel** (People page, visible). Three signals, all evidence-first:
   problems reported outside one's own lane (reporter data, validity-weighted so
   junk tickets don't count), PR reviews given **unprompted** (review present,
   no review_request preceding it — the WP4 data distinguishes these), and
   blockers raised early (transition into Blocked/Discussion Needed, or a
   flagged Slack ask, *before* the due date rather than after). Self-assignment
   was considered and deliberately excluded by Angel.
2. **Consequences are real: renewal and hours decisions.** Therefore the system
   must be announced to the team before it takes effect — measured-in-secret
   then acted-on reads as ambush and poisons the remaining team. The announce-
   ment is also the intervention: behavior changes the day measurement is known.
3. **Full leaderboard, visible to everyone.** Within-role ranking, all names.
   Combined with #2 this is a strong regime; expect it to move the middle and
   possibly shed the bottom — plan hiring buffer accordingly. The integrity
   page stays Angel-only regardless.
4. **Standup attendance joins the visible metrics.** Attendance is derived from
   transcript speakers (never the invite list). Two numbers per person:
   attendance rate, and **no-notice absences** — absent from the transcript
   with no message that day in the standup/team channel (Slack join is cheap).
   Meetings are mandatory and scheduled; showing up is a fact, not surveillance.
   Missing-with-notice is normal life and is not flagged.

**Goodhart caution, stated for the record:** snippets, attendance and
proactivity counts are presence signals and will be gamed the moment they carry
weight — that is fine and even useful (a gamed snippet is still a written
weekly goal; a gamed attendance is still a person at standup). The output and
integrity layers (size-weighted delivery, cycle time, Clockify-vs-delivery,
standup truth-check) are the ones that stay hard to fake, and they carry the
renewal decisions. Presence signals inform; output signals decide.

## 7. Sprint planning no-shows and the PM problem (Angel, 16 Aug, late)

Context: people skip sprint planning without notice; Mihai is not authoritative
and is distracted, so process enforcement through him does not happen.

Three responses, none of which require Mihai to become someone he is not:

1. **Sprint planning counts as a tracked meeting** in the attendance work
   (speaker-derived, no-notice absences flagged) — same as standup. The 12 Aug
   standup transcript already records planning being missed "due to scheduling
   conflicts", so the evidence trail exists today.
2. **Silence = consent.** The dashboard's existing Sprint Planner publishes the
   sprint as a visible, per-person plan. Policy to announce with the rest:
   miss planning without notice and the published plan stands as your
   commitment — carry-over against it is measured (already a scorecard
   component). Attendance stops being the only enforcement; absence costs
   influence over your own week, which is a natural consequence rather than a
   punishment Mihai has to deliver.
3. **The PM rubric is also Angel's read on Mihai.** Triage latency, orphaned
   tickets, empty epics, team estimate coverage, sprint dates set, carry-over
   rate — every one of these is currently visibly failing (97 no-priority
   tickets, sprint with no dates, 5 stuck in triage, 45 epics needing
   attention). The same scorecard that makes engineers legible makes the PM
   function legible. Whether the answer is coaching, narrowing his scope, or a
   different PM, the decision gets to ride on ninety days of the same evidence
   as everyone else's.

The general principle, stated once: **authority the PM doesn't have is replaced
by visibility, not by asking him to be tougher.** The system publishes
commitments, records absences, and measures follow-through; Angel applies
consequences at renewal time. Nobody has to chase anybody.
