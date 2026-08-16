# Dashboard redesign — proposal

Prepared for Angel Vossough, 16 Aug 2026.
Branch: `redesign/dashboard-and-kpis`. Companion document: `KPI_SPEC.md`.

---

## 0. Open decisions — I need you for these

Everything else in this document I decided and implemented. These five I could not.

| # | Decision | Why I can't make it | Default I assumed |
|---|---|---|---|
| 1 | **Do engineers see the Integrity panel?** | You said engineers see their own page and you see the org view. The integrity flags (board grooming, estimate inflation) are accusations. Showing them changes behaviour; hiding them means engineers can't contest a flag they never saw. | Hidden from engineers. Flags are yours alone, with evidence, for a one-on-one. |
| 2 | **`ACTIVE_MERCHANTS` roster** | You said TheWinesGood, Yiannis, Capital Fine Wine and Little International Wine are live and the rest are disabled. I need the exact strings as the catalogue spells them. | Implemented as an env var, **unset**, so behaviour is unchanged until you set it. See §3.4. |
| 3 | **Is logged time trustworthy at all?** | Estimate-accuracy metrics compare `original_estimate` against `time_spent`. If engineers don't log time in Jira, half of `estimate_accuracy.py` is dead on arrival and the invoice is the only hours record. | Assumed partially logged. Every function returns "insufficient data" rather than a flattering zero. |
| 4 | **Does Devin review every PR, or only some?** | `devin_findings` reports findings as a *share of judged PRs*. If Devin only runs on some repos, low findings means "not reviewed", not "clean". | Assumed partial. `reviews_fetched` distinguishes the two, but the roll-up needs your answer to be meaningful. |
| 5 | **Sprint dates in Jira** | "Availability vs Commitment" is empty because the sprint has no start/end dates. Capacity cannot be computed without them. This is a Jira config fix, not a code fix. | Left as-is; flagged in §3.3. |

---

## 1. Three live bugs, in priority order

**1. GitHub token returning 403.** Mid-session the PR tiles went from `126 merged (7d)` / `611 (30d)` to `—`, with the message *"PR charts need a GitHub token. (403 Client Error: Forbidden)"*. Same token, same session, ~40 minutes apart. That points at rate limiting or an expiring credential rather than a permissions change. Every PR metric in this proposal depends on that token, and §4 roughly triples the query cost, so this needs diagnosing before the new metrics ship. `pr_quality` degrades gracefully (falls back to the lean query, reports `unknown` rather than `0`) but it cannot invent data.

**2. `undefined` as a chart title.** The Ticket Composition donut rendered the literal word `undefined` where its title should be. Root cause found and fixed: `theme.plot()` set `title_font` unconditionally, which creates a `layout.title` object with no `text`; Streamlit's frontend then wraps `String(spec.layout.title.text)` in bold, and `String(undefined)` is the string `"undefined"`. Any chart without an explicit title hit this. `snapshot.py:720` has the same construct on the PDF path and should get the same guard.

**3. `/engineering` returns "Page not found."** The default page is served at `/` while its `url_path` is `"engineering"`, so the URL in your address bar 404s. Fixed by registering a second hidden page at that path.

Two smaller ones, both fixed: the Backlog Cleanup card renders `Epic: nan` (a float NaN passing through `x or 'none'`, which returns the NaN), and the Business page is titled "Jira Ticket Health Dashboard" above orders and ad spend.

---

## 2. Why it looks busy — the actual mechanism

Your read was right; here is what's producing it.

**Nineteen sections on one scrolling page, all at the same visual level.** Every section is an `st.subheader` separated by `st.divider()`. Nothing on the Engineering page is styled as more important than anything else, so the eye has no entry point and the page reads as one undifferentiated wall. Sections 3 and 15 have no heading at all.

**Roughly twenty KPI tiles before the first chart.** `_render_resolved_summary` fires 4, `_render_metrics` 6, `_render_pr_hygiene` 4, `_render_ticket_quality` 4. A tile is a claim on attention; twenty of them is none.

**The type scale had no ratio.** Sizes in use were 0.76, 0.78, 0.85, 0.95, 1, 1.25 and 2rem — seven sizes, no system. Worse, the smallest of them carried the load-bearing text: `.kpi-note` at 0.78rem holds every honest qualification on the page ("no owner set", "in dev/staging/prod, idle >6d"), at 40% of the size of the number it qualifies. Meanwhile `st.dataframe` had no override at all, so tables rendered smaller than body copy — on a page that is mostly tables.

**Pie charts with 23 slices.** "Who resolved tickets" and "Who merged PRs" both draw one slice per person, most of them unreadable slivers with a scrolling legend. A pie stops communicating past about six categories. This was the single worst offender.

**No maximum width.** `layout="wide"` with no constraint means a six-card KPI strip stretches across a 1900px monitor into a thin ribbon of tiny labels.

---

## 3. What changed on the branch

### 3.1 Type scale

One scale, one ratio: **13 / 14 / 15 / 17 / 20 / 32px** as named tokens in `theme.py`, referenced everywhere instead of ad-hoc rem values. Captions and metadata went from ~12.2px to 13px, body to 17px, and table cells now land at ~15px.

Table text needed a different lever than CSS. Streamlit draws `st.dataframe` on a **canvas** via glide-data-grid, so no stylesheet can reach the cells — the size derives from the root font size. Setting `baseFontSize = 17` in `config.toml` puts cells at ~14.9px and lifts the widget chrome with it. For the same reason a 600-weight header row is unreachable; a tinted header background separates it instead.

### 3.2 Charts

- **`rank_bar()`** replaces all three unreadable pies: horizontal bars, sorted descending, top 10 with the tail collapsed into a single `Other (N)` bar pinned last, value labels outside each bar, category labels never rotated.
- **`CATEGORICAL`** — one palette of 8 colourblind-safe hues (Okabe-Ito derived, with its blue swapped for your `#2563eb` and its illegible yellow for a deep violet), ordered so adjacent entries differ in luminance. Replaces five competing colour sets. The same hues go into `chartCategoricalColors` so Streamlit's native charts and Plotly figures finally agree.
- Rotated vertical axis labels on the per-assignee and team-sprint charts are now horizontal bars.
- `_STAGE_COLORS` re-hued — "Ready for Production" and "In Progress" were the *identical* green. All nine status pills verified at ≥4.5:1 contrast.

### 3.3 Honesty fixes

- **Estimate coverage now agrees with Policy Compliance.** The KPI strip's 49% and the Estimate Policy section's 61% were different numbers with no note saying so: the strip counted epics and initiatives that `hygiene.estimate_policy` deliberately exempts, via a dead code branch that never ran. The strip now calls `estimate_policy` directly.
- **Bots are out of person tables.** `devin-ai-integration` and `github-actions` were listed as people in "PR status by person", holding 12 and 7 open PRs. They now count in org totals but not in person rankings, and each site prints how many were excluded rather than silently dropping them.

### 3.4 The merchant roster — your point from tonight

"87% of 6,532 priced products cost more here than the market" was measured across **every merchant the feed remembers, including the disabled ones.** A switched-off shop's prices are nobody's decision, so they shouldn't be in the denominator of the number that section exists to state.

Implemented as `ACTIVE_MERCHANTS` (comma-separated). Unset, nothing changes. Set, disabled shops are dropped before anything is counted, and a caption states how many offers were left out and which roster produced the figure. Give me the four exact names and I'll wire them in.

Two related findings on that page: **"Their Vivino price" can never populate** — you said Vivino has blocked you, but the tab shows "Pick a merchant above", implying it would work if you did. It should say it's unavailable and why. And in **"Most clicked"**, four of ten rows show `(none)` as the wine with a blank merchant, so 40% of your most-clicked table is unidentifiable.

---

## 4. The information architecture I recommend

**Not implemented — this is the part to decide before anyone builds it.** Restructuring 19 sections is a large change and I didn't want to make it unilaterally overnight.

Split the Engineering page into five, using the `st.navigation` mechanism already in place:

| Page | Contains | Answers |
|---|---|---|
| **Today** | 6 tiles, and the three things needing a decision now | "What do I do this morning?" |
| **People** | Leaderboard, per-person scorecard, capacity | "Who is actually delivering?" |
| **Delivery** | Composition, epics, prioritized queue, stale, backlog cleanup | "What's the state of the work?" |
| **Code** | PRs, hygiene, Devin findings, review citizenship | "Is the code any good?" |
| **Planning** | Sprint planner, sprint capacity | "What are we committing to?" |
| **Integrity** | Gaming signals with evidence — CEO only | "Is anyone playing the metrics?" |

Two structural notes. First, **there is no leaderboard today** — the KPI score exists only per-person, one person at a time, so you cannot compare engineers anywhere in the app. For your stated problem that's the single most important missing view. Second, the Engineering page currently renders a *completely different* page in Individual scope, which is the right instinct; the split above just makes it explicit.

The "Today" page should lead with the number that actually matters. Right now the most alarming fact on the dashboard — **73 of 79 open PRs have no approving review, and 21 have never been reviewed at all** — sits mid-page as plain text. 92% of open PRs unreviewed is arguably the headline finding of the entire system.

---

## 5. Rollout order

1. **Diagnose the GitHub 403.** Everything downstream depends on it.
2. **Merge this branch.** Bug fixes and visual system, 446 tests passing, no metric definitions changed.
3. **Set `ACTIVE_MERCHANTS`.** One env var, immediate correction to a headline number.
4. **Wire the integrity panel** (`integrity.py` is built and tested; it needs a UI and the `changelog` column, which is now plumbed).
5. **Ship the new PR metrics** (`pr_quality.py` built and tested; needs the extended GraphQL, which is written).
6. **Then** restructure the IA, once you've decided §4.

Do not do 6 before 1–5. The metrics are the thing you actually need; the layout is what makes them readable.
