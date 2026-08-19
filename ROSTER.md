# Roster — confirmed by Angel, 16 Aug 2026

Source of truth for `JIRA_ROLES` and `GITHUB_LOGIN_MAP` until moved into env.
Everyone is hourly **except** Angel (CEO) and Arsalan (CTO).

## Active — engineering (12)

| Person | Role | Notes |
|---|---|---|
| Tam | platform (backend + search + MLOps + DevOps) | touches every repo · GitHub **Phelan164** (confirmed) — #1 producer, 153 merged/30d |
| Shawn | backend | GitHub **tungph** (confirmed) |
| Mohsen Davoudi | frontend | GitHub **MohsenStack** (confirmed) |
| David | frontend | GitHub **ahref13** (confirmed) |
| Farid Shahidi | frontend + mobile | GitHub **faridsh69** (confirmed) |
| Ali | mobile (React Native) | GitHub **alivinovoss** (confirmed) |
| Jal Haidar | CRM/Medusa backend + merchant data | works with Actenzo (NL data provider) · GitHub **jal-vino** (confirmed) |
| Anouar Kacem | CRM/Medusa backend (marketplace) | Tunisia · GitHub **anouar-source** (confirmed) |
| Santi Caamaño | QA — automated | Uruguay · transparency concern (Angel) · GitHub **SantiVinoVoss** (confirmed) |
| Mehdi Ordikhani | AI recommendation (ML team) | very expensive/hr · GitHub **Morse-vv** (confirmed) |
| Gaston | infrastructure | expensive/hr · padding concern (Angel) · GitHub **gsalgado-cloudacio** (confirmed) |
| Dina QA | QA — manual | Vietnam · cheap, very good |

## Active — non-engineering

| Person | Role | Notes |
|---|---|---|
| Mihai Manea | PM — all boards (ML/Marketplace/App/Mobile) | owns the departed-staff cleanup below |
| Robert Surpateanu | designer | very expensive/hr · GitHub **robertsurpe** (confirmed) |
| Alesya Kasovich | designer | cheap/hr |
| Igor Taborsak | SEO | |
| Evmorfia Kostaki, Matthew, Sylvia | wine experts team | |
| zoe | advisor | |
| Jim | ML advisor | not in Jira assignee data |
| Praveen Rai | biz dev | super expensive/hr · **the only person not on Clockify** |
| Angel Vossough | CEO | GitHub **avosmod8** (confirmed) — #2 producer, 132 merged/30d · exempt |
| Arsalan | CTO | GitHub **arsalanvm** (confirmed) · exempt from hourly-incentive metrics |

## Former / inactive — 21 names still present in board data

Sai Shankar (test eng), Sarju (frontend), Yantao He (CRM/Medusa), Dan O'Sullivan
(Medusa), Shivanand (ML), Jon Wang (frontend), Kevin Cai (frontend), Ramin Shahid
(head of marketplace), Amir (mobile), Christina Lo (designer), Aleksei Pinchuk
(frontend), Saji (UX), Mark (frontend), Courtney McNeil (designer), Lotte Karolina
(wine), Jennifer (HR), Eva van Wielink (wine), Stanislav (ML), Saeid Parsa (ML),
Armine Aproyan (PM), **Haichen Song (frontend — corrected: earlier recorded as
active backend, Angel corrected to former)**, Dat / Đào Nguyễn Anh (QA — fired;
still appearing in standup invites and giving updates as recently as 13 Aug per
Fireflies, so confirm his access and invites are actually revoked).

**This is the headline:** 21 former people vs 12 active engineers in the same
assignee data. Verified 19 Aug 2026: the former names hold **zero open
tickets** — the earlier "21 ghosts holding tickets" line was wrong and is
retired. What they do hold is historical *resolutions*, which is how
Sai Shankar collects 194 "resolved in 30d" — second highest in the company —
whenever anyone moves a ticket still assigned to him. That is the
current-assignee attribution flaw, demonstrated by the data itself, and it is
fixed by crediting the changelog author of the resolving transition.

The real board residue, measured rather than assumed:

| Residue | Count |
|---|---|
| Open tickets with no assignee | 88 |
| Open tickets with no priority | 168 |
| Open tickets idle > 90 days | 94 |
| Open tickets assigned to former staff (ghost-assigned) | 0 |

## Unmapped GitHub logins (confirm or retire)

`lawrnsfeng` and `VossBackend` — real PR authors, still no owner named; likely
former staff or service accounts, confirm or retire their access. `VossQABot`,
`devin-ai-integration`, `github-actions` are bots (already filtered).

## Env values (paste-ready; unconfirmed logins marked)

```
JIRA_ROLES="platform=Tam;backend=Shawn;frontend=Mohsen Davoudi,David;frontend-mobile=Farid Shahidi;mobile=Ali;crm-backend=Jal Haidar,Anouar Kacem;qa-automated=Santi Caamaño;qa-manual=Dina QA;ai-recommendation=Mehdi Ordikhani;infrastructure=Gaston;designer=Robert Surpateanu,Alesya Kasovich;pm=Mihai Manea;seo=Igor Taborsak;wine=Evmorfia Kostaki,Matthew,Sylvia;advisor=zoe;exec=Angel Vossough,Arsalan"

GITHUB_LOGIN_MAP="Tam=Phelan164;Shawn=tungph;David=ahref13;Mehdi Ordikhani=Morse-vv;Angel Vossough=avosmod8;Arsalan=arsalanvm;Mohsen Davoudi=MohsenStack;Farid Shahidi=faridsh69;Ali=alivinovoss;Jal Haidar=jal-vino;Anouar Kacem=anouar-source;Santi Caamaño=SantiVinoVoss;Gaston=gsalgado-cloudacio;Robert Surpateanu=robertsurpe"  # all confirmed by Angel 16 Aug 2026
```

## Consequences for the plan (feeds DEVIN_PLAN.md)

1. **New pre-WP task for Mihai:** departed-staff sweep — reassign or close every
   open ticket held by the 21 former people (55 "unassigned" undercounts the
   ownerless problem; ghost-assigned is worse than unassigned).
2. **WP3 gains a rule:** resolutions credited to a person on the former list are
   flagged, not counted — and attribution by changelog author makes the Sai
   artifact impossible going forward.
3. **WP5 rubric table needs two more columns:** designer (Robert is expensive and
   his design tickets were among the longest-idle on the board) and
   infrastructure (Gaston — infra work rarely maps to tickets/PRs one-to-one, so
   his rubric leans on estimate accuracy and hours-vs-delivery, exactly the
   padding checks).
4. **Scorecard population is 12 engineers + 2 designers + PM + QA** — small
   enough that peer-relative components need care (min 3 peers per role family;
   fold frontend+mobile for comparison purposes).
5. Angel and Arsalan appear in output charts but are exempt from
   contractor-incentive metrics (estimate padding, hours ratios).

## Late confirmations (16 Aug, second pass)

Every login above is now **confirmed by Angel** except `lawrnsfeng` and
`VossBackend`, which remain unowned. Devin reviews **every PR in every repo**,
so AI-review findings can be a *scored* component for all code roles, not
evidence-only. Integrity-page access: **decided — Angel only.** WP6 implements a second
admin credential (the shared dashboard password stays for everyone else); the
integrity computations must not run at all for non-admin sessions.

## Time tracking (revealed 16 Aug)

The team bills hours through **Clockify** — everyone except Praveen Rai. This is
the invoiced-hours source the KPI spec flagged as the only true cross-check:
Clockify hours per person per week, joined against delivered work (size-weighted
PRs, resolved tickets by changelog author) and cycle time, closes the loop that
Jira estimates and Jira worklogs cannot. Needs a Clockify API key (workspace
reports endpoint, read-only) — see DEVIN_PLAN WP11.
