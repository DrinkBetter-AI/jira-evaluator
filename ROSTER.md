# Roster — confirmed by Angel, 16 Aug 2026

Source of truth for `JIRA_ROLES` and `GITHUB_LOGIN_MAP` until moved into env.
Everyone is hourly **except** Angel (CEO) and Arsalan (CTO).

## Active — engineering (12)

| Person | Role | Notes |
|---|---|---|
| Tam | platform (backend + search + MLOps + DevOps) | touches every repo · GitHub **Phelan164** (confirmed) — #1 producer, 153 merged/30d |
| Shawn | backend | |
| Mohsen Davoudi | frontend | GitHub MohsenStack (unconfirmed) |
| David | frontend | GitHub **ahref13** (confirmed) |
| Farid Shahidi | frontend + mobile | GitHub faridsh69 (unconfirmed) |
| Ali | mobile (React Native) | GitHub alivinovoss (unconfirmed) |
| Jal Haidar | CRM/Medusa backend + merchant data | works with Actenzo (NL data provider) · GitHub jal-vino (unconfirmed) |
| Anouar Kacem | CRM/Medusa backend (marketplace) | Tunisia · GitHub anouar-source (unconfirmed) |
| Santi Caamaño | QA — automated | Uruguay · transparency concern (Angel) · GitHub SantiVinoVoss (unconfirmed) |
| Mehdi Ordikhani | AI recommendation (ML team) | very expensive/hr |
| Gaston | infrastructure | expensive/hr · padding concern (Angel) · GitHub gsalgado-cloudacio (unconfirmed) |
| Dina QA | QA — manual | Vietnam · cheap, very good |

## Active — non-engineering

| Person | Role | Notes |
|---|---|---|
| Mihai Manea | PM — all boards (ML/Marketplace/App/Mobile) | owns the departed-staff cleanup below |
| Robert Surpateanu | designer | very expensive/hr · GitHub robertsurpe (unconfirmed) |
| Alesya Kasovich | designer | cheap/hr |
| Igor Taborsak | SEO | |
| Evmorfia Kostaki, Matthew, Sylvia | wine experts team | |
| zoe | advisor | |
| Jim | ML advisor | not in Jira assignee data |
| Angel Vossough | CEO | GitHub **avosmod8** (confirmed) — #2 producer, 132 merged/30d · exempt |
| Arsalan | CTO | GitHub arsalanvm (unconfirmed) · exempt from hourly-incentive metrics |

## Former / inactive — 21 names still present in board data

Sai Shankar (test eng), Sarju (frontend), Yantao He (CRM/Medusa), Dan O'Sullivan
(Medusa), Shivanand (ML), Jon Wang (frontend), Kevin Cai (frontend), Ramin Shahid
(head of marketplace), Amir (mobile), Christina Lo (designer), Aleksei Pinchuk
(frontend), Saji (UX), Mark (frontend), Courtney McNeil (designer), Lotte Karolina
(wine), Jennifer (HR), Eva van Wielink (wine), Stanislav (ML), Saeid Parsa (ML),
Armine Aproyan (PM), **Haichen Song (frontend — corrected: earlier recorded as
active backend, Angel corrected to former)**.

**This is the headline:** 21 former people vs 12 active engineers in the same
assignee data. The "23 assignees" scope, the unassigned counts, the stale queues
and the resolved rankings all mix ghosts with staff. Sai Shankar's 194 "resolved
in 30d" — second highest in the company — is a departed person collecting credit
whenever anyone moves a ticket still assigned to him: the current-assignee
attribution flaw, demonstrated by the data itself.

## Unmapped GitHub logins (confirm or retire)

`tungph`, `Morse-vv`, `lawrnsfeng`, `VossBackend` — real PR authors, no owner
named. `VossQABot`, `devin-ai-integration`, `github-actions` are bots (already
filtered).

## Env values (paste-ready; unconfirmed logins marked)

```
JIRA_ROLES="platform=Tam;backend=Shawn;frontend=Mohsen Davoudi,David;frontend-mobile=Farid Shahidi;mobile=Ali;crm-backend=Jal Haidar,Anouar Kacem;qa-automated=Santi Caamaño;qa-manual=Dina QA;ai-recommendation=Mehdi Ordikhani;infrastructure=Gaston;designer=Robert Surpateanu,Alesya Kasovich;pm=Mihai Manea;seo=Igor Taborsak;wine=Evmorfia Kostaki,Matthew,Sylvia;advisor=zoe;exec=Angel Vossough,Arsalan"

GITHUB_LOGIN_MAP="Tam=Phelan164;David=ahref13;Angel Vossough=avosmod8;Arsalan=arsalanvm;Mohsen Davoudi=MohsenStack;Farid Shahidi=faridsh69;Ali=alivinovoss;Jal Haidar=jal-vino;Anouar Kacem=anouar-source;Santi Caamaño=SantiVinoVoss;Gaston=gsalgado-cloudacio;Robert Surpateanu=robertsurpe"
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
