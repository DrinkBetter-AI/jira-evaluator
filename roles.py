"""The roster and the rubric table: who plays what role, and how they're scored.

WP5's gap, closed here: ``JIRA_ROLES`` names 16 role keys and the scorecard in
``kpi.py`` only ever knew how to score one shape of person (a code-producing
engineer). This module is the roster (who is who, active or former, Jira name
<-> GitHub login both ways) and the rubric table (which of the 16 keys get
scored, on what, and which are explicitly left unscored).

The trap this task exists to close: ``roles_template.env`` used to carry a
stale 10-key template (``mlops-ai``, ``recommendation``, ``bizdev``) that did
not match ``ROSTER.md`` at all, so a rubric lookup keyed off that template
silently dropped six roles - nobody saw an error, the six roles just never
got scored. The fix isn't only "use the new file"; it's that nothing in this
module is allowed to key off anything *other* than the live, parsed roster.
``rubric_for_role`` returns an explicit ``"unclassified"`` status - never a
silent pass-through - for any role string it doesn't recognise, and
``tests/test_roles.py`` parametrises over the roles actually parsed out of
``JIRA_ROLES`` at collection time, so a role key added to the env without a
rubric decision fails the build instead of failing silently in production.

Defaults are baked in from ``roles_template.env`` (regenerated 20 Aug 2026
from the org-chart upload)
as literal strings below, because the Cloud Run env vars for this app have
not been set - that is the actual deployed state today, not a hypothetical -
and the dashboard has to be correct anyway. Each of the four env vars falls
back to its own baked default independently; when the env var is present it
wins over the default for that var alone.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Mapping

# ---------------------------------------------------------------------------
# Baked-in defaults, copied verbatim from roles_template.env (20 Aug 2026
# regen, org-chart upload). Not read from disk at runtime - a missing or stale file on a
# deploy target must not be able to blank out the roster.
# ---------------------------------------------------------------------------

_ROLES_VAR = "JIRA_ROLES"
_LOGIN_VAR = "GITHUB_LOGIN_MAP"
_FORMER_VAR = "JIRA_FORMER_STAFF"
_UNMAPPED_VAR = "GITHUB_UNMAPPED_AUTHORS"

_DEFAULT_JIRA_ROLES = (
    "platform=Tam;backend=Shawn;frontend=Mohsen Davoudi,David;"
    "frontend-mobile=Farid Shahidi;mobile=Ali;"
    "crm-backend=Jal Haidar,Anouar Kacem;qa-automated=Santi Caamaño;"
    "qa-manual=Dina QA;ai-recommendation=Mehdi Ordikhani,Shiva Naduvin;"
    "infrastructure=Gaston;designer=Robert Surpateanu,Alesya Kasovich;"
    "pm=Mihai Manea;seo=Igor Taborsak;"
    "wine=Evmorfia Kostaki,Matthew,Sylvia;advisor=zoe,Jim;"
    "exec=Angel Vossough,Arsalan"
)

_DEFAULT_GITHUB_LOGIN_MAP = (
    "Tam=Phelan164;Shawn=tungph;David=ahref13;Mehdi Ordikhani=Morse-vv;"
    "Angel Vossough=avosmod8;Arsalan=arsalanvm;Mohsen Davoudi=MohsenStack;"
    "Farid Shahidi=faridsh69;Ali=alivinovoss;Jal Haidar=jal-vino;"
    "Anouar Kacem=anouar-source;Santi Caamaño=SantiVinoVoss;"
    "Gaston=gsalgado-cloudacio;Robert Surpateanu=robertsurpe"
)

_DEFAULT_JIRA_FORMER_STAFF = (
    "Sai Shankar;Sarju;Yantao He;Dan O'Sullivan;Shivanand;Jon Wang;Kevin Cai;"
    "Ramin Shahid;Amir;Christina Lo;Aleksei Pinchuk;Saji;Mark;Courtney McNeil;"
    "Lotte Karolina;Jennifer;Eva van Wielink;Stanislav;Saeid Parsa;"
    "Armine Aproyan;Haichen Song;Dat"
)

_DEFAULT_GITHUB_UNMAPPED_AUTHORS = "lawrnsfeng;VossBackend"

# The 16 valid role keys, in the order roles_template.env documents them.
ROLE_ORDER: tuple[str, ...] = (
    "platform",
    "backend",
    "frontend",
    "frontend-mobile",
    "mobile",
    "crm-backend",
    "qa-automated",
    "qa-manual",
    "ai-recommendation",
    "infrastructure",
    "designer",
    "pm",
    "seo",
    "wine",
    "advisor",
    "exec",
)

ROLE_LABELS: dict[str, str] = {
    "platform": "Platform",
    "backend": "Backend",
    "frontend": "Frontend",
    "frontend-mobile": "Frontend (mobile-capable)",
    "mobile": "Mobile",
    "crm-backend": "Marketplace / CRM backend",
    "qa-automated": "QA (automated)",
    "qa-manual": "QA (manual)",
    "ai-recommendation": "AI / recommendation",
    "infrastructure": "Infrastructure",
    "designer": "Design",
    "pm": "PM",
    "seo": "SEO",
    "wine": "Wine experts",
    "advisor": "Advisor",
    "exec": "Exec",
}

# Roles graded on the same rubric as an ordinary code-producing engineer -
# the rubric kpi.components()/kpi.overall() already implement. Farid sits in
# frontend-mobile rather than plain frontend, and Tam sits in platform rather
# than plain backend, but both ship PRs against Jira tickets the same way
# everyone else in this group does, so they share its rubric.
CODE_ROLES: frozenset[str] = frozenset(
    {
        "platform",
        "backend",
        "frontend",
        "frontend-mobile",
        "mobile",
        "crm-backend",
        "ai-recommendation",
    }
)

# Roles listed on the People page but with no rubric yet - explicitly
# unscored, never scored as zero. SEO (Igor) and the roles with no Jira
# delivery work at all (wine, advisor) all land here.
NO_RUBRIC_ROLES: frozenset[str] = frozenset({"seo", "wine", "advisor"})

EXEC_ROLE = "exec"
EXEC_REASON = "exec — not scored"

# Below this share of a rubric's documented weight, ``overall`` withholds a
# headline rather than average over a fragment - the same 60% rule
# kpi.MIN_SCORABLE_WEIGHT already enforces for the code rubric, generalised
# here to any rubric's own total.
MIN_SCORABLE_FRACTION = 0.6

# Minimum peers a cohort needs before a peer-relative comparison means
# anything. Distinct from kpi.MIN_PEERS (which gates a different, ticket-count
# peer percentile inside the code rubric) - same value, different question,
# kept as its own constant so this module has no import-time dependency on
# kpi.py.
MIN_PEERS = 3

# frontend + frontend-mobile + mobile share one comparison cohort (ROSTER.md:
# three people, three different keys, one team - splitting them leaves each
# alone). Every other role, including platform and backend, stands on its
# own; nothing else folds automatically.
_FOLDED_COHORT_ID = "frontend+frontend-mobile+mobile"
_FOLD_ROLES: frozenset[str] = frozenset({"frontend", "frontend-mobile", "mobile"})


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Person:
    """One roster entry: a Jira display name, a role, and active status.

    ``role`` is ``None`` for former staff (they carry no role once departed -
    the roster's job is to say they're gone, not to keep scoring them) and
    for a GitHub login mapped to nobody the roles/former lists name.
    """

    name: str
    role: str | None
    active: bool
    github_login: str | None = None

    @property
    def key(self) -> str:
        """Case-insensitive identity used for all roster lookups."""
        return self.name.strip().lower()


@dataclass(frozen=True)
class Roster:
    """The parsed roster: every name in ``JIRA_ROLES``/``JIRA_FORMER_STAFF``,
    plus the GitHub login round trip.

    Build with :func:`load_roster`, not by hand - the constructor takes
    already-parsed data because parsing and lookup are different failure
    modes and mixing them made the stale-template bug this task fixes harder
    to see.
    """

    people: dict[str, Person]
    unmapped_logins: tuple[str, ...]
    login_index: dict[str, str] = field(default_factory=dict)

    def person(self, name: str) -> Person | None:
        return self.people.get(str(name).strip().lower())

    def role_of(self, name: str) -> str | None:
        who = self.person(name)
        return who.role if who else None

    def is_active(self, name: str) -> bool | None:
        """True/False for a known person, ``None`` if the name isn't on the roster at all."""
        who = self.person(name)
        return who.active if who else None

    def login_for(self, name: str) -> str | None:
        who = self.person(name)
        return who.github_login if who else None

    def name_for_login(self, login: str) -> str | None:
        key = self.login_index.get(str(login).strip().lower())
        who = self.people.get(key) if key else None
        return who.name if who else None

    def roles(self) -> tuple[str, ...]:
        """Distinct role keys actually present in the roster, sorted."""
        return tuple(sorted({p.role for p in self.people.values() if p.role}))

    def people_in_role(self, role: str, active_only: bool = True) -> tuple[Person, ...]:
        return tuple(
            p
            for p in self.people.values()
            if p.role == role and (p.active or not active_only)
        )


def _env_value(env: Mapping[str, str] | None, name: str, default: str) -> str:
    source = env if env is not None else os.environ
    value = source.get(name)
    return value if value not in (None, "") else default


def load_roster(env: Mapping[str, str] | None = None) -> Roster:
    """Parse the roster from env (falling back per-variable to the baked defaults).

    Order matters: ``JIRA_ROLES`` seeds active people, ``JIRA_FORMER_STAFF``
    is applied after and overrides same-name entries to ``active=False,
    role=None`` (nobody should be both listed as active and flagged
    departed), and ``GITHUB_LOGIN_MAP`` is applied last, attaching a login to
    whichever entry already exists for that name. Names are matched
    case-insensitively for roster lookups; a malformed pair costs that pair,
    not the whole parse.
    """
    roles_raw = _env_value(env, _ROLES_VAR, _DEFAULT_JIRA_ROLES)
    logins_raw = _env_value(env, _LOGIN_VAR, _DEFAULT_GITHUB_LOGIN_MAP)
    former_raw = _env_value(env, _FORMER_VAR, _DEFAULT_JIRA_FORMER_STAFF)
    unmapped_raw = _env_value(env, _UNMAPPED_VAR, _DEFAULT_GITHUB_UNMAPPED_AUTHORS)

    people: dict[str, Person] = {}

    for pair in roles_raw.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        role, _, names = pair.partition("=")
        role = role.strip()
        if not role:
            continue
        for name in names.split(","):
            name = name.strip()
            if not name:
                continue
            entry = Person(name=name, role=role, active=True)
            people[entry.key] = entry

    for name in former_raw.split(";"):
        name = name.strip()
        if not name:
            continue
        entry = Person(name=name, role=None, active=False)
        people[entry.key] = entry

    login_index: dict[str, str] = {}
    for pair in logins_raw.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, _, login = pair.partition("=")
        name = name.strip()
        login = login.strip()
        if not name or not login:
            continue
        key = name.strip().lower()
        existing = people.get(key)
        if existing is not None:
            people[key] = dataclasses.replace(existing, github_login=login)
        login_index[login.lower()] = key

    unmapped = tuple(sorted({n.strip() for n in unmapped_raw.split(";") if n.strip()}))

    return Roster(people=people, unmapped_logins=unmapped, login_index=login_index)


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Weight:
    """One scored component of a rubric.

    ``blind_spot`` is required and non-empty: KPI_SPEC.md §1 rule 3 says
    every metric has to name what it can't see, so a component that skips
    this fails construction instead of shipping silently unqualified.
    """

    name: str
    points: float
    blind_spot: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("rubric component name cannot be blank")
        if self.points <= 0:
            raise ValueError(f"rubric component '{self.name}' must carry positive weight")
        if not self.blind_spot.strip():
            raise ValueError(f"rubric component '{self.name}' has no documented blind spot")


@dataclass(frozen=True)
class Rubric:
    """A role's scoring rubric: named, weighted components summing to ``total``.

    The sum is checked at construction and raises on mismatch - it does not
    round or renormalise into agreement. A rubric whose weights don't add up
    is a bug in the table; the fix is to fix the table, and finding that out
    at import time beats finding it out from a scorecard that quietly
    over- or under-weights someone.
    """

    rubric_id: str
    label: str
    components: tuple[Weight, ...]
    total: float = 100.0

    def __post_init__(self) -> None:
        names = [c.name for c in self.components]
        if len(names) != len(set(names)):
            raise ValueError(f"rubric '{self.rubric_id}' has duplicate component names")
        summed = sum(c.points for c in self.components)
        if abs(summed - self.total) > 1e-9:
            raise ValueError(
                f"rubric '{self.rubric_id}' weights sum to {summed}, not the "
                f"documented total {self.total} - fix the table, do not round"
            )

    def weights(self) -> dict[str, float]:
        return {c.name: c.points for c in self.components}


# The code rubric. These nine components/weights are a hand-kept copy of
# kpi.WEIGHTS, not an import of it - roles.py must not import kpi.py, because
# kpi.py imports roles.py for rubric selection and a cycle would follow.
# tests/test_kpi.py cross-checks this copy against kpi.WEIGHTS on every run
# so the two cannot silently drift apart.
CODE_RUBRIC = Rubric(
    rubric_id="code",
    label="Engineering (platform / backend / frontend / mobile / marketplace / ML)",
    components=(
        Weight(
            "Delivery",
            10.0,
            "rewards volatility off a low self-baseline and cannot see ticket difficulty at all",
        ),
        Weight(
            "Delivery vs team",
            10.0,
            "a percentile is blind to difficulty too - whoever draws the hard tickets resolves fewer and ranks lower for it",
        ),
        Weight(
            "Rework",
            20.0,
            "a reopen that gets silently re-resolved can vanish from this - see integrity.reresolve_events for the honest read",
        ),
        Weight(
            "Weekly updates",
            15.0,
            "idle_days resets on any field edit, including a cosmetic one - see integrity.status_age_days / cosmetic_touches",
        ),
        Weight(
            "Staleness",
            10.0,
            "same idle_days blind spot as Weekly updates: a groomed ticket reads fresh here too",
        ),
        Weight(
            "Carry-over",
            10.0,
            "only counts sprint history while a ticket stays open; pulling it out of every sprint erases the trail",
        ),
        Weight(
            "Estimates",
            10.0,
            "checks presence, not accuracy - pairs with estimate_accuracy.py, which is the actual padding cross-check",
        ),
        Weight(
            "Devin-ready docs",
            10.0,
            "a declared quality score, graded by whoever scores it - not a machine-recorded fact",
        ),
        Weight(
            "Urgent response",
            5.0,
            "priority is self-set; inflating it doesn't help this component but does help almost everything else on the board",
        ),
    ),
)

QA_AUTOMATED_RUBRIC = Rubric(
    rubric_id="qa-automated",
    label="QA (automated)",
    components=(
        Weight(
            "Defect escape rate",
            30.0,
            "only counts bugs someone eventually filed and linked back - a bug nobody ever reported is invisible here",
        ),
        Weight(
            "Verification cycle time",
            20.0,
            "measures time-in-status only, not CI queue time or flaky-rerun delay outside this person's control",
        ),
        Weight(
            "Automation coverage added",
            20.0,
            "counts test files/checks touched in a PR, not whether the tests assert anything - a no-op test passes this",
        ),
        Weight(
            "Rework after verification",
            20.0,
            "sees only reopen/re-resolve events still visible in the current status - see integrity.reresolve_events for the exploit this alone would miss",
        ),
        Weight(
            "Estimate accuracy",
            10.0,
            "logged time is self-reported too; this checks consistency between two declared numbers, not ground truth",
        ),
    ),
)

QA_MANUAL_RUBRIC = Rubric(
    rubric_id="qa-manual",
    label="QA (manual)",
    components=(
        Weight(
            "Defect escape rate",
            35.0,
            "only counts bugs someone eventually filed and linked back - a bug nobody ever reported is invisible here",
        ),
        Weight(
            "Bug validity rate",
            25.0,
            "'valid' is decided by whoever triages the bug, not this person - a strict triager can make thorough QA look worse",
        ),
        Weight(
            "Verification cycle time",
            20.0,
            "measures time-in-status only, not the time actually spent testing by hand",
        ),
        Weight(
            "Rework after verification",
            10.0,
            "sees only reopen/re-resolve events still visible in the current status",
        ),
        Weight(
            "Estimate accuracy",
            10.0,
            "logged time is self-reported too; this checks consistency between two declared numbers, not ground truth",
        ),
    ),
)

PM_RUBRIC = Rubric(
    rubric_id="pm",
    label="PM",
    components=(
        Weight(
            "Triage latency",
            30.0,
            "measures time to the first triage action in the changelog, not whether the triage decision itself was right",
        ),
        Weight(
            "Board hygiene",
            30.0,
            "hygiene is nudged by a PM, not enforced - a low score can mean an uncooperative team as much as an absent PM",
        ),
        Weight(
            "Estimate coverage",
            20.0,
            "checks that an estimate exists before work starts, not whether it's accurate",
        ),
        Weight(
            "Stale queue rate",
            20.0,
            "status_age_days can't distinguish 'nobody is working this' from 'deliberately deprioritized' - both look identical here",
        ),
    ),
)

DESIGNER_RUBRIC = Rubric(
    rubric_id="designer",
    label="Design",
    components=(
        Weight(
            "Cycle time",
            35.0,
            "counts time-in-status; a ticket blocked on stakeholder feedback reads the same as one nobody is touching",
        ),
        Weight(
            "Estimate accuracy",
            30.0,
            "logged time is self-reported; this is a cross-check between two declared numbers, not a ground-truth measurement",
        ),
        Weight(
            "Staleness",
            20.0,
            "the same idle-days measurement the engineering rubric uses, with the same cosmetic-edit blind spot",
        ),
        Weight(
            "Handoff completeness",
            15.0,
            "a declared quality/handoff score, graded by whoever scores it - not a machine fact",
        ),
    ),
)

INFRASTRUCTURE_RUBRIC = Rubric(
    rubric_id="infrastructure",
    label="Infrastructure",
    components=(
        Weight(
            "Estimate accuracy",
            35.0,
            "logged time is self-reported; this detects inconsistency between two declared numbers and one recorded one, not ground truth (KPI_SPEC.md §3.2)",
        ),
        Weight(
            "Hours vs delivered output",
            30.0,
            "infra work often has no PR or ticket to divide hours by at all - a real measurement gap, not a false positive, when there's nothing to compare against",
        ),
        Weight(
            "Estimate churn",
            20.0,
            "only sees estimates revised inside Jira; a padded estimate set once and never touched again is invisible to churn",
        ),
        Weight(
            "Cycle time",
            15.0,
            "the one input here that resists gaming by editing fields, but it says nothing about hours billed against it",
        ),
    ),
)

# Role key -> rubric, for every role with a rubric decided.
ROLE_RUBRIC: dict[str, Rubric] = {
    **{role: CODE_RUBRIC for role in CODE_ROLES},
    "qa-automated": QA_AUTOMATED_RUBRIC,
    "qa-manual": QA_MANUAL_RUBRIC,
    "pm": PM_RUBRIC,
    "designer": DESIGNER_RUBRIC,
    "infrastructure": INFRASTRUCTURE_RUBRIC,
}

# Defensive check, run at import time: every one of the 16 documented role
# keys must be classified as scored, explicitly no-rubric, or exec - the
# exact three-way split this module exists to guarantee. If a role is ever
# added to ROLE_ORDER without updating one of the three tables below, import
# fails loudly here instead of the gap surfacing as a silently-dropped role
# in production, which is the original bug this task closes.
_classified = set(ROLE_RUBRIC) | NO_RUBRIC_ROLES | {EXEC_ROLE}
if _classified != set(ROLE_ORDER):
    _missing = set(ROLE_ORDER) - _classified
    _extra = _classified - set(ROLE_ORDER)
    raise AssertionError(
        "roles.py's rubric classification doesn't match ROLE_ORDER - "
        f"missing: {sorted(_missing) or 'none'}, unexpected: {sorted(_extra) or 'none'}"
    )
del _classified


@dataclass(frozen=True)
class RubricLookup:
    """The result of asking "what rubric applies to this role/person".

    ``status`` is one of:
    - ``"scored"``: ``rubric`` is set, score them on it.
    - ``"no_rubric"``: role is real and listed, but has no rubric yet
      (seo, wine, advisor) - render the "no rubric defined" sentinel, never
      a zero.
    - ``"exec"``: never scored, by design - board attribution only.
    - ``"role_unknown"``: the person isn't on the roster at all.
    - ``"unclassified"``: the role string is real (came from a live roster
      parse) but isn't one of the 16 keys this module knows how to classify -
      this should be unreachable given the import-time check above, and its
      existence is what makes the "adding a role without a rubric fails the
      build" test possible.
    """

    role: str | None
    status: str
    rubric: Rubric | None
    reason: str


NO_RUBRIC_REASON = "no rubric defined"


def rubric_for_role(role: str | None) -> RubricLookup:
    if role is None:
        return RubricLookup(role=None, status="role_unknown", rubric=None, reason="role unknown")
    if role == EXEC_ROLE:
        return RubricLookup(role=role, status="exec", rubric=None, reason=EXEC_REASON)
    if role in NO_RUBRIC_ROLES:
        return RubricLookup(
            role=role,
            status="no_rubric",
            rubric=None,
            reason=f"{NO_RUBRIC_REASON} for role '{role}'",
        )
    rubric = ROLE_RUBRIC.get(role)
    if rubric is not None:
        return RubricLookup(
            role=role,
            status="scored",
            rubric=rubric,
            reason=f"scored on the {rubric.rubric_id} rubric",
        )
    return RubricLookup(
        role=role,
        status="unclassified",
        rubric=None,
        reason=f"role '{role}' has no rubric decision recorded - not one of the 16 known keys",
    )


def rubric_for_person(roster: Roster, person: str) -> RubricLookup:
    """Which rubric applies to ``person``, resolved through the roster.

    Anyone the roster doesn't recognise comes back ``"role_unknown"`` - not
    scored, and never scored wrongly by default.
    """
    return rubric_for_role(roster.role_of(person))


# ---------------------------------------------------------------------------
# Overall score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverallResult:
    score: float | None
    reason: str


def overall(
    person: str,
    roster: Roster | None = None,
    component_scores: Mapping[str, float] | None = None,
) -> OverallResult:
    """Weighted mean of ``component_scores`` against ``person``'s rubric, or ``None`` and why.

    This is the generic engine every non-code rubric (QA, PM, designer,
    infrastructure) scores against once its own component-computation lands;
    the code rubric keeps using ``kpi.components()``/``kpi.overall()``
    directly since those already exist and this function would only be a
    slower path to the same arithmetic for that one rubric.

    ``component_scores`` maps a rubric component name to a 0-100 score. A
    name the rubric doesn't recognise is ignored, not an error - callers
    computing a subset of components (because some inputs aren't wired up
    yet) get "insufficient data" instead of a crash. Below
    ``MIN_SCORABLE_FRACTION`` of the rubric's total weight, the score is
    withheld the same way ``kpi.overall`` withholds a headline built on a
    fragment of the code rubric.
    """
    ros = roster if roster is not None else load_roster()
    lookup = rubric_for_person(ros, person)

    if lookup.status == "exec":
        return OverallResult(None, EXEC_REASON)
    if lookup.status == "no_rubric":
        return OverallResult(None, lookup.reason)
    if lookup.status in ("role_unknown", "unclassified"):
        return OverallResult(None, "role unknown")

    rubric = lookup.rubric
    assert rubric is not None  # only "scored" reaches here
    weights = rubric.weights()
    scores = component_scores or {}
    covered = sum(weights[name] for name in scores if name in weights)
    if covered < rubric.total * MIN_SCORABLE_FRACTION:
        return OverallResult(
            None,
            f"not scored: only {covered:.0f} of {rubric.total:.0f} points of the "
            f"{rubric.rubric_id} rubric's weight had data "
            f"({rubric.total * MIN_SCORABLE_FRACTION:.0f} needed)",
        )
    weighted_sum = sum(weights[name] * scores[name] for name in scores if name in weights)
    return OverallResult(
        weighted_sum / covered,
        f"scored on {covered:.0f} of {rubric.total:.0f} points of the {rubric.rubric_id} rubric's weight",
    )


# ---------------------------------------------------------------------------
# Cohorts
# ---------------------------------------------------------------------------


def cohort_key(role: str | None) -> str | None:
    """The comparison-cohort id for a role: itself, unless it folds.

    frontend, frontend-mobile and mobile fold into one cohort id
    (ROSTER.md: three people, three keys, one team). Every other role,
    platform and backend included, maps to itself - nothing else folds
    automatically, and nothing here widens a cohort silently; that's
    :func:`peer_cohort`'s job to report, not this function's to do.
    """
    if role is None:
        return None
    return _FOLDED_COHORT_ID if role in _FOLD_ROLES else role


@dataclass(frozen=True)
class CohortResult:
    """The peer group for one person's role-relative comparisons.

    ``sufficient`` is False below :data:`MIN_PEERS`; the caller gets the real
    ``peer_count`` either way; and the requested person is never counted as
    their own peer, and the cohort is never widened past its documented fold
    without saying so in ``reason``.
    """

    person: str
    role: str | None
    cohort: str | None
    peers: tuple[str, ...]
    peer_count: int
    sufficient: bool
    reason: str


def peer_cohort(roster: Roster, person: str) -> CohortResult:
    who = roster.person(person)
    if who is None or who.role is None:
        return CohortResult(
            person=person,
            role=who.role if who else None,
            cohort=None,
            peers=(),
            peer_count=0,
            sufficient=False,
            reason="role unknown",
        )

    cohort = cohort_key(who.role)
    members = tuple(
        sorted(
            p.name
            for p in roster.people.values()
            if p.active and p.key != who.key and cohort_key(p.role) == cohort
        )
    )
    count = len(members)
    sufficient = count >= MIN_PEERS
    reason = (
        f"{count} active peer(s) in cohort '{cohort}'"
        if sufficient
        else f"insufficient peers: {count} found in cohort '{cohort}', need {MIN_PEERS}"
    )
    return CohortResult(
        person=person,
        role=who.role,
        cohort=cohort,
        peers=members,
        peer_count=count,
        sufficient=sufficient,
        reason=reason,
    )
