"""Team membership for tickets.

Jira knows nothing about "the Marketplace team". Two signals can stand in for it:
who the ticket is assigned to (``JIRA_TEAM_PEOPLE``) and which project it lives in
(``JIRA_TEAM_PROJECTS``). People win where both apply, because part-time engineers
here work across several projects while their team stays the same.

``JIRA_TEAM_PEOPLE`` is ``"Team=Name,Name;Team=Name"``. Names are matched loosely
against the Jira display name - a configured ``Farid`` matches ``Farid Shahidi`` -
so the roster does not have to mirror Jira's spelling.
"""

from __future__ import annotations

import pandas as pd


UNASSIGNED_TEAM = "Other"
NO_OWNER_TEAM = "Unassigned work"
# People who left: their open tickets are real work with nobody behind it.
FORMER_TEAM = "Former staff"

_NO_OWNER = {"", "unassigned", "none"}

# Teams that used to be tracked separately and are now one. Applied after both
# routing paths, so a ticket routed by project key lands in the same row as one
# routed by assignee - a raw ``CRM`` project key is the same team as Anouar.
TEAM_ALIASES = {
    "crm": "Marketplace",
    "leadership": "Business strategy",
    "business": "Business strategy",
}

# The VinoVoss roster as of this writing; overridden wholesale by JIRA_TEAM_PEOPLE.
DEFAULT_TEAM_PEOPLE = (
    # The CRM is the merchant side of the marketplace, and leadership sets the
    # business direction: two rows each would split one team's work in half.
    "Marketplace=Shawn,Shown,David,Mohsen,Gaston,Anouar,Jal;"
    "App=Ali,Farid;"
    "Design=Robert,Alesya;"
    "QA=Santi,Dina;"
    "ML=Tam,Mehdi,Jim;"
    "Business strategy=Zoe,Praveen,Igor,Jason,Kenesha,Whitney,Jennifer,Nancy,"
    "Matthew,Sylvia,Evmorfia,Angel,Arsalan,Mihai,Jeff;"
    # Full Jira display names, verified against the instance: a bare "Dan" would
    # file a future Dan Someone-Else's tickets under people who have left.
    f"{FORMER_TEAM}=Armine Aproyan,Saji,Sai Shankar,Saeid Parsa,Haichen Song,"
    "Yantao He,Dan O'Sullivan,Ramin Shahid,Shivanand"
)


def _parse_groups(spec: str) -> dict[str, str]:
    """``"Team=a,b;Team=c"`` -> ``{"a": "Team", "b": "Team", "c": "Team"}``."""
    mapping: dict[str, str] = {}
    for group in str(spec or "").split(";"):
        team, _, members = group.partition("=")
        team = team.strip()
        if not team:
            continue
        for member in members.split(","):
            member = member.strip()
            if member:
                mapping[member.lower()] = team
    return mapping


def parse_team_projects(spec: str) -> dict[str, str]:
    """Project key -> team name, from the ``Team=KEY,KEY;Team=KEY`` spec."""
    return {key.upper(): team for key, team in _parse_groups(spec).items()}


def parse_team_people(spec: str) -> dict[str, str]:
    """Person name -> team name, from the ``Team=Name,Name;Team=Name`` spec."""
    return _parse_groups(spec)


def _active_aliases(
    project_teams: dict[str, str], people_teams: dict[str, str]
) -> dict[str, str]:
    """The merges that still apply once the deployment has had its say.

    A team someone names in ``JIRA_TEAM_PEOPLE`` or ``JIRA_TEAM_PROJECTS`` is a
    deliberate answer to "who owns this", so a historical merge here must not
    quietly rename it. The alias survives only where the config is silent -
    notably the bare project key a ticket falls back to.
    """
    configured = {
        str(team).strip().lower()
        for team in (*project_teams.values(), *people_teams.values())
    }
    return {
        name: target
        for name, target in TEAM_ALIASES.items()
        if name not in configured
    }


def _team_for_person(display_name: str, people_teams: dict[str, str]) -> str | None:
    """Match a Jira display name against the roster, full name or first name."""
    name = display_name.strip().lower()
    if not name or name in _NO_OWNER:
        return None
    if name in people_teams:
        return people_teams[name]
    tokens = set(name.replace(".", " ").split())
    matched: set[str] = set()
    for person, team in people_teams.items():
        # Subset either way, so a roster "Mehdi Ordikhani" still finds Jira's
        # "Mehdi Ordikhani Fard" rather than falling through to the project.
        person_tokens = set(person.replace(".", " ").split())
        if person_tokens and (person_tokens <= tokens or tokens <= person_tokens):
            matched.add(team)
    # A bare roster first name can match two people on different teams; naming
    # one of them would be a coin toss dressed up as a fact, so the ticket falls
    # back to its project instead. Fix by spelling the name out in the roster.
    if len(matched) == 1:
        return matched.pop()
    return None


def add_team(
    df: pd.DataFrame,
    project_teams: dict[str, str],
    people_teams: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Attach a ``team`` column: assignee first, then project, then the raw key."""
    out = df.copy()
    people_teams = people_teams or {}

    owners = (
        out.get("assignee", pd.Series("", index=out.index)).fillna("").astype(str)
    )
    by_person = owners.map(lambda name: _team_for_person(name, people_teams))

    if "project_key" in out.columns:
        keys = out["project_key"].fillna("").astype(str).str.strip().str.upper()
        by_project = keys.map(project_teams) if project_teams else pd.Series(
            pd.NA, index=out.index
        )
        by_project = by_project.fillna(keys.where(keys.ne(""), UNASSIGNED_TEAM))
    else:
        by_project = pd.Series(UNASSIGNED_TEAM, index=out.index)

    # An unowned ticket has no team of its own; it belongs to whoever picks it up,
    # so it is called out rather than silently attributed to a project's team.
    no_owner = owners.str.strip().str.lower().isin(_NO_OWNER)
    team = by_person.fillna(by_project).mask(no_owner, NO_OWNER_TEAM).astype(str)
    aliases = _active_aliases(project_teams, people_teams)
    out["team"] = team.map(lambda name: aliases.get(name.strip().lower(), name))
    return out


def team_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-team open counts, idle pressure and estimate coverage."""
    columns = ["team", "open", "people", "avg_idle", "idle_30d", "unassigned", "no_estimate"]
    if df.empty or "team" not in df.columns:
        return pd.DataFrame(columns=columns)

    frame = df.copy()
    frame["_idle"] = pd.to_numeric(frame.get("idle_days"), errors="coerce").fillna(0.0)
    owners = frame.get("assignee", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["_unassigned"] = owners.str.strip().str.lower().isin(_NO_OWNER).astype(int)
    # "Unassigned" is a placeholder name, not a teammate, so it must not be counted.
    frame["_owner"] = owners.where(frame["_unassigned"].eq(0), pd.NA)
    frame["_idle30"] = (frame["_idle"] >= 30).astype(int)
    if "policy_violation" in frame.columns:
        frame["_no_estimate"] = frame["policy_violation"].fillna(False).astype(int)
    else:
        frame["_no_estimate"] = 0

    grouped = frame.groupby("team", dropna=False).agg(
        open=("key", "count"),
        people=("_owner", "nunique"),
        avg_idle=("_idle", "mean"),
        idle_30d=("_idle30", "sum"),
        unassigned=("_unassigned", "sum"),
        no_estimate=("_no_estimate", "sum"),
    )
    rollup = grouped.reset_index()
    rollup["avg_idle"] = rollup["avg_idle"].round(1)
    return rollup.sort_values("open", ascending=False).reset_index(drop=True)[columns]
