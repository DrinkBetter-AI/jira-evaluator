"""Which pull requests belong to one engineer's focused view.

The Individual scope shows a person only their own work, and PRs are the one
place that takes matching: GitHub knows authors by login, Jira by display name,
and nothing connects the two. Two routes, used together:

- ``GITHUB_LOGIN_MAP`` names the connection outright
  (``Tam=Phelan164, Mehdi Ordikhani=mordikh|mehdi-o``).
- A PR that names a Jira ticket assigned to the person is theirs to care
  about even when the author login is unmapped.
"""

from __future__ import annotations

import os

import pandas as pd

from capacity import same_person

_LOGIN_MAP_VAR = "GITHUB_LOGIN_MAP"


def parse_login_map(raw: str | None = None) -> dict[str, set[str]]:
    """Display-name -> github logins, from ``GITHUB_LOGIN_MAP``.

    Format: ``Name=login`` pairs separated by commas; one name may carry
    several logins separated by ``|``. Names and logins compare
    case-insensitively. A malformed pair costs that pair, not the map.
    """
    text = raw if raw is not None else os.getenv(_LOGIN_MAP_VAR, "")
    mapping: dict[str, set[str]] = {}
    for pair in text.split(","):
        name, _, logins = pair.partition("=")
        name = name.strip().lower()
        entries = {login.strip().lower() for login in logins.split("|") if login.strip()}
        if name and entries:
            mapping.setdefault(name, set()).update(entries)
    return mapping


def logins_for(person: str, login_map: dict[str, set[str]] | None = None) -> set[str]:
    """The GitHub logins mapped to ``person``, matched loosely on the name.

    A map key matches by name-token subset either way ("Tam" matches
    "Tam Nguyen", "Mehdi Ordikhani" matches "Mehdi Ordikhani Fard") - the
    same forgiving rule the team roster uses for Jira names.
    """
    mapping = parse_login_map() if login_map is None else login_map
    name = str(person or "").strip().lower()
    if not name:
        return set()
    found: set[str] = set()
    for key, logins in mapping.items():
        if same_person(key, name):
            found |= logins
    return found


def personal_prs(
    prs: pd.DataFrame,
    person: str,
    tickets: pd.DataFrame,
    login_map: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """The PRs that are ``person``'s: authored by them, or on their tickets.

    ``prs`` should already carry ``jira_key`` (see
    :func:`pr_hygiene.add_hygiene_fields`); without it only authorship counts.
    """
    if prs.empty:
        return prs
    logins = logins_for(person, login_map)
    author = prs.get("author", pd.Series(index=prs.index, dtype=object))
    authored = author.fillna("").astype(str).str.strip().str.lower().isin(logins)

    on_their_ticket = pd.Series(False, index=prs.index)
    if "jira_key" in prs.columns and not tickets.empty and {"key", "assignee"} <= set(
        tickets.columns
    ):
        owners = (
            tickets[["key", "assignee"]]
            .dropna(subset=["key"])
            .drop_duplicates(subset="key", keep="last")
        )
        owner_by_key = {
            str(key): str(owner or "").strip().lower()
            for key, owner in zip(owners["key"], owners["assignee"])
        }
        name = str(person or "").strip().lower()
        on_their_ticket = (
            prs["jira_key"].fillna("").astype(str).map(owner_by_key.get).fillna("")
            == name
        ) & prs["jira_key"].fillna("").astype(bool)

    return prs[authored | on_their_ticket]
