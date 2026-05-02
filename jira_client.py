from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
import yaml


DEFAULT_FIELDS = [
    "summary",
    "status",
    "priority",
    "assignee",
    "reporter",
    "created",
    "updated",
    "duedate",
    "issuetype",
    "labels",
    "resolution",
    "statuscategorychangedate",
]


class JiraConfigError(ValueError):
    """Raised when Jira config is missing or invalid."""


def _first_non_empty(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalize_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return cleaned
    return f"https://{cleaned}"


def load_jira_profile(
    creds_path: str | Path = "~/.creds/vinovoss.yml",
    profile_name: str = "ML-TEAM-MANAGEMENT",
) -> dict[str, str]:
    path = Path(creds_path).expanduser()
    if not path.exists():
        raise JiraConfigError(f"Credentials file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    jira_section = data.get("Jira")
    if not isinstance(jira_section, dict):
        raise JiraConfigError("Missing 'Jira' section in credentials file.")

    profile = jira_section.get(profile_name)
    if not isinstance(profile, dict):
        available = ", ".join(jira_section.keys()) or "none"
        raise JiraConfigError(
            f"Jira profile '{profile_name}' not found. Available profiles: {available}"
        )

    base_url = _first_non_empty(profile, ["base_url", "url", "domain", "host"])
    email = _first_non_empty(profile, ["email", "username", "user"])
    api_token = _first_non_empty(profile, ["api_token", "token", "password", "apiKey"])

    if not base_url or not email or not api_token:
        raise JiraConfigError(
            "Jira profile is missing one of required values: base URL, email, API token."
        )

    return {
        "base_url": _normalize_base_url(str(base_url)),
        "email": str(email),
        "api_token": str(api_token),
    }


@dataclass
class JiraClient:
    base_url: str
    email: str
    api_token: str

    @classmethod
    def from_yaml(
        cls,
        creds_path: str | Path = "~/.creds/vinovoss.yml",
        profile_name: str = "ML-TEAM-MANAGEMENT",
    ) -> "JiraClient":
        cfg = load_jira_profile(creds_path=creds_path, profile_name=profile_name)
        return cls(**cfg)

    def _session(self) -> requests.Session:
        session = requests.Session()
        session.auth = (self.email, self.api_token)
        session.headers.update({"Accept": "application/json"})
        return session

    def search_issues(
        self,
        jql: str,
        fields: list[str] | None = None,
        max_results: int = 1000,
        page_size: int = 100,
    ) -> pd.DataFrame:
        fields = fields or DEFAULT_FIELDS
        issues: list[dict[str, Any]] = []
        seen_issue_keys: set[str] = set()

        search_url = f"{self.base_url}/rest/api/3/search/jql"
        legacy_search_url = f"{self.base_url}/rest/api/3/search"

        with self._session() as session:
            use_new_api = True
            next_page_token: str | None = None
            start_at = 0
            total = None
            seen_page_signatures: set[tuple[Any, ...]] = set()

            while True:
                remaining = max_results - len(issues)
                if remaining <= 0:
                    break

                batch_size = min(page_size, remaining)
                if use_new_api:
                    params = {
                        "jql": jql,
                        "fields": ",".join(fields),
                        "maxResults": batch_size,
                    }
                    if next_page_token:
                        params["nextPageToken"] = next_page_token
                    response = session.get(search_url, params=params, timeout=30)

                    # If the tenant doesn't support the new endpoint shape, fallback.
                    if response.status_code in {404, 405, 410}:
                        use_new_api = False
                        continue
                else:
                    params = {
                        "jql": jql,
                        "fields": ",".join(fields),
                        "startAt": start_at,
                        "maxResults": batch_size,
                    }
                    response = session.get(legacy_search_url, params=params, timeout=30)

                if response.status_code >= 400:
                    details = response.text[:500]
                    raise RuntimeError(
                        f"Jira search failed ({response.status_code}): {details}"
                    )

                payload = response.json()
                batch = payload.get("issues", [])

                if not batch:
                    break

                for issue in batch:
                    issue_key = issue.get("key")
                    if issue_key and issue_key in seen_issue_keys:
                        continue
                    if issue_key:
                        seen_issue_keys.add(issue_key)
                    issues.append(issue)

                if use_new_api:
                    signature = (
                        batch[0].get("key"),
                        batch[-1].get("key"),
                        len(batch),
                        payload.get("nextPageToken"),
                    )
                    if signature in seen_page_signatures:
                        break
                    seen_page_signatures.add(signature)

                    next_page_token = payload.get("nextPageToken")
                    if not next_page_token:
                        break
                else:
                    total = payload.get("total", total)
                    start_at += len(batch)

                    if total is not None and start_at >= total:
                        break
                    if total is None and len(batch) < batch_size:
                        break

        return self._issues_to_dataframe(issues)

    def _issues_to_dataframe(self, issues: list[dict[str, Any]]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for issue in issues:
            fields = issue.get("fields") or {}
            status = fields.get("status") or {}
            status_category = status.get("statusCategory") or {}

            assignee = fields.get("assignee") or {}
            reporter = fields.get("reporter") or {}
            priority = fields.get("priority") or {}
            issue_type = fields.get("issuetype") or {}
            resolution = fields.get("resolution") or {}

            rows.append(
                {
                    "key": issue.get("key"),
                    "summary": fields.get("summary"),
                    "status": status.get("name"),
                    "status_category": status_category.get("name"),
                    "priority": priority.get("name"),
                    "assignee": assignee.get("displayName") or assignee.get("accountId") or "Unassigned",
                    "assignee_account_id": assignee.get("accountId"),
                    "reporter": reporter.get("displayName") or reporter.get("accountId"),
                    "created": fields.get("created"),
                    "updated": fields.get("updated"),
                    "due_date": fields.get("duedate"),
                    "issue_type": issue_type.get("name"),
                    "labels": ", ".join(fields.get("labels", [])),
                    "resolution": resolution.get("name"),
                    "status_category_changed_date": fields.get("statuscategorychangedate"),
                    "ticket_url": f"{self.base_url}/browse/{issue.get('key')}",
                }
            )

        return pd.DataFrame(rows)
