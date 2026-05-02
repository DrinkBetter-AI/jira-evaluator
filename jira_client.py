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

    def update_issue(self, key: str, fields: dict[str, Any]) -> None:
        """Update arbitrary fields on a single Jira issue."""
        url = f"{self.base_url}/rest/api/3/issue/{key}"
        with self._session() as session:
            session.headers["Content-Type"] = "application/json"
            response = session.put(url, json={"fields": fields}, timeout=30)
        if response.status_code not in {200, 204}:
            raise RuntimeError(
                f"Failed to update {key} ({response.status_code}): {response.text[:300]}"
            )

    def get_issue(self, key: str, fields: list[str] | None = None) -> dict[str, Any]:
        """Fetch a Jira issue payload for a key."""
        url = f"{self.base_url}/rest/api/3/issue/{key}"
        params: dict[str, str] = {}
        if fields:
            params["fields"] = ",".join(fields)

        with self._session() as session:
            response = session.get(url, params=params, timeout=30)

        if response.status_code >= 400:
            raise RuntimeError(
                f"Failed to fetch issue {key} ({response.status_code}): {response.text[:300]}"
            )
        return response.json() or {}

    def get_issue_snapshot(self, key: str) -> dict[str, Any]:
        """Return a compact live snapshot used for audit/revert."""
        issue = self.get_issue(key, fields=["summary", "status", "priority", "updated"])
        fields = issue.get("fields") or {}
        priority = fields.get("priority") or {}
        status = fields.get("status") or {}
        return {
            "key": key,
            "summary": fields.get("summary"),
            "status": status.get("name"),
            "priority": priority.get("name"),
            "priority_id": priority.get("id"),
            "updated": fields.get("updated"),
        }

    def set_priority(self, key: str, priority_name: str) -> None:
        self.update_issue(key, {"priority": {"name": priority_name}})

    def set_priority_by_id(self, key: str, priority_id: str) -> None:
        self.update_issue(key, {"priority": {"id": priority_id}})

    def bulk_update_priority(
        self,
        keys: list[str],
        priority_name: str,
    ) -> tuple[list[str], dict[str, str]]:
        """Set priority on each ticket in keys.
        Returns (succeeded_keys, {key: error_message}) for failed ones.
        """
        succeeded: list[str] = []
        failed: dict[str, str] = {}
        for key in keys:
            try:
                self.set_priority(key, priority_name)
                succeeded.append(key)
            except Exception as exc:  # noqa: BLE001
                failed[key] = str(exc)
        return succeeded, failed

    def get_issue_transitions(self, key: str) -> list[dict[str, str]]:
        """Return available transitions for a Jira issue key."""
        url = f"{self.base_url}/rest/api/3/issue/{key}/transitions"
        with self._session() as session:
            response = session.get(url, timeout=30)

        if response.status_code >= 400:
            raise RuntimeError(
                f"Failed to fetch transitions for {key} ({response.status_code}): {response.text[:300]}"
            )

        payload = response.json() or {}
        transitions = payload.get("transitions", [])
        return [
            {
                "id": str(t.get("id", "")),
                "name": str(t.get("name", "")),
                "to_status": str((t.get("to") or {}).get("name", "")),
            }
            for t in transitions
            if t.get("id")
        ]

    def transition_issue(self, key: str, transition_id: str) -> None:
        """Transition a Jira issue using a transition id."""
        url = f"{self.base_url}/rest/api/3/issue/{key}/transitions"
        with self._session() as session:
            session.headers["Content-Type"] = "application/json"
            response = session.post(
                url,
                json={"transition": {"id": transition_id}},
                timeout=30,
            )

        if response.status_code not in {200, 204}:
            raise RuntimeError(
                f"Failed to transition {key} ({response.status_code}): {response.text[:300]}"
            )

    def transition_issue_to_status(self, key: str, to_status_name: str) -> None:
        """Transition a Jira issue to a target status name when valid."""
        target = to_status_name.strip().lower()
        transitions = self.get_issue_transitions(key)
        matched = next(
            (t for t in transitions if t.get("to_status", "").strip().lower() == target),
            None,
        )
        if not matched:
            available = ", ".join(
                sorted({t.get("to_status", "") for t in transitions if t.get("to_status")})
            )
            raise RuntimeError(
                "No valid transition to target status. "
                + (f"Available: {available}" if available else "No transitions available")
            )
        self.transition_issue(key, matched["id"])

    def bulk_transition_status(
        self,
        keys: list[str],
        to_status_name: str,
    ) -> tuple[list[str], dict[str, str]]:
        """Transition each ticket to the given target status when possible.
        Returns (succeeded_keys, {key: error_message}) for failed ones.
        """
        target = to_status_name.strip().lower()
        succeeded: list[str] = []
        failed: dict[str, str] = {}

        for key in keys:
            try:
                self.transition_issue_to_status(key, target)
                succeeded.append(key)
            except Exception as exc:  # noqa: BLE001
                failed[key] = str(exc)

        return succeeded, failed

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
