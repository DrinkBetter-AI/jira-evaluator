from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd
import requests
import yaml

from write_access import require_writes_enabled


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
    "timetracking",
    "customfield_10020",
    "parent",
    "project",
]

_ML_SPRINT_NAME_RE = re.compile(r"^ML\s+Sprint\s+\d+$", re.IGNORECASE)

DEFAULT_CREDS_PATH = "~/.creds/vinovoss.yml"
DEFAULT_PROFILE_NAME = "ML-TEAM-MANAGEMENT"


def _is_ml_sprint_name(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(_ML_SPRINT_NAME_RE.match(text))


def _is_ignored_sprint_rollover_item(item: dict[str, Any]) -> bool:
    """Return True when a changelog item is only ML sprint rollover (X -> X+1 style)."""
    field_name = str(item.get("field") or "").strip().lower()
    if field_name not in {"sprint", "customfield_10020"}:
        return False

    from_value = item.get("fromString")
    to_value = item.get("toString")
    return _is_ml_sprint_name(from_value) and _is_ml_sprint_name(to_value)


def _extract_last_meaningful_activity(issue: dict[str, Any]) -> Any:
    """Pick the latest changelog timestamp that reflects meaningful activity.

    Ignored activity: sprint rollover between ML sprints (both from/to are ML Sprint names).
    Kept activity: all other changes, including None -> ML Sprint assignment.
    """
    changelog = issue.get("changelog") or {}
    histories = changelog.get("histories") or []
    if not isinstance(histories, list):
        return None

    meaningful_timestamps: list[Any] = []
    for history in histories:
        if not isinstance(history, dict):
            continue
        items = history.get("items") or []
        if not isinstance(items, list) or not items:
            continue

        has_meaningful_item = any(
            isinstance(item, dict) and not _is_ignored_sprint_rollover_item(item)
            for item in items
        )
        if not has_meaningful_item:
            continue

        created = history.get("created")
        if created:
            meaningful_timestamps.append(created)

    if not meaningful_timestamps:
        return None
    return max(meaningful_timestamps)


class JiraConfigError(ValueError):
    """Raised when Jira config is missing or invalid."""


def _coerce_sprint_id(sprint_id: int | str) -> str:
    """Normalize sprint IDs so Agile API paths always use integer-like IDs."""
    text = str(sprint_id).strip()
    if not text:
        raise ValueError("Sprint ID is missing.")

    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]

    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"Invalid sprint ID '{sprint_id}'. Expected a numeric ID.")

    return text


def _first_non_empty(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return cleaned
    return f"https://{cleaned}"


def load_jira_env() -> dict[str, str] | None:
    """Read Jira credentials from the environment, or return None when unset.

    Requires JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN to all be present.
    """
    base_url = os.getenv("JIRA_BASE_URL", "").strip()
    email = os.getenv("JIRA_EMAIL", "").strip()
    api_token = os.getenv("JIRA_API_TOKEN", "").strip()
    if not (base_url and email and api_token):
        return None
    return {
        "base_url": normalize_base_url(base_url),
        "email": email,
        "api_token": api_token,
    }


def load_jira_profile(
    creds_path: str | Path = DEFAULT_CREDS_PATH,
    profile_name: str = DEFAULT_PROFILE_NAME,
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
        "base_url": normalize_base_url(str(base_url)),
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
        creds_path: str | Path = DEFAULT_CREDS_PATH,
        profile_name: str = DEFAULT_PROFILE_NAME,
    ) -> "JiraClient":
        """Build a client from the named YAML profile, as asked for."""
        return cls(**load_jira_profile(creds_path=creds_path, profile_name=profile_name))

    @classmethod
    def resolve(
        cls,
        creds_path: str | Path = DEFAULT_CREDS_PATH,
        profile_name: str = DEFAULT_PROFILE_NAME,
    ) -> "JiraClient":
        """Environment credentials if the deployment supplies them, else the profile.

        Kept apart from ``from_yaml`` so a caller naming a specific profile gets
        that profile: on Cloud Run there is no credentials file and the env vars
        are the only source, while locally the profile still wins by absence.
        """
        cfg = load_jira_env()
        if cfg is None:
            cfg = load_jira_profile(creds_path=creds_path, profile_name=profile_name)
        return cls(**cfg)

    def _session(self) -> requests.Session:
        session = requests.Session()
        session.auth = (self.email, self.api_token)
        session.headers.update({"Accept": "application/json"})
        return session

    def approximate_count(self, jql: str) -> int:
        """Jira's fast issue count for a JQL, independent of paging.

        Uses ``/search/approximate-count`` (documented as approximate for large
        result sets) so the number is not silently capped by ``max_results``. On
        a tenant that doesn't expose that endpoint (404/405/410) it falls back to
        the ``total`` of a ``maxResults=0`` search, mirroring ``search_issues``'
        legacy fallback, so the resolved tiles still show a number rather than a
        permanent "—".
        """
        url = f"{self.base_url}/rest/api/3/search/approximate-count"
        with self._session() as session:
            response = session.post(url, json={"jql": jql}, timeout=30)
            if response.status_code in {404, 405, 410}:
                return self._legacy_total(session, jql)
            response.raise_for_status()
            return int(response.json().get("count", 0))

    def _legacy_total(self, session: requests.Session, jql: str) -> int:
        """Total matches via the legacy search's ``total`` field (maxResults=0)."""
        response = session.get(
            f"{self.base_url}/rest/api/3/search",
            params={"jql": jql, "maxResults": 0},
            timeout=30,
        )
        response.raise_for_status()
        return int(response.json().get("total", 0))

    def search_issues(
        self,
        jql: str,
        fields: list[str] | None = None,
        max_results: int = 1000,
        page_size: int = 100,
        expand: str | None = None,
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
                    if expand:
                        params["expand"] = expand
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
                    if expand:
                        params["expand"] = expand
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
        require_writes_enabled()
        url = f"{self.base_url}/rest/api/3/issue/{key}"
        with self._session() as session:
            session.headers["Content-Type"] = "application/json"
            response = session.put(url, json={"fields": fields}, timeout=30)
        if response.status_code not in {200, 204}:
            raise RuntimeError(
                f"Failed to update {key} ({response.status_code}): {response.text[:300]}"
            )

    def add_issues_to_sprint(self, sprint_id: int | str, issue_keys: list[str]) -> None:
        """Add issues to a Jira sprint via the Agile API."""
        if not issue_keys:
            return
        require_writes_enabled()

        normalized_sprint_id = _coerce_sprint_id(sprint_id)
        url = f"{self.base_url}/rest/agile/1.0/sprint/{normalized_sprint_id}/issue"
        payload = {"issues": issue_keys}
        with self._session() as session:
            session.headers["Content-Type"] = "application/json"
            response = session.post(url, json=payload, timeout=30)

        if response.status_code not in {200, 201, 204}:
            raise RuntimeError(
                f"Failed to add issues to sprint {normalized_sprint_id} ({response.status_code}): {response.text[:300]}"
            )

    def move_issues_to_backlog(self, issue_keys: list[str]) -> None:
        """Move issues out of their current non-closed sprint and back to backlog."""
        if not issue_keys:
            return
        require_writes_enabled()

        url = f"{self.base_url}/rest/agile/1.0/backlog/issue"
        payload = {"issues": issue_keys}
        with self._session() as session:
            session.headers["Content-Type"] = "application/json"
            response = session.post(url, json=payload, timeout=30)

        if response.status_code not in {200, 201, 204}:
            raise RuntimeError(
                f"Failed to move issues to backlog ({response.status_code}): {response.text[:300]}"
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
        require_writes_enabled()
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

    def get_all_statuses(self) -> list[str]:
        """Return all status names configured in this Jira instance."""
        url = f"{self.base_url}/rest/api/3/status"
        with self._session() as session:
            response = session.get(url, timeout=30)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Failed to fetch statuses ({response.status_code}): {response.text[:300]}"
            )
        return sorted({s["name"] for s in response.json() if s.get("name")})

    def get_all_priorities(self) -> list[str]:
        """Return all priority names configured in this Jira instance."""
        url = f"{self.base_url}/rest/api/3/priority/search"
        with self._session() as session:
            response = session.get(url, timeout=30)
        if response.status_code >= 400:
            url = f"{self.base_url}/rest/api/3/priority"
            with self._session() as session2:
                response = session2.get(url, timeout=30)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Failed to fetch priorities ({response.status_code}): {response.text[:300]}"
            )
        payload = response.json()
        items = payload.get("values", payload) if isinstance(payload, dict) else payload
        return [p["name"] for p in items if p.get("name")]

    def get_all_users(self, page_size: int = 1000) -> list[dict[str, str]]:
        """Return Jira users with display names and account ids."""
        url = f"{self.base_url}/rest/api/3/users/search"
        start_at = 0
        users: list[dict[str, str]] = []

        with self._session() as session:
            while True:
                response = session.get(
                    url,
                    params={
                        "query": "",
                        "startAt": start_at,
                        "maxResults": page_size,
                    },
                    timeout=30,
                )
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"Failed to fetch users ({response.status_code}): {response.text[:300]}"
                    )

                batch = response.json() or []
                if not isinstance(batch, list) or not batch:
                    break

                for user in batch:
                    account_id = str(user.get("accountId") or "").strip()
                    display_name = str(user.get("displayName") or "").strip()
                    if account_id and display_name:
                        users.append(
                            {
                                "account_id": account_id,
                                "display_name": display_name,
                            }
                        )

                if len(batch) < page_size:
                    break
                start_at += page_size

        dedup: dict[tuple[str, str], dict[str, str]] = {}
        for user in users:
            key = (user["account_id"], user["display_name"])
            dedup[key] = user
        return list(dedup.values())

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
            timetracking = fields.get("timetracking") or {}
            parent = fields.get("parent") or {}
            parent_fields = parent.get("fields") or {}
            parent_status = parent_fields.get("status") or {}
            # ``parent`` is the epic for stories and tasks but the containing
            # story for a sub-task, so only an Epic parent is an epic link.
            parent_type = (parent_fields.get("issuetype") or {}).get("name") or ""
            is_epic_parent = parent_type.strip().lower() == "epic"
            project = fields.get("project") or {}
            time_spent_sec = timetracking.get("timeSpentSeconds") or 0
            orig_est_sec = timetracking.get("originalEstimateSeconds") or 0
            completion_pct = round(time_spent_sec / orig_est_sec * 100, 1) if orig_est_sec > 0 else None

            # Parse sprint info from customfield_10020 (array of sprint objects).
            sprints_raw = fields.get("customfield_10020") or []
            future_sprints: list[dict[str, Any]] = []
            active_sprints: list[dict[str, Any]] = []
            closed_sprints: list[dict[str, Any]] = []
            for sp in sprints_raw:
                if not isinstance(sp, dict):
                    continue
                state = (sp.get("state") or "").lower()
                if state == "future":
                    future_sprints.append(sp)
                elif state == "active":
                    active_sprints.append(sp)
                elif state == "closed":
                    closed_sprints.append(sp)

            chosen_sprint = (future_sprints or active_sprints or closed_sprints or [None])[-1]
            sprint_name = chosen_sprint.get("name") if chosen_sprint else None
            sprint_state = chosen_sprint.get("state") if chosen_sprint else None
            sprint_id = chosen_sprint.get("id") if chosen_sprint else None
            sprint_board_id = chosen_sprint.get("boardId") if chosen_sprint else None
            # Sprint window drives per-person available hours in the capacity view.
            sprint_start = chosen_sprint.get("startDate") if chosen_sprint else None
            sprint_end = chosen_sprint.get("endDate") if chosen_sprint else None

            # Closed sprints this still-open ticket has already passed through. Not
            # conditioned on being in a sprint now: a ticket carried through five
            # sprints and then dropped out of planning is the most abandoned case
            # there is, and scoring it zero would hide exactly that.
            carry_over_count = len(closed_sprints)
            last_meaningful_activity = _extract_last_meaningful_activity(issue)

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
                    "last_meaningful_activity": last_meaningful_activity,
                    "due_date": fields.get("duedate"),
                    "issue_type": issue_type.get("name"),
                    "project_key": project.get("key"),
                    "project_name": project.get("name"),
                    "parent_key": parent.get("key"),
                    "parent_type": parent_type or None,
                    "epic_key": parent.get("key") if is_epic_parent else None,
                    "epic_summary": parent_fields.get("summary") if is_epic_parent else None,
                    "epic_status": parent_status.get("name") if is_epic_parent else None,
                    "labels": ", ".join(fields.get("labels", [])),
                    "resolution": resolution.get("name"),
                    "status_category_changed_date": fields.get("statuscategorychangedate"),
                    "original_estimate": timetracking.get("originalEstimate"),
                    "logged_time": timetracking.get("timeSpent"),
                    "completion_pct": completion_pct,
                    "original_estimate_sec": orig_est_sec,
                    "time_spent_sec": time_spent_sec,
                    "sprint_id": sprint_id,
                    "sprint_name": sprint_name,
                    "sprint_state": sprint_state,
                    "sprint_board_id": sprint_board_id,
                    "sprint_start": sprint_start,
                    "sprint_end": sprint_end,
                    "carry_over_count": carry_over_count,
                    "ticket_url": f"{self.base_url}/browse/{issue.get('key')}",
                }
            )

        return pd.DataFrame(rows)
