import os
from pathlib import Path

from jira_client import DEFAULT_CREDS_PATH, DEFAULT_PROFILE_NAME, JiraClient


EPIC_KEY = "MB-4102"
EPIC_KEY_2 = "MB-4290"
CSV_FILENAME = Path(f"jira_epic_tickets_extended_{EPIC_KEY}.csv")


def main() -> None:
    jql = f'"parent" in ({EPIC_KEY}, {EPIC_KEY_2}) ORDER BY created ASC'

    # Same resolution as the dashboard, so the script also runs in the container,
    # where the credentials arrive as environment variables and no YAML exists.
    client = JiraClient.resolve(
        creds_path=os.getenv("JIRA_CREDS_PATH", DEFAULT_CREDS_PATH),
        profile_name=os.getenv("JIRA_PROFILE", DEFAULT_PROFILE_NAME),
    )

    df = client.search_issues(
        jql=jql,
        fields=[
            "summary",
            "issuetype",
            "assignee",
            "priority",
            "status",
            "created",
            "updated",
            "duedate",
            "reporter",
            "labels",
            "resolution",
            "statuscategorychangedate",
        ],
        max_results=1000,
        page_size=100,
    )

    if df.empty:
        print("No Jira issues returned for the selected JQL.")
        return

    export = df.rename(
        columns={
            "key": "Key",
            "summary": "Summary",
            "issue_type": "Type",
            "assignee": "Assignee",
            "priority": "Priority",
            "status": "Status",
            "due_date": "Due Date",
            "created": "Created",
            "updated": "Updated",
            "reporter": "Reporter",
            "labels": "Labels",
            "resolution": "Resolution",
        }
    )

    export.to_csv(CSV_FILENAME, index=False)
    print(f"Exported {len(export)} issues to {CSV_FILENAME}")


if __name__ == "__main__":
    main()