from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from change_audit import (
    append_operation,
    finalize_operation,
    load_operations,
    new_operation_record,
    summarize_operations,
)
from jira_client import DEFAULT_FIELDS, JiraClient, JiraConfigError
from transformations import add_ticket_health_fields


DEFAULT_JQL = """assignee = 712020:24fedc7d-c0b0-46c0-99a1-1b5b29efdc47
AND statusCategory != Done
ORDER BY updated ASC"""

CORE_TEAM_MEMBERS = [
    "Tam",
    "Mehdi Ordikhani",
    "Shivanand",
]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_tickets(
    creds_path: str,
    profile_name: str,
    jql: str,
    max_results: int,
    page_size: int,
) -> pd.DataFrame:
    client = JiraClient.from_yaml(creds_path=creds_path, profile_name=profile_name)
    return client.search_issues(
        jql=jql,
        fields=DEFAULT_FIELDS,
        max_results=max_results,
        page_size=page_size,
    )


def _render_metrics(df: pd.DataFrame) -> None:
    total_open = int(len(df))
    avg_idle = float(df["idle_days"].mean()) if total_open else 0.0
    max_idle = float(df["idle_days"].max()) if total_open else 0.0
    oldest = float(df["ticket_age_days"].max()) if total_open else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Open Tickets", total_open)
    m2.metric("Average Idle Days", f"{avg_idle:.1f}")
    m3.metric("Max Idle Days", f"{max_idle:.1f}")
    m4.metric("Oldest Ticket Age", f"{oldest:.1f}")


def _render_bubble_chart(df: pd.DataFrame, color_by: str = "priority") -> None:
    if df.empty:
        st.info("No data available for staleness bubble chart.")
        return

    plot_df = df.copy()

    # Assign a numeric Y position per status so we can add jitter around it.
    # Fixed order bottom (0) → top (n).
    STATUS_ORDER = [
        "Backlog",
        "DISCUSSION NEEDED",
        "To Do",
        "In Progress",
        "IN DEV ENV",
        "Review in Staging",
        "Code Review",
        "Ready for Production",
    ]
    statuses = plot_df["status"].fillna("Unknown")
    # Any status not in the fixed list gets appended at the top.
    extra = [s for s in statuses.unique() if s not in STATUS_ORDER]
    full_order = STATUS_ORDER + extra
    status_to_y = {s: i for i, s in enumerate(full_order)}
    rng = np.random.default_rng(seed=42)
    plot_df["y_jitter"] = (
        statuses.map(status_to_y).astype(float)
        + rng.uniform(-0.35, 0.35, size=len(plot_df))
    )
    plot_df["status_label"] = statuses

    # Normalise bubble size so smallest is still visible.
    age = plot_df["ticket_age_days"].clip(lower=1)
    plot_df["bubble_size"] = ((age - age.min()) / (age.max() - age.min() + 1e-9) * 28 + 6).round(1)

    fig = px.scatter(
        plot_df,
        x="idle_days",
        y="y_jitter",
        size="bubble_size",
        color=color_by,
        custom_data=["key", "summary", "assignee", "status_label", "priority", "ticket_age_days", "idle_days"],
        title="Staleness vs Workflow Status",
        labels={"idle_days": "Idle Days", "y_jitter": "Status"},
        size_max=34,
        opacity=0.3,
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br>"
            "Assignee: %{customdata[2]}<br>"
            "Status: %{customdata[3]}<br>"
            "Priority: %{customdata[4]}<br>"
            "Age: %{customdata[5]:.1f} days<br>"
            "Idle: %{customdata[6]:.1f} days"
            "<extra></extra>"
        )
    )

    # Replace numeric Y ticks with status labels.
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(status_to_y.values()),
        ticktext=list(status_to_y.keys()),
        title="Status",
    )
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

def _apply_action_with_audit(
    client: JiraClient,
    action_type: str,
    selected_keys: list[str],
    target: str,
    source_status: str | None = None,
    parent_operation_id: str | None = None,
) -> tuple[list[str], dict[str, str], dict[str, object]]:
    operation = new_operation_record(
        action_type=action_type,
        target=target,
        selected_keys=selected_keys,
        source_status=source_status,
        parent_operation_id=parent_operation_id,
    )

    items: list[dict[str, object]] = []
    succeeded: list[str] = []
    failed: dict[str, str] = {}

    for key in selected_keys:
        before_snapshot: dict[str, object] = {}
        try:
            before_snapshot = client.get_issue_snapshot(key)

            if action_type == "priority":
                client.set_priority(key, target)
            elif action_type == "status":
                client.transition_issue_to_status(key, target)
            elif action_type == "revert_priority":
                prior_id = str(target).strip()
                if prior_id:
                    client.set_priority_by_id(key, prior_id)
                else:
                    raise RuntimeError("Cannot revert: original priority id is missing.")
            elif action_type == "revert_status":
                client.transition_issue_to_status(key, target)
            else:
                raise RuntimeError(f"Unsupported action type: {action_type}")

            after_snapshot = client.get_issue_snapshot(key)
            items.append(
                {
                    "key": key,
                    "success": True,
                    "before": before_snapshot,
                    "after": after_snapshot,
                }
            )
            succeeded.append(key)
        except Exception as exc:  # noqa: BLE001
            failed[key] = str(exc)
            items.append(
                {
                    "key": key,
                    "success": False,
                    "before": before_snapshot,
                    "error": str(exc),
                }
            )

    operation = finalize_operation(operation, items)
    append_operation(operation)
    return succeeded, failed, operation


def main() -> None:
    st.set_page_config(page_title="Jira Ticket Health Dashboard", layout="wide")
    st.title("Jira Ticket Health Dashboard")
    st.caption("Visual monitoring for stale, idle, and high-risk tickets.")

    with st.container(border=True):
        st.subheader("Data Query")
        q_col1, q_col2 = st.columns([3, 1])
        with q_col1:
            jql = st.text_area("JQL", value=DEFAULT_JQL, height=110)
            selected_core_team = st.multiselect(
                "Team members (check/uncheck)",
                options=CORE_TEAM_MEMBERS,
                default=CORE_TEAM_MEMBERS,
            )
        with q_col2:
            max_results = st.number_input("Max tickets", min_value=1, max_value=5000, value=1000, step=100)
            page_size = st.number_input("Page size", min_value=10, max_value=200, value=100, step=10)
            refresh_clicked = st.button("Refresh Data", use_container_width=True)

    if refresh_clicked:
        st.cache_data.clear()

    try:
        raw_df = fetch_tickets(
            creds_path="~/.creds/vinovoss.yml",
            profile_name="ML-TEAM-MANAGEMENT",
            jql=jql,
            max_results=int(max_results),
            page_size=int(page_size),
        )
    except JiraConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to fetch Jira issues: {exc}")
        st.stop()

    if raw_df.empty:
        st.warning("No tickets returned for the current JQL.")
        st.stop()

    df = add_ticket_health_fields(raw_df)

    st.subheader("Filters")
    f1, f2, f3, f4, f5 = st.columns(5)
    assignees = sorted(df["assignee"].dropna().unique().tolist())
    statuses = sorted(df["status"].dropna().unique().tolist())
    priorities = sorted(df["priority"].dropna().unique().tolist())

    selected_assignees = f1.multiselect("Assignee", options=assignees, default=[])
    selected_statuses = f2.multiselect("Status", options=statuses, default=[])
    selected_priorities = f3.multiselect("Priority", options=priorities, default=[])
    min_idle = f4.slider("Min idle days", min_value=0, max_value=180, value=0)
    min_age = f5.slider("Min ticket age", min_value=0, max_value=365, value=0)

    color_by = st.radio("Bubble color", options=["priority", "assignee"], horizontal=True)

    filtered = df.copy()
    if selected_core_team:
        selected_core_team_lower = {name.strip().lower() for name in selected_core_team}
        filtered = filtered[
            filtered["assignee"].fillna("").astype(str).str.strip().str.lower().isin(selected_core_team_lower)
        ]
    else:
        filtered = filtered.iloc[0:0]

    if selected_assignees:
        filtered = filtered[filtered["assignee"].isin(selected_assignees)]
    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]
    if selected_priorities:
        filtered = filtered[filtered["priority"].isin(selected_priorities)]

    filtered = filtered[(filtered["idle_days"] >= min_idle) & (filtered["ticket_age_days"] >= min_age)]

    _render_metrics(filtered)

    _render_bubble_chart(filtered, color_by=color_by)

    st.subheader("Raw Tickets")
    raw_columns = [
        "key",
        "summary",
        "status",
        "status_category",
        "priority",
        "assignee",
        "reporter",
        "ticket_age_days",
        "idle_days",
        "created",
        "updated"
    ]
    raw_display_df = filtered.sort_values(["idle_days", "ticket_age_days"], ascending=[False, False])[raw_columns]
    raw_display_df = raw_display_df.rename(columns={"created": "Created at", "updated": "Updated at"})
    st.dataframe(raw_display_df, use_container_width=True)

    st.divider()
    st.subheader("Suggested First Action")

    PRIORITY_OPTIONS = ["Highest", "High", "Normal", "Low", "Lowest"]
    action_type = st.selectbox(
        "Action",
        options=["Set None-priority tickets", "Change status"],
        index=0,
        help="Default action keeps the first cleanup flow: None priority -> Normal.",
    )

    status_options = sorted(filtered["status"].dropna().astype(str).unique().tolist())
    normalized_priority = filtered["priority"].fillna("").astype(str).str.strip().str.lower()
    none_priority_keys = sorted(filtered[normalized_priority.isin(["", "none"])]["key"].tolist())

    with st.container(border=True):
        if action_type == "Set None-priority tickets":
            st.markdown("**Detected tickets without priority**")
            st.caption(
                f"{len(none_priority_keys)} ticket(s) in the current view have no priority set."
            )
            if none_priority_keys:
                preview = ", ".join(none_priority_keys[:15])
                suffix = " ..." if len(none_priority_keys) > 15 else ""
                st.caption(f"Sample: {preview}{suffix}")

            selected_keys = st.multiselect(
                "Tickets to update",
                options=none_priority_keys,
                default=none_priority_keys,
                help="Remove any tickets you do not want to update.",
            )

            target_priority = st.selectbox(
                "Suggested priority",
                options=PRIORITY_OPTIONS,
                index=2,
                help="Normal is selected by default as the first cleanup action.",
            )
            target_label = f"priority '{target_priority}'"
        else:
            st.markdown("**Change ticket status**")
            if not status_options:
                st.info("No statuses available in the current filtered view.")
                source_status = None
                target_status = None
                selected_keys = []
            else:
                source_status = st.selectbox("From status", options=status_options, index=0)
                to_options = [s for s in status_options if s != source_status] or status_options
                target_status = st.selectbox("To status", options=to_options, index=0)

                source_keys = sorted(filtered[filtered["status"] == source_status]["key"].tolist())
                selected_keys = st.multiselect(
                    "Tickets to update",
                    options=source_keys,
                    default=source_keys,
                    help="Only tickets currently in the selected source status are listed.",
                )
                target_label = f"status '{source_status}' -> '{target_status}'"

        apply_suggestion = st.button(
            f"Apply to {len(selected_keys)} ticket(s)",
            disabled=not selected_keys,
            type="primary",
        )

    if apply_suggestion and selected_keys:
        client = JiraClient.from_yaml(
            creds_path="~/.creds/vinovoss.yml",
            profile_name="ML-TEAM-MANAGEMENT",
        )
        with st.spinner(f"Updating {len(selected_keys)} tickets..."):
            if action_type == "Set None-priority tickets":
                succeeded, failed, operation = _apply_action_with_audit(
                    client=client,
                    action_type="priority",
                    selected_keys=selected_keys,
                    target=target_priority,
                )
            else:
                succeeded, failed, operation = _apply_action_with_audit(
                    client=client,
                    action_type="status",
                    selected_keys=selected_keys,
                    target=target_status,
                    source_status=source_status,
                )

        if succeeded:
            st.success(
                f"Updated {len(succeeded)} ticket(s) to {target_label}. Operation ID: {operation['operation_id']}"
            )
        if failed:
            for key, err in failed.items():
                st.error(f"{key}: {err}")

        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("Change History and Revert")
    operations = load_operations(limit=30)
    if not operations:
        st.info("No write operations have been logged yet.")
    else:
        st.dataframe(pd.DataFrame(summarize_operations(operations)), use_container_width=True)

        op_options = {
            (
                f"{op.get('created_at', '')} | {op.get('action_type', '')} | "
                f"{op.get('target', '')} | success={op.get('success_count', 0)} | "
                f"id={str(op.get('operation_id', ''))[:8]}"
            ): op
            for op in operations
            if op.get("success_count", 0) > 0
        }

        if not op_options:
            st.caption("No successful operation available for revert.")
        else:
            selected_label = st.selectbox(
                "Select operation to revert",
                options=list(op_options.keys()),
            )
            selected_operation = op_options[selected_label]
            confirm_revert = st.checkbox("I understand revert may partially fail due to Jira workflow rules.")

            revert_clicked = st.button("Revert selected operation", disabled=not confirm_revert)

            if revert_clicked:
                client = JiraClient.from_yaml(
                    creds_path="~/.creds/vinovoss.yml",
                    profile_name="ML-TEAM-MANAGEMENT",
                )

                revert_succeeded: list[str] = []
                revert_failed: dict[str, str] = {}
                parent_id = selected_operation.get("operation_id")
                successful_items = [it for it in selected_operation.get("items", []) if it.get("success")]

                with st.spinner(f"Reverting {len(successful_items)} ticket(s)..."):
                    for item in successful_items:
                        key = str(item.get("key", ""))
                        before = item.get("before") or {}
                        try:
                            if selected_operation.get("action_type") == "priority":
                                original_priority_id = before.get("priority_id")
                                if not original_priority_id:
                                    raise RuntimeError("Original priority id missing in audit record.")

                                rev_succeeded, rev_failed, rev_op = _apply_action_with_audit(
                                    client=client,
                                    action_type="revert_priority",
                                    selected_keys=[key],
                                    target=str(original_priority_id),
                                    parent_operation_id=str(parent_id),
                                )
                            elif selected_operation.get("action_type") == "status":
                                original_status = before.get("status")
                                if not original_status:
                                    raise RuntimeError("Original status missing in audit record.")

                                rev_succeeded, rev_failed, rev_op = _apply_action_with_audit(
                                    client=client,
                                    action_type="revert_status",
                                    selected_keys=[key],
                                    target=str(original_status),
                                    parent_operation_id=str(parent_id),
                                )
                            else:
                                raise RuntimeError("Selected operation type is not revertible by this tool.")

                            revert_succeeded.extend(rev_succeeded)
                            revert_failed.update(rev_failed)
                        except Exception as exc:  # noqa: BLE001
                            revert_failed[key] = str(exc)

                if revert_succeeded:
                    st.success(f"Reverted {len(revert_succeeded)} ticket(s).")
                if revert_failed:
                    for key, err in revert_failed.items():
                        st.error(f"Revert failed for {key}: {err}")

                st.cache_data.clear()
                st.rerun()

    st.caption(
        "Team member filter uses Jira assignee display names from fetched data. "
        "For stricter JQL filtering, use assignee account IDs in JQL."
    )


if __name__ == "__main__":
    main()
