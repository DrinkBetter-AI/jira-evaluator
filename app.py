from __future__ import annotations

import re

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


DEFAULT_JQL = """statusCategory != Done
ORDER BY updated ASC"""

FETCH_SCHEMA_VERSION = 2


@st.cache_data(ttl=300, show_spinner=False)
def fetch_tickets(
    creds_path: str,
    profile_name: str,
    jql: str,
    max_results: int,
    page_size: int,
    schema_version: int,
) -> pd.DataFrame:
    client = JiraClient.from_yaml(creds_path=creds_path, profile_name=profile_name)
    _ = schema_version
    result = client.search_issues(
        jql=jql,
        fields=DEFAULT_FIELDS,
        max_results=max_results,
        page_size=page_size,
    )
    for col in ["sprint_id", "sprint_name", "sprint_state", "sprint_board_id"]:
        if col not in result.columns:
            result[col] = pd.NA
    return result


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


def _fmt_seconds(secs: float) -> str:
    """Convert seconds to a human-readable h/m string."""
    if secs <= 0:
        return "—"
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    if h and m:
        return f"{h}h {m}m"
    if h:
        return f"{h}h"
    return f"{m}m"


def _parse_estimate_to_seconds(value: str) -> float | None:
    """Parse Jira estimate text like '2h', '1d 2h', '30m' into seconds."""
    text = str(value or "").strip().lower()
    if not text:
        return None

    # Jira-style units. We use common defaults: 1d = 8h, 1w = 5d.
    unit_seconds = {
        "m": 60,
        "h": 3600,
        "d": 8 * 3600,
        "w": 5 * 8 * 3600,
    }
    tokens = re.findall(r"(\d+)\s*([mhdw])", text)
    if not tokens:
        return None

    matched = " ".join(f"{num}{unit}" for num, unit in tokens)
    normalized = re.sub(r"\s+", "", text)
    if re.sub(r"\s+", "", matched) != normalized:
        return None

    total = 0.0
    for num, unit in tokens:
        total += int(num) * unit_seconds[unit]
    return total


def _render_sprint_capacity(df: pd.DataFrame) -> None:
    """Show sprint capacity breakdown for a selected future/active sprint, grouped by assignee."""
    required_cols = {"sprint_id", "sprint_name", "sprint_state", "sprint_board_id"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        st.info("Sprint data is not fully loaded yet. Refresh data to enable sprint capacity editing.")
        return

    sprint_df = df[df["sprint_name"].notna()].copy() if "sprint_name" in df.columns else pd.DataFrame()
    if sprint_df.empty:
        st.info("No sprint data found. Ensure your Jira board uses sprints and the sprint field is enabled.")
        return

    sprint_df["sprint_state"] = sprint_df["sprint_state"].fillna("").astype(str)
    non_closed = sprint_df[sprint_df["sprint_state"].str.lower().isin(["future", "active"])]
    target_df = non_closed if not non_closed.empty else sprint_df

    state_rank = {"future": 0, "active": 1, "closed": 2, "": 3}
    sprint_options_df = (
        target_df[["sprint_id", "sprint_name", "sprint_state", "sprint_board_id"]]
        .drop_duplicates()
        .assign(
            state_rank=lambda frame: frame["sprint_state"].str.lower().map(state_rank).fillna(9),
            sprint_label=lambda frame: frame["sprint_name"] + " (" + frame["sprint_state"].str.title().replace("", "Unknown") + ")",
        )
        .sort_values(["state_rank", "sprint_name"])
    )
    sprint_labels = sprint_options_df["sprint_label"].tolist()
    default_idx = 0
    selected_label = st.selectbox("Sprint", options=sprint_labels, index=default_idx)
    selected_row = sprint_options_df.loc[sprint_options_df["sprint_label"] == selected_label].iloc[0]

    scoped = target_df[
        (target_df["sprint_name"] == selected_row["sprint_name"])
        & (target_df["sprint_state"] == selected_row["sprint_state"])
    ].copy()

    st.markdown(f"**Selected sprint:** {selected_row['sprint_name']} ({str(selected_row['sprint_state']).title()})")

    is_ml_sprint = str(selected_row["sprint_name"]).startswith("ML Sprint")

    editable = str(selected_row["sprint_state"]).lower() in {"future", "active"}
    ticket_editor_df = df.copy()
    ticket_editor_df = ticket_editor_df[
        ticket_editor_df["issue_type"].fillna("").astype(str).str.strip().str.lower() != "epic"
    ].copy()
    ticket_editor_df["in_selected_sprint"] = (
        ticket_editor_df["sprint_id"].fillna(-1).astype(str) == str(selected_row["sprint_id"])
    )
    ticket_editor_df["include"] = ticket_editor_df["in_selected_sprint"]
    ticket_editor_df["estimate_edit"] = ticket_editor_df["original_estimate"].fillna("")
    ticket_editor_df = ticket_editor_df[
        ["include", "key", "summary", "assignee", "status", "estimate_edit", "logged_time", "issue_type"]
    ].sort_values(["include", "assignee", "key"], ascending=[False, True, True])

    st.markdown("##### Sprint Tickets")
    st.caption("Check tickets to preview keeping them in or moving them into the selected sprint. Uncheck them to preview sending them to backlog.")
    editor_key = f"sprint_editor_{selected_row['sprint_id']}"
    editor_version_key = f"{editor_key}_version"
    if editor_version_key not in st.session_state:
        st.session_state[editor_version_key] = 0
    editor_widget_key = f"{editor_key}_{st.session_state[editor_version_key]}"
    edited_tickets = st.data_editor(
        ticket_editor_df,
        use_container_width=True,
        hide_index=True,
        disabled=(not editable) or (not is_ml_sprint) or ["key", "summary", "assignee", "status", "logged_time", "issue_type"],
        column_config={
            "include": st.column_config.CheckboxColumn("In Sprint"),
            "key": "Key",
            "summary": "Summary",
            "assignee": "Assignee",
            "status": "Status",
            "estimate_edit": st.column_config.TextColumn(
                "Estimate",
                help="Editable Jira estimate format (examples: 2h, 1d 2h, 30m)",
            ),
            "logged_time": "Logged",
            "issue_type": "Type",
        },
        key=editor_widget_key,
    )

    desired_in_sprint = set(edited_tickets.loc[edited_tickets["include"], "key"].astype(str).tolist())
    # Compare against only the tickets shown in Sprint Tickets to avoid hidden-row drift
    # (e.g., excluded Epic rows creating phantom backlog deltas).
    current_in_sprint = set(
        ticket_editor_df.loc[ticket_editor_df["include"], "key"].astype(str).tolist()
    )
    to_add = sorted(desired_in_sprint - current_in_sprint)
    to_backlog = sorted(current_in_sprint - desired_in_sprint)

    original_estimate_by_key = (
        ticket_editor_df.set_index("key")["estimate_edit"].fillna("").astype(str).str.strip().to_dict()
    )
    edited_estimate_by_key = (
        edited_tickets.set_index("key")["estimate_edit"].fillna("").astype(str).str.strip().to_dict()
    )
    parsed_estimate_seconds_by_key: dict[str, float] = {}
    invalid_estimate_keys: list[str] = []
    for key, value in edited_estimate_by_key.items():
        parsed = _parse_estimate_to_seconds(value)
        if value and parsed is None:
            invalid_estimate_keys.append(str(key))
            continue
        if parsed is not None:
            parsed_estimate_seconds_by_key[str(key)] = parsed

    estimate_updates: dict[str, str] = {}
    skipped_blank_estimates: list[str] = []
    for key, new_value in edited_estimate_by_key.items():
        old_value = original_estimate_by_key.get(key, "")
        if new_value == old_value:
            continue
        if not new_value:
            skipped_blank_estimates.append(key)
            continue
        estimate_updates[str(key)] = new_value

    changed_keys = set(to_add) | set(to_backlog) | set(estimate_updates.keys())
    if changed_keys:
        st.caption("Pending row changes (highlighted)")
        changed_preview = edited_tickets[edited_tickets["key"].astype(str).isin(changed_keys)].copy()

        def _change_type(key: str) -> str:
            parts: list[str] = []
            if key in to_add:
                parts.append("Add to sprint")
            if key in to_backlog:
                parts.append("Move to backlog")
            if key in estimate_updates:
                parts.append("Estimate edited")
            return " + ".join(parts)

        changed_preview["change_type"] = changed_preview["key"].astype(str).map(_change_type)
        changed_preview = changed_preview[
            ["change_type", "include", "key", "summary", "assignee", "status", "estimate_edit", "logged_time", "issue_type"]
        ]

        def _highlight_row(_: pd.Series) -> list[str]:
            return ["background-color: rgba(255, 165, 0, 0.13); font-weight: 600;"] * len(changed_preview.columns)

        st.dataframe(
            changed_preview.style.apply(_highlight_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )

    preview_scoped = df[df["key"].isin(desired_in_sprint)].copy()
    all_sprint_tickets = df[df["sprint_name"].notna()].copy()
    preview_scoped["estimate_seconds_live"] = (
        preview_scoped["key"].astype(str).map(parsed_estimate_seconds_by_key)
        .fillna(preview_scoped["original_estimate_sec"])
        .fillna(0.0)
    )
    all_sprint_tickets["estimate_seconds_live"] = (
        all_sprint_tickets["key"].astype(str).map(parsed_estimate_seconds_by_key)
        .fillna(all_sprint_tickets["original_estimate_sec"])
        .fillna(0.0)
    )

    workload_status_options = sorted(
        pd.Index(pd.concat([preview_scoped["status"], all_sprint_tickets["status"]]).dropna().unique()).tolist()
    )
    default_workload_statuses = [status for status in ["To Do", "In Progress"] if status in workload_status_options]
    if not default_workload_statuses:
        default_workload_statuses = workload_status_options

    if not editable:
        st.info("Sprint membership editing is only available for future or active sprints.")
    else:
        st.caption("`Apply sprint selection` writes sprint membership and estimate edits to Jira.")

    if not is_ml_sprint:
        st.warning(
            f"Sprint editing is restricted to **ML Sprint** boards. "
            f"'{selected_row['sprint_name']}' cannot be modified from this dashboard."
        )

    if skipped_blank_estimates:
        st.caption(
            f"Blank estimate edits are ignored for {len(skipped_blank_estimates)} ticket(s). "
            "Use a Jira estimate format like `2h` or `1d 2h`."
        )
    if invalid_estimate_keys:
        st.caption(
            f"Invalid estimate format for {len(invalid_estimate_keys)} ticket(s); "
            "live totals keep previous values for those rows."
        )

    action_col1, action_col2 = st.columns([4, 1])
    with action_col1:
        apply_sprint_selection = st.button(
            f"Apply sprint selection ({len(to_add)} add, {len(to_backlog)} backlog, {len(estimate_updates)} estimates)",
            disabled=(not editable) or (not is_ml_sprint) or (not to_add and not to_backlog and not estimate_updates),
            type="primary",
            key=f"apply_sprint_{selected_row['sprint_id']}",
        )
    with action_col2:
        reset_sprint_changes = st.button(
            "Reset changes",
            disabled=(not editable) or (not is_ml_sprint),
            key=f"reset_sprint_{selected_row['sprint_id']}",
            help="Discard unsaved checkbox/estimate edits in Sprint Tickets.",
        )

    if reset_sprint_changes:
        st.session_state[editor_version_key] = int(st.session_state.get(editor_version_key, 0)) + 1
        st.rerun()

    if apply_sprint_selection:
        client = JiraClient.from_yaml(
            creds_path="~/.creds/vinovoss.yml",
            profile_name="ML-TEAM-MANAGEMENT",
        )
        with st.spinner("Updating sprint membership..."):
            parts: list[str] = []
            had_success = False
            try:
                if to_add:
                    client.add_issues_to_sprint(selected_row["sprint_id"], to_add)
                    parts.append(f"added {len(to_add)}")
                    had_success = True
                if to_backlog:
                    client.move_issues_to_backlog(to_backlog)
                    parts.append(f"moved {len(to_backlog)} to backlog")
                    had_success = True
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to update sprint membership: {exc}")

            estimate_success = 0
            estimate_failed: dict[str, str] = {}
            for key, estimate in estimate_updates.items():
                try:
                    client.update_issue(key, {"timetracking": {"originalEstimate": estimate}})
                    estimate_success += 1
                except Exception as exc:  # noqa: BLE001
                    estimate_failed[key] = str(exc)

            if estimate_success:
                parts.append(f"updated {estimate_success} estimate(s)")
                had_success = True
            if estimate_failed:
                for key, err in estimate_failed.items():
                    st.error(f"Estimate update failed for {key}: {err}")

            if parts:
                st.success("Update completed: " + ", ".join(parts))
            if had_success:
                st.cache_data.clear()
                st.rerun()

    calc_col1, calc_col2 = st.columns([2, 3])
    with calc_col1:
        workload_statuses = st.multiselect(
            "Statuses counted in hours",
            options=workload_status_options,
            default=default_workload_statuses,
            help="Use this to focus sprint effort on work that still needs attention.",
            key=f"workload_statuses_{selected_row['sprint_id']}",
        )
    with calc_col2:
        st.caption("Hour totals and the assignee workload table use the selected statuses. `Tickets in sprint` stays as the full selected-sprint count.")

    if workload_statuses:
        preview_workload = preview_scoped[preview_scoped["status"].isin(workload_statuses)].copy()
        all_sprint_workload = all_sprint_tickets[all_sprint_tickets["status"].isin(workload_statuses)].copy()
    else:
        preview_workload = preview_scoped.iloc[0:0].copy()
        all_sprint_workload = all_sprint_tickets.iloc[0:0].copy()

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        (
            '<div style="font-size: 0.875rem; opacity: 0.85; margin-bottom: 0.2rem;">Tickets in sprint</div>'
            f'<div style="font-size: 2.2rem; font-weight: 700; line-height: 1.05;">'
            f'{len(preview_workload)} '
            '<span style="font-size: 0.75rem; font-weight: 500; opacity: 0.75; vertical-align: middle;">out of</span> '
            f'{len(preview_scoped)}</div>'
        ),
        unsafe_allow_html=True,
    )
    c2.metric(
        "Total estimated (sprint)",
        _fmt_seconds(preview_workload["estimate_seconds_live"].fillna(0).sum()),
    )
    c3.metric(
        "Grand Total",
        _fmt_seconds(all_sprint_workload["estimate_seconds_live"].fillna(0).sum()),
    )

    # Per-assignee breakdown
    st.markdown("##### Capacity per Assignee")
    agg = (
        preview_workload.groupby("assignee")
        .agg(
            tickets=("key", "count"),
            estimated_sec=("estimate_seconds_live", "sum"),
            logged_sec=("time_spent_sec", "sum"),
        )
        .reset_index()
    )
    agg["Total Estimated"] = agg["estimated_sec"].apply(_fmt_seconds)
    agg["Total Logged"] = agg["logged_sec"].apply(_fmt_seconds)
    agg["Remaining"] = (agg["estimated_sec"] - agg["logged_sec"]).clip(lower=0).apply(_fmt_seconds)
    agg = agg.rename(columns={"assignee": "Assignee", "tickets": "Tickets"})
    st.dataframe(
        agg[["Assignee", "Tickets", "Total Estimated", "Total Logged", "Remaining"]],
        use_container_width=True,
        hide_index=True,
    )


_PRIORITY_BUCKET_MAP = {
    "low": "Normal",
    "lowest": "Normal",
    "normal": "Normal",
    "medium": "Normal",
    "high": "High",
    "highest": "Urgent",
    "urgent": "Urgent",
}
_BUCKET_COLORS = {"Normal": "#2ECC71", "High": "#F5A623", "Urgent": "#E74C3C"}


def _render_bubble_chart(df: pd.DataFrame, color_by: str = "priority", agg_priority: bool = False) -> None:
    if df.empty:
        st.info("No data available for staleness bubble chart.")
        return

    plot_df = df.copy()

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

    age = plot_df["ticket_age_days"].clip(lower=1)
    plot_df["bubble_size"] = ((age - age.min()) / (age.max() - age.min() + 1e-9) * 31 + 3).round(1)

    if agg_priority and color_by == "priority":
        plot_df["priority_bucket"] = (
            plot_df["priority"].fillna("none").astype(str).str.strip().str.lower()
            .map(_PRIORITY_BUCKET_MAP)
            .fillna("Normal")
        )
        fig = px.scatter(
            plot_df,
            x="idle_days",
            y="y_jitter",
            size="bubble_size",
            color="priority_bucket",
            color_discrete_map=_BUCKET_COLORS,
            category_orders={"priority_bucket": ["Normal", "High", "Urgent"]},
            custom_data=["key", "summary", "assignee", "status_label", "priority", "ticket_age_days", "idle_days"],
            title="Staleness vs Workflow Status (Aggregated Priority)",
            labels={"idle_days": "Idle Days", "y_jitter": "Status", "priority_bucket": "Priority"},
            size_max=34,
            opacity=0.3,
        )
    else:
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

    refresh_clicked = st.button("Refresh Data")

    if refresh_clicked:
        st.cache_data.clear()

    jql = DEFAULT_JQL
    max_results = 1000
    page_size = 100

    try:
        raw_df = fetch_tickets(
            creds_path="~/.creds/vinovoss.yml",
            profile_name="ML-TEAM-MANAGEMENT",
            jql=jql,
            max_results=max_results,
            page_size=page_size,
            schema_version=FETCH_SCHEMA_VERSION,
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

    ML_TEAM_MEMBERS = ["Tam", "Shivanand", "Mehdi Ordikhani"]
    default_assignees = [m for m in ML_TEAM_MEMBERS if m in assignees]
    selected_assignees = f1.multiselect("Assignee", options=assignees, default=default_assignees)
    selected_statuses = f2.multiselect("Status", options=statuses, default=[])
    selected_priorities = f3.multiselect("Priority", options=priorities, default=[])
    min_idle = f4.slider("Min idle days", min_value=0, max_value=180, value=0)
    min_age = f5.slider("Min ticket age", min_value=0, max_value=365, value=0)

    color_by = st.radio("Bubble color", options=["priority", "assignee"], horizontal=True)

    filtered = df.copy()
    if selected_assignees:
        filtered = filtered[filtered["assignee"].isin(selected_assignees)]
    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]
    if selected_priorities:
        filtered = filtered[filtered["priority"].isin(selected_priorities)]

    filtered = filtered[(filtered["idle_days"] >= min_idle) & (filtered["ticket_age_days"] >= min_age)]

    _render_metrics(filtered)

    agg_priority = st.checkbox(
        "Aggregate Priorities (Normal / High / Urgent)",
        value=False,
        help="Buckets: Normal = None/Low/Normal · High = High · Urgent = Highest/Urgent",
    )
    _render_bubble_chart(filtered, color_by=color_by, agg_priority=agg_priority)

    st.divider()
    st.subheader("Sprint Capacity")
    _render_sprint_capacity(filtered)

    st.subheader("Raw Tickets")
    raw_columns = [
        "key",
        "summary",
        "status",
        "priority",
        "assignee",
        "reporter",
        "original_estimate",
        "logged_time",
        "completion_pct",
        "ticket_age_days",
        "idle_days",
        "created",
        "updated",
    ]
    SORT_DISPLAY = {
        "key": "Key",
        "summary": "Summary",
        "status": "Status",
        "priority": "Priority",
        "assignee": "Assignee",
        "reporter": "Reporter",
        "original_estimate": "Original Estimate",
        "logged_time": "Logged Time",
        "completion_pct": "Completion %",
        "ticket_age_days": "Age (days)",
        "idle_days": "Idle (days)",
        "created": "Created at",
        "updated": "Updated at",
    }
    SORT_DISPLAY_INV = {v: k for k, v in SORT_DISPLAY.items()}

    with st.expander("Sort order (SQL-style ORDER BY)", expanded=False):
        n_sorts = st.number_input("Number of sort levels", min_value=1, max_value=5, value=2, step=1)
        sort_cols: list[str] = []
        sort_asc: list[bool] = []
        sort_row_cols = st.columns(int(n_sorts))
        for i, col_widget in enumerate(sort_row_cols):
            with col_widget:
                default_col = list(SORT_DISPLAY.values())[min(i, len(SORT_DISPLAY) - 1)]
                chosen_label = st.selectbox(
                    f"Level {i + 1} column",
                    options=list(SORT_DISPLAY.values()),
                    index=list(SORT_DISPLAY.values()).index(default_col),
                    key=f"sort_col_{i}",
                )
                direction = st.radio(
                    "Direction",
                    options=["ASC", "DESC"],
                    index=0,
                    horizontal=True,
                    key=f"sort_dir_{i}",
                )
                sort_cols.append(SORT_DISPLAY_INV[chosen_label])
                sort_asc.append(direction == "ASC")

    raw_display_df = (
        filtered[raw_columns]
        .sort_values(sort_cols, ascending=sort_asc)
        .rename(columns={"created": "Created at", "updated": "Updated at", "completion_pct": "Completion %", "original_estimate": "Original Estimate", "logged_time": "Logged Time"})
    )
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
