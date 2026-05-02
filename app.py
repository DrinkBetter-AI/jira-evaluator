from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from jira_client import DEFAULT_FIELDS, JiraClient, JiraConfigError
from transformations import DEFAULT_ACTIVE_STATUSES, add_ticket_health_fields


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
    zombies = int(df["is_zombie"].sum()) if total_open else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Open Tickets", total_open)
    m2.metric("Average Idle Days", f"{avg_idle:.1f}")
    m3.metric("Max Idle Days", f"{max_idle:.1f}")
    m4.metric("Oldest Ticket Age", f"{oldest:.1f}")
    m5.metric("Zombie Tickets", zombies)


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



def _render_zombie_table(df: pd.DataFrame) -> None:
    cols = [
        "key",
        "summary",
        "status",
        "priority",
        "assignee",
        "ticket_age_days",
        "idle_days",
        "reporter",
        "due_date",
        "risk_score",
    ]

    zombies = (
        df[df["is_zombie"]]
        .sort_values(["idle_days", "risk_score"], ascending=[False, False])
        .loc[:, cols]
    )
    st.subheader("Zombie Tickets")
    if zombies.empty:
        st.success("No zombie tickets detected with the current threshold and filters.")
        return

    st.dataframe(zombies, use_container_width=True)


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
            zombie_idle_threshold = st.slider("Zombie threshold (idle days)", min_value=1, max_value=30, value=3)
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

    df = add_ticket_health_fields(raw_df, zombie_idle_threshold=zombie_idle_threshold)

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

    show_only_zombies = st.checkbox("Show only zombie tickets", value=False)
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
    if show_only_zombies:
        filtered = filtered[filtered["is_zombie"]]

    _render_metrics(filtered)

    _render_bubble_chart(filtered, color_by=color_by)

    _render_zombie_table(filtered)

    st.subheader("Raw Tickets")
    raw_columns = [
        "key",
        "summary",
        "status",
        "status_category",
        "priority",
        "assignee",
        "created",
        "updated",
        "due_date",
        "ticket_age_days",
        "idle_days",
        "is_zombie",
        "risk_score",
    ]
    st.dataframe(
        filtered.sort_values(["risk_score", "idle_days"], ascending=[False, False])[raw_columns],
        use_container_width=True,
    )

    st.caption(
        f"Active status set for zombie logic: {', '.join(sorted(DEFAULT_ACTIVE_STATUSES))}"
    )
    st.caption(
        "Team member filter uses Jira assignee display names from fetched data. "
        "For stricter JQL filtering, use assignee account IDs in JQL."
    )


if __name__ == "__main__":
    main()
