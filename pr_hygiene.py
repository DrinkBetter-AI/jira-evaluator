"""Which open PRs are untraceable, stalled, or unowned.

The dashboard already says who has PRs open and which lack an approval. This
answers the questions that decide whether the repos stay reviewable: can this PR
be traced back to a Jira ticket, has it been sitting long enough that nobody is
really working on it, and is anyone on the hook for reviewing it.

A key is looked for in the title, the branch name and the body, because teams
put it in whichever of the three their tooling fills in. When the set of real
Jira project keys is known it is used to match, so a string like ``UTF-8`` or
``COVID-19`` in a title does not read as a ticket reference.
"""

from __future__ import annotations

import os
import re

import pandas as pd


def _positive_float(value: str | None, *, default: float) -> float:
    """Read a positive number setting; a typo costs the setting, not the app."""
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


# Default thresholds. Age is when a PR stops being "in progress" and starts
# being a merge conflict waiting to happen; idle is when it stops moving at all.
STALE_AGE_DAYS = _positive_float(os.getenv("PR_STALE_AGE_DAYS"), default=14.0)
STALE_IDLE_DAYS = _positive_float(os.getenv("PR_STALE_IDLE_DAYS"), default=7.0)

# Jira keys: uppercase project key, hyphen, number. Bounded so "A-1" inside a
# longer token, or a lowercase word, does not count.
_GENERIC_KEY_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})-(\d+)\b")


def _key_pattern(project_keys: list[str] | None) -> re.Pattern[str]:
    """Match a key from ``project_keys``, or any key-shaped token if none given.

    The known-key match ignores case, because branch tooling routinely lowers it
    (``mb-1234-fix-login``) and that branch is one of the three places the key is
    looked for. The explicit key list is what keeps ``utf-8`` out; the generic
    fallback stays case-sensitive, where nothing constrains it.
    """
    if not project_keys:
        return _GENERIC_KEY_RE
    keys = sorted({k.strip().upper() for k in project_keys if k and k.strip()}, key=len, reverse=True)
    if not keys:
        return _GENERIC_KEY_RE
    return re.compile(
        r"\b(" + "|".join(re.escape(k) for k in keys) + r")-(\d+)\b",
        re.IGNORECASE,
    )


def find_jira_key(text: str, pattern: re.Pattern[str]) -> str:
    """First Jira key in ``text``, upper-cased, or an empty string."""
    match = pattern.search(text or "")
    return f"{match.group(1).upper()}-{match.group(2)}" if match else ""


def add_hygiene_fields(
    prs: pd.DataFrame,
    project_keys: list[str] | None = None,
    stale_age_days: float = STALE_AGE_DAYS,
    stale_idle_days: float = STALE_IDLE_DAYS,
) -> pd.DataFrame:
    """Add ``jira_key``, ``has_jira_key``, ``is_stale`` and ``is_unowned``."""
    if prs.empty:
        return prs
    frame = prs.copy()
    pattern = _key_pattern(project_keys)

    def _key(row: pd.Series) -> str:
        for field in ("title", "branch", "body"):
            found = find_jira_key(str(row.get(field, "") or ""), pattern)
            if found:
                return found
        return ""

    frame["jira_key"] = frame.apply(_key, axis=1)
    frame["has_jira_key"] = frame["jira_key"].astype(bool)

    age = frame.get("age_days", pd.Series(0.0, index=frame.index)).fillna(0.0)
    idle = frame.get("idle_days", pd.Series(0.0, index=frame.index)).fillna(0.0)
    frame["is_stale"] = (age > stale_age_days) | (idle > stale_idle_days)
    # Why a PR is stale matters for what to do about it: an old but active PR
    # needs splitting, an idle one needs chasing or closing.
    frame["stale_reason"] = [
        ", ".join(
            reason
            for reason, hit in (
                (f"open >{stale_age_days:.0f}d", a > stale_age_days),
                (f"untouched >{stale_idle_days:.0f}d", i > stale_idle_days),
            )
            if hit
        )
        for a, i in zip(age, idle)
    ]

    # Unowned: nobody was asked to review and nobody has. A PR with a review,
    # even a rejecting one, has someone's attention.
    requests = frame.get("review_requests", pd.Series(0, index=frame.index))
    reviews = frame.get("total_reviews", pd.Series(0, index=frame.index))
    frame["is_unowned"] = (
        pd.Series(requests, index=frame.index).fillna(0).astype(int) == 0
    ) & (pd.Series(reviews, index=frame.index).fillna(0).astype(int) == 0)
    return frame


# The priorities worth interrupting someone for. Both names exist in this
# instance, and "urgent" is here because a renamed top priority should not
# quietly stop being urgent.
CRITICAL_PRIORITIES = frozenset({"highest", "high", "urgent", "critical", "blocker"})

# Where work is close enough to shipping that a stalled PR holds up a release
# rather than merely sitting in someone's branch.
# "In Progress" is deliberately absent: it means someone is writing the code,
# not that the code is somewhere it can hold up a release.
IN_FLIGHT_STATUSES = frozenset(
    {
        "in dev env",
        "code review",
        "review",
        "design review",
        "review in staging",
        "ready for production",
    }
)


def _normalized(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def critical_in_flight(prs: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    """Open PRs whose ticket is high priority and already in dev, review or staging.

    The PR list on its own cannot say which PRs matter - GitHub knows nothing
    about priority - so the ticket each PR names supplies it. A PR with no key
    cannot be judged this way at all and is left to the untraceable tab.
    """
    columns = ["priority", "ticket_status"]
    if prs.empty or "jira_key" not in prs.columns:
        return prs.iloc[0:0].assign(**{column: "" for column in columns})
    keyed = prs[prs["jira_key"].astype(bool)]
    if keyed.empty or tickets.empty or "key" not in tickets.columns:
        return keyed.iloc[0:0].assign(**{column: "" for column in columns})

    facts = (
        tickets[["key", "priority", "status"]]
        .drop_duplicates(subset="key", keep="last")
        .rename(columns={"status": "ticket_status"})
    )
    merged = keyed.merge(
        facts, left_on="jira_key", right_on="key", how="inner", suffixes=("", "_ticket")
    )
    if merged.empty:
        return merged.assign(**{column: "" for column in columns if column not in merged})
    critical = _normalized(merged["priority"]).isin(CRITICAL_PRIORITIES)
    in_flight = _normalized(merged["ticket_status"]).isin(IN_FLIGHT_STATUSES)
    return merged[critical & in_flight].drop(columns=["key"], errors="ignore")


def hygiene_by_person(prs: pd.DataFrame) -> pd.DataFrame:
    """Per-author counts of each problem, worst offender first."""
    if prs.empty:
        return prs
    rollup = (
        prs.assign(no_key=~prs["has_jira_key"])
        .groupby("author")
        .agg(
            open_prs=("number", "size"),
            no_jira_key=("no_key", "sum"),
            stale=("is_stale", "sum"),
            unowned=("is_unowned", "sum"),
            oldest_days=("age_days", "max"),
        )
        .reset_index()
    )
    for column in ("no_jira_key", "stale", "unowned"):
        rollup[column] = rollup[column].astype(int)
    rollup["problems"] = rollup[["no_jira_key", "stale", "unowned"]].sum(axis=1)
    return rollup.sort_values(["problems", "oldest_days"], ascending=[False, False])
