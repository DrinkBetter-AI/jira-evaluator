"""How close logged time came to the estimate, and what that is worth knowing.

Jira already carries both numbers - ``original_estimate_sec`` and
``time_spent_sec`` - and the dashboard shows them in one collapsed table behind
a checkbox. Nothing compares them. For a team of hourly contractors that
comparison is the whole point of asking for an estimate in the first place:
an estimate nobody is ever measured against is a form to fill in, not a
commitment.

What these numbers are, stated plainly so nobody over-reads them:

- the estimate is self-reported;
- the logged time is self-reported;
- the only machine-recorded quantity in the neighbourhood is the diff a PR
  changed, and it measures typing rather than thinking.

So this module detects *inconsistency* between two self-reported figures and one
machine-recorded one. It cannot detect padding on its own, because a person who
estimates eight hours and logs eight hours for two hours of work is perfectly
self-consistent. What it does catch is the thing that is hard to keep
consistent: a ratio that sits far from the team's, a person whose tickets always
finish comfortably inside an estimate nobody else can hit, or hours that buy an
order of magnitude less delivered change than everyone else's.

A ratio well under 1 means the estimates were padded or the work was easier than
thought; well over 1 means it was underestimated. Both are worth fixing. Only
the first is a billing risk, and only when the logged hours are what gets
invoiced.
"""

from __future__ import annotations

import pandas as pd

import pr_hygiene

# Below this share of the estimate, a ticket "finished early" in the sense the
# padding index counts. 0.6 is a deliberate margin: estimating is hard and
# beating an estimate by a fifth is ordinary good luck.
UNDER_RUN_RATIO = 0.6

# No verdict on fewer tickets than this. Three tickets cannot separate a padder
# from someone who had a quiet fortnight, and a scorecard that pretends
# otherwise is worse than no scorecard.
MIN_TICKETS_FOR_VERDICT = 5

# Modified z-score cut-off (Iglewicz and Hoaglin). Applied to the team's own
# distribution, so the bar moves with the team instead of being a number someone
# picked once and forgot.
OUTLIER_Z = 3.5

_DONE_CATEGORIES = frozenset({"done", "complete", "completed", "resolved"})


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def is_done(tickets: pd.DataFrame) -> pd.Series:
    """Which tickets are finished, by resolution or by status category.

    This matters more than it looks. On an unfinished ticket, time logged so far
    against the estimate is not an accuracy ratio at all - it is progress - and
    counting it would mark every ticket started yesterday as a padded estimate.
    """
    if tickets.empty:
        return pd.Series(dtype=bool)
    resolved = (
        tickets["resolution"].notna() & (tickets["resolution"].astype(str).str.strip() != "")
        if "resolution" in tickets.columns
        else pd.Series(False, index=tickets.index)
    )
    category = (
        tickets["status_category"].fillna("").astype(str).str.strip().str.lower()
        if "status_category" in tickets.columns
        else pd.Series("", index=tickets.index)
    )
    return resolved | category.isin(_DONE_CATEGORIES)


def accuracy_ratio(tickets: pd.DataFrame, completed_only: bool = True) -> pd.DataFrame:
    """Per ticket: logged hours over estimated hours.

    Only tickets that carry both numbers can appear. A ticket with no estimate
    is a different problem, already measured elsewhere as estimate coverage; a
    ticket with an estimate and no logged time is either not started or not
    logged, and either way says nothing about accuracy.

    ``completed_only`` keeps unfinished tickets out, because on those the ratio
    is a progress bar rather than an outcome.
    """
    columns = [
        "key",
        "assignee",
        "summary",
        "status",
        "estimate_hours",
        "logged_hours",
        "ratio",
        "under_ran",
        "over_ran",
    ]
    if tickets.empty:
        return _empty(columns)
    # A clean index: the ticket frames get filtered and concatenated upstream,
    # and duplicate labels would turn the masks below into a cross product.
    tickets = tickets.reset_index(drop=True)
    frame = tickets[is_done(tickets)] if completed_only else tickets
    if frame.empty:
        return _empty(columns)
    estimate = _numeric(frame, "original_estimate_sec")
    logged = _numeric(frame, "time_spent_sec")
    usable = (estimate > 0) & (logged > 0)
    frame = frame[usable.fillna(False)]
    if frame.empty:
        return _empty(columns)
    estimate_hours = _numeric(frame, "original_estimate_sec") / 3600.0
    logged_hours = _numeric(frame, "time_spent_sec") / 3600.0
    ratio = logged_hours / estimate_hours
    out = pd.DataFrame(
        {
            "key": frame.get("key", pd.Series("", index=frame.index)),
            "assignee": frame.get("assignee", pd.Series("Unassigned", index=frame.index)),
            "summary": frame.get("summary", pd.Series("", index=frame.index)),
            "status": frame.get("status", pd.Series("", index=frame.index)),
            "estimate_hours": estimate_hours.round(2),
            "logged_hours": logged_hours.round(2),
            "ratio": ratio.round(3),
            "under_ran": ratio < UNDER_RUN_RATIO,
            "over_ran": ratio > 1.0,
        }
    )
    return out[columns].reset_index(drop=True)


def accuracy_by_person(
    tickets: pd.DataFrame,
    completed_only: bool = True,
    min_tickets: int = MIN_TICKETS_FOR_VERDICT,
) -> pd.DataFrame:
    """The ratio distribution per person: median, quartiles, spread, count.

    The median rather than the mean, because one ticket someone forgot to close
    for a month would otherwise decide the number. The interquartile range is
    the part worth reading next to it: a person whose ratios run 0.9 to 1.1 is
    estimating well, and a person whose median is 1.0 with an IQR of 2.5 is not
    estimating at all - the two look identical if only the median is shown.

    ``enough_data`` is false where the sample is under ``min_tickets``; the row
    is still returned, because knowing somebody logs against almost nothing is
    itself a finding.
    """
    columns = [
        "assignee",
        "tickets",
        "median_ratio",
        "p25_ratio",
        "p75_ratio",
        "iqr",
        "estimated_hours",
        "logged_hours",
        "enough_data",
    ]
    if tickets.empty:
        return _empty(columns)
    detail = accuracy_ratio(tickets, completed_only=completed_only)
    if detail.empty:
        return _empty(columns)
    grouped = detail.groupby("assignee", dropna=False)
    out = pd.DataFrame(
        {
            "tickets": grouped.size(),
            "median_ratio": grouped["ratio"].median(),
            "p25_ratio": grouped["ratio"].quantile(0.25),
            "p75_ratio": grouped["ratio"].quantile(0.75),
            "estimated_hours": grouped["estimate_hours"].sum().round(1),
            "logged_hours": grouped["logged_hours"].sum().round(1),
        }
    )
    out["iqr"] = (out["p75_ratio"] - out["p25_ratio"]).round(3)
    out["enough_data"] = out["tickets"] >= int(min_tickets)
    out = out.reset_index()
    return out[columns].sort_values("median_ratio", ignore_index=True)


def padding_index(
    tickets: pd.DataFrame,
    completed_only: bool = True,
    min_tickets: int = MIN_TICKETS_FOR_VERDICT,
    under_run_ratio: float = UNDER_RUN_RATIO,
) -> pd.DataFrame:
    """Per person: median ratio, how often they finish well inside the estimate.

    Two columns that have to be read together. A median ratio under 1 on its own
    is unremarkable - most people pad a little, and finishing early is a virtue.
    A median under 1 *and* a large share of tickets landing under 60% of the
    estimate means the estimates are consistently generous, every time,
    regardless of the work. That is the pattern an estimate-padding incentive
    produces, and it is the only pattern here worth raising.

    ``verdict`` is deliberately blank below ``min_tickets``: with fewer tickets
    than that there is no verdict to give, and printing one anyway is how a
    dashboard ends up being argued with instead of used.

    None of this proves padding. It shows a person whose estimates are shaped
    differently from everyone else's, on numbers they supplied themselves.
    """
    columns = [
        "assignee",
        "tickets",
        "median_ratio",
        "under_run_share",
        "estimated_hours",
        "logged_hours",
        "enough_data",
        "verdict",
    ]
    if tickets.empty:
        return _empty(columns)
    detail = accuracy_ratio(tickets, completed_only=completed_only)
    if detail.empty:
        return _empty(columns)
    detail = detail.assign(under=detail["ratio"] < float(under_run_ratio))
    grouped = detail.groupby("assignee", dropna=False)
    out = pd.DataFrame(
        {
            "tickets": grouped.size(),
            "median_ratio": grouped["ratio"].median(),
            "under_run_share": grouped["under"].mean(),
            "estimated_hours": grouped["estimate_hours"].sum().round(1),
            "logged_hours": grouped["logged_hours"].sum().round(1),
        }
    )
    out["enough_data"] = out["tickets"] >= int(min_tickets)

    def _verdict(row: pd.Series) -> str:
        if not row["enough_data"]:
            return ""
        if row["median_ratio"] < under_run_ratio and row["under_run_share"] >= 0.5:
            return "estimates consistently generous"
        if row["median_ratio"] > 1.5:
            return "estimates consistently low"
        return "estimates broadly hold"

    out["verdict"] = out.apply(_verdict, axis=1)
    out = out.reset_index()
    return out[columns].sort_values(
        ["under_run_share", "tickets"], ascending=False, ignore_index=True
    )


def _modified_z(values: pd.Series) -> pd.Series:
    """Robust z-score against the median, using the median absolute deviation.

    Median-based on purpose: with a handful of people, one extreme value drags a
    mean-and-standard-deviation rule far enough that the outlier stops looking
    like one.

    A MAD of zero is the common case on a small team - eight people at the same
    rate and one nowhere near it - and dividing by it would flag everybody or
    nobody depending on rounding. There it falls back to the mean absolute
    deviation form of the same statistic. When both are zero the team is
    genuinely uniform, there is no outlier to find, and the score is missing
    rather than invented.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    clean = numeric.dropna()
    unknown = pd.Series(pd.NA, index=values.index, dtype="Float64")
    if len(clean) < 3:
        return unknown
    median = clean.median()
    deviations = (clean - median).abs()
    mad = deviations.median()
    if mad and not pd.isna(mad):
        return (0.6745 * (numeric - median) / mad).astype("Float64")
    mean_ad = deviations.mean()
    if mean_ad and not pd.isna(mean_ad):
        return ((numeric - median) / (1.253314 * mean_ad)).astype("Float64")
    return unknown


def hours_per_delivered_line(
    tickets: pd.DataFrame,
    prs: pd.DataFrame,
    project_keys: list[str] | None = None,
    completed_only: bool = True,
    outlier_z: float = OUTLIER_Z,
) -> pd.DataFrame:
    """Logged hours against the lines a ticket's PRs actually changed.

    This is the only place the self-reported side of the ledger meets something
    a machine recorded. The Jira key links them: hours logged on a ticket, and
    the diff of every PR naming that ticket. Key matching is :mod:`pr_hygiene`'s,
    imported rather than repeated.

    Outliers are flagged against the team's own median with a median absolute
    deviation, not a fixed threshold, because "hours per line" has no defensible
    absolute value. Anyone flagged is simply far from how the rest of this team
    converts hours into change - in either direction.

    What it cannot do: measure difficulty. A day spent finding a one-line race
    condition and a day spent generating boilerplate land at opposite ends of
    this column, and the hard one looks worse. It also mis-reads any ticket
    whose PR includes a lockfile, a migration or vendored code. Treat a flag as
    a reason to open the PR, and never as a number to bill against.
    """
    columns = [
        "assignee",
        "key",
        "logged_hours",
        "prs",
        "changed_lines",
        "hours_per_100_lines",
        "modified_z",
        "is_outlier",
    ]
    if tickets.empty or prs.empty:
        return _empty(columns)

    tickets = tickets.reset_index(drop=True)
    prs = prs.reset_index(drop=True)
    frame = tickets[is_done(tickets)] if completed_only else tickets
    logged = _numeric(frame, "time_spent_sec")
    frame = frame[(logged > 0).fillna(False)]
    if frame.empty or "key" not in frame.columns:
        return _empty(columns)

    keyed = prs if "jira_key" in prs.columns else pr_hygiene.add_hygiene_fields(prs, project_keys)
    if "jira_key" not in keyed.columns or "changed_lines" not in keyed.columns:
        return _empty(columns)
    keyed = keyed[keyed["jira_key"].astype(str).str.strip().astype(bool)]
    keyed = keyed[pd.to_numeric(keyed["changed_lines"], errors="coerce").notna()]
    if keyed.empty:
        return _empty(columns)
    per_key = (
        keyed.assign(changed_lines=pd.to_numeric(keyed["changed_lines"], errors="coerce"))
        .groupby("jira_key", dropna=False)
        .agg(prs=("changed_lines", "size"), changed_lines=("changed_lines", "sum"))
        .reset_index()
    )

    joined = pd.DataFrame(
        {
            "assignee": frame.get("assignee", pd.Series("Unassigned", index=frame.index)),
            "key": frame["key"].astype(str),
            "logged_hours": (_numeric(frame, "time_spent_sec") / 3600.0).round(2),
        }
    ).merge(per_key, left_on="key", right_on="jira_key", how="inner")
    if joined.empty:
        return _empty(columns)
    joined = joined.drop(columns=["jira_key"])
    # Zero-line PRs exist (a revert of a revert, an empty merge); a ticket whose
    # PRs changed nothing has no rate, rather than an infinite one.
    lines = joined["changed_lines"].where(joined["changed_lines"] > 0)
    joined["hours_per_100_lines"] = (100.0 * joined["logged_hours"] / lines).round(2)
    joined["modified_z"] = _modified_z(joined["hours_per_100_lines"])
    # The flag is what a page filters on, so an unanswerable row is "not
    # flagged", never a missing boolean; ``modified_z`` stays missing to show
    # the question could not be answered rather than that it came back clean.
    joined["is_outlier"] = (
        (joined["modified_z"].abs() > float(outlier_z)).fillna(False).astype(bool)
    )
    return joined[columns].sort_values(
        "hours_per_100_lines", ascending=False, ignore_index=True, na_position="last"
    )
