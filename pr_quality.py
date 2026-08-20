"""What a pull request was worth, and what it cost to review.

The dashboard counts PRs. A README typo and a 900-line feature both count as
one, nobody is measured on the reviewing they do or fail to do, and the AI
reviewer's findings - the closest thing the team has to an objective read on how
much is wrong with a PR before it merges - are not looked at at all.

This module is the arithmetic for those questions, over the frames
:mod:`github_client` already returns. Pure pandas: no network, no Streamlit, no
thresholds that fire an alert. Every function returns a frame the page can sort,
because each one is a prompt to look at a person's PRs, not a verdict on them:

- size, so output can be weighted rather than counted, and so five trivial PRs
  in a day are visible as five trivial PRs;
- the AI reviewer's verdicts per PR and per author;
- reviews given, which is real work that currently earns nobody anything;
- reciprocity, which is a question ("why do these two only ever review each
  other?") and never an accusation;
- self-merges and unapproved merges;
- abandoned PRs, which cost the same hours as shipped ones;
- traceability of *merged* work back to a Jira ticket;
- reviews nobody asked for, which is the proactivity signal review counts
  alone cannot show;
- a PR going back to draft after a reviewer was already asked - the specific
  ordering that turns "hide a PR in draft" from a click into a fact; and
- size converted into points, count and median bound together so a count
  alone can never leave this module.

Two honest limits run through all of it. Diff size measures typing, not
thinking: a one-line fix to a race condition can be the hardest work of the
month, and generated files or a lockfile can make a trivial change enormous.
And any measure derived from reviews is a measure of two people at once - a PR
with no findings may be clean work, or may be work nobody read.
"""

from __future__ import annotations

import fnmatch
from typing import NamedTuple

import pandas as pd

import pr_hygiene

# Size bands, in changed lines (additions + deletions). The boundaries are
# conventional rather than derived; what matters is that they are stable, so a
# person's mix can be compared with their own last month.
TRIVIAL_MAX = 10  # exclusive: under ten changed lines
SMALL_MAX = 100
MEDIUM_MAX = 400
LARGE_MAX = 1000  # over this is "oversized": too big to review honestly
SIZE_BANDS = ("trivial", "small", "medium", "large", "oversized")

# Points a merged PR in each band is worth, for :func:`delivered_points`.
# Convex up to "large" (bigger, well-scoped work is worth more than the same
# work split small, which is exactly what KPI_SPEC exploit #3 rewards today),
# flat from "large" to "oversized": past LARGE_MAX a diff is "too big to
# review honestly" per the band's own docstring, so it stops earning more for
# being bigger still - that would reward the dump, not the work.
SIZE_POINTS = {
    "trivial": 1,
    "small": 3,
    "medium": 8,
    "large": 20,
    "oversized": 20,
}

# The AI reviewer's login. GitHub apps post as ``name[bot]``, so this is a glob.
# Configurable because the reviewer can be swapped for another one.
AI_REVIEWER_PATTERNS = ("devin-ai-integration*",)

# Branches a merge into means "shipped". Anything else is a merge into someone's
# own stack, which is bookkeeping rather than delivery.
TRUNK_BRANCHES = frozenset({"main", "master", "develop", "development"})

_APPROVED = "APPROVED"
_CHANGES_REQUESTED = "CHANGES_REQUESTED"
_COMMENTED = "COMMENTED"


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _rows(prs: pd.DataFrame) -> pd.DataFrame:
    """The frame with a clean index.

    Several of these are meant to be fed two fetches concatenated together - the
    merged PRs and the abandoned ones - and a caller who forgets ``ignore_index``
    hands over duplicate labels, which would silently turn an index-aligned mask
    into a cross product.
    """
    return prs.reset_index(drop=True)


def _is_ai(login: object, patterns: tuple[str, ...] | list[str]) -> bool:
    name = str(login or "").strip().lower()
    return any(fnmatch.fnmatch(name, str(p).lower()) for p in patterns)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """A numeric column, or an all-missing one when the fetch never carried it.

    Missing stays missing on purpose: ``fillna(0)`` here would turn a throttled
    response into a team that shipped zero lines.
    """
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _authors(frame: pd.DataFrame) -> pd.Series:
    if "author" not in frame.columns:
        return pd.Series("unknown", index=frame.index, dtype="object")
    return frame["author"].fillna("unknown").astype(str)


def _merged_mask(prs: pd.DataFrame) -> pd.Series:
    """Which rows are merged PRs, from ``state`` when present, ``merged_at`` if not."""
    if "state" in prs.columns:
        state = prs["state"].fillna("").astype(str).str.upper()
        if state.str.strip().any():
            return state == "MERGED"
    if "merged_at" in prs.columns:
        return prs["merged_at"].notna()
    return pd.Series(False, index=prs.index)


# --------------------------------------------------------------------------- #
# Size
# --------------------------------------------------------------------------- #


def size_band(changed_lines: float | None) -> str:
    """The band a diff of this size falls in; empty string when size is unknown."""
    if changed_lines is None or pd.isna(changed_lines):
        return ""
    lines = float(changed_lines)
    if lines < TRIVIAL_MAX:
        return "trivial"
    if lines < SMALL_MAX:
        return "small"
    if lines < MEDIUM_MAX:
        return "medium"
    if lines <= LARGE_MAX:
        return "large"
    return "oversized"


def classify_sizes(prs: pd.DataFrame) -> pd.DataFrame:
    """Per-PR size, with the band. Rows whose size was never fetched band as ``""``."""
    columns = ["author", "number", "url", "title", "changed_lines", "changed_files", "size_band"]
    if prs.empty:
        return _empty(columns)
    prs = _rows(prs)
    lines = _numeric(prs, "changed_lines")
    out = pd.DataFrame(
        {
            "author": _authors(prs),
            "number": prs.get("number", pd.Series(pd.NA, index=prs.index)),
            "url": prs.get("url", pd.Series("", index=prs.index)),
            "title": prs.get("title", pd.Series("", index=prs.index)),
            "changed_lines": lines,
            "changed_files": _numeric(prs, "changed_files"),
        }
    )
    out["size_band"] = [size_band(value) for value in lines]
    return out


def size_bands(prs: pd.DataFrame) -> pd.DataFrame:
    """Per person: how many PRs in each size band, and their median size.

    This is what a raw PR count cannot show. Splitting one change into five
    small PRs to make the count look busy shows up here as a spike of trivial
    PRs against an unchanged median; a person who only ever ships oversized PRs
    shows up as work nobody can review properly. Both are conversations, not
    conclusions - a run of trivial PRs is exactly what a week of config and
    copy fixes looks like.

    ``unsized`` counts PRs whose diff size the API never returned, so a small
    band count can be read as "few" rather than "few that we know of".
    """
    columns = ["author", "prs", "unsized", *SIZE_BANDS, "median_changed_lines", "trivial_share"]
    if prs.empty:
        return _empty(columns)
    detail = classify_sizes(prs)
    grouped = detail.groupby("author", dropna=False)
    out = pd.DataFrame({"prs": grouped.size()})
    for band in SIZE_BANDS:
        out[band] = grouped["size_band"].apply(lambda s, b=band: int((s == b).sum()))
    out["unsized"] = grouped["changed_lines"].apply(lambda s: int(s.isna().sum()))
    out["median_changed_lines"] = grouped["changed_lines"].median()
    sized = out["prs"] - out["unsized"]
    # Share of *sized* PRs, so a throttled fetch reads as a smaller sample and
    # not as a person who suddenly stopped splitting their work.
    out["trivial_share"] = (out["trivial"] / sized.where(sized > 0)).astype("Float64")
    out = out.reset_index()
    return out[columns].sort_values(["trivial", "prs"], ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# What the AI reviewer found
# --------------------------------------------------------------------------- #


def devin_findings(
    prs: pd.DataFrame, reviewer_patterns: tuple[str, ...] | list[str] = AI_REVIEWER_PATTERNS
) -> pd.DataFrame:
    """Per PR: how many times the AI reviewer weighed in, and how it ruled.

    The CEO's complaint is that PRs "usually have so many issues". This is the
    nearest thing to a direct measurement of that, and until now it was not even
    fetched: the AI reviewer's verdicts arrive as ordinary GitHub reviews, so
    counting the CHANGES_REQUESTED ones per PR says how much was wrong with the
    work before a human spent time on it.

    Read it with two caveats. An AI reviewer's finding is not automatically a
    defect - it comments on style and on things a human would wave through - so
    the useful signal is a person's rate against the team's, not their absolute
    count. And ``ai_reviews`` of zero can mean a clean PR or a PR the reviewer
    never ran on; ``reviews_fetched`` says which frames could answer at all.
    """
    columns = [
        "author",
        "number",
        "url",
        "reviews_fetched",
        "ai_reviews",
        "ai_changes_requested",
        "ai_commented",
        "ai_approved",
        "review_threads",
        "comments",
    ]
    if prs.empty:
        return _empty(columns)
    prs = _rows(prs)
    if "reviews" not in prs.columns:
        # Lean payload: say so once, per PR, instead of reporting zeros.
        out = pd.DataFrame(
            {
                "author": _authors(prs),
                "number": prs.get("number", pd.Series(pd.NA, index=prs.index)),
                "url": prs.get("url", pd.Series("", index=prs.index)),
                "reviews_fetched": False,
            }
        )
        for column in ("ai_reviews", "ai_changes_requested", "ai_commented", "ai_approved"):
            out[column] = pd.Series(pd.NA, index=prs.index, dtype="Int64")
        out["review_threads"] = _numeric(prs, "review_threads")
        out["comments"] = _numeric(prs, "comments")
        return out[columns].reset_index(drop=True)

    rows = []
    for _, pr in prs.iterrows():
        reviews = pr.get("reviews")
        fetched = isinstance(reviews, list)
        ai = [r for r in reviews or [] if _is_ai(r.get("reviewer"), reviewer_patterns)]
        states = [str(r.get("state") or "").upper() for r in ai]
        rows.append(
            {
                "author": str(pr.get("author") or "unknown"),
                "number": pr.get("number"),
                "url": pr.get("url") or "",
                "reviews_fetched": fetched,
                "ai_reviews": len(ai) if fetched else pd.NA,
                "ai_changes_requested": states.count(_CHANGES_REQUESTED) if fetched else pd.NA,
                "ai_commented": states.count(_COMMENTED) if fetched else pd.NA,
                "ai_approved": states.count(_APPROVED) if fetched else pd.NA,
                "review_threads": pr.get("review_threads"),
                "comments": pr.get("comments"),
            }
        )
    out = pd.DataFrame(rows)
    for column in ("ai_reviews", "ai_changes_requested", "ai_commented", "ai_approved"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    for column in ("review_threads", "comments"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[columns]


def devin_findings_by_author(
    prs: pd.DataFrame, reviewer_patterns: tuple[str, ...] | list[str] = AI_REVIEWER_PATTERNS
) -> pd.DataFrame:
    """Per person: the share of their PRs the AI reviewer asked for changes on.

    A rate, not a total, so someone who ships more work is not automatically the
    worst engineer on the page. ``prs_judged`` is the denominator that actually
    had reviews fetched; if it is much smaller than the person's PR count, the
    rate is a sample and should be read as one.
    """
    columns = [
        "author",
        "prs_judged",
        "prs_ai_reviewed",
        "prs_changes_requested",
        "changes_requested_share",
        "ai_findings",
        "median_review_threads",
    ]
    if prs.empty:
        return _empty(columns)
    detail = devin_findings(prs, reviewer_patterns)
    judged = detail[detail["reviews_fetched"].fillna(False).astype(bool)]
    if judged.empty:
        return _empty(columns)
    grouped = judged.groupby("author", dropna=False)
    out = pd.DataFrame(
        {
            "prs_judged": grouped.size(),
            "prs_ai_reviewed": grouped["ai_reviews"].apply(lambda s: int((s.fillna(0) > 0).sum())),
            "prs_changes_requested": grouped["ai_changes_requested"].apply(
                lambda s: int((s.fillna(0) > 0).sum())
            ),
            "ai_findings": grouped["ai_changes_requested"].sum(),
            "median_review_threads": grouped["review_threads"].median(),
        }
    )
    out["changes_requested_share"] = out["prs_changes_requested"] / out["prs_judged"]
    out = out.reset_index()
    return out[columns].sort_values(
        ["changes_requested_share", "prs_judged"], ascending=False, ignore_index=True
    )


# --------------------------------------------------------------------------- #
# Reviewing as work
# --------------------------------------------------------------------------- #


def _review_events(
    prs: pd.DataFrame, reviewer_patterns: tuple[str, ...] | list[str]
) -> pd.DataFrame:
    """One row per review, with who wrote it, on whose PR, and how late it was.

    The clock starts when the PR asked for a reviewer (``review_ready_at``) and
    falls back to when it was opened, because a PR that sat in draft for a week
    was not waiting on anybody.
    """
    rows = []
    prs = _rows(prs)
    if "reviews" not in prs.columns:
        return pd.DataFrame(
            columns=[
                "reviewer",
                "author",
                "number",
                "url",
                "state",
                "submitted_at",
                "body",
                "is_ai",
                "is_self",
                "hours_to_review",
            ]
        )
    for _, pr in prs.iterrows():
        reviews = pr.get("reviews")
        if not isinstance(reviews, list):
            continue
        author = str(pr.get("author") or "unknown")
        opened = pr.get("review_ready_at")
        if opened is None or pd.isna(opened):
            opened = pr.get("created_at")
        for review in reviews:
            submitted = pd.to_datetime(review.get("submitted_at"), utc=True, errors="coerce")
            hours = pd.NA
            if pd.notna(submitted) and opened is not None and pd.notna(opened):
                delta = (submitted - pd.Timestamp(opened)).total_seconds() / 3600.0
                # A review stamped before the PR asked for one (a re-request
                # after the fact) is not a negative wait; it is unmeasurable.
                hours = delta if delta >= 0 else pd.NA
            reviewer = str(review.get("reviewer") or "unknown")
            rows.append(
                {
                    "reviewer": reviewer,
                    "author": author,
                    "number": pr.get("number"),
                    "url": pr.get("url") or "",
                    "state": str(review.get("state") or "").upper(),
                    "submitted_at": submitted,
                    "body": str(review.get("body") or ""),
                    "is_ai": _is_ai(reviewer, reviewer_patterns),
                    "is_self": reviewer == author,
                    "hours_to_review": hours,
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["hours_to_review"] = pd.to_numeric(frame["hours_to_review"], errors="coerce")
    return frame


def review_citizenship(
    prs: pd.DataFrame,
    reviewer_patterns: tuple[str, ...] | list[str] = AI_REVIEWER_PATTERNS,
    include_ai: bool = False,
) -> pd.DataFrame:
    """Per person: reviews given, people reviewed, and how fast they respond.

    Reviewing is work, and it is the work that keeps everyone else unblocked. It
    is also entirely unmeasured today, which makes never reviewing anybody the
    rational choice for an hourly contractor: it earns nothing and costs time.
    Putting it on the page is most of the fix.

    ``distinct_authors_reviewed`` matters as much as the count: twenty reviews
    spread over one colleague is a different behaviour from twenty spread over
    six. ``median_hours_to_first_review`` counts only the reviews where this
    person was first on the PR - being second says nothing about response time.

    The AI reviewer is excluded by default; it reviews everything instantly and
    would otherwise sit at the top of a leaderboard meant for humans. Self
    reviews are always excluded.
    """
    columns = [
        "reviewer",
        "reviews_given",
        "prs_reviewed",
        "distinct_authors_reviewed",
        "approvals_given",
        "changes_requested_given",
        "median_hours_to_first_review",
    ]
    if prs.empty:
        return _empty(columns)
    events = _review_events(prs, reviewer_patterns)
    if events.empty:
        return _empty(columns)
    events = events[~events["is_self"]]
    if not include_ai:
        events = events[~events["is_ai"]]
    if events.empty:
        return _empty(columns)

    # Who got there first on each PR, among the reviews being counted.
    first = (
        events.dropna(subset=["submitted_at"])
        .sort_values("submitted_at")
        .drop_duplicates(subset=["number", "url"], keep="first")
    )
    first_hours = first.groupby("reviewer", dropna=False)["hours_to_review"].median()

    grouped = events.groupby("reviewer", dropna=False)
    out = pd.DataFrame(
        {
            "reviews_given": grouped.size(),
            "prs_reviewed": grouped["url"].nunique(),
            "distinct_authors_reviewed": grouped["author"].nunique(),
            "approvals_given": grouped["state"].apply(lambda s: int((s == _APPROVED).sum())),
            "changes_requested_given": grouped["state"].apply(
                lambda s: int((s == _CHANGES_REQUESTED).sum())
            ),
        }
    )
    out["median_hours_to_first_review"] = first_hours
    out = out.reset_index()
    return out[columns].sort_values(
        ["reviews_given", "distinct_authors_reviewed"], ascending=False, ignore_index=True
    )


class Reciprocity(NamedTuple):
    """``pairs`` (who reviews whom) and ``by_person`` (how concentrated that is)."""

    pairs: pd.DataFrame
    by_person: pd.DataFrame


def reciprocity(
    prs: pd.DataFrame,
    reviewer_patterns: tuple[str, ...] | list[str] = AI_REVIEWER_PATTERNS,
) -> Reciprocity:
    """Who reviews whom, and who only ever reviews one person.

    A closed approval loop - two people who review each other and nobody else,
    approving without comment - is what a review process looks like when it has
    stopped being a review process. This finds the shape of that: the pair
    matrix, whether each pair goes both ways, and per person how concentrated
    their reviewing is on one colleague.

    Read this as a prompt to look, never as proof. On a team of four, everybody
    reviewing everybody still produces a high concentration figure, and two
    people who own the same subsystem *should* review each other more than they
    review anyone else. The honest reading is: pull up these PRs and see whether
    the approvals have anything in them.

    ``rubber_stamp_approvals`` counts approvals whose review body is empty and
    which left no review thread on the PR. That is a weak signal on its own - a
    genuinely clean small PR is approved exactly this way - so it is worth
    something only next to the size of what was approved.
    """
    pair_columns = ["reviewer", "author", "reviews", "approvals", "share_of_reviewer", "mutual"]
    person_columns = [
        "reviewer",
        "reviews_given",
        "top_partner",
        "top_partner_share",
        "concentration",
        "approvals_given",
        "rubber_stamp_approvals",
        "rubber_stamp_share",
    ]
    if prs.empty:
        return Reciprocity(_empty(pair_columns), _empty(person_columns))
    prs = _rows(prs)
    events = _review_events(prs, reviewer_patterns)
    if events.empty:
        return Reciprocity(_empty(pair_columns), _empty(person_columns))
    events = events[~events["is_self"] & ~events["is_ai"]]
    if events.empty:
        return Reciprocity(_empty(pair_columns), _empty(person_columns))

    # An approval with nothing written on it, on a PR that drew no review
    # threads at all. Where thread counts were not fetched, an empty body alone
    # has to do, and the column is that much softer.
    if {"url", "review_threads"} <= set(prs.columns):
        threads = prs.set_index("url")["review_threads"]
        threads = threads[~threads.index.duplicated(keep="first")]
        pr_threads = events["url"].map(threads)
    else:
        pr_threads = pd.Series(pd.NA, index=events.index)
    events = events.assign(
        rubber_stamp=(events["state"] == _APPROVED)
        & (events["body"].str.strip() == "")
        & (pd.to_numeric(pr_threads, errors="coerce").fillna(0) == 0)
    )

    pairs = (
        events.groupby(["reviewer", "author"], dropna=False)
        .agg(
            reviews=("state", "size"),
            approvals=("state", lambda s: int((s == _APPROVED).sum())),
        )
        .reset_index()
    )
    totals = pairs.groupby("reviewer")["reviews"].sum()
    pairs["share_of_reviewer"] = pairs["reviews"] / pairs["reviewer"].map(totals)
    seen = set(zip(pairs["reviewer"], pairs["author"]))
    pairs["mutual"] = [(a, r) in seen for r, a in zip(pairs["reviewer"], pairs["author"])]
    pairs = pairs.sort_values(["reviews", "share_of_reviewer"], ascending=False, ignore_index=True)

    top = pairs.sort_values("reviews", ascending=False).drop_duplicates("reviewer", keep="first")
    grouped = events.groupby("reviewer", dropna=False)
    by_person = pd.DataFrame(
        {
            "reviews_given": grouped.size(),
            "approvals_given": grouped["state"].apply(lambda s: int((s == _APPROVED).sum())),
            "rubber_stamp_approvals": grouped["rubber_stamp"].sum().astype(int),
        }
    ).reset_index()
    by_person = by_person.merge(
        top[["reviewer", "author", "share_of_reviewer"]].rename(
            columns={"author": "top_partner", "share_of_reviewer": "top_partner_share"}
        ),
        on="reviewer",
        how="left",
    )
    # Herfindahl over the reviewer's targets: 1.0 is one colleague and nobody
    # else, 1/n is reviewing n people evenly.
    concentration = (
        pairs.assign(square=pairs["share_of_reviewer"] ** 2).groupby("reviewer")["square"].sum()
    )
    by_person["concentration"] = by_person["reviewer"].map(concentration)
    by_person["rubber_stamp_share"] = (
        by_person["rubber_stamp_approvals"]
        / by_person["approvals_given"].where(by_person["approvals_given"] > 0)
    ).astype("Float64")
    by_person = by_person[person_columns].sort_values(
        ["concentration", "reviews_given"], ascending=False, ignore_index=True
    )
    return Reciprocity(pairs, by_person)


def unprompted_reviews(
    prs: pd.DataFrame,
    reviewer_patterns: tuple[str, ...] | list[str] = AI_REVIEWER_PATTERNS,
) -> pd.DataFrame:
    """Per person: reviews given that nobody asked for, with the PRs as evidence.

    DEVIN_PLAN §6 calls this "the proactivity signal - the rest is
    compliance": ``review_citizenship`` counts reviews given, but a review
    given because a person was assigned one and a review given because
    someone went looking for work to unblock are the same row in that table.
    A review counts as unprompted here when nothing in the PR's timeline
    asked *this* reviewer for one at or before the moment they submitted it.

    Ordering is the whole signal, and it runs one direction only: a review
    request logged *after* the review still leaves the review unprompted - it
    was volunteered before anyone asked, and a request filed afterwards (a
    maintainer formalising it, a re-request for the record) does not undo
    that. A request at or before the review is what makes a review
    compliance instead.

    Read alongside :func:`reciprocity` on purpose - ``top_partner_share`` and
    ``concentration`` are joined onto the per-person result, because ten
    unprompted reviews aimed entirely at one colleague is the closed
    approval loop reciprocity already looks for, not proactivity, and the
    two should never be read apart.

    Evidence travels twice: ``prs`` is the distinct PR numbers, and ``pr_refs``
    the same PRs as ``(number, url)`` pairs. A PR number is unique only within
    one repository, so ``pr_refs`` is what a link should be built from; ``prs``
    is for counting and for display where the repo is already established.

    Blind spot: this needs the extended timeline payload
    (``timeline_events``/``extended_fetched`` from :mod:`github_client`'s
    extended query). A PR fetched on the lean or detail payload contributes
    nothing here - not zero unprompted reviews, no evidence at all - and a
    request made outside GitHub (Slack, standup, "can you look at this")
    cannot be seen either, so this undercounts compliance, never proactivity.
    """
    columns = [
        "reviewer",
        "unprompted_reviews",
        "prs",
        "pr_refs",
        "top_partner",
        "top_partner_share",
        "concentration",
    ]
    if prs.empty:
        return _empty(columns)
    prs = _rows(prs)
    if "timeline_events" not in prs.columns:
        return _empty(columns)

    rows = []
    for _, row in prs.iterrows():
        events = row.get("timeline_events")
        reviews = row.get("reviews")
        if not isinstance(events, list) or not isinstance(reviews, list):
            continue
        author = str(row.get("author") or "unknown")
        requested_at: dict[str, list[pd.Timestamp]] = {}
        for event in events:
            if event.get("type") != "review_requested":
                continue
            reviewer = event.get("requested_reviewer")
            at = pd.to_datetime(event.get("created_at"), utc=True, errors="coerce")
            if not reviewer or pd.isna(at):
                continue
            requested_at.setdefault(str(reviewer), []).append(at)
        for review in reviews:
            reviewer = str(review.get("reviewer") or "unknown")
            if reviewer == author or _is_ai(reviewer, reviewer_patterns):
                continue
            submitted = pd.to_datetime(review.get("submitted_at"), utc=True, errors="coerce")
            if pd.isna(submitted):
                continue
            # Prompted only by a request at or before the review; one logged
            # afterwards does not retroactively make the review compliance.
            asked = requested_at.get(reviewer, [])
            prompted = any(at <= submitted for at in asked)
            if not prompted:
                rows.append(
                    {
                        "reviewer": reviewer,
                        "number": row.get("number"),
                        "url": row.get("url"),
                    }
                )
    if not rows:
        return _empty(columns)

    detail = pd.DataFrame(rows)
    grouped = detail.groupby("reviewer", dropna=False)
    out = pd.DataFrame(
        {
            "unprompted_reviews": grouped.size(),
            "prs": grouped["number"].apply(
                lambda s: tuple(sorted({int(n) for n in s if pd.notna(n)}))
            ),
        }
    ).reset_index()
    # ``pr_refs`` is the same evidence keyed the way GitHub actually identifies a
    # PR. ``prs`` collapses on the number alone, which is unique only inside one
    # repository: two repos both having a #42 is ordinary, and a reader following
    # a bare number can land on the wrong PR entirely. Carrying the URL from the
    # row the review was found on removes the guess - nothing downstream has to
    # resolve a number back to a URL, so nothing downstream can resolve it wrong.
    refs = {
        reviewer: tuple(
            sorted(
                {
                    (int(number), str(url))
                    for number, url in zip(group["number"], group["url"])
                    if pd.notna(number) and url
                }
            )
        )
        for reviewer, group in detail.groupby("reviewer", dropna=False)
    }
    out["pr_refs"] = out["reviewer"].map(refs)

    _, by_person = reciprocity(prs, reviewer_patterns)
    out = out.merge(
        by_person[["reviewer", "top_partner", "top_partner_share", "concentration"]],
        on="reviewer",
        how="left",
    )
    return out[columns].sort_values("unprompted_reviews", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# How work got merged
# --------------------------------------------------------------------------- #


def flag_self_merges(
    prs: pd.DataFrame,
    trunk_branches: frozenset[str] | set[str] = TRUNK_BRANCHES,
    reviewer_patterns: tuple[str, ...] | list[str] = AI_REVIEWER_PATTERNS,
) -> pd.DataFrame:
    """Per merged PR: who merged it, whether anyone else approved, where it landed."""
    columns = [
        "author",
        "number",
        "url",
        "merged_by",
        "base_branch",
        "is_self_merge",
        "outside_approval",
        "ai_only_approval",
        "merged_off_trunk",
    ]
    if prs.empty:
        return _empty(columns)
    prs = _rows(prs)
    merged = prs[_merged_mask(prs)]
    if merged.empty:
        return _empty(columns)

    rows = []
    for _, pr in merged.iterrows():
        author = str(pr.get("author") or "unknown")
        merged_by = pr.get("merged_by")
        reviews = pr.get("reviews")
        if isinstance(reviews, list):
            approvals = [
                r for r in reviews if str(r.get("state") or "").upper() == _APPROVED
            ]
            human = [
                r
                for r in approvals
                if str(r.get("reviewer") or "") != author
                and not _is_ai(r.get("reviewer"), reviewer_patterns)
            ]
            outside = bool(human)
            ai_only = bool(approvals) and not human
        else:
            # Lean payload. GitHub does not let an author approve their own PR,
            # so an approving-review count above zero is somebody else - but it
            # cannot tell a human approval from the AI reviewer's.
            count = pd.to_numeric(pd.Series([pr.get("approving_reviews")]), errors="coerce").iloc[0]
            outside = bool(count and count > 0) if pd.notna(count) else pd.NA
            ai_only = pd.NA
        base = str(pr.get("base_branch") or "")
        rows.append(
            {
                "author": author,
                "number": pr.get("number"),
                "url": pr.get("url") or "",
                "merged_by": merged_by,
                "base_branch": base,
                "is_self_merge": (str(merged_by) == author) if merged_by else pd.NA,
                "outside_approval": outside,
                "ai_only_approval": ai_only,
                "merged_off_trunk": (base.lower() not in trunk_branches) if base else pd.NA,
            }
        )
    return pd.DataFrame(rows)[columns]


def self_merge(
    prs: pd.DataFrame,
    trunk_branches: frozenset[str] | set[str] = TRUNK_BRANCHES,
    reviewer_patterns: tuple[str, ...] | list[str] = AI_REVIEWER_PATTERNS,
) -> pd.DataFrame:
    """Per person: how much of their merged work merged itself.

    Two different failures, kept apart on purpose. Pressing merge on your own PR
    after a colleague approved it is normal and often correct. Merging your own
    PR that nobody approved is the review process not happening, and it is the
    one worth a branch-protection rule rather than a conversation.

    ``merged_off_trunk`` catches the variant where a PR is merged into the
    author's own feature branch: the merge count goes up and nothing shipped.

    ``unknown_approval`` counts merged PRs whose reviews were not fetched, so a
    zero in the unapproved column can be told apart from an unanswered question.
    """
    columns = [
        "author",
        "merged_prs",
        "self_merged",
        "self_merge_share",
        "merged_without_outside_approval",
        "unapproved_share",
        "ai_only_approval",
        "merged_off_trunk",
        "unknown_approval",
    ]
    if prs.empty:
        return _empty(columns)
    detail = flag_self_merges(prs, trunk_branches, reviewer_patterns)
    if detail.empty:
        return _empty(columns)
    grouped = detail.groupby("author", dropna=False)

    def _true(series: pd.Series) -> int:
        return int(series.apply(lambda v: v is True).sum())

    out = pd.DataFrame(
        {
            "merged_prs": grouped.size(),
            "self_merged": grouped["is_self_merge"].apply(_true),
            "merged_without_outside_approval": grouped["outside_approval"].apply(
                lambda s: int(s.apply(lambda v: v is False).sum())
            ),
            "ai_only_approval": grouped["ai_only_approval"].apply(_true),
            "merged_off_trunk": grouped["merged_off_trunk"].apply(_true),
            "unknown_approval": grouped["outside_approval"].apply(
                lambda s: int(s.apply(lambda v: v is not True and v is not False).sum())
            ),
        }
    )
    out["self_merge_share"] = out["self_merged"] / out["merged_prs"]
    out["unapproved_share"] = out["merged_without_outside_approval"] / out["merged_prs"]
    out = out.reset_index()
    return out[columns].sort_values(
        ["merged_without_outside_approval", "self_merged"], ascending=False, ignore_index=True
    )


def abandoned_rate(prs: pd.DataFrame) -> pd.DataFrame:
    """Per person: closed-without-merging as a share of everything they closed.

    Feed this the merged frame and the closed-unmerged frame concatenated;
    still-open PRs are ignored, because a PR that has not been decided yet is
    not abandoned. An abandoned PR costs the same hours as a shipped one and
    leaves no trace in any count on the page today.

    A high rate is a question, not a fault: superseded work, spikes and
    duplicates all close unmerged for good reasons. What it reliably catches is
    a person whose hours keep landing in branches that never ship.
    """
    columns = ["author", "closed_prs", "merged", "abandoned", "abandoned_rate"]
    if prs.empty:
        return _empty(columns)
    prs = _rows(prs)
    merged = _merged_mask(prs)
    if "state" in prs.columns and prs["state"].fillna("").astype(str).str.strip().any():
        closed_unmerged = prs["state"].fillna("").astype(str).str.upper() == "CLOSED"
    else:
        # No state column: a row with a close time and no merge time is one.
        closed = (
            prs["closed_at"].notna()
            if "closed_at" in prs.columns
            else pd.Series(False, index=prs.index)
        )
        closed_unmerged = closed & ~merged
    decided = prs[merged | closed_unmerged]
    if decided.empty:
        return _empty(columns)
    frame = pd.DataFrame(
        {
            "author": _authors(decided),
            "merged": merged[decided.index].astype(int),
            "abandoned": closed_unmerged[decided.index].astype(int),
        }
    )
    out = frame.groupby("author", dropna=False).agg(
        closed_prs=("merged", "size"), merged=("merged", "sum"), abandoned=("abandoned", "sum")
    )
    out["abandoned_rate"] = out["abandoned"] / out["closed_prs"]
    out = out.reset_index()
    return out[columns].sort_values(
        ["abandoned_rate", "abandoned"], ascending=False, ignore_index=True
    )


def traceability(
    prs: pd.DataFrame, project_keys: list[str] | None = None
) -> pd.DataFrame:
    """Per person: the share of their MERGED PRs that name a Jira ticket.

    The open-PR view already asks this. Merged work never has been asked, which
    is the half that matters: an untraceable merged PR is work that shipped with
    no ticket behind it, so the hours billed against it cannot be checked
    against anything.

    Key matching is :mod:`pr_hygiene`'s, imported rather than repeated, so the
    two views cannot drift into disagreeing about what a Jira key looks like.

    ``judgeable`` is the honest denominator. The merged fetch does not carry the
    branch or the body unless it was asked to (``hygiene=True``), and a PR whose
    key was never looked for must not count as a PR that failed to name one; a
    key visible in the title still counts, since finding one is proof either way.
    """
    columns = ["author", "merged_prs", "judgeable", "with_key", "traceability", "not_judgeable"]
    if prs.empty:
        return _empty(columns)
    prs = _rows(prs)
    merged = prs[_merged_mask(prs)]
    if merged.empty:
        return _empty(columns)
    keyed = (
        merged
        if "has_jira_key" in merged.columns
        else pr_hygiene.add_hygiene_fields(merged, project_keys)
    )
    has_key = keyed["has_jira_key"].fillna(False).astype(bool)
    if "hygiene_fetched" in keyed.columns:
        looked = keyed["hygiene_fetched"].fillna(False).astype(bool) | has_key
    else:
        looked = pd.Series(True, index=keyed.index)
    frame = pd.DataFrame(
        {
            "author": _authors(keyed),
            "judgeable": looked.astype(int),
            "with_key": (has_key & looked).astype(int),
        }
    )
    out = frame.groupby("author", dropna=False).agg(
        merged_prs=("judgeable", "size"),
        judgeable=("judgeable", "sum"),
        with_key=("with_key", "sum"),
    )
    out["not_judgeable"] = out["merged_prs"] - out["judgeable"]
    out["traceability"] = (
        out["with_key"] / out["judgeable"].where(out["judgeable"] > 0)
    ).astype("Float64")
    out = out.reset_index()
    return out[columns].sort_values(
        ["traceability", "merged_prs"], ascending=[True, False], ignore_index=True
    )


# --------------------------------------------------------------------------- #
# Hiding in draft
# --------------------------------------------------------------------------- #


class DraftTransitions(NamedTuple):
    """Per-PR draft round-trips, and what became of the ones that look deliberate.

    ``detail`` is one row per PR: how many times it went back into draft
    (``draft_round_trips``), and whether that happened *after* a reviewer was
    asked for (``after_review_request``) - a PR opened as a draft and marked
    ready once has neither, and is not in this file for anything but a normal
    open. ``outcome`` runs :func:`abandoned_rate` on exactly the PRs flagged
    ``after_review_request``, so "of the PRs that hid in draft after being
    asked for review, how many actually shipped" reuses the real
    abandoned-work arithmetic rather than a second one invented here.
    """

    detail: pd.DataFrame
    outcome: pd.DataFrame


def draft_transitions(prs: pd.DataFrame) -> DraftTransitions:
    """KPI_SPEC exploit #6, made visible: hiding a PR in draft after asking for review.

    ``_open_query`` carries ``draft:false``, so one click removes a PR from
    Open, Stuck, Never-reviewed and every hygiene tab - and the removal is
    silent, because a draft PR simply stops matching the query rather than
    appearing anywhere flagged. This counts every draft round trip per PR
    from the timeline and isolates the ones where the PR went to draft
    *after* someone was already asked to review it - ordering is the whole
    signal named in KPI_SPEC: a PR that starts life as a draft is an
    engineer working in the open, and is not this at all.

    Innocent readings exist for the flagged case too - a reviewer found a
    real problem and the author correctly pulled the PR back to fix it - so
    this is a queue to open, not a verdict. What it removes is the silence:
    today that PR vanishes from every view instead of turning up in one.

    Blind spot: needs the extended timeline payload
    (``timeline_events``/``extended_fetched``). A PR fetched on the lean or
    detail payload is silently absent from both frames, not silently "no
    round trips" - there is no column here that would let the two be told
    apart, which is itself worth knowing before reading a quiet page as a
    clean one.
    """
    detail_columns = ["author", "number", "url", "draft_round_trips", "after_review_request"]
    empty_outcome = abandoned_rate(pd.DataFrame())
    if prs.empty:
        return DraftTransitions(_empty(detail_columns), empty_outcome)
    prs = _rows(prs)
    if "timeline_events" not in prs.columns:
        return DraftTransitions(_empty(detail_columns), empty_outcome)

    rows = []
    flagged_positions = []
    for idx, row in prs.iterrows():
        events = row.get("timeline_events")
        if not isinstance(events, list):
            continue
        draft_times = sorted(
            t
            for t in (
                pd.to_datetime(e.get("created_at"), utc=True, errors="coerce")
                for e in events
                if e.get("type") == "converted_to_draft"
            )
            if pd.notna(t)
        )
        request_times = sorted(
            t
            for t in (
                pd.to_datetime(e.get("created_at"), utc=True, errors="coerce")
                for e in events
                if e.get("type") == "review_requested"
            )
            if pd.notna(t)
        )
        after_request = any(d > r for d in draft_times for r in request_times)
        rows.append(
            {
                "author": str(row.get("author") or "unknown"),
                "number": row.get("number"),
                "url": row.get("url") or "",
                "draft_round_trips": len(draft_times),
                "after_review_request": after_request,
            }
        )
        if after_request:
            flagged_positions.append(idx)

    if not rows:
        return DraftTransitions(_empty(detail_columns), empty_outcome)
    detail = pd.DataFrame(rows)[detail_columns]
    outcome = abandoned_rate(prs.loc[flagged_positions]) if flagged_positions else empty_outcome
    return DraftTransitions(detail, outcome)


# --------------------------------------------------------------------------- #
# Size in points
# --------------------------------------------------------------------------- #


class DeliveredPoints(NamedTuple):
    """One person's size-weighted output - count and median size, bound together.

    KPI_SPEC §3.3 is explicit: "Report count and median size together,
    always. A count alone is exploit #3" - five trivial PRs beating one real
    one on every raw count the dashboard has. A ``NamedTuple`` with no
    default on ``median_changed_lines`` makes a count-only reading a
    ``TypeError`` at construction, not a value some later caller can quietly
    drop: there is no second, scalar-returning function anywhere in this
    module that hands back ``points`` or ``prs`` alone.

    Blind to the same thing :func:`size_bands` is blind to: diff size
    measures typing, not thinking or difficulty. A high score here is a
    volume of well-scoped, mergeable work; it is not evidence the work was
    hard, or good.
    """

    author: str
    prs: int
    median_changed_lines: float
    points: float
    trivial_share: float


def delivered_points(prs: pd.DataFrame) -> list[DeliveredPoints]:
    """Per person: size-band counts converted into points, via :func:`size_bands`.

    Built directly on :func:`size_bands` rather than reclassifying diffs a
    second time, so the two can never disagree about which band a PR falls
    in. ``SIZE_POINTS`` weights the bands so splitting one change into five
    trivial PRs (exploit #3) scores below shipping it as one - five trivial
    PRs earn ``5 * SIZE_POINTS["trivial"]``, well under one medium PR's
    ``SIZE_POINTS["medium"]`` for a fraction of the real work, and nowhere
    close to what the same total size shipped as one PR would earn.

    Returns a plain list, not a frame: a ``DataFrame`` would let a caller
    select the ``prs`` column alone and hand it around as if it were the
    complete picture, which is the exact shortcut KPI_SPEC §3.3 rules out.
    Every element is a whole :class:`DeliveredPoints`, count and median
    inseparable.
    """
    bands = size_bands(prs)
    if bands.empty:
        return []
    out = [
        DeliveredPoints(
            author=str(row["author"]),
            prs=int(row["prs"]),
            median_changed_lines=(
                float(row["median_changed_lines"]) if pd.notna(row["median_changed_lines"]) else float("nan")
            ),
            points=float(sum(row[band] * SIZE_POINTS[band] for band in SIZE_BANDS)),
            trivial_share=(
                float(row["trivial_share"]) if pd.notna(row["trivial_share"]) else float("nan")
            ),
        )
        for _, row in bands.iterrows()
    ]
    out.sort(key=lambda d: d.points, reverse=True)
    return out
