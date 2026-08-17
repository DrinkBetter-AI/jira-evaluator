"""What to do about the numbers on the landing page, named one by one.

The attention tiles counted problems - "75 of 80 open PRs have no approving
review" - and a count is not a move. A reader who cannot get from the tile to
the pull request it is about has been informed and left stuck, which is the
complaint this module answers: every tile now has the named work behind it,
each row addressed to a person and carrying its own link.

Deliberately free of Streamlit so the wording and the ranking can be tested
without drawing a page.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Ranked by cost of inaction, so a tile a reader can only half-act on still
# opens with the item that has been waiting longest.


@dataclass(frozen=True)
class Action:
    """One move, addressed to somebody, with the thing it is about linked.

    ``days`` is how long the item has been waiting; it is what the ranking
    reads, and what the row shows, because "waiting 12 days" is the argument
    for doing it before the one waiting an hour.
    """

    kind: str
    verb: str
    subject: str
    url: str
    detail: str
    owner: str
    days: float

    @property
    def sentence(self) -> str:
        """The action as one line a reader can act on without the table."""
        return f"{self.verb} {self.subject} — {self.detail}"


def _days(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce").fillna(0.0)


def _counts(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0, index=frame.index, dtype="int64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)


def _live(open_prs: pd.DataFrame) -> pd.DataFrame:
    """Open PRs minus drafts - a draft has not been handed to a reviewer yet."""
    if open_prs.empty or "is_draft" not in open_prs.columns:
        return open_prs
    return open_prs[~open_prs["is_draft"].fillna(False).astype(bool)]


def _pr_name(row: pd.Series) -> str:
    repo = str(row.get("repo") or "").strip()
    number = row.get("number")
    tag = f"#{int(number)}" if pd.notna(number) else ""
    return f"{repo}{tag}" if repo else (tag or "a pull request")


def review_actions(open_prs: pd.DataFrame) -> list[Action]:
    """The open PRs nobody has approved, oldest wait first.

    Two different moves hide in one tile: a PR with no reviewer requested needs
    somebody named, and a PR whose reviewer has not looked needs chasing. The
    verb says which, because they are asked of different people.
    """
    live = _live(open_prs)
    if live.empty:
        return []

    approvals = _counts(live, "approving_reviews")
    reviews = _counts(live, "total_reviews")
    requests = _counts(live, "review_requests")
    age = _days(live, "age_days")

    waiting = live[approvals == 0]
    actions: list[Action] = []
    for index, row in waiting.iterrows():
        asked = int(requests.get(index, 0)) > 0
        looked = int(reviews.get(index, 0)) > 0
        author = str(row.get("author") or "unknown")
        days = float(age.get(index, 0.0))
        if not asked:
            verb, detail = (
                "Name a reviewer for",
                f"open {days:.0f}d, {author} asked nobody",
            )
        elif not looked:
            verb, detail = (
                "Chase the reviewer on",
                f"open {days:.0f}d, requested but never looked at",
            )
        else:
            verb, detail = (
                "Get a decision on",
                f"open {days:.0f}d, reviewed but not approved",
            )
        actions.append(
            Action(
                kind="review",
                verb=verb,
                subject=_pr_name(row),
                url=str(row.get("url") or ""),
                detail=detail,
                owner=author,
                days=days,
            )
        )
    return sorted(actions, key=lambda action: action.days, reverse=True)


def triage_actions(triage: pd.DataFrame, *, url_for) -> list[Action]:
    """Tickets sitting in a triage status: somebody has to say whether they matter."""
    if triage is None or triage.empty:
        return []
    age = _days(triage, "ticket_age_days")
    actions = []
    for index, row in triage.iterrows():
        key = str(row.get("key") or "").strip()
        if not key:
            continue
        days = float(age.get(index, 0.0))
        summary = str(row.get("summary") or "").strip()
        actions.append(
            Action(
                kind="triage",
                verb="Accept or close",
                subject=key,
                url=url_for(key),
                detail=f"{days:.0f}d in triage — {summary[:60]}" if summary else f"{days:.0f}d in triage",
                owner="",
                days=days,
            )
        )
    return sorted(actions, key=lambda action: action.days, reverse=True)


def ownership_actions(ownerless: pd.DataFrame, *, url_for) -> list[Action]:
    """Open tickets belonging to nobody, the oldest first."""
    if ownerless.empty:
        return []
    age = _days(ownerless, "ticket_age_days")
    actions = []
    for index, row in ownerless.iterrows():
        key = str(row.get("key") or "").strip()
        if not key:
            continue
        days = float(age.get(index, 0.0))
        status = str(row.get("status") or "").strip() or "no status"
        actions.append(
            Action(
                kind="ownership",
                verb="Assign an owner to",
                subject=key,
                url=url_for(key),
                detail=f"{days:.0f}d old, in {status}, nobody's score carries it",
                owner="",
                days=days,
            )
        )
    return sorted(actions, key=lambda action: action.days, reverse=True)


def stalled_actions(stalled: pd.DataFrame, *, url_for) -> list[Action]:
    """Owned tickets that have not moved: their owner owes an answer."""
    if stalled.empty:
        return []
    idle = _days(stalled, "idle_days")
    actions = []
    for index, row in stalled.iterrows():
        key = str(row.get("key") or "").strip()
        if not key:
            continue
        owner = str(row.get("assignee") or "").strip()
        days = float(idle.get(index, 0.0))
        actions.append(
            Action(
                kind="stalled",
                verb=("Ask " + owner + " about") if owner else "Find out what happened to",
                subject=key,
                url=url_for(key),
                detail=f"no movement in {days:.0f}d",
                owner=owner,
                days=days,
            )
        )
    return sorted(actions, key=lambda action: action.days, reverse=True)


def rank(queues: dict[str, list[Action]], limit: int = 5) -> list[Action]:
    """The few actions to open the page with, one problem at a time.

    Not simply the longest-waiting items: a 1,200-day unreviewed PR backlog
    would fill every slot and the reader would never hear about untriaged bugs.
    Each queue offers its oldest, then its next, so a short list still spans
    the problems - and within a round the longest wait goes first.
    """
    remaining = {name: list(items) for name, items in queues.items() if items}
    chosen: list[Action] = []
    while remaining and len(chosen) < limit:
        round_heads = sorted(
            ((name, items[0]) for name, items in remaining.items()),
            key=lambda pair: pair[1].days,
            reverse=True,
        )
        for name, head in round_heads:
            if len(chosen) >= limit:
                break
            chosen.append(head)
            remaining[name] = remaining[name][1:]
            if not remaining[name]:
                del remaining[name]
    return chosen


def as_frame(actions: list[Action]) -> pd.DataFrame:
    """The actions as a table, with the link in its own column for st.dataframe."""
    return pd.DataFrame(
        [
            {
                "Do this": action.verb,
                "Item": action.subject,
                "Open": action.url,
                "Why": action.detail,
                "Waiting (days)": round(action.days, 1),
                "Owner": action.owner or "—",
            }
            for action in actions
        ]
    )
