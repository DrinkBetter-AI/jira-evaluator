"""Where the orphan tickets belong, and which epics have nothing left in them.

Two halves of the same tidy-up. A ticket with no epic is invisible in planning:
it belongs to no piece of work anyone is tracking. Rather than list the orphans
and leave the filing to a person, this suggests a parent for each one from the
epics that already exist, and says which words earned the suggestion so the
guess can be judged in a second rather than trusted.

Nothing here writes to Jira. Every suggestion is a suggestion.
"""

from __future__ import annotations

import math
import re

import pandas as pd

from epics import EXCLUDED_ISSUE_TYPES, NO_EPIC


# How much of what a ticket is about the epic has to account for. A share rather
# than a raw total, because a raw total rises with the number of epics in the
# instance and would quietly loosen as the backlog grows.
MIN_CONFIDENCE = 0.34
MAX_REASON_WORDS = 4

# Words that appear across every epic in a Jira instance and so carry no signal
# about which epic a ticket belongs to.
_STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have how if in into is it its
    of on or that the their then there these this to was were what when which
    who will with
    add added adding all any app bug change changes check create creating
    current data ensure error fix fixed fixing implement improve issue issues
    make missing new not now page prod production remove report set should
    show support task tests update updated updating use user users via work
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'&/-]*")


def _words(text: str) -> set[str]:
    """Distinctive lowercase words in a summary, without the boilerplate."""
    found = _WORD_RE.findall(str(text or "").lower())
    return {word for word in found if len(word) > 2 and word not in _STOPWORDS}


def _open_children(df: pd.DataFrame) -> pd.DataFrame:
    """Tickets that can have an epic: no epics, initiatives or sub-tasks."""
    if "issue_type" not in df.columns:
        return df
    kinds = (
        df["issue_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[-_\s]", "", regex=True)
    )
    return df[~kinds.isin(EXCLUDED_ISSUE_TYPES)]


def _epic_vocabulary(children: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    """Each epic's words: its own name, plus the summaries of what is in it.

    A child's summary is the better evidence of the two. "Checkout" may never
    appear in an epic called *Merchant Payments*, but it will appear in half the
    tickets already filed under it, and a new checkout ticket belongs there.
    """
    parented = children[children["epic_key"].astype(str).str.strip().ne("")]
    vocabulary: dict[str, dict[str, set[str]]] = {}
    for (epic_key, epic_name), group in parented.groupby(
        ["epic_key", "epic_summary"], dropna=False
    ):
        key = str(epic_key).strip()
        if not key:
            continue
        # A missing parent summary arrives as NaN, which is truthy: named from it
        # directly, the epic would be displayed as the word "nan" and would carry
        # "nan" into its vocabulary as if it were a word the epic is about.
        name = "" if pd.isna(epic_name) else str(epic_name).strip()
        entry = vocabulary.setdefault(key, {"name": name or key, "words": set()})
        if name:
            entry["name"] = name
            entry["words"] |= _words(name)
        for summary in group["summary"]:
            entry["words"] |= _words(summary)
    return vocabulary


def _weights(vocabulary: dict[str, dict[str, set[str]]]) -> dict[str, float]:
    """Inverse frequency: a word in every epic tells you nothing about which one."""
    total = len(vocabulary) or 1
    counts: dict[str, int] = {}
    for entry in vocabulary.values():
        for word in entry["words"]:
            counts[word] = counts.get(word, 0) + 1
    return {word: math.log(total / count) for word, count in counts.items()}


def suggest_parents(
    df: pd.DataFrame, min_confidence: float = MIN_CONFIDENCE
) -> pd.DataFrame:
    """Every parentless ticket, with the epic its words point at.

    Scored against the epics of its own project only: Jira keys do not cross
    projects, and a confident match in someone else's project is a suggestion
    nobody can act on.
    """
    columns = [
        "key",
        "summary",
        "status",
        "assignee",
        "project_key",
        "suggested_epic",
        "suggested_epic_key",
        "confidence",
        "why",
    ]

    # Typed even when empty: the confidence column is rendered as a number, and
    # an all-object empty frame is a state the caller can reach (every ticket
    # filed, some epic still sitting empty).
    def _blank() -> pd.DataFrame:
        return pd.DataFrame(
            {column: pd.Series(dtype="float" if column == "confidence" else "object")
             for column in columns}
        )
    if df.empty or "epic_key" not in df.columns:
        return _blank()

    children = _open_children(df).copy()
    if children.empty:
        return _blank()
    children["epic_key"] = children["epic_key"].fillna("").astype(str).str.strip()
    orphans = children[children["epic_key"].eq("")]
    if orphans.empty:
        return _blank()

    vocabulary = _epic_vocabulary(children)
    weights = _weights(vocabulary)
    projects = (
        children[children["epic_key"].ne("")]
        .groupby("epic_key")["project_key"]
        .agg(lambda keys: str(keys.iloc[0] or "").strip().upper())
        .to_dict()
    )

    # A word no epic uses is still part of what the ticket is about, and the most
    # distinctive part; unweighted it would count as nothing and let an epic that
    # matches one word out of six look like a complete explanation.
    unseen = math.log(len(vocabulary)) if len(vocabulary) > 1 else 1.0

    rows = []
    for _, ticket in orphans.iterrows():
        words = _words(ticket.get("summary"))
        about = sum(weights.get(word, unseen) for word in words)
        project = str(ticket.get("project_key") or "").strip().upper()
        best_key, best_score, best_shared = "", 0.0, set()
        for epic_key, entry in vocabulary.items():
            if project and projects.get(epic_key, project) != project:
                continue
            shared = words & entry["words"]
            score = sum(weights.get(word, 0.0) for word in shared)
            if score > best_score:
                best_key, best_score, best_shared = epic_key, score, shared
        matched = sorted(best_shared, key=lambda word: -weights.get(word, 0.0))
        rows.append(
            {
                "key": ticket.get("key"),
                "summary": ticket.get("summary"),
                "status": ticket.get("status"),
                "assignee": ticket.get("assignee"),
                "project_key": ticket.get("project_key"),
                "suggested_epic": vocabulary[best_key]["name"] if best_key else "",
                "suggested_epic_key": best_key,
                "confidence": round(best_score / about, 2) if about else 0.0,
                "why": ", ".join(matched[:MAX_REASON_WORDS]),
            }
        )

    out = pd.DataFrame(rows, columns=columns)
    # A weak match is worse than none: it invites a wrong parent to be accepted
    # because the table offered one. Those tickets keep their row and lose the guess.
    weak = out["confidence"] < float(min_confidence)
    out.loc[weak, ["suggested_epic", "suggested_epic_key", "why"]] = ""
    out.loc[weak, "confidence"] = 0.0
    return out.sort_values(
        ["confidence", "key"], ascending=[False, True]
    ).reset_index(drop=True)


def empty_epics(df: pd.DataFrame) -> pd.DataFrame:
    """Epics with no open children left - closeable, or forgotten.

    The dashboard loads open work only, so "no open children" means the epic has
    nothing left to do here; whether that is finished or abandoned is what the
    idle column is for.
    """
    columns = ["key", "summary", "status", "assignee", "idle_days", "ticket_age_days"]
    if df.empty or "issue_type" not in df.columns:
        return pd.DataFrame(columns=columns)

    kinds = (
        df["issue_type"].fillna("").astype(str).str.strip().str.lower().str.replace(
            r"[-_\s]", "", regex=True
        )
    )
    epics = df[kinds.eq("epic")]
    if epics.empty:
        return pd.DataFrame(columns=columns)

    parented = set(
        df.get("epic_key", pd.Series(dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
    ) - {""}
    empty = epics[~epics["key"].astype(str).isin(parented)]
    present = [column for column in columns if column in empty.columns]
    return (
        empty[present]
        .sort_values("idle_days" if "idle_days" in present else "key", ascending=False)
        .reset_index(drop=True)
    )


def orphan_summary(suggestions: pd.DataFrame) -> pd.DataFrame:
    """The suggested epics, largest pile first - one epic to file many tickets."""
    columns = ["suggested_epic", "suggested_epic_key", "tickets", "keys"]
    if suggestions.empty:
        return pd.DataFrame(columns=columns)
    matched = suggestions[suggestions["suggested_epic_key"].ne("")]
    if matched.empty:
        return pd.DataFrame(columns=columns)
    grouped = matched.groupby(["suggested_epic", "suggested_epic_key"]).agg(
        tickets=("key", "count"),
        keys=("key", lambda values: ", ".join(sorted(values)[:8])),
    )
    return (
        grouped.reset_index()
        .sort_values("tickets", ascending=False)
        .reset_index(drop=True)[columns]
    )


__all__ = ["suggest_parents", "empty_epics", "orphan_summary", "NO_EPIC"]
