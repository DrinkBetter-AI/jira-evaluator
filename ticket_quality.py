"""How clear a ticket is, and whether it is clear enough to hand to Devin.

Angel's ask of the team was that a ticket state its goal and acceptance criteria
plainly enough that someone who was not in the conversation can pick it up. This
grades every ticket against that bar and names what is missing, so the gap is
something a person can fix rather than something they have to be chased about.

The bar is five things: a summary that says what the work is, a description long
enough to be a brief, explicit acceptance criteria, an estimate, and a parent
epic. Ownership is scored separately - an unowned ticket can still be perfectly
well written, and is in fact the best kind to hand off.

"Devin-able" is deliberately stricter than a good score: an agent has no one to
ask, so a ticket only qualifies when the goal and the finish line are both
written down, and when the work is the kind an agent can actually do. Tickets
that hinge on a design decision, an outage, or a conversation are marked No
however well written they are.
"""

from __future__ import annotations

import re

import pandas as pd

# What a description has to clear to count as a brief rather than a title
# repeated. Short enough that a genuinely small, well-stated ticket passes.
MIN_DESCRIPTION_CHARS = 120

# A summary of two words ("fix login") names a subject, not a task.
MIN_SUMMARY_WORDS = 4

_ACCEPTANCE_RE = re.compile(
    r"acceptance criteria|acceptance:|definition of done|\bDoD\b|"
    r"expected (?:result|behaviou?r|outcome)|success criteria|"
    r"steps to reproduce|given .*\bwhen\b.*\bthen\b",
    re.IGNORECASE | re.DOTALL,
)

# A checklist or a numbered list of outcomes is acceptance criteria whether or
# not anyone wrote the words.
_CHECKLIST_RE = re.compile(r"^\s*(?:[-*\u2022]\s*\[[ xX]\]|\d+[.)]\s+|[-*\u2022]\s+)", re.MULTILINE)
_MIN_CHECKLIST_ITEMS = 3

# Work that needs a human in the room. Matched on the summary, where the intent
# of a ticket is usually stated.
#
# Deliberately narrow: a word earns its place only if it names the activity
# rather than describing the subject. "align" and "policy" were tried and
# dropped - they demoted "Rename schemas (permanent Medusa schema alignment)"
# and "Define cost-control policy", where only the second is a conversation.
_HUMAN_ONLY_RE = re.compile(
    r"\b(?:discuss|decide|decision|meeting|sync|workshop|"
    r"research|investigate|explore|spike|proposal|propose|planning|"
    r"design|mockup|wireframe|brainstorm|interview|hiring|"
    r"onboard|offboard|coordinate|follow up|followup|review with|"
    r"roadmap|strategy|kick ?off|kickoff|retrospective|prioriti[sz]e|"
    r"presentation|training|"
    r"outage|incident|escalat|customer call|demo)\b",
    re.IGNORECASE,
)

# Containers describe other tickets' work, so the clarity bar does not apply.
CONTAINER_ISSUE_TYPES = {"epic", "initiative", "toplevelinitiative"}

_CRITERIA = ("has_summary", "has_description", "has_acceptance", "has_estimate", "has_epic")

# What to tell someone whose ticket failed, phrased as the thing to go do.
MISSING_LABELS = {
    "has_summary": "summary too vague",
    "has_description": "no real description",
    "has_acceptance": "no acceptance criteria",
    "has_estimate": "no estimate",
    "has_epic": "no epic",
}


def _text(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[column].fillna("").astype(str)


def _is_container(df: pd.DataFrame) -> pd.Series:
    types = _text(df, "issue_type").str.lower().str.replace(r"[^a-z]", "", regex=True)
    return types.isin(CONTAINER_ISSUE_TYPES)


def has_acceptance_criteria(text: str) -> bool:
    """True when the ticket says how anyone would know it is finished."""
    if not text:
        return False
    if _ACCEPTANCE_RE.search(text):
        return True
    return len(_CHECKLIST_RE.findall(text)) >= _MIN_CHECKLIST_ITEMS


def _estimate_seconds(df: pd.DataFrame) -> pd.Series:
    """Estimate in seconds, from whichever column the fetch produced.

    Mirrors ``hygiene._estimate_seconds``: a row can carry only the human string
    ("2h"), which still means an estimate exists.
    """
    seconds = pd.Series(0.0, index=df.index)
    for column in ("estimate_seconds", "original_estimate_sec"):
        if column in df.columns:
            seconds = seconds.where(
                seconds > 0, pd.to_numeric(df[column], errors="coerce").fillna(0.0)
            )
    if "original_estimate" in df.columns:
        has_text = _text(df, "original_estimate").str.strip().ne("")
        seconds = seconds.where(~(seconds.le(0) & has_text), 1.0)
    return seconds


def score_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 0-5 ``quality_score``, what is ``missing``, and ``devinable``.

    Epics and initiatives are scored ``NA`` rather than badly: they are meant to
    be short pointers to other work.
    """
    if df.empty:
        return df

    out = df.copy()
    summary = _text(out, "summary").str.strip()
    description = _text(out, "description").str.strip()

    out["has_summary"] = summary.str.split().str.len().ge(MIN_SUMMARY_WORDS)
    out["has_description"] = description.str.len().ge(MIN_DESCRIPTION_CHARS)
    out["has_acceptance"] = description.map(has_acceptance_criteria)
    out["has_estimate"] = _estimate_seconds(out) > 0
    out["has_epic"] = _text(out, "epic_key").str.strip().ne("")

    out["quality_score"] = out[list(_CRITERIA)].sum(axis=1).astype(int)
    out["missing"] = [
        ", ".join(MISSING_LABELS[c] for c in _CRITERIA if not row[c])
        for _, row in out[list(_CRITERIA)].iterrows()
    ]

    # An agent needs the goal and the finish line in writing, and work that does
    # not depend on being in the room. The estimate and the epic matter for
    # planning, not for whether the ticket can be executed, so they are left out.
    # Written acceptance criteria stand in for the description length bar: a
    # short ticket that says what done looks like is not a vague ticket.
    needs_a_person = summary.map(lambda s: bool(_HUMAN_ONLY_RE.search(s)))
    ready = out["has_summary"] & out["has_acceptance"] & description.ne("")
    out["devinable"] = pd.Series("No", index=out.index, dtype="object")
    # Written well enough to execute, but the goal reads as a conversation:
    # worth a look rather than a flat no.
    out.loc[ready & needs_a_person, "devinable"] = "Maybe"
    out.loc[ready & ~needs_a_person, "devinable"] = "Yes"
    # Everything a person would need is there except the finish line - the one
    # gap that is usually a minute's work to close.
    out.loc[
        ~ready & out["has_summary"] & out["has_description"] & ~needs_a_person,
        "devinable",
    ] = "Maybe"

    containers = _is_container(out)
    out.loc[containers, ["quality_score", "devinable"]] = [pd.NA, "n/a"]
    out.loc[containers, "missing"] = ""
    return out


def quality_by_person(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-reporter quality, worst average first.

    Grouped by reporter rather than assignee: the person who wrote the ticket is
    the one who can say what "done" means.
    """
    if scored.empty:
        return scored

    gradable = scored[scored["quality_score"].notna()].copy()
    if gradable.empty:
        return gradable

    gradable["quality_score"] = gradable["quality_score"].astype(int)
    rollup = (
        gradable.assign(
            ready=gradable["devinable"].eq("Yes"),
            no_criteria=~gradable["has_acceptance"],
            no_description=~gradable["has_description"],
        )
        .groupby(gradable["reporter"].fillna("Unknown"))
        .agg(
            tickets=("key", "size"),
            avg_score=("quality_score", "mean"),
            ready_for_devin=("ready", "sum"),
            no_acceptance_criteria=("no_criteria", "sum"),
            no_description=("no_description", "sum"),
        )
        .reset_index()
        .rename(columns={"reporter": "Reporter"})
    )
    rollup["avg_score"] = rollup["avg_score"].round(1)
    for column in ("ready_for_devin", "no_acceptance_criteria", "no_description"):
        rollup[column] = rollup[column].astype(int)
    return rollup.sort_values(["avg_score", "tickets"], ascending=[True, False])
