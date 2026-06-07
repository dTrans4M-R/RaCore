"""A $0, deterministic memory extractor: the write-policy floor (``docs/memory.md`` §3).

It recognises a small set of **explicit** self-statement patterns — a stated preference, a
``my <slot> is <value>`` fact, an occupation, or an explicit ``remember that ...`` request — and
proposes one ``MemoryItem`` per match. Each item gets a content-hash ID (so re-stating a fact is
idempotent), a conflict ``key`` slot when the statement names one, and the turn as provenance.

It deliberately favours **precision over recall**: it captures what the user said outright — the
highest-salience, most reusable memory — and infers nothing. That missing recall (implicit facts,
paraphrased preferences, episodic summaries) is exactly what a paid LLM extractor adds behind the
same ``MemoryExtractor`` port — the "paid only improves" lever. So personalization works at $0,
and spend only widens what is remembered, never enables it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from racore.core.ids import content_id
from racore.core.types import MemoryItem, MemoryKind

if TYPE_CHECKING:
    from racore.core.types import MemoryTurn

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# "remember (that) X" / "please remember X" — an explicit request; X is re-run through the rules.
_REMEMBER_RE = re.compile(r"^(?:please\s+)?remember\s+(?:that\s+)?(?P<inner>.+)$", re.IGNORECASE)

# Rules tried in order; the first to match a sentence wins (so "my name is X" is a name, not a
# generic "my <slot> is" fact). Each exposes a ``value`` group (the remembered content); the
# my-fact rule also exposes ``slot`` (its conflict key) and ``verb``.
_NAME_RE = re.compile(
    r"\b(?:my\s+name\s+is|call\s+me)\s+(?P<value>[A-Za-z][\w'-]*)\b", re.IGNORECASE
)
_MY_FACT_RE = re.compile(
    r"\bmy\s+(?P<slot>[A-Za-z][A-Za-z '\-]*?)\s+"
    r"(?P<verb>is|are|was|were|ends|begins|starts)\b\s+(?P<rest>.+?)\s*$",
    re.IGNORECASE,
)
_WORK_RE = re.compile(r"\bi\s+work\s+(?P<prep>in|as|at|on)\s+(?P<value>.+?)\s*$", re.IGNORECASE)
_PREFERENCE_RE = re.compile(
    r"\bi\s+(?:really\s+|always\s+)?(?:prefer|like|love|enjoy|want|favou?r)\s+(?P<value>.+?)\s*$",
    re.IGNORECASE,
)

# Below this, a candidate is not worth storing (the salience gate). Every rule here clears it; the
# floor is the seam at which weaker or model-proposed signals get dropped, not stored as noise.
_SALIENCE_FLOOR = 0.5
_WRAPPER_BOOST = 0.1  # an explicit "remember ..." raises confidence in whatever it wraps.


class RuleBasedMemoryExtractor:
    """Pulls explicit user self-statements from a turn into candidate memories. $0, stdlib."""

    async def extract(self, turns: list[MemoryTurn]) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        for turn in turns:
            for sentence in _sentences(turn.user_text):
                item = _from_sentence(sentence, turn)
                if item is not None and item.salience >= _SALIENCE_FLOOR:
                    items.append(item)
        return items


def _from_sentence(sentence: str, turn: MemoryTurn) -> MemoryItem | None:
    """One candidate memory from one sentence, or ``None`` if it states nothing durable."""
    # A question is asking, not stating — "What field do I work in?" must never be mistaken for a
    # self-statement. Filtering it keeps precision high (and stops a probe from polluting memory).
    if sentence.rstrip().endswith("?"):
        return None

    boost = 0.0
    wrapped = _REMEMBER_RE.match(sentence)
    if wrapped is not None:
        sentence, boost = wrapped.group("inner"), _WRAPPER_BOOST

    parsed = _match_rules(sentence)
    if parsed is None:
        # An explicit "remember ..." with no recognised shape is still worth keeping verbatim.
        if boost == 0.0:
            return None
        kind, content, key, salience = MemoryKind.SEMANTIC, _clean(sentence), "", 0.7
    else:
        kind, content, key, salience = parsed
    if not content:
        return None
    return _build(turn, kind, content, key, min(salience + boost, 0.97))


def _match_rules(text: str) -> tuple[MemoryKind, str, str, float] | None:
    """Return ``(kind, content, key, salience)`` for the first rule that matches ``text``.

    Each rule requires a non-empty captured value, so a fragment like "I work in" (no object) does
    not produce a hollow memory."""
    name = _NAME_RE.search(text)
    if name is not None and (value := _clean(name.group("value"))):
        return MemoryKind.PROFILE, f"name is {value}", "name", 0.9

    fact = _MY_FACT_RE.search(text)
    if (
        fact is not None
        and (slot := _clean(fact.group("slot")))
        and (rest := _clean(fact.group("rest")))
    ):
        return (
            MemoryKind.SEMANTIC,
            f"{slot} {fact.group('verb').lower()} {rest}",
            slot.lower(),
            0.85,
        )

    work = _WORK_RE.search(text)
    if work is not None and (value := _clean(work.group("value"))):
        return MemoryKind.PROFILE, f"works {work.group('prep').lower()} {value}", "work", 0.8

    pref = _PREFERENCE_RE.search(text)
    if pref is not None and (value := _clean(pref.group("value"))):
        return MemoryKind.PROFILE, f"prefers {value}", "", 0.8
    return None


def _build(
    turn: MemoryTurn, kind: MemoryKind, content: str, key: str, salience: float
) -> MemoryItem:
    return MemoryItem(
        id=content_id(turn.tenant_id, turn.user_id, content),
        tenant_id=turn.tenant_id,
        user_id=turn.user_id,
        kind=kind,
        content=content,
        source=turn.source,
        key=key,
        salience=salience,
    )


def _sentences(text: str) -> list[str]:
    return [s for s in (part.strip() for part in _SENTENCE_SPLIT_RE.split(text.strip())) if s]


def _clean(value: str) -> str:
    """Trim surrounding whitespace and trailing sentence punctuation, preserving inner casing."""
    return value.strip().rstrip(".!?,;:").strip()
