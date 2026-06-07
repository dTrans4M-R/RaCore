"""RuleBasedMemoryExtractor: the $0 write-policy floor (ADR-0026).

Proves the deterministic extractor turns explicit user self-statements into candidate memories
with the right kind, content, conflict ``key``, and provenance — and, crucially, extracts
*nothing* from an ordinary question, so asking does not pollute memory. Implicit-fact recall is
deliberately out of scope here; that is what a paid LLM extractor adds behind the same port.
"""

from __future__ import annotations

import asyncio

from racore.adapters.memory_extract import RuleBasedMemoryExtractor
from racore.core.types import MemoryItem, MemoryKind, MemoryTurn


def _extract(text: str, *, source: str = "turn-1") -> list[MemoryItem]:
    extractor = RuleBasedMemoryExtractor()
    turn = MemoryTurn(tenant_id="t", user_id="u", source=source, user_text=text)
    return asyncio.run(extractor.extract([turn]))


def test_stated_preference_becomes_a_profile_memory() -> None:
    (item,) = _extract("I prefer bullet-point summaries.")
    assert item.kind == MemoryKind.PROFILE
    assert "bullet" in item.content
    assert item.salience >= 0.8
    assert item.key == ""  # a bare preference has no stable slot; it accumulates.
    assert item.source == "turn-1"  # provenance is carried, so the memory is grounded.


def test_my_fact_keeps_its_slot_and_value_casing() -> None:
    (item,) = _extract("My fiscal year ends in March.")
    assert item.kind == MemoryKind.SEMANTIC
    assert item.content == "fiscal year ends in March"  # original casing preserved for "March".
    assert item.key == "fiscal year"  # the slot a later conflicting fact supersedes on (slice 2).


def test_name_and_occupation_are_profile_facts() -> None:
    (name,) = _extract("My name is Ada.")
    assert name.content == "name is Ada"
    assert name.key == "name"

    (work,) = _extract("I work in mergers and acquisitions.")
    assert work.content == "works in mergers and acquisitions"
    assert work.key == "work"


def test_remember_wrapper_boosts_salience_without_double_storing() -> None:
    (item,) = _extract("Remember that I prefer dark mode.")
    assert item.content == "prefers dark mode"  # the inner preference, extracted once.
    assert item.salience >= 0.9  # an explicit "remember ..." raises confidence over a bare one.


def test_remember_wrapper_keeps_an_unrecognised_fact_verbatim() -> None:
    (item,) = _extract("Remember that the launch is on Friday.")
    assert item.kind == MemoryKind.SEMANTIC
    assert item.content == "the launch is on Friday"


def test_a_question_extracts_nothing() -> None:
    # Asking about something must never be mistaken for stating a fact about oneself.
    assert _extract("What is the largest planet?") == []
    assert _extract("Do you prefer tea or coffee?") == []


def test_multiple_sentences_yield_multiple_memories() -> None:
    items = _extract("My name is Ada. I prefer bullet summaries.")
    assert len(items) == 2
    assert {i.key for i in items} == {"name", ""}
