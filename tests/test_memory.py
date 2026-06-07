"""FileMemoryStore: persistence round-trip, per-user isolation, superseded exclusion.

Exercises the memory adapter directly. Memory is wired into the pipeline but inert in
Phase 0 (the write policy is empty until Phase 4); these tests prove the durable substrate
it will build on, including the isolation boundary from ``docs/memory.md`` §7.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from racore.adapters.memory import FileMemoryStore
from racore.core.types import MemoryItem, MemoryKind

if TYPE_CHECKING:
    from pathlib import Path


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    asyncio.run(_write_then_read_roundtrip(tmp_path))


async def _write_then_read_roundtrip(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    items = [
        MemoryItem(
            id="m1",
            tenant_id="t",
            user_id="u",
            kind=MemoryKind.SEMANTIC,
            content="fiscal year ends in March",
            source="turn-1",
            salience=0.9,
            embedding=(0.1, 0.2, 0.3),
        ),
        MemoryItem(
            id="m2",
            tenant_id="t",
            user_id="u",
            kind=MemoryKind.PROFILE,
            content="prefers bullet summaries",
            source="turn-2",
        ),
    ]
    await store.write("t", "u", items)

    got = await store.read("t", "u", "tell me about the fiscal year", k=5)

    assert len(got) == 2
    # Ranked by query overlap: the fiscal-year fact comes first and round-trips intact.
    assert got[0].content == "fiscal year ends in March"
    assert got[0].kind == MemoryKind.SEMANTIC
    assert got[0].embedding == (0.1, 0.2, 0.3)
    assert got[0].salience == 0.9


def test_users_are_isolated(tmp_path: Path) -> None:
    asyncio.run(_users_are_isolated(tmp_path))


async def _users_are_isolated(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    await store.write(
        "t",
        "alice",
        [
            MemoryItem(
                id="a",
                tenant_id="t",
                user_id="alice",
                kind=MemoryKind.PROFILE,
                content="alice private note",
                source="x",
            )
        ],
    )

    # Bob must never see Alice's memory.
    assert await store.read("t", "bob", "private note", k=5) == []


def test_superseded_items_are_hidden(tmp_path: Path) -> None:
    asyncio.run(_superseded_items_are_hidden(tmp_path))


async def _superseded_items_are_hidden(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    await store.write(
        "t",
        "u",
        [
            MemoryItem(
                id="old",
                tenant_id="t",
                user_id="u",
                kind=MemoryKind.SEMANTIC,
                content="old fact",
                source="x",
                superseded_by="new",
            )
        ],
    )

    assert await store.read("t", "u", "old fact", k=5) == []


def test_a_newer_fact_supersedes_the_same_slot(tmp_path: Path) -> None:
    asyncio.run(_newer_fact_supersedes(tmp_path))


async def _newer_fact_supersedes(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    ada = MemoryItem(
        id="ada",
        tenant_id="t",
        user_id="u",
        kind=MemoryKind.PROFILE,
        content="name is Ada",
        source="turn-1",
        key="name",
    )
    bob = MemoryItem(
        id="bob",
        tenant_id="t",
        user_id="u",
        kind=MemoryKind.PROFILE,
        content="name is Bob",
        source="turn-2",
        key="name",
    )
    await store.write("t", "u", [ada])
    await store.write("t", "u", [bob])

    # The slot has one current value — the newer fact — not two contradictory ones.
    assert [m.content for m in await store.read("t", "u", "name", k=5)] == ["name is Bob"]

    # The superseded fact is kept on disk, pointing at its successor, with its provenance intact.
    superseded = next(m for m in store._load("t", "u") if m.id == "ada")
    assert superseded.superseded_by == "bob"
    assert superseded.source == "turn-1"


def test_distinct_slots_and_keyless_facts_coexist(tmp_path: Path) -> None:
    asyncio.run(_distinct_slots_coexist(tmp_path))


async def _distinct_slots_coexist(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    facts = [
        MemoryItem(
            id="n",
            tenant_id="t",
            user_id="u",
            kind=MemoryKind.PROFILE,
            content="name is Ada",
            source="x",
            key="name",
        ),
        MemoryItem(
            id="f",
            tenant_id="t",
            user_id="u",
            kind=MemoryKind.SEMANTIC,
            content="fiscal year ends in March",
            source="x",
            key="fiscal year",
        ),
        MemoryItem(
            id="p1",
            tenant_id="t",
            user_id="u",
            kind=MemoryKind.PROFILE,
            content="prefers tea",
            source="x",
            key="",
        ),
        MemoryItem(
            id="p2",
            tenant_id="t",
            user_id="u",
            kind=MemoryKind.PROFILE,
            content="prefers window seats",
            source="x",
            key="",
        ),
    ]
    await store.write("t", "u", facts)

    # Different slots don't collide, and keyless preferences accumulate rather than supersede.
    got = {m.id for m in await store.read("t", "u", "anything", k=10)}
    assert got == {"n", "f", "p1", "p2"}


def test_read_ranks_by_salience_within_equal_relevance(tmp_path: Path) -> None:
    asyncio.run(_ranks_by_salience(tmp_path))


async def _ranks_by_salience(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    await store.write(
        "t",
        "u",
        [
            # Both overlap the query "tea" exactly once, so salience is the tie-breaker.
            MemoryItem(
                id="weak",
                tenant_id="t",
                user_id="u",
                kind=MemoryKind.PROFILE,
                content="I like tea",
                source="x",
                salience=0.3,
            ),
            MemoryItem(
                id="strong",
                tenant_id="t",
                user_id="u",
                kind=MemoryKind.PROFILE,
                content="tea is essential",
                source="x",
                salience=0.9,
            ),
        ],
    )

    got = await store.read("t", "u", "tea", k=5)
    assert [m.id for m in got] == ["strong", "weak"]
