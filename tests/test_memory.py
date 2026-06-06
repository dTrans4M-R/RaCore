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
