"""An in-memory vector store for dev and the $0 demo (the pgvector adapter mirrors it).

Upserts are keyed by the chunk's content-hash ID, so re-ingesting unchanged content
overwrites in place rather than duplicating (ADR-0011). Search is exact cosine over every
stored vector — fine for a demo corpus, not for production scale.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from racore.core.types import Chunk, EmbeddedChunk, Retrieval, Vector

if TYPE_CHECKING:
    from collections.abc import Mapping

# Chunk fields that a filter may constrain. Unknown filter keys match nothing, so a
# typo fails closed rather than silently returning everything.
_FILTERABLE = ("doc_id", "source")


class InMemoryVectorStore:
    """Per-tenant dict of ``chunk_id -> EmbeddedChunk`` with brute-force cosine search."""

    def __init__(self) -> None:
        self._by_tenant: dict[str, dict[str, EmbeddedChunk]] = {}

    async def upsert(self, items: list[EmbeddedChunk], tenant_id: str) -> None:
        store = self._by_tenant.setdefault(tenant_id, {})
        for item in items:
            store[item.chunk.id] = item

    async def chunk_ids(self, tenant_id: str) -> set[str]:
        return set(self._by_tenant.get(tenant_id, {}))

    async def delete(self, chunk_ids: list[str], tenant_id: str) -> None:
        store = self._by_tenant.get(tenant_id)
        if store is None:
            return
        for chunk_id in chunk_ids:
            store.pop(chunk_id, None)

    async def search(
        self,
        vector: Vector,
        k: int,
        tenant_id: str,
        filters: Mapping[str, str] | None = None,
    ) -> list[Retrieval]:
        store = self._by_tenant.get(tenant_id, {})
        active = filters or {}
        scored = [
            Retrieval(chunk=item.chunk, score=_cosine(vector, item.vector))
            for item in store.values()
            if _passes(item.chunk, active)
        ]
        # Stable sort: ties keep upsert order, so results are deterministic.
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]


def _passes(chunk: Chunk, filters: Mapping[str, str]) -> bool:
    fields = {"doc_id": chunk.doc_id, "source": chunk.source}
    return all(key in _FILTERABLE and fields[key] == value for key, value in filters.items())


def _cosine(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("cosine on vectors of different dimensions")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
