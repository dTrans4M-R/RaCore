"""A no-op reranker: it trusts the retriever's order and just enforces ``top_k``.

It exists so the rerank stage is present and timed from day one (ADR-0010); a real
cross-encoder or hosted reranker is a drop-in replacement that actually re-orders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from racore.core.types import Retrieval


class NoopReranker:
    """Returns the first ``top_k`` candidates unchanged."""

    async def rerank(self, query: str, candidates: list[Retrieval], top_k: int) -> list[Retrieval]:
        # query is unused — by definition this reranker does not look at relevance.
        return candidates[:top_k]
