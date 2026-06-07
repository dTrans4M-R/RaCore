"""Phase 3, slice 1: incremental re-index by content-hash diff (ADR-0023).

The freshness guarantee made executable. Re-ingesting unchanged content must embed nothing
and touch nothing; re-ingesting a *changed* corpus must touch only the delta (add the new,
delete the gone) and leave no stale chunk searchable. A counting embedder proves the work is
proportional to the change, not the corpus.
"""

from __future__ import annotations

import asyncio

from racore.adapters.chunkers import FixedWindowChunker
from racore.adapters.embeddings import MockEmbeddingProvider
from racore.adapters.judges import SubstringEntailmentJudge
from racore.adapters.llm import ExtractiveLLM
from racore.adapters.rerankers import NoopReranker
from racore.adapters.sources import InMemoryDocumentSource
from racore.adapters.vectorstores import InMemoryVectorStore
from racore.core.pipeline import Pipeline
from racore.core.types import InputType, Vector


class _CountingEmbedder:
    """Wraps the $0 mock embedder and records every text it is asked to embed, so a test can
    assert that an unchanged re-ingest does *no* embedding work — the incremental win."""

    def __init__(self) -> None:
        self._inner = MockEmbeddingProvider()
        self.embedded: list[str] = []

    async def embed(self, texts: list[str], input_type: InputType) -> list[Vector]:
        self.embedded.extend(texts)
        return await self._inner.embed(texts, input_type)


def _pipeline() -> tuple[Pipeline, _CountingEmbedder, InMemoryVectorStore]:
    embedder = _CountingEmbedder()
    store = InMemoryVectorStore()
    pipeline = Pipeline(
        embedder=embedder,
        store=store,
        reranker=NoopReranker(),
        chunker=FixedWindowChunker(),
        llm=ExtractiveLLM(),
        judge=SubstringEntailmentJudge(),
    )
    return pipeline, embedder, store


def test_reingest_unchanged_embeds_nothing() -> None:
    asyncio.run(_reingest_unchanged_embeds_nothing())


async def _reingest_unchanged_embeds_nothing() -> None:
    pipeline, embedder, _ = _pipeline()
    source = InMemoryDocumentSource(
        [("d1", "Mercury is the smallest planet."), ("d2", "Mars is the red planet.")]
    )

    first = await pipeline.ingest(source, prune=True)
    assert (first.added, first.unchanged, first.deleted) == (2, 0, 0)
    assert first.chunks == 2
    assert len(embedder.embedded) == 2  # both chunks embedded on the first pass.

    second = await pipeline.ingest(source, prune=True)
    # Idempotent: nothing new, nothing stale — and crucially, no embedding work happened.
    assert (second.added, second.unchanged, second.deleted) == (0, 2, 0)
    assert len(embedder.embedded) == 2  # still 2 — the re-ingest embedded zero chunks.


def test_reindex_touches_only_the_delta() -> None:
    asyncio.run(_reindex_touches_only_the_delta())


async def _reindex_touches_only_the_delta() -> None:
    pipeline, embedder, store = _pipeline()

    v1 = InMemoryDocumentSource(
        [
            ("d1", "Mercury is the smallest planet."),
            ("d2", "Mars is the red planet."),
            ("d3", "Jupiter is the largest planet."),
        ]
    )
    await pipeline.ingest(v1, prune=True)
    assert len(embedder.embedded) == 3
    embedded_after_v1 = len(embedder.embedded)

    # v2: d1 unchanged, d2 edited, d3 removed, d4 added.
    v2 = InMemoryDocumentSource(
        [
            ("d1", "Mercury is the smallest planet."),
            ("d2", "Mars is the rusty red planet."),
            ("d4", "Venus is the hottest planet."),
        ]
    )
    report = await pipeline.ingest(v2, prune=True)

    # Only the delta is touched: d2' and d4 added; old-d2 and d3 chunks deleted; d1 unchanged.
    assert (report.added, report.unchanged, report.deleted) == (2, 1, 2)
    # Embedding work was proportional to the change, not the corpus: only the 2 new chunks.
    assert len(embedder.embedded) - embedded_after_v1 == 2

    # Freshness guarantee: the index now matches v2 exactly — no stale content survives.
    (probe,) = await embedder.embed(["planet"], InputType.QUERY)
    hits = await store.search(probe, k=100, tenant_id="default")
    sources = {r.chunk.source for r in hits}
    assert sources == {"d1", "d2", "d4"}  # d3 is gone.
    d2_hit = next(r for r in hits if r.chunk.source == "d2")
    assert d2_hit.chunk.text == "Mars is the rusty red planet."  # the edited text, not the old.


def test_default_ingest_is_additive_and_never_deletes() -> None:
    asyncio.run(_default_ingest_is_additive())


async def _default_ingest_is_additive() -> None:
    # prune defaults to False: separate sources accumulate in one tenant; nothing is pruned.
    pipeline, _, store = _pipeline()

    a = await pipeline.ingest(InMemoryDocumentSource([("a", "Alpha doc.")]))
    assert (a.added, a.deleted) == (1, 0)

    b = await pipeline.ingest(InMemoryDocumentSource([("b", "Beta doc.")]))
    # The second source's doc was absent from the store, so it is added; a's doc is *not* pruned
    # even though it wasn't in this fetch — additive ingest never deletes.
    assert (b.added, b.deleted) == (1, 0)
    assert len(await store.chunk_ids("default")) == 2
