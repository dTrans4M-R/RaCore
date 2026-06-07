"""The grounding-gated answer cache, end to end through the pipeline (Phase 5 slice 3).

The cache's whole claim is that it is *safe* — it serves a repeat ask fast, but never a stale or
wrong answer. These tests pin both halves: the latency win (a hit skips the generate stage) and the
correctness gate (an answer is evicted exactly when the evidence it stood on changes, and never
otherwise). The gate is structural, riding the content-hash IDs (ADR-0011): edit the grounding
evidence and the cached answer auto-invalidates; touch an unrelated document and it stays valid.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from racore.adapters.cache import GroundingGatedCache
from racore.adapters.chunkers import FixedWindowChunker
from racore.adapters.embeddings import MockEmbeddingProvider
from racore.adapters.judges import SubstringEntailmentJudge
from racore.adapters.llm import ExtractiveLLM
from racore.adapters.memory import FileMemoryStore
from racore.adapters.memory_extract import RuleBasedMemoryExtractor
from racore.adapters.rerankers import NoopReranker
from racore.adapters.sources import InMemoryDocumentSource
from racore.adapters.vectorstores import InMemoryVectorStore
from racore.core.pipeline import Pipeline
from racore.core.types import Answer, Query

if TYPE_CHECKING:
    from pathlib import Path


def _pipeline() -> Pipeline:
    return Pipeline(
        embedder=MockEmbeddingProvider(),
        store=InMemoryVectorStore(),
        reranker=NoopReranker(),
        chunker=FixedWindowChunker(),
        llm=ExtractiveLLM(),
        judge=SubstringEntailmentJudge(),
        cache=GroundingGatedCache(),
    )


def _stages(answer: Answer) -> set[str]:
    return {t.stage for t in answer.timings}


def _was_generated(answer: Answer) -> bool:
    """A fresh answer runs the generate stage; a cache hit never does."""
    return "generate" in _stages(answer)


def test_repeat_query_is_served_from_cache(tmp_path: Path) -> None:
    async def go() -> None:
        pipe = _pipeline()
        await pipe.ingest(
            InMemoryDocumentSource([("sky", "Mercury is closest to the Sun.")]), tenant_id="t"
        )
        q = Query(text="What is closest to the Sun?", tenant_id="t")

        first = await pipe.answer(q)
        second = await pipe.answer(q)

        assert "Mercury" in first.text and "Mercury" in second.text
        assert _was_generated(first)  # cold: full pipeline ran
        assert not _was_generated(second)  # warm: served from cache, generate skipped
        assert _stages(second) == {"cache"}  # the hit is a single fast stage

    asyncio.run(go())


def test_editing_the_grounding_evidence_invalidates_the_cache(tmp_path: Path) -> None:
    async def go() -> None:
        pipe = _pipeline()
        q = Query(text="What is closest to the Sun?", tenant_id="t")

        await pipe.ingest(
            InMemoryDocumentSource([("sky", "Mercury is closest to the Sun.")]), tenant_id="t"
        )
        first = await pipe.answer(q)
        assert "Mercury" in first.text

        # Re-ingest the same source with corrected text and prune: the old chunk's content hash is
        # gone, a new one takes its place. The cached "Mercury" answer stood on the old chunk, so it
        # must be evicted and the question re-answered against the current evidence.
        await pipe.ingest(
            InMemoryDocumentSource([("sky", "Venus is closest to the Sun.")]),
            tenant_id="t",
            prune=True,
        )
        second = await pipe.answer(q)

        assert "Venus" in second.text
        assert "Mercury" not in second.text
        assert _was_generated(second)  # the stale entry was not served

    asyncio.run(go())


def test_an_unrelated_ingest_keeps_the_cache_valid(tmp_path: Path) -> None:
    async def go() -> None:
        pipe = _pipeline()
        q = Query(text="What is closest to the Sun?", tenant_id="t")

        await pipe.ingest(
            InMemoryDocumentSource([("sky", "Mercury is closest to the Sun.")]), tenant_id="t"
        )
        await pipe.answer(q)  # warms the cache

        # Add an unrelated document (additive). It touches none of the chunks the cached answer was
        # grounded on, so the entry stays valid and the repeat ask is still served fast.
        await pipe.ingest(
            InMemoryDocumentSource([("trivia", "The Great Wall is in China.")]), tenant_id="t"
        )
        second = await pipe.answer(q)

        assert "Mercury" in second.text
        assert not _was_generated(second)

    asyncio.run(go())


def test_cache_is_isolated_per_tenant(tmp_path: Path) -> None:
    async def go() -> None:
        pipe = _pipeline()
        await pipe.ingest(
            InMemoryDocumentSource([("sky", "Mercury is closest to the Sun.")]), tenant_id="t1"
        )
        await pipe.ingest(
            InMemoryDocumentSource([("sky", "Venus is closest to the Sun.")]), tenant_id="t2"
        )
        q1 = Query(text="What is closest to the Sun?", tenant_id="t1")
        q2 = Query(text="What is closest to the Sun?", tenant_id="t2")

        await pipe.answer(q1)  # caches t1's "Mercury" answer
        t2 = await pipe.answer(q2)

        # Same normalized question, different tenant: t1's entry must never serve t2.
        assert "Venus" in t2.text
        assert "Mercury" not in t2.text
        assert _was_generated(t2)

    asyncio.run(go())


def test_an_abstain_is_not_cached(tmp_path: Path) -> None:
    async def go() -> None:
        pipe = _pipeline()
        q = Query(text="What is closest to the Sun?", tenant_id="t")

        first = await pipe.answer(q)  # empty corpus -> abstains
        assert first.abstained

        # The abstain must not have been cached: once the corpus gains the answer, the next ask is
        # answered correctly rather than replaying a stale "I don't know".
        await pipe.ingest(
            InMemoryDocumentSource([("sky", "Mercury is closest to the Sun.")]), tenant_id="t"
        )
        second = await pipe.answer(q)
        assert not second.abstained
        assert "Mercury" in second.text

    asyncio.run(go())


def test_personalized_answers_bypass_the_shared_cache(tmp_path: Path) -> None:
    async def go() -> None:
        # A pipeline with both memory and the cache: a per-user answer must never be stored where a
        # different user could be served it.
        pipe = Pipeline(
            embedder=MockEmbeddingProvider(),
            store=InMemoryVectorStore(),
            reranker=NoopReranker(),
            chunker=FixedWindowChunker(),
            llm=ExtractiveLLM(),
            judge=SubstringEntailmentJudge(),
            memory=FileMemoryStore(tmp_path),
            extractor=RuleBasedMemoryExtractor(),
            cache=GroundingGatedCache(),
        )
        await pipe.answer(Query(text="My name is Bob.", tenant_id="t", user_id="alice"))
        mine = await pipe.answer(Query(text="What is my name?", tenant_id="t", user_id="alice"))
        assert "Bob" in mine.text

        # A different user asks the identical question. If alice's answer had leaked into the shared
        # cache, carol would be told "Bob". It must not — carol has no such memory.
        carol = await pipe.answer(Query(text="What is my name?", tenant_id="t", user_id="carol"))
        assert "Bob" not in carol.text

    asyncio.run(go())
