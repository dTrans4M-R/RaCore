"""The two pipelines wired through the real ports: ``ingest()`` and ``answer()``.

Two cross-cutting concerns are built in from day one rather than bolted on:

* **Per-stage timing** (ADR-0010) — every stage records its wall-clock duration via the
  ``_Stopwatch`` so a latency regression is localizable to the stage that caused it.
* **Content-hash IDs** (ADR-0011) — documents and chunks are identified by a hash of
  their content (minted in the ingest stages via ``racore.core.ids``), so re-ingesting
  unchanged content is an idempotent upsert.

Grounding, relevance, and memory are *stages on the main answer path*, not add-ons —
matching ``docs/architecture.md`` §5. The grounding logic itself lives in
``racore.core.grounding``; this module only orchestrates the ports and times the stages.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from racore.core import grounding
from racore.core.types import (
    Answer,
    EmbeddedChunk,
    GroundingReport,
    IngestReport,
    InputType,
    LLMRequest,
    StageTiming,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from racore.core.ports import (
        Chunker,
        DocumentSource,
        EmbeddingProvider,
        EntailmentJudge,
        LLMProvider,
        MemoryStore,
        Reranker,
        VectorStore,
    )
    from racore.core.types import LLMResponse, MemoryItem, Query

_SYSTEM = (
    "Answer the question using only the numbered evidence provided. Cite each claim with "
    "its evidence marker like [1]. If the evidence does not contain the answer, say you "
    "don't know rather than guessing."
)

_ABSTAIN_TEXT = "I don't know — I couldn't find supporting evidence in the corpus."


class _Stopwatch:
    """Accumulates per-stage durations for one pipeline run (ADR-0010)."""

    def __init__(self) -> None:
        self._timings: list[StageTiming] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._timings.append(StageTiming(stage=name, millis=elapsed_ms))

    @property
    def timings(self) -> tuple[StageTiming, ...]:
        return tuple(self._timings)


@dataclass(frozen=True, slots=True)
class Pipeline:
    """Holds the adapters and orchestrates ingest/answer over them.

    ``memory`` is optional: when absent (or when a query carries no ``user_id``) the
    memory read/write stages are skipped, so the corpus-only path stays simple.
    ``drop_unsupported`` switches the grounding stage from *flag* (default — unsupported
    claims are reported but kept) to *drop* (the answer is rebuilt from supported claims).
    """

    embedder: EmbeddingProvider
    store: VectorStore
    reranker: Reranker
    chunker: Chunker
    llm: LLMProvider
    judge: EntailmentJudge
    memory: MemoryStore | None = None
    drop_unsupported: bool = False

    async def ingest(self, source: DocumentSource, tenant_id: str = "default") -> IngestReport:
        """``fetch -> chunk -> embed -> upsert``, timing each stage."""
        sw = _Stopwatch()

        with sw.stage("fetch"):
            documents = await source.fetch()

        with sw.stage("chunk"):
            chunks = await self.chunker.chunk(documents)

        with sw.stage("embed"):
            vectors = await self.embedder.embed([c.text for c in chunks], InputType.DOCUMENT)

        embedded = [EmbeddedChunk(chunk=c, vector=v) for c, v in zip(chunks, vectors, strict=True)]

        with sw.stage("upsert"):
            await self.store.upsert(embedded, tenant_id)

        return IngestReport(documents=len(documents), chunks=len(chunks), timings=sw.timings)

    async def answer(self, query: Query) -> Answer:
        """``memory.read -> understand -> embed -> retrieve -> rerank -> assemble ->
        generate -> verify -> memory.write``. Returns a streamable ``Answer`` (ADR-0009),
        or abstains when no usable context survives reranking."""
        sw = _Stopwatch()
        use_memory = self.memory is not None and query.user_id is not None

        memories: list[MemoryItem] = []
        if use_memory:
            assert self.memory is not None and query.user_id is not None  # narrowed above
            with sw.stage("memory.read"):
                memories = await self.memory.read(query.tenant_id, query.user_id, query.text, k=3)

        with sw.stage("understand"):
            search_text = _understand(query, memories)

        with sw.stage("embed"):
            (query_vector,) = await self.embedder.embed([search_text], InputType.QUERY)

        with sw.stage("retrieve"):
            candidates = await self.store.search(
                query_vector, query.k, query.tenant_id, query.filters
            )

        with sw.stage("rerank"):
            ranked = await self.reranker.rerank(query.text, candidates, query.k)

        if not ranked:
            return Answer(
                text=_ABSTAIN_TEXT,
                citations=(),
                grounding=GroundingReport(supported_claims=(), unsupported_claims=()),
                timings=sw.timings,
                retrievals=(),
                abstained=True,
            )

        with sw.stage("assemble"):
            context = grounding.assemble(query.text, ranked)

        with sw.stage("generate"):
            (response,) = await self.llm.generate(
                [LLMRequest(query=query.text, context=context, system=_SYSTEM)]
            )

        with sw.stage("verify"):
            outcome = await grounding.verify(
                response, context, self.judge, drop_unsupported=self.drop_unsupported
            )

        if use_memory:
            assert self.memory is not None and query.user_id is not None  # narrowed above
            with sw.stage("memory.write"):
                await self.memory.write(query.tenant_id, query.user_id, _learn(query, response))

        return Answer(
            text=outcome.text,
            citations=outcome.citations,
            grounding=outcome.report,
            timings=sw.timings,
            retrievals=tuple(ranked),
            usage=response.usage,
        )


# --- stage helpers ----------------------------------------------------------------


def _understand(query: Query, memories: list[MemoryItem]) -> str:
    """Query-understanding seam. Phase 0 is the identity transform; query rewrite and
    memory-conditioned expansion land in Phase 2. ``memories`` is accepted now so the
    signature is stable once it starts to matter."""
    return query.text


def _learn(query: Query, response: LLMResponse) -> list[MemoryItem]:
    """Write policy for Phase 0: remember nothing. A salience-gated extractor that
    proposes memories from the turn lands in Phase 4 (``docs/memory.md`` §3). Kept as a
    seam so the ``memory.write`` stage and its timing already exist."""
    return []
