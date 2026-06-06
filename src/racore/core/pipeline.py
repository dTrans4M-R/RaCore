"""The two pipelines wired through the real ports: ``ingest()`` and ``answer()``.

Two cross-cutting concerns are built in from day one rather than bolted on:

* **Per-stage timing** (ADR-0010) — every stage records its wall-clock duration via the
  ``_Stopwatch`` so a latency regression is localizable to the stage that caused it.
* **Content-hash IDs** (ADR-0011) — documents and chunks are identified by a hash of
  their content (minted in the ingest stages via ``racore.core.ids``), so re-ingesting
  unchanged content is an idempotent upsert.

Grounding, relevance, and memory are *stages on the main answer path*, not add-ons —
matching ``docs/architecture.md`` §5.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from racore.core.types import (
    Answer,
    Citation,
    EmbeddedChunk,
    Evidence,
    GroundedContext,
    GroundingReport,
    IngestReport,
    InputType,
    LLMRequest,
    LLMResponse,
    MemoryItem,
    Query,
    Retrieval,
    StageTiming,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from racore.core.ports import (
        Chunker,
        DocumentSource,
        EmbeddingProvider,
        LLMProvider,
        MemoryStore,
        Reranker,
        VectorStore,
    )

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
    """

    embedder: EmbeddingProvider
    store: VectorStore
    reranker: Reranker
    chunker: Chunker
    llm: LLMProvider
    memory: MemoryStore | None = None

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
            context = _assemble(query.text, ranked)

        with sw.stage("generate"):
            (response,) = await self.llm.generate(
                [LLMRequest(query=query.text, context=context, system=_SYSTEM)]
            )

        with sw.stage("verify"):
            citations, grounding = _verify(response, context)

        if use_memory:
            assert self.memory is not None and query.user_id is not None  # narrowed above
            with sw.stage("memory.write"):
                await self.memory.write(query.tenant_id, query.user_id, _learn(query, response))

        return Answer(
            text=response.text,
            citations=citations,
            grounding=grounding,
            timings=sw.timings,
            retrievals=tuple(ranked),
        )


# --- stage helpers ----------------------------------------------------------------


def _understand(query: Query, memories: list[MemoryItem]) -> str:
    """Query-understanding seam. Phase 0 is the identity transform; query rewrite and
    memory-conditioned expansion land in Phase 2. ``memories`` is accepted now so the
    signature is stable once it starts to matter."""
    return query.text


def _assemble(query_text: str, ranked: list[Retrieval]) -> GroundedContext:
    """Turn ranked retrievals into cited evidence the LLM can ground in."""
    evidences = tuple(
        Evidence(
            quote=r.chunk.text,
            doc_id=r.chunk.doc_id,
            chunk_id=r.chunk.id,
            start=r.chunk.start,
            end=r.chunk.end,
            source=r.chunk.source,
        )
        for r in ranked
    )
    return GroundedContext(query=query_text, evidences=evidences)


_MARKER_RE = re.compile(r"\s*\[\d+\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _verify(
    response: LLMResponse, context: GroundedContext
) -> tuple[tuple[Citation, ...], GroundingReport]:
    """Resolve cited markers to citations and check faithfulness deterministically.

    A claim (a sentence of the answer, with markers stripped) is *supported* when it
    appears verbatim within one of the cited evidence quotes. This is the deterministic
    half of the faithfulness check in ``docs/evaluation.md`` §2; an LLM-judge for
    paraphrased support lands in Phase 1.
    """
    citations = tuple(
        Citation(marker=m, evidence=context.evidences[m - 1])
        for m in response.cited_markers
        if 1 <= m <= len(context.evidences)
    )
    cited_quotes = [_normalize(c.evidence.quote) for c in citations]

    supported: list[str] = []
    unsupported: list[str] = []
    for claim in _claims(response.text):
        norm = _normalize(claim)
        if any(norm in quote for quote in cited_quotes):
            supported.append(claim)
        else:
            unsupported.append(claim)

    return citations, GroundingReport(
        supported_claims=tuple(supported), unsupported_claims=tuple(unsupported)
    )


def _claims(text: str) -> list[str]:
    """Split an answer into claim sentences, with citation markers removed."""
    cleaned = _MARKER_RE.sub("", text).strip()
    if not cleaned:
        return []
    return [s for s in (part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned)) if s]


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for substring comparison."""
    return " ".join(text.lower().split())


def _learn(query: Query, response: LLMResponse) -> list[MemoryItem]:
    """Write policy for Phase 0: remember nothing. A salience-gated extractor that
    proposes memories from the turn lands in Phase 4 (``docs/memory.md`` §3). Kept as a
    seam so the ``memory.write`` stage and its timing already exist."""
    return []
