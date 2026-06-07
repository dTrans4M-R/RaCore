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

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from racore.core import grounding
from racore.core.ports import UsageReporter
from racore.core.types import (
    Answer,
    EmbeddedChunk,
    GroundingReport,
    IngestReport,
    InputType,
    LLMRequest,
    RelevanceCheck,
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
        RelevanceGate,
        Reranker,
        VectorStore,
    )
    from racore.core.types import LLMResponse, MemoryItem, Query, TokenUsage

_SYSTEM = (
    "Answer the question using only the numbered evidence provided. Be concise: state each fact "
    "once, in its own sentence, and do not restate or pad the evidence. End every sentence with "
    "the evidence marker that supports it, like [1] — every sentence must carry its own citation. "
    "If the evidence does not contain the answer, say you don't know rather than guessing."
)

_ABSTAIN_TEXT = "I don't know — I couldn't find supporting evidence in the corpus."

# A generated answer that opens with the mandated "I don't know" is the model declining to
# answer (the system prompt instructs exactly that on missing evidence). We *record* that as an
# abstention so a correct refusal isn't scored as an ungrounded answer — a measurement-integrity
# signal (ADR-0013), distinct from the Phase 2 capability of *deciding* when to abstain.
_REFUSAL_RE = re.compile(r"^\W*i\s+don'?t\s+know\b", re.IGNORECASE)


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
    ``gate`` is optional: when present it decides — after rerank, before generate — whether
    the retrieved evidence warrants an answer, and an abstain short-circuits generation
    (ADR-0021). When absent the pipeline answers whenever any context survives rerank.
    """

    embedder: EmbeddingProvider
    store: VectorStore
    reranker: Reranker
    chunker: Chunker
    llm: LLMProvider
    judge: EntailmentJudge
    memory: MemoryStore | None = None
    drop_unsupported: bool = False
    gate: RelevanceGate | None = None

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
        # Discard ingest-time embedding usage: cost/answer is per-answer. Draining here keeps the
        # embedder's meter clean so the first answer doesn't absorb the corpus's embed cost.
        _drain(self.embedder)

        with sw.stage("upsert"):
            await self.store.upsert(embedded, tenant_id)

        return IngestReport(documents=len(documents), chunks=len(chunks), timings=sw.timings)

    async def answer(self, query: Query) -> Answer:
        """``memory.read -> understand -> embed -> retrieve -> rerank -> relevance ->
        assemble -> generate -> verify -> memory.write``. Returns a streamable ``Answer``
        (ADR-0009), or abstains when no usable context survives reranking — or, when a
        relevance gate is configured, when it judges the surviving context too weak to
        answer (short-circuiting generation)."""
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
        query_usages = _drain(
            self.embedder
        )  # this answer's query-embed cost (ingest already drained)

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
                usages=tuple(query_usages),  # the query embed still cost something
            )

        if self.gate is not None:
            with sw.stage("relevance"):
                (should_answer,) = await self.gate.should_answer(
                    [RelevanceCheck(query=query.text, retrievals=tuple(ranked))]
                )
            if not should_answer:
                # Proactive abstention (ADR-0021): the surviving evidence isn't relevant
                # enough to answer, so skip the expensive generate/verify stages entirely.
                # The retrievals are kept so the harness still scores what retrieval reached.
                return Answer(
                    text=_ABSTAIN_TEXT,
                    citations=(),
                    grounding=GroundingReport(supported_claims=(), unsupported_claims=()),
                    timings=sw.timings,
                    retrievals=tuple(ranked),
                    abstained=True,
                    usages=(*query_usages, *_drain(self.gate)),
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

        # Sum every billed component this answer touched: query embed, the generator, and the judge.
        generator_usage = [response.usage] if response.usage is not None else []
        usages = (*query_usages, *generator_usage, *_drain(self.judge))
        return Answer(
            text=outcome.text,
            citations=outcome.citations,
            grounding=outcome.report,
            timings=sw.timings,
            retrievals=tuple(ranked),
            usages=usages,
            abstained=_is_refusal(response.text),
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


def _is_refusal(text: str) -> bool:
    """True when the model declined to answer (opens with the mandated "I don't know").

    This records what the model did so a correct refusal on a no-evidence question isn't
    scored as an ungrounded answer (ADR-0013). It is a heuristic on the phrasing the system
    prompt mandates — Phase 2 replaces it with a robust abstention decision."""
    return _REFUSAL_RE.match(text) is not None


def _drain(component: object) -> list[TokenUsage]:
    """Drain a component's accumulated token usage if it bills (the optional ``UsageReporter``
    port, ADR-0018). Free adapters don't implement it and contribute nothing, so the $0 stack
    reports a true $0 and the harness can price every paid component, not just the generator."""
    if isinstance(component, UsageReporter):
        return component.drain_usage()
    return []
