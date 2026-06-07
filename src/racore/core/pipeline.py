"""The two pipelines wired through the real ports: ``ingest()`` and ``answer()``.

Two cross-cutting concerns are built in from day one rather than bolted on:

* **Per-stage timing** (ADR-0010) — every stage records its wall-clock duration via the
  ``_Stopwatch`` so a latency regression is localizable to the stage that caused it.
* **Content-hash IDs** (ADR-0011) — documents and chunks are identified by a hash of
  their content (minted in the ingest stages via ``racore.core.ids``), so re-ingesting
  unchanged content is an idempotent upsert. ``ingest()`` diffs against the store on this
  hash to re-index incrementally (ADR-0023): embed only what changed, prune what's gone.

Grounding, relevance, and memory are *stages on the main answer path*, not add-ons —
matching ``docs/architecture.md`` §5. The grounding logic itself lives in
``racore.core.grounding``; this module only orchestrates the ports and times the stages.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from racore.core import grounding
from racore.core.ids import content_id
from racore.core.ports import UsageReporter
from racore.core.types import (
    Answer,
    CacheKey,
    Chunk,
    EmbeddedChunk,
    GroundingReport,
    IngestReport,
    InputType,
    LLMRequest,
    MemoryTurn,
    RelevanceCheck,
    Retrieval,
    StageTiming,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from racore.core.ports import (
        AnswerCache,
        Chunker,
        DocumentSource,
        EmbeddingProvider,
        EntailmentJudge,
        LLMProvider,
        MemoryExtractor,
        MemoryStore,
        RelevanceGate,
        Reranker,
        VectorStore,
    )
    from racore.core.types import MemoryItem, Query, TokenUsage

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

# Token relevance for memory injection: which remembered facts bear on the current query.
_WORD_RE = re.compile(r"\w+")


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
    memory read/write stages are skipped, so the corpus-only path stays simple. When present
    with an ``extractor``, ``answer()`` reads the user's relevant memories in (as labelled
    grounded evidence) and learns durable facts from the turn back out (ADR-0026).
    ``drop_unsupported`` switches the grounding stage from *flag* (default — unsupported
    claims are reported but kept) to *drop* (the answer is rebuilt from supported claims).
    ``gate`` is optional: when present it decides — after rerank, before generate — whether
    the retrieved evidence warrants an answer, and an abstain short-circuits generation
    (ADR-0021). When absent the pipeline answers whenever any context survives rerank.
    ``cache`` is optional: when present, a repeat ask whose grounding is still intact is served
    without re-running the expensive generate path (the latency lever, ADR-0020/0031). Only
    corpus-scoped answers are cached — a query carrying a ``user_id`` depends on per-user memory
    and bypasses the shared cache.
    """

    embedder: EmbeddingProvider
    store: VectorStore
    reranker: Reranker
    chunker: Chunker
    llm: LLMProvider
    judge: EntailmentJudge
    memory: MemoryStore | None = None
    extractor: MemoryExtractor | None = None
    drop_unsupported: bool = False
    gate: RelevanceGate | None = None
    cache: AnswerCache | None = None

    async def ingest(
        self, source: DocumentSource, tenant_id: str = "default", *, prune: bool = False
    ) -> IngestReport:
        """``fetch -> chunk -> diff -> embed -> upsert [-> prune]``, timing each stage.

        Incremental by content hash (ADR-0023): only chunks whose ID is not already stored are
        embedded and upserted, so re-ingesting unchanged content does no embedding work and is a
        true no-op. With ``prune=True`` the fetch is treated as the tenant's *complete* corpus and
        any stored chunk absent from it is deleted — so an edited or removed document leaves no
        stale chunk behind (the freshness guarantee). ``prune=False`` (the default) is purely
        additive: it never deletes, so several sources can be ingested into one tenant across calls.
        """
        sw = _Stopwatch()

        with sw.stage("fetch"):
            documents = await source.fetch()

        with sw.stage("chunk"):
            chunks = await self.chunker.chunk(documents)

        with sw.stage("diff"):
            stored = await self.store.chunk_ids(tenant_id)
            new_chunks = [c for c in chunks if c.id not in stored]
            stale = stored - {c.id for c in chunks} if prune else set()

        embedded: list[EmbeddedChunk] = []
        if new_chunks:
            with sw.stage("embed"):
                vectors = await self.embedder.embed(
                    [c.text for c in new_chunks], InputType.DOCUMENT
                )
            embedded = [
                EmbeddedChunk(chunk=c, vector=v) for c, v in zip(new_chunks, vectors, strict=True)
            ]
            # Discard ingest-time embedding usage: cost/answer is per-answer. Draining here keeps
            # the embedder's meter clean so the first answer doesn't absorb the corpus's embed cost.
            _drain(self.embedder)

        with sw.stage("upsert"):
            await self.store.upsert(embedded, tenant_id)

        if stale:
            with sw.stage("prune"):
                await self.store.delete(list(stale), tenant_id)

        return IngestReport(
            documents=len(documents),
            chunks=len(chunks),
            timings=sw.timings,
            added=len(new_chunks),
            unchanged=len(chunks) - len(new_chunks),
            deleted=len(stale),
        )

    async def answer(self, query: Query) -> Answer:
        """``memory.read -> understand -> embed -> retrieve -> rerank -> relevance ->
        assemble -> generate -> verify -> memory.write``. Returns a streamable ``Answer``
        (ADR-0009), or abstains when no usable context survives reranking — or, when a
        relevance gate is configured, when it judges the surviving context too weak to
        answer (short-circuiting generation)."""
        sw = _Stopwatch()
        use_memory = self.memory is not None and query.user_id is not None

        # Corpus-scoped answers are cacheable; a per-user query depends on memory that the key can't
        # capture, so it bypasses the shared cache (ADR-0020/0031).
        cacheable = self.cache is not None and query.user_id is None
        cache_key: CacheKey | None = None
        if cacheable:
            assert self.cache is not None  # narrowed by `cacheable`
            cache_key = _cache_key(query)
            with sw.stage("cache"):
                live_ids = frozenset(await self.store.chunk_ids(query.tenant_id))
                hit = await self.cache.get(cache_key, live_ids)
            if hit is not None:
                # Served without embed/retrieve/generate; re-stamp the timing so the latency win is
                # what the harness measures, not the original generation's cost.
                return replace(hit, timings=sw.timings)

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

        if use_memory:
            # The user's relevant memories enter the same grounded-evidence channel as the corpus
            # (labelled by source), so the $0 model can use them and grounding still verifies them
            # — and a personal question can be answered even when retrieval found nothing. Memories
            # irrelevant to this query are filtered, so they never displace a corpus answer.
            ranked = _inject_memory(query.text, memories, ranked)

        if not ranked:
            await self._remember(sw, query)
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
                await self._remember(sw, query)
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

        await self._remember(sw, query)

        # Sum every billed component this answer touched: query embed, the relevance gate (if it
        # ran and bills), the generator, and the judge.
        generator_usage = [response.usage] if response.usage is not None else []
        usages = (*query_usages, *_drain(self.gate), *generator_usage, *_drain(self.judge))
        answer = Answer(
            text=outcome.text,
            citations=outcome.citations,
            grounding=outcome.report,
            timings=sw.timings,
            retrievals=tuple(ranked),
            usages=usages,
            abstained=_is_refusal(response.text),
        )
        if cacheable and cache_key is not None:
            assert self.cache is not None  # narrowed by `cacheable`
            # Gate the entry on exactly the chunks the answer *cited*: if any of them later changes
            # content (new hash) or is removed, the grounding gate evicts it (ADR-0031). An answer
            # that cited nothing has no grounding to gate on, so `put` declines to store it.
            grounded_ids = frozenset(c.evidence.chunk_id for c in answer.citations)
            await self.cache.put(cache_key, answer, grounded_ids)
        return answer

    async def _remember(self, sw: _Stopwatch, query: Query) -> None:
        """Learn durable facts from this turn (the write policy, ADR-0026), on every exit path.

        A no-op unless a memory store *and* an extractor are configured and the query is scoped to
        a user. It runs even when the pipeline abstains: a user stating a fact should be remembered
        whether or not this turn produced an answer."""
        if self.memory is None or self.extractor is None or query.user_id is None:
            return
        with sw.stage("memory.write"):
            turn = MemoryTurn(
                tenant_id=query.tenant_id,
                user_id=query.user_id,
                source=content_id(query.tenant_id, query.user_id, query.text),
                user_text=query.text,
            )
            items = await self.extractor.extract([turn])
            if items:
                await self.memory.write(query.tenant_id, query.user_id, items)


# --- stage helpers ----------------------------------------------------------------


def _cache_key(query: Query) -> CacheKey:
    """The exact-match cache identity for a query: tenant + normalized text + sorted filters.

    Normalizing (lowercase, whitespace-collapsed) collapses trivially-different phrasings of the
    same exact ask onto one entry; the tenant is part of the key so a hit can't cross tenants."""
    normalized = " ".join(query.text.lower().split())
    filters = tuple(sorted(query.filters.items()))
    return CacheKey(tenant_id=query.tenant_id, query=normalized, filters=filters)


def _understand(query: Query, memories: list[MemoryItem]) -> str:
    """Query-understanding seam (identity transform). Memory personalizes the answer through
    evidence injection (see ``_inject_memory``), not query rewriting; a memory-conditioned query
    expansion can still slot in here later. ``memories`` is accepted so the signature is stable."""
    return query.text


def _inject_memory(
    query_text: str, memories: list[MemoryItem], ranked: list[Retrieval]
) -> list[Retrieval]:
    """Prepend the memories relevant to *this* query as top-priority, labelled evidence (ADR-0026).

    Memory is the user's own stated facts, so a relevant memory leads the corpus evidence: a
    personal question is answered from what we know about the user. Relevance is gated by token
    overlap, so a memory unrelated to this query never displaces a corpus answer; principled
    recency-relevance-salience ranking is the next slice. Each kept memory becomes a ``Retrieval``
    over a synthetic chunk sourced ``memory/<turn>`` — the same grounded-evidence channel as the
    corpus, kept distinct by that label — so grounding verifies it like any other evidence and a
    remembered fact is never invented (``docs/memory.md`` §5)."""
    injected = [
        Retrieval(
            chunk=Chunk(
                id=item.id,
                doc_id=item.id,
                text=item.content,
                ordinal=0,
                start=0,
                end=len(item.content),
                source=f"memory/{item.source}",
                created_at=item.created_at,
            ),
            score=1.0 + item.salience,  # above corpus cosine scores: a relevant memory leads.
        )
        for item in memories
        if _overlap(query_text, item.content) > 0
    ]
    return injected + ranked


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _overlap(query_text: str, content: str) -> int:
    """Token overlap between a query and a memory's content — the relevance gate for injection."""
    return len(_tokens(query_text) & _tokens(content))


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
