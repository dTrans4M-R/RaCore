"""Ports — the plugin boundaries of the engine (``docs/architecture.md`` §4).

Each port is a ``Protocol`` (structural typing — an adapter satisfies a port by shape,
with no inheritance). The core depends only on these protocols; concrete providers live
under ``racore.adapters``.

Two rules are frozen here on purpose (ADR-0009), because changing them after adapters
exist would mean rewriting every adapter and caller:

* **Async-first** — every method that does (or may grow to do) I/O is ``async def``. A
  synchronous facade may wrap the async core later; never the reverse.
* **Batch-first** — I/O methods take and return *lists*, so batching is the default path
  and a single item is just a batch of one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from racore.core.types import (
        Answer,
        CacheKey,
        Chunk,
        ClaimCheck,
        Document,
        EmbeddedChunk,
        EvalCase,
        EvalResult,
        InputType,
        LLMRequest,
        LLMResponse,
        MemoryItem,
        MemoryTurn,
        RelevanceCheck,
        Retrieval,
        TokenUsage,
        Vector,
    )


class EmbeddingProvider(Protocol):
    """Map text to dense vectors. ``input_type`` lets a provider embed a query and a
    document asymmetrically when it supports that."""

    async def embed(self, texts: list[str], input_type: InputType) -> list[Vector]: ...


class VectorStore(Protocol):
    """Upsert embedded chunks and run filtered similarity search, namespaced per tenant.

    ``chunk_ids`` and ``delete`` make **incremental re-index** a first-class operation
    (ADR-0023): the pipeline reads what is already stored, embeds only chunks whose content
    hash is new, and (when pruning) deletes chunks that no longer appear in the source. Because
    IDs are content hashes (ADR-0011), this diff is exact — unchanged content keeps its ID and
    is skipped, changed content mints a new ID and replaces the old.
    """

    async def upsert(self, items: list[EmbeddedChunk], tenant_id: str) -> None: ...

    async def search(
        self,
        vector: Vector,
        k: int,
        tenant_id: str,
        filters: Mapping[str, str] | None = None,
    ) -> list[Retrieval]: ...

    async def chunk_ids(self, tenant_id: str) -> set[str]:
        """All chunk IDs currently stored for ``tenant_id`` (empty when the tenant is unknown)."""
        ...

    async def delete(self, chunk_ids: list[str], tenant_id: str) -> None:
        """Remove the given chunk IDs from ``tenant_id``. IDs not present are ignored."""
        ...


class Reranker(Protocol):
    """Re-order retrieval candidates by relevance to the query, keeping the top ``top_k``."""

    async def rerank(
        self, query: str, candidates: list[Retrieval], top_k: int
    ) -> list[Retrieval]: ...


class Chunker(Protocol):
    """Split documents into retrievable chunks. Batch-first (a list of documents in, a
    flat list of chunks out — each chunk carries its ``doc_id``). Async because real,
    structure-aware chunkers may call out to layout/OCR services."""

    async def chunk(self, documents: list[Document]) -> list[Chunk]: ...


class DocumentSource(Protocol):
    """Fetch and extract raw documents from somewhere (a path, a connector, the web)."""

    async def fetch(self) -> list[Document]: ...


class MemoryStore(Protocol):
    """Per-(tenant, user) memory: read the most relevant items, write new ones. Isolation
    at this boundary is a hard requirement (``docs/memory.md`` §7)."""

    async def read(self, tenant_id: str, user_id: str, query: str, k: int) -> list[MemoryItem]: ...

    async def write(self, tenant_id: str, user_id: str, items: list[MemoryItem]) -> None: ...


class MemoryExtractor(Protocol):
    """Propose durable per-user memories from a conversational turn — the *write policy*.

    What is worth remembering is a judgement (``docs/memory.md`` §3), so it is a port, not a
    hardcoded rule: the $0 ``RuleBasedMemoryExtractor`` is the floor (explicit self-statements),
    and a salience-judging, implicit-fact-aware LLM extractor drops in behind the same port as the
    "paid only improves" lever. Batch-first and async (ADR-0009). Each returned ``MemoryItem``
    carries its provenance ``source`` so a remembered fact is grounded, never invented."""

    async def extract(self, turns: list[MemoryTurn]) -> list[MemoryItem]: ...


class LLMProvider(Protocol):
    """Grounded generation. Batch-first; the return type carries cited markers so the
    pipeline can resolve them to citations. Streaming is additive later (ADR-0009)."""

    async def generate(self, requests: list[LLMRequest]) -> list[LLMResponse]: ...


class EntailmentJudge(Protocol):
    """Decide, per claim, whether its cited evidence entails (supports) it.

    The pluggable half of the grounding stage (``docs/evaluation.md`` §2). Batch-first — a
    list of checks in, one verdict per check out, in the same order — and ``async`` so a real
    LLM-judge for paraphrased support drops in behind the same port without touching callers
    (ADR-0009). The deterministic ``$0`` judges live in ``racore.adapters.judges``.
    """

    async def judge(self, checks: list[ClaimCheck]) -> list[bool]: ...


class RelevanceGate(Protocol):
    """Decide, per query, whether the retrieved evidence supports answering — or whether the
    system should abstain ("I don't know").

    The Relevance pillar's *abstain* clause made a pipeline decision rather than a prompt
    suffix (``docs/architecture.md`` §5). It is slotted between rerank and generate so an
    abstain **short-circuits the expensive generation stage** — a latency *and* cost win on
    no-evidence queries, not only a trust feature (ADR-0021).

    Batch-first and ``async`` (ADR-0009): the deterministic ``$0`` gate in
    ``racore.adapters.relevance`` makes a free arithmetic decision, while a semantic LLM gate
    drops in behind the same port — and is escalated to only for the uncertain cases — without
    touching callers. One bool per check, in order: ``True`` = answer, ``False`` = abstain.
    """

    async def should_answer(self, checks: list[RelevanceCheck]) -> list[bool]: ...


class AnswerCache(Protocol):
    """Serve a previously-computed answer to skip the expensive generate path on a repeat ask.

    The latency lever (``docs/latency.md`` §5a), made *safe* by gating on grounding rather than on
    similarity (ADR-0020). An entry is keyed by ``CacheKey`` (tenant + normalized question), so a
    hit can never cross the tenant boundary. The gate is the second argument to ``get``: the caller
    passes the chunk IDs *currently* in the store, and the cache returns the entry only if every
    chunk it was grounded on is still present. Because IDs are content hashes (ADR-0011), "still
    present" means "byte-identical" — so editing or removing the evidence an answer stood on mints a
    new ID and **auto-invalidates** the cached answer, while an unrelated ingest leaves it valid.
    No blunt whole-corpus flush is needed for correctness; ``invalidate`` is the explicit escape
    hatch (e.g. a config or policy change).

    Batch-free here by design: caching is a per-answer decision on the hot path, not an I/O batch.
    ``async`` so a shared/remote cache (Redis, a CDN edge) drops in behind the same port.
    """

    async def get(self, key: CacheKey, live_ids: frozenset[str]) -> Answer | None: ...

    async def put(self, key: CacheKey, answer: Answer, grounded_ids: frozenset[str]) -> None: ...

    async def invalidate(self, tenant_id: str) -> None: ...


class Evaluator(Protocol):
    """Score a set of answered cases on one metric. The eval harness runs several of
    these to cover the metric taxonomy in ``docs/evaluation.md`` §2."""

    name: str

    async def evaluate(self, cases: list[EvalCase]) -> EvalResult: ...


@runtime_checkable
class UsageReporter(Protocol):
    """*Optional* capability: an adapter that bills can report (and reset) the token usage of the
    calls it has made since the last drain.

    It is deliberately **not** part of the I/O ports — adding usage to ``embed`` / ``judge`` return
    types would break the frozen batch-first signatures (ADR-0009). Instead the pipeline drains its
    embedder and judge after each answer and folds the result into ``Answer.usages``, so cost/answer
    counts *every* paid component, not just the generator. The free ``$0`` adapters don't implement
    it, so they contribute nothing and the demo stays a true $0. ``runtime_checkable`` so the
    pipeline can ``isinstance``-test a component before draining it.
    """

    def drain_usage(self) -> list[TokenUsage]: ...
