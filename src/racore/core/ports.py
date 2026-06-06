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
        Retrieval,
        TokenUsage,
        Vector,
    )


class EmbeddingProvider(Protocol):
    """Map text to dense vectors. ``input_type`` lets a provider embed a query and a
    document asymmetrically when it supports that."""

    async def embed(self, texts: list[str], input_type: InputType) -> list[Vector]: ...


class VectorStore(Protocol):
    """Upsert embedded chunks and run filtered similarity search, namespaced per tenant."""

    async def upsert(self, items: list[EmbeddedChunk], tenant_id: str) -> None: ...

    async def search(
        self,
        vector: Vector,
        k: int,
        tenant_id: str,
        filters: Mapping[str, str] | None = None,
    ) -> list[Retrieval]: ...


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
