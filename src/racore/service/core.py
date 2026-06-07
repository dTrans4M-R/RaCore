"""The service facade: one transport-agnostic object over the async pipeline (ADR-0009).

``RaCoreService`` is the synchronous-to-wire seam ADR-0009 anticipated ("a synchronous facade may
wrap the async core for simple embedding; never the reverse"). It is *not* a second pipeline: all
retrieval, grounding, and memory logic stays in ``racore.core``. What lives here is only
product-surface concern — mapping a wire request onto the core objects, scoping it to a tenant, and
the guard rails an external caller needs (memory endpoints require a memory store to be configured).

The same service backs every transport: the ASGI app (``racore.service.asgi``), an embedded import,
or a future CLI — so multi-tenancy and the memory loop are exercised once, here, and inherited by
all of them. Tenancy is not a feature added here; it is the ``tenant_id`` the store and memory
already namespace on (``InMemoryVectorStore._by_tenant``, one memory file per ``(tenant, user)``).
The service simply refuses to let a request cross that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from racore.adapters.chunkers import FixedWindowChunker
from racore.adapters.embeddings import MockEmbeddingProvider
from racore.adapters.judges import SubstringEntailmentJudge
from racore.adapters.llm import ExtractiveLLM
from racore.adapters.memory import FileMemoryStore
from racore.adapters.memory_extract import RuleBasedMemoryExtractor
from racore.adapters.rerankers import NoopReranker
from racore.adapters.sources import InMemoryDocumentSource
from racore.adapters.vectorstores import InMemoryVectorStore
from racore.core.ids import content_id
from racore.core.pipeline import Pipeline
from racore.core.types import MemoryTurn, Query

if TYPE_CHECKING:
    from pathlib import Path

    from racore.core.ports import MemoryStore
    from racore.core.types import Answer, IngestReport, MemoryItem
    from racore.service.types import (
        AnswerRequest,
        IngestRequest,
        MemoryReadRequest,
        MemoryWriteRequest,
    )


class ServiceError(Exception):
    """A request the service can't fulfil for a client-side reason (a transport maps it to a 4xx).

    ``code`` is a stable, machine-readable string so a transport can switch on it without
    string-matching the human message."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class RaCoreService:
    """Wraps a configured ``Pipeline`` and exposes the four product operations.

    Holds no state of its own — the pipeline's adapters hold the corpus and memory — so a single
    service instance is safe to share across concurrent requests (the async core does the I/O)."""

    pipeline: Pipeline

    async def ingest(self, request: IngestRequest) -> IngestReport:
        """Ingest a batch of documents into ``request.tenant_id``'s corpus (incremental)."""
        source = InMemoryDocumentSource([(doc.source, doc.text) for doc in request.documents])
        return await self.pipeline.ingest(source, request.tenant_id, prune=request.prune)

    async def answer(self, request: AnswerRequest) -> Answer:
        """Answer one question. Returns the streamable ``Answer`` (ADR-0009) so a transport can
        stream the text and let citations/grounding follow a beat later (ADR-0020)."""
        return await self.pipeline.answer(
            Query(
                text=request.text,
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                k=request.k,
            )
        )

    async def read_memory(self, request: MemoryReadRequest) -> tuple[MemoryItem, ...]:
        """Return a user's most relevant memories — only ever that ``(tenant, user)``'s file."""
        memory = self._require_memory()
        items = await memory.read(request.tenant_id, request.user_id, request.query, request.k)
        return tuple(items)

    async def write_memory(self, request: MemoryWriteRequest) -> tuple[MemoryItem, ...]:
        """Extract durable memories from ``request.text`` and persist them; return what was stored.

        Mirrors the pipeline's implicit write policy (ADR-0026) for an explicit "teach me this"
        call: the same extractor decides what is worth keeping, so an empty list back means the
        text stated nothing durable — not an error."""
        memory = self._require_memory()
        if self.pipeline.extractor is None:
            raise ServiceError("no memory extractor is configured", code="extractor_not_configured")
        turn = MemoryTurn(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            source=content_id(request.tenant_id, request.user_id, request.text),
            user_text=request.text,
        )
        items = await self.pipeline.extractor.extract([turn])
        await memory.write(request.tenant_id, request.user_id, items)
        return tuple(items)

    def _require_memory(self) -> MemoryStore:
        if self.pipeline.memory is None:
            raise ServiceError("memory is not configured", code="memory_not_configured")
        return self.pipeline.memory


def demo_service(memory_dir: Path) -> RaCoreService:
    """The default $0 stack wired with per-user memory — for tests, demos, and local runs.

    Deterministic and zero-spend (ADR-0007): the mock embedder, in-memory store, extractive LLM,
    strict substring judge, plus a file-backed memory store and the rule-based extractor so the
    memory endpoints work out of the box. Swapping in real providers (Voyage, Claude, pgvector) is
    a constructor change behind the same ports — never a fork of this facade."""
    pipeline = Pipeline(
        embedder=MockEmbeddingProvider(),
        store=InMemoryVectorStore(),
        reranker=NoopReranker(),
        chunker=FixedWindowChunker(),
        llm=ExtractiveLLM(),
        judge=SubstringEntailmentJudge(),
        memory=FileMemoryStore(memory_dir),
        extractor=RuleBasedMemoryExtractor(),
    )
    return RaCoreService(pipeline=pipeline)
