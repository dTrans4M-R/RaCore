"""Request value types for the service facade — the transport-agnostic wire contract.

These are deliberately separate from ``racore.core.types``: a caller sends *text* and lets the
engine mint the content-hash IDs (ADR-0011) and assemble the ``Document``/``Query`` objects, so the
wire contract exposes only what a client actually provides — never an internal identifier. The
service maps these onto the core objects and returns the rich core results (``Answer``,
``IngestReport``, ``MemoryItem``) unchanged; turning those into JSON stays the transport layer's job
(``racore.service.asgi``). Frozen, slotted, stdlib-only — the same discipline as the core types.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentInput:
    """One document to ingest: its text and a provenance label. The content-hash ID is minted by
    the engine at fetch time (ADR-0011), so a caller never supplies — or can forge — one."""

    text: str
    source: str


@dataclass(frozen=True, slots=True)
class IngestRequest:
    """Ingest a batch of documents into one tenant's corpus.

    ``prune=True`` treats the batch as that tenant's *complete* corpus (the freshness reconcile,
    ADR-0023): chunks no longer present are deleted. The default is purely additive."""

    tenant_id: str
    documents: tuple[DocumentInput, ...]
    prune: bool = False


@dataclass(frozen=True, slots=True)
class AnswerRequest:
    """Answer a question, scoped to a tenant and (optionally) a user.

    A ``user_id`` opts the turn into the memory loop: the user's relevant memories are read in as
    grounded evidence and durable facts are learned back out (ADR-0026). Without one, it is a plain
    corpus answer — byte-identical to the pre-memory path."""

    text: str
    tenant_id: str = "default"
    user_id: str | None = None
    k: int = 5


@dataclass(frozen=True, slots=True)
class MemoryReadRequest:
    """Read a user's most relevant memories — the inspect/debug half of the memory surface."""

    tenant_id: str
    user_id: str
    query: str
    k: int = 5


@dataclass(frozen=True, slots=True)
class MemoryWriteRequest:
    """Teach the system a fact directly: the extractor proposes durable memories from ``text`` and
    the store persists them. This is the explicit-write counterpart to the implicit learning that
    happens on every ``answer`` turn (ADR-0026)."""

    tenant_id: str
    user_id: str
    text: str
