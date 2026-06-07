"""The service facade, end to end on the $0 stack (Phase 5 slice 1).

These prove the two things the facade is responsible for: that the four operations map cleanly onto
the engine, and that the tenant/user boundary the store and memory namespace on is *enforced* — a
request for one tenant never sees another's corpus, and a request for one user never sees another's
memory. The same guarantees are then inherited by the HTTP transport (``test_asgi``).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from racore.eval.harness import demo_pipeline
from racore.service import (
    AnswerRequest,
    DocumentInput,
    IngestRequest,
    MemoryReadRequest,
    MemoryWriteRequest,
    RaCoreService,
    ServiceError,
    demo_service,
)

if TYPE_CHECKING:
    from pathlib import Path


def _ingest(service: RaCoreService, tenant_id: str, *docs: tuple[str, str]) -> None:
    asyncio.run(
        service.ingest(
            IngestRequest(
                tenant_id=tenant_id,
                documents=tuple(DocumentInput(text=text, source=source) for source, text in docs),
            )
        )
    )


def test_ingest_then_answer_is_grounded(tmp_path: Path) -> None:
    service = demo_service(tmp_path)
    _ingest(service, "acme", ("acme/eiffel", "The Eiffel Tower is in Paris."))

    answer = asyncio.run(
        service.answer(AnswerRequest(text="Where is the Eiffel Tower?", tenant_id="acme"))
    )

    assert not answer.abstained
    assert "Paris" in answer.text
    assert answer.grounding.is_grounded


def test_corpus_is_isolated_per_tenant(tmp_path: Path) -> None:
    service = demo_service(tmp_path)
    _ingest(service, "acme", ("acme/eiffel", "The Eiffel Tower is in Paris."))
    _ingest(service, "globex", ("globex/wall", "The Great Wall is in China."))

    # Answering inside acme can only ever reach acme's corpus...
    acme = asyncio.run(
        service.answer(AnswerRequest(text="Where is the Eiffel Tower?", tenant_id="acme"))
    )
    assert all(r.chunk.source.startswith("acme/") for r in acme.retrievals)

    # ...and globex cannot retrieve acme's document even when asked acme's question.
    globex = asyncio.run(
        service.answer(AnswerRequest(text="Where is the Eiffel Tower?", tenant_id="globex"))
    )
    assert all(r.chunk.source != "acme/eiffel" for r in globex.retrievals)
    assert "Paris" not in globex.text


def test_memory_is_isolated_per_user(tmp_path: Path) -> None:
    service = demo_service(tmp_path)
    stored = asyncio.run(
        service.write_memory(
            MemoryWriteRequest(tenant_id="t", user_id="alice", text="My name is Bob.")
        )
    )
    assert any("Bob" in item.content for item in stored)

    # Alice sees her own memory; Carol — a different user in the same tenant — sees nothing.
    alice = asyncio.run(
        service.read_memory(MemoryReadRequest(tenant_id="t", user_id="alice", query="name"))
    )
    carol = asyncio.run(
        service.read_memory(MemoryReadRequest(tenant_id="t", user_id="carol", query="name"))
    )
    assert any("Bob" in item.content for item in alice)
    assert carol == ()


def test_written_memory_personalizes_a_later_answer(tmp_path: Path) -> None:
    service = demo_service(tmp_path)
    asyncio.run(
        service.write_memory(MemoryWriteRequest(tenant_id="t", user_id="u", text="My name is Bob."))
    )

    answer = asyncio.run(
        service.answer(AnswerRequest(text="What is my name?", tenant_id="t", user_id="u"))
    )

    assert "Bob" in answer.text
    assert any(c.evidence.source.startswith("memory/") for c in answer.citations)


def test_memory_endpoints_require_a_memory_store() -> None:
    # A corpus-only pipeline (no memory configured) must refuse the memory operations with a
    # machine-readable code, not crash — so a transport can map it to a clean 4xx.
    service = RaCoreService(pipeline=demo_pipeline())
    with pytest.raises(ServiceError) as excinfo:
        asyncio.run(service.read_memory(MemoryReadRequest(tenant_id="t", user_id="u", query="x")))
    assert excinfo.value.code == "memory_not_configured"
