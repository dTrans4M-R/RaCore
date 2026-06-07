"""Per-request observability events from the service layer (Phase 5 slice 4).

A test ``Observer`` collects events; the assertions pin that each operation emits one well-formed
event carrying the operational metrics an operator slices on — cache hit-rate, abstain rate, tokens,
grounding — per tenant. The point is that these fall out of the per-stage timings the pipeline
already records, so observability cost the core nothing.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from racore.service import (
    AnswerRequest,
    DocumentInput,
    IngestRequest,
    MemoryWriteRequest,
    ServiceEvent,
    demo_service,
)

if TYPE_CHECKING:
    from pathlib import Path


class _Collector:
    """A test sink that just remembers every event."""

    def __init__(self) -> None:
        self.events: list[ServiceEvent] = []

    def observe(self, event: ServiceEvent) -> None:
        self.events.append(event)


def test_each_operation_emits_one_event(tmp_path: Path) -> None:
    async def go() -> None:
        sink = _Collector()
        service = demo_service(tmp_path, observer=sink)

        await service.ingest(
            IngestRequest(
                tenant_id="acme",
                documents=(
                    DocumentInput(text="The Eiffel Tower is in Paris.", source="acme/eiffel"),
                ),
            )
        )
        await service.answer(AnswerRequest(text="Where is the Eiffel Tower?", tenant_id="acme"))
        await service.write_memory(
            MemoryWriteRequest(tenant_id="acme", user_id="u", text="My name is Bob.")
        )

        operations = [e.operation for e in sink.events]
        assert operations == ["ingest", "answer", "write_memory"]
        assert all(e.tenant_id == "acme" for e in sink.events)
        assert all(e.request_id for e in sink.events)  # every event is traceable

    asyncio.run(go())


def test_answer_event_reports_grounding_and_tokens(tmp_path: Path) -> None:
    async def go() -> None:
        sink = _Collector()
        service = demo_service(tmp_path, observer=sink)
        await service.ingest(
            IngestRequest(
                tenant_id="t",
                documents=(DocumentInput(text="Mercury is closest to the Sun.", source="sky"),),
            )
        )
        await service.answer(AnswerRequest(text="What is closest to the Sun?", tenant_id="t"))

        event = next(e for e in sink.events if e.operation == "answer")
        detail = event.detail
        assert detail["abstained"] is False
        assert detail["grounded"] is True
        assert detail["cache_hit"] is False  # cold answer
        n_citations = detail["n_citations"]
        assert isinstance(n_citations, int) and n_citations >= 1
        # The $0 stack bills nothing, so token counts are present and zero — a true $0, not a guess.
        assert detail["tokens_in"] == 0
        assert detail["tokens_out"] == 0

    asyncio.run(go())


def test_cache_hit_is_visible_in_the_event(tmp_path: Path) -> None:
    async def go() -> None:
        sink = _Collector()
        service = demo_service(tmp_path, observer=sink)
        await service.ingest(
            IngestRequest(
                tenant_id="t",
                documents=(DocumentInput(text="Mercury is closest to the Sun.", source="sky"),),
            )
        )
        ask = AnswerRequest(text="What is closest to the Sun?", tenant_id="t")
        await service.answer(ask)  # cold
        await service.answer(ask)  # warm

        answers = [e for e in sink.events if e.operation == "answer"]
        assert answers[0].detail["cache_hit"] is False
        assert answers[1].detail["cache_hit"] is True

    asyncio.run(go())


def test_abstain_is_reported(tmp_path: Path) -> None:
    async def go() -> None:
        sink = _Collector()
        service = demo_service(tmp_path, observer=sink)  # empty corpus
        await service.answer(AnswerRequest(text="What is closest to the Sun?", tenant_id="t"))

        event = next(e for e in sink.events if e.operation == "answer")
        assert event.detail["abstained"] is True

    asyncio.run(go())


def test_event_serializes_to_a_flat_dict(tmp_path: Path) -> None:
    async def go() -> None:
        sink = _Collector()
        service = demo_service(tmp_path, observer=sink)
        await service.ingest(
            IngestRequest(
                tenant_id="t",
                documents=(DocumentInput(text="Mercury is closest to the Sun.", source="sky"),),
            )
        )

        event = sink.events[0]
        flat = event.to_dict()
        # The op-specific detail is flattened alongside the envelope fields, ready for one log line.
        assert flat["operation"] == "ingest"
        assert flat["tenant_id"] == "t"
        assert flat["added"] == 1

    asyncio.run(go())


def test_no_observer_means_no_emission(tmp_path: Path) -> None:
    # The default service has no observer; answering must work and emit nothing (and not error).
    async def go() -> None:
        service = demo_service(tmp_path)
        assert service.observer is None
        result = await service.answer(AnswerRequest(text="anything?", tenant_id="t"))
        assert result.abstained  # empty corpus, but the point is it ran cleanly with no sink

    asyncio.run(go())
