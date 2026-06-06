"""The Phase 0 end-to-end test: ingest -> ask -> answer with a citation -> baseline.

This is the definition-of-done in ``docs/roadmap.md`` made executable. Tests are plain
sync functions that drive the async pipeline via ``asyncio.run`` so the suite needs no
async-test plugin (keeping the dependency surface small, CLAUDE.md §4).
"""

from __future__ import annotations

import asyncio

from racore.core.types import InputType, Query
from racore.eval import (
    HarnessReport,
    default_evaluators,
    demo_pipeline,
    golden_dataset,
    golden_source,
    run,
)


def test_ingest_then_answer_with_citation() -> None:
    asyncio.run(_ingest_then_answer_with_citation())


async def _ingest_then_answer_with_citation() -> None:
    pipeline = demo_pipeline()

    report = await pipeline.ingest(golden_source())
    assert report.documents == 7
    assert report.chunks == 7  # each one-sentence doc fits in a single window.

    answer = await pipeline.answer(Query(text="Which is the smallest planet?"))

    # The answer is real, on-topic, and not an abstention.
    assert not answer.abstained
    assert "Mercury" in answer.text

    # It carries at least one citation that resolves to the verbatim retrieved span.
    assert answer.citations
    citation = answer.citations[0]
    assert answer.retrievals
    assert citation.evidence.quote == answer.retrievals[0].chunk.text
    assert citation.evidence.doc_id == answer.retrievals[0].chunk.doc_id

    # Every claim is grounded (the extractive answer is a cited span), and the citation it
    # made is correct.
    assert answer.grounding.is_grounded
    assert answer.grounding.faithfulness == 1.0
    assert answer.grounding.citation_correctness == 1.0

    # Per-stage timing is present (ADR-0010).
    stages = {timing.stage for timing in answer.timings}
    assert {"embed", "retrieve", "rerank", "generate", "verify"} <= stages
    assert answer.total_millis >= 0.0

    # The Answer is streamable (ADR-0009): deltas reconstruct the text exactly.
    streamed = "".join([token async for token in answer.stream()])
    assert streamed == answer.text


def test_reingest_is_idempotent() -> None:
    asyncio.run(_reingest_is_idempotent())


async def _reingest_is_idempotent() -> None:
    # Content-hash IDs (ADR-0011): re-ingesting identical content must not duplicate.
    pipeline = demo_pipeline()
    source = golden_source()

    await pipeline.ingest(source)
    (probe,) = await pipeline.embedder.embed(["planet"], InputType.QUERY)
    before = await pipeline.store.search(probe, 100, "default")

    await pipeline.ingest(source)
    after = await pipeline.store.search(probe, 100, "default")

    assert len(before) == len(after) == 7


def test_eval_baseline_reports_quality_and_ops() -> None:
    report = asyncio.run(_baseline())
    metrics = {result.name: result.score for result in report.results}

    # Quality: retrieval, grounding, and answer are all solid on the golden set.
    assert metrics["retrieval.hit@k"] == 1.0
    assert metrics["grounding.faithfulness"] == 1.0
    assert metrics["grounding.citation_correctness"] == 1.0
    assert metrics["answer.correctness"] == 1.0

    # Ops: the $0 stack is free, and percentiles are ordered.
    assert report.n_cases == 8
    assert report.cost_per_answer_usd == 0.0
    assert report.latency_p95_ms >= report.latency_p50_ms
    assert report.stage_millis  # per-stage breakdown exists.

    # Known Phase 0 gap, surfaced not hidden: no abstention logic yet, so the negative
    # controls produce false answers. Phase 2 closes this.
    assert metrics["refusal.accuracy"] < 1.0


async def _baseline() -> HarnessReport:
    pipeline = demo_pipeline()
    await pipeline.ingest(golden_source())
    return await run(pipeline, golden_dataset(), default_evaluators())
