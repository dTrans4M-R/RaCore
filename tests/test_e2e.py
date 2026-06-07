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
    assert report.documents == 28
    assert report.chunks == 28  # each one-sentence doc fits in a single window.

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

    first = await pipeline.ingest(source)
    assert (first.added, first.unchanged, first.deleted) == (28, 0, 0)
    (probe,) = await pipeline.embedder.embed(["planet"], InputType.QUERY)
    before = await pipeline.store.search(probe, 100, "default")

    second = await pipeline.ingest(source)
    # Re-ingesting identical content is a true no-op: every chunk is unchanged, none re-embedded.
    assert (second.added, second.unchanged, second.deleted) == (0, 28, 0)
    after = await pipeline.store.search(probe, 100, "default")

    assert len(before) == len(after) == 28


def test_eval_baseline_reports_quality_and_ops() -> None:
    report = asyncio.run(_baseline())
    metrics = {result.name: result.score for result in report.results}

    # Grounding is perfect: the extractive $0 generator only ever quotes the evidence it
    # cites, so every claim is supported and every citation is right — regardless of whether
    # the *right* evidence was retrieved.
    assert metrics["grounding.faithfulness"] == 1.0
    assert metrics["grounding.citation_correctness"] == 1.0

    # Retrieval is now a real, measurable gap (this is the point of the harder corpus): on a
    # corpus with distractors and paraphrase-gap questions the lexical $0 retriever no longer
    # reaches every relevant doc (recall@k < 1), and even when it does, a distractor often
    # wins rank 1, so the extractive answer — which quotes the top hit — is sometimes wrong
    # (answer.correctness < 1). These are the numbers hybrid retrieval + reranking must beat.
    assert 0.9 <= metrics["retrieval.recall@k"] < 1.0
    assert 0.8 <= metrics["answer.correctness"] < 1.0

    # Rank-aware metrics expose what recall@k hides: the right doc is often retrieved but not
    # at rank 1 (a distractor wins), so nDCG@k and MRR sit *below* recall@k. This is the gap a
    # reranker closes — and which recall@k, being position-blind, cannot see.
    assert 0.0 < metrics["retrieval.ndcg@k"] < metrics["retrieval.recall@k"]
    assert 0.0 < metrics["retrieval.mrr"] < metrics["retrieval.recall@k"]

    # Ops: the $0 stack is free, and percentiles are ordered.
    assert report.n_cases == 17
    assert report.cost_per_answer_usd == 0.0
    assert report.latency_p95_ms >= report.latency_p50_ms
    assert report.stage_millis  # per-stage breakdown exists.

    # Known gap, surfaced not hidden: no abstention logic yet, so the negative controls
    # produce false answers. The robust abstention decision is Phase 2.
    assert metrics["refusal.accuracy"] < 1.0


def test_report_carries_per_case_detail() -> None:
    report = asyncio.run(_baseline())

    # One outcome per golden row, addressable by id, retained for drill-down.
    assert len(report.per_case) == 17
    assert {"q1", "n1"} <= {c.id for c in report.per_case}

    # Verbose render surfaces the per-case block; the default render stays compact.
    verbose = report.render(verbose=True)
    assert "Per-case" in verbose
    assert "[q1]" in verbose
    assert (
        "top=" in verbose
    )  # the top retrieval score is shown so threshold bands can be calibrated
    assert "Per-case" not in report.render(verbose=False)

    # The detail is machine-readable too (for JSON dumps / regression diffs).
    assert isinstance(report.to_dict()["per_case"], list)


async def _baseline() -> HarnessReport:
    pipeline = demo_pipeline()
    await pipeline.ingest(golden_source())
    return await run(pipeline, golden_dataset(), default_evaluators())
