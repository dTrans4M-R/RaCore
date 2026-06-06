"""The evaluation harness: run a pipeline over a dataset and report quality + ops.

This is the moat made concrete (``docs/evaluation.md``): every answer is scored against the
golden set, and latency/cost are reported alongside quality so a regression in either is
visible. ``demo_pipeline()`` assembles the standard $0 adapter stack so the same wiring
backs both ``python -m racore.eval`` and the end-to-end test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from racore.adapters.chunkers import FixedWindowChunker
from racore.adapters.embeddings import MockEmbeddingProvider
from racore.adapters.judges import SubstringEntailmentJudge
from racore.adapters.llm import ExtractiveLLM
from racore.adapters.rerankers import NoopReranker
from racore.adapters.vectorstores import InMemoryVectorStore
from racore.core.pipeline import Pipeline
from racore.core.types import EvalCase, EvalResult, GoldenRow, Query

if TYPE_CHECKING:
    from racore.core.ports import Evaluator


def demo_pipeline() -> Pipeline:
    """The standard deterministic, zero-cost pipeline.

    Grounding uses the strict ``SubstringEntailmentJudge`` so the faithfulness baseline is
    never inflated by a lenient check; the paraphrase-tolerant judge is opt-in.
    """
    return Pipeline(
        embedder=MockEmbeddingProvider(),
        store=InMemoryVectorStore(),
        reranker=NoopReranker(),
        chunker=FixedWindowChunker(),
        llm=ExtractiveLLM(),
        judge=SubstringEntailmentJudge(),
    )


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """Per-row detail behind the aggregates, so a number like faithfulness=0.75 can be
    traced to the specific rows — and the specific unsupported claims — that produced it."""

    id: str
    question: str
    answerable: bool
    abstained: bool
    answer_correct: bool
    faithfulness: float
    citation_correctness: float
    unsupported_claims: tuple[str, ...]
    answer: str


@dataclass(frozen=True, slots=True)
class HarnessReport:
    """Aggregate outcome of a harness run: quality scores plus latency/cost, with per-case
    detail retained so a regression can be localized to the row that caused it."""

    n_cases: int
    results: tuple[EvalResult, ...]
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    cost_per_answer_usd: float
    stage_millis: tuple[tuple[str, float], ...]
    per_case: tuple[CaseOutcome, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "cases": self.n_cases,
            "cost_per_answer_usd": self.cost_per_answer_usd,
            "metrics": {
                r.name: {"score": r.score, "details": dict(r.details)} for r in self.results
            },
            "latency_ms": {
                "p50": self.latency_p50_ms,
                "p95": self.latency_p95_ms,
                "mean": self.latency_mean_ms,
            },
            "stage_ms": dict(self.stage_millis),
            "per_case": [
                {
                    "id": c.id,
                    "question": c.question,
                    "answerable": c.answerable,
                    "abstained": c.abstained,
                    "answer_correct": c.answer_correct,
                    "faithfulness": c.faithfulness,
                    "citation_correctness": c.citation_correctness,
                    "unsupported_claims": list(c.unsupported_claims),
                    "answer": c.answer,
                }
                for c in self.per_case
            ],
        }

    def render(self, *, verbose: bool = False) -> str:
        lines = [
            "RaCore - evaluation baseline",
            "============================",
            f"Cases: {self.n_cases}    Cost/answer: ${self.cost_per_answer_usd:.6f}",
            "",
            "Quality",
        ]
        for result in self.results:
            detail = "  ".join(f"{k}={v}" for k, v in result.details.items())
            lines.append(f"  {result.name:<32} {result.score:6.3f}   ({detail})")
        lines += [
            "",
            "Latency (ms)",
            f"  p50 {self.latency_p50_ms:7.3f}   p95 {self.latency_p95_ms:7.3f}"
            f"   mean {self.latency_mean_ms:7.3f}",
            "",
            "Per-stage mean (ms)",
        ]
        lines.extend(f"  {stage:<14} {millis:7.3f}" for stage, millis in self.stage_millis)
        if verbose and self.per_case:
            lines += ["", "Per-case (-v)"]
            lines.extend(_render_case(c) for c in self.per_case)
        return "\n".join(lines)


async def run(
    pipeline: Pipeline,
    dataset: list[GoldenRow],
    evaluators: list[Evaluator],
    tenant_id: str = "default",
) -> HarnessReport:
    """Answer every row, then score the run with each evaluator."""
    cases: list[EvalCase] = []
    for row in dataset:
        answer = await pipeline.answer(Query(text=row.question, tenant_id=tenant_id))
        cases.append(EvalCase(row=row, answer=answer))

    results = tuple([await evaluator.evaluate(cases) for evaluator in evaluators])

    latencies = sorted(case.answer.total_millis for case in cases)
    return HarnessReport(
        n_cases=len(cases),
        results=results,
        latency_p50_ms=_percentile(latencies, 50.0),
        latency_p95_ms=_percentile(latencies, 95.0),
        latency_mean_ms=_mean(latencies),
        cost_per_answer_usd=0.0,  # the Phase 0 stack makes no paid calls (ADR-0007).
        stage_millis=_stage_means(cases),
        per_case=tuple(_case_outcome(case) for case in cases),
    )


def _case_outcome(case: EvalCase) -> CaseOutcome:
    """Flatten one answered case into the per-row facts worth eyeballing."""
    row, answer = case.row, case.answer
    correct = (
        _normalize(row.expected_answer) in _normalize(answer.text)
        if row.answerable
        else answer.abstained  # a negative control is "correct" only when it abstains.
    )
    return CaseOutcome(
        id=row.id,
        question=row.question,
        answerable=row.answerable,
        abstained=answer.abstained,
        answer_correct=correct,
        faithfulness=answer.grounding.faithfulness,
        citation_correctness=answer.grounding.citation_correctness,
        unsupported_claims=answer.grounding.unsupported_claims,
        answer=answer.text,
    )


def _render_case(case: CaseOutcome) -> str:
    mark = "ok " if case.answer_correct else "BAD"
    head = (
        f"  [{case.id}] {mark} faith={case.faithfulness:.2f} cite={case.citation_correctness:.2f}"
        f" abstain={'Y' if case.abstained else 'n'}  {case.question}"
    )
    body = [f"        A: {_truncate(case.answer, 140)}"]
    body.extend(
        f"        - unsupported: {_truncate(claim, 120)}" for claim in case.unsupported_claims
    )
    return "\n".join([head, *body])


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _truncate(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


def _stage_means(cases: list[EvalCase]) -> tuple[tuple[str, float], ...]:
    """Mean duration per stage across all answered cases, in first-seen order."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for case in cases:
        for timing in case.answer.timings:
            totals[timing.stage] = totals.get(timing.stage, 0.0) + timing.millis
            counts[timing.stage] = counts.get(timing.stage, 0) + 1
    return tuple((stage, totals[stage] / counts[stage]) for stage in totals)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fraction


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
