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
from racore.eval.metrics import recall_at_k
from racore.eval.pricing import cost_usd

if TYPE_CHECKING:
    from racore.core.ports import Evaluator
    from racore.core.types import TokenUsage


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
    retrieval_recall: float
    top_score: float
    relevant_sources: tuple[str, ...]
    retrieved_sources: tuple[str, ...]
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
    cost_per_answer_usd: float | None  # None when a paid run used a model with no known price.
    stage_millis: tuple[tuple[str, float], ...]
    tokens_in_per_answer: float = 0.0
    tokens_out_per_answer: float = 0.0
    per_case: tuple[CaseOutcome, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "cases": self.n_cases,
            "cost_per_answer_usd": self.cost_per_answer_usd,
            "tokens_per_answer": {
                "in": self.tokens_in_per_answer,
                "out": self.tokens_out_per_answer,
            },
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
                    "retrieval_recall": c.retrieval_recall,
                    "top_score": c.top_score,
                    "relevant_sources": list(c.relevant_sources),
                    "retrieved_sources": list(c.retrieved_sources),
                    "unsupported_claims": list(c.unsupported_claims),
                    "answer": c.answer,
                }
                for c in self.per_case
            ],
        }

    def render(self, *, verbose: bool = False) -> str:
        cost = (
            "n/a (unpriced model)"
            if self.cost_per_answer_usd is None
            else f"${self.cost_per_answer_usd:.6f}"
        )
        lines = [
            "RaCore - evaluation baseline",
            "============================",
            f"Cases: {self.n_cases}    Cost/answer: {cost}",
        ]
        if self.tokens_in_per_answer or self.tokens_out_per_answer:
            lines.append(
                f"Tokens/answer: in {self.tokens_in_per_answer:.0f}"
                f"   out {self.tokens_out_per_answer:.0f}"
            )
        lines += ["", "Quality"]
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
    per_answer_usages = [case.answer.usages for case in cases]
    return HarnessReport(
        n_cases=len(cases),
        results=results,
        latency_p50_ms=_percentile(latencies, 50.0),
        latency_p95_ms=_percentile(latencies, 95.0),
        latency_mean_ms=_mean(latencies),
        cost_per_answer_usd=_cost_per_answer(per_answer_usages),
        stage_millis=_stage_means(cases),
        tokens_in_per_answer=_mean([_sum_tokens(u, "input") for u in per_answer_usages]),
        tokens_out_per_answer=_mean([_sum_tokens(u, "output") for u in per_answer_usages]),
        per_case=tuple(_case_outcome(case) for case in cases),
    )


def _sum_tokens(usages: tuple[TokenUsage, ...], side: str) -> float:
    """Total input or output tokens an answer spent across all its billed components."""
    return float(sum(u.input_tokens if side == "input" else u.output_tokens for u in usages))


def _cost_per_answer(per_answer_usages: list[tuple[TokenUsage, ...]]) -> float | None:
    """Mean USD/answer, **summed across every billed component** (generator + any paid embedder or
    judge, ADR-0018). ``0.0`` when nothing was billed (the $0 stack reports no usage); ``None`` when
    any billed component's model has no price-table entry — never a fake $0 on a paid run."""
    answer_costs: list[float] = []
    for usages in per_answer_usages:
        total = 0.0
        for usage in usages:
            cost = cost_usd(usage)
            if cost is None:
                return None  # a billed component we can't price -> the whole figure is unknown
            total += cost
        answer_costs.append(total)
    return _mean(answer_costs) if answer_costs else 0.0


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
        retrieval_recall=recall_at_k(case),
        top_score=answer.retrievals[0].score if answer.retrievals else 0.0,
        relevant_sources=row.relevant_sources,
        retrieved_sources=tuple(r.chunk.source for r in answer.retrievals),
        unsupported_claims=answer.grounding.unsupported_claims,
        answer=answer.text,
    )


def _render_case(case: CaseOutcome) -> str:
    mark = "ok " if case.answer_correct else "BAD"
    head = (
        f"  [{case.id}] {mark} recall={case.retrieval_recall:.2f} faith={case.faithfulness:.2f}"
        f" cite={case.citation_correctness:.2f} abstain={'Y' if case.abstained else 'n'}"
        f" top={case.top_score:.3f}"
        f"  {case.question}"
    )
    body = [f"        A: {_truncate(case.answer, 140)}"]
    if case.answerable:
        missed = [s for s in case.relevant_sources if s not in case.retrieved_sources]
        top = case.retrieved_sources[0] if case.retrieved_sources else "(none)"
        body.append(f"        retrieved top: {top}   missed relevant: {missed or 'none'}")
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
