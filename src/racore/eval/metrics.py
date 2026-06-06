"""Evaluators — one per metric, each satisfying the ``Evaluator`` port.

They cover the layers of ``docs/evaluation.md`` §2 we can measure: retrieval (recall@k plus
the rank-aware nDCG@k / MRR), grounding faithfulness, citation correctness, answer correctness,
and refusal accuracy. Each returns a single ``EvalResult`` with a score in ``[0, 1]`` plus
supporting detail. Keeping retrieval, grounding, and answer separate is deliberate: a good
final answer can hide a broken retriever, and vice versa — and a *real* generator can hide it
twice over by reading past a bad rank-1, which is why the retriever gets rank-aware metrics of
its own rather than being judged through answer correctness.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from racore.core.types import EvalCase, EvalResult

if TYPE_CHECKING:
    from racore.core.ports import Evaluator


class RetrievalEvaluator:
    """Recall@k: of the sources relevant to a question, what fraction were retrieved.

    The mean over answerable rows. For a single-relevant-source row this is the classic
    hit@k (1.0 iff the source surfaced); for a multi-source row it is the fraction of the
    relevant set that surfaced. Recall@k answers *did the right documents get reached at
    all* — it is blind to rank, so a buried-but-present relevant doc still scores 1.0.
    Rank-position quality (nDCG@k / MRR) is the next retrieval slice; until then watch
    answer.correctness, which depends on the rank-1 result, to feel what recall@k hides.
    """

    name = "retrieval.recall@k"

    async def evaluate(self, cases: list[EvalCase]) -> EvalResult:
        recalls = [
            recall_at_k(case) for case in cases if case.row.answerable and case.row.relevant_sources
        ]
        return EvalResult(
            name=self.name,
            score=_mean(recalls),
            details={"answerable_cases": float(len(recalls))},
        )


class FaithfulnessEvaluator:
    """The headline metric: fraction of answer claims backed by a cited span."""

    name = "grounding.faithfulness"

    async def evaluate(self, cases: list[EvalCase]) -> EvalResult:
        scores = [case.answer.grounding.faithfulness for case in cases if not case.answer.abstained]
        faithfulness = _mean(scores)
        return EvalResult(
            name=self.name,
            score=faithfulness,
            details={
                "answered_cases": float(len(scores)),
                "unsupported_claim_rate": round(1.0 - faithfulness, 4),
            },
        )


class CitationCorrectnessEvaluator:
    """Of the citations an answer actually made, how many point at supporting evidence.

    Distinct from faithfulness: faithfulness penalizes uncited claims, while this scores only
    *cited* claims — it asks whether the citations that exist are right (``docs/evaluation.md``
    §2). Averaged over answered cases that cited at least one source.
    """

    name = "grounding.citation_correctness"

    async def evaluate(self, cases: list[EvalCase]) -> EvalResult:
        scored = [
            case.answer.grounding.citation_correctness
            for case in cases
            if not case.answer.abstained and case.answer.grounding.cited_claims > 0
        ]
        return EvalResult(
            name=self.name,
            score=_mean(scored),
            details={"cited_cases": float(len(scored))},
        )


class AnswerCorrectnessEvaluator:
    """Correctness: does the expected answer appear in the produced answer text?"""

    name = "answer.correctness"

    async def evaluate(self, cases: list[EvalCase]) -> EvalResult:
        correct = [
            _normalize(case.row.expected_answer) in _normalize(case.answer.text)
            for case in cases
            if case.row.answerable
        ]
        return EvalResult(
            name=self.name,
            score=_mean(correct),
            details={"answerable_cases": float(len(correct))},
        )


class RefusalEvaluator:
    """Refusal accuracy + the damaging false-answer-on-no-evidence rate."""

    name = "refusal.accuracy"

    async def evaluate(self, cases: list[EvalCase]) -> EvalResult:
        correct = [case.answer.abstained == (not case.row.answerable) for case in cases]
        false_answers = [not case.answer.abstained for case in cases if not case.row.answerable]
        return EvalResult(
            name=self.name,
            score=_mean(correct),
            details={
                "negative_controls": float(len(false_answers)),
                "false_answer_on_no_evidence_rate": _mean(false_answers, default=0.0),
            },
        )


class NDCGEvaluator:
    """nDCG@k: rank-weighted retrieval quality — credit for a relevant doc decays with its rank.

    Where recall@k is blind to position (a relevant doc at rank 5 scores the same as at rank 1),
    nDCG discounts each relevant hit by ``log2(rank + 1)`` and normalises against the ideal
    ranking. Moving the right doc *up* raises the score — so this is the metric a reranker
    improves and recall@k cannot see. Mean over answerable rows with ≥1 relevant source.
    """

    name = "retrieval.ndcg@k"

    async def evaluate(self, cases: list[EvalCase]) -> EvalResult:
        scores = [
            ndcg_at_k(case) for case in cases if case.row.answerable and case.row.relevant_sources
        ]
        return EvalResult(
            name=self.name,
            score=_mean(scores),
            details={"answerable_cases": float(len(scores))},
        )


class MRREvaluator:
    """Mean Reciprocal Rank: ``1 / rank`` of the *first* relevant doc retrieved (0 if none).

    A pure top-of-list signal — exactly what a generator that leans on the rank-1 result
    depends on. Mean over answerable rows with ≥1 relevant source.
    """

    name = "retrieval.mrr"

    async def evaluate(self, cases: list[EvalCase]) -> EvalResult:
        scores = [
            reciprocal_rank(case)
            for case in cases
            if case.row.answerable and case.row.relevant_sources
        ]
        return EvalResult(
            name=self.name,
            score=_mean(scores),
            details={"answerable_cases": float(len(scores))},
        )


def default_evaluators() -> list[Evaluator]:
    """The standard evaluator panel: retrieval (recall + rank-aware), grounding, answer, refusal."""
    return [
        RetrievalEvaluator(),
        NDCGEvaluator(),
        MRREvaluator(),
        FaithfulnessEvaluator(),
        CitationCorrectnessEvaluator(),
        AnswerCorrectnessEvaluator(),
        RefusalEvaluator(),
    ]


def recall_at_k(case: EvalCase) -> float:
    """Fraction of a row's relevant sources that appear anywhere in its retrieved set.

    1.0 when the row has no relevant sources (a negative control), so callers can score it
    uniformly; in practice the evaluator filters those out before averaging.
    """
    relevant = set(case.row.relevant_sources)
    if not relevant:
        return 1.0
    retrieved = {r.chunk.source for r in case.answer.retrievals}
    return len(relevant & retrieved) / len(relevant)


def ndcg_at_k(case: EvalCase) -> float:
    """Normalised discounted cumulative gain over the retrieved list (binary relevance).

    DCG sums ``1 / log2(rank + 1)`` for each retrieved doc that is relevant; IDCG is the same
    sum for the best possible ranking (every relevant doc packed at the top, capped at the
    number of retrieved slots). 1.0 for a row with no relevant sources; 0.0 when nothing
    relevant was retrieved.
    """
    relevant = set(case.row.relevant_sources)
    if not relevant:
        return 1.0
    retrieved = [r.chunk.source for r in case.answer.retrievals]
    dcg = sum(1.0 / math.log2(rank + 2) for rank, src in enumerate(retrieved) if src in relevant)
    ideal_hits = min(len(relevant), len(retrieved))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def reciprocal_rank(case: EvalCase) -> float:
    """``1 / rank`` (1-based) of the first relevant doc in the retrieved list; 0.0 if none."""
    relevant = set(case.row.relevant_sources)
    for rank, retrieval in enumerate(case.answer.retrievals, start=1):
        if retrieval.chunk.source in relevant:
            return 1.0 / rank
    return 0.0


def _mean(values: list[bool] | list[float], default: float = 1.0) -> float:
    """Mean of ``values``; ``default`` when empty (vacuously true by default)."""
    if not values:
        return default
    return sum(float(value) for value in values) / len(values)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
