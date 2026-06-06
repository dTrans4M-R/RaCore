"""Evaluators — one per metric, each satisfying the ``Evaluator`` port.

They cover the layers of ``docs/evaluation.md`` §2 we can measure: retrieval hit-rate,
grounding faithfulness, citation correctness, answer correctness, and refusal accuracy. Each
returns a single ``EvalResult`` with a score in ``[0, 1]`` plus supporting detail. Keeping
retrieval, grounding, and answer separate is deliberate: a good final answer can hide a
broken retriever, and vice versa.
"""

from __future__ import annotations

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


def default_evaluators() -> list[Evaluator]:
    """The standard Phase 0 evaluator panel."""
    return [
        RetrievalEvaluator(),
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


def _mean(values: list[bool] | list[float], default: float = 1.0) -> float:
    """Mean of ``values``; ``default`` when empty (vacuously true by default)."""
    if not values:
        return default
    return sum(float(value) for value in values) / len(values)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
