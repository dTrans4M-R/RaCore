"""Rank-aware retrieval metrics: nDCG@k and MRR see what recall@k cannot.

Recall@k is blind to position — a relevant doc at rank 5 scores like one at rank 1. These
tests pin that nDCG@k and MRR fall when the right doc is buried, which is the signal a reranker
improves (and the reason a real generator's answer.correctness can't stand in for retrieval
quality — a capable reader answers correctly even from a bad rank-1). Plain sync helpers are
tested directly; the evaluator's filtering is driven via ``asyncio.run`` (CLAUDE.md §4).
"""

from __future__ import annotations

import asyncio
import math

import pytest

from racore.core.types import Answer, Chunk, EvalCase, GoldenRow, GroundingReport, Retrieval
from racore.eval.metrics import NDCGEvaluator, ndcg_at_k, recall_at_k, reciprocal_rank


def _case(relevant: tuple[str, ...], retrieved: tuple[str, ...]) -> EvalCase:
    """An EvalCase whose answer retrieved ``retrieved`` (in rank order) for a row whose ground
    truth is ``relevant``. Only the source labels and their order matter to these metrics."""
    retrievals = tuple(
        Retrieval(
            chunk=Chunk(id=s, doc_id=s, text="", ordinal=0, start=0, end=0, source=s), score=1.0
        )
        for s in retrieved
    )
    answer = Answer(
        text="",
        citations=(),
        grounding=GroundingReport(supported_claims=(), unsupported_claims=()),
        timings=(),
        retrievals=retrievals,
    )
    return EvalCase(row=GoldenRow("x", "q?", "a", relevant), answer=answer)


def test_rank1_is_perfect_on_every_metric() -> None:
    case = _case(("a",), ("a", "b", "c"))
    assert recall_at_k(case) == 1.0
    assert reciprocal_rank(case) == 1.0
    assert ndcg_at_k(case) == 1.0


def test_burying_the_relevant_doc_drops_rank_metrics_but_not_recall() -> None:
    case = _case(("a",), ("b", "a", "c"))  # the right doc is present, but at rank 2
    assert recall_at_k(case) == 1.0  # recall@k can't tell — it only checks presence
    assert reciprocal_rank(case) == pytest.approx(0.5)  # 1 / rank
    assert ndcg_at_k(case) == pytest.approx(1.0 / math.log2(3))  # discounted to rank 2


def test_missing_relevant_doc_is_zero_on_rank_metrics() -> None:
    case = _case(("a",), ("b", "c"))
    assert recall_at_k(case) == 0.0
    assert reciprocal_rank(case) == 0.0
    assert ndcg_at_k(case) == 0.0


def test_multi_source_separates_recall_from_rank() -> None:
    # Two relevant docs; only one was retrieved, but it landed at rank 1.
    case = _case(("a", "b"), ("a", "x", "y"))
    assert recall_at_k(case) == 0.5  # half the relevant set surfaced
    assert reciprocal_rank(case) == 1.0  # the first hit is still at the top
    # nDCG normalises against the ideal (both relevant packed at the top), so one top hit out
    # of two is partial credit — a different lens than recall's 0.5.
    idcg = 1.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(case) == pytest.approx(1.0 / idcg)


def test_evaluator_excludes_negative_controls() -> None:
    asyncio.run(_evaluator_excludes_controls())


async def _evaluator_excludes_controls() -> None:
    cases = [
        _case(("a",), ("a", "b")),  # answerable, relevant doc at rank 1
        EvalCase(  # negative control: no relevant sources -> not scored
            row=GoldenRow("n", "q?", "", (), answerable=False),
            answer=_case((), ("a",)).answer,
        ),
    ]
    result = await NDCGEvaluator().evaluate(cases)
    assert result.details["answerable_cases"] == 1.0  # only the answerable row counted
    assert result.score == 1.0
