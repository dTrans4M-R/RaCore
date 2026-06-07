"""Relevance gate: the proactive abstention decision (Phase 2, ADR-0021).

Two things are covered: the deterministic ``ThresholdRelevanceGate``'s signals (an absolute
score floor plus a margin over the runner-up), and the pipeline short-circuit — an abstain
must skip the expensive ``generate``/``verify`` stages, which is the latency win, not only a
trust feature.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from racore.adapters.relevance import ThresholdRelevanceGate
from racore.core.types import Chunk, Query, RelevanceCheck, Retrieval
from racore.eval import demo_pipeline, golden_source


def _check(*scores: float) -> RelevanceCheck:
    """A check whose retrievals carry the given scores in descending-rank order."""
    retrievals = tuple(
        Retrieval(
            chunk=Chunk(id=f"s{i}", doc_id=f"s{i}", text=f"text {i}", ordinal=0, start=0, end=0),
            score=score,
        )
        for i, score in enumerate(scores)
    )
    return RelevanceCheck(query="q", retrievals=retrievals)


async def _decide(gate: ThresholdRelevanceGate, check: RelevanceCheck) -> bool:
    (decision,) = await gate.should_answer([check])
    return decision


# --- the deterministic gate's signals ---------------------------------------------


def test_empty_retrieval_abstains() -> None:
    gate = ThresholdRelevanceGate()
    assert asyncio.run(_decide(gate, RelevanceCheck(query="q", retrievals=()))) is False


def test_neutral_defaults_answer_whenever_something_retrieved() -> None:
    # Defaults (0, 0): the gate never introduces a false refusal — it abstains only on empty
    # retrieval, so it is safe to wire into any (even uncalibrated) stack.
    gate = ThresholdRelevanceGate()
    assert asyncio.run(_decide(gate, _check(0.01))) is True


def test_absolute_floor_abstains_below_threshold() -> None:
    gate = ThresholdRelevanceGate(min_score=0.5)
    assert asyncio.run(_decide(gate, _check(0.40, 0.10))) is False
    assert asyncio.run(_decide(gate, _check(0.60, 0.10))) is True


def test_margin_abstains_on_a_flat_distribution() -> None:
    # The top clears the floor, but the runner-up is right behind it: the retriever could not
    # discriminate, so a confident answer isn't warranted even though the score is fine.
    gate = ThresholdRelevanceGate(min_score=0.5, min_margin=0.1)
    assert asyncio.run(_decide(gate, _check(0.60, 0.58))) is False
    assert asyncio.run(_decide(gate, _check(0.60, 0.45))) is True


def test_lone_retrieval_margin_is_measured_against_zero() -> None:
    # With no competition the margin is the top score itself.
    gate = ThresholdRelevanceGate(min_score=0.3, min_margin=0.2)
    assert asyncio.run(_decide(gate, _check(0.60))) is True


def test_batch_order_is_preserved() -> None:
    gate = ThresholdRelevanceGate(min_score=0.5)
    decisions = asyncio.run(gate.should_answer([_check(0.6), _check(0.4), _check(0.9)]))
    assert decisions == [True, False, True]


def test_invalid_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError):
        ThresholdRelevanceGate(min_score=1.5)
    with pytest.raises(ValueError):
        ThresholdRelevanceGate(min_margin=-0.1)


# --- the pipeline short-circuit (the latency win) ---------------------------------


def test_gate_short_circuits_generation() -> None:
    asyncio.run(_gate_short_circuits_generation())


async def _gate_short_circuits_generation() -> None:
    # A floor above any real score forces an abstain on a genuinely retrieved result.
    pipeline = dataclasses.replace(demo_pipeline(), gate=ThresholdRelevanceGate(min_score=0.99))
    await pipeline.ingest(golden_source())

    answer = await pipeline.answer(Query(text="Which is the smallest planet?"))

    assert answer.abstained
    assert answer.citations == ()
    # Retrieval DID run and is retained for eval — the gate abstained, not empty retrieval.
    assert answer.retrievals
    stages = {timing.stage for timing in answer.timings}
    assert "relevance" in stages
    # The win: the ~1.5s generation (and verify) never ran.
    assert "generate" not in stages
    assert "verify" not in stages


def test_permissive_gate_answers_normally() -> None:
    asyncio.run(_permissive_gate_answers_normally())


async def _permissive_gate_answers_normally() -> None:
    # A present-but-neutral gate must not disturb the happy path.
    pipeline = dataclasses.replace(demo_pipeline(), gate=ThresholdRelevanceGate())
    await pipeline.ingest(golden_source())

    answer = await pipeline.answer(Query(text="Which is the smallest planet?"))

    assert not answer.abstained
    assert "Mercury" in answer.text
    stages = {timing.stage for timing in answer.timings}
    assert {"relevance", "generate", "verify"} <= stages
