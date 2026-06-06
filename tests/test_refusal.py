"""Refusal recording: a model "I don't know" is logged as an abstention (ADR-0013).

Measurement integrity — a correct refusal on a no-evidence question must not be scored as an
ungrounded answer (which dragged faithfulness *and* refusal accuracy down before). Phase 2
replaces the phrase heuristic with a real abstention decision.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import TYPE_CHECKING

from racore.adapters.sources import InMemoryDocumentSource
from racore.core.pipeline import _is_refusal
from racore.core.types import GoldenRow, LLMResponse, Query
from racore.eval import (
    HarnessReport,
    default_evaluators,
    demo_pipeline,
    golden_source,
    run,
)

if TYPE_CHECKING:
    from racore.core.types import LLMRequest


def test_is_refusal_detects_the_mandated_phrase() -> None:
    assert _is_refusal("I don't know. The evidence does not cover Neptune.")
    assert _is_refusal("I don't know.")
    assert _is_refusal("  I don't know — no supporting evidence.")
    # A real answer is not a refusal, even when it talks about knowing.
    assert not _is_refusal("Mercury is the smallest planet [1].")
    assert not _is_refusal("We know Mercury is closest to the Sun [1].")


class _RefusingLLM:
    async def generate(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        return [
            LLMResponse(text="I don't know. The evidence does not cover that.", cited_markers=())
            for _ in requests
        ]


def test_pipeline_records_a_refusal_as_abstention() -> None:
    asyncio.run(_pipeline_refusal())


async def _pipeline_refusal() -> None:
    pipeline = dataclasses.replace(demo_pipeline(), llm=_RefusingLLM())
    await pipeline.ingest(golden_source())

    answer = await pipeline.answer(Query(text="How many rings does Neptune have?"))

    # Recorded as abstention even though retrieval returned (irrelevant) chunks.
    assert answer.abstained


# A focused fixture: retrieval here is unambiguous, so this test exercises refusal
# *recording* — not retrieval quality. (The harder golden set in datasets.py, where
# retrieval is deliberately imperfect, is exercised by the eval-baseline tests.)
_SIMPLE_CORPUS = (
    ("planets/mercury", "Mercury is the smallest planet in the Solar System."),
    ("planets/jupiter", "Jupiter is the largest planet in the Solar System."),
)
_SIMPLE_ROWS = (
    GoldenRow("q1", "Which is the smallest planet?", "Mercury", ("planets/mercury",)),
    GoldenRow("q2", "What is the largest planet?", "Jupiter", ("planets/jupiter",)),
    GoldenRow("n1", "How many rings does Neptune have?", "", (), answerable=False),
    GoldenRow("n2", "In what year did the first Moon landing happen?", "", (), answerable=False),
)


class _ScriptedLLM:
    """Answers from evidence, but refuses on the fixture's two no-evidence questions."""

    async def generate(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        responses: list[LLMResponse] = []
        for request in requests:
            query = request.query.lower()
            if "neptune" in query or "moon landing" in query:
                responses.append(
                    LLMResponse(
                        text="I don't know. The evidence does not cover that.", cited_markers=()
                    )
                )
            else:
                quote = request.context.evidences[0].quote
                responses.append(LLMResponse(text=f"{quote} [1]", cited_markers=(1,)))
        return responses


def test_recorded_refusals_make_faithfulness_and_refusal_honest() -> None:
    report = asyncio.run(_scripted_run())
    metrics = {result.name: result.score for result in report.results}

    # Negative controls now abstain -> excluded from faithfulness, credited by refusal.
    assert metrics["grounding.faithfulness"] == 1.0
    assert metrics["refusal.accuracy"] == 1.0
    assert metrics["answer.correctness"] == 1.0


async def _scripted_run() -> HarnessReport:
    pipeline = dataclasses.replace(demo_pipeline(), llm=_ScriptedLLM())
    await pipeline.ingest(InMemoryDocumentSource(_SIMPLE_CORPUS))
    return await run(pipeline, list(_SIMPLE_ROWS), default_evaluators())
