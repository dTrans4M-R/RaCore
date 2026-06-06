"""Token + cost accounting: the price table, and the harness's per-answer cost/tokens.

The $0 stack reports no usage (cost stays a true 0.0); a real provider reports tokens that
the harness prices from the verified table. An unpriced model yields ``None`` — never a
misleading $0 on a paid run.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import TYPE_CHECKING

import pytest

from racore.core.types import LLMResponse, TokenUsage
from racore.eval import (
    HarnessReport,
    default_evaluators,
    demo_pipeline,
    golden_dataset,
    golden_source,
    run,
)
from racore.eval.pricing import cost_usd

if TYPE_CHECKING:
    from racore.core.types import LLMRequest


def test_haiku_cost_matches_published_rates() -> None:
    # Haiku 4.5 is $1/MTok input, $5/MTok output; a dated snapshot matches by prefix.
    usage = TokenUsage(
        input_tokens=1_000_000, output_tokens=1_000_000, model="claude-haiku-4-5-20251001"
    )
    assert cost_usd(usage) == pytest.approx(6.0)


def test_unknown_or_unset_model_is_unpriced() -> None:
    assert cost_usd(TokenUsage(10, 10, model="some-local-llm")) is None
    assert cost_usd(TokenUsage(10, 10)) is None  # model="" -> the $0 stack, unpriced


class _CostedLLM:
    """A stand-in that reports fixed token usage so the harness has real numbers to price."""

    async def generate(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        return [
            LLMResponse(
                text=f"{request.context.evidences[0].quote} [1]",
                cited_markers=(1,),
                usage=TokenUsage(input_tokens=100, output_tokens=20, model="claude-haiku-4-5"),
            )
            for request in requests
        ]


def test_harness_reports_real_cost_and_tokens() -> None:
    report = asyncio.run(_run_costed())

    # (100 input * $1 + 20 output * $5) / 1e6 = $0.0002 per answer.
    assert report.cost_per_answer_usd == pytest.approx(0.0002)
    assert report.tokens_in_per_answer == 100.0
    assert report.tokens_out_per_answer == 20.0

    rendered = report.render()
    assert "$0.000200" in rendered
    assert "Tokens/answer" in rendered


async def _run_costed() -> HarnessReport:
    pipeline = dataclasses.replace(demo_pipeline(), llm=_CostedLLM())
    await pipeline.ingest(golden_source())
    return await run(pipeline, golden_dataset(), default_evaluators())
