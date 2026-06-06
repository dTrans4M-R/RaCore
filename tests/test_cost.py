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
from racore.eval.harness import _cost_per_answer
from racore.eval.pricing import cost_usd

if TYPE_CHECKING:
    from racore.core.types import InputType, LLMRequest, Vector


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


# --- component cost accounting (ADR-0018): cost counts every paid component, not just the LLM ---


def test_voyage_embeddings_are_priced_input_only() -> None:
    # voyage-3.5 = $0.06 / MTok input, output-free; longest-prefix beats the bare "voyage-3".
    assert cost_usd(TokenUsage(1_000_000, 0, "voyage-3.5")) == pytest.approx(0.06)
    assert cost_usd(TokenUsage(1_000_000, 0, "voyage-3-large")) == pytest.approx(0.18)


def test_cost_per_answer_sums_every_billed_component() -> None:
    generator = TokenUsage(100, 20, "claude-haiku-4-5")  # $0.0002
    embedder = TokenUsage(1_000_000, 0, "voyage-3.5")  # $0.06
    assert _cost_per_answer([(generator, embedder)]) == pytest.approx(0.0002 + 0.06)


def test_an_unpriced_billed_component_makes_cost_unknown_not_zero() -> None:
    priced = TokenUsage(100, 20, "claude-haiku-4-5")
    unpriced = TokenUsage(1_000, 0, "some-local-embedder")
    # The whole figure is None (reported as n/a) — never a misleading partial $0 on a paid run.
    assert _cost_per_answer([(priced, unpriced)]) is None


class _UsageEmbedder:
    """A mock embedder that also reports token usage (the UsageReporter port)."""

    def __init__(self) -> None:
        self._usage: list[TokenUsage] = []

    async def embed(self, texts: list[str], input_type: InputType) -> list[Vector]:
        self._usage.append(TokenUsage(input_tokens=len(texts), output_tokens=0, model="voyage-3.5"))
        return [
            (1.0, 0.0) for _ in texts
        ]  # constant 2-d vectors; retrieval quality is irrelevant here

    def drain_usage(self) -> list[TokenUsage]:
        drained, self._usage = self._usage, []
        return drained


def test_pipeline_folds_embedder_and_generator_usage_into_cost() -> None:
    report = asyncio.run(_run_embedder_and_generator())

    # Per answer: generator (100 in / 20 out) + one query-embed token, summed across components.
    assert report.tokens_in_per_answer == 101.0
    assert report.tokens_out_per_answer == 20.0
    # Cost now includes the embedder, so it is strictly above the generator-only $0.0002 — the
    # misleading $0/understatement is gone.
    assert report.cost_per_answer_usd is not None
    assert report.cost_per_answer_usd > 0.0002


async def _run_embedder_and_generator() -> HarnessReport:
    pipeline = dataclasses.replace(demo_pipeline(), embedder=_UsageEmbedder(), llm=_CostedLLM())
    await pipeline.ingest(golden_source())
    return await run(pipeline, golden_dataset(), default_evaluators())
