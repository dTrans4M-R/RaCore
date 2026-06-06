"""``python -m racore.eval`` — ingest the golden corpus and print the baseline.

Runnable definition-of-done: a thin end-to-end slice through the real ports that emits
quality *and* latency/cost numbers. The default (``--llm mock``) stays zero external spend;
``--llm anthropic`` swaps a real generator in behind the same port to stress-test grounding
(it paraphrases, so the strict substring judge will surface a faithfulness gap that
``--judge overlap`` can close). The provider/judge are selected here, not baked into the core.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
from typing import TYPE_CHECKING

from racore.adapters.judges import SubstringEntailmentJudge, TokenOverlapEntailmentJudge
from racore.eval.datasets import golden_dataset, golden_source
from racore.eval.harness import HarnessReport, demo_pipeline, run
from racore.eval.metrics import default_evaluators

if TYPE_CHECKING:
    from racore.core.pipeline import Pipeline

_JUDGES = {"substring": SubstringEntailmentJudge, "overlap": TokenOverlapEntailmentJudge}


def _build_pipeline(llm_kind: str, judge_kind: str, model: str | None = None) -> Pipeline:
    """Start from the $0 stack and swap only the LLM and/or judge that were selected."""
    pipeline = dataclasses.replace(demo_pipeline(), judge=_JUDGES[judge_kind]())
    if llm_kind == "anthropic":
        # Lazy import keeps the SDK optional — the mock path never reaches here.
        from racore.adapters.llm_anthropic import AnthropicConfig, AnthropicLLM

        config = AnthropicConfig(model=model) if model else AnthropicConfig()
        return dataclasses.replace(pipeline, llm=AnthropicLLM(config))
    return pipeline


async def _run(llm_kind: str, judge_kind: str, model: str | None = None) -> HarnessReport:
    pipeline = _build_pipeline(llm_kind, judge_kind, model)
    await pipeline.ingest(golden_source())
    return await run(pipeline, golden_dataset(), default_evaluators())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m racore.eval",
        description="Ingest the golden corpus and print the evaluation baseline.",
    )
    parser.add_argument(
        "--llm",
        choices=["mock", "anthropic"],
        default="mock",
        help="generator: 'mock' (extractive, $0, default) or 'anthropic' (real, opt-in).",
    )
    parser.add_argument(
        "--judge",
        choices=["substring", "overlap"],
        default="substring",
        help="entailment judge: 'substring' (exact, default) or 'overlap' (paraphrase-tolerant).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="override the provider model id (anthropic only), e.g. claude-haiku-4-5.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print per-case detail (question, answer, the claims that were unsupported).",
    )
    args = parser.parse_args()

    report = asyncio.run(_run(args.llm, args.judge, args.model))
    header = f"# llm={args.llm}  judge={args.judge}"
    if args.llm == "anthropic" and args.model:
        header += f"  model={args.model}"
    print(header)
    print(report.render(verbose=args.verbose))


if __name__ == "__main__":
    main()
