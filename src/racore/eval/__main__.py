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
from racore.adapters.relevance import CascadeRelevanceGate, ThresholdRelevanceGate
from racore.eval.datasets import golden_dataset, golden_source
from racore.eval.harness import HarnessReport, demo_pipeline, run
from racore.eval.metrics import default_evaluators

if TYPE_CHECKING:
    from collections.abc import Callable

    from racore.core.pipeline import Pipeline
    from racore.core.ports import EntailmentJudge, RelevanceGate

_JUDGES: dict[str, Callable[[], EntailmentJudge]] = {
    "substring": SubstringEntailmentJudge,
    "overlap": TokenOverlapEntailmentJudge,
}


def _make_judge(judge_kind: str, model: str | None) -> EntailmentJudge:
    """Build the selected entailment judge. 'llm' is opt-in and lazily imports the SDK."""
    if judge_kind == "llm":
        from racore.adapters.judge_anthropic import AnthropicEntailmentJudge
        from racore.adapters.llm_anthropic import AnthropicConfig

        config = AnthropicConfig(model=model) if model else AnthropicConfig()
        return AnthropicEntailmentJudge(config)
    return _JUDGES[judge_kind]()


def _make_gate(
    gate_kind: str,
    model: str | None,
    min_score: float,
    margin: float,
    high: float,
) -> RelevanceGate | None:
    """Build the selected relevance gate. 'none' (default) leaves the pipeline ungated;
    'llm'/'cascade' are opt-in and lazily import the SDK."""
    if gate_kind == "none":
        return None
    if gate_kind == "threshold":
        return ThresholdRelevanceGate(min_score=min_score, min_margin=margin)
    from racore.adapters.llm_anthropic import AnthropicConfig
    from racore.adapters.relevance_anthropic import AnthropicRelevanceGate

    config = AnthropicConfig(model=model) if model else AnthropicConfig()
    llm_gate = AnthropicRelevanceGate(config)
    if gate_kind == "llm":
        return llm_gate
    # cascade: free score-band decisions, with the LLM gate only in the gray zone [min_score, high).
    return CascadeRelevanceGate(llm_gate, low=min_score, high=high)


def _build_pipeline(
    llm_kind: str,
    judge_kind: str,
    model: str | None = None,
    embedder_kind: str = "mock",
    embed_model: str | None = None,
    gate_kind: str = "none",
    gate_min_score: float = 0.0,
    gate_margin: float = 0.0,
    gate_high: float = 1.0,
) -> Pipeline:
    """Start from the $0 stack and swap only the embedder/LLM/judge/gate that were selected."""
    pipeline = dataclasses.replace(demo_pipeline(), judge=_make_judge(judge_kind, model))
    if embedder_kind == "voyage":
        # Lazy import keeps the SDK optional — the mock path never reaches here.
        from racore.adapters.embeddings_voyage import VoyageConfig, VoyageEmbeddingProvider

        vconfig = VoyageConfig(model=embed_model) if embed_model else VoyageConfig()
        pipeline = dataclasses.replace(pipeline, embedder=VoyageEmbeddingProvider(vconfig))
    if llm_kind == "anthropic":
        from racore.adapters.llm_anthropic import AnthropicConfig, AnthropicLLM

        config = AnthropicConfig(model=model) if model else AnthropicConfig()
        pipeline = dataclasses.replace(pipeline, llm=AnthropicLLM(config))
    gate = _make_gate(gate_kind, model, gate_min_score, gate_margin, gate_high)
    if gate is not None:
        pipeline = dataclasses.replace(pipeline, gate=gate)
    return pipeline


async def _run(
    llm_kind: str,
    judge_kind: str,
    model: str | None = None,
    embedder_kind: str = "mock",
    embed_model: str | None = None,
    gate_kind: str = "none",
    gate_min_score: float = 0.0,
    gate_margin: float = 0.0,
    gate_high: float = 1.0,
) -> HarnessReport:
    pipeline = _build_pipeline(
        llm_kind,
        judge_kind,
        model,
        embedder_kind,
        embed_model,
        gate_kind,
        gate_min_score,
        gate_margin,
        gate_high,
    )
    await pipeline.ingest(golden_source())
    return await run(pipeline, golden_dataset(), default_evaluators())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m racore.eval",
        description="Ingest the golden corpus and print the evaluation baseline.",
    )
    parser.add_argument(
        "--embedder",
        choices=["mock", "voyage"],
        default="mock",
        help="embedder: 'mock' (lexical, $0, default) or 'voyage' (real semantic, opt-in).",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help="override the embedding model id (voyage only), e.g. voyage-3-large.",
    )
    parser.add_argument(
        "--llm",
        choices=["mock", "anthropic"],
        default="mock",
        help="generator: 'mock' (extractive, $0, default) or 'anthropic' (real, opt-in).",
    )
    parser.add_argument(
        "--judge",
        choices=["substring", "overlap", "llm"],
        default="substring",
        help="entailment judge: 'substring' (exact, default), 'overlap' (paraphrase-tolerant), "
        "or 'llm' (semantic, opt-in — uses Claude to decide entailment).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="override the Anthropic model id (used by --llm anthropic, --judge llm, and the "
        "--gate llm/cascade gates), e.g. claude-haiku-4-5.",
    )
    parser.add_argument(
        "--gate",
        choices=["none", "threshold", "llm", "cascade"],
        default="none",
        help="relevance gate (proactive abstention): 'none' (default), 'threshold' ($0, "
        "score + margin), 'llm' (semantic, opt-in), or 'cascade' (free score bands + the LLM "
        "gate only in the gray zone).",
    )
    parser.add_argument(
        "--gate-min-score",
        type=float,
        default=0.0,
        help="relevance gate score floor (threshold gate) and cascade gray-zone lower bound.",
    )
    parser.add_argument(
        "--gate-margin",
        type=float,
        default=0.0,
        help="relevance gate: required margin of the top score over the runner-up (threshold).",
    )
    parser.add_argument(
        "--gate-high",
        type=float,
        default=1.0,
        help="cascade gray-zone upper bound: a top score >= this answers without a paid call.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print per-case detail (question, answer, the claims that were unsupported).",
    )
    args = parser.parse_args()

    report = asyncio.run(
        _run(
            args.llm,
            args.judge,
            args.model,
            args.embedder,
            args.embed_model,
            args.gate,
            args.gate_min_score,
            args.gate_margin,
            args.gate_high,
        )
    )
    header = f"# embedder={args.embedder}  llm={args.llm}  judge={args.judge}  gate={args.gate}"
    if args.embedder == "voyage" and args.embed_model:
        header += f"  embed_model={args.embed_model}"
    if args.llm == "anthropic" and args.model:
        header += f"  model={args.model}"
    if args.gate in {"threshold", "cascade"}:
        header += f"  gate_min_score={args.gate_min_score}"
    if args.gate == "threshold" and args.gate_margin:
        header += f"  gate_margin={args.gate_margin}"
    if args.gate == "cascade":
        header += f"  gate_high={args.gate_high}"
    print(header)
    print(report.render(verbose=args.verbose))


if __name__ == "__main__":
    main()
