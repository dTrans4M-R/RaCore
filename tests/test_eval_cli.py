"""The eval CLI's provider/judge selection (``python -m racore.eval --llm … --judge …``).

Pins the wiring in ``_build_pipeline`` without any external spend: selecting ``anthropic``
constructs the adapter but must *not* import the SDK (the SDK loads lazily, only when a
client is built), so this proves the selection path is safe on a clean install.
"""

from __future__ import annotations

from racore.adapters.judges import SubstringEntailmentJudge, TokenOverlapEntailmentJudge
from racore.adapters.llm import ExtractiveLLM
from racore.adapters.llm_anthropic import AnthropicLLM
from racore.eval.__main__ import _build_pipeline


def test_mock_selection_keeps_extractive_llm_and_picks_judge() -> None:
    pipeline = _build_pipeline("mock", "overlap")
    assert isinstance(pipeline.llm, ExtractiveLLM)
    assert isinstance(pipeline.judge, TokenOverlapEntailmentJudge)


def test_anthropic_selection_swaps_llm_without_importing_sdk() -> None:
    pipeline = _build_pipeline("anthropic", "substring", model="claude-haiku-4-5")
    assert isinstance(pipeline.llm, AnthropicLLM)
    assert isinstance(pipeline.judge, SubstringEntailmentJudge)
