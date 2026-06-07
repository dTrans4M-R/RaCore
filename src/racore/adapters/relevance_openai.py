"""A local-or-hosted LLM relevance gate over the OpenAI-compatible protocol.

The semantic sibling of ``AnthropicRelevanceGate``: same port (``RelevanceGate``), same verbatim
prompt and verdict-parse (the shared ``_relevance_llm`` helpers, so the two gates can never
disagree on the same evidence), but it talks chat-completions — so the *same* adapter runs against
a **local** Ollama/vLLM/LM Studio model ($0, private) or the hosted OpenAI API, chosen only by
``base_url``. That makes the embedder-independent abstention check available with **zero per-call
spend** when self-hosted: the cheap-but-smart middle tier of the cascade (ADR-0022).

Lazy SDK (``racore[openai]``), an injected client ``Protocol`` (offline-testable), and
``UsageReporter`` so any hosted tokens are priced into cost/answer (ADR-0018). Mirrors
``AnthropicRelevanceGate`` structurally; only the client call shape differs.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from racore.adapters._relevance_llm import GATE_SYSTEM, parse_verdict, render_gate_prompt
from racore.adapters.llm_openai import (
    OpenAIConfig,
    _build_client,
    _extract_text,
    _extract_usage,
)

if TYPE_CHECKING:
    from racore.adapters.llm_openai import _Client
    from racore.core.types import RelevanceCheck, TokenUsage


class OpenAIRelevanceGate:
    """Semantic answer-vs-abstain via an OpenAI-compatible model, behind the ``RelevanceGate``
    port (batch-first). Local by default; hosted by config.

    ``max_evidence`` caps how many top passages are shown to the model, bounding the per-query
    cost; the reranked order means the most relevant evidence is kept.
    """

    def __init__(
        self,
        config: OpenAIConfig | None = None,
        *,
        client: _Client | None = None,
        max_evidence: int = 5,
    ) -> None:
        self._config = config or OpenAIConfig()
        self._client: _Client | None = client
        self._usage: list[TokenUsage] = []
        self._max_evidence = max_evidence

    async def should_answer(self, checks: list[RelevanceCheck]) -> list[bool]:
        # Empty retrieval is an abstain by construction — skip the network/cost for those.
        if not any(check.retrievals for check in checks):
            return [False] * len(checks)
        client = self._client_or_build()
        return list(await asyncio.gather(*(self._one(client, check) for check in checks)))

    async def _one(self, client: _Client, check: RelevanceCheck) -> bool:
        if not check.retrievals:
            return False  # nothing retrieved -> abstain, no LLM call (no cost)
        response = await client.chat.completions.create(
            model=self._config.model,
            max_tokens=8,  # a one-word verdict; keeps the per-query cost minimal
            temperature=0.0,
            messages=[
                {"role": "system", "content": GATE_SYSTEM},
                {"role": "user", "content": render_gate_prompt(check, self._max_evidence)},
            ],
        )
        # Record the gate call's tokens so cost/answer counts it (zero for a self-hosted model).
        self._usage.append(_extract_usage(response, self._config.model))
        return parse_verdict(_extract_text(response))

    def drain_usage(self) -> list[TokenUsage]:
        """Usage accumulated since the last drain; resets on read (the ``UsageReporter`` port)."""
        drained, self._usage = self._usage, []
        return drained

    def _client_or_build(self) -> _Client:
        if self._client is None:
            self._client = _build_client(self._config)
        return self._client
