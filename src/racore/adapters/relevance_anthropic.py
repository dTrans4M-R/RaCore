"""An opt-in LLM relevance gate (Claude) for the ``RelevanceGate`` port.

The deterministic ``ThresholdRelevanceGate`` decides from retrieval *scores*, which only
separate relevant from irrelevant when the embedder is semantic (ADR-0021). This adapter asks
the question directly — *does this evidence actually answer the query?* — on meaning, so it is
**embedder-independent**: it closes the refusal gap even on the lexical ``$0`` stack, where no
score threshold can. It is the cascade's expensive tier, escalated to only for the gray-zone
cases the cheap gate can't decide.

Mirrors ``AnthropicEntailmentJudge`` and reuses ``AnthropicLLM``'s client plumbing: optional
extra (``racore[anthropic]``), lazy SDK, a narrow injected client ``Protocol`` (offline-testable),
and ``UsageReporter`` so the gate's tokens are priced into cost/answer (ADR-0018). One adapter
behind the port — an OpenAI-compatible **local** LLM (Ollama, vLLM) is a future drop-in here.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from racore.adapters.llm_anthropic import (
    AnthropicConfig,
    _build_client,
    _extract_text,
    _extract_usage,
)

if TYPE_CHECKING:
    from racore.adapters.llm_anthropic import _Client
    from racore.core.types import RelevanceCheck, TokenUsage

# A one-word verdict keeps the call tiny (max_tokens below) and cheap. The instruction is
# strict: "answer" only when the evidence contains the answer, not when it is merely on-topic —
# otherwise the gate would wave through the near-miss distractors it exists to catch.
_GATE_SYSTEM = (
    "You are a strict relevance gate for a retrieval system. Decide whether the EVIDENCE "
    "contains the information needed to answer the QUESTION. Say it can be answered only if the "
    "evidence actually states the answer — not if it is merely on the same topic. Reply with "
    "exactly one word: ANSWER or ABSTAIN."
)
_ABSTAIN_RE = re.compile(r"\babstain\b", re.IGNORECASE)
_ANSWER_RE = re.compile(r"\banswer\b", re.IGNORECASE)


class AnthropicRelevanceGate:
    """Semantic answer-vs-abstain via Claude, behind the ``RelevanceGate`` port (batch-first).

    ``max_evidence`` caps how many top passages are shown to the model, bounding the per-query
    cost; the reranked order means the most relevant evidence is kept.
    """

    def __init__(
        self,
        config: AnthropicConfig | None = None,
        *,
        client: _Client | None = None,
        max_evidence: int = 5,
    ) -> None:
        self._config = config or AnthropicConfig()
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
        message = await client.messages.create(
            model=self._config.model,
            max_tokens=8,  # a one-word verdict; keeps the per-query cost minimal
            temperature=0.0,
            system=_GATE_SYSTEM,
            messages=[{"role": "user", "content": _render(check, self._max_evidence)}],
        )
        # Record the gate call's tokens so cost/answer counts it, not just the generator.
        self._usage.append(_extract_usage(message, self._config.model))
        return _should_answer(_extract_text(message))

    def drain_usage(self) -> list[TokenUsage]:
        """Usage accumulated since the last drain; resets on read (the ``UsageReporter`` port)."""
        drained, self._usage = self._usage, []
        return drained

    def _client_or_build(self) -> _Client:
        if self._client is None:
            self._client = _build_client(self._config)
        return self._client


def _render(check: RelevanceCheck, max_evidence: int) -> str:
    """Render the question and its top retrieved passages into the gate's user turn."""
    passages = "\n".join(f"- {r.chunk.text}" for r in check.retrievals[:max_evidence])
    return f"QUESTION: {check.query}\n\nEVIDENCE:\n{passages}"


def _should_answer(verdict: str) -> bool:
    """Parse the one-word verdict; default to abstain on anything ambiguous (the trust-safe
    direction — better a missed answer than a fabricated one)."""
    if _ABSTAIN_RE.search(verdict):
        return False
    return bool(_ANSWER_RE.search(verdict))
