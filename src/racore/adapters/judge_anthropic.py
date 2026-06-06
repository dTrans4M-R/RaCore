"""An opt-in LLM entailment judge (Claude) for the ``EntailmentJudge`` port.

The deterministic ``$0`` judges (substring, token-overlap) are *lexical*: they cannot verify a
faithful **paraphrase**. "Mercury is the world that sits closest to our star" is correctly grounded
by "...the closest to the Sun", but shares too few words for token overlap, so it is wrongly scored
unsupported — which *understates* the headline faithfulness metric under a paraphrasing model.
ADR-0012 left the ``EntailmentJudge`` port pluggable for exactly this; this adapter is the real
semantic judge that decides entailment instead of lexical overlap.

Mirrors ``AnthropicLLM`` (and reuses its client plumbing): optional extra (``racore[anthropic]``),
lazy SDK, narrow client ``Protocol`` (offline-testable), **one adapter behind the port** — not a
lock-in. An OpenAI/local judge is a future drop-in behind the same `EntailmentJudge` port.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from racore.adapters.llm_anthropic import AnthropicConfig, _build_client, _extract_text

if TYPE_CHECKING:
    from racore.adapters.llm_anthropic import _Client
    from racore.core.types import ClaimCheck

# One-word verdict so the call is tiny (max_tokens below) and cheap. The instruction is strict:
# "supported" must be entailment, not mere topical relatedness, so the judge stays honest.
_VERDICT_SYSTEM = (
    "You are a strict grounding judge. Decide whether the EVIDENCE supports the CLAIM. "
    "The claim is supported only if the evidence states it or directly entails it — not if it "
    "is merely related. Reply with exactly one word: SUPPORTED or UNSUPPORTED."
)
# "unsupported" contains "supported", so check the negative first and require word boundaries.
_UNSUPPORTED_RE = re.compile(r"\bunsupported\b", re.IGNORECASE)
_SUPPORTED_RE = re.compile(r"\bsupported\b", re.IGNORECASE)


class AnthropicEntailmentJudge:
    """Semantic entailment via Claude, behind the ``EntailmentJudge`` port (batch-first)."""

    def __init__(
        self, config: AnthropicConfig | None = None, *, client: _Client | None = None
    ) -> None:
        self._config = config or AnthropicConfig()
        self._client: _Client | None = client

    async def judge(self, checks: list[ClaimCheck]) -> list[bool]:
        # An uncited claim is unsupportable by construction (no evidence to entail it) — match the
        # deterministic judges, and skip the network/cost entirely if nothing was cited at all.
        if not any(check.evidence for check in checks):
            return [False] * len(checks)
        client = self._client_or_build()
        return list(await asyncio.gather(*(self._one(client, check) for check in checks)))

    async def _one(self, client: _Client, check: ClaimCheck) -> bool:
        if not check.evidence:
            return False  # uncited -> unsupported, no LLM call (no cost)
        message = await client.messages.create(
            model=self._config.model,
            max_tokens=8,  # a one-word verdict; keeps the per-claim cost minimal
            temperature=0.0,
            system=_VERDICT_SYSTEM,
            messages=[{"role": "user", "content": _render(check)}],
        )
        return _is_supported(_extract_text(message))

    def _client_or_build(self) -> _Client:
        if self._client is None:
            self._client = _build_client(self._config)
        return self._client


def _render(check: ClaimCheck) -> str:
    """Render one claim and its cited evidence into the judge's user turn."""
    evidence = "\n".join(f"- {quote}" for quote in check.evidence)
    return f"CLAIM: {check.claim}\n\nEVIDENCE:\n{evidence}"


def _is_supported(verdict: str) -> bool:
    """Parse the model's one-word verdict; default to unsupported on anything ambiguous."""
    if _UNSUPPORTED_RE.search(verdict):
        return False
    return bool(_SUPPORTED_RE.search(verdict))
