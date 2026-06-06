"""An extractive, deterministic LLM stand-in for the $0 demo path.

It does not generate free text — it selects the sentence from the top cited evidence that
best overlaps the question and returns it verbatim with a ``[1]`` citation. That keeps the
answer trivially faithful (the answer *is* a cited span), which is the right Phase 0
baseline: it isolates retrieval quality from generation quality. A real provider
(anthropic/openai) is a drop-in swap behind the same ``LLMProvider`` port.
"""

from __future__ import annotations

import re

from racore.core.types import LLMRequest, LLMResponse

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"\w+")

_NO_EVIDENCE = "I don't know — no evidence was provided."


class ExtractiveLLM:
    """Answers by quoting the best-matching sentence of the top evidence."""

    async def generate(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        return [self._answer_one(request) for request in requests]

    def _answer_one(self, request: LLMRequest) -> LLMResponse:
        evidences = request.context.evidences
        if not evidences:
            return LLMResponse(text=_NO_EVIDENCE, cited_markers=())
        sentence = _best_sentence(evidences[0].quote, request.query)
        return LLMResponse(text=f"{sentence} [1]", cited_markers=(1,))


def _best_sentence(quote: str, query: str) -> str:
    """Return the sentence of ``quote`` with the most token overlap with ``query``.

    Ties resolve to the earliest sentence, so the result is deterministic. The returned
    text is a verbatim slice of ``quote``, which is what makes the answer faithful.
    """
    sentences = [s for s in (part.strip() for part in _SENTENCE_SPLIT_RE.split(quote)) if s]
    if not sentences:
        return quote.strip()
    query_tokens = _tokens(query)
    return max(sentences, key=lambda sentence: len(query_tokens & _tokens(sentence)))


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))
