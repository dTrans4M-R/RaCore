"""Deterministic, $0 relevance gate — the abstention decision's strict floor.

``ThresholdRelevanceGate`` satisfies the ``RelevanceGate`` port using only the reranked
scores the retriever already produced, so it costs nothing and adds no latency. It is the
floor of a cascade: a semantic LLM gate (``racore.adapters.relevance_anthropic``) drops in
behind the same port for the cases this one can't decide.

The thresholds are deliberately **per-embedder**. A *lexical* embedder's cosine scores are
not calibrated to semantic relevance — a no-answer question that shares vocabulary with a
distractor can outscore a faithful paraphrase of a real answer (measured; ADR-0021) — so on
that stack the only honest setting catches the clearly-empty case and nothing more. A
*semantic* embedder (Voyage, or a local model) separates relevant from irrelevant cleanly,
which is where a score threshold earns its keep. The defaults are therefore **neutral**
(answer whenever anything was retrieved), so adding the gate never introduces a false refusal
on an uncalibrated stack; callers raise the thresholds for their embedder against the eval
harness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from racore.core.types import RelevanceCheck


class ThresholdRelevanceGate:
    """Answer iff the top retrieval clears an absolute floor *and* stands out from the pack.

    Two cheap signals, both read off the reranked scores at $0:

    * **Absolute floor** (``min_score``) — the best evidence must be at least this relevant,
      which catches the empty / weak-match case (nothing worth answering from).
    * **Margin** (``min_margin``) — the top score must beat the runner-up by this much. A flat
      distribution (everything scored about the same) means the retriever could not
      discriminate, so a confident answer isn't warranted even when the absolute score is fine.

    Both default to ``0.0`` — a neutral gate that abstains only on empty retrieval (which the
    pipeline already does), so it is safe to wire into any stack and tune upward per embedder.
    """

    def __init__(self, min_score: float = 0.0, min_margin: float = 0.0) -> None:
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        if min_margin < 0.0:
            raise ValueError("min_margin must be >= 0")
        self._min_score = min_score
        self._min_margin = min_margin

    async def should_answer(self, checks: list[RelevanceCheck]) -> list[bool]:
        return [self._answer(check) for check in checks]

    def _answer(self, check: RelevanceCheck) -> bool:
        if not check.retrievals:
            return False  # nothing retrieved -> nothing to answer from.
        top = check.retrievals[0].score
        if top < self._min_score:
            return False
        # A lone retrieval has no competition, so its margin is measured against zero.
        runner_up = check.retrievals[1].score if len(check.retrievals) > 1 else 0.0
        return (top - runner_up) >= self._min_margin
