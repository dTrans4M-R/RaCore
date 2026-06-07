"""Provider-neutral pieces shared by every LLM ``RelevanceGate`` adapter.

The gate's *verdict* is decided by two things only: the instruction it gives the model and how
it reads the reply. Those must be **identical** across providers — an Anthropic gate and a local
OpenAI-compatible gate that disagreed on the same evidence would be a silent correctness bug. So
the prompt, the rendering, and the one-word parse live here once; each adapter supplies only its
own client call and token accounting (which *are* provider-specific). This abstraction earns its
place now that a second concrete gate exists (ADR-0022), not in anticipation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from racore.core.types import RelevanceCheck

# A one-word verdict keeps the call tiny (the adapters cap max_tokens) and cheap. The instruction
# is strict: "answer" only when the evidence contains the answer, not when it is merely on-topic —
# otherwise the gate would wave through the near-miss distractors it exists to catch.
GATE_SYSTEM = (
    "You are a strict relevance gate for a retrieval system. Decide whether the EVIDENCE "
    "contains the information needed to answer the QUESTION. Say it can be answered only if the "
    "evidence actually states the answer — not if it is merely on the same topic. Reply with "
    "exactly one word: ANSWER or ABSTAIN."
)
_ABSTAIN_RE = re.compile(r"\babstain\b", re.IGNORECASE)
_ANSWER_RE = re.compile(r"\banswer\b", re.IGNORECASE)


def render_gate_prompt(check: RelevanceCheck, max_evidence: int) -> str:
    """Render the question and its top retrieved passages into the gate's user turn.

    ``max_evidence`` caps how many top passages are shown, bounding per-query cost; the reranked
    order means the most relevant evidence is the evidence kept.
    """
    passages = "\n".join(f"- {r.chunk.text}" for r in check.retrievals[:max_evidence])
    return f"QUESTION: {check.query}\n\nEVIDENCE:\n{passages}"


def parse_verdict(verdict: str) -> bool:
    """Parse the one-word verdict; default to abstain on anything ambiguous (the trust-safe
    direction — better a missed answer than a fabricated one)."""
    if _ABSTAIN_RE.search(verdict):
        return False
    return bool(_ANSWER_RE.search(verdict))
