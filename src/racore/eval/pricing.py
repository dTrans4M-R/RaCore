"""Per-model token pricing for the harness's cost accounting.

USD per million tokens (input, output) at base Claude API rates — no prompt-cache or Batch
discounts. Verified against platform.claude.com/docs/en/docs/about-claude/pricing on
2026-06-07; update when prices change. A model with no entry yields a cost of ``None`` so the
harness reports tokens only and never prints a misleading $0 for a paid run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from racore.core.types import TokenUsage

_PER_MTOK = 1_000_000.0

# Model-id prefix -> (input $/MTok, output $/MTok). Dated snapshots (e.g.
# ``claude-haiku-4-5-20251001``) match their family prefix; the longest matching prefix wins.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-3-5": (0.80, 4.0),
}


def cost_usd(usage: TokenUsage) -> float | None:
    """USD for one response's tokens, or ``None`` when the model has no known price."""
    price = _lookup(usage.model)
    if price is None:
        return None
    input_per_mtok, output_per_mtok = price
    return (usage.input_tokens * input_per_mtok + usage.output_tokens * output_per_mtok) / _PER_MTOK


def _lookup(model: str) -> tuple[float, float] | None:
    matches = [(key, price) for key, price in PRICES.items() if model.startswith(key)]
    if not matches:
        return None
    return max(matches, key=lambda item: len(item[0]))[1]  # longest (most specific) prefix wins.
