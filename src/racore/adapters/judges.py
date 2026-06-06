"""Deterministic, $0 entailment judges for the grounding stage.

Each satisfies the ``EntailmentJudge`` port: given claims paired with the evidence they
cited, decide per claim whether that evidence supports it. Both judges are pure, local, and
free — a real LLM-judge for harder paraphrase/inference is a drop-in swap behind the same
port later.

* ``SubstringEntailmentJudge`` — exact: the claim must appear verbatim in a cited quote. It
  is the strict default, so the headline faithfulness number is never inflated by a lenient
  check.
* ``TokenOverlapEntailmentJudge`` — paraphrase-tolerant: most of the claim's *content* words
  must occur in one cited quote. It accepts reordering and filler that the substring check
  rejects, standing in for an LLM entailment judge at zero cost.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from racore.core.types import ClaimCheck

_TOKEN_RE = re.compile(r"\w+")

# Function words carry no support signal; counting them would let a claim "match" a quote on
# filler alone. Stripped before scoring token overlap (not used by the exact judge).
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
)


class SubstringEntailmentJudge:
    """Supported iff the normalized claim is a verbatim substring of a cited quote."""

    async def judge(self, checks: list[ClaimCheck]) -> list[bool]:
        return [
            any(_normalize(check.claim) in _normalize(quote) for quote in check.evidence)
            for check in checks
        ]


class TokenOverlapEntailmentJudge:
    """Supported iff a cited quote covers at least ``threshold`` of the claim's content words.

    Tolerates word order and filler differences, so a faithful paraphrase passes where exact
    substring matching would fail. ``threshold`` defaults to 0.8 — high enough that a quote
    must contain almost all of the claim's substance, not just share a few common words.
    """

    def __init__(self, threshold: float = 0.8) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self._threshold = threshold

    async def judge(self, checks: list[ClaimCheck]) -> list[bool]:
        return [self._supported(check) for check in checks]

    def _supported(self, check: ClaimCheck) -> bool:
        claim_tokens = _content_tokens(check.claim)
        if not claim_tokens:
            return False
        return any(
            len(claim_tokens & _content_tokens(quote)) / len(claim_tokens) >= self._threshold
            for quote in check.evidence
        )


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for substring comparison."""
    return " ".join(text.lower().split())


def _content_tokens(text: str) -> frozenset[str]:
    """Lowercased word tokens with stopwords removed."""
    return frozenset(_TOKEN_RE.findall(text.lower())) - _STOPWORDS
