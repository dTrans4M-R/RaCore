"""A grounding-gated in-memory answer cache (ADR-0020, ADR-0031).

This is the always-safe latency lever from ``docs/latency.md`` §5a, implemented so that
correctness is structural, not a tuning knob. The cache keys an answer by ``(tenant, normalized
question, filters)`` — the exact-match tier — and stores alongside each entry the set of chunk IDs
the answer was *grounded on*. On lookup it serves the entry only if every one of those chunks is
still present in the live store.

Because chunk IDs are content hashes (ADR-0011), "still present" is byte-exact: editing or removing
the evidence an answer stood on changes its hash, so the cached answer **auto-invalidates** the
moment its grounding moves — while an unrelated ingest, which touches none of those chunks, leaves
the entry valid. That is the whole thesis of grounding-gated caching: gate on whether the answer
*still holds*, not on whether a new question merely *looks similar*. The classic semantic-cache
traps (negation, specificity) are a different axis — candidate selection by embedding similarity —
left to an opt-in tier once a semantic embedder makes its threshold measurable (ADR-0031).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from racore.core.types import Answer, CacheKey


class GroundingGatedCache:
    """An in-memory ``AnswerCache`` whose entries are valid only while their grounding survives."""

    def __init__(self) -> None:
        # key -> (answer, the chunk IDs it was grounded on). The dev/$0 store; a Redis/edge adapter
        # mirrors the same port.
        self._entries: dict[CacheKey, tuple[Answer, frozenset[str]]] = {}

    async def get(self, key: CacheKey, live_ids: frozenset[str]) -> Answer | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        answer, grounded_ids = entry
        if grounded_ids <= live_ids:
            return answer  # every chunk this answer stood on is still present, unchanged.
        # The evidence moved out from under it (edited or removed): it no longer holds. Evict so the
        # next ask regenerates against the current corpus.
        del self._entries[key]
        return None

    async def put(self, key: CacheKey, answer: Answer, grounded_ids: frozenset[str]) -> None:
        # Cache only an answer that actually stands on current evidence. An abstain, or an answer
        # with no grounding to gate on, is never stored: there would be nothing to invalidate it,
        # and (for an abstain) it would wrongly keep saying "I don't know" after the corpus gains
        # the answer.
        if not grounded_ids or answer.abstained:
            return
        self._entries[key] = (answer, grounded_ids)

    async def invalidate(self, tenant_id: str) -> None:
        """Drop every entry for a tenant — the explicit flush for changes the grounding gate can't
        see (a policy/config change, a model swap). Routine corpus edits don't need it."""
        self._entries = {
            key: value for key, value in self._entries.items() if key.tenant_id != tenant_id
        }
