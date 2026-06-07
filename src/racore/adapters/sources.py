"""An in-memory document source: seed it with ``(source, text)`` pairs, fetch Documents.

Useful for tests, the eval corpus, and quick demos. Real connectors (PDF, SEC-EDGAR,
web, S3) implement the same ``fetch`` port. IDs are content hashes minted at fetch time
(ADR-0011), so the same text always yields the same document ID.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from racore.core.ids import content_id
from racore.core.types import Document

if TYPE_CHECKING:
    from collections.abc import Sequence


class InMemoryDocumentSource:
    """Holds raw ``(source, text)`` pairs and emits them as Documents on ``fetch()``.

    ``add`` accepts an optional ``created_at`` (epoch seconds) so a test or demo can seed
    documents with known ages and exercise the freshness path (ADR-0024) deterministically —
    real connectors supply the source's own publish/modified time here instead.
    """

    def __init__(self, documents: Sequence[tuple[str, str]] | None = None) -> None:
        # Each entry is (source, text, created_at): the constructor's pairs default to an unset
        # (0.0) timestamp; ``add(..., created_at=)`` sets a real one.
        self._raw: list[tuple[str, str, float]] = [(s, t, 0.0) for s, t in (documents or [])]

    def add(self, source: str, text: str, *, created_at: float = 0.0) -> None:
        self._raw.append((source, text, created_at))

    async def fetch(self) -> list[Document]:
        return [
            Document(id=content_id(source, text), text=text, source=source, created_at=created_at)
            for source, text, created_at in self._raw
        ]
