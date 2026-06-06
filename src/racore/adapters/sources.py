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
    """Holds raw ``(source, text)`` pairs and emits them as Documents on ``fetch()``."""

    def __init__(self, documents: Sequence[tuple[str, str]] | None = None) -> None:
        # Each entry is (source, text): source is the provenance label, text is the body.
        self._raw: list[tuple[str, str]] = list(documents or [])

    def add(self, source: str, text: str) -> None:
        self._raw.append((source, text))

    async def fetch(self) -> list[Document]:
        return [
            Document(id=content_id(source, text), text=text, source=source)
            for source, text in self._raw
        ]
