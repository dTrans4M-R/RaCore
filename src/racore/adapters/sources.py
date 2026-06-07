"""Document sources: an in-memory one for tests/demos, and a live filesystem connector.

Both implement the same ``fetch`` port; real remote connectors (PDF, SEC-EDGAR, web, S3) are
further adapters behind it. IDs are content hashes minted at fetch time (ADR-0011), so the same
text always yields the same document ID — which is what makes incremental re-index work (ADR-0023).
"""

from __future__ import annotations

from pathlib import Path
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


class FileSystemDocumentSource:
    """A live document source backed by a directory of text files.

    Each ``fetch()`` reflects the directory's *current* contents, so re-ingesting (``prune=True``)
    after files are added, edited, or removed picks up exactly the delta (ADR-0023). Every file's
    modified time (``st_mtime``) becomes its document's ``created_at``, so the age of the evidence
    behind an answer is **real**, not synthetic (ADR-0024) — this is what makes the freshness path
    concrete end-to-end. Text only; structured formats (PDF, HTML) are heavier adapters behind the
    same port. A missing root (or one that isn't a directory) yields an empty corpus, not an error,
    so a connector pointed at a not-yet-created folder degrades gracefully.
    """

    def __init__(
        self, root: str | Path, *, pattern: str = "**/*.txt", encoding: str = "utf-8"
    ) -> None:
        self._root = Path(root)
        self._pattern = pattern
        self._encoding = encoding

    async def fetch(self) -> list[Document]:
        if not self._root.is_dir():
            return []
        documents: list[Document] = []
        for path in sorted(self._root.glob(self._pattern)):
            if not path.is_file():
                continue
            text = path.read_text(encoding=self._encoding)
            source = path.relative_to(self._root).as_posix()  # stable, portable provenance label.
            documents.append(
                Document(
                    id=content_id(source, text),
                    text=text,
                    source=source,
                    created_at=path.stat().st_mtime,
                )
            )
        return documents
