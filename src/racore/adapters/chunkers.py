"""A fixed-window chunker that splits on word boundaries and tracks exact char offsets.

Offsets matter: a chunk's ``start``/``end`` index back into the source document, so a
retrieval can be quoted verbatim and cited (``docs/architecture.md`` §3). A structural,
page/section-aware chunker is a later drop-in; this one keeps Phase 0 honest and simple.
"""

from __future__ import annotations

import re

from racore.core.ids import content_id
from racore.core.types import Chunk, Document

_WORD_RE = re.compile(r"\S+")


class FixedWindowChunker:
    """Sliding window of ``window`` words with ``overlap`` words of carry-over."""

    def __init__(self, window: int = 60, overlap: int = 12) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if not 0 <= overlap < window:
            raise ValueError("overlap must be in [0, window)")
        self._window = window
        self._overlap = overlap
        self._step = window - overlap

    async def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._chunk_one(document))
        return chunks

    def _chunk_one(self, document: Document) -> list[Chunk]:
        spans = [(m.start(), m.end()) for m in _WORD_RE.finditer(document.text)]
        if not spans:
            return []

        chunks: list[Chunk] = []
        for ordinal, i in enumerate(range(0, len(spans), self._step)):
            window = spans[i : i + self._window]
            start = window[0][0]
            end = window[-1][1]
            text = document.text[start:end]
            chunks.append(
                Chunk(
                    id=content_id(document.id, str(ordinal), text),
                    doc_id=document.id,
                    text=text,
                    ordinal=ordinal,
                    start=start,
                    end=end,
                    source=document.source,
                )
            )
            if i + self._window >= len(spans):
                break  # last window already reached the end; avoid a trailing overlap-only chunk.
        return chunks
