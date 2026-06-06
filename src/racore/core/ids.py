"""Deterministic, content-derived identifiers (ADR-0011).

Document and chunk IDs are a hash of their content, so re-ingesting unchanged
content is a no-op (same ID -> idempotent upsert) and incremental indexing works
later with no schema change. Centralised here so every ingest stage that mints an
ID uses the same function.
"""

from __future__ import annotations

import hashlib

_ID_LEN = 16  # 64 bits of SHA-256 hex — ample collision margin for a document corpus.


def content_id(*parts: str) -> str:
    """Return a stable ID derived from ``parts``.

    Parts are joined with a NUL separator so that ``("a", "bc")`` and ``("ab", "c")``
    hash differently — boundaries are part of the identity.
    """
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
    return digest[:_ID_LEN]
