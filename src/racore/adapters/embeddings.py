"""A deterministic, dependency-free embedding provider for the $0 demo path.

It hashes each *distinct* token into a fixed-width binary bag-of-words vector and
L2-normalises it, so cosine similarity between a query and a chunk reflects their shared
vocabulary. Presence rather than term-frequency is deliberate: it stops a sentence that
repeats a stopword ("the … the … the") from dominating similarity. It is not semantic — it
is a stand-in that makes the pipeline runnable and the eval baseline reproducible. A real
model (voyage/openai/local-bge) is a drop-in replacement.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from racore.core.types import InputType, Vector

_TOKEN_RE = re.compile(r"\w+")


class MockEmbeddingProvider:
    """Hashing bag-of-words embedder. Deterministic across processes and runs."""

    def __init__(self, dim: int = 1024) -> None:
        if dim <= 0:
            raise ValueError("embedding dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str], input_type: InputType) -> list[Vector]:
        # input_type is ignored here — the mock treats queries and documents identically.
        # The parameter stays so real, asymmetric providers are a drop-in swap.
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> Vector:
        buckets = [0.0] * self._dim
        for token in set(_TOKEN_RE.findall(text.lower())):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dim
            buckets[index] = 1.0
        norm = math.sqrt(sum(value * value for value in buckets))
        if norm == 0.0:
            return tuple(buckets)
        return tuple(value / norm for value in buckets)
