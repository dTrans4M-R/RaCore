"""Concrete adapters that satisfy the core ports.

The Phase 0 set is deterministic and runs with **zero external spend** (ADR-0007): a
mock embedder, an in-memory vector store, a noop reranker, a fixed-window chunker, an
in-memory document source, a file-backed memory store, an extractive LLM, and the
deterministic entailment judges that back the grounding stage. Each is a drop-in target
for a real provider later (voyage/openai/pgvector/anthropic).
"""

from __future__ import annotations

from racore.adapters.chunkers import FixedWindowChunker
from racore.adapters.embeddings import MockEmbeddingProvider
from racore.adapters.judges import SubstringEntailmentJudge, TokenOverlapEntailmentJudge
from racore.adapters.llm import ExtractiveLLM
from racore.adapters.memory import FileMemoryStore
from racore.adapters.rerankers import NoopReranker
from racore.adapters.sources import InMemoryDocumentSource
from racore.adapters.vectorstores import InMemoryVectorStore

__all__ = [
    "ExtractiveLLM",
    "FileMemoryStore",
    "FixedWindowChunker",
    "InMemoryDocumentSource",
    "InMemoryVectorStore",
    "MockEmbeddingProvider",
    "NoopReranker",
    "SubstringEntailmentJudge",
    "TokenOverlapEntailmentJudge",
]
