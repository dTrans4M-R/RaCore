"""Concrete adapters that satisfy the core ports.

The default set is deterministic and runs with **zero external spend** (ADR-0007): a
mock embedder, an in-memory vector store, a noop reranker, a fixed-window chunker, an
in-memory document source, a file-backed memory store, an extractive LLM, and the
deterministic entailment judges that back the grounding stage. ``AnthropicLLM`` is the
first **opt-in real provider** — importing it here is safe (the SDK loads lazily, only when
a client is built), so the default install stays dependency-free.
"""

from __future__ import annotations

from racore.adapters.chunkers import FixedWindowChunker
from racore.adapters.embeddings import MockEmbeddingProvider
from racore.adapters.judges import SubstringEntailmentJudge, TokenOverlapEntailmentJudge
from racore.adapters.llm import ExtractiveLLM
from racore.adapters.llm_anthropic import AnthropicConfig, AnthropicLLM
from racore.adapters.memory import FileMemoryStore
from racore.adapters.rerankers import NoopReranker
from racore.adapters.sources import InMemoryDocumentSource
from racore.adapters.vectorstores import InMemoryVectorStore

__all__ = [
    "AnthropicConfig",
    "AnthropicLLM",
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
