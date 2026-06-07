"""Concrete adapters that satisfy the core ports.

The default set is deterministic and runs with **zero external spend** (ADR-0007): a
mock embedder, an in-memory vector store, a noop reranker, a fixed-window chunker, an
in-memory document source, a file-backed memory store, an extractive LLM, and the
deterministic entailment judges that back the grounding stage. ``AnthropicLLM`` (generation) and
``VoyageEmbeddingProvider`` (embeddings) are the **opt-in real providers** — importing them here is
safe (each SDK loads lazily, only when a client is built), so the default install stays
dependency-free.
"""

from __future__ import annotations

from racore.adapters.chunkers import FixedWindowChunker
from racore.adapters.embeddings import MockEmbeddingProvider
from racore.adapters.embeddings_voyage import VoyageConfig, VoyageEmbeddingProvider
from racore.adapters.judge_anthropic import AnthropicEntailmentJudge
from racore.adapters.judges import SubstringEntailmentJudge, TokenOverlapEntailmentJudge
from racore.adapters.llm import ExtractiveLLM
from racore.adapters.llm_anthropic import AnthropicConfig, AnthropicLLM
from racore.adapters.llm_openai import OpenAIConfig
from racore.adapters.memory import FileMemoryStore
from racore.adapters.memory_extract import RuleBasedMemoryExtractor
from racore.adapters.memory_extract_anthropic import AnthropicMemoryExtractor
from racore.adapters.relevance import CascadeRelevanceGate, ThresholdRelevanceGate
from racore.adapters.relevance_anthropic import AnthropicRelevanceGate
from racore.adapters.relevance_openai import OpenAIRelevanceGate
from racore.adapters.rerankers import NoopReranker
from racore.adapters.sources import FileSystemDocumentSource, InMemoryDocumentSource
from racore.adapters.vectorstores import InMemoryVectorStore

__all__ = [
    "AnthropicConfig",
    "AnthropicEntailmentJudge",
    "AnthropicLLM",
    "AnthropicMemoryExtractor",
    "AnthropicRelevanceGate",
    "CascadeRelevanceGate",
    "ExtractiveLLM",
    "FileMemoryStore",
    "FileSystemDocumentSource",
    "FixedWindowChunker",
    "InMemoryDocumentSource",
    "InMemoryVectorStore",
    "MockEmbeddingProvider",
    "NoopReranker",
    "OpenAIConfig",
    "OpenAIRelevanceGate",
    "RuleBasedMemoryExtractor",
    "SubstringEntailmentJudge",
    "ThresholdRelevanceGate",
    "TokenOverlapEntailmentJudge",
    "VoyageConfig",
    "VoyageEmbeddingProvider",
]
