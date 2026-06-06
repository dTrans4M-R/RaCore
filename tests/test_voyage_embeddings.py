"""The opt-in Voyage embedding adapter, typed against a narrow client Protocol so it is
testable offline (a fake client — no SDK, no network, no key), with a guarded contract test
that runs only when the optional extra is installed. Mirrors ``test_anthropic_llm.py``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys

import pytest

from racore.adapters.embeddings_voyage import (
    VoyageConfig,
    VoyageEmbeddingProvider,
    _build_client,
)
from racore.core.types import InputType

_HAS_VOYAGE = importlib.util.find_spec("voyageai") is not None


class _FakeResult:
    def __init__(self, embeddings: list[list[float]], total_tokens: int = 0) -> None:
        self.embeddings = embeddings
        self.total_tokens = total_tokens


class _FakeAsyncClient:
    """Records the (batch size, model, input_type) it was called with; returns deterministic
    vectors so assertions are exact. Satisfies the adapter's ``_AsyncClient`` protocol by shape."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, str | None]] = []

    async def embed(self, texts: list[str], *, model: str, input_type: str | None) -> _FakeResult:
        self.calls.append((len(texts), model, input_type))
        # total_tokens scales with the batch so the usage assertion below is meaningful.
        return _FakeResult([[float(len(t)), 1.0] for t in texts], total_tokens=len(texts) * 5)


def test_embed_returns_one_vector_per_text_in_a_single_batched_call() -> None:
    asyncio.run(_embed_batch())


async def _embed_batch() -> None:
    client = _FakeAsyncClient()
    provider = VoyageEmbeddingProvider(VoyageConfig(model="voyage-3.5"), client=client)

    vectors = await provider.embed(["aa", "bbb"], InputType.DOCUMENT)

    assert vectors == [(2.0, 1.0), (3.0, 1.0)]
    # Batch-first: the whole list goes in one call, with the document input_type mapped to wire.
    assert client.calls == [(2, "voyage-3.5", "document")]


def test_embed_records_drainable_token_usage() -> None:
    asyncio.run(_usage_case())


async def _usage_case() -> None:
    client = _FakeAsyncClient()
    provider = VoyageEmbeddingProvider(VoyageConfig(model="voyage-3.5"), client=client)

    await provider.embed(["aa", "bbb"], InputType.DOCUMENT)  # total_tokens = 2 * 5 = 10
    drained = provider.drain_usage()

    assert len(drained) == 1
    # Embeddings bill input only; the model tags it for pricing.
    assert (drained[0].input_tokens, drained[0].output_tokens) == (10, 0)
    assert drained[0].model == "voyage-3.5"
    assert (
        provider.drain_usage() == []
    )  # draining resets, so per-answer accounting doesn't double-count


def test_query_and_document_input_types_map_to_voyage_wire_values() -> None:
    asyncio.run(_input_types())


async def _input_types() -> None:
    client = _FakeAsyncClient()
    provider = VoyageEmbeddingProvider(client=client)

    await provider.embed(["q"], InputType.QUERY)
    await provider.embed(["d"], InputType.DOCUMENT)

    # The provider embeds queries and documents asymmetrically, as Voyage supports.
    assert [call[2] for call in client.calls] == ["query", "document"]


def test_empty_input_skips_the_client_entirely() -> None:
    asyncio.run(_empty())


async def _empty() -> None:
    client = _FakeAsyncClient()
    provider = VoyageEmbeddingProvider(client=client)

    assert await provider.embed([], InputType.DOCUMENT) == []
    assert client.calls == []  # no key, no client build, no network for an empty batch


def test_missing_sdk_raises_a_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate the optional extra being absent: importing voyageai fails.
    monkeypatch.setitem(sys.modules, "voyageai", None)
    with pytest.raises(RuntimeError, match=r"racore\[voyage\]"):
        _build_client(VoyageConfig())


@pytest.mark.skipif(not _HAS_VOYAGE, reason="optional 'voyage' extra not installed")
def test_real_sdk_client_satisfies_adapter_contract() -> None:
    # Runs only when `uv sync --extra voyage` was done. No network: the constructor is offline,
    # and we only assert the client shape the adapter relies on (an async ``embed``) still exists.
    client = _build_client(VoyageConfig(api_key="not-used-offline"))
    assert hasattr(client, "embed")
