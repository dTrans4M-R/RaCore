"""An opt-in Voyage AI adapter for the ``EmbeddingProvider`` port.

The real-embedding drop-in ADR-0007 anticipated: the ``$0`` ``MockEmbeddingProvider`` stays the
default, and swapping to a real *semantic* embedder is a one-line change at the composition root
because both satisfy the same ``EmbeddingProvider`` port. Nothing in the core or the default test
path depends on this module, and the Voyage SDK is an **optional extra**
(``pip install 'racore[voyage]'``) — importing this module never imports the SDK; that happens
lazily only when a real client is built.

Provider-agnostic by construction (ADR-0001): this is *one* adapter behind the port, exactly as
``AnthropicLLM`` is one adapter behind ``LLMProvider``. A local embedder (Ollama, BGE) or OpenAI is
a future drop-in with no core change — choosing Voyage here is "the best-results adapter first," not
a lock-in.

Configurable at three levels: **provider** (inject any ``EmbeddingProvider``), **adapter**
(``VoyageConfig`` — model, timeout, key), and **client** (a real ``AsyncClient`` built lazily from
config, or any object with the same tiny shape injected by tests to stay offline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from racore.core.types import InputType

if TYPE_CHECKING:
    from typing import Any

    from racore.core.types import Vector


@dataclass(frozen=True, slots=True)
class VoyageConfig:
    """Tunables for the Voyage adapter. ``api_key=None`` lets the SDK read ``VOYAGE_API_KEY``
    from the environment (load a gitignored ``.env`` via, e.g., ``uv run --env-file .env``)."""

    # Pinned model, not a moving alias, so eval runs are reproducible. Override with --embed-model.
    model: str = "voyage-3.5"
    timeout_s: float = 30.0
    api_key: str | None = None


# Voyage embeds queries and documents asymmetrically; map our InputType to its wire value.
_INPUT_TYPE = {InputType.QUERY: "query", InputType.DOCUMENT: "document"}


class _EmbeddingsResult(Protocol):
    @property
    def embeddings(self) -> list[list[float]]: ...


class _AsyncClient(Protocol):
    """The sliver of the Voyage async client this adapter uses — so the real SDK client and a
    test fake satisfy it structurally, with no SDK import at type-check time."""

    async def embed(
        self, texts: list[str], *, model: str, input_type: str | None
    ) -> _EmbeddingsResult: ...


class VoyageEmbeddingProvider:
    """Semantic embeddings via Voyage AI, behind the ``EmbeddingProvider`` port (batch-first)."""

    def __init__(
        self, config: VoyageConfig | None = None, *, client: _AsyncClient | None = None
    ) -> None:
        self._config = config or VoyageConfig()
        self._client: _AsyncClient | None = client

    async def embed(self, texts: list[str], input_type: InputType) -> list[Vector]:
        if not texts:  # an empty batch needs no client, key, or network round-trip.
            return []
        client = self._client_or_build()
        result = await client.embed(
            texts, model=self._config.model, input_type=_INPUT_TYPE.get(input_type)
        )
        return [tuple(vector) for vector in result.embeddings]

    def _client_or_build(self) -> _AsyncClient:
        if self._client is None:
            self._client = _build_client(self._config)
        return self._client


def _build_client(config: VoyageConfig) -> _AsyncClient:
    try:
        from voyageai import AsyncClient
    except ImportError as exc:
        raise RuntimeError(
            "The Voyage adapter needs its SDK. Install the optional extra: "
            "pip install 'racore[voyage]' (or `uv add voyageai`)."
        ) from exc

    kwargs: dict[str, Any] = {"timeout": config.timeout_s}
    if config.api_key is not None:
        kwargs["api_key"] = config.api_key
    # The real client is usable as ``_AsyncClient`` (it has an async ``embed``); cast past mypy's
    # strictness about the precise SDK signature vs our permissive protocol. (The SDK's loose
    # re-export of ``AsyncClient`` is handled by the scoped mypy override in pyproject.) Keeps the
    # gate green whether or not the optional extra is installed.
    return cast("_AsyncClient", AsyncClient(**kwargs))
