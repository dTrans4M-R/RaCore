"""Client plumbing for OpenAI-*compatible* LLM endpoints — the "paid <-> local" seam.

The chat-completions protocol this speaks is the lingua franca of local model runtimes: Ollama,
vLLM, LM Studio and llama.cpp all expose it, and so does the hosted OpenAI API. So a single
adapter reaches a **local ($0)** model or a **paid** hosted one with no code change — only
``base_url`` (and, for hosted, ``api_key``) differ. That is the cheap middle tier between the
deterministic ``ThresholdRelevanceGate`` (no model) and a frontier gate like Claude: a free,
private, semantic gate you self-host.

This module is the shared plumbing only — config, a narrow client ``Protocol`` (so a test fake and
the real SDK both satisfy it with no SDK import at type-check time), lazy client construction, and
the response/usage mapping. The first consumer is ``OpenAIRelevanceGate``; a generator or judge on
the same protocol would land here too. The SDK is an **optional extra** (``racore[openai]``),
imported lazily, so the core install stays dependency-free (ADR-0007, ADR-0022). Mirrors
``llm_anthropic`` but in the OpenAI request/response shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from racore.core.types import TokenUsage

if TYPE_CHECKING:
    from typing import Any


@dataclass(frozen=True, slots=True)
class OpenAIConfig:
    """Tunables for the OpenAI-compatible adapter.

    Defaults target a **local** Ollama endpoint so the $0 path works out of the box once a model
    is pulled (``ollama pull llama3.2``). Point ``base_url`` at vLLM/LM Studio, or at
    ``https://api.openai.com/v1`` with a real ``api_key`` and an OpenAI ``model`` (e.g.
    ``gpt-4o-mini``), to use the same adapter against the hosted API.
    """

    # A pinned local default, overridable per run with `--model`. Even a small instruct model
    # (1-3B) is enough for the gate's one-word ANSWER/ABSTAIN verdict.
    model: str = "llama3.2"
    max_tokens: int = 1024
    temperature: float = 0.0  # deterministic-as-possible for reproducible eval runs.
    timeout_s: float = 30.0
    api_key: str | None = None
    base_url: str | None = "http://localhost:11434/v1"


class _Completions(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _Chat(Protocol):
    @property
    def completions(self) -> _Completions: ...


class _Client(Protocol):
    """The sliver of the OpenAI client this adapter uses — ``client.chat.completions.create`` —
    so the real ``AsyncOpenAI`` and a test fake satisfy it structurally, with no SDK import at
    type-check time. The nested members are read-only properties so a fake's narrower types match
    covariantly."""

    @property
    def chat(self) -> _Chat: ...


def _build_client(config: OpenAIConfig) -> _Client:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI-compatible adapter needs its SDK. Install the optional extra: "
            "pip install 'racore[openai]' (or `uv add openai`)."
        ) from exc

    kwargs: dict[str, Any] = {"timeout": config.timeout_s}
    if config.base_url is not None:
        kwargs["base_url"] = config.base_url
    # Local servers (Ollama, vLLM, LM Studio) ignore the key, but the SDK refuses to build without
    # a non-empty one; a harmless placeholder keeps the local path zero-config. For the hosted API,
    # set `api_key` in the config.
    kwargs["api_key"] = config.api_key if config.api_key is not None else "not-needed"
    # The real client satisfies ``_Client`` (it has ``chat.completions.create``); cast past mypy's
    # strictness about the exact SDK signature vs our permissive protocol. Keeps the gate green
    # whether or not the optional extra is installed.
    return cast("_Client", AsyncOpenAI(**kwargs))


def _extract_text(response: Any) -> str:
    """Read the assistant text off a chat-completions response (empty string if absent)."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return (content or "").strip()


def _extract_usage(response: Any, model: str) -> TokenUsage:
    """Read prompt/completion token counts off the response's ``usage`` (zero if absent).

    Local runtimes may omit usage; a zero count then prices to $0, which is honest — a self-hosted
    call has no per-token charge anyway.
    """
    raw = getattr(response, "usage", None)
    return TokenUsage(
        input_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
        model=model,
    )
