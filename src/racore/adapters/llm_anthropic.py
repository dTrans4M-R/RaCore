"""An opt-in Anthropic (Claude) adapter for the ``LLMProvider`` port.

This is the real-generation drop-in that ADR-0007 anticipated: the ``$0`` extractive stack
stays the default, and swapping to a real model is a one-line change at the composition root
because both satisfy the same port. Nothing in the core or the default test path depends on
this module, and the Anthropic SDK is an **optional extra** (``pip install 'racore[anthropic]'``)
— importing this module never imports the SDK; that happens lazily only when a real client is
built.

Configurable at three levels (the point of ports-and-adapters):

* **provider** — the pipeline depends on ``LLMProvider``, so this is interchangeable with
  ``ExtractiveLLM`` (or any other adapter) by injection alone;
* **adapter** — ``AnthropicConfig`` carries model, token budget, temperature, timeout, key, and
  base URL, with sane defaults (``temperature=0.0`` for eval reproducibility);
* **client** — a real ``AsyncAnthropic`` is built lazily from config, but any object with the
  same tiny shape can be injected (used by the tests to stay deterministic and offline).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from racore.core.types import LLMResponse

if TYPE_CHECKING:
    from typing import Any

    from racore.core.types import LLMRequest

_MARKER_RE = re.compile(r"\[(\d+)\]")

# Used only if a request arrives without a system prompt; the pipeline always supplies one.
_FALLBACK_SYSTEM = (
    "Answer the question using only the numbered evidence provided. Cite each claim with its "
    "evidence marker like [1]. If the evidence does not contain the answer, say you don't know."
)


@dataclass(frozen=True, slots=True)
class AnthropicConfig:
    """Tunables for the Anthropic adapter. ``api_key=None`` lets the SDK read
    ``ANTHROPIC_API_KEY`` from the environment (load it from a gitignored ``.env`` via, e.g.,
    ``uv run --env-file .env``)."""

    # Pinned snapshot, not a moving alias, so eval runs are reproducible. Override per run with
    # `--model`; e.g. `claude-haiku-4-5` (alias) or a newer dated snapshot.
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 1024
    temperature: float = 0.0  # deterministic-as-possible for reproducible eval runs.
    timeout_s: float = 30.0
    api_key: str | None = None
    base_url: str | None = None


class _Messages(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _Client(Protocol):
    """The sliver of the Anthropic client this adapter actually uses — so the real SDK client
    and a test fake satisfy it structurally, with no SDK import at type-check time. ``messages``
    is a read-only property so a fake's narrower ``messages`` type matches covariantly."""

    @property
    def messages(self) -> _Messages: ...


class AnthropicLLM:
    """Grounded generation via Claude, behind the ``LLMProvider`` port (batch-first)."""

    def __init__(
        self, config: AnthropicConfig | None = None, *, client: _Client | None = None
    ) -> None:
        self._config = config or AnthropicConfig()
        self._client: _Client | None = client

    async def generate(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        client = self._client_or_build()
        return list(await asyncio.gather(*(self._one(client, request) for request in requests)))

    async def _one(self, client: _Client, request: LLMRequest) -> LLMResponse:
        message = await client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            system=request.system or _FALLBACK_SYSTEM,
            messages=[{"role": "user", "content": _render_prompt(request)}],
        )
        text = _extract_text(message)
        return LLMResponse(text=text, cited_markers=_cited_markers(text))

    def _client_or_build(self) -> _Client:
        if self._client is None:
            self._client = _build_client(self._config)
        return self._client


def _build_client(config: AnthropicConfig) -> _Client:
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise RuntimeError(
            "The Anthropic adapter needs its SDK. Install the optional extra: "
            "pip install 'racore[anthropic]' (or `uv add anthropic`)."
        ) from exc

    kwargs: dict[str, Any] = {"timeout": config.timeout_s}
    if config.api_key is not None:
        kwargs["api_key"] = config.api_key
    if config.base_url is not None:
        kwargs["base_url"] = config.base_url
    # The real client is usable as ``_Client`` (it has ``messages.create``); cast past mypy's
    # strictness about the precise SDK signature vs our permissive protocol. Keeps the gate green
    # whether or not the optional extra is installed.
    return cast("_Client", AsyncAnthropic(**kwargs))


def _render_prompt(request: LLMRequest) -> str:
    """Render the question and 1-indexed evidence into a single user turn the model can cite."""
    lines = [f"Question: {request.query}", "", "Evidence:"]
    lines.extend(f"[{i}] {ev.quote}" for i, ev in enumerate(request.context.evidences, start=1))
    lines += ["", "Answer using only the evidence above; cite each claim with its [n] marker."]
    return "\n".join(lines)


def _cited_markers(text: str) -> tuple[int, ...]:
    """Every distinct ``[n]`` marker in the answer, in first-seen order."""
    ordered: dict[int, None] = {}
    for raw in _MARKER_RE.findall(text):
        ordered[int(raw)] = None
    return tuple(ordered)


def _extract_text(message: Any) -> str:
    """Concatenate the text blocks of an Anthropic message response."""
    blocks = getattr(message, "content", []) or []
    return "".join(block.text for block in blocks if getattr(block, "type", None) == "text").strip()
