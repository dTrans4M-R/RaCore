"""The Anthropic adapter's request/response mapping, verified offline with a fake client.

No SDK, no network, no key: a fake that mimics the tiny ``client.messages.create`` shape is
injected, so these run in the default $0 gate like everything else. They pin what the adapter
controls — the request it builds from config + evidence, and how it parses the answer into
text + cited markers. The real network path is exercised by hand via ``python -m racore.eval
--llm anthropic`` with a key.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from typing import TYPE_CHECKING

import pytest

from racore.adapters.llm_anthropic import AnthropicConfig, AnthropicLLM, _extract_text
from racore.core.types import Evidence, GroundedContext, LLMRequest

if TYPE_CHECKING:
    from typing import Any

_HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None


class _FakeBlock:
    def __init__(self, text: str, type_: str = "text") -> None:
        self.type = type_
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeMessage:
    def __init__(self, blocks: list[_FakeBlock], usage: _FakeUsage | None = None) -> None:
        self.content = blocks
        self.usage = usage


class _FakeMessages:
    """Records every create() call and replays a canned answer with token usage."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeMessage:
        self.calls.append(kwargs)
        return _FakeMessage([_FakeBlock(self._reply)], _FakeUsage(11, 7))


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self.messages = _FakeMessages(reply)


def _evidence(quote: str) -> Evidence:
    return Evidence(quote=quote, doc_id="d", chunk_id="c", start=0, end=len(quote), source="s")


def test_generate_renders_evidence_and_parses_markers() -> None:
    asyncio.run(_generate_case())


async def _generate_case() -> None:
    client = _FakeClient("Mercury is the smallest planet [1]. It is closest to the Sun [1].")
    llm = AnthropicLLM(AnthropicConfig(model="claude-haiku-4-5", max_tokens=512), client=client)
    context = GroundedContext(
        query="Which is the smallest planet?",
        evidences=(_evidence("Mercury is the smallest planet in the Solar System."),),
    )

    (response,) = await llm.generate(
        [LLMRequest(query="Which is the smallest planet?", context=context, system="SYS")]
    )

    assert response.text == "Mercury is the smallest planet [1]. It is closest to the Sun [1]."
    assert response.cited_markers == (1,)  # deduped

    # Token usage is carried through, tagged with the model that priced it.
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens) == (11, 7)
    assert response.usage.model == "claude-haiku-4-5"

    # The outgoing request carried our config, the system prompt, and numbered evidence.
    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["max_tokens"] == 512
    assert call["temperature"] == 0.0
    assert call["system"] == "SYS"
    user_turn = call["messages"][0]["content"]
    assert "Which is the smallest planet?" in user_turn
    assert "[1] Mercury is the smallest planet in the Solar System." in user_turn


def test_cited_markers_are_deduped_in_first_seen_order() -> None:
    asyncio.run(_markers_case())


async def _markers_case() -> None:
    client = _FakeClient("Gamma [2]. Alpha [1]. Beta again [2].")
    llm = AnthropicLLM(client=client)
    context = GroundedContext(query="q", evidences=(_evidence("x"), _evidence("y")))

    (response,) = await llm.generate([LLMRequest(query="q", context=context)])

    assert response.cited_markers == (2, 1)


def test_generate_is_batch_first() -> None:
    asyncio.run(_batch_case())


async def _batch_case() -> None:
    client = _FakeClient("Answer [1].")
    llm = AnthropicLLM(client=client)
    context = GroundedContext(query="q", evidences=(_evidence("x"),))

    responses = await llm.generate(
        [LLMRequest(query="q1", context=context), LLMRequest(query="q2", context=context)]
    )

    assert len(responses) == 2
    assert all(r.cited_markers == (1,) for r in responses)
    assert len(client.messages.calls) == 2  # one create() per request


def test_extract_text_concatenates_text_blocks_only() -> None:
    # The response may carry non-text blocks (e.g. thinking/tool_use); only text counts.
    message = _FakeMessage(
        [_FakeBlock("Hello "), _FakeBlock("ignored", type_="thinking"), _FakeBlock("world.")]
    )
    assert _extract_text(message) == "Hello world."


def test_missing_sdk_raises_a_friendly_error() -> None:
    asyncio.run(_missing_sdk_case())


async def _missing_sdk_case() -> None:
    # Without an injected client the adapter must import the SDK; if it's absent, the error
    # should name the optional extra rather than surfacing a bare ImportError. Forcing
    # ``sys.modules['anthropic'] = None`` makes ``import anthropic`` raise, regardless of
    # whether the extra happens to be installed.
    llm = AnthropicLLM()
    request = LLMRequest(query="q", context=GroundedContext(query="q", evidences=(_evidence("x"),)))
    sentinel = object()
    saved: object = sys.modules.get("anthropic", sentinel)
    sys.modules["anthropic"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match=r"racore\[anthropic\]"):
            await llm.generate([request])
    finally:
        if saved is sentinel:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = saved  # type: ignore[assignment]


@pytest.mark.skipif(not _HAS_ANTHROPIC, reason="optional 'anthropic' extra not installed")
def test_real_sdk_client_satisfies_adapter_contract() -> None:
    # Runs only when `uv sync --extra anthropic` was done. No network: the constructor is
    # offline, and we only assert the client shape the adapter relies on still exists — this
    # catches an SDK that renamed `messages.create` out from under us.
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key="not-used-no-network-call")
    assert hasattr(client, "messages")
    assert callable(getattr(client.messages, "create", None))
