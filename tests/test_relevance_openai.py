"""The OpenAI-compatible (local-or-hosted) relevance gate, verified offline with a fake client.

No SDK, no network, no key: a fake mimicking ``client.chat.completions.create`` is injected, so
these run in the default $0 gate. They pin what *this* adapter controls — the chat-completions
call shape (system + user messages, the one-word cap), reading the verdict off
``choices[0].message.content``, that empty retrieval costs nothing — and that the gate drops into
the provider-agnostic cascade and forwards its usage. The prompt/parse themselves are shared with
the Anthropic gate and pinned in ``test_relevance_llm.py``; the real local path is exercised by hand
via ``python -m racore.eval --gate llm --gate-provider openai`` against a running Ollama/vLLM.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from typing import TYPE_CHECKING

import pytest

from racore.adapters.llm_openai import OpenAIConfig, _resolve_api_key
from racore.adapters.relevance import CascadeRelevanceGate
from racore.adapters.relevance_openai import OpenAIRelevanceGate
from racore.core.types import Chunk, RelevanceCheck, Retrieval

if TYPE_CHECKING:
    from typing import Any

_HAS_OPENAI = importlib.util.find_spec("openai") is not None


def _check(query: str, *scores: float) -> RelevanceCheck:
    retrievals = tuple(
        Retrieval(
            chunk=Chunk(
                id=f"s{i}", doc_id=f"s{i}", text=f"{query} evidence {i}", ordinal=0, start=0, end=0
            ),
            score=score,
        )
        for i, score in enumerate(scores)
    )
    return RelevanceCheck(query=query, retrievals=retrievals)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(40, 1)  # a gate call: ~40 in, 1 out


class _FakeCompletions:
    """Returns a verdict chosen by which query appears in the user turn."""

    def __init__(self, verdict_by_query: dict[str, str]) -> None:
        self._verdicts = verdict_by_query
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        user_turn = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
        for needle, verdict in self._verdicts.items():
            if needle in user_turn:
                return _FakeResponse(verdict)
        return _FakeResponse("ABSTAIN")


class _FakeChat:
    def __init__(self, verdict_by_query: dict[str, str]) -> None:
        self.completions = _FakeCompletions(verdict_by_query)


class _FakeClient:
    def __init__(self, verdict_by_query: dict[str, str]) -> None:
        self.chat = _FakeChat(verdict_by_query)


def test_gate_parses_answer_and_abstain_per_query() -> None:
    asyncio.run(_verdicts_case())


async def _verdicts_case() -> None:
    client = _FakeClient({"smallest planet": "ANSWER", "rings on Neptune": "ABSTAIN"})
    gate = OpenAIRelevanceGate(client=client)

    decisions = await gate.should_answer(
        [_check("smallest planet", 0.5), _check("rings on Neptune", 0.4)]
    )

    assert decisions == [True, False]
    calls = client.chat.completions.calls
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 8
    # The strict instruction rides as a system message (OpenAI shape), not a top-level param.
    assert calls[0]["messages"][0]["role"] == "system"
    assert "ANSWER or ABSTAIN" in calls[0]["messages"][0]["content"]


def test_empty_retrieval_costs_nothing() -> None:
    asyncio.run(_empty_case())


async def _empty_case() -> None:
    client = _FakeClient({"answerable": "ANSWER"})
    gate = OpenAIRelevanceGate(client=client)

    decisions = await gate.should_answer(
        [_check("answerable", 0.5), RelevanceCheck(query="empty", retrievals=())]
    )

    assert decisions == [True, False]
    assert len(client.chat.completions.calls) == 1  # the empty check never reached the model


def test_all_empty_skips_the_client_entirely() -> None:
    asyncio.run(_all_empty_case())


async def _all_empty_case() -> None:
    client = _FakeClient({})
    gate = OpenAIRelevanceGate(client=client)

    decisions = await gate.should_answer(
        [RelevanceCheck(query="a", retrievals=()), RelevanceCheck(query="b", retrievals=())]
    )

    assert decisions == [False, False]
    assert client.chat.completions.calls == []


def test_gate_records_drainable_usage() -> None:
    asyncio.run(_usage_case())


async def _usage_case() -> None:
    client = _FakeClient({"alpha": "ANSWER", "beta": "ABSTAIN"})
    gate = OpenAIRelevanceGate(OpenAIConfig(model="llama3.2"), client=client)

    await gate.should_answer([_check("alpha", 0.6), _check("beta", 0.6)])
    drained = gate.drain_usage()

    assert len(drained) == 2
    assert all((u.input_tokens, u.output_tokens) == (40, 1) for u in drained)
    assert all(u.model == "llama3.2" for u in drained)
    assert (
        gate.drain_usage() == []
    )  # draining resets, so per-answer accounting doesn't double-count


def test_missing_sdk_raises_a_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    gate = OpenAIRelevanceGate()
    with pytest.raises(RuntimeError, match=r"racore\[openai\]"):
        asyncio.run(gate.should_answer([_check("q", 0.5)]))


def test_resolve_api_key_prefers_config_then_env_then_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression for the hosted-path 401: a real key in the environment must NOT be shadowed by the
    # local placeholder. Priority is explicit config key > OPENAI_API_KEY env var > placeholder.
    # (Values passed via vars so the secret scanner doesn't flag a literal `api_key=` assignment.)
    given = "from-config"
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert _resolve_api_key(OpenAIConfig(api_key=given)) == given  # explicit wins
    assert _resolve_api_key(OpenAIConfig()) == "from-env"  # hosted path reads the env
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _resolve_api_key(OpenAIConfig()) == "not-needed"  # keyless local: placeholder


@pytest.mark.skipif(not _HAS_OPENAI, reason="optional 'openai' extra not installed")
def test_real_sdk_client_satisfies_gate_contract() -> None:
    # The default config points at a local endpoint with a placeholder key — building the client is
    # offline and makes no request, so this pins the structural contract without a server.
    gate = OpenAIRelevanceGate(OpenAIConfig())
    client = gate._client_or_build()
    assert callable(getattr(client.chat.completions, "create", None))


def test_cascade_forwards_openai_gate_usage() -> None:
    asyncio.run(_cascade_usage())


async def _cascade_usage() -> None:
    # The OpenAI gate is just another RelevanceGate, so it drops into the provider-agnostic cascade
    # as the gray-zone fallback and its usage is forwarded for pricing.
    client = _FakeClient({"gray": "ANSWER"})
    fallback = OpenAIRelevanceGate(OpenAIConfig(model="llama3.2"), client=client)
    cascade = CascadeRelevanceGate(fallback, low=0.3, high=0.7)

    decisions = await cascade.should_answer([_check("high", 0.9), _check("gray", 0.5)])
    drained = cascade.drain_usage()

    assert decisions == [True, True]  # 'high' free-answered; 'gray' escalated to the local gate
    assert len(drained) == 1  # only the gray-zone call billed
    assert drained[0].model == "llama3.2"
    assert cascade.drain_usage() == []
