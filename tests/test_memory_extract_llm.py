"""AnthropicMemoryExtractor offline (ADR-0030): parsing, provenance, cost, question-skip.

Exercises the LLM extractor with an injected fake client — no SDK, no network, no key — so the
JSON-parsing, provenance, usage-reporting, and question-skipping contracts are gated in CI. The
real-API recall lift (the "paid only improves" number) is recorded in ADR-0030 from a live run.
"""

from __future__ import annotations

import asyncio
from typing import Any

from racore.adapters.memory_extract_anthropic import AnthropicMemoryExtractor
from racore.core.types import MemoryKind, MemoryTurn


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeUsage:
    input_tokens = 42
    output_tokens = 9


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]
        self.usage = _FakeUsage()


class _FakeMessages:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0

    async def create(self, **_kwargs: Any) -> _FakeMessage:
        self.calls += 1
        return _FakeMessage(self._reply)


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self.messages = _FakeMessages(reply)


def _turn(text: str) -> MemoryTurn:
    return MemoryTurn(tenant_id="t", user_id="u", source="turn-1", user_text=text)


def test_parses_an_implicit_fact_the_rule_floor_would_miss() -> None:
    reply = (
        '[{"kind": "semantic", "content": "usually meets on Tuesday mornings", "key": "schedule"}]'
    )
    extractor = AnthropicMemoryExtractor(client=_FakeClient(reply))

    (item,) = asyncio.run(extractor.extract([_turn("We usually sync on Tuesday mornings.")]))
    assert item.content == "usually meets on Tuesday mornings"
    assert item.key == "schedule"
    assert item.kind == MemoryKind.SEMANTIC
    assert item.source == "turn-1"  # provenance carried — memory is grounded.
    assert extractor.drain_usage()  # the call's tokens were recorded (UsageReporter).


def test_a_question_is_skipped_without_a_paid_call() -> None:
    client = _FakeClient('[{"content": "should not happen"}]')
    extractor = AnthropicMemoryExtractor(client=client)

    assert asyncio.run(extractor.extract([_turn("What is my name?")])) == []
    assert client.messages.calls == 0  # no model call on a question
    assert extractor.drain_usage() == []


def test_non_json_output_is_tolerated_as_no_memories() -> None:
    extractor = AnthropicMemoryExtractor(client=_FakeClient("I couldn't find anything durable."))
    assert asyncio.run(extractor.extract([_turn("I prefer tea.")])) == []
