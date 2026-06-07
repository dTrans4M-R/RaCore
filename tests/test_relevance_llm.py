"""The opt-in LLM relevance gate + the cascade, verified offline with a fake Claude client.

No SDK, no network, no key: a fake mimicking ``client.messages.create`` is injected, so these
run in the default $0 gate. They pin what the adapter controls — the per-query prompt, the
one-word verdict parse, that empty retrieval costs nothing — and that the cascade escalates
*only* the gray zone and forwards the fallback's billed usage. The real network path is exercised
by hand via ``python -m racore.eval --gate llm`` with a key.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from typing import TYPE_CHECKING

import pytest

from racore.adapters.llm_anthropic import AnthropicConfig
from racore.adapters.relevance import CascadeRelevanceGate
from racore.adapters.relevance_anthropic import AnthropicRelevanceGate, _should_answer
from racore.core.types import Chunk, RelevanceCheck, Retrieval

if TYPE_CHECKING:
    from typing import Any

_HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None


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


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]
        self.usage = _FakeUsage(40, 1)  # a gate call: ~40 in, 1 out


class _FakeMessages:
    """Returns a verdict chosen by which query appears in the prompt."""

    def __init__(self, verdict_by_query: dict[str, str]) -> None:
        self._verdicts = verdict_by_query
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeMessage:
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        for needle, verdict in self._verdicts.items():
            if needle in prompt:
                return _FakeMessage(verdict)
        return _FakeMessage("ABSTAIN")


class _FakeClient:
    def __init__(self, verdict_by_query: dict[str, str]) -> None:
        self.messages = _FakeMessages(verdict_by_query)


class _RecordingGate:
    """A stub RelevanceGate that records the checks it was handed (to assert escalation)."""

    def __init__(self, verdict: bool = True) -> None:
        self.seen: list[RelevanceCheck] = []
        self._verdict = verdict

    async def should_answer(self, checks: list[RelevanceCheck]) -> list[bool]:
        self.seen.extend(checks)
        return [self._verdict] * len(checks)


# --- the LLM gate -----------------------------------------------------------------


def test_gate_parses_answer_and_abstain_per_query() -> None:
    asyncio.run(_verdicts_case())


async def _verdicts_case() -> None:
    client = _FakeClient({"smallest planet": "ANSWER", "rings on Neptune": "ABSTAIN"})
    gate = AnthropicRelevanceGate(client=client)

    decisions = await gate.should_answer(
        [_check("smallest planet", 0.5), _check("rings on Neptune", 0.4)]
    )

    assert decisions == [True, False]
    assert len(client.messages.calls) == 2
    assert client.messages.calls[0]["max_tokens"] == 8
    assert "ANSWER or ABSTAIN" in client.messages.calls[0]["system"]


def test_empty_retrieval_costs_nothing() -> None:
    asyncio.run(_empty_case())


async def _empty_case() -> None:
    client = _FakeClient({"answerable": "ANSWER"})
    gate = AnthropicRelevanceGate(client=client)

    decisions = await gate.should_answer(
        [_check("answerable", 0.5), RelevanceCheck(query="empty", retrievals=())]
    )

    assert decisions == [True, False]
    assert len(client.messages.calls) == 1  # the empty-retrieval check never reached the model


def test_all_empty_skips_the_client_entirely() -> None:
    asyncio.run(_all_empty_case())


async def _all_empty_case() -> None:
    client = _FakeClient({})
    gate = AnthropicRelevanceGate(client=client)

    decisions = await gate.should_answer(
        [RelevanceCheck(query="a", retrievals=()), RelevanceCheck(query="b", retrievals=())]
    )

    assert decisions == [False, False]
    assert client.messages.calls == []


def test_gate_records_drainable_usage() -> None:
    asyncio.run(_usage_case())


async def _usage_case() -> None:
    client = _FakeClient({"alpha": "ANSWER", "beta": "ABSTAIN"})
    gate = AnthropicRelevanceGate(AnthropicConfig(model="claude-haiku-4-5"), client=client)

    await gate.should_answer([_check("alpha", 0.6), _check("beta", 0.6)])
    drained = gate.drain_usage()

    assert len(drained) == 2
    assert all((u.input_tokens, u.output_tokens) == (40, 1) for u in drained)
    assert all(u.model == "claude-haiku-4-5" for u in drained)
    assert (
        gate.drain_usage() == []
    )  # draining resets, so per-answer accounting doesn't double-count


def test_should_answer_parses_the_one_word_verdict() -> None:
    assert _should_answer("ANSWER")
    assert _should_answer("Answer: yes")
    assert not _should_answer("ABSTAIN")
    assert not _should_answer("I would abstain.")
    assert not _should_answer("not sure")  # ambiguous defaults to abstain (trust-safe)


def test_missing_sdk_raises_a_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    gate = AnthropicRelevanceGate()
    with pytest.raises(RuntimeError, match=r"racore\[anthropic\]"):
        asyncio.run(gate.should_answer([_check("q", 0.5)]))


@pytest.mark.skipif(not _HAS_ANTHROPIC, reason="optional 'anthropic' extra not installed")
def test_real_sdk_client_satisfies_gate_contract() -> None:
    gate = AnthropicRelevanceGate(AnthropicConfig(api_key="not-used-offline"))
    client = gate._client_or_build()
    assert callable(getattr(client.messages, "create", None))


# --- the cascade ------------------------------------------------------------------


def test_cascade_escalates_only_the_gray_zone() -> None:
    asyncio.run(_cascade_gray_zone())


async def _cascade_gray_zone() -> None:
    fallback = _RecordingGate(verdict=True)
    cascade = CascadeRelevanceGate(fallback, low=0.3, high=0.7)

    decisions = await cascade.should_answer(
        [
            _check("low", 0.10),  # below low -> abstain, free
            _check("gray", 0.50),  # gray zone -> escalate to fallback (says answer)
            _check("high", 0.90),  # at/above high -> answer, free
            RelevanceCheck(query="empty", retrievals=()),  # empty -> abstain, free
        ]
    )

    assert decisions == [False, True, True, False]
    # Only the single gray-zone check reached the (expensive) fallback.
    assert [c.query for c in fallback.seen] == ["gray"]


def test_cascade_forwards_fallback_usage() -> None:
    asyncio.run(_cascade_usage())


async def _cascade_usage() -> None:
    client = _FakeClient({"gray": "ANSWER"})
    fallback = AnthropicRelevanceGate(AnthropicConfig(model="claude-haiku-4-5"), client=client)
    cascade = CascadeRelevanceGate(fallback, low=0.3, high=0.7)

    await cascade.should_answer([_check("high", 0.9), _check("gray", 0.5)])
    drained = cascade.drain_usage()

    # The free 'high' decision billed nothing; only the gray-zone LLM call did.
    assert len(drained) == 1
    assert drained[0].model == "claude-haiku-4-5"
    assert cascade.drain_usage() == []


def test_cascade_rejects_invalid_bands() -> None:
    with pytest.raises(ValueError):
        CascadeRelevanceGate(_RecordingGate(), low=0.8, high=0.2)
    with pytest.raises(ValueError):
        CascadeRelevanceGate(_RecordingGate(), low=-0.1, high=0.5)
