"""The opt-in LLM entailment judge, verified offline with a fake Claude client.

No SDK, no network, no key: a fake mimicking ``client.messages.create`` is injected, so these
run in the default $0 gate. They pin what the adapter controls — the per-claim prompt it builds,
how it parses the one-word verdict, and that an uncited claim costs nothing (no call). The real
network path is exercised by hand via ``python -m racore.eval --judge llm`` with a key.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from typing import TYPE_CHECKING

import pytest

from racore.adapters.judge_anthropic import AnthropicEntailmentJudge, _is_supported
from racore.adapters.llm_anthropic import AnthropicConfig
from racore.core.types import ClaimCheck

if TYPE_CHECKING:
    from typing import Any

_HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None


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
        self.usage = _FakeUsage(30, 1)  # a verdict call: ~30 in, 1 out


class _FakeMessages:
    """Returns a verdict chosen by which claim appears in the prompt (order-independent)."""

    def __init__(self, verdict_by_claim: dict[str, str]) -> None:
        self._verdicts = verdict_by_claim
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeMessage:
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        for needle, verdict in self._verdicts.items():
            if needle in prompt:
                return _FakeMessage(verdict)
        return _FakeMessage("UNSUPPORTED")


class _FakeClient:
    def __init__(self, verdict_by_claim: dict[str, str]) -> None:
        self.messages = _FakeMessages(verdict_by_claim)


def test_parses_supported_and_unsupported_per_claim() -> None:
    asyncio.run(_verdicts_case())


async def _verdicts_case() -> None:
    client = _FakeClient({"closest to our star": "SUPPORTED", "Saturn has rings": "UNSUPPORTED"})
    judge = AnthropicEntailmentJudge(client=client)

    verdicts = await judge.judge(
        [
            ClaimCheck(
                claim="Mercury sits closest to our star", evidence=("...closest to the Sun.",)
            ),
            ClaimCheck(claim="Saturn has rings", evidence=("Saturn is a gas giant.",)),
        ]
    )

    assert verdicts == [True, False]
    assert len(client.messages.calls) == 2  # one create() per cited claim
    # Each call carried the strict one-word system prompt and a tiny token budget.
    assert client.messages.calls[0]["max_tokens"] == 8
    assert "SUPPORTED or UNSUPPORTED" in client.messages.calls[0]["system"]


def test_uncited_claim_is_unsupported_without_a_call() -> None:
    asyncio.run(_uncited_case())


async def _uncited_case() -> None:
    client = _FakeClient({"cited": "SUPPORTED"})
    judge = AnthropicEntailmentJudge(client=client)

    verdicts = await judge.judge(
        [
            ClaimCheck(claim="a cited claim", evidence=("cited evidence",)),
            ClaimCheck(claim="an uncited claim", evidence=()),  # no evidence -> no call
        ]
    )

    assert verdicts == [True, False]
    assert len(client.messages.calls) == 1  # the uncited claim never reached the model


def test_all_uncited_skips_the_client_entirely() -> None:
    asyncio.run(_all_uncited_case())


async def _all_uncited_case() -> None:
    client = _FakeClient({})
    judge = AnthropicEntailmentJudge(client=client)

    verdicts = await judge.judge(
        [ClaimCheck(claim="x", evidence=()), ClaimCheck(claim="y", evidence=())]
    )

    assert verdicts == [False, False]
    assert client.messages.calls == []  # no key, no network when nothing is cited


def test_judge_records_drainable_usage_for_cited_claims_only() -> None:
    asyncio.run(_usage_case())


async def _usage_case() -> None:
    client = _FakeClient({"alpha": "SUPPORTED", "beta": "UNSUPPORTED"})
    judge = AnthropicEntailmentJudge(AnthropicConfig(model="claude-haiku-4-5"), client=client)

    await judge.judge(
        [
            ClaimCheck(claim="alpha", evidence=("ev",)),
            ClaimCheck(claim="beta", evidence=("ev",)),
            ClaimCheck(claim="uncited", evidence=()),  # no call -> no usage to record
        ]
    )
    drained = judge.drain_usage()

    assert len(drained) == 2  # one usage per cited claim; the uncited one cost nothing
    assert all((u.input_tokens, u.output_tokens) == (30, 1) for u in drained)
    assert all(u.model == "claude-haiku-4-5" for u in drained)
    assert (
        judge.drain_usage() == []
    )  # draining resets, so per-answer accounting doesn't double-count


def test_is_supported_parses_the_one_word_verdict() -> None:
    assert _is_supported("SUPPORTED")
    assert _is_supported("The claim is SUPPORTED.")
    assert not _is_supported("UNSUPPORTED")
    assert not _is_supported("unsupported.")  # 'supported' is a substring — must not match
    assert not _is_supported("I'm not sure")  # ambiguous defaults to unsupported


def test_missing_sdk_raises_a_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    judge = AnthropicEntailmentJudge()
    with pytest.raises(RuntimeError, match=r"racore\[anthropic\]"):
        asyncio.run(judge.judge([ClaimCheck(claim="x", evidence=("ev",))]))


@pytest.mark.skipif(not _HAS_ANTHROPIC, reason="optional 'anthropic' extra not installed")
def test_real_sdk_client_satisfies_judge_contract() -> None:
    # Runs only with `uv sync --extra anthropic`. No network: build the client offline and assert
    # the shape the judge relies on (messages.create) exists.
    judge = AnthropicEntailmentJudge(AnthropicConfig(api_key="not-used-offline"))
    client = judge._client_or_build()
    assert callable(getattr(client.messages, "create", None))
