"""An opt-in LLM memory extractor (Claude) for the ``MemoryExtractor`` port — the paid lever.

The $0 ``RuleBasedMemoryExtractor`` catches **explicit** self-statements but misses facts stated
**implicitly** ("we usually sync on Tuesdays" → a schedule preference; "I had to give up gluten" →
a dietary constraint). This adapter asks the model to extract durable, reusable facts from a turn,
so the same ``MemoryExtractor`` port now covers the implicit recall the rule floor leaves on the
table — bought only when you pay for it (``docs/memory.md`` §3, ADR-0030). It is **one adapter
behind the port**, not a lock-in: an OpenAI/local extractor is a future drop-in here.

Mirrors ``AnthropicLLM``/``AnthropicEntailmentJudge`` and reuses their client plumbing: optional
extra (``racore[anthropic]``), lazy SDK, narrow client ``Protocol`` (offline-testable), and a
``UsageReporter`` so the harness prices the extractor like any other billed component.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from racore.adapters.llm_anthropic import (
    AnthropicConfig,
    _build_client,
    _extract_text,
    _extract_usage,
)
from racore.core.ids import content_id
from racore.core.types import MemoryItem, MemoryKind

if TYPE_CHECKING:
    from racore.adapters.llm_anthropic import _Client
    from racore.core.types import MemoryTurn, TokenUsage

_SYSTEM = (
    "You extract durable, reusable facts a user states about themselves, for long-term memory. "
    "Return ONLY a JSON array; each element is an object with keys: "
    '"kind" (one of "profile" or "semantic"), '
    '"content" (a concise third-person fact, e.g. "prefers bullet summaries", '
    '"usually meets on Tuesday mornings", "avoids gluten"), and '
    '"key" (a short slot name for a fact that can later change, e.g. "name", "schedule"; '
    "otherwise an empty string). "
    "Extract only durable, user-specific facts — never questions, pleasantries, or one-off "
    "requests. If there is nothing worth remembering, return []."
)
# A model-proposed fact clears the salience floor: the model judged it durable enough to return.
_LLM_SALIENCE = 0.8
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class AnthropicMemoryExtractor:
    """Durable-fact extraction via Claude, behind the ``MemoryExtractor`` port (batch-first)."""

    def __init__(
        self, config: AnthropicConfig | None = None, *, client: _Client | None = None
    ) -> None:
        self._config = config or AnthropicConfig()
        self._client: _Client | None = client
        self._usage: list[TokenUsage] = []

    async def extract(self, turns: list[MemoryTurn]) -> list[MemoryItem]:
        # A question states nothing durable; skip it without a (paid) call — and never let a probe
        # pollute memory. Mirrors the rule extractor's "questions are not self-statements".
        targets = [turn for turn in turns if _is_statement(turn.user_text)]
        if not targets:
            return []
        client = self._client_or_build()
        per_turn = await asyncio.gather(*(self._one(client, turn) for turn in targets))
        return [item for items in per_turn for item in items]

    async def _one(self, client: _Client, turn: MemoryTurn) -> list[MemoryItem]:
        message = await client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            system=_SYSTEM,
            messages=[{"role": "user", "content": turn.user_text}],
        )
        self._usage.append(_extract_usage(message, self._config.model))
        return _parse(_extract_text(message), turn)

    def drain_usage(self) -> list[TokenUsage]:
        """Usage accumulated since the last drain; resets on read (the ``UsageReporter`` port)."""
        drained, self._usage = self._usage, []
        return drained

    def _client_or_build(self) -> _Client:
        if self._client is None:
            self._client = _build_client(self._config)
        return self._client


def _is_statement(text: str) -> bool:
    """A non-empty turn that isn't a bare question — worth a (paid) extraction call."""
    stripped = text.strip()
    return bool(stripped) and not stripped.endswith("?")


def _parse(reply: str, turn: MemoryTurn) -> list[MemoryItem]:
    """Parse the model's JSON array into memories; tolerate prose or malformed output as empty."""
    match = _ARRAY_RE.search(reply)
    if match is None:
        return []
    try:
        raw = json.loads(match.group())
    except (ValueError, TypeError):
        return []
    items: list[MemoryItem] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        content = str(entry.get("content", "")).strip()
        if not content:
            continue
        items.append(
            MemoryItem(
                id=content_id(turn.tenant_id, turn.user_id, content),
                tenant_id=turn.tenant_id,
                user_id=turn.user_id,
                kind=_kind(entry.get("kind")),
                content=content,
                source=turn.source,
                key=str(entry.get("key", "")).strip(),
                salience=_LLM_SALIENCE,
            )
        )
    return items


def _kind(value: object) -> MemoryKind:
    try:
        return MemoryKind(str(value).lower())
    except ValueError:
        return MemoryKind.SEMANTIC
