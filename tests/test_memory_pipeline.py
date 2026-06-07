"""The Phase 4 write->read personalization loop, end to end through the pipeline (ADR-0026).

The headline number: a question answerable only from a user's stated preference is wrong/abstained
with memory OFF and correct with memory ON — that delta is the personalization lift, demonstrated
on the **$0** stack (the remembered fact enters the same grounded-evidence channel the extractive
model already reads). The other tests guard the two ways this can go wrong: leaking one user's
memory to another, and letting an unrelated memory hijack a corpus answer.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from racore.adapters.chunkers import FixedWindowChunker
from racore.adapters.embeddings import MockEmbeddingProvider
from racore.adapters.judges import SubstringEntailmentJudge
from racore.adapters.llm import ExtractiveLLM
from racore.adapters.memory import FileMemoryStore
from racore.adapters.memory_extract import RuleBasedMemoryExtractor
from racore.adapters.rerankers import NoopReranker
from racore.adapters.sources import InMemoryDocumentSource
from racore.adapters.vectorstores import InMemoryVectorStore
from racore.core.pipeline import Pipeline
from racore.core.types import Query

if TYPE_CHECKING:
    from pathlib import Path

_CORPUS = (
    ("planets/jupiter", "Jupiter is the largest planet in the Solar System."),
    ("planets/mercury", "Mercury is the smallest planet and closest to the Sun."),
)


def _pipeline(tmp_path: Path) -> Pipeline:
    return Pipeline(
        embedder=MockEmbeddingProvider(),
        store=InMemoryVectorStore(),
        reranker=NoopReranker(),
        chunker=FixedWindowChunker(),
        llm=ExtractiveLLM(),
        judge=SubstringEntailmentJudge(),
        memory=FileMemoryStore(tmp_path),
        extractor=RuleBasedMemoryExtractor(),
    )


def test_personalization_lift_on_the_zero_cost_stack(tmp_path: Path) -> None:
    asyncio.run(_personalization_lift(tmp_path))


async def _personalization_lift(tmp_path: Path) -> None:
    pipe = _pipeline(tmp_path)
    await pipe.ingest(InMemoryDocumentSource(_CORPUS), tenant_id="t")

    # Turn 1: the user states a durable preference. We do not score this turn; we assert it learned.
    await pipe.answer(
        Query(
            text="Remember that I prefer bullet-point summaries.",
            tenant_id="t",
            user_id="u",
        )
    )
    assert pipe.memory is not None
    stored = await pipe.memory.read("t", "u", "summaries", k=5)
    assert any("bullet" in item.content for item in stored)

    probe = "How should you format summaries for me?"

    # Memory ON: the remembered preference is injected as labelled evidence and answers the probe.
    on = await pipe.answer(Query(text=probe, tenant_id="t", user_id="u"))
    assert "bullet" in on.text.lower()
    assert not on.abstained
    assert any(c.evidence.source.startswith("memory/") for c in on.citations)

    # Memory OFF: a different user has no such memory, so the same probe cannot answer "bullet".
    off = await pipe.answer(Query(text=probe, tenant_id="t", user_id="other"))
    assert "bullet" not in off.text.lower()


def test_an_unrelated_memory_does_not_hijack_a_corpus_answer(tmp_path: Path) -> None:
    asyncio.run(_no_corpus_pollution(tmp_path))


async def _no_corpus_pollution(tmp_path: Path) -> None:
    pipe = _pipeline(tmp_path)
    await pipe.ingest(InMemoryDocumentSource(_CORPUS), tenant_id="t")
    await pipe.answer(Query(text="I prefer bullet-point summaries.", tenant_id="t", user_id="u"))

    # The memory shares no words with this question, so it must not be injected: the answer is the
    # corpus fact, cited to the corpus — never the irrelevant preference.
    ans = await pipe.answer(Query(text="What is the largest planet?", tenant_id="t", user_id="u"))
    assert "Jupiter" in ans.text
    assert all(not c.evidence.source.startswith("memory/") for c in ans.citations)


def test_a_stated_fact_is_learned_even_when_the_turn_abstains(tmp_path: Path) -> None:
    asyncio.run(_learns_on_abstain(tmp_path))


async def _learns_on_abstain(tmp_path: Path) -> None:
    # No corpus at all: the statement turn cannot be answered, so it abstains — but the write policy
    # runs on every exit path, so the fact is still learned.
    pipe = _pipeline(tmp_path)
    result = await pipe.answer(Query(text="My name is Ada.", tenant_id="t", user_id="u"))
    assert result.abstained

    assert pipe.memory is not None
    stored = await pipe.memory.read("t", "u", "name", k=5)
    assert any("Ada" in item.content for item in stored)
