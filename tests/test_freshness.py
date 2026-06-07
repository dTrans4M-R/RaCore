"""Phase 3, slice 2: staleness surfaced (ADR-0024).

Freshness is a fact carried on each chunk and a judgment made against an injected ``now`` — never
the wall clock — so these tests are fully deterministic. They prove the timestamp propagates
source → document → chunk → retrieval (so it rides on every ``Answer``), that the pure helpers
judge staleness correctly, and that the harness surfaces the stalest evidence age when given a
reference ``now``.
"""

from __future__ import annotations

import asyncio

from racore.adapters.sources import InMemoryDocumentSource
from racore.core.freshness import age_seconds, stale, stalest_age
from racore.core.types import Chunk, GoldenRow, Query, Retrieval
from racore.eval import default_evaluators, demo_pipeline, run

# A fixed reference clock so every assertion is reproducible (ADR-0010: now is an input).
_NOW = 1_000_000_000.0
_DAY = 86_400.0


def _retrieval(source: str, created_at: float) -> Retrieval:
    chunk = Chunk(
        id=source,
        doc_id=source,
        text=source,
        ordinal=0,
        start=0,
        end=0,
        source=source,
        created_at=created_at,
    )
    return Retrieval(chunk=chunk, score=1.0)


def test_age_seconds_handles_unset_and_skew() -> None:
    assert age_seconds(0.0, _NOW) is None  # unset -> unknown age, not stale.
    assert age_seconds(_NOW - 10 * _DAY, _NOW) == 10 * _DAY
    assert age_seconds(_NOW + 5 * _DAY, _NOW) == 0.0  # future timestamp (clock skew) clamps to 0.


def test_stalest_age_and_stale_filter() -> None:
    retrievals = [
        _retrieval("fresh", _NOW - 10 * _DAY),
        _retrieval("stale", _NOW - 400 * _DAY),
        _retrieval("undated", 0.0),
    ]
    # The answer is only as fresh as its oldest evidence.
    assert stalest_age(retrievals, _NOW) == 400 * _DAY

    # A 30-day policy flags only the 400-day span; the undated one is never flagged.
    flagged = stale(retrievals, _NOW, max_age_seconds=30 * _DAY)
    assert [r.chunk.source for r in flagged] == ["stale"]

    # No timestamps at all -> nothing to judge.
    assert stalest_age([_retrieval("u", 0.0)], _NOW) is None


def test_timestamp_propagates_to_retrieval() -> None:
    asyncio.run(_timestamp_propagates_to_retrieval())


async def _timestamp_propagates_to_retrieval() -> None:
    pipeline = demo_pipeline()
    source = InMemoryDocumentSource()
    source.add("doc_fresh", "Neptune is a planet.", created_at=_NOW - 10 * _DAY)
    source.add("doc_stale", "Saturn is a planet.", created_at=_NOW - 400 * _DAY)
    await pipeline.ingest(source, prune=True)

    answer = await pipeline.answer(Query(text="Which is a planet?", k=5))

    # The freshness fact rode all the way through to the retrievals on the Answer.
    ages = {r.chunk.source: r.chunk.created_at for r in answer.retrievals}
    assert ages["doc_fresh"] == _NOW - 10 * _DAY
    assert ages["doc_stale"] == _NOW - 400 * _DAY
    assert stalest_age(answer.retrievals, _NOW) == 400 * _DAY


def test_harness_surfaces_evidence_age_only_when_now_given() -> None:
    asyncio.run(_harness_surfaces_evidence_age())


async def _harness_surfaces_evidence_age() -> None:
    pipeline = demo_pipeline()
    source = InMemoryDocumentSource()
    source.add("doc_fresh", "Neptune is a planet.", created_at=_NOW - 10 * _DAY)
    source.add("doc_stale", "Saturn is a planet.", created_at=_NOW - 400 * _DAY)
    await pipeline.ingest(source, prune=True)

    row = GoldenRow(
        id="f1",
        question="Which is a planet?",
        expected_answer="planet",
        relevant_sources=("doc_fresh",),
    )

    # With a reference now, the stalest evidence age is computed and rendered.
    dated = await run(pipeline, [row], default_evaluators(), now=_NOW)
    assert dated.per_case[0].evidence_age_s == 400 * _DAY
    assert "stalest evidence:" in dated.render(verbose=True)

    # Without a now, freshness stays unreported — the default eval path is unchanged.
    undated = await run(pipeline, [row], default_evaluators())
    assert undated.per_case[0].evidence_age_s is None
    assert "stalest evidence:" not in undated.render(verbose=True)
