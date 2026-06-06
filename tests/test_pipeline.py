"""Pipeline behaviour that the end-to-end happy path doesn't cover: abstention."""

from __future__ import annotations

import asyncio

from racore.core.types import Query
from racore.eval import demo_pipeline


def test_abstains_when_no_context() -> None:
    asyncio.run(_abstains_when_no_context())


async def _abstains_when_no_context() -> None:
    # Nothing ingested -> the store is empty -> retrieval yields nothing to ground on.
    pipeline = demo_pipeline()

    answer = await pipeline.answer(Query(text="Which is the smallest planet?"))

    assert answer.abstained
    assert answer.citations == ()
    assert answer.retrievals == ()
    # Even an abstention is timed up to the point it bailed out.
    assert {"embed", "retrieve", "rerank"} <= {t.stage for t in answer.timings}
