"""``python -m racore.eval`` — ingest the golden corpus and print the baseline.

This is the Phase 0 definition-of-done in runnable form: a thin end-to-end slice through
the real ports that emits quality *and* latency/cost numbers with zero external spend.
"""

from __future__ import annotations

import asyncio

from racore.eval.datasets import golden_dataset, golden_source
from racore.eval.harness import HarnessReport, demo_pipeline, run
from racore.eval.metrics import default_evaluators


async def _baseline() -> HarnessReport:
    pipeline = demo_pipeline()
    await pipeline.ingest(golden_source())
    return await run(pipeline, golden_dataset(), default_evaluators())


def main() -> None:
    report = asyncio.run(_baseline())
    print(report.render())


if __name__ == "__main__":
    main()
