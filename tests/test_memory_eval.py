"""The memory personalization-lift harness — Phase 4's headline number (ADR-0028, ADR-0029).

Proves two things on the $0 stack: explicit self-statements go from unanswerable (memory off) to
correct (memory on), and — honestly — the rule floor's recall *ceiling*: it catches every explicit
fact but none of the implicit ones, so the headline is +0.625, not a misleading +1.000. That gap is
what the paid LLM extractor (ADR-0030) is measured against.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from racore.eval.memory import memory_demo_pipeline, run_memory_lift

if TYPE_CHECKING:
    from pathlib import Path


def test_zero_cost_floor_catches_explicit_facts_and_misses_implicit(tmp_path: Path) -> None:
    report = asyncio.run(run_memory_lift(memory_demo_pipeline(tmp_path)))

    explicit = [c for c in report.per_case if not c.implicit]
    implicit = [c for c in report.per_case if c.implicit]
    assert explicit and implicit  # the eval mixes both kinds on purpose.

    # Every explicit self-statement is extracted and answered (cited to a memory/ source); the
    # memory-off control never answers a personal question.
    assert all(c.extracted and c.on_correct and c.used_memory for c in explicit)
    assert not any(c.off_correct for c in report.per_case)

    # The honest ceiling: the rule floor extracts *none* of the implicit facts — so the headline is
    # the floor's real recall, not a perfect 1.0. This is the gap the paid extractor must close.
    assert not any(c.extracted for c in implicit)
    assert report.extraction_recall == len(explicit) / report.scenarios
    assert report.lift == len(explicit) / report.scenarios
    assert report.correctness_off == 0.0
