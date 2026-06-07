"""The memory personalization-lift harness — Phase 4's headline number (ADR-0028).

Proves, on the $0 stack, that questions answerable only from a user's stated facts go from
unanswerable (memory off) to correct (memory on): the lift the DoD requires "measured".
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from racore.eval.memory import memory_demo_pipeline, run_memory_lift

if TYPE_CHECKING:
    from pathlib import Path


def test_memory_lift_on_the_zero_cost_stack(tmp_path: Path) -> None:
    report = asyncio.run(run_memory_lift(memory_demo_pipeline(tmp_path)))

    # Every probe is answerable only from the remembered fact: correct with memory on, not without.
    assert report.correctness_on == 1.0
    assert report.correctness_off == 0.0
    assert report.lift == 1.0
    assert report.used_memory_rate == 1.0  # every "on" answer cited a memory/ source.
