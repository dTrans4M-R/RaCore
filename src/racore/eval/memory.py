"""Memory evaluation: the personalization-lift number (``docs/memory.md`` §6).

Unlike the corpus eval (one pipeline over a flat golden set), personalization is measured **per user
and across turns**: state a fact, then ask a question only that fact can answer, and compare the
answer with memory **on** (the user who stated it) against memory **off** (a user who never did).
The gap is the lift. It runs on the **$0 stack**, so the headline is the proof that personalization
is bought at zero cost — a paid extractor/embedder only widens which questions clear the bar.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from racore.adapters.memory import FileMemoryStore
from racore.adapters.memory_extract import RuleBasedMemoryExtractor
from racore.core.types import Query
from racore.eval.datasets import golden_source
from racore.eval.harness import demo_pipeline

if TYPE_CHECKING:
    from pathlib import Path

    from racore.core.pipeline import Pipeline

# A user who never stated anything: the memory-off control. Probes are questions, so answering as
# this user never writes anything (the extractor ignores questions), keeping the control empty.
_OFF_USER = "__no_memory__"


@dataclass(frozen=True, slots=True)
class MemoryScenario:
    """A user states ``setup`` facts, then asks ``question`` — answerable only from those facts."""

    id: str
    user_id: str
    setup: tuple[str, ...]
    question: str
    expected: str  # a substring the correct, memory-grounded answer must contain.


MEMORY_SCENARIOS: tuple[MemoryScenario, ...] = (
    MemoryScenario("name", "bob", ("My name is Bob.",), "What is my name?", "Bob"),
    MemoryScenario(
        "pref",
        "ada",
        ("Remember that I prefer bullet-point summaries.",),
        "How should you format summaries for me?",
        "bullet",
    ),
    MemoryScenario(
        "fiscal",
        "carol",
        ("My fiscal year ends in March.",),
        "When does my fiscal year end?",
        "March",
    ),
    MemoryScenario(
        "work",
        "dan",
        ("I work in mergers and acquisitions.",),
        "What field do I work in?",
        "mergers",
    ),
    MemoryScenario(
        "flight",
        "erin",
        ("Remember that my flight is on Friday.",),
        "When is my flight?",
        "Friday",
    ),
)


@dataclass(frozen=True, slots=True)
class MemoryCaseOutcome:
    """Per-scenario detail: was the probe answered correctly with memory on vs off, and was a
    remembered fact actually used (cited from a ``memory/`` source)?"""

    id: str
    question: str
    on_correct: bool
    off_correct: bool
    used_memory: bool
    on_answer: str
    off_answer: str


@dataclass(frozen=True, slots=True)
class MemoryLiftReport:
    scenarios: int
    correctness_on: float
    correctness_off: float
    used_memory_rate: float
    per_case: tuple[MemoryCaseOutcome, ...]

    @property
    def lift(self) -> float:
        """Personalization lift: correctness with memory on minus the same probes off."""
        return self.correctness_on - self.correctness_off

    def render(self, *, verbose: bool = False) -> str:
        lines = [
            "RaCore - memory personalization lift",
            "====================================",
            f"Scenarios: {self.scenarios}",
            "",
            f"  answer.correctness (memory on)    {self.correctness_on:6.3f}",
            f"  answer.correctness (memory off)   {self.correctness_off:6.3f}",
            f"  personalization lift             {self.lift:+6.3f}",
            f"  stored-fact recall (mem used)     {self.used_memory_rate:6.3f}",
        ]
        if verbose:
            lines += ["", "Per-scenario (-v)"]
            for case in self.per_case:
                lines.append(
                    f"  [{case.id}] on={'ok ' if case.on_correct else 'BAD'}"
                    f" off={'ok' if case.off_correct else 'BAD'}"
                    f" used={'Y' if case.used_memory else 'n'}  {case.question}"
                )
                lines.append(f"        on:  {_truncate(case.on_answer)}")
                lines.append(f"        off: {_truncate(case.off_answer)}")
        return "\n".join(lines)


def memory_demo_pipeline(base_dir: Path) -> Pipeline:
    """The $0 stack with memory enabled: a file-backed store + the rule-based extractor."""
    return replace(
        demo_pipeline(),
        memory=FileMemoryStore(base_dir),
        extractor=RuleBasedMemoryExtractor(),
    )


async def run_memory_lift(
    pipeline: Pipeline,
    scenarios: tuple[MemoryScenario, ...] = MEMORY_SCENARIOS,
    *,
    tenant_id: str = "default",
) -> MemoryLiftReport:
    """Ingest the shared corpus, then measure each scenario's correctness with memory on vs off."""
    await pipeline.ingest(golden_source(), tenant_id)
    outcomes: list[MemoryCaseOutcome] = []
    for scenario in scenarios:
        for statement in scenario.setup:
            await pipeline.answer(
                Query(text=statement, tenant_id=tenant_id, user_id=scenario.user_id)
            )
        on = await pipeline.answer(
            Query(text=scenario.question, tenant_id=tenant_id, user_id=scenario.user_id)
        )
        off = await pipeline.answer(
            Query(text=scenario.question, tenant_id=tenant_id, user_id=_OFF_USER)
        )
        outcomes.append(
            MemoryCaseOutcome(
                id=scenario.id,
                question=scenario.question,
                on_correct=_contains(on.text, scenario.expected),
                off_correct=_contains(off.text, scenario.expected),
                used_memory=any(c.evidence.source.startswith("memory/") for c in on.citations),
                on_answer=on.text,
                off_answer=off.text,
            )
        )
    return MemoryLiftReport(
        scenarios=len(outcomes),
        correctness_on=_mean([o.on_correct for o in outcomes]),
        correctness_off=_mean([o.off_correct for o in outcomes]),
        used_memory_rate=_mean([o.used_memory for o in outcomes]),
        per_case=tuple(outcomes),
    )


def _contains(answer: str, expected: str) -> bool:
    return expected.lower() in answer.lower()


def _mean(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def _truncate(text: str, limit: int = 100) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."
