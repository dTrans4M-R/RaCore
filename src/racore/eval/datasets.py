"""The Phase 0 golden dataset: a tiny synthetic corpus plus question/answer/source rows.

The corpus is public-domain general knowledge, written from scratch — no client or user
data ever enters this repo (CLAUDE.md §6). It is deliberately small and unambiguous so the
baseline is reproducible. Two rows are negative controls (``answerable=False``): their
answer is absent from the corpus, so a mature system must abstain. Phase 0 has no
abstention logic yet, so they intentionally expose that gap in the baseline (Phase 2 closes
it). See ``docs/evaluation.md`` §3.
"""

from __future__ import annotations

from racore.adapters.sources import InMemoryDocumentSource
from racore.core.types import GoldenRow

# (source, text) — each document is one self-contained fact.
CORPUS: tuple[tuple[str, str], ...] = (
    (
        "planets/mercury",
        "Mercury is the smallest planet in the Solar System and the closest to the Sun.",
    ),
    (
        "planets/jupiter",
        "Jupiter is the largest planet in the Solar System, a gas giant with a Great Red Spot.",
    ),
    (
        "planets/mars",
        "Mars is called the Red Planet because iron oxide on its surface gives it a reddish hue.",
    ),
    (
        "planets/saturn",
        "Saturn is famous for its prominent ring system, made largely of ice particles.",
    ),
    (
        "planets/venus",
        "Venus has the hottest surface of any planet, with a thick carbon dioxide atmosphere.",
    ),
    (
        "moons/titan",
        "Titan is the largest moon of Saturn and has a dense nitrogen-rich atmosphere.",
    ),
    (
        "stars/sun",
        "The Sun is a yellow dwarf star that holds over 99 percent of the Solar System's mass.",
    ),
)

GOLDEN: tuple[GoldenRow, ...] = (
    GoldenRow("q1", "Which is the smallest planet?", "Mercury", "planets/mercury"),
    GoldenRow("q2", "What is the largest planet?", "Jupiter", "planets/jupiter"),
    GoldenRow("q3", "Why is Mars called the Red Planet?", "iron oxide", "planets/mars"),
    GoldenRow("q4", "What is Saturn famous for?", "ring system", "planets/saturn"),
    GoldenRow("q5", "Which planet has the hottest surface?", "Venus", "planets/venus"),
    GoldenRow("q6", "What is the largest moon of Saturn?", "Titan", "moons/titan"),
    # Negative controls — the corpus has no Neptune or Moon-landing fact.
    GoldenRow("n1", "How many rings does Neptune have?", "", "", answerable=False),
    GoldenRow("n2", "In what year did the first Moon landing happen?", "", "", answerable=False),
)


def golden_source() -> InMemoryDocumentSource:
    """A document source seeded with the golden corpus."""
    return InMemoryDocumentSource(CORPUS)


def golden_dataset() -> list[GoldenRow]:
    """The golden rows as a fresh list."""
    return list(GOLDEN)
