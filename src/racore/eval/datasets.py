"""The golden dataset: a synthetic Solar-System corpus plus question/answer/source rows.

The corpus is public-domain general knowledge, written from scratch — no client or user
data ever enters this repo (CLAUDE.md §6). Unlike the Phase 0 toy corpus (7 unambiguous
docs, where ``k=5`` made retrieval trivially perfect), this set is built to make retrieval
*non-trivial* so the next slices have something to improve and measure:

* **Distractors** — documents that share salient words with a question but do not answer it
  (other moons for a "largest moon" query, other "giants", a Neptune doc that never mentions
  rings). A purely lexical retriever has to discriminate, not just match.
* **Paraphrase-gap rows** — questions whose wording barely overlaps the answer document
  ("closest to our star" vs "closest to the Sun"). The $0 lexical embedder scores these
  poorly, which is exactly the gap a real semantic embedder closes; until then the right
  document is often buried below rank 1, so the extractive answer quotes a distractor.
* **Multi-source rows** — questions whose answer spans more than one document, so recall@k
  can read below 1.0 (and rank-weighted nDCG@k / MRR have judgments to score next slice).

Negative controls (``answerable=False``, empty ``relevant_sources``) have a topically-near
distractor in the corpus but no answer, so a mature system must abstain rather than seize the
near-miss. See ``docs/evaluation.md`` §3.
"""

from __future__ import annotations

from racore.adapters.sources import InMemoryDocumentSource
from racore.core.types import GoldenRow

# (source, text) — each document is one self-contained fact. Clusters of topically-near
# docs are deliberate: they are the distractors that make ranking matter.
CORPUS: tuple[tuple[str, str], ...] = (
    # --- planets ---------------------------------------------------------------
    (
        "planets/mercury",
        "Mercury is the smallest planet in the Solar System and the closest to the Sun.",
    ),
    (
        "planets/venus",
        "Venus has the hottest surface of any planet, with a thick carbon dioxide atmosphere.",
    ),
    (
        "planets/earth",
        "Earth is the only planet known to harbour life, its surface mostly covered by oceans "
        "of liquid water.",
    ),
    (
        "planets/mars",
        "Mars is called the Red Planet because iron oxide on its surface gives it a reddish hue.",
    ),
    (
        "planets/jupiter",
        "Jupiter is the largest planet in the Solar System, a gas giant with a Great Red Spot.",
    ),
    (
        "planets/saturn",
        "Saturn is famous for its prominent ring system, made largely of ice particles.",
    ),
    (
        "planets/uranus",
        "Uranus is an ice giant that rotates on its side, tipped almost ninety degrees over.",
    ),
    (
        "planets/neptune",
        "Neptune, the windiest ice giant, is lashed by the fastest gusts in the Solar System.",
    ),
    # --- planet distractors (share a planet's name, answer a different question) -
    (
        "planets/mercury-temperature",
        "Mercury swings from scorching days to freezing nights because it keeps almost no air.",
    ),
    (
        "planets/saturn-hexagon",
        "A six-sided jet stream forms a lasting hexagon around Saturn's north pole.",
    ),
    (
        "planets/jupiter-moons",
        "Jupiter holds the most moons of any planet, among them the four large Galilean moons.",
    ),
    (
        "planets/mars-air",
        "Mars keeps only a thin atmosphere, under one percent the density of air at sea level.",
    ),
    # --- moons -----------------------------------------------------------------
    (
        "moons/titan",
        "Titan is the largest moon of Saturn and wears a dense, nitrogen-rich atmosphere.",
    ),
    (
        "moons/ganymede",
        "Ganymede, which circles Jupiter, is the largest moon in the whole Solar System.",
    ),
    (
        "moons/europa",
        "Europa is an icy moon of Jupiter believed to hide a salty ocean under its frozen shell.",
    ),
    (
        "moons/luna",
        "The Moon is Earth's only natural satellite and the brightest object in the night sky.",
    ),
    (
        "moons/phobos",
        "Phobos is the larger of the two small moons of Mars and drifts slowly inward each year.",
    ),
    (
        "moons/titan-lakes",
        "Titan is dotted with lakes and rivers of liquid methane across its frigid landscape.",
    ),
    # --- giants and composition (distractors for each other) -------------------
    (
        "facts/gas-giants",
        "The gas giants are Jupiter and Saturn, worlds of mostly hydrogen and helium with no "
        "solid surface.",
    ),
    (
        "facts/ice-giants",
        "The ice giants are Uranus and Neptune, rich in frozen water, ammonia, and methane.",
    ),
    # --- stars and light -------------------------------------------------------
    (
        "stars/sun",
        "The Sun is a yellow dwarf star that holds over 99 percent of the Solar System's mass.",
    ),
    (
        "stars/proxima",
        "Proxima Centauri is the nearest star beyond the Sun, about 4.2 light-years away.",
    ),
    (
        "facts/sunlight",
        "Light from the Sun takes roughly eight minutes to cross the gap and reach the Earth.",
    ),
    # --- general distractors ---------------------------------------------------
    (
        "facts/great-red-spot",
        "The Great Red Spot is a colossal storm on Jupiter that has churned for centuries.",
    ),
    (
        "facts/asteroid-belt",
        "The asteroid belt lies between Mars and Jupiter and holds countless rocky bodies.",
    ),
    (
        "facts/solar-system-age",
        "The Solar System formed about 4.6 billion years ago from a collapsing cloud of dust.",
    ),
    (
        "facts/dwarf-planet",
        "Pluto was reclassified as a dwarf planet in 2006 and circles beyond Neptune.",
    ),
    (
        "facts/comets",
        "Comets are icy bodies that grow bright tails of gas and dust when they near the Sun.",
    ),
)

GOLDEN: tuple[GoldenRow, ...] = (
    # --- direct rows: the answer document shares the question's key words --------
    GoldenRow("q1", "Which is the smallest planet?", "Mercury", ("planets/mercury",)),
    GoldenRow("q2", "What is the largest planet?", "Jupiter", ("planets/jupiter",)),
    GoldenRow("q3", "Why is Mars called the Red Planet?", "iron oxide", ("planets/mars",)),
    GoldenRow("q4", "What is Saturn famous for?", "ring system", ("planets/saturn",)),
    GoldenRow("q5", "Which planet has the hottest surface?", "Venus", ("planets/venus",)),
    GoldenRow("q6", "What is the largest moon of Saturn?", "Titan", ("moons/titan",)),
    GoldenRow(
        "q7",
        "What is the largest moon in the Solar System?",
        "Ganymede",
        ("moons/ganymede",),
    ),
    # --- paraphrase-gap rows: question wording diverges from the answer doc ------
    # The lexical $0 embedder shares few content words with the answer here, so the right
    # doc is easily out-ranked by a distractor that happens to share more — the gap a real
    # semantic embedder (and reranking) closes.
    GoldenRow("p1", "Which world sits closest to our star?", "Mercury", ("planets/mercury",)),
    GoldenRow("p2", "Why does Mars look rusty?", "iron oxide", ("planets/mars",)),
    GoldenRow("p3", "Which planet spins tipped over on its side?", "Uranus", ("planets/uranus",)),
    GoldenRow(
        "p4",
        "How long does light from the Sun take to get here?",
        "eight minutes",
        ("facts/sunlight",),
    ),
    GoldenRow(
        "p5",
        "Which moon might hold an underground sea?",
        "Europa",
        ("moons/europa",),
    ),
    # --- multi-source rows: the answer spans more than one document --------------
    GoldenRow(
        "m1",
        "Which two planets are the gas giants?",
        "Jupiter and Saturn",
        # planets/saturn's doc is about its rings, not its classification, so it is not a
        # relevant source here; the summary doc carries Saturn, and Jupiter's doc says "gas
        # giant" outright.
        ("facts/gas-giants", "planets/jupiter"),
    ),
    GoldenRow(
        "m2",
        "Which planets are the ice giants?",
        "Uranus and Neptune",
        ("facts/ice-giants", "planets/uranus", "planets/neptune"),
    ),
    # --- negative controls: a near-miss distractor exists, but no answer ---------
    # The corpus has a Neptune doc (winds), a Moon doc (satellite), and a Pluto doc (dwarf
    # planet) — none answer these, so the system must abstain rather than grab the near-miss.
    GoldenRow("n1", "How many rings does Neptune have?", "", (), answerable=False),
    GoldenRow("n2", "In what year did the first Moon landing happen?", "", (), answerable=False),
    GoldenRow("n3", "What is the surface temperature on Pluto?", "", (), answerable=False),
)


def golden_source() -> InMemoryDocumentSource:
    """A document source seeded with the golden corpus."""
    return InMemoryDocumentSource(CORPUS)


def golden_dataset() -> list[GoldenRow]:
    """The golden rows as a fresh list."""
    return list(GOLDEN)
