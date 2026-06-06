"""Domain objects shared by the core orchestration and every adapter.

Pure-stdlib, immutable value objects. No provider or I/O concern lives here — see
``racore.core.ports`` for the plugin boundaries and ``racore.core.pipeline`` for how
these flow through ingest/answer.

Why frozen dataclasses, not pydantic, in Phase 0: the walking skeleton runs with zero
runtime dependencies (ADR-0007, $0 demo). A typed config surface (pydantic) arrives
with the config schema in a later phase. See ``docs/architecture.md`` §3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

# A dense embedding. A tuple (not a list) so domain objects stay immutable and the
# vector is hashable alongside the rest of the object.
type Vector = tuple[float, ...]


class InputType(StrEnum):
    """Embedding input role. Some providers embed a query differently from a document."""

    DOCUMENT = "document"
    QUERY = "query"


class MemoryKind(StrEnum):
    """Kinds of per-user memory. See ``docs/memory.md`` §2."""

    PROFILE = "profile"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    WORKING = "working"


# --- ingestion-side objects -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Document:
    """A source document after fetch + extraction. ``id`` is a content hash (ADR-0011)."""

    id: str
    text: str
    source: str  # where it came from — a URI, path, or connector key (provenance).
    metadata: Mapping[str, str] = field(default_factory=dict)
    created_at: float = 0.0  # ingest/freshness timestamp, epoch seconds; 0.0 = unset.


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable span of a ``Document``. ``start``/``end`` are char offsets into it,
    so a retrieval can be traced back to a verbatim source span for citation."""

    id: str
    doc_id: str
    text: str
    ordinal: int  # position within the document, 0-based.
    start: int
    end: int
    source: str = ""  # carried from the parent document for convenient provenance.


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A chunk paired with its embedding, ready for upsert into a vector store."""

    chunk: Chunk
    vector: Vector


# --- query-side objects -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Query:
    """A question to answer, scoped to a tenant and (optionally) a user."""

    text: str
    tenant_id: str = "default"
    user_id: str | None = None
    k: int = 5  # how many candidates to retrieve.
    filters: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Retrieval:
    """A retrieved chunk and its relevance score (higher = more relevant)."""

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Evidence:
    """A verbatim span that can back a claim, with enough provenance to click through."""

    quote: str
    doc_id: str
    chunk_id: str
    start: int
    end: int
    source: str = ""


@dataclass(frozen=True, slots=True)
class GroundedContext:
    """Assembled, citeable context handed to the LLM. ``evidences`` are 1-indexed by the
    citation marker that refers to them (evidence ``[i]`` is ``evidences[i - 1]``)."""

    query: str
    evidences: tuple[Evidence, ...]

    def marker_for(self, index: int) -> int:
        """1-based citation marker for ``evidences[index]``."""
        return index + 1


@dataclass(frozen=True, slots=True)
class Citation:
    """An inline ``[marker]`` in an answer, resolved to the evidence it points at."""

    marker: int
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """What an ``LLMProvider`` receives: a question and the cited context to ground in."""

    query: str
    context: GroundedContext
    system: str = ""


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """What an ``LLMProvider`` returns: answer text and the 1-based evidence markers it
    relied on, so the pipeline can resolve them to ``Citation`` objects."""

    text: str
    cited_markers: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class GroundingReport:
    """Per-claim support verdict for an answer. Faithfulness is the headline metric:
    the fraction of claims with a supporting cited span (1.0 when there are no claims)."""

    supported_claims: tuple[str, ...]
    unsupported_claims: tuple[str, ...]

    @property
    def faithfulness(self) -> float:
        total = len(self.supported_claims) + len(self.unsupported_claims)
        return 1.0 if total == 0 else len(self.supported_claims) / total

    @property
    def is_grounded(self) -> bool:
        return len(self.unsupported_claims) == 0


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Wall-clock duration of one pipeline stage (ADR-0010), in milliseconds."""

    stage: str
    millis: float


@dataclass(frozen=True, slots=True)
class Answer:
    """The result of ``pipeline.answer()`` — a *streamable* type (ADR-0009).

    The text is materialised here, but callers consume it through ``stream()`` so that
    real streaming LLM adapters can land later without changing this signature. The
    ``retrievals`` are retained for evaluation and click-to-source debugging.
    """

    text: str
    citations: tuple[Citation, ...]
    grounding: GroundingReport
    timings: tuple[StageTiming, ...]
    retrievals: tuple[Retrieval, ...] = ()
    abstained: bool = False

    async def stream(self) -> AsyncIterator[str]:
        """Yield the answer as text deltas.

        Phase 0 replays the already-materialised text token-by-token so callers can
        code against the same streaming interface a live LLM adapter will expose. The
        deltas concatenate back to ``text`` exactly.
        """
        for token in re.findall(r"\S+\s*|\s+", self.text):
            yield token

    @property
    def total_millis(self) -> float:
        return sum(t.millis for t in self.timings)


@dataclass(frozen=True, slots=True)
class IngestReport:
    """Outcome of ``pipeline.ingest()`` — what landed, and how long each stage took."""

    documents: int
    chunks: int
    timings: tuple[StageTiming, ...]

    @property
    def total_millis(self) -> float:
        return sum(t.millis for t in self.timings)


# --- memory -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """A single per-(tenant, user) memory. Schema mirrors ``docs/memory.md`` §4."""

    id: str
    tenant_id: str
    user_id: str
    kind: MemoryKind
    content: str
    source: str  # turn id / document ref — provenance; memory is grounded too.
    salience: float = 0.0
    embedding: Vector = ()
    created_at: float = 0.0
    last_used_at: float = 0.0
    superseded_by: str | None = None


# --- evaluation -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoldenRow:
    """One row of the golden dataset: a question with its expected answer and source.

    ``answerable=False`` marks a negative control — the corpus does not contain the
    answer, so the system must abstain rather than fabricate (see ``docs/evaluation.md``).
    """

    id: str
    question: str
    expected_answer: str
    expected_source: str
    answerable: bool = True


@dataclass(frozen=True, slots=True)
class EvalCase:
    """A golden row paired with the answer the pipeline produced for it."""

    row: GoldenRow
    answer: Answer


@dataclass(frozen=True, slots=True)
class EvalResult:
    """One evaluator's verdict over a run: a named score in ``[0, 1]`` plus detail."""

    name: str
    score: float
    details: Mapping[str, float] = field(default_factory=dict)
