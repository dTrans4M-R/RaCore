"""The grounding stage: assemble cited context, then verify each claim against what it cited.

Split out of ``pipeline.py`` so the claim/citation bookkeeping has one home and the
entailment check is a pluggable port (``EntailmentJudge``). Two things make this more than
the Phase 0 substring check:

* **Per-claim attribution.** Each answer sentence is checked only against the evidence *it
  cited*, not a pool of every cited quote — so a sentence citing ``[2]`` is judged against
  evidence 2. A sentence that cites nothing is unsupportable by construction.
* **Drop or flag.** Unsupported claims are always reported (flagged); with
  ``drop_unsupported`` the answer text is rebuilt from the supported claims only.

The entailment decision itself is delegated to an ``EntailmentJudge`` (exact substring,
token overlap, or — later — an LLM judge), so this module stays about *what* a claim is and
*which* evidence backs it, never *how* support is decided.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from racore.core.types import (
    Citation,
    ClaimCheck,
    Evidence,
    GroundedContext,
    GroundingReport,
)

if TYPE_CHECKING:
    from racore.core.ports import EntailmentJudge
    from racore.core.types import LLMResponse, Retrieval


@dataclass(frozen=True, slots=True)
class GroundingOutcome:
    """What the grounding stage produces: the (possibly claim-filtered) answer text, the
    resolved citations, and the per-claim support report."""

    text: str
    citations: tuple[Citation, ...]
    report: GroundingReport


def assemble(query_text: str, ranked: list[Retrieval]) -> GroundedContext:
    """Turn ranked retrievals into 1-indexed, citeable evidence for the LLM to ground in."""
    evidences = tuple(
        Evidence(
            quote=r.chunk.text,
            doc_id=r.chunk.doc_id,
            chunk_id=r.chunk.id,
            start=r.chunk.start,
            end=r.chunk.end,
            source=r.chunk.source,
        )
        for r in ranked
    )
    return GroundedContext(query=query_text, evidences=evidences)


async def verify(
    response: LLMResponse,
    context: GroundedContext,
    judge: EntailmentJudge,
    *,
    drop_unsupported: bool = False,
) -> GroundingOutcome:
    """Resolve citations and judge each claim against the evidence it cited.

    A claim is *supported* when the judge entails it from one of its own cited quotes;
    everything else — uncited claims and claims whose citation doesn't hold up — is
    unsupported. Faithfulness counts supported / all claims; citation correctness counts the
    supported subset of *cited* claims (``docs/evaluation.md`` §2).
    """
    claims = _parse_claims(response.text)
    checks = [
        ClaimCheck(claim=claim.text, evidence=_cited_quotes(claim, context)) for claim in claims
    ]
    verdicts = await judge.judge(checks)

    supported: list[str] = []
    unsupported: list[str] = []
    kept: list[str] = []
    cited_claims = 0
    correct_citations = 0
    for claim, ok in zip(claims, verdicts, strict=True):
        if claim.markers:
            cited_claims += 1
        if ok:
            supported.append(claim.text)
            correct_citations += (
                1  # support implies a citation held up (judge fails on no evidence).
            )
            kept.append(_render(claim))
        else:
            unsupported.append(claim.text)
            if not drop_unsupported:
                kept.append(_render(claim))

    report = GroundingReport(
        supported_claims=tuple(supported),
        unsupported_claims=tuple(unsupported),
        cited_claims=cited_claims,
        correct_citations=correct_citations,
    )
    text = " ".join(kept) if drop_unsupported else response.text
    return GroundingOutcome(
        text=text, citations=_resolve_citations(response, context), report=report
    )


# --- claim parsing ----------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MARKER_RE = re.compile(r"\[(\d+)\]")
# Markers at the very start of a fragment belong to the *previous* sentence: a marker that
# trails a claim ("... Sun. [1] Next claim.") lands at the head of the next split fragment.
_LEADING_MARKERS_RE = re.compile(r"^(?:\s*\[\d+\])+\s*")


@dataclass(frozen=True, slots=True)
class _Claim:
    """One answer sentence: its text with markers stripped, and the markers it cited."""

    text: str
    markers: tuple[int, ...]


def _parse_claims(text: str) -> list[_Claim]:
    """Split an answer into claim sentences, attaching each citation marker to its claim."""
    claims: list[_Claim] = []
    for raw in _SENTENCE_SPLIT_RE.split(text.strip()):
        fragment = raw.strip()
        if not fragment:
            continue
        lead = _LEADING_MARKERS_RE.match(fragment)
        if lead:
            _attach_markers(claims, _markers_in(lead.group()))
            fragment = fragment[lead.end() :].strip()
        body = _MARKER_RE.sub("", fragment).strip()
        markers = _markers_in(fragment)
        if body:
            claims.append(_Claim(text=body, markers=markers))
        else:
            _attach_markers(claims, markers)  # a marker-only tail: still the prior claim's.
    return claims


def _attach_markers(claims: list[_Claim], markers: tuple[int, ...]) -> None:
    """Append ``markers`` to the most recent claim, if any (else they have no owner)."""
    if claims and markers:
        prev = claims[-1]
        claims[-1] = _Claim(text=prev.text, markers=prev.markers + markers)


def _markers_in(text: str) -> tuple[int, ...]:
    return tuple(int(m) for m in _MARKER_RE.findall(text))


def _render(claim: _Claim) -> str:
    """Re-attach a claim's markers to its text (used when rebuilding a dropped answer)."""
    return claim.text + "".join(f" [{m}]" for m in claim.markers)


# --- evidence resolution ----------------------------------------------------------


def _cited_quotes(claim: _Claim, context: GroundedContext) -> tuple[str, ...]:
    """The quote text of each in-range marker the claim cited (empty if it cited nothing)."""
    return tuple(
        context.evidences[m - 1].quote for m in claim.markers if 1 <= m <= len(context.evidences)
    )


def _resolve_citations(response: LLMResponse, context: GroundedContext) -> tuple[Citation, ...]:
    """Map the response's declared markers to the evidence they point at, dropping any that
    fall outside the available evidence."""
    return tuple(
        Citation(marker=m, evidence=context.evidences[m - 1])
        for m in response.cited_markers
        if 1 <= m <= len(context.evidences)
    )
