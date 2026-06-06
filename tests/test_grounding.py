"""The Phase 1 grounding stage: entailment judges, per-claim attribution, drop vs flag.

Grounding is where "every claim is cited *and* the citation holds up" becomes a number.
These tests pin the two things Phase 0 couldn't do: judging each claim against the evidence
it actually cited (not a pool), and dropping unsupported claims rather than only flagging
them. Plain sync wrappers drive the async API via ``asyncio.run`` (CLAUDE.md §4).
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from racore.adapters.judges import SubstringEntailmentJudge, TokenOverlapEntailmentJudge
from racore.core import grounding
from racore.core.types import (
    Chunk,
    ClaimCheck,
    Evidence,
    GroundedContext,
    LLMRequest,
    LLMResponse,
    Query,
    Retrieval,
)
from racore.eval import demo_pipeline, golden_source

_MERCURY = "Mercury is the smallest planet."
_JUPITER = "Jupiter is the largest planet."


def _evidence(quote: str) -> Evidence:
    return Evidence(quote=quote, doc_id="d", chunk_id="c", start=0, end=len(quote), source="s")


def _context(*quotes: str) -> GroundedContext:
    return GroundedContext(query="q", evidences=tuple(_evidence(q) for q in quotes))


# --- entailment judges ------------------------------------------------------------


def test_substring_judge_requires_verbatim_evidence() -> None:
    asyncio.run(_substring_judge())


async def _substring_judge() -> None:
    judge = SubstringEntailmentJudge()
    quote = "Iron oxide on the surface of Mars gives it a reddish hue."
    verdicts = await judge.judge(
        [
            ClaimCheck(claim="a reddish hue", evidence=(quote,)),  # verbatim slice
            ClaimCheck(claim="Mars looks red from rust", evidence=(quote,)),  # paraphrase
            ClaimCheck(claim="anything at all", evidence=()),  # nothing cited
        ]
    )
    assert verdicts == [True, False, False]


def test_overlap_judge_accepts_paraphrase_substring_rejects() -> None:
    asyncio.run(_overlap_vs_substring())


async def _overlap_vs_substring() -> None:
    # A faithful paraphrase: every content word of the claim is in the quote, reordered.
    quote = "Mars is called the Red Planet because iron oxide gives it a reddish hue."
    paraphrase = [ClaimCheck(claim="Iron oxide gives Mars its reddish hue.", evidence=(quote,))]
    unrelated = [ClaimCheck(claim="Saturn has a prominent ring system.", evidence=(quote,))]

    # Same paraphrase: the exact judge rejects it, the overlap judge accepts it.
    assert await SubstringEntailmentJudge().judge(paraphrase) == [False]
    assert await TokenOverlapEntailmentJudge().judge(paraphrase) == [True]
    # Overlap is not a free pass: an unrelated claim still fails.
    assert await TokenOverlapEntailmentJudge().judge(unrelated) == [False]


def test_overlap_judge_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        TokenOverlapEntailmentJudge(threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        TokenOverlapEntailmentJudge(threshold=1.5)


# --- assemble + per-claim verification --------------------------------------------


def test_assemble_maps_each_retrieval_to_one_evidence() -> None:
    ranked = [_chunk_retrieval("alpha", 0), _chunk_retrieval("beta", 1)]
    context = grounding.assemble("q", ranked)
    assert tuple(e.quote for e in context.evidences) == ("alpha", "beta")
    assert context.evidences[0].chunk_id == "c0"
    assert context.marker_for(0) == 1


def _chunk_retrieval(text: str, ordinal: int) -> Retrieval:
    chunk = Chunk(
        id=f"c{ordinal}", doc_id="d", text=text, ordinal=ordinal, start=0, end=len(text), source="s"
    )
    return Retrieval(chunk=chunk, score=1.0)


def test_verify_attributes_each_claim_to_its_own_citation() -> None:
    asyncio.run(_verify_attribution())


async def _verify_attribution() -> None:
    context = _context(_MERCURY, _JUPITER)
    # Each claim cites the marker that actually supports it.
    right = LLMResponse(text=f"{_MERCURY} [1] {_JUPITER} [2]", cited_markers=(1, 2))
    outcome = await grounding.verify(right, context, SubstringEntailmentJudge())
    assert outcome.report.faithfulness == 1.0
    assert outcome.report.citation_correctness == 1.0
    assert outcome.report.cited_claims == 2

    # Swap the markers: both claims now cite evidence that does *not* support them. Pooling
    # every cited quote would call these supported; per-claim attribution does not.
    swapped = LLMResponse(text=f"{_MERCURY} [2] {_JUPITER} [1]", cited_markers=(1, 2))
    outcome = await grounding.verify(swapped, context, SubstringEntailmentJudge())
    assert outcome.report.supported_claims == ()
    assert outcome.report.faithfulness == 0.0
    assert outcome.report.citation_correctness == 0.0
    assert outcome.report.cited_claims == 2


def test_uncited_claim_is_unsupported_but_not_a_citation_error() -> None:
    asyncio.run(_uncited_claim())


async def _uncited_claim() -> None:
    context = _context(_MERCURY, _JUPITER)
    # First claim cited and correct; second claim asserts something with no citation at all.
    response = LLMResponse(text=f"{_MERCURY} [1] Jupiter is enormous.", cited_markers=(1,))
    outcome = await grounding.verify(response, context, SubstringEntailmentJudge())

    assert outcome.report.supported_claims == (_MERCURY,)
    assert outcome.report.unsupported_claims == ("Jupiter is enormous.",)
    # Faithfulness drops (an uncited assertion is unsupported)...
    assert outcome.report.faithfulness == 0.5
    # ...but citation correctness stays 1.0: the one citation made was right.
    assert outcome.report.cited_claims == 1
    assert outcome.report.citation_correctness == 1.0


def test_drop_unsupported_rewrites_answer_flag_keeps_it() -> None:
    asyncio.run(_drop_vs_flag())


async def _drop_vs_flag() -> None:
    context = _context(_MERCURY, _JUPITER)
    response = LLMResponse(text=f"{_MERCURY} [1] Jupiter is enormous.", cited_markers=(1,))

    flagged = await grounding.verify(response, context, SubstringEntailmentJudge())
    # Flag (default): text is untouched, the bad claim is reported.
    assert flagged.text == response.text
    assert flagged.report.unsupported_claims == ("Jupiter is enormous.",)

    dropped = await grounding.verify(
        response, context, SubstringEntailmentJudge(), drop_unsupported=True
    )
    # Drop: the answer is rebuilt from supported claims only; the bad claim is gone but
    # still reported, and the surviving claim keeps its citation marker.
    assert "Jupiter is enormous" not in dropped.text
    assert dropped.text == f"{_MERCURY} [1]"
    assert dropped.report.unsupported_claims == ("Jupiter is enormous.",)


# --- the drop policy end-to-end through the pipeline -------------------------------


class _PartlyGroundedLLM:
    """An LLM stand-in that emits one grounded sentence plus one fabricated one, so the
    pipeline's drop policy has something real to remove."""

    async def generate(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        responses = []
        for request in requests:
            quote = request.context.evidences[0].quote  # verbatim -> supported
            text = f"{quote} [1] The planet is made of green cheese [1]."
            responses.append(LLMResponse(text=text, cited_markers=(1,)))
        return responses


def test_pipeline_drops_unsupported_claims_when_configured() -> None:
    asyncio.run(_pipeline_drop())


async def _pipeline_drop() -> None:
    pipeline = dataclasses.replace(demo_pipeline(), llm=_PartlyGroundedLLM(), drop_unsupported=True)
    await pipeline.ingest(golden_source())

    answer = await pipeline.answer(Query(text="Which is the smallest planet?"))

    assert not answer.abstained
    assert "Mercury" in answer.text  # the grounded sentence survives...
    assert "green cheese" not in answer.text  # ...the fabricated one is dropped...
    # ...and the drop is recorded, not silent.
    assert any("green cheese" in claim for claim in answer.grounding.unsupported_claims)
    assert answer.grounding.faithfulness == 0.5
