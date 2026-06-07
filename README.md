# RaCore

> A reusable, provider-agnostic engine for building **grounded, real-time, memory-aware**
> retrieval systems you can actually trust — with **evidence and evaluation as first-class
> concerns**, not afterthoughts.

RaCore is a **clean-room, independently-owned engine**. It plugs into any application via an
in-process Python SDK or an HTTP service, and shares **no code** with any downstream product.

---

## Why this exists (the one-paragraph thesis)

Naive RAG — *chunk → embed → cosine search → stuff into a prompt* — is commoditizing into a
library call. What is **not** commoditizing, and is becoming the backbone of serious AI products,
is the **reliability layer**: getting the *right, current, permissioned, trustworthy* information
into a model's context **with provenance**, knowing **when not to answer**, **remembering** the
user across turns, and being able to **prove all of it with numbers**. Bigger context windows and
smarter models do not remove that problem — they make it more central. RaCore is a bet on that layer.

## The pillars

| Pillar | What it means | Where it lives |
|---|---|---|
| **Grounding** | every claim is backed by a verbatim source + citation; unsupported claims are dropped or flagged | `core/grounding.py` |
| **Freshness** | knowledge stays current via incremental indexing + connectors; staleness is visible | `adapters/sources/` |
| **Relevance** | retrieve the right context, filter the rest, and **abstain** when there's no support | retrieval + rerank |
| **Memory** *(differentiator)* | per-user persistent memory that personalizes across sessions | `adapters/memory/` |
| **Evaluation** *(the moat)* | retrieval / grounding / answer / memory quality measured + gated in CI | `eval/` |

## Repo shape

```
RaCore/                     ← this repo (the engine; Apache-2.0)
  src/racore/               ← provider-agnostic core + pluggable adapters
  docs/                     ← architecture, evaluation, memory, roadmap, decisions
  tests/
client-<name>/              ← (separate repo) a consumer's config + plugins — NOT here
```

A consumer (including any product that adopts RaCore) depends on a **released, pinned version** of
this package and contributes only **config + plugins** in its own repo. Engine changes always land
here, never in a consumer repo. See [`CLAUDE.md`](CLAUDE.md) for the house rules that enforce this.

## Status

**Phase 0 — Foundation: the walking skeleton is live**, and **Phase 1 — Grounding is underway.** A
thin end-to-end slice runs through the real ports at **$0** external spend — ingest → retrieve →
rerank → ground → cite → answer — with per-stage timing (ADR-0010) and content-hash IDs (ADR-0011),
plus an eval harness that prints a baseline over a golden set.

Grounding is now a real, pluggable stage (`core/grounding.py`, ADR-0012): each claim is attributed
to the evidence *it* cited and judged against only that span, unsupported claims are **dropped or
flagged**, and the entailment check is a swappable `EntailmentJudge` port — a strict substring judge
by default (so faithfulness is never inflated) with a paraphrase-tolerant token-overlap judge ready
for richer generators, and a real LLM judge as a future drop-in.

On the golden set today, **grounding faithfulness and citation correctness are 1.0** — the extractive
$0 generator only ever quotes the evidence it cites. Retrieval, by contrast, is now a deliberate and
**measurable** gap: the corpus carries distractors and paraphrase-gap questions, so the lexical $0
retriever reads **recall@k ≈ 0.94** (ADR-0014). The rank-aware view confirms the weakness recall
hides — **nDCG@k ≈ 0.86, MRR ≈ 0.89**, both below recall, because the right doc is frequently
retrieved but not ranked first (ADR-0015). Those are the numbers a real embedding adapter, hybrid
retrieval, and a reranker must beat. The first of these — an opt-in **Voyage** semantic embedder —
is now wired behind the `EmbeddingProvider` port (ADR-0016); run it with `--embedder voyage` to
measure the lift, while the `$0` mock stays the default. Refusal accuracy is intentionally **below
1.0** — there is no abstention logic yet, so the
harness *surfaces* that gap for Phase 2 rather than hiding it. Build order and per-phase "definition
of done" are in [`docs/roadmap.md`](docs/roadmap.md).

## Docs

- [`docs/architecture.md`](docs/architecture.md) — ports-and-adapters design, core types, the ingest/answer pipelines, how apps connect.
- [`docs/evaluation.md`](docs/evaluation.md) — the measurement strategy (the differentiator).
- [`docs/latency.md`](docs/latency.md) — latency & streaming: replying promptly without awkward pauses.
- [`docs/memory.md`](docs/memory.md) — the per-user persistent memory subsystem.
- [`docs/roadmap.md`](docs/roadmap.md) — the phased build plan + the public learning-doc outline.
- [`docs/decisions.md`](docs/decisions.md) — the architecture decision log (why each choice was made).

## Getting started (for the builder)

1. **Python 3.12+** and [`uv`](https://docs.astral.sh/uv/). Node 20+ later for the demo UI.
2. `uv sync` — creates the venv and installs the dev toolchain (ruff, mypy, pytest, pre-commit).
3. `uv run pre-commit install` — activates the commit-time quality gate.
4. `uv run pytest` — runs the test suite. In the lean (default) environment this is **green with one
   skip**: a single contract test that pins the real Anthropic SDK's client shape runs only when the
   optional `anthropic` extra is installed, so it is skipped here by design (ADR-0007 keeps the paid
   SDK out of the core gate). To include it: `uv run --extra anthropic pytest` (skip → pass).
5. `uv run python -m racore.eval` — ingests the golden corpus and prints the baseline metrics.

### Quickstart (ingest → ask → grounded answer)

```python
import asyncio

from racore.adapters import InMemoryDocumentSource
from racore.core.types import Query
from racore.eval import demo_pipeline


async def main() -> None:
    pipeline = demo_pipeline()  # deterministic, $0 in-memory adapter stack
    await pipeline.ingest(
        InMemoryDocumentSource(
            [("notes/saturn", "Saturn is famous for its prominent ring system made of ice.")]
        )
    )
    answer = await pipeline.answer(Query(text="What is Saturn famous for?"))
    print(answer.text)
    for citation in answer.citations:
        print(f"  [{citation.marker}] {citation.evidence.source}: {citation.evidence.quote}")


asyncio.run(main())
```

Swap any adapter (embeddings, vector store, reranker, LLM, …) for a real provider without touching
the core — that boundary is the ports-and-adapters design in
[`docs/architecture.md`](docs/architecture.md).

### Stress-testing grounding with a real model (opt-in)

The `$0` stack quotes evidence verbatim, so grounding scores a perfect 1.0 — which proves the wiring,
not that grounding is hard to fool. To challenge it with a real, *paraphrasing* generator behind the
same `LLMProvider` port:

```bash
# put your key in a gitignored .env first:  ANTHROPIC_API_KEY=sk-ant-...
# `--extra anthropic` installs the optional SDK for the run (the core stays dependency-free);
# `--env-file .env` loads the key. Same run, two judges:
uv run --extra anthropic --env-file .env python -m racore.eval --llm anthropic --judge substring
uv run --extra anthropic --env-file .env python -m racore.eval --llm anthropic --judge overlap
```

The strict `substring` judge surfaces the **faithfulness gap** — a real generator reformats sentences
rather than quoting, so exact-match grounding collapses (faithfulness ≈ **0.0** on a Haiku run, the
honest extreme). The paraphrase-tolerant `overlap` judge recovers most of it (faithfulness ≈ **0.75**,
citation-correctness ≈ **0.75** on the same answers) — same answers, two judges, which is exactly why
the entailment check is a swappable port. (Illustrative; values move with the model and run.) Two
things the real model reveals: **answer correctness rises to ≈ 1.0** — a capable reader answers
correctly even when a distractor wins rank 1, which is why retrieval keeps its own rank-aware metrics
rather than being judged through the answer; and **retrieval recall@k stays ≈ 0.94** regardless of
generator or judge (it is upstream of both). The model and its params are configurable via
`AnthropicConfig`
(default `claude-haiku-4-5-20251001`, `temperature=0.0`, both pinned for reproducible eval); override
per run with `--model <id>`. The core and the default test path never import the SDK.

The harness reports **real tokens/answer and cost/answer** from the provider's reported usage, priced
by a verified table (`eval/pricing.py`). The `$0` stack reports a true $0; a model with no price entry
shows tokens with `cost n/a` rather than a fake $0. Cost accounting is provider-agnostic — any adapter
that fills in `LLMResponse.usage` is priced the same way; supporting a new provider is a table entry,
not a code change. Add `-v` for per-case detail (which question, the answer, the unsupported claims).

Cost/answer counts **every billed component** — the generator, a paid embedder (`--embedder voyage`),
and the LLM judge (`--judge llm`) — summed and priced from each one's reported token usage (ADR-0018).
The `$0` default stack reports a **true $0**; if any paid component's model isn't in the price table,
the figure shows **`n/a`** rather than a misleading $0. Billed adapters surface usage through an
*optional* `UsageReporter` port, so the frozen batch-first I/O signatures stay unchanged and adding a
new paid component is a price-table row plus that one method — never a core change.

**Verifying grounding semantically (opt-in).** The lexical judges (`substring`, `overlap`) can't
credit a faithful paraphrase — "Mercury is the *world* closest to our *star*" is grounded by "…closest
to the *Sun*" but scores 0.0. The `llm` judge uses Claude to decide entailment on *meaning*:

```bash
uv run --extra anthropic --env-file .env python -m racore.eval --llm anthropic --judge llm -v
# the full stack — best retrieval + real generator + semantic judge, all at once:
uv run --extra voyage --extra anthropic --env-file .env python -m racore.eval --embedder voyage --llm anthropic --judge llm -v
```

It's one adapter behind the same `EntailmentJudge` port as the deterministic judges (ADR-0017); the
strict `substring` judge stays the default floor. On a real Haiku run the judge lifts faithfulness
from ≈ **0.78** (overlap, lexical) to ≈ **0.93–0.96** (llm, semantic) — meeting the ≥ 0.95 target
([`docs/evaluation.md`](docs/evaluation.md) §5), with `citation_correctness` ≈ 1.0. The small residual
is a model citation-style quirk (a verbatim restatement left uncited), not a judging error. The
generation prompt now asks for concise, **one-citation-per-sentence** answers to curb it (ADR-0019) —
the parser stays strict, since an uncited claim genuinely *is* unsupported. The LLM judge adds latency
(per-claim calls); its tokens are counted in cost/answer alongside the generator and embedder (ADR-0018).

### Upgrading retrieval with a real embedder (opt-in)

The `$0` embedder is purely lexical (shared-vocabulary cosine), so it buries answer docs that
paraphrase the question — "world/star" never matches "planet/Sun" (recall@k ≈ 0.94, nDCG@k ≈ 0.86).
A real *semantic* embedder is a drop-in behind the `EmbeddingProvider` port; the first one is Voyage:

```bash
# put your key in the gitignored .env:  VOYAGE_API_KEY=pa-...
uv run --extra voyage --env-file .env python -m racore.eval --embedder voyage -v
# combine with a real generator (one line; backslash continuations don't work in PowerShell):
uv run --extra voyage --extra anthropic --env-file .env python -m racore.eval --embedder voyage --llm anthropic --judge overlap -v
```

This is **not** a lock-in: Voyage is one adapter behind the port, exactly as `AnthropicLLM` is one of
many LLM adapters. OpenAI, Cohere, or a **local model** (Ollama `nomic-embed-text`, BGE) drop in the
same way with zero core change. The `$0` `MockEmbeddingProvider` stays the default; `--embedder voyage`
opts in. Model and key are configurable via `VoyageConfig` (default `voyage-3.5`); override the model
with `--embed-model <id>`. The core and the default test path never import the SDK (ADR-0016).

## License

**Apache-2.0** — see [`LICENSE`](LICENSE). You keep an irrevocable right to reuse this engine;
anyone may adopt it. Consumer-specific customization is separate code owned by the consumer.
