# Roadmap

> **Breadth-first walking skeleton, then depth-per-pillar.** Never build what you can't measure.
> Each phase produces, at once: a **mastered component**, a **doc chapter**, and a **demo
> capability** — so learning and the asset compound together.

---

## Phases at a glance

| Phase | Goal | Definition of done | Doc chapter |
|---|---|---|---|
| **0 — Foundation** | the spine + measurement | E2E thin slice runs; eval harness produces a baseline number; one passing test | Ch 0–1 |
| **1 — Grounding** | every claim cited + verified | faithfulness + citation-correctness measured; unsupported claims dropped/flagged | Ch 2 |
| **2 — Relevance & refusal** | right context, or abstain | abstention accuracy measured; false-answer-on-no-evidence rate down | Ch 3 |
| **3 — Freshness** | knowledge stays current | incremental re-index works; staleness surfaced; a live connector | Ch 4 |
| **4 — Memory** | per-user personalization | fact-recall + personalization-lift measured | Ch 5 |
| **5 — Productization** | multi-tenant + packaged | API/SDK, observability, cost/latency dashboard, open-core release | Ch 6–7 |

## Phase 0 — Foundation (build this first)

The walking skeleton that makes "just right from the start" real. Concrete deliverables:

- `pyproject.toml` (pip-installable), package skeleton, dev tooling, **the quality gate** — done.
- `core/types.py` + `core/ports.py` — the domain objects and the port `Protocol`s
  (**async-first, batch-first** — see ADR-0009).
- `core/pipeline.py` — minimal `ingest()` + `answer()` wired through the **real ports**, with
  **per-stage timing** built in (ADR-0010) and **content-hash IDs** at ingest (ADR-0011).
- A **complete in-memory adapter set** (mock/local embeddings, in-memory vector store, noop
  reranker, a simple chunker, a PDF/text source, a file memory store, one LLM provider) → the slice
  runs with **$0** external spend.
- `eval/harness.py` + `eval/metrics.py` + a tiny **golden dataset** (5–10 Q→answer→source rows),
  reporting quality **and** latency/cost.
- **One passing end-to-end test**: ingest a sample document → ask a question → get an answer **with a
  citation** → harness prints a **baseline** retrieval + grounding number.

**DoD:** `pytest` green; `python -m racore.eval` prints baseline metrics; README quickstart works.

After Phase 0, every later phase is *fill-in*, not *re-architect*.

## The public learning doc (chapters map 1:1 to phases)

Working title: **"Grounded RAG: building retrieval systems you can actually trust."** You never
write a chapter you haven't built, or build a thing you can't measure.

- **Ch 0 — Why naive RAG fails in production** *(P0)* — the strawman + real failure modes; sets the stakes.
- **Ch 1 — Architecture & the three pillars** *(P0)* — grounding / freshness / relevance + ports-and-adapters.
- **Ch 2 — Grounding & provenance** *(P1)* — hybrid retrieval, rerank, citations, faithfulness eval, drop-ungrounded.
- **Ch 3 — Relevance & refusal** *(P2)* — query understanding, filtering, knowing when to abstain.
- **Ch 4 — Keeping knowledge current** *(P3)* — incremental indexing, connectors, staleness.
- **Ch 5 — Agent memory done right** *(P4)* — per-user memory, write/compaction, memory vs. retrieval.
- **Ch 6 — Evaluation: how you prove any of this** *(cross-cutting)* — datasets, metrics, regression gates. **The moat.**
- **Ch 7 — Productizing** *(P5)* — multi-tenant, cost/latency, observability, open-core.

## Sequencing notes

- Lead with **grounding (Phase 1)** — it's the strongest, most differentiated, most credible pillar.
- Don't over-polish one component before the skeleton exists; depth comes *per phase* on top of a
  working, measured spine.
- Ship Ch 0–2 publicly after Phase 1 — that's enough proof to demonstrate the approach.
