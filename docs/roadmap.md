# Roadmap

> **Breadth-first walking skeleton, then depth-per-pillar. Never build what you can't measure.**
> Each phase produced, at once, a **mastered component** and a **demo capability** — so capability and
> proof compounded together, one pillar at a time.

This is the engine's **technical** roadmap: the phase plan, each phase's definition of done, and
what's next. The realized per-phase detail and headline numbers live in the [README status](../README.md#project-status);
the *why* behind every decision is the [decision log](decisions.md).

---

## Build philosophy

1. **Breadth before depth.** A thin end-to-end slice through the real ports first (Phase 0), so every
   later phase is *fill-in*, not *re-architect*.
2. **Eval-first.** The measurement harness is wired on day one. A change you can't measure doesn't ship:
   baseline → change → re-measure → keep only if a metric improved without regressing another.
3. **Each phase = a component + a capability + a number.** No phase is "done" on vibes; it's done when
   its definition of done is a measured fact.

## Phases — all shipped ✅

| Phase | Goal | Definition of done | Status |
|---|---|---|---|
| **0 — Foundation** | the spine + measurement | E2E thin slice runs; eval harness produces a baseline number; one passing test | ✅ done |
| **1 — Grounding** | every claim cited + verified | faithfulness + citation-correctness measured; unsupported claims dropped/flagged | ✅ done |
| **2 — Relevance & refusal** | right context, or abstain | abstention accuracy measured; false-answer-on-no-evidence rate down | ✅ done |
| **3 — Freshness** | knowledge stays current | incremental re-index works; staleness surfaced; a live connector | ✅ done |
| **4 — Memory** | per-user personalization | fact-recall + personalization-lift measured | ✅ done |
| **5 — Productizing** | multi-tenant + packaged | service/API surface; multi-tenancy end-to-end; grounding-gated caching; observability; open-core line | ✅ done |

Each definition of done is met; the [README status](../README.md#project-status) carries the headline
result per phase, and the [decision log](decisions.md) records the ADRs behind them. The engine now
stands on all four pillars — **grounded · relevant · current · memory-aware** — each proven by a
number, and is deployable as a dependency-free service.

## Why this order (sequencing rationale)

- **Lead with grounding (Phase 1)** — the strongest, most differentiated, most credible pillar, and the
  one the rest depend on (you can't measure relevance or memory honestly without it).
- **Don't over-polish a component before the skeleton exists** — depth comes *per phase* on top of a
  working, measured spine, never ahead of it.
- **Freshness and memory after relevance** — both are only measurable once retrieval and refusal are
  honest (a stale or personalized answer still has to be grounded and relevant first).
- **Productizing last** — wiring a transport over a *proven* engine is additive; doing it earlier would
  have meant productizing something unmeasured.

## Beyond the phases — what's next

The five phases are complete. These are the measured next steps, each **deferred deliberately** until a
concrete case makes it measurable — the same "don't build what you can't measure" rule that shaped the
phases (e.g. the reranker waits for a corpus where it has a gap to close):

- **Memory — recency-aware ranking.** Fold an injected `now` into `MemoryStore.read` over a *dated*
  memory corpus, so a fresh relevant fact can outrank a stale exact one. Deferred until there's a `now`
  seam and dated data to measure against, so it ships with a number.
- **Memory — compaction / TTL.** Fold many episodic items into summaries and expire working memory
  (memory.md §3). Needs memory *at scale* to show the win.
- **Caching — the semantic answer tier.** Selecting a cached candidate by embedding similarity, then
  re-validating it by grounding (the opt-in tier in productizing.md §4). Needs a semantic embedder to
  tune its similarity threshold; the always-safe grounding gate it would reuse is already built.
- **Relevance — tighten the cascade.** A larger false-refusal corpus to trust the gate bands past the
  honest, un-overfit 0.94.
- **Freshness — a filter/penalty + a gated metric.** A freshness-aware re-ranking penalty and a *gated*
  freshness number, measured on a genuinely dated corpus rather than synthetic ages.
- **Retrieval — hybrid + reranker.** BM25/dense hybrid (→ recall) and a reranker (→ nDCG/MRR), once a
  harder, keyword-heavy corpus (rare terms, IDs, acronyms) gives them a measurable gap to close.
- **Evaluation — public benchmarks.** Sanity-check the engine against the field with CUAD (contract
  clause QA) and a small financial-QA set, alongside the bespoke golden regression set.
- **Productizing — the managed operation.** Auth, rate-limiting, quotas, autoscaling, and dashboards
  over the observability stream live *around* the engine (the open-core line, productizing.md §6), not
  inside it.

Every one of these stays gated: baseline → change → re-measure → keep only if a metric improved without
regressing another.
