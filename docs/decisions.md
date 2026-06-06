# Architecture Decision Log (ADRs)

> Short, dated records of *why* each foundational choice was made — so future-you (or a collaborator)
> doesn't re-litigate settled questions. Add a new ADR when a decision changes; don't edit history.

Format: **Context → Decision → Consequences.**

---

### ADR-0001 — Ports-and-adapters core
**Context:** the engine must be reusable across consumers and swappable across providers (Voyage/
OpenAI/pgvector/etc.) without forks. **Decision:** a provider-agnostic core behind `Protocol` ports;
every varying concern is an adapter. **Consequences:** customization = config + plugins; a need that
can't be config/plugin signals a new port, not a fork; slight upfront interface-design cost.

### ADR-0002 — Python core + thin TS/Next.js demo
**Context:** best RAG/eval ecosystem, employability, and in-process embedding into other Python apps.
**Decision:** Python 3.12 engine; Next.js/TypeScript only for the demo UI. **Consequences:** matches
the broader ecosystem; one extra language at the edge; the engine ships as a pip package.

### ADR-0003 — Contracts / financial filings as the flagship corpus
**Context:** the demo needs one corpus that proves grounding under pressure. **Decision:** lead with
contracts/financial filings (high grounding stakes). **Consequences:** harder corpus = stronger
proof; CUAD available as a public benchmark; the engine stays generic (this is just showcase config).

### ADR-0004 — Apache-2.0 license for the core
**Context:** the engine should be maximally adoptable and trusted, while remaining the author's
reusable, independently-owned asset. **Decision:** license the core **Apache-2.0**. **Consequences:**
anyone may adopt it; the author retains copyright and an irrevocable right to reuse published code;
consumer-specific customization is separate, consumer-owned code. *(Commercial/IP process is tracked
outside this technical log.)*

### ADR-0005 — Eval-first / walking-skeleton
**Context:** RAG quality is invisible without measurement; "expert depth" must be provable.
**Decision:** build a thin end-to-end slice **and** the eval harness in Phase 0, before optimizing any
component; gate regressions in CI. **Consequences:** every later change is measured; slightly slower
start, far faster and safer iteration after.

### ADR-0006 — Memory as a first-class subsystem
**Context:** per-user persistent memory is the least-saturated, most differentiating capability.
**Decision:** treat memory as a first-class port + pipeline stage (read before retrieval, write
after), with provenance, compaction, and conflict resolution — not a bolt-on. **Consequences:** more
design surface; a genuinely differentiated result. See [`memory.md`](memory.md).

### ADR-0007 — Provider-agnostic with free/local defaults for the public demo
**Context:** the public showcase must not cost money to run. **Decision:** default the demo to a
**local embedding model + in-memory/pgvector store**, with Voyage/OpenAI/Anthropic as drop-in adapter
upgrades. **Consequences:** the public demo costs **$0**; paid providers are opt-in config, not a
dependency.

### ADR-0008 — Engine name: RaCore (neutral, independently owned)
**Context:** an earlier working name tied the engine to a specific product brand, which would weaken
neutral adoption and muddy ownership. **Decision:** name the engine **RaCore** — neutral, brand-free,
owned by the author personally; any product (including the author's own) is a **downstream consumer**,
not the owner. **Consequences:** clean-room boundary stays intact; engine remains sellable to other
consumers; package is `racore`; no product code or product branding enters this repo.

### ADR-0009 — Async-first, batch-first ports with a streamable answer type
**Context:** the costly scaling rework in RAG engines is not swapping infrastructure (ports already
cover that) — it's the **method signatures**. Converting sync→async, single→batch, or
buffered→streaming after adapters exist forces a rewrite of every adapter and caller. **Decision:**
from day one — (a) every I/O port method is `async def`; a thin **sync facade** may wrap the async
core, never the reverse; (b) I/O ports are **batch-first** (take/return lists); (c) `answer()` returns
a **streamable result type** even before streaming is implemented. **Consequences:** concurrency,
throughput batching, and streaming are all additive later — no signature churn; small upfront
discipline cost. These signatures are effectively frozen once adapters depend on them.

### ADR-0010 — Per-stage timing + latency/cost as gated metrics
**Context:** "patches won't add latency" is not a property of architecture — it's a property of a
**guardrail**. You also cannot fix a latency regression you can't localize. **Decision:** instrument
every pipeline stage with timing from Phase 0; record p50/p95 latency and cost/answer in the eval
harness; **gate** them in CI so a regression fails the build like a faithfulness drop does.
**Consequences:** latency/cost regressions are caught and attributable per stage; the harness carries
an ops dimension, not just a quality dimension.

### ADR-0011 — Stable content-hash IDs for idempotent ingest
**Context:** Phase 3 incremental re-indexing needs idempotent upserts; random IDs at ingest would
force a rework to support it. **Decision:** derive document and chunk IDs deterministically from a
content hash at ingest time. **Consequences:** re-ingesting unchanged content is a no-op; incremental
freshness and dedup work later with no schema change; a one-line decision made early.

### ADR-0012 — Grounding as a pluggable stage with an `EntailmentJudge` port
**Context:** Phase 0 inlined one deterministic substring check in the pipeline: it pooled *all* cited
quotes per answer, so it could not attribute a claim to the specific evidence it cited, could not
tell an uncited claim from a supported one, and left no seam for a paraphrase-aware or LLM judge.
Phase 1's definition of done is *faithfulness + citation-correctness measured; unsupported claims
dropped/flagged.* **Decision:** extract grounding into `core/grounding.py` as a real stage that (a)
attributes each claim to the markers it cited and judges it against **only** that evidence, (b) makes
entailment an `EntailmentJudge` **port** with deterministic $0 adapters — `SubstringEntailmentJudge`
(exact, the strict default) and `TokenOverlapEntailmentJudge` (paraphrase-tolerant) — and (c) supports
a **drop-or-flag** policy for unsupported claims; plus add **citation correctness** as a gated metric
beside faithfulness. **Consequences:** a real LLM entailment judge is a drop-in adapter with no caller
change; the strict default keeps the headline faithfulness number honest; faithfulness (penalizes
uncited claims) and citation-correctness (scores only the citations actually made) are now distinct,
separately gated numbers; the drop policy can rewrite an answer down to its supported claims. The
golden-set baseline is unchanged, because the extractive $0 path quotes evidence verbatim.

### ADR-0013 — Record model refusals as abstentions (measurement integrity)
**Context:** A capable LLM, told by the system prompt to say "I don't know" when the evidence
lacks the answer, correctly refuses on negative-control questions. But the pipeline set
`abstained` only on *empty retrieval*, so a textual refusal was scored as a normal answer —
its ungrounded refusal sentences drove faithfulness down, and refusal accuracy counted the
correct refusal as a false answer. Per-case eval (`-v`) on a real Haiku run made it concrete:
all six answerable questions scored faithfulness 1.0 (overlap judge), and the aggregate 0.75
came *entirely* from two correctly-refused controls. **Decision:** detect the mandated "I
don't know" refusal in the generated answer and set `abstained=True`. Faithfulness already
excludes abstained cases; refusal accuracy credits an abstention on a no-evidence question.
This **records what the model did** — deliberately distinct from the Phase 2 capability of
**deciding** when to abstain (retrieval-confidence, query understanding), which stays
deferred. Detection is a heuristic on the phrasing the system prompt mandates.
**Consequences:** faithfulness and refusal aggregates reflect reality — a correct refusal is
no longer double-penalized (with a scripted-refusal pipeline both read 1.0). The $0 mock
baseline is unchanged (the extractive LLM never refuses). The heuristic could miss a reworded
refusal or false-positive on an answer that literally opens "I don't know"; acceptable for a
seed, hardened when Phase 2 builds a real abstention decision.
