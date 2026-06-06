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
