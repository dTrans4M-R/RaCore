# RaCore

> A reusable, provider-agnostic engine for building **grounded, real-time, memory-aware** retrieval
> systems you can actually trust — with **evidence and evaluation as first-class concerns**, not
> afterthoughts.

RaCore is the reusable **core** of a retrieval system: the part that gets the *right, current,
trustworthy* information into a language model's context **with provenance**, knows **when not to
answer**, **remembers** the user across sessions, and can **prove all of it with numbers**. It runs
with **zero external spend and zero runtime dependencies** by default, and every real provider
(embeddings, vector store, LLM, …) is an opt-in drop-in behind a stable interface.

It is a **clean-room, independently-owned** engine: it plugs into any application via an in-process
Python API or an HTTP service, and shares **no code** with any downstream product.

> *Why the name?* **RaCore** is **RA**G + **Core**, with the *G* quietly dropped — so the engine that's
> all about retrieval actually rolls off the tongue. 🙂

- **New here?** Read [What is RaCore?](#what-is-racore) → [Key concepts](#key-concepts-plain-english)
  → [Quickstart](#quickstart). That's the 10-minute on-ramp.
- **Evaluating it?** Skim [The five pillars](#the-five-pillars) and [Project status](#project-status).
- **Going deep?** The [Documentation](#documentation) index links every technical doc and the full
  decision log.

---

## Table of contents

- [What is RaCore?](#what-is-racore)
- [Why RaCore exists](#why-racore-exists)
- [Key concepts (plain English)](#key-concepts-plain-english)
- [The five pillars](#the-five-pillars)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Project status](#project-status)
- [Going further: real providers (opt-in)](#going-further-real-providers-opt-in)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Development and the quality gate](#development-and-the-quality-gate)
- [Ownership and the clean-room boundary](#ownership-and-the-clean-room-boundary)
- [License](#license)

---

## What is RaCore?

**RAG** — *retrieval-augmented generation* — is the standard way to make a language model answer from
*your* documents instead of only its training data: you **retrieve** the relevant text and put it in
the model's prompt so it can generate a grounded answer. The naive version (chunk a document, embed
it, do a cosine search, stuff the top hits into a prompt) is a few lines of code and is fast becoming
a commodity library call.

**RaCore is the layer that makes RAG trustworthy enough to ship.** Naive RAG works on the three
questions you tried and quietly fails on the fourth — it will confidently make things up, serve stale
information, forget who it's talking to, and give you no way to know any of that happened. RaCore is
the engine that closes those gaps and, crucially, **measures** that it closed them:

- every claim in an answer is **traced to a source** you can click through to,
- the system **says "I don't know"** instead of fabricating when the evidence isn't there,
- the index stays **current** as documents change,
- it **remembers** each user across sessions, and
- all of the above is **scored by an evaluation harness** that blocks a change if a number regresses.

**Who it's for:** engineers building a retrieval/QA/assistant product who need it to be *correct and
provable*, not just demoable — especially in high-stakes domains (contracts, filings, support,
internal knowledge) where a confidently wrong answer is the failure that matters.

## Why RaCore exists

Naive RAG is commoditizing into a library call. What is **not** commoditizing — and is becoming the
backbone of serious AI products — is the **reliability layer**: getting the *right, current,
permissioned, trustworthy* information into a model's context **with provenance**, knowing **when not
to answer**, **remembering** the user across turns, and being able to **prove all of it with numbers**.
Bigger context windows and smarter models don't remove that problem — they make it more central.
RaCore is a focused bet on that layer, built one measured failure mode at a time.

## Key concepts (plain English)

The few terms that recur throughout the docs, in everyday language:

| Term | In plain English |
|---|---|
| **RAG** | *Retrieval-augmented generation.* Find the relevant text first, then let the model answer from it (not from memory). |
| **Grounding** | Making sure every statement in an answer is actually backed by a retrieved source — and dropping or flagging any that isn't. |
| **Citation / provenance** | The pointer from a claim back to the exact source span it came from, so you can verify it. |
| **Faithfulness** | The headline quality score: the fraction of an answer's claims that are genuinely supported by their cited evidence. |
| **Abstain / refusal** | The system choosing to say *"I don't know"* when the evidence doesn't support an answer — instead of guessing. |
| **Freshness** | Keeping the index current as documents are added, edited, or removed — and making the *age* of an answer's evidence visible. |
| **Memory** | Per-user facts the system remembers across sessions (your name, preferences, constraints), kept separate from the shared document corpus. |
| **Evaluation / the harness** | The test rig that scores retrieval, grounding, answers, and memory against a known-answer dataset — the project's source of truth. |
| **Ports & adapters** | The plugin design: the core depends on *interfaces* (ports); each concrete provider (Voyage, OpenAI, pgvector…) is an *adapter*. Swap providers without touching the core. |
| **The `$0` stack** | The default set of adapters that runs entirely locally with no API spend — so the engine is fully usable, and testable, for free. |
| **Multi-tenancy** | One deployment serving many independent customers, with each tenant's documents and each user's memory strictly isolated. |

## The five pillars

| Pillar | What it means | Learn more |
|---|---|---|
| **Grounding** | every claim is backed by a verbatim source + citation; unsupported claims are dropped or flagged | [architecture.md](docs/architecture.md) · [evaluation.md](docs/evaluation.md) |
| **Freshness** | knowledge stays current via incremental indexing + live connectors; staleness is visible | [freshness.md](docs/freshness.md) |
| **Relevance** | retrieve the right context, filter the rest, and **abstain** when there's no support | [architecture.md](docs/architecture.md) |
| **Memory** *(differentiator)* | per-user persistent memory that personalizes across sessions | [memory.md](docs/memory.md) |
| **Evaluation** *(the moat)* | retrieval / grounding / answer / memory quality measured **and gated** in CI | [evaluation.md](docs/evaluation.md) |

The engine is built **breadth-first** (a thin end-to-end slice), then **depth-per-pillar** — so
capability and proof compound together. See the [roadmap](docs/roadmap.md).

## Quickstart

**Prerequisites:** Python 3.12+ and [`uv`](https://docs.astral.sh/uv/). Then:

```bash
uv sync                  # create the venv + install the dev toolchain
uv run pytest            # run the test suite (green, $0, offline)
uv run python -m racore.eval   # ingest the golden corpus + print baseline metrics
```

### Embed it in Python (ingest → ask → grounded answer)

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

Every adapter (embeddings, vector store, reranker, LLM, judge, …) swaps for a real provider **without
touching the core** — that boundary is the ports-and-adapters design in
[architecture.md](docs/architecture.md).

### Run it as a service (HTTP, $0, zero dependencies)

The same engine behind a transport-agnostic facade and a **dependency-free ASGI app** — no web
framework, so any ASGI server runs it (see [productizing.md](docs/productizing.md)):

```python
from pathlib import Path

from racore.service import create_app, demo_service

# The $0 stack wired with per-user memory + the grounding-gated cache.
app = create_app(demo_service(Path("./.racore-memory")))
# then:  uvicorn module:app   (uvicorn is a deploy choice, never a package dependency)
```

```bash
curl -s localhost:8000/ingest -d '{"tenant_id":"acme","documents":[{"source":"sky","text":"Mercury is closest to the Sun."}]}'
curl -sN localhost:8000/answer -d '{"text":"What is closest to the Sun?","tenant_id":"acme"}'
#   → SSE: `token` events stream the text, then one `done` event carries citations + grounding.
```

Routes: `GET /health`, `POST /ingest`, `POST /answer` (SSE), `GET`/`POST /memory`. Every call is
tenant-scoped, repeat asks are served from the grounding-gated cache, and each request emits a
structured observability event. No socket is needed to *test* it — the suite drives the ASGI app
through the raw protocol with fake `receive`/`send`. The full HTTP contract — request/response
schemas and the SSE event shapes — is documented in [`docs/openapi.yaml`](docs/openapi.yaml)
(OpenAPI 3.1), kept honest by a drift-guard test that asserts it against the served routes.

## How it works

**Ports & adapters (hexagonal).** The core orchestration knows nothing about any concrete provider.
Everything that varies is a **port** (an interface); a provider is an **adapter** behind it.
Customization is swapping or configuring adapters — never forking the core. The ports include
`EmbeddingProvider`, `VectorStore`, `Reranker`, `Chunker`, `DocumentSource`, `MemoryStore`,
`MemoryExtractor`, `LLMProvider`, `EntailmentJudge`, `RelevanceGate`, `AnswerCache`, and `Evaluator`
(full table in [architecture.md](docs/architecture.md) §4).

**Two pipelines:**

- **`ingest(source, tenant)`** — fetch → chunk → diff → embed → upsert (→ prune). The diff is on a
  content-hash ID, so re-ingesting unchanged content does no work and editing a document leaves no
  stale copy behind (incremental re-index).
- **`answer(query, tenant, user)`** — memory.read → understand → embed → retrieve → rerank →
  relevance gate → assemble → generate → verify → memory.write. Grounding, relevance, and memory are
  *stages on this path*, not add-ons. It returns a **streamable** `Answer` with citations and a
  grounding report — or abstains when the evidence is too weak.

Two things are built in from day one rather than bolted on: **per-stage timing** (so a latency
regression is localizable) and **content-hash IDs** (so re-ingest is idempotent and caching can be
gated by grounding). That discipline is why later phases — streaming, caching, the HTTP service —
were *additive*, with no change to the core. Diagrams and the full data model are in
[architecture.md](docs/architecture.md).

## Project status

All five build phases are in; the engine stands on all four pillars (grounded · relevant · current ·
memory-aware), each proven by a number, and is deployable as a service. Everything below runs at
**$0** and is verified by an offline test suite (135 tests today).

| Phase | Capability | Status | Headline result |
|---|---|---|---|
| **0 — Foundation** | the spine + measurement (walking skeleton) | ✅ done | E2E $0 slice + eval baseline; per-stage timing; content-hash IDs |
| **1 — Grounding** | every claim cited and verified | ✅ done | faithfulness **1.0** ($0 stack); **0.96** on a real paraphrasing model |
| **2 — Relevance & refusal** | the right context, or abstain | ✅ done | fabrication-on-no-evidence driven **3/3 → 0 at $0**; the gate is faster *and* better when escalated |
| **3 — Freshness** | knowledge stays current | ✅ done | incremental re-index (unchanged corpus re-embeds **nothing**); staleness surfaced; a live filesystem connector |
| **4 — Memory** | per-user personalization | ✅ done | personalization lift **+0.625** ($0 floor) → **+1.000** with a paid extractor — "$0 best-positioned, paid only improves" |
| **5 — Productizing** | multi-tenant + packaged | ✅ done | service facade + dependency-free ASGI/SSE + grounding-gated caching + observability — **zero core changes** |

The thread tying every phase together: **no change ships without a number.** Each capability was
built against the eval harness — baseline first, change second, kept only if a metric improved
without regressing another. The per-phase definition of done is in [roadmap.md](docs/roadmap.md); the
*why* behind every decision is the [decision log](docs/decisions.md).

<details>
<summary><b>Detailed phase-by-phase status</b> (click to expand)</summary>

**Phase 1 — Grounding** is a real, pluggable stage (`core/grounding.py`): each claim is attributed to
the evidence *it* cited and judged against only that span, unsupported claims are **dropped or
flagged**, and the entailment check is a swappable `EntailmentJudge` port — a strict substring judge
by default (so faithfulness is never inflated), a paraphrase-tolerant token-overlap judge, and an
opt-in LLM judge. On the golden set the `$0` extractive generator scores faithfulness and citation
correctness **1.0** (it only ever quotes what it cites). Retrieval is a deliberate, *measurable* gap:
the lexical `$0` retriever reads **recall@k ≈ 0.94 / nDCG@k ≈ 0.86 / MRR ≈ 0.89** on a corpus with
distractors and paraphrase-gap questions — the numbers a real embedder must beat. An opt-in **Voyage**
embedder lifts them to **1.0 / 0.99 / 1.0**. With a real paraphrasing model the LLM judge lifts
faithfulness to ≈ **0.96**, meeting the ≥ 0.95 target.

**Phase 2 — Relevance & refusal** adds a `RelevanceGate` port between rerank and generate: on abstain
it **short-circuits the expensive generation stage** — a latency *and* cost win, not only a trust
feature. The deterministic `$0` threshold gate stays neutral on a lexical embedder (whose scores can't
separate no-answer questions from faithful paraphrases, so it never forces a false refusal); the
opt-in **LLM gate** decides on *meaning* and lifts refusal accuracy **0.824 → 1.000**. A **cascade**
runs the paid gate only in the uncertain gray zone, so confident cases stay free — faster, better, and
cheaper than the pure LLM gate. The same adapter targets a **local** model (Ollama/vLLM/LM Studio) at
**$0 per call**.

**Phase 3 — Freshness.** `ingest()` re-indexes **incrementally** by diffing the source against the
store on the content-hash ID: unchanged chunks are skipped (a re-ingest of an unchanged corpus embeds
*nothing*), changed chunks re-embedded, and `prune=True` deletes chunks the source dropped — so an
edited or removed document leaves no stale content searchable. **Staleness is surfaced**: a freshness
timestamp rides `Document → Chunk → Retrieval`, and a pure `core/freshness.py` judges age against an
explicit `now` (never the wall clock, so eval stays deterministic). A first **live connector**,
`FileSystemDocumentSource`, drives both — each file's modified time becomes its real freshness
timestamp. See [freshness.md](docs/freshness.md).

**Phase 4 — Memory** makes the engine personalize per user. A pluggable `MemoryExtractor` turns a
turn's explicit self-statements into durable memories (a `$0` rule-based floor; an LLM extractor is
the paid drop-in that only *widens* recall), and a user's *relevant* memories are injected into the
answer as labelled, overlap-gated `memory/` evidence — the **same grounded channel as the corpus**, so
the extractive `$0` model can use them yet grounding still verifies them (a remembered fact is never
invented). New facts in a known slot **supersede** the old, keeping provenance for audit. Measured: on
the `$0` stack, questions answerable only from a stated fact go from unanswerable to correct. Honest
about the floor's ceiling — the rule extractor catches every *explicit* self-statement but misses
*implicit* facts, reading **extraction recall 0.625 / lift +0.625**; the opt-in **LLM extractor**
recovers it to **1.000 / +1.000** without regressing the explicit ones. See [memory.md](docs/memory.md).

**Phase 5 — Productizing** turns the proven engine into a deployable service **without touching
`core/`**. A transport-agnostic `RaCoreService` facade drives the pipeline; over it sits a
**dependency-free ASGI app** (no web framework) exposing `/ingest`, `/answer` (**SSE** — tokens stream
for a fast first paint, then a `done` event carries citations + grounding), and `/memory`.
**Multi-tenancy** is enforced and proven *through the HTTP round-trip*. **Caching** is the latency
lever, made safe by gating on grounding rather than similarity: a `GroundingGatedCache` stores each
answer with the chunk IDs it was grounded on and — because IDs are content hashes — **auto-invalidates**
the instant that evidence is edited or removed, while an unrelated ingest leaves it valid; a hit skips
the dominant `generate` stage, and personalized answers bypass the shared cache. **Per-request
observability** falls out of the per-stage trace for free: one structured event per call to a pluggable
`Observer` (a `$0` stdlib JSON sink by default). See [productizing.md](docs/productizing.md).

</details>

## Going further: real providers (opt-in)

The `$0` default stack proves the wiring; real providers prove the engine under pressure. Each is a
drop-in behind a port — the core and the default test path never import the SDK. Put keys in a
gitignored `.env`, then opt in per run.

<details>
<summary><b>Stress-test grounding with a real model, upgrade retrieval, run the relevance gate</b> (click to expand)</summary>

**A real, paraphrasing generator** behind the `LLMProvider` port challenges grounding (the `$0`
extractive generator quotes verbatim, so it can't be unfaithful):

```bash
# .env:  ANTHROPIC_API_KEY=sk-ant-...
uv run --extra anthropic --env-file .env python -m racore.eval --llm anthropic --judge substring
uv run --extra anthropic --env-file .env python -m racore.eval --llm anthropic --judge overlap
```

The strict `substring` judge surfaces the faithfulness gap (a real model reformats rather than quoting,
so exact-match collapses to ≈ 0.0); the paraphrase-tolerant `overlap` judge recovers most of it; the
semantic **`llm` judge** credits faithful paraphrases lexical judges miss, lifting faithfulness to ≈
**0.93–0.96** (the ≥ 0.95 target). The model reveals two things: **answer correctness rises to ≈ 1.0**
(a capable reader recovers even when a distractor wins rank 1 — which is why retrieval keeps its own
rank-aware metrics), and **retrieval recall@k stays ≈ 0.94** regardless of generator or judge.

**A real semantic embedder** (Voyage) behind the `EmbeddingProvider` port fixes the lexical retriever's
paraphrase blindness — recall@k **0.94 → 1.0**, nDCG@k **0.86 → 0.99**:

```bash
# .env:  VOYAGE_API_KEY=pa-...
uv run --extra voyage --env-file .env python -m racore.eval --embedder voyage -v
```

This is **not** a lock-in — OpenAI, Cohere, or a local model (Ollama `nomic-embed-text`, BGE) drop in
the same way with zero core change.

**The relevance gate** (proactive abstention) — paid or local:

```bash
uv run --extra anthropic --env-file .env python -m racore.eval --gate llm -v
uv run --extra anthropic --env-file .env python -m racore.eval --gate cascade --gate-high 0.5
# local, $0 per call (needs Ollama running):
ollama pull qwen2.5:7b
uv run --extra openai python -m racore.eval --gate llm --gate-provider openai --model qwen2.5:7b -v
```

`--gate llm` lifts refusal accuracy **0.824 → 1.000**; `--gate cascade` escalates to the paid gate only
in the gray zone. Gate quality scales with the model — prefer a 7–8B local model or run on a semantic
embedder (cleaner evidence). The same adapter targets the hosted OpenAI API with `--gate-base-url`.

**Honest cost accounting:** cost/answer counts **every billed component** (generator + paid embedder +
LLM judge), summed and priced from each one's reported token usage. The `$0` stack reports a true `$0`;
a model with no price-table entry shows tokens with `cost n/a`, never a fake `$0`. Add `-v` for
per-case detail. The full set of flags and a verified price table are documented in
[evaluation.md](docs/evaluation.md).

</details>

The full menu of opt-in providers, flags, and the measurement methodology lives in
[evaluation.md](docs/evaluation.md) and the [decision log](docs/decisions.md).

## Repository layout

```
RaCore/                     ← this repo (the engine; Apache-2.0)
  src/racore/
    core/      types · ports · pipeline · grounding · freshness · ids   (provider-agnostic, $0)
    adapters/  embeddings · vectorstores · rerankers · chunkers · sources · memory ·
               memory_extract · judges · relevance · llm · cache        (one family per port)
    eval/      harness · metrics · datasets · pricing                   (the measurement rig)
    service/   core (facade) · asgi (HTTP/SSE) · observability · types  (the productizing layer)
  docs/        architecture · evaluation · memory · freshness · latency · productizing · roadmap · decisions
  tests/       one module per behaviour, mirroring the package
client-<name>/              ← (separate repo) a consumer's config + plugins — NOT here
```

A consumer (including any product that adopts RaCore) depends on a **released, pinned version** of this
package and contributes only **config + plugins** in its own repo. Engine changes always land here,
never in a consumer repo — see [Ownership](#ownership-and-the-clean-room-boundary).

## Documentation

Read in roughly this order:

- [architecture.md](docs/architecture.md) — ports-and-adapters design, core types, the ingest/answer pipelines, how apps connect.
- [evaluation.md](docs/evaluation.md) — the measurement strategy: the metric taxonomy, datasets, the harness, and the targets (**the differentiator**).
- [memory.md](docs/memory.md) — the per-user persistent memory subsystem.
- [freshness.md](docs/freshness.md) — keeping the index current (incremental re-index, staleness, connectors).
- [latency.md](docs/latency.md) — latency & streaming: replying promptly without awkward pauses.
- [productizing.md](docs/productizing.md) — the service surface: facade, ASGI/SSE, multi-tenancy, grounding-gated caching, observability, and the open-core line.
- [openapi.yaml](docs/openapi.yaml) — the HTTP API contract (OpenAPI 3.1): every route, schema, and SSE event shape, drift-guarded against the code.
- [roadmap.md](docs/roadmap.md) — the phased build plan and per-phase definition of done.
- [decisions.md](docs/decisions.md) — the architecture decision log (**why** each choice was made); the deepest reference.

## Development and the quality gate

```bash
uv sync                       # set up / update the environment
uv run pytest                 # tests
uv run ruff format .          # format
uv run ruff check --fix .     # lint + autofix
uv run mypy                   # type-check (strict)
uv run pre-commit run --all   # run the full gate locally
uv run pre-commit install     # activate the commit-time gate
```

The **pre-commit gate** runs the full test suite, `mypy --strict`, `ruff` format + lint, and a secret
scan before any commit lands — the same gate locally and in CI. Latency and cost are **gated** metrics,
not just reported ones: a change that regresses p95 latency or cost-per-answer fails the build exactly
like one that drops faithfulness. The house rules that keep the engine clean, simple, and legally
yours are in [`CLAUDE.md`](CLAUDE.md).

> **Note on the lean install:** `uv sync` installs the dependency-free `$0` environment, where the
> suite is green with one skip — a single contract test that pins the real Anthropic SDK's client shape
> runs only under `uv run --extra anthropic pytest` (the paid SDK is kept out of the core gate).

## Ownership and the clean-room boundary

RaCore is **independently owned** and **Apache-2.0** licensed. It is a **clean room**: it contains no
client data, no client-specific logic, and no code copied from any other product. Consumer-specific
work (the config + plugins that adapt the engine to a particular use case) lives in that consumer's own
repo, built against a **released, pinned** version of this package — never by editing the engine inside
a gig. That boundary is both good engineering and what keeps the engine reusable and sellable. The
rules that enforce it (including a pre-commit guard) are in [`CLAUDE.md`](CLAUDE.md).

## License

**Apache-2.0** — see [`LICENSE`](LICENSE). You keep an irrevocable right to reuse this engine; anyone
may adopt it. Consumer-specific customization is separate code owned by the consumer.
