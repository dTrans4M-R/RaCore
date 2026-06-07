# Architecture

> Provider-agnostic core, pluggable everything-that-varies, measurement built in from day one.

---

## 1. Principles

1. **Ports & adapters (hexagonal).** The core orchestration logic knows nothing about Voyage, OpenAI,
   pgvector, or PDFs. Everything that varies is a **port** (an interface); a concrete provider is an
   **adapter**. Customization = swapping/configuring adapters, never forking the core.
2. **Eval-first.** The evaluation module is wired on day one, not bolted on. If you can't measure a
   change, you don't ship it. → [`evaluation.md`](evaluation.md)
3. **Config-as-customization.** A typed (pydantic) config schema is the client-facing surface. A client
   need that can't be expressed as config or a plugin is a signal to add a new **port**, not a fork.
4. **Grounding and memory are pipeline stages, not add-ons.** They sit in the main `answer()` path.

## 2. Component view

```mermaid
flowchart TB
    subgraph core["core/ (provider-agnostic)"]
        T["types.py — domain objects"]
        P["ports.py — interfaces"]
        PL["pipeline.py — ingest() / answer()"]
        G["grounding.py — citations + faithfulness"]
    end
    subgraph adapters["adapters/ (pluggable)"]
        EM["embeddings"]:::a
        VS["vectorstores"]:::a
        RR["rerankers"]:::a
        CH["chunkers"]:::a
        SR["sources / connectors"]:::a
        ME["memory stores"]:::a
        LL["llm providers"]:::a
    end
    EV["eval/ — harness + metrics"]
    API["api/ — HTTP service + in-process SDK"]
    WEB["demo-web/ — Next.js showcase"]

    PL --> P
    P -.implemented by.-> adapters
    EV -.measures.-> PL
    API --> PL
    WEB --> API
    classDef a fill:#eef,stroke:#88a;
```

## 3. Core domain types (`core/types.py`)

`Document` · `Chunk` · `EmbeddedChunk` · `Query` · `Retrieval` (chunk + score + source span) ·
`Evidence` (verbatim quote + offset + doc ref) · `GroundedContext` (assembled, cited) · `Citation` ·
`Answer` (text + citations + grounding_report) · `GroundingReport` (per-claim support state) ·
`MemoryItem` (→ [`memory.md`](memory.md)) · `EvalResult`.

## 4. Ports — the plugin boundaries (`core/ports.py`)

`Protocol`s (structural typing — adapters need no inheritance):

| Port | Responsibility | Example adapters |
|---|---|---|
| `EmbeddingProvider` | text → vectors (`document` vs `query` input types) | voyage · openai · local-bge · mock |
| `VectorStore` | upsert · filtered similarity search · content-hash diff (`chunk_ids`/`delete`) for incremental re-index, per tenant | pgvector · inmemory |
| `Reranker` | re-order candidates by query relevance | voyage-rerank · cross-encoder · noop |
| `Chunker` | document → structure-aware chunks | structural (page/section) · fixed-window |
| `DocumentSource` | fetch + extract raw documents, with a freshness timestamp | filesystem · inmemory · pdf · sec-edgar · web · s3 |
| `MemoryStore` | per-user read/write/compaction | pg-memory · file-memory |
| `MemoryExtractor` | turn → durable per-user memories (the write policy) | rule-based ($0) · llm |
| `LLMProvider` | grounded generation (streaming) | anthropic · openai |
| `EntailmentJudge` | per-claim: does cited evidence support it? | substring · token-overlap · llm-judge |
| `RelevanceGate` | answer-vs-abstain on the retrieved evidence (proactive refusal) | threshold · anthropic · openai (local/hosted) · cascade |
| `Evaluator` | score a run against a dataset | retrieval · grounding · answer |

## 5. The two pipelines (`core/pipeline.py`)

**`ingest(source, tenant)`** — `DocumentSource.fetch → extract → Chunker.chunk → diff → EmbeddingProvider.embed
→ VectorStore.upsert [→ prune]`, recording metadata + a freshness timestamp. The diff is on the content-hash
ID (ADR-0011): only changed chunks are embedded, and `prune=True` deletes chunks the source dropped, so
re-index is incremental and leaves no stale content (ADR-0023). That freshness timestamp is carried
`Document → Chunk → Retrieval`, so the age of an answer's evidence is visible on the `Answer`; staleness is
judged against an explicit `now` (`core/freshness.py`), never the wall clock, keeping eval deterministic (ADR-0024).

**`answer(query, tenant, user)`** — grounding, relevance, and memory are first-class. When a query
carries a `user_id`, `memory.read` injects the user's *relevant* memories (gated by overlap) as
labelled `memory/<turn>` evidence in the same grounded channel as the corpus — so a personal fact is
used by the model yet still verified by grounding (never invented), and personalization works at $0.
`memory.write` then learns durable facts from the turn via a `MemoryExtractor`, on every exit path
including abstain (ADR-0026):

```mermaid
flowchart LR
    Q["query"] --> M["memory.read<br/>(personalize)"]
    M --> U["understand<br/>rewrite / expand"]
    U --> H["hybrid retrieve<br/>(dense + sparse)"]
    H --> R["rerank<br/>(relevance)"]
    R --> G["relevance gate<br/>(answer? / abstain)"]
    G --> A["grounding.assemble<br/>(attach evidence + citations)"]
    A --> L["llm.generate<br/>(answer w/ inline [n], stream)"]
    L --> V["grounding.verify<br/>(faithfulness → drop/flag unsupported)"]
    V --> W["memory.write<br/>(learn)"]
    W --> O(["Answer + citations + grounding_report"])
    R -.no usable context.-> X["abstain — 'I don't know'"]
    G -.evidence too weak<br/>(short-circuit generate).-> X
```

## 6. How applications connect

One core, exposed two ways:

- **In-process SDK** (`api/sdk.py`) — a Python object; lowest latency. **This is how Lorinks (or any
  Python app) embeds the engine.**
- **HTTP / gRPC service** (`api/service.py`) — `POST /ingest`, `POST /answer` (SSE streaming),
  `GET/POST /memory`. For non-Python apps or process isolation.

**Multi-tenancy:** a `tenant_id` at the boundary namespaces the vector store and the memory store, so
one deployment serves many clients without co-mingling data. Per-user memory is keyed `(tenant, user)`.

## 7. Folder layout

```
src/grounded_rag/
  core/      types.py · ports.py · pipeline.py · grounding.py
  adapters/  embeddings/ vectorstores/ rerankers/ chunkers/ sources/ memory/ llm/
  config/    schema.py            # pydantic — the customization surface
  eval/      harness.py · metrics.py · datasets/
  api/       service.py · sdk.py
tests/
demo-web/    # Next.js: chat + click-to-source citations + memory panel
examples/contracts/   # flagship demo config + seed filings
```

## 8. Tech choices (see [`decisions.md`](decisions.md) for the why)

Python 3.12 core · pydantic config · FastAPI service · **pgvector** (prod) / **in-memory** (dev) ·
a **local embedding model** as the default so the public demo costs **$0** (Voyage/OpenAI are drop-in
upgrades) · Next.js + TypeScript for `demo-web/`.
