# Productizing — from a proven engine to a deployable service

> Phases 1–4 proved the four pillars with numbers. Phase 5 makes the engine **deployable** without
> touching `core/`: a transport-agnostic service facade, a dependency-free HTTP surface, multi-tenancy
> exercised end-to-end, grounding-gated caching for latency, and per-request observability. This is
> the payoff of the day-one discipline (async-first, batch-first, streamable `Answer`, per-stage
> timing): productizing is **wiring + adapters**, not a re-architecture
> ([`latency.md`](latency.md) §6). The decisions are ADR-0031 – ADR-0033.

---

## 1. The shape: one core, many transports

The engine is the product; a transport is an adapter over it. So Phase 5 adds two thin layers under
`src/racore/service/`, and **nothing** under `core/`:

```
caller (HTTP client · embedded import · CLI)
        │
        ▼
racore.service.asgi    ← transport adapter: HTTP/SSE ⇄ wire JSON   (ADR-0031)
        │
        ▼
racore.service.core    ← RaCoreService: the transport-agnostic facade
        │
        ▼
racore.core.pipeline   ← the engine (untouched)
```

`RaCoreService` maps a wire request onto the core objects, drives the `Pipeline`, and returns the rich
core result. It holds **no** retrieval/grounding/memory logic — that all stays in `core`. What lives
here is product-surface concern only: request mapping, tenant scoping, the guard rails an external
caller needs, caching, and observability. Because the facade is the single choke point, every
transport inherits those behaviours identically — the HTTP app, an in-process import, or a future CLI.

## 2. The HTTP surface (dependency-free ASGI)

`racore.service.asgi` is a hand-rolled ASGI app — **no web framework**. A 4-route surface doesn't
justify a framework and its transitive dependencies, and a zero-dependency app keeps the engine's
$0/stdlib footprint (ADR-0007). Any ASGI server runs it (`uvicorn racore.service.asgi:app`-style); the
server is a *deploy* choice, never a package dependency — exactly how provider SDKs are opt-in extras.

| Route | Method | Body / params | Returns |
|---|---|---|---|
| `/health` | GET | — | `{"status": "ok"}` |
| `/ingest` | POST | `{tenant_id, documents:[{text, source}], prune?}` | `IngestReport` JSON |
| `/answer` | POST | `{text, tenant_id?, user_id?, k?}` | **SSE** stream |
| `/memory` | GET | `?tenant_id&user_id&q&k` | `{items:[…]}` |
| `/memory` | POST | `{tenant_id, user_id, text}` | `{stored:[…]}` |

`/answer` streams **Server-Sent Events**: `token` events carry the answer text for a fast
time-to-first-token, then a single `done` event delivers the citations + grounding a beat later —
the "text flows, trust badges fill in" shape from [`latency.md`](latency.md) §4, realized by the
streamable `Answer` that has existed since Phase 0. Errors map to clean 4xx with machine-readable
codes (`invalid_json`, `invalid_request`, `not_found`, `method_not_allowed`), and a `ServiceError`
code (e.g. `memory_not_configured`) passes straight through.

Because the app is just an async callable, it is **offline-testable**: the suite drives it through the
raw ASGI protocol with fake `receive`/`send` — no socket, no HTTP client — and asserts tenant
isolation and the memory loop survive the round-trip. Same injected-fake discipline as every provider
adapter.

## 3. Multi-tenancy — enforced, not bolted on

Tenancy was never a Phase-5 feature to *add*; it is the `tenant_id` the vector store and memory store
have namespaced on since Phase 0 (`InMemoryVectorStore._by_tenant`; one memory file per
`(tenant, user)`). The service's job is only to **refuse to let a request cross that boundary** and to
fail closed (a typed `ServiceError`) when a memory operation is asked of a pipeline without a store.
The tests prove a query in tenant A can never retrieve tenant B's corpus, and one user can never read
another's memory — *through the HTTP round-trip*, not just in the core.

## 4. Grounding-gated caching (the latency lever)

Caching is the biggest perceived-latency win and the most dangerous one: a cache that serves a
*similar* question's answer can be confidently wrong (negation, specificity, scope traps —
[`latency.md`](latency.md) §5a). The resolution (ADR-0020, ADR-0032) is to **gate the cache by whether
the answer still holds, not by how similar a query looks** — and the engine already owns the gate.

`GroundingGatedCache` keys an answer on `(tenant, normalized_query, filters)` — the always-safe
exact-match tier — and stores alongside it the set of chunk IDs the answer was **grounded on**. On
lookup it serves the entry only if every one of those chunks is still present in the store. Because IDs
are content hashes (ADR-0011), "still present" is byte-exact:

- **Edit or remove the grounding evidence** → its hash changes → the cached answer **auto-invalidates**
  and the question is re-answered against the current corpus.
- **Ingest an unrelated document** → none of the answer's grounding chunks change → the entry stays
  valid and the repeat ask is served fast.

A repeat ask is served running a single `cache` stage instead of the full pipeline — skipping
`generate`, the ~1.2 s stage that dominates a real-model run. Never cached: an abstain (it would
replay "I don't know" after the corpus gains the answer) or an uncited answer (nothing to gate on).
Bypassed entirely: any query with a `user_id`, because a personalized answer depends on per-user
memory the key can't capture and must never land in the shared cache.

The **semantic** tier — selecting a candidate by embedding similarity, then re-validating it by
grounding — is deliberately deferred until a semantic embedder makes its threshold measurable (the
same "don't build what you can't measure" discipline that deferred the reranker and recency ranking).
The grounding gate built here is the safety mechanism that tier will reuse.

## 5. Observability — the operator's view, for free

A deployed engine has to answer "is it working, for whom, at what cost?" — the same numbers the eval
harness gates in aggregate (latency, cache behaviour, tokens, abstain rate, grounding), but **per
request and sliced by tenant**. Observability is purely additive at the service layer (ADR-0033): the
pipeline already records per-stage timings (ADR-0010) and token usage (ADR-0018) on the result it
returns, so `RaCoreService` builds one structured `ServiceEvent` per call *from that result* and hands
it to a pluggable `Observer` port. `duration_ms` is the operation's own per-stage total, not a second
clock, so it can't drift from the harness.

The default sink, `LoggingObserver`, emits one JSON line per event through stdlib `logging` — $0, no
dependency; a statsd / OpenTelemetry / billing sink is another adapter behind the same port. A key
field, `cache_hit`, is read straight off the stage trace (a hit runs exactly one `cache` stage), so
hit-rate, p95 latency, abstain rate, and tokens-per-answer per tenant all fall out of these lines with
no extra instrumentation — the dividend of having made per-stage timing first-class on day one.

## 6. The open-core line

The architecture *is* the open-core boundary, drawn cleanly because the engine carries no transport
weight:

- **The open engine (this repo, Apache-2.0).** The full pillars — grounding, relevance, freshness,
  memory — the eval harness, the service facade, and a dependency-free HTTP app you can run under any
  ASGI server and embed in any Python process. Everything needed to build a trustworthy retrieval
  system, free, with no runtime dependency you didn't choose.
- **The managed operation (the paid part).** What a *deployment* needs and the engine deliberately
  leaves out: authentication, rate limiting, request quotas, autoscaling, a hosted control plane,
  dashboards over the observability stream, SLAs, and the tuned provider stack. None of it requires
  forking the engine — it wraps the same service the open repo ships.

That a zero-dependency ASGI app is the surface is the open-core line in microcosm: the engine is pure
and embeddable; the operation around it is the product you pay someone to run.

## 7. Honest status — built vs. deferred

- **Built (gated, $0/offline):** the service facade; the dependency-free ASGI surface with SSE
  streaming; multi-tenancy proven through HTTP; grounding-gated exact-match caching with
  content-hash invalidation; per-request observability with a stdlib JSON sink.
- **Deferred, with a reason:** the semantic answer-cache tier (needs a semantic embedder to tune its
  threshold); auth / rate-limiting / quotas (an edge concern, kept out of the engine on purpose); a
  runnable server entrypoint (BYO ASGI server, by design); cost (USD) in the event stream (the price
  table is the billing layer's, kept out of the service); trace-header propagation into `request_id`;
  and HTTP-level fields (status, path) on the event.

None of the built work touched `src/racore/core/` — which is the whole point.
