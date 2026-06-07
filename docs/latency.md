# Latency & streaming — replying promptly without awkward pauses

> Modern RAG is judged partly on **feel**: the answer must start almost immediately and flow, not
> arrive after a buffered pause. This note records where RaCore spends time, why "prompt-feel" is an
> **additive** capability rather than a rebuild, and the concrete path to it. It is forward-looking in
> parts — the foundation is built; the streaming/caching adapters are not yet wired. Productionizing
> this is **Phase 5** (see [`roadmap.md`](roadmap.md)); the groundwork is load-bearing today.

---

## 1. Principle: perceived latency is *time-to-first-token*, not total time

A user tolerates a 2-second total answer if it **starts streaming in ~300–500 ms** and flows as it is
generated. The "awkward pause" is a blank screen during a *buffered* response — the system finishing
the whole answer before showing anything. So the target is:

- **Time-to-first-token (TTFT)** + a steady token stream — the *perceived* speed, and
- p50/p95 **total** latency — already a **gated** metric (ADR-0010), so regressions fail the build.

Optimizing for TTFT is mostly orthogonal to total time: you can have a 2 s total that *feels* instant.

## 2. Where the time actually goes (measured)

Per-stage timing is built into the pipeline (ADR-0010), so this is observed, not guessed. A real
full-stack run (Voyage embedder + Claude generator + LLM entailment judge), p50 ≈ 2.2 s:

| Stage | ~Time | Whose latency |
|---|---|---|
| understand | ~0 ms | RaCore |
| **embed** (query, Voyage API) | ~350 ms | provider network |
| retrieve + rerank + assemble | ~10 ms | **RaCore** |
| **generate** (Claude) | ~1.2 s | provider |
| **verify** (LLM judge) | ~750 ms | provider |

The `$0` deterministic stack runs the whole thing in **~4 ms**. So **RaCore's own overhead is ~10 ms**;
the ~2 s is almost entirely **provider round-trips**. The lever is therefore *not* "make the engine
faster" — it is "**stop making the user wait** for the providers."

## 3. Why prompt-feel is additive, not a rebuild (the ADR-0009 payoff)

The costly, irreversible scaling rework in a RAG engine is not infrastructure — it is the **method
signatures**. Converting buffered→streaming or sync→async *after* adapters exist forces a rewrite of
every adapter and caller. RaCore made those decisions up front (ADR-0009), specifically so streaming
would be additive:

- **`answer()` returns a streamable `Answer` type** with `async def stream()`, from day one — *before*
  streaming was implemented. Today `stream()` replays the already-materialised text token-by-token (a
  placeholder), but the **public interface is already streaming-shaped**.
- **Async-first** — every I/O port is `async def`, so non-blocking streaming and concurrency are
  native, not a retrofit.
- **Latency is gated** (ADR-0010) — once prompt, it *stays* prompt, because a p95 regression fails the
  gate like a faithfulness drop.

Net: wiring real streaming is a **new adapter path**, with **no churn to existing signatures or the
core** — the whole point of the discipline.

## 4. The grounding-vs-streaming tension (and the resolution)

This is the subtle part, and it exists *because* grounding is a first-class stage. The answer pipeline
is `generate → verify`, and verification needs the **whole** answer to attribute each claim to its
cited evidence. Streaming raw generation shows the user text *before* it has been grounding-checked.

The resolution (recorded as ADR-0020):

- **Grounding decorates a concurrent stream.** Stream the answer text immediately for a fast TTFT, run
  `verify` *concurrently*, and let **citations / grounding badges populate a beat later** — exactly how
  a trustworthy product feels (the answer flows; citation chips fill in). Because grounding is a
  separate stage that annotates the `Answer` (it does not produce the text), this composes cleanly.
- **The judge cost is opt-in and off the hot path.** The deterministic substring/overlap judges are
  ~0.1 ms — free in-path. Only the *optional* LLM judge (~750 ms) is expensive, and in production it
  runs **async / on-demand** (e.g. verify after the stream, or when the user clicks "check grounding"),
  never blocking the first token.
- **`drop_unsupported` is the documented exception.** You cannot stream text you might later delete, so
  drop-mode buffers. Flag-mode (the default) streams fine.

## 5. The levers (all architecture-ready, all additive)

1. **Streaming LLM adapter.** Anthropic/OpenAI stream tokens over SSE; a streaming implementation
   behind `LLMProvider` feeds `Answer.stream()` real deltas → **TTFT ~hundreds of ms**. The single
   biggest perceived-latency win. The batch `generate()` stays for eval/batch use; streaming is a new
   additive method, not a change to it.
2. **Kill the embedder round-trip.** A **local embedder** (Ollama / BGE behind the same
   `EmbeddingProvider` port) or a query-embedding **cache** removes the ~350 ms network hop — a drop-in,
   exactly as Voyage was.
3. **Caching** — a real win, but a correctness *hazard* if done naively. See §5a; it must be gated by
   grounding, not by similarity.
4. **Async / on-demand grounding.** Keep the LLM judge out of the blocking path (see §4).

## 5a. Caching, safely — why "similar" is not "same"

Caching is the most tempting latency lever and the most dangerous one. The hazard: a **semantic** cache
keyed on embedding similarity will happily serve a previous answer for a *different* question, because
**users don't phrase things like machines** — two questions can be embedding-near yet have *different
correct answers*. The classic traps, where similarity is high but the answer must differ:

- **Negation:** "Is X covered?" vs "Is X **not** covered?" — near-identical embeddings, *opposite*
  answers.
- **Specificity:** "What's the refund policy?" vs "...for **digital** goods?" — the cached general
  answer is wrong for the specific ask.
- **Entities / scope:** "reset my password" vs "reset my **admin** password".

In RaCore's flagship domain — contracts and financial filings (ADR-0003) — a wrong cache hit isn't a
slow answer, it's a **confidently wrong** one, which is the failure mode the whole engine exists to
prevent. So the instinct to be wary of semantic caching is **correct**.

The resolution is to **gate the cache by correctness, not by similarity** — and RaCore already owns the
gate. Tiered, from zero-risk to opt-in:

1. **Exact / normalized-match cache (always safe).** Key on the normalized query (lowercased,
   whitespace-collapsed) plus tenant + filters. Identical re-asks — refreshes, dashboards, repeated
   questions — hit with **zero** correctness risk and capture a large share of real repeat traffic.
2. **Cache the *retrieval*, not the answer (low risk).** Reuse the cached query embedding and/or
   retrieved chunks (the ~350 ms embed + retrieve) but still **regenerate** the answer. Cuts latency;
   the answer is always freshly grounded.
3. **Semantic answer cache (opt-in, grounding-gated).** A high similarity threshold is *necessary but
   not sufficient*. On a candidate hit, **retrieve for the new query and re-run the grounding /
   faithfulness check on the cached answer against the new evidence**; serve it only if it still holds,
   else regenerate. This converts "trust the embedding" into "verify the answer is still supported" —
   and the negation/specificity traps fail that check, so they regenerate. **Semantic caching is only
   as safe as your ability to detect when the cached answer no longer holds — which is exactly what
   RaCore measures.** Off by default for high-stakes corpora.

This is a **`Cache` port with a pluggable policy** (ports-and-adapters): the *stakes* choose the
aggressiveness via config, never a fork — conservative (exact-match only) for contracts/filings,
more aggressive where a wrong answer is cheap.

## 6. Built vs. future (honest status)

- **Built today:** the streamable `Answer` type, the async core, per-stage timing, and **gated** p50/p95
  latency + cost/answer.
- **Not yet wired (additive, no core change):** a real streaming LLM adapter, a local/cached embedder,
  async/on-demand grounding, and a cache layer.

None of the future work requires touching `src/racore/core/` — which is the dividend of the ADR-0009 /
ADR-0010 discipline. "Prompt replies without awkward pauses" is a **wiring + adapter task**, not a
re-architecture.

## 7. The discipline that keeps it prompt

Latency is a **gated** metric, not a reported one (ADR-0010): a PR that regresses p95 fails the build
exactly like one that drops faithfulness. So promptness, once achieved, is protected by the same
eval-first guardrail as quality — see [`evaluation.md`](evaluation.md).
