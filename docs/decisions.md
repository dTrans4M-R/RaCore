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

### ADR-0014 — A harder eval corpus, a relevance-set golden schema, and recall@k
**Context:** the Phase 0 golden corpus was 7 unambiguous docs; with `k=5` over 7 docs the
retriever returned almost everything, so hit@k was a saturated 1.0 and a reranker or hybrid
retriever would have **no measurable gap to close**. Retrieval depth (real embeddings, hybrid
dense+sparse, reranking) is the remaining Phase 1 work, and "no change ships without a number"
needs a corpus where ranking actually matters. **Decision:** (a) grow the golden corpus to ~28
synthetic, public-domain Solar-System docs with deliberate **distractors** (docs that share a
question's words without answering it) and **paraphrase-gap** questions (wording that diverges
from the answer doc); (b) generalize the golden schema from a single `expected_source` to a
`relevant_sources` **set**, so multi-source questions are expressible and graded judgments
exist; (c) replace the rank-blind hit@k metric with **recall@k** over that set, keeping
rank-weighted **nDCG@k / MRR** for the slice that lands the reranker; (d) surface per-case
retrieval detail (recall, retrieved-vs-relevant) in the harness `-v` output. **Consequences:**
the $0 baseline now reads **recall@k ≈ 0.94** and **answer.correctness ≈ 0.86** — a real,
attributable gap (the lexical retriever buries some relevant docs below `k`, and a distractor
often wins rank 1, so the extractive answer quotes the wrong doc), while grounding stays 1.0
(the extractive generator grounds in whatever it cites). Those two numbers are the targets a
real embedding adapter, hybrid retrieval, and a reranker must beat. Relevance judgments are
content-defensible — a doc is marked relevant only if it states the fact. Behaviour tests that
need perfect retrieval (refusal recording) use a small focused fixture, not the hard set, so
each test exercises one thing.

### ADR-0015 — Rank-aware retrieval metrics (nDCG@k, MRR) added *before* the reranker
**Context:** ADR-0014 deferred nDCG@k / MRR to "the slice that lands the reranker." A real-model
run (Claude Haiku, 2026-06-07) showed that ordering was wrong. Two things make the existing
metrics blind to rank: **answer.correctness rose from 0.857 (extractive mock) to 1.000 (Claude)**
because a capable generator reads *all* `k` evidences and answers correctly even when a distractor
wins rank 1 — so end-to-end correctness **launders** retrieval rank quality; and **recall@k is
position-blind** by construction. A reranker only reorders within top-k, so it moves *neither*
recall@k nor LLM answer.correctness — with only those metrics, a reranker's benefit is unmeasurable.
**Decision:** add **nDCG@k** (relevant hit discounted by `log2(rank+1)`, normalised to the ideal
ranking) and **MRR** (`1/rank` of the first relevant doc) as gated evaluators **now**, before
building the retriever — measurement before optimization (ADR-0005). Binary relevance over
`relevant_sources`. **Consequences:** the $0 baseline now reads **recall@k 0.940, nDCG@k 0.863,
MRR 0.889** — the rank-aware pair sits *below* recall, surfacing the buried-but-present docs recall
hides. These are the gated numbers a reranker (and hybrid retrieval) must lift, and the reason the
retriever is judged directly rather than through a forgiving generator. Graded (multi-level)
relevance can extend nDCG later with no interface change.

### ADR-0016 — Real embedding adapter (Voyage), opt-in behind the `EmbeddingProvider` port
**Context:** the $0 `MockEmbeddingProvider` is purely lexical (shared-vocabulary cosine), so it buries
answer documents that paraphrase the question — "closest to our **star**" never matches "closest to
the **Sun**". That is the measurable gap (recall@k 0.940, nDCG@k 0.863) the retrieval-depth work
exists to close, and ADR-0007 requires the $0 default to remain. A real *semantic* embedder is the
first lever. **Decision:** add `VoyageEmbeddingProvider` as the first opt-in real embedder behind the
`EmbeddingProvider` port — Voyage AI (Anthropic's recommended embeddings), pinned model `voyage-3.5`,
shipped as an **optional extra** (`racore[voyage]`, `voyageai>=0.4`) with a lazy SDK import, a narrow
async client `Protocol` (so it is fully typed and offline-testable with a fake client), and
query/document `input_type` asymmetry. The mock stays the default; selection is `--embedder voyage`,
mirroring `--llm anthropic`. **Consequences:** provider-agnostic, *not* a lock-in (ADR-0001) — Voyage
is one adapter; OpenAI, Cohere, or a **local** embedder (Ollama `nomic-embed-text`, BGE) are drop-ins
behind the same port with zero core change, exactly as `AnthropicLLM` is one of many LLM adapters. The
core stays dependency-free; only `racore[voyage]` pulls the SDK (and its numpy/pillow deps). The
nDCG@k / MRR lift is measured by an opt-in `--embedder voyage` run (needs `VOYAGE_API_KEY`). Embedding
*cost* accounting is deferred — the port returns vectors, not usage — and noted for a later slice.

### ADR-0017 — LLM entailment judge (Claude) behind the `EntailmentJudge` port
**Context:** the deterministic ``$0`` judges are *lexical*, so on a paraphrasing model they
**understate the headline faithfulness metric**. Measured on a real Claude run: correct, grounded
answers score **0.0** — p1 "Mercury is the **world** that sits closest to our **star**" and p5
"Europa might hold an **underground sea**" are entailed by the evidence ("closest to the **Sun**",
"hide a salty **ocean**") but share too few words for substring or token-overlap to credit them.
ADR-0012 left the `EntailmentJudge` port pluggable for exactly this. **Decision:** add
`AnthropicEntailmentJudge` as the first *semantic* judge behind the port — a per-claim
SUPPORTED/UNSUPPORTED verdict from Claude (`max_tokens=8`), batched, with an **uncited-claim
short-circuit** (no call, no cost, matching the deterministic judges), reusing the Anthropic client
plumbing; optional extra, lazy SDK, offline-testable; selected via `--judge llm`. The strict
substring judge stays the **default** (the un-foolable floor). **Consequences:** faithfulness can now
be judged on *meaning*, so the lexical understatement (p1/p5 = 0.0) should resolve to supported —
**validated against the golden set** on a real `--judge llm` run. Provider-agnostic: an OpenAI/local
judge is a drop-in behind the same port (not a lock-in). LLM-as-judge brings its own validity
question (does it agree with ground truth?) — which is why the deterministic judges remain as the
honest floor and the golden set is the check. **Cost caveat:** the judge makes per-claim LLM calls
that, like Voyage embeddings, are **not yet priced** into cost/answer — the `EntailmentJudge` /
`EmbeddingProvider` ports return verdicts / vectors, not usage; only the generator's tokens are
priced today. Surfacing and pricing component usage is the next measurement-integrity slice.

### ADR-0018 — Honest cost accounting across all billed components (optional `UsageReporter`)
**Context:** cost/answer priced only the **generator** (`LLMResponse.usage`). Once a paid embedder
(Voyage) and judge (LLM) landed, a paid run **understated** cost and a `$0`-LLM + Voyage run reported a
misleading **$0.000000** — violating the "never a fake $0 on a paid run" principle, which matters
because honest measurement *is* the moat. The I/O ports return vectors/verdicts, not usage, and
ADR-0009 froze those batch-first signatures. **Decision:** add an **optional** `UsageReporter` port
(`drain_usage() -> list[TokenUsage]`, `runtime_checkable`), *separate* from the I/O ports so the frozen
signatures don't change. Billed adapters (`VoyageEmbeddingProvider`, `AnthropicEntailmentJudge`)
accumulate per-call usage and expose `drain_usage()`; the free `$0` adapters don't implement it. The
pipeline drains its embedder (per query) and judge after each answer, folds them with the generator's
usage into **`Answer.usages`** (replacing the single `Answer.usage`); `ingest()` drains the embedder so
per-answer accounting doesn't absorb the corpus's one-time embed cost. The harness **sums and prices
every component per answer**; if *any* billed component's model is unpriced, the whole figure is `None`
(n/a), never a partial $0. Voyage embedding prices were added to the table (input-only). **Consequences:**
cost/answer is now honest across generator + embedder + judge; the `$0` default stays a true $0 (free
adapters report nothing). Adding a new paid component = implement `UsageReporter` + a price-table row,
no core change. A provider with no price entry shows tokens with cost n/a, not a fake $0. The drain
pattern is stateful but contained (the harness answers sequentially; appends within one call are
asyncio-safe). Ingest-time embedding cost is currently discarded (cost/answer is per-answer); tracking
it is a later addition.

### ADR-0019 — Fix the faithfulness residual at the generation prompt, not the parser/judge
**Context:** with the LLM judge, faithfulness landed at **0.93–0.96**, wobbling around the ≥0.95 target.
The sole residual (Finding G) is the model appending a **verbatim restatement of the evidence as an
uncited second sentence** (`"claim. [1] restatement."`): the inter-sentence marker is the *first*
sentence's trailing citation, so the restatement is genuinely uncited and correctly scored
unsupported. Two tempting "fixes" were **rejected**: (a) also attaching the leading marker to the
*following* sentence — but `test_grounding`'s swap case (`"Mercury… [2] Jupiter… [1]"`) proves that
wrong: it would hand the Jupiter sentence the `[2]` it never intended and make a *swapped* citation
look correct; (b) crediting an uncited sentence because it appears verbatim in the evidence — the
"pool every quote" anti-pattern ADR-0012 deliberately removed. The parser is right; the model simply
left a claim uncited. **Decision:** fix it **upstream, at the generation prompt** — instruct the model
to be concise, state each fact once, not restate the evidence, and **end every sentence with its
supporting `[n]` marker**. Keep `_parse_claims` strict and the deterministic substring judge as the
floor. **Consequences:** a verbose answer no longer bleeds faithfulness through an uncited restatement,
so the headline metric should sit **stably ≥0.95** (validated on a real `--judge llm` run). The fix
does **not** relax the metric — an uncited claim is still unsupported (`test_grounding`'s
`test_uncited_claim_is_unsupported…` guards that); it improves the *generator's* citation discipline.
Prompt effects are non-deterministic, so validation is **empirical** (the eval harness on a real
model), not a unit test; the deterministic gate is unaffected because the `$0`/mock generators ignore
the system prompt.

### ADR-0020 — Streaming is additive; grounding decorates a concurrent stream
**Context:** modern RAG is judged on responsiveness — the answer must start promptly and flow, not
arrive after a buffered pause. Measured latency (~2.2 s full stack) is **~99 % provider round-trips**
(generate ~1.2 s, LLM judge ~0.75 s, embedder ~0.35 s); RaCore's own overhead is ~10 ms. ADR-0009
already froze a streamable `Answer` type + async ports so streaming is *additive*; the open question is
how streaming coexists with grounding, which verifies the **whole** answer (`generate → verify`).
**Decision:** (a) real token streaming is a **new additive path** — a streaming LLM adapter behind
`LLMProvider` feeds `Answer.stream()` real deltas; the batch `generate()` stays for eval/batch, with no
change to existing signatures or the core. (b) **Grounding decorates the stream:** stream the text
immediately for a fast time-to-first-token, run `verify` concurrently, and let citations/grounding
resolve a beat later (flag mode). Deterministic judges are ~0.1 ms (free in-path); the **opt-in** LLM
judge (~0.75 s) runs async / on-demand, off the blocking path. (c) `drop_unsupported` buffers (you
cannot stream text you may delete) — the documented exception. (d) Latency stays **gated** (ADR-0010).
**Consequences:** time-to-first-token drops to ~hundreds of ms — the perceived-latency win — with no
core rebuild; grounding stays first-class without blocking the stream; the engine stays
provider-agnostic (a local/cached embedder and the streaming adapter are drop-ins). Productionizing
this is **Phase 5**; the foundation is load-bearing now. Full analysis in [`latency.md`](latency.md).

### ADR-0021 — Relevance gate: proactive abstention as a pipeline decision, built as a cascade
**Context:** Phase 2 is the *proactive* refusal decision — knowing when **not** to answer. Until now the
pipeline only abstained on *empty* retrieval and *recorded* a refusal the model volunteered (ADR-0013);
the `$0` stack still false-answers all three negative controls (refusal accuracy **0.824**). The obvious
design — abstain when the top retrieval score is below a threshold — was tested first, and **measurement
killed the premise**: on the lexical `$0` embedder the negative controls *interleave* with the
legitimate paraphrase rows (top-1 `n1`=0.113, `n2`=0.289, `n3`=0.378 vs answerable `p2`=0.135,
`p1`/`p5`=0.218, `m1`=0.378). To catch all three negatives needs a floor > 0.378, which falsely refuses
five answerable rows; to avoid any false refusal needs a floor < 0.135, which catches only one negative.
**No single threshold separates them**, because a lexical embedder's cosine reflects word overlap, not
relevance — "surface temperature on Pluto" (no answer) lexically matches the Pluto/surface docs and
scores *high*. This is the relevance analogue of the substring judge scoring 0.0 on a real model
(ADR-0017): the cheap deterministic signal has a known blind spot, surfaced by measurement, not hidden.

**Decision:** add a `RelevanceGate` port — `should_answer(checks) -> list[bool]`, async + batch-first
(ADR-0009) — slotted **between rerank and generate**, and build it as a **cost-escalating cascade**, not
one gate:
- **Tier 1 — `ThresholdRelevanceGate`** (stdlib, `$0`, this slice): decides from two cheap signals on
  the reranked scores — an absolute **floor** (`min_score`) and a **margin** over the runner-up
  (`min_margin`, so a *flat* distribution the retriever couldn't discriminate doesn't earn a confident
  answer). Thresholds are **per-embedder** and default to **neutral** (`0.0/0.0` → abstain only on empty
  retrieval), so wiring the gate in **never introduces a false refusal** on an uncalibrated stack; a
  semantic embedder (Voyage / local) is where a score threshold separates cleanly, calibrated against the
  eval harness.
- **Tier 2 — LLM gate** (paid, next slice): a semantic "is the answer in this evidence?" judgement,
  escalated to **only for the uncertain gray-zone** cases tier 1 can't decide — the embedder-independent
  robustness, with the paid-call rate (hence added latency) bounded to the hard minority.
- A local LLM (Ollama, OpenAI-compatible) drops in behind the same port as a sibling of the paid gate;
  an in-process cross-encoder is deferred (heaviest dependency in the repo, and Finding D shows no
  measurable retrieval gap for it yet).

**Latency is a first-class reason for the design, not an afterthought:** the gate sits *before* generate
so an abstain **short-circuits the dominant ~1.5 s generation stage** — a latency *and* cost win on
exactly the queries where generating was pointless (and harmful). The `relevance` stage is timed like
every other (ADR-0010), so its cost is **gated**, and the cascade keeps the expensive tier off the
common path. (Speculative `gate ∥ generate` for the answerable hot path is a Phase-5 option, ADR-0020.)

**Consequences:** the seam, the `$0` deterministic floor, and the latency short-circuit land now;
because the default `demo_pipeline()` ships **no** gate, the mock baseline is **unchanged** (refusal
stays an honest 0.824 — the lexical stack genuinely can't gate on score, and we don't fake it). The gate
is scored by the existing `RefusalEvaluator` (both error directions), and any calibrated threshold is
kept only if **refusal accuracy rises without regressing answer correctness** — over-abstention is the
risk watched. The robust closing of the 0.824 gap comes with the tier-2 LLM gate, validated empirically
on a real run, mirroring how the LLM entailment judge followed the deterministic judges.
**Validated (2026-06-07, real Claude gate, mock embedder + mock generator to isolate the gate):**
`--gate llm` took refusal accuracy **0.824 → 1.000** (false-answer-on-no-evidence **1.0 → 0.0**) with
answer correctness **held at 0.857** and faithfulness/citation unchanged — it abstains on all three
negative controls and lets every answerable row through, on the *lexical* embedder where no score
threshold can, confirming the gate is embedder-independent. The cost is honest (~$0.00022/answer, the
gate's tokens priced via `UsageReporter`) and the `relevance` stage is ~1.1 s/query — every query pays,
which is exactly the cascade's reason to exist: `--gate cascade --gate-high 0.5` held refusal at
**1.000** while cutting gate cost to **~$0.00016/answer** (138 vs 195 tokens/answer) by answering the
confident high-score rows for free. On a semantic embedder the free bands widen, realizing the latency
win; the residual `p2`/`p5` wrong answers are the mock extractive generator quoting a distractor
(Finding A), not the gate, which correctly let them through. **Cascade-safety finding (folded into the
code):** the cascade's *free-answer* high band is **opt-in and off by default** (`high=None`). A high
retrieval score is **not** proof the answer is present — a Voyage run with `--gate-high 0.5` let "how
many rings does Neptune have?" through on the high-scoring "Saturn's ring system" doc (a semantic false
positive), re-introducing a false answer that the full-stack run caught only via the generator's *own*
refusal (ADR-0013). The free-*abstain* low band stays safe (worst case a false refusal, never a
fabrication); the free-answer band must be calibrated on the harness before opting in, so the default
never trades away the abstention guarantee for a cost saving.

### ADR-0022 — Local relevance gate over the OpenAI-compatible protocol (paid ↔ local)
**Context:** ADR-0021 closed the refusal gap with a *paid* semantic gate (Claude) and named the obvious
follow-on: "a local LLM (Ollama, OpenAI-compatible) drops in behind the same port as a sibling of the
paid gate." That sibling matters for the product's core promise — *the `$0` stack must be best-positioned
so paid only improves quality.* Between the `$0` `ThresholdRelevanceGate` (no model, but blind on a
lexical embedder) and the frontier paid gate there was no **free, semantic** tier. A self-hosted small
model is exactly that tier: embedder-independent abstention at **zero per-call spend**.

**Decision:** add `OpenAIRelevanceGate` behind the same `RelevanceGate` port, speaking the
**chat-completions** protocol. That protocol is the lingua franca of local runtimes — Ollama, vLLM, LM
Studio, llama.cpp all expose it, and so does the hosted OpenAI API — so **one** adapter reaches a local
($0) model or a hosted one, selected only by `base_url` (and, for hosted, `api_key`). The SDK is an
optional extra (`racore[openai]`, newest stable `openai>=2.41`), imported lazily, with an injected client
`Protocol` so the adapter is fully typed and offline-testable; the core install stays dependency-free
(ADR-0007). Defaults target a local Ollama endpoint (`http://localhost:11434/v1`, placeholder key) so the
$0 path works once a model is pulled.

Because a *second* concrete LLM gate now exists, the verdict-deciding pieces — the strict one-word system
prompt, the evidence rendering, the ANSWER/ABSTAIN parse — are lifted into a provider-neutral
`adapters/_relevance_llm.py` and shared **verbatim** by both gates. This is the abstraction the house
rule sanctions ("add it when the second case exists, not in anticipation"): two gates that disagreed on
the same evidence because their prompts drifted would be a silent correctness bug. Each adapter keeps only
its own client call and token accounting, which genuinely *are* provider-specific. CLI:
`--gate-provider {anthropic,openai}` + `--gate-base-url`; the default run is unchanged (`anthropic`).

**Consequences:** the "paid ↔ local" reach is delivered with no heavy dependency and no new lock-in —
swapping the gate's brain is a config change, not a fork. Cost stays honest via `UsageReporter`
(ADR-0018): a self-hosted call reports zero tokens → prices to `$0`, which is *true* (no per-token
charge), while a hosted OpenAI call is priced like any other. Verified offline with a fake client (call
shape, verdict parse, empty-retrieval skip, cascade composition + usage forwarding) **and** against the
real `AsyncOpenAI` 2.41 constructor (the contract test, offline — it builds the client, makes no request);
mypy `--strict` passes with and without the extra. Live validation against a running local model is the
operator's manual step (`--gate llm --gate-provider openai`), the same opt-in pattern as the paid gate —
no server runs in CI.

**Live local-vs-hosted runs (2026-06-07, mock stack) — $0 buys the gate's *safety*; its *helpfulness
and speed* scale with the model.** Four gates over the same lexical `$0` stack:

| gate | refusal.acc | answer.corr | fabrication-on-no-evidence | false refusals | gate latency |
|---|---|---|---|---|---|
| none (baseline) | 0.824 | 0.857 | **3 / 3 negatives** | 0 | 0 ms |
| Ollama `llama3.2` (3B) | 0.647 | 0.429 | **0** | 6 | ~4 s |
| Ollama `qwen2.5:7b` | 0.824 | 0.643 | **0** | 3 | **~10 s** |
| `gpt-4o-mini` / Claude (hosted) | **1.000** | 0.857 | **0** | **0** | ~1 s |

The decisive column is **fabrication-on-no-evidence: every gate, local or hosted, drives it 3/3 → 0** —
the cardinal grounding property is bought at `$0` by even a 3B model. What scales with model strength is
the *secondary* axis: false refusals fall **6 → 3 → 0** (qwen's three holdouts `q4`/`q6`/`p1` are
paraphrase / noisy-evidence cases a mid-size model won't dig the answer out of), and only a frontier
model reaches the no-false-refusal ideal. Two measurement nuances the numbers force into the open:
(1) `refusal.accuracy` weights a false refusal and a fabrication **equally, but the product does not** —
so qwen's `0.824` (all negatives caught, three *safe* over-refusals) is *qualitatively better* than the
baseline's identical `0.824` (three *fabrications*); the symmetric metric understates the gate's value,
and an asymmetric refusal score is a future eval refinement. (2) the 7B gate ran at **~10 s/query** on
commodity CPU — safe but too slow for an interactive bot, the project's stated latency concern. The two
`BAD` answer rows (`p2`/`p5`) are **identical across qwen and the hosted gates** because they are
Finding A (the mock generator quoting a rank-1 distractor), *not* a gate error — so the hosted gate is a
**perfect** gate here and the `0.857` ceiling is the mock generator, which Voyage already lifts by
putting the right doc at rank-1.

**Conclusion — aligned with the product goal.** `$0` already buys *safety*; paid only *improves*
helpfulness and latency, which is precisely "the `$0` stack is best-positioned so paid only improves,"
not "paid is required for correctness." The path to a usable **fully-`$0`** gate is a **semantic
embedder** (clean evidence should flip the local 7B's three holdouts) run through the **cascade**
(gray-zone-only calls bound the ~10 s cost) — the next experiment. This mirrors the project's pattern: a
cheap component's blind spot surfaced by the eval harness, not hidden (cf. the lexical judge, ADR-0017;
the lexical gate, ADR-0021). (Fixed in passing: `_resolve_api_key` now resolves explicit key >
`OPENAI_API_KEY` env > local placeholder — the placeholder had shadowed a real key and 401'd the hosted
path.) The in-process cross-encoder gate remains deferred (heaviest dependency; Finding D shows no
measurable retrieval gap yet).

**Voyage-embedder follow-up (2026-06-07, `qwen2.5:7b` gate) — clean evidence lifts the local gate but it
plateaus *below* the frontier: the wall is the model, not the embedder.** The pre-registered test was "does
a semantic embedder flip qwen's three false refusals (q4/q6/p1)?" Swapping the lexical mock for Voyage
(retrieval **0.940 → 1.000** recall, nDCG **→ 0.988**, MRR **→ 1.000**) lifted the local 7B to **refusal
0.882 / answer 0.857** (from 0.824 / 0.643) and, with the right doc now at rank-1, erased *both* Finding-A
wrong answers (p2 asteroid-belt → Mars, p5 Ganymede → Europa). But it closed only **one** of the three
false refusals: **q4 flipped** to ANSWER once the evidence was clean, while **q6 and p1 still refused on
perfect rank-1 evidence** — q6 against a passage that *verbatim* states "Titan is the largest moon of
Saturn," p1 needing only the synonym map world→planet / star→Sun. Those two are a **7B judgment ceiling**,
not an evidence problem, so no embedder fixes them; the gate plateaus at **0.882 vs the hosted gate's
1.000**. The product reading is sharpened, not changed: `$0` buys safety and *most* helpfulness, and the
frontier premium is now isolated to exactly the calibration/paraphrase rows a mid-size model can't judge.
The remaining open issue is **latency** (~7 s/query), not quality — to be addressed by the cascade (free
score bands carry the confident cases; the slow LLM runs only the gray zone). To make those bands
tunable, the harness `-v` output now prints the per-case **top retrieval score** (`top=`): a threshold
can't be calibrated blind.

**Calibrated cascade on Voyage (2026-06-07) — the synthesis: faster AND better AND cheaper than the pure
LLM gate.** The per-case `top=` scores confirmed ADR-0021's trap in the data: n2/n3 (0.443/0.488) sit
below every answerable, but **n1 "rings on Neptune" (0.526) outscores the lowest legitimate answerable p5
(0.514)** — so no single floor separates them. The cascade's two bands do: **low 0.50** free-abstains the
clear negatives, **high 0.60** free-*answers* the confident rows (every negative is ≤ 0.526, well below),
and only the thin gray zone [0.50, 0.60) — p5, n1, p1, p2 — reaches the LLM. With local `qwen2.5:7b` in
that gray zone: **refusal 0.941, answer 0.929, fabrication 0.0, p50 latency 0.41 s (from ~9 s), 4 LLM calls
instead of 17, ~50 tokens/answer.** Two findings worth keeping:
- **The cascade beats the pure LLM gate on quality, not just speed** (0.941 vs 0.882). The high band
  free-answered **q6** — a verbatim "Titan is the largest moon of Saturn" that qwen *itself* had wrongly
  refused. For a high-confidence retrieval the *score* is a better judge than a mid-size LLM, which
  over-thinks; the LLM earns its keep only in the ambiguous middle. That is the cascade's deep
  justification, now measured.
- **The one residual is p1** (0.592, gray zone, qwen refuses). **Confirmed empirically: `--gate-high 0.55`
  frees p1 → refusal/answer 1.000 / 1.000 at just 2 gray-zone LLM calls (p5, n1) — the architecture reaches
  the ceiling.** But **0.941, not 1.000, is the *shippable* number**, for two reasons. (a) **Overfit:**
  high=0.55 clears n1 (0.526) by only 0.024, and that margin rests on **three negatives**; a larger/harder
  negative set must re-validate before the high band is trusted live (ADR-0021's off-by-default stance
  holds). (b) **Latency tail:** even with calls this rare, **p95 ≈ 20 s** (vs p50 0.34 s), because each
  *local* gray-zone call is slow and variable. The cascade makes the slow calls rare but can't make them
  fast — so a production cascade wants a **hosted or faster gray-zone judge** to crush the tail. That is
  precisely where paid "only improves": it buys **robustness and p95**, not the achievable ceiling.

This closes the Phase-2 gate as designed: a free deterministic floor + a free high band carry the
confident majority at interactive speed and `$0`, and a smart-but-slow LLM is reserved for the ambiguous
minority — fast, cheap, safe, and good. Paid models (a stronger gray-zone judge) or a local embedder are
the only remaining levers to reach 1.000, which is exactly "the `$0` stack is best-positioned, paid only
improves."

### ADR-0023 — Incremental re-index by content-hash diff (Phase 3, slice 1)

**Context.** Phase 3 (Freshness) DoD: "incremental re-index works." The old `ingest()` re-embedded
*every* chunk on every run and only ever upserted — so an **edited** document left its previous chunks in
the index (its content-hash ID changed, ADR-0011, so the old ID was orphaned, never overwritten) and a
**removed** document was never cleaned up. Re-ingest was neither incremental (it paid to embed unchanged
content) nor correct (stale chunks stayed searchable). Both are freshness bugs.

**Decision.** Make `ingest()` diff the fetch against the store on the content-hash ID — the operation
ADR-0011 was chosen to enable. The `VectorStore` port gains two methods, `chunk_ids(tenant)` and
`delete(ids, tenant)`. `ingest()` now embeds **only** chunks whose ID is not already stored, skips the
rest, and — when called with `prune=True` — deletes any stored chunk absent from the fetch. `IngestReport`
gains `added / unchanged / deleted` so the behaviour is **measurable**, per the no-change-without-a-number
rule. `prune` defaults to **False** (purely additive — several sources can share a tenant across calls, the
Phase-0 contract, unchanged); `prune=True` treats the fetch as the tenant's *complete* corpus and is the
freshness path. Whole-tenant reconcile is the demo semantics; a connector-scoped (per-namespace) sync is a
later refinement to add when a second real connector exists — not before (no premature abstraction).

**Measured (`tests/test_ingest_incremental.py`, $0/stdlib, no API).** First ingest of a 3-doc corpus →
`added=3`. Re-ingest unchanged → `added=0, unchanged=3, deleted=0` and the counting embedder records
**zero** new embeddings — re-ingest is a true no-op, not just idempotent storage. Edit one doc, remove one,
add one → `added=2, unchanged=1, deleted=2`, embedding work is the 2 new chunks only (proportional to the
*change*, not the corpus), and a post-sync search returns exactly the new corpus — the removed doc and the
pre-edit text are gone (the freshness guarantee). The existing baseline eval and the idempotent-reingest
e2e test are unchanged, since the default path is still additive and first-ingest numbers are identical.

### ADR-0024 — Staleness surfaced: freshness as a carried fact + an injected-`now` judgment (Phase 3, slice 2)

**Context.** Phase 3 DoD also requires "staleness surfaced." `Document.created_at` (epoch seconds) existed
as a seam from Phase 0 but was never set or propagated, so the age of the evidence behind an answer was
invisible. The hard constraint: computing staleness needs a clock, but the eval harness must stay
**deterministic** (ADR-0010) — a run has to score the same today and next year — so `time.time()` must not
leak into core logic.

**Decision.** Split the **fact** from the **judgment**. The fact — a timestamp — is carried on the chunk:
`Chunk` gains `created_at`, propagated `Document → Chunk` by the chunker, so it rides on every
`Retrieval` and therefore on every `Answer` with no new field. It is deliberately **excluded from the
content-hash ID**, so re-stamping identical content neither changes its identity nor forces a re-embed;
the consequence — content-identical re-ingest keeps the chunk's *first-seen* timestamp — is the intended
semantics (freshness tracks **content change**, not re-fetch time; a separate "last-seen" recency is a
future refinement, not built now). The judgment lives in a pure `core/freshness.py` (`age_seconds`,
`stalest_age`, `stale`) whose every function takes `now` as an **explicit argument** — no wall-clock read
— so staleness is the caller's policy and the facts stay reproducible. An *unset* timestamp (`0.0`) means
"age unknown" and is treated as **not stale** (absence of a date is not evidence of staleness). The harness
`run()` gains an optional `now`; when given, each case's stalest evidence age is computed and shown in `-v`
("stalest evidence: Nd old"). `now` defaults to `None`, so the $0 mock corpus (no dates) reports nothing
and the **baseline output and numbers are unchanged**.

**Measured (`tests/test_freshness.py`, $0/stdlib, no API, fixed `now`).** Helpers: unset → `None`, a 10-day
span → `10·86400`, a future timestamp clamps to `0.0` (connector clock skew reads fresh, never negative);
a 30-day `stale()` policy flags only the 400-day span and never the undated one. End-to-end: seed a source
with dated docs (`add(..., created_at=)`), ingest, answer — the timestamps arrive intact on
`answer.retrievals`, and `stalest_age` reads 400 days. Harness: with a reference `now` the per-case
`evidence_age_s` is populated and `-v` renders "stalest evidence:"; with `now=None` it is `None` and the
line is absent — the default path is provably untouched. Real dates (and a gated freshness metric) arrive
with the live connector in slice 3, where a source supplies its own publish/modified time.

### ADR-0025 — A live filesystem connector (Phase 3, slice 3)

**Context.** Phase 3 DoD's third clause: "a live connector." Slices 1–2 built incremental re-index and
staleness, but every prior `DocumentSource` was `InMemoryDocumentSource` — seeded by hand, with synthetic
(or absent) dates. Nothing exercised the phase against a *real, changing* source, and the freshness
timestamps were never real. The first live connector had to (a) reflect a source's current contents on each
fetch so incremental re-index is demonstrable end-to-end, (b) supply **real** `created_at` so staleness is
genuine, and (c) stay $0/stdlib and **offline-testable** — a network connector (HTTP/S3) can't be tested
deterministically in CI.

**Decision.** Ship `FileSystemDocumentSource` (in `adapters/sources.py`): each `fetch()` globs a directory
(`**/*.txt` by default), reads each file's text, derives a stable posix-relative `source` label, and uses
the file's `st_mtime` as `created_at`. A directory is the right *first* live connector precisely because it
is genuinely live (a fetch sees whatever is on disk *now* — adds, edits, removes since last ingest), its
mtimes are real freshness data, and it needs no network — so it's fully deterministic under a temp dir with
`os.utime`. A missing/non-directory root yields an empty corpus rather than raising, so a connector pointed
at a not-yet-created folder degrades gracefully. Text only; PDF/HTML extraction and remote connectors
(HTTP `Last-Modified`, S3, SEC-EDGAR) are further adapters behind the same `fetch` port — added when each
concrete case arrives, not speculatively (no premature abstraction). No new runtime dependency
(`pathlib`/`os` are stdlib).

**Measured (`tests/test_filesystem_source.py`, $0/stdlib, no network, temp dir + `os.utime`).** Fetch reads
text files recursively with posix labels, excludes non-matching extensions, and carries each file's real
mtime into `created_at`. The capstone ties all three slices together over one real folder: ingest 3 files
(`prune=True`) → `added=3`; the answer's stalest evidence is the genuinely 400-day-old file (real mtime →
chunk → retrieval → `stalest_age`); then keep one file, edit one, remove one, add one and re-ingest →
`added=2, unchanged=1, deleted=2`, and a search confirms the index matches the folder exactly — the removed
file and the pre-edit text are gone. **Phase 3 (Freshness) DoD is now fully met: incremental re-index
works, staleness is surfaced, and a live connector drives both.**

### ADR-0026 — Memory write policy as a port; memory injected as labelled grounded evidence (Phase 4, slice 1)

**Context.** Phase 4 DoD: "per-user personalization; fact-recall + personalization-lift measured." The
`MemoryStore` port and `FileMemoryStore` substrate existed from Phase 0, but the two pipeline seams were
inert stubs — `_understand` ignored the memories it was handed, and `_learn` returned `[]`, so nothing was
ever remembered or used. The hard parts (`docs/memory.md` §3) had to be built: *what is worth remembering*
(write policy), and *how a remembered fact reaches the answer* — under two house constraints. First, the
$0 stack must demonstrate the value (every component earns its place at $0; paid only improves), yet the
default generator is the **extractive** `ExtractiveLLM`, which can only quote evidence — it can't read a
"system context" section. Second, memory must stay **grounded** (`docs/memory.md` §6: never let the model
invent a memory) and **isolated** per `(tenant, user)`.

**Decision.** Three choices.

1. **The write policy is a port, not a hardcoded rule.** *What to remember* is a judgement, and we already
   know the second concrete case (an LLM extractor), so — exactly as Phase 2 introduced `RelevanceGate` with
   a $0 floor plus an LLM drop-in — add a `MemoryExtractor` port now. The $0 `RuleBasedMemoryExtractor`
   recognises **explicit** self-statements (a stated preference, `my <slot> is <value>`, an occupation, an
   explicit `remember that …`) and proposes one `MemoryItem` each, with a content-hash ID (idempotent
   re-statement), a salience weight gated by a floor (the write policy — weak signals are dropped, not
   stored as noise), and the turn as provenance. It favours **precision over recall** and infers nothing;
   that missing recall (implicit/paraphrased facts, episodic summaries) is precisely what a paid LLM
   extractor adds behind the same port — the "paid only improves" lever. So personalization is bought at
   $0, and spend only *widens* what is remembered, never *enables* it.

2. **Memory reaches the answer as labelled grounded evidence, not a separate prompt channel.** A relevant
   memory is wrapped in a `Retrieval` over a synthetic chunk sourced `memory/<turn>` and prepended to the
   reranked corpus evidence. This is the one design that satisfies both constraints at once: the $0
   extractive model *can* use it (it's just evidence it can quote), and grounding *still verifies* it (a
   remembered claim must be entailed by the injected memory text — never invented). Injection is
   **relevance-gated** by token overlap, so a memory unrelated to the current query is never injected and
   can't hijack a corpus answer; when a memory *is* relevant it leads (score `1 + salience`), because a
   personal question is answered from what we know about the user. Because injection happens *before* the
   "no usable context" check, a purely personal question is answerable even when corpus retrieval finds
   nothing. The `memory/` source label keeps it distinct from corpus evidence (`docs/memory.md` §5).

3. **Learn on every exit path, with no clock.** `_remember` runs before *every* return — including both
   abstain paths — so a user stating a fact is remembered whether or not the turn produced an answer. The
   extractor takes no wall-clock time (consistent with ADR-0024): timestamps and recency ranking are
   deferred to slice 2, so this slice stays deterministic. `MemoryItem` gains a `key` slot (the conflict
   field a newer same-slot fact will supersede on) and a `MemoryTurn` input type; neither is part of the
   content-hash ID.

**Measured (`tests/test_memory_extract.py`, `tests/test_memory_pipeline.py`; $0, no API).** The extractor
turns explicit statements into the right kind/content/key/provenance and extracts **nothing** from an
ordinary question (asking never pollutes memory). End to end on the **$0 stack**, a probe answerable only
from a stated preference shows the personalization lift: with memory **off** (a different user) the answer
cannot say "bullet"; with memory **on** it answers "bullet", cited to a `memory/` source — and an
*unrelated* memory leaves a corpus answer (and its corpus citation) untouched. The standard corpus-only
eval (no `user_id`) is **byte-identical** to the Phase 3 baseline (recall 0.940, faithfulness 1.000,
answer 0.857, refusal 0.824) — the personalization path adds capability without touching the corpus path.
Gate green: **102 passed**.

**Deferred (slices 2–3).** Conflict resolution (a newer fact superseding the older on its `key`, keeping
the old one's provenance) and recency-relevance-salience memory ranking (slice 2); a first-class harness
personalization-lift metric + `--memory` CLI and the opt-in paid LLM extractor, plus compaction/TTL
(slice 3). This slice builds the loop and proves the lift; the next two make conflict, ranking, and the
headline number first-class.

### ADR-0027 — Conflict resolution by slot; salience-aware memory ranking (Phase 4, slice 2)

**Context.** Slice 1 gave each memory a `key` slot but did not yet act on it, and `read` ranked by pure
relevance with recency as the only tie-break. Two `docs/memory.md` §3 hard parts remained: **conflict
resolution** ("a new fact contradicts an old one → supersede, don't append both; keep the old one's
provenance for audit") and **ranking** ("by recency × relevance × salience, not pure similarity"). Without
conflict handling, a user correcting a fact (`my name is Ada` → `actually, my name is Bob`) would leave
*both* in memory and the reader could surface the stale one.

**Decision.** Two changes, plus an honest deferral.

1. **Supersede-on-write, in the store.** `FileMemoryStore.write` is where set consistency is enforced: when
   an incoming item has a non-empty `key`, every stored non-superseded item in the same slot gets its
   `superseded_by` set to the new item's ID. The superseded item is **kept on disk** (provenance for audit);
   `read` already filters `superseded_by != None`, so it simply stops surfacing. Keyless facts (e.g. an
   open-ended preference with no stable slot) accumulate rather than collide. The store — not the extractor —
   owns this because it owns the persisted set; a `pg-memory` adapter mirrors it with a SQL update.

2. **Salience-aware ranking.** `read` now sorts by `(relevance, salience, recency)` as ordered keys, so
   among equally-relevant facts the more salient one leads — ranking is no longer pure similarity.

3. **Recency-weighted ranking deferred to slice 3 — deliberately.** The full blend where a *fresh relevant*
   fact can outrank a *stale exact* one needs (a) an injected `now` so the judgment stays deterministic
   (the ADR-0024 rule), and (b) a **dated memory corpus** to measure the weighting against. Shipping a tuned
   weight vector with no number to justify it would violate the house rule ("no change ships without a
   number"). So recency ranking is paired with the slice-3 harness lift metric and dated scenarios that can
   actually measure it — exactly as Phase 3 paired its freshness *filter* with the live connector that gave
   it real dates.

**Measured (`tests/test_memory.py`, `tests/test_memory_pipeline.py`; $0, no API).** Supersede keeps one
current value per slot while retaining the old one on disk with its `superseded_by` link and original
provenance; distinct slots and keyless preferences coexist without false collisions; `read` breaks an
equal-relevance tie by salience. End to end through the pipeline, a user correcting their name is answered
"Bob" and never "Ada" — extractor `key` + store supersede + injection composing. Gate green: **106 passed**.

### ADR-0028 — Personalization-lift eval, and "no extraction from a question" (Phase 4, slice 3a)

**Context.** Phase 4 DoD: "fact-recall + personalization-lift **measured**." The corpus harness runs one
pipeline over a flat golden set with **no `user_id`**, so it structurally cannot measure personalization,
which is per-user and cross-turn (state a fact on one turn, rely on it the next). Slices 1–2 built and
unit-proved the loop; the headline number needed a first-class harness. Building it surfaced a precision
gap in the slice-1 extractor: a probe like "What field do I work in?" matched the occupation rule, so
merely *asking* could write (and, via slot conflict, corrupt) a memory.

**Decision.** Two changes.

1. **A dedicated memory eval (`eval/memory.py`), separate from the corpus path.** A `MemoryScenario` is a
   user's `setup` statements plus a `question` only those facts can answer. `run_memory_lift` answers each
   probe twice — as the **stating user** (memory on) and as a reserved user who never stated anything
   (memory off; per-user isolation keeps it empty) — and reports `lift = correctness_on − correctness_off`,
   plus a stored-fact-recall rate (did the on-answer actually cite a `memory/` source?). It is a separate
   harness, not an `Evaluator`, because the unit of measurement is an **on/off pair across turns**, not a
   per-row score over a single run. CLI: `python -m racore.eval --memory [-v]` over a throwaway store; the
   default corpus baseline is untouched.

2. **The extractor ignores questions.** A sentence ending in `?` is asking, not stating, so it yields no
   memory — and each rule now requires a non-empty captured value. This keeps precision high and, crucially,
   stops a probe (or any question) from polluting memory. It is the right rule regardless of the eval:
   questions are not self-statements.

**Measured (`tests/test_memory_eval.py`, `tests/test_memory_extract.py`; $0, no API).** On the $0 stack
across 5 scenarios (name, preference, fiscal-year, occupation, a "remember that" flight fact): **lift
+1.000** (correctness on **1.000** / off **0.000**), **stored-fact recall 1.000** — every on-answer is
grounded in a `memory/` citation, every off-answer falls back to an irrelevant corpus doc. Asking "What
field do I work in?" now extracts nothing. Gate green: **107 passed**.

**Deferred (slice 3b).** Recency-aware ranking with an injected `now` over a *dated* memory corpus (the
piece held back from slice 2 so it ships with a number), the opt-in **LLM extractor** as the
"paid only improves" lever (it lifts *recall* of implicit/paraphrased facts the rule floor misses, on the
same scenarios), and compaction/TTL. This slice fixes the number the lift is measured by; 3b moves it.

### ADR-0029 — Expose the rule floor's extraction-recall ceiling before closing it (Phase 4, slice 3b-i)

**Context.** ADR-0028's lift was **+1.000** — but over five scenarios that were *all* explicit
self-statements, the exact shape the $0 rule extractor was built to catch. A number that is perfect
because the test contains only the cases the code already handles is "perfect for the wrong reason."
Before building a paid extractor that "only improves," we have to show — with a number — what the $0
floor *cannot* do. (Same discipline as the harder retrieval corpus that turned a saturated hit@k=1.0
into a real recall@k gap before the embedder was built.)

**Decision.** Add **implicit-fact** scenarios to `eval/memory.py` — durable facts stated *without* a
self-statement pattern ("we usually sync on Tuesday mornings", "I had to give up gluten last year",
"Python is the only language I'm comfortable writing") — and an **`extraction_recall`** metric (did a
scenario's setup leave any stored memory for that user?). The rule floor structurally misses these (no
"I prefer / my X is / remember that"), so the headline now reads the floor's *real* recall, not a
saturated 1.0.

**Measured (`tests/test_memory_eval.py`; $0, no API).** On the $0 rule floor over 8 scenarios (5 explicit,
3 implicit): **extraction recall 0.625**, **lift +0.625** (down from the misleading +1.000) — every
explicit fact extracted and answered (cited to `memory/`), **none** of the implicit facts extracted; the
memory-off control stays 0.000. That 0.625 is the honest ceiling the LLM extractor is measured against
(ADR-0030). Gate green: **107 passed**.
