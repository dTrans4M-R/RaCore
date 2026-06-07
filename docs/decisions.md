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

**Live local run (2026-06-07, Ollama `llama3.2` 3B, mock stack) — the gate's quality is model-bound.**
The adapter works end-to-end and cost is honestly `n/a`/`$0`, but the 3B model **over-abstained**:
refusal accuracy **0.647** (six false refusals on answerable rows where the right doc *was* retrieved)
vs the Claude gate's **1.000**, dragging answer correctness to 0.429 — both *below* the no-gate baseline
(0.824 / 0.857), at ~4 s/query. The failure is a weak model told to judge strictly refusing the noisy
evidence a *lexical* embedder retrieves; the protective direction held (all three negative controls
caught, false-answer rate 0.0 — it over-refuses, never fabricates). So the local gate is only as good as
its model: the path to a usable local gate is a **stronger model (7-8B)** and/or a **semantic embedder**
(cleaner evidence, fewer gray-zone calls via the cascade), not an adapter change. This mirrors the
project's pattern — the cheap component has a measured blind spot, surfaced by the eval harness, not
hidden (cf. the lexical judge, ADR-0017; the lexical gate, ADR-0021). The in-process cross-encoder gate
remains deferred (heaviest dependency; Finding D shows no measurable retrieval gap yet).
