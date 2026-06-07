# Evaluation — the moat

> 95% of "RAG engineers" have no eval harness. Being able to **prove** retrieval quality, grounding
> faithfulness, and hallucination rate — with numbers, in CI — is the scarcest, most valuable skill in
> this space. This is the differentiator, so it's a first-class module from day one.

---

## 1. Principle: measure before you optimize

Every change is judged against a dataset, not vibes. Workflow: **baseline → change → re-measure →
keep only if a metric improved without regressing another.** A change you can't measure doesn't ship.

## 2. The metric taxonomy

Evaluate each stage independently — a good final answer can hide a broken retriever (and vice-versa).

| Layer | Metric | Question it answers |
|---|---|---|
| **Retrieval** | recall@k, nDCG@k, MRR, context precision | Did the right chunks get retrieved, ranked high? |
| **Grounding** | **faithfulness / attribution**, citation correctness, unsupported-claim rate | Is every claim actually supported by a cited source? |
| **Answer** | correctness (exact-match where possible, else LLM-judge), completeness | Is the answer right and complete? |
| **Relevance / refusal** | abstention accuracy, **false-answer-on-no-evidence rate** | Does it say "I don't know" when it should, instead of hallucinating? |
| **Memory** | stored-fact recall, personalization lift, staleness | Does it remember the user and use it correctly? → [`memory.md`](memory.md) |
| **Ops** | cost/answer, p50/p95 latency, tokens | Is it affordable and fast enough? |

**Faithfulness is the headline metric** — for each claim in the answer, is there a retrieved span that
entails it? Measured deterministically where the citation maps to a verbatim span, and with an
LLM-judge (entailment) for paraphrased support. The unsupported-claim rate is your hallucination gauge.

**Abstentions are excluded from faithfulness.** A correct refusal ("I don't know" on a no-evidence
question) makes no grounded claims — scoring it for faithfulness would punish the right behaviour. The
pipeline records a model refusal as an abstention (ADR-0013), faithfulness skips abstained cases, and
refusal accuracy credits them. Read the two together: low faithfulness with high refusal accuracy can
mean the system is correctly *declining*, not hallucinating. (Found via per-case `-v`: a real model
scored 1.0 on every answerable row, and the only drag on the aggregate was two correct refusals being
mis-scored — which is what this records.)

## 3. Datasets

- **Golden set** — hand-built `(question, expected_answer, relevant_sources)` rows, where
  `relevant_sources` is the *set* of documents that should be retrieved (one for a single-fact row,
  several for a multi-source row). This is your regression bedrock. It is deliberately **not** an easy
  corpus: it carries **distractors** (docs that share a question's words without answering it) and
  **paraphrase-gap** questions (wording that diverges from the answer doc), so retrieval is non-trivial
  and has real headroom for hybrid retrieval and reranking to close (ADR-0014). It is scored both by
  position-blind **recall@k** and by the rank-aware **nDCG@k / MRR** (ADR-0015) — the latter expose the
  buried-but-present docs recall hides, and are the metrics a reranker moves. Today's $0 baseline:
  recall@k ≈ 0.94, nDCG@k ≈ 0.86, MRR ≈ 0.89, answer-correctness ≈ 0.86, grounding 1.0.
- **Public benchmarks** — for the contracts/financial corpus, **CUAD** (contract clause QA) is a
  strong fit; add a small financial-QA set for numbers. Use them to sanity-check against the field.
- **Negative controls** — questions whose answer is **not** in the corpus. The system must **abstain**,
  not fabricate. This directly tests relevance + refusal (and catches the most damaging failure mode).
- **Synthetic generation** — bootstrap Q→A→span from the corpus with an LLM, then human-verify a sample.

## 4. The harness (`eval/`)

```
harness.run(dataset, pipeline_config) ->
    for each row: answer = pipeline.answer(row.question)
    score: retrieval metrics + grounding metrics + answer/refusal metrics
    emit: per-row results + aggregate report (JSON + human table)
```

- **Regression gate:** the pre-commit gate runs the full test suite — which asserts the key numbers,
  including the honest ones like memory's 0.625 floor — locally and in CI, so a change that drops a
  metric **fails the build**. Critically, **latency and cost are gated too** (ADR-0010): a PR that
  regresses p95 latency or cost-per-answer fails exactly like one that drops faithfulness. This is what
  turns "I think it's grounded" into a guarantee.
- **Determinism:** the `$0` stack is reproducible end to end, and anything time-dependent takes its
  clock as an **injected argument** rather than reading the wall clock (the fact-vs-judgment split,
  ADR-0024) — the precondition for gating on a number that must mean the same thing a year from now.
- **Live report:** the harness (and the service's per-request observability, ADR-0033) surface the
  numbers — clients and engineers both trust a system that shows its work.

## 5. What "good" looks like (targets to set, then beat)

Set explicit thresholds per phase (e.g. faithfulness ≥ 0.95 on the golden set, false-answer-on-no-
evidence ≤ 2%, recall@10 ≥ 0.9). The exact numbers matter less than: **they exist, they're tracked
over time, and a regression blocks merge.**
