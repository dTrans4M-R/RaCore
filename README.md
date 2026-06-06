# RaCore

> A reusable, provider-agnostic engine for building **grounded, real-time, memory-aware**
> retrieval systems you can actually trust — with **evidence and evaluation as first-class
> concerns**, not afterthoughts.

RaCore is a **clean-room, independently-owned engine**. It plugs into any application via an
in-process Python SDK or an HTTP service, and shares **no code** with any downstream product.

---

## Why this exists (the one-paragraph thesis)

Naive RAG — *chunk → embed → cosine search → stuff into a prompt* — is commoditizing into a
library call. What is **not** commoditizing, and is becoming the backbone of serious AI products,
is the **reliability layer**: getting the *right, current, permissioned, trustworthy* information
into a model's context **with provenance**, knowing **when not to answer**, **remembering** the
user across turns, and being able to **prove all of it with numbers**. Bigger context windows and
smarter models do not remove that problem — they make it more central. RaCore is a bet on that layer.

## The pillars

| Pillar | What it means | Where it lives |
|---|---|---|
| **Grounding** | every claim is backed by a verbatim source + citation; unsupported claims are dropped or flagged | `core/grounding.py` |
| **Freshness** | knowledge stays current via incremental indexing + connectors; staleness is visible | `adapters/sources/` |
| **Relevance** | retrieve the right context, filter the rest, and **abstain** when there's no support | retrieval + rerank |
| **Memory** *(differentiator)* | per-user persistent memory that personalizes across sessions | `adapters/memory/` |
| **Evaluation** *(the moat)* | retrieval / grounding / answer / memory quality measured + gated in CI | `eval/` |

## Repo shape

```
RaCore/                     ← this repo (the engine; Apache-2.0)
  src/racore/               ← provider-agnostic core + pluggable adapters
  docs/                     ← architecture, evaluation, memory, roadmap, decisions
  tests/
client-<name>/              ← (separate repo) a consumer's config + plugins — NOT here
```

A consumer (including any product that adopts RaCore) depends on a **released, pinned version** of
this package and contributes only **config + plugins** in its own repo. Engine changes always land
here, never in a consumer repo. See [`CLAUDE.md`](CLAUDE.md) for the house rules that enforce this.

## Status

**Phase 0 — Foundation (scaffolding).** Guardrails, tooling, and the repo structure are in place;
the walking skeleton (the first end-to-end slice) is the next step. Build order and per-phase
"definition of done" are in [`docs/roadmap.md`](docs/roadmap.md).

## Docs

- [`docs/architecture.md`](docs/architecture.md) — ports-and-adapters design, core types, the ingest/answer pipelines, how apps connect.
- [`docs/evaluation.md`](docs/evaluation.md) — the measurement strategy (the differentiator).
- [`docs/memory.md`](docs/memory.md) — the per-user persistent memory subsystem.
- [`docs/roadmap.md`](docs/roadmap.md) — the phased build plan + the public learning-doc outline.
- [`docs/decisions.md`](docs/decisions.md) — the architecture decision log (why each choice was made).

## Getting started (for the builder)

1. **Python 3.12+** and [`uv`](https://docs.astral.sh/uv/). Node 20+ later for the demo UI.
2. `uv sync` — creates the venv and installs the dev toolchain (ruff, mypy, pytest, pre-commit).
3. `uv run pre-commit install` — activates the commit-time quality gate.
4. `uv run pytest` — runs the test suite (smoke test today; the E2E slice after Phase 0).
5. Phase 0 produces the walking skeleton: `core/` types + ports + pipeline, an in-memory adapter
   set (so it runs at **$0**), the eval-harness stub, and one passing end-to-end test
   (ingest → retrieve → ground → cite → answer) that prints a **baseline** number.

## License

**Apache-2.0** — see [`LICENSE`](LICENSE). You keep an irrevocable right to reuse this engine;
anyone may adopt it. Consumer-specific customization is separate code owned by the consumer.
