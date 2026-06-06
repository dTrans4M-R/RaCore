# CLAUDE.md — house rules for RaCore

> Read this before touching anything. These rules keep the engine **clean, simple, scalable, and
> legally yours**. When a request conflicts with a rule here, surface the conflict instead of
> silently following it.

## 0. What this project is (and is not)

- RaCore is a **provider-agnostic, clean-room retrieval engine**, owned independently. It is the
  reusable core only.
- It contains **no client data, no client-specific logic, and no code copied from any other
  product** — ever. That clean-room boundary is what keeps the engine legally yours and sellable.
- Strategy, business, and IP notes live in a **separate private repo** (`racore-strategy`). They
  must **never** appear here. The pre-commit gate blocks files named `vision-and-business*`,
  `roadmap-full*`, or `*strategy*` from being committed to this repo.

## 1. Architecture rules (non-negotiable)

1. **Ports & adapters.** The core (`src/racore/core/`) knows nothing about any concrete provider.
   Everything that varies is a **port** (a `Protocol`); a provider is an **adapter**. See
   [`docs/architecture.md`](docs/architecture.md).
2. **Customization = config or plugin, never a fork.** If a need can't be expressed as config or a
   plugin, that's the signal to add a **new port** — not to fork or special-case the core.
3. **Async-first.** Every port method that does I/O is `async def`. A synchronous facade may wrap
   the async core for simple embedding; never the reverse. (ADR-0009)
4. **Batch-first signatures.** I/O ports take and return **lists** (`embed(texts) -> vectors`), so
   batching is the default path, not a retrofit. Single-item is a batch of one. (ADR-0009)
5. **`answer()` returns a streamable result type** from day one, even before streaming is wired —
   so adding streaming later never changes the public signature. (ADR-0009)
6. **Stable content-hash IDs.** Documents and chunks get deterministic IDs derived from content, so
   re-ingest is idempotent and incremental indexing works later without rework. (ADR-0011)
7. **Per-stage timing is built into the pipeline**, not bolted on — every stage emits a duration so
   latency regressions are localizable. (ADR-0010)

## 2. Measurement rules (the moat)

- **No change ships without a number.** Behaviour changes are judged against the eval harness, not
  vibes: baseline → change → re-measure → keep only if a metric improved without regressing another.
- **Latency and cost are gated metrics**, not just reported ones. A PR that regresses p95 latency or
  cost/answer fails the gate exactly like one that drops faithfulness. (ADR-0010)
- **Faithfulness is the headline metric.** See [`docs/evaluation.md`](docs/evaluation.md).

## 3. Code style — keep it clean and simple

- **Formatting & linting:** `ruff format` + `ruff check` are the source of truth. Don't hand-format.
- **Types:** full type hints; `mypy --strict` must pass. Public functions get precise signatures.
- **Simplicity over cleverness:** small modules, one responsibility each. **No premature
  abstraction** — add a port/layer when a second concrete case exists, not in anticipation.
- **No dead code, no commented-out blocks, no TODO without an issue reference.**
- **Comments explain *why*, not *what*.** Match the density and idiom of surrounding code.

## 4. Dependencies

- **No new runtime dependency without a clear justification** (what it buys, why stdlib/existing
  deps don't). Prefer the standard library and what's already in `pyproject.toml`.
- Pin sensibly; keep the core's dependency surface small (adapters may pull provider SDKs).

## 5. Testing

- **Every behaviour change ships with a test.** Tests live in `tests/`, mirror the package layout.
- `uv run pytest` must be green before any commit. The pre-commit gate enforces this.

## 6. Secrets & data

- **Never commit secrets.** API keys go in `.env` (gitignored); the gate runs a secret scanner.
- **Never commit real client/user data.** Test fixtures are synthetic or public-domain only.

## 7. Git & IP hygiene

- **Engine work lands in this repo.** Consumer/client work lands in that consumer's own repo as
  config + plugins against a **released, pinned** RaCore version — never by editing the engine inside
  a gig. This is both good engineering and the practice that keeps the IP boundary clean.
- Small, focused commits. Conventional, imperative messages ("add reranker port", not "changes").
- Don't push or open PRs unless asked.

## 8. Docs

- Keep docs **technical and current**. When a decision changes, **add an ADR** to
  [`docs/decisions.md`](docs/decisions.md) — don't rewrite history.
- Update the relevant doc in the same change that alters the behaviour it describes.

## Quick commands

```
uv sync                       # set up / update the environment
uv run pytest                 # tests
uv run ruff format .          # format
uv run ruff check --fix .     # lint + autofix
uv run mypy                   # type-check
uv run pre-commit run --all   # run the full gate locally
```
