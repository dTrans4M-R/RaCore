# Freshness — keeping the knowledge base current, and knowing how old an answer is

> A retrieval system is only as trustworthy as the corpus behind it. Two failures matter: the index
> drifting out of sync with its sources (stale or duplicated content surviving a re-ingest), and an
> answer built on evidence that is simply too old, with no way to tell. This note records how RaCore
> keeps the index current **incrementally** and makes the **age** of an answer's evidence visible. This
> is **Phase 3** (see [`roadmap.md`](roadmap.md)) — built and measured, not forward-looking.

---

## 1. Incremental re-index by content-hash diff (ADR-0023)

Documents and chunks are identified by a hash of their content (ADR-0011). That single decision is what
makes incremental re-index fall out for free: `ingest()` diffs the fetched corpus against what the store
already holds, on the chunk ID.

- **Unchanged** content keeps its ID → it is **skipped**, not re-embedded. Re-ingesting an unchanged
  corpus embeds *nothing* — a true no-op, not just an idempotent overwrite.
- **Changed** content mints a **new** ID → the new chunk is embedded and upserted, and (with
  `prune=True`) the orphaned old chunk is **deleted**, so an edit leaves no stale duplicate.
- **Removed** documents leave chunks absent from the fetch → `prune=True` deletes them.

`IngestReport` reports `added / unchanged / deleted` so the behaviour is **measured**, not assumed. The
`VectorStore` port carries the two methods this needs — `chunk_ids()` and `delete()`. `prune` defaults to
**False** (purely additive: several sources can share a tenant across calls — the Phase-0 contract); the
freshness path opts in with `prune=True`, which treats the fetch as the tenant's complete corpus. A
connector-scoped (per-namespace) reconcile is a later refinement, to add when a second real connector
needs it — not before.

**Deliberate semantics:** a content-identical re-ingest keeps the chunk's *first-seen* timestamp — freshness
tracks **content change**, not re-fetch time. A separate "last-seen" recency is a future refinement.

## 2. Staleness as a carried fact + an injected judgment (ADR-0024)

Freshness has two halves, kept apart on purpose:

- **The fact** — a timestamp. `created_at` (epoch seconds) is carried `Document → Chunk → Retrieval`, so
  the age of an answer's evidence rides on every `Answer` with no extra plumbing. It is **excluded from the
  content-hash ID**, so re-stamping identical content never changes identity or forces a re-embed.
- **The judgment** — staleness. The pure helpers in `core/freshness.py` (`age_seconds`, `stalest_age`,
  `stale`) each take `now` as an **explicit argument**. Nothing reads the wall clock, so a chunk's age is a
  fixed fact and the eval harness scores the same today and next year (the fact-vs-judgment split, ADR-0024).

An *unset* timestamp (`0.0`) means "age unknown" and is treated as **not stale** — absence of a date is not
evidence of staleness. The harness `run(now=…)` surfaces the stalest evidence age per case in `-v`; with
`now=None` (the default) nothing is reported, so the $0 mock corpus — which carries no dates — leaves the
baseline output and numbers unchanged.

## 3. A live connector supplies the real dates (ADR-0025)

`FileSystemDocumentSource` is the first live source: each `fetch()` reflects a directory's *current*
contents, and every file's modified time (`st_mtime`) becomes its document's `created_at`. So the two
mechanisms above become concrete end-to-end — incremental re-index over a folder that actually changes, and
staleness measured from real mtimes. It is text-only and stdlib (`pathlib`/`os`), with no network, so it is
fully deterministic under a temp dir. Remote connectors (HTTP `Last-Modified`, S3, SEC-EDGAR) and
structured extractors (PDF, HTML) are further adapters behind the same `fetch` port.

## 4. What's measured, and what's next

`tests/test_ingest_incremental.py`, `tests/test_freshness.py`, and `tests/test_filesystem_source.py` prove
the three mechanisms — all $0/stdlib, no API. The capstone ingests a real folder, shows the answer's
stalest evidence is the genuinely oldest file, then edits/removes/adds files and confirms the re-ingest
touches only the delta and leaves the index matching the folder exactly.

Deferred until they have a corpus that makes them meaningful: a freshness-aware **filter/penalty** (prefer
or drop stale evidence in retrieval — policy, beyond visibility) and a **gated** freshness metric in the
default eval panel. Both want real dates from a connector rather than synthetic ages, so they follow the
live-connector slice rather than precede it.
