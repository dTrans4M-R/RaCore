# Memory — the differentiator

> Per-user persistent memory — the engineered cousin of a `memory.md` — is the least-saturated, most
> current part of this space. Vanilla RAG retrieves from a shared corpus; **memory is learned, private,
> per-user state that personalizes across sessions.** Getting it right (with provenance, compaction,
> and conflict handling) is what makes the project *not* generic.

---

## 1. Memory vs. retrieval (keep them distinct)

| | Retrieval (RAG) | Memory |
|---|---|---|
| Source | shared corpus (docs the user uploaded/owns) | the user's own history + stated facts |
| Scope | per tenant | per **(tenant, user)** |
| Lifecycle | indexed once, queried | continuously written, compacted, decayed |
| Failure if wrong | misses context | feels "forgetful" or "creepy/wrong" |

They meet in the `answer()` pipeline: memory is read **before** retrieval (it personalizes the query
and the answer) and written **after** (it learns from the turn).

## 2. Memory types

- **Profile / preferences** — durable facts about the user ("works in M&A", "prefers bullet summaries").
- **Semantic facts** — discrete things learned ("their fiscal year ends in March").
- **Episodic** — summaries of past conversations/sessions.
- **Working / task** — short-lived state within an active task (often not persisted).

## 3. The hard parts (where most implementations fail)

- **Write policy — what's worth remembering.** Not every turn. A **salience** judgement (stable,
  reusable, user-specific) decides what gets stored, so memory doesn't bloat into noise. *Built
  (ADR-0026):* a pluggable `MemoryExtractor` port — a $0 rule-based floor for explicit
  self-statements (precision over recall), with a salience floor as the gate; an LLM extractor for
  implicit facts is the paid drop-in.
- **Compaction / summarization.** Periodically fold many episodic items into concise summaries; cap
  size. Unbounded memory is as bad as none.
- **Conflict resolution.** New fact contradicts an old one → **supersede**, don't append both (mirror
  the document pipeline's "newer supersedes prior"). Keep the old one's provenance for audit.
- **Retrieval of memory.** Rank by **recency × relevance × salience**, not pure similarity — a stale
  exact match shouldn't beat a fresh relevant one.
- **Forgetting / TTL.** Working memory expires; episodic decays; profile persists. Explicit, not implicit.
- **Provenance.** A "remembered" fact carries where it came from (which turn/source). **Never let the
  model invent a memory** — memory is grounded too.

## 4. Schema (`MemoryItem`)

```
MemoryItem:
  id, tenant_id, user_id
  type: profile | semantic | episodic | working
  content: str               # the fact / summary, concise
  source: str                # turn id / document ref — provenance
  key: str                   # the slot a fact occupies ("name"); a newer same-key fact supersedes
  salience: float            # why it was kept
  embedding: Vector          # for relevance retrieval
  created_at, last_used_at   # for recency + decay
  superseded_by: id | null   # conflict resolution
```

## 5. Read & write paths

- **Read** (`MemoryStore.read(tenant, user, query)`) → inject the top-ranked items into the system
  context before generation, clearly labelled as "known about the user" (separate from retrieved corpus
  evidence, so the two never blur).
- **Write** (`MemoryStore.write(...)`) → after the answer, an extractor proposes candidate memories from
  the turn; the salience gate + conflict check decide store / update / skip.

## 6. Evaluation

- **Stored-fact recall** — after telling the system a fact, does it use it correctly N turns later?
- **Personalization lift** — answer quality with memory on vs. off, on a per-user eval set.
- **Staleness / conflict** — when a fact changes, does the system supersede rather than contradict itself?

→ folds into the harness in [`evaluation.md`](evaluation.md).

## 7. Privacy & security

Memory is the most sensitive data in the system. **Strict per-(tenant, user) isolation** at the store
boundary; **deletion / right-to-forget** as a first-class operation; never use one user's memory to
answer another's query. This isolation is also why memory can't just live in a giant shared context —
it's a structural reason retrieval/memory systems outlast bigger context windows.
