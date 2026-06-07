"""Phase 3, slice 3: a live filesystem connector (ADR-0025).

The capstone that ties the phase together over a real directory: each fetch reflects the folder's
current contents (so re-ingest is incremental, slice 1) and every file's modified time becomes its
document's freshness timestamp (so staleness is real, not synthetic, slice 2). Offline and $0 — a
temp dir with mtimes set explicitly via ``os.utime`` keeps it deterministic.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from racore.adapters.sources import FileSystemDocumentSource
from racore.core.freshness import stalest_age
from racore.core.types import InputType, Query
from racore.eval import demo_pipeline

if TYPE_CHECKING:
    from pathlib import Path

_NOW = 1_000_000_000.0
_DAY = 86_400.0


def _write(root: Path, name: str, text: str, mtime: float) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_reads_text_files_with_real_mtimes(tmp_path: Path) -> None:
    asyncio.run(_reads_text_files_with_real_mtimes(tmp_path))


async def _reads_text_files_with_real_mtimes(tmp_path: Path) -> None:
    _write(tmp_path, "a.txt", "Neptune is a planet.", _NOW - 10 * _DAY)
    _write(tmp_path, "sub/b.txt", "Saturn is a planet.", _NOW - 400 * _DAY)
    _write(tmp_path, "ignore.md", "not matched by the glob", _NOW)

    docs = await FileSystemDocumentSource(tmp_path).fetch()

    by_source = {d.source: d for d in docs}
    assert set(by_source) == {"a.txt", "sub/b.txt"}  # recursive, posix labels, .md excluded.
    assert by_source["a.txt"].text == "Neptune is a planet."
    # The freshness timestamp is the file's real mtime (tolerant of filesystem mtime granularity).
    assert abs(by_source["a.txt"].created_at - (_NOW - 10 * _DAY)) < 1.0
    assert abs(by_source["sub/b.txt"].created_at - (_NOW - 400 * _DAY)) < 1.0


def test_missing_root_is_an_empty_corpus(tmp_path: Path) -> None:
    docs = asyncio.run(FileSystemDocumentSource(tmp_path / "does-not-exist").fetch())
    assert docs == []


def test_live_reindex_and_real_staleness(tmp_path: Path) -> None:
    asyncio.run(_live_reindex_and_real_staleness(tmp_path))


async def _live_reindex_and_real_staleness(tmp_path: Path) -> None:
    pipeline = demo_pipeline()
    source = FileSystemDocumentSource(tmp_path)

    # v1: three files, one of them genuinely old.
    _write(tmp_path, "a.txt", "Neptune is a planet.", _NOW - 10 * _DAY)
    _write(tmp_path, "b.txt", "Saturn is a planet.", _NOW - 400 * _DAY)
    _write(tmp_path, "d.txt", "Mars is a planet.", _NOW - 5 * _DAY)
    r1 = await pipeline.ingest(source, prune=True)
    assert (r1.added, r1.unchanged, r1.deleted) == (3, 0, 0)

    # Staleness is real and end-to-end: the answer's oldest evidence is the 400-day file.
    answer = await pipeline.answer(Query(text="Which is a planet?", k=5))
    age = stalest_age(answer.retrievals, _NOW)
    assert age is not None and abs(age - 400 * _DAY) < 1.0

    # Mutate the live folder: keep a, edit b, remove d, add c.
    _write(tmp_path, "b.txt", "Saturn is the ringed planet.", _NOW - 1 * _DAY)
    (tmp_path / "d.txt").unlink()
    _write(tmp_path, "c.txt", "Venus is a planet.", _NOW - 2 * _DAY)

    r2 = await pipeline.ingest(source, prune=True)
    # Only the delta is touched: edited-b + new-c added; old-b + removed-d deleted; a unchanged.
    assert (r2.added, r2.unchanged, r2.deleted) == (2, 1, 2)

    # The index now matches the folder exactly — the removed file and pre-edit text are gone.
    (probe,) = await pipeline.embedder.embed(["planet"], InputType.QUERY)
    hits = await pipeline.store.search(probe, k=100, tenant_id="default")
    sources = {r.chunk.source for r in hits}
    assert sources == {"a.txt", "b.txt", "c.txt"}  # d.txt is gone.
    b_hit = next(r for r in hits if r.chunk.source == "b.txt")
    assert b_hit.chunk.text == "Saturn is the ringed planet."  # the edited text, not the old.
