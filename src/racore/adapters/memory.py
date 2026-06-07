"""A file-backed per-(tenant, user) memory store (JSON on disk).

One file per ``(tenant, user)`` enforces the isolation boundary memory requires
(``docs/memory.md`` §7): a query for one user can only ever load that user's file. Reads
rank by query relevance then recency and skip superseded items; writes are idempotent by
ID. The full salience/compaction/conflict machinery lands in Phase 4 — this is the
durable substrate it will build on (the pg-memory adapter mirrors the same port).
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from racore.core.types import MemoryItem, MemoryKind, Vector

if TYPE_CHECKING:
    from pathlib import Path

_TOKEN_RE = re.compile(r"\w+")
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_-]")


class FileMemoryStore:
    """Persists memories as one JSON array per ``(tenant, user)`` under ``base_dir``."""

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir

    async def read(self, tenant_id: str, user_id: str, query: str, k: int) -> list[MemoryItem]:
        items = [item for item in self._load(tenant_id, user_id) if item.superseded_by is None]
        query_tokens = _tokens(query)
        # Rank by relevance, then salience, then recency — not pure similarity (docs/memory.md §3):
        # among equally-relevant facts the more salient one leads. The full recency-weighted blend
        # (where a fresh relevant fact can outrank a stale exact one) needs an injected `now` and a
        # dated corpus to measure against, and lands in slice 3 (ADR-0027).
        items.sort(
            key=lambda item: (
                _overlap(query_tokens, item.content),
                item.salience,
                item.last_used_at,
            ),
            reverse=True,
        )
        return items[:k]

    async def write(self, tenant_id: str, user_id: str, items: list[MemoryItem]) -> None:
        if not items:
            return
        merged = {item.id: item for item in self._load(tenant_id, user_id)}
        for item in items:
            # Conflict resolution (docs/memory.md §3): a newer fact in the same slot supersedes the
            # older rather than contradicting it. The superseded item is kept on disk with its
            # provenance for audit; `read` filters it out. Keyless facts just accumulate.
            if item.key:
                for prior_id, prior in merged.items():
                    if (
                        prior.superseded_by is None
                        and prior.key == item.key
                        and prior_id != item.id
                    ):
                        merged[prior_id] = replace(prior, superseded_by=item.id)
            merged[item.id] = item  # idempotent upsert by content/turn-derived ID.
        self._dump(tenant_id, user_id, list(merged.values()))

    # --- persistence helpers ------------------------------------------------------

    def _path(self, tenant_id: str, user_id: str) -> Path:
        name = f"{_safe(tenant_id)}__{_safe(user_id)}.json"
        return self._base / name

    def _load(self, tenant_id: str, user_id: str) -> list[MemoryItem]:
        path = self._path(tenant_id, user_id)
        if not path.exists():
            return []
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        return [_decode(entry) for entry in raw]

    def _dump(self, tenant_id: str, user_id: str, items: list[MemoryItem]) -> None:
        self._base.mkdir(parents=True, exist_ok=True)
        payload = [_encode(item) for item in items]
        self._path(tenant_id, user_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _encode(item: MemoryItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "tenant_id": item.tenant_id,
        "user_id": item.user_id,
        "kind": item.kind.value,
        "content": item.content,
        "source": item.source,
        "key": item.key,
        "salience": item.salience,
        "embedding": list(item.embedding),
        "created_at": item.created_at,
        "last_used_at": item.last_used_at,
        "superseded_by": item.superseded_by,
    }


def _decode(entry: Any) -> MemoryItem:
    embedding: Vector = tuple(float(value) for value in entry["embedding"])
    return MemoryItem(
        id=str(entry["id"]),
        tenant_id=str(entry["tenant_id"]),
        user_id=str(entry["user_id"]),
        kind=MemoryKind(entry["kind"]),
        content=str(entry["content"]),
        source=str(entry["source"]),
        key=str(entry.get("key", "")),  # tolerate pre-ADR-0026 files written without a slot.
        salience=float(entry["salience"]),
        embedding=embedding,
        created_at=float(entry["created_at"]),
        last_used_at=float(entry["last_used_at"]),
        superseded_by=entry["superseded_by"],
    )


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _overlap(query_tokens: set[str], content: str) -> int:
    return len(query_tokens & _tokens(content))


def _safe(value: str) -> str:
    return _UNSAFE_RE.sub("_", value)
