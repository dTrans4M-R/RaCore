"""Freshness: judge the age of retrieved evidence (Phase 3, ADR-0024).

Two things are kept deliberately apart. A timestamp is a **fact** carried on each chunk
(``created_at``, epoch seconds) — a chunk's age at a given moment is fixed and reproducible.
Staleness is a **judgment** that needs a reference ``now`` and a maximum acceptable age — that
policy is the caller's, supplied explicitly, so nothing here reads the wall clock. Keeping the
clock out of the core is what lets the eval harness stay deterministic (ADR-0010): a run scores
the same today and next year because ``now`` is an input, not ``time.time()``.

An *unset* timestamp (``0.0``) means "age unknown", which is treated as **not stale** — absence
of a date is not evidence of staleness, and the $0 mock corpus carries none, so these helpers are
dormant (and the baseline unchanged) until a source supplies real dates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from racore.core.types import Retrieval


def age_seconds(created_at: float, now: float) -> float | None:
    """Age of ``created_at`` at reference ``now``; ``None`` when the timestamp is unset (``0.0``).

    Clamped at ``0.0`` so a slightly-future timestamp (clock skew at a connector) reads as fresh,
    never negative.
    """
    if created_at <= 0.0:
        return None
    return max(0.0, now - created_at)


def stalest_age(retrievals: Sequence[Retrieval], now: float) -> float | None:
    """The largest evidence age across ``retrievals`` at ``now`` — the freshness an answer is only
    as good as. ``None`` when no retrieval carries a timestamp."""
    ages = [age for r in retrievals if (age := age_seconds(r.chunk.created_at, now)) is not None]
    return max(ages) if ages else None


def stale(
    retrievals: Sequence[Retrieval], now: float, max_age_seconds: float
) -> tuple[Retrieval, ...]:
    """The retrievals whose evidence is older than ``max_age_seconds`` at ``now``.

    Untimestamped evidence is never flagged (unknown age ≠ stale). The result preserves input
    order, so the caller can report or drop the stale spans while keeping rank.
    """
    return tuple(
        r
        for r in retrievals
        if (age := age_seconds(r.chunk.created_at, now)) is not None and age > max_age_seconds
    )
