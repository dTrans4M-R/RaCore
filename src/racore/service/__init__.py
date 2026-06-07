"""The product surface: a thin, transport-agnostic facade over the engine (Phase 5).

``RaCoreService`` turns a wire request into core objects, drives the ``Pipeline``, and returns the
rich core result — so corpus multi-tenancy and the per-user memory loop are implemented once and
inherited by every transport. ``racore.service.asgi`` is the HTTP adapter over it; an embedded
import or a CLI are equally valid callers. The core is never touched: this is wiring, not a
re-architecture (``docs/latency.md`` §6).
"""

from __future__ import annotations

from racore.service.asgi import ASGIApplication, create_app
from racore.service.core import RaCoreService, ServiceError, demo_service
from racore.service.types import (
    AnswerRequest,
    DocumentInput,
    IngestRequest,
    MemoryReadRequest,
    MemoryWriteRequest,
)

__all__ = [
    "ASGIApplication",
    "AnswerRequest",
    "DocumentInput",
    "IngestRequest",
    "MemoryReadRequest",
    "MemoryWriteRequest",
    "RaCoreService",
    "ServiceError",
    "create_app",
    "demo_service",
]
