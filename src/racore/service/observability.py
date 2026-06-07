"""Per-request observability for the service layer (Phase 5 slice 4).

The same numbers the eval harness gates in aggregate — latency, cache behaviour, tokens, abstain
rate, grounding — are what an operator needs *per request* in production, sliced by tenant. This
module turns each service call into one structured ``ServiceEvent`` and hands it to a pluggable
``Observer``. It is purely additive: the pipeline already records per-stage timings (ADR-0010) and
token usage (ADR-0018) on the result it returns, so the event is built from that — no core change,
no second clock, and zero cost when no observer is wired (the default).

The default sink (``LoggingObserver``) emits a JSON line through stdlib ``logging`` — $0, no
dependency. A statsd / OpenTelemetry / billing sink is another adapter behind the same port.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

_LOGGER = logging.getLogger("racore.service")


@dataclass(frozen=True, slots=True)
class ServiceEvent:
    """One structured record of a handled request.

    ``detail`` carries the operation-specific metrics (an answer reports cache-hit / abstained /
    tokens / grounding; an ingest reports added/unchanged/deleted), kept as a mapping so a new
    operation doesn't change this type. ``duration_ms`` is the operation's own per-stage total
    (ADR-0010), not a re-measured wall clock, so it stays consistent with what the harness reports.
    """

    request_id: str
    operation: str
    tenant_id: str
    user_id: str | None
    duration_ms: float
    detail: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "operation": self.operation,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "duration_ms": self.duration_ms,
            **self.detail,
        }


class Observer(Protocol):
    """A sink for service events. Sync and fire-and-forget: emitting an observation must never block
    or fail a request, so a remote sink buffers internally rather than awaiting on the hot path."""

    def observe(self, event: ServiceEvent) -> None: ...


class LoggingObserver:
    """The $0 default sink: one JSON line per event through stdlib ``logging`` at INFO.

    Structured (JSON) so a log pipeline can index and aggregate it — cache hit-rate, p95 latency,
    cost, and abstain rate per tenant all fall out of these lines without any extra instrumentation.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or _LOGGER

    def observe(self, event: ServiceEvent) -> None:
        self._logger.info(json.dumps(event.to_dict()))
