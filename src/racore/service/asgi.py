"""A thin, dependency-free ASGI surface over ``RaCoreService`` (Phase 5 slice 2).

The HTTP transport is an *adapter*, not the product: it maps requests onto the service facade and
serializes the rich core results back out. Hand-rolled against the raw ASGI protocol on purpose —

* **Zero runtime dependencies.** No web framework, matching the engine's $0/stdlib core (ADR-0007).
  Any ASGI server runs it (``uvicorn racore.service.asgi:app``-style) — the server is a deploy
  choice, never a package dependency. That is the open-core line in microcosm: the engine is pure
  and embeddable; the managed operation around it is the paid part.
* **Offline-testable.** Because it is just an async callable, tests drive it through the protocol
  with fake ``receive``/``send`` — no socket, no client library — exactly the injected-fake pattern
  used for every provider adapter.

Routes (all JSON in/out except the answer stream):

* ``GET  /health``  → liveness.
* ``POST /ingest``  → ingest a batch of documents into a tenant's corpus.
* ``POST /answer``  → **Server-Sent Events**: ``token`` events stream the text for a fast
  time-to-first-token, then one ``done`` event carries the citations + grounding a beat later —
  the ADR-0020 "text flows, trust badges fill in" shape, made real by the streamable ``Answer``.
* ``GET  /memory``  / ``POST /memory`` → read / teach per-user memories.

Tenant and user are explicit request fields; the service enforces isolation, so the transport
stays a dumb mapper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qs

from racore.service.core import ServiceError
from racore.service.types import (
    AnswerRequest,
    DocumentInput,
    IngestRequest,
    MemoryReadRequest,
    MemoryWriteRequest,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, MutableMapping

    from racore.core.types import Answer, IngestReport, MemoryItem
    from racore.service.core import RaCoreService

    # The minimal slice of the ASGI spec this surface needs. Aliased here so the handlers read
    # against names, not raw mappings; ASGI servers pass plain dicts that satisfy these.
    Scope = MutableMapping[str, Any]
    Message = MutableMapping[str, Any]
    Receive = Callable[[], Awaitable[Message]]
    Send = Callable[[Message], Awaitable[None]]
    Handler = Callable[[Scope, Receive, Send], Awaitable[None]]

_JSON_HEADER = (b"content-type", b"application/json")
_SSE_HEADERS = [
    (b"content-type", b"text/event-stream"),
    (b"cache-control", b"no-cache"),
]

# The canonical (method, path) -> handler-method-name table, in one place. ``__call__`` dispatches
# off it and ``ROUTES`` exposes its keys, so the documented surface (``docs/openapi.yaml``) and the
# served surface are checked against a single source by the contract test (ADR-0034) — neither can
# drift without the gate noticing.
_ROUTES: dict[tuple[str, str], str] = {
    ("GET", "/health"): "_health",
    ("POST", "/ingest"): "_ingest",
    ("POST", "/answer"): "_answer",
    ("GET", "/memory"): "_read_memory",
    ("POST", "/memory"): "_write_memory",
}
ROUTES: frozenset[tuple[str, str]] = frozenset(_ROUTES)


@dataclass(frozen=True, slots=True)
class ASGIApplication:
    """An ASGI app bound to one service. Stateless beyond the service, so it is safe to share."""

    service: RaCoreService

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope_type = scope["type"]
        if scope_type == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope_type != "http":
            return  # websockets etc. are out of scope for this surface.

        method: str = scope["method"]
        path: str = scope["path"]
        if len(path) > 1:
            path = path.rstrip("/")

        handler_name = _ROUTES.get((method, path))
        if handler_name is not None:
            handler = cast("Handler", getattr(self, handler_name))
            await handler(scope, receive, send)
        elif any(known_path == path for _, known_path in _ROUTES):
            await _error(send, 405, "method_not_allowed", f"{method} is not allowed on {path}")
        else:
            await _error(send, 404, "not_found", f"no route for {path}")

    # --- route handlers -----------------------------------------------------------

    async def _health(self, scope: Scope, receive: Receive, send: Send) -> None:
        await _json(send, 200, {"status": "ok"})

    async def _ingest(self, scope: Scope, receive: Receive, send: Send) -> None:
        payload = await _json_body(receive, send)
        if payload is None:
            return
        try:
            documents = tuple(
                DocumentInput(text=str(doc["text"]), source=str(doc["source"]))
                for doc in payload["documents"]
            )
            request = IngestRequest(
                tenant_id=str(payload["tenant_id"]),
                documents=documents,
                prune=bool(payload.get("prune", False)),
            )
        except (KeyError, TypeError) as exc:
            await _error(send, 400, "invalid_request", f"malformed ingest request: {exc}")
            return
        report = await self.service.ingest(request)
        await _json(send, 200, _ingest_json(report))

    async def _answer(self, scope: Scope, receive: Receive, send: Send) -> None:
        payload = await _json_body(receive, send)
        if payload is None:
            return
        try:
            user = payload.get("user_id")
            request = AnswerRequest(
                text=str(payload["text"]),
                tenant_id=str(payload.get("tenant_id", "default")),
                user_id=None if user is None else str(user),
                k=int(payload.get("k", 5)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            await _error(send, 400, "invalid_request", f"malformed answer request: {exc}")
            return

        answer = await self.service.answer(request)
        # Stream the text first (fast TTFT), then deliver the grounding metadata in a final event —
        # the answer flows, the citation chips fill in a beat later (ADR-0020 §4).
        await send({"type": "http.response.start", "status": 200, "headers": _SSE_HEADERS})
        async for delta in answer.stream():
            await _sse(send, "token", {"text": delta}, more=True)
        await _sse(send, "done", _answer_json(answer), more=False)

    async def _read_memory(self, scope: Scope, receive: Receive, send: Send) -> None:
        params = parse_qs(bytes(scope.get("query_string", b"")).decode())
        tenant = _first(params, "tenant_id")
        user = _first(params, "user_id")
        if not tenant or not user:
            await _error(send, 400, "invalid_request", "tenant_id and user_id are required")
            return
        try:
            k = int(_first(params, "k") or "5")
        except ValueError:
            await _error(send, 400, "invalid_request", "k must be an integer")
            return
        request = MemoryReadRequest(tenant_id=tenant, user_id=user, query=_first(params, "q"), k=k)
        try:
            items = await self.service.read_memory(request)
        except ServiceError as exc:
            await _error(send, 400, exc.code, str(exc))
            return
        await _json(send, 200, {"items": [_memory_json(item) for item in items]})

    async def _write_memory(self, scope: Scope, receive: Receive, send: Send) -> None:
        payload = await _json_body(receive, send)
        if payload is None:
            return
        try:
            request = MemoryWriteRequest(
                tenant_id=str(payload["tenant_id"]),
                user_id=str(payload["user_id"]),
                text=str(payload["text"]),
            )
        except (KeyError, TypeError) as exc:
            await _error(send, 400, "invalid_request", f"malformed memory request: {exc}")
            return
        try:
            items = await self.service.write_memory(request)
        except ServiceError as exc:
            await _error(send, 400, exc.code, str(exc))
            return
        await _json(send, 200, {"stored": [_memory_json(item) for item in items]})

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        """Answer the ASGI lifespan protocol so a real server's startup/shutdown doesn't hang.

        There is nothing to warm up or tear down — the service holds no connections — so this just
        acknowledges each phase."""
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return


def create_app(service: RaCoreService) -> ASGIApplication:
    """Build the ASGI app for a configured service — the factory an ASGI server points at."""
    return ASGIApplication(service=service)


# --- protocol helpers -------------------------------------------------------------


async def _read_body(receive: Receive) -> bytes:
    chunks: list[bytes] = []
    more = True
    while more:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        chunks.append(bytes(message.get("body", b"")))
        more = bool(message.get("more_body", False))
    return b"".join(chunks)


async def _json_body(receive: Receive, send: Send) -> dict[str, Any] | None:
    """Read and parse a JSON object body, or send a 400 and return ``None``."""
    raw = await _read_body(receive)
    if not raw:
        await _error(send, 400, "invalid_json", "request body is empty")
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        await _error(send, 400, "invalid_json", str(exc))
        return None
    if not isinstance(parsed, dict):
        await _error(send, 400, "invalid_json", "request body must be a JSON object")
        return None
    return parsed


async def _json(send: Send, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode()
    headers = [_JSON_HEADER, (b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _error(send: Send, status: int, code: str, message: str) -> None:
    await _json(send, status, {"error": {"code": code, "message": message}})


async def _sse(send: Send, event: str, data: dict[str, Any], *, more: bool) -> None:
    # JSON-encode the payload so a token containing a newline can't break SSE framing.
    body = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
    await send({"type": "http.response.body", "body": body, "more_body": more})


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key)
    return values[0] if values else ""


# --- serializers (core objects -> wire JSON) --------------------------------------


def _ingest_json(report: IngestReport) -> dict[str, Any]:
    return {
        "documents": report.documents,
        "chunks": report.chunks,
        "added": report.added,
        "unchanged": report.unchanged,
        "deleted": report.deleted,
        "timings": [{"stage": t.stage, "ms": t.millis} for t in report.timings],
        "total_ms": report.total_millis,
    }


def _answer_json(answer: Answer) -> dict[str, Any]:
    return {
        "text": answer.text,
        "abstained": answer.abstained,
        "citations": [
            {
                "marker": c.marker,
                "source": c.evidence.source,
                "quote": c.evidence.quote,
                "doc_id": c.evidence.doc_id,
            }
            for c in answer.citations
        ],
        "grounding": {
            "faithfulness": answer.grounding.faithfulness,
            "citation_correctness": answer.grounding.citation_correctness,
            "is_grounded": answer.grounding.is_grounded,
            "unsupported_claims": list(answer.grounding.unsupported_claims),
        },
        "timings": [{"stage": t.stage, "ms": t.millis} for t in answer.timings],
        "total_ms": answer.total_millis,
    }


def _memory_json(item: MemoryItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "kind": item.kind.value,
        "content": item.content,
        "key": item.key,
        "salience": item.salience,
        "source": item.source,
    }
