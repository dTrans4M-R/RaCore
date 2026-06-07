"""The ASGI surface, driven through the raw protocol — no server, no HTTP client (slice 2).

Each test builds a scope, feeds the body through a fake ``receive``, and collects the app's
``send`` messages, then asserts on the wire bytes. This proves the transport maps cleanly onto the
service *and* that the tenant/user isolation the facade enforces survives the HTTP round-trip. The
answer route is checked for its SSE shape: token events reconstruct the text, a final ``done`` event
carries the grounding.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from racore.service.asgi import create_app
from racore.service.core import demo_service

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path

    from racore.service.asgi import ASGIApplication


async def _request(
    app: ASGIApplication,
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    raw: bytes | None = None,
    query: dict[str, str] | None = None,
) -> tuple[int, dict[bytes, bytes], bytes]:
    """Call the app once and return ``(status, headers, body_bytes)``."""
    if raw is not None:
        body = raw
    elif json_body is not None:
        body = json.dumps(json_body).encode()
    else:
        body = b""
    scope: MutableMapping[str, Any] = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": urlencode(query or {}).encode(),
        "headers": [],
    }
    sent: list[MutableMapping[str, Any]] = []

    async def receive() -> MutableMapping[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    await app(scope, receive, send)

    start = next(m for m in sent if m["type"] == "http.response.start")
    headers: dict[bytes, bytes] = {key: value for key, value in start["headers"]}
    out = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return int(start["status"]), headers, out


def _json(body: bytes) -> Any:
    return json.loads(body)


def _parse_sse(body: bytes) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.decode().split("\n\n"):
        if not block.strip():
            continue
        name = ""
        data = "{}"
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        events.append((name, json.loads(data)))
    return events


def _app(tmp_path: Path) -> ASGIApplication:
    return create_app(demo_service(tmp_path))


def test_health(tmp_path: Path) -> None:
    async def go() -> None:
        status, _headers, body = await _request(_app(tmp_path), "GET", "/health")
        assert status == 200
        assert _json(body) == {"status": "ok"}

    asyncio.run(go())


def test_ingest_then_answer_streams_sse(tmp_path: Path) -> None:
    async def go() -> None:
        app = _app(tmp_path)
        status, _h, body = await _request(
            app,
            "POST",
            "/ingest",
            json_body={
                "tenant_id": "acme",
                "documents": [{"source": "acme/eiffel", "text": "The Eiffel Tower is in Paris."}],
            },
        )
        assert status == 200
        assert _json(body)["added"] == 1

        status, headers, body = await _request(
            app,
            "POST",
            "/answer",
            json_body={"text": "Where is the Eiffel Tower?", "tenant_id": "acme"},
        )
        assert status == 200
        assert headers[b"content-type"] == b"text/event-stream"

        events = _parse_sse(body)
        streamed = "".join(data["text"] for name, data in events if name == "token")
        done = next(data for name, data in events if name == "done")
        assert streamed == done["text"]  # the token stream reconstructs the full answer
        assert "Paris" in done["text"]
        assert done["citations"]  # grounding arrives in the final event
        assert not done["abstained"]

    asyncio.run(go())


def test_answer_is_tenant_isolated_over_http(tmp_path: Path) -> None:
    async def go() -> None:
        app = _app(tmp_path)
        await _request(
            app,
            "POST",
            "/ingest",
            json_body={
                "tenant_id": "acme",
                "documents": [{"source": "acme/eiffel", "text": "The Eiffel Tower is in Paris."}],
            },
        )
        # A different tenant has an empty corpus, so the same question must abstain — no Paris leak.
        _status, _h, body = await _request(
            app,
            "POST",
            "/answer",
            json_body={"text": "Where is the Eiffel Tower?", "tenant_id": "globex"},
        )
        done = next(data for name, data in _parse_sse(body) if name == "done")
        assert done["abstained"]
        assert "Paris" not in done["text"]

    asyncio.run(go())


def test_memory_roundtrip_over_http(tmp_path: Path) -> None:
    async def go() -> None:
        app = _app(tmp_path)
        status, _h, body = await _request(
            app,
            "POST",
            "/memory",
            json_body={"tenant_id": "t", "user_id": "u", "text": "My name is Bob."},
        )
        assert status == 200
        assert any("Bob" in item["content"] for item in _json(body)["stored"])

        # The owning user reads it back...
        _s, _hh, mine = await _request(
            app, "GET", "/memory", query={"tenant_id": "t", "user_id": "u", "q": "name"}
        )
        assert any("Bob" in item["content"] for item in _json(mine)["items"])

        # ...a different user in the same tenant sees nothing.
        _s2, _h2, other = await _request(
            app, "GET", "/memory", query={"tenant_id": "t", "user_id": "other", "q": "name"}
        )
        assert _json(other)["items"] == []

    asyncio.run(go())


def test_written_memory_personalizes_an_answer_over_http(tmp_path: Path) -> None:
    async def go() -> None:
        app = _app(tmp_path)
        await _request(
            app,
            "POST",
            "/memory",
            json_body={"tenant_id": "t", "user_id": "u", "text": "My name is Bob."},
        )
        _s, _h, body = await _request(
            app,
            "POST",
            "/answer",
            json_body={"text": "What is my name?", "tenant_id": "t", "user_id": "u"},
        )
        done = next(data for name, data in _parse_sse(body) if name == "done")
        assert "Bob" in done["text"]
        assert any(c["source"].startswith("memory/") for c in done["citations"])

    asyncio.run(go())


def test_unknown_route_is_404(tmp_path: Path) -> None:
    async def go() -> None:
        status, _h, body = await _request(_app(tmp_path), "GET", "/nope")
        assert status == 404
        assert _json(body)["error"]["code"] == "not_found"

    asyncio.run(go())


def test_wrong_method_is_405(tmp_path: Path) -> None:
    async def go() -> None:
        status, _h, body = await _request(_app(tmp_path), "GET", "/ingest")
        assert status == 405
        assert _json(body)["error"]["code"] == "method_not_allowed"

    asyncio.run(go())


def test_bad_json_is_400(tmp_path: Path) -> None:
    async def go() -> None:
        status, _h, body = await _request(_app(tmp_path), "POST", "/answer", raw=b"{not json")
        assert status == 400
        assert _json(body)["error"]["code"] == "invalid_json"

    asyncio.run(go())
