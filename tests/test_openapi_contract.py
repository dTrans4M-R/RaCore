"""The OpenAPI document must describe exactly the surface the ASGI app serves (ADR-0034).

A documented API is only worth the bytes if it can't drift from the code. This loads the
hand-authored ``docs/openapi.yaml`` and asserts its ``(method, path)`` set equals
``asgi.ROUTES`` — the same table ``ASGIApplication`` dispatches off — in *both* directions, so a
route added or removed without a matching doc change fails the gate. A few structural checks guard
the response contract the UI / client generators depend on (the SSE media type on ``/answer``, a
machine-readable error on every fallible route, and no dangling ``$ref``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from racore.service.asgi import ROUTES

if TYPE_CHECKING:
    from collections.abc import Iterator

_SPEC_PATH = Path(__file__).resolve().parents[1] / "docs" / "openapi.yaml"


def _spec() -> dict[str, Any]:
    loaded = yaml.safe_load(_SPEC_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _documented_routes(spec: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (method.upper(), path)
        for path, operations in spec["paths"].items()
        for method in operations
    }


def _iter_refs(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                yield value
            else:
                yield from _iter_refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_refs(item)


def test_documented_surface_matches_served_surface() -> None:
    assert _documented_routes(_spec()) == set(ROUTES)


def test_answer_route_streams_server_sent_events() -> None:
    answer = _spec()["paths"]["/answer"]["post"]
    assert "text/event-stream" in answer["responses"]["200"]["content"]


def test_every_fallible_route_documents_a_machine_readable_error() -> None:
    spec = _spec()
    for method, path in ROUTES:
        if path == "/health":
            continue  # liveness can't fail on a client error.
        operation = spec["paths"][path][method.lower()]
        assert "400" in operation["responses"], f"{method} {path} lacks a documented 400"


def test_all_refs_resolve() -> None:
    spec = _spec()
    schemas = spec["components"]["schemas"]
    responses = spec["components"].get("responses", {})
    for ref in _iter_refs(spec):
        section, name = ref.rsplit("/", 2)[-2:]
        pool = schemas if section == "schemas" else responses
        assert name in pool, f"dangling $ref: {ref}"
