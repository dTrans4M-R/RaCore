"""RaCore — a grounded, provider-agnostic retrieval engine.

This package is the provider-agnostic core. Concrete providers live under
``racore.adapters``; the orchestration and domain types live under ``racore.core``.
See ``docs/architecture.md`` for the ports-and-adapters design.
"""

from __future__ import annotations

__version__ = "0.0.0"

__all__ = ["__version__"]
