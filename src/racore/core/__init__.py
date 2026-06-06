"""Provider-agnostic core: domain types, port protocols, and the pipeline.

Nothing in this package may import a concrete provider or an adapter — the
dependency arrow points *into* the core only. See ``docs/architecture.md``.
"""

from __future__ import annotations
