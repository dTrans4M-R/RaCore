"""Smoke test: the package imports and exposes a version.

Placeholder until the Phase 0 walking skeleton lands (see ``docs/roadmap.md``).
It exists so the pre-commit / CI gate has something green to run from day one —
the eval harness and end-to-end test replace it in Phase 0.
"""

from __future__ import annotations

import racore


def test_package_exposes_version() -> None:
    assert racore.__version__
