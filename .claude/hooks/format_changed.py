#!/usr/bin/env python
"""PostToolUse hook: auto-format a Python file right after Claude edits it.

Best-effort and non-blocking by design: any failure (ruff not installed yet, no
venv, etc.) is swallowed and the hook exits 0 so it never interrupts editing.
The authoritative gate is pre-commit; this just keeps the working tree tidy.
"""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    path = (data.get("tool_input") or {}).get("file_path")
    if not path or not path.endswith(".py"):
        return 0

    for cmd in (
        ["uv", "run", "ruff", "format", path],
        ["uv", "run", "ruff", "check", "--fix", path],
    ):
        try:
            subprocess.run(cmd, capture_output=True, timeout=60, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return 0  # toolchain not ready — stay silent
    return 0


if __name__ == "__main__":
    sys.exit(main())
