"""The evaluation module — wired on day one, not bolted on (ADR-0005).

Public surface for running the golden-set baseline and scoring a pipeline run.
"""

from __future__ import annotations

from racore.eval.datasets import golden_dataset, golden_source
from racore.eval.harness import HarnessReport, demo_pipeline, run
from racore.eval.metrics import default_evaluators

__all__ = [
    "HarnessReport",
    "default_evaluators",
    "demo_pipeline",
    "golden_dataset",
    "golden_source",
    "run",
]
