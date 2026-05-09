"""Cheap perf_counter-based wall timers for orchestration (CPU) phases."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Generator


def wall_ms_from_delta(seconds: float) -> float:
    return round(seconds * 1000.0, 6)


@dataclass
class StepOrchestrationTimers:
    """Per-step accumulators (reset each step before rollout)."""

    rollout_wall_ms: float = 0.0
    loss_compute_wall_ms: float = 0.0

    def reset(self) -> None:
        self.rollout_wall_ms = 0.0
        self.loss_compute_wall_ms = 0.0

    @contextmanager
    def rollout_scope(self) -> Generator[None, None, None]:
        t0 = perf_counter()
        try:
            yield
        finally:
            self.rollout_wall_ms = wall_ms_from_delta(perf_counter() - t0)

    @contextmanager
    def loss_compute_scope(self) -> Generator[None, None, None]:
        t0 = perf_counter()
        try:
            yield
        finally:
            self.loss_compute_wall_ms = wall_ms_from_delta(perf_counter() - t0)
