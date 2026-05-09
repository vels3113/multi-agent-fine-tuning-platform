"""Tests for orchestration wall timers."""
from __future__ import annotations

import time

from src.instrumentation.orchestration_timers import StepOrchestrationTimers, wall_ms_from_delta


def test_wall_ms_from_delta():
    assert wall_ms_from_delta(0.5) == 500.0


def test_rollout_and_loss_scopes():
    t = StepOrchestrationTimers()
    t.reset()
    with t.rollout_scope():
        time.sleep(0.01)
    with t.loss_compute_scope():
        time.sleep(0.005)
    assert t.rollout_wall_ms >= 5.0
    assert t.loss_compute_wall_ms >= 1.0


def test_reset_clears():
    t = StepOrchestrationTimers()
    with t.rollout_scope():
        pass
    assert t.rollout_wall_ms >= 0
    t.reset()
    assert t.rollout_wall_ms == 0.0
    assert t.loss_compute_wall_ms == 0.0
