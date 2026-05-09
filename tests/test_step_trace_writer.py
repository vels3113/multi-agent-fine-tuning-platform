"""Tests for JSONL step trace writer."""
from __future__ import annotations

import json

from src.instrumentation.step_trace_writer import StepTraceWriter


def test_append_creates_file_and_lines(tmp_path):
    p = tmp_path / "s" / "steps.jsonl"
    w = StepTraceWriter(str(p))
    w.append(
        {
            "step": 1,
            "profiled": False,
            "tracing_version": "1.0.0",
            "timer_rollout_wall_ms": 1.0,
            "timer_loss_compute_wall_ms": 2.0,
        }
    )
    w.append(
        {
            "step": 2,
            "profiled": True,
            "tracing_version": "1.0.0",
            "timer_rollout_wall_ms": 3.0,
            "timer_loss_compute_wall_ms": 4.0,
        }
    )
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 1
    assert json.loads(lines[1])["step"] == 2
