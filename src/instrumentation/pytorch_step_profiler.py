"""Bounded PyTorch profiler: full-step context, aggregate forward/backward (or fallback) only."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

TRACING_VERSION = "1.0.0"


def should_profile_step(step: int, profile_every_n: int) -> bool:
    """Profile step 1 and every Nth step (10 -> 1,10,20,...)."""
    if profile_every_n <= 0:
        return False
    return step == 1 or (step % profile_every_n == 0)


def _event_self_cuda_us(ev: Any) -> float:
    t = getattr(ev, "self_cuda_time_total", 0) or 0
    return float(t)


def _event_self_cpu_us(ev: Any) -> float:
    t = getattr(ev, "self_cpu_time_total", 0) or 0
    return float(t)


def parse_profiler_key_averages(key_averages) -> dict[str, Any]:
    """Derive ms aggregates from profiler.key_averages() (training-agnostic heuristics)."""
    forward_us = 0.0
    backward_us = 0.0
    cuda_total_us = 0.0
    keyword_hit = False
    top_name: str | None = None
    top_cuda_us = 0.0

    for ev in key_averages:
        name = str(getattr(ev, "key", "") or "")
        lname = name.lower()
        sc = _event_self_cuda_us(ev)
        cuda_total_us += sc

        if "backward" in lname:
            backward_us += sc
            keyword_hit = True
        elif "forward" in lname:
            forward_us += sc
            keyword_hit = True

        if sc > top_cuda_us:
            top_cuda_us = sc
            top_name = name or None

    cuda_total_ms = round(cuda_total_us / 1000.0, 6) if cuda_total_us else None
    note: str | None = None
    if keyword_hit:
        forward_ms = round(forward_us / 1000.0, 6)
        backward_ms = round(backward_us / 1000.0, 6)
    elif cuda_total_us > 0:
        forward_ms, backward_ms = None, None
        note = "forward_backward_unavailable"
    else:
        forward_ms, backward_ms = None, None
        cpu_total_us = sum(_event_self_cpu_us(ev) for ev in key_averages)
        if cpu_total_us > 0:
            note = "cpu_only"
        cuda_total_ms = None

    top_op_ms = round(top_cuda_us / 1000.0, 6) if top_cuda_us and top_name else None

    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "cuda_total_ms": cuda_total_ms,
        "top_op_name": top_name,
        "top_op_ms": top_op_ms,
        "profiler_note": note,
    }


@dataclass
class PytorchStepProfiler:
    """Starts profiler before rollout, stops after loss, for one step only."""

    profile_every_n: int
    export_chrome_trace: bool = False
    chrome_trace_path: str | None = None

    _active: Any = None
    _profiled_step: int | None = None

    def profile_this_step(self, step: int) -> bool:
        return should_profile_step(step, self.profile_every_n)

    def on_rollout_start(self, step: int) -> None:
        if not self.profile_this_step(step):
            return
        import torch

        activities: list = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        kwargs: dict[str, Any] = {
            "activities": activities,
            "record_shapes": False,
            "profile_memory": False,
            "with_stack": False,
        }

        prof = torch.profiler.profile(**kwargs)
        prof.__enter__()
        self._active = prof
        self._profiled_step = step

    def on_loss_end(self, step: int) -> dict[str, Any] | None:
        if self._active is None or self._profiled_step != step:
            return None
        prof = self._active
        try:
            prof.__exit__(None, None, None)
            if self.export_chrome_trace and self.chrome_trace_path:
                try:
                    prof.export_chrome_trace(self.chrome_trace_path)
                except Exception as exc:
                    logger.warning("export_chrome_trace failed: %s", exc)
            aggregates = parse_profiler_key_averages(prof.key_averages())
            aggregates["profiled"] = True
            return aggregates
        except Exception as exc:
            logger.warning("Profiler teardown failed: %s", exc)
            return {
                "forward_ms": None,
                "backward_ms": None,
                "cuda_total_ms": None,
                "top_op_name": None,
                "top_op_ms": None,
                "profiler_note": "profiler_error",
                "profiled": True,
            }
        finally:
            self._active = None
            self._profiled_step = None

    def noop_loss_aggregates(self) -> dict[str, Any]:
        return {
            "forward_ms": None,
            "backward_ms": None,
            "cuda_total_ms": None,
            "top_op_name": None,
            "top_op_ms": None,
            "profiler_note": None,
            "profiled": False,
        }


def step_tracing_disabled_by_env() -> bool:
    import os

    v = os.environ.get("STEP_TRACING", "").strip().lower()
    return v in ("0", "false", "off", "no")
