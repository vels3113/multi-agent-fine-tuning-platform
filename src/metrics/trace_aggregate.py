"""Aggregate P3b JSONL step traces into D1 components."""

from __future__ import annotations

import json
from typing import Any, Iterable


def _mean(vals: list[float]) -> float | None:
    return round(sum(vals) / len(vals), 6) if vals else None


def load_steps_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def aggregate_d1_from_jsonl(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, float | None], list[str]]:
    """Compute D1 means from timers + profiler fields.

    Returns ``(d1_dict, notes)`` — notes capture partial coverage reasons.
    """
    notes: list[str] = []
    rollout_ms: list[float] = []
    loss_wall_ms: list[float] = []
    compute_vals: list[float] = []

    j_fwd_bwd_cuda = 0
    j_cuda_only = 0

    for row in rows:
        tr = row.get("timer_rollout_wall_ms")
        tl = row.get("timer_loss_compute_wall_ms")
        if tr is not None:
            rollout_ms.append(float(tr))
        if tl is not None:
            loss_wall_ms.append(float(tl))

        fwd = row.get("forward_ms")
        bwd = row.get("backward_ms")
        cuda_t = row.get("cuda_total_ms")
        prof = bool(row.get("profiled"))

        if prof and cuda_t is not None:
            compute_vals.append(float(cuda_t))
            j_cuda_only += 1
        elif prof and fwd is not None and bwd is not None:
            compute_vals.append(float(fwd) + float(bwd))
            j_fwd_bwd_cuda += 1

    wall_totals: list[float] = []
    cpu_overheads: list[float] = []
    for row in rows:
        tr = row.get("timer_rollout_wall_ms")
        tl = row.get("timer_loss_compute_wall_ms")
        if tr is None or tl is None:
            continue
        wall = float(tr) + float(tl)
        wall_totals.append(wall)

        fwd = row.get("forward_ms")
        bwd = row.get("backward_ms")
        cuda_t = row.get("cuda_total_ms")
        prof = bool(row.get("profiled"))
        comp: float | None = None
        if prof and cuda_t is not None:
            comp = float(cuda_t)
        elif prof and fwd is not None and bwd is not None:
            comp = float(fwd) + float(bwd)
        if comp is not None:
            cpu_overheads.append(max(0.0, wall - comp))

    if j_fwd_bwd_cuda > 0 and j_cuda_only > 0:
        notes.append(
            "Mixed cuda_total_ms vs forward_ms+backward_ms across profiled steps; "
            "compute_ms_mean uses whichever is present per step."
        )

    if not rows:
        notes.append("No JSONL rows — D1 timers empty.")

    d1: dict[str, float | None] = {
        "compute_ms_mean": _mean(compute_vals),
        "memory_ms_mean": None,
        "cpu_overhead_ms_mean": _mean(cpu_overheads) if cpu_overheads else None,
        "wall_total_ms_mean": _mean(wall_totals) if wall_totals else None,
        "rollout_wall_ms_mean": _mean(rollout_ms) if rollout_ms else None,
        "loss_compute_wall_ms_mean": _mean(loss_wall_ms) if loss_wall_ms else None,
    }

    if d1["compute_ms_mean"] is None:
        notes.append(
            "compute_ms_mean unavailable (no profiled steps with cuda_total_ms or forward+backward)."
        )
    notes.append(
        "memory_ms_mean requires rocprof memory-bound timing aligned to steps — not inferred from JSONL alone."
    )

    return d1, notes


def rollout_vs_loss_hint(rows: Iterable[dict[str, Any]]) -> str | None:
    """Optional B* hint from timer proportions."""
    rollout_ms: list[float] = []
    loss_wall_ms: list[float] = []
    for row in rows:
        tr = row.get("timer_rollout_wall_ms")
        tl = row.get("timer_loss_compute_wall_ms")
        if tr is None or tl is None:
            continue
        rollout_ms.append(float(tr))
        loss_wall_ms.append(float(tl))
    if not rollout_ms:
        return None
    mr = sum(rollout_ms) / len(rollout_ms)
    ml = sum(loss_wall_ms) / len(loss_wall_ms)
    tot = mr + ml
    if tot <= 0:
        return None
    rpct = round(100.0 * mr / tot, 1)
    return (
        f"Mean rollout wall ~{rpct}% of (rollout+loss) wall — "
        "interpret as generation/data-path vs loss-compute balance (qualitative)."
    )


def jsonl_fwd_bwd_split(rows: Iterable[dict[str, Any]]) -> tuple[float | None, float | None]:
    """Summarize JSONL forward/backward ms where both present (corroboration only)."""
    fwd_sum = 0.0
    bwd_sum = 0.0
    n = 0
    for row in rows:
        if not row.get("profiled"):
            continue
        f = row.get("forward_ms")
        b = row.get("backward_ms")
        if f is None or b is None:
            continue
        fwd_sum += float(f)
        bwd_sum += float(b)
        n += 1
    if n == 0:
        return None, None
    tot = fwd_sum + bwd_sum
    if tot <= 0:
        return None, None
    return round(100.0 * fwd_sum / tot, 2), round(100.0 * bwd_sum / tot, 2)
