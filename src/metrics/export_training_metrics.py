"""Build demo/wandb/training_metrics.json from W&B-style row exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Canonical W&B keys → demo JSON keys (METRIC_CATALOG)
_WANDB_TO_DEMO = {
    "train/task_pass_rate": "test_pass_rate",
    "train/joint_reward": "joint_reward",
    "train/reward_std_across_agents": "reward_std_across_agents",
    "train/syntactic_correctness_ratio": "syntactic_correctness",
    "hardware/gpu_util_pct": "hardware/gpu_util_pct",
    "train/loss": "train/loss",
}


def _step_from_row(row: dict[str, Any]) -> int | None:
    if "_step" in row and row["_step"] is not None:
        return int(row["_step"])
    if "step" in row and row["step"] is not None:
        return int(row["step"])
    return None


def wandb_rows_to_training_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert history rows (each dict may mix scalar keys) into export shape."""
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = _step_from_row(row)
        if step is None:
            continue
        slot = by_step.setdefault(step, {"step": step})
        for wb_key, demo_key in _WANDB_TO_DEMO.items():
            if wb_key in row and row[wb_key] is not None:
                slot[demo_key] = row[wb_key]

    steps = sorted(by_step.values(), key=lambda s: s["step"])
    return {"schema_version": "1.0.0", "wandb_run_id": None, "steps": steps}


def export_training_metrics_from_file(in_path: str, out_path: str) -> None:
    data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, list):
        out = wandb_rows_to_training_metrics(data)
    elif isinstance(data, dict) and "rows" in data:
        out = wandb_rows_to_training_metrics(data["rows"])
    else:
        raise ValueError("Expected JSON array of W&B rows or {\"rows\": [...]}")

    Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
