"""Merge generation_stats.json + pass_at_k.json into baseline_metrics.json.

Can be run standalone or imported by the dashboard data preparation pipeline.

Usage:
    python -m baseline.merge_metrics <artifacts_dir>
    python -m baseline.merge_metrics  # defaults to artifacts/P1b relative to platform root
"""
import json
import os
import sys


def merge(artifacts_dir: str) -> dict:
    with open(os.path.join(artifacts_dir, "generation_stats.json")) as f:
        stats = json.load(f)
    with open(os.path.join(artifacts_dir, "pass_at_k.json")) as f:
        pass_at_k = json.load(f)

    metrics = {
        "test_pass_rate": pass_at_k.get("pass@1", 0.0),
        **{k: stats[k] for k in ("syntactic_correctness_ratio", "token_throughput_per_sec",
                                  "num_problems", "num_runs", "model", "dataset", "timestamp")},
    }

    out_path = os.path.join(artifacts_dir, "baseline_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    platform_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(platform_root, "artifacts", "P1b")
    metrics = merge(artifacts_dir)
    print(json.dumps(metrics, indent=2))
