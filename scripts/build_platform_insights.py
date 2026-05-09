#!/usr/bin/env python3
"""CLI: aggregate JSONL + rocprof/smi artifacts → platform insights summary JSON."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics.platform_insights_builder import build_platform_insights


def main() -> int:
    parser = argparse.ArgumentParser(description="Build demo/platform_insights/summary.json inputs.")
    parser.add_argument("--steps-jsonl", required=True, help="Path to P3b steps.jsonl")
    parser.add_argument("--rocprof-dir", default=None, help="Directory with kernel CSV + hardware JSON")
    parser.add_argument("--smi-csv", default=None, help="Path to gpu utilization CSV")
    parser.add_argument("--hardware-json", default=None, help="Explicit hardware_counters.json path")
    parser.add_argument("--kernel-csv", default=None, help="Explicit kernel_timeline.csv path")
    parser.add_argument("--out", required=True, help="Output summary JSON path")
    args = parser.parse_args()

    summary = build_platform_insights(
        steps_jsonl=args.steps_jsonl,
        rocprof_dir=args.rocprof_dir,
        smi_csv=args.smi_csv,
        hardware_counters_json=args.hardware_json,
        kernel_timeline_csv=args.kernel_csv,
    )
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
