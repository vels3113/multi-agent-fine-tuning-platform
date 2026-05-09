#!/usr/bin/env python3
"""CLI: convert W&B-style row JSON into demo/wandb/training_metrics.json shape."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics.export_training_metrics import export_training_metrics_from_file


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wandb-rows-json", required=True, help="JSON array of rows or {\"rows\": [...]}")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    export_training_metrics_from_file(args.wandb_rows_json, args.out)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
