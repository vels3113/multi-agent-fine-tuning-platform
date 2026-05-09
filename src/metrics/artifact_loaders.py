"""Load frozen demo-shaped artifacts (CSV/JSON) for platform insights."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

PEAK_HBM_TB_S = 5.2


def mean_gpu_util_from_smi_csv(path: str | Path) -> float | None:
    """Mean utilization from a CSV containing ``gpu_utilization_pct`` or ``GPU utilization %``."""
    path = Path(path)
    if not path.is_file():
        return None
    vals: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        cols = {h.strip().lower(): h for h in reader.fieldnames}
        key = None
        for candidate in ("gpu_utilization_pct", "gpu util %", "gpu utilization %"):
            if candidate in cols:
                key = cols[candidate]
                break
        if key is None:
            # fallback: first numeric column
            key = reader.fieldnames[0]
        for row in reader:
            raw = row.get(key, "").strip().rstrip("%")
            if raw == "":
                continue
            try:
                vals.append(float(raw))
            except ValueError:
                continue
    return round(sum(vals) / len(vals), 4) if vals else None


def _float_from_obj(obj: dict[str, Any]) -> float | None:
    for k in (
        "hbm_bandwidth_utilization_pct",
        "hbw_utilization_pct",
        "hbm_utilization_pct",
        "bandwidth_utilization_pct",
    ):
        if k in obj and obj[k] is not None:
            try:
                return float(obj[k])
            except (TypeError, ValueError):
                pass
    # achieved TB/s vs peak
    ach = obj.get("hbm_achieved_tbps") or obj.get("achieved_tbps")
    if ach is not None:
        try:
            return float(ach) / PEAK_HBM_TB_S * 100.0
        except (TypeError, ValueError):
            pass
    return None


def load_hbm_utilization_pct(json_path: str | Path) -> float | None:
    """Load B2-style utilization % from ``hardware_counters.json`` (flexible shapes)."""
    json_path = Path(json_path)
    if not json_path.is_file():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    vals: list[float] = []

    if isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            for s in data["samples"]:
                if isinstance(s, dict):
                    v = _float_from_obj(s)
                    if v is not None:
                        vals.append(v)
        else:
            v = _float_from_obj(data)
            if v is not None:
                vals.append(v)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                v = _float_from_obj(item)
                if v is not None:
                    vals.append(v)

    return round(sum(vals) / len(vals), 4) if vals else None


def forward_backward_pct_from_kernel_csv(path: str | Path) -> tuple[float | None, float | None]:
    """Sum kernel durations by forward vs backward name heuristic (B3)."""
    path = Path(path)
    if not path.is_file():
        return None, None

    fwd_ns = 0.0
    bwd_ns = 0.0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None, None
        fn_lower = [h.lower() for h in reader.fieldnames]

        def pick(*names: str) -> str | None:
            for want in names:
                for orig in reader.fieldnames:
                    if orig.lower() == want.lower():
                        return orig
            return None

        name_col = pick("kernelname", "name", "kernel")
        dur_col = pick("totaldurationns", "total_duration_ns", "duration_ns", "duration")
        if name_col is None or dur_col is None:
            return None, None

        for row in reader:
            name = (row.get(name_col) or "").lower()
            raw_d = row.get(dur_col, "")
            try:
                dur = float(raw_d)
            except (TypeError, ValueError):
                continue
            if "backward" in name or "bwd" in name:
                bwd_ns += dur
            elif "forward" in name or "fwd" in name:
                fwd_ns += dur

    tot = fwd_ns + bwd_ns
    if tot <= 0:
        return None, None
    return round(100.0 * fwd_ns / tot, 2), round(100.0 * bwd_ns / tot, 2)


def find_first_file(directory: str | Path, suffixes: tuple[str, ...]) -> Path | None:
    """Return first file matching suffix under directory (non-recursive)."""
    d = Path(directory)
    if not d.is_dir():
        return None
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in suffixes:
            return p
    return None
