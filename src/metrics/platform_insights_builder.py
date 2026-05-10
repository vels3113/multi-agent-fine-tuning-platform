"""Build ``summary.json`` payload for demo/platform_insights."""

from __future__ import annotations

from pathlib import Path

from src.metrics.artifact_loaders import (
    find_first_file,
    forward_backward_pct_from_kernel_csv,
    load_hbm_utilization_pct,
    mean_gpu_util_from_smi_csv,
)
from src.metrics.trace_aggregate import (
    aggregate_d1_from_jsonl,
    jsonl_fwd_bwd_split,
    load_steps_jsonl,
    rollout_vs_loss_hint,
)

SUMMARY_SCHEMA_VERSION = "1.0.0"


def build_platform_insights(
    *,
    steps_jsonl: str | Path,
    rocprof_dir: str | Path | None = None,
    smi_csv: str | Path | None = None,
    hardware_counters_json: str | Path | None = None,
    kernel_timeline_csv: str | Path | None = None,
) -> dict:
    """Aggregate traces + optional rocprof/smi artifacts into a summary dict."""
    steps_jsonl = Path(steps_jsonl)
    rows = load_steps_jsonl(str(steps_jsonl))
    d1, d1_note_parts = aggregate_d1_from_jsonl(rows)

    sj = str(Path(steps_jsonl).resolve())
    source_notes: list[str] = [f"steps_jsonl_path={sj} rows={len(rows)}"]

    rocprof_dir = Path(rocprof_dir) if rocprof_dir else None

    hw_path = Path(hardware_counters_json) if hardware_counters_json else None
    if hw_path is None and rocprof_dir and rocprof_dir.is_dir():
        cand = find_first_file(rocprof_dir, (".json",))
        if cand and "hardware" in cand.name.lower():
            hw_path = cand
        elif cand:
            hw_path = cand

    kt_path = Path(kernel_timeline_csv) if kernel_timeline_csv else None
    if kt_path is None and rocprof_dir and rocprof_dir.is_dir():
        found = find_first_file(rocprof_dir, (".csv",))
        if found and "timeline" in found.name.lower():
            kt_path = found
        elif found:
            kt_path = found

    b2 = load_hbm_utilization_pct(hw_path) if hw_path and hw_path.is_file() else None
    if b2 is None and rocprof_dir:
        source_notes.append("B2: hardware_counters.json missing or unparsed.")

    b3_f, b3_b = (None, None)
    if kt_path and kt_path.is_file():
        b3_f, b3_b = forward_backward_pct_from_kernel_csv(kt_path)
        source_notes.append(f"B3 kernel CSV: {kt_path.name}")
    else:
        source_notes.append("B3: kernel timeline CSV missing — forward/backward split null.")

    smi_path = Path(smi_csv) if smi_csv else None
    b1 = mean_gpu_util_from_smi_csv(smi_path) if smi_path and smi_path.is_file() else None
    if b1 is None:
        source_notes.append("B1: smi CSV missing or unparsed.")

    jf, jb = jsonl_fwd_bwd_split(rows)
    b3_note = None
    if jf is not None and jb is not None and b3_f is not None and b3_b is not None:
        b3_note = (
            f"JSONL corroboration fwd/bwd ~{jf}%/{jb}% vs rocprof kernel heuristic {b3_f}%/{b3_b}%."
        )
    elif jf is not None and jb is not None:
        b3_note = f"JSONL-only fwd/bwd split ~{jf}%/{jb}% (rocprof authoritative when present)."

    d2_h = round(100.0 - b2, 4) if b2 is not None else None
    if d2_h is not None:
        hint = (
            f"HBM headroom ~{d2_h:.1f}% vs 5.2 TB/s peak — batch/seq can grow before saturating memory BW."
        )
    else:
        hint = "HBM headroom unknown without hardware counter snapshot — capture rocprof counters into demo/rocprof/hardware_counters.json."

    b_star = rollout_vs_loss_hint(rows)

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "source_notes": source_notes,
        "b1_gpu_util_pct_mean": b1,
        "b2_hbm_bandwidth_utilization_pct": b2,
        "b3_forward_pct": b3_f,
        "b3_backward_pct": b3_b,
        "b3_corroboration_note": b3_note,
        "d1": d1,
        "d1_notes": " ".join(d1_note_parts).strip(),
        "d2_hbm_headroom_pct": d2_h,
        "next_10x_hint": hint,
        "b_star_data_loading_hint": b_star,
    }
