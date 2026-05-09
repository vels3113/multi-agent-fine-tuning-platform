"""P3c: platform insights summary validates against artifacts/P3c schema."""
import json
import os

import pytest

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

from src.metrics.platform_insights_builder import build_platform_insights

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SCHEMA_PATH = os.path.join(ROOT, "artifacts", "P3c", "platform_insights_summary.schema.json")


@pytest.fixture
def tiny_steps_jsonl(tmp_path):
    p = tmp_path / "steps.jsonl"
    rows = [
        {
            "step": 1,
            "tracing_version": "1.0.0",
            "profiled": True,
            "forward_ms": 10.0,
            "backward_ms": 30.0,
            "cuda_total_ms": None,
            "timer_rollout_wall_ms": 80.0,
            "timer_loss_compute_wall_ms": 120.0,
        },
        {
            "step": 2,
            "tracing_version": "1.0.0",
            "profiled": True,
            "forward_ms": None,
            "backward_ms": None,
            "cuda_total_ms": 95.0,
            "timer_rollout_wall_ms": 70.0,
            "timer_loss_compute_wall_ms": 110.0,
        },
    ]
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


@pytest.fixture
def rocprof_fixtures(tmp_path):
    csv_path = tmp_path / "kernel_timeline.csv"
    csv_path.write_text(
        "KernelName,TotalDurationNs\n"
        "fwd_attention,100\n"
        "backward_attention,300\n",
        encoding="utf-8",
    )
    hw_path = tmp_path / "hardware_counters.json"
    hw_path.write_text(
        json.dumps({"hbm_bandwidth_utilization_pct": 40.0}),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def smi_csv(tmp_path):
    p = tmp_path / "smi.csv"
    p.write_text(
        "gpu_utilization_pct\n"
        "70.0\n"
        "72.0\n",
        encoding="utf-8",
    )
    return str(p)


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_summary_validates_schema(tiny_steps_jsonl, rocprof_fixtures, smi_csv):
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    summary = build_platform_insights(
        steps_jsonl=tiny_steps_jsonl,
        rocprof_dir=str(rocprof_fixtures),
        smi_csv=smi_csv,
    )
    jsonschema.validate(summary, schema)

    assert summary["b1_gpu_util_pct_mean"] == pytest.approx(71.0)
    assert summary["b2_hbm_bandwidth_utilization_pct"] == pytest.approx(40.0)
    assert summary["b3_forward_pct"] == pytest.approx(25.0)
    assert summary["b3_backward_pct"] == pytest.approx(75.0)
    assert summary["d2_hbm_headroom_pct"] == pytest.approx(60.0)
    assert summary["d1"]["compute_ms_mean"] is not None


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_summary_partial_when_missing_rocprof(tiny_steps_jsonl, smi_csv):
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    summary = build_platform_insights(
        steps_jsonl=tiny_steps_jsonl,
        rocprof_dir=None,
        smi_csv=smi_csv,
    )
    jsonschema.validate(summary, schema)
    assert summary["b2_hbm_bandwidth_utilization_pct"] is None
    assert summary["b3_forward_pct"] is None


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_summary_sample_fixture_if_present():
    sample = os.path.join(ROOT, "artifacts", "P3c", "sample-exports", "summary.sample.json")
    if not os.path.isfile(sample):
        pytest.skip("no committed sample export")
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    data = json.loads(open(sample, encoding="utf-8").read())
    jsonschema.validate(data, schema)
