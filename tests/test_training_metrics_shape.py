"""P3c: training_metrics export shape vs artifacts/P3c schema."""
import json
import os

import pytest

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

from src.metrics.export_training_metrics import wandb_rows_to_training_metrics
from src.metrics.training_step_extract import infer_task_pass_rate

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SCHEMA_PATH = os.path.join(ROOT, "artifacts", "P3c", "training_metrics.schema.json")


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_wandb_rows_export_validates_schema():
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    rows = [
        {
            "_step": 1,
            "train/loss": 0.5,
            "train/joint_reward": 1.25,
            "train/reward_std_across_agents": 0.05,
            "train/task_pass_rate": 80.0,
            "hardware/gpu_util_pct": 71.2,
        },
        {
            "_step": 2,
            "train/joint_reward": 1.30,
            "train/reward_std_across_agents": 0.07,
        },
    ]
    export = wandb_rows_to_training_metrics(rows)
    jsonschema.validate(export, schema)
    assert len(export["steps"]) == 2
    assert export["steps"][0]["test_pass_rate"] == 80.0


def test_infer_task_pass_rate_from_flags():
    assert infer_task_pass_rate([{"passed": True}, {"passed": False}]) == 50.0


def test_infer_task_pass_rate_ratio_normalized():
    assert infer_task_pass_rate({"test_pass_rate": 0.85}) == 85.0


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_sample_fixture_if_present():
    """Optional committed samples under artifacts/P3c/sample-exports/."""
    sample = os.path.join(ROOT, "artifacts", "P3c", "sample-exports", "training_metrics.sample.json")
    if not os.path.isfile(sample):
        pytest.skip("no committed sample export")
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    data = json.loads(open(sample, encoding="utf-8").read())
    jsonschema.validate(data, schema)
