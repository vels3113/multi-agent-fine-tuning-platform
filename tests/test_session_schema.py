"""Tests for session_schema.json — baseline, training, and combined sessions all validate."""
import json
import os
import pytest

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "session_schema.json")

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def _load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def _runtime():
    return {"hostname": "h", "num_gpus": 1, "peak_gpu_memory_mb": None,
            "gpu_utilization_pct": None, "total_duration_sec": None,
            "latest_checkpoint": None, "wandb_run_id": None}


def _baseline_only_session():
    return {
        "session_id": "00000000-0000-4000-8000-000000000000",
        "timestamp": "2026-05-08T00:00:00Z",
        "user": None,
        "stage": {"baseline": True, "training": False},
        "config": {
            "baseline": {
                "model": "Qwen/Qwen3-1.7B",
                "dataset": "openai/openai_humaneval",
                "num_problems": 164,
                "num_runs": 5,
                "batch_size": 16,
                "max_new_tokens": 2048,
                "stop_sequences": ["\ndef "],
                "seed": 42,
            }
        },
        "metrics": {"test_pass_rate": 0.073, "num_problems": 164, "num_runs": 5},
        "runtime": {**_runtime(), "peak_gpu_memory_mb": 4000.0,
                    "gpu_utilization_pct": 100, "total_duration_sec": 2720.0},
    }


def _training_only_session():
    return {
        "session_id": "00000000-0000-4000-8000-000000000001",
        "timestamp": "2026-05-08T01:00:00Z",
        "user": None,
        "stage": {"baseline": False, "training": True},
        "config": {
            "training": {
                "model": "Qwen/Qwen3-1.7B",
                "seed": 42,
                "num_agents": 2,
                "num_train_epochs": 1,
                "reward_func": "dummy",
                "model_params": {"enable_thinking": False, "joint_mode": "aligned"},
                "dataset": {"type": "inline", "prompts": ["Write a function."]},
            }
        },
        "metrics": {},
        "runtime": _runtime(),
    }


def _combined_session():
    return {
        "session_id": "00000000-0000-4000-8000-000000000002",
        "timestamp": "2026-05-08T02:00:00Z",
        "user": None,
        "stage": {"baseline": True, "training": True},
        "config": {
            "baseline": {
                "model": "Qwen/Qwen3-1.7B",
                "seed": 42,
                "num_runs": 5,
            },
            "training": {
                "model": "Qwen/Qwen3-1.7B",
                "seed": 42,
                "num_agents": 2,
                "model_params": {"enable_thinking": False, "joint_mode": "aligned"},
            },
        },
        "metrics": {},
        "runtime": _runtime(),
    }


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_baseline_only_session_validates():
    jsonschema.validate(_baseline_only_session(), _load_schema())


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_training_only_session_validates():
    jsonschema.validate(_training_only_session(), _load_schema())


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_combined_session_validates():
    jsonschema.validate(_combined_session(), _load_schema())


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_config_baseline_missing_model_fails():
    schema = _load_schema()
    session = _baseline_only_session()
    del session["config"]["baseline"]["model"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(session, schema)


def test_p2b_smoke_config_scope_locks():
    """P2b smoke config must satisfy all RULES.md scope locks."""
    import yaml, os
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "demo", "configs", "p2b-smoke.yaml"
    )
    # File doesn't exist yet — this test will fail until Task 1 is complete
    assert os.path.exists(config_path), "demo/configs/p2b-smoke.yaml not created yet"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["num_agents"] == 2
    assert cfg["model_params"]["enable_thinking"] is False
    assert cfg["model_params"]["joint_mode"] == "aligned"
    assert cfg["model_params"]["num_turns"] == 1
    assert cfg["model"] == "Qwen/Qwen3-1.7B"


def test_run_py_stores_cfg_under_training_key():
    """Session created by run.py stores cfg under config['training'], not at top level."""
    training_cfg = {
        "seed": 42,
        "model": "Qwen/Qwen3-1.7B",
        "num_agents": 2,
        "num_train_epochs": 1,
        "reward_func": "dummy",
        "model_params": {"enable_thinking": False, "joint_mode": "aligned"},
        "dataset": {"type": "inline", "prompts": ["fn"]},
    }
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.session.session import Session
    s = Session.start({"training": training_cfg}, stage={"baseline": False, "training": True})
    assert "training" in s.config
    assert s.config["training"]["model_params"]["joint_mode"] == "aligned"
    assert "baseline" not in s.config


def test_runtime_has_wandb_run_id_field():
    """P3a: wandb_run_id must be present in runtime schema and Session.start()."""
    import json, os
    schema_path = os.path.join(os.path.dirname(__file__), "..", "session_schema.json")
    with open(schema_path) as f:
        schema = json.load(f)
    runtime_props = schema["properties"]["runtime"]["properties"]
    assert "wandb_run_id" in runtime_props
    assert runtime_props["wandb_run_id"]["type"] == ["string", "null"]


def test_session_start_includes_wandb_run_id():
    from src.session.session import Session
    s = Session.start(config={"training": {"model": "test", "seed": 0}},
                      stage={"baseline": False, "training": True})
    assert "wandb_run_id" in s.runtime
    assert s.runtime["wandb_run_id"] is None
