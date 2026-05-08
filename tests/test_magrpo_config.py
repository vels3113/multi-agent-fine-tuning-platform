"""Tests for 2-agent MAGRPOConfig construction from YAML."""
import dataclasses
import pytest
from comlrl.trainers.reinforce.magrpo import MAGRPOConfig

_MAGRPO_FIELDS = {f.name for f in dataclasses.fields(MAGRPOConfig)}

EXAMPLE_CFG = {
    "seed": 42,
    "model": "Qwen/Qwen3-1.7B",
    "num_agents": 2,
    "num_train_epochs": 1,
    "reward_func": "dummy",
    "model_params": {
        "enable_thinking": False,
        "joint_mode": "aligned",
        "num_generations": 4,
        "max_new_tokens": 512,
        "num_turns": 1,
    },
    "dataset": {"type": "inline", "prompts": ["Write a function."]},
}


def _build_trainer_cfg(cfg: dict) -> MAGRPOConfig:
    model_params = cfg.get("model_params", {})
    return MAGRPOConfig(
        num_train_epochs=cfg["num_train_epochs"],
        num_agents=cfg["num_agents"],
        **{k: v for k, v in model_params.items()
           if k not in ("enable_thinking",) and k in _MAGRPO_FIELDS},
    )


def test_num_agents_is_2():
    assert _build_trainer_cfg(EXAMPLE_CFG).num_agents == 2


def test_enable_thinking_not_a_magrpo_field():
    assert "enable_thinking" not in _MAGRPO_FIELDS


def test_joint_mode_forwarded_if_field_exists():
    if "joint_mode" not in _MAGRPO_FIELDS:
        pytest.skip("joint_mode is a MAGRPOTrainer param on this CoMLRL version")
    assert _build_trainer_cfg(EXAMPLE_CFG).joint_mode == "aligned"
