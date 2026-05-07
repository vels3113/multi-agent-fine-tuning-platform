import sys
import os
import argparse
import yaml
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from comlrl.trainers.reinforce.magrpo import MAGRPOTrainer, MAGRPOConfig

# ── Reward registry ──────────────────────────────────────────────────────────

def _reward_dummy(completions, **kwargs):
    return [1.0] * len(completions)

def _reward_length(completions, **kwargs):
    return [float(len(c)) for c in completions]

REWARD_REGISTRY = {
    "dummy": _reward_dummy,
    "length": _reward_length,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict) -> Dataset:
    ds_cfg = cfg["dataset"]
    if ds_cfg.get("type") == "inline":
        return Dataset.from_dict({"prompt": ds_cfg["prompts"]})
    return load_dataset(ds_cfg["name"], split=ds_cfg.get("split", "train"))


def build_reward_fn(name: str):
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward_func '{name}'. Available: {list(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[name]


def build_tokenizer(model_name: str, model_params: dict):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Silence pad_token_id warning (see P0a findings)
    if hasattr(tokenizer, "eos_token_id"):
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def assert_no_think_tokens(text: str, context: str = ""):
    if "<think>" in text or "</think>" in text:
        raise AssertionError(
            f"Thinking token detected in {context}: {text[:200]!r}"
        )

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])

    model_name = cfg["model"]
    model_params = cfg.get("model_params", {})
    enable_thinking = model_params.get("enable_thinking", False)

    tokenizer = build_tokenizer(model_name, model_params)
    dataset = build_dataset(cfg)
    reward_fn = build_reward_fn(cfg["reward_func"])

    trainer_cfg = MAGRPOConfig(
        num_train_epochs=cfg["num_train_epochs"],
        num_agents=cfg["num_agents"],
        **{k: v for k, v in model_params.items()
           if k not in ("enable_thinking",) and k in MAGRPOConfig.__dataclass_fields__},
    )

    print(f"enable_thinking={enable_thinking} | model={model_name}")

    trainer = MAGRPOTrainer(
        agent_model=model_name,
        num_agents=cfg["num_agents"],
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_func=reward_fn,
        args=trainer_cfg,
    )

    # Apply enable_thinking on all generation paths before training
    if not enable_thinking:
        for attr in ("model", "ref_model"):
            m = getattr(trainer, attr, None)
            if m is not None and hasattr(m, "generation_config"):
                m.generation_config.update({"enable_thinking": False})

    # Wrap trainer to assert no <think> tokens in every rollout batch
    if not enable_thinking:
        _original_step = trainer.training_step
        def _guarded_step(model, inputs):
            result = _original_step(model, inputs)
            for seq in inputs.get("input_ids", []):
                decoded = tokenizer.decode(seq, skip_special_tokens=False)
                assert_no_think_tokens(decoded, context="training rollout")
            return result
        trainer.training_step = _guarded_step

    trainer.train()
    print("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
