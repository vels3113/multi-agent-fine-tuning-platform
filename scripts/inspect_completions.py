"""
Diagnostic script: intercepts every rollout completion during a run.py training
loop, prints prompt→completion pairs, flags <think> tokens, and writes a full
report to artifacts/P1a/verification-logs/completions-inspection.txt.

No commits, no side effects beyond the log file.
Usage (inside Docker via make or docker-run.sh):
    python scripts/inspect_completions.py --config configs/p1a-thinking-gate.yaml
"""
import sys
import os
import argparse
import dataclasses

import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import load_config, build_dataset, build_reward_fn, build_tokenizer

from comlrl.trainers.reinforce.magrpo import MAGRPOTrainer, MAGRPOConfig

LOG = "artifacts/P1a/verification-logs/completions-inspection.txt"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _has_think(text: str) -> bool:
    return THINK_OPEN in text or THINK_CLOSE in text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p1a-thinking-gate.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])

    model_name = cfg["model"]
    model_params = cfg.get("model_params", {})
    enable_thinking = model_params.get("enable_thinking", False)

    tokenizer = build_tokenizer(model_name, model_params)
    dataset = build_dataset(cfg)
    reward_fn = build_reward_fn(cfg["reward_func"])

    _magrpo_fields = {f.name for f in dataclasses.fields(MAGRPOConfig)}
    trainer_cfg = MAGRPOConfig(
        num_train_epochs=cfg["num_train_epochs"],
        num_agents=cfg["num_agents"],
        **{k: v for k, v in model_params.items()
           if k not in ("enable_thinking",) and k in _magrpo_fields},
    )

    trainer = MAGRPOTrainer(
        agent_model=model_name,
        num_agents=cfg["num_agents"],
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_func=reward_fn,
        args=trainer_cfg,
    )

    if not enable_thinking:
        for attr in ("model", "ref_model"):
            m = getattr(trainer, attr, None)
            if m is not None and hasattr(m, "generation_config"):
                m.generation_config.update({"enable_thinking": False})

    records = []
    think_count = 0
    call_count = 0

    _orig_gen = trainer._generate_completions

    def _capturing_gen(*a, **kw):
        nonlocal think_count, call_count
        result = _orig_gen(*a, **kw)
        call_count += 1
        prompts = result.get("prompts", [])
        completions = result.get("completions", [])
        for i, prompt in enumerate(prompts):
            batch = completions[i] if i < len(completions) else []
            for j, text in enumerate(batch):
                flagged = _has_think(text)
                if flagged:
                    think_count += 1
                records.append({
                    "call": call_count,
                    "prompt_idx": i,
                    "completion_idx": j,
                    "prompt": prompt,
                    "completion": text,
                    "think_detected": flagged,
                })
        return result

    trainer._generate_completions = _capturing_gen

    # Intercept _update_from_samples to stop after generation (no weight update)
    _orig_update = trainer._update_from_samples

    def _noop_update(*a, **kw):
        pass

    trainer._update_from_samples = _noop_update

    print(f"Running with enable_thinking={enable_thinking}, model={model_name}")
    print("(weight updates disabled — generation only)\n")

    trainer.train()

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    lines = []
    lines.append(f"Completions Inspection Report")
    lines.append(f"Config:          {args.config}")
    lines.append(f"Model:           {model_name}")
    lines.append(f"enable_thinking: {enable_thinking}")
    lines.append(f"Total calls:     {call_count}")
    lines.append(f"Total records:   {len(records)}")
    lines.append(f"<think> found:   {think_count}")
    lines.append("=" * 60)

    for r in records:
        flag = "  *** <think> DETECTED ***" if r["think_detected"] else ""
        lines.append(f"\n[call={r['call']} prompt={r['prompt_idx']} completion={r['completion_idx']}]{flag}")
        lines.append(f"PROMPT:     {r['prompt']!r}")
        lines.append(f"COMPLETION: {r['completion']!r}")

    lines.append("\n" + "=" * 60)
    if think_count == 0:
        lines.append("RESULT: PASS — no <think> tokens in any completion")
    else:
        lines.append(f"RESULT: FAIL — {think_count} completion(s) contained <think> tokens")

    report = "\n".join(lines)
    print(report)

    with open(LOG, "w") as f:
        f.write(report + "\n")

    print(f"\nReport saved to {LOG}")
    sys.exit(0 if think_count == 0 else 1)


if __name__ == "__main__":
    main()
