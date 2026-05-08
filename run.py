import sys
import os
import argparse
import dataclasses
import yaml
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from comlrl.trainers.reinforce.magrpo import MAGRPOTrainer, MAGRPOConfig
from src.utils import build_tokenizer, assert_no_think_tokens
from src.session.session import Session
from src.training.checkpoint import CheckpointManager
from src.training.guards import check_loss, check_kl, check_reward_collapse
from src.training.watchdog import Watchdog
from src.instrumentation.smi_poller import SmiPoller
from src.instrumentation.wandb_logger import WandbLogger

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
    if ds_cfg.get("type") == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", split=ds_cfg.get("split", "train"))
        max_problems = ds_cfg.get("max_problems")
        if max_problems:
            ds = ds.select(range(min(max_problems, len(ds))))
        return ds.rename_column("text", "prompt")
    return load_dataset(ds_cfg["name"], split=ds_cfg.get("split", "train"))


def build_reward_fn(name: str):
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward_func '{name}'. Available: {list(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[name]



# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="Path to YAML config (required for fresh runs; omit when --resume-session is used)")
    parser.add_argument("--sessions-dir", default=None,
                        help="Directory to write session JSON (skipped if not set)")
    parser.add_argument("--resume-session", default=None,
                        help="Session ID to resume (reads config from session.config['training'])")
    args = parser.parse_args()

    if args.sessions_dir and args.resume_session:
        session = Session.load(args.resume_session, args.sessions_dir)
        cfg = session.config["training"]
        print(f"Resumed session {session.session_id}")
    elif args.config:
        cfg = load_config(args.config)
        session = (
            Session.start({"training": cfg}, stage={"baseline": False, "training": True})
            if args.sessions_dir else None
        )
        if session:
            print(f"Started session {session.session_id}")
    else:
        raise SystemExit("error: --config is required when not using --resume-session")

    torch.manual_seed(cfg["seed"])
    model_name = cfg["model"]
    model_params = cfg.get("model_params", {})
    enable_thinking = model_params.get("enable_thinking", False)

    tokenizer = build_tokenizer(model_name)
    dataset = build_dataset(cfg)
    reward_fn = build_reward_fn(cfg["reward_func"])

    _magrpo_fields = {f.name for f in dataclasses.fields(MAGRPOConfig)}
    trainer_cfg = MAGRPOConfig(
        num_train_epochs=cfg["num_train_epochs"],
        num_agents=cfg["num_agents"],
        **{k: v for k, v in model_params.items()
           if k not in ("enable_thinking",) and k in _magrpo_fields},
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

    # Wrap _generate_completions to assert no <think> tokens in every rollout
    if not enable_thinking:
        _original_gen = trainer._generate_completions
        def _guarded_gen(*args, **kwargs):
            completions = _original_gen(*args, **kwargs)
            for text in (completions if isinstance(completions, list) else [completions]):
                if isinstance(text, str):
                    assert_no_think_tokens(text, context="rollout completion")
            return completions
        trainer._generate_completions = _guarded_gen

    ckpt_cfg = cfg.get("checkpoint", {})
    ckpt_dir = ckpt_cfg.get("output_dir")
    save_every = ckpt_cfg.get("save_steps", 1)
    reward_guard_enabled = cfg.get("reward_guard", True)
    cm = CheckpointManager(ckpt_dir, keep=2) if ckpt_dir else None

    shm_name = cfg.get("watchdog_shm", "magrpo_heartbeat")
    watchdog = Watchdog(shm_name=shm_name, interval=2.0)
    watchdog.start(initial_workers=len(__import__("multiprocessing").active_children()),
                   initial_batch=0)

    # rocm-smi background polling — populates gpu_utilization_pct in session
    smi_poll_interval = cfg.get("smi_poll_interval", 5.0)
    smi_poller = SmiPoller(interval=smi_poll_interval)
    smi_poller.start()

    # W&B logging — optional, degrades gracefully without WANDB_API_KEY
    wandb_cfg = cfg.get("wandb", {})
    wb = WandbLogger(cfg=wandb_cfg)
    wandb_run_id = wb.init(run_config=cfg)
    if wandb_run_id:
        os.environ["WANDB_RUN_ID"] = wandb_run_id
        print(f"W&B run: {wandb_run_id}", flush=True)

    # Wrap _compute_loss_with_gradients to log per-step loss, run guards, and save checkpoints
    _loss_history = []
    _step_counter = [0]
    _kl_baseline = [None]
    _original_loss = trainer._compute_loss_with_gradients

    def _logging_loss(agent, completions_data, returns):
        loss = _original_loss(agent, completions_data, returns)
        _step_counter[0] += 1
        loss_val = loss.item() if hasattr(loss, "item") else float(loss)
        _loss_history.append(loss_val)
        if _step_counter[0] % 8 == 1 or _step_counter[0] <= 3:
            print(f"step={_step_counter[0]} loss={loss_val:.4f}", flush=True)

        rewards = [float(r) for r in returns] if returns is not None else []
        try:
            check_loss(loss_val, step=_step_counter[0])
            check_kl(current_kl=loss_val, baseline_kl=_kl_baseline[0],
                     threshold_multiplier=20.0, step=_step_counter[0])
            if reward_guard_enabled:
                check_reward_collapse(rewards, step=_step_counter[0])
        except ValueError as exc:
            print(f"[GUARD] {exc} — skipping checkpoint save", flush=True)
            return loss

        if _kl_baseline[0] is None and loss_val > 0:
            _kl_baseline[0] = loss_val

        if cm and save_every and _step_counter[0] % save_every == 0:
            wandb_id = os.environ.get("WANDB_RUN_ID")
            cm.save(trainer, step=_step_counter[0], wandb_run_id=wandb_id)

        # Log step metrics to W&B (includes loss + smi snapshot)
        smi_snap = smi_poller.get_stats()
        wb_metrics = {"train/loss": loss_val, "train/step": _step_counter[0]}
        if smi_snap["gpu_utilization_pct_mean"] is not None:
            wb_metrics["hardware/gpu_util_pct"] = smi_snap["gpu_utilization_pct_mean"]
        if smi_snap["vram_used_mb_mean"] is not None:
            wb_metrics["hardware/vram_used_mb"] = smi_snap["vram_used_mb_mean"]
        wb.log(wb_metrics, step=_step_counter[0])

        return loss

    trainer._compute_loss_with_gradients = _logging_loss

    trainer.train()
    if _loss_history:
        print(f"loss_first={_loss_history[0]:.4f} loss_last={_loss_history[-1]:.4f} "
              f"loss_mean={sum(_loss_history)/len(_loss_history):.4f} steps={len(_loss_history)}")
    print("Training complete.")

    watchdog.stop()

    smi_poller.stop()
    smi_stats = smi_poller.get_stats()

    wb.finish()

    # Capture peak VRAM immediately after training, before any save/cleanup resets stats
    peak_vram_mb = None
    if torch.cuda.is_available():
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
        print(f"peak_gpu_memory_mb={peak_vram_mb}")

    # Final checkpoint save
    latest_ckpt = None
    if cm:
        wandb_id = os.environ.get("WANDB_RUN_ID")
        latest_ckpt = cm.save(trainer, step=_step_counter[0], wandb_run_id=wandb_id)
        print(f"Checkpoint saved to {latest_ckpt}")

    if session is not None:
        if peak_vram_mb is not None:
            session.runtime["peak_gpu_memory_mb"] = peak_vram_mb
        if latest_ckpt is not None:
            session.runtime["latest_checkpoint"] = latest_ckpt
        if smi_stats.get("gpu_utilization_pct_mean") is not None:
            session.runtime["gpu_utilization_pct"] = smi_stats["gpu_utilization_pct_mean"]
        if wandb_run_id is not None:
            session.runtime["wandb_run_id"] = wandb_run_id
        session.update({}, args.sessions_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
