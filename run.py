from __future__ import annotations

import sys
import os
import argparse
import dataclasses
import statistics
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
from src.instrumentation.pytorch_step_profiler import (
    PytorchStepProfiler,
    TRACING_VERSION,
    step_tracing_disabled_by_env,
)
from src.instrumentation.step_trace_writer import StepTraceWriter
from src.instrumentation.orchestration_timers import StepOrchestrationTimers
from src.metrics.training_step_extract import (
    infer_syntactic_correctness_ratio,
    infer_task_pass_rate,
)

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
    if ds_cfg.get("type") == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split=ds_cfg.get("split", "test"))
        max_problems = ds_cfg.get("max_problems")
        if max_problems:
            ds = ds.select(range(min(max_problems, len(ds))))
        return ds
    return load_dataset(ds_cfg["name"], split=ds_cfg.get("split", "train"))


def build_reward_fn(name: str):
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward_func '{name}'. Available: {list(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[name]


def _resolve_trace_output_jsonl(tracing_cfg: dict, session, workspace_root: str) -> str:
    rel_or_abs = tracing_cfg.get("trace_output_dir")
    if not rel_or_abs:
        trace_parent = os.path.join(workspace_root, "artifacts", "P3b", "sample-traces")
    elif os.path.isabs(rel_or_abs):
        trace_parent = rel_or_abs
    else:
        trace_parent = os.path.join(workspace_root, rel_or_abs)
    sid = session.session_id if session else "no-session"
    return os.path.join(trace_parent, sid, "steps.jsonl")


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

    _loss_history: list[float] = []
    _step_counter: list[int] = [0]
    _kl_baseline: list[float | None] = [None]

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

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tracing_cfg = cfg.get("tracing") or {}
    trace_enabled = bool(tracing_cfg.get("enabled", False)) and not step_tracing_disabled_by_env()
    step_profiler: PytorchStepProfiler | None = None
    trace_writer: StepTraceWriter | None = None
    orch: StepOrchestrationTimers | None = None
    trace_log_wandb = False

    if trace_enabled:
        pevery = int(tracing_cfg.get("profile_every_n_steps", 10))
        export_chrome = bool(tracing_cfg.get("export_chrome_trace", False))
        chrome_path = tracing_cfg.get("chrome_trace_path")
        if chrome_path and not os.path.isabs(str(chrome_path)):
            chrome_path = os.path.join(workspace_root, str(chrome_path))
        step_profiler = PytorchStepProfiler(
            profile_every_n=pevery,
            export_chrome_trace=export_chrome,
            chrome_trace_path=chrome_path,
        )
        orch = StepOrchestrationTimers()
        trace_log_wandb = bool(tracing_cfg.get("log_to_wandb", False))
        jsonl_path = _resolve_trace_output_jsonl(tracing_cfg, session, workspace_root)
        trace_writer = StepTraceWriter(jsonl_path)
        print(f"P3b tracing enabled → {jsonl_path}", flush=True)

        _gen_for_trace = trainer._generate_completions

        def _traced_generate(*args, **kwargs):
            trace_step = _step_counter[0] + 1
            assert orch is not None and step_profiler is not None
            orch.reset()
            step_profiler.on_rollout_start(trace_step)
            with orch.rollout_scope():
                return _gen_for_trace(*args, **kwargs)

        trainer._generate_completions = _traced_generate

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
    _original_loss = trainer._compute_loss_with_gradients

    def _logging_loss(agent, completions_data, returns):
        trace_step = _step_counter[0] + 1
        if trace_enabled:
            assert orch is not None and step_profiler is not None
            with orch.loss_compute_scope():
                loss = _original_loss(agent, completions_data, returns)
            prof_part = step_profiler.on_loss_end(trace_step)
            if prof_part is None:
                prof_part = step_profiler.noop_loss_aggregates()
        else:
            loss = _original_loss(agent, completions_data, returns)
            prof_part = None

        _step_counter[0] += 1
        loss_val = loss.item() if hasattr(loss, "item") else float(loss)
        _loss_history.append(loss_val)
        if _step_counter[0] % 8 == 1 or _step_counter[0] <= 3:
            print(f"step={_step_counter[0]} loss={loss_val:.4f}", flush=True)

        rewards = [float(r) for r in returns] if returns is not None else []

        if trace_enabled and trace_writer is not None and orch is not None and prof_part is not None:
            row = {
                "step": _step_counter[0],
                "tracing_version": TRACING_VERSION,
                "profiled": bool(prof_part["profiled"]),
                "forward_ms": prof_part["forward_ms"],
                "backward_ms": prof_part["backward_ms"],
                "cuda_total_ms": prof_part["cuda_total_ms"],
                "top_op_name": prof_part["top_op_name"],
                "top_op_ms": prof_part["top_op_ms"],
                "profiler_note": prof_part["profiler_note"],
                "timer_rollout_wall_ms": orch.rollout_wall_ms,
                "timer_loss_compute_wall_ms": orch.loss_compute_wall_ms,
            }
            if session is not None:
                row["session_id"] = session.session_id
            trace_writer.append(row)

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
        if rewards:
            wb_metrics["train/joint_reward"] = float(statistics.mean(rewards))
        if len(rewards) >= 2:
            wb_metrics["train/reward_std_across_agents"] = float(statistics.pstdev(rewards))
        tpr = infer_task_pass_rate(completions_data)
        if tpr is not None:
            wb_metrics["train/task_pass_rate"] = float(tpr)
        syn = infer_syntactic_correctness_ratio(completions_data)
        if syn is not None:
            wb_metrics["train/syntactic_correctness_ratio"] = float(syn)

        if smi_snap["gpu_utilization_pct_mean"] is not None:
            wb_metrics["hardware/gpu_util_pct"] = smi_snap["gpu_utilization_pct_mean"]
        if smi_snap["vram_used_mb_mean"] is not None:
            wb_metrics["hardware/vram_used_mb"] = smi_snap["vram_used_mb_mean"]

        if trace_enabled and trace_log_wandb and prof_part and prof_part.get("profiled"):
            for key in ("forward_ms", "backward_ms", "cuda_total_ms", "top_op_ms"):
                val = prof_part.get(key)
                if val is not None:
                    wb_metrics[f"trace/{key}"] = val

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
