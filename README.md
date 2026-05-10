# multi-agent-fine-tuning-platform

Training and evaluation platform for multi-agent fine-tuning of open models on AMD MI300X.

## ROCm-related tooling

Training and profiling assume **ROCm in Docker** on AMD hardware. Host access to the GPU typically uses `/dev/kfd` and `/dev/dri` with `render` / `video` groups; `platform/scripts/docker-run.sh` encodes this for routine commands.

### `rocm-smi` (routine monitoring)

- **What it answers:** GPU utilization, VRAM use, and device visibility — suitable for dashboards and session records alongside step metrics.
- **How this repo uses it:** When `smi_poll_interval` is set in the training YAML, `src/instrumentation/smi_poller.py` runs `rocm-smi --showuse --showmeminfo vram --json` on a background thread and attaches parsed values to the session (and to W&B when configured).
- **Operational note:** If a container sees no GPUs, retry with Docker networking / IPC tweaks documented in the P0a gate scripts (e.g. `--ipc=host` once); silent empty device lists are a common pitfall.

### `rocprof` / `rocprofv2` / `rocprofv3` (episodic profiling)

- **What it answers:** Kernel execution, HIP activity, and overlap — questions **`rocm-smi` alone cannot** (e.g. after a ROCm or PyTorch bump, “did the kernel mix change?”).
- **How this repo uses it:** **`scripts/rocprof-stack-profile.sh`** wraps `run.py` with `rocprof`. Set `ROCPROF_CMD` (`rocprofv3` default, or `rocprofv2`). Output goes under `ROCPROF_OUTPUT_DIR` (see script header). This is for **short, bounded** runs; `--hip-trace` output can be **very large**.
- **Privileges:** `rocprofv3` commonly requires a **privileged** container and GPU devices; align with `project/implementation/P3a/P3a-1-Setup-Instruction.md`. Perf-counter enumeration may still be restricted without privilege — then rely on `rocm-smi` + PyTorch profiler for evidence.

---

## Requirements

- AMD MI300X server with `rocm:latest` Docker image pre-loaded
- CoMLRL repo cloned (handled by `make install-comlrl`)
- Optional: Weights & Biases API key in the environment for P3a-style runs (`WANDB_API_KEY`, plus `WANDB_PROJECT` / `WANDB_ENTITY` as needed)

## Structure

```
run.py                              # MAGRPO training entry point (config-driven)
session_schema.json                 # JSON Schema (draft-07) for session files

src/
  utils.py                          # Shared helpers: build_tokenizer, assert_no_think_tokens
  session/session.py                # Session: records config, metrics, and runtime per run
  training/
    checkpoint.py                   # Checkpoint rotation + resume metadata
    watchdog.py                     # Heartbeat / worker liveness (shared-memory)
    guards.py                       # Loss / KL / reward-collapse checks around training steps
    supervisor.py                   # Stall detection utilities (used in tests; extend for orchestration)
  instrumentation/
    smi_poller.py                   # Background rocm-smi polling → session + W&B hardware metrics
    wandb_logger.py                 # Optional W&B init / step logging
  metrics/
    trace_aggregate.py              # P3b JSONL → D1 aggregates (pure)
    platform_insights_builder.py    # Merge traces + rocprof/smi → summary dict

baseline/
  agent.py                          # Agent: model + tokenizer wrapper (multi-agent ready)
  eval_baseline.py                  # Single-agent HumanEval evaluator; writes samples + session
  merge_metrics.py                  # Merges generation stats into a single baseline_metrics.json
  metrics.py                        # compute_syntactic_ratio, compute_token_throughput

configs/
  p1b-baseline.example.yaml         # HumanEval baseline config template
  p3a-instrumentation.example.yaml  # P3a reference: W&B + smi polling + checkpoint layout

scripts/
  docker-run.sh                     # docker run wrapper (ROCm devices, workspace + CoMLRL mounts)
  smoke.sh                          # P1a thinking-mode smoke test
  install-comlrl.sh                 # clone CoMLRL if not present
  inspect_completions.py            # diagnostic: log every completion, flag <think> / redacted thinking tokens
  rocprof-stack-profile.sh          # Episodic rocprof wrapper for stack comparisons (see script header)
  build_platform_insights.py        # P3c: JSONL + rocprof/smi → demo/platform_insights/summary.json shape
  export_training_metrics.py        # P3c: W&B row JSON → demo/wandb/training_metrics.json shape

tests/                              # Host-side pytest suite (torch mocked where needed)
```

## Running the baseline evaluator

The baseline evaluator runs inside Docker via the demo runner (`demo/scripts/run_baseline.sh`).
To run it manually:

```bash
source ../project/.env
bash scripts/docker-run.sh \
  "pip install pyyaml datasets human-eval -q && \
   python -m baseline.eval_baseline \
     --config configs/p1b-baseline.example.yaml \
     --problems-path artifacts/P1b/problems.jsonl \
     --sessions-dir /sessions"
```

Pass `EXTRA_DOCKER_ARGS="-v /path/to/sessions:/sessions"` to mount the sessions directory.

## Session storage

Every run writes a JSON record to a sessions directory. Baseline runs store the evaluator YAML under `config.baseline`; training runs store the full training YAML under `config.training` (and may omit `baseline`). Example baseline-oriented record (fields may be null until the run finishes):

```json
{
  "session_id": "<uuid4>",
  "timestamp": "2026-05-07T22:14:50Z",
  "user": null,
  "stage": {"baseline": true, "training": false},
  "config": {
    "baseline": {"model": "Qwen/Qwen3-1.7B", "seed": 42}
  },
  "metrics": {"test_pass_rate": 0.0732, "token_throughput_per_sec": 612.7},
  "runtime": {
    "hostname": "amd-mi300x",
    "num_gpus": 1,
    "peak_gpu_memory_mb": 4187.3,
    "gpu_utilization_pct": null,
    "total_duration_sec": 120.5,
    "latest_checkpoint": null,
    "wandb_run_id": null
  }
}
```

Schema: `session_schema.json`. A sample P1b baseline session path is referenced from the demo layout (`demo/sessions/...`).

**Training** (`run.py`):

```bash
# Start a new training session
bash scripts/docker-run.sh \
  "pip install pyyaml datasets -q && \
   python run.py --config configs/p1a-thinking-gate.yaml --sessions-dir /sessions"

# Resume after checkpoint recovery
bash scripts/docker-run.sh \
  "pip install pyyaml datasets -q && \
   python run.py --sessions-dir /sessions --resume-session <session-id>"
```

**Evaluation** (`baseline/eval_baseline.py`): pass `--sessions-dir` to write a baseline session.

### Instrumentation and profiling (quick pointers)

- Training configs may set `smi_poll_interval` (seconds) for `rocm-smi` polling and a `wandb:` block; see `configs/p3a-instrumentation.example.yaml`.
- For episodic stack profiling with **rocprof**, use `scripts/rocprof-stack-profile.sh` from inside Docker (see **ROCm-related tooling** above for `ROCPROF_CMD`, output directories, and privilege requirements).

### Analytics exports (P3c)

- Catalog + schemas: `artifacts/P3c/METRIC_CATALOG.md`, `artifacts/P3c/*.schema.json`.
- **Episodic rocprof → demo:** After a run, copy kernel timeline CSV and hardware-counter JSON into `demo/rocprof/` (paths in storyboard cold-start). Align episodic captures with the training steps you annotate in gate notes — see `METRIC_CATALOG.md` §capture workflow.
- **Platform insights:** On CPU, from repo root or `platform/` (adjust paths):

  ```bash
  python scripts/build_platform_insights.py \
    --steps-jsonl ../artifacts/P3b/sample-traces/<SESSION_ID>/steps.jsonl \
    --rocprof-dir ../demo/rocprof \
    --smi-csv ../demo/rocm_smi/gpu_utilization_log.csv \
    --out ../demo/platform_insights/summary.json
  ```

- **W&B → frozen curves JSON:**

  ```bash
  python scripts/export_training_metrics.py \
    --wandb-rows-json ./wandb_history_rows.json \
    --out ../demo/wandb/training_metrics.json
  ```

  (`wandb_history_rows.json` is a JSON array of per-step dicts using `_step` or `step`, with keys matching `METRIC_CATALOG.md`.)

## Running tests

From `platform/`:

```bash
uvx pytest tests/ -v
```

Tests run on the host (no Docker required). `torch` is mocked where not available.

## Custom docker run

```bash
bash scripts/docker-run.sh "python your_script.py --arg value"
```

Set `EXTRA_DOCKER_ARGS` to inject additional `docker run` flags (e.g. volume mounts):

```bash
EXTRA_DOCKER_ARGS="-v /host/sessions:/sessions" bash scripts/docker-run.sh "python run.py ..."
```

All commands run inside `rocm:latest` with the workspace and CoMLRL mounted.
