# multi-agent-fine-tuning-platform

Training and evaluation platform for MAGRPO-based multi-agent fine-tuning of Qwen3-1.7B on AMD MI300X.

## Requirements

- AMD MI300X server with `rocm:latest` Docker image pre-loaded
- CoMLRL repo cloned (handled by `make install-comlrl`)
- Environment variables sourced from `project/.env` (see `.env.example`)

## Quickstart

```bash
source ../project/.env

make install-comlrl   # clone CoMLRL once per fresh server
make smoke            # run P1a thinking-mode gate test
make train CONFIG=configs/p1a-thinking-gate.yaml
```

## Structure

```
run.py                              # MAGRPO training entry point (config-driven)
session.py                          # Session: records config, metrics, and runtime per run
session_schema.json                 # JSON Schema (draft-07) for session files
utils.py                            # Shared helpers: build_tokenizer, assert_no_think_tokens

baseline/
  agent.py                          # Agent: model + tokenizer wrapper (multi-agent ready)
  eval_baseline.py                  # Single-agent HumanEval evaluator; writes samples + session
  merge_metrics.py                  # Merges generation stats into a single baseline_metrics.json
  metrics.py                        # compute_syntactic_ratio, compute_token_throughput

configs/
  p1a-thinking-gate.yaml            # Qwen3-1.7B, enable_thinking=false (gate config)
  p1b-baseline.example.yaml         # HumanEval baseline config template

scripts/
  docker-run.sh                     # docker run wrapper (ROCm devices, workspace + CoMLRL mounts)
  smoke.sh                          # P1a thinking-mode smoke test
  install-comlrl.sh                 # clone CoMLRL if not present
  inspect_completions.py            # diagnostic: log every completion, flag <think> tokens
  analyze_p1b.py                    # offline analysis of P1b baseline artifacts

tests/
  smoke_test_thinking.py            # asserts no <think> tokens in any completion
  test_baseline.py                  # unit tests for metrics functions
  test_session.py                   # unit tests for Session (start, update, load)
  conftest.py                       # adds platform/ to sys.path for test imports
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

Every run writes a JSON record to a sessions directory:

```json
{
  "session_id": "<uuid4>",
  "timestamp": "2026-05-07T22:14:50Z",
  "stage": {"baseline": true, "training": false},
  "config": {"model": "Qwen/Qwen3-1.7B", "num_problems": 164, ...},
  "metrics": {"test_pass_rate": 0.0732, "token_throughput_per_sec": 612.7, ...},
  "runtime": {"hostname": "amd-mi300x", "num_gpus": 1, "peak_gpu_memory_mb": 4187.3, ...}
}
```

Schema: `session_schema.json`. The P1b baseline session is at `demo/sessions/a7c4f91b-3e2d-4b8a-8f6e-5d0c7a1b9e3f.json`.

**Training** (`run.py`):

```bash
# Start a new training session
bash scripts/docker-run.sh \
  "python run.py --config configs/p1a-thinking-gate.yaml --sessions-dir /sessions"

# Resume after checkpoint recovery
bash scripts/docker-run.sh \
  "python run.py --config configs/p1a-thinking-gate.yaml \
     --sessions-dir /sessions \
     --resume-session <session-id>"
```

**Evaluation** (`baseline/eval_baseline.py`): pass `--sessions-dir` to write a baseline session.

## Running tests

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
