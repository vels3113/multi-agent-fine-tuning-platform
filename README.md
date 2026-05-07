# multi-agent-fine-tuning-platform

Training platform for MAGRPO-based multi-agent fine-tuning of Qwen3-1.7B on AMD MI300X.

## Requirements

- AMD MI300X server with `rocm:latest` Docker image pre-loaded
- CoMLRL repo cloned (handled by `make install-comlrl`)
- Environment variables sourced from `project/.env` (see `.env.example`)

## Quickstart

```bash
source ../project/.env

make install-comlrl   # clone CoMLRL once per fresh server
make smoke            # run P1a thinking-mode gate test
make inspect          # capture all completions without weight updates (diagnostic)
make train CONFIG=configs/p1a-thinking-gate.yaml
```

## Structure

```
run.py                        # config-driven training entry point
configs/
  p1a-thinking-gate.yaml      # Qwen3-1.7B, enable_thinking=false
scripts/
  docker-run.sh               # docker run wrapper (mounts, devices, CoMLRL)
  smoke.sh                    # runs the P1a smoke test inside Docker
  install-comlrl.sh           # clone CoMLRL if not present
  inspect_completions.py      # diagnostic: log every completion, flag <think>
tests/
  smoke_test_thinking.py      # asserts no <think> tokens in any completion
```

## Custom docker run

```bash
bash scripts/docker-run.sh "python your_script.py --arg value"
```

All commands run inside `rocm:latest` with the workspace and CoMLRL mounted.
