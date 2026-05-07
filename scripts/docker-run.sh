#!/usr/bin/env bash
# Run a command inside the ROCm container with workspace and CoMLRL mounted.
# Usage: ./scripts/docker-run.sh <command> [args...]
# Requires: IMAGE_TAG, COMLRL_REPO_PATH set in environment (source project/.env)
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-rocm:latest}"
COMLRL_REPO_PATH="${COMLRL_REPO_PATH:-/root/ma-workspace/comlrl-repo}"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"

exec docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add render --group-add video \
  -v "$WORKSPACE:/workspace" \
  -v "$COMLRL_REPO_PATH:/comlrl" \
  -e HF_HOME=/workspace/.hf_cache \
  -w /workspace \
  "$IMAGE_TAG" \
  bash -c "pip install -e /comlrl -q && $*"
