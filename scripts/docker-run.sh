#!/usr/bin/env bash
# Run a command inside the ROCm container with workspace and CoMLRL mounted.
# Usage: ./scripts/docker-run.sh <command> [args...]
# Requires: IMAGE_TAG, COMLRL_REPO_PATH set in environment (source project/.env)
set -euo pipefail

# TODO: test and setup dkms status according to https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html
dkms status | grep "installed" || { echo "FAIL: install `amdgpu-dkms` before running ROCm docker"; exit 1 }

# TODO: fix image
IMAGE_TAG="${IMAGE_TAG:-rocm:latest}"
COMLRL_REPO_PATH="${COMLRL_REPO_PATH:-/root/ma-workspace/comlrl-repo}"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"

python3 -m compileall -q "$WORKSPACE" || { echo "FAIL: syntax error in platform source — fix before running Docker"; exit 1; }

exec docker run --rm \
  # ROCm Docker parameters
  --ipc=host \
  --privileged=true \
  --shm-size=128GB \
  --network=host \
  --device /dev/kfd --device /dev/dri \
  # --security-opt seccomp=unconfined \
  # custom parameters
  --group-add render --group-add video \
  -v "$WORKSPACE:/workspace" \
  -v "$COMLRL_REPO_PATH:/comlrl" \
  ${EXTRA_DOCKER_ARGS:-} \
  -e HF_HOME=/workspace/.hf_cache \
  -e TRANSFORMERS_VERBOSITY=error \
  -w /workspace \
  "$IMAGE_TAG" \
  bash -c "pip install -e /comlrl -q && $*"
