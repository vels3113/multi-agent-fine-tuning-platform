#!/usr/bin/env bash
# platform/scripts/rocprof-stack-profile.sh
#
# Episodic ROCm profiling helper: run the same training command you use for a pinned
# stack (PyTorch + ROCm + CoMLRL versions), capture rocprof output under a labeled
# directory. No scoring, thresholds, or bump/pin decisions — only data collection.
#
# Typical use: run once per stack candidate (e.g. before/after a ROCm bump), then
# compare outputs manually or attach to a vendor ticket. Default --hip-trace can
# produce very large files; use a short smoke config and expect multi-GB traces.
#
# Usage (inside Docker, repo at /workspace — see P3a validation doc):
#   ROCPROF_CMD=rocprofv3 bash /workspace/platform/scripts/rocprof-stack-profile.sh \
#     --config /workspace/platform/configs/p3a-instrumentation.example.yaml \
#     --sessions-dir /workspace/artifacts/P3a/profile_sessions
#
# Environment:
#   ROCPROF_CMD        rocprofv3 (default) or rocprofv2
#   ROCPROF_OUTPUT_DIR base output directory (default: /workspace/artifacts/P3a/rocprof)
#   WORKSPACE          repo root (default: /workspace)
#   STACK_LABEL        label prepended to timestamp subdir (default: profile) — for your notes only
#
# Requires: docker run --privileged (and GPU devices) for rocprofv3 as documented in P3a-1.
set -euo pipefail

ROCPROF_CMD="${ROCPROF_CMD:-rocprofv3}"
WORKSPACE="${WORKSPACE:-/workspace}"
ROCPROF_OUTPUT_DIR="${ROCPROF_OUTPUT_DIR:-${WORKSPACE}/artifacts/P3a/rocprof}"
STACK_LABEL="${STACK_LABEL:-profile}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="${ROCPROF_OUTPUT_DIR}/${STACK_LABEL}_${TIMESTAMP}"

mkdir -p "$ROCPROF_OUTPUT_DIR"

echo "rocprof-stack-profile.sh: command=$ROCPROF_CMD label=$STACK_LABEL"
echo "rocprof-stack-profile.sh: output directory $OUT"

if [ "$ROCPROF_CMD" = "rocprofv3" ]; then
  exec "$ROCPROF_CMD" \
    --hip-trace \
    --output-directory "$OUT" \
    --output-format json \
    -- python3 "${WORKSPACE}/platform/run.py" "$@"
else
  exec "$ROCPROF_CMD" \
    --hip-trace \
    -d "$OUT" \
    python3 "${WORKSPACE}/platform/run.py" "$@"
fi
