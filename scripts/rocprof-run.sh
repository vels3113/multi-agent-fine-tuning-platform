#!/usr/bin/env bash
# platform/scripts/rocprof-run.sh
# Launch platform/run.py under rocprofv3 for kernel-level profiling.
# Requires --privileged docker flag for hardware perf counters.
#
# Usage (from workspace root inside Docker):
#   ROCPROF_CMD=rocprofv3 bash /workspace/platform/scripts/rocprof-run.sh \
#     --config /workspace/demo/configs/p3a-smoke.yaml \
#     --sessions-dir /workspace/demo/sessions/p3a
#
# Environment:
#   ROCPROF_CMD        rocprofv3 (default) or rocprofv2
#   ROCPROF_OUTPUT_DIR output directory for profiling data (default: /workspace/artifacts/P3a/rocprof)
#   WORKSPACE          repo root (default: /workspace)
set -euo pipefail

ROCPROF_CMD="${ROCPROF_CMD:-rocprofv3}"
WORKSPACE="${WORKSPACE:-/workspace}"
ROCPROF_OUTPUT_DIR="${ROCPROF_OUTPUT_DIR:-${WORKSPACE}/artifacts/P3a/rocprof}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)

mkdir -p "$ROCPROF_OUTPUT_DIR"

echo "rocprof-run.sh: using $ROCPROF_CMD"
echo "rocprof-run.sh: output directory $ROCPROF_OUTPUT_DIR/$TIMESTAMP"

if [ "$ROCPROF_CMD" = "rocprofv3" ]; then
  exec "$ROCPROF_CMD" \
    --hip-trace \
    --output-directory "${ROCPROF_OUTPUT_DIR}/${TIMESTAMP}" \
    --output-format json \
    -- python3 "${WORKSPACE}/platform/run.py" "$@"
else
  # rocprofv2 fallback
  exec "$ROCPROF_CMD" \
    --hip-trace \
    -d "${ROCPROF_OUTPUT_DIR}/${TIMESTAMP}" \
    python3 "${WORKSPACE}/platform/run.py" "$@"
fi
