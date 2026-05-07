#!/usr/bin/env bash
# Run the P1a smoke test inside the ROCm container.
# Usage: ./scripts/smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/docker-run.sh" "pip install pyyaml datasets -q && python tests/smoke_test_thinking.py"
