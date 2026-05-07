#!/usr/bin/env bash
# Clone CoMLRL into COMLRL_REPO_PATH if not already present.
# Usage: ./scripts/install-comlrl.sh
# Requires: COMLRL_REPO_PATH, COMLRL_REPO set in environment (source project/.env)
set -euo pipefail

COMLRL_REPO_PATH="${COMLRL_REPO_PATH:-/root/ma-workspace/comlrl-repo}"
COMLRL_REPO="${COMLRL_REPO:-https://github.com/OpenMLRL/CoMLRL}"

if [ -d "$COMLRL_REPO_PATH/.git" ]; then
  echo "CoMLRL already present at $COMLRL_REPO_PATH ($(git -C "$COMLRL_REPO_PATH" log --oneline -1))"
  exit 0
fi

echo "Cloning CoMLRL into $COMLRL_REPO_PATH ..."
git clone "$COMLRL_REPO" "$COMLRL_REPO_PATH"
echo "Done: $(git -C "$COMLRL_REPO_PATH" log --oneline -1)"
