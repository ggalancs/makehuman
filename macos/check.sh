#!/bin/bash
# Project checks. Run before every commit.
#
#   macos/check.sh         unit tests + coverage gate (no window opens)
#   macos/check.sh --e2e   also drive the real app end to end (opens the MakeHuman window)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
cd "$REPO"

"$PY" -c "import pytest, pytest_cov" 2>/dev/null || {
  echo "Missing test dependencies: install them with" >&2
  echo "  uv pip install --python $PY -r requirements-dev.txt" >&2
  exit 1
}

# Code added by this fork must stay at >= 90% line coverage.
"$PY" -m pytest tests/unit -q \
  --cov=download_system_assets --cov-report=term-missing --cov-fail-under=90

if [ "${1:-}" = "--e2e" ]; then
  MH_RUN_E2E=1 "$PY" -m pytest tests/e2e -q
fi
