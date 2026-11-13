#!/bin/bash
# Line coverage of the whole MakeHuman application: unit tests plus the
# end-to-end run of the real app (opens the MakeHuman window).
#
#   macos/coverage.sh [--fail-under N]
#
# Data and the report go to local/coverage/ (not tracked by git).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
OUT="$REPO/local/coverage"
FAIL_UNDER="${2:-0}"
[ "${1:-}" = "--fail-under" ] || FAIL_UNDER=0
cd "$REPO"
mkdir -p "$OUT"

"$PY" -m pytest tests/unit -q --cov=makehuman --cov-config=.coveragerc --cov-report=
mv .coverage "$OUT/unit.cov"
"$PY" tests/e2e/run_smoke.py --coverage "$OUT/e2e.cov"
"$PY" -m coverage combine --keep --rcfile=.coveragerc --data-file="$OUT/combined.cov" "$OUT/unit.cov" "$OUT/e2e.cov" >/dev/null
"$PY" -m coverage report --rcfile=.coveragerc --data-file="$OUT/combined.cov" > "$OUT/report.txt" || true
grep TOTAL "$OUT/report.txt"
"$PY" -m coverage report --rcfile=.coveragerc --data-file="$OUT/combined.cov" --fail-under="$FAIL_UNDER" >/dev/null
