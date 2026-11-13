#!/bin/bash
# Line coverage of the whole MakeHuman application: the unit tests plus the
# end-to-end run of the real app (opens the MakeHuman window on the main screen,
# without taking the keyboard focus).
#
#   macos/coverage.sh [--fail-under N]     (default 90)
#
# Fails when a test fails or when the combined coverage is below N percent.
# Data and the report go to local/coverage/ (not tracked by git).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
OUT="$REPO/local/coverage"
FAIL_UNDER=90
if [ "${1:-}" = "--fail-under" ]; then
  FAIL_UNDER="${2:?--fail-under needs a number}"
fi
cd "$REPO"
mkdir -p "$OUT"

COVERAGE_FILE="$OUT/unit.cov" "$PY" -m pytest tests/unit -q -p no:cacheprovider \
  --cov=makehuman --cov-config=.coveragerc --cov-report=
"$PY" tests/e2e/run_smoke.py --coverage "$OUT/e2e.cov"
# combine starts from scratch (no --append): nothing to delete first
"$PY" -m coverage combine --keep --rcfile=.coveragerc --data-file="$OUT/combined.cov" \
  "$OUT/unit.cov" "$OUT/e2e.cov" >/dev/null

status=0
"$PY" -m coverage report --rcfile=.coveragerc --data-file="$OUT/combined.cov" \
  --fail-under="$FAIL_UNDER" > "$OUT/report.txt" || status=$?
grep TOTAL "$OUT/report.txt"
if [ "$status" -ne 0 ]; then
  echo "Whole-app line coverage is below ${FAIL_UNDER}% (report: $OUT/report.txt)" >&2
fi
exit "$status"
