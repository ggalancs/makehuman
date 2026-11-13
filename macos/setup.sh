#!/bin/bash
# Set up MakeHuman from this source checkout on macOS (Apple Silicon or Intel).
#
#   macos/setup.sh                 venv + dependencies + system assets + compiled data
#   macos/setup.sh --skip-assets   do not download the system assets pack (~270 MB)
#   PYTHON=3.13 macos/setup.sh     choose the Python version (uv) or interpreter path
#
# Safe to run again: the venv is reused, existing asset files are never
# overwritten and the compile steps just refresh the binary caches.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"
PY_WANTED="${PYTHON:-3.12}"
SKIP_ASSETS=0
for arg in "$@"; do
  case "$arg" in
    --skip-assets) SKIP_ASSETS=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n==> %s\n' "$*"; }

# 1. Virtual environment ---------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  say "Creating $VENV (Python $PY_WANTED)"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PY_WANTED" "$VENV"
  else
    PY_BIN="$PY_WANTED"
    case "$PY_BIN" in [0-9]*) PY_BIN="python$PY_BIN" ;; esac
    command -v "$PY_BIN" >/dev/null 2>&1 || {
      echo "Python $PY_WANTED not found. Install it (https://www.python.org or 'brew install python@$PY_WANTED') or install uv." >&2
      exit 1
    }
    "$PY_BIN" -m venv "$VENV"
  fi
fi
PY="$VENV/bin/python"

# 2. Refuse an x86_64 interpreter on Apple Silicon (it would run under Rosetta)
if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
  arch="$("$PY" -c 'import platform; print(platform.machine())')"
  if [ "$arch" != "arm64" ]; then
    echo "The Python in $VENV is $arch, not arm64: it would run under Rosetta." >&2
    echo "Remove that venv and run this script again with a native (arm64) Python." >&2
    exit 1
  fi
fi

# 3. Dependencies ----------------------------------------------------------
say "Installing dependencies"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PY" -r "$REPO/requirements.txt"
else
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r "$REPO/requirements.txt"
fi
"$PY" - <<'EOF'
import platform, numpy, OpenGL
from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
print("Python %s (%s), numpy %s, PyQt %s / Qt %s, PyOpenGL %s" % (
    platform.python_version(), platform.machine(), numpy.__version__,
    PYQT_VERSION_STR, QT_VERSION_STR, OpenGL.__version__))
EOF

# 4. Assets ------------------------------------------------------------------
cd "$REPO/makehuman"
if [ "$SKIP_ASSETS" = "0" ]; then
  say "Installing the MakeHuman system assets (hair, clothes, skins, proxies...)"
  "$PY" download_system_assets.py
fi

# 5. Binary caches (much faster start-up) ------------------------------------
say "Compiling targets, meshes and proxies"
"$PY" compile_targets.py >/dev/null
"$PY" compile_models.py >/dev/null
"$PY" compile_proxies.py >/dev/null

say "Done. Start MakeHuman with:  $REPO/macos/makehuman"
