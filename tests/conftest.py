"""
Shared test setup: make the MakeHuman source folders importable the same way
makehuman.py does, without a display and without touching the real user home.
"""

import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MH_DIR = os.path.join(REPO, "makehuman")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# getpath reads MH_HOME_LOCATION when first asked for the home folder
_HOME = tempfile.mkdtemp(prefix="mh-unit-home-")
os.environ["MH_HOME_LOCATION"] = _HOME

for sub in reversed(["", "lib", "apps", "shared", os.path.join("apps", "gui"), "core"]):
    sys.path.insert(0, os.path.join(MH_DIR, sub))

from core import G  # noqa: E402

G.args = {}  # glmodule reads its OpenGL debug flags from here at import time


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_HOME, ignore_errors=True)  # created above by mkdtemp
