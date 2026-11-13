import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_smoke  # noqa: E402

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.environ.get("MH_RUN_E2E") != "1",
                       reason="opens the MakeHuman window; set MH_RUN_E2E=1 to run"),
]


def test_makehuman_end_to_end():
    assert run_smoke.run(sys.executable, timeout=int(os.environ.get("MH_E2E_TIMEOUT", "600"))) == 0
