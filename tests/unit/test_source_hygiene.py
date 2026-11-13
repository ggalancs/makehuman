"""Every module must compile cleanly on the running Python, warnings included."""

import os
import warnings

import pytest

from conftest import MH_DIR


def python_sources():
    for root, dirs, files in os.walk(MH_DIR):
        rel = os.path.relpath(root, MH_DIR)
        if rel == "data" or rel.startswith("data" + os.sep):
            dirs[:] = []
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


SOURCES = sorted(python_sources())


def test_sources_found():
    assert len(SOURCES) > 150  # 197 modules at the time of writing


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: os.path.relpath(p, MH_DIR))
def test_compiles_without_warnings(path):
    with open(path, encoding="utf-8") as f:
        source = f.read()
    with warnings.catch_warnings():
        # Invalid escape sequences are SyntaxWarnings since Python 3.12 and will become errors
        warnings.simplefilter("error")
        compile(source, path, "exec")
