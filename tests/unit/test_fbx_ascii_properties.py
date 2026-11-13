"""ASCII FBX property lines: the shared property tables hold bytes for the binary
writer; the ASCII writer must put their text in the file, not b'...' reprs."""

import importlib
import os
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The plugin's modules use relative imports and its __init__ needs the running app:
# load them as part of a stand-in package whose path is the plugin folder.
PACKAGE = "_fbx_exporter_under_test"
_package = types.ModuleType(PACKAGE)
_package.__path__ = [os.path.join(REPO, "makehuman", "plugins", "9_export_fbx")]
sys.modules.setdefault(PACKAGE, _package)

fbx_utils_bin = importlib.import_module(PACKAGE + ".fbx_utils_bin")


@pytest.mark.parametrize("args, expected", [
    ((b"Size", "p_double", 100), 'P: "Size", "double", "Number", "", 100'),
    ((b"Color", "p_color_rgb", (0.8, 0.8, 0.8)), 'P: "Color", "ColorRGB", "Color", "", 0.8,0.8,0.8'),
    ((b"Path", "p_string_url", "/tmp/eye.png"), 'P: "Path", "KString", "Url", "", "/tmp/eye.png"'),
    ((b"Path", "p_string_url", b"/tmp/eye.png"), 'P: "Path", "KString", "Url", "", "/tmp/eye.png"'),
    (("Lcl Rotation", "p_double", 1, True), 'P: "Lcl Rotation", "double", "Number", "A", 1'),
], ids=["double", "color", "url", "bytes-url", "animatable"])
def test_ascii_property_line(args, expected):
    assert fbx_utils_bin.get_ascii_property(*args) == expected


def test_ascii_properties_block_has_no_bytes_reprs():
    text = fbx_utils_bin.get_ascii_properties([(b"Size", "p_double", 1), (b"Path", "p_string_url", b"x.png")], indent=2)
    assert "b'" not in text
    assert text.splitlines()[0].startswith("        P: \"Size\"")
