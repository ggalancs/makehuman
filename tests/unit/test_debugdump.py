"""lib/debugdump.py: the makehuman-debug.txt written at startup."""

import os
import sys
import types

import numpy
import pytest

import debugdump
import getpath
import mhversion
from core import G


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MH_HOME_LOCATION", str(tmp_path))
    for name, value in [("MH_VERSION", "1.3.0 test"), ("MH_SHORT_VERSION", "alpha"),
                        ("MH_MESH_VERSION", "hm08"), ("MH_FROZEN", "No"), ("MH_RELEASE", "No")]:
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(mhversion.MHVersion, "getFullVersionStr", lambda self: "1.3.0 alpha (b:c)")
    return tmp_path


def dump_text():
    with open(os.path.join(getpath.getPath(), "makehuman-debug.txt"), encoding="utf-8") as f:
        return f.read()


def test_reset_writes_a_fresh_file(home):
    dump = debugdump.DebugDump()
    dump.reset()
    text = dump_text()
    lines = text.splitlines()
    assert lines[:4] == ["VERSION: 1.3.0 test", "SHORT VERSION: alpha",
                         "LONG VERSION: 1.3.0 alpha (b:c)", "BASEMESH VERSION: hm08"]
    assert "HOME LOCATION: %s" % getpath.getHomePath() in lines
    assert "NUMPY.VERSION: %s" % numpy.__version__ in lines
    assert dump.debug is None  # closed
    dump.reset()  # second run appends to the same file
    assert dump_text().count("VERSION: 1.3.0 test\n") == 2


def test_platform_specific_lines(home, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    debugdump.DebugDump().reset()
    assert "PLATFORM.MAC_VER: " in dump_text()


def test_windows_version_line(home, monkeypatch):
    import platform
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(platform, "win32_ver", lambda: ("10", "10.0.19045", "SP0", "Multiprocessor"))
    debugdump.DebugDump().reset()
    assert "PLATFORM.WIN32_VER: 10 10.0.19045 SP0 Multiprocessor" in dump_text()


def test_linux_distribution_from_distro(home, monkeypatch):
    fake = types.ModuleType("distro")
    fake.linux_distribution = lambda: ("Ubuntu", "24.04", "noble")
    monkeypatch.setitem(sys.modules, "distro", fake)
    monkeypatch.setattr(sys, "platform", "linux")
    debugdump.DebugDump().reset()
    assert "PLATFORM.LINUX_DISTRIBUTION: Ubuntu 24.04 noble" in dump_text()


def test_linux_distribution_unknown_without_distro(home, monkeypatch):
    monkeypatch.setitem(sys.modules, "distro", None)  # import distro -> ImportError
    monkeypatch.setattr(sys, "platform", "linux")
    debugdump.DebugDump().reset()
    assert "PLATFORM.LINUX_DISTRIBUTION: Unknown" in dump_text()


def test_old_numpy_is_refused(home, monkeypatch):
    monkeypatch.setattr(numpy, "__version__", "1.5.1")
    with pytest.raises(debugdump.DependencyError) as err:
        debugdump.DebugDump().reset()
    assert str(err.value) == repr("MakeHuman requires at least numpy version 1.6")


def test_append_gl_reports_settings(home, monkeypatch):
    import OpenGL
    monkeypatch.setattr(G, "args", {"noshaders": True})
    monkeypatch.setattr(G, "preStartupSettings", {"noShaders": False, "noSampleBuffers": True},
                        raising=False)
    dump = debugdump.DebugDump()
    dump.reset()
    dump.appendGL()
    lines = dump_text().splitlines()
    assert lines[-3:] == ["PYOPENGL.VERSION: %s" % OpenGL.__version__,
                          "NOSHADERS: set via command line", "NOSAMPLEBUFFERS: set via setting"]
    monkeypatch.setattr(G, "args", {})
    monkeypatch.setattr(G, "preStartupSettings", {"noShaders": True, "noSampleBuffers": False})
    dump.appendGL()
    assert dump_text().splitlines()[-2:] == ["NOSHADERS: set via setting", "NOSAMPLEBUFFERS: not set"]


@pytest.mark.parametrize("qt_conf", [None, "[Paths]\nPrefix = .\n"])
def test_append_qt(home, tmp_path, monkeypatch, qt_conf):
    import qtui
    workdir = tmp_path / "work"
    workdir.mkdir()
    if qt_conf:
        (workdir / "qt.conf").write_text(qt_conf, encoding="utf-8")
    monkeypatch.chdir(workdir)
    monkeypatch.delenv("QT_PLUGIN_PATH", raising=False)
    dump = debugdump.DebugDump()
    dump.reset()
    dump.appendQt()
    text = dump_text()
    assert "QT.VERSION: %s" % qtui.getQtVersionString() in text
    assert "QT.JPG_SUPPORT: " in text and "QT.SVG_SUPPORT: " in text
    assert "QT.PLUGIN_PATH_ENV: \n" in text
    if qt_conf:
        assert "QT.CONF: [Paths]\n         Prefix = ." in text
    else:
        assert "QT.CONF: NOT PRESENT" in text


def test_append_message(home):
    dump = debugdump.DebugDump()
    dump.reset()
    dump.appendMessage("GL.VENDOR: Apple")
    assert dump_text().endswith("GL.VENDOR: Apple\n")


def test_append_message_with_percent_sign(home):
    dump = debugdump.DebugDump()
    dump.reset()
    dump.appendMessage("GL.EXTENSION: 100% supported")
    assert dump_text().endswith("GL.EXTENSION: 100% supported\n")


def test_undecodable_arguments_are_skipped(home):
    class Undecodable(object):
        def __str__(self):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    dump = debugdump.DebugDump()
    dump.reset()
    dump.open()
    dump.write("BROKEN: %s", Undecodable())
    dump.write("AFTER: %s", "ok")
    dump.close()
    text = dump_text()
    assert "BROKEN" not in text and text.endswith("AFTER: ok\n")
