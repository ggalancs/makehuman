"""Running from a source checkout on macOS behaves like any other platform."""

import os
import sys

import pytest

import makehuman


@pytest.fixture(autouse=True)
def restore(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(makehuman, "stdout_filename", None)
    monkeypatch.setattr(makehuman, "stderr_filename", None)
    monkeypatch.setenv("PATH", os.environ["PATH"])
    cwd = os.getcwd()
    yield
    os.chdir(cwd)


def test_source_run_on_darwin_changes_to_the_source_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delattr(sys, "frozen", raising=False)
    os.chdir(tmp_path)
    makehuman.set_sys_path()
    assert os.path.realpath(os.getcwd()) == os.path.realpath(os.path.dirname(makehuman.__file__))
    assert sys.path[:2] == ["./", "./lib"]


def test_py2app_bundle_keeps_the_bundle_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    (tmp_path / "MacOS").mkdir()
    (tmp_path / "Resources" / "makehuman").mkdir(parents=True)
    os.chdir(tmp_path / "MacOS")
    makehuman.set_sys_path()
    assert os.path.realpath(os.getcwd()) == os.path.realpath(tmp_path / "Resources" / "makehuman")
    assert sys.path[0] == "./lib"


def test_source_run_on_darwin_keeps_the_terminal(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delattr(sys, "frozen", raising=False)
    makehuman.get_platform_paths()
    assert makehuman.stdout_filename is None and makehuman.stderr_filename is None


def test_app_bundle_on_darwin_logs_to_files(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    makehuman.get_platform_paths()
    assert makehuman.stdout_filename.endswith("makehuman-output.txt")
    assert makehuman.stderr_filename.endswith("makehuman-error.txt")


def test_redirected_streams_are_line_buffered(tmp_path, monkeypatch):
    monkeypatch.setattr(makehuman, "stdout_filename", str(tmp_path / "out.txt"))
    monkeypatch.setattr(makehuman, "stderr_filename", str(tmp_path / "err.txt"))
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    makehuman.redirect_standard_streams()
    try:
        print("progress line")
        print("error line", file=sys.stderr)
        # no flush: the text must already be on disk
        assert (tmp_path / "out.txt").read_text() == "progress line\n"
        assert (tmp_path / "err.txt").read_text() == "error line\n"
    finally:
        sys.stdout.close()
        sys.stderr.close()
