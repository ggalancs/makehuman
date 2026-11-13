"""lib/xdg_parser.py: XDG user folders on Linux, nothing elsewhere."""

import os
import runpy
import sys

import xdg_parser


def run_as(monkeypatch, platform, home, run_name="xdg_under_test"):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home), 1))
    monkeypatch.setenv("HOME", str(home))
    return runpy.run_path(xdg_parser.__file__, run_name=run_name)


def write_user_dirs(home):
    (home / "Docs").mkdir()
    (home / "Pictures").mkdir()
    (home / ".config").mkdir()
    (home / ".config" / "user-dirs.dirs").write_text(
        "# written by xdg-user-dirs-update\n"
        'XDG_DOCUMENTS_DIR="$HOME/Docs"\n'
        'XDG_PICTURES_DIR="$HOME/Pictures"\n'
        'XDG_MUSIC_DIR="$HOME/NoSuchFolder"\n'
        "SOMETHING_ELSE=1\n", encoding="utf-8")


def test_linux_reads_existing_user_dirs(tmp_path, monkeypatch):
    write_user_dirs(tmp_path)
    ns = run_as(monkeypatch, "linux", tmp_path)
    assert ns["isLinux"] is True
    assert ns["CONFIG_PATH"] == str(tmp_path / ".config" / "user-dirs.dirs")
    # Variables are expanded; folders that do not exist are left out
    assert ns["XDG_PATHS"] == {"DOCUMENTS": str(tmp_path / "Docs"),
                               "PICTURES": str(tmp_path / "Pictures")}
    assert ns["XDG_DIRS"] == ["DOCUMENTS", "PICTURES"]


def test_linux_without_config_file(tmp_path, monkeypatch):
    ns = run_as(monkeypatch, "linux", tmp_path)
    assert ns["XDG_PATHS"] == {}
    assert ns["XDG_DIRS"] == []
    assert ns["get_userdirs"](str(tmp_path / "missing")) == {}


def test_other_platforms_have_no_user_dirs(tmp_path, monkeypatch):
    write_user_dirs(tmp_path)
    ns = run_as(monkeypatch, "darwin", tmp_path)
    assert ns["isLinux"] is False
    assert ns["get_userdirs"](str(tmp_path / ".config" / "user-dirs.dirs")) == {}
    assert ns["XDG_DIRS"] == []


def test_main_prints_a_table_on_linux(tmp_path, monkeypatch, capsys):
    write_user_dirs(tmp_path)
    run_as(monkeypatch, "linux", tmp_path, run_name="__main__")
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "Found XDG_DIRS: DOCUMENTS, PICTURES "
    assert "DOCUMENTS  --->   %s" % (tmp_path / "Docs") in out
    assert "PICTURES   --->   %s" % (tmp_path / "Pictures") in out


def test_main_elsewhere_explains_itself(tmp_path, monkeypatch, capsys):
    run_as(monkeypatch, "darwin", tmp_path, run_name="__main__")
    assert capsys.readouterr().out == "This module should be used on Linux platforms only!\n"


def test_imported_module_matches_this_platform():
    if sys.platform.startswith("linux"):
        assert xdg_parser.isLinux
    else:
        assert not xdg_parser.isLinux and xdg_parser.XDG_PATHS == {} and xdg_parser.XDG_DIRS == []
