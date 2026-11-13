"""makehuman.py: version strings, arguments, licenses and the start-up sequence.

tests/unit/test_startup_paths.py covers set_sys_path() and the macOS paths.
"""

import os
import sys
import types

import numpy as np
import pytest

import makehuman
from core import G

MH_DIR = os.path.dirname(os.path.realpath(makehuman.__file__))


@pytest.fixture(autouse=True)
def restore(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(makehuman, "stdout_filename", None)
    monkeypatch.setattr(makehuman, "stderr_filename", None)
    monkeypatch.setenv("PATH", os.environ["PATH"])
    monkeypatch.chdir(os.getcwd())


# --- versions ---------------------------------------------------------------------------

def test_version_strings(monkeypatch):
    monkeypatch.setattr(makehuman, "__version__", "1.3.0")
    monkeypatch.setattr(makehuman, "versionSub", "alpha")
    assert makehuman.getVersionDigitsStr() == "1.3.0"
    assert makehuman.getVersionSubStr() == "alpha"
    assert makehuman._versionStr() == "1.3.0 alpha"
    assert makehuman.getVersion() == ["1", "3", "0"]
    assert makehuman.getShortVersion() == "alpha"
    assert makehuman.getShortVersion(noSub=True) == "v1.3.0"
    monkeypatch.setattr(makehuman, "versionSub", "Release Candidate")
    assert makehuman.getShortVersion() == "release_candidate"
    monkeypatch.setattr(makehuman, "versionSub", "")
    assert makehuman._versionStr() == "1.3.0"
    assert makehuman.getShortVersion() == "v1.3.0"
    assert makehuman.getBasemeshVersion() == "hm08"


def test_release_version_string_is_the_number(monkeypatch):
    monkeypatch.setattr(makehuman, "release", True)
    monkeypatch.setattr(makehuman, "versionSub", "final")
    assert makehuman.isRelease() is True
    assert makehuman.getVersionStr() == makehuman.getVersionDigitsStr() + " final"


def test_nightly_version_string_is_branch_and_commit(monkeypatch):
    import mhversion

    class Stub(object):
        currentBranch = "topic"
        currentShortCommit = "abc123"

    monkeypatch.setattr(mhversion, "MHVersion", Stub)
    monkeypatch.setattr(makehuman, "release", False)
    assert makehuman.getVersionStr() == "topic:abc123"
    monkeypatch.setattr(makehuman, "release", True)
    assert makehuman.getVersionStr(full=True) == "topic:abc123"


def test_build_detection_and_cwd(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert makehuman.isBuild() is False
    assert makehuman.getCwd() == MH_DIR
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/Applications/MH/bin/makehuman.exe")
    assert makehuman.isBuild() is True
    assert makehuman.getCwd() == "/Applications/MH/bin"


def test_frozen_build_prefers_its_own_libraries(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "makehuman.exe"))
    monkeypatch.setenv("PATH", "/usr/bin")
    makehuman.set_sys_path()
    assert os.path.realpath(os.getcwd()) == os.path.realpath(str(tmp_path))
    assert os.environ["PATH"] == os.pathsep.join([".", str(tmp_path), "/usr/bin"])
    assert sys.path[:6] == ["./", "./lib", "./apps", "./shared", "./apps/gui", "./core"]


def test_windows_logs_to_the_user_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("MH_HOME_LOCATION", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "win32")
    makehuman.get_platform_paths()
    assert makehuman.stdout_filename == os.path.join(str(tmp_path / "makehuman" / "v1py3"), "python_out.txt")
    assert makehuman.stderr_filename.endswith("python_err.txt")


def test_close_standard_streams(tmp_path, monkeypatch):
    out = open(str(tmp_path / "o"), "w")
    err = open(str(tmp_path / "e"), "w")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    makehuman.close_standard_streams()
    assert out.closed and err.closed


def test_make_user_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MH_HOME_LOCATION", str(tmp_path))
    import getpath
    makehuman.make_user_dir()
    for sub in ("", "data", "plugins"):
        assert os.path.isdir(getpath.getPath(sub))
    makehuman.make_user_dir()  # existing folders are fine


def test_init_logging(monkeypatch, caplog):
    import log
    calls = []
    monkeypatch.setattr(log, "init", lambda: calls.append("init"))
    with caplog.at_level("INFO"):
        makehuman.init_logging()
    assert calls == ["init"]
    assert [r.getMessage() for r in caplog.records] == ["Initialized logging"]


# --- debug dump --------------------------------------------------------------------------

def test_debug_dump_resets_the_dump(monkeypatch):
    import debugdump
    calls = []
    monkeypatch.setattr(debugdump.dump, "reset", lambda: calls.append("reset"))
    makehuman.debug_dump()
    assert calls == ["reset"]


def test_debug_dump_exits_on_missing_dependency(monkeypatch, capsys, caplog):
    import debugdump

    def old_numpy():
        raise debugdump.DependencyError("numpy too old")

    monkeypatch.setattr(debugdump.dump, "reset", old_numpy)
    with pytest.raises(SystemExit) as stop:
        makehuman.debug_dump()
    assert stop.value.code == -1
    assert "Dependency error: 'numpy too old'" in capsys.readouterr().err
    assert any("Dependency error" in r.getMessage() for r in caplog.records)


def test_debug_dump_other_errors_are_logged(monkeypatch, caplog):
    import debugdump

    def broken():
        raise IOError("disk full")

    monkeypatch.setattr(debugdump.dump, "reset", broken)
    makehuman.debug_dump()  # does not stop start-up
    assert any(r.getMessage() == "Could not create debug dump" and r.exc_info
               for r in caplog.records)


# --- command line ------------------------------------------------------------------------

def test_no_arguments(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["makehuman.py"])
    assert makehuman.parse_arguments() == {}


def test_flags_and_model_file(monkeypatch):
    monkeypatch.setattr(makehuman, "release", False)
    monkeypatch.setattr(sys, "argv", ["makehuman.py", "--noshaders", "--debugnumpy", "-t", "my.mhm"])
    args = makehuman.parse_arguments()
    assert args == {"license": False, "noshaders": True, "multisampling": False,
                    "debugopengl": False, "fullloggingopengl": False, "debugnumpy": True,
                    "runtests": True, "mhmFile": "my.mhm"}


def test_release_has_no_test_option(monkeypatch, capsys):
    monkeypatch.setattr(makehuman, "release", True)
    monkeypatch.setattr(sys, "argv", ["makehuman.py", "--runtests"])
    with pytest.raises(SystemExit) as stop:
        makehuman.parse_arguments()
    assert stop.value.code == 2
    assert "unrecognized arguments: --runtests" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["makehuman.py", "--multisampling"])
    assert "runtests" not in makehuman.parse_arguments()


def test_license_option_prints_and_exits(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["makehuman.py", "--license"])
    with pytest.raises(SystemExit) as stop:
        makehuman.parse_arguments()
    assert stop.value.code == 0
    assert capsys.readouterr().out == "\n" + makehuman.getCopyrightMessage() + "\n\n"


def test_version_option(monkeypatch, capsys):
    monkeypatch.setattr(makehuman, "release", True)
    monkeypatch.setattr(sys, "argv", ["makehuman.py", "--version"])
    with pytest.raises(SystemExit) as stop:
        makehuman.parse_arguments()
    assert stop.value.code == 0
    assert capsys.readouterr().out.strip() == makehuman.getVersionStr()


def test_copyright_messages():
    short = makehuman.getCopyrightMessage(short=True)
    assert short.startswith("MakeHuman Copyright (C) 2001-2020")
    assert short.endswith("use the option --license")
    full = makehuman.getCopyrightMessage()
    assert "GNU Affero General Public License" in full and len(full) > len(short)


# --- asset licenses --------------------------------------------------------------------------

def test_default_asset_license():
    lic = makehuman.getAssetLicense()
    assert lic.asDict() == {"author": "MakeHuman Team", "license": "CC0",
                            "copyright": "(c) www.makehumancommunity.org 2001-2020",
                            "homepage": "http://www.makehumancommunity.org"}
    assert lic.properties == ["author", "license", "copyright", "homepage"]
    assert not lic.isCustomized()
    assert str(lic).splitlines() == ["MakeHuman asset license:", "Author: MakeHuman Team",
                                     "License: CC0",
                                     "Copyright: (c) www.makehumancommunity.org 2001-2020",
                                     "Homepage: http://www.makehumancommunity.org"]


def test_asset_license_from_properties_is_not_customized():
    lic = makehuman.getAssetLicense({"author": "Jane", "license": "CC-BY"})
    assert (lic.author, lic.license) == ("Jane", "CC-BY")
    assert not lic.isCustomized()


def test_changing_a_license_marks_it_customized():
    lic = makehuman.LicenseInfo()
    lic.author = "MakeHuman Team"  # same value
    assert not lic.isCustomized()
    lic.author = "Jane"
    assert lic.author == "Jane" and lic.isCustomized()
    lic.setProperty("unknown", "ignored")
    assert not hasattr(lic, "unknown")
    lic._customized = False  # internal attributes can be set
    assert not lic.isCustomized()
    with pytest.raises(KeyError):
        lic.brandNewProperty = 1


def test_license_from_json_and_copy():
    lic = makehuman.LicenseInfo().fromJson({"license": "AGPL", "name": "not a license key"})
    assert lic.license == "AGPL" and lic.isCustomized()
    dup = lic.copy()
    assert dup.asDict() == lic.asDict() and dup.isCustomized()
    dup.author = "Other"
    assert lic.author == "MakeHuman Team"


@pytest.mark.parametrize("line, key, value", [
    ("# author Jane Doe", "author", "Jane Doe"),
    ("// license: CC-BY 4.0", "license", "CC-BY 4.0"),
    ("homepage http://example.invalid", "homepage", "http://example.invalid"),
])
def test_license_from_comment_lines(line, key, value):
    lic = makehuman.LicenseInfo()
    lic.updateFromComment(line)
    assert getattr(lic, key) == value


def test_comment_without_words_changes_nothing():
    lic = makehuman.LicenseInfo()
    lic.updateFromComment("#   ")
    lic.updateFromComment("# unrelated words")
    assert not lic.isCustomized()


def test_numpy_string_round_trip_with_unicode():
    lic = makehuman.LicenseInfo().fromDict({"author": "José 山田", "license": "CC0"})
    packed = lic.toNumpyString()
    text, index = packed
    assert text.dtype == np.dtype("S1") and index.dtype == np.uint32
    assert makehuman.LicenseInfo().fromNumpyString(packed).asDict() == lic.asDict()
    assert makehuman.LicenseInfo().fromNumpyString(text, index).author == "José 山田"


def test_numpy_string_survives_np_save(tmp_path):
    lic = makehuman.LicenseInfo().fromDict({"copyright": "© 2026"})
    np.save(str(tmp_path / "l.npy"), lic.toNumpyString())
    loaded = np.load(str(tmp_path / "l.npy"), allow_pickle=True)
    assert makehuman.LicenseInfo().fromNumpyString(loaded).copyright == "© 2026"


# --- software licenses -------------------------------------------------------------------------

def test_wordwrap():
    long_line = " ".join(["word%02d" % i for i in range(30)])  # 209 characters
    text = makehuman._wordwrap("short line\n" + long_line + "\n\n\n", chars_per_line=40)
    lines = text.split("\n")
    assert lines[0] == "short line"
    assert all(len(l) <= 40 for l in lines)
    assert " ".join(lines[1:]) == long_line
    assert not text.endswith("\n")


def test_software_license_from_the_source_folder(monkeypatch):
    monkeypatch.chdir(MH_DIR)
    plain = makehuman.getSoftwareLicense()
    assert plain.startswith("LICENSE\n=======\n")
    assert "B. The license for the source code as such" in plain
    assert all(len(l) <= 80 for l in plain.split("\n"))
    assert makehuman.getSoftwareLicense(richtext=True).startswith("<h2>MakeHuman software license</h2>")


def test_missing_software_license(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert makehuman.getSoftwareLicense().startswith("Error: License file license.txt is not found")
    assert '<span style="color: red;">' in makehuman.getSoftwareLicense(richtext=True)


def test_third_party_licenses(monkeypatch):
    monkeypatch.chdir(MH_DIR)
    plain = makehuman.getThirdPartyLicenses()
    for name in ("PyQt5", "Qt5", "Numpy", "PyOpenGL", "Transformations", "pyFBX"):
        assert name + " (" in plain
    assert "Error" not in plain
    rich = makehuman.getThirdPartyLicenses(richtext=True)
    assert rich.startswith("<h2>Third-party licenses</h2>")
    assert '<a id="Numpy"><h3>Numpy</h3></a>' in rich and "<hr" in rich


def test_third_party_licenses_incomplete(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert makehuman.getThirdPartyLicenses().endswith("external licenses folder is not found, "
                                                      "this is an incomplete MakeHuman distribution!")
    (tmp_path / "licenses").mkdir()
    (tmp_path / "licenses" / "numpy-license.txt").write_text("Numpy terms", encoding="utf-8")
    text = makehuman.getThirdPartyLicenses(richtext=True)
    assert "Numpy terms" in text
    assert text.count('<span style="color: red;">Error: License file') == 5


def test_credits(tmp_path, monkeypatch):
    monkeypatch.chdir(MH_DIR)
    assert makehuman.getCredits().strip()
    assert makehuman.getCredits(richtext=True).endswith('<hr style="border: 1px solid #ffa02f;">')
    monkeypatch.chdir(tmp_path)
    assert makehuman.getCredits().startswith("Error: Credits file")
    assert makehuman.getCredits(richtext=True).startswith('<span style="color: red;">')


# --- main() ------------------------------------------------------------------------------------

@pytest.fixture
def startup(monkeypatch):
    """Replace every start-up step so main() can be followed without opening the GUI."""
    steps = []
    for name in ("set_sys_path", "make_user_dir", "get_platform_paths",
                 "redirect_standard_streams", "init_logging", "debug_dump",
                 "close_standard_streams"):
        monkeypatch.setattr(makehuman, name, (lambda n: lambda: steps.append(n))(name))
    args = {"debugnumpy": False}
    monkeypatch.setattr(makehuman, "parse_arguments", lambda: steps.append("parse_arguments") or args)

    class App(object):
        def __init__(self):
            steps.append("MHApplication")

        def run(self):
            steps.append("run")

    monkeypatch.setitem(sys.modules, "mhmain", types.SimpleNamespace(MHApplication=App))
    monkeypatch.setattr(G, "args", G.args)
    for var in ("MH_VERSION", "MH_SHORT_VERSION", "MH_MESH_VERSION", "MH_FROZEN", "MH_RELEASE"):
        monkeypatch.delenv(var, raising=False)
    old_err = np.geterr()
    yield steps, args
    np.seterr(**old_err)


def test_main_runs_the_start_up_sequence(startup, monkeypatch, capsys):
    steps, args = startup
    monkeypatch.setattr(makehuman, "getVersionStr", lambda: "topic:abc")
    monkeypatch.delattr(sys, "frozen", raising=False)
    np.seterr(all="raise")
    makehuman.main()
    assert steps == ["set_sys_path", "make_user_dir", "get_platform_paths",
                     "redirect_standard_streams", "parse_arguments", "init_logging",
                     "debug_dump", "MHApplication", "run", "close_standard_streams"]
    assert os.environ["MH_VERSION"] == "topic:abc"
    assert os.environ["MH_SHORT_VERSION"] == makehuman.getShortVersion()
    assert os.environ["MH_MESH_VERSION"] == "hm08"
    assert os.environ["MH_FROZEN"] == "No"
    assert os.environ["MH_RELEASE"] == ("Yes" if makehuman.isRelease() else "No")
    assert G.args is args
    assert np.geterr()["divide"] == "ignore"  # numpy warnings silenced
    assert capsys.readouterr().out.startswith(makehuman.getCopyrightMessage(short=True))


def test_main_keeps_numpy_errors_when_debugging(startup, monkeypatch):
    steps, args = startup
    args["debugnumpy"] = True
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    np.seterr(all="raise")
    makehuman.main()
    assert np.geterr()["divide"] == "raise"
    assert os.environ["MH_FROZEN"] == "Yes"


def test_main_stops_when_start_up_fails(startup, monkeypatch, capsys):
    steps, _ = startup

    def broken():
        raise OSError("read-only home")

    monkeypatch.setattr(makehuman, "make_user_dir", broken)
    makehuman.main()
    assert steps == ["set_sys_path"]
    err = capsys.readouterr().err
    assert "error: read-only home" in err and "Traceback" in err
