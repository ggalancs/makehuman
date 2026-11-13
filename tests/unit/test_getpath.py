"""lib/getpath.py: home folder discovery per platform and path helpers."""

import os
import runpy
import sys
import types

import pytest

import getpath


def real(path):
    return getpath.formatPath(os.path.realpath(str(path)))


# --- the optional makehuman.conf read at import time -----------------------------

def run_module(monkeypatch, platform, home):
    # The module's own default arguments ask for the home folder at import time
    monkeypatch.delenv("MH_HOME_LOCATION", raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home), 1))
    return runpy.run_path(getpath.__file__, run_name="getpath_under_test")


@pytest.mark.parametrize("platform, conf", [
    ("linux", ".config/makehuman.conf"),
    ("darwin", "Library/Application Support/MakeHuman/makehuman.conf"),
])
def test_conf_file_moves_the_home_folder(tmp_path, monkeypatch, platform, conf):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / conf).parent.mkdir(parents=True)
    (tmp_path / conf).write_text(str(target) + "\n  ignored second line\n", encoding="utf-8")
    ns = run_module(monkeypatch, platform, tmp_path)
    assert ns["configFile"] == str(tmp_path / conf)
    assert ns["getHomePath"]() == os.path.normpath(str(target))


def test_conf_file_on_windows_lives_in_local_appdata(tmp_path, monkeypatch):
    target = tmp_path / "mh home"
    target.mkdir()
    (tmp_path / "makehuman.conf").write_text(str(target), encoding="utf-8")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    ns = run_module(monkeypatch, "win32", tmp_path)
    assert ns["configFile"] == os.path.join(str(tmp_path), "makehuman.conf")
    assert ns["getHomePath"]() == os.path.normpath(str(target))


def test_conf_file_pointing_nowhere_is_ignored(tmp_path, monkeypatch):
    (tmp_path / ".config").mkdir()
    (tmp_path / ".config" / "makehuman.conf").write_text(str(tmp_path / "missing"), encoding="utf-8")
    ns = run_module(monkeypatch, "linux", tmp_path)
    assert ns["configPath"] == str(tmp_path / "missing")
    assert ns["getHomePath"]() == str(tmp_path)  # the platform default, not the missing folder


def test_unknown_platform_has_no_conf_file(tmp_path, monkeypatch):
    ns = run_module(monkeypatch, "sunos5", tmp_path)
    assert ns["configFile"] == ""
    assert ns["configPath"] == ""
    assert ns["getHomePath"]() == str(tmp_path)


# --- getHomePath / getPath per platform ---------------------------------------------

@pytest.fixture
def no_home_override(monkeypatch):
    monkeypatch.delenv("MH_HOME_LOCATION", raising=False)
    monkeypatch.setattr(getpath, "__home_path", None)


def test_env_variable_overrides_and_is_cached(tmp_path, monkeypatch, no_home_override):
    monkeypatch.setenv("MH_HOME_LOCATION", str(tmp_path))
    assert getpath.getHomePath() == getpath.formatPath(str(tmp_path))
    monkeypatch.delenv("MH_HOME_LOCATION")
    assert getpath.getHomePath() == getpath.formatPath(str(tmp_path))  # cached


def test_darwin_home_is_the_user_folder(monkeypatch, no_home_override):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert getpath.getHomePath() == os.path.expanduser("~")


def test_linux_home_prefers_xdg_documents(tmp_path, monkeypatch, no_home_override):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(getpath, "XDG_PATHS", {"DOCUMENTS": str(tmp_path)})
    assert getpath.getHomePath() == str(tmp_path)


def test_linux_home_without_xdg_documents(monkeypatch, no_home_override):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(getpath, "XDG_PATHS", {})
    assert getpath.getHomePath() == os.path.expanduser("~")


class FakeKey(object):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_winreg(value=None, type_=None):
    reg = types.ModuleType("winreg")
    reg.HKEY_CURRENT_USER = "HKCU"
    reg.REG_SZ, reg.REG_EXPAND_SZ = 1, 2
    reg.opened = []

    def OpenKey(root, name):
        reg.opened.append((root, name))
        return FakeKey()

    def QueryValueEx(key, name):
        assert name == "Personal"
        if value is None:
            raise FileNotFoundError(name)
        return value, type_

    reg.OpenKey = OpenKey
    reg.QueryValueEx = QueryValueEx
    reg.ExpandEnvironmentStrings = lambda s: s.replace("%USERPROFILE%", "C:\\Users\\me")
    return reg


@pytest.mark.parametrize("value, type_, expected", [
    ("D:\\Docs", 1, "D:/Docs"),
    ("%USERPROFILE%\\My Docs", 2, "C:/Users/me/My Docs"),
    (None, None, "C:/Users/me/Documents"),
])
def test_windows_home_comes_from_the_registry(monkeypatch, no_home_override, value, type_, expected):
    reg = fake_winreg(value, type_)
    monkeypatch.setitem(sys.modules, "winreg", reg)
    monkeypatch.setattr(sys, "platform", "win32")
    assert getpath.getHomePath() == expected
    assert reg.opened == [("HKCU", r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders")]


def test_windows_home_with_unexpected_registry_type(monkeypatch, no_home_override):
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg("x", 7))
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="user folder"):
        getpath.getHomePath()


@pytest.mark.parametrize("platform, tail", [
    ("win32", "/makehuman/v1py3"),
    ("darwin", "/Documents/MakeHuman/v1py3"),
    ("linux", "/makehuman/v1py3"),
])
def test_user_folder_layout_per_platform(tmp_path, monkeypatch, platform, tail):
    monkeypatch.setenv("MH_HOME_LOCATION", str(tmp_path))
    monkeypatch.setattr(sys, "platform", platform)
    home = getpath.formatPath(str(tmp_path))
    assert getpath.getPath() == home + tail
    assert getpath.getPath("plugins") == home + tail + "/plugins"
    assert getpath.getDataPath() == home + tail + "/data"
    assert getpath.getDataPath("skins") == home + tail + "/data/skins"


def test_system_paths_are_relative_to_the_working_folder():
    assert getpath.getSysPath() == "."
    assert getpath.getSysPath("licenses") == "licenses"
    assert getpath.getSysDataPath() == "data"
    assert getpath.getSysDataPath("skins/default.mhmat") == "data/skins/default.mhmat"


# --- string helpers ---------------------------------------------------------------------

def test_path_to_unicode():
    assert getpath.pathToUnicode("already text") == "already text"
    assert getpath.pathToUnicode("caf\u00e9".encode("utf-8")) == "caf\u00e9"


@pytest.mark.parametrize("patch_target", ["getfilesystemencoding", "getdefaultencoding", "locale"])
def test_path_to_unicode_tries_other_encodings(monkeypatch, patch_target):
    latin = "caf\u00e9".encode("latin-1")  # not valid UTF-8
    if patch_target == "locale":
        import locale
        monkeypatch.setattr(locale, "getpreferredencoding", lambda *a: "latin-1")
    else:
        for name in ("getfilesystemencoding", "getdefaultencoding"):
            monkeypatch.setattr(sys, name, lambda: "utf-8")
        monkeypatch.setattr(sys, patch_target, lambda: "latin-1")
    assert getpath.pathToUnicode(latin) == "caf\u00e9"


def test_undecodable_bytes_are_returned_unchanged(monkeypatch):
    import locale
    monkeypatch.setattr(sys, "getfilesystemencoding", lambda: "utf-8")
    monkeypatch.setattr(sys, "getdefaultencoding", lambda: "utf-8")
    monkeypatch.setattr(locale, "getpreferredencoding", lambda *a: "utf-8")
    assert getpath.pathToUnicode(b"\xff\xfe") == b"\xff\xfe"


def test_format_path():
    assert getpath.formatPath(None) is None
    assert getpath.formatPath("a/./b/../c") == "a/c"
    assert getpath.formatPath("a\\b") == "a/b"


def test_canonical_and_local_paths(tmp_path, monkeypatch):
    (tmp_path / "data" / "skins").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert getpath.canonicalPath("data/skins") == real(tmp_path / "data" / "skins")
    assert getpath.localPath(str(tmp_path / "data" / "skins")) == "data/skins"


def test_commonprefix_and_subpaths(tmp_path):
    assert getpath.commonprefix(["/a/b/c", "/a/b/d"]) == "/a/b"
    assert getpath.commonprefix(["/a/bc", "/a/b"]) == "/a"  # not a string prefix
    (tmp_path / "b" / "c").mkdir(parents=True)
    (tmp_path / "bc").mkdir()
    assert getpath.isSubPath(str(tmp_path / "b" / "c"), str(tmp_path / "b"))
    assert not getpath.isSubPath(str(tmp_path / "bc"), str(tmp_path / "b"))
    assert getpath.isSamePath(str(tmp_path / "b" / ".." / "bc"), str(tmp_path / "bc"))
    assert not getpath.isSamePath(str(tmp_path / "b"), str(tmp_path / "bc"))


# --- relative paths and file search -----------------------------------------------------

@pytest.fixture
def tree(tmp_path):
    for rel in ["user/skins/a.mhmat", "sys/skins/a.mhmat", "sys/skins/b.mhmat",
                "sys/skins/b.thumb", "sys/skins/young/c.MHMAT", "outside/x.mhmat"]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rel)
    return tmp_path


def test_relative_path_uses_the_first_matching_root(tree):
    user, sys_ = str(tree / "user"), str(tree / "sys")
    assert getpath.getRelativePath(str(tree / "sys/skins/b.mhmat"), [user, sys_]) == "skins/b.mhmat"
    assert getpath.getRelativePath(str(tree / "user/skins/a.mhmat"), user) == "skins/a.mhmat"
    outside = str(tree / "outside/x.mhmat")
    assert getpath.getRelativePath(outside, [user, sys_]) == outside
    assert getpath.getRelativePath(outside, [user, sys_], strict=True) is None


def test_find_file_is_the_inverse(tree):
    user, sys_ = str(tree / "user"), str(tree / "sys")
    assert getpath.findFile("skins/a.mhmat", [user, sys_]) == getpath.formatPath(user + "/skins/a.mhmat")
    assert getpath.findFile("skins/b.mhmat", [user, sys_]) == getpath.formatPath(sys_ + "/skins/b.mhmat")
    assert getpath.findFile("skins/b.mhmat", sys_) == getpath.formatPath(sys_ + "/skins/b.mhmat")
    assert getpath.findFile("skins/zzz.mhmat", [user, sys_]) == "skins/zzz.mhmat"
    assert getpath.findFile("skins/zzz.mhmat", [user, sys_], strict=True) is None


def test_thorough_find_file(tree, monkeypatch):
    sys_ = str(tree / "sys")
    found = getpath.thoroughFindFile("skins/b.mhmat", sys_, searchDefaultPaths=False)
    assert found == real(tree / "sys/skins/b.mhmat")
    # Absolute paths are accepted as they are
    absolute = str(tree / "outside/x.mhmat")
    assert getpath.thoroughFindFile(absolute, [], searchDefaultPaths=False) == real(absolute)
    # A leading data/ folder is stripped when the data folder is searched directly
    assert getpath.thoroughFindFile("data/skins/b.mhmat", [sys_], searchDefaultPaths=False) == \
        real(tree / "sys/skins/b.mhmat")
    # Nothing found: the most probable name comes back
    assert getpath.thoroughFindFile("data/skins/none.mhmat", [sys_], False) == "data/skins/none.mhmat"


def test_thorough_find_file_searches_default_paths(tree, monkeypatch):
    monkeypatch.chdir(tree / "sys")  # getSysPath() is "."
    assert getpath.thoroughFindFile("skins/b.mhmat") == real(tree / "sys/skins/b.mhmat")


def test_thorough_find_file_accepts_windows_separators(tree):
    found = getpath.thoroughFindFile("skins\\b.mhmat", [str(tree / "sys")], searchDefaultPaths=False)
    assert found == real(tree / "sys/skins/b.mhmat")


def rel(paths, root):
    return sorted(os.path.relpath(p, str(root)).replace(os.sep, "/") for p in paths)


def test_search_recursive(tree):
    found = getpath.search(str(tree / "sys"), ".mhmat")
    assert rel(found, tree) == ["sys/skins/a.mhmat", "sys/skins/b.mhmat", "sys/skins/young/c.MHMAT"]


def test_search_flat_skips_missing_folders(tree):
    found = getpath.search([str(tree / "sys/skins"), str(tree / "missing")], ["mhmat", "thumb"],
                           recursive=False)
    assert rel(found, tree) == ["sys/skins/a.mhmat", "sys/skins/b.mhmat", "sys/skins/b.thumb"]


@pytest.mark.parametrize("recursive", [True, False])
@pytest.mark.parametrize("preferred", ["thumb", "mhmat"])  # either listing order hits the swap
def test_search_mutex_extensions_keeps_the_preferred_one(tree, recursive, preferred):
    folder = str(tree / "sys/skins") if not recursive else str(tree / "sys")
    exts = [preferred] + [e for e in ("thumb", "mhmat") if e != preferred]
    found = getpath.search(folder, exts, recursive=recursive, mutexExtensions=True)
    expected = ["sys/skins/a.mhmat", "sys/skins/b.%s" % preferred]
    if recursive:
        expected.append("sys/skins/young/c.MHMAT")  # the extension check ignores case
    assert rel(found, tree) == expected


def test_search_mutex_extensions_with_uppercase_extension(tmp_path):
    (tmp_path / "d.MHMAT").write_text("x")
    (tmp_path / "d.thumb").write_text("x")
    found = list(getpath.search(str(tmp_path), ["thumb", "mhmat"], mutexExtensions=True))
    assert rel(found, tmp_path) == ["d.thumb"]


def test_jailed_path(tree):
    jail = [str(tree / "user"), str(tree / "sys")]
    b = str(tree / "sys/skins/b.mhmat")
    assert getpath.getJailedPath(b, str(tree / "sys/skins"), jail) == "b.mhmat"
    # Not below relativeTo, but inside the jail: relative to the jail root
    assert getpath.getJailedPath(b, str(tree / "user/skins"), jail) == "skins/b.mhmat"
    assert getpath.getJailedPath(str(tree / "outside/x.mhmat"), str(tree / "sys"), jail) is None
