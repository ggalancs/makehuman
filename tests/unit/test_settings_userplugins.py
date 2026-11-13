"""plugins/5_settings_userplugins.py without the GUI: unpacking a zipped plugin into
the user plugins folder (refusing unsafe or clashing archives) and listing the
user plugins."""

import importlib.util
import os
import zipfile

import pytest

import getpath

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLUGIN = os.path.join(REPO, "makehuman", "plugins", "5_settings_userplugins.py")


@pytest.fixture(scope="module")
def plugin():
    spec = importlib.util.spec_from_file_location("settings_userplugins_under_test", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_zip(path, members):
    with zipfile.ZipFile(str(path), "w") as zf:
        for name, text in members.items():
            zf.writestr(zipfile.ZipInfo(name), text)
    return str(path)


@pytest.fixture
def dest(tmp_path):
    folder = tmp_path / "plugins"
    folder.mkdir()
    return folder


def test_a_plugin_zip_is_unpacked(plugin, tmp_path, dest):
    archive = make_zip(tmp_path / "p.zip", {"9_p/__init__.py": "def load(app): pass\n", "9_p/data.txt": "x"})
    assert plugin.decompress(archive, str(dest)) == 0
    assert (dest / "9_p" / "__init__.py").read_text() == "def load(app): pass\n"
    assert (dest / "9_p" / "data.txt").is_file()


def test_not_a_zip(plugin, tmp_path, dest):
    fake = tmp_path / "fake.zip"
    fake.write_text("not a zip")
    assert plugin.decompress(str(fake), str(dest)) == 1


def test_missing_destination(plugin, tmp_path, dest):
    archive = make_zip(tmp_path / "p.zip", {"9_p/__init__.py": ""})
    assert plugin.decompress(archive, str(dest / "missing")) == 2


@pytest.mark.parametrize("member", ["../evil.py", "a/../../evil.py", "/abs/evil.py"],
                         ids=["parent", "nested-parent", "absolute"])
def test_unsafe_paths_are_refused_and_nothing_is_extracted(plugin, tmp_path, dest, member):
    archive = make_zip(tmp_path / "evil.zip", {"9_ok/__init__.py": "", member: "boom"})
    assert plugin.decompress(archive, str(dest)) == 3
    assert list(dest.iterdir()) == []
    assert not (tmp_path / "evil.py").exists()


def test_existing_files_are_never_overwritten(plugin, tmp_path, dest):
    (dest / "9_p").mkdir()
    (dest / "9_p" / "__init__.py").write_text("mine")
    archive = make_zip(tmp_path / "p.zip", {"9_p/__init__.py": "theirs"})
    assert plugin.decompress(archive, str(dest)) == 4
    assert (dest / "9_p" / "__init__.py").read_text() == "mine"


def test_user_plugins_are_packages_and_modules(plugin, tmp_path, dest, monkeypatch):
    (dest / "9_package").mkdir()
    (dest / "9_package" / "__init__.py").write_text("")
    (dest / "9_folder_without_init").mkdir()
    (dest / "_private").mkdir()
    (dest / "_private" / "__init__.py").write_text("")
    (dest / "9_module.py").write_text("")
    (dest / "_hidden.py").write_text("")
    (dest / "notes.txt").write_text("")
    monkeypatch.setattr(getpath, "getPath", lambda sub="": str(dest) if sub == "plugins" else sub)
    found = sorted(plugin.getUserPlugins())
    assert found == [("9_module", str(dest / "9_module.py")),
                     ("9_package", str(dest / "9_package" / "__init__.py"))]
