"""plugins/1_convert_model_tags.py: at startup, MHM files whose tags line uses the
old ",," separator are rewritten (tags separated by ";", spaces restored), after a
backup copy "<file>~" is made. Files are left alone when there is nothing to do
or when the backup cannot be made."""

import importlib.util
import logging
import os

import pytest

import getpath

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLUGIN = os.path.join(REPO, "makehuman", "plugins", "1_convert_model_tags.py")


@pytest.fixture
def plugin():
    spec = importlib.util.spec_from_file_location("convert_model_tags_under_test", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def models(tmp_path, monkeypatch):
    folder = tmp_path / "models"
    (folder / "sub").mkdir(parents=True)
    real = getpath.getPath
    monkeypatch.setattr(getpath, "getPath", lambda sub="": str(folder) if sub == "models" else real(sub))
    return folder


OLD = "version v1.1.0\ntags young,,woman tall\nname Ann\n"
NEW = "version v1.1.0\ntags young woman;tall\nname Ann\n"


def test_old_tags_are_converted_with_a_backup(plugin, models, caplog):
    path = models / "sub" / "ann.mhm"
    path.write_text(OLD)
    with caplog.at_level(logging.INFO):
        plugin.load(None)
    assert path.read_text() == NEW
    assert (models / "sub" / "ann.mhm~").read_text() == OLD
    assert "Successfully converted" in caplog.text


@pytest.mark.parametrize("text", [NEW, "version v1.1.0\nname Bob\n", "tags single\n"],
                         ids=["already-new", "no-tags", "one-tag"])
def test_files_without_old_tags_are_left_alone(plugin, models, text):
    path = models / "bob.mhm"
    path.write_text(text)
    plugin.load(None)
    assert path.read_text() == text
    assert not (models / "bob.mhm~").exists()


def test_existing_backup_is_never_overwritten(plugin, models, caplog):
    path = models / "ann.mhm"
    path.write_text(OLD)
    (models / "ann.mhm~").write_text("an older backup\n")
    plugin.load(None)
    assert path.read_text() == OLD  # not converted: the tags line has to be fixed by hand
    assert (models / "ann.mhm~").read_text() == "an older backup\n"
    assert "already exists" in caplog.text and "should be fixed in a text editor" in caplog.text


def test_failed_backup_leaves_the_file_unchanged(plugin, models, caplog, monkeypatch):
    path = models / "ann.mhm"
    path.write_text(OLD)

    def no_copy(src, dst):
        raise OSError("read-only folder")

    monkeypatch.setattr(plugin.shutil, "copy2", no_copy)
    plugin.load(None)  # the missing backup is not compared (that raised FileNotFoundError)
    assert path.read_text() == OLD
    assert not (models / "ann.mhm~").exists()
    assert "Failed to backup" in caplog.text


def test_unwritable_file_is_reported(plugin, models, caplog, monkeypatch):
    path = models / "ann.mhm"
    path.write_text(OLD)
    real_open = open

    def read_only_open(file, mode="r", *args, **kwargs):
        if str(file) == str(path) and "w" in mode:
            raise PermissionError("read-only file")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", read_only_open)
    plugin.load(None)
    assert path.read_text() == OLD
    assert "Cannot write changes" in caplog.text
