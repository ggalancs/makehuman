"""lib/mhversion.py: version, branch and commit from git or a VERSION file."""

import json

import pytest

import gitutils
import makehuman
import mhversion


@pytest.fixture
def no_git(monkeypatch, tmp_path):
    monkeypatch.setattr(gitutils, "findGitDir", lambda: None)
    monkeypatch.chdir(tmp_path)  # no data/VERSION here


@pytest.fixture
def fake_git(monkeypatch, tmp_path):
    monkeypatch.setattr(gitutils, "findGitDir", lambda: "/src/.git")
    monkeypatch.setattr(gitutils, "getCurrentBranch", lambda: "topic")
    monkeypatch.setattr(gitutils, "getCurrentCommit",
                        lambda short=True: "abcd1234" if short else "abcd1234" + "0" * 32)
    monkeypatch.chdir(tmp_path)


def test_defaults_without_git_or_version_file(no_git, monkeypatch):
    monkeypatch.setattr(makehuman, "release", False)
    v = mhversion.MHVersion()
    assert (v.currentBranch, v.currentShortCommit, v.currentLongCommit) == ("UNKNOWN",) * 3
    assert v.version == makehuman.getVersionDigitsStr()
    assert v.versionSub == makehuman.getVersionSubStr()
    assert v.getFullVersionStr() == "%s %s (UNKNOWN:UNKNOWN)" % (v.version, v.versionSub)
    assert v.fullTitle == "MakeHuman Community " + v.getFullVersionStr()


def test_release_title_is_short(no_git, monkeypatch):
    monkeypatch.setattr(makehuman, "release", True)
    v = mhversion.MHVersion()
    assert v.isRelease is True
    assert v.fullTitle == "MakeHuman Community " + makehuman.getVersionDigitsStr()


def test_git_checkout_is_never_a_release(fake_git, monkeypatch):
    monkeypatch.setattr(makehuman, "release", True)
    v = mhversion.MHVersion()
    assert v.isRelease is False
    assert (v.currentBranch, v.currentShortCommit) == ("topic", "abcd1234")
    assert v.currentLongCommit == "abcd1234" + "0" * 32
    assert v.fullTitle.endswith("(topic:abcd1234)")


def test_git_without_answers_keeps_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(gitutils, "findGitDir", lambda: "/src/.git")
    monkeypatch.setattr(gitutils, "getCurrentBranch", lambda: None)
    monkeypatch.setattr(gitutils, "getCurrentCommit", lambda short=True: None)
    monkeypatch.chdir(tmp_path)
    v = mhversion.MHVersion()
    assert (v.currentBranch, v.currentShortCommit, v.currentLongCommit) == ("UNKNOWN",) * 3


def test_version_file_overrides_everything(fake_git, tmp_path):
    path = tmp_path / "VERSION"
    path.write_text(json.dumps({
        "currentShortCommit": "feed0001", "currentLongCommit": "feed0001" * 5,
        "currentBranch": "release-1.3", "title": "MakeHuman Custom",
        "version": "9.9.9", "isRelease": True}), encoding="utf-8")
    v = mhversion.MHVersion(str(path))
    assert (v.currentBranch, v.currentShortCommit) == ("release-1.3", "feed0001")
    assert v.currentLongCommit == "feed0001" * 5
    assert v.fullTitle == "MakeHuman Custom 9.9.9"


def test_default_version_file_is_in_the_data_folder(no_git, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "VERSION").write_text('{"currentBranch": "from-data"}', encoding="utf-8")
    assert mhversion.MHVersion().currentBranch == "from-data"


def test_write_version_file_round_trips(fake_git, tmp_path):
    v = mhversion.MHVersion()
    v.title = "Special"
    out = tmp_path / "VERSION.json"
    v.writeVersionFile(str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == {"currentShortCommit": "abcd1234", "currentLongCommit": "abcd1234" + "0" * 32,
                    "currentBranch": "topic", "title": "Special",
                    "version": makehuman.getVersionDigitsStr(), "isRelease": False}
    again = mhversion.MHVersion(str(out))
    assert again.title == "Special" and again.currentBranch == "topic"


def test_write_version_file_leaves_out_unknowns(no_git, tmp_path):
    (tmp_path / "data").mkdir()
    v = mhversion.MHVersion()
    v.isRelease = None
    v.writeVersionFile()  # default: data/VERSION
    data = json.loads((tmp_path / "data" / "VERSION").read_text(encoding="utf-8"))
    assert data == {"version": makehuman.getVersionDigitsStr()}


def test_write_version_file_uses_its_own_path(no_git, tmp_path):
    path = tmp_path / "own.json"
    v = mhversion.MHVersion(str(path))
    v.writeVersionFile()
    assert json.loads(path.read_text(encoding="utf-8"))["isRelease"] == makehuman.isRelease()
