"""lib/targets.py: discovering morph targets in folders or in the targets.npz archive."""

import logging
import os
import zipfile

import pytest

import getpath
import targets

MH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "makehuman")

TREE = [
    "targets/macrodetails/universal-female-young-averagemuscle-averageweight.target",
    "targets/macrodetails/universal-male-young-averagemuscle-averageweight.target",
    "targets/nose/nose_trans_up.target",
    "targets/nose/nose-trans-up.PNG",
    "targets/nose/readme.txt",
    "poses/pose-a.target",
]


def make_tree(root):
    for rel in TREE:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0 0.0 0.0 0.0\n", encoding="utf-8")
    return str(root)


def keys(components):
    return sorted(tuple(c.key) for c in components)


@pytest.fixture
def tree(tmp_path):
    return make_tree(tmp_path / "data")


@pytest.fixture
def files_targets(tree):
    # No targets.npz in this folder: Targets falls back to crawling .target files
    return targets.Targets(tree)


def test_component_update_sorts_keys_from_variables():
    c = targets.Component()
    for part in ["universal", "female", "young", "maxmuscle", "target"]:
        c.update(part)
    assert c.key == ["universal"]
    assert c.data == {"gender": "female", "age": "young", "muscle": "maxmuscle"}
    assert sorted(c.getVariables()) == ["female", "maxmuscle", "young"]
    c.finish("x.target")
    assert c.path == "x.target"
    assert c.data["race"] is None and c.data["bodyproportions"] is None
    assert len(c.data) == 9  # every category is present after finishing
    assert repr(c) == repr((["universal"], c.data, "x.target"))


def test_component_clone_is_independent_child():
    parent = targets.Component()
    parent.update("nose")
    child = parent.clone()
    child.update("up")
    child.update("male")
    assert child.parent is parent and not child.isRoot() and parent.isRoot()
    assert parent.key == ["nose"] and parent.data == {}
    assert child.key == ["nose", "up"] and child.data == {"gender": "male"}


def test_component_rejects_conflicting_values():
    c = targets.Component()
    c.set_data("gender", "male")
    c.set_data("gender", "male")  # same value again is fine
    with pytest.raises(RuntimeError, match="gender already set"):
        c.set_data("gender", "female")


@pytest.mark.parametrize("method", ["list_dir", "is_dir", "is_file", "real_path"])
def test_crawler_base_is_abstract(method):
    with pytest.raises(NotImplementedError):
        getattr(targets.TargetsCrawler("."), method)("x")


def test_files_crawler_finds_targets_and_images(files_targets, tree):
    t = files_targets
    assert len(t.targets) == 4
    assert sorted(t.groups) == [("macrodetails", "universal"), ("nose", "nose", "trans", "up"),
                                ("poses", "pose", "a")]
    female, male = sorted(t.groups[("macrodetails", "universal")], key=lambda c: c.path)
    assert female.data["gender"] == "female" and male.data["gender"] == "male"
    assert female.path == os.path.join(tree, TREE[0]).replace("\\", "/")
    # images are indexed by lower-case file name; other files are ignored
    assert list(t.images) == ["nose-trans-up.png"]


def test_lookup_by_group_accepts_str_tuple_and_list(files_targets):
    expected = files_targets.groups[("nose", "nose", "trans", "up")]
    assert files_targets.getTargetsByGroup("nose-nose-trans-up") is expected
    assert files_targets.getTargetsByGroup(("nose", "nose", "trans", "up")) is expected
    assert files_targets.getTargetsByGroup(["nose", "nose", "trans", "up"]) is expected
    with pytest.raises(KeyError):
        files_targets.getTargetsByGroup("nose")


def test_find_targets_by_partial_group(files_targets):
    t = files_targets
    assert keys(t.findTargets("macrodetails")) == [("macrodetails", "universal")] * 2
    assert keys(t.findTargets(["nose"])) == [("nose", "nose", "trans", "up")]
    assert len(t.findTargets(())) == 4  # root lists everything
    assert t.findTargets("nothing-here") == []


def test_conflicting_file_name_aborts_crawl(tmp_path):
    bad = tmp_path / "data" / "targets" / "nose-male-female.target"
    bad.parent.mkdir(parents=True)
    bad.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="gender already set"):
        targets.FilesTargetsCrawler(str(tmp_path / "data")).findTargets()


def test_debug_output_lists_groups_and_dependencies(files_targets, caplog):
    caplog.set_level(logging.DEBUG)
    files_targets.debugGroups()
    assert "nose-nose-trans-up" in caplog.text
    caplog.clear()
    files_targets.debugTargets(showData=True)
    assert "Group: macrodetails-universal" in caplog.text
    assert "'gender': 'female'" in caplog.text
    assert "data: {" in caplog.text


def make_npz(folder, names):
    os.makedirs(folder, exist_ok=True)
    with zipfile.ZipFile(os.path.join(folder, "targets.npz"), "w") as z:
        for name in names:
            z.writestr(name, b"")


def test_zipped_crawler_reads_archive_and_image_list(tmp_path):
    data = str(tmp_path / "data")
    make_npz(data, ["targets/targets.license.npy",
                    "targets/nose/nose-trans-up.index.npy",
                    "targets/nose/nose-trans-up.vector.npy",
                    "targets/macrodetails/universal-female-young.index.npy",
                    "expression/units/smile.index.npy"])
    with open(os.path.join(data, "images.list"), "w", encoding="utf-8") as f:
        f.write(os.path.join(data, "targets", "nose", "nose-trans-up.png") + "\n")
        f.write(os.path.join(data, "targets", "nose", "notes.txt") + "\n")

    t = targets.Targets(data)
    assert sorted(t.groups) == [("expression", "units", "smile"), ("macrodetails", "universal"),
                                ("nose", "nose", "trans", "up")]
    [smile] = t.groups[("expression", "units", "smile")]
    assert smile.path == data + "/expression/units/smile.target"
    assert t.images == {"nose-trans-up.png": data + "/targets/nose/nose-trans-up.png"}


def test_zipped_crawler_tree_navigation(tmp_path):
    data = str(tmp_path / "data")
    make_npz(data, ["targets/a/b-c.index.npy"])
    open(os.path.join(data, "images.list"), "w").close()
    crawler = targets.ZippedTargetsCrawler(data, "targets.npz")
    assert sorted(crawler.list_dir(data)) == ["targets"]
    assert crawler.namei(".") == crawler.namei("") == {"targets": {"a": {"b-c.target": None}}}
    assert crawler.namei(["targets", "a"]) == {"b-c.target": None}
    assert crawler.lookupPath(data) == ""
    assert crawler.is_dir(data + "/targets/a")
    assert crawler.is_file(data + "/targets/a/b-c.target")
    assert not crawler.is_file(data + "/targets")


def test_zipped_crawler_requires_archive(tmp_path):
    with pytest.raises(Exception, match="npz archive"):
        targets.ZippedTargetsCrawler(str(tmp_path), "targets.npz")


def test_system_targets_come_from_the_npz(monkeypatch):
    monkeypatch.chdir(MH_DIR)
    monkeypatch.setattr(targets, "_targets", None)
    t = targets.getTargets()
    assert targets.getTargets() is t  # cached
    with zipfile.ZipFile(getpath.getSysDataPath("targets.npz")) as z:
        n_index = sum(1 for n in z.namelist() if n.endswith(".index.npy"))
    assert len(t.targets) == n_index
    universal = t.getTargetsByGroup("macrodetails-universal")
    assert "data/targets/macrodetails/universal-female-young-averagemuscle-averageweight.target" in \
        [c.path for c in universal]
    assert all(c.data["gender"] in ("male", "female") for c in universal)
    assert len(t.findTargets("macrodetails")) > len(universal)
    assert t.images["fedora_diffuse.png"].endswith("fedora_diffuse.png")
