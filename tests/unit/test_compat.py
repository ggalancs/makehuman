"""apps/compat.py: loading .mhm files saved by older MakeHuman releases."""

import os

import pytest

import compat
import progress
from core import G

MH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "makehuman")


class FakeApp(object):
    def __init__(self):
        self.calls = []

    def progress(self, *args, **kwargs):
        self.calls.append(args)


@pytest.fixture(autouse=True)
def mh_env(monkeypatch):
    monkeypatch.chdir(MH_DIR)  # the modifier mapping is read from data/ relative to cwd
    monkeypatch.setattr(G, "app", FakeApp(), raising=False)
    monkeypatch.setattr(progress, "current_Progress_", None)


@pytest.fixture
def loader():
    # Fresh instance: the class caches the CSV mapping on first use
    return compat.MHM10Loader()


def run(loader, line):
    out = []
    loader.loadProperty(line.split(), out.append, False)
    return out


@pytest.mark.parametrize("text, expected", [
    ("v1.0.2", [1, 0, 2]),
    ("1.1", [1, 1]),
    (" V1.2 ", [1, 2]),
    ("1.0.0.7", [1, 0, 0]),
    ("1.x.3", [1]),
    ("alpha", None),
    ("", None),
])
def test_parse_version(text, expected):
    assert compat._parse_version(text) == expected


@pytest.mark.parametrize("version, cls", [
    ([1, 0, 3], compat.MHM10Loader),
    ([1, 0], compat.MHM10Loader),
    ([1, 1], compat.MHM11Loader),
    ([1, 2, 0], compat.MHM12Loader),
])
def test_get_loader_by_version(version, cls):
    assert isinstance(compat.getMHMLoader(version), cls)


@pytest.mark.parametrize("version", [[2, 0], [1], [0, 9]])
def test_get_loader_rejects_unknown_versions(version):
    with pytest.raises(RuntimeError, match="No suitable MHM"):
        compat.getMHMLoader(version)


def test_abstract_loader_requires_implementation():
    base = compat.MHMLoader()
    with pytest.raises(NotImplementedError):
        base.loadProperty([], None)
    with pytest.raises(NotImplementedError):
        base.getAcceptedVersion()


def test_modifier_mapping_is_read_from_csv_and_cached(loader):
    mapping = loader.getModifierMapping()
    assert mapping["macrodetails/Gender"] == ("macrodetails/Gender", False)
    assert mapping["armslegs/l-foot-trans-forward|backward"] == \
        ("armslegs/l-foot-trans-backward|forward", True)
    assert "Original modifier name (1.0)" not in mapping  # header row dropped
    assert loader.getModifierMapping() is mapping


@pytest.mark.parametrize("savename, target, expected", [
    ("macro", "Gender", "macrodetails/Gender"),            # explicit mapping table
    ("gendered", "BreastSize", "breast/BreastSize"),
    ("measure", "bust", "measure/measure-bust-decrease|increase"),
    ("armslegs", "l-foot-trans-forward-backward", "armslegs/l-foot-trans-forward|backward"),  # two-token prefix
    ("face", "r-eye-move-in-out", "eyes/r-eye-move-in|out"),
    ("torso", "bulge-decr-incr", "pelvis/bulge-decr|incr"),  # one-token prefix
    ("face", "nose-scale-vert-incr-decr", "nose/nose-scale-vert-incr|decr"),  # no prefix mapping
    ("face", "mouth-open", "mouth/mouth-open"),            # no trailing pair
])
def test_target_to_modifier(loader, savename, target, expected):
    assert loader.targetToModifier(savename, target) == expected


def test_map_modifier_inverts_when_flagged(loader):
    assert loader.mapModifier("armslegs/l-foot-trans-forward|backward", 0.25) == \
        ("armslegs/l-foot-trans-backward|forward", -0.25)
    assert loader.mapModifier("macrodetails/Age", 0.5) == ("macrodetails/Age", 0.5)
    assert loader.mapModifier("does/not-exist", 1.0) is None


def test_v10_modifier_lines_are_translated(loader):
    assert run(loader, "armslegs l-foot-trans-forward-backward 0.300000") == \
        [["modifier", "armslegs/l-foot-trans-backward|forward", -0.3]]
    assert run(loader, "macro Gender 0.75") == [["modifier", "macrodetails/Gender", 0.75]]


def test_v10_unmapped_modifier_is_passed_with_a_warning(loader, caplog):
    out = run(loader, "face nonexistent-target 0.1")
    assert out == [["modifier", "nonexistent/nonexistent-target", 0.1]]
    assert "No 1.0 -> 1.1 mapping" in caplog.text


def test_v10_custom_targets(loader):
    assert run(loader, "custom mytarget 0.2") == [["modifier", "custom/mytarget", 0.2]]


@pytest.mark.parametrize("line", ["tags one two", "camera 0.0 0.0", "subdivide True"])
def test_v10_passthrough_properties(loader, line):
    assert run(loader, line) == [line.split()]


def test_v10_skeleton_is_replaced(loader):
    assert run(loader, "skeleton game.json") == [["skeleton", "cmu_mb.mhskel"]]


def test_v10_skeleton_without_replacement_is_dropped(loader, caplog):
    assert run(loader, "skeleton xonotic.json") == []
    assert "no good replacement" in caplog.text
    assert run(loader, "skeleton unknown.json") == []


def test_v10_skin_material(loader):
    old = "skins/young_caucasian_female/young_caucasian_female_sweat.mhmat"
    assert run(loader, "skinMaterial " + old) == \
        [["skinMaterial", "skins/young_caucasian_female/young_caucasian_female.mhmat"]]
    assert run(loader, "skinMaterial skins/other.mhmat") == [["skinMaterial", "skins/other.mhmat"]]


def test_v10_proxy_uuid_replacement(loader):
    out = run(loader, "hair fhair01 c4808d0d-bbbc-4e62-82a1-8610b06d0e05")
    assert out == [["hair", "fhair01", "6c6b0bb1-3608-40b8-9108-4bfd8fa71e39"]]
    assert run(loader, "genitals x 1234") == [["genitals", "x", "1234"]]


def test_v10_unknown_property_passthrough(loader):
    assert run(loader, "eyebrows eyebrow001 abc") == [["eyebrows", "eyebrow001", "abc"]]


def test_v11_tags_become_name():
    out = []
    compat.MHM11Loader().loadProperty(["tags", "bob"], out.append, False)
    compat.MHM11Loader().loadProperty(["modifier", "x", "0.5"], out.append, False)
    assert out == [["name", "bob"], ["modifier", "x", "0.5"]]
    assert compat.MHM11Loader().getAcceptedVersion() == (1, 1)


def test_v12_passthrough():
    out = []
    compat.MHM12Loader().loadProperty(["tags", "bob"], out.append, False)
    assert out == [["tags", "bob"]]


def test_load_mhm_dispatches_every_line_and_reports_progress():
    out = []
    lines = ["tags old file\n", "skeleton basic.json\n", "macro Age 0.5\n"]
    compat.loadMHM("v1.0.1", lines, out.append)
    assert out == [["tags", "old", "file"],
                   ["skeleton", "default.mhskel"],
                   ["modifier", "macrodetails/Age", 0.5]]
    assert len(G.app.calls) == 3  # one progress step per line


def test_load_mhm_rejects_unparsable_version():
    with pytest.raises(RuntimeError, match="Failed to parse version"):
        compat.loadMHM("vX", ["tags a"], lambda d: None)
