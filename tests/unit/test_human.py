"""apps/human.py: the Human model.

Covers the macro values (gender, age, ...), the registry of modifiers and their
dependencies, target application on the mesh, proxies, skeletons and poses, and
reading and writing .mhm files.

The human is built from the real base mesh, modifiers and targets in
makehuman/data, as mhmain.loadHuman() and the modelling plugins do. Only the
running application is replaced (FakeApp): progress bar, camera and the
load/save handlers that plugins register.
"""

import logging
import os
import sys
import uuid

import numpy as np
import pytest

from core import G

MH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "makehuman")


class Camera(object):
    def __init__(self):
        self.rotation = [0.0, 0.0, 0.0]
        self.translation = np.zeros(3, dtype=np.float32)
        self.zoomFactor = 1.0

    def getRotation(self):
        return self.rotation

    def setRotation(self, rot):
        self.rotation = list(rot)

    def setZoomFactor(self, zoom):
        self.zoomFactor = zoom


class FakeApp(object):
    """The parts of the running application that Human and Progress use."""

    def __init__(self):
        self.loadHandlers = {}
        self.saveHandlers = []
        self.modelCamera = Camera()
        self.selectedHuman = None

    def progress(self, *args, **kwargs):
        pass

    def redraw(self):
        pass


def without_gl(mp):
    """Unit tests have no OpenGL context. If another test imported the shader module,
    materials would try to compile shaders (material.getShaderObj) and crash."""
    shader = sys.modules.get("shader")
    if shader is not None:
        mp.setattr(shader.Shader, "_supported", False)


@pytest.fixture(scope="module")
def app():
    mp = pytest.MonkeyPatch()
    mp.chdir(MH_DIR)  # data/... paths are relative to makehuman/
    without_gl(mp)
    fake = FakeApp()
    mp.setattr(G, "app", fake)
    yield fake
    mp.undo()


def new_human():
    import files3d
    import getpath
    import human
    return human.Human(files3d.loadMesh(getpath.getSysDataPath("3dobjs/base.obj"), maxFaces=5))


def with_modifiers(h):
    import getpath
    import humanmodifier
    humanmodifier.loadModifiers(getpath.getSysDataPath("modifiers/modeling_modifiers.json"), h)
    h.applyAllTargets()
    return h


@pytest.fixture(scope="module")
def shared(app):
    """Attaching the modelling modifiers takes ~0.5 s, so tests share one human."""
    h = with_modifiers(new_human())
    return h, h.meshData.coord.copy()


@pytest.fixture
def human(app, shared):
    h = shared[0]
    app.selectedHuman = h
    app.loadHandlers = {}
    app.saveHandlers = []
    app.modelCamera = Camera()
    yield h
    h.symmetryModeEnabled = False
    h.blockEthnicUpdates = False
    h.resetMeshValues()
    h.updateMacroModifiers()
    h.applyAllTargets()


@pytest.fixture
def default_coords(shared):
    return shared[1]


@pytest.fixture
def bare(app):
    """A human without modifiers: cheap, for proxies, skeletons and poses."""
    h = new_human()
    app.selectedHuman = h
    return h


@pytest.fixture
def fresh(app):
    """A human with modifiers that the test may take apart."""
    h = with_modifiers(new_human())
    app.selectedHuman = h
    return h


class Events(object):
    def __init__(self):
        self.changing = []
        self.changed = []

    def names(self, which="changed"):
        return [e.change for e in getattr(self, which)]


def listen(h, monkeypatch):
    ev = Events()
    monkeypatch.setattr(h, "onChanging", ev.changing.append, raising=False)
    monkeypatch.setattr(h, "onChanged", ev.changed.append, raising=False)
    return ev


def data(rel):
    return "data/" + rel


def load_skeleton(h, name="default"):
    import skeleton
    return skeleton.load(data("rigs/%s.mhskel" % name), h.meshData)


def load_proxy(h, rel, ptype):
    import proxy
    pxy = proxy.loadProxy(h, data(rel), type=ptype)
    pxy.loadMeshAndObject(h)
    return pxy


# --- identity ----------------------------------------------------------------------------

def test_name_tags_and_uuid(bare):
    h = bare
    assert (h.getName(), h.getUuid(), h.getTags()) == ("", "", set())
    h.humanName = "Ada"
    assert h.getName() == "Ada"
    h.setTags({"Tall", "x" * 30})
    assert h.tags == {"Tall", "x" * 25}  # max. tag length is 25
    with pytest.raises(ValueError):
        h.setTags(["not", "a", "set"])
    h.addTag("BLUE eyes and a very long tag")
    assert "blue eyes and a very long" in h.getTags()
    h.clearTags()
    assert h.getTags() == set()
    new = h.newUuid()
    assert str(uuid.UUID(new)) == new and h.getUuid() == new
    h.setUuid("custom")
    assert h.getUuid() == "custom"


# --- macro values without modifiers ----------------------------------------------------------

def test_gender(bare, monkeypatch):
    h = bare
    ev = listen(h, monkeypatch)
    h.setGender(1.5, updateModifier=False)
    assert (h.getGender(), h.maleVal, h.femaleVal) == (1.0, 1.0, 0.0)
    assert h.getDominantGender() == "male"
    h.setGender(1.0, updateModifier=False)  # unchanged: no event
    assert ev.names("changing") == ["gender"]
    h.setGender(0.2, updateModifier=False)
    assert h.getDominantGender() == "female" and h.femaleVal == pytest.approx(0.8)
    h.setGender(0.5, updateModifier=False)
    assert h.getDominantGender() is None


@pytest.mark.parametrize("age, baby, child, young, old", [
    (0.0, 1.0, 0.0, 0.0, 0.0),
    (0.1875, 0.0, 1.0, 0.0, 0.0),
    (0.5, 0.0, 0.0, 1.0, 0.0),
    (0.75, 0.0, 0.0, 0.5, 0.5),
    (1.0, 0.0, 0.0, 0.0, 1.0),
])
def test_age_blends_baby_child_young_old(bare, age, baby, child, young, old):
    bare.setAge(age, updateModifier=False)
    got = (bare.babyVal, bare.childVal, bare.youngVal, bare.oldVal)
    assert got == pytest.approx((baby, child, young, old), abs=1e-3)


def test_age_in_years(bare, monkeypatch):
    h = bare
    ev = listen(h, monkeypatch)
    for age, years in [(0.0, 1.0), (0.25, 13.0), (0.5, 25.0), (0.75, 57.5), (1.0, 90.0)]:
        h.setAge(age, updateModifier=False)
        assert h.getAgeYears() == pytest.approx(years)
    h.setAgeYears(10, updateModifier=False)
    assert h.getAge() == pytest.approx(0.1875)
    h.setAgeYears("57.5", updateModifier=False)
    assert h.getAge() == pytest.approx(0.75)
    h.setAge(0.75, updateModifier=False)  # unchanged: no event
    assert ev.names("changing").count("age") == 7  # five ages, two setAgeYears
    for bad in (0.5, 91):
        with pytest.raises(RuntimeError, match="Invalid age"):
            h.setAgeYears(bad)


@pytest.mark.parametrize("name, attrs", [
    ("Weight", ("minweightVal", "averageweightVal", "maxweightVal")),
    ("Muscle", ("minmuscleVal", "averagemuscleVal", "maxmuscleVal")),
    ("Height", ("minheightVal", "averageheightVal", "maxheightVal")),
    ("BreastSize", ("mincupVal", "averagecupVal", "maxcupVal")),
    ("BreastFirmness", ("minfirmnessVal", "averagefirmnessVal", "maxfirmnessVal")),
    ("BodyProportions", ("uncommonproportionsVal", "regularproportionsVal", "idealproportionsVal")),
])
def test_three_way_macro_values(bare, monkeypatch, name, attrs):
    h = bare
    ev = listen(h, monkeypatch)
    setter, getter = getattr(h, "set" + name), getattr(h, "get" + name)
    for value, expected in [(0.0, (1, 0, 0)), (0.25, (0.5, 0.5, 0)),
                            (0.75, (0, 0.5, 0.5)), (1.0, (0, 0, 1))]:
        setter(value, updateModifier=False)
        assert getter() == value
        assert tuple(getattr(h, a) for a in attrs) == pytest.approx(expected)
    setter(7.0, updateModifier=False)  # clamped to 1.0, which it already is: no event
    setter(-1.0, updateModifier=False)
    assert getter() == 0.0
    event = name[0].lower() + name[1:]
    assert ev.names("changing") == [event] * 5


# --- ethnic values ---------------------------------------------------------------------------

def ethnic(h):
    return (h.getAfrican(), h.getAsian(), h.getCaucasian())


def test_ethnic_values_are_normalized(bare):
    h = bare
    h.setCaucasian(0.5, updateModifier=False)
    assert ethnic(h) == pytest.approx((0.25, 0.25, 0.5))
    h.setAfrican(0.75, updateModifier=False)
    assert ethnic(h) == pytest.approx((0.75, 1 / 12., 1 / 6.))
    assert sum(ethnic(h)) == pytest.approx(1.0)


def test_ethnic_values_without_sync(bare):
    h = bare
    h.setAsian(0.9, sync=False, updateModifier=False)
    assert ethnic(h) == pytest.approx((1 / 3., 0.9, 1 / 3.))
    h.blockEthnicUpdates = True
    h.setAfrican(2.0, updateModifier=False)  # clamped, not normalized
    assert ethnic(h) == pytest.approx((1.0, 0.9, 1 / 3.))


def test_ethnic_edge_cases(bare):
    h = bare
    h.africanVal = h.asianVal = h.caucasianVal = 0.0
    h._setEthnicVals()  # all zero cannot be: back to thirds
    assert ethnic(h) == pytest.approx((1 / 3.,) * 3)
    h.setAsian(1.0, updateModifier=False)
    assert ethnic(h) == pytest.approx((0.0, 1.0, 0.0))
    h.setAsian(1.0, updateModifier=False)  # the others are 0 and stay 0
    assert ethnic(h) == pytest.approx((0.0, 1.0, 0.0))
    h.setAsian(0.4, updateModifier=False)  # others were 0: they share the rest
    assert ethnic(h) == pytest.approx((0.3, 0.4, 0.3))
    h.africanVal = h.caucasianVal = 0.0
    h.setAsian(0.0, updateModifier=False)  # everything 0 again
    assert ethnic(h) == pytest.approx((1 / 3.,) * 3)


@pytest.mark.parametrize("african, asian, caucasian, dominant", [
    (0.2, 0.5, 0.3, "asian"),
    (0.1, 0.3, 0.6, "caucasian"),
    (0.2, 0.4, 0.4, None),
    (0.5, 0.2, 0.3, "african"),
    (0.3, 0.1, 0.6, "caucasian"),
    (0.4, 0.2, 0.4, None),
    (0.2, 0.2, 0.6, "caucasian"),
    (1 / 3., 1 / 3., 1 / 3., None),
])
def test_dominant_ethnicity(bare, african, asian, caucasian, dominant):
    bare.africanVal, bare.asianVal, bare.caucasianVal = african, asian, caucasian
    assert bare.getEthnicity() == dominant


# --- macro setters through the modifiers ----------------------------------------------------

@pytest.mark.parametrize("name, modifier", [
    ("Gender", "macrodetails/Gender"),
    ("Age", "macrodetails/Age"),
    ("Weight", "macrodetails-universal/Weight"),
    ("Muscle", "macrodetails-universal/Muscle"),
    ("Height", "macrodetails-height/Height"),
    ("BreastSize", "breast/BreastSize"),
    ("BreastFirmness", "breast/BreastFirmness"),
    ("BodyProportions", "macrodetails-proportions/BodyProportions"),
    ("Caucasian", "macrodetails/Caucasian"),
    ("African", "macrodetails/African"),
    ("Asian", "macrodetails/Asian"),
])
def test_macro_setters_reshape_the_mesh(human, default_coords, name, modifier):
    getattr(human, "set" + name)(0.9)
    assert getattr(human, "get" + name)() == pytest.approx(0.9)
    assert human.getModifier(modifier).getValue() == pytest.approx(0.9)
    moved = np.abs(human.meshData.coord - default_coords).max()
    assert moved > 0.01


def test_gender_selects_male_or_female_targets(human):
    def tokens(path):
        return os.path.basename(path)[:-len(".target")].split("-")

    human.setGender(1.0)
    stack = human.targetsDetailStack
    assert stack and all("male" in tokens(p) and "female" not in tokens(p) for p in stack)
    race = [p for p in stack if tokens(p)[0] in ("african", "asian", "caucasian")]
    assert sorted(tokens(p)[0] for p in race) == ["african", "asian", "caucasian"]
    assert sum(stack[p] for p in race) == pytest.approx(1.0)  # the race mix of a male
    human.setGender(0.0)
    assert all("female" in tokens(p) for p in human.targetsDetailStack)


def test_height_measure_follows_the_height_modifier(human):
    default = human.getHeightCm()
    assert 150 < default < 190
    human.setHeight(1.0)
    assert human.getHeightCm() > default + 10


def test_body_measures(human):
    bbox = human.getBoundingBox()
    assert human.getHeightCm() == pytest.approx(10 * (bbox[1][1] - bbox[0][1]))
    bsa = human.getBsa()
    assert 1.4 < bsa < 2.2  # m^2, an adult body surface
    assert human.getWeightKg() == pytest.approx(bsa * bsa * 3600 / human.getHeightCm())
    volume = human.getVolume()
    human.setWeight(1.0)
    assert human.getBsa() > bsa and human.getVolume() > volume


# --- details and the modifier registry ----------------------------------------------------------

def test_details_are_stored_by_canonical_path(bare):
    import getpath
    path = data("targets/nose/nose-base-up.target")
    bare.setDetail(path, 0.5)
    assert bare.getDetail(getpath.canonicalPath(path)) == 0.5
    assert list(bare.targetsDetailStack) == [getpath.canonicalPath(path)]
    bare.setDetail(path, 0)
    assert bare.targetsDetailStack == {}
    bare.setDetail(path, 0)  # removing twice is fine
    assert bare.getDetail(path) == 0.0


def test_modifier_lookup(human, caplog):
    import humanmodifier
    names = human.modifierNames
    assert "macrodetails/Gender" in names and names == human.getModifierNames()
    assert len(human.modifiers) == len(names)
    assert human.getModifier("nose/nose-trans-down|up").fullName == "nose/nose-trans-down|up"
    with pytest.raises(KeyError):
        human.getModifier("nose/no-such-thing")
    assert {"macrodetails", "nose", "breast", "armslegs"} <= set(human.modifierGroups)
    nose = human.getModifiersByGroup("nose")
    assert nose and all(m.groupName == "nose" for m in nose)
    with caplog.at_level(logging.WARNING):
        assert human.getModifiersByGroup("wings") == []
    assert "Modifier group wings does not exist." in caplog.text
    macros = human.getModifiersByType(humanmodifier.MacroModifier)
    assert sorted(m.name for m in macros) == sorted([
        "Gender", "Age", "African", "Asian", "Caucasian", "Muscle", "Weight", "Height",
        "BodyProportions", "BreastSize", "BreastFirmness"])
    assert human.getModifiersByType(humanmodifier.MacroModifier) is macros  # cached


def test_adding_a_modifier_twice_is_refused(human, caplog):
    with pytest.raises(RuntimeError, match="already attached"):
        human.addModifier(human.getModifier("nose/nose-trans-down|up"))


def test_modifier_dependencies(human):
    height = human.getModifier("macrodetails-height/Height")
    gender = human.getModifier("macrodetails/Gender")
    assert human.getModifierDependencies(height) == {"macrodetails", "macrodetails-universal"}
    assert human.getModifierDependencies(height, filter=["macrodetails"]) == {"macrodetails"}
    assert human.getModifierDependencies(height, filter=["macrodetails", "macrodetails-universal"]) \
        == {"macrodetails", "macrodetails-universal"}
    assert human.getModifierDependencies(human.getModifier("nose/nose-trans-down|up")) == set()
    affected = human.getModifiersAffectedBy(gender)
    assert {"macrodetails-universal", "macrodetails-height", "macrodetails-proportions",
            "breast"} <= set(affected)
    assert "macrodetails" not in affected  # its own group
    assert human.getModifiersAffectedBy(gender, filter=["breast", "nose"]) == ["breast"]


def test_unmapped_dependencies_are_reported(bare, caplog):
    import humanmodifier
    height = humanmodifier.MacroModifier("macrodetails-height", "Height")
    height.setHuman(bare)
    with caplog.at_level(logging.ERROR):
        assert bare.getModifierDependencies(height) == set()
    assert "Error var gender not mapped" in caplog.text


def test_two_groups_cannot_control_one_variable(bare, caplog):
    import humanmodifier
    humanmodifier.MacroModifier("macrodetails", "Gender").setHuman(bare)
    with caplog.at_level(logging.ERROR):
        # the universal targets depend on gender too, so this is a valid gender modifier
        humanmodifier.MacroModifier("macrodetails-universal", "Gender").setHuman(bare)
    assert "multiple modifier groups setting var gender (macrodetails-universal and macrodetails)" \
        in caplog.text
    assert bare._modifier_varMapping["gender"] == "macrodetails"


def test_remove_a_simple_modifier(bare):
    import humanmodifier
    nose = humanmodifier.UniversalModifier("nose", "nose-trans", "down", "up")
    nose.setHuman(bare)
    nose.setValue(1.0)
    up = nose.r_targets[0][0]
    assert bare.getDetail(up) == 1.0
    bare.removeModifier(nose)
    assert "nose/nose-trans-down|up" not in bare.modifierNames
    assert "nose" not in bare.modifierGroups
    assert bare.getDetail(up) == 0.0


def test_remove_a_macro_modifier_without_dependencies(bare):
    import humanmodifier
    gender = humanmodifier.MacroModifier("macrodetails", "Gender")
    gender.macroDependencies = []  # keep this one independent
    gender.setHuman(bare)
    assert bare._modifier_varMapping == {"gender": "macrodetails"}
    bare.removeModifier(gender)
    assert bare._modifier_varMapping == {}
    assert bare.modifierNames == []


def test_removing_an_unknown_modifier_is_ignored(bare, caplog):
    import humanmodifier
    stray = humanmodifier.UniversalModifier("nose", "nose-trans", "down", "up")
    with caplog.at_level(logging.DEBUG):
        bare.removeModifier(stray)
    assert "Failed to remove modifier nose/nose-trans-down|up" in caplog.text


def test_remove_a_modifier_with_dependencies(fresh):
    height = fresh.getModifier("macrodetails-height/Height")
    fresh.setHeight(1.0)
    active = [t[0] for t in height.targets if fresh.getDetail(t[0])]
    assert active
    fresh.removeModifier(height)
    assert [p for p in active if fresh.getDetail(p)] == []
    assert "macrodetails-height" not in fresh.getModifiersAffectedBy(fresh.getModifier("macrodetails/Gender"))
    fresh.setGender(1.0)  # must still work after the removal


# --- applying targets ------------------------------------------------------------------------

def test_apply_all_targets_adds_weighted_offsets(bare, monkeypatch):
    import algos3d
    h = bare
    ev = listen(h, monkeypatch)
    h.applyAllTargets()
    rest = h.meshData.coord.copy()
    path = data("targets/nose/nose-base-up.target")
    target = algos3d.getTarget(h.meshData, path)
    h.setDetail(path, 0.5)
    h.applyAllTargets()
    expected = rest.copy()
    expected[target.verts] += 0.5 * target.data
    np.testing.assert_allclose(h.meshData.coord, expected, atol=1e-5)
    np.testing.assert_allclose(h.getRestposeCoordinates(), expected, atol=1e-5)
    assert ev.names().count("targets") == 2
    h.setDetail(path, 0)
    h.applyAllTargets(update=False)
    np.testing.assert_allclose(h.meshData.coord, rest, atol=1e-6)


def test_subdivision(bare, monkeypatch):
    h = bare
    ev = listen(h, monkeypatch)
    h.setSubdivided(True)
    assert h.isSubdivided() and ev.names() == ["smooth"]
    h.setSubdivided(True)  # no change, no event
    assert ev.names() == ["smooth"]
    sub = h.getSubdivisionMesh(False)
    before = sub.coord.copy()
    h.setDetail(data("targets/nose/nose-base-up.target"), 1.0)
    h.applyAllTargets()
    assert not np.allclose(sub.coord, before)  # the smoothed mesh follows the targets
    h.setSubdivided(False)
    assert not h.isSubdivided() and ev.names().count("smooth") == 2


def test_symmetry(human):
    left = human.getModifier("eyes/l-eye-bag-decr|incr")
    right = human.getModifier("eyes/r-eye-bag-decr|incr")
    left.setValue(0.7)
    human.applySymmetryRight()  # left to right
    assert right.getValue() == pytest.approx(0.7)
    right.setValue(-0.3)
    human.applySymmetryLeft()  # right to left
    assert left.getValue() == pytest.approx(-0.3)


def test_part_name_for_group_name(bare):
    assert bare.getPartNameForGroupName("l-eye-lower") == "l-eye"
    assert bare.getPartNameForGroupName("r-upperarm") == "r-upperarm"
    assert bare.getPartNameForGroupName("wing") is None


def test_joints_without_skeleton(bare):
    joints = bare.getJoints()
    assert "joint-pelvis" in joints and all(j.startswith("joint-") for j in joints)
    group = bare.meshData.getFaceGroup("joint-head")
    verts = bare.meshData.getVerticesForGroups([group.name])
    expected = bare.getRestposeCoordinates()[verts].mean(axis=0)
    np.testing.assert_allclose(bare.getJointPosition("joint-head"), expected, rtol=1e-6)
    np.testing.assert_allclose(bare.getJointPosition("head"), expected, rtol=1e-6)


def test_reset_restores_the_default_human(human, default_coords, monkeypatch):
    import material
    ev = listen(human, monkeypatch)
    human.setName("Someone")
    human.addTag("tagged")
    human.newUuid()
    human.setGender(1.0)
    human.material = material.fromFile(data("skins/young_caucasian_female/young_caucasian_female.mhmat"))
    assert human.material.name != human._defaultMaterial.name
    human.resetMeshValues()
    assert (human.getName(), human.getTags(), human.getUuid()) == ("", set(), "")
    assert human.getGender() == 0.5 and human.targetsDetailStack == {}
    assert human.material.name == human._defaultMaterial.name
    assert ev.names("changing")[-1] == "reset" and ev.names()[-1] == "reset"
    assert "material" in ev.names("changing") and "material" in ev.names()


# --- skeletons ---------------------------------------------------------------------------------

def test_base_skeleton(bare, monkeypatch):
    h = bare
    assert h.getVertexWeights() is None
    ev = listen(h, monkeypatch)
    skel = load_skeleton(h)
    h.setBaseSkeleton(skel)
    assert ev.names("changing")[0] == "skeleton" and "skeleton" in ev.names()
    assert h.getBaseSkeleton() is skel
    weights = h.getVertexWeights()
    assert weights is not None and "spine05" in weights.data
    assert h.getVertexWeights(skel) is weights
    pelvis = skel.getJointPosition("pelvis", h)
    np.testing.assert_allclose(h.getJointPosition("pelvis"), pelvis)
    # Changing the shape moves the joints
    head_before = skel.getBone("head").getRestHeadPos()
    h.setDetail(data("targets/macrodetails/height/female-young-averagemuscle-averageweight-maxheight.target"), 1.0)
    h.applyAllTargets()
    assert skel.getBone("head").getRestHeadPos()[1] > head_before[1] + 0.5


def test_base_skeleton_can_be_removed(bare):
    bare.setBaseSkeleton(load_skeleton(bare))
    bare.setBaseSkeleton(None)
    assert bare.getBaseSkeleton() is None and bare.getVertexWeights() is None
    assert bare.getBoundMeshes() == [bare.meshData.name]


def test_user_skeleton(bare, monkeypatch):
    h = bare
    h.setBaseSkeleton(load_skeleton(h))
    ev = listen(h, monkeypatch)
    user = load_skeleton(h, "game_engine")
    h.setSkeleton(user)
    assert ev.names("changing") == ["user-skeleton"] and ev.names() == ["user-skeleton"]
    assert user.dirty is True
    assert h.getSkeleton() is user and user.dirty is False  # joints were updated
    h.applyAllTargets()
    assert user.dirty is True
    weights = h.getVertexWeights(user)
    assert set(weights.data) <= set(b.name for b in user.getBones())
    assert "spine05" not in weights.data
    h.setSkeleton(None)
    assert h.getSkeleton() is None


# --- poses ---------------------------------------------------------------------------------------

@pytest.fixture
def posed(bare):
    import bvh
    h = bare
    h.setBaseSkeleton(load_skeleton(h))
    anim = bvh.load(data("poses/benchmark.bvh"), "auto").createAnimationTrack(h.getBaseSkeleton())
    h.addAnimation(anim)
    return h, anim


def test_pose_changes_the_mesh_but_not_the_rest_shape(posed, monkeypatch):
    h, anim = posed
    rest = h.meshData.coord.copy()
    h.setSkeleton(load_skeleton(h, "game_engine"))
    ev = listen(h, monkeypatch)
    h.setActiveAnimation(anim.name)
    h.setPosed(True)
    assert h.isPosed()
    assert ev.names("changing")[:2] == ["poseChange", "poseState"]
    changed = dict((e.change, e) for e in ev.changed)
    assert changed["poseChange"].pose == anim.name and changed["poseState"].state is True
    assert np.abs(h.meshData.coord - rest).max() > 1.0
    np.testing.assert_allclose(h.getRestposeCoordinates(), rest)
    assert h.skeleton.dirty is True
    h.refreshPose()
    assert ev.names()[-1] == "poseRefresh"
    h.setPosed(False)
    np.testing.assert_allclose(h.meshData.coord, rest, atol=1e-4)


def test_targets_on_a_posed_human(posed):
    h, anim = posed
    h.setActiveAnimation(anim.name)
    h.setPosed(True)
    posed_coords = h.meshData.coord.copy()
    h.setDetail(data("targets/nose/nose-base-up.target"), 1.0)
    h.applyAllTargets()
    assert h.isPosed()
    assert not np.allclose(h.meshData.coord, posed_coords)
    h.setPosed(False)
    import algos3d
    target = algos3d.getTarget(h.meshData, data("targets/nose/nose-base-up.target"))
    np.testing.assert_allclose(h.getRestposeCoordinates()[target.verts],
                               new_human().meshData.coord[target.verts] + target.data, atol=1e-4)


def test_refresh_pose_updates_the_smoothed_mesh(posed):
    h, anim = posed
    h.setSubdivided(True)
    sub = h.getSubdivisionMesh(False)
    before = sub.coord.copy()
    h.setActiveAnimation(anim.name)
    h.setPosed(True)
    h.refreshPose()
    assert np.abs(sub.coord - before).max() > 1.0


# --- proxies ---------------------------------------------------------------------------------------

SIMPLE_PROXIES = [
    ("Hair", "hair/short01/short01.mhclo"),
    ("Eyes", "eyes/low-poly/low-poly.mhclo"),
    ("Eyebrows", "eyebrows/eyebrow001/eyebrow001.mhclo"),
    ("Eyelashes", "eyelashes/eyelashes01/eyelashes01.mhclo"),
    ("Teeth", "teeth/teeth_base/teeth_base.mhclo"),
    ("Tongue", "tongue/tongue01/tongue01.mhclo"),
]


@pytest.mark.parametrize("ptype, rel", SIMPLE_PROXIES)
def test_simple_proxies(bare, monkeypatch, ptype, rel):
    h = bare
    h.setBaseSkeleton(load_skeleton(h))
    ev = listen(h, monkeypatch)
    pxy = load_proxy(h, rel, ptype)
    setattr(h, ptype.lower() + "Proxy", pxy)
    assert getattr(h, "get%sProxy" % ptype)() is pxy
    assert h.getTypedSimpleProxies(ptype.lower()) is pxy
    assert h.getProxies() == [pxy] and h.getProxyObjects() == [pxy.object]
    assert h.getObjects() == [h, pxy.object]
    seed = pxy.object.getSeedMesh().name
    assert seed in h.getBoundMeshes()  # posed with the human
    assert [e.proxy for e in ev.changed if e.change == "proxyChange"] == [ptype.lower()]
    getattr(h, "set%sProxy" % ptype)(None)
    assert h.getProxies() == [] and seed not in h.getBoundMeshes()


def test_unknown_simple_proxy_type(bare):
    assert bare.getTypedSimpleProxies("wings") is None
    assert bare.getTypedSimpleProxies("hair") is None


def test_replacing_a_proxy_unbinds_the_old_one(bare):
    h = bare
    h.setBaseSkeleton(load_skeleton(h))
    first = load_proxy(h, "eyebrows/eyebrow001/eyebrow001.mhclo", "Eyebrows")
    second = load_proxy(h, "eyebrows/eyebrow002/eyebrow002.mhclo", "Eyebrows")
    h.eyebrowsProxy = first
    h.eyebrowsProxy = second
    bound = h.getBoundMeshes()
    assert second.object.getSeedMesh().name in bound
    assert first.object.getSeedMesh().name not in bound


def test_clothes(bare, monkeypatch):
    h = bare
    ev = listen(h, monkeypatch)
    shoes = load_proxy(h, "clothes/shoes01/shoes01.mhclo", "Clothes")
    hat = load_proxy(h, "clothes/fedora01/fedora01.mhclo", "Clothes")
    h.addClothesProxy(shoes)
    h.addClothesProxy(hat)
    assert set(h.clothesProxies) == {shoes.getUuid(), hat.getUuid()}
    h.clothesProxies.clear()  # a copy
    assert len(h.clothesProxies) == 2
    assert set(h.getProxies()) == {shoes, hat}
    h.removeClothesProxy(shoes.getUuid())
    assert list(h.clothesProxies) == [hat.getUuid()]
    h.removeClothesProxy("no-such-uuid")
    actions = [(e.change, e.proxy, e.action, e.proxy_obj) for e in ev.changed
               if e.change == "proxyChange"]
    assert actions == [("proxyChange", "clothes", "add", shoes), ("proxyChange", "clothes", "add", hat),
                       ("proxyChange", "clothes", "remove", shoes),
                       ("proxyChange", "clothes", "remove", None)]


def test_body_proxy(bare, monkeypatch):
    h = bare
    h.setBaseSkeleton(load_skeleton(h))
    ev = listen(h, monkeypatch)
    pxy = load_proxy(h, "proxymeshes/proxy741/proxy741.proxy", "Proxymeshes")
    h.setProxy(pxy)
    assert h.isProxied() and h.getProxy() is pxy
    assert h.mesh is h.getProxyMesh() and h.mesh.getVertexCount() < h.meshData.getVertexCount()
    assert pxy in h.getProxies() and pxy not in h.getProxies(includeHumanProxy=False)
    assert h.getProxyMesh().name in h.getBoundMeshes()
    assert [e.proxy for e in ev.changed if e.change == "proxyChange"] == ["human"]
    assert not h.hasGenitals()
    pxy.tags.add("genitals")
    assert h.hasGenitals()
    other = load_proxy(h, "proxymeshes/female_generic/female_generic.proxy", "Proxymeshes")
    old_mesh = h.getProxyMesh().name
    h.setProxy(other)
    assert old_mesh not in h.getBoundMeshes() and h.getProxy() is other
    h.setProxy(None)
    assert not h.isProxied() and h.mesh is h.meshData
    assert not h.hasGenitals()


class View(object):
    """The 3D view objects are attached to when the app adds them (app.addObject)."""

    def isVisible(self):
        return True


def test_proxy_objects_follow_the_human(bare, monkeypatch):
    h = bare
    eyes = load_proxy(h, "eyes/low-poly/low-poly.mhclo", "Eyes")
    h.eyesProxy = eyes
    view = View()
    h._view = eyes.object._view = lambda: view
    seen = []
    for name in ("onShown", "onHidden", "onTranslated", "onRotated"):
        monkeypatch.setattr(h, name, (lambda n: lambda e: seen.append(n))(name), raising=False)
    h.hide()
    assert not h.visible and not eyes.object.visible
    assert not h.mesh.visibility and not eyes.object.mesh.visibility
    h.show()
    assert h.visible and eyes.object.visible
    assert h.mesh.visibility and eyes.object.mesh.visibility
    eyes.object.setPosition([0.5, 0.0, 0.0])
    h.setPosition([1.0, 2.0, 3.0])
    assert list(h.getPosition()) == [1.0, 2.0, 3.0]
    assert list(eyes.object.getPosition()) == [1.5, 2.0, 3.0]  # moved by the same delta
    h.setRotation([0.0, 90.0, 0.0])
    assert list(eyes.object.getRotation()) == [0.0, 90.0, 0.0]
    h.setSolid(False)
    assert not eyes.object.isSolid()
    assert seen == ["onHidden", "onShown", "onTranslated", "onRotated"]


def test_subdividing_also_subdivides_proxies(bare):
    h = bare
    eyes = load_proxy(h, "eyes/low-poly/low-poly.mhclo", "Eyes")
    h.eyesProxy = eyes
    h.setSubdivided(True)
    assert h.isSubdivided() and eyes.object.isSubdivided()
    h.setSubdivided(False)
    assert not eyes.object.isSubdivided()


def test_objects_without_visible_faces_can_be_left_out(bare):
    h = bare
    eyes = load_proxy(h, "eyes/low-poly/low-poly.mhclo", "Eyes")
    h.eyesProxy = eyes
    mesh = eyes.object.mesh
    mesh.changeFaceMask(np.zeros(mesh.getFaceCount(), dtype=bool))
    assert h.getObjects() == [h, eyes.object]
    assert h.getObjects(excludeZeroFaceObjs=True) == [h]


def test_vertex_weights_for_other_bound_meshes(bare):
    import files3d
    import guicommon
    h = bare
    h.setBaseSkeleton(load_skeleton(h))
    # A mesh added by a plugin, without proxy: left to that plugin
    extra = guicommon.Object(files3d.loadMesh(data("3dobjs/axis.obj")))
    h._updateMeshVertexWeights(extra.mesh)
    assert extra.mesh.name not in h.getBoundMeshes()
    # A bound proxy mesh whose object went away is dropped
    eyes = load_proxy(h, "eyes/low-poly/low-poly.mhclo", "Eyes")
    h.eyesProxy = eyes
    seed = eyes.object.getSeedMesh()
    assert seed.name in h.getBoundMeshes()
    seed.object = None
    h._updateMeshVertexWeights(seed)
    assert seed.name not in h.getBoundMeshes()


def test_new_base_skeleton_reweights_proxies(bare):
    h = bare
    eyes = load_proxy(h, "eyes/low-poly/low-poly.mhclo", "Eyes")
    h.eyesProxy = eyes
    seed = eyes.object.getSeedMesh().name
    _, weights = h.getBoundMesh(seed)
    assert weights is None  # no skeleton yet
    h.setBaseSkeleton(load_skeleton(h))
    _, weights = h.getBoundMesh(seed)
    assert weights is not None and len(weights.data) > 0


def test_reset_drops_proxies_and_bound_meshes(bare):
    h = bare
    h.setBaseSkeleton(load_skeleton(h))
    h.eyesProxy = load_proxy(h, "eyes/low-poly/low-poly.mhclo", "Eyes")
    h.resetMeshValues()
    assert h.getProxies() == [] and h.getBoundMeshes() == [h.meshData.name]


# --- debug helpers --------------------------------------------------------------------------------

def test_trace_helpers_log_the_stack_and_buffer(bare, caplog):
    import algos3d
    import getpath
    h = bare
    loaded = data("targets/nose/nose-base-up.target")
    algos3d.getTarget(h.meshData, loaded)
    h.setDetail(loaded, 0.5)
    h.setDetail("/nowhere/never-loaded.target", 0.25)
    with caplog.at_level(logging.DEBUG):
        h.traceStack()
        h.traceBuffer(vertsToList=1)
    text = caplog.text
    assert "human.targetsDetailStack:" in text
    assert "   %s: 0.5" % getpath.canonicalPath(loaded) in text
    assert " ??? /nowhere/never-loaded.target: 0.25" in text
    assert "algos3d.targetBuffer:" in text
    target = algos3d._targetBuffer[getpath.canonicalPath(loaded)]
    assert "   %d : " % target.verts[0] in text


# --- MHM files --------------------------------------------------------------------------------------

def mhm(tmp_path, text, name="model.mhm"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_save_writes_the_model(human, app, tmp_path, monkeypatch):
    import makehuman
    ev = listen(human, monkeypatch)
    human.setName("Ada Lovelace")
    human.setTags({"tall", "scientist"})
    human.setUuid("1234-abcd")
    human.setGender(0.8)
    human.getModifier("nose/nose-trans-down|up").setValue(0.3)
    app.modelCamera.rotation = [10.0, 20.0, 0.0]
    app.modelCamera.translation[:] = [1.0, 2.0, 3.0]
    app.modelCamera.zoomFactor = 2.5

    def handler(h, f):
        assert h is human
        f.write("skinMaterial skins/default.mhmat")  # no newline: added
        f.writelines("eyebrows eyebrow001 1234\n")
        assert f.closed is False  # other attributes reach the real file

    app.saveHandlers = [handler]
    path = str(tmp_path / "ada.mhm")
    human.save(path)
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    assert lines[0] == "# Written by MakeHuman %s" % makehuman.getVersionStr()
    assert lines[1:6] == ["version %s" % makehuman.getShortVersion(noSub=True), "uuid 1234-abcd",
                          "name Ada Lovelace", "tags scientist;tall",
                          "camera 10.0 20.0 1.0 2.0 3.0 2.5"]
    modifiers = [l for l in lines if l.startswith("modifier ")]
    assert "modifier macrodetails/Gender 0.800000" in modifiers
    assert "modifier nose/nose-trans-down|up 0.300000" in modifiers
    assert len(modifiers) == 12  # the 11 macro modifiers are always written, plus the nose
    # handler lines get their newline; the last line too, because f is the wrapper by then
    assert lines[-4:] == ["skinMaterial skins/default.mhmat", "eyebrows eyebrow001 1234",
                          "subdivide False", ""]
    saves = [(e.change, e.path) for e in ev.changing + ev.changed if e.change == "save"]
    assert saves == [("save", path), ("save", path)]


def test_save_and_load_round_trip(human, app, tmp_path, monkeypatch):
    human.setName("Round Trip")
    human.addTag("one")
    human.addTag("two")
    human.newUuid()
    human.setAge(0.9)
    human.setBreastSize(1.0)
    human.setCaucasian(0.8)
    human.getModifier("nose/nose-trans-down|up").setValue(-0.4)
    human.applyAllTargets()
    app.modelCamera.rotation = [5.0, 6.0, 0.0]
    app.modelCamera.translation[:] = [0.5, 0.25, 0.0]
    app.modelCamera.zoomFactor = 1.5
    human.setSubdivided(True)
    saved_uuid = human.getUuid()
    saved_coords = human.meshData.coord.copy()
    path = str(tmp_path / "rt.mhm")
    human.save(path)

    human.resetMeshValues()
    app.modelCamera = Camera()
    ev = listen(human, monkeypatch)

    def on_changed(event):
        ev.changed.append(event)
        if event.change == "load":
            # What the modifier task views do (guimodifier.onHumanChanged). Needed:
            # the file lists breast/* before macrodetails/* and lines load without
            # dependency updates, so breast targets are first computed for the old age.
            human.updateMacroModifiers()

    monkeypatch.setattr(human, "onChanged", on_changed)
    human.load(path)
    assert (human.getName(), human.getTags(), human.getUuid()) == ("Round Trip", {"one", "two"}, saved_uuid)
    assert human.getAge() == pytest.approx(0.9)
    assert human.getCaucasian() == pytest.approx(0.8, abs=1e-5)
    assert sum(ethnic(human)) == pytest.approx(1.0)
    assert human.getModifier("nose/nose-trans-down|up").getValue() == pytest.approx(-0.4)
    assert app.modelCamera.rotation == [5.0, 6.0, 0.0]
    assert list(app.modelCamera.translation) == [0.5, 0.25, 0.0]
    assert app.modelCamera.zoomFactor == 1.5
    assert human.isSubdivided()
    np.testing.assert_allclose(human.meshData.coord, saved_coords, atol=1e-5)
    assert ev.names("changing")[0] == "load" and ev.changing[0].path == path
    assert "load" in ev.names()


def test_load_handlers(human, app, tmp_path):
    calls = []

    def skin(h, values, strict):
        calls.append(("skin", values, strict))

    def clothes(h, values, strict):
        calls.append(("clothes", values, strict))

    app.loadHandlers = {"skinMaterial": skin, "clothes": clothes, "clothesHideFaces": clothes}
    path = mhm(tmp_path, "# a comment\nversion v1.3.0\nskinMaterial skins/default.mhmat\n"
                         "clothes shoes01 1234\n\n")
    human.load(path)
    assert ("skin", ["skinMaterial", "skins/default.mhmat"], False) in calls
    assert ("clothes", ["clothes", "shoes01", "1234"], False) in calls
    # every handler hears about the start (once per key) and the end (once per handler)
    starts = [c[0] for c in calls if c[1] == ["status", "started"]]
    ends = [c[0] for c in calls if c[1] == ["status", "finished"]]
    assert sorted(starts) == ["clothes", "clothes", "skin"]
    assert sorted(ends) == ["clothes", "skin"]


def test_load_without_update_leaves_the_mesh(human, tmp_path):
    before = human.meshData.coord.copy()
    human.load(mhm(tmp_path, "version v1.3.0\nmodifier macrodetails/Gender 1.000000\n"), update=False)
    assert human.getGender() == 1.0
    np.testing.assert_array_equal(human.meshData.coord, before)


def test_unknown_lines_warn_or_fail(human, tmp_path, caplog):
    path = mhm(tmp_path, "version v1.3.0\nwings 2\nmodifier nose/no-such-modifier 0.5\n")
    with caplog.at_level(logging.WARNING):
        human.load(path)
    assert "Unknown property in MHM file: ['wings', '2']" in caplog.text
    assert "Unknown modifier specified in MHM file: nose/no-such-modifier" in caplog.text
    with pytest.raises(RuntimeError, match="Unknown property in MHM file"):
        human.load(path, strict=True)


@pytest.mark.parametrize("when", ["started", "property", "finished"])
def test_failing_load_handler(human, app, tmp_path, caplog, when):
    def fussy(h, values, strict):
        if values[0] == "status" and values[1] == when or values[0] == "skinMaterial" and when == "property":
            raise ValueError("handler failed at " + when)

    app.loadHandlers = {"skinMaterial": fussy}
    path = mhm(tmp_path, "version v1.3.0\nskinMaterial x\nmodifier macrodetails/Gender 1.0\n")
    with caplog.at_level(logging.WARNING):
        human.load(path)
    assert human.getGender() == 1.0  # loading went on
    assert "Exception while" in caplog.text
    human.resetMeshValues()
    with pytest.raises(ValueError, match="handler failed at " + when):
        human.load(path, strict=True)


def test_older_files_load_through_compat(human, tmp_path, caplog):
    path = mhm(tmp_path, "version v1.1.0\nmodifier macrodetails/Gender 0.800000\n"
                         "modifier macrodetails-universal/Muscle 0.700000\n")
    with caplog.at_level(logging.INFO):
        human.load(path)
    assert "attempting to load with backward compatibility" in caplog.text
    assert (human.getGender(), human.getMuscle()) == (pytest.approx(0.8), pytest.approx(0.7))


@pytest.mark.parametrize("text", [
    "\nversion v1.3.0\nmodifier macrodetails/Gender 1.000000\n",
    "modifier macrodetails/Gender 1.000000\n",
], ids=["blank-line-first", "no-version-line"])
def test_files_with_blank_lines_or_without_version_load(human, tmp_path, text):
    human.load(mhm(tmp_path, text))
    assert human.getGender() == 1.0
