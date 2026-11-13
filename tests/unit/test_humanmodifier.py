"""apps/humanmodifier.py: modifiers turn slider values into weighted targets.

Uses the real base mesh, targets and modifier definitions from makehuman/data.
Only the running application is replaced (FakeApp: progress bar and redraw).
"""

import json
import logging
import os
import sys

import numpy as np
import pytest

from core import G

MH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "makehuman")


class FakeApp(object):
    def __init__(self):
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
    mp.chdir(MH_DIR)
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


@pytest.fixture(scope="module")
def shared(app):
    import getpath
    import humanmodifier
    h = new_human()
    humanmodifier.loadModifiers(getpath.getSysDataPath("modifiers/modeling_modifiers.json"), h)
    h.applyAllTargets()
    return h


@pytest.fixture
def human(app, shared):
    app.selectedHuman = shared
    yield shared
    shared.symmetryModeEnabled = False
    shared.blockEthnicUpdates = False
    shared.resetMeshValues()
    shared.updateMacroModifiers()
    shared.applyAllTargets()


@pytest.fixture
def bare(app):
    h = new_human()
    app.selectedHuman = h
    return h


def nose(h=None):
    import humanmodifier
    m = humanmodifier.UniversalModifier("nose", "nose-trans", "down", "up")
    if h is not None:
        m.setHuman(h)
    return m


def target(h, path):
    import algos3d
    return algos3d.getTarget(h.meshData, path)


# --- the base class -----------------------------------------------------------------------------

def test_base_modifier_is_abstract():
    import humanmodifier
    m = humanmodifier.Modifier("group/sub", "a/b")
    assert (m.groupName, m.name, m.fullName) == ("group-sub", "a-b", "group-sub/a-b")
    assert (m.getMin(), m.getMax(), m.getDefaultValue()) == (0.0, 1.0, 0)
    assert not m.isMacro()
    with pytest.raises(NotImplementedError):
        m.clampValue(0.5)
    with pytest.raises(NotImplementedError):
        m.getFactors(0.5)
    assert str(m) == repr(m) == "Modifier group-sub/a-b"


@pytest.mark.parametrize("name, side, opposite", [
    ("l-eye-bag-decr|incr", "l", "eyes/r-eye-bag-decr|incr"),
    ("r-eye-bag-decr|incr", "r", "eyes/l-eye-bag-decr|incr"),
    ("eye-bag-l-r", "l", "eyes/eye-bag-r-l"),
    ("nose-trans-down|up", None, None),
])
def test_symmetry_names(name, side, opposite):
    import humanmodifier
    m = humanmodifier.Modifier("eyes", name)
    assert m.getSymmetricOpposite() == opposite  # before getSymmetrySide: computes it
    assert m.getSymmetrySide() == side
    assert m.getSymmetrySide() == side  # cached


def test_get_target_weights():
    import humanmodifier
    targets = [("a.target", ["male", "young"]), ("b.target", ["female"])]
    factors = {"male": 0.5, "young": 0.4, "female": 0.25}
    assert humanmodifier.getTargetWeights(targets, factors) == {"a.target": 0.2, "b.target": 0.25}
    assert humanmodifier.getTargetWeights(targets, factors, value=2.0) == {"a.target": 0.4, "b.target": 0.5}
    with pytest.raises(KeyError):
        humanmodifier.getTargetWeights(targets, {"male": 1.0})
    assert humanmodifier.getTargetWeights(targets, {"male": 0.5}, ignoreNotfound=True) == \
        {"a.target": 0.5, "b.target": 1.0}


# --- universal (target) modifiers -------------------------------------------------------------------

def test_universal_two_sided_modifier(bare):
    m = nose(bare)
    assert m.fullName == "nose/nose-trans-down|up"
    assert (m.left, m.right, m.center) == ("nose-nose-trans-down", "nose-nose-trans-up", None)
    assert [os.path.basename(t[0]) for t in m.l_targets] == ["nose-trans-down.target"]
    assert [os.path.basename(t[0]) for t in m.r_targets] == ["nose-trans-up.target"]
    assert m.targets == m.l_targets + m.r_targets
    assert (m.getMin(), m.getMax()) == (-1.0, 1.0)
    assert (m.clampValue(-3), m.clampValue(3)) == (-1.0, 1.0)
    down, up = m.l_targets[0][0], m.r_targets[0][0]
    m.setValue(0.6)
    assert (bare.getDetail(down), bare.getDetail(up)) == (0.0, pytest.approx(0.6))
    assert m.getValue() == pytest.approx(0.6)
    m.setValue(-0.25)
    assert (bare.getDetail(down), bare.getDetail(up)) == (pytest.approx(0.25), 0.0)
    assert m.getValue() == pytest.approx(-0.25)
    assert m.resetValue() == pytest.approx(-0.25)
    assert m.getValue() == 0 and bare.targetsDetailStack == {}


def test_universal_three_way_modifier(bare):
    import humanmodifier
    m = humanmodifier.UniversalModifier("nose", "nose-trans", "down", "up", "forward")
    m.setHuman(bare)
    assert m.fullName == "nose/nose-trans-down|forward|up"
    center = m.c_targets[0][0]
    assert os.path.basename(center) == "nose-trans-forward.target"
    m.setValue(0.0)
    assert bare.getDetail(center) == 1.0  # centre target fully on at 0
    m.setValue(-0.5)
    assert bare.getDetail(center) == 0.5 and bare.getDetail(m.l_targets[0][0]) == 0.5


def test_universal_one_sided_modifier(bare):
    import humanmodifier
    m = humanmodifier.UniversalModifier("stomach", "stomach-pregnant-incr")
    m.setHuman(bare)
    assert m.left is None and m.getMin() == 0.0
    assert m.clampValue(-1.0) == 0.0
    m.setValue(0.7)
    assert m.getValue() == pytest.approx(0.7)


def test_targets_and_dependencies_of_a_group(app):
    import humanmodifier
    MTM = humanmodifier.ManagedTargetModifier
    assert MTM.findTargets(None) == []
    assert MTM.findTargets("nose-no-such-target") == []
    (path, deps), = MTM.findTargets("nose-nose-trans-up")
    assert path.endswith("nose/nose-trans-up.target") and deps == ["nose-nose-trans-up"]
    breast = MTM.findTargets("breast")
    assert breast and all("breast" in deps for _, deps in breast)
    assert MTM.findMacroDependencies(None) == set()
    assert MTM.findMacroDependencies("nose-nose-trans-up") == set()
    assert MTM.findMacroDependencies("macrodetails-height") == {"gender", "age", "muscle", "weight", "height"}


def test_dependent_targets_follow_the_macros(human):
    """Breast targets depend on gender, age, muscle and weight."""
    size = human.getModifier("breast/BreastSize")
    size.setValue(1.0)
    weights = dict((p, human.getDetail(p)) for p, _ in size.targets if human.getDetail(p))
    assert weights and all("maxcup" in p for p in weights)
    human.setGender(0.0)  # propagates to the breast group
    female = dict((p, human.getDetail(p)) for p, _ in size.targets if human.getDetail(p))
    assert female != weights
    assert sum(female.values()) == pytest.approx(1.0)  # all female now: full weight


# --- macro modifiers ------------------------------------------------------------------------------------

def test_macro_modifier(human):
    import humanmodifier
    gender = human.getModifier("macrodetails/Gender")
    assert isinstance(gender, humanmodifier.MacroModifier) and gender.isMacro()
    assert (gender.variable, gender.getMacroVariable(), gender.macroVariable) == ("Gender", "gender", "gender")
    assert "gender" not in gender.macroDependencies and "age" in gender.macroDependencies
    assert gender.getDefaultValue() == 0.5
    assert gender.clampValue(3) == 1.0 and gender.clampValue(-3) == 0.0
    factors = gender.getFactors(0.3)
    assert factors["macrodetails"] == 1.0 and factors["male"] == human.maleVal
    gender.setValue(0.3)
    assert human.getGender() == pytest.approx(0.3) and gender.getValue() == pytest.approx(0.3)
    gender.buildLists()  # nothing to collect for macro modifiers
    assert gender.verts is None


def test_race_modifiers_control_the_race_variable(human):
    asian = human.getModifier("macrodetails/Asian")
    assert asian.getMacroVariable() == "race"
    assert asian.getDefaultValue() == pytest.approx(1 / 3.)


def test_unknown_macro_variable(app):
    import humanmodifier
    m = humanmodifier.MacroModifier("custom", "Wings")
    assert m.getMacroVariable() is None and not m.isMacro()
    assert m.targets == [] and m.macroDependencies == []


def test_resetting_one_ethnic_modifier_resets_all_three(human):
    human.setAsian(0.8)
    african = human.getModifier("macrodetails/African")
    old = african.resetValue()
    assert old == {"macrodetails/African": pytest.approx(0.1), "macrodetails/Asian": pytest.approx(0.8),
                   "macrodetails/Caucasian": pytest.approx(0.1)}
    third = pytest.approx(1 / 3.)
    assert (human.getAfrican(), human.getAsian(), human.getCaucasian()) == (third, third, third)
    assert human.blockEthnicUpdates is False


def test_similar_modifiers(human):
    import humanmodifier
    asian = human.getModifier("macrodetails/Asian")
    similar = asian.getSimilar()
    assert {m.fullName for m in similar} == {"macrodetails/African", "macrodetails/Caucasian"}
    assert all(isinstance(m, humanmodifier.EthnicModifier) for m in similar)


# --- realtime updates (slider dragging) --------------------------------------------------------------

def test_update_value_moves_only_the_affected_vertices(bare, monkeypatch):
    h = bare
    m = nose(h)
    h.applyAllTargets()
    rest = h.meshData.coord.copy()
    events = []
    monkeypatch.setattr(h, "onChanging", events.append, raising=False)
    m.updateValue(0.5)
    up = target(h, m.r_targets[0][0])
    expected = rest.copy()
    expected[up.verts] += 0.5 * up.data
    np.testing.assert_allclose(h.meshData.coord, expected, atol=1e-5)
    assert m.verts is not None and set(m.verts) >= set(up.verts)
    assert len(m.faces) > 0
    assert [(e.change, e.modifier) for e in events] == [("modifier", "nose/nose-trans-down|up")]
    m.updateValue(0.5)  # same value: nothing to apply
    np.testing.assert_allclose(h.meshData.coord, expected, atol=1e-5)
    m.updateValue(-0.5, updateNormals=0, skipUpdate=True)  # no event when skipping the update
    down = target(h, m.l_targets[0][0])
    expected = rest.copy()
    expected[down.verts] += 0.5 * down.data
    np.testing.assert_allclose(h.meshData.coord, expected, atol=1e-5)
    assert len(events) == 2


def test_build_lists_collects_the_vertices_of_all_targets(bare):
    m = nose(bare)
    m.buildLists()
    up, down = target(bare, m.r_targets[0][0]), target(bare, m.l_targets[0][0])
    assert sorted(m.verts) == sorted(set(up.verts) | set(down.verts))
    first = m.verts
    m.buildLists()  # only once
    assert m.verts is first


def test_update_value_on_a_posed_human(bare):
    import bvh
    import skeleton
    h = bare
    m = nose(h)
    h.setBaseSkeleton(skeleton.load("data/rigs/default.mhskel", h.meshData))
    anim = bvh.load("data/poses/benchmark.bvh", "auto").createAnimationTrack(h.getBaseSkeleton())
    h.addAnimation(anim)
    h.setActiveAnimation(anim.name)
    h.setPosed(True)
    rest = h.getRestposeCoordinates().copy()
    posed = h.meshData.coord.copy()
    m.updateValue(1.0)
    up = target(h, m.r_targets[0][0])
    quick = h.meshData.coord.copy()
    others = np.setdiff1d(np.arange(len(posed)), up.verts)
    np.testing.assert_array_equal(quick[others], posed[others])
    # The offsets are posed with the vertices (fast approximation while dragging) ...
    assert not np.allclose(quick[up.verts] - posed[up.verts], up.data, atol=1e-3)
    np.testing.assert_array_equal(h.getRestposeCoordinates(), rest)
    # ... and agree with properly re-applying all targets afterwards
    h.applyAllTargets()
    np.testing.assert_allclose(h.meshData.coord, quick, atol=1e-4)


def test_realtime_updates_only_reach_realtime_groups(human):
    human.setHeight(1.0)
    height = human.getModifier("macrodetails-height/Height")
    universal = human.getModifier("macrodetails-universal/Weight")
    gender = human.getModifier("macrodetails/Gender")

    def active(m):
        return sorted(os.path.basename(p) for p, _ in m.targets if human.getDetail(p))

    height_before = active(height)
    gender.updateValue(1.0)  # slider dragged
    assert all("-male-" in "-" + t for t in active(universal))  # realtime group updated
    assert active(height) == height_before  # not realtime: waits for the release
    gender.setValue(1.0)  # slider released
    assert active(height) != height_before
    assert all(t.startswith("male-") for t in active(height))


# --- undoable actions ------------------------------------------------------------------------------------

def test_modifier_action_do_and_undo(human):
    import humanmodifier
    m = human.getModifier("nose/nose-trans-down|up")
    posted = []
    action = humanmodifier.ModifierAction(m, 0.0, 0.8, lambda: posted.append(m.getValue()))
    assert action.name == "Change modifier"
    assert action.do() is True
    assert m.getValue() == pytest.approx(0.8) and posted == [pytest.approx(0.8)]
    assert action.undo() is True
    assert m.getValue() == 0.0 and len(posted) == 2


def test_modifier_action_with_symmetry(human):
    import humanmodifier
    left = human.getModifier("eyes/l-eye-bag-decr|incr")
    right = human.getModifier("eyes/r-eye-bag-decr|incr")
    human.symmetryModeEnabled = True
    action = humanmodifier.ModifierAction(left, 0.0, 0.6, lambda: None)
    action.do()
    assert left.getValue() == pytest.approx(0.6) and right.getValue() == pytest.approx(0.6)
    action.undo()
    assert left.getValue() == 0.0 and right.getValue() == 0.0
    left.setValue(0.4)
    right.setValue(0.4)
    reset = humanmodifier.ModifierAction(left, None, None, lambda: None)
    reset.do()
    assert left.getValue() == 0.0 and right.getValue() == 0.0
    assert reset.before == pytest.approx(0.4)
    reset.undo()
    assert left.getValue() == pytest.approx(0.4) and right.getValue() == pytest.approx(0.4)


def test_modifier_action_reset_of_ethnic_values(human):
    import humanmodifier
    human.setAfrican(0.6)
    before = (human.getAfrican(), human.getAsian(), human.getCaucasian())
    action = humanmodifier.ModifierAction(human.getModifier("macrodetails/African"), None, None,
                                          lambda: None)
    action.do()
    assert human.getAfrican() == pytest.approx(1 / 3.)
    assert isinstance(action.before, dict) and len(action.before) == 3
    action.undo()
    after = (human.getAfrican(), human.getAsian(), human.getCaucasian())
    assert after == pytest.approx(before, abs=1e-6)


# --- simple modifiers --------------------------------------------------------------------------------------

def test_simple_modifier(bare):
    import humanmodifier
    m = humanmodifier.SimpleModifier("custom", "data/targets/nose", "nose-base-up.target")
    assert m.fullName == "custom/nose-base-up"
    assert m.filename == os.path.join("data/targets/nose", "nose-base-up.target")
    assert m.targets == [(m.filename, ["dummy"])]
    assert m.getFactors(0.3) == {"dummy": 1.0}
    assert (m.clampValue(-1), m.clampValue(2)) == (0.0, 1.0)
    m.setHuman(bare)
    m.setValue(0.4)
    assert bare.getDetail(m.filename) == pytest.approx(0.4)
    assert m.getValue() == pytest.approx(0.4)
    m.setValue(0.9, skipDependencies=True)
    assert bare.getDetail(m.filename) == pytest.approx(0.9)
    deep = humanmodifier.SimpleModifier("custom", "data/targets", "nose\\nose-base-up.target")
    assert deep.name == "nose-nose-base-up"


# --- loading definitions -----------------------------------------------------------------------------------

def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def test_load_modifiers_from_a_definition_file(bare, tmp_path, caplog):
    import humanmodifier
    defs = write_json(tmp_path / "my_modifiers.json", [
        {"group": "nose", "modifiers": [
            {"target": "nose-trans", "min": "down", "max": "up", "defaultValue": 0.25},
            {"target": "nose-trans", "min": "in", "max": "out", "modifierType": "UniversalModifier"}]},
        {"group": "macrodetails", "modifiers": [
            {"macrovar": "Gender"},
            {"macrovar": "Asian", "modifierType": "EthnicModifier"}]},
        {"group": "custom", "modifiers": [{"macrovar": "Wings"}]},
    ])
    write_json(tmp_path / "my_modifiers_desc.json", {
        "nose/nose-trans-down|up": "Moves the nose up or down",
        "nose/no-such-modifier": "orphan description"})
    with caplog.at_level(logging.DEBUG):
        mods = humanmodifier.loadModifiers(defs, bare)
    assert [type(m).__name__ for m in mods] == ["UniversalModifier", "UniversalModifier",
                                                "MacroModifier", "EthnicModifier", "MacroModifier"]
    assert [m.fullName for m in mods] == ["nose/nose-trans-down|up", "nose/nose-trans-in|out",
                                          "macrodetails/Gender", "macrodetails/Asian", "custom/Wings"]
    assert mods[0].getDefaultValue() == 0.25
    assert mods[0].description == "Moves the nose up or down"
    assert all(m.human is bare for m in mods)
    assert sorted(bare.modifierNames) == sorted(m.fullName for m in mods)
    text = caplog.text
    assert "Expected modifier custom/Wings to be a macro modifier" in text
    assert "Loaded description for nose/no-such-modifier but modifier does not exist!" in text
    assert "Loaded 1 modifier descriptions" in text
    assert "No description defined for modifier nose/nose-trans-in|out!" in text
    assert "Loaded 5 modifiers from file" in text


def test_load_modifiers_without_human_or_descriptions(app, tmp_path, caplog):
    import humanmodifier
    defs = write_json(tmp_path / "plain.json", [
        {"group": "nose", "modifiers": [{"target": "nose-trans", "min": "down", "max": "up"}]}])
    with caplog.at_level(logging.WARNING):
        mods = humanmodifier.loadModifiers(defs, None)
    assert len(mods) == 1 and mods[0].human is None
    assert "No description defined for modifier nose/nose-trans-down|up!" in caplog.text


@pytest.mark.parametrize("name, count", [("modeling", 249), ("bodyshapes", 22), ("measurement", 20)])
def test_shipped_modifier_definitions(app, name, count):
    """Every shipped modifier controls targets that exist on disk."""
    import getpath
    import humanmodifier
    mods = humanmodifier.loadModifiers(getpath.getSysDataPath("modifiers/%s_modifiers.json" % name), None)
    assert len(mods) == count
    assert len(set(m.fullName for m in mods)) == count
    for m in mods:
        assert m.targets, m.fullName
        for path, _ in m.targets:
            assert os.path.isfile(path), path


def test_debug_modifiers_lists_every_modifier(human, app, caplog):
    import humanmodifier
    with caplog.at_level(logging.DEBUG):
        humanmodifier.debugModifiers()
    text = caplog.text
    assert "MacroModifier macrodetails/Gender:" in text
    assert "    controls: gender" in text
    assert text.count("    description: ") == len(human.modifiers)
