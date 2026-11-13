"""shared/animation.py: animation tracks, poses, pose units, vertex weights and skinning."""

import json
import logging
import os
import sys

import numpy as np
import pytest

import animation
import bvh
import files3d
import getpath
import human
import progress
import skeleton
import transformations as tm
from core import G

MH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "makehuman")


@pytest.fixture(autouse=True)
def no_opengl_shaders(monkeypatch):
    """Unit tests have no OpenGL context. If another test file imported the real shader
    module, materials would try to compile GLSL (material.getShaderObj) and segfault."""
    shader = sys.modules.get("shader")
    if shader is not None and hasattr(shader, "Shader"):
        monkeypatch.setattr(shader.Shader, "_supported", False)



class FakeApp(object):
    def __init__(self, selectedHuman):
        self.selectedHuman = selectedHuman
        self.progress_calls = 0

    def progress(self, *args, **kwargs):
        self.progress_calls += 1


@pytest.fixture
def mh(monkeypatch):
    """A headless Human with the default skeleton, as G.app.selectedHuman."""
    monkeypatch.chdir(MH_DIR)
    monkeypatch.setattr(progress, "current_Progress_", None)
    h = human.Human(files3d.loadMesh(getpath.getSysDataPath("3dobjs/base.obj"), maxFaces=5))
    monkeypatch.setattr(G, "app", FakeApp(h), raising=False)
    h.setBaseSkeleton(skeleton.load(getpath.getSysDataPath("rigs/default.mhskel"), h.meshData))
    return h


@pytest.fixture
def base(mh):
    return mh.getBaseSkeleton()


@pytest.fixture
def benchmark(base):
    return bvh.load(getpath.getSysDataPath("poses/benchmark.bvh")).createAnimationTrack(base)


@pytest.fixture
def face_units(base):
    """The face pose units, built as the expression plugin does."""
    units = bvh.load(getpath.getSysDataPath("poseunits/face-poseunits.bvh"), allowTranslation="none")
    track = units.createAnimationTrack(base, name="Expression-Face-PoseUnits")
    names = json.load(open(getpath.getSysDataPath("poseunits/face-poseunits.json"), encoding="utf-8"))["framemapping"]
    return animation.PoseUnit(track.name, track._data, names)


def rot(angle_deg, axis):
    return tm.rotation_matrix(np.radians(angle_deg), axis)


def track(frames):
    """AnimationTrack from a list of frames, each a list of 3x4 (or 4x4) matrices."""
    data = np.array([m for frame in frames for m in frame], dtype=np.float32)
    return animation.AnimationTrack("t", data, len(frames), 10)


def translation(x, y=0.0, z=0.0):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (x, y, z)
    return m


# --- AnimationTrack ----------------------------------------------------------

def test_track_layout_and_accessors():
    t = track([[translation(0), translation(1)], [translation(2), translation(3)], [translation(4), translation(5)]])
    assert (t.nFrames, t.nBones, t.dataLen) == (3, 2, 6)
    assert t.data.shape == (6, 3, 4)  # the constant last row is not stored
    assert t.getAtFramePos(1)[1][0, 3] == 3
    assert t.getAtFramePos(2.7)[0][0, 3] == 4  # frame positions are truncated
    assert t.getPlaytime() == pytest.approx(0.3)
    assert not t.isPose() and t.isLooping()
    assert t.description == "t animation"


@pytest.mark.parametrize("data, frames, message", [
    (np.zeros((1, 3, 4)), 2, "contains no or not enough data"),
    (np.zeros((5, 3, 4)), 2, "does not have the proper length"),
    (np.zeros((4, 3, 3)), 2, "does not have the proper dimensions"),
])
def test_track_validation(data, frames, message):
    with pytest.raises(RuntimeError, match=message):
        animation.AnimationTrack("bad", data, frames, 30)


def test_frame_index_and_interpolation():
    t = track([[translation(0)], [translation(10)], [translation(20)]])  # 10 fps
    assert t.getFrameIndexAtTime(0.15) == (1, pytest.approx(0.5))
    assert t.getFrameIndexAtTime(0.35)[0] == 0  # loops: frame 3 -> 0
    assert t.getAtTime(0.15)[0][0, 3] == 10  # no interpolation by default
    t.interpolationType = 1
    assert t.getAtTime(0.15)[0][0, 3] == pytest.approx(15)
    assert t.getAtTime(0.25)[0][0, 3] == pytest.approx(10)  # interpolates towards frame 0 when looping
    assert t.getAtTime(0.1)[0][0, 3] == pytest.approx(10)
    t.interpolationType = 2
    assert t.getAtTime(0.15) is None  # logarithmic interpolation is a TODO
    t.loop = False
    assert t.getFrameIndexAtTime(1.0) == (2, 0)  # clamped to the last frame


def test_set_looping():
    t = track([[translation(0)], [translation(1)]])
    t.setLooping(False)
    assert not t.isLooping()


def test_sparsify():
    t = track([[translation(i)] for i in range(4)])
    t.sparsify(5)
    assert t.nFrames == 2


def test_sparsify_rejects_higher_framerate():
    t = track([[translation(0)], [translation(1)]])
    with pytest.raises(RuntimeError, match="higher than old framerate"):
        t.sparsify(20)
    assert t.nFrames == 2 and t.frameRate == 10


def test_scale_affects_only_translations():
    t = track([[rot(30, [0, 1, 0]) @ translation(1, 2, 3)], [translation(4)]])
    rotation_before = t.data[0][:, :3].copy()
    t.scale(2.0)
    np.testing.assert_allclose(t.data[0][:, 3], 2 * (rot(30, [0, 1, 0]) @ translation(1, 2, 3))[:3, 3], atol=1e-6)
    np.testing.assert_allclose(t.data[1][:, 3], [8, 0, 0])
    np.testing.assert_allclose(t.data[0][:, :3], rotation_before)


def test_bake_matches_skeleton_skinning_matrices(base, benchmark):
    benchmark.bake(base)
    assert benchmark.isBaked()
    assert base.isInRestPose()  # the previous (rest) pose is restored afterwards
    assert G.app.progress_calls > 0
    base.setPose(benchmark.getAtFramePos(0, noBake=True))
    expected = np.array([b.matPoseVerts[:3, :4] for b in base.getBones()])
    np.testing.assert_allclose(benchmark.data, expected, atol=1e-5)
    benchmark.disableBaking = True
    assert not benchmark.isBaked()
    np.testing.assert_allclose(benchmark.data, benchmark._data)
    benchmark.resetBaked()
    benchmark.bake(base)  # disabled: no-op
    assert benchmark._data_baked is None


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="bake() saves the pose with getPose() (bone-local matrices) but restores it with "
                          "setPose(), which expects global-axis matrices (animation.py:139,150), so a "
                          "non-rest pose is not restored")
def test_bake_restores_a_posed_skeleton(base, benchmark):
    other = bvh.load(getpath.getSysDataPath("poses/sit01.bvh")).createAnimationTrack(base)
    base.setPose(benchmark.getAtFramePos(0, noBake=True))
    before = np.array([b.matPoseGlobal for b in base.getBones()])
    other.bake(base)
    np.testing.assert_allclose([b.matPoseGlobal for b in base.getBones()], before, atol=1e-4)


def test_bake_rejects_other_skeleton(base):
    t = track([[translation(0)]])
    with pytest.raises(RuntimeError, match="number of bones in animation data differs"):
        t.bake(base)


def test_join_animations(caplog):
    a = track([[translation(1), translation(2)]])
    b = track([[translation(3), translation(4)], [translation(5), translation(6)]])
    b.frameRate = 24
    joined = animation.joinAnimations(a, b)
    assert joined.name == "t_t" and joined.nFrames == 3 and joined.frameRate == 10
    assert [m[0, 3] for m in joined.data] == [1, 2, 3, 4, 5, 6]
    assert "framerates differ" in caplog.text
    with pytest.raises(RuntimeError, match="same bone count"):
        animation.joinAnimations(a, track([[translation(0)]]))


# --- Poses -------------------------------------------------------------------

def test_pose_basics():
    p = animation.Pose("p", np.array([translation(1), translation(2)]))
    assert p.isPose() and p.nBones == 2
    assert p.getData()[1][0, 3] == 2
    with pytest.raises(NotImplementedError):
        p.sparsify(1)


def test_mix_poses_replaces_selected_bones():
    p1 = animation.Pose("a", np.array([translation(i) for i in range(4)]))
    p2 = animation.Pose("b", np.array([translation(10 + i) for i in range(4)]))
    mixed = animation.mixPoses(p1, p2, [1, 3])
    assert mixed.name == "a_mix_b"
    assert [m[0, 3] for m in mixed.getData()] == [0, 11, 2, 13]
    assert [m[0, 3] for m in p1.getData()] == [0, 1, 2, 3]  # originals untouched
    with pytest.raises(RuntimeError, match="not compatible"):
        animation.mixPoses(p1, animation.Pose("c", np.array([translation(0)])), [0])


def test_blend_poses_trivial_cases():
    p = animation.Pose("p", np.array([translation(7)]))
    assert animation.blendPoses([p], []) is None
    assert animation.blendPoses([p], [1.0])[0][0, 3] == 7


@pytest.mark.xfail(strict=True, raises=(TypeError, ValueError),
                   reason="blendPoses() multiplies Pose objects and unpacks them as (index, pose) pairs "
                          "(animation.py:442-443); it can only handle a single pose")
def test_blend_two_poses():
    a = animation.Pose("a", animation.emptyPose())
    b = animation.Pose("b", animation.emptyPose())
    assert animation.blendPoses([a, b], [0.5, 0.5]) is not None


def test_pose_units(face_units):
    names = face_units.getPoseNames()
    assert names[0] == "Rest" and face_units.nFrames == len(names) == 60
    with pytest.raises(NotImplementedError):
        face_units.sparsify(1)
    smile = names.index("MouthLeftPullUp")
    np.testing.assert_allclose(face_units.getUnitPose("MouthLeftPullUp"), face_units.getUnitPose(smile))
    affected = face_units.getAffectedBones()
    assert len(affected) == 60 and affected[0] == []  # the rest frame moves nothing
    assert face_units.getAffectedBones(smile) == affected[smile] and len(affected[smile]) > 0
    unit = face_units.getUnitPose(smile)
    for b_idx in range(face_units.nBones):
        assert (b_idx in affected[smile]) != animation.isRest(unit[b_idx])


def test_blended_pose_single_and_pair(face_units):
    names = face_units.getPoseNames()
    left = names.index("MouthLeftPullUp")
    full = face_units.getBlendedPose(["MouthLeftPullUp"], [1.0])
    assert isinstance(full, animation.Pose) and full.name == "Expression-Face-PoseUnits-blended"
    np.testing.assert_allclose(full.getData()[:, :, :3], face_units.getUnitPose(left)[:, :, :3], atol=1e-5)
    none = face_units.getBlendedPose([left], [0.0], only_data=True)  # frame indices work too
    np.testing.assert_allclose(none, animation.emptyPose(face_units.nBones), atol=1e-6)

    brow = names.index("LeftBrowDown")
    pair = face_units.getBlendedPose([left, brow], [1.0, 1.0], only_data=True)
    for b_idx in face_units.getAffectedBones(brow):
        if b_idx not in face_units.getAffectedBones(left):
            np.testing.assert_allclose(pair[b_idx][:, :3], face_units.getUnitPose(brow)[b_idx][:, :3], atol=1e-5)
    for m in pair:
        np.testing.assert_allclose(m[:, :3] @ m[:, :3].T, np.eye(3), atol=1e-5)


def test_blended_pose_of_three_and_half_weight(face_units):
    names = ["MouthLeftPullUp", "MouthRightPullUp", "NasolabialDeepener"]
    three = face_units.getBlendedPose(names, [0.5, 0.5, 0.5], only_data=True)
    assert three.shape == (face_units.nBones, 3, 4)
    half = face_units.getBlendedPose(["MouthLeftPullUp"], [0.5], only_data=True)
    unit = face_units.getUnitPose("MouthLeftPullUp")
    b_idx = face_units.getAffectedBones(face_units.getPoseNames().index("MouthLeftPullUp"))[0]
    full_angle = tm.rotation_from_matrix(np.vstack([unit[b_idx], [0, 0, 0, 1]]))[0]
    half_angle = tm.rotation_from_matrix(np.vstack([half[b_idx], [0, 0, 0, 1]]))[0]
    assert abs(half_angle) == pytest.approx(abs(full_angle) / 2, abs=1e-4)  # slerp halfway


@pytest.mark.xfail(strict=True, raises=IndexError,
                   reason="non-additive blending adds the remainder weight element-wise to the weights "
                          "array instead of appending it (animation.py:385), then indexes past its end")
def test_non_additive_blending(face_units):
    face_units.getBlendedPose(["MouthLeftPullUp", "NasolabialDeepener"], [0.3, 0.3], additiveBlending=False)


def test_non_additive_blending_with_full_weights(face_units):
    out = face_units.getBlendedPose(["MouthLeftPullUp"], [2.0], additiveBlending=False, only_data=True)
    # weights are normalised to sum 1
    np.testing.assert_allclose(out, face_units.getBlendedPose(["MouthLeftPullUp"], [1.0], only_data=True),
                               atol=1e-6)


def test_expression_from_unit_poses(face_units, tmp_path):
    pose = animation.poseFromUnitPose("expr", getpath.getSysDataPath("expressions/smile01.mhpose"), face_units)
    assert pose.name == "smile01" and pose.nBones == face_units.nBones and pose.nFrames == 1
    assert "delight" in pose.tags and pose.license.license == "CC0"
    assert dict(pose.unitposes) == {"MouthLeftPullUp": 0.598, "NasolabialDeepener": 0.05, "MouthRightPullUp": 0.226}
    expected = face_units.getBlendedPose(list(pose.unitposes), list(pose.unitposes.values()), only_data=True)
    np.testing.assert_allclose(pose.getData(), expected)

    empty = tmp_path / "empty.mhpose"
    empty.write_text(json.dumps({"name": "e", "unit_poses": {}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="at least one entry"):
        animation.poseFromUnitPose("e", str(empty), face_units)


# --- Vertex weights ----------------------------------------------------------

def test_weights_are_normalised_merged_and_filtered():
    w = animation.VertexBoneWeights({
        "a": [(0, 1.0), (1, 3.0), (0, 1.0), (2, 1e-6)],
        "b": [(1, 1.0), (2, 1.0)],
        "empty": [],
    }, rootBone="a")
    assert w.vertexCount == 3 and list(w.data) == ["a", "b"]
    verts, weights = w.data["a"]
    assert verts.tolist() == [0, 1] and verts.dtype == np.uint32
    np.testing.assert_allclose(weights, [1.0, 0.75])  # doubles merged, tiny weight dropped
    np.testing.assert_allclose(w.data["b"][1], [0.25, 1.0], atol=1e-5)
    assert w.getMaxNumberVertexWeights() == 2


def test_unweighted_vertices_go_to_root(caplog):
    caplog.set_level(logging.DEBUG)
    w = animation.VertexBoneWeights({"spine": [(1, 0.5)], "root": [(3, 1.0)]}, vertexCount=5)
    verts, weights = w.data["root"]
    assert sorted(verts.tolist()) == [0, 2, 3, 4]
    np.testing.assert_allclose(weights, 1.0)
    assert "3 unweighted vertices. [0, 2, 4]" in caplog.text
    caplog.clear()
    big = animation.VertexBoneWeights({"spine": [(0, 1.0)]}, vertexCount=150, rootBone="hips")
    assert len(big.data["hips"][0]) == 149
    assert "149 unweighted vertices." in caplog.text and "[" not in caplog.text


@pytest.mark.xfail(strict=True, raises=IndexError,
                   reason="the check for already-built (verts, weights) tuples reads element [2] of a "
                          "2-tuple (animation.py:647), so passing built data crashes")
def test_prebuilt_weights_are_accepted():
    data = {"root": (np.array([0, 1], np.uint32), np.array([1.0, 1.0], np.float32))}
    assert animation.VertexBoneWeights(data).vertexCount == 2


def test_weights_file_round_trip(tmp_path):
    w = animation.VertexBoneWeights({"a": [(0, 0.5), (1, 1.0)], "b": [(0, 0.5)]}, rootBone="a")
    w.name, w.version, w.description = "mine", "7", "test weights"
    w.license.author = "Tester"
    path = str(tmp_path / "w.mhw")
    w.toFile(path)
    raw = json.load(open(path, encoding="utf-8"))
    assert raw["weights"]["b"] == [[0, 0.5]] and raw["author"] == "Tester"
    back = animation.VertexBoneWeights.fromFile(path, rootBone="a")
    assert (back.name, back.version, back.description) == ("mine", "7", "test weights")
    assert back.license.author == "Tester"
    for bone in ("a", "b"):
        assert back.data[bone][0].tolist() == w.data[bone][0].tolist()
        np.testing.assert_allclose(back.data[bone][1], w.data[bone][1])
    copy = w.create({"c": [(0, 1.0)]})
    assert copy.rootBone == "a" and isinstance(copy, animation.VertexBoneWeights)


def test_compiled_weights_keep_the_strongest(mh, base):
    w = animation.VertexBoneWeights({"root": [(0, 0.5)], "spine05": [(0, 0.3)], "spine04": [(0, 0.2)],
                                     "head": [(1, 1.0)]}, vertexCount=2)
    assert w.compiled() is None and not w.isCompiled()
    compiled = w.compiled(2, skel=base)
    assert w.isCompiled(2) and w.compiled(2) is compiled
    idx = base.getBoneToIdxMapping()
    assert (compiled[0]["b_idx1"], compiled[0]["b_idx2"]) == (idx["root"], idx["spine05"])
    assert (compiled[0]["wght1"], compiled[0]["wght2"]) == (pytest.approx(0.625), pytest.approx(0.375))
    assert (compiled[1]["b_idx1"], compiled[1]["wght1"], compiled[1]["wght2"]) == (idx["head"], 1.0, 0.0)
    w.compileData(base, nWeights=99)  # clamped to the maximum present (3)
    assert w.isCompiled(99) and w.isCompiled(3)
    w.clearCompiled()
    assert not w.isCompiled(3)


def test_compiling_skips_unknown_bones(base, caplog):
    w = animation.VertexBoneWeights({"root": [(0, 1.0)], "no-such-bone": [(1, 1.0)]})
    compiled = w._compileVertexWeights(w.data, base, 1)  # vertexCount derived from the data
    assert len(compiled) == 2 and compiled[1]["wght1"] == 0
    assert "Bone no-such-bone not found in skeleton" in caplog.text


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="_compileVertexWeights only has record layouts for 1-6 and 13 weights; any other "
                          "count (7-12, e.g. the default rig's own maximum of 12) falls back to a single "
                          "weight and crashes (animation.py:765-766, 811)")
def test_compile_seven_weights_per_vertex(base):
    w = animation.VertexBoneWeights({b.name: [(0, 1.0)] for b in base.getBones()[:7]}, vertexCount=1)
    w.compileData(base)
    assert len(w.compiled().dtype) == 14


@pytest.mark.parametrize("n_weights", [1, 2, 3, 4, 5, 6, 13])
@pytest.mark.parametrize("columns", [3, 4])
def test_fast_skinning_matches_reference_skinning(mh, base, benchmark, n_weights, columns):
    rng = np.random.RandomState(n_weights)
    n_verts = 30
    bones = [b.name for b in base.getBones()]
    data = {}
    for v in range(n_verts):
        for b in rng.choice(len(bones), n_weights, replace=False):
            data.setdefault(bones[b], []).append((v, float(rng.uniform(0.1, 1.0))))
    weights = animation.VertexBoneWeights(data, vertexCount=n_verts)
    assert weights.getMaxNumberVertexWeights() == n_weights

    base.setPose(benchmark.getAtFramePos(0, noBake=True))
    skin_mats = np.array([b.matPoseVerts[:3, :4] for b in base.getBones()])
    coords = np.ones((n_verts, 4), dtype=np.float32)
    coords[:, :3] = mh.getRestposeCoordinates()[:n_verts]
    expected = base.skinMesh(coords, weights.data)

    compiled = weights.compiled(n_weights, skel=base)
    fast = animation.skinMesh(coords[:, :columns].copy(), compiled, skin_mats)
    if columns == 4:
        np.testing.assert_allclose(fast, expected, atol=1e-4)
    else:  # without the homogeneous coordinate only the rotation part applies
        assert fast.shape == (n_verts, 3)
        np.testing.assert_allclose(fast, base.skinMesh(coords * [1, 1, 1, 0], weights.data), atol=1e-4)
        assert not np.allclose(fast, expected[:, :3], atol=1e-2)  # translations really were left out


# --- Helpers -----------------------------------------------------------------

def test_empty_tracks_and_rest_detection():
    t = animation.emptyTrack(2, 3)
    assert t.shape == (6, 3, 4) and all(animation.isRest(m) for m in t)
    assert animation.isRest(animation.emptyPose())
    assert animation.isRest(np.eye(4)) and not animation.isRest(translation(1))
    assert not animation.isRest(rot(10, [1, 0, 0])[:3, :4])
    assert animation.animationRelativeToPose(None, None) is None


MHP = """\
# a comment
head matrix 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
neck01 matrix 0 -1 0 0 1 0 0 0 0 0 1 0 0 0 0 1
neck02 quat 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
too short
"""


def test_load_mhp_pose(base, tmp_path, caplog):
    path = tmp_path / "p.mhp"
    path.write_text(MHP, encoding="utf-8")
    pose = animation.loadPoseFromMhpFile(str(path), base)
    assert pose.name == "p" and pose.nBones == base.getBoneCount()
    neck = base.getBone("neck01").index
    # Z and Y rotation components are flipped for MHP files
    np.testing.assert_allclose(pose.getData()[neck], [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0]])
    np.testing.assert_allclose(pose.getData()[base.getBone("head").index], np.eye(4)[:3])
    assert "Unknown keyword in mhp file: quat" in caplog.text
    assert "Too short line in mhp file: too short" in caplog.text


def test_load_valid_mhp_logs_no_error(base, tmp_path, caplog):
    path = tmp_path / "ok.mhp"
    path.write_text("head matrix 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n", encoding="utf-8")
    animation.loadPoseFromMhpFile(str(path), base)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


# --- AnimatedMesh (through Human) -------------------------------------------

def reference_skin(mh, pose_track):
    skel = mh.getBaseSkeleton()
    skel.setPose(pose_track.getAtFramePos(0, noBake=True))
    coords = skel.skinMesh(mh.getRestposeCoordinates(), skel.getVertexWeights().data)
    skel.setToRestPose()
    return coords


def test_posing_the_human_skins_its_mesh(mh, benchmark):
    expected = reference_skin(mh, benchmark)
    rest = mh.getRestposeCoordinates().copy()
    mh.addAnimation(benchmark)
    assert mh.hasAnimation(benchmark.name) and mh.getAnimations() == [benchmark.name]
    assert mh.getAnimation(benchmark.name) is benchmark
    mh.setActiveAnimation(benchmark.name)
    assert mh.getActiveAnimation() is benchmark and mh.isPoseable()
    mh.setPosed(True)
    assert mh.isPosed() and mh.posed
    # fast path: baked animation + the 6 strongest weights per vertex (the rig has up to 12)
    assert benchmark.isBaked()
    counts = mh.getBaseSkeleton().getVertexWeights()._wCounts
    few = counts <= 6
    assert few.sum() > 0.9 * len(counts)
    np.testing.assert_allclose(mh.meshData.coord[few], expected[few], atol=2e-3)
    assert np.abs(mh.meshData.coord[~few] - expected[~few]).max() < 0.5  # approximated, still close
    np.testing.assert_allclose(mh.getRestposeCoordinates(), rest)
    assert not mh.getBaseSkeleton().isInRestPose()  # skeleton synced to the pose

    mh.setPosed(False)
    np.testing.assert_allclose(mh.meshData.coord, rest, atol=1e-6)
    assert mh.getBaseSkeleton().isInRestPose()


def test_unbaked_animation_uses_slow_skinning(mh, benchmark):
    expected = reference_skin(mh, benchmark)
    benchmark.disableBaking = True
    mh.addAnimation(benchmark)
    mh.setActiveAnimation(benchmark.name)
    mh.setPosed(True)
    np.testing.assert_allclose(mh.meshData.coord, expected, atol=1e-4)


def test_playback_time_and_frames(mh):
    frames = [[translation(0)] * mh.getBaseSkeleton().getBoneCount(),
              [translation(1)] * mh.getBaseSkeleton().getBoneCount()]
    anim = track(frames)  # 10 fps
    mh.setToFrame(1)  # no active animation: ignored
    assert mh.getTime() == 0
    mh.addAnimation(anim)
    mh.setActiveAnimation("t")
    mh.setToFrame(1, update=False)
    assert mh.getTime() == pytest.approx(0.1)
    mh.setToTime(0.0)
    np.testing.assert_allclose(mh.getPoseState(noBake=True)[0][:, 3], [0, 0, 0])
    mh.update(0.1)
    assert mh.getTime() == pytest.approx(0.1)
    np.testing.assert_allclose(mh.getPoseState(noBake=True)[0][:, 3], [1, 0, 0])
    mh.setAnimateInPlace(True)
    np.testing.assert_allclose(mh.getPoseState(noBake=True)[:, :, 3], 0)
    mh.setAnimateInPlace(False)
    mh.resetTime()
    assert mh.getTime() == 0
    mh.setToFrame(1)
    mh.resetBakedAnimations()
    assert not anim.isBaked()


def test_removing_animations(mh, benchmark):
    other = animation.Pose("other", animation.emptyPose(mh.getBaseSkeleton().getBoneCount()))
    mh.addAnimation(benchmark)
    mh.addAnimation(other)
    mh.setActiveAnimation("other")
    mh.removeAnimation("other")
    assert mh.getActiveAnimation() is None and mh.getAnimations() == [benchmark.name]
    mh.setActiveAnimation(benchmark.name)
    mh.removeAnimation(benchmark.name)
    mh.addAnimation(benchmark)
    mh.setActiveAnimation(benchmark.name)
    mh.removeAnimations()
    assert mh.getAnimations() == [] and mh.getActiveAnimation() is None


def test_bound_mesh_management(mh, caplog):
    import module3d
    extra = module3d.Object3D("extra")
    extra.setCoords(np.zeros((3, 3), dtype=np.float32))
    extra.setUVs(np.zeros((1, 2), dtype=np.float32))
    extra.setFaces([[0, 1, 2, 0]], [[0, 0, 0, 0]])
    weights = animation.VertexBoneWeights({"root": [(0, 1.0)]}, vertexCount=2)
    mh.addBoundMesh(extra, weights)
    assert "Vertex count of bound mesh extra (3) and vertex its weights (2) differs" in caplog.text
    assert "extra" in mh.getBoundMeshes()
    assert mh.getBoundMesh("extra") == (extra, weights)
    assert mh.containsBoundMesh(extra)
    assert mh.getBoundMesh("nope") == (None, None)

    mh.addBoundMesh(extra, weights)  # same mesh again replaces it
    assert "Attempt to add the same bound mesh extra twice" in caplog.text
    assert mh.getBoundMeshes().count("extra") == 1

    new_weights = animation.VertexBoneWeights({"root": [(0, 1.0)]}, vertexCount=3)
    animation.AnimatedMesh.updateVertexWeights(mh, "extra", new_weights)  # Human overrides the name
    assert mh.getBoundMesh("extra")[1] is new_weights
    new_weights.compileData(mh.getBaseSkeleton(), 1)
    mh.resetCompiledWeights()
    assert not new_weights.isCompiled(1)

    extra.changeCoords(np.ones((3, 3), dtype=np.float32))
    mh._updateOriginalMeshCoords("extra", np.full((3, 3), 2.0, dtype=np.float32))
    np.testing.assert_allclose(mh.getRestCoordinates("extra"), 2.0)
    mh.refreshStaticMeshes(refresh_pose=False)
    np.testing.assert_allclose(mh.getRestCoordinates("extra"), 1.0)

    extra.setCoords(np.zeros((4, 3), dtype=np.float32))  # mesh changed behind our back
    mh.removeBoundMesh("extra")  # restoring its rest coordinates fails silently
    assert "extra" not in mh.getBoundMeshes()
    mh.removeBoundMesh("extra")
    assert "Cannot remove bound mesh extra" in caplog.text


def test_posing_skips_hidden_and_unweighted_meshes(mh, benchmark, caplog):
    import module3d
    loose = module3d.Object3D("loose")
    loose.setCoords(np.ones((3, 3), dtype=np.float32))
    loose.setUVs(np.zeros((1, 2), dtype=np.float32))
    loose.setFaces([[0, 1, 2, 0]], [[0, 0, 0, 0]])
    mh.addBoundMesh(loose, None)
    mh.addAnimation(benchmark)
    mh.setActiveAnimation(benchmark.name)
    mh.onlyAnimateVisible = True
    mh.meshData.setVisibility(False)
    rest = mh.meshData.coord.copy()
    mh.setPosed(True)
    assert "No weights assigned to bound mesh loose" in caplog.text
    np.testing.assert_allclose(mh.meshData.coord, rest)  # hidden: not skinned
    np.testing.assert_allclose(loose.coord, 1.0)


def test_skinning_errors_are_logged_and_raised(mh, benchmark, caplog):
    import module3d
    small = module3d.Object3D("small")
    small.setCoords(np.ones((3, 3), dtype=np.float32))
    small.setUVs(np.zeros((1, 2), dtype=np.float32))
    small.setFaces([[0, 1, 2, 0]], [[0, 0, 0, 0]])
    too_many = animation.VertexBoneWeights({"root": [(4, 1.0)]}, vertexCount=5)
    mh.addBoundMesh(small, too_many)
    mh.addAnimation(benchmark)
    mh.setActiveAnimation(benchmark.name)
    with pytest.raises(ValueError):
        mh.setPosed(True)
    assert "Error skinning mesh small" in caplog.text


def test_animated_mesh_without_skeleton():
    import module3d
    mesh = module3d.Object3D("m")
    mesh.setCoords(np.zeros((3, 3), dtype=np.float32))
    mesh.setUVs(np.zeros((1, 2), dtype=np.float32))
    mesh.setFaces([[0, 1, 2, 0]], [[0, 0, 0, 0]])
    am = animation.AnimatedMesh(None, mesh, None)
    am.addAnimation(animation.Pose("p", animation.emptyPose()))
    am.setActiveAnimation("p")
    assert not am.isPoseable() and not am.isPosed()
    mesh.changeCoords(np.ones((3, 3), dtype=np.float32))
    am.refreshPose()  # no skeleton: back to rest coordinates
    np.testing.assert_allclose(mesh.coord, 0)
    assert am.getActiveAnimation() is None
    am._posed = True
    am._AnimatedMesh__currentAnim = animation.Pose("p", animation.emptyPose())
    am._pose()  # posed but without skeleton: nothing to do
    np.testing.assert_allclose(mesh.coord, 0)
