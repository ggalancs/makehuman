"""shared/skeleton.py: .mhskel rigs, bone matrices, posing, skinning and weight remapping."""

import json
import logging
import math
import os
import sys

import numpy as np
import pytest

import animation
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

    def progress(self, *args, **kwargs):
        pass


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


def load_rig(name, mh):
    return skeleton.load(getpath.getSysDataPath("rigs/%s.mhskel" % name), mh.meshData)


def rot(angle_deg, axis):
    return tm.rotation_matrix(math.radians(angle_deg), axis)


def assert_rotation(m):
    np.testing.assert_allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-5)
    assert np.linalg.det(m[:3, :3]) == pytest.approx(1.0, abs=1e-5)


def small_skeleton(**bone_kwargs):
    """root -> mid -> tip along the spine helpers of the base mesh."""
    skel = skeleton.Skeleton("small")
    skel.addBone("root", None, "pelvis", "spine-4")
    skel.addBone("mid", "root", "spine-4", "spine-3", **bone_kwargs)
    skel.addBone("tip", "mid", "spine-3", "spine-2")
    skel.build()
    return skel


# --- Loading -----------------------------------------------------------------

def test_load_default_rig(base, mh):
    assert base.name == "MakeHuman default skeleton" and base.version == 110
    assert base.description.startswith("MakeHuman default skeleton")
    assert base.getBoneCount() == 163 and [b.name for b in base.roots] == ["root"]
    assert len(base.joint_pos_idxs) == 326 and len(base.planes) == 163
    assert base.hasCustomVertexWeights()
    assert base.vertexWeights.vertexCount == mh.meshData.getVertexCount()
    assert base.vertexWeights.rootBone == "root"
    bones = base.getBones()
    assert [b.index for b in bones] == list(range(163))
    assert all(b.parent.index < b.index for b in bones if b.parent)  # breadth first
    assert base.getJointNames()[:3] == ["root", "spine05", "spine04"]  # depth first
    assert base.getBoneToIdxMapping()["spine05"] == base.getBone("spine05").index
    assert repr(base) == "  <Skeleton MakeHuman default skeleton>"


def test_rest_matrices_are_consistent(base):
    for bone in base.getBones():
        rest = bone.matRestGlobal
        assert_rotation(rest)
        np.testing.assert_allclose(rest[:3, 3], bone.headPos, atol=1e-5)
        np.testing.assert_allclose(rest[:3, 1] * bone.length, bone.tailPos - bone.headPos, atol=1e-4)
        assert bone.getLength() == pytest.approx(bone.length)
        if bone.parent:
            np.testing.assert_allclose(bone.parent.matRestGlobal @ bone.matRestRelative, rest, atol=1e-5)
        # in rest pose skinning matrices are identities and posed = rest positions
        np.testing.assert_allclose(bone.matPoseVerts, np.eye(4), atol=1e-4)
        np.testing.assert_allclose(bone.getHead(), bone.getRestHeadPos(), atol=1e-4)
        np.testing.assert_allclose(bone.getTail(), bone.getRestTailPos(), atol=1e-4)
    assert base.isInRestPose()


def test_joint_positions_from_vertex_indices_and_helpers(base, mh):
    idxs = base.joint_pos_idxs["head____head"]
    np.testing.assert_allclose(base.getJointPosition("head____head", mh),
                               mh.getRestposeCoordinates()[idxs].mean(axis=0))
    helper = mh.meshData.getVerticesForGroups(["joint-pelvis"])
    expected = mh.meshData.coord[helper].mean(axis=0)
    np.testing.assert_allclose(base.getJointPosition("pelvis", mh), expected, atol=1e-6)
    np.testing.assert_allclose(base.getJointPosition("joint-pelvis", mh, rest_coord=False), expected, atol=1e-6)
    np.testing.assert_allclose(base.getJointPosition("head____head", mh, rest_coord=False),
                               mh.meshData.getCoords(idxs).mean(axis=0))


def test_joint_position_errors(base, mh, caplog):
    with pytest.raises(RuntimeError, match="no reference vertices or joint name"):
        base.getJointPosition("", mh)
    np.testing.assert_array_equal(base.getJointPosition("no-such-joint", mh), [0, 0, 0])
    assert "Cannot find position for joint joint-no-such-joint" in caplog.text


def write_rig(tmp_path, data, name="custom.mhskel"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_from_file_skips_orphans_and_bad_planes(mh, tmp_path, caplog):
    path = write_rig(tmp_path, {
        "name": "custom", "version": "3", "description": "d", "plane_map_strategy": 2,
        "license": "AGPL3", "author": "Someone",
        "joints": {"j_head": [0, 1, 2], "empty": []},
        "planes": {"p1": ["pelvis", "spine-4", "spine-3"]},
        "bones": {
            "child": {"head": "spine-4", "tail": "spine-3", "parent": "base", "rotation_plane": "p1"},
            "base": {"head": "pelvis", "tail": "spine-4"},
            "orphan": {"head": "spine-3", "tail": "spine-2", "parent": "nobody"},
            "flat": {"head": "j_head", "tail": "spine-1", "parent": "base",
                     "rotation_plane": [None, None, None], "reference": "spine05",
                     "weights_reference": "spine04"},
        },
    })
    skel = skeleton.load(path, mh.meshData)
    assert (skel.name, skel.version, skel.description, skel.plane_map_strategy) == ("custom", 3, "d", 2)
    assert skel.license.author == "Someone"
    assert sorted(skel.bones) == ["base", "child", "flat"]
    assert "invalid parent bone (orphan)" in caplog.text
    assert "Invalid rotation plane specified for bone flat" in caplog.text
    assert skel.getBone("flat").roll == 0
    assert skel.joint_pos_idxs == {"j_head": [0, 1, 2]}  # empty joint lists are dropped
    assert skel.vertexWeights is None and not skel.hasCustomVertexWeights()
    flat = skel.getBone("flat")
    assert flat.reference_bones == ["spine05"] and flat.weight_reference_bones == ["spine04"]
    assert skel.getBone("child").weight_reference_bones == []  # falls back to reference_bones
    # the plane normal is used as bone X axis
    child = skel.getBone("child")
    np.testing.assert_allclose(child.get_normal(), skeleton.get_normal(skel, "p1", skel.planes), atol=1e-6)


def test_peek_metadata():
    path = os.path.join(MH_DIR, "data", "rigs", "game_engine.mhskel")
    assert skeleton.peekMetadata(path) == ("Game_engine", "Skeleton optimized for game engines",
                                           {"game", "detailed hands", "makehuman™"})


def test_save_and_reload_round_trip(mh, base, tmp_path):
    game = load_rig("game_engine", mh)
    game.getBone("thigh_l").roll = "upperleg01.L____plane"
    out = str(tmp_path / "copy.mhskel")
    game.toFile(out, ref_weights=base.getVertexWeights())
    saved = json.load(open(out, encoding="utf-8"))
    assert saved["weights_file"] == "copy_weights.mhw"
    assert os.path.isfile(str(tmp_path / "copy_weights.mhw"))
    assert saved["bones"]["thigh_l"]["rotation_plane"] == "upperleg01.L____plane"
    assert "parent" not in saved["bones"]["Root"]

    again = skeleton.load(out, mh.meshData)
    assert again.hasCustomVertexWeights()
    assert [b.name for b in again.getBones()] == [b.name for b in game.getBones()]
    for a, b in zip(game.getBones(), again.getBones()):
        assert (a.headJoint, a.tailJoint) == (b.headJoint, b.tailJoint)
        assert (a.parent and a.parent.name) == (b.parent and b.parent.name)
        assert sorted(a.reference_bones) == sorted(b.reference_bones)
        assert sorted(a.weight_reference_bones) == sorted(b.weight_reference_bones)
        np.testing.assert_allclose(a.headPos, b.headPos, atol=1e-5)
    for bone, (verts, weights) in game.getVertexWeights().data.items():
        v2, w2 = again.vertexWeights.data[bone]
        assert verts.tolist() == v2.tolist()
        np.testing.assert_allclose(weights, w2, atol=1e-6)


# --- Vertex weights ----------------------------------------------------------

def test_vertex_weights_are_remapped_through_reference_bones(mh, base, caplog):
    caplog.set_level(logging.DEBUG)
    ref = base.getVertexWeights()
    skel = skeleton.Skeleton("remap")
    skel.addBone("root", None, "pelvis", "spine-4")  # no references: mapped by name
    skel.addBone("torso", "root", "spine-4", "spine-3", reference_bones=["spine05", "spine04"])
    skel.addBone("ghost", "root", "spine-3", "spine-2", weight_reference_bones=["nonexistent"])
    skel.addBone("loner", "root", "spine-2", "spine-1")  # no references, unknown name
    skel.build()
    assert skel.getVertexWeights() is None
    weights = skel.getVertexWeights(ref)
    assert skel.vertexWeights is weights  # remembered on first remap
    assert set(weights.data) == {"root", "torso"}
    torso_verts = set(weights.data["torso"][0].tolist())
    assert torso_verts == set(ref.data["spine05"][0].tolist()) | set(ref.data["spine04"][0].tolist())
    assert "No weights could be mapped to bone ghost" in caplog.text
    assert "No explicit weight reference bone mapping for bone loner" in caplog.text
    # without force_remap the stored weights are returned
    assert skel.getVertexWeights(ref) is weights
    assert skel.getVertexWeights(ref, force_remap=True) is not weights


def test_auto_build_weight_references(mh, base):
    game = load_rig("game_engine", mh)
    explicit = {b.name for b in game.getBones() if b._weight_reference_bones is not None}
    game.autoBuildWeightReferences(base)
    spine3 = game.getBone("spine_03")
    assert sorted(spine3.reference_bones) == ["spine01", "spine02"]
    # breast bones are not referenced by the game rig: their weights go to spine_03
    assert {"breast.L", "breast.R"} <= set(spine3.weight_reference_bones)
    for name in explicit:  # bones with explicit weight references are left alone
        bone = game.getBone(name)
        assert bone._weight_reference_bones is not None
    every_ref = [r for b in game.getBones() for r in b.weight_reference_bones]
    assert len(every_ref) >= len({r for b in game.getBones() for r in b.reference_bones})


# --- Reference planes --------------------------------------------------------

@pytest.mark.parametrize("strategy", [1, 2, 3])
def test_reference_planes_strategies(mh, base, strategy, caplog):
    game = load_rig("game_engine", mh)
    game.plane_map_strategy = strategy
    refs = {b.name: list(b.reference_bones) for b in game.getBones()}
    game.addReferencePlanes(base)
    thigh = game.getBone("thigh_l")
    rolls = [base.getBone(r).roll for r in refs["thigh_l"]]
    expected = {1: rolls[0], 2: rolls[-1], 3: rolls}[strategy]
    assert thigh.roll == expected
    for plane in (rolls if strategy == 3 else [expected]):
        assert len(game.planes[plane]) == 3
        for joint in game.planes[plane]:
            assert game.joint_pos_idxs[joint] == base.joint_pos_idxs[joint]
    # 'Root' has no references and no bone of that name in the default rig
    assert "Root" in caplog.text
    for bone in game.getBones():
        assert_rotation(bone.matRestGlobal)


def test_reference_plane_joints_are_renamed_on_conflict(mh, base):
    skel = small_skeleton(reference_bones=["spine05"])
    plane = base.getBone("spine05").roll
    clashing = base.planes[plane][0]
    skel.joint_pos_idxs[clashing] = [0]  # a different joint with the same name
    skel.joint_pos_idxs[clashing + "_1"] = [1]  # and its first alternative name is taken too
    skel.plane_map_strategy = 1
    skel.addReferencePlanes(base)
    assert skel.getBone("mid").roll == plane
    renamed = clashing + "_2"
    assert skel.planes[plane][0] == renamed
    assert skel.joint_pos_idxs[clashing] == [0] and skel.joint_pos_idxs[clashing + "_1"] == [1]
    assert skel.joint_pos_idxs[renamed] == base.joint_pos_idxs[clashing]


def test_reference_plane_strategy_3_with_unresolved_refs(mh, base, caplog):
    skel = small_skeleton(reference_bones=["nope1", "nope2"])
    skel.addReferencePlanes(base)
    assert skel.getBone("mid").roll == []
    assert "has no resolved references" in caplog.text
    np.testing.assert_allclose(skel.getBone("mid").get_normal(), [0, 1, 0])


@pytest.mark.parametrize("strategy", [1, 2])
def test_reference_plane_unresolved_single_ref(mh, base, strategy, caplog):
    skel = small_skeleton(reference_bones=["nope"])
    skel.plane_map_strategy = strategy
    skel.addReferencePlanes(base)
    assert "unresolved reference nope" in caplog.text
    assert skel.getBone("mid").roll == 0


def test_bone_normal_from_list_of_planes(mh):
    skel = small_skeleton()
    skel.planes = {"a": ["pelvis", "spine-4", "spine-1"], "flat": ["pelvis", "pelvis", "pelvis"]}
    mid = skel.getBone("mid")
    mid.roll = ["a", "flat"]  # degenerate planes are ignored in the average
    np.testing.assert_allclose(mid.get_normal(), skeleton.get_normal(skel, "a", skel.planes), atol=1e-6)
    mid.roll = ["flat"]
    np.testing.assert_allclose(mid.get_normal(), [0, 1, 0])
    mid.roll = "flat"
    np.testing.assert_allclose(mid.get_normal(), [0, 1, 0])


def test_get_normal_of_unknown_plane(mh, caplog):
    np.testing.assert_allclose(skeleton.get_normal(small_skeleton(), "missing", {}), [0, 1, 0])
    assert "No plane with name missing" in caplog.text


# --- Cloning, scaling, posing ------------------------------------------------

def test_scaled_clone(base):
    big = base.scaled(2.0)
    assert big.scale == 2.0 and big.getBoneCount() == base.getBoneCount()
    for name in ("root", "head", "upperarm01.L"):
        np.testing.assert_allclose(big.getBone(name).headPos, 2 * base.getBone(name).headPos, atol=1e-4)
        # bone orientations are copied from the original
        np.testing.assert_allclose(big.getBone(name).matRestGlobal[:3, :3],
                                   base.getBone(name).matRestGlobal[:3, :3], atol=1e-5)
    same = base.clone()
    np.testing.assert_allclose(same.getBone("head").matRestGlobal, base.getBone("head").matRestGlobal,
                               atol=1e-5)


def test_set_pose_with_translation_and_skinning(base, mh):
    n = base.getBoneCount()
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    poses[0, :3, 3] = [0.5, -1.0, 2.0]  # move the root only
    base.setPose(poses)
    assert not base.isInRestPose()
    coords = mh.getRestposeCoordinates()
    skinned = base.skinMesh(coords, base.getVertexWeights().data)  # 3 columns: converted internally
    np.testing.assert_allclose(skinned, coords + [0.5, -1.0, 2.0], atol=1e-3)
    hom = np.ones((len(coords), 4), dtype=np.float32)
    hom[:, :3] = coords
    np.testing.assert_allclose(base.skinMesh(hom, base.getVertexWeights().data), skinned, atol=1e-5)
    np.testing.assert_allclose(base.getPose()[0][:3, 3],
                               np.linalg.inv(base.getBone("root").matRestGlobal)[:3, :3] @ [0.5, -1.0, 2.0],
                               atol=1e-5)
    base.setToRestPose()
    assert base.isInRestPose()


def test_set_pose_rotation_only_and_world_rotation(base):
    n = base.getBoneCount()
    poses = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    neck = base.getBone("neck01")
    poses[neck.index] = rot(30, [0, 1, 0])[:3, :3]
    base.setPose(poses)  # 3x3 matrices: rotations without translation
    # the global rotation is applied to the bone in world space
    np.testing.assert_allclose(neck.matPoseGlobal[:3, :3], rot(30, [0, 1, 0])[:3, :3] @ neck.matRestGlobal[:3, :3],
                               atol=1e-5)
    np.testing.assert_allclose(neck.getHead(), neck.getRestHeadPos(), atol=1e-4)
    assert not neck.isInRestPose() and base.getBone("root").isInRestPose()


def test_skin_mesh_ignores_unknown_bones(base, mh, caplog):
    coords = mh.getRestposeCoordinates()[:5]
    mapping = {"no-such-bone": (np.array([0, 1]), np.array([1.0, 1.0]))}
    np.testing.assert_allclose(base.skinMesh(coords, mapping), np.zeros((5, 3)))
    assert "Could not skin bone no-such-bone" in caplog.text


def test_create_from_pose_bakes_pose_into_rest(base):
    neck = base.getBone("neck01")
    neck.matPose = rot(25, [1, 0, 0]).astype(np.float32)
    base.update()
    baked = base.createFromPose()
    for name in ("neck01", "neck02", "head", "spine01"):
        np.testing.assert_allclose(baked.getBone(name).matRestGlobal, base.getBone(name).matPoseGlobal, atol=1e-4)
    assert baked.isInRestPose()


def test_canonical_bone_names():
    skel = skeleton.Skeleton("x")
    skel.bones = {"Upper Arm-L": skeleton.Bone.__new__(skeleton.Bone)}
    skel.canonalizeBoneNames()
    assert list(skel.bones) == ["upper_arm_l"] and skel.bones["upper_arm_l"].name == "upper_arm_l"


def test_bone_lookup_helpers(mh):
    skel = small_skeleton(reference_bones="spine05")
    assert skel.getBoneByReference("tip").name == "tip"
    assert skel.getBoneByReference("spine05").name == "mid"
    assert skel.getBoneByReference("unknown") is None
    with pytest.raises(RuntimeError, match="already contains a bone named mid"):
        skel.addBone("mid", None, "pelvis", "spine-4")
    with pytest.raises(NotImplementedError):
        skel.transformed(np.eye(4))
    assert skel.compare(skel) is None


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="Skeleton.display() calls bone.display(), which Bone does not define "
                          "(skeleton.py:444)")
def test_display_logs_every_bone(mh, caplog):
    caplog.set_level(logging.DEBUG)
    skel = small_skeleton()
    skel.display()
    assert "<Skeleton small" in caplog.text
    assert "<Bone mid>" in caplog.text
    assert "Bone root" in caplog.text  # Bone.display() logs the matrices


# --- Bone API ----------------------------------------------------------------

def test_bone_basics(mh):
    skel = small_skeleton()
    root, mid, tip = (skel.getBone(n) for n in ("root", "mid", "tip"))
    assert repr(mid) == "  <Bone mid>" and mid.getName() == "mid"
    assert root.isRoot() and not root.hasParent() and root.hasChildren()
    assert mid.hasParent() and not tip.hasChildren()
    assert (root.level, mid.level, tip.level) == (0, 1, 2)
    np.testing.assert_allclose(mid.getRestOffset(), mid.getRestHeadPos() - root.getRestHeadPos())
    np.testing.assert_allclose(root.getRestOffset(), root.getRestHeadPos())
    assert np.linalg.norm(mid.getRestDirection()) == pytest.approx(1.0)
    q = mid.getRestOrientationQuat()
    np.testing.assert_allclose(tm.quaternion_matrix(q)[:3, :3], mid.matRestGlobal[:3, :3], atol=1e-5)
    assert -math.pi <= mid.getRoll() <= math.pi
    assert mid.planes is skel.planes
    mid.matRestGlobal = np.diag([-1.0, -1.0, 1.0, 1.0])  # half turn about Z
    assert mid.getRoll() == pytest.approx(math.pi)
    mid.matRestGlobal = rot(40, [0, 1, 0])
    assert mid.getRoll() == pytest.approx(math.radians(40))


def test_bind_and_relative_matrices(mh):
    mid = small_skeleton().getBone("mid")
    bindmat, bindinv = mid.getBindMatrix([1, 0, 0])
    expected = mid.matRestGlobal.copy()
    expected[:3, 3] += [1, 0, 0]
    np.testing.assert_allclose(bindinv, expected.T, atol=1e-6)
    np.testing.assert_allclose(bindmat @ bindinv, np.eye(4), atol=1e-5)
    np.testing.assert_allclose(mid.getRelativeMatrix(), mid.matRestRelative, atol=1e-5)
    np.testing.assert_allclose(mid.parent.getRelativeMatrix(), mid.parent.matRestGlobal, atol=1e-6)


@pytest.mark.parametrize("orientation, rotation", [
    (0, np.eye(4)), ("yUpFaceZ", np.eye(4)),
    (1, rot(90, [0, 1, 0])), ("yUpFaceX", rot(90, [0, 1, 0])),
    (2, rot(90, [1, 0, 0])), ("zUpFaceNegY", rot(90, [1, 0, 0])),
    (3, rot(90, [0, 0, 1]) @ rot(90, [1, 0, 0])), ("zUpFaceX", rot(90, [0, 0, 1]) @ rot(90, [1, 0, 0])),
])
def test_rest_matrix_orientations(mh, orientation, rotation):
    mid = small_skeleton().getBone("mid")
    offset = np.array([0.0, 1.0, 0.0])
    moved = mid.matRestGlobal.copy()
    moved[:3, 3] += offset
    np.testing.assert_allclose(mid.getRestMatrix(orientation, "y", offset), rotation @ moved, atol=1e-6)
    x_axis = mid.getRestMatrix(orientation, "x", offset)
    np.testing.assert_allclose(x_axis, rotation @ moved @ (rot(-90, [1, 0, 0]) @ rot(90, [0, 1, 0])), atol=1e-6)
    glob = mid.getRestMatrix(orientation, "G", offset)
    np.testing.assert_allclose(glob[:3, :3], np.eye(3))
    np.testing.assert_allclose(glob[:, 3], rotation @ moved[:, 3], atol=1e-6)


def test_rest_matrix_invalid_arguments(mh, caplog):
    mid = small_skeleton().getBone("mid")
    assert mid.getRestMatrix("sideways") is None
    assert mid.getRestMatrix(0, "z") is None
    assert "invalid meshOrientation" in caplog.text and "invalid localBoneAxis" in caplog.text


def test_pose_quaternion_and_rotation_getters(mh):
    bone = small_skeleton().getBone("mid")
    q = tm.quaternion_about_axis(math.pi / 2, [0, 0, 1])
    bone.setPoseQuaternion(q)
    np.testing.assert_allclose(bone.matPose, rot(90, [0, 0, 1]), atol=1e-6)
    np.testing.assert_allclose(bone.getPoseQuaternion(), q, atol=1e-6)
    qw, qx, qy, qz, ax, ay, az = bone.getRotation()  # quaternion * 1000, then Euler 'sxyz' in degrees
    assert qw == pytest.approx(1000 * math.cos(math.pi / 4)) and qz == pytest.approx(1000 * math.sin(math.pi / 4))
    assert (qx, qy) == (pytest.approx(0, abs=1e-6), pytest.approx(0, abs=1e-6))
    assert (ax, ay, az) == (pytest.approx(0, abs=1e-6), pytest.approx(0, abs=1e-6), pytest.approx(90))


def test_set_rotation_composes_static_zyx(mh):
    bone = small_skeleton().getBone("mid")
    a, b, c = 0.3, -0.2, 0.5
    bone.setRotation((a, b, c))
    expected = tm.euler_matrix(a, b, c, axes="szyx")[:3, :3]
    np.testing.assert_allclose(bone.matPose[:3, :3], expected, atol=1e-6)
    assert_rotation(bone.matPose)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="Bone.setRotation documents its angles as x, y, z but passes them to "
                          "euler_matrix(..., 'szyx'), which applies the first angle about Z "
                          "(skeleton.py:1050-1051); getRotation() reads them back as 'sxyz'")
def test_set_rotation_x_angle_rotates_about_x(mh):
    bone = small_skeleton().getBone("mid")
    bone.setRotation((math.radians(90), 0.0, 0.0))
    np.testing.assert_allclose(bone.matPose[:3, :3], rot(90, [1, 0, 0])[:3, :3], atol=1e-6)


@pytest.mark.parametrize("index, axis", [(1, [1, 0, 0]), (2, [0, 1, 0]), (3, [0, 0, 1])])
def test_set_rotation_index_euler(mh, index, axis):
    bone = small_skeleton().getBone("mid")
    assert bone.setRotationIndex(index, 30, useQuat=False) == 1000.0
    np.testing.assert_allclose(bone.matPose[:3, :3], rot(30, axis)[:3, :3], atol=1e-6)


def test_set_rotation_index_quaternion(mh, caplog):
    caplog.set_level(logging.DEBUG)
    bone = small_skeleton().getBone("mid")
    w = bone.setRotationIndex(1, 500, useQuat=True)  # x component 0.5, w renormalised
    assert w == pytest.approx(1000 * math.sqrt(0.75))
    np.testing.assert_allclose(bone.matPose, rot(60, [1, 0, 0]), atol=1e-6)
    bone.matPose = tm.quaternion_matrix([-0.5, 0.5, 0.5, 0.5])
    assert bone.setRotationIndex(2, 2000, useQuat=True) == pytest.approx(0)  # clamped to a unit quaternion


def test_rotate_local_and_world(mh):
    skel = small_skeleton()
    bone = skel.getBone("mid")
    bone.rotate(90, 2, rotWorld=False)
    np.testing.assert_allclose(bone.matPose[:3, :3], rot(90, [0, 0, 1])[:3, :3], atol=1e-6)
    skel.setToRestPose()
    bone.rotate(45, 1, rotWorld=True)
    skel.update()
    np.testing.assert_allclose(bone.matPoseGlobal[:3, :3], rot(45, [0, 1, 0])[:3, :3] @ bone.matRestGlobal[:3, :3],
                               atol=1e-5)
    np.testing.assert_allclose(bone.getPoseFromGlobal(), bone.matPose, atol=1e-5)
    root = skel.getBone("root")
    np.testing.assert_allclose(root.getPoseFromGlobal(), root.matPose, atol=1e-5)


def test_rotate_rest_changes_rest_orientation(mh):
    skel = small_skeleton()
    root, mid = skel.getBone("root"), skel.getBone("mid")
    root_before = root.matRestGlobal.copy()
    mid_relative = mid.matRestRelative.copy()
    turn = rot(90, [0, 1, 0])
    root.rotateRest(turn)
    np.testing.assert_allclose(root.matRestGlobal, root_before @ turn, atol=1e-5)
    mid.rotateRest(np.eye(4))  # children pick up the parent's new rest orientation
    np.testing.assert_allclose(mid.matRestGlobal, root_before @ turn @ mid_relative, atol=1e-5)
    np.testing.assert_allclose(mid.matPoseVerts, np.eye(4), atol=1e-5)  # still in rest pose


def test_pole_target_rotates_about_bone_axis(mh):
    bone = small_skeleton().getBone("mid")
    y_before = bone.matPoseGlobal[:3, 1].copy()
    head = bone.getHead()
    goal = head + y_before
    bone.poleTargetCorrect(head, goal, head + bone.matPoseGlobal[:3, 0] + bone.matPoseGlobal[:3, 2], 0.0)
    np.testing.assert_allclose(bone.matPoseGlobal[:3, 1], y_before, atol=1e-5)  # still along the bone
    assert_rotation(bone.matPoseGlobal)
    # with the pole on the bone axis nothing happens
    m = bone.matPoseGlobal.copy()
    bone.poleTargetCorrect(head, goal, goal, 1.0)
    np.testing.assert_allclose(bone.matPoseGlobal, m)


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="Bone.stretchTo uses self.roll_angle, which nothing defines (skeleton.py:1101); "
                          "the IK helper has no callers left")
def test_stretch_to(mh):
    bone = small_skeleton().getBone("mid")
    bone.stretchTo(bone.getHead() + [0, 1, 0], True)


def test_update_tolerates_singular_rest_matrix(mh, caplog):
    caplog.set_level(logging.DEBUG)
    bone = small_skeleton().getBone("root")
    previous = bone.matPoseVerts
    bone.matRestGlobal = np.zeros((4, 4))
    bone.update()
    assert bone.matPoseVerts is previous
    assert "Cannot calculate pose verts matrix for bone root" in caplog.text


# --- Module helpers ----------------------------------------------------------

def test_axis_conversions():
    np.testing.assert_allclose(skeleton.toZisUp3([0, 1, 0]), [0, 0, 1])
    np.testing.assert_allclose(skeleton.fromZisUp4(np.eye(4))[:3, :3], skeleton.YZRotation[:3, :3])
    np.testing.assert_allclose(skeleton.fromZisUp4(skeleton.ZYRotation), np.eye(4))


def test_get_matrix_builds_orthonormal_base():
    m = skeleton.getMatrix(np.array([1.0, 2, 3]), np.array([1.0, 4, 3]), np.array([1.0, 0.2, 0]))
    assert_rotation(m)
    np.testing.assert_allclose(m[:3, 1], [0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(m[:3, 0], [1, 0, 0], atol=1e-6)  # normal, made perpendicular
    np.testing.assert_allclose(m[:3, 3], [1, 2, 3])


@pytest.mark.parametrize("tail, roll", [
    ([0, 0, -3], 0),       # along +Y in Z-up space
    ([0, 0, 3], 0),        # along -Y in Z-up space
    ([2, 0, 0], 0.5),      # general direction, with roll
    ([0, 0, 0], 0),        # zero length
])
def test_legacy_get_matrix(tail, roll):
    head = np.array([0.0, 0.0, 0.0])
    length, m = skeleton._getMatrix(head, np.asarray(tail, dtype=float), roll)
    assert length == pytest.approx(np.linalg.norm(tail))
    assert_rotation(m)
    np.testing.assert_allclose(m[:3, 3], head)


@pytest.mark.parametrize("tail, normal, expected", [
    ([0, 0, -1], [-1, 0, 0], 0),                       # half-turn quaternion: reported as 0
    ([-0.322, -0.384, 1.134], [-0.747, 0.394, -0.536], None),
    ([1.624, -0.612, -0.528], [-0.423, 0.315, -0.85], None),   # wraps from above pi
])
def test_get_roll_to(tail, normal, expected):
    roll = skeleton.get_roll_to(np.zeros(3), np.array(tail, dtype=float), np.array(normal, dtype=float))
    assert -math.pi <= roll <= math.pi
    if expected is not None:
        assert roll == expected


def test_get_roll_to_warns_on_non_orthogonal_input(caplog):
    skeleton.get_roll_to(np.zeros(3), np.array([1.0, 0, 0]), np.array([2.0, 0, 0]))
    assert "not orthogonal" in caplog.text


def test_assert_orthogonal():
    assert skeleton.assertOrthogonal(rot(30, [1, 1, 0])[:3, :3])
    with pytest.raises(AssertionError, match="Not orthogonal"):
        skeleton.assertOrthogonal(np.diag([1.0, 2.0, 1.0]))


def test_normalize_quaternion_keeps_sign():
    q = [0.0, 0.6, 0.0, 0.0]
    skeleton._normalizeQuaternion(q)
    assert q[0] == pytest.approx(0.8)
    q = [-1.0, 0.0, 0.8, 0.0]
    skeleton._normalizeQuaternion(q)
    assert q[0] == pytest.approx(-0.6)
    q = [1.0, 2.0, 0.0, 0.0]
    skeleton._normalizeQuaternion(q)
    assert q[0] == 0


def test_copy_normal(mh, base):
    same = base.getBone("head")
    np.testing.assert_allclose(skeleton.copy_normal(same, base), same.matRestGlobal[:3, 0])
    one = small_skeleton(reference_bones="neck01")
    np.testing.assert_allclose(skeleton.copy_normal(one.getBone("mid"), base),
                               base.getBone("neck01").matRestGlobal[:3, 0])
    two = small_skeleton(reference_bones=["neck01", "neck02"])
    expected = (base.getBone("neck01").matRestGlobal[:3, 0] + base.getBone("neck02").matRestGlobal[:3, 0]) / 2
    np.testing.assert_allclose(skeleton.copy_normal(two.getBone("mid"), base), expected, atol=1e-6)
    np.testing.assert_allclose(skeleton.copy_normal(two.getBone("tip"), base), [0, 1, 0])  # unmapped


def test_copy_normal_cancelling_references_fall_back(mh, base):
    skel = small_skeleton(reference_bones=["upperarm01.L", "upperarm01.R"])
    left, right = base.getBone("upperarm01.L"), base.getBone("upperarm01.R")
    right.matRestGlobal = right.matRestGlobal.copy()
    right.matRestGlobal[:3, 0] = -left.matRestGlobal[:3, 0]
    np.testing.assert_allclose(skeleton.copy_normal(skel.getBone("mid"), base), [0, 1, 0])


def test_vertex_bone_weights_import_alias():
    assert skeleton.VertexBoneWeights is animation.VertexBoneWeights
