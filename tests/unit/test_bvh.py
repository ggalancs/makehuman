"""shared/bvh.py: BVH motion files (parsing, axis conversion, export, tracks)."""

import os
import sys
from math import pi

import numpy as np
import pytest

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


# Root with translation, two joints named "Spine" (the second becomes Spine_1)
SMALL = """\
HIERARCHY
ROOT Hips
{
\tOFFSET 0 0 0
\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
\tJOINT Spine
\t{
\t\tOFFSET 0 1 0
\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\tJOINT Head
\t\t{
\t\t\tOFFSET 0 2 0
\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\tEnd Site
\t\t\t{
\t\t\t\tOFFSET 0 0.5 0
\t\t\t}
\t\t}
\t}
\tJOINT Spine
\t{
\t\tOFFSET 1 0 0
\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\tEnd Site
\t\t{
\t\t\tOFFSET 0 -1 0
\t\t}
\t}
}
MOTION
Frames: 2
Frame Time: 0.5
1 2 3 0 0 0 0 0 0 0 0 0 0 0 0
4 5 6 90 0 0 0 90 0 0 0 0 0 0 0
"""

# Translation-only root, a joint with a translation and a single rotation channel
TRANSLATING = """\
HIERARCHY
ROOT A
{
\tOFFSET 0 0 0
\tCHANNELS 3 Xposition Yposition Zposition
\tJOINT B
\t{
\t\tOFFSET 0 1 0
\t\tCHANNELS 2 Yposition Xrotation
\t\tEnd Site
\t\t{
\t\t\tOFFSET 0 1 0
\t\t}
\t}
}
MOTION
Frames: 1
Frame Time: 0.1
1 2 3 7 45
"""


def rot(angle_deg, axis):
    return tm.rotation_matrix(angle_deg * pi / 180, axis)[:3, :3]


def write(tmp_path, text, name="small.bvh"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


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
    skel = skeleton.load(getpath.getSysDataPath("rigs/default.mhskel"), h.meshData)
    h.setBaseSkeleton(skel)
    return h


def test_parse_hierarchy_and_motion(tmp_path):
    b = bvh.load(write(tmp_path, SMALL), convertFromZUp=False)
    assert b.name == "small"
    assert b.frameCount == 2 and b.frameTime == 0.5
    assert sorted(b.joints) == ["Head", "Hips", "Spine", "Spine_1"]  # end sites are not named joints
    assert [j.name for j in b.getJointsBVHOrder()] == \
        ["Hips", "Spine", "Head", "End effector", "Spine_1", "End effector"]
    assert [j.name for j in b.getJoints()] == \
        ["Hips", "Spine", "Spine_1", "Head", "End effector", "End effector"]  # breadth first
    head = b.getJoint("Head")
    np.testing.assert_allclose(head.position, [0, 3, 0])
    np.testing.assert_allclose(head.children[0].position, [0, 3.5, 0])
    np.testing.assert_allclose(head.matRestGlobal[:3, 3], [0, 3, 0])
    np.testing.assert_allclose(b.getJoint("Spine_1").children[0].offset, [0, -1, 0])
    assert head.hasParent() and not head.isRoot() and head.hasChildren() and not head.isEndConnector()
    assert b.rootJoint.isRoot() and b.rootJoint.getName() == "Hips"


def test_channel_data_becomes_pose_matrices(tmp_path):
    b = bvh.load(write(tmp_path, SMALL), convertFromZUp=False)
    hips, spine = b.getJoint("Hips"), b.getJoint("Spine")
    assert hips.matrixPoses.shape == (2, 3, 4)
    assert hips.rotOrder == "syxz"
    np.testing.assert_allclose(hips.matrixPoses[0][:, 3], [1, 2, 3])
    np.testing.assert_allclose(hips.matrixPoses[1][:, 3], [4, 5, 6])
    np.testing.assert_allclose(hips.matrixPoses[1][:, :3], rot(90, [0, 0, 1]), atol=1e-6)
    np.testing.assert_allclose(spine.matrixPoses[1][:, :3], rot(90, [1, 0, 0]), atol=1e-6)
    np.testing.assert_allclose(spine.matrixPoses[:, :, 3], 0)  # translation only on the root


def test_z_up_conversion(tmp_path):
    b = bvh.load(write(tmp_path, SMALL), convertFromZUp=True)
    hips = b.getJoint("Hips")
    np.testing.assert_allclose(b.getJoint("Spine").offset, [0, 0, -1])  # (x, y, z) -> (x, z, -y)
    np.testing.assert_allclose(b.getJoint("Head").position, [0, 0, -3])
    np.testing.assert_allclose(hips.matrixPoses[1][:, 3], [4, 6, -5])
    assert hips.rotOrder == "szxy"
    # rotation about Z-up's Z axis is a rotation about Y in MakeHuman's Y-up space
    np.testing.assert_allclose(hips.matrixPoses[1][:, :3], rot(90, [0, 1, 0]), atol=1e-6)
    np.testing.assert_allclose(b.getJoint("Spine").matrixPoses[1][:, :3], rot(90, [1, 0, 0]), atol=1e-6)


def test_auto_axis_detects_y_up(tmp_path):
    b = bvh.load(write(tmp_path, SMALL))  # 'Head' found via its capitalised name
    assert b.convertFromZUp is False
    np.testing.assert_allclose(b.getJoint("Head").position, [0, 3, 0])


def test_auto_axis_detects_z_up_and_converts(tmp_path):
    z_up = SMALL.replace("OFFSET 0 0.5 0", "OFFSET 0 0 0.5")  # head points along +Z
    b = bvh.load(write(tmp_path, z_up))
    assert b.convertFromZUp is True
    np.testing.assert_allclose(b.getJoint("Head").position, [0, 0, -3])
    np.testing.assert_allclose(b.getJoint("Head").children[0].position, [0, 0.5, -3])


def test_auto_axis_skips_reference_joints_without_children(tmp_path):
    text = """\
HIERARCHY
ROOT spine01
{
\tOFFSET 0 0 0
\tCHANNELS 3 Zrotation Xrotation Yrotation
\tJOINT lowerleg02.L
\t{
\t\tOFFSET 0 0 2
\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t}
}
MOTION
Frames: 1
Frame Time: 1
0 0 0 0 0 0
"""
    b = bvh.load(write(tmp_path, text))
    # lowerleg02.L is checked first but is a leaf, so spine01 -> lowerleg02.L (+Z) decides: Z-up
    assert b.convertFromZUp is True


def test_auto_axis_with_unknown_joint_names(tmp_path):
    b = bvh.load(write(tmp_path, TRANSLATING))
    assert b.convertFromZUp is False


def test_real_poses_are_z_up(mh):
    b = bvh.load(getpath.getSysDataPath("poses/standing01.bvh"))
    assert b.convertFromZUp is True
    assert b.frameCount == 1 and len(b.joints) == 163
    head, root = b.getJoint("head"), b.getJoint("root")
    assert head.position[1] > root.position[1] + 5  # head above the root once in Y-up


def test_translation_policies(tmp_path):
    path = write(tmp_path, TRANSLATING)
    only = bvh.load(path, convertFromZUp=False)
    np.testing.assert_allclose(only.getJoint("A").matrixPoses[0], np.eye(4)[:3] + [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]])
    # B: no translation (not root) and a lone rotation channel is ignored
    np.testing.assert_allclose(only.getJoint("B").matrixPoses[0], np.eye(4)[:3])

    every = bvh.load(path, convertFromZUp=False, allowTranslation="all")
    np.testing.assert_allclose(every.getJoint("B").matrixPoses[0][:, 3], [0, 7, 0])

    none = bvh.load(path, convertFromZUp=False, allowTranslation="none")
    np.testing.assert_allclose(none.getJoint("A").matrixPoses[0], np.eye(4)[:3])

    zup = bvh.load(path, convertFromZUp=True, allowTranslation="all")
    np.testing.assert_allclose(zup.getJoint("A").matrixPoses[0][:, 3], [1, 3, -2])
    np.testing.assert_allclose(zup.getJoint("B").matrixPoses[0][:, 3], [0, 0, -7])


def test_illegal_axis_option(tmp_path):
    with pytest.raises(RuntimeError, match="illegal option"):
        bvh.load(write(tmp_path, SMALL), convertFromZUp="sideways")


@pytest.mark.parametrize("broken, message", [
    (SMALL.replace("HIERARCHY", "MOTION", 1), "Expected HIERARCHY found MOTION"),
    (SMALL.replace("\t\tEnd Site\n\t\t{\n\t\t\tOFFSET 0 -1 0", "\t\tBogus\n\t\t{\n\t\t\tOFFSET 0 -1 0"),
     "Expected JOINT, End Site or } found Bogus"),
    (SMALL.replace("4 5 6 90 0 0 0 90 0 0 0 0 0 0 0", "4 5 6 90"), "Expected frame data length for joint"),
])
def test_malformed_files(tmp_path, broken, message):
    with pytest.raises(RuntimeError, match=message):
        bvh.load(write(tmp_path, broken), convertFromZUp=False)


def test_channel_count_mismatch_is_rejected(tmp_path):
    broken = SMALL.replace("CHANNELS 3 Zrotation Xrotation Yrotation", "CHANNELS 3 Zrotation Xrotation", 1)
    with pytest.raises(RuntimeError, match="Expected 3 channels"):
        bvh.load(write(tmp_path, broken), convertFromZUp=False)


def test_joint_lookup_by_canonical_name(tmp_path):
    b = bvh.load(write(tmp_path, SMALL.replace("Head", "Upper-Head")), convertFromZUp=False)
    assert b.getJointByCanonicalName("upper head") is b.getJoint("Upper-Head")
    assert b.getJointByCanonicalName("nothing") is None
    assert b.containsJoint("Spine_1") and not b.containsJoint("Neck")


def test_write_and_reload_round_trip(tmp_path):
    original = bvh.load(write(tmp_path, SMALL), convertFromZUp=False)
    out = str(tmp_path / "copy.bvh")
    original.writeToFile(out)
    text = open(out, encoding="utf-8").read()
    assert text.startswith("HIERARCHY\nROOT Hips\n")
    assert "End Site" in text and "Frames: 2" in text and "Frame Time: 0.500000" in text
    copy = bvh.load(out, convertFromZUp=False)
    assert [j.name for j in copy.getJointsBVHOrder()] == [j.name for j in original.getJointsBVHOrder()]
    for name in original.joints:
        a, b = original.getJoint(name), copy.getJoint(name)
        assert a.channels == b.channels
        np.testing.assert_allclose(a.offset, b.offset)
        np.testing.assert_allclose(a.frames, b.frames)
        np.testing.assert_allclose(a.matrixPoses, b.matrixPoses, atol=1e-6)


def test_scale_y_up_rig(tmp_path):
    b = bvh.load(write(tmp_path, SMALL), convertFromZUp=False)
    b.scale(2.0)
    np.testing.assert_allclose(b.getJoint("Head").position, [0, 6, 0])
    np.testing.assert_allclose(b.getJoint("Head").children[0].position, [0, 7, 0])
    np.testing.assert_allclose(b.getJoint("Hips").matrixPoses[1][:, 3], [8, 10, 12])
    np.testing.assert_allclose(b.getJoint("Spine").matrixPoses[1][:, :3], rot(90, [1, 0, 0]), atol=1e-6)


def test_offset_moves_the_whole_rig(tmp_path):
    b = bvh.load(write(tmp_path, SMALL), convertFromZUp=False)
    b.offset(np.array([1.0, 0.0, -2.0], dtype=np.float32))
    np.testing.assert_allclose(b.getJoint("Hips").position, [1, 0, -2])
    np.testing.assert_allclose(b.getJoint("Head").position, [1, 3, -2])
    np.testing.assert_allclose(b.getJoint("Hips").matrixPoses[1][:, 3], [4, 5, 6])


def test_scale_by_one_keeps_a_converted_rig_unchanged(tmp_path):
    b = bvh.load(write(tmp_path, SMALL), convertFromZUp=True)
    before = b.getJoint("Head").position.copy()
    b.scale(1.0)
    np.testing.assert_allclose(b.getJoint("Head").position, before)


def test_animation_track_without_skeleton(tmp_path):
    b = bvh.load(write(tmp_path, SMALL), convertFromZUp=False)
    track = b.createAnimationTrack()
    assert track.name == "small"
    assert track.nFrames == 2 and track.nBones == 4  # end effectors carry no animation
    assert track.frameRate == 2.0
    # per frame, joints in breadth-first order: Hips, Spine, Spine_1, Head
    np.testing.assert_allclose(track.getAtFramePos(1)[1][:, :3], rot(90, [1, 0, 0]), atol=1e-6)


def test_animation_track_from_name_list(tmp_path):
    b = bvh.load(write(tmp_path, SMALL), convertFromZUp=False)
    track = b.createAnimationTrack(["Hips", "Spine_7", "Missing", None, "head"], name="sel")
    assert track.name == "sel" and track.nBones == 5
    frame = track.getAtFramePos(1)
    np.testing.assert_allclose(frame[0][:, 3], [4, 5, 6])
    # "Spine_7" loses the duplicate suffix and maps to Spine
    np.testing.assert_allclose(frame[1][:, :3], rot(90, [1, 0, 0]), atol=1e-6)
    np.testing.assert_allclose(frame[2], np.eye(4)[:3])  # unknown joint: rest pose
    np.testing.assert_allclose(frame[3], np.eye(4)[:3])  # no name: rest pose


def test_animation_track_for_skeleton_with_reference_bones(mh, tmp_path):
    b = bvh.load(write(tmp_path, SMALL), convertFromZUp=False)
    skel = skeleton.Skeleton("refs")
    skel.addBone("Hips", None, "pelvis", "spine-4")
    skel.addBone("pair", "Hips", "spine-4", "spine-3", reference_bones=["Hips", "Spine"])
    skel.addBone("triple", "Hips", "spine-4", "spine-3", reference_bones=["Hips", "Spine", "Head"])
    skel.addBone("single", "pair", "spine-3", "spine-2", reference_bones="Spine")
    skel.addBone("lost", "pair", "spine-3", "spine-2", reference_bones=["Nope"])
    skel.addBone("Other", "pair", "spine-2", "spine-1")
    skel.build()
    track = b.createAnimationTrack(skel)
    assert track.nBones == skel.getBoneCount() == 6
    frame1 = track.getAtFramePos(1)
    poses = {"Hips": rot(90, [0, 0, 1]), "Spine": rot(90, [1, 0, 0]), "Head": np.eye(3)}
    by_name = dict(zip([bone.name for bone in skel.getBones()], frame1))

    def combined(bone_name):
        # quaternions are multiplied in the order the bone stores its references
        result = np.eye(3)
        for ref in skel.getBone(bone_name).reference_bones:
            result = poses[ref] @ result
        return result

    np.testing.assert_allclose(by_name["Hips"][:, :3], poses["Hips"], atol=1e-6)
    np.testing.assert_allclose(by_name["pair"][:, :3], combined("pair"), atol=1e-6)
    np.testing.assert_allclose(by_name["triple"][:, :3], combined("triple"), atol=1e-6)
    np.testing.assert_allclose(by_name["single"][:, :3], poses["Spine"], atol=1e-6)
    np.testing.assert_allclose(by_name["lost"], np.eye(4)[:3])
    np.testing.assert_allclose(by_name["Other"], np.eye(4)[:3])


def test_real_pose_on_default_and_game_skeletons(mh):
    b = bvh.load(getpath.getSysDataPath("poses/benchmark.bvh"))
    base = mh.getBaseSkeleton()
    track = b.createAnimationTrack(base)
    assert track.nBones == base.getBoneCount() and track.nFrames == 1
    idx = base.getBone("upperleg01.R").index
    np.testing.assert_allclose(track.data[idx], b.getJoint("upperleg01.R").matrixPoses[0])

    game = skeleton.load(getpath.getSysDataPath("rigs/game_engine.mhskel"), mh.meshData)
    game_track = b.createAnimationTrack(game)
    assert game_track.nBones == game.getBoneCount()
    for m in game_track.data:  # every combined pose is still a rotation
        np.testing.assert_allclose(m[:, :3] @ m[:, :3].T, np.eye(3), atol=1e-4)


@pytest.mark.parametrize("dummy_joints", [True, False])
def test_export_skeleton_pose_and_reload(mh, tmp_path, dummy_joints):
    base = mh.getBaseSkeleton()
    pose = bvh.load(getpath.getSysDataPath("poses/benchmark.bvh")).createAnimationTrack(base)
    exported = bvh.createFromSkeleton(base, pose, dummyJoints=dummy_joints)
    dummies = [name for name in exported.joints if name.startswith("__")]
    assert bool(dummies) == dummy_joints
    assert len(exported.joints) == base.getBoneCount() + len(dummies)
    assert exported.frameCount == 1

    out = str(tmp_path / "exported.bvh")
    exported.writeToFile(out)
    reloaded = bvh.load(out, convertFromZUp=False)
    again = reloaded.createAnimationTrack(base)
    np.testing.assert_allclose(again.data[:, :, :3], pose.data[:, :, :3], atol=2e-4)
    np.testing.assert_allclose(again.data[0, :, 3], pose.data[0, :, 3], atol=1e-4)  # root translation
    np.testing.assert_allclose(reloaded.getJoint("root").position, base.getBone("root").getRestHeadPos(),
                               atol=1e-4)


def test_export_skeleton_without_animation_is_rest_pose(mh):
    base = mh.getBaseSkeleton()
    exported = bvh.createFromSkeleton(base)
    assert exported.frameCount == 1 and exported.frameTime == 1.0
    root = exported.getJoint("root")
    assert len(root.channels) == 6 and list(root.frames) == [0.0] * 6
    track = exported.createAnimationTrack(base)
    np.testing.assert_allclose(track.data, np.tile(np.eye(4)[:3], (base.getBoneCount(), 1, 1)), atol=1e-6)


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="createSkeleton passes joint positions (arrays) where Skeleton.addBone expects "
                          "joint names; Bone.updateJointPositions then evaluates 'not ndarray' "
                          "(bvh.py:140 -> skeleton.py:426)")
def test_create_skeleton_from_bvh(mh, tmp_path):
    b = bvh.load(write(tmp_path, SMALL), convertFromZUp=False)
    skel = b.createSkeleton()
    assert skel.containsBone("Hips")
