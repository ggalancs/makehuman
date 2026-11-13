"""shared/skeleton_drawing.py: meshes that visualize a skeleton (one shape per
bone) and joint positions (one diamond per joint)."""

import os
import types

import numpy as np
import pytest

import geometry3d
import skeleton_drawing as sd

MH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "makehuman")


def translate(x, y, z):
    m = np.identity(4)
    m[:3, 3] = (x, y, z)
    return m


# 90 degrees about z: (x, y, z) -> (-y, x, z)
ROT_Z90 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def bone(name, mat, length):
    return types.SimpleNamespace(name=name, matPoseGlobal=mat, getLength=lambda: length)


def skeleton(*bones):
    return types.SimpleNamespace(getBones=lambda: list(bones))


# Bone "up": head at (1, 2, 3), pointing +y, length 2.
# Bone "side": head at the origin, rotated so it points -x, length 1.
UP = bone("up", translate(1, 2, 3), 2.0)
SIDE = bone("side", ROT_Z90, 1.0)

# Prism vertices of UP: shape vector * 2 + (1, 2, 3)
UP_PRISM = [(1, 2, 3), (1.28, 2.5, 3), (1, 2.5, 3.28), (0.72, 2.5, 3), (1, 2.5, 2.72), (1, 4, 3)]
# Prism vertices of SIDE: (x, y, z) -> (-y, x, z)
SIDE_PRISM = [(0, 0, 0), (-0.25, 0.14, 0), (-0.25, 0, 0.14), (-0.25, -0.14, 0), (-0.25, 0, -0.14), (-1, 0, 0)]


@pytest.fixture
def axis_cache(monkeypatch):
    """Start without the cached axis mesh; it loads data/3dobjs/axis.obj."""
    monkeypatch.chdir(MH_DIR)
    monkeypatch.setattr(sd, "_axismesh_", None)


def faces_point_outwards(mesh, first_vert, nverts, first_face, nfaces):
    """Every face normal of one (convex) bone shape points away from its centre."""
    centre = mesh.coord[first_vert:first_vert + nverts].mean(axis=0)
    for f in range(first_face, first_face + nfaces):
        centroid = mesh.coord[mesh.fvert[f]].mean(axis=0)
        if np.dot(mesh.fnorm[f], centroid - centre) <= 0:
            return False
    return True


# --- one bone -------------------------------------------------------------------------

def test_prism_of_a_translated_bone():
    points, faces, colors = sd._shapeFromBone(UP, "Prism")
    assert np.allclose(points, UP_PRISM)
    assert faces.dtype == np.uint16 and faces.tolist() == [list(f) for f in sd.SHAPE_FACES["Prism"]]
    assert colors.shape == (6, 4) and (colors == 255).all()


def test_prism_of_a_rotated_bone():
    points, _, _ = sd._shapeFromBone(SIDE, "Prism")
    assert np.allclose(points, SIDE_PRISM, atol=1e-6)


@pytest.mark.parametrize("shape, half", [("Box", 0.1), ("Line", 0.03)])
def test_box_and_line_are_square_beams(shape, half):
    points, faces, _ = sd._shapeFromBone(UP, shape)
    # x and z within +-half*length of the axis, y from the head to the tail
    expected = [(1 + sx * half * 2, 2 + y * 2, 3 + sz * half * 2)
                for sx in (-1, 1) for y in (0, 1) for sz in (-1, 1)]
    assert np.allclose(points, expected)
    assert faces.shape == (6, 4)


# --- whole skeleton -------------------------------------------------------------------

def test_mesh_from_skeleton_prism():
    mesh = sd.meshFromSkeleton(skeleton(UP, SIDE), "Prism")
    assert mesh.name == "SkeletonMesh" and mesh.boneShape == "Prism"
    assert mesh.getVertexCount() == 12 and mesh.getFaceCount() == 16
    assert np.allclose(mesh.coord, UP_PRISM + SIDE_PRISM, atol=1e-6)
    prism = np.array(sd.SHAPE_FACES["Prism"])
    assert (mesh.fvert[:8] == prism).all() and (mesh.fvert[8:] == prism + 6).all()
    assert [g.name for g in mesh.faceGroups] == ["up", "side"]
    assert mesh.group.tolist() == [0] * 8 + [1] * 8
    assert (mesh.color == 255).all()
    assert mesh.priority == 30 and mesh.cameraMode == 0
    # Normals were computed, and the shapes are wound outwards
    assert faces_point_outwards(mesh, 0, 6, 0, 8) and faces_point_outwards(mesh, 6, 6, 8, 8)
    assert len(mesh.r_coord) > 0  # index buffer built


@pytest.mark.parametrize("shape", ["Box", "Line"])
def test_mesh_from_skeleton_beams(shape):
    mesh = sd.meshFromSkeleton(skeleton(UP, SIDE), shape)
    assert mesh.getVertexCount() == 16 and mesh.getFaceCount() == 12
    assert (mesh.fvert[6:] == np.array(sd.SHAPE_FACES[shape]) + 8).all()
    assert faces_point_outwards(mesh, 0, 8, 0, 6) and faces_point_outwards(mesh, 8, 8, 6, 6)


@pytest.mark.xfail(strict=True, raises=KeyError, reason=(
    "skeleton_drawing.py:206 looks up SHAPE_FACES['Cube'], but SHAPE_FACES only has "
    "Prism, Box and Line although SHAPE_VECTORS has Cube and the docstring "
    "(skeleton_drawing.py:180) lists it as allowed"))
def test_mesh_from_skeleton_cube():
    mesh = sd.meshFromSkeleton(skeleton(UP), "Cube")
    assert mesh.getVertexCount() == 8 and mesh.getFaceCount() == 6


@pytest.mark.parametrize("length, size", [(5.0, 1.0), (0.5, 0.5), (0.01, 0.1)])
def test_axis_shape_is_scaled_by_the_clamped_bone_length(axis_cache, length, size):
    reference = geometry3d.AxisMesh(scale=0.5)
    points, faces, colors = sd._shapeFromBone(bone("b", translate(1, 2, 3), length), "axis")
    assert np.allclose(points, reference.coord * size + (1, 2, 3), atol=1e-6)
    assert (faces == reference.fvert).all() and (colors == reference.color).all()
    # The colours mark the three axes
    assert {tuple(c) for c in colors} == {(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)}


def test_mesh_from_skeleton_axis(axis_cache):
    mesh = sd.meshFromSkeleton(skeleton(UP, SIDE), "axis")
    n = geometry3d.AxisMesh().getVertexCount()
    assert mesh.getVertexCount() == 2 * n and mesh.boneShape == "axis"
    assert (mesh.color[:n] == mesh.color[n:]).all() and (mesh.color[:, :3] != 255).any()
    # SIDE: unit length, so its axes are the reference ones turned (x, y, z) -> (-y, x, z)
    ref = geometry3d.AxisMesh(scale=0.5).coord
    assert np.allclose(mesh.coord[n:], np.stack([-ref[:, 1], ref[:, 0], ref[:, 2]], axis=1), atol=1e-6)


# --- vertex to bone mapping -----------------------------------------------------------

def test_vertex_bone_mapping_of_a_prism_mesh():
    skel = skeleton(UP, SIDE)
    weights = sd.getVertBoneMapping(skel, sd.meshFromSkeleton(skel, "Prism"))
    assert weights.vertexCount == 12
    verts, w = weights.data["up"]
    assert verts.tolist() == list(range(6)) and np.allclose(w, 1)
    verts, w = weights.data["side"]
    assert verts.tolist() == list(range(6, 12)) and np.allclose(w, 1)


def test_vertex_bone_mapping_of_an_axis_mesh(axis_cache):
    skel = skeleton(UP, SIDE)
    mesh = sd.meshFromSkeleton(skel, "axis")
    sd._axismesh_ = None  # getVertBoneMapping loads it again on its own
    weights = sd.getVertBoneMapping(skel, mesh)
    assert weights.data["side"][0].tolist() == list(range(130, 260))
    assert weights.vertexCount == mesh.getVertexCount() == 260


def test_vertex_bone_mapping_needs_a_skeleton_mesh():
    with pytest.raises(RuntimeError, match="not a skeleton mesh"):
        sd.getVertBoneMapping(skeleton(UP), types.SimpleNamespace())


# --- joints ---------------------------------------------------------------------------

def test_mesh_from_joints_with_names():
    mesh = sd.meshFromJoints(np.array([(0, 0, 0), (1, 2, 3)], dtype=float), ["joint-a", "joint-b"])
    assert mesh.name == "jointsMesh" and mesh.getVertexCount() == 12 and mesh.getFaceCount() == 16
    # Diamond of half-size 0.14 around each joint
    assert np.allclose(mesh.coord[6:], [(1, 1.86, 3), (1.14, 2, 3), (1, 2, 3.14),
                                        (0.86, 2, 3), (1, 2, 2.86), (1, 2.14, 3)])
    assert (mesh.fvert[8:] == np.array(sd.SHAPE_FACES["Prism"]) + 6).all()
    assert [g.name for g in mesh.faceGroups] == ["joint-a", "joint-b"]
    assert mesh.group.tolist() == [0] * 8 + [1] * 8
    assert mesh.priority == 30 and mesh.cameraMode == 0
    assert faces_point_outwards(mesh, 0, 6, 0, 8)


def test_mesh_from_joints_without_names_has_no_groups():
    mesh = sd.meshFromJoints([(0, 0, 0)])
    assert mesh.faceGroupCount == 0 and mesh.group.tolist() == [0] * 8
    assert np.allclose(mesh.coord[[0, 5]], [(0, -0.14, 0), (0, 0.14, 0)])
