"""Tests for shared/mesh_operations.py: surface area and volume of meshes."""

import numpy as np
import pytest

import mesh_operations
import module3d

# Unit cube, vertex index = x + 2*y + 4*z, faces wound counter-clockwise
# seen from outside (outward normals)
CUBE_COORDS = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
CUBE_QUADS = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
FACE_GROUPS = ["bottom", "top", "front", "back", "left", "right"]


def make_mesh(coords, faces, verts_per_face, groups=None):
    mesh = module3d.Object3D("test", verts_per_face)
    mesh.setCoords(coords)
    names = groups or ["all"] * len(faces)
    fgs = {}
    for name in names:
        if name not in fgs:
            fgs[name] = mesh.createFaceGroup(name)
    mesh.setFaces(faces, None, [fgs[n].idx for n in names])
    return mesh


def cube(size=(1.0, 1.0, 1.0), offset=(0.0, 0.0, 0.0)):
    coords = np.array(CUBE_COORDS, dtype=float) * size + offset
    return make_mesh(coords, CUBE_QUADS, 4, FACE_GROUPS)


def triangulated_cube(size=(1.0, 1.0, 1.0), offset=(0.0, 0.0, 0.0)):
    coords = np.array(CUBE_COORDS, dtype=float) * size + offset
    tris = []
    for a, b, c, d in CUBE_QUADS:
        tris += [(a, b, c), (c, d, a)]
    return make_mesh(coords, tris, 3)


def test_unit_cube_surface_and_volume():
    mesh = cube()
    assert np.isclose(mesh_operations.calculateSurface(mesh), 6.0)
    assert np.isclose(mesh_operations.calculateVolume(mesh), 1.0)


@pytest.mark.parametrize("build", [cube, triangulated_cube], ids=["quads", "triangles"])
def test_box_surface_and_volume(build):
    mesh = build(size=(2.0, 3.0, 4.0))
    assert np.isclose(mesh_operations.calculateSurface(mesh), 2 * (6 + 8 + 12))
    assert np.isclose(mesh_operations.calculateVolume(mesh), 24.0)


@pytest.mark.parametrize("build", [cube, triangulated_cube], ids=["quads", "triangles"])
def test_measures_do_not_depend_on_position(build):
    # The signed tetrahedra cancel out for a closed mesh wherever the origin is
    mesh = build(size=(2.0, 3.0, 4.0), offset=(10.0, -7.0, 3.5))
    assert np.isclose(mesh_operations.calculateVolume(mesh), 24.0, rtol=1e-5)
    assert np.isclose(mesh_operations.calculateSurface(mesh), 52.0, rtol=1e-5)


def test_volume_does_not_depend_on_winding():
    mesh = cube(size=(2.0, 2.0, 2.0))
    flipped = make_mesh(mesh.coord, mesh.fvert[:, ::-1], 4)
    assert np.isclose(mesh_operations.calculateVolume(flipped), 8.0)


def test_surface_of_a_face_mask():
    mesh = cube(size=(2.0, 3.0, 4.0))
    mask = np.zeros(6, dtype=bool)
    mask[[0, 1]] = True  # bottom and top, 2 x 3 each
    assert np.isclose(mesh_operations.calculateSurface(mesh, faceMask=mask), 12.0)


def test_surface_of_vertex_groups():
    mesh = cube(size=(2.0, 3.0, 4.0))
    assert np.isclose(mesh_operations.calculateSurface(mesh, vertGroups=["left"]), 12.0)
    assert np.isclose(mesh_operations.calculateSurface(mesh, vertGroups=["left", "front"]), 12.0 + 8.0)


def test_volume_of_a_subset_matches_its_tetrahedra():
    # The top face alone: tetrahedra from the origin to a 2x3 face at z=4
    mesh = cube(size=(2.0, 3.0, 4.0))
    assert np.isclose(mesh_operations.calculateVolume(mesh, vertGroups=["top"]), 2 * 3 * 4 / 3.0)
    mask = np.array([False, True, False, False, False, False])
    assert np.isclose(mesh_operations.calculateVolume(mesh, faceMask=mask), 8.0)


def test_lines_are_rejected():
    lines = make_mesh([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [(0, 1), (1, 2)], 2)
    with pytest.raises(RuntimeError):
        mesh_operations.calculateSurface(lines)
    with pytest.raises(RuntimeError):
        mesh_operations.calculateVolume(lines)


def test_heron_formula_on_known_triangles():
    tris = np.array([
        [[0, 0, 0], [3, 0, 0], [0, 4, 0]],  # 3-4-5 right triangle, area 6
        [[0, 0, 0], [2, 0, 0], [1, np.sqrt(3), 0]],  # equilateral, side 2
    ], dtype=np.float32)
    lengths = mesh_operations._sideLengthsFromTris(tris)
    assert np.allclose(lengths, [[3, 5, 4], [2, 2, 2]])
    assert np.isclose(mesh_operations._surfaceOfTris(lengths), 6 + np.sqrt(3), rtol=1e-6)


def test_find_vert_index_needs_all_three_coordinates_to_match():
    coords = [(1, 2, 3), (1, 2, 4), (1, 2, 3), (3, 2, 1)]
    mesh = make_mesh(coords, [(0, 1, 3)], 3)
    assert sorted(mesh_operations.findVertIndex(mesh, np.array([1, 2, 3]))) == [0, 2]
    assert mesh_operations.findVertIndex(mesh, np.array([1, 2, 5])) == []
    assert mesh_operations.findVertIndex(mesh, np.array([3, 2, 1])) == [3]
