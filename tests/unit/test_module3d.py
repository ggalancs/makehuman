"""Tests for core/module3d.py: Object3D, the mesh container behind every
MakeHuman model, and FaceGroup."""

import gc
import types

import numpy as np
import pytest

import module3d
from core import G

# Tangent computation divides by zero on axis-aligned UVs (see the xfail tests
# below); keep the RuntimeWarnings it raises from burying everything else.
pytestmark = [pytest.mark.filterwarnings("ignore::RuntimeWarning:module3d"),
              pytest.mark.filterwarnings("ignore:invalid value encountered in reduce:RuntimeWarning")]

# Unit cube, vertex index = x + 2*y + 4*z, faces wound counter-clockwise seen
# from outside, so face normals point outwards
CUBE_COORDS = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
CUBE_QUADS = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
CUBE_GROUPS = ["bottom", "top", "front", "back", "left", "right"]
OUTWARD = np.array([(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)], dtype=float)
UVS = [(0, 0), (1, 0), (1, 1), (0, 1)]


def make_cube(name="cube"):
    mesh = module3d.Object3D(name, 4)
    mesh.setCoords(CUBE_COORDS)
    mesh.setUVs(UVS)
    groups = [mesh.createFaceGroup(g) for g in CUBE_GROUPS]
    mesh.setFaces(CUBE_QUADS, [(0, 1, 2, 3)] * 6, [g.idx for g in groups])
    mesh.updateIndexBuffer()
    return mesh


class Holder(object):
    """Stands in for the gui3d.Object a mesh belongs to (weakly referenced)."""

    def __init__(self, diffuse=(1.0, 1.0, 1.0), opacity=1.0, transparent=False):
        self.material = types.SimpleNamespace(
            diffuseColor=types.SimpleNamespace(values=list(diffuse)),
            opacity=opacity, transparent=transparent)


# --- construction and topology ---------------------------------------------

def test_new_object_is_empty():
    mesh = module3d.Object3D("empty")
    assert mesh.vertsPerPrimitive == 4
    assert mesh.getVertexCount() == 0 and mesh.faceGroupCount == 0
    assert mesh.visibility is True and mesh.pickable is False
    assert mesh.object is None
    assert str(mesh) == "object3D Mesh named: empty, nverts: 0, nfaces: 0"


def test_cube_topology():
    mesh = make_cube()
    assert mesh.getVertexCount() == 8 and mesh.getFaceCount() == 6
    assert mesh.getUVCount() == 4 and mesh.hasUVs()
    assert mesh.fvert.dtype == np.uint32 and mesh.fvert.shape == (6, 4)
    # Every corner of a cube touches three faces
    assert mesh.nfaces.tolist() == [3] * 8
    for v in range(8):
        faces = mesh.vface[v, :mesh.nfaces[v]]
        assert sorted(faces) == sorted(f for f, quad in enumerate(CUBE_QUADS) if v in quad)
    # vface is trimmed to the minimum of four slots
    assert mesh.MAX_FACES == 4 and mesh.vface.shape == (8, 4)
    assert mesh.vertsPerFaceForExport == 4


def test_quads_with_a_repeated_vertex_are_exported_as_triangles():
    mesh = module3d.Object3D("tris", 4)
    mesh.setCoords([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.setFaces([(0, 1, 2, 0)])
    assert mesh.vertsPerFaceForExport == 3
    assert not mesh.hasUVs()


def test_too_many_faces_on_one_vertex_is_rejected():
    # A fan of 9 triangles around vertex 0, more than MAX_FACES (8) allows
    mesh = module3d.Object3D("fan", 3)
    angles = np.linspace(0, 2 * np.pi, 10)[:-1]
    mesh.setCoords([(0, 0, 0)] + [(np.cos(a), np.sin(a), 0) for a in angles])
    faces = [(0, 1 + i, 1 + (i + 1) % 9) for i in range(9)]
    with pytest.raises(RuntimeError, match="Incompatible mesh topology"):
        mesh.setFaces(faces)


def test_high_valence_mesh_with_raised_max_faces():
    mesh = module3d.Object3D("fan", 3)
    mesh.MAX_FACES = 12
    angles = np.linspace(0, 2 * np.pi, 10)[:-1]
    mesh.setCoords([(0, 0, 0)] + [(np.cos(a), np.sin(a), 0) for a in angles])
    mesh.setFaces([(0, 1 + i, 1 + (i + 1) % 9) for i in range(9)])
    assert mesh.nfaces[0] == 9
    # Resized to the larger of the face count and the pole count (9 neighbours)
    assert mesh.MAX_FACES == 9 and mesh.vface.shape[1] == 9


def test_set_faces_accepts_an_empty_face_list():
    mesh = module3d.Object3D("empty", 4)
    mesh.setCoords([(0, 0, 0)])
    mesh.setFaces([])
    assert mesh.getFaceCount() == 0


# --- face groups -------------------------------------------------------------

def test_face_groups():
    mesh = make_cube()
    top = mesh.getFaceGroup("top")
    assert str(top) == "facegroup top"
    assert top.idx == 1 and top.parent is mesh and top.object is mesh
    assert mesh.getFaceGroup("nope") is None
    assert sorted(mesh.getFaceGroups()) == sorted(CUBE_GROUPS)
    assert [fg.name for fg in mesh.faceGroups] == CUBE_GROUPS
    assert mesh.faceGroupCount == 6
    top.setColor([255, 0, 0, 255])
    assert top.color.dtype == np.uint8 and top.color.tolist() == [255, 0, 0, 255]
    top.object = None
    assert top.object is None and top.parent is None


def test_face_group_reference_to_its_mesh_is_weak():
    mesh = make_cube()
    fg = mesh.getFaceGroup("top")
    del mesh
    gc.collect()
    assert fg.object is None


def test_group_masks_and_lookups():
    mesh = make_cube()
    assert mesh.getGroupMaskForGroups(["top", "left"]).tolist() == [False, True, False, False, True, False]
    assert mesh.getFaceMaskForGroups(["top"]).tolist() == [False, True, False, False, False, False]
    assert mesh.getFacesForGroups(["top", "bottom"]).tolist() == [0, 1]
    assert mesh.getVerticesForGroups(["top"]).tolist() == [4, 5, 6, 7]
    vmask, fmask = mesh.getVertexAndFaceMasksForGroups(["front"])
    assert np.flatnonzero(vmask).tolist() == [0, 1, 4, 5]
    assert np.flatnonzero(fmask).tolist() == [2]
    assert np.array_equal(mesh.getVertexMaskForGroups(["front"]), vmask)


def test_faces_for_vertices():
    mesh = make_cube()
    assert mesh.getFacesForVertices([0]).tolist() == [0, 2, 4]
    assert mesh.getFacesForVertices([0, 7]).tolist() == list(range(6))
    assert mesh.getFaceMaskForVertices([7]).tolist() == [False, True, False, True, False, True]


def test_face_and_vertex_masks():
    mesh = make_cube()
    mask = np.ones(6, dtype=bool)
    mask[1] = False  # hide the top
    mesh.changeFaceMask(mask)
    assert mesh.getFaceCount(excludeMaskedFaces=True) == 5
    assert mesh.getVertexCount(excludeMaskedVerts=True) == 8  # top corners shared with sides
    mesh.changeFaceMask(False, [0, 2, 3, 4, 5])
    assert mesh.getFaceMask().tolist() == [False] * 6
    assert mesh.getVertexCount(excludeMaskedVerts=True) == 0
    mesh.changeFaceMask(True, [1])
    assert mesh.getFaceMask([0, 1]).tolist() == [False, True]
    assert mesh.getVertexMaskForFaceMask(mesh.getFaceMask()).tolist() == [False] * 4 + [True] * 4
    assert mesh.getVerticesForFaceMask(mesh.getFaceMask()).tolist() == [4, 5, 7, 6]


# --- accessors and update flags ---------------------------------------------

def test_accessors():
    mesh = make_cube()
    assert np.array_equal(mesh.getCoords(), np.array(CUBE_COORDS, dtype=np.float32))
    assert mesh.getCoords([7]).tolist() == [[1, 1, 1]]
    assert mesh.getNormals().shape == (8, 3) and mesh.getNormals([0]).shape == (1, 3)
    assert mesh.getUVs([2]).tolist() == [[1, 1]] and mesh.getUVs().shape == (4, 2)
    assert mesh.getFaceVerts([1]).tolist() == [[4, 5, 7, 6]]
    assert mesh.getFaceVerts().shape == (6, 4)
    assert mesh.getFaceUVs([0]).tolist() == [[0, 1, 2, 3]] and mesh.getFaceUVs().shape == (6, 4)


def test_change_coords_marks_only_what_changed():
    mesh = make_cube()
    mesh.update()  # sync everything: flags back to False
    assert mesh.ucoor is False
    mesh.changeCoords([[5, 5, 5]], [3])
    assert mesh.coord[3].tolist() == [5, 5, 5]
    assert mesh.ucoor.dtype == bool and np.flatnonzero(mesh.ucoor).tolist() == [3]
    mesh.changeCoords([[6, 6, 6]], [4])
    assert np.flatnonzero(mesh.ucoor).tolist() == [3, 4]
    mesh.sync_coord()
    assert mesh.ucoor is False
    # Every unwelded copy of vertex 3 got the new position
    copies = np.flatnonzero(mesh.vmap == 3)
    assert len(copies) > 0 and np.all(mesh.r_coord[copies] == 5)
    mesh.changeCoords(np.zeros((8, 3)))
    assert mesh.ucoor is True and not mesh.coord.any()


def test_mark_coords_with_indices_for_normals_and_colors():
    mesh = make_cube()
    mesh.update()
    mesh.markCoords((np.array([1, 2]),), norm=True, colr=True)  # np.where-style tuple
    assert np.flatnonzero(mesh.unorm).tolist() == [1, 2]
    assert mesh.utang is mesh.unorm
    assert np.flatnonzero(mesh.ucolr).tolist() == [1, 2]
    mesh.markCoords([5], norm=True, colr=True)
    assert np.flatnonzero(mesh.unorm).tolist() == [1, 2, 5]
    mesh.vnorm[:] = (0, 0, 1)
    mesh.color[:] = (1, 2, 3, 4)
    mesh.vtang[:] = (1, 0, 0, 1)
    mesh.sync_norms()
    mesh.sync_color()
    mesh.sync_tangents()
    marked = np.isin(mesh.vmap, [1, 2, 5])
    assert np.all(mesh.r_vnorm[marked] == (0, 0, 1))
    assert not np.any(np.all(mesh.r_vnorm[~marked] == (0, 0, 1), axis=1))  # not synced
    assert np.all(mesh.r_color[marked] == (1, 2, 3, 4))
    assert np.all(mesh.r_vtang[marked] == (1, 0, 0, 1))
    assert mesh.unorm is False and mesh.ucolr is False and mesh.utang is False
    # Whole-mesh marks override partial ones
    mesh.markCoords([0], coor=True)
    mesh.markCoords(None, coor=True, norm=True, colr=True)
    assert mesh.ucoor is True and mesh.unorm is True and mesh.ucolr is True


def test_mark_uvs():
    mesh = make_cube()
    mesh.update()
    assert mesh.utexc is False
    mesh.texco[2] = (0.5, 0.25)
    mesh.markUVs((np.array([2]),))
    assert np.flatnonzero(mesh.utexc).tolist() == [2]
    mesh.markUVs([3])
    assert np.flatnonzero(mesh.utexc).tolist() == [2, 3]
    mesh.sync_texco()
    assert mesh.utexc is False
    assert np.all(mesh.r_texco[mesh.tmap == 2] == (0.5, 0.25))
    mesh.markUVs()
    assert mesh.utexc is True


def test_syncs_do_nothing_without_flags_or_buffers():
    mesh = make_cube()
    mesh.update()
    before = mesh.r_coord.copy()
    mesh.coord[:] = 9  # changed without marking: not synced
    mesh.sync_all()
    assert np.array_equal(mesh.r_coord, before)
    bare = module3d.Object3D("bare")
    bare.setCoords(CUBE_COORDS)  # no faces, so no unwelded buffers yet
    for sync in (bare.sync_coord, bare.sync_norms, bare.sync_color):
        sync()
    assert bare.ucoor is True and bare.unorm is True and bare.ucolr is True
    bare.has_uv = True
    bare.utexc = bare.utang = True
    bare.sync_texco()
    bare.sync_tangents()
    assert bare.utexc is True and bare.utang is True
    nouv = module3d.Object3D("nouv")
    nouv.setCoords(CUBE_COORDS)
    nouv.setFaces(CUBE_QUADS)
    nouv.updateIndexBuffer()
    nouv.utang = nouv.utexc = True
    nouv.sync_tangents()
    nouv.sync_texco()
    assert nouv.utang is True and nouv.utexc is True  # nothing to sync without UVs


def test_index_buffer_unwelds_per_uv():
    mesh = make_cube()
    # 8 positions, each used with the UV slot of its place in each quad
    assert len(mesh.vmap) == len(mesh.tmap) == len(mesh.r_coord)
    assert np.array_equal(mesh.r_coord, mesh.coord[mesh.vmap])
    assert np.array_equal(mesh.r_texco, mesh.texco[mesh.tmap])
    # Each face corner references an unwelded vertex with the right position and UV
    for f in range(6):
        for c in range(4):
            r = mesh.r_faces[f, c]
            assert mesh.vmap[r] == mesh.fvert[f, c] and mesh.tmap[r] == mesh.fuvs[f, c]
    # inverse_vmap is cached and inverts vmap
    inv = mesh.inverse_vmap
    assert inv is mesh.inverse_vmap
    for original, unwelded in inv.items():
        assert all(mesh.vmap[u] == original for u in unwelded)
    assert sum(len(u) for u in inv.values()) == len(mesh.vmap)


def test_index_buffer_groups_visible_faces():
    mesh = make_cube()
    mask = np.ones(6, dtype=bool)
    mask[[0, 3]] = False
    mesh.changeFaceMask(mask)
    mesh.updateIndexBufferFaces()
    assert len(mesh.index) == 4
    # One face per group, hidden groups get no faces
    assert mesh.grpix[:, 1].tolist() == [0, 1, 1, 0, 1, 1]
    start = mesh.grpix[4, 0]
    assert np.array_equal(mesh.index[start], mesh.r_faces[4])
    mesh.changeFaceMask(np.zeros(6, dtype=bool))
    mesh.updateIndexBufferFaces()
    assert len(mesh.index) == 0 and mesh.grpix.shape == (0, 2)


# --- colors, flags, material ---------------------------------------------------

def test_set_color():
    mesh = make_cube()
    mesh.setColor([10, 20, 30])  # RGB gets an opaque alpha
    assert np.all(mesh.color == (10, 20, 30, 255))
    assert np.all(mesh.r_color == (10, 20, 30, 255))
    per_vertex = np.arange(32, dtype=np.uint8).reshape(8, 4)
    mesh.setColor(per_vertex)
    assert np.array_equal(mesh.color, per_vertex)
    assert np.array_equal(mesh.r_color, per_vertex[mesh.vmap])


def test_r_color_diff_multiplies_vertex_colors_by_the_material():
    mesh = make_cube()
    holder = Holder(diffuse=(0.5, 1.0, 0.0), opacity=0.5)
    mesh.object = holder
    mesh.setColor([200, 100, 50, 255])
    first = mesh.r_color_diff
    assert np.all(first == (100, 100, 0, 127))
    assert mesh.r_color_diff is first  # cached while the material is unchanged
    holder.material.diffuseColor.values = [1.0, 1.0, 1.0]
    holder.material.opacity = 1.0
    assert np.all(mesh.r_color_diff == (200, 100, 50, 255))


def test_material_and_transparency():
    mesh = make_cube()
    holder = Holder()
    mesh.object = holder
    assert mesh.material is holder.material
    mesh.setTransparentPrimitives(2)
    assert mesh.transparentPrimitives == 2
    holder.material.transparent = True
    assert mesh.transparentPrimitives == 6  # all faces
    # A mesh without object of its own uses its parent's material
    child = make_cube("child")
    child.parent = mesh
    assert child.material is holder.material


def test_object_reference_is_weak():
    mesh = make_cube()
    holder = Holder()
    mesh.object = holder
    assert mesh.object is holder
    del holder
    gc.collect()
    assert mesh.object is None
    mesh.object = None
    assert mesh.object is None


def test_flags_and_properties():
    mesh = make_cube()
    mesh.setVisibility(False)
    assert mesh.visibility is False
    mesh.visibility = True
    assert mesh.visibility is True
    mesh.setPickable(True)
    assert mesh.pickable is True
    mesh.priority = 30
    assert mesh.getPriority() == 30
    mesh.setCameraProjection(1)
    assert mesh.cameraMode == 1


def test_get_camera_uses_the_camera_mode(monkeypatch):
    cameras = [object(), object()]
    monkeypatch.setattr(G, "cameras", cameras)
    mesh = make_cube()
    assert mesh.getCamera() is cameras[0]
    mesh.setCameraProjection(1)
    assert mesh.getCamera() is cameras[1]


# --- geometry: normals, tangents, bounding box --------------------------------

def test_face_and_vertex_normals_of_a_cube_point_outwards():
    mesh = make_cube()
    mesh.calcNormals()
    unit_fnorm = mesh.fnorm / np.linalg.norm(mesh.fnorm, axis=1)[:, None]
    assert np.allclose(unit_fnorm, OUTWARD)
    # Vertex normals are the normalised sum of the three adjacent face normals
    expected = (np.array(CUBE_COORDS) - 0.5) * 2 / np.sqrt(3)
    assert np.allclose(mesh.vnorm, expected)
    assert np.allclose(np.linalg.norm(mesh.vnorm, axis=1), 1.0)


def test_normals_follow_a_deformation_of_some_vertices():
    mesh = make_cube()
    mesh.calcNormals()
    # Push the top face up into a roof: only vertices 4..7 move
    mesh.changeCoords([[0, 0, 2], [1, 0, 1], [0, 1, 2], [1, 1, 1]], [4, 5, 6, 7])
    faces = mesh.getFacesForVertices([4, 5, 6, 7])
    mesh.calcNormals(1, 1, np.array([4, 5, 6, 7]), faces)
    top = mesh.fnorm[1] / np.linalg.norm(mesh.fnorm[1])
    assert np.allclose(top, np.array([1, 0, 1]) / np.sqrt(2))
    # The bottom face did not change
    assert np.allclose(mesh.fnorm[0] / np.linalg.norm(mesh.fnorm[0]), [0, 0, -1])


def test_calc_normals_flags():
    mesh = make_cube()
    mesh.calcNormals(recalcVertexNormals=0, recalcFaceNormals=1)
    assert mesh.fnorm.any() and not mesh.vnorm.any()
    other = make_cube()
    other.calcFaceNormals()
    other.calcNormals(recalcVertexNormals=1, recalcFaceNormals=0)
    assert other.vnorm.any()


ROTATED_UVS = [(0.1, 0.2), (0.9, 0.3), (0.8, 0.9), (0.2, 0.7)]  # no UV edge axis aligned


def test_vertex_tangents_carry_a_handedness_sign_and_are_synced():
    # Only what holds regardless of the tangent bugs below: their direction
    # and length are checked by the xfail tests
    mesh = make_cube()
    mesh.setUVs(ROTATED_UVS)
    mesh.calcNormals()
    assert set(np.unique(mesh.vtang[:, 3])) <= {-1.0, 1.0}
    # Tangents are only marked for syncing, then synced with the rest
    assert mesh.utang is True
    mesh.update()
    assert mesh.utang is False
    assert np.array_equal(mesh.r_vtang, mesh.vtang[mesh.vmap], equal_nan=True)


def test_vertex_tangents_for_a_subset_only_touch_that_subset():
    mesh = make_cube()
    mesh.setUVs(ROTATED_UVS)
    mesh.calcNormals()
    mesh.update()
    mesh.vtang[:] = 0
    mesh.calcVertexTangents(np.array([0, 3]))
    assert np.flatnonzero(mesh.vtang.any(axis=1)).tolist() == [0, 3]
    assert np.flatnonzero(mesh.utang).tolist() == [0, 3]


def test_tangents_of_a_subset_match_the_full_computation():
    # Mirrored UVs give left-handed tangents (w = -1). Recomputing an index array of
    # vertices, as a slider edit does, must give the same tangents and signs.
    mesh = make_cube()
    uvs = np.array(ROTATED_UVS, dtype=np.float32)
    uvs[:, 0] = 1 - uvs[:, 0]
    mesh.setUVs(uvs)
    mesh.calcNormals()
    full = mesh.vtang.copy()
    assert (full[:, 3] == -1).any() and np.isfinite(full).all()
    mesh.vtang[:] = 0
    mesh.calcVertexTangents(np.arange(mesh.getVertexCount()))
    np.testing.assert_allclose(mesh.vtang, full, atol=1e-6)


def test_vertex_tangents_are_perpendicular_to_the_normals():
    mesh = make_cube()
    mesh.setUVs(ROTATED_UVS)
    mesh.calcNormals()
    assert np.allclose(np.sum(mesh.vtang[:, :3] * mesh.vnorm, axis=1), 0.0, atol=1e-5)


def test_no_tangents_without_uvs():
    mesh = module3d.Object3D("nouv")
    mesh.setCoords(CUBE_COORDS)
    mesh.setFaces(CUBE_QUADS)
    mesh.calcNormals()
    assert not mesh.vtang.any()


def flat_quad():
    quad = module3d.Object3D("quad", 4)
    quad.setCoords([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
    quad.setUVs(UVS)
    quad.setFaces([(0, 1, 2, 3)], [(0, 1, 2, 3)], [0])
    return quad


def test_tangents_of_a_textured_quad_are_finite():
    quad = flat_quad()
    quad.calcNormals()
    assert np.all(np.isfinite(quad.vtang))


def test_tangent_follows_the_u_direction():
    # u = x and v = 1 + y: the tangent (dP/du) is +x. These UVs make the
    # line-411 bug harmless (w3.v - w1.v == w1.v), isolating this one
    tri = module3d.Object3D("tri", 3)
    tri.setCoords([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    tri.setUVs([(0, 1), (1, 1), (0, 2)])
    tri.setFaces([(0, 1, 2)], [(0, 1, 2)], [0])
    tri.calcNormals()
    assert np.allclose(tri.vtang, [(1, 0, 0, 1)] * 3, atol=1e-5)


def test_calculate_tangents_flag_disables_tangents():
    mesh = make_cube()
    mesh.setUVs(ROTATED_UVS)
    mesh.calculateTangents = False
    mesh.calcNormals()
    assert not mesh.vtang.any()


def test_bounding_box_and_center():
    mesh = make_cube()
    mesh.changeCoords(np.array(CUBE_COORDS, dtype=float) * (2, 4, 6) + (1, 1, 1))
    assert mesh.calcBBox().tolist() == [[1, 1, 1], [3, 5, 7]]
    assert mesh.getCenter().tolist() == [-2, -3, -4]  # the offset that centres it
    # A stray vertex outside every face only counts when asked for
    coords = np.vstack([mesh.coord, [[100, 100, 100]]])
    mesh.setCoords(coords)
    mesh.setFaces(CUBE_QUADS, [(0, 1, 2, 3)] * 6, list(range(6)))
    assert mesh.calcBBox().tolist() == [[1, 1, 1], [3, 5, 7]]
    assert mesh.calcBBox(onlyVisible=False).tolist() == [[1, 1, 1], [100, 100, 100]]
    assert mesh.calcBBox([0, 8], onlyVisible=False).tolist() == [[1, 1, 1], [100, 100, 100]]
    top = mesh.getFaceMaskForGroups(["top"]) if mesh.faceGroupCount else np.arange(6) == 1
    assert mesh.calcBBox(fixedFaceMask=top).tolist() == [[1, 1, 7], [3, 5, 7]]
    mesh.changeFaceMask(np.zeros(6, dtype=bool))
    assert mesh.calcBBox().tolist() == [[0, 0, 0], [0, 0, 0]]


def test_dot_v3():
    a = np.array([[1, 2, 3], [0, 1, 0]], dtype=float)
    b = np.array([[4, 5, 6], [1, 0, 0]], dtype=float)
    assert module3d.dot_v3(a, b).tolist() == [32, 0]


# --- clone, filtering and vertex maps ----------------------------------------

def test_clone_copies_geometry_and_scales_it():
    mesh = make_cube()
    holder = Holder()
    mesh.object = holder
    mesh.getFaceGroup("top").setColor([1, 2, 3, 255])
    mesh.setPickable(True)
    mesh.priority = 7
    mask = np.ones(6, dtype=bool)
    mask[0] = False
    mesh.changeFaceMask(mask)
    other = mesh.clone(scale=2.0)
    assert other is not mesh and type(other) is module3d.Object3D
    assert np.array_equal(other.coord, mesh.coord * 2)
    assert np.array_equal(other.fvert, mesh.fvert) and np.array_equal(other.fuvs, mesh.fuvs)
    assert np.array_equal(other.face_mask, mask)
    assert other.pickable and other.priority == 7 and other.object is holder
    assert other.getFaceGroup("top").color.tolist() == [1, 2, 3, 255]
    assert other.getFaceGroup("bottom").color is None
    # Deep copies: changing the clone leaves the original alone
    other.coord[0] = 42
    other.getFaceGroup("top").color[0] = 99
    assert mesh.coord[0].tolist() == [0, 0, 0]
    assert mesh.getFaceGroup("top").color[0] == 1
    assert mesh.parent_map is None and mesh.inverse_parent_map is None


def test_clone_of_a_fully_hidden_mesh_is_rejected():
    mesh = make_cube()
    mesh.changeFaceMask(np.zeros(6, dtype=bool))
    with pytest.raises(RuntimeError):
        mesh.clone(filterMaskedVerts=True)


def test_clone_filtering_masked_vertices_builds_vertex_maps():
    mesh = make_cube()
    mesh.setColor(np.arange(32, dtype=np.uint8).reshape(8, 4))
    only_top = mesh.getFaceMaskForGroups(["top"])
    mesh.changeFaceMask(only_top)
    sub = mesh.clone(scale=3.0, filterMaskedVerts=True)
    assert sub.getVertexCount() == 4 and sub.getFaceCount() == 1
    assert sub.parent is mesh
    pmap = sub.parent_map
    assert pmap.tolist() == [4, 5, 6, 7]
    assert np.array_equal(sub.coord, mesh.coord[pmap] * 3)
    assert np.array_equal(sub.color, mesh.color[pmap])
    # The remapped face describes the same corners
    assert np.array_equal(pmap[sub.fvert[0]], mesh.fvert[only_top][0])
    inv = sub.inverse_parent_map
    assert inv.tolist() == [-1, -1, -1, -1, 0, 1, 2, 3]
    assert sub.parent_map_weights.tolist() == [1, 1, 1, 1]
    # UVs are compacted to the ones still used
    assert np.array_equal(sub.texco, mesh.texco[np.unique(mesh.fuvs[only_top])])


def test_vertex_maps_chain_to_the_topmost_parent():
    mesh = make_cube()
    mask = np.ones(6, dtype=bool)
    mask[0] = False  # drop the bottom: still all 8 vertices
    mesh.changeFaceMask(mask)
    first = mesh.clone(filterMaskedVerts=True)
    first.changeFaceMask(first.getFaceMaskForGroups(["top"]))
    second = module3d.Object3D("second")
    first.filterMaskedVerts(second)
    # filterMaskedVerts links to the original, not to the intermediate copy
    assert second.parent is mesh
    first._parent_map = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint32)
    chained = module3d.Object3D("chained")
    chained.parent = first
    chained._parent_map = np.array([7, 6], dtype=np.uint32)
    chained._inverse_parent_map = np.array([-1] * 6 + [1, 0], dtype=np.int32)
    first._inverse_parent_map = np.arange(8, dtype=np.int32)
    assert chained.parent_map.tolist() == [7, 6]
    assert chained.inverse_parent_map.tolist() == [-1] * 6 + [1, 0]
    # Without an explicit map the identity is assumed at each level
    plain = module3d.Object3D("plain")
    plain.setCoords(np.zeros((8, 3)))
    plain.parent = first
    assert plain.parent_map.tolist() == list(range(8))
    assert plain.inverse_parent_map.tolist() == list(range(8))


def test_inverse_parent_map_with_multi_column_maps():
    root = make_cube("root")
    mid = module3d.Object3D("mid")
    mid.parent = root
    mid._inverse_parent_map = np.array([0, -1, 1, -1, 2, -1, 3, -1], dtype=np.int32)
    leaf = module3d.Object3D("leaf")
    leaf.parent = mid
    leaf._inverse_parent_map = np.array([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=np.int32)
    result = leaf.inverse_parent_map
    assert result.shape == (8, 2)
    assert result[[0, 2, 4, 6]].tolist() == [[10, 11], [20, 21], [30, 31], [40, 41]]
    assert np.all(result[[1, 3, 5, 7]] == -1)


def test_transformed_applies_a_matrix_to_a_copy():
    mesh = make_cube()
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)  # 90 degrees about z
    turned = mesh.transformed(rot)
    assert np.allclose(turned.coord, mesh.coord @ rot.T)
    assert np.array_equal(mesh.coord, np.array(CUBE_COORDS, dtype=np.float32))
    full = np.identity(4)
    full[:3, :3] = rot
    full[:3, 3] = (10, 20, 30)
    moved = mesh.transformed(full)
    assert np.allclose(moved.coord, mesh.coord @ rot.T + (10, 20, 30))
    # Normals were recomputed for the new orientation
    unit = moved.fnorm / np.linalg.norm(moved.fnorm, axis=1)[:, None]
    assert np.allclose(unit, OUTWARD @ rot.T)


def test_vertex_weights_follow_the_vertex_map():
    class Weights(object):
        def __init__(self, data, count=None):
            self.data, self.count = data, count

        def create(self, data, count):
            return Weights(data, count)

    mesh = make_cube()
    weights = Weights({
        "bottom_bone": (np.array([0, 1, 2, 3]), np.array([1.0, 1.0, 0.5, 0.5])),
        "top_bone": (np.array([4, 7]), np.array([0.25, 1.0])),
    })
    assert mesh.getVertexWeights(weights) is weights  # no parent: unchanged
    mesh.changeFaceMask(mesh.getFaceMaskForGroups(["top"]))
    sub = mesh.clone(filterMaskedVerts=True)
    mapped = sub.getVertexWeights(weights)
    assert mapped.count == 4
    assert list(mapped.data) == ["top_bone"]  # no bottom vertex survived
    assert mapped.data["top_bone"] == [(0, 0.25), (3, 1.0)]
