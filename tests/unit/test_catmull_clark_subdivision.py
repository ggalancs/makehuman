"""Tests for apps/catmull_clark_subdivision.py, the smooth (subdivided) mesh.

The cube has a closed-form Catmull-Clark result, so positions are checked
exactly; the rest are topological counts and invariants (Euler
characteristic, affine invariance, vertex maps).
"""

import types

import numpy as np
import pytest

import catmull_clark_subdivision as cks
import module3d

# Tangent computation of module3d divides by zero on axis-aligned UVs (see
# test_module3d.py); keep its RuntimeWarnings out of these tests' output.
pytestmark = [pytest.mark.filterwarnings("ignore::RuntimeWarning:module3d"),
              pytest.mark.filterwarnings("ignore:invalid value encountered in reduce:RuntimeWarning")]

# Unit cube centred on the origin, vertex index = x + 2*y + 4*z, faces wound
# counter-clockwise seen from outside
CUBE_COORDS = [(x - 0.5, y - 0.5, z - 0.5) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
CUBE_QUADS = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
CUBE_GROUPS = ["bottom", "top", "front", "back", "left", "right"]
UVS = [(0, 0), (1, 0), (1, 1), (0, 1)]


class Holder(object):
    """Stands in for the gui3d.Object owning the mesh: SubdivisionObject reads
    the material through it. Kept alive here because the link is a weakref."""

    material = types.SimpleNamespace(transparent=False)


HOLDER = Holder()


def make_mesh(coords, faces, uvs=UVS, fuvs=None, groups=None):
    mesh = module3d.Object3D("seed", 4)
    mesh.setCoords(coords)
    mesh.setUVs(uvs)
    names = groups or ["all"] * len(faces)
    fgs = {}
    for name in names:
        if name not in fgs:
            fgs[name] = mesh.createFaceGroup(name)
    mesh.setFaces(faces, fuvs if fuvs is not None else [(0, 1, 2, 3)] * len(faces),
                  [fgs[n].idx for n in names])
    mesh.updateIndexBuffer()
    mesh.calcNormals()
    mesh.object = HOLDER
    return mesh


def make_cube():
    return make_mesh(CUBE_COORDS, CUBE_QUADS, groups=CUBE_GROUPS)


def make_grid(n=2):
    """Flat n x n grid of unit quads in the z=0 plane, an open mesh."""
    coords = [(x, y, 0) for y in range(n + 1) for x in range(n + 1)]
    uvs = [(x / n, y / n) for y in range(n + 1) for x in range(n + 1)]
    quads = []
    for y in range(n):
        for x in range(n):
            a = y * (n + 1) + x
            quads.append((a, a + 1, a + n + 2, a + n + 1))
    return make_mesh(coords, quads, uvs=uvs, fuvs=quads)


def edges_of(fvert):
    return {tuple(sorted((int(f[i]), int(f[(i + 1) % 4])))) for f in fvert for i in range(4)}


# --- the cube --------------------------------------------------------------------

def test_cube_counts_and_euler_characteristic():
    sub = cks.createSubdivisionObject(make_cube())
    assert isinstance(sub, cks.SubdivisionObject)
    assert sub.name == "seed.sub"
    # V' = V + F + E, F' = 4F, and the surface is still a sphere: V - E + F = 2
    assert sub.getVertexCount() == 8 + 6 + 12
    assert sub.getFaceCount() == 24 and sub.fvert.shape == (24, 4)
    edges = edges_of(sub.fvert)
    assert len(edges) == 48
    assert sub.getVertexCount() - len(edges) + sub.getFaceCount() == 2
    # Every new face has one original corner, two edge points and a face point
    assert np.all(sub.fvert[:, 0] < sub.cbase)
    assert np.all((sub.fvert[:, 2] >= sub.cbase) & (sub.fvert[:, 2] < sub.ebase))
    assert np.all(sub.fvert[:, [1, 3]] >= sub.ebase)


def test_cube_positions_match_the_closed_form():
    sub = cks.createSubdivisionObject(make_cube())
    corners = sub.coord[:sub.cbase]
    centres = sub.coord[sub.cbase:sub.ebase]
    edge_points = sub.coord[sub.ebase:]
    # Original corners: (F + 2R + (n - 3)P) / n with n = 3, F = P/3 and
    # R = 2P/3 gives (P/3 + 4P/3) / 3 = 5P/9
    assert np.allclose(corners, np.array(CUBE_COORDS) * 5 / 9)
    # Face points: the face centres
    assert np.allclose(np.sort(np.abs(centres).sum(axis=1)), 0.5)
    assert np.allclose(np.abs(centres).max(axis=1), 0.5)
    # Edge points: average of the endpoints and the two adjacent face points
    for p in edge_points:
        assert np.allclose(np.sort(np.abs(p)), [0.0, 0.375, 0.375])


def test_cube_subdivision_keeps_its_symmetry_and_shrinks_inside_the_hull():
    sub = cks.createSubdivisionObject(make_cube())
    assert np.allclose(sub.coord.mean(axis=0), 0.0, atol=1e-6)
    assert np.all(np.abs(sub.coord) <= 0.5 + 1e-6)
    # Normals still point away from the centre
    unit = sub.fnorm / np.linalg.norm(sub.fnorm, axis=1)[:, None]
    centroids = sub.coord[sub.fvert].mean(axis=1)
    assert np.all(np.sum(unit * centroids, axis=1) > 0)
    assert np.allclose(np.linalg.norm(sub.vnorm, axis=1), 1.0)


def test_cube_uvs_are_interpolated():
    sub = cks.createSubdivisionObject(make_cube())
    assert sub.hasUVs()
    assert np.allclose(sub.texco[:sub.tcbase], UVS)  # original UVs kept
    assert np.allclose(sub.texco[sub.tcbase:sub.tebase], (0.5, 0.5))  # face centres
    # UV edge points are midpoints of the four UV square edges
    expected = {(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)}
    assert {tuple(t) for t in np.round(sub.texco[sub.tebase:], 6)} == expected
    assert np.array_equal(sub.r_texco, sub.texco[sub.tmap])


def test_groups_and_face_mask_are_inherited():
    seed = make_cube()
    mask = np.ones(6, dtype=bool)
    mask[1] = False
    seed.changeFaceMask(mask)
    sub = cks.createSubdivisionObject(seed)
    assert [fg.name for fg in sub.faceGroups] == CUBE_GROUPS
    assert sub.group.tolist() == np.repeat(seed.group, 4).tolist()
    assert sub.face_mask.tolist() == np.repeat(mask, 4).tolist()


def test_vertex_maps_to_the_seed_mesh():
    seed = make_cube()
    sub = cks.createSubdivisionObject(seed)
    pmap = sub.parent_map
    weights = sub.parent_map_weights
    assert pmap.shape == (26, 4)
    # Corners map to themselves with weight 1
    assert pmap[:8, 0].tolist() == list(range(8)) and np.all(pmap[:8, 1:] == -1)
    assert np.all(weights[:8] == 1.0)
    # Face points map to the four corners of their face, weight 1/4
    for i, face in enumerate(seed.fvert):
        assert sorted(pmap[8 + i]) == sorted(face)
    assert np.all(weights[8:14] == 0.25)
    # Edge points map to the two ends of their edge, weight 1/2
    ends = {tuple(sorted(row[:2])) for row in pmap[14:]}
    assert ends == edges_of(seed.fvert)
    assert np.all(pmap[14:, 2:] == -1) and np.all(weights[14:] == 0.5)
    # Every reference in the forward map appears in the inverse map
    inv = sub.inverse_parent_map
    assert inv.shape == (8, 1 + 2 * seed.MAX_FACES)
    assert inv[:, 0].tolist() == list(range(8))
    for s_idx, row in enumerate(pmap):
        for p_idx in row[row >= 0]:
            assert s_idx in inv[p_idx]
    # Each corner is referenced by 1 corner, 3 face points and 3 edge points
    assert np.all((inv >= 0).sum(axis=1) == 7)


def test_vertex_weights_of_the_subdivided_mesh():
    class Weights(object):
        def __init__(self, data, count=None):
            self.data, self.count = data, count

        def create(self, data, count):
            return Weights(data, count)

    sub = cks.createSubdivisionObject(make_cube())
    weights = Weights({"bone": (np.array([0]), np.array([1.0]))})
    mapped = sub.getVertexWeights(weights)
    assert mapped.count == 26
    got = dict(mapped.data["bone"])
    # Vertex 0 itself, its 3 face points (1/4) and its 3 edge points (1/2)
    assert got[0] == 1.0
    assert sorted(got.values()) == [0.25] * 3 + [0.5] * 3 + [1.0]


# --- open meshes, masks and updates --------------------------------------------

def test_flat_grid_stays_flat_with_straight_borders():
    seed = make_grid(3)
    sub = cks.createSubdivisionObject(seed)
    assert sub.getFaceCount() == 4 * 9
    assert sub.getVertexCount() == 16 + 9 + 24
    assert np.allclose(sub.coord[:, 2], 0.0)
    assert np.all((sub.coord[:, :2] >= 0) & (sub.coord[:, :2] <= 3))
    # Vertices in the middle of a straight border do not move, and edge points
    # on the border are plain midpoints
    for i, p in enumerate(seed.coord):
        on_border = p[0] in (0, 3) or p[1] in (0, 3)
        corner = p[0] in (0, 3) and p[1] in (0, 3)
        if on_border and not corner:
            assert np.allclose(sub.coord[i], p)
    # On a regular flat grid every edge point (border or interior) is the
    # midpoint of its edge and every face point the centre of its quad
    ends = sub.parent_map[sub.ebase:, :2]
    assert np.allclose(sub.coord[sub.ebase:], seed.coord[ends].mean(axis=1))
    corners = sub.parent_map[sub.cbase:sub.ebase]
    assert np.allclose(sub.coord[sub.cbase:sub.ebase], seed.coord[corners].mean(axis=1))
    # Interior vertices of a regular flat grid do not move either
    assert np.allclose(sub.coord[5], seed.coord[5])


def test_concave_border_vertex_stays_in_the_plane():
    # An L of three quads: the inner corner has 3 faces but 4 edges
    coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (0, 2, 0), (1, 2, 0)]
    quads = [(0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6)]
    seed = make_mesh(coords, quads)
    sub = cks.createSubdivisionObject(seed)
    assert seed.nfaces[4] == 3 and sub.nedges[4] == 4
    assert np.allclose(sub.coord[:, 2], 0.0)
    # (sum of the two border edge midpoints + P) / 3
    assert np.allclose(sub.coord[4], ((1.5, 1, 0) + np.array((1, 1.5, 0)) + (1, 1, 0)) / 3)


def make_strip():
    """Three unit quads in a row; vertex index = 2*x + y, so the vertices only
    used by the last quad (6 and 7) come last, like the human's helpers."""
    coords = [(x, y, 0) for x in range(4) for y in range(2)]
    quads = [(2 * x, 2 * x + 2, 2 * x + 3, 2 * x + 1) for x in range(3)]
    return make_mesh(coords, quads, uvs=[c[:2] for c in coords], fuvs=quads)


def test_static_face_mask_leaves_faces_out():
    seed = make_strip()
    static = np.array([True, True, False])
    sub = cks.createSubdivisionObject(seed, staticFaceMask=static)
    assert sub.staticFaceMask is static
    assert sub.face_map.tolist() == [0, 1]
    assert sub.face_rmap.tolist() == [0, 1, -1]
    assert sub.vtx_map.tolist() == [0, 1, 2, 3, 4, 5]
    assert sub.getFaceCount() == 8
    # 6 corners + 2 face points + 7 edges (the two quads share one)
    assert sub.getVertexCount() == 6 + 2 + 7
    assert np.all(sub.coord[:, 0] <= 2)  # nothing from the third quad
    inv = sub.inverse_parent_map
    assert inv[:, 0].tolist() == [0, 1, 2, 3, 4, 5, -1, -1]
    assert np.all(inv[6:] == -1)  # vertices left out are not referenced


@pytest.mark.xfail(strict=True, raises=(IndexError, AssertionError), reason=(
    "catmull_clark_subdivision.py:314 maps the seed's (already original) face "
    "vertex indices through vtx_map again, and line 316 stores the edge "
    "endpoints as local (vtx_rmap) indices without mapping them back. Both are "
    "identities only when the statically visible vertices are a prefix "
    "0..n-1, as for the default human (body 0..13379, helpers after); any "
    "other static face mask raises IndexError or builds a wrong parent_map"))
def test_static_face_mask_hiding_vertices_from_the_middle():
    seed = make_cube()
    static = seed.getFaceMaskForGroups(["top", "front"])  # vertices 0, 1, 4..7
    sub = cks.createSubdivisionObject(seed, staticFaceMask=static)
    pmap = sub.parent_map
    assert pmap[:sub.cbase, 0].tolist() == [0, 1, 4, 5, 6, 7]
    for i, face in enumerate(seed.fvert[static]):
        assert sorted(pmap[sub.cbase + i]) == sorted(face)
    ends = {tuple(sorted(row[:2])) for row in pmap[sub.ebase:]}
    assert ends == edges_of(seed.fvert[static])


def test_dynamic_face_mask_is_remapped_from_the_seed():
    seed = make_cube()
    sub = cks.createSubdivisionObject(seed)
    mask = np.zeros(6, dtype=bool)
    mask[[0, 5]] = True
    sub.changeFaceMask(mask)
    assert sub.face_mask.tolist() == np.repeat(mask, 4).tolist()
    direct = np.zeros(24, dtype=bool)
    direct[3] = True
    sub.changeFaceMask(direct, remapFromUnsubdivided=False)
    assert np.flatnonzero(sub.face_mask).tolist() == [3]


def test_subdivision_is_affine_invariant():
    seed = make_cube()
    sub = cks.createSubdivisionObject(seed)
    before = sub.coord.copy()
    m = np.array([[2.0, 0.3, 0.0], [0.0, 1.0, -0.5], [0.1, 0.0, 3.0]])
    t = np.array([5.0, -2.0, 1.0])
    seed.changeCoords(seed.coord @ m.T + t)
    cks.updateSubdivisionObject(sub)
    assert np.allclose(sub.coord, before @ m.T + t, atol=1e-5)
    assert np.array_equal(sub.r_coord, sub.coord[sub.vmap])  # synced for drawing


def test_clone_resubdivides_the_cloned_seed():
    seed = make_cube()
    sub = cks.createSubdivisionObject(seed)
    copy = sub.clone(scale=2.0)
    assert isinstance(copy, cks.SubdivisionObject)
    assert copy.parent is not seed
    assert np.allclose(copy.coord, sub.coord * 2)
    seed.changeFaceMask(seed.getFaceMaskForGroups(["top"]))
    filtered = sub.clone(filterMaskedVerts=True)
    assert filtered.getFaceCount() == 4 and filtered.getVertexCount() == 4 + 1 + 4


def test_meshes_that_cannot_be_subdivided_are_returned_unchanged():
    tris = module3d.Object3D("tris", 4)
    tris.setCoords([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    tris.setUVs(UVS)
    tris.setFaces([(0, 1, 2, 0)], [(0, 1, 2, 0)])
    assert cks.createSubdivisionObject(tris) is tris
    no_uv = module3d.Object3D("no_uv", 4)
    no_uv.setCoords(CUBE_COORDS)
    no_uv.setFaces(CUBE_QUADS)
    assert cks.createSubdivisionObject(no_uv) is no_uv


def test_too_low_pole_count_is_reported():
    seed = make_cube()
    seed.MAX_FACES = 2  # each cube corner has three edges
    with pytest.raises(RuntimeError, match="Pole-count too low"):
        cks.createSubdivisionObject(seed)


def test_reverse_n_to_m_map():
    refs = np.array([[0, 1], [1, 2], [2, 0]])
    out = -np.ones((3, 2), dtype=np.int32)
    cks._reverse_n_to_m_map(refs, out, offset=10)
    # Row v lists (offset +) the rows of refs that mention v
    assert out.tolist() == [[10, 12], [10, 11], [11, 12]]
