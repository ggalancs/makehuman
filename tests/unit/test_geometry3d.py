"""Tests for core/geometry3d.py: the helper meshes (background rectangle,
frames, grids, axes and a debug cube)."""

import os

import numpy as np
import pytest

import geometry3d
import transformations as tm
from core import G

MH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "makehuman")

# Tangent computation of module3d divides by zero on axis-aligned UVs (see
# test_module3d.py); keep its RuntimeWarnings out of these tests' output.
pytestmark = [pytest.mark.filterwarnings("ignore::RuntimeWarning:module3d"),
              pytest.mark.filterwarnings("ignore:invalid value encountered in reduce:RuntimeWarning")]


def bbox2d(mesh):
    lo, hi = mesh.coord.min(axis=0), mesh.coord.max(axis=0)
    return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]


def unit_normal(mesh):
    mesh.calcFaceNormals()
    n = mesh.fnorm[0]
    return n / np.linalg.norm(n)


# --- RectangleMesh -----------------------------------------------------------

def test_rectangle_from_the_origin():
    r = geometry3d.RectangleMesh(4, 2, texture="bg.png")
    assert r.name == "rectangle_bg.png"
    assert r.coord.tolist() == [[0, 0, 0], [4, 0, 0], [4, 2, 0], [0, 2, 0]]
    assert r.texco.tolist() == [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert r.fvert.tolist() == [[0, 1, 2, 3]] and r.fuvs.tolist() == [[0, 1, 2, 3]]
    assert r.getFaceGroups() == ["rectangle"] and r.cameraMode == 1
    assert np.allclose(unit_normal(r), [0, 0, 1])  # facing the camera
    assert len(r.r_coord) == 4  # index buffer ready for drawing


def test_centered_rectangle():
    r = geometry3d.RectangleMesh(4, 2, centered=True)
    assert r.coord.tolist() == [[-2, -1, 0], [2, -1, 0], [2, 1, 0], [-2, 1, 0]]
    assert r.getOffset() == (0, 0)
    width, height = r.getSize()
    assert width == 4 and abs(height) == 2


def test_move_shifts_the_rectangle():
    r = geometry3d.RectangleMesh(4, 2, centered=True)
    r.move(1.5, -3)
    assert bbox2d(r) == [-0.5, -4, 3.5, -2]
    assert r.getOffset() == (1.5, -3)
    assert np.array_equal(r.r_coord, r.coord[r.vmap])  # synced for drawing


def test_resize_keeps_the_centre_of_a_centered_rectangle():
    r = geometry3d.RectangleMesh(4, 2, centered=True)
    r.move(1, 1)
    r.resize(10, 6)
    assert bbox2d(r) == [-4, -2, 6, 4]
    assert r.getOffset() == (1, 1)
    assert np.allclose(unit_normal(r), [0, 0, 1])


def test_set_position_centres_the_rectangle_there():
    r = geometry3d.RectangleMesh(4, 2, centered=True)
    r.setPosition(5, 7)
    assert bbox2d(r) == [3, 6, 7, 8]
    assert r.getOffset() == (5, 7)
    r.resetPosition()
    assert bbox2d(r) == [-2, -1, 2, 1]


@pytest.mark.xfail(strict=True, reason=(
    "geometry3d.py:181 getSize() returns (width, y0 - y1), a negative "
    "height; setPosition()/resetPosition() rebuild the vertices with it, which "
    "swaps the top and bottom rows: the face turns away from the camera and "
    "the texture (UV 0,0 at vertex 0) is mirrored vertically. Used by "
    "0_modeling_background.py setBackgroundPosition"))
def test_set_position_keeps_the_rectangle_orientation():
    r = geometry3d.RectangleMesh(4, 2, centered=True)
    before = r.coord.copy()
    r.setPosition(0, 0)  # where it already is
    assert np.array_equal(r.coord, before)
    assert np.allclose(unit_normal(r), [0, 0, 1])


@pytest.mark.xfail(strict=True, reason=(
    "geometry3d.py:181/191: with getSize()'s negative height, getOffset() of "
    "an uncentered rectangle is its top-left corner (0, height) instead of its "
    "origin, so resize() to the same size moves it up by its height"))
def test_resize_to_the_same_size_keeps_an_uncentered_rectangle_in_place():
    r = geometry3d.RectangleMesh(4, 2)
    before = r.coord.copy()
    r.resize(4, 2)
    assert np.array_equal(r.coord, before)


def test_rotated_rectangle():
    rot = tm.rotation_matrix(np.pi / 2, [1, 0, 0])  # stand it up in the x-z plane
    r = geometry3d.RectangleMesh(4, 2, centered=True, rotation=rot)
    assert np.allclose(r.coord, [[-2, 0, -1], [2, 0, -1], [2, 0, 1], [-2, 0, 1]])
    # Size and offset are measured in the rectangle's own (unrotated) frame
    assert r.getSize()[0] == pytest.approx(4)
    assert np.allclose(r._originalVerts(r.coord), [[-2, -1, 0], [2, -1, 0], [2, 1, 0], [-2, 1, 0]], atol=1e-6)
    r.move(1, 1)
    assert np.allclose(r.coord, [[-1, 0, 0], [3, 0, 0], [3, 0, 2], [-1, 0, 2]], atol=1e-6)
    assert np.allclose(r.getOffset(), (1, 1))
    # The world-space box is the rotated one
    box = r._bbox(ignore_rotation=False)
    assert np.allclose(box, [[-1, 0, 0], [3, 0, 2]], atol=1e-6)


def test_bbox_of_an_empty_rectangle():
    r = geometry3d.RectangleMesh(4, 2)
    r.coord = np.zeros((0, 3), dtype=np.float32)
    assert r._bbox().tolist() == [[0, 0, 0], [0, 0, 0]]


def test_set_colors_per_corner():
    r = geometry3d.RectangleMesh(4, 2)
    r.setColors([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.5], [1.0, 1.0, 1.0])
    assert r.color.tolist() == [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 127], [255, 255, 255, 255]]
    assert np.array_equal(r.r_color, r.color[r.vmap])


# --- FrameMesh and Cube ---------------------------------------------------------

def test_frame_is_four_lines_around_the_rectangle():
    f = geometry3d.FrameMesh(4, 2)
    assert f.vertsPerPrimitive == 2 and f.getFaceCount() == 4
    corners = [tuple(c) for c in f.coord]
    assert corners == [(0, 0, 0), (4, 0, 0), (4, 2, 0), (0, 2, 0)]
    segments = {frozenset(e) for e in f.fvert.tolist()}
    assert segments == {frozenset(s) for s in [(0, 1), (1, 2), (2, 3), (3, 0)]}
    f.move(1, -1)
    assert bbox2d(f) == [1, -1, 5, 1]
    f.resize(10, 5)
    assert bbox2d(f) == [0, 0, 10, 5]


def test_cube_dimensions_and_faces():
    c = geometry3d.Cube(2)
    assert c.width == c.height == c.depth == 2
    box = geometry3d.Cube(1, 2, 3, texture="t")
    assert box.name == "cube_t" and box.cameraMode == 0
    assert box.calcBBox().tolist() == [[0, 0, 0], [1, 2, 3]]
    assert box.getFaceCount() == 6
    # Every vertex is a corner of exactly three faces, every face has 4 corners
    assert box.nfaces.tolist() == [3] * 8
    assert all(len(set(f)) == 4 for f in box.fvert.tolist())
    box.resize(4, 5, 6)
    assert box.calcBBox().tolist() == [[0, 0, 0], [4, 5, 6]]


@pytest.mark.xfail(strict=True, reason=(
    "geometry3d.py:326 calls setFaces(f, fg.idx): the group index lands in "
    "the uvs parameter, so the cube claims UVs while every face corner uses "
    "UV 0 (RectangleMesh and GridMesh pass setFaces(f, uvs, fg.idx))"))
def test_cube_uv_mapping_is_not_degenerate():
    box = geometry3d.Cube(1)
    assert not box.hasUVs() or len(np.unique(box.fuvs)) > 1


# --- GridMesh ------------------------------------------------------------------------

class FakeCamera(object):
    def __init__(self, zoom=1.0, radius=1.0, inclination=10.0, view=None):
        self.zoomFactor = zoom
        self.radius = radius
        self.inclination = inclination
        self.view = view

    def getVerticalInclination(self):
        return self.inclination

    def isInFrontView(self):
        return self.view == "front"

    def isInBackView(self):
        return self.view == "back"

    def isInSideView(self):
        return self.view == "side"

    def isInTopView(self):
        return self.view == "top"

    def isInBottomView(self):
        return self.view == "bottom"


@pytest.fixture
def camera(monkeypatch):
    cam = FakeCamera()
    monkeypatch.setattr(G, "cameras", [cam, cam])
    return cam


def is_axis_aligned(mesh):
    a, b = mesh.coord[mesh.fvert[:, 0]], mesh.coord[mesh.fvert[:, 1]]
    return np.all(np.sum(~np.isclose(a, b), axis=1) == 1)


@pytest.mark.parametrize("plane, flat_axis", [(0, 2), (1, 1)], ids=["back", "ground"])
def test_grid_lines(plane, flat_axis):
    g = geometry3d.GridMesh(4, 4, spacing=2, offset=-10, plane=plane)
    assert g.name == ("ground_grid" if plane else "back_grid")
    assert g.vertsPerPrimitive == 2
    assert g.getFaceCount() == 5 + 5  # 5 lines each way
    assert g.getVertexCount() == 2 * 10 and g.mainGridEnd == 20
    assert np.all(g.coord[:, flat_axis] == -10)  # the whole grid sits at the offset
    assert is_axis_aligned(g)
    in_plane = [a for a in range(3) if a != flat_axis]
    # Centred on the origin: lines at -4, -2, 0, 2, 4
    assert sorted(set(g.coord[:, in_plane].ravel().tolist())) == [-4, -2, 0, 2, 4]
    assert not g.hasSubGrid()
    assert g.cameraMode == 0


def test_grid_subdivisions_fall_between_main_lines():
    g = geometry3d.GridMesh(2, 2, spacing=1, plane=1, subgrids=4, static=True)
    assert g.cameraMode == 1
    assert g.hasSubGrid()
    sub_lines = (2 * (4 - 1)) * 2  # 3 extra lines per box, 2 boxes, both ways
    assert g.getFaceCount() == 3 + 3 + sub_lines
    sub = g.coord[g.mainGridEnd:]
    assert len(sub) == 2 * sub_lines
    assert is_axis_aligned(g)
    positions = set(np.round(np.concatenate([sub[:, 0], sub[:, 2]]), 6).tolist())
    assert {-0.75, -0.5, -0.25, 0.25, 0.5, 0.75} <= positions
    main = set(g.coord[:g.mainGridEnd][:, [0, 2]].ravel().tolist())
    assert main == {-1, 0, 1}


def test_grid_colors():
    g = geometry3d.GridMesh(2, 2, subgrids=2)
    g.setMainColor([1.0, 0.0, 0.0])
    g.setSubColor([0.0, 0.0, 1.0, 0.5])
    assert np.all(g.color[:g.mainGridEnd] == (255, 0, 0, 255))
    assert np.all(g.color[g.mainGridEnd:] == (0, 0, 255, 127))
    plain = geometry3d.GridMesh(2, 2)
    plain.setSubColor([0.0, 1.0, 0.0])  # no subgrid: nothing to colour
    assert np.all(plain.color == 255)


def test_grid_visibility_flag(camera):
    g = geometry3d.GridMesh(2, 2)
    assert g.visibility is True
    g.visibility = False
    assert g.visibility is False


def test_subgrid_is_hidden_when_zoomed_out(camera):
    g = geometry3d.GridMesh(2, 2, subgrids=2)
    g.minSubgridZoom = 1.0
    camera.zoomFactor, camera.radius = 0.05, 1.0  # 0.5 < 1: too far for the subgrid
    assert g.visibility is True
    assert np.all(g.face_mask[:g.mainGridEnd // 2])
    assert not np.any(g.face_mask[g.mainGridEnd // 2:])
    assert len(g.index) == g.mainGridEnd // 2
    camera.zoomFactor = 0.5  # 5 >= 1
    assert g.visibility is True
    assert np.all(g.face_mask)
    assert len(g.index) == g.getFaceCount()


def test_ground_grid_hides_below_the_ground_and_when_seen_edge_on(camera):
    g = geometry3d.GridMesh(2, 2, plane=1)
    g.restrictVisibleAboveGround = True
    camera.inclination = -5
    assert g.visibility is False
    camera.inclination = 200
    assert g.visibility is False
    camera.inclination = 30
    assert g.visibility is True
    for view in ("front", "back", "side"):
        camera.view = view
        assert g.visibility is False
    camera.view = "top"
    assert g.visibility is True


def test_back_grid_restricted_to_horizontal_views(camera):
    g = geometry3d.GridMesh(2, 2, plane=0)
    g.restrictVisibleToCamera = True
    for view, visible in [("front", True), ("back", True), ("side", True), ("top", False), (None, False)]:
        camera.view = view
        assert g.visibility is visible, view


@pytest.mark.xfail(strict=True, reason=(
    "geometry3d.py:501 tests `camera.isInTopView() and "
    "camera.isInBottomView()`, which can never both hold, so an unrestricted "
    "vertical grid is never hidden when looked at from above or below (the "
    "ground-grid branch above uses `or`). Latent: MakeHuman sets "
    "restrictVisibleToCamera on its back grid"))
def test_unrestricted_back_grid_hides_in_top_view(camera):
    g = geometry3d.GridMesh(2, 2, plane=0)
    camera.view = "top"
    assert g.visibility is False


# --- AxisMesh ------------------------------------------------------------------------

def test_axis_mesh_colors_and_scale(monkeypatch):
    monkeypatch.chdir(MH_DIR)  # the mesh is loaded from data/3dobjs, relative to makehuman/
    axis = geometry3d.AxisMesh()
    assert axis.getVertexCount() == 130 and axis.getFaceCount() == 100
    groups = {name.lower(): axis.getVerticesForGroups([name]) for name in axis.getFaceGroups()}
    expected = {"red": (255, 0, 0, 255), "green": (0, 255, 0, 255), "blue": (0, 0, 255, 255)}
    for colour, rgba in expected.items():
        verts = next(v for name, v in groups.items() if colour in name)
        assert len(verts) > 0
        assert np.all(axis.color[verts] == rgba)
    assert np.array_equal(axis.r_color, axis.color[axis.vmap])
    big = geometry3d.AxisMesh(scale=2.0)
    assert np.allclose(big.coord, axis.coord * 2)
    assert np.allclose(big.r_coord, big.coord[big.vmap])
