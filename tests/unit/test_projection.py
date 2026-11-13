"""shared/projection.py: software rasterizer and texture-map baking (light maps,
UV-map export, seam masks) used by the SSS render and the map exporters.

There is no OpenGL context in unit tests: the hardware paths (mh.renderSkin,
mh.Texture) are replaced by recorders, and every expected image is computed
independently, by hand or with plain numpy barycentric coordinates."""

import types

import numpy as np
import pytest

import image
import mh
import module3d
import progress
import projection
from core import G
from material import Color

# Lit quad of the light-map tests: unit square in the z=0 plane, normals +z
QUAD_COORDS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
# Its UV layout: a skewed quad, chosen so that no pixel centre of a 16x16 map
# falls on an edge (the rasterizer only fills pixels strictly inside)
QUAD_UVS = [(0.117, 0.111), (0.855, 0.1), (0.883, 0.859), (0.106, 0.772)]
# Light straight above vertex 0: shade = cos(angle to +z) * 256, clipped to
# 255 and truncated to uint8. Vertex 0: 1 -> 255; vertices 1 and 3:
# 1/sqrt(2)*256 = 181.02 -> 181; vertex 2: 1/sqrt(3)*256 = 147.80 -> 147.
LIGHT = (0.0, 0.0, 1.0)
SHADES = [255, 181, 147, 181]


class FakeApp(object):
    def __init__(self):
        self.modelCamera = types.SimpleNamespace(eye=(0, 0, 10), focus=(0, 0, 0),
                                                 getRotation=lambda: [0, 0, 0])
        self.selectedHuman = None

    def progress(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def app(monkeypatch):
    fake = FakeApp()
    monkeypatch.setattr(G, "app", fake, raising=False)
    monkeypatch.setattr(progress, "current_Progress_", None)
    # Never ask the real glmodule: without a GL context its calls crash the process
    monkeypatch.setattr(mh, "hasRenderSkin", lambda: False)
    monkeypatch.setattr(mh, "renderSkin", _no_gl)
    monkeypatch.setattr(mh, "Texture", _no_gl)
    return fake


def _no_gl(*args, **kwargs):
    raise AssertionError("hardware render called with hasRenderSkin() False")


class Renderer(object):
    """Stands in for mh.renderSkin: records each call and returns the image
    given for the light map (color passed) or for the mask (no color)."""

    def __init__(self, lightmap=None, mask=None, error=None):
        self.lightmap, self.mask, self.error = lightmap, mask, error
        self.calls = []

    def __call__(self, dst, vertsPerPrimitive, verts, index=None, color=None, clearColor=None):
        self.calls.append(dict(dst=dst, vpp=vertsPerPrimitive, verts=np.array(verts),
                               index=None if index is None else np.array(index),
                               color=None if color is None else np.array(color),
                               clearColor=clearColor))
        if self.error:
            raise self.error
        src = self.lightmap if color is not None else self.mask
        return image.Image(data=np.array(src, dtype=np.uint8))


def use_gl(monkeypatch, renderer):
    monkeypatch.setattr(mh, "hasRenderSkin", lambda: True)
    monkeypatch.setattr(mh, "renderSkin", renderer)
    return renderer


def quad_mesh(uvs=QUAD_UVS):
    mesh = module3d.Object3D("quad", 4)
    mesh.setCoords(QUAD_COORDS)
    mesh.setUVs(uvs)
    skin = mesh.createFaceGroup("skin")
    mesh.setFaces([(0, 1, 2, 3)], [(0, 1, 2, 3)], [skin.idx])
    mesh.vnorm[:] = (0, 0, 1)
    mesh.updateIndexBuffer()
    return mesh


# --- independent reference rasterization ----------------------------------------------

def barycentric(tri, W, H):
    """Weights (3, H, W) of every pixel centre w.r.t. triangle tri (3 x (x, y)),
    by Cramer's rule."""
    ys, xs = np.mgrid[0:H, 0:W] + 0.5
    (x0, y0), (x1, y1), (x2, y2) = tri
    det = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    l0 = ((xs - x2) * (y1 - y2) - (x1 - x2) * (ys - y2)) / det
    l1 = ((x0 - x2) * (ys - y2) - (xs - x2) * (y0 - y2)) / det
    return np.stack([l0, l1, 1 - l0 - l1])


def uv_pixels(uvs, W, H):
    """Texture-space pixel coordinates: x = u*W, y = H - v*H (row 0 at the top)."""
    return [(u * W, H - v * H) for u, v in uvs]


def quad_triangles(uvs, W, H):
    p = uv_pixels(uvs, W, H)
    return [((p[0], p[1], p[2]), (0, 1, 2)), ((p[2], p[3], p[0]), (2, 3, 0))]


def expected_shading(uvs, W, H, shades):
    """Per-pixel RGB of a quad lit with per-vertex grey levels, and its coverage."""
    out = np.zeros((H, W, 3))
    covered = np.zeros((H, W), dtype=bool)
    for tri, vidx in quad_triangles(uvs, W, H):
        w = barycentric(tri, W, H)
        assert np.abs(w).min() > 4e-3, "premise: no pixel centre on a triangle edge"
        inside = np.all(w > 0, axis=0)
        value = sum(w[k] * shades[v] for k, v in enumerate(vidx))
        out[inside] = np.floor(value[inside])[:, None]
        covered |= inside
    return out, covered


def dilate(mask):
    """8-neighbour dilation without wrap-around."""
    padded = np.pad(mask, 1)
    H, W = mask.shape
    return np.any([padded[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)], axis=0)


# --- small helpers --------------------------------------------------------------------

def test_v4to3_divides_by_w():
    assert projection.v4to3(np.matrix([[2.0], [4.0], [6.0], [2.0]])).tolist() == [1, 2, 3]


def test_vnormalize_rows():
    v = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, -2.0]])
    assert np.allclose(projection.vnormalize(v), [[0.6, 0.8, 0], [0, 0, -1]])


def test_color_shader_interpolates_vertex_colors():
    # uvw[k] is the weight of the vertex opposite edge k: vertex 2 for k=0,
    # vertex 0 for k=1, vertex 1 for k=2
    colors = np.array([[[100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 100, 0]]], dtype=float)
    shader = projection.ColorShader(colors)
    uvw = np.array([[[0.2, 0.5, 0.3]]])
    assert np.allclose(shader.shade(0, None, uvw), [[[50, 30, 20, 0]]])


def test_mask_shader_fills_the_block():
    xy = np.zeros((2, 3, 2), dtype=int)
    assert projection.MaskShader().shade(0, xy, None).tolist() == [[[255] * 4] * 3] * 2
    out = projection.MaskShader([1, 2, 3, 4]).shade(0, xy, None)
    assert out.shape == (2, 3, 4) and (out == [1, 2, 3, 4]).all()


# --- rasterizer -----------------------------------------------------------------------

TRI = [(1.2, 0.7), (6.6, 2.1), (3.3, 7.4)]


@pytest.mark.parametrize("winding", [[0, 1, 2], [0, 2, 1]], ids=["ccw", "cw"])
def test_rasterize_fills_pixel_centres_strictly_inside(winding):
    dst = image.Image(data=np.zeros((8, 8, 4), dtype=np.uint8))
    coords = np.array([TRI])[:, winding, :]
    projection.RasterizeTriangles(dst, coords, projection.MaskShader([10, 20, 30, 40]))
    w = barycentric(TRI, 8, 8)
    assert np.abs(w).min() > 1e-3
    inside = np.all(w > 0, axis=0)
    assert 10 < inside.sum() < 40  # a real triangle, not all or nothing
    assert (dst.data[inside] == [10, 20, 30, 40]).all()
    assert (dst.data[~inside] == 0).all()


def test_rasterize_interpolates_colors_barycentrically():
    dst = image.Image(data=np.zeros((8, 8, 4), dtype=np.uint8))
    colors = np.array([[[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]]], dtype=np.uint8)
    projection.RasterizeTriangles(dst, np.array([TRI]), projection.ColorShader(colors))
    w = barycentric(TRI, 8, 8)
    inside = np.all(w > 0, axis=0)
    expected = np.floor(np.moveaxis(w, 0, -1) @ colors[0].astype(float))
    assert np.abs(dst.data[inside].astype(int) - expected[inside]).max() <= 1
    assert (dst.data[~inside] == 0).all()
    # Pure red at the pixel nearest vertex 0, so the weights are not all equal
    assert dst.data[1, 2, 0] > 150 and dst.data[1, 2, 2] < 60


# --- mesh queries ---------------------------------------------------------------------

def test_get_faces_skips_joint_and_helper_groups():
    mesh = module3d.Object3D("groups", 4)
    mesh.setCoords([(i, 0, 0) for i in range(4)])
    names = ["body", "joint-ankle", "helper-tights", "eyes", "jointless-body"]
    groups = [mesh.createFaceGroup(n) for n in names]
    mesh.setFaces([(0, 1, 2, 3)] * 5, None, [g.idx for g in groups])
    # Only the prefix counts: "jointless-body" starts with "joint" too
    assert projection.getFaces(mesh).tolist() == [0, 3]


def test_get_camera_is_the_eye_direction_in_object_space(app):
    # Object rotated 90 degrees about y and moved to x=5; camera on +z.
    # Inverse transform of eye (0,0,10) is (-10,0,-5), of focus (0,0,0) is
    # (0,0,-5): the camera looks from -x in object space.
    mesh = types.SimpleNamespace(transform=np.matrix([[0, 0, 1, 5],
                                                      [0, 1, 0, 0],
                                                      [-1, 0, 0, 0],
                                                      [0, 0, 0, 1]], dtype=float))
    assert np.allclose(projection.getCamera(mesh), [-1, 0, 0])


# --- seam fixing ----------------------------------------------------------------------

def seam_case():
    """6x4 RGB image with a masked band in columns 2-3: column 2 rows hold
    40, 80, 120, 160 and column 3 holds 200. Everything else is black."""
    data = np.zeros((4, 6, 3), dtype=np.uint8)
    data[:, 2] = np.array([40, 80, 120, 160])[:, None]
    data[:, 3] = 200
    mask = np.zeros((4, 6, 3), dtype=np.uint8)
    mask[:, 2:4] = 255
    return data, mask


# One step of expansion: column 1 takes the mean of its masked neighbours in
# column 2 (rows wrap around): (160+40+80)/3=93.3, (40+80+120)/3=80,
# (80+120+160)/3=120, (120+160+40)/3=106.7. Column 4 only sees 200s.
SEAM_COL1 = [93, 80, 120, 107]


def check_seam_result(out):
    data, _ = seam_case()
    assert (out[:, 2:4] == data[:, 2:4]).all()          # masked pixels untouched
    assert (out[:, 1] == np.array(SEAM_COL1)[:, None]).all()
    assert (out[:, 4] == 200).all()
    assert (out[:, 0] == 0).all() and (out[:, 5] == 0).all()  # two pixels away: still empty


def test_fix_seams_with_a_mask_image_grows_one_pixel():
    data, mask = seam_case()
    out, newmask = projection.fixSeams(image.Image(data=data), image.Image(data=mask), border=1)
    check_seam_result(out.data)
    assert (newmask.data[:, 1:5] == 255).all() and (newmask.data[:, [0, 5]] == 0).all()


def test_fix_seams_border_zero_changes_nothing():
    data, mask = seam_case()
    out, newmask = projection.fixSeams(image.Image(data=data), image.Image(data=mask), border=0)
    assert (out.data == data).all() and (newmask.data == mask).all()


# --- masks ----------------------------------------------------------------------------

def test_map_mask_soft_covers_the_uv_quad():
    mask = projection.mapMaskSoft((16, 16), quad_mesh())
    _, covered = expected_shading(QUAD_UVS, 16, 16, SHADES)
    assert 60 < covered.sum() < 200
    assert mask.data.shape == (16, 16, 4)
    assert (mask.data[covered] == [255, 255, 255, 255]).all()
    assert (mask.data[~covered] == [0, 0, 0, 255]).all()


def test_map_mask_soft_defaults_to_the_selected_human(app):
    app.selectedHuman = types.SimpleNamespace(mesh=quad_mesh())
    _, covered = expected_shading(QUAD_UVS, 16, 16, SHADES)
    assert (projection.mapMaskSoft((16, 16)).data[..., 0] == covered * 255).all()


def test_map_mask_uses_the_hardware_renderer(monkeypatch, app):
    mesh = quad_mesh()
    app.selectedHuman = types.SimpleNamespace(mesh=mesh)
    r = use_gl(monkeypatch, Renderer(mask=np.full((3, 2, 3), 7)))
    out = projection.mapMask((2, 3))
    assert out.data.shape == (3, 2, 3) and (out.data == 7).all()
    call, = r.calls
    assert call["dst"] == (2, 3) and call["vpp"] == 4 and call["color"] is None
    assert (call["verts"] == mesh.r_texco).all() and (call["index"] == mesh.index).all()
    assert call["clearColor"] == (0, 0, 0, 0)


def test_map_mask_falls_back_to_software_when_the_renderer_fails(monkeypatch):
    r = use_gl(monkeypatch, Renderer(error=RuntimeError("no FBO")))
    out = projection.mapMask((16, 16), quad_mesh())
    assert len(r.calls) == 1
    _, covered = expected_shading(QUAD_UVS, 16, 16, SHADES)
    assert (out.data[..., 0] == covered * 255).all()


# --- light maps -----------------------------------------------------------------------

def test_map_lighting_soft_shades_each_vertex_and_interpolates():
    mesh = quad_mesh()
    out = projection.mapLightingSoft(LIGHT, mesh, (16, 16), border=0)
    expected, covered = expected_shading(QUAD_UVS, 16, 16, SHADES)
    assert out.data.shape == (16, 16, 4)
    diff = np.abs(out.data[..., :3].astype(int) - expected)
    assert diff[covered].max() <= 1                       # float truncation
    assert (out.data[~covered] == 0).all()                # transparent black outside
    assert out.data[covered, 3].min() >= 254              # opaque inside
    assert out.data[..., 0].max() > 230 and out.data[covered, 0].min() < 170  # really a gradient
    # The vertex colours used for baking are reset to white afterwards
    assert (mesh.color == 255).all()


def test_map_lighting_soft_border_grows_into_the_neighbours(app):
    app.selectedHuman = types.SimpleNamespace(mesh=quad_mesh())
    flat = projection.mapLightingSoft(LIGHT, None, (16, 16), border=0).data
    grown = projection.mapLightingSoft(LIGHT, None, (16, 16), border=1).data
    covered = flat[..., 3] > 0
    ring = dilate(covered) & ~covered
    assert (grown[covered, :3] == flat[covered, :3]).all()
    assert (grown[ring, :3] > 0).all()
    assert (grown[~dilate(covered), :3] == 0).all()
    # Each ring pixel is the rounded mean of its covered neighbours
    y, x = np.argwhere(ring)[0]
    neigh = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if covered[y + dy, x + dx]]
    mean = np.mean([flat[p][:3] for p in neigh], axis=0)
    assert np.abs(grown[y, x, :3] - np.floor(mean + 0.5)).max() <= 1


def test_map_lighting_gl_passes_the_shading_to_the_renderer(monkeypatch):
    mesh = quad_mesh()
    data, mask = seam_case()
    r = use_gl(monkeypatch, Renderer(lightmap=data, mask=mask))
    out = projection.mapLightingGL(LIGHT, mesh, (6, 4), border=1)
    light, masking = r.calls
    assert light["dst"] == (6, 4) and light["vpp"] == 4
    assert light["clearColor"] == (0, 0, 0, 0)
    assert (light["index"] == mesh.index).all() and (light["verts"] == mesh.r_texco).all()
    # Each unwelded vertex sent to the GPU carries the shade of its own corner
    corners = [QUAD_COORDS.index(tuple(int(c) for c in v)) for v in mesh.r_coord]
    assert sorted(corners) == [0, 1, 2, 3]
    assert [tuple(c) for c in light["color"]] == [(SHADES[k],) * 3 + (255,) for k in corners]
    assert masking["color"] is None and masking["dst"] == (6, 4)
    check_seam_result(out.data)
    assert (mesh.color == 255).all()


def test_map_lighting_gl_defaults_to_the_selected_human(monkeypatch, app):
    mesh = quad_mesh()
    app.selectedHuman = types.SimpleNamespace(mesh=mesh)
    data, mask = seam_case()
    r = use_gl(monkeypatch, Renderer(lightmap=data, mask=mask))
    projection.mapLightingGL(res=(6, 4))
    # Default light position (-10.99, 20, 20): vertex 0 sees it at
    # 20/sqrt(10.99^2+20^2+20^2) = 20/30.344 = 0.6591, *256 = 168.7 -> 168
    assert tuple(r.calls[0]["color"][0]) == (168, 168, 168, 255)


def test_map_lighting_picks_software_without_render_to_texture():
    mesh = quad_mesh()
    expected, covered = expected_shading(QUAD_UVS, 16, 16, SHADES)
    out = projection.mapLighting(LIGHT, mesh, (16, 16), border=0)
    assert np.abs(out.data[..., :3].astype(int) - expected)[covered].max() <= 1


def test_map_lighting_prefers_the_hardware_renderer(monkeypatch):
    data, mask = seam_case()
    r = use_gl(monkeypatch, Renderer(lightmap=data, mask=mask))
    out = projection.mapLighting(LIGHT, quad_mesh(), (6, 4), border=1)
    assert len(r.calls) == 2
    check_seam_result(out.data)


def test_map_lighting_falls_back_to_software_when_the_renderer_fails(monkeypatch):
    r = use_gl(monkeypatch, Renderer(error=RuntimeError("no FBO")))
    out = projection.mapLighting(LIGHT, quad_mesh(), (16, 16), border=0)
    assert len(r.calls) == 2  # light map, then the mask inside the software fallback
    expected, covered = expected_shading(QUAD_UVS, 16, 16, SHADES)
    assert np.abs(out.data[..., :3].astype(int) - expected)[covered].max() <= 1


# --- scene light maps -----------------------------------------------------------------

class LightmapRecorder(object):
    """Stands in for projection.mapLighting: returns an RGB image per light."""

    def __init__(self, images):
        self.images = images  # lightpos tuple (rounded) -> HxWx3 array
        self.calls = []

    def __call__(self, lightpos, mesh, res, border):
        key = tuple(round(float(c), 6) + 0.0 for c in lightpos)
        self.calls.append((key, mesh, res, border))
        return image.Image(data=np.array(self.images[key], dtype=np.uint8))


def scene_object(diffuse=(1.0, 1.0, 1.0), ao=None, aoIntensity=1.0):
    material = types.SimpleNamespace(diffuseColor=Color(*diffuse),
                                     shaderConfig={"ambientOcclusion": ao is not None},
                                     aoMapTexture=ao, aoMapIntensity=aoIntensity)
    return types.SimpleNamespace(mesh=quad_mesh(), material=material)


def a_scene(ambience, lights=()):
    return types.SimpleNamespace(
        environment=types.SimpleNamespace(ambience=Color(*ambience)),
        lights=[types.SimpleNamespace(position=p, color=Color(*c)) for p, c in lights])


def test_scene_without_lights_is_ambience_times_diffuse(monkeypatch, app):
    monkeypatch.setattr(projection, "mapLighting", LightmapRecorder({}))
    app.selectedHuman = scene_object(diffuse=(1.0, 0.6, 0.2))
    out = projection.mapSceneLighting(a_scene((0.2, 0.2, 0.2)), res=(4, 3))
    # 0.2*255 = 51, 0.12*255 = 30.6 -> 31, 0.04*255 = 10.2 -> 10
    assert out.data.shape == (3, 4, 3)
    assert (out.data == [51, 31, 10]).all()


def test_scene_ambience_is_darkened_by_the_ao_map(monkeypatch):
    monkeypatch.setattr(projection, "mapLighting", LightmapRecorder({}))
    ao = image.Image(data=np.array([[[0], [255]], [[100], [255]]], dtype=np.uint8))
    obj = scene_object(diffuse=(1.0, 0.6, 0.2), ao=ao, aoIntensity=0.5)
    out = projection.mapSceneLighting(a_scene((0.2, 0.2, 0.2)), obj, res=(2, 2))
    # AO mixed half-way with white: 0 -> 128, 100 -> 178, 255 -> 255; then
    # (51, 31, 10) * that / 255, rounded
    assert out.data[0, 0].tolist() == [26, 16, 5]
    assert out.data[1, 0].tolist() == [36, 22, 7]
    assert out.data[0, 1].tolist() == [51, 31, 10] == out.data[1, 1].tolist()


def test_scene_ao_flag_without_a_map_is_ignored(monkeypatch):
    monkeypatch.setattr(projection, "mapLighting", LightmapRecorder({}))
    obj = scene_object(diffuse=(1.0, 0.6, 0.2))
    obj.material.shaderConfig["ambientOcclusion"] = True
    out = projection.mapSceneLighting(a_scene((0.2, 0.2, 0.2)), obj, res=(2, 2))
    assert (out.data == [51, 31, 10]).all()


def test_scene_one_light_adds_ambience_to_the_unlit_part(monkeypatch):
    rec = LightmapRecorder({(1.0, 2.0, 3.0): np.full((3, 4, 3), 200)})
    monkeypatch.setattr(projection, "mapLighting", rec)
    obj = scene_object()
    out = projection.mapSceneLighting(a_scene((0.2, 0.2, 0.2), [((1, 2, 3), (1.0, 0.5, 0.25))]),
                                      obj, res=(4, 3), border=2)
    # light = 200 * colour = (200, 100, 50); ambience 51;
    # result = light + (255 - light) * 51/255 = (211, 131, 91)
    assert (out.data == [211, 131, 91]).all()
    (pos, mesh, res, border), = rec.calls
    assert pos == (1.0, 2.0, 3.0) and mesh is obj.mesh and res == (4, 3) and border == 2


@pytest.mark.parametrize("rotation, moved", [
    ([0, 90, 0], (-3.0, 2.0, 1.0)),   # undo +90 about y: (x, y, z) -> (-z, y, x)
    ([90, 0, 0], (1.0, 3.0, -2.0)),   # undo +90 about x: (x, y, z) -> (x, z, -y)
    ([0, 0, 90], (2.0, -1.0, 3.0)),   # undo +90 about z: (x, y, z) -> (y, -x, z)
], ids=["y", "x", "z"])
def test_scene_lights_are_moved_into_the_model_frame(monkeypatch, app, rotation, moved):
    rec = LightmapRecorder({moved: np.full((1, 1, 3), 255)})
    monkeypatch.setattr(projection, "mapLighting", rec)
    app.modelCamera.getRotation = lambda: rotation
    projection.mapSceneLighting(a_scene((0.2, 0.2, 0.2), [((1, 2, 3), (1, 1, 1))]),
                                scene_object(), res=(1, 1))
    assert rec.calls[0][0] == moved


def test_scene_several_lights_are_summed_and_normalized(monkeypatch):
    left = np.zeros((2, 2, 3))
    left[:, 0] = 100
    rec = LightmapRecorder({(1.0, 0.0, 0.0): left, (0.0, 1.0, 0.0): np.full((2, 2, 3), 50)})
    monkeypatch.setattr(projection, "mapLighting", rec)
    scn = a_scene((0.2, 0.2, 0.2), [((1, 0, 0), (1, 1, 1)), ((0, 1, 0), (1, 1, 1))])
    out = projection.mapSceneLighting(scn, scene_object(), res=(2, 2))
    # Sum: 150 left, 50 right; normalized so the brightest is 255: 255 and
    # 85. Ambience: 255 + 0*0.2 = 255; 85 + 170*0.2 = 119
    assert (out.data[:, 0] == 255).all() and (out.data[:, 1] == 119).all()
    assert [c[0] for c in rec.calls] == [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]


def test_scene_lighting_with_the_software_renderer():
    out = projection.mapSceneLighting(a_scene((0.2, 0.2, 0.2), [(LIGHT, (1, 1, 1))]),
                                      scene_object(), res=(16, 16), border=0)
    assert out.data.shape == (16, 16, 3)


# --- UV map ---------------------------------------------------------------------------

def test_rasterize_hlines_draws_one_pixel_per_column():
    # From (0.5, 2.5) to (6.5, 5.5): y = 0.5*x + 2.5 at pixel centres x+0.5
    # gives rows 2,3,3,4,4,5,5 for columns 0..6 (row 0 = bottom of the image)
    for edge in ([(0.5, 2.5), (6.5, 5.5)], [(6.5, 5.5), (0.5, 2.5)]):  # both directions
        dst = image.Image(data=np.zeros((8, 8, 3), dtype=np.uint8))
        e = np.array([edge], dtype=float)
        projection.rasterizeHLines(dst, e, e[:, 1] - e[:, 0])
        expected = np.zeros((8, 8), dtype=bool)
        for x, y in enumerate([2, 3, 3, 4, 4, 5, 5]):
            expected[7 - y, x] = True
        assert ((dst.data == 255).all(axis=-1) == expected).all()
        assert ((dst.data == 0).all(axis=-1) == ~expected).all()


def test_rasterize_vlines_draws_one_pixel_per_row():
    # From (2.5, 0.5) to (5.5, 6.5): x = 0.5*y + 2.25 at y+0.5 gives columns
    # 2,3,3,4,4,5,5 for rows 0..6
    for edge in ([(2.5, 0.5), (5.5, 6.5)], [(5.5, 6.5), (2.5, 0.5)]):
        dst = image.Image(data=np.zeros((8, 8, 3), dtype=np.uint8))
        e = np.array([edge], dtype=float)
        projection.rasterizeVLines(dst, e, e[:, 1] - e[:, 0])
        expected = np.zeros((8, 8), dtype=bool)
        for y, x in enumerate([2, 3, 3, 4, 4, 5, 5]):
            expected[7 - y, x] = True
        assert ((dst.data == 255).all(axis=-1) == expected).all()


SQUARE_UVS = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]


def square_outline():
    """UV square 0.25..0.75 on the 2048 map: u,v = 0.25 -> 512, 0.75 -> 1536.
    Bottom edge at image row 2047-512, top at 2047-1536, sides in columns 512
    and 1536; every edge covers the 1024 pixels from 512 to 1535."""
    expected = np.zeros((2048, 2048), dtype=bool)
    expected[1535, 512:1536] = True
    expected[511, 512:1536] = True
    expected[512:1536, 512] = True
    expected[512:1536, 1536] = True
    return expected


def test_map_uv_soft_draws_the_uv_edges():
    out = projection.mapUVSoft(quad_mesh(SQUARE_UVS))
    assert out.data.shape == (2048, 2048, 3)
    lit = (out.data == 255).all(axis=-1)
    assert (lit == square_outline()).all()
    assert ((out.data == 0).all(axis=-1) == ~lit).all()


def test_map_uv_soft_with_uvs_on_the_texture_border():
    out = projection.mapUVSoft(quad_mesh([(0, 0), (1, 0), (1, 1), (0, 1)]))
    lit = (out.data == 255).all(axis=-1)
    assert lit[0].all() and lit[:, -1].all()


class FakeTexture(object):
    def __init__(self, size=None, components=4):
        self.size, self.components = size, components


def test_map_uv_gl_renders_each_edge_once(monkeypatch):
    # Two quads sharing the edge 1-2
    mesh = module3d.Object3D("pair", 4)
    mesh.setCoords([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)])
    mesh.setUVs([(0, 0), (0.5, 0), (0.5, 1), (0, 1), (1, 0), (1, 1)])
    mesh.setFaces([(0, 1, 2, 3), (1, 4, 5, 2)], [(0, 1, 2, 3), (1, 4, 5, 2)])
    mesh.updateIndexBuffer()
    monkeypatch.setattr(mh, "Texture", FakeTexture)
    r = use_gl(monkeypatch, Renderer(mask=np.full((2, 2, 4), 9)))
    out = projection.mapUVGL(mesh)
    assert out.data.shape == (2, 2, 3) and (out.data == 9).all()  # alpha dropped
    call, = r.calls
    assert isinstance(call["dst"], FakeTexture)
    assert call["dst"].size == (2048, 2048) and call["dst"].components == 3
    assert call["vpp"] == 2 and call["clearColor"] == (0, 0, 0, 255)
    assert (call["verts"] == mesh.r_texco).all()
    # Every side of every rendered quad, undirected, exactly once
    sides = {frozenset((int(q[i]), int(q[(i + 1) % 4]))) for q in mesh.index for i in range(4)}
    assert len(sides) == 7
    edges = call["index"]
    assert edges.dtype == np.uint32 and edges.flags["C_CONTIGUOUS"]
    assert sorted(map(frozenset, edges.tolist()), key=sorted) == sorted(sides, key=sorted)


def test_map_uv_dispatches_to_hardware_or_software(monkeypatch):
    mesh = quad_mesh(SQUARE_UVS)
    soft = projection.mapUV(mesh)
    assert ((soft.data == 255).all(axis=-1) == square_outline()).all()

    monkeypatch.setattr(mh, "Texture", FakeTexture)
    r = use_gl(monkeypatch, Renderer(mask=np.full((2, 2, 3), 9)))
    assert projection.mapUV(mesh).data.shape == (2, 2, 3) and len(r.calls) == 1

    use_gl(monkeypatch, Renderer(error=RuntimeError("no FBO")))
    fallback = projection.mapUV(mesh)
    assert ((fallback.data == 255).all(axis=-1) == square_outline()).all()
