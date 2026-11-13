"""lib/object3d.py: the render-side view of a mesh that glmodule draws.

Object3D only forwards to its module3d mesh (the parent) and to the
gui3d.Object holding it; the tests give each source attribute a distinct
value so a property that reads the wrong one fails. GL is never touched:
glmodule's draw/pick and shader support are replaced by recorders."""

import types

import numpy as np
import pytest

import glmodule
import object3d
import shader
from core import G


class FakeObject(object):
    """Stands in for gui3d.Object; counts how often its shader is fetched."""

    def __init__(self, shader_path=None, shader_obj=None, error=None):
        self.shader = shader_path
        self.material = types.SimpleNamespace(shaderChanged=1000.0)
        self._shader_obj, self._error = shader_obj, error
        self.fetches = 0

    @property
    def shaderObj(self):
        self.fetches += 1
        if self._error:
            raise self._error
        return self._shader_obj


def make(obj=None, **parent_attrs):
    parent = types.SimpleNamespace(object=obj if obj is not None else FakeObject(), **parent_attrs)
    return object3d.Object3D(parent), parent


@pytest.fixture
def shaders(monkeypatch):
    def set_supported(flag):
        monkeypatch.setattr(shader.Shader, "supported", staticmethod(lambda: flag))
    set_supported(True)
    return set_supported


@pytest.fixture
def logged(monkeypatch):
    out = {"error": [], "warning": [], "debug": []}
    for level in out:
        monkeypatch.setattr(object3d.log, level,
                            lambda msg, *args, _l=level, **kw: out[_l].append(msg % args if args else msg))
    return out


# --- forwarded properties -------------------------------------------------------------

PARENT_PROPS = [("name", "name"), ("verts", "r_coord"), ("norms", "r_vnorm"),
                ("color", "r_color"), ("color_diff", "r_color_diff"), ("UVs", "r_texco"),
                ("tangents", "r_vtang"), ("primitives", "index"), ("groups", "grpix"),
                ("vertsPerPrimitive", "vertsPerPrimitive"), ("visibility", "visibility"),
                ("cameraMode", "cameraMode"), ("pickable", "pickable"),
                ("nTransparentPrimitives", "transparentPrimitives"), ("priority", "priority")]

OBJECT_PROPS = ["shadeless", "depthless", "shaderParameters", "excludeFromProduction", "solid",
                "alphaToCoverage", "transform", "lockRotation", "x", "y", "z", "rx", "ry", "rz",
                "sx", "sy", "sz", "material", "cull", "texture"]


def test_properties_read_their_own_parent_attribute():
    values = {src: ("parent", src) for _, src in PARENT_PROPS}
    o, parent = make(**values)
    for prop, src in PARENT_PROPS:
        assert getattr(o, prop) == ("parent", src), prop


def test_properties_read_their_own_object_attribute():
    obj = FakeObject()
    for name in OBJECT_PROPS:
        setattr(obj, name, ("object", name))
    o, _ = make(obj)
    assert o.object is obj
    for name in OBJECT_PROPS:
        assert getattr(o, name) == ("object", name), name


def test_derived_properties():
    obj = FakeObject()
    obj.loc, obj.rot, obj.scale = np.array([1.0, 2, 3]), np.array([10.0, 20, 30]), np.array([2.0, 2, 2])
    obj.shaderConfig = {"vertexColors": "vc", "diffuse": "tex"}
    o, parent = make(obj, index=np.zeros((5, 4)), hasUVs=lambda: "uv?", name="body",
                     _faceGroups=[types.SimpleNamespace(colorID=(1, 2, 3), color=(9, 9, 9, 9))] * 2)
    assert o.nPrimitives == 5 and o.hasUVs == "uv?"
    assert o.translation.tolist() == [1, 2, 3] and o.rotation.tolist() == [10, 20, 30]
    assert o.scale.tolist() == [2, 2, 2]
    assert o.useVertexColors == "vc" and o.isTextured == "tex"
    assert o.clrid(1) == (1, 2, 3) and o.gcolor(1) == (9, 9, 9, 9)
    assert str(o) == "<Object3D body>"


def test_shader_changed_tracks_the_material_stamp():
    o, parent = make()
    material = parent.object.material
    assert o.shaderChanged  # never synchronised yet (-1)
    o.shaderChanged = False
    assert not o.shaderChanged and o.shader_mtime == 1000.0
    material.shaderChanged = 2000.0  # the material marks its shader dirty
    assert o.shaderChanged
    o.shaderChanged = False
    o.shaderChanged = True
    assert o.shader_mtime == -1 and o.shaderChanged


# --- shader object --------------------------------------------------------------------

def test_no_shader_without_glsl_support(shaders):
    shaders(False)
    o, parent = make(FakeObject("data/shaders/glsl/x", types.SimpleNamespace(shaderId=42)))
    assert o.shaderObj is None and o.shader == 0
    assert parent.object.fetches == 0


def test_no_shader_path_means_fixed_function(shaders):
    o, parent = make(FakeObject(None))
    assert o.shaderObj is None and o.shader == 0
    assert parent.object.fetches == 0
    assert not o.shaderChanged  # marked synchronised


def test_shader_is_fetched_once_and_cached(shaders):
    prog = types.SimpleNamespace(shaderId=42)
    o, parent = make(FakeObject("glsl/litsphere", prog))
    assert o.shaderObj is prog and o.shader == 42
    assert parent.object.fetches == 1
    o.shaderObj, o.shaderObj
    assert parent.object.fetches == 1


def test_shader_is_fetched_again_when_the_material_changes_it(shaders):
    o, parent = make(FakeObject("glsl/litsphere", types.SimpleNamespace(shaderId=42)))
    o.shaderObj
    parent.object.material.shaderChanged = 2000.0
    o.shaderObj
    assert parent.object.fetches == 2
    o.shaderObj
    assert parent.object.fetches == 2


def test_shader_is_fetched_again_when_the_path_changes(shaders):
    other = types.SimpleNamespace(shaderId=7)
    o, parent = make(FakeObject("glsl/a", types.SimpleNamespace(shaderId=42)))
    assert o.shader == 42
    parent.object.shader, parent.object._shader_obj = "glsl/b", other
    assert o.shaderObj is other and o.shader == 7


def test_broken_shader_falls_back_and_is_not_retried(shaders, logged):
    o, parent = make(FakeObject("glsl/broken", error=RuntimeError("compile error")))
    assert o.shaderObj is None and o.shader == 0
    assert parent.object.fetches == 1
    assert len(logged["error"]) == 1
    assert logged["warning"] == ["Failed to initialize shader (glsl/broken), falling back to fixed function shading."]
    parent.object.material.shaderChanged = 2000.0  # even if the material changes...
    assert o.shaderObj is None and parent.object.fetches == 1  # ...a failed path is not retried
    # ...until the path changes
    parent.object.shader, parent.object._error = "glsl/fixed", None
    parent.object._shader_obj = types.SimpleNamespace(shaderId=5)
    assert o.shader == 5


# --- drawing --------------------------------------------------------------------------

def test_draw_and_pick_hand_the_object_to_glmodule(monkeypatch):
    calls = []
    monkeypatch.setattr(glmodule, "drawMesh", lambda *a, **k: calls.append(("draw", a, k)) or "drawn")
    monkeypatch.setattr(glmodule, "pickMesh", lambda *a, **k: calls.append(("pick", a, k)) or "picked")
    o, _ = make()
    assert o.draw(1, flag=True) == "drawn" and o.pick() == "picked"
    assert calls == [("draw", (o, 1), {"flag": True}), ("pick", (o,), {})]


def quads(positions, axis):
    """One unit quad per position, perpendicular to `axis`, centred there."""
    corners = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    verts = []
    for p in positions:
        for a, b in corners:
            v = [a, b]
            v.insert(axis, p)
            verts.append(v)
    index = np.arange(4 * len(positions), dtype=np.uint32).reshape(-1, 4)
    return np.array(verts, dtype=np.float32), index


def sorted_mesh(monkeypatch, positions, axis, transparent, rot=(0.0, 0.0, 0.0)):
    monkeypatch.setattr(G, "cameras", [types.SimpleNamespace(eye=(0.0, 0.0, 10.0))])
    verts, index = quads(positions, axis)
    obj = FakeObject()
    obj.loc, (obj.rx, obj.ry, obj.rz) = np.zeros(3), rot
    o, parent = make(obj, r_coord=verts, index=index, transparentPrimitives=transparent)
    original = index.copy()
    o.sortFaces()
    return parent.index, original


def test_sort_faces_draws_the_farthest_transparent_face_first(monkeypatch):
    # Quads at z = 7, 1 (opaque) and -4, 3, 0 (transparent); camera at z=10.
    # Squared distances of the transparent ones: 196.5, 49.5, 100.5
    index, original = sorted_mesh(monkeypatch, [7, 1, -4, 3, 0], axis=2, transparent=3)
    assert (index[:2] == original[:2]).all()  # opaque part untouched
    assert (index[2:] == original[[2, 4, 3]]).all()


def test_sort_faces_undoes_the_model_rotation(monkeypatch):
    # Model turned 90 degrees about y: in model space the camera (0, 0, 10)
    # sits at (-10, 0, 0). Quads at x = -6, 2, 5 are 4, 12, 15 away.
    index, original = sorted_mesh(monkeypatch, [-6, 2, 5], axis=0, transparent=3, rot=(0.0, 90.0, 0.0))
    assert (index == original[[2, 1, 0]]).all()


def test_sort_faces_without_transparent_faces_does_nothing(monkeypatch):
    index, original = sorted_mesh(monkeypatch, [-4, 3, 0], axis=2, transparent=0)
    assert (index == original).all()


# --- scene membership -----------------------------------------------------------------

def test_attach_and_detach(monkeypatch, logged):
    monkeypatch.setattr(G, "world", [])
    mesh = types.SimpleNamespace(object3d=None)
    object3d.Object3D.attach(mesh)
    attached = mesh.object3d
    assert isinstance(attached, object3d.Object3D) and attached.parent is mesh
    assert G.world == [attached]
    object3d.Object3D.attach(mesh)  # already attached: no second entry
    assert G.world == [attached] and mesh.object3d is attached
    assert logged["debug"] == ["mesh is already attached"]
    object3d.Object3D.detach(mesh)
    assert G.world == [] and mesh.object3d is None
    object3d.Object3D.detach(mesh)
    assert logged["debug"][-1] == "mesh is not attached"
