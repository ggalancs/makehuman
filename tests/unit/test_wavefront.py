"""shared/wavefront.py: Wavefront OBJ import and export."""

import os
import sys
import types

import numpy as np
import pytest

import guicommon
import material
import wavefront


@pytest.fixture(autouse=True)
def no_opengl_shaders(monkeypatch):
    """Unit tests have no OpenGL context. If another test file imported the real shader
    module, materials would try to compile GLSL (material.getShaderObj) and segfault."""
    shader = sys.modules.get("shader")
    if shader is not None and hasattr(shader, "Shader"):
        monkeypatch.setattr(shader.Shader, "_supported", False)


QUADS_AND_TRI = """\
# two quads and a triangle, with UVs and groups
o testobj
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
v 2 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
g left
usemtl skin
f 1/1 2/2 3/3 4/4
g right
f 2/1 5/2 6/3 3/4
f 2/1 6/3 3/4
"""

TRIS_NO_UV = """\
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
f 1 2 3
f 2//1 4//1 3//1
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def quad_mesh(tmp_path):
    return wavefront.loadObjFile(write(tmp_path, "quads.obj", QUADS_AND_TRI))


def attach(mesh):
    """Exporting reads mesh.material, which lives on the scene object (held weakly)."""
    obj = guicommon.Object(mesh)
    return obj


def face_group_names(mesh):
    return [fg.name for fg in mesh.faceGroups]


def test_load_quads_triangles_uvs_and_groups(quad_mesh):
    m = quad_mesh
    assert m.name == "testobj"  # from the 'o' line
    assert m.getVertexCount() == 6 and m.getFaceCount() == 3
    np.testing.assert_allclose(m.coord[4], [2, 0, 0])
    # A triangle is stored as a quad whose last vertex repeats the first
    assert m.fvert.tolist() == [[0, 1, 2, 3], [1, 4, 5, 2], [1, 5, 2, 1]]
    assert m.has_uv
    assert m.fuvs.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 3, 0]]
    np.testing.assert_allclose(m.texco, [[0, 0], [1, 0], [1, 1], [0, 1]])
    assert face_group_names(m) == ["left", "right"]
    assert m.group.tolist() == [0, 1, 1]
    assert m.vertsPerFaceForExport == 4


def test_load_triangles_without_uvs_go_to_dummy_group(tmp_path):
    m = wavefront.loadObjFile(write(tmp_path, "tris.obj", TRIS_NO_UV))
    assert m.name == "tris"  # file name when there is no 'o' line
    assert not m.has_uv
    assert m.fvert.tolist() == [[0, 1, 2, 0], [1, 3, 2, 1]]  # 'v//vn' carries no UV
    assert face_group_names(m) == ["default-dummy-group"]
    assert m.vertsPerFaceForExport == 3


def test_load_into_existing_object_reuses_it(tmp_path):
    import module3d
    target = module3d.Object3D("given")
    result = wavefront.loadObjFile(write(tmp_path, "t.obj", TRIS_NO_UV), target)
    assert result is target and target.getFaceCount() == 2


def test_loose_vertices_are_rejected(tmp_path):
    path = write(tmp_path, "loose.obj", TRIS_NO_UV + "v 9 9 9\n")
    with pytest.raises(RuntimeError, match=r"loose vertices.*\[4\]"):
        wavefront.loadObjFile(path)


def test_export_round_trip_with_normals_and_mtl(tmp_path, quad_mesh):
    keep = attach(quad_mesh)
    keep.material.diffuseColor = (0.5, 0.25, 1.0)
    out = str(tmp_path / "out.obj")
    wavefront.writeObjFile(out, quad_mesh)

    text = open(out, encoding="utf-8").read()
    assert text.startswith("# MakeHuman exported OBJ")
    assert "mtllib out.mtl" in text
    assert text.count("\nvn ") == 6
    assert "f 1/1/1 2/2/2 3/3/3 4/4/4" in text

    mtl = open(str(tmp_path / "out.mtl"), encoding="utf-8").read()
    assert "newmtl %s" % keep.material.name in mtl
    assert "Kd 0.5 0.25 1" in mtl and "d 1" in mtl  # no texture: alpha is the opacity

    back = wavefront.loadObjFile(out)
    np.testing.assert_allclose(back.coord, quad_mesh.coord, atol=1e-4)
    assert back.fvert.tolist() == quad_mesh.fvert.tolist()
    np.testing.assert_allclose(back.texco, quad_mesh.texco, atol=1e-6)
    assert face_group_names(back) == [quad_mesh.name]  # one group per exported mesh


def test_export_filters_masked_faces_and_unused_vertices(tmp_path, quad_mesh):
    keep = attach(quad_mesh)
    quad_mesh.changeFaceMask(np.array([True, False, False]))
    out = str(tmp_path / "masked.obj")
    wavefront.writeObjFile(out, [quad_mesh], writeMTL=False)
    back = wavefront.loadObjFile(out)
    assert back.getFaceCount() == 1 and back.getVertexCount() == 4
    np.testing.assert_allclose(back.coord, quad_mesh.coord[:4], atol=1e-4)
    assert not os.path.exists(str(tmp_path / "masked.mtl"))
    assert keep


def test_export_unfiltered_keeps_vertices_but_skips_masked_faces(tmp_path, quad_mesh):
    keep = attach(quad_mesh)
    quad_mesh.changeFaceMask(np.array([True, False, True]))
    out = str(tmp_path / "unfiltered.obj")
    wavefront.writeObjFile(out, quad_mesh, writeMTL=False, filterMaskedFaces=False)
    text = open(out, encoding="utf-8").read()
    assert text.count("\nv ") == 6
    assert text.count("\nf ") == 2
    assert keep


def test_export_with_config_scale_offset_and_no_normals(tmp_path, quad_mesh):
    keep = attach(quad_mesh)
    config = types.SimpleNamespace(scale=10.0, feetOnGround=True, offset=np.array([0.0, 1.0, 0.0]),
                                   useNormals=False, copyTextureToNewLocation=lambda p: p)
    out = str(tmp_path / "scaled.obj")
    wavefront.writeObjFile(out, quad_mesh, writeMTL=False, config=config)
    text = open(out, encoding="utf-8").read()
    assert "vn " not in text
    assert "f 1/1 2/2 3/3 4/4" in text
    back = wavefront.loadObjFile(out)
    np.testing.assert_allclose(back.coord, quad_mesh.coord * 10 + [0, 1, 0], atol=1e-4)
    assert keep


class FlatMesh(object):
    """Duck-typed mesh for the writer's UV-less branches.

    A real Object3D cannot reach them: clone() hands fuvs to setFaces()
    unconditionally, so every clone reports has_uv (see the xfail below).
    """
    def __init__(self, name, coord, fvert, mask=None):
        self.name = name
        self.coord = np.asarray(coord, dtype=np.float32)
        self.vnorm = np.tile(np.float32([0, 0, 1]), (len(coord), 1))
        self.fvert = np.asarray(fvert)
        self.fuvs = np.zeros_like(self.fvert)
        self.texco = np.zeros((0, 2), dtype=np.float32)
        self.has_uv = False
        self.face_mask = np.ones(len(fvert), bool) if mask is None else np.asarray(mask)
        self.vertsPerFaceForExport = 3
        self.material = material.Material(name + "_mat")

    def clone(self, scale=1.0, filterMaskedVerts=False):
        return self


@pytest.mark.parametrize("use_normals, face_lines", [
    (True, ["f 1//1 2//2 3//3", "f 5//5 6//6 7//7"]),
    (False, ["f 1 2 3", "f 5 6 7"]),
])
def test_export_meshes_without_uvs(tmp_path, use_normals, face_lines):
    tri = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    a = FlatMesh("a", tri, [[0, 1, 2], [1, 3, 2]], mask=[True, False])
    b = FlatMesh("b", tri, [[0, 1, 2]])
    config = types.SimpleNamespace(scale=1.0, feetOnGround=False, useNormals=use_normals)
    out = tmp_path / "nouv.obj"
    with open(str(out), "w", encoding="utf-8") as fp:  # an open file is accepted too
        wavefront.writeObjFile(fp, [a, b], writeMTL=False, config=config)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if line.startswith("f ")] == face_lines  # masked face skipped
    assert not any(line.startswith("vt ") for line in lines)
    assert sum(line.startswith("vn ") for line in lines) == (8 if use_normals else 0)
    assert "g a" in lines and "usemtl b_mat" in lines


def test_export_two_meshes_offsets_indices(tmp_path, quad_mesh):
    other = wavefront.loadObjFile(write(tmp_path, "quads2.obj", QUADS_AND_TRI))
    keep = (attach(quad_mesh), attach(other))
    other.changeFaceMask(np.array([True, True, False]))
    config = types.SimpleNamespace(scale=1.0, feetOnGround=False, useNormals=False)
    out = str(tmp_path / "two.obj")
    wavefront.writeObjFile(out, [quad_mesh, other], writeMTL=False, config=config,
                           filterMaskedFaces=False)
    lines = open(out, encoding="utf-8").read().splitlines()
    faces = [line for line in lines if line.startswith("f ")]
    assert len(faces) == 5  # the masked face of the second mesh is skipped
    # vertices and UVs of the second mesh are numbered after the 6 verts / 4 UVs of the first
    assert faces[3] == "f 7/5 8/6 9/7 10/8"
    back = wavefront.loadObjFile(out)
    assert back.getVertexCount() == 12
    assert keep


def test_export_real_mesh_without_uvs(tmp_path):
    mesh = wavefront.loadObjFile(write(tmp_path, "tris.obj", TRIS_NO_UV))
    keep = attach(mesh)
    out = str(tmp_path / "nouv.obj")
    wavefront.writeObjFile(out, mesh, writeMTL=False, filterMaskedFaces=False)
    assert "f 1//1 2//2 3//3" in open(out, encoding="utf-8").read()
    assert keep


class FakeTexConfig(object):
    def __init__(self):
        self.copied = []

    def copyTextureToNewLocation(self, path):
        self.copied.append(path)
        return "textures/" + os.path.basename(path)


def test_write_material_with_textures_uses_path_config():
    mat = material.Material("cloth")
    mat._diffuseTexture = "/abs/diffuse.png"
    mat._specularMapTexture = "/abs/spec.png"
    mat._normalMapTexture = "/abs/normal.png"
    mat.specularColor = (0.1, 0.2, 0.3)
    conf = FakeTexConfig()

    import io
    buf = io.StringIO()
    wavefront.writeMaterial(buf, mat, conf)
    lines = buf.getvalue().splitlines()
    assert "newmtl cloth" in lines
    assert "d 0" in lines  # textured: alpha forced to 0 for Blender
    assert "Ks 0.1 0.2 0.3" in lines
    assert "map_Kd textures/diffuse.png" in lines
    assert "map_D textures/diffuse.png" in lines
    assert "map_Ks textures/spec.png" in lines
    assert not any(line.startswith("map_Disp") for line in lines)
    # the normal map is copied along although no MTL key references it
    assert conf.copied == ["/abs/diffuse.png", "/abs/diffuse.png", "/abs/spec.png", "/abs/normal.png"]


def test_write_texture_without_config_writes_path_verbatim():
    import io
    buf = io.StringIO()
    wavefront.writeTexture(buf, "map_Kd", "some/file.png")
    wavefront.writeTexture(buf, "map_Ks", None)
    assert buf.getvalue() == "map_Kd some/file.png\n"


def test_write_material_with_normal_map_and_no_config():
    import io
    mat = material.Material("n")
    mat._normalMapTexture = "/abs/normal.png"
    buf = io.StringIO()
    wavefront.writeMaterial(buf, mat)
    assert "newmtl n" in buf.getvalue()


def test_export_to_file_object_with_default_mtl(tmp_path, quad_mesh):
    keep = attach(quad_mesh)
    with open(str(tmp_path / "f.obj"), "w", encoding="utf-8") as fp:
        wavefront.writeObjFile(fp, quad_mesh)
    assert keep
