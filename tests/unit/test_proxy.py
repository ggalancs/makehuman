"""shared/proxy.py: .mhclo/.proxy parsing, compiled .mhpxy files, fitting and weights."""

import logging
import os
import sys
import shutil

import numpy as np
import pytest

import animation
import files3d
import getpath
import human
import progress
import proxy
import skeleton
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


FEDORA = "data/clothes/fedora01/fedora01.mhclo"

SYNTHETIC = """\
# This is a test proxy
# license AGPL3
name My cloth
uuid 1234-abcd
description A test cloth
tag Hat
tag Summer Wear
version 111
max_pole 5
special_pose foot heels
obj_file thing.obj
material thing.mhmat
vertexboneweights_file thing.jsonw
backface_culling False
transparent True
x_scale 0 1 2.0
y_scale 2 3 4.0
z_scale 4 5 0.5
basemesh hm08
shapekey something
unknownkey 1
weights bone1
10 0.5
verts 0
100
101 102 103 0.5 0.25 0.25
104 105 106 0.2 0.3 0.5 0.1 0.2 0.3
delete_verts
1 - 3 7
"""


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
def user_data(tmp_path, monkeypatch):
    """Per-user data folder in tmp_path (compiled proxies are only written there)."""
    monkeypatch.setenv("MH_HOME_LOCATION", str(tmp_path))
    data = getpath.getDataPath()
    os.makedirs(data)
    # getJailedPath() captured the user data folder when getpath was imported
    monkeypatch.setattr(getpath.getJailedPath, "__defaults__", ([data, getpath.getSysDataPath()],))
    return data


def write_asset(folder, text=SYNTHETIC, name="thing.mhclo"):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "thing.mhmat"), "w", encoding="utf-8") as f:
        f.write("name thingmat\ndiffuseColor 0.1 0.2 0.3\n")
    animation.VertexBoneWeights({"root": [(0, 1.0), (1, 1.0), (2, 1.0)]}).toFile(os.path.join(folder, "thing.jsonw"))
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def same_tmatrix(a, b):
    for attr in ("scaleData", "shearData", "lShearData", "rShearData"):
        x, y = getattr(a, attr), getattr(b, attr)
        if x is None or y is None:
            assert x is y, attr
            continue
        assert len(x) == len(y) == 3, attr
        for ex, ey in zip(x, y):
            if ex is None or ey is None:
                assert ex is ey, attr
            else:
                np.testing.assert_allclose(np.array(ex, dtype=float), np.array(ey, dtype=float), err_msg=attr)


# --- Text proxies ------------------------------------------------------------

def test_parse_text_proxy(mh, tmp_path, caplog):
    path = write_asset(str(tmp_path / "asset"))
    p = proxy.loadTextProxy(mh, path, "Clothes")
    folder = os.path.realpath(str(tmp_path / "asset"))
    assert (p.name, p.uuid, p.description) == ("My cloth", "1234-abcd", "A test cloth")
    assert p.license.license == "AGPL3"  # read from the comment header
    assert p.tags == ["hat", "summer wear"] and p.version == 111
    assert p.z_depth == 50 and "does not specify a Z depth" in caplog.text
    assert p.max_pole == 10  # doubled for neighbour calculations
    assert p.special_pose == {"foot": "heels"}
    assert p.basemesh == "hm08" and p.type == "Clothes"
    assert p._obj_file == os.path.join(folder, "thing.obj")
    assert p.material.name == "thingmat" and p.material.diffuseColor.values == pytest.approx([0.1, 0.2, 0.3])
    assert p.material_file.endswith("thing.mhmat")
    assert p.hasCustomVertexWeights() and p.vertexBoneWeights_file.endswith("thing.jsonw")
    assert p.vertexBoneWeights.data["root"][0].tolist() == [0, 1, 2]
    assert p.tmatrix.scaleData == [(0, 1, 2.0), (2, 3, 4.0), (4, 5, 0.5)]
    assert p.ref_vIdxs.tolist() == [[100, 0, 1], [101, 102, 103], [104, 105, 106]]
    np.testing.assert_allclose(p.weights, [[1, 0, 0], [0.5, 0.25, 0.25], [0.2, 0.3, 0.5]])
    np.testing.assert_allclose(p.offsets, [[0, 0, 0], [0, 0, 0], [0.1, 0.2, 0.3]])
    assert np.flatnonzero(p.deleteVerts).tolist() == [1, 2, 3, 7]
    assert p.vertWeights[100] == [(0, 1)] and p.vertWeights[104] == [(2, 0.2)]
    for message in ('"backface_culling"', '"transparent"', '"shapekey" used in proxy file',
                    "Unknown keyword unknownkey"):
        assert message in caplog.text
    assert repr(p) == "<Proxy My cloth Clothes %s 1234-abcd>" % path
    assert p.getUuid() == "1234-abcd"


def test_fitting_uses_weights_offsets_and_scale(mh, tmp_path):
    p = proxy.loadTextProxy(mh, write_asset(str(tmp_path / "asset")))
    h = mh.getRestposeCoordinates()
    scale = np.diag([abs(h[0][0] - h[1][0]) / 2.0, abs(h[2][1] - h[3][1]) / 4.0, abs(h[4][2] - h[5][2]) / 0.5])
    np.testing.assert_allclose(p.tmatrix.getMatrix(h), scale)
    coords = p.getCoords()
    np.testing.assert_allclose(coords[0], h[100], atol=1e-6)
    np.testing.assert_allclose(coords[1], 0.5 * h[101] + 0.25 * h[102] + 0.25 * h[103], atol=1e-6)
    np.testing.assert_allclose(coords[2], 0.2 * h[104] + 0.3 * h[105] + 0.5 * h[106] + scale @ [0.1, 0.2, 0.3],
                               atol=1e-6)
    # fit_to_posed uses the current (possibly posed) mesh instead of the rest coordinates
    mh.meshData.changeCoords(mh.meshData.coord + [0, 1, 0])
    np.testing.assert_allclose(p.getCoords(fit_to_posed=True)[0], h[100] + [0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(p.getCoords()[0], h[100], atol=1e-6)


def test_update_moves_a_mesh_onto_the_fit(mh):
    p = proxy.loadProxy(mh, FEDORA)
    mesh = files3d.loadMesh(p.obj_file, maxFaces=p.max_pole)
    assert mesh.getVertexCount() == len(p.ref_vIdxs)
    p.update(mesh)
    np.testing.assert_allclose(mesh.coord, p.getCoords(), atol=1e-5)


def test_license_keyword_lines(mh, tmp_path):
    path = write_asset(str(tmp_path / "asset"), SYNTHETIC.replace("tag Hat\n", "author Jane Doe\ntag Hat\n"))
    assert proxy.loadTextProxy(mh, path).license.author == "Jane Doe"


def test_text_proxy_missing_file(mh, tmp_path, caplog):
    assert proxy.loadTextProxy(mh, str(tmp_path / "nope.mhclo")) is None
    assert "Cannot open" in caplog.text


def test_shear_keywords_and_matrix(mh, tmp_path):
    text = SYNTHETIC.replace("x_scale 0 1 2.0\ny_scale 2 3 4.0\nz_scale 4 5 0.5\n",
                             "shear_x 0 1 0.0 1.0\nshear_y 2 3 0.0 1.0\nshear_z 4 5 0.0 1.0\n"
                             "l_shear_x 0 1 0.0 2.0\nl_shear_y 2 3 0.0 2.0\nl_shear_z 4 5 0.0 2.0\n"
                             "r_shear_x 0 1 0.0 3.0\nr_shear_y 2 3 0.0 3.0\nr_shear_z 4 5 0.0 3.0\n")
    p = proxy.loadTextProxy(mh, write_asset(str(tmp_path / "asset"), text))
    t = p.tmatrix
    assert t.scaleData is None
    assert t.shearData == [(0, 1, 0.0, 1.0), (2, 3, 0.0, 1.0), (4, 5, 0.0, 1.0)]
    assert t.lShearData[2] == (4, 5, 0.0, 2.0) and t.rShearData[0] == (0, 1, 0.0, 3.0)
    # target box twice the source box on every axis -> uniform scale by 2
    h = np.zeros((6, 3))
    h[1] = h[3] = h[5] = 2.0
    np.testing.assert_allclose(t.getMatrix(h), 2 * np.eye(3), atol=1e-6)
    t.shearData = None
    np.testing.assert_allclose(t.getMatrix(h), np.eye(3), atol=1e-6)  # left box: source 0..2
    t.lShearData = None
    np.testing.assert_allclose(t.getMatrix(h), np.eye(3) * 2 / 3, atol=1e-6)  # right box: source 0..3
    t.rShearData = None
    assert t.getMatrix(h) is proxy.Unit3


def test_ref_vert_helpers(mh):
    weights = {}
    single = proxy.ProxyRefVert(mh).fromSingle(["7"], 0, weights)
    triple = proxy.ProxyRefVert(mh).fromTriple("1 2 3 0.2 0.3 0.5".split(), 1, weights)
    offset = proxy.ProxyRefVert(mh).fromTriple("1 2 3 1 0 0 0.5 0 0".split(), 2, weights)
    assert single.getWeights() == (1.0, 0.0, 0.0) and triple.getWeights() == (0.2, 0.3, 0.5)
    assert single._verts == (7, 0, 1) and offset._offset.tolist() == [0.5, 0, 0]
    assert weights[7] == [(0, 1)] and weights[1] == [(1, 0.2), (2, 1.0)]


@pytest.mark.xfail(strict=True, raises=IndexError,
                   reason="ProxyRefVert.getCoord indexes the coordinates with a tuple (a 3-D index, not "
                          "three rows) and dots rows with the weights the wrong way round (proxy.py:756); "
                          "it has no callers, Proxy.getCoords does the fitting")
def test_ref_vert_get_coord(mh):
    weights = {}
    triple = proxy.ProxyRefVert(mh).fromTriple("1 2 3 0.2 0.3 0.5".split(), 1, weights)
    offset = proxy.ProxyRefVert(mh).fromTriple("1 2 3 1 0 0 0.5 0 0".split(), 2, weights)
    h = mh.getRestposeCoordinates()
    np.testing.assert_allclose(triple.getCoord(np.eye(3)), 0.2 * h[1] + 0.3 * h[2] + 0.5 * h[3], atol=1e-6)
    np.testing.assert_allclose(offset.getCoord(2 * np.eye(3)), h[1] + [1.0, 0, 0], atol=1e-6)


def test_proxy_requires_a_human():
    with pytest.raises(RuntimeError, match="valid human"):
        proxy.Proxy("x.mhclo", "Clothes", None)


def test_proxy_without_file_and_seed_mesh(mh):
    p = proxy.Proxy("", "Proxymeshes", mh)
    assert p.mtime is None and p.getMesh() is None
    assert p.material_file is None and p.vertexBoneWeights_file is None
    assert p.getUuid() == p.name  # no uuid: the name identifies it
    assert p.getSeedMesh() is None  # human has no proxy mesh
    p.type = "Converter"
    assert p.getSeedMesh() is None
    p.type = "Clothes"
    with pytest.raises(NameError, match="Unknown proxy type"):
        p.getSeedMesh()


def test_obj_file_of_proxy_without_mesh(mh):
    assert proxy.Proxy("", "Clothes", mh).obj_file is None


def test_load_mesh_and_object(mh):
    p = proxy.loadProxy(mh, FEDORA)
    mesh, obj = p.loadMeshAndObject(mh)
    assert obj is p.object and obj.proxy is p
    assert obj.material.name == p.material.name  # the object keeps its own copy
    assert obj.material.diffuseTexture == p.material.diffuseTexture
    assert p.getMesh() is mesh and mesh.priority == p.z_depth
    assert mesh.getVertexCount() == len(p.ref_vIdxs)


def test_load_mesh_and_object_failure(mh, tmp_path, caplog):
    p = proxy.loadTextProxy(mh, write_asset(str(tmp_path / "asset")))  # thing.obj does not exist
    assert p.loadMeshAndObject(mh) == (False, False)
    assert "Failed to load" in caplog.text


# --- Compiled proxies --------------------------------------------------------

def test_compiled_and_text_versions_agree(mh, monkeypatch):
    compiled = proxy.loadProxy(mh, FEDORA)
    assert compiled.file.endswith(".mhpxy")
    text = proxy.loadTextProxy(mh, FEDORA)
    assert (compiled.name, compiled.uuid, compiled.z_depth, compiled.max_pole) == \
        (text.name, text.uuid, text.z_depth, text.max_pole)
    assert compiled.tags == set(text.tags) and compiled.material.name == text.material.name
    assert compiled.ref_vIdxs.tolist() == text.ref_vIdxs.tolist()
    np.testing.assert_allclose(compiled.weights, text.weights, atol=1e-6)
    np.testing.assert_allclose(compiled.offsets, text.offsets, atol=1e-6)
    same_tmatrix(compiled.tmatrix, text.tmatrix)
    np.testing.assert_allclose(compiled.getCoords(), text.getCoords(), atol=1e-5)
    assert os.path.basename(compiled.obj_file) == "fedora.obj"


def test_binary_round_trip(mh, user_data):
    folder = os.path.join(user_data, "clothes", "thing")
    original = proxy.loadTextProxy(mh, write_asset(folder))
    original.license.author = "Ñandú"
    npz = os.path.join(folder, "thing.mhpxy")
    proxy.saveBinaryProxy(original, npz)
    stored = np.load(npz)
    assert int(stored["num_refverts"]) == 3 and str(stored["obj_file"].tobytes(), "utf8") == "thing.obj"

    open(os.path.join(folder, "thing.obj"), "w").close()  # so that obj_file resolves
    back = proxy.loadBinaryProxy(npz, mh, "Clothes")
    for attr in ("name", "uuid", "description", "basemesh", "version", "z_depth", "max_pole", "special_pose"):
        assert getattr(back, attr) == getattr(original, attr), attr
    assert back.tags == set(original.tags) and back.license.author == "Ñandú"
    assert back.ref_vIdxs.tolist() == original.ref_vIdxs.tolist()
    np.testing.assert_allclose(back.weights, original.weights)
    np.testing.assert_allclose(back.offsets, original.offsets)
    assert back.deleteVerts.tolist() == original.deleteVerts.tolist()
    same_tmatrix(back.tmatrix, original.tmatrix)
    assert back.material.name == "thingmat"
    assert back.obj_file == getpath.canonicalPath(os.path.join(folder, "thing.obj"))
    assert back.vertexBoneWeights.data["root"][0].tolist() == [0, 1, 2]
    assert back.vertWeights[104] == [(2, pytest.approx(0.2))]  # reverse mapping rebuilt


def test_binary_exact_fit_is_stored_compactly(mh, user_data):
    folder = os.path.join(user_data, "clothes", "exact")
    text = "uuid x-1\nname exact\nz_depth -1\nobj_file thing.obj\nverts 0\n5\n6\n7\n"
    original = proxy.loadTextProxy(mh, write_asset(folder, text))
    original.z_depth = -1  # not stored: loading falls back to 50
    npz = os.path.join(folder, "exact.mhpxy")
    proxy.saveBinaryProxy(original, npz)
    stored = np.load(npz)
    assert int(stored["num_refverts"]) == 1 and stored["ref_vIdxs"].tolist() == [5, 6, 7]
    assert "z_depth" not in stored and "material_file" not in stored and "deleteVerts" not in stored
    back = proxy.loadBinaryProxy(npz, mh, "Clothes")
    assert back.ref_vIdxs.tolist() == [[5, 0, 0], [6, 0, 0], [7, 0, 0]]
    np.testing.assert_allclose(back.weights[:, 0], 1)
    assert back.z_depth == 50 and not back.hasCustomVertexWeights()
    np.testing.assert_allclose(back.getCoords(), mh.getRestposeCoordinates()[[5, 6, 7]], atol=1e-6)


def test_binary_proxy_uv_layers_are_loaded(mh, user_data):
    # Saving uvLayers is broken (see the xfail below); inject them into a compiled file instead
    folder = os.path.join(user_data, "clothes", "uvload")
    p = proxy.loadTextProxy(mh, write_asset(folder))
    npz = os.path.join(folder, "thing.mhpxy")
    proxy.saveBinaryProxy(p, npz)
    arrays = dict(np.load(npz))
    arrays["uvLayers_str"], arrays["uvLayers_idx"] = proxy._packStringList(["a.mhuv", "b.mhuv"])
    with open(npz, "wb") as f:
        np.savez_compressed(f, **arrays)
    assert proxy.loadBinaryProxy(npz, mh, "Clothes").uvLayers == {0: "a.mhuv", 1: "b.mhuv"}


def test_binary_proxy_with_uv_layers(mh, user_data):
    folder = os.path.join(user_data, "clothes", "uv")
    p = proxy.loadTextProxy(mh, write_asset(folder, SYNTHETIC + "uvLayer 1 alt.mhuv\n"))
    assert p.uvLayers == {1: os.path.join(os.path.realpath(folder), "alt.mhuv")}
    proxy.saveBinaryProxy(p, os.path.join(folder, "thing.mhpxy"))


def test_uv_layer_keyword_without_index(mh, tmp_path):
    p = proxy.loadTextProxy(mh, write_asset(str(tmp_path / "a"), SYNTHETIC + "uvLayer alt.mhuv\n"))
    assert p.uvLayers == {0: os.path.join(os.path.realpath(str(tmp_path / "a")), "alt.mhuv")}


def test_load_proxy_compiles_in_user_folder(mh, user_data, monkeypatch, caplog):
    path = write_asset(os.path.join(user_data, "clothes", "thing"))
    first = proxy.loadProxy(mh, path)
    npz = path[:-6] + ".mhpxy"
    assert first.file == path and os.path.isfile(npz)
    assert "compiled proxy file missing" in caplog.text

    def no_text(*args, **kwargs):
        raise AssertionError("text proxy parsed although an up-to-date .mhpxy exists")
    monkeypatch.setattr(proxy, "loadTextProxy", no_text)
    second = proxy.loadProxy(mh, path)
    assert second.file == npz and second.ref_vIdxs.tolist() == first.ref_vIdxs.tolist()


def test_load_proxy_rebuilds_out_of_date_file(mh, tmp_path, caplog):
    folder = str(tmp_path / "fedora")
    shutil.copytree(os.path.join(MH_DIR, "data", "clothes", "fedora01"), folder)
    mhclo = os.path.join(folder, "fedora01.mhclo")
    os.utime(os.path.join(folder, "fedora01.mhpxy"), (1000000, 1000000))
    p = proxy.loadProxy(mh, mhclo)
    assert p.file == mhclo  # parsed from text
    assert "out of date" in caplog.text
    assert os.path.getmtime(os.path.join(folder, "fedora01.mhpxy")) == 1000000  # not in the user folder


def test_load_proxy_failed_compile_leaves_no_file(mh, user_data, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    path = write_asset(os.path.join(user_data, "clothes", "uv"), SYNTHETIC + "uvLayer 1 alt.mhuv\n")

    def half_written(proxy_, npzpath):
        open(npzpath, "wb").close()
        raise OSError("disk full")

    monkeypatch.setattr(proxy, "saveBinaryProxy", half_written)
    p = proxy.loadProxy(mh, path)
    assert p is not None and p.uvLayers
    assert "unable to save compiled proxy" in caplog.text
    assert not os.path.exists(path[:-6] + ".mhpxy")


def test_load_proxy_with_uv_layers_compiles_and_reloads(mh, user_data, caplog):
    caplog.set_level(logging.DEBUG)
    path = write_asset(os.path.join(user_data, "clothes", "uv"), SYNTHETIC + "uvLayer 1 alt.mhuv\n")
    first = proxy.loadProxy(mh, path)
    assert os.path.isfile(path[:-6] + ".mhpxy")
    assert "unable to save compiled proxy" not in caplog.text
    again = proxy.loadProxy(mh, path)  # now from the compiled file
    assert "Problem loading binary proxy" not in caplog.text.split("Compiling binary proxy")[-1]
    # the compiled file keeps the layers in order but numbers them from 0 (nothing reads the numbers)
    assert list(first.uvLayers) == [1] and list(again.uvLayers) == [0]
    assert os.path.basename(again.uvLayers[0]) == "alt.mhuv"


def test_load_proxy_with_broken_file_returns_none(mh, tmp_path, caplog):
    path = write_asset(str(tmp_path / "bad"), "verts 0\n1 2 x 0.1 0.2 0.3\n")
    assert proxy.loadProxy(mh, path) is None
    assert "Unable to load proxy file" in caplog.text


def test_proxymesh_extension_and_special_files(mh):
    assert proxy.getAsciiFileExtension("Proxymeshes") == ".proxy"
    assert proxy.getAsciiFileExtension("Hair") == ".mhclo"
    p = proxy.loadProxy(mh, "data/proxymeshes/proxy741/proxy741.proxy", "Proxymeshes")
    assert p.type == "Proxymeshes" and p.file.endswith(".mhpxy")
    suit = proxy.loadProxy(mh, "data/clothes/female_casualsuit01/female_casualsuit01.mhclo")
    assert suit.deleteVerts.any() and suit.deleteVerts.shape == (mh.meshData.getVertexCount(),)


# --- Metadata ----------------------------------------------------------------

def test_peek_metadata_binary_and_text(mh):
    uuid, tags = proxy.peekMetadata("data/clothes/fedora01/fedora01.mhpxy", "Clothes")
    text_uuid, text_tags = proxy.peekMetadata(FEDORA)
    assert uuid == text_uuid == "566dbd52-71d1-4d76-a799-0474a5a384db"
    assert tags == text_tags and "makehuman™" in tags


def test_peek_metadata_prefers_newer_text(mh, tmp_path, caplog):
    folder = str(tmp_path / "fedora")
    shutil.copytree(os.path.join(MH_DIR, "data", "clothes", "fedora01"), folder)
    with open(os.path.join(folder, "fedora01.mhclo"), "a", encoding="utf-8") as f:
        f.write("\n")
    os.utime(os.path.join(folder, "fedora01.mhpxy"), (1000000, 1000000))
    uuid, tags = proxy.peekMetadata(os.path.join(folder, "fedora01.mhpxy"), "Clothes")
    assert uuid == "566dbd52-71d1-4d76-a799-0474a5a384db"
    assert "out of date" in caplog.text


def test_peek_metadata_stops_at_vertex_data(tmp_path):
    path = tmp_path / "t.mhclo"
    path.write_text("uuid abc\n\ntag One\nverts 0\ntag Never\n", encoding="utf-8")
    assert proxy.peekMetadata(str(path)) == ("abc", {"one"})


def test_peek_metadata_of_corrupt_binary(tmp_path, caplog):
    path = tmp_path / "broken.mhpxy"
    path.write_bytes(b"uuid not-a-zip\n")
    assert proxy.peekMetadata(str(path)) == ("not-a-zip", set())  # falls back to reading it as text
    assert "Problem loading metadata from binary proxy" in caplog.text


# --- TMatrix serialisation ---------------------------------------------------

def test_tmatrix_numpy_round_trip_with_prefix():
    t = proxy.TMatrix()
    t.scaleData = [(1, 2, 0.5), None, (5, 6, 1.5)]
    t.shearData = [(1, 2, 0.1, 0.2)] * 3
    t.lShearData = [(3, 4, 0.3, 0.4)] * 3
    t.rShearData = [(5, 6, 0.5, 0.6)] * 3
    store = {}
    t.toNumpyStruct(store, prefix="p")
    assert sorted(store) == ["p_tmat_lshear", "p_tmat_lshear_idx", "p_tmat_rshear", "p_tmat_rshear_idx",
                             "p_tmat_scale", "p_tmat_scale_idx", "p_tmat_shear", "p_tmat_shear_idx"]
    back = proxy.TMatrix()
    back.fromNumpyStruct(store, prefix="p")
    assert back.scaleData[1] is None  # NaN marks a missing axis
    same_tmatrix(back, t)
    empty = proxy.TMatrix()
    empty.fromNumpyStruct({})
    assert empty.scaleData is None and empty.shearData is None


def test_tmatrix_unpacks_truncated_arrays():
    t = proxy.TMatrix()
    t.fromNumpyStruct({"tmat_scale": np.float32([2.0]), "tmat_scale_idx": np.uint32([1, 2]),
                       "tmat_shear": np.float32([0.1, 0.2]), "tmat_shear_idx": np.uint32([3, 4])})
    assert t.scaleData == [(1, 2, 2.0), None, None]
    assert t.shearData == [(3, 4, pytest.approx(0.1), pytest.approx(0.2)), None, None]


def test_tmatrix_with_missing_shear_axis_packs_nan():
    t = proxy.TMatrix()
    t.shearData = [None, (1, 2, 0.1, 0.2), None]
    store = {}
    t.toNumpyStruct(store)
    assert np.isnan(store["tmat_shear"][:2]).all() and store["tmat_shear_idx"][:2].tolist() == [0, 0]


def test_verts_to_numpy():
    out = proxy.vertsToNumpy([np.array((1, 2, 3)), np.array((4, 5, 6))])
    assert out.dtype == np.float32 and out.tolist() == [[1, 4], [2, 5], [3, 6]]


# --- Masks, file names and weights -------------------------------------------

def test_transfer_vertex_mask(mh, tmp_path):
    p = proxy.loadTextProxy(mh, write_asset(str(tmp_path / "asset")))
    mask = np.ones(mh.meshData.getVertexCount(), dtype=bool)
    assert proxy.transferVertexMaskToProxy(mask, p).tolist() == [True, True, True]
    mask[[100, 101, 104, 105]] = False  # exact-fit vertex follows its reference
    assert proxy.transferVertexMaskToProxy(mask, p).tolist() == [False, True, False]


def test_file_name_with_extension_is_kept():
    assert proxy._getFileName("/a", "dir/file.obj", ".mhclo") == os.path.join("/a", "dir/file.obj")


def test_file_name_without_extension_gets_suffix():
    assert proxy._getFileName("/a", "thing", ".obj") == os.path.join("/a", "thing.obj")


def test_file_path_prefers_existing_alternative_extension(tmp_path):
    (tmp_path / "mesh.npz").write_bytes(b"")
    path = proxy._getFilePath("mesh.obj", str(tmp_path), [".npz", "obj"])
    assert path == getpath.formatPath(os.path.join(str(tmp_path), "mesh.obj"))
    assert proxy._getFilePath(None, str(tmp_path)) is None


def test_vertex_weights_follow_the_fit(mh):
    p = proxy.loadProxy(mh, FEDORA)
    body = mh.getVertexWeights()
    weights = p.getVertexWeights(body)
    assert weights.vertexCount == len(p.ref_vIdxs)
    totals = np.zeros(weights.vertexCount)
    for verts, w in weights.data.values():
        np.add.at(totals, verts, w)
    np.testing.assert_allclose(totals, 1, atol=1e-4)
    # an exact-fit vertex inherits the bones of its reference vertex
    exact = int(np.flatnonzero(p.weights[:, 0] == 1)[0]) if (p.weights[:, 0] == 1).any() else None
    if exact is not None:
        ref = int(p.ref_vIdxs[exact, 0])
        body_bones = {b for b, (v, _) in body.data.items() if ref in v}
        proxy_bones = {b for b, (v, _) in weights.data.items() if exact in v}
        assert proxy_bones == body_bones
    assert p.weightsCache is None


def test_vertex_weights_cache(mh, caplog):
    caplog.set_level(logging.DEBUG)
    p = proxy.loadProxy(mh, FEDORA)
    body = mh.getVertexWeights()
    skel = mh.getBaseSkeleton()
    first = p.getVertexWeights(body, skel, allowCache=True)
    assert p.weightsCache is first and p.cacheSkel is skel
    assert p.getVertexWeights(body, skel, allowCache=True) is first
    other = skeleton.load(getpath.getSysDataPath("rigs/cmu_mb.mhskel"), mh.meshData)
    third = p.getVertexWeights(body, other, allowCache=True)
    assert third is not first and "The skeleton is different" in caplog.text
    assert p.getVertexWeights(body, None, allowCache=True) is not third  # no skeleton: recalculated


def test_custom_vertex_weights_are_remapped_to_the_skeleton(mh):
    shoes = proxy.loadProxy(mh, "data/clothes/shoes02/shoes02.mhclo")
    assert shoes.hasCustomVertexWeights()
    base_w = shoes.getVertexWeights(mh.getVertexWeights())
    assert set(base_w.data) <= {b.name for b in mh.getBaseSkeleton().getBones()}
    game = skeleton.load(getpath.getSysDataPath("rigs/game_engine.mhskel"), mh.meshData)
    game_w = shoes.getVertexWeights(mh.getVertexWeights(), game)
    assert set(game_w.data) <= {b.name for b in game.getBones()}
    assert game_w.vertexCount == base_w.vertexCount


def test_reverse_mapping_is_rebuilt_from_arrays(mh, tmp_path):
    p = proxy.loadTextProxy(mh, write_asset(str(tmp_path / "asset")))
    expected = {k: list(v) for k, v in p.vertWeights.items()}
    p._reloadReverseMapping()
    got = {int(k): [(int(a), pytest.approx(float(b))) for a, b in v] for k, v in p.vertWeights.items()}
    # the text parser skipped the dummy refs of exact-fit vertices; the rebuild adds them with weight 0
    assert got[100] == [(0, 1.0)] and got[0] == [(0, 0.0)] and got[1] == [(0, 0.0)]
    for k in (101, 102, 103, 104, 105, 106):
        assert got[k] == expected[k]
