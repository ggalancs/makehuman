"""core/files3d.py: loading meshes from compiled .npz files with OBJ fallback."""

import logging
import os

import numpy as np
import pytest

import files3d
import getpath
import module3d

OBJ = """\
o sample
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
g front
f 1/1 2/2 3/3 4/4
g side-ñ
f 2/1 5/2 6/3 3/4
"""


@pytest.fixture
def user_home(tmp_path, monkeypatch):
    """Point the per-user MakeHuman folder at a temporary directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("MH_HOME_LOCATION", str(home))
    return home


def write_obj(folder, name="sample.obj"):
    os.makedirs(str(folder), exist_ok=True)
    path = os.path.join(str(folder), name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(OBJ)
    return path


def set_mtime(path, t):
    os.utime(path, (t, t))


def assert_same_mesh(a, b):
    np.testing.assert_allclose(a.coord, b.coord)
    assert a.fvert.tolist() == b.fvert.tolist()
    assert a.fuvs.tolist() == b.fuvs.tolist()
    assert a.group.tolist() == b.group.tolist()
    np.testing.assert_allclose(a.texco, b.texco)
    assert [g.name for g in a.faceGroups] == [g.name for g in b.faceGroups]
    assert a.has_uv == b.has_uv


def test_pack_unpack_string_list():
    names = ["front", "side-ñ", "", "x"]
    text, index = files3d.packStringList(names)
    assert list(index) == [0, 5, 12, 12]  # byte offsets: ñ is two bytes
    assert files3d.unpackStringList(text, index) == names
    assert files3d.unpackStringList(*files3d.packStringList([])) == []


def test_binary_round_trip(tmp_path):
    original = files3d.loadMesh(write_obj(tmp_path), maxFaces=6)
    npz = str(tmp_path / "copy.npz")
    files3d.saveBinaryMesh(original, npz)

    data = np.load(npz)
    # loading the OBJ shrinks MAX_FACES to the topology's needs (never below 4)
    assert original.MAX_FACES == 4
    assert "fuvs" in data.files and int(data["MAX_FACES"][0]) == 4

    restored = module3d.Object3D("restored")
    files3d.loadBinaryMesh(restored, npz)
    assert_same_mesh(original, restored)
    assert restored.MAX_FACES == 4
    assert restored.nfaces.tolist() == original.nfaces.tolist()
    np.testing.assert_allclose(restored.vnorm, original.vnorm, atol=1e-6)


def test_binary_mesh_without_uvs_omits_fuvs(tmp_path):
    original = files3d.loadMesh(write_obj(tmp_path))
    original.has_uv = False
    npz = str(tmp_path / "nouv.npz")
    files3d.saveBinaryMesh(original, npz)
    assert "fuvs" not in np.load(npz).files
    restored = module3d.Object3D("r")
    files3d.loadBinaryMesh(restored, npz)
    assert not restored.has_uv


def test_load_mesh_outside_user_folder_is_not_compiled(tmp_path, user_home, caplog):
    path = write_obj(tmp_path / "elsewhere")
    mesh = files3d.loadMesh(path)
    assert mesh.name == "sample" and mesh.path == path
    assert mesh.getFaceCount() == 2
    assert not os.path.exists(path[:-4] + ".npz")
    assert "compiled file missing" in caplog.text


def test_load_mesh_in_user_folder_compiles_then_uses_binary(user_home, monkeypatch):
    path = write_obj(getpath.getPath("data/custom"))
    assert getpath.isSubPath(path, getpath.getPath(""))
    first = files3d.loadMesh(path)
    npz = path[:-4] + ".npz"
    assert os.path.isfile(npz)

    def no_text(*args):
        raise AssertionError("OBJ parsed although an up-to-date npz exists")
    monkeypatch.setattr(files3d, "loadTextMesh", no_text)
    second = files3d.loadMesh(path)
    assert_same_mesh(first, second)
    assert second.name == "sample.obj"  # binary meshes are named after the file


def test_out_of_date_npz_is_rebuilt(user_home, caplog):
    path = write_obj(getpath.getPath("data/custom"))
    npz = path[:-4] + ".npz"
    files3d.loadMesh(path)
    set_mtime(npz, 1000000)
    set_mtime(path, 2000000)
    mesh = files3d.loadMesh(path)
    assert "out of date" in caplog.text
    assert mesh.getFaceCount() == 2
    assert os.path.getmtime(npz) > 2000000  # recompiled


def test_corrupt_npz_falls_back_to_obj(tmp_path, caplog):
    path = write_obj(tmp_path)
    with open(path[:-4] + ".npz", "wb") as f:
        f.write(b"not a zip file")
    set_mtime(path, 1000000)  # npz is newer, so it is tried
    mesh = files3d.loadMesh(path)
    assert mesh and mesh.getFaceCount() == 2
    assert "Problem loading binary mesh" in caplog.text


def test_unwritable_npz_is_reported_but_mesh_loads(user_home, caplog):
    caplog.set_level(logging.DEBUG)  # log.notice is below WARNING
    path = write_obj(getpath.getPath("data/custom"))
    os.makedirs(path[:-4] + ".npz")  # a directory where the compiled file should go
    mesh = files3d.loadMesh(path)
    assert mesh.getFaceCount() == 2
    assert "unable to save compiled mesh" in caplog.text


def test_missing_file_returns_false(tmp_path, caplog):
    assert files3d.loadMesh(str(tmp_path / "nope.obj")) is False
    assert "Unable to load obj file" in caplog.text


def test_load_into_given_object(tmp_path):
    target = module3d.Object3D("given")
    result = files3d.loadMesh(write_obj(tmp_path), obj=target, maxFaces=5)
    assert result is target and target.getFaceCount() == 2


def test_max_faces_limits_the_pole_count(tmp_path, caplog):
    # vertices 2 and 3 are shared by two faces: a limit of one face per vertex must fail
    assert files3d.loadMesh(write_obj(tmp_path), maxFaces=1) is False
    assert "too many faces attached to one vertex" in caplog.text


def test_system_base_mesh_loads_from_npz(monkeypatch):
    monkeypatch.chdir(os.path.join(os.path.dirname(__file__), "..", "..", "makehuman"))
    calls = []
    real = files3d.loadBinaryMesh
    monkeypatch.setattr(files3d, "loadBinaryMesh", lambda obj, p: calls.append(p) or real(obj, p))
    mesh = files3d.loadMesh(getpath.getSysDataPath("3dobjs/base.obj"), maxFaces=5)
    assert calls == [getpath.getSysDataPath("3dobjs/base.npz")]
    assert mesh.getVertexCount() == 19158
    assert mesh.getFaceGroup("body") is not None
