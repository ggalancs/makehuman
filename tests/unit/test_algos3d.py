"""Tests for core/algos3d.py: morph targets (loading, caching, applying,
saving) and resetting a mesh to its base shape."""

import os
import time
import types
import zipfile

import numpy as np
import pytest

import algos3d
import makehuman
import module3d

# Tangent computation of module3d divides by zero on axis-aligned UVs (see
# test_module3d.py); keep its RuntimeWarnings out of these tests' output.
pytestmark = [pytest.mark.filterwarnings("ignore::RuntimeWarning:module3d"),
              pytest.mark.filterwarnings("ignore:invalid value encountered in reduce:RuntimeWarning")]

CUBE_COORDS = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
CUBE_QUADS = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
CUBE_GROUPS = ["bottom", "top", "front", "back", "left", "right"]

TARGET_TEXT = """\
# This is a target file for MakeHuman
# basemesh hm08
4 0.000000 0.000000 1.000000
5 0.000000 0.000000 1.000000
6 0.500000 -0.250000 2.000000

garbage line
7 0.000000 0.000000 1.000000 extra
"""


def make_cube():
    mesh = module3d.Object3D("cube", 4)
    mesh.setCoords(CUBE_COORDS)
    mesh.setUVs([(0, 0), (1, 0), (1, 1), (0, 1)])
    groups = [mesh.createFaceGroup(g) for g in CUBE_GROUPS]
    mesh.setFaces(CUBE_QUADS, [(0, 1, 2, 3)] * 6, [g.idx for g in groups])
    mesh.updateIndexBuffer()
    mesh.calcNormals()
    mesh.update()
    return mesh


@pytest.fixture(autouse=True)
def isolated_target_state(monkeypatch):
    """Target keeps the compiled archive and a cache in class/module state:
    give every test its own, with no archive (npzfile False) by default."""
    monkeypatch.setattr(algos3d.Target, "npzfile", False)
    monkeypatch.setattr(algos3d.Target, "npztime", None)
    monkeypatch.setattr(algos3d.Target, "npzdir", None)
    monkeypatch.setattr(algos3d, "_targetBuffer", {})


@pytest.fixture
def target_file(tmp_path):
    path = tmp_path / "raise-top.target"
    path.write_text(TARGET_TEXT, encoding="utf-8")
    return str(path)


# --- loading -------------------------------------------------------------------------

def test_load_text_target(target_file):
    t = algos3d.Target(make_cube(), target_file)
    assert t.verts.tolist() == [4, 5, 6]  # malformed lines skipped
    assert np.allclose(t.data, [(0, 0, 1), (0, 0, 1), (0.5, -0.25, 2)])
    assert t.morphFactor == -1
    assert repr(t) == "<Target raise-top.target>"
    # Faces touching the moved vertices: everything but the bottom
    assert sorted(t.faces.tolist()) == [1, 2, 3, 4, 5]


def test_default_license_of_a_text_target(monkeypatch, target_file):
    t = algos3d.Target(make_cube(), target_file)
    monkeypatch.setattr(algos3d.Target, "npzfile", None)  # archive not opened (yet)
    assert not t.license.isCustomized()
    assert (t.license.license, t.license.author) == ("AGPL3", "MakeHuman")


@pytest.mark.xfail(strict=True, raises=TypeError, reason=(
    "algos3d.py:117 tests `'targets/targets.license' in Target.npzfile`, but "
    "npzfile is False (not None) once _load_binary found no archive, so "
    "Target.license raises TypeError in that supported fallback mode"))
def test_license_of_a_target_when_there_is_no_archive(target_file):
    t = algos3d.Target(make_cube(), target_file)  # fixture: npzfile False
    assert t.license.license == "AGPL3"


def test_custom_license_comments_are_kept(tmp_path):
    path = tmp_path / "custom.target"
    path.write_text("# author Jane Doe\n# license CC-BY\n0 1 0 0\n", encoding="utf-8")
    t = algos3d.Target(make_cube(), str(path))
    assert t.license.isCustomized()
    assert t.license.author == "Jane Doe" and t.license.license == "CC-BY"


def test_unreadable_target_is_empty(tmp_path):
    t = algos3d.Target(make_cube(), str(tmp_path / "missing.target"))
    assert len(t.verts) == 0
    assert t.apply(make_cube(), 1.0) is False


def test_unnamed_target_is_empty():
    t = algos3d.Target(None, None)  # how compile_targets.py makes its scratch target
    assert t.verts == [] and t.name is None


def test_binary_round_trip_rounds_to_millimetres(target_file, tmp_path):
    t = algos3d.Target(make_cube(), target_file)
    t.data = t.data + 0.0004  # below the 1e-3 resolution of the binary format
    iname, vname, lname = t._save_binary(target_file)
    assert lname is None
    assert iname.endswith(".index.npy") and vname.endswith(".vector.npy")
    assert np.load(iname).dtype == np.uint16 and np.load(vname).dtype == np.int16
    loaded = algos3d.Target(make_cube(), target_file)  # npzfile False: per-file binaries
    assert loaded.verts.tolist() == [4, 5, 6]
    assert np.allclose(loaded.data, np.round((t.data) * 1e3) * 1e-3)
    assert not np.allclose(loaded.data, t.data)  # it really came from the binaries


def test_binary_save_with_license(tmp_path):
    path = tmp_path / "custom.target"
    path.write_text("# author Jane Doe\n0 1 0 0\n", encoding="utf-8")
    t = algos3d.Target(make_cube(), str(path))
    iname, vname, lname = t._save_binary(str(path))
    assert lname == str(tmp_path / "custom.license.npy")
    text, index = np.load(lname, allow_pickle=True)
    assert makehuman.LicenseInfo().fromNumpyString(text, index).author == "Jane Doe"


def test_binary_save_failure_is_logged_not_raised(tmp_path):
    t = algos3d.Target(None, None)
    t.verts, t.data = np.array([1]), np.zeros((1, 3))
    assert t._save_binary(str(tmp_path / "no" / "such" / "dir.target")) is None


@pytest.mark.parametrize("stale", ["index", "vector"])
def test_stale_or_missing_binaries_fall_back_to_text(target_file, stale):
    t = algos3d.Target(make_cube(), target_file)
    t.data = t.data * 2  # binaries now disagree with the text
    iname, vname, _ = t._save_binary(target_file)
    old = time.time() - 100
    os.utime(iname if stale == "index" else vname, (old, old))
    loaded = algos3d.Target(make_cube(), target_file)
    assert np.allclose(loaded.data[2], (0.5, -0.25, 2))  # the text, not the doubled binaries
    os.remove(vname)
    os.utime(iname, None)
    assert np.allclose(algos3d.Target(make_cube(), target_file).data[2], (0.5, -0.25, 2))
    os.remove(iname)
    assert np.allclose(algos3d.Target(make_cube(), target_file).data[2], (0.5, -0.25, 2))


def write_archive(tmp_path, entries):
    """Write a targets.npz-style zip of .npy files, like compile_targets.py."""
    archive = tmp_path / "targets.npz"
    with zipfile.ZipFile(archive, "w") as z:
        for name, array in entries.items():
            npy = tmp_path / "entry.npy"
            np.save(npy, array)
            z.write(npy, name + ".npy")
            os.remove(npy)
    return str(archive)


def use_archive(monkeypatch, tmp_path, entries, mtime=None):
    archive = write_archive(tmp_path, entries)
    monkeypatch.setattr(algos3d.Target, "npzfile", np.load(archive))
    monkeypatch.setattr(algos3d.Target, "npzdir", str(tmp_path))
    monkeypatch.setattr(algos3d.Target, "npztime", mtime if mtime is not None else time.time() + 100)
    monkeypatch.chdir(tmp_path)  # the staleness check uses the archive-relative name


def test_load_from_archive(monkeypatch, tmp_path):
    use_archive(monkeypatch, tmp_path, {
        "targets/up.index": np.array([0, 3], dtype=np.uint16),
        "targets/up.vector": np.array([[0, 0, 1000], [0, 500, 0]], dtype=np.int16),
    })
    t = algos3d.Target(make_cube(), str(tmp_path / "targets" / "up.target"))  # no text file at all
    assert t.verts.tolist() == [0, 3]
    assert np.allclose(t.data, [(0, 0, 1), (0, 0.5, 0)])


def test_archive_is_ignored_when_the_text_target_is_newer(monkeypatch, tmp_path):
    (tmp_path / "targets").mkdir()
    (tmp_path / "targets" / "up.target").write_text("1 0 0 7\n", encoding="utf-8")
    use_archive(monkeypatch, tmp_path, {
        "targets/up.index": np.array([0], dtype=np.uint16),
        "targets/up.vector": np.array([[0, 0, 1000]], dtype=np.int16),
    }, mtime=time.time() - 100)
    t = algos3d.Target(make_cube(), str(tmp_path / "targets" / "up.target"))
    assert t.verts.tolist() == [1] and np.allclose(t.data, [(0, 0, 7)])


@pytest.mark.parametrize("missing", ["index", "vector"])
def test_incomplete_archive_entry_falls_back_to_text(monkeypatch, tmp_path, missing):
    (tmp_path / "targets").mkdir()
    (tmp_path / "targets" / "up.target").write_text("2 0 7 0\n", encoding="utf-8")
    entries = {"targets/up.index": np.array([0], dtype=np.uint16),
               "targets/up.vector": np.array([[0, 0, 1000]], dtype=np.int16)}
    del entries["targets/up." + missing]
    use_archive(monkeypatch, tmp_path, entries)
    t = algos3d.Target(make_cube(), str(tmp_path / "targets" / "up.target"))
    assert t.verts.tolist() == [2] and np.allclose(t.data, [(0, 7, 0)])


def test_archive_is_opened_once_from_the_data_folder(monkeypatch, tmp_path):
    archive = write_archive(tmp_path, {"targets/up.index": np.array([5], dtype=np.uint16),
                                       "targets/up.vector": np.array([[1000, 0, 0]], dtype=np.int16)})
    monkeypatch.setattr(algos3d.Target, "npzfile", None)
    monkeypatch.setattr(algos3d, "getSysDataPath", lambda sub: archive)
    monkeypatch.chdir(tmp_path)
    t = algos3d.Target(make_cube(), str(tmp_path / "targets" / "up.target"))
    assert t.verts.tolist() == [5]
    assert algos3d.Target.npzdir == str(tmp_path)
    assert algos3d.Target.npztime == os.path.getmtime(archive)


def test_missing_archive_disables_it(monkeypatch, target_file, tmp_path):
    monkeypatch.setattr(algos3d.Target, "npzfile", None)
    monkeypatch.setattr(algos3d, "getSysDataPath", lambda sub: str(tmp_path / "nope.npz"))
    t = algos3d.Target(make_cube(), target_file)
    assert algos3d.Target.npzfile is False
    assert t.verts.tolist() == [4, 5, 6]


def license_entry():
    # Exactly what compile_targets.py stores for targets/targets.license
    return makehuman.getAssetLicense().toNumpyString()


@pytest.mark.xfail(strict=True, raises=ValueError, reason=(
    "algos3d.py:119 (and :171) read a license entry that compile_targets.py "
    "saved as an object array, but Target.npzfile comes from np.load() with "
    "the default allow_pickle=False (numpy >= 1.16.3): reading "
    "Target.license of any archived target raises ValueError, and an "
    "archived target with its own license silently falls back to text"))
def test_license_of_an_archived_target(monkeypatch, tmp_path):
    use_archive(monkeypatch, tmp_path, {
        "targets/targets.license": license_entry(),
        "targets/up.index": np.array([0], dtype=np.uint16),
        "targets/up.vector": np.array([[0, 0, 1000]], dtype=np.int16),
    })
    t = algos3d.Target(make_cube(), str(tmp_path / "targets" / "up.target"))
    assert t.license.license == makehuman.getAssetLicense().license


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "algos3d.py:171: same allow_pickle problem for a per-target license entry; "
    "the ValueError is swallowed by Target._load, which falls back to the "
    "text file, and without one the target loads empty"))
def test_archived_target_with_its_own_license(monkeypatch, tmp_path):
    custom = makehuman.getAssetLicense({"author": "Jane Doe"})
    use_archive(monkeypatch, tmp_path, {
        "targets/up.index": np.array([0], dtype=np.uint16),
        "targets/up.vector": np.array([[0, 0, 1000]], dtype=np.int16),
        "targets/up.license": custom.toNumpyString(),
    })
    t = algos3d.Target(make_cube(), str(tmp_path / "targets" / "up.target"))
    assert list(t.verts) == [0]
    assert t.license.author == "Jane Doe"


def test_license_property_without_an_archive_license(monkeypatch, tmp_path):
    use_archive(monkeypatch, tmp_path, {
        "targets/up.index": np.array([0], dtype=np.uint16),
        "targets/up.vector": np.array([[0, 0, 1000]], dtype=np.int16),
    })
    t = algos3d.Target(make_cube(), str(tmp_path / "targets" / "up.target"))
    lic = t.license
    assert lic.license == "AGPL3" and lic.author == "MakeHuman"
    custom = makehuman.getAssetLicense({"author": "X"})
    t.setLicense(custom)
    assert t.license is custom


def test_default_target_license():
    lic = algos3d.defaultTargetLicense()
    assert (lic.license, lic.author) == ("AGPL3", "MakeHuman")
    assert "Joel Palmius" in lic.copyright
    assert not lic.isCustomized()


# --- applying ------------------------------------------------------------------------

def test_apply_moves_only_the_target_vertices(target_file):
    mesh = make_cube()
    t = algos3d.Target(mesh, target_file)
    assert t.apply(mesh, 0.5) is True
    assert t.morphFactor == 0.5
    expected = np.array(CUBE_COORDS, dtype=float)
    expected[[4, 5, 6]] += 0.5 * np.array([(0, 0, 1), (0, 0, 1), (0.5, -0.25, 2)])
    assert np.allclose(mesh.coord, expected)
    assert np.allclose(mesh.r_coord, mesh.coord[mesh.vmap])  # update() synced it
    # The top is no longer flat: its normal tilted (recalculated by apply)
    top = mesh.fnorm[1] / np.linalg.norm(mesh.fnorm[1])
    assert not np.allclose(top, [0, 0, 1])
    # Applying the opposite factor undoes it
    t.apply(mesh, -0.5)
    assert np.allclose(mesh.coord, CUBE_COORDS, atol=1e-6)


def test_apply_with_scale_and_without_updates(target_file):
    mesh = make_cube()
    t = algos3d.Target(mesh, target_file)
    synced = mesh.r_coord.copy()
    t.apply(mesh, 1.0, update=False, calcNormals=False, scale=(2.0, 1.0, 0.5))
    assert np.allclose(mesh.coord[6], np.array((0, 1, 1)) + (1.0, -0.25, 1.0))
    assert np.array_equal(mesh.r_coord, synced)  # not pushed to the render buffers
    assert mesh.ucoor is not False  # but marked for the next update


def test_apply_restricted_to_a_face_group(target_file):
    mesh = make_cube()
    t = algos3d.Target(mesh, target_file)
    t.apply(mesh, 1.0, faceGroupToUpdateName="back")  # back face: vertices 2, 3, 6, 7
    moved = np.flatnonzero(np.any(mesh.coord != np.array(CUBE_COORDS, dtype=np.float32), axis=1))
    assert moved.tolist() == [6]


def test_apply_with_zero_factor_only_refreshes(target_file):
    mesh = make_cube()
    t = algos3d.Target(mesh, target_file)
    assert t.apply(mesh, 0.0) is True
    assert np.array_equal(mesh.coord, np.array(CUBE_COORDS, dtype=np.float32))


def test_apply_in_a_posed_direction(target_file):
    """With an animated mesh the target offsets are posed (skinned) first."""
    rot_z = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    weights = np.zeros(8, dtype=[("b_idx1", np.uint32), ("wght1", np.float32)])
    weights["wght1"] = 1.0
    calls = []

    class Mapping(object):
        def isCompiled(self, n):
            return False

        def compileData(self, skel, n):
            calls.append(("compile", skel, n))

        def compiled(self, n):
            return weights

    class Track(object):
        def isBaked(self):
            return False

        def bake(self, skel):
            calls.append(("bake", skel))

    skeleton = object()
    animated = types.SimpleNamespace(
        getBoundMesh=lambda name: (None, Mapping()),
        getBaseSkeleton=lambda: skeleton,
        getActiveAnimation=lambda: Track(),
        getPoseState=lambda: np.array([rot_z]))
    mesh = make_cube()
    t = algos3d.Target(mesh, target_file)
    t.apply(mesh, 1.0, animatedMesh=animated)
    # (0.5, -0.25, 2) turned 90 degrees about z is (0.25, 0.5, 2)
    assert np.allclose(mesh.coord[6], np.array((0, 1, 1)) + (0.25, 0.5, 2))
    assert ("compile", skeleton, 4) in calls and ("bake", skeleton) in calls


def test_translation_targets_are_cached(target_file):
    mesh = make_cube()
    algos3d.loadTranslationTarget(mesh, target_file, 1.0)
    assert np.allclose(mesh.coord[4], (0, 0, 2))
    cached = algos3d.getTarget(mesh, target_file)
    assert algos3d.getTarget(mesh, target_file) is cached
    # Edits to the file are not seen until the cache entry is refreshed
    with open(target_file, "w", encoding="utf-8") as f:
        f.write("0 9 9 9\n")
    assert algos3d.getTarget(mesh, target_file).verts.tolist() == [4, 5, 6]
    algos3d.refreshCachedTarget(target_file)
    assert algos3d.getTarget(mesh, target_file).verts.tolist() == [0]
    algos3d.refreshCachedTarget(target_file + ".unknown")  # nothing cached: no error


def test_load_translation_target_with_nothing_to_do(target_file):
    mesh = make_cube()
    algos3d.loadTranslationTarget(mesh, target_file, 0.0, update=0)
    assert algos3d._targetBuffer == {}  # not even loaded
    assert np.array_equal(mesh.coord, np.array(CUBE_COORDS, dtype=np.float32))


# --- saving and resetting --------------------------------------------------------------

def test_save_translation_target_round_trip(tmp_path):
    mesh = make_cube()
    mesh.coord[2] += (0.25, 0.0, -0.5)
    mesh.coord[7] += (0.0005, 0.0, 0.0)  # below epsilon: not saved
    path = str(tmp_path / "saved.target")
    algos3d.saveTranslationTarget(mesh, path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert text.startswith("# MakeHuman asset license:")
    assert "# basemesh hm08" in text
    loaded = algos3d.Target(mesh, path)
    assert loaded.verts.tolist() == [2]
    assert np.allclose(loaded.data, [(0.25, 0.0, -0.5)])
    assert not hasattr(loaded, "_license")  # the default license header is not "custom"


def test_save_unchanged_mesh_writes_only_the_header(tmp_path):
    path = str(tmp_path / "empty.target")
    algos3d.saveTranslationTarget(make_cube(), path)
    loaded = algos3d.Target(make_cube(), path)
    assert len(loaded.verts) == 0


def test_save_to_an_unwritable_path_returns_none(tmp_path):
    mesh = make_cube()
    mesh.coord[0] += 1
    assert algos3d.saveTranslationTarget(mesh, str(tmp_path / "missing" / "x.target")) is None


@pytest.mark.xfail(strict=True, raises=UnboundLocalError, reason=(
    "algos3d.py:442-447: the groupToSave branch is a TODO (`pass`), so "
    "vertsToSave is never assigned and saving a group crashes"))
def test_save_translation_target_for_a_group(tmp_path):
    mesh = make_cube()
    mesh.coord[[4, 0]] += 1
    path = str(tmp_path / "group.target")
    algos3d.saveTranslationTarget(mesh, path, groupToSave=mesh.getFaceGroup("top"))
    assert algos3d.Target(mesh, path).verts.tolist() == [4]


@pytest.mark.parametrize("update, calc_norm", [(None, None), (True, True)])
def test_reset_obj_restores_the_base_shape(update, calc_norm):
    mesh = make_cube()
    mesh.changeCoords(mesh.coord * 3)
    mesh.update()
    algos3d.resetObj(mesh, update=update, calcNorm=calc_norm)
    assert np.array_equal(mesh.coord, np.array(CUBE_COORDS, dtype=np.float32))
    if update:
        assert np.array_equal(mesh.r_coord, mesh.coord[mesh.vmap])
    else:
        assert mesh.ucoor is True and not np.array_equal(mesh.r_coord, mesh.coord[mesh.vmap])
