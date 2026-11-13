"""compile_targets.py, compile_models.py, compile_proxies.py and create_pylint_log.py.

The compile scripts write next to the data they find under ./data, so every test
runs them inside a temporary copy of the few assets it needs.
"""

import os
import runpy
import shutil
import sys
import zipfile

import numpy as np
import pytest

MH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "makehuman")
DATA = os.path.join(MH_DIR, "data")


def script(name):
    return os.path.join(MH_DIR, name)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))  # the scripts prepend their folders
    monkeypatch.chdir(tmp_path)
    return tmp_path


def copy_data(workdir, rel):
    dest = workdir / "data" / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(os.path.join(DATA, rel), str(dest))  # keeps mtimes: compiled files stay newer
    return dest


def text_target(path):
    """(indices, vectors) read the plain way from a .target file."""
    idx, vec = [], []
    with open(str(path), encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4 and not line.startswith("#"):
                idx.append(int(parts[0]))
                vec.append([float(v) for v in parts[1:]])
    return np.array(idx), np.array(vec)


# --- shared helpers ------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["compile_targets.py", "compile_models.py", "compile_proxies.py"])
def test_get_all_files_groups_by_pattern(workdir, name):
    for rel in ["a/x.target", "a/b/y.target", "a/b/z.png", "a/c.txt"]:
        (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (workdir / rel).write_text("")
    ns = runpy.run_path(script(name), run_name="as_module")
    targets, images = ns["getAllFiles"]("a", ["*.target", "*.png"])
    assert sorted(targets) == [os.path.join("a", "b", "y.target"), os.path.join("a", "x.target")]
    assert images == [os.path.join("a", "b", "z.png")]
    assert ns["getFiles"]("root", ["p.png", "q.obj"], "*.obj") == [os.path.join("root", "q.obj")]


# --- compile_targets.py ----------------------------------------------------------------------

def test_compile_targets_builds_the_archive_and_image_list(workdir, capsys):
    up = copy_data(workdir, "targets/nose/nose-base-up.target")
    down = copy_data(workdir, "targets/nose/nose-base-down.target")
    copy_data(workdir, "targets/chin/images/chin-height-decr-incr.png")
    custom = workdir / "data" / "targets" / "custom" / "mine.target"
    custom.parent.mkdir(parents=True)
    custom.write_text("# author Jane Doe\n# license CC-BY\n3 0.001 0.002 -0.003\n10 0.5 0 0\n",
                      encoding="utf-8")

    runpy.run_path(script("compile_targets.py"), run_name="__main__")

    npz = workdir / "data" / "targets.npz"
    with zipfile.ZipFile(str(npz)) as z:
        names = set(z.namelist())
    assert names == {
        "targets/targets.license.npy",
        "targets/nose/nose-base-up.index.npy", "targets/nose/nose-base-up.vector.npy",
        "targets/nose/nose-base-down.index.npy", "targets/nose/nose-base-down.vector.npy",
        "targets/custom/mine.index.npy", "targets/custom/mine.vector.npy",
        "targets/custom/mine.license.npy",
    }
    archive = np.load(str(npz), allow_pickle=True)
    idx, vec = text_target(up)
    np.testing.assert_array_equal(archive["targets/nose/nose-base-up.index"], idx)
    # Stored as int16 thousandths, the way algos3d reads them back (* 1e-3)
    np.testing.assert_allclose(archive["targets/nose/nose-base-up.vector"] * 1e-3, vec, atol=5e-4)
    np.testing.assert_array_equal(archive["targets/custom/mine.vector"], [[1, 2, -3], [500, 0, 0]])

    import makehuman
    lic = makehuman.LicenseInfo().fromNumpyString(archive["targets/custom/mine.license"])
    assert (lic.author, lic.license) == ("Jane Doe", "CC-BY")
    default = makehuman.LicenseInfo().fromNumpyString(archive["targets/targets.license"])
    assert default.asDict() == makehuman.getAssetLicense().asDict()

    # Intermediate .npy files are cleaned up
    leftovers = [f for _, _, files in os.walk(str(workdir / "data")) for f in files if f.endswith(".npy")]
    assert leftovers == []
    assert (workdir / "data" / "images.list").read_text(encoding="utf-8") == \
        "data/targets/chin/images/chin-height-decr-incr.png\n"
    out = capsys.readouterr().out
    assert out.count("converted target") == 3 and out.rstrip().endswith("All done.")
    assert down.exists() and up.exists()  # sources stay


def test_compiled_archive_loads_like_the_text_target(workdir, monkeypatch):
    up = copy_data(workdir, "targets/nose/nose-base-up.target")
    runpy.run_path(script("compile_targets.py"), run_name="__main__")
    import algos3d
    monkeypatch.setattr(algos3d.Target, "npzfile", None)  # use our archive, not the real one
    monkeypatch.setattr(algos3d.Target, "npzdir", None, raising=False)
    monkeypatch.setattr(algos3d.Target, "npztime", None, raising=False)
    os.utime(str(up), (1, 1))  # the source must be older than the archive
    t = algos3d.Target(None, None)
    t._load_binary("data/targets/nose/nose-base-up.target")
    idx, vec = text_target(up)
    np.testing.assert_array_equal(t.verts, idx)
    np.testing.assert_allclose(t.data, vec, atol=5e-4)


def test_malformed_target_reports_the_real_error(workdir):
    bad = workdir / "data" / "targets" / "bad.target"
    bad.parent.mkdir(parents=True)
    bad.write_text("1 0.1 nan-ish 0.3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        runpy.run_path(script("compile_targets.py"), run_name="__main__")


# --- compile_models.py -------------------------------------------------------------------------

def test_compile_models_writes_npz_meshes(workdir, capsys):
    obj = copy_data(workdir, "3dobjs/axis.obj")
    brow = copy_data(workdir, "eyebrows/eyebrow001/eyebrow001.obj")
    runpy.run_path(script("compile_models.py"), run_name="__main__")

    import files3d
    import module3d
    for path in (obj, brow):
        npz = path.with_suffix(".npz")
        assert npz.is_file()
        text = module3d.Object3D("text")
        files3d.loadTextMesh(text, str(path))
        binary = module3d.Object3D("binary")
        files3d.loadBinaryMesh(binary, str(npz))
        np.testing.assert_allclose(binary.coord, text.coord)
        np.testing.assert_array_equal(binary.fvert, text.fvert)
        assert [g.name for g in binary.faceGroups] == [g.name for g in text.faceGroups]
    out = capsys.readouterr().out
    assert out.count("converted mesh") == 2 and "All done." in out


def test_compile_mesh_reports_unloadable_files(workdir, capsys):
    ns = runpy.run_path(script("compile_models.py"), run_name="as_module")
    broken = workdir / "broken.obj"
    broken.write_text("v 0 0 0\nf 1 2 3 4 5 6 7\n", encoding="utf-8")
    assert ns["compileMesh"](str(broken)) is False
    assert "Could not load OBJ file %s" % broken in capsys.readouterr().out
    assert not (workdir / "broken.npz").exists()


def test_compile_mesh_reports_unwritable_output(workdir, capsys):
    ns = runpy.run_path(script("compile_models.py"), run_name="as_module")
    obj = copy_data(workdir, "3dobjs/axis.obj")
    (workdir / "data" / "3dobjs" / "axis.npz").mkdir()  # a folder where the file should go
    assert ns["compileMesh"](str(obj)) is False
    assert "Unable to save compiled mesh for file %s" % obj in capsys.readouterr().out


# --- compile_proxies.py ------------------------------------------------------------------------

@pytest.fixture
def proxy_workdir(workdir, monkeypatch):
    # No OpenGL context here: keep materials from compiling shaders (see test_human.py)
    shader = sys.modules.get("shader")
    if shader is not None:
        monkeypatch.setattr(shader.Shader, "_supported", False)
    copy_data(workdir, "3dobjs/base.obj")
    copy_data(workdir, "3dobjs/base.npz")
    copy_data(workdir, "skins/default.mhmat")
    for name in ("eyebrow001.mhclo", "eyebrow001.obj", "eyebrow001.npz", "eyebrow001.mhmat"):
        copy_data(workdir, "eyebrows/eyebrow001/" + name)
    return workdir


def test_compile_proxies_writes_binary_proxies(proxy_workdir, capsys):
    runpy.run_path(script("compile_proxies.py"), run_name="__main__")
    out_file = proxy_workdir / "data" / "eyebrows" / "eyebrow001" / "eyebrow001.mhpxy"
    assert out_file.is_file()
    out = capsys.readouterr().out
    assert "converted proxy data/eyebrows/eyebrow001/eyebrow001.mhclo" in out
    assert out.rstrip().endswith("All done.")

    import files3d
    import proxy
    from human import Human
    human = Human(files3d.loadMesh("data/3dobjs/base.obj"))
    text = proxy.loadTextProxy(human, "data/eyebrows/eyebrow001/eyebrow001.mhclo", type=None)
    binary = proxy.loadBinaryProxy(str(out_file), human, type="Eyebrows")
    assert binary.uuid == text.uuid and binary.name == text.name
    np.testing.assert_array_equal(binary.ref_vIdxs, text.ref_vIdxs)
    np.testing.assert_allclose(binary.weights, text.weights)
    np.testing.assert_allclose(binary.offsets, text.offsets)


def test_compile_proxy_reports_unloadable_files(proxy_workdir, capsys):
    ns = runpy.run_path(script("compile_proxies.py"), run_name="as_module")
    import files3d
    human = ns["Human"](files3d.loadMesh("data/3dobjs/base.obj"))
    broken = proxy_workdir / "broken.proxy"
    broken.write_text("name broken\nverts 0\n12 34\n", encoding="utf-8")
    assert ns["compileProxy"](str(broken), human) is False
    out = capsys.readouterr().out
    assert "Could not load proxy file %s." % broken in out and "Traceback" in out
    assert not (proxy_workdir / "broken.mhpxy").exists()


def test_compile_proxy_removes_half_written_output(proxy_workdir, monkeypatch, capsys):
    ns = runpy.run_path(script("compile_proxies.py"), run_name="as_module")
    import files3d
    import proxy
    human = ns["Human"](files3d.loadMesh("data/3dobjs/base.obj"))

    def fail_halfway(pxy, path):
        with open(path, "wb") as f:
            f.write(b"partial")
        raise IOError("disk full")

    monkeypatch.setattr(proxy, "saveBinaryProxy", fail_halfway)
    mhclo = "data/eyebrows/eyebrow001/eyebrow001.mhclo"
    assert ns["compileProxy"](mhclo, human) is False
    assert "Unable to save compiled proxy for file %s" % mhclo in capsys.readouterr().out
    assert not (proxy_workdir / "data" / "eyebrows" / "eyebrow001" / "eyebrow001.mhpxy").exists()


# --- create_pylint_log.py ---------------------------------------------------------------------

def test_pylint_log_lists_every_python_file(workdir, monkeypatch):
    for rel in ["a.py", "pkg/b.py", "pkg/deep/c.py", "pkg/readme.txt", "notpy.pyc"]:
        (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (workdir / rel).write_text("")
    commands = []
    monkeypatch.setattr(os, "system", commands.append)
    runpy.run_path(script("create_pylint_log.py"), run_name="__main__")
    assert len(commands) == 1
    head = "pylint3 > pylint.log "
    assert commands[0].startswith(head)
    files = commands[0][len(head):].split(" ")
    assert sorted(files) == sorted([os.path.join(".", "a.py"), os.path.join(".", "pkg", "b.py"),
                                    os.path.join(".", "pkg", "deep", "c.py")])
