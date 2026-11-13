import functools
import hashlib
import http.server
import os
import socket
import stat
import threading
import zipfile

import pytest

import download_system_assets as dsa


def make_zip(path, entries, symlinks=()):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
        for name in symlinks:
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, "target")
    return path


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def closed_port_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return "http://127.0.0.1:%d/pack.zip" % port  # nothing listens there any more


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield root, "http://127.0.0.1:%d" % httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def pack(tmp_path):
    return make_zip(tmp_path / "pack.zip", {
        "hair/": b"",  # directory entries, as real zip tools write them
        "hair/bob01/bob01.mhclo": b"clo",
        "skins/young/young.mhmat": b"mat",
        "packs/makehuman_system_assets.json": b"{}",
    })


def test_pinned_hash_is_a_sha256():
    assert len(dsa.PACK_SHA256) == 64
    int(dsa.PACK_SHA256, 16)


def test_default_data_dir_is_next_to_the_script():
    d = dsa.default_data_dir()
    assert d == os.path.join(os.path.dirname(os.path.abspath(dsa.__file__)), "data")
    assert os.path.isdir(os.path.join(d, "targets"))


def test_sha256_of(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"x" * (dsa.CHUNK + 10))
    assert dsa.sha256_of(f) == hashlib.sha256(b"x" * (dsa.CHUNK + 10)).hexdigest()


@pytest.mark.parametrize("name", ["/etc/passwd", "\\abs", "../x", "a/../../b", "a\\..\\b", "C:/x"])
def test_check_member_rejects_paths_outside_target(name):
    with pytest.raises(dsa.AssetError, match="Unsafe path"):
        dsa.check_member(zipfile.ZipInfo(name))


def test_check_member_rejects_symlinks():
    info = zipfile.ZipInfo("hair/link")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(dsa.AssetError, match="Symbolic link"):
        dsa.check_member(info)


def test_check_member_accepts_normal_paths():
    dsa.check_member(zipfile.ZipInfo("hair/bob01/bob01.mhclo"))
    dsa.check_member(zipfile.ZipInfo("hair/"))


def test_install_extracts_counts_and_never_overwrites(tmp_path, pack):
    data = tmp_path / "data"
    (data / "hair" / "bob01").mkdir(parents=True)
    (data / "hair" / "bob01" / "bob01.mhclo").write_bytes(b"clo")       # identical
    (data / "skins" / "young").mkdir(parents=True)
    (data / "skins" / "young" / "young.mhmat").write_bytes(b"mine")     # locally modified
    logged = []
    counts = dsa.install(pack, str(data), log=logged.append)
    assert counts == {"installed": 1, "identical": 1, "kept": 1}
    assert (data / "skins" / "young" / "young.mhmat").read_bytes() == b"mine"
    assert (data / "packs" / "makehuman_system_assets.json").read_bytes() == b"{}"
    assert any("Keeping existing file" in m for m in logged)
    # idempotent
    assert dsa.install(pack, str(data), log=logged.append) == {"installed": 0, "identical": 2, "kept": 1}


def test_install_validates_every_member_before_writing(tmp_path):
    z = make_zip(tmp_path / "evil.zip", {"hair/good.mhclo": b"ok", "../evil": b"x"})
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(dsa.AssetError):
        dsa.install(z, str(data))
    assert list(data.iterdir()) == []
    assert not (tmp_path / "evil").exists()


def test_install_rejects_symlink_entries(tmp_path):
    z = make_zip(tmp_path / "link.zip", {"hair/a": b"a"}, symlinks=["hair/link"])
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(dsa.AssetError, match="Symbolic link"):
        dsa.install(z, str(data))
    assert list(data.iterdir()) == []


def test_download_falls_back_to_the_next_mirror(tmp_path, server, pack):
    root, base = server
    (root / "pack.zip").write_bytes(open(pack, "rb").read())
    dest = tmp_path / "out.zip"
    logged = []
    used = dsa.download([closed_port_url(), base + "/pack.zip"], dest, timeout=5, log=logged.append)
    assert used == base + "/pack.zip"
    assert dest.read_bytes() == open(pack, "rb").read()
    assert any("failed" in m for m in logged)


def test_download_reports_every_mirror_when_all_fail(tmp_path, server):
    _root, base = server
    dead = closed_port_url()
    with pytest.raises(dsa.AssetError) as e:
        dsa.download([dead, base + "/missing.zip"], tmp_path / "out.zip", timeout=5, log=lambda m: None)
    assert dead in str(e.value) and "404" in str(e.value)


def test_main_installs_from_local_zip(tmp_path, pack):
    data = tmp_path / "data"
    data.mkdir()
    logged = []
    rc = dsa.main(["--zip", str(pack), "--data-dir", str(data), "--sha256", sha(pack)], log=logged.append)
    assert rc == 0
    assert (data / "hair" / "bob01" / "bob01.mhclo").read_bytes() == b"clo"
    assert "Installed 3 files" in logged[-2]


def test_main_refuses_a_pack_with_the_wrong_hash(tmp_path, pack):
    data = tmp_path / "data"
    data.mkdir()
    logged = []
    rc = dsa.main(["--zip", str(pack), "--data-dir", str(data)], log=logged.append)
    assert rc == 3
    assert list(data.iterdir()) == []
    assert any(sha(pack) in m for m in logged)


def test_main_requires_an_existing_data_dir(tmp_path, pack):
    rc = dsa.main(["--zip", str(pack), "--data-dir", str(tmp_path / "nope")], log=lambda m: None)
    assert rc == 2


def test_main_downloads_and_removes_its_temporary_folder(tmp_path, server, pack, monkeypatch):
    root, base = server
    (root / "pack.zip").write_bytes(open(pack, "rb").read())
    data = tmp_path / "data"
    data.mkdir()
    made = []
    real_mkdtemp = dsa.tempfile.mkdtemp
    monkeypatch.setattr(dsa.tempfile, "mkdtemp", lambda **kw: made.append(real_mkdtemp(dir=tmp_path, **kw)) or made[-1])
    rc = dsa.main(["--data-dir", str(data), "--sha256", sha(pack)], urls=[base + "/pack.zip"], log=lambda m: None)
    assert rc == 0
    assert (data / "skins" / "young" / "young.mhmat").exists()
    assert made and not os.path.exists(made[0])


def test_main_returns_1_when_no_mirror_works(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    logged = []
    rc = dsa.main(["--data-dir", str(data)], urls=[closed_port_url()], log=logged.append)
    assert rc == 1
    assert "Could not download" in logged[-1]


def test_main_records_the_install_and_skips_the_next_run(tmp_path, pack):
    data = tmp_path / "data"
    data.mkdir()
    digest = sha(pack)
    assert dsa.installed_pack(str(data)) is None
    assert dsa.main(["--zip", str(pack), "--data-dir", str(data), "--sha256", digest], log=lambda m: None) == 0
    assert dsa.installed_pack(str(data)) == digest
    logged = []
    # Already installed: no download is attempted (the only mirror is dead)
    rc = dsa.main(["--data-dir", str(data), "--sha256", digest], urls=[closed_port_url()], log=logged.append)
    assert rc == 0 and "already installed" in logged[-1]


def test_force_and_a_new_pack_reinstall(tmp_path, pack):
    data = tmp_path / "data"
    data.mkdir()
    digest = sha(pack)
    dsa.main(["--zip", str(pack), "--data-dir", str(data), "--sha256", digest], log=lambda m: None)
    (data / "hair" / "bob01" / "bob01.mhclo").unlink()
    assert dsa.main(["--zip", str(pack), "--data-dir", str(data), "--sha256", digest, "--force"], log=lambda m: None) == 0
    assert (data / "hair" / "bob01" / "bob01.mhclo").exists()
    # A marker from a different pack does not count as installed
    (data / dsa.MARKER).write_text("0" * 64)
    logged = []
    assert dsa.main(["--zip", str(pack), "--data-dir", str(data), "--sha256", digest], log=logged.append) == 0
    assert "Installed" in logged[-2]


def test_failed_install_writes_no_marker(tmp_path):
    z = make_zip(tmp_path / "evil.zip", {"../evil": b"x"})
    data = tmp_path / "data"
    data.mkdir()
    assert dsa.main(["--zip", str(z), "--data-dir", str(data), "--sha256", sha(z)], log=lambda m: None) == 1
    assert dsa.installed_pack(str(data)) is None
