"""A material whose texture file is missing must not abort an export."""

import logging
import os

from PyQt5 import QtGui

import material
from core import G


class FakeApp(object):
    def progress(self, *args, **kwargs):
        pass


def write_png(path):
    img = QtGui.QImage(2, 2, QtGui.QImage.Format_RGB32)
    img.fill(0xff336699)
    assert img.save(str(path))


def test_export_textures_skips_missing_files(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(G, "app", FakeApp())  # Progress reports to the app's progress bar
    src = tmp_path / "asset"
    src.mkdir()
    write_png(src / "diffuse.png")
    mhmat = src / "thing.mhmat"
    mhmat.write_text("name thing\n"
                     "diffuseTexture diffuse.png\n"
                     "normalmapTexture missing_normal.png\n", encoding="utf-8")
    mat = material.fromFile(str(mhmat))
    # Point the normal map at a file that does not exist, as in incomplete community assets
    mat._normalMap = str(src / "missing_normal.png")
    out = tmp_path / "export" / "textures"
    with caplog.at_level(logging.WARNING):
        textures = mat.exportTextures(str(out))
    assert "normalmapTexture" not in textures
    assert os.path.basename(textures["diffuseTexture"]) == "diffuse.png"
    assert (out / "diffuse.png").is_file()
    assert any("missing_normal.png" in r.getMessage() for r in caplog.records)
