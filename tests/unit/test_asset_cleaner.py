"""
Unit tests for plugins/8_asset_downloader/assetcleaner.py. AssetCleaner is a
skeleton: no caller in the plugin uses it yet and its fix-up steps are empty.
These tests pin the contract it has today (nothing missing, no textures,
cleanAsset dispatching on the asset type), so a change to it is noticed.
Harness and fixtures: test_asset_harness.py.
"""

import types

import pytest

from test_asset_harness import (  # noqa: F401  (qapp and env are pytest fixtures)
    FILES, asset_json, assetcleaner, clothes_files, env, qapp, remoteasset)


def remote(env, nid, kind="clothes", category=None, files=None):
    extra = {} if category is None else {"category": category}
    raw = asset_json(nid, "Asset %d" % nid, type=kind, files=files or {}, **extra)
    return remoteasset.RemoteAsset(types.SimpleNamespace(root=env.root), raw)


def test_constructor_reads_the_asset(env):
    asset = remote(env, 1, files=clothes_files(1, "shirt"))
    cleaner = assetcleaner.AssetCleaner(asset)
    assert cleaner.asset is asset
    assert cleaner.assetType == "clothes"
    assert cleaner._mhmat_as_string == ""


def test_nothing_is_reported_missing_and_no_textures_are_found(env):
    cleaner = assetcleaner.AssetCleaner(remote(env, 2))
    assert cleaner.checkForMissingFiles() == []
    assert cleaner._getTextureTuples() == []
    cleaner._mhmat_as_string = None
    assert cleaner._getTextureTuples() == []
    cleaner._mhmat_as_string = "diffuseTexture shirt.png\n"
    assert cleaner._getTextureTuples() == []  # parsing is not implemented yet


def test_fix_up_steps_do_nothing(env, tmp_path):
    asset = remote(env, 3, files=clothes_files(3, "shirt"))
    cleaner = assetcleaner.AssetCleaner(asset)
    before = dict(asset.localFiles)
    for step in (cleaner._cleanMHMAT, cleaner._cleanMHCLO, cleaner._fixClothes, cleaner._fixMaterial):
        assert step() is None
    assert asset.localFiles == before


@pytest.mark.parametrize("kind, category, expected", [
    ("clothes", None, ["clothes"]),
    ("clothes", "hair", ["clothes"]),
    ("clothes", "eyebrows", ["clothes"]),
    ("clothes", "eyelashes", ["clothes"]),
    ("clothes", "teeth", ["clothes"]),
    ("material", None, ["material"]),
    ("pose", None, []),
    ("skin", None, []),
])
def test_clean_asset_dispatches_on_the_type(env, monkeypatch, kind, category, expected):
    cleaner = assetcleaner.AssetCleaner(remote(env, 4, kind=kind, category=category))
    calls = []
    monkeypatch.setattr(cleaner, "_fixClothes", lambda: calls.append("clothes"))
    monkeypatch.setattr(cleaner, "_fixMaterial", lambda: calls.append("material"))
    cleaner.cleanAsset()
    assert calls == expected
