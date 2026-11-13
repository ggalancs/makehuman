"""
Unit tests for plugins/8_asset_downloader/remoteasset.py: how one entry of
the community asset database (assets.json) becomes a RemoteAsset — generic
fields, clothes subtypes, material ownership, local file layout and the list
of files to download. Harness and fixtures: test_asset_harness.py.
"""

import os
import types

import pytest

from test_asset_harness import (  # noqa: F401  (qapp and env are pytest fixtures)
    FILES, asset_json, clothes_files, env, mhapi, qapp, remoteasset)


def make(env, raw, assetdb=None):
    parent = types.SimpleNamespace(root=env.root)  # RemoteAsset only reads parent.root
    return remoteasset.RemoteAsset(parent, raw, assetdb=assetdb)


def url(nid, name):
    return "%s/assets/%d/%s" % (FILES, nid, name)


# --------------------------------------------------------------------------
# Generic fields
# --------------------------------------------------------------------------

def test_missing_fields_get_defaults(env):
    a = make(env, {"nid": 7})
    assert a.getType() == "unknown"
    assert a.getLicense() == "unknown"
    assert a.getTitle() == "-- unknown title --"
    assert a.getDescription() == "--"
    assert a.getAuthor() == a.getUsername() == "unknown author"
    assert a.uid == -1
    assert a.getChanged() is None and a.getCreated() is None
    assert a.getId() == 7
    assert a.remoteFiles == {} and a.localFiles == {}
    assert a.getThumbPath() is None and a.getScreenshotPath() is None
    assert os.path.isdir(os.path.join(env.root, "7"))  # per-asset cache folder


def test_fields_are_read_from_the_json(env):
    raw = asset_json(12, "Red Shirt", author="bob", license="CC-BY", description="A shirt",
                     changed="2021-02-03 04:05:06", category="Shirt")
    a = make(env, raw)
    assert (a.getId(), a.getTitle(), a.getAuthor(), a.getUsername()) == (12, "Red Shirt", "bob", "bob")
    assert (a.getLicense(), a.getDescription()) == ("CC-BY", "A shirt")
    assert a.getChanged() == "2021-02-03 04:05:06"
    assert a.getCreated() == "2019-01-01 10:00:00"
    assert a.uid == 1012
    assert a.rawJson is raw


def test_existing_cache_folder_is_reused(env):
    os.makedirs(os.path.join(env.root, "8"))
    with open(os.path.join(env.root, "8", "thumb.png"), "w") as f:
        f.write("keep")
    make(env, asset_json(8, "X"))
    with open(os.path.join(env.root, "8", "thumb.png")) as f:
        assert f.read() == "keep"


# --------------------------------------------------------------------------
# Clothes subtypes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("category, kind", [
    ("Eyebrows", "eyebrows"),
    ("EYELASHES", "eyelashes"),
    ("teeth", "teeth"),
    ("Hair", "hair"),
    ("Shoes", "clothes"),
])
def test_clothes_category_is_lowercased_and_picks_the_type(env, category, kind):
    a = make(env, asset_json(20, "Thing", category=category))
    assert a.getCategory() == category.lower()
    assert a.getType() == kind


def test_clothes_without_category(env):
    a = make(env, asset_json(21, "Thing"))
    assert a.getCategory() == ""
    assert a.getType() == "clothes"


def test_hair_category_installs_in_the_hair_folder(env):
    a = make(env, asset_json(22, " Long  Bob ", category="hair", files={"mhclo": url(22, "bob.mhclo")}))
    assert a.getInstallPath() == env.data_path("hair", "Long_Bob")
    assert a.localFiles["mhclo"] == os.path.join(env.data_path("hair", "Long_Bob"), "bob.mhclo")


# --------------------------------------------------------------------------
# Files: remote names, local layout
# --------------------------------------------------------------------------

def test_file_layout(env):
    files = clothes_files(30, "shirt")
    files["render"] = url(30, "Shirt Render.PNG")
    files["diffuse"] = url(30, "shirt_diffuse.png")
    a = make(env, asset_json(30, "Red Shirt", files=files))
    install = env.data_path("clothes", "Red_Shirt")
    cache = os.path.join(env.root, "30")

    assert set(a.remoteFiles) == {"mhclo", "obj", "thumb", "screenshot", "diffuse"}
    assert a.remoteFiles["screenshot"] == files["render"]  # "render" is stored as screenshot
    assert a.localFiles == {
        "mhclo": os.path.join(install, "shirt.mhclo"),
        "obj": os.path.join(install, "shirt.obj"),
        "diffuse": os.path.join(install, "shirt_diffuse.png"),
        "thumb": os.path.join(cache, "thumb.png"),  # whatever the remote extension
        "screenshot": os.path.join(cache, "screenshot.png"),  # extension lowercased
    }
    assert a.getThumbPath() == os.path.join(cache, "thumb.png")
    assert a.getScreenshotPath() == os.path.join(cache, "screenshot.png")


def test_illustration_is_a_screenshot_too(env):
    a = make(env, asset_json(31, "Pic", files={"illustration": url(31, "pic.JPEG")}))
    assert a.remoteFiles == {"screenshot": url(31, "pic.JPEG")}
    assert a.getScreenshotPath() == os.path.join(env.root, "31", "screenshot.jpeg")


def test_converted_jpg_screenshot_is_preferred(env):
    os.makedirs(os.path.join(env.root, "32"))
    jpg = os.path.join(env.root, "32", "screenshot.jpg")
    with open(jpg, "wb") as f:
        f.write(b"jpg")
    a = make(env, asset_json(32, "Pic", files={"render": url(32, "pic.png")}))
    assert a.getScreenshotPath() == jpg


@pytest.mark.parametrize("kind, key, name, folder", [
    ("clothes", "mhclo", "a.mhclo", ("clothes", "My_Asset")),
    ("material", "mhmat", "a.mhmat", ("materials", "My_Asset")),
    ("skin", "mhmat", "a.mhmat", ("skins", "My_Asset")),
    ("pose", "bvh", "a.bvh", ("poses", "My_Asset")),
    ("expression", "mhpose", "a.mhpose", ("expressions", "My_Asset")),
    ("rig", "mhskel", "a.mhskel", ("rigs", "My_Asset")),
    ("proxy", "file", "a.proxy", ("proxymeshes",)),
    ("target", "file", "a.target", ("custom",)),
])
def test_pertinent_file_and_install_path_by_type(env, kind, key, name, folder):
    a = make(env, asset_json(40, "My Asset", type=kind, files={key: url(40, name), "thumb": url(40, "t.png")}))
    assert a.getPertinentFileName() == name
    assert a.getInstallPath() == env.data_path(*folder)
    assert a.localFiles[key] == os.path.join(env.data_path(*folder), name)


def test_model_installs_in_the_models_folder(env):
    a = make(env, asset_json(41, "My Model", type="model", files={"mhm": url(41, "m.mhm")}))
    assert a.getPertinentFileName() == "m.mhm"
    assert a.getInstallPath() == os.path.join(mhapi.locations.getUserHomePath("models"), "My_Model")


def test_pertinent_file_missing(env):
    a = make(env, asset_json(42, "No Mhclo", files={"obj": url(42, "a.obj")}))
    assert a.getPertinentFileName() is None


def test_install_path_is_computed_once(env, monkeypatch):
    a = make(env, asset_json(43, "Cached", files={"mhclo": url(43, "a.mhclo")}))
    first = a.getInstallPath()
    monkeypatch.setattr(mhapi.assets, "getAssetLocation", lambda *args: pytest.fail("not cached"))
    assert a.getInstallPath() == first


# --------------------------------------------------------------------------
# Download tuples
# --------------------------------------------------------------------------

def keys_of(asset, tuples):
    by_remote = dict((r, k) for k, r in asset.remoteFiles.items())
    for r, l in tuples:
        assert l == asset.localFiles[by_remote[r]]  # each tuple is (remote, local) of one key
    return sorted(by_remote[r] for r, l in tuples)


@pytest.mark.parametrize("kwargs, expected", [
    ({}, ["mhclo", "obj", "screenshot", "thumb"]),
    ({"onlyMeta": True}, ["screenshot", "thumb"]),
    ({"onlyMeta": True, "excludeThumb": True}, ["screenshot"]),
    ({"onlyMeta": True, "excludeScreenshot": True}, ["thumb"]),
    ({"excludeThumb": True, "excludeScreenshot": True}, ["mhclo", "obj"]),
])
def test_download_tuples_filters(env, kwargs, expected):
    a = make(env, asset_json(50, "Shirt", files=clothes_files(50, "shirt")))
    assert keys_of(a, a.getDownloadTuples(**kwargs)) == expected


def test_download_tuples_skip_existing_files_unless_asked(env):
    a = make(env, asset_json(51, "Shirt", files=clothes_files(51, "shirt")))
    os.makedirs(a.getInstallPath())
    for key in ("thumb", "mhclo"):
        with open(a.localFiles[key], "w") as f:
            f.write("x")
    assert keys_of(a, a.getDownloadTuples()) == ["obj", "screenshot"]
    assert keys_of(a, a.getDownloadTuples(ignoreExisting=False)) == ["mhclo", "obj", "screenshot", "thumb"]


def test_download_tuples_of_an_asset_without_files(env):
    assert make(env, asset_json(52, "Empty")).getDownloadTuples() == []


# --------------------------------------------------------------------------
# Materials: where they are installed
# --------------------------------------------------------------------------

def material(nid, belongs_to=None, title="Blue Fabric"):
    extra = {} if belongs_to is None else {"belongs_to": belongs_to}
    return asset_json(nid, title, type="material", files={"mhmat": url(nid, "fabric.mhmat")}, **extra)


def test_material_without_owner(env):
    a = make(env, material(60))
    assert a.belongs_to_metadata == {"belonging_is_assigned": False}
    assert a.getInstallPath() == env.data_path("materials", "Blue_Fabric")


def test_material_not_assigned_ignores_the_core_asset(env):
    a = make(env, material(61, {"belonging_is_assigned": False, "belongs_to_core_asset": "clothes/fedora01"}))
    assert a.belongs_to_metadata["belonging_is_assigned"] is False
    assert a.getInstallPath() == env.data_path("materials", "Blue_Fabric")


def test_material_for_a_core_asset_installs_next_to_it(env):
    a = make(env, material(62, {"belonging_is_assigned": True, "belongs_to_core_asset": " clothes/fedora01 "}))
    target = os.path.abspath(env.data_path("clothes", "fedora01"))
    assert a.getInstallPath() == target
    assert a.localFiles["mhmat"] == os.path.join(target, "fabric.mhmat")


@pytest.mark.parametrize("core", ["clothes/not_a_core_hat", "", "   "])
def test_material_for_an_unknown_core_asset_keeps_the_default_folder(env, core):
    a = make(env, material(63, {"belonging_is_assigned": True, "belongs_to_core_asset": core}))
    assert a.getInstallPath() == env.data_path("materials", "Blue_Fabric")


@pytest.mark.xfail(strict=True, raises=IndexError, reason=(
    "remoteasset.py:136 checks the core asset with `in` on the whole text of "
    "meshAssetSubdirs, a substring test: 'fedora01' passes and line 143 then "
    "takes parts[1] of a path without '/', raising IndexError while the asset "
    "database loads; partial paths such as 'clothes/fedora0' pass too and "
    "install into a folder that is not a core asset"))
def test_material_core_asset_must_be_a_whole_listed_path(env):
    a = make(env, material(64, {"belonging_is_assigned": True, "belongs_to_core_asset": "fedora01"}))
    assert a.getInstallPath() == env.data_path("materials", "Blue_Fabric")


def test_material_owned_by_a_known_asset_installs_next_to_it(env):
    owner = make(env, asset_json(70, "Owner Coat", files={"mhclo": url(70, "coat.mhclo")}))
    db = types.SimpleNamespace(assetsById={70: owner})
    a = make(env, material(71, {"belonging_is_assigned": True, "belongs_to_id": 70}), assetdb=db)
    assert a.getInstallPath() == env.data_path("clothes", "Owner_Coat")
    assert a.localFiles["mhmat"] == os.path.join(env.data_path("clothes", "Owner_Coat"), "fabric.mhmat")


def test_material_owned_by_an_unknown_asset_keeps_the_default_folder(env):
    db = types.SimpleNamespace(assetsById={})
    a = make(env, material(72, {"belonging_is_assigned": True, "belongs_to_id": 999}), assetdb=db)
    assert a.getInstallPath() == env.data_path("materials", "Blue_Fabric")
