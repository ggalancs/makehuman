"""
Unit tests for plugins/8_asset_downloader/assetdb.py: loading the remote and
local asset databases, filtering, synchronising with the (local fake of the)
community server, and downloading one asset. The server is an http.server on
127.0.0.1; the plugin's hard-coded makehumancommunity.org and tuxfamily.org
URLs are mapped to it. Harness and fixtures: test_asset_harness.py.
"""

import datetime
import json
import os
import socket

import pytest

from test_asset_harness import (  # noqa: F401  (qapp and env are pytest fixtures)
    ASSETS_JSON, FILES, SCREEN_SEED, THUMB_SEED, asset_json, assetdb, clothes_files,
    env, mhapi, png_bytes, qapp)

ALL_TYPES = [t for t in mhapi.assets.getAssetTypes() if t != "node_setups_and_blender_specific"]


def fmt(ts):
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def days_ago(days):
    return (datetime.datetime.now() - datetime.timedelta(days)).strftime('%Y-%m-%d %H:%M:%S')


def clothes(nid, title, category="Shirt", **kw):
    return asset_json(nid, title, category=category, files=clothes_files(nid, "item%d" % nid), **kw)


def material(nid, title, belongs_to=None, **kw):
    extra = {} if belongs_to is None else {"belongs_to": belongs_to}
    extra.update(kw)
    return asset_json(nid, title, type="material",
                      files={"mhmat": "%s/assets/%d/mat%d.mhmat" % (FILES, nid, nid),
                             "thumb": "%s/assets/%d/mat%d.thumb" % (FILES, nid, nid)}, **extra)


def ids(assets):
    return sorted(a.getId() for a in assets)


def subsequence(needles, haystack):
    it = iter(haystack)
    return all(any(n == h for h in it) for n in needles)


class Calls(object):
    def __init__(self):
        self.finished = []
        self.progress = []

    def on_finished(self, *args):
        self.finished.append(args)

    def on_progress(self, prog):
        self.progress.append(prog)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def test_without_any_database(env):
    db = assetdb.AssetDB(None)
    assert db.isSynchronized is False
    assert sorted(db.remoteAssets) == sorted(ALL_TYPES)
    assert all(v == {} for v in db.remoteAssets.values())
    assert db.localAssets == {}
    assert db.getKnownAuthors() == [] and db.getKnownClothesCategories() == []
    assert db.root == os.path.join(mhapi.locations.getUserDataPath(), "community-assets")
    assert not os.path.exists(db.localdb)
    assert "Remote json does not exist locally" in env.plugin_log()


def test_remote_database_is_indexed(env):
    env.write_remote_db([
        clothes(101, "Red Shirt", author="bob"),
        clothes(102, "Blue Dress", category="Dress", author="alice"),
        clothes(103, "Long Hair", category="Hair", author="carol"),
        material(104, "Felt", author="alice"),
        asset_json(105, "Shader nodes", type="node_setups_and_blender_specific", author="zed"),
    ])
    db = assetdb.AssetDB(None)
    assert db.isSynchronized is True
    assert sorted(db.remoteAssets["clothes"]) == [101, 102]
    assert sorted(db.remoteAssets["hair"]) == [103]
    assert sorted(db.remoteAssets["material"]) == [104]
    assert sorted(db.assetsById) == [101, 102, 103, 104]
    assert db.remoteAssets["clothes"][101] is db.assetsById[101]
    assert db.getKnownAuthors() == ["alice", "bob", "carol"]  # sorted, no duplicates, no zed
    assert db.getKnownClothesCategories() == ["dress", "shirt"]  # hair became its own type

    authors = db.getKnownAuthors()
    authors.append("mallory")  # a copy
    assert db.getKnownAuthors() == ["alice", "bob", "carol"]


def test_unknown_asset_type_is_rejected(env):
    env.write_remote_db([asset_json(1, "Ship", type="spaceship")])
    with pytest.raises(ValueError, match="Asset type not known: spaceship"):
        assetdb.AssetDB(None)


def test_local_database_is_rebuilt_from_installed_files(env):
    env.write_remote_db([
        clothes(101, "Red Shirt"),
        clothes(102, "Blue Dress"),
        asset_json(103, "No Mhclo", files={"obj": FILES + "/assets/103/a.obj"}),
        clothes(104, "Not Installed"),
    ])
    shirt_dir = env.data_path("clothes", "Red_Shirt")
    os.makedirs(shirt_dir)
    mhclo = os.path.join(shirt_dir, "item101.mhclo")
    with open(mhclo, "w") as f:
        f.write("mhclo")
    os.utime(mhclo, (1500000000, 1500000000))
    os.makedirs(env.data_path("clothes", "Blue_Dress"))  # folder without its mhclo
    os.makedirs(env.data_path("clothes", "No_Mhclo"))

    db = assetdb.AssetDB(None)

    expected = {"101": {"file": mhclo, "modified": fmt(1500000000)}}
    assert db.localAssets["clothes"] == expected  # read back from local.json: string keys
    on_disk = env.read_json("local.json")
    assert sorted(on_disk) == sorted(ALL_TYPES)
    assert on_disk["clothes"] == expected
    assert all(on_disk[t] == {} for t in ALL_TYPES if t != "clothes")


def test_existing_local_database_is_loaded_as_is(env):
    env.write_remote_db([clothes(101, "Red Shirt")])
    os.makedirs(env.data_path("clothes", "Red_Shirt"))
    with open(env.data_path("clothes", "Red_Shirt", "item101.mhclo"), "w") as f:
        f.write("installed, but local.json says otherwise")
    local = {"clothes": {"999": {"file": "/x", "modified": "2000-01-01 00:00:00"}}}
    env.write_local_db(local)
    db = assetdb.AssetDB(None)
    assert db.localAssets == local
    assert env.read_json("local.json") == local  # not rewritten


def test_loading_a_missing_local_database_gives_empty_types(env):
    db = assetdb.AssetDB(None)
    db._loadLocalDB()
    assert sorted(db.localAssets) == sorted(mhapi.assets.getAssetTypes())
    assert all(v == {} for v in db.localAssets.values())
    assert "Local json does not exist" in env.plugin_log()


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------

@pytest.fixture
def catalog(env):
    assets = [
        clothes(101, "Red Shirt", author="bob", license="CC0", description="A red cotton shirt", changed=days_ago(3)),
        clothes(102, "Blue Dress", category="Dress", author="alice", license="CC-BY", description="Evening dress", changed=days_ago(20)),
        clothes(103, "Green Shirt", author="alice", license="AGPL", description="Shirt with RED buttons", changed=days_ago(60)),
        clothes(104, "Old Coat", category="Coat", author="bob", changed=days_ago(200)),
        clothes(105, "Ancient Hat", category="Hat", author="carol", changed=days_ago(1000)),
        clothes(106, "Undated Hat", category="Hat", author="carol", changed=""),
        clothes(107, "Nodate Hat", category="Hat", author="carol"),
        material(201, "Fedora Felt", {"belonging_is_assigned": True, "belongs_to_core_asset": "clothes/fedora01"}),
        material(202, "Loose Fabric"),
        material(203, "Coat Lining", {"belonging_is_assigned": True, "belongs_to_id": 104}),
    ]
    del assets[6]["changed"]
    env.write_remote_db(assets)
    env.write_local_db(dict((t, {}) for t in ALL_TYPES))
    return assetdb.AssetDB(None)


@pytest.mark.parametrize("kwargs, expected", [
    ({}, [101, 102, 103, 104, 105, 106, 107]),
    ({"subtype": "shirt"}, [101, 103]),
    ({"author": "alice"}, [102, 103]),
    ({"title": "SHIRT"}, [101, 103]),
    ({"desc": "red"}, [101, 103]),
    ({"license": "CC-BY"}, [102]),
    ({"author": "bob", "title": "coat"}, [104]),
    ({"author": "nobody"}, []),
])
def test_filters(catalog, kwargs, expected):
    assert ids(catalog.getFilteredAssets("clothes", **kwargs)) == expected


@pytest.mark.parametrize("changed, expected", [
    ("One week", [101, 106, 107]),
    ("one month", [101, 102, 106, 107]),
    ("Three months", [101, 102, 103, 106, 107]),
    ("One year", [101, 102, 103, 104, 106, 107]),
    ("whenever", [101, 102, 103, 104, 105, 106, 107]),  # anything else: a century
])
def test_changed_within(catalog, changed, expected):
    assert ids(catalog.getFilteredAssets("clothes", changed=changed)) == expected


def test_material_subtype_keeps_materials_for_core_assets(catalog):
    assert ids(catalog.getFilteredAssets("material")) == [201, 202, 203]
    assert ids(catalog.getFilteredAssets("material", subtype="for core asset")) == [201]


def test_downloaded_filter(catalog):
    catalog.localAssets["clothes"]["101"] = {"file": "/x", "modified": "2000-01-01 00:00:00"}
    assert ids(catalog.getFilteredAssets("clothes", isDownloaded="yes")) == [101]
    assert ids(catalog.getFilteredAssets("clothes", isDownloaded="no")) == [102, 103, 104, 105, 106, 107]


def test_unknown_or_empty_types_give_nothing(catalog):
    assert catalog.getFilteredAssets("spaceship") == []
    assert catalog.getFilteredAssets("node_setups_and_blender_specific") == []
    assert catalog.getFilteredAssets("hair") == []


def test_download_tuples_of_the_whole_database(catalog):
    tuples = catalog.getDownloadTuples(onlyMeta=True, excludeScreenshot=True)
    expected = set()
    for nid in range(101, 108):
        expected.add((clothes_files(nid, "item%d" % nid)["thumb"], os.path.join(catalog.root, str(nid), "thumb.png")))
    for nid in (201, 202, 203):
        expected.add(("%s/assets/%d/mat%d.thumb" % (FILES, nid, nid), os.path.join(catalog.root, str(nid), "thumb.png")))
    assert set(tuples) == expected and len(tuples) == len(expected)
    assert len(catalog.getDownloadTuples()) == 7 * 4 + 3 * 2


def test_material_owned_by_an_earlier_asset_installs_next_to_it(env, catalog):
    lining = catalog.remoteAssets["material"][203]
    assert lining.getInstallPath() == env.data_path("clothes", "Old_Coat")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "remoteasset.py:146-154 resolves belongs_to_id against assetdb.assetsById, "
    "which assetdb.py:98-121 fills while it walks the JSON; a material listed "
    "before the asset it belongs to falls back to data/materials/<title>, so "
    "the install folder depends on the key order of assets.json"))
def test_material_owned_by_a_later_asset_installs_next_to_it(env):
    env.write_remote_db([
        material(50, "Coat Lining", {"belonging_is_assigned": True, "belongs_to_id": 60}),
        clothes(60, "Old Coat"),
    ])
    db = assetdb.AssetDB(None)
    assert db.remoteAssets["material"][50].getInstallPath() == env.data_path("clothes", "Old_Coat")


# --------------------------------------------------------------------------
# Synchronising with the server
# --------------------------------------------------------------------------

def server_catalog(env, assets):
    env.publish(ASSETS_JSON, json.dumps(dict((str(a["nid"]), a) for a in assets)))
    for a in assets:
        env.publish_files(a)


def sync(env, db, **kwargs):
    calls = Calls()
    db.synchronizeRemote(None, calls.on_finished, calls.on_progress, **kwargs)
    env.pump_until(lambda: calls.finished)
    return calls


def test_first_synchronization_uses_the_seed_zips(env):
    assets = [clothes(101, "Red Shirt", author="bob"), clothes(102, "Blue Dress", author="alice")]
    server_catalog(env, assets)
    seed_thumb = png_bytes("#00ff00")
    env.publish_zip(THUMB_SEED, {"101/thumb.png": seed_thumb, "remote.json": b'{"stale": 1}'})
    env.publish_zip(SCREEN_SEED, {"101/screenshot.png": b"seed screen", "remote.json": b'{"stale": 2}'})

    db = assetdb.AssetDB(None)
    assert not os.path.exists(db.root)
    calls = sync(env, db, downloadScreenshots=True)

    assert calls.finished == [()]
    a102 = clothes_files(102, "item102")
    assert env.fetcher.opened[:3] == [THUMB_SEED, SCREEN_SEED, ASSETS_JSON]
    assert sorted(env.fetcher.opened[3:]) == sorted([a102["thumb"], a102["render"]])  # 101's came in the zips

    assert env.read_json("remote.json") == dict((str(a["nid"]), a) for a in assets)
    for leftover in ("remote.json.keep", "asset-db-thumbnails.zip", "asset-db-screenshots.zip"):
        assert not os.path.exists(os.path.join(db.root, leftover))
    with open(os.path.join(db.root, "101", "thumb.png"), "rb") as f:
        assert f.read() == seed_thumb
    with open(os.path.join(db.root, "101", "screenshot.png"), "rb") as f:
        assert f.read() == b"seed screen"
    assert os.path.exists(os.path.join(db.root, "102", "thumb.png"))
    assert os.path.exists(os.path.join(db.root, "102", "screenshot.png"))

    assert db.isSynchronized
    assert sorted(db.remoteAssets["clothes"]) == [101, 102]
    assert db.getKnownAuthors() == ["alice", "bob"]
    assert sorted(env.read_json("local.json")) == sorted(ALL_TYPES)
    assert db.localAssets["clothes"] == {}

    assert subsequence([
        (0.5, "Unzipping seed zips"),
        (4 / 6.0, "Unzipping seed zips"),
        (5 / 6.0, "Checking for additional files to download"),
        (1.0, "Checking for additional files to download"),
        (0.1, "Rebuilding local asset DB"),
        (0.5, "Loading local asset DB"),
        (1.0, "Loading local asset DB"),
    ], env.app.progress_calls), env.app.progress_calls


def test_first_synchronization_without_screenshots(env):
    server_catalog(env, [clothes(101, "Red Shirt")])
    env.publish_zip(THUMB_SEED, {"999/thumb.png": b"other"})
    db = assetdb.AssetDB(None)
    calls = sync(env, db, downloadScreenshots=False)
    assert calls.finished == [()]
    assert env.fetcher.opened == [THUMB_SEED, ASSETS_JSON, clothes_files(101, "item101")["thumb"]]
    assert os.path.exists(os.path.join(db.root, "999", "thumb.png"))
    assert not os.path.exists(os.path.join(db.root, "101", "screenshot.png"))


def test_first_synchronization_without_any_pictures(env):
    server_catalog(env, [clothes(101, "Red Shirt")])
    db = assetdb.AssetDB(None)
    calls = sync(env, db, downloadScreenshots=False, downloadThumbnails=False)
    assert calls.finished == [()]
    assert env.fetcher.opened == [ASSETS_JSON]
    assert sorted(db.remoteAssets["clothes"]) == [101]
    assert not os.path.exists(os.path.join(db.root, "101", "thumb.png"))


def test_resynchronization_fetches_the_database_and_new_pictures(env):
    env.write_remote_db([clothes(101, "Red Shirt")])
    os.makedirs(os.path.join(env.root, "101"), exist_ok=True)
    with open(os.path.join(env.root, "101", "thumb.png"), "wb") as f:
        f.write(b"old thumb")
    os.makedirs(env.data_path("clothes", "Blue_Dress"))
    with open(env.data_path("clothes", "Blue_Dress", "item102.mhclo"), "w") as f:
        f.write("installed")
    db = assetdb.AssetDB(None)
    assert sorted(db.remoteAssets["clothes"]) == [101]

    server_catalog(env, [clothes(101, "Red Shirt"), clothes(102, "Blue Dress")])
    calls = sync(env, db, downloadScreenshots=False)
    assert calls.finished == [()]
    assert env.fetcher.opened == [ASSETS_JSON, clothes_files(102, "item102")["thumb"]]
    assert sorted(db.remoteAssets["clothes"]) == [101, 102]
    with open(os.path.join(env.root, "101", "thumb.png"), "rb") as f:
        assert f.read() == b"old thumb"  # existing pictures are kept
    assert list(db.localAssets["clothes"]) == ["102"]


def test_failed_resynchronization_keeps_the_previous_database(env):
    env.write_remote_db([clothes(101, "Red Shirt")])
    os.makedirs(os.path.join(env.root, "101"))
    with open(os.path.join(env.root, "101", "thumb.png"), "wb") as f:
        f.write(b"cached")  # otherwise the reloaded old database asks for it
    before = env.read_json("remote.json")
    db = assetdb.AssetDB(None)
    sync(env, db, downloadScreenshots=False)  # assets.json is not on the server: 404
    assert env.fetcher.opened == [ASSETS_JSON]
    assert env.read_json("remote.json") == before
    assert sorted(db.remoteAssets["clothes"]) == [101]


def test_synchronization_reports_a_failed_database_download(env):
    db = assetdb.AssetDB(None)
    calls = sync(env, db, downloadScreenshots=False, downloadThumbnails=False)
    assert env.fetcher.opened == [ASSETS_JSON]
    assert calls.finished == [(404, ASSETS_JSON)]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "assetdb.py:328 stores the onProgress callback of synchronizeRemote but "
    "_syncRemote1Progress (332) and _syncRemote2Progress (403) never call it; "
    "no effect in the app today because the view's _onSyncProgress is empty"))
def test_synchronization_reports_progress_to_the_caller(env):
    server_catalog(env, [clothes(101, "Red Shirt")])
    db = assetdb.AssetDB(None)
    calls = sync(env, db, downloadScreenshots=False)
    assert calls.finished == [()]
    assert calls.progress


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "assetdb.py:76-131 (_loadRemoteDB) resets remoteAssets on every load but "
    "not knownAuthors, knownClothesCategories or assetsById, so after a "
    "resynchronisation the author and subtype lists keep entries that are no "
    "longer in the database (choosing one lists nothing)"))
def test_resynchronization_forgets_authors_and_categories_that_are_gone(env):
    env.write_remote_db([clothes(101, "Red Shirt", author="alice"),
                         clothes(102, "Blue Dress", category="Dress", author="bob")])
    db = assetdb.AssetDB(None)
    assert db.getKnownAuthors() == ["alice", "bob"]
    server_catalog(env, [clothes(101, "Red Shirt", author="alice")])
    sync(env, db, downloadScreenshots=False)
    assert sorted(db.remoteAssets["clothes"]) == [101]
    assert db.getKnownAuthors() == ["alice"]
    assert db.getKnownClothesCategories() == ["shirt"]


# --------------------------------------------------------------------------
# Downloading one asset
# --------------------------------------------------------------------------

@pytest.fixture
def shirt_db(env):
    raw = clothes(101, "Red Shirt")
    env.write_remote_db([raw])
    db = assetdb.AssetDB(None)  # no local.json yet: rebuilt, nothing installed
    return db, raw


def download(env, db, asset, with_progress=True):
    calls = Calls()
    db.downloadItem(None, asset, calls.on_finished, calls.on_progress if with_progress else None)
    env.pump_until(lambda: calls.finished)
    return calls


def test_download_item_installs_the_asset(env, shirt_db):
    db, raw = shirt_db
    thumb = png_bytes("#0000ff")
    env.publish_files(raw, {"thumb": thumb})
    asset = db.remoteAssets["clothes"][101]
    calls = download(env, db, asset)

    assert calls.finished == [()]
    assert calls.progress and calls.progress[0] == 0.0
    files = raw["files"]
    assert sorted(env.fetcher.opened) == sorted([files["mhclo"], files["obj"], files["thumb"]])  # no screenshot
    install = env.data_path("clothes", "Red_Shirt")
    with open(os.path.join(install, "item101.mhclo"), "rb") as f:
        assert f.read() == b"mhclo of Red Shirt"
    with open(os.path.join(install, "item101.obj"), "rb") as f:
        assert f.read() == b"obj of Red Shirt"
    with open(os.path.join(install, "item101.thumb"), "rb") as f:
        assert f.read() == thumb  # the cached thumbnail, copied next to the mhclo
    mhclo = os.path.join(install, "item101.mhclo")
    record = {"file": mhclo, "modified": fmt(os.path.getmtime(mhclo))}
    assert db.localAssets["clothes"][101] == record
    assert env.read_json("local.json")["clothes"] == {"101": record}


def test_download_item_without_thumbnail(env):
    raw = asset_json(301, "Bare Pose", type="pose", files={"bvh": FILES + "/assets/301/walk.bvh"})
    env.write_remote_db([raw])
    db = assetdb.AssetDB(None)
    env.publish_files(raw)
    calls = download(env, db, db.remoteAssets["pose"][301], with_progress=False)
    assert calls.finished == [()]
    bvh = env.data_path("poses", "Bare_Pose", "walk.bvh")
    assert db.localAssets["pose"][301]["file"] == bvh
    assert not os.path.exists(env.data_path("poses", "Bare_Pose", "walk.thumb"))


def test_download_item_passes_http_errors_on(env, shirt_db):
    db, raw = shirt_db
    env.publish(raw["files"]["thumb"], png_bytes())  # the mhclo is missing: 404
    before = env.read_json("local.json")
    calls = download(env, db, db.remoteAssets["clothes"][101], with_progress=False)
    assert calls.finished == [(404, raw["files"]["mhclo"])]
    assert 101 not in db.localAssets["clothes"] and "101" not in db.localAssets["clothes"]
    assert env.read_json("local.json") == before


@pytest.mark.xfail(strict=True, raises=FileNotFoundError, reason=(
    "assetdb.py:465-468: when the main file did not arrive (a non-HTTP error "
    "that DownloadThread swallows and reports as 'OK', downloadtask.py:153) "
    "_downloadFinished logs 'File does not exist post download' and then "
    "calls os.path.getmtime on it anyway; the FileNotFoundError is raised in "
    "a Qt slot, where PyQt aborts MakeHuman since no excepthook is installed"))
def test_download_item_with_an_unreachable_file_reports_failure(env, shirt_db):
    db, raw = shirt_db
    env.publish_files(raw)  # every file is on the server...
    asset = db.remoteAssets["clothes"][101]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    asset.remoteFiles["mhclo"] = "http://127.0.0.1:%d/item101.mhclo" % port  # ...but the mhclo host is down
    calls = download(env, db, asset, with_progress=False)
    assert calls.finished and calls.finished[0] and calls.finished[0][0] > 0
    assert 101 not in db.localAssets["clothes"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "assetdb.py:472 records a fresh download under the integer node id, but "
    "getFilteredAssets (lines 265 and 268) looks for str(assetId), the form "
    "local.json gives after a reload; until MakeHuman restarts or resyncs, the "
    "asset just downloaded is missing from 'Already downloaded: yes' and still "
    "listed under 'no'"))
def test_downloaded_asset_is_listed_as_downloaded(env, shirt_db):
    db, raw = shirt_db
    env.publish_files(raw)
    calls = download(env, db, db.remoteAssets["clothes"][101], with_progress=False)
    assert calls.finished == [()]
    assert ids(db.getFilteredAssets("clothes", isDownloaded="yes")) == [101]
    assert ids(db.getFilteredAssets("clothes", isDownloaded="no")) == []
