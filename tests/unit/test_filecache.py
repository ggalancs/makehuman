"""lib/filecache.py: the metadata cache behind the asset libraries."""

import logging
import os
import pickle

import pytest

import filecache
import getpath


def touch(path, text="x", mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(str(path), (mtime, mtime))
    return path


def key(path):
    return getpath.canonicalPath(str(path))


class Reader(object):
    """getMetadata callback: (name, tags) read from the file, counting calls."""

    def __init__(self):
        self.read = []

    def __call__(self, filename):
        self.read.append(os.path.basename(filename))
        with open(filename, encoding="utf-8") as f:
            name, _, tags = f.read().partition(";")
        return (name, set(tags.split(",")) if tags else set())


# --- FileCache ---------------------------------------------------------------------------

def test_update_reads_each_file_once(tmp_path):
    a = touch(tmp_path / "lib" / "a.mhclo", "Alpha;red,big", mtime=1000)
    b = touch(tmp_path / "lib" / "sub" / "b.mhclo", "Beta;", mtime=1000)
    touch(tmp_path / "lib" / "ignored.txt")
    cache = filecache.FileCache(str(tmp_path / "c.cache"))
    reader = Reader()
    cache.update(str(tmp_path / "lib"), ".MHCLO", reader)
    assert sorted(reader.read) == ["a.mhclo", "b.mhclo"]
    assert cache[key(a)] == (1000, "Alpha", {"red", "big"})
    assert cache.getMetadata(str(b)) == (1000, "Beta", set())
    assert key(a) in cache and len(cache) == 2
    assert sorted(cache.keys()) == sorted([key(a), key(b)])
    assert dict(cache.items())[key(b)][1] == "Beta"
    cache.update([str(tmp_path / "lib")], ["mhclo"], reader)
    assert len(reader.read) == 2  # nothing changed, nothing re-read


def test_update_rereads_changed_files_and_drops_removed_ones(tmp_path):
    a = touch(tmp_path / "a.mhclo", "Alpha;", mtime=1000)
    b = touch(tmp_path / "b.mhclo", "Beta;", mtime=1000)
    cache = filecache.FileCache(str(tmp_path / "c.cache"))
    reader = Reader()
    cache.update(str(tmp_path), "mhclo", reader)
    touch(a, "Alpha2;new", mtime=2000)
    b.unlink()
    cache.update(str(tmp_path), "mhclo", reader, removeOldEntries=False)
    assert cache[key(a)] == (2000, "Alpha2", {"new"})
    assert key(b) in cache  # kept on request
    cache.update(str(tmp_path), "mhclo", reader)
    assert key(b) not in cache and len(cache) == 1


def test_newer_alternative_file_overrides(tmp_path):
    """With several extensions, the first is preferred unless another one is newer."""
    touch(tmp_path / "hair.mhpxy", "Binary;", mtime=1000)
    touch(tmp_path / "hair.mhclo", "Text;", mtime=2000)
    touch(tmp_path / "old.mhpxy", "OldBinary;", mtime=3000)
    touch(tmp_path / "old.mhclo", "OldText;", mtime=1000)
    cache = filecache.FileCache(str(tmp_path / "c.cache"))
    cache.update(str(tmp_path), ["mhpxy", "mhclo"], Reader())
    assert cache.getMetadata(str(tmp_path / "hair.mhpxy")) == (2000, "Text", set())
    assert cache.getMetadata(str(tmp_path / "old.mhpxy")) == (3000, "OldBinary", set())


def test_metadata_file_can_differ_from_the_asset(tmp_path):
    asset = touch(tmp_path / "pose.bvh", "not metadata", mtime=1000)
    touch(tmp_path / "pose.meta", "Pose;", mtime=5000)
    cache = filecache.FileCache(str(tmp_path / "c.cache"))
    cache.get_metadata_filename = lambda f: os.path.splitext(f)[0] + ".meta"
    assert cache.getMetadataFile(str(asset)).endswith("pose.meta")
    cache.update(str(tmp_path), "bvh", lambda f: ("named", f))
    assert cache[key(asset)][0] == 5000  # modification time of the metadata file


def test_unreadable_file_is_reported_and_skipped(tmp_path, caplog):
    good = touch(tmp_path / "good.mhclo", "Good;")
    bad = touch(tmp_path / "bad.mhclo", "Bad;")

    def reader(filename):
        if filename.endswith("bad.mhclo"):
            raise ValueError("broken header")
        return ("Good", set())

    cache = filecache.FileCache(str(tmp_path / "c.cache"))
    with caplog.at_level(logging.ERROR):
        cache.update(str(tmp_path), "mhclo", reader)
    assert key(good) in cache and key(bad) not in cache
    assert any("bad.mhclo" in r.getMessage() for r in caplog.records)


def test_save_and_load_round_trip(tmp_path):
    a = touch(tmp_path / "a.mhclo", "Alpha;x")
    path = str(tmp_path / "c.cache")
    cache = filecache.FileCache(path, version=7)
    cache.update(str(tmp_path), "mhclo", Reader())
    cache.get_metadata_filename = lambda f: f  # not pickled
    filecache.saveCache(cache)
    loaded = filecache.loadCache(path, 7)
    assert loaded.version == 7 and loaded.filepath == path
    assert loaded[key(a)][1:] == ("Alpha", {"x"})


def test_load_drops_caches_of_another_version(tmp_path, caplog):
    path = str(tmp_path / "c.cache")
    old = filecache.FileCache(path, version=1)
    old._cache["x"] = (1, "y")
    old.save()
    with caplog.at_level(logging.DEBUG):
        fresh = filecache.loadCache(path, expected_version=2)
    assert fresh.version == 2 and len(fresh) == 0
    assert any("different version" in r.getMessage() for r in caplog.records)
    assert len(filecache.loadCache(path)) == 1  # default version is CACHE_FORMAT_VERSION (1)


@pytest.mark.parametrize("content", [b"not a pickle", None])
def test_load_starts_over_on_bad_or_missing_files(tmp_path, content):
    path = tmp_path / "c.cache"
    if content is not None:
        path.write_bytes(content)
    cache = filecache.loadCache(str(path))
    assert isinstance(cache, filecache.FileCache)
    assert len(cache) == 0 and cache.filepath == str(path)
    assert cache.version == filecache.CACHE_FORMAT_VERSION


def test_cleanup_keeps_existing_entries(tmp_path):
    keep = touch(tmp_path / "keep.mhclo")
    cache = filecache.FileCache(str(tmp_path / "c.cache"))
    cache._cache = {key(keep): (1, "k")}
    cache.cleanup()
    assert list(cache.keys()) == [key(keep)]


CLEANUP_BUG = ("FileCache.cleanup (filecache.py:81-84) deletes from the dict it is iterating: "
               "any missing file raises 'dictionary changed size during iteration'")


def test_cleanup_drops_missing_entries(tmp_path):
    keep = touch(tmp_path / "keep.mhclo")
    cache = filecache.FileCache(str(tmp_path / "c.cache"))
    cache._cache = {key(keep): (1, "k"), str(tmp_path / "gone.mhclo"): (1, "g")}
    cache.cleanup()
    assert list(cache.keys()) == [key(keep)]


def test_library_stores_its_cache_after_an_asset_was_deleted(home):
    """What happens at exit (onUnload) after the user deleted an asset."""
    lib_dir = home / "assets"
    touch(lib_dir / "hat.mhclo", "Hat;")
    cape = touch(lib_dir / "cape.mhclo", "Cape;")
    lib = Library(str(lib_dir))
    lib.getName(str(cape))
    cape.unlink()
    lib.onUnload()
    stored = filecache.loadCache(os.path.join(getpath.getPath("cache"), "lib.cache"))
    assert len(stored) == 1


# --- MetadataCacher -------------------------------------------------------------------------

class Library(filecache.MetadataCacher):
    def __init__(self, folder, cache_name="lib.cache"):
        super(Library, self).__init__(["mhclo"], cache_name)
        self.folder = folder
        self.reader = Reader()

    def getMetadataImpl(self, filename):
        return self.reader(filename)

    def getSearchPaths(self):
        return [self.folder]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MH_HOME_LOCATION", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return tmp_path


def test_base_class_must_be_specialised():
    base = filecache.MetadataCacher(["x"], "x.cache")
    with pytest.raises(NotImplementedError):
        base.getMetadataImpl("f")
    with pytest.raises(NotImplementedError):
        base.getSearchPaths()
    assert base.getMetadataFile("f.x") == "f.x"
    assert base.getFileExtensions() == ["x"]
    assert base.getNameFromMetadata(("n", {"t"})) == "n"
    assert base.getTagsFromMetadata(("n", {"t"})) == {"t"}


def test_library_reads_names_and_tags(home):
    lib_dir = home / "assets"
    touch(lib_dir / "hat.mhclo", "Hat;head,red")
    touch(lib_dir / "shoe.mhclo", "Shoe;feet,red")
    lib = Library(str(lib_dir))
    assert lib.getName(str(lib_dir / "hat.mhclo")) == "Hat"
    assert lib.getTags(str(lib_dir / "shoe.mhclo")) == {"feet", "red"}
    assert lib.getTags() == {"head", "feet", "red"}  # all tags
    assert sorted(lib.reader.read) == ["hat.mhclo", "shoe.mhclo"]


def test_library_lazily_adds_files_outside_its_paths(home):
    lib_dir = home / "assets"
    touch(lib_dir / "hat.mhclo", "Hat;")
    extra = touch(home / "elsewhere" / "cape.mhclo", "Cape;back")
    lib = Library(str(lib_dir))
    assert lib.getTags(str(extra)) == {"back"}
    assert lib.getName(str(lib_dir / "hat.mhclo")) == "Hat"


def test_library_refreshes_stale_entries(home):
    lib_dir = home / "assets"
    hat = touch(lib_dir / "hat.mhclo", "Hat;", mtime=1000)
    lib = Library(str(lib_dir))
    assert lib.getName(str(hat)) == "Hat"
    touch(hat, "Better Hat;", mtime=2000)
    assert lib.getName(str(hat)) == "Better Hat"


def test_library_unknown_file(home, caplog):
    lib = Library(str(home / "assets"))
    with caplog.at_level(logging.WARNING):
        assert lib.getName(str(home / "nope.mhclo")) == ""
        assert lib.getTags(str(home / "nope.mhclo")) == set()
    assert any("Could not get metadata" in r.getMessage() for r in caplog.records)


def test_library_cache_survives_a_restart(home):
    lib_dir = home / "assets"
    touch(lib_dir / "hat.mhclo", "Hat;head")
    lib = Library(str(lib_dir))
    lib.getName(str(lib_dir / "hat.mhclo"))
    lib.onUnload()
    stored = os.path.join(getpath.getPath("cache"), "lib.cache")
    assert os.path.isfile(stored)
    with open(stored, "rb") as f:
        assert len(pickle.load(f)) == 1
    again = Library(str(lib_dir))
    assert again.getName(str(lib_dir / "hat.mhclo")) == "Hat"
    assert again.reader.read == []  # served from the stored cache
    again.updateFileCache()  # explicit refresh with the defaults: nothing changed
    assert again.reader.read == []


def test_empty_library_stores_nothing(home):
    lib = Library(str(home / "assets"))
    lib.storeCache()  # never loaded
    lib.loadCache()
    lib.storeCache()  # loaded but empty
    assert not os.path.exists(getpath.getPath("cache"))
