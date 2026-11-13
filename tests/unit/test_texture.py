"""lib/texture.py: OpenGL texture objects and the path-keyed texture cache.

There is no OpenGL context in unit tests, and a real GL call without one
crashes the process (segfault), so every GL entry point texture.py uses is
replaced by a recorder. The recorders stay installed until every Texture made
by a test is gone: Texture.__del__ calls glDeleteTextures."""

import gc
import os
import sys
import types

import numpy as np
import pytest
from OpenGL.GL import (GL_ALPHA, GL_ALPHA8, GL_LINEAR, GL_PROXY_TEXTURE_1D,
                       GL_PROXY_TEXTURE_2D, GL_REPEAT, GL_RGB, GL_RGBA, GL_TEXTURE_1D,
                       GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
                       GL_TEXTURE_WIDTH, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
                       GL_UNPACK_ALIGNMENT, GL_UNSIGNED_BYTE)

import debugdump
import image
import texture

RECORDED = ("glPixelStorei", "glBindTexture", "glTexImage1D", "glTexImage2D",
            "gluBuild1DMipmaps", "gluBuild2DMipmaps", "glTexParameteri",
            "glTexSubImage1D", "glTexSubImage2D")


class InertTexture(texture.Texture):
    """Textures still alive when a test ends (e.g. held by a failure
    traceback) are switched to this class before the recorders go away."""

    def __del__(self):
        pass


class FakeGL(object):
    def __init__(self):
        self.calls = []
        self.next_id = 100
        self.npot = True          # GL_ARB_texture_non_power_of_two available
        self.proxy_width = 64     # what the proxy-texture query answers
        self.npot_queries = 0
        self.deleted = []
        self.delete_error = None
        self.dump = []

    def recorder(self, name):
        def call(*args):
            self.calls.append((name,) + args)
        return call

    def gen(self, n):
        self.next_id += 1
        return self.next_id

    def delete(self, textureId):
        if self.delete_error:
            raise self.delete_error
        self.deleted.append(textureId)

    def proxy(self, target, level, pname):
        self.calls.append(("glGetTexLevelParameteriv", target, level, pname))
        return self.proxy_width

    def query_npot(self):
        self.npot_queries += 1
        return self.npot

    def names(self):
        return [c[0] for c in self.calls]

    def args(self, name):
        return [c[1:] for c in self.calls if c[0] == name]


@pytest.fixture(autouse=True)
def gl(monkeypatch):
    fake = FakeGL()
    for name in RECORDED:
        monkeypatch.setattr(texture, name, fake.recorder(name))
    monkeypatch.setattr(texture, "glGenTextures", fake.gen)
    monkeypatch.setattr(texture, "glDeleteTextures", fake.delete)
    monkeypatch.setattr(texture, "glGetTexLevelParameteriv", fake.proxy)
    monkeypatch.setattr(texture, "glInitTextureNonPowerOfTwoARB", fake.query_npot)
    monkeypatch.setattr(texture.Texture, "_npot", None)
    monkeypatch.setattr(texture.Texture, "_powers", None)
    monkeypatch.setattr(texture, "_textureCache", {})
    monkeypatch.setattr(debugdump, "dump", types.SimpleNamespace(appendMessage=fake.dump.append))
    yield fake
    # Runs before monkeypatch restores the real GL functions
    fake.delete_error = None
    texture._textureCache.clear()
    gc.collect()
    for obj in gc.get_objects():
        if type(obj) is texture.Texture:
            obj.__class__ = InertTexture


@pytest.fixture
def logged(monkeypatch):
    """Messages sent to texture.log, by level."""
    out = {"error": [], "warning": [], "notice": [], "message": [], "debug": []}
    for level in out:
        monkeypatch.setattr(texture.log, level,
                            lambda msg, *args, _l=level, **kw: out[_l].append(msg % args if args else msg))
    return out


def rgba(rows):
    return image.Image(data=np.array(rows, dtype=np.uint8))


# 3x2 RGBA with distinct pixels: row 0 red/green/blue, row 1 greys
PIXELS = [[[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
          [[10, 10, 10, 255], [20, 20, 20, 255], [30, 30, 30, 255]]]


def png(tmp_path, rows=PIXELS, name="t.png"):
    path = str(tmp_path / name)
    rgba(rows).save(path)
    return path


def uploaded(fake, name):
    """Pixel array of the last upload call of that name (last argument)."""
    return fake.args(name)[-1][-1]


# --- object life cycle ----------------------------------------------------------------

@pytest.mark.parametrize("npot, word", [(True, "enabled"), (False, "not available")])
def test_first_texture_queries_npot_support_once(gl, npot, word):
    gl.npot = npot
    a, b = texture.Texture(), texture.Texture()
    assert gl.npot_queries == 1 and texture.Texture._npot is npot
    assert gl.dump == ["GL.EXTENSION: GL_ARB_texture_non_power_of_two %s" % word]
    assert texture.Texture._powers == [2 ** i for i in range(20)]
    assert (a.textureId, b.textureId) == (101, 102)
    assert (a.width, a.height, a.modified) == (0, 0, None)
    assert gl.calls == []  # no image, no size: nothing uploaded


def test_debug_dump_failure_is_only_logged(gl, logged, monkeypatch):
    monkeypatch.setattr(debugdump, "dump", None)
    t = texture.Texture()
    assert t.textureId == 101
    assert len(logged["error"]) == 1
    assert logged["error"][0].startswith("Failed to write GL debug info to debug dump:")


def test_deleting_a_texture_frees_its_gl_name(gl):
    t = texture.Texture()
    del t
    gc.collect()
    assert gl.deleted == [101]


def test_gl_errors_while_deleting_are_swallowed(gl, monkeypatch):
    raised = []
    monkeypatch.setattr(sys, "unraisablehook", raised.append)
    gl.delete_error = RuntimeError("context gone")
    t = texture.Texture()
    del t
    gc.collect()
    assert raised == []


@pytest.mark.parametrize("components, expected", [
    (1, (GL_ALPHA8, GL_ALPHA)), (3, (3, GL_RGB)), (4, (4, GL_RGBA))])
def test_get_format(components, expected):
    assert texture.Texture.getFormat(components) == expected


@pytest.mark.parametrize("components", [2, 5])
def test_get_format_rejects_other_layouts(components):
    with pytest.raises(RuntimeError, match="Unsupported pixel format"):
        texture.Texture.getFormat(components)


# --- allocation and upload ------------------------------------------------------------

LINEAR_REPEAT = [(GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                 (GL_TEXTURE_WRAP_S, GL_REPEAT), (GL_TEXTURE_WRAP_T, GL_REPEAT)]


def test_sized_texture_allocates_zeroed_2d_storage(gl):
    t = texture.Texture(size=(4, 2), components=3)
    assert gl.names() == ["glPixelStorei", "glBindTexture", "glTexImage2D",
                          "glGetTexLevelParameteriv", "glTexImage2D"] + ["glTexParameteri"] * 4 + ["glBindTexture"]
    assert gl.args("glPixelStorei") == [(GL_UNPACK_ALIGNMENT, 1)]
    assert gl.args("glBindTexture") == [(GL_TEXTURE_2D, 101), (GL_TEXTURE_2D, 0)]
    proxy, real = gl.args("glTexImage2D")
    assert proxy[:8] == (GL_PROXY_TEXTURE_2D, 0, 3, 4, 2, 0, GL_RGB, GL_UNSIGNED_BYTE)
    assert real[:8] == (GL_TEXTURE_2D, 0, 3, 4, 2, 0, GL_RGB, GL_UNSIGNED_BYTE)
    assert real[8].dtype == np.uint8 and real[8].size == 4 * 2 * 3 and not real[8].any()
    assert gl.args("glGetTexLevelParameteriv") == [(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)]
    assert gl.args("glTexParameteri") == [(GL_TEXTURE_2D,) + p for p in LINEAR_REPEAT]
    assert (t.width, t.height) == (4, 2)


def test_image_is_uploaded_bottom_row_first(gl):
    img = rgba(PIXELS)
    t = texture.Texture(img)
    real = gl.args("glTexImage2D")[-1]
    assert real[:8] == (GL_TEXTURE_2D, 0, 4, 3, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE)
    assert np.array_equal(real[8], np.array(PIXELS[::-1], dtype=np.uint8))
    assert (t.width, t.height) == (3, 2)
    assert "gluBuild2DMipmaps" not in gl.names()


def test_npot_size_without_npot_support_cannot_be_allocated_empty(gl):
    gl.npot = False
    t = texture.Texture()
    with pytest.raises(RuntimeError, match="Non-power-of-two textures not supported"):
        t.initTexture(3, 2)
    assert (t.width, t.height) == (0, 0)
    assert "glTexImage2D" not in gl.names()


def test_npot_image_without_npot_support_is_uploaded_as_mipmaps(gl, logged):
    gl.npot = False
    t = texture.Texture(rgba(PIXELS))
    assert "glTexImage2D" not in gl.names() and "glGetTexLevelParameteriv" not in gl.names()
    (args,) = gl.args("gluBuild2DMipmaps")
    assert args[:6] == (GL_TEXTURE_2D, 4, 3, 2, GL_RGBA, GL_UNSIGNED_BYTE)
    assert np.array_equal(args[6], np.array(PIXELS[::-1], dtype=np.uint8))
    assert (t.width, t.height) == (3, 2)
    assert any("Non-power-of-two" in m for m in logged["debug"])


def test_power_of_two_image_without_npot_support_is_uploaded_directly(gl):
    gl.npot = False
    texture.Texture(rgba([[[1, 2, 3, 4]] * 4] * 2))  # 4x2
    assert "gluBuild2DMipmaps" not in gl.names()
    assert gl.args("glTexImage2D")[-1][:5] == (GL_TEXTURE_2D, 0, 4, 4, 2)


def test_too_large_for_the_driver_falls_back_to_mipmaps(gl, logged):
    gl.proxy_width = 0
    texture.Texture(rgba(PIXELS))
    assert [c[0] for c in gl.args("glTexImage2D")] == [GL_PROXY_TEXTURE_2D]  # proxy only
    assert len(gl.args("gluBuild2DMipmaps")) == 1
    assert logged["notice"] == ["texture size (3 x 2) too large, building mipmaps"]


ROW = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]]


def test_one_row_image_becomes_a_1d_texture(gl):
    t = texture.Texture(rgba(ROW))
    assert gl.args("glBindTexture") == [(GL_TEXTURE_1D, 101), (GL_TEXTURE_1D, 0)]
    proxy, real = gl.args("glTexImage1D")
    assert proxy[:7] == (GL_PROXY_TEXTURE_1D, 0, 4, 5, 0, GL_RGBA, GL_UNSIGNED_BYTE)
    assert real[:7] == (GL_TEXTURE_1D, 0, 4, 5, 0, GL_RGBA, GL_UNSIGNED_BYTE)
    assert np.array_equal(real[7], np.array(ROW, dtype=np.uint8))
    assert gl.args("glGetTexLevelParameteriv") == [(GL_PROXY_TEXTURE_1D, 0, GL_TEXTURE_WIDTH)]
    assert gl.args("glTexParameteri") == [(GL_TEXTURE_1D,) + p for p in LINEAR_REPEAT]
    assert "glTexImage2D" not in gl.names() and (t.width, t.height) == (5, 1)


def test_1d_texture_too_large_falls_back_to_mipmaps(gl, logged):
    gl.proxy_width = 0
    texture.Texture(rgba(ROW))
    assert [c[0] for c in gl.args("glTexImage1D")] == [GL_PROXY_TEXTURE_1D]
    (args,) = gl.args("gluBuild1DMipmaps")
    assert args[:5] == (GL_TEXTURE_1D, 4, 5, GL_RGBA, GL_UNSIGNED_BYTE)
    assert logged["notice"] == ["texture size (5) too large, building mipmaps"]


def test_1d_npot_without_npot_support_is_uploaded_as_mipmaps(gl):
    gl.npot = False
    texture.Texture(rgba(ROW))
    assert "glTexImage1D" not in gl.names()
    assert len(gl.args("gluBuild1DMipmaps")) == 1


def test_load_image_from_a_path(gl, tmp_path):
    t = texture.Texture()
    t.loadImage(png(tmp_path))
    assert np.array_equal(uploaded(gl, "glTexImage2D"), np.array(PIXELS[::-1], dtype=np.uint8))
    assert (t.width, t.height) == (3, 2)


# --- sub-images -----------------------------------------------------------------------

def test_load_sub_image_2d(gl):
    t = texture.Texture(size=(8, 8))
    gl.calls.clear()
    t.loadSubImage(rgba(PIXELS), 1, 2)
    assert gl.names() == ["glBindTexture", "glTexSubImage2D", "glBindTexture"]
    assert gl.args("glBindTexture") == [(GL_TEXTURE_2D, 101), (GL_TEXTURE_2D, 0)]
    (args,) = gl.args("glTexSubImage2D")
    assert args[:8] == (GL_TEXTURE_2D, 0, 1, 2, 3, 2, GL_RGBA, GL_UNSIGNED_BYTE)
    assert np.array_equal(args[8], np.array(PIXELS[::-1], dtype=np.uint8))


def test_load_sub_image_1d_from_a_path(gl, tmp_path):
    t = texture.Texture(size=(8, 1))
    gl.calls.clear()
    t.loadSubImage(png(tmp_path, ROW), 3, 0)
    assert gl.args("glBindTexture") == [(GL_TEXTURE_1D, 101), (GL_TEXTURE_1D, 0)]
    (args,) = gl.args("glTexSubImage1D")
    assert args[:6] == (GL_TEXTURE_1D, 0, 3, 5, GL_RGBA, GL_UNSIGNED_BYTE)
    assert np.array_equal(args[6], np.array(ROW, dtype=np.uint8))


def test_load_sub_image_needs_a_gl_texture(gl):
    t = texture.Texture()
    t.textureId = 0
    with pytest.raises(RuntimeError, match="Texture is empty"):
        t.loadSubImage(rgba(PIXELS), 0, 0)
    assert gl.calls == []


# --- the texture cache: files ---------------------------------------------------------

def test_missing_file_gives_no_texture(gl, logged, tmp_path):
    path = str(tmp_path / "nope.png")
    assert texture.getTexture(path) is None
    assert logged["error"] == ["Cannot get texture for file path %s, no such file." % path]
    assert texture._textureCache == {} and gl.calls == []


def test_file_texture_is_created_once_and_cached(gl, tmp_path):
    path = png(tmp_path)
    t = texture.getTexture(path)
    assert isinstance(t, texture.Texture) and texture._textureCache == {path: t}
    assert t.modified == os.path.getmtime(path)
    assert np.array_equal(uploaded(gl, "glTexImage2D"), np.array(PIXELS[::-1], dtype=np.uint8))
    uploads = len(gl.args("glTexImage2D"))
    assert texture.getTexture(path) is t
    assert len(gl.args("glTexImage2D")) == uploads  # nothing re-uploaded


def test_file_changed_on_disk_is_reloaded_into_the_same_texture(gl, tmp_path):
    path = png(tmp_path)
    t = texture.getTexture(path)
    newer = [[[9, 9, 9, 255]] * 3] * 2
    png(tmp_path, newer)
    os.utime(path, (t.modified + 10, t.modified + 10))
    assert texture.getTexture(path) is t
    assert np.array_equal(uploaded(gl, "glTexImage2D"), np.array(newer, dtype=np.uint8))
    assert t.modified == os.path.getmtime(path)


def test_file_that_became_unreadable_is_not_returned(gl, logged, tmp_path):
    path = png(tmp_path)
    t = texture.getTexture(path)
    with open(path, "wb") as f:
        f.write(b"not an image")
    os.utime(path, (t.modified + 10, t.modified + 10))
    old = t.modified
    assert texture.getTexture(path) is None
    assert texture._textureCache[path] is t and t.modified == old  # retried next time
    assert len(logged["error"]) == 1


def test_unreadable_file_is_remembered_as_false(gl, logged, tmp_path):
    path = str(tmp_path / "bad.png")
    with open(path, "wb") as f:
        f.write(b"not an image")
    assert texture.getTexture(path) is False
    assert texture._textureCache == {path: False}
    assert logged["error"] == ["Error loading texture %s" % path]
    assert texture.getTexture(path) is False and len(logged["error"]) == 1  # not retried
    assert "glTexImage2D" not in gl.names()


def test_an_explicit_cache_is_used(gl, tmp_path):
    path = png(tmp_path)
    mine = {"other": None}
    t = texture.getTexture(path, cache=mine)
    assert mine[path] is t and path not in texture._textureCache


def test_an_empty_explicit_cache_is_used(gl, tmp_path):
    path = png(tmp_path)
    mine = {}
    t = texture.getTexture(path, cache=mine)
    assert mine.get(path) is t and path not in texture._textureCache


# --- the texture cache: in-memory images ----------------------------------------------

def test_image_without_source_path_always_gets_a_new_texture(gl, logged):
    img = rgba(PIXELS)
    a, b = texture.getTexture(img), texture.getTexture(img)
    assert a is not b and texture._textureCache == {}
    assert len(logged["warning"]) == 2 and "sourcePath" in logged["warning"][0]


def test_dynamic_image_is_cached_by_source_path_until_modified(gl):
    img = rgba(PIXELS)
    img.sourcePath = "Internal_Lightmap"
    t = texture.getTexture(img)
    assert texture._textureCache == {"Internal_Lightmap": t} and t.modified == img.modified
    uploads = len(gl.args("glTexImage2D"))
    assert texture.getTexture(img) is t and len(gl.args("glTexImage2D")) == uploads

    img.data[...] = 7
    img.modified = t.modified + 1
    assert texture.getTexture(img) is t
    assert (uploaded(gl, "glTexImage2D") == 7).all()
    assert t.modified == img.modified


def test_dynamic_image_without_modification_time_is_stamped_now(gl):
    import time
    img = rgba(PIXELS)
    img.sourcePath = "Internal_Mask"
    del img.modified
    before = time.time()
    t = texture.getTexture(img)
    assert before <= t.modified <= time.time()
    assert texture.getTexture(img) is t  # cached and, without a stamp, never reloaded


# --- reloading ------------------------------------------------------------------------

def test_reload_textures_reloads_every_cached_file(gl, logged, tmp_path):
    a, b = png(tmp_path, name="a.png"), png(tmp_path, name="b.png")
    ta, tb = texture.getTexture(a), texture.getTexture(b)
    dyn = rgba(PIXELS)
    dyn.sourcePath = "Internal_Not_A_File"
    texture.getTexture(dyn)
    png(tmp_path, [[[1, 1, 1, 255]] * 3] * 2, name="a.png")
    png(tmp_path, [[[2, 2, 2, 255]] * 3] * 2, name="b.png")
    gl.calls.clear()
    texture.reloadTextures()
    pixels = [c[-1] for c in gl.args("glTexImage2D") if c[0] == GL_TEXTURE_2D]
    assert len(pixels) == 2 and (pixels[0] == [1, 1, 1, 255]).all() and (pixels[1] == [2, 2, 2, 255]).all()
    # The in-memory image has no file to reload from: logged, not raised
    assert logged["error"] == ["Error loading texture Internal_Not_A_File"]
    assert logged["message"] == ["Reloading all textures"]


def test_reload_texture_reloads_one_file(gl, logged, tmp_path):
    a, b = png(tmp_path, name="a.png"), png(tmp_path, name="b.png")
    texture.getTexture(a), texture.getTexture(b)
    png(tmp_path, [[[5, 5, 5, 255]] * 3] * 2, name="b.png")
    gl.calls.clear()
    texture.reloadTexture(b)
    assert (uploaded(gl, "glTexImage2D") == [5, 5, 5, 255]).all()
    assert len(gl.args("glTexImage2D")) == 2  # proxy + real, only for b
    assert logged["message"] == ["Reloading texture %s" % b]


def test_reload_texture_not_in_the_cache_is_an_error(gl, logged, tmp_path):
    assert texture.reloadTexture(str(tmp_path / "x.png")) is None
    assert logged["error"] == ["Cannot reload non-existing texture %s" % (tmp_path / "x.png")]
    assert gl.calls == []


def test_reload_texture_of_a_now_unreadable_file_is_logged(gl, logged, tmp_path):
    path = png(tmp_path)
    texture.getTexture(path)
    with open(path, "wb") as f:
        f.write(b"broken")
    texture.reloadTexture(path)
    assert logged["error"] == ["Error loading texture %s" % path]


@pytest.mark.parametrize("reload_all", [True, False], ids=["reloadTextures", "reloadTexture"])
def test_reload_skips_files_that_failed_to_load(gl, tmp_path, reload_all):
    bad = str(tmp_path / "bad.png")
    with open(bad, "wb") as f:
        f.write(b"not an image")
    assert texture.getTexture(bad) is False
    good = png(tmp_path, name="good.png")
    texture.getTexture(good)
    gl.calls.clear()
    if reload_all:
        texture.reloadTextures()
        assert len(gl.args("glTexImage2D")) == 2  # good.png reloaded
    else:
        texture.reloadTexture(bad)
    assert texture._textureCache[bad] is False
