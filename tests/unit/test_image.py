"""lib/image.py and lib/image_qt.py: the Image container and its Qt back end."""

import sys

import numpy as np
import pytest
from PyQt5 import QtGui

import image
import image_qt

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


def rgba(rows):
    return image.Image(data=np.array(rows, dtype=np.uint8))


@pytest.fixture
def png(tmp_path):
    """3x2 RGBA PNG with distinct colours."""
    img = rgba([[[255, 0, 0, 255], [0, 255, 0, 128], [0, 0, 255, 255]],
                [[10, 20, 30, 255], [40, 50, 60, 255], [255, 255, 255, 0]]])
    path = str(tmp_path / "t.png")
    img.save(path)
    return path, img.data.copy()


def qcolor(qimg, x, y):
    c = QtGui.QColor.fromRgba(qimg.pixel(x, y))
    return (c.red(), c.green(), c.blue(), c.alpha())


@pytest.mark.parametrize("kwargs, shape", [
    ({}, (2, 3, 4)),
    ({"bitsPerPixel": 24}, (2, 3, 3)),
    ({"components": 1}, (2, 3, 1)),
    ({"components": 2, "bitsPerPixel": 8}, (2, 3, 2)),
])
def test_empty_image(kwargs, shape):
    img = image.Image(width=3, height=2, **kwargs)
    assert img.data.shape == shape
    assert img.size == (3, 2) and img.width == 3 and img.height == 2
    assert img.components == shape[2] and img.bitsPerPixel == 8 * shape[2]
    assert img.isEmpty
    img.markModified()
    assert not img.isEmpty


def test_empty_image_with_unsupported_depth():
    with pytest.raises(NotImplementedError):
        image.Image(width=1, height=1, bitsPerPixel=16)


def test_load_from_path_keeps_pixels_and_source(png):
    path, expected = png
    img = image.Image(path)
    assert img.sourcePath == path and not img.isEmpty
    assert img.data.shape == (2, 3, 4)
    opaque = expected[..., 3] == 255
    assert (img.data[opaque] == expected[opaque]).all()
    assert (img.data[..., 3] == expected[..., 3]).all()
    # data= also accepts a path
    assert (image.Image(data=path).data == img.data).all()


def test_copy_and_shared_construction(png):
    path, _ = png
    src = image.Image(path)
    copy = image.Image(src)
    shared = image.Image(data=src)
    src[(0, 0)] = (1, 2, 3, 4)
    assert copy[(0, 0)] != (1, 2, 3, 4)
    assert shared[(0, 0)] == (1, 2, 3, 4)


def test_from_qimage_without_alpha_has_three_channels():
    q = QtGui.QImage(2, 1, QtGui.QImage.Format_RGB32)
    q.fill(QtGui.QColor(10, 20, 30))
    img = image.Image(q)
    assert img.components == 3
    assert img[(1, 0)] == (10, 20, 30, 255)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError, match="unable to load image"):
        image.Image(str(tmp_path / "missing.png"))


def test_save_round_trip_and_thumb_is_png(tmp_path):
    img = rgba([[[1, 2, 3, 255], [4, 5, 6, 255]]])
    thumb = str(tmp_path / "x.thumb")
    img.save(thumb)
    assert open(thumb, "rb").read(8) == b"\x89PNG\r\n\x1a\n"  # .thumb is written as PNG
    assert (image.Image(thumb).data == img.data).all()


def test_save_failure_raises(tmp_path):
    with pytest.raises(RuntimeError, match="error saving image"):
        image.Image(width=1, height=1).save(str(tmp_path / "no-such-dir" / "x.png"))


@pytest.mark.parametrize("pixel, expected", [
    ((200,), (200, 200, 200, 255)),
    ((200, 128), (200, 200, 200, 128)),
    ((255, 0, 0), (255, 0, 0, 255)),
])
def test_to_qimage_grey_and_rgb(pixel, expected):
    data = np.array([[pixel] * 4] * 2, dtype=np.uint8)
    q = image.Image(data=data).toQImage()
    assert (q.width(), q.height()) == (4, 2)
    assert qcolor(q, 3, 1) == expected


def test_to_qimage_rgba_keeps_red_red():
    q = rgba([[[255, 0, 0, 255]]]).toQImage()
    assert qcolor(q, 0, 0) == (255, 0, 0, 255)


def test_to_qimage_grey_odd_width_rows_are_aligned():
    data = (np.arange(6, dtype=np.uint8).reshape(2, 3, 1) + 1) * 40
    q = image.Image(data=data).toQImage()
    assert [qcolor(q, x, 1)[0] for x in range(3)] == data[1, :, 0].tolist()


def test_resize_nearest():
    img = rgba([[[1, 1, 1, 1], [2, 2, 2, 2]],
                [[3, 3, 3, 3], [4, 4, 4, 4]]])
    big = img.resized(4, 4)
    assert big.size == (4, 4)
    assert big.data[:, :, 0].tolist() == [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
    before = img.modified
    img.resize(1, 1)
    assert img.size == (1, 1) and img.data[0, 0, 0] == 4 and img.modified >= before


def test_resize_bilinear_through_qt():
    img = image.Image(data=np.full((2, 2, 3), 100, dtype=np.uint8))
    big = img.resized(6, 4, image.FILTER_BILINEAR)
    assert big.size == (6, 4) and big.components == 4  # Qt hands back RGBA
    assert (big.data[..., :3] == 100).all()


def test_blit_copies_into_place():
    dest = image.Image(data=np.zeros((4, 4, 3), dtype=np.uint8))
    src = image.Image(data=np.full((2, 2, 3), 9, dtype=np.uint8))
    dest.blit(src, 1, 2)
    assert dest.data[..., 0].tolist() == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 9, 9, 0], [0, 9, 9, 0]]


def test_blit_rejects_other_channel_count():
    dest = image.Image(data=np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="incorrect format"):
        dest.blit(image.Image(data=np.zeros((1, 1, 4), dtype=np.uint8)), 0, 0)


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="blit clips the target region but still assigns the whole source "
                          "(image.py:233), so a source crossing the border cannot be blitted")
def test_blit_clips_at_the_border():
    dest = image.Image(data=np.zeros((4, 4, 3), dtype=np.uint8))
    dest.blit(image.Image(data=np.full((3, 3, 3), 9, dtype=np.uint8)), 2, 2)
    assert dest.data[2:, 2:, 0].tolist() == [[9, 9], [9, 9]]


def test_flips():
    img = image.Image(data=np.arange(4, dtype=np.uint8).reshape(2, 2, 1))
    assert img.flip_vertical().data[..., 0].tolist() == [[2, 3], [0, 1]]
    assert img.flip_horizontal().data[..., 0].tolist() == [[1, 0], [3, 2]]


@pytest.mark.parametrize("pixel, expected", [
    ((1, 2, 3, 4), (1, 2, 3, 4)),
    ((1, 2, 3), (1, 2, 3, 255)),
    ((7, 8), (7, 7, 7, 8)),
    ((7,), (7, 7, 7, 255)),
    ((1, 2, 3, 4, 5), None),
])
def test_getitem_per_channel_count(pixel, expected):
    img = image.Image(data=np.array([[pixel]], dtype=np.uint8))
    assert img[(0, 0)] == expected


@pytest.mark.parametrize("key, exc", [
    ((0,), TypeError), ([0, 0], TypeError), ((0.0, 0), TypeError),
    ((1, 0), IndexError), ((0, -1), IndexError),
])
def test_item_access_validation(key, exc):
    img = image.Image(data=np.zeros((1, 1, 3), dtype=np.uint8))
    with pytest.raises(exc):
        img[key]
    with pytest.raises(exc):
        img[key] = (1, 2, 3)


def test_setitem_requires_tuple_and_marks_modified():
    img = image.Image(data=np.zeros((1, 2, 3), dtype=np.uint8))
    with pytest.raises(TypeError, match="tuple expected"):
        img[(1, 0)] = [1, 2, 3]
    img.modified = 0
    img[(1, 0)] = (4, 5, 6)
    assert img[(1, 0)] == (4, 5, 6, 255) and img.modified > 0


@pytest.mark.parametrize("src, components, expected", [
    ((30, 60, 90, 128), 3, (30, 60, 90)),
    ((30, 60, 90), 1, (60,)),            # mean of the colour channels
    ((30, 60, 90, 128), 2, (60, 128)),   # keeps alpha
    ((50,), 3, (50, 50, 50)),
    ((50,), 2, (50, 255)),               # opaque alpha added
    ((50, 7), 4, (50, 50, 50, 7)),
    ((1, 2, 3), 4, (1, 2, 3, 255)),
])
def test_convert(src, components, expected):
    img = image.Image(data=np.array([[src]], dtype=np.uint8))
    out = img.convert(components)
    assert out.components == components
    assert tuple(out.data[0, 0].tolist()) == expected


def test_convert_to_same_format_returns_self():
    img = image.Image(width=1, height=1)
    assert img.convert(4) is img


def test_is_qpixmap_without_qt_loaded(monkeypatch):
    monkeypatch.delitem(sys.modules, "PyQt5")
    assert image._isQPixmap(object()) is False


def test_image_qt_load_rejects_null_qimage():
    with pytest.raises(RuntimeError, match="unable to load image"):
        image_qt.load(QtGui.QImage())


@pytest.mark.parametrize("components", [1, 2, 3, 4])
def test_image_qt_to_qimage_channel_layout(components):
    px = {1: (90,), 2: (90, 200), 3: (10, 20, 30), 4: (10, 20, 30, 200)}[components]
    data = np.array([[px, px]], dtype=np.uint8)
    q = image_qt.toQImage(data)
    expected = {1: (90, 90, 90, 255), 2: (90, 90, 90, 200),
                3: (10, 20, 30, 255), 4: (10, 20, 30, 200)}[components]
    assert qcolor(q, 1, 0) == expected


def test_image_qt_save_and_load_round_trip(tmp_path):
    data = np.array([[[10, 20, 30, 255], [200, 100, 50, 255]]], dtype=np.uint8)
    path = str(tmp_path / "rt.png")
    image_qt.save(path, data)
    assert (image_qt.load(path) == data).all()


def test_image_qt_resized_fast_and_smooth():
    img = image.Image(data=np.full((2, 2, 3), 80, dtype=np.uint8))
    fast = image_qt.resized(img, 4, 2)
    smooth = image_qt.resized(img, 4, 2, filter=image.FILTER_BILINEAR)
    for out in (fast, smooth):
        assert out.shape[:2] == (2, 4) and (out[..., :3] == 80).all()
