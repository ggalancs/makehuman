"""Image loading/conversion through Qt used np.fromstring/tostring (removed in numpy 2)."""

import numpy as np
import pytest
from PyQt5 import QtGui

import image
import image_qt


@pytest.fixture
def png(tmp_path):
    qimg = QtGui.QImage(3, 2, QtGui.QImage.Format_ARGB32)
    colours = [[(255, 0, 0, 255), (0, 255, 0, 128), (0, 0, 255, 255)],
               [(10, 20, 30, 255), (40, 50, 60, 0), (255, 255, 255, 255)]]
    for y, row in enumerate(colours):
        for x, (r, g, b, a) in enumerate(row):
            qimg.setPixel(x, y, QtGui.qRgba(r, g, b, a))
    path = str(tmp_path / "t.png")
    assert qimg.save(path)
    return path, np.array(colours, dtype=np.uint8)


def test_load_returns_rgba_pixels(png):
    path, expected = png
    data = image_qt.load(path)
    assert data.shape == (2, 3, 4)
    assert (data[expected[..., 3] == 255] == expected[expected[..., 3] == 255]).all()
    assert (data[..., 3] == expected[..., 3]).all()


def test_to_qimage_round_trip(png):
    path, expected = png
    data = image_qt.load(path)
    q = image_qt.toQImage(data)
    assert (q.width(), q.height()) == (3, 2)
    c = QtGui.QColor(q.pixel(0, 0))
    assert (c.red(), c.green(), c.blue()) == (255, 0, 0)


@pytest.mark.parametrize("components", [1, 2, 3, 4])
def test_to_qimage_all_channel_counts(components):
    data = np.full((2, 2, components), 100, dtype=np.uint8)
    q = image_qt.toQImage(data)
    assert (q.width(), q.height()) == (2, 2)


@pytest.mark.parametrize("components", [3, 4])
def test_image_to_qimage(png, components):
    path, _ = png
    img = image.Image(path).convert(components) if hasattr(image.Image(path), "convert") else image.Image(path)
    q = img.toQImage()
    assert (q.width(), q.height()) == (3, 2)
