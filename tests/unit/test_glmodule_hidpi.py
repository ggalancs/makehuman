"""Logical (Qt) pixels vs framebuffer (device) pixels on Retina displays."""

import numpy as np
import pytest

import glmodule
from core import G


class FakeCanvas(object):
    def __init__(self, ratio):
        self.ratio = ratio

    def devicePixelRatioF(self):
        return self.ratio


@pytest.fixture
def window(monkeypatch):
    def setup(width, height, ratio):
        monkeypatch.setattr(G, "windowWidth", width)
        monkeypatch.setattr(G, "windowHeight", height)
        monkeypatch.setattr(G, "canvas", FakeCanvas(ratio) if ratio else None)
    return setup


def test_ratio_defaults_to_one_without_canvas(window):
    window(100, 50, None)
    assert glmodule.devicePixelRatio() == 1.0
    assert glmodule.framebufferSize() == (100, 50)


def test_framebuffer_size_on_retina(window):
    window(100, 50, 2.0)
    assert glmodule.framebufferSize() == (200, 100)


def test_fractional_ratio_rounds(window):
    window(101, 51, 1.5)
    assert glmodule.framebufferSize() == (152, 76)


@pytest.mark.parametrize("ratio,logical,expected", [
    (1.0, (0, 0), (0, 49)),
    (1.0, (99, 49), (99, 0)),
    (2.0, (0, 0), (0, 99)),
    (2.0, (99, 49), (198, 1)),
    (2.0, (50, 25), (100, 49)),
])
def test_logical_to_framebuffer_pixel(window, ratio, logical, expected):
    window(100, 50, ratio)
    assert glmodule.toFramebufferPixel(*logical) == expected


@pytest.mark.parametrize("logical", [(-1, 0), (0, -1), (100, 0), (0, 50)])
def test_outside_the_canvas_is_none(window, logical):
    window(100, 50, 2.0)
    assert glmodule.toFramebufferPixel(*logical) is None


def test_picked_colour_comes_from_the_scaled_position(window, monkeypatch):
    window(100, 50, 2.0)
    buf = np.zeros((100, 200, 3), dtype=np.uint8)
    # logical (30, 10) -> framebuffer column 60, row 100 - 1 - 20 = 79
    buf[79, 60] = (8, 16, 24)
    monkeypatch.setattr(glmodule, "pickingBuffer", buf)
    monkeypatch.setattr(glmodule, "pickingBufferDirty", False)
    assert glmodule.getPickedColor(30, 10) == (8, 16, 24)
    assert glmodule.getPickedColor(31, 10) == (0, 0, 0)
    assert glmodule.getPickedColor(500, 10) == (0, 0, 0)
