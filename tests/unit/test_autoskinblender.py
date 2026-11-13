"""apps/autoskinblender.py: skin litsphere and colour follow the ethnic sliders."""

import numpy as np
import pytest

import autoskinblender


class Ethnic(object):
    """Only the three getters the blender reads from the human."""

    def __init__(self, caucasian, african, asian):
        self.values = [caucasian, african, asian]

    def getCaucasian(self):
        return self.values[0]

    def getAfrican(self):
        return self.values[1]

    def getAsian(self):
        return self.values[2]


@pytest.fixture
def blender(monkeypatch):
    import os
    monkeypatch.chdir(os.path.join(os.path.dirname(__file__), "..", "..", "makehuman"))
    return lambda c, af, asn: autoskinblender.EthnicSkinBlender(Ethnic(c, af, asn))


def pixels(img):
    return np.asarray(img.data, dtype=np.float64)


def test_single_ethnicity_uses_its_own_litsphere(blender):
    b = blender(0.0, 1.0, 0.0)
    tex = b.getLitsphereTexture()
    assert tex is b.skinCache["african"]
    assert tex.sourcePath == "data/litspheres/adaptive_skin_tone.png"
    np.testing.assert_allclose(b.getDiffuseColor().values, autoskinblender.africanColor, rtol=1e-6)


def test_two_ethnicities_mix_the_litspheres(blender):
    b = blender(0.5, 0.0, 0.5)
    expected = 0.5 * pixels(b.skinCache["caucasian"]) + 0.5 * pixels(b.skinCache["asian"])
    np.testing.assert_allclose(pixels(b.getLitsphereTexture()), expected, atol=1.0)
    np.testing.assert_allclose(b.getDiffuseColor().values,
                               0.5 * autoskinblender.caucasianColor + 0.5 * autoskinblender.asianColor,
                               rtol=1e-5)


def test_three_ethnicities(blender):
    third = 1.0 / 3
    b = blender(third, third, third)
    tex = pixels(b.getLitsphereTexture())
    caucasian = pixels(b.skinCache["caucasian"])
    african = pixels(b.skinCache["african"])
    asian = pixels(b.skinCache["asian"])
    expected = third * caucasian + third * african + third * asian
    assert np.abs(tex - expected).mean() < 1.5  # 8-bit rounding in two mixing steps
    assert not np.allclose(tex, caucasian)


def test_texture_is_only_rebuilt_when_ethnicity_changes(blender):
    b = blender(1.0, 0.0, 0.0)
    first = b.getLitsphereTexture()
    assert b.checkUpdate() is False
    assert b.getLitsphereTexture() is first
    b.human.values = [0.0, 0.0, 1.0]
    assert b.getLitsphereTexture() is b.skinCache["asian"]
    np.testing.assert_allclose(b.getDiffuseColor().values, autoskinblender.asianColor, rtol=1e-6)
    assert b.checkUpdate() is False
