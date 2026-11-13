"""shared/image_operations.py: pixel arithmetic, masks and blur used by texture baking."""

import numpy as np
import pytest

import image
import image_operations as op
import progress
from core import G


def img(rows):
    return image.Image(data=np.array(rows, dtype=np.uint8))


def grey(values):
    """2-D list of ints -> 1-channel Image."""
    return image.Image(data=np.array(values, dtype=np.uint8)[..., None])


class FakeApp(object):
    def progress(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def no_progress_stack(monkeypatch):
    monkeypatch.setattr(G, "app", FakeApp(), raising=False)
    monkeypatch.setattr(progress, "current_Progress_", None)


def test_clip_and_normalize():
    data = np.array([[[-20], [100], [300]]])
    assert op.clip(data).data[..., 0].tolist() == [[0, 100, 255]]
    assert op.clipData(data).max() == 255
    assert op.normalize(grey([[0, 50, 100]])).data[..., 0].tolist() == [[0, 127, 255]]  # 50*2.55+0.5 rounds down in float


def test_invert_keeps_alpha():
    assert op.invert(img([[[10, 20, 30]]])).data.tolist() == [[[245, 235, 225]]]
    assert op.invert(img([[[10, 20, 30, 40]]])).data.tolist() == [[[245, 235, 225, 40]]]
    assert op.invert(img([[[10, 40]]])).data.tolist() == [[[245, 40]]]


def test_mix_weights_and_channel_sync():
    a = img([[[100, 100, 100]]])
    b = grey([[200]])  # synchronised to RGB before mixing
    assert op.mix(a, b, 0.25).data.tolist() == [[[175, 175, 175]]]
    assert op.mix(a, b, 1, 1).data.tolist() == [[[300 % 256] * 3]]  # no clipping: wraps in uint8
    assert op.mix(None, None, 0.5) is None


def test_mix_with_missing_image_uses_black():
    a = grey([[100, 200], [50, 0]])  # square: see the xfail about non-square images
    assert op.mix(a, None, 0.5).data[..., 0].tolist() == [[50, 100], [25, 0]]
    assert op.mix(None, a, 0.5).data[..., 0].tolist() == [[50, 100], [25, 0]]


def test_multiply_with_missing_image_uses_white():
    a = grey([[100, 200], [255, 0]])
    b = grey([[255, 128], [128, 128]])
    assert op.multiply(a, b).data[..., 0].tolist() == [[100, 100], [128, 0]]
    assert op.multiply(a, None).data[..., 0].tolist() == a.data[..., 0].tolist()
    assert op.multiply(None, None) is None


def test_bitwise_operations():
    a = grey([[0b1100, 0b1010], [0, 255]])
    b = grey([[0b1010, 0b0110], [7, 1]])
    assert op.bitwiseAnd(a, b).data[..., 0].tolist() == [[0b1000, 0b0010], [0, 1]]
    assert op.bitwiseOr(a, b).data[..., 0].tolist() == [[0b1110, 0b1110], [7, 255]]
    assert op.bitwiseAnd(a, None).data[..., 0].tolist() == a.data[..., 0].tolist()
    assert op.bitwiseOr(None, a).data[..., 0].tolist() == a.data[..., 0].tolist()
    assert op.bitwiseAnd(None, None) is None and op.bitwiseOr(None, None) is None


def test_mix_with_missing_non_square_image():
    a = grey([[100, 200, 50]])
    out = op.mix(a, None, 0.5)
    assert out.data.shape == (1, 3, 1)
    assert out.data[..., 0].tolist() == [[50, 100, 25]]


def test_compose_takes_channel_i_from_image_i():
    r = img([[[11, 12, 13]]])
    g = img([[[21, 22, 23]]])
    b = grey([[99]])  # single-channel images contribute their only channel
    a = img([[[41, 42, 43, 44]]])
    assert op.compose([r, g, b, a]).data.tolist() == [[[11, 22, 99, 44]]]


def test_compose_rejects_images_without_the_channel():
    with pytest.raises(TypeError, match="not enough channels"):
        op.compose([grey([[1]]), grey([[2]]), img([[[1, 2]]]), img([[[1, 2]]])])


def test_compose_rgb_image_has_no_fourth_channel():
    rgb = img([[[1, 2, 3]]])
    with pytest.raises(TypeError, match="not enough channels"):
        op.compose([rgb, rgb, rgb, rgb])


def test_color_as_image():
    out = op.colorAsImage((1.0, 0.5, 0.0), width=3, height=2)
    assert out.data.shape == (2, 3, 3)
    assert out.data[1, 2].tolist() == [255, 128, 0]
    rgba = op.colorAsImage([10, 20, 30, 40, 50], width=1, height=1)  # extra channels dropped
    assert rgba.data.tolist() == [[[10, 20, 30, 40]]]


def test_color_as_image_fills_existing_image():
    target = image.Image(width=2, height=2, components=3)
    target._is_empty = True
    assert op.colorAsImage([1, 2, 3], image=target) is target
    assert (target.data == [1, 2, 3]).all() and not target.isEmpty
    with pytest.raises(TypeError, match="same amount of channels"):
        op.colorAsImage([1, 2], image=target)
    with pytest.raises(TypeError, match="Specify either"):
        op.colorAsImage([1, 2, 3])


def test_alpha_and_channel_extraction():
    a = img([[[1, 2, 3, 4]]])
    assert op.getAlpha(a).data.tolist() == [[[4]]]
    assert op.getAlpha(img([[[1, 2, 3]]])).data.tolist() == [[[255]]]
    assert op.getChannel(a, 1).data.tolist() == [[[2]]]
    assert op.getBlack(grey([[5, 5], [5, 5]])).data.tolist() == [[[0], [0]], [[0], [0]]]


def test_add_and_remove_alpha():
    rgba = img([[[1, 2, 3, 4]]])
    assert op.removeAlpha(rgba).data.tolist() == [[[1, 2, 3]]]
    assert op.removeAlpha(img([[[9, 8]]])).data.tolist() == [[[9]]]
    assert op.removeAlpha(img([[[1, 2, 3]]])).components == 3
    assert op.addAlpha(img([[[1, 2, 3]]])).data.tolist() == [[[1, 2, 3, 255]]]
    assert op.addAlpha(grey([[7]])).data.tolist() == [[[7, 255]]]
    assert op.addAlpha(img([[[1, 2, 3]]]), grey([[77]])).data.tolist() == [[[1, 2, 3, 77]]]


def test_resized_and_synchronize_channels():
    a = grey([[1, 2], [3, 4]])
    assert op.resized(a, 1, 1).data.tolist() == [[[4]]]
    x, y = op.synchronizeChannels(img([[[1, 2, 3, 4]]]), grey([[5]]))
    assert x.components == y.components == 4
    x, y = op.synchronizeChannels(grey([[5]]), img([[[1, 2, 3]]]))
    assert x.components == y.components == 3


def test_jitter_sum_adds_the_3x3_neighbourhood():
    data = np.eye(3, dtype=np.uint8)[..., None]
    # rolls wrap around the borders, so every pixel sees the whole diagonal
    assert op.jitterSum(data)[..., 0].tolist() == [[3, 3, 3]] * 3
    single = np.zeros((5, 5, 1), dtype=np.uint8)
    single[2, 2] = 1
    out = op.jitterSum(single, step=2)[..., 0]
    assert out[0, 0] == out[2, 4] == out[4, 2] == 1 and out[2, 2] == 1 and out.sum() == 9


def test_grow_and_shrink_mask():
    dot = np.zeros((7, 7), dtype=np.uint8)
    dot[3, 3] = 255
    grown = op.growMask(grey(dot), radius=1)
    assert (grown.data[2:5, 2:5] == 255).all() and grown.data.sum() == 9 * 255
    shrunk = op.shrinkMask(grown, radius=1)
    assert shrunk.data[..., 0].tolist() == dot.tolist()
    assert op.growMask(grey(dot), radius=2).data.sum() == 25 * 255


def test_expand_bleeds_colours_beyond_the_mask():
    colour = np.zeros((5, 5, 3), dtype=np.uint8)
    colour[2, 2] = (200, 100, 50)
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 255
    out, new_mask = op.expand(image.Image(data=colour), grey(mask), 1)
    assert (out.data[1:4, 1:4] == (200, 100, 50)).all()
    assert out.data[0].sum() == 0
    assert (new_mask.data[1:4, 1:4] == 255).all() and new_mask.data.sum() == 9 * 255


def test_blurred_runs_on_this_numpy():
    # used to call numpy.trapz, removed in numpy 2; the OpenGL renderer's sub-surface scattering needs it
    out = op.blurred(grey(np.full((6, 6), 100.0)))
    assert out.data.shape[:2] == (6, 6)


@pytest.fixture
def trapz_restored(monkeypatch):
    # Exercise the rest of the blur with the numpy 2 name of the same function
    monkeypatch.setattr(np, "trapz", np.trapezoid, raising=False)


@pytest.mark.parametrize("kernel", [5, 15])
def test_blurred_algorithm_with_trapz_restored(trapz_restored, kernel):
    spike = np.zeros((9, 9), dtype=np.uint8)
    spike[4, 4] = 255
    out = op.blurred(grey(spike), level=10.0, kernelSize=kernel).data[..., 0]
    assert out.shape == (9, 9)
    assert out.min() >= 0 and out.max() <= 255
    assert out.argmax() == 4 * 9 + 4  # still centred
    np.testing.assert_allclose(out, out.T, atol=1e-6)
    np.testing.assert_allclose(out, out[::-1, ::-1], atol=1e-6)
    assert 0 < out[4, 4] < 255 and out[3, 4] > 0  # spread to the neighbours


def test_blurred_clamps_kernel_size(trapz_restored):
    noise = grey(np.random.RandomState(0).randint(0, 255, (6, 7)))
    def blur(k):
        return op.blurred(noise, level=4.0, kernelSize=k).data
    assert np.array_equal(blur(3), blur(5))
    assert np.array_equal(blur(40), blur(30))
    assert not np.array_equal(blur(5), blur(30))
