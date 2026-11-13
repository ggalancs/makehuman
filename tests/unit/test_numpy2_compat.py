"""Regression tests for the numpy 2 / Python 3.12 fixes."""

import numpy as np
import pytest

import files3d
import makehuman
import proxy
import transformations
import unique

NAMES = ["body", "helper-r-eye", "", "ñandú", "日本", "x"]


def test_pack_string_list_round_trip_including_non_ascii():
    text, index = files3d.packStringList(NAMES)
    assert text.dtype == np.dtype("S1") and index.dtype == np.uint32
    assert files3d.unpackStringList(text, index) == NAMES


def test_pack_string_list_matches_the_old_numpy_1_layout_for_ascii():
    # numpy 1.x: np.fromstring(''.join(names), dtype='S1') with character offsets
    names = ["body", "joint-head", "helper-tongue"]
    text, index = files3d.packStringList(names)
    assert text.tobytes() == "".join(names).encode()
    assert list(index) == [0, 4, 14]


def test_pack_string_list_empty():
    text, index = files3d.packStringList([])
    assert len(text) == 0 and len(index) == 0
    assert files3d.unpackStringList(text, index) == []


def test_proxy_string_helpers_round_trip():
    s1 = proxy._toS1("Café ñ")
    assert s1.dtype == np.dtype("S1")
    assert str(s1.tobytes(), "utf8") == "Café ñ"
    text, index = proxy._packStringList(NAMES)
    assert proxy._unpackStringList(text, index) == NAMES


def test_license_info_round_trip_with_non_ascii_author():
    lic = makehuman.LicenseInfo()
    lic.author = "José Núñez"
    lic.homepage = "https://example.org/ü"
    packed = lic.toNumpyString()
    restored = makehuman.LicenseInfo().fromNumpyString(packed)
    assert restored.author == "José Núñez"
    assert restored.homepage == "https://example.org/ü"
    assert restored.license == "CC0"


def test_unique_on_empty_arrays():
    a = np.array([])
    u, inv, idx = unique.unique(a, return_inverse=True, return_index=True)
    assert u.size == 0 and inv.dtype == bool and idx.dtype == bool
    u, inv = unique.unique(a, return_inverse=True)
    assert inv.dtype == bool
    assert unique.unique(a).size == 0


IDENTITY = np.identity(4).tolist()


@pytest.mark.parametrize("call", [
    lambda: transformations.translation_from_matrix(IDENTITY),
    lambda: transformations.decompose_matrix(IDENTITY),
    lambda: transformations.rotation_from_matrix(IDENTITY),
    lambda: transformations.scale_from_matrix(np.diag([2.0, 2.0, 2.0, 1.0]).tolist()),
    lambda: transformations.projection_from_matrix(transformations.projection_matrix([0, 0, 0], [1, 0, 0]).tolist()),
    lambda: transformations.quaternion_from_matrix(IDENTITY),
    lambda: transformations.is_same_transform(IDENTITY, IDENTITY),
    lambda: transformations.affine_matrix_from_points([[0, 1, 0], [0, 0, 1], [0, 0, 0]], [[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
    lambda: transformations.superimposition_matrix([[0, 1, 2], [0, 0, 1], [0, 0, 0]], [[0, 1, 2], [0, 0, 1], [0, 0, 0]]),
    lambda: transformations.angle_between_vectors([1, 0, 0], [0, 1, 0]),
    lambda: transformations.projection_matrix([0, 0, 0], [0, 0, 1], perspective=[0, 0, 5]),
    lambda: transformations.reflection_from_matrix(transformations.reflection_matrix([0, 0, 0], [1, 0, 0]).tolist()),
    lambda: transformations.shear_from_matrix(transformations.shear_matrix(0.5, [1, 0, 0], [0, 0, 0], [0, 0, 1]).tolist()),
])
def test_transformations_accept_lists(call):
    # numpy 2 raises "Unable to avoid copy" for np.array(list, copy=False)
    call()


def test_transformations_results_are_still_correct():
    m = transformations.translation_matrix([1, 2, 3]).tolist()
    assert np.allclose(transformations.translation_from_matrix(m), [1, 2, 3])
    assert np.isclose(transformations.angle_between_vectors([1, 0, 0], [0, 1, 0]), np.pi / 2)
    out = np.empty(3)
    transformations.unit_vector([3.0, 0.0, 4.0], out=out)
    assert np.allclose(out, [0.6, 0, 0.8])
