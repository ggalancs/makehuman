"""Tests for lib/unique.py, MakeHuman's copy of numpy.unique for old numpy.

numpy.unique itself serves as the oracle: the copy must give the same
answers for every combination of flags.
"""

import numpy as np
import pytest

import unique

RNG = np.random.default_rng(1234)
ARRAYS = [
    np.array([1, 2, 6, 4, 2, 3, 2]),
    np.array([[1, 1], [2, 3]]),
    RNG.integers(0, 20, size=200),
    RNG.integers(0, 5, size=(7, 9)).astype(np.uint32),
    np.array([0.5, -1.0, 0.5, 3.25]),
    np.array(["b", "a", "b", "c", "a"]),
]


@pytest.mark.parametrize("ar", ARRAYS, ids=lambda a: "%s%s" % (a.dtype, a.shape))
def test_matches_numpy_unique(ar):
    before = ar.copy()
    assert np.array_equal(unique.unique(ar), np.unique(ar))
    assert np.array_equal(ar, before)  # the input is not sorted in place


@pytest.mark.parametrize("ar", ARRAYS, ids=lambda a: "%s%s" % (a.dtype, a.shape))
def test_return_index_points_at_first_occurrences(ar):
    u, idx = unique.unique(ar, return_index=True)
    flat = ar.flatten()
    assert np.array_equal(u, np.unique(ar))
    assert np.array_equal(flat[idx], u)
    # mergesort is stable, so each index is the first occurrence of its value
    for value, i in zip(u, idx):
        assert i == np.flatnonzero(flat == value)[0]


@pytest.mark.parametrize("ar", ARRAYS, ids=lambda a: "%s%s" % (a.dtype, a.shape))
def test_return_inverse_rebuilds_the_flattened_input(ar):
    u, inv = unique.unique(ar, return_inverse=True)
    assert np.array_equal(u[inv], ar.flatten())


@pytest.mark.parametrize("ar", ARRAYS, ids=lambda a: "%s%s" % (a.dtype, a.shape))
def test_return_index_and_inverse_together(ar):
    u, idx, inv = unique.unique(ar, return_index=True, return_inverse=True)
    flat = ar.flatten()
    assert np.array_equal(flat[idx], u)
    assert np.array_equal(u[inv], flat)


def test_plain_sequences_without_flags_use_a_set():
    result = unique.unique([3, 1, 3, 2, 1])
    assert isinstance(result, np.ndarray)
    assert result.tolist() == [1, 2, 3]
    assert unique.unique(("b", "a", "b")).tolist() == ["a", "b"]


def test_plain_sequences_with_flags_are_converted_to_arrays():
    u, idx, inv = unique.unique([5, 5, 1, 9, 1], return_index=True, return_inverse=True)
    assert u.tolist() == [1, 5, 9]
    assert idx.tolist() == [2, 0, 3]
    assert inv.tolist() == [1, 1, 0, 2, 0]


def test_empty_input_returns_empty_results():
    empty = np.array([], dtype=np.int64)
    assert unique.unique(empty).size == 0
    u, idx = unique.unique(empty, return_index=True)
    assert u.size == 0 and idx.size == 0
    u, idx, inv = unique.unique(empty, return_index=True, return_inverse=True)
    assert u.size == idx.size == inv.size == 0


def test_numpy_unique_is_left_alone_on_modern_numpy():
    # The module only monkeypatches numpy.unique for numpy < 1.6.2
    assert np.unique is not unique.unique
