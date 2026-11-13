"""Tests for core/transformations.py (Gohlke's transformations, 2012.01.01).

Most of the module is exercised by its own doctests, which are run here one
docstring at a time with a fixed random seed (several examples draw random
matrices) and numpy's 1.13 print style (the expected outputs were written
for numpy 1.x reprs such as ``array([ 1.,  1.])`` and plain ``1.0`` scalars).
The remaining tests cover branches the doctests do not reach: error paths,
gimbal lock, every Euler axis convention and the Arcball helpers.
"""

import doctest
import math
import random

import numpy as np
import pytest

import transformations as tm

_DOCTESTS = doctest.DocTestFinder().find(tm, globs={"random": random, **vars(tm)})


def test_the_module_still_has_its_doctests():
    # Guards the parametrised test below against silently collecting nothing
    names = {t.name.split(".")[-1] for t in _DOCTESTS if t.examples}
    assert len(names) > 40
    assert {"rotation_matrix", "decompose_matrix", "quaternion_slerp", "Arcball"} <= names


@pytest.mark.parametrize("test", [t for t in _DOCTESTS if t.examples], ids=lambda t: t.name)
def test_doctest(test):
    random.seed(0)
    np.random.seed(0)
    report = []
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    # Copy the globals so one docstring's names cannot leak into the next
    test = doctest.DocTest(test.examples, dict(test.globs), test.name,
                           test.filename, test.lineno, test.docstring)
    with np.printoptions(suppress=True, precision=5, legacy="1.13"):
        result = runner.run(test, out=report.append)
    assert result.attempted == len(test.examples)
    assert result.failed == 0, "".join(report)


# --- rotations -------------------------------------------------------------

@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 3, 4]])
def test_rotation_from_matrix_round_trip_for_every_axis_branch(axis):
    angle, point = 0.7, [1.0, -2.0, 0.5]
    R0 = tm.rotation_matrix(angle, axis, point)
    a, d, p = tm.rotation_from_matrix(R0)
    R1 = tm.rotation_matrix(a, d, p)
    assert tm.is_same_transform(R0, R1)
    # The recovered axis is the input axis up to sign (angle flips with it)
    assert np.allclose(abs(np.dot(d, tm.unit_vector(axis))), 1.0)
    assert np.isclose(abs(a), angle)


def test_rotation_matrix_is_orthonormal_and_keeps_its_axis_fixed():
    R = tm.rotation_matrix(1.1, [1, 2, 3], [4, 5, 6])
    R33 = R[:3, :3]
    assert np.allclose(R33 @ R33.T, np.identity(3))
    assert np.isclose(np.linalg.det(R33), 1.0)
    # Points on the rotation axis do not move
    for t in (0.0, 1.0, -2.5):
        p = np.array([4, 5, 6, 1.0]) + t * np.array([1, 2, 3, 0.0])
        assert np.allclose(R @ p, p)


def test_rotation_from_matrix_rejects_matrices_without_rotation_axis():
    with pytest.raises(ValueError):
        tm.rotation_from_matrix(-np.identity(4))  # 3x3 part has no eigenvalue 1
    # The 3x3 part rotates about z (eigenvalue 1) but the whole 4x4 has none
    M = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]], float)
    with pytest.raises(ValueError):
        tm.rotation_from_matrix(M)


def test_reflection_from_matrix_error_paths():
    with pytest.raises(ValueError):
        tm.reflection_from_matrix(np.identity(4))  # nothing is mirrored
    with pytest.raises(ValueError):
        tm.reflection_from_matrix(-np.identity(4))  # no fixed point


def test_reflection_is_an_involution():
    M = tm.reflection_matrix([1, 2, 3], [0, 1, 1])
    assert np.allclose(M @ M, np.identity(4))
    assert np.isclose(np.linalg.det(M[:3, :3]), -1.0)


def test_scale_from_matrix_without_fixed_point():
    with pytest.raises(ValueError):
        tm.scale_from_matrix(np.diag([2.0, 2.0, 2.0, 2.0]))


def test_projection_from_matrix_error_paths():
    with pytest.raises(ValueError):
        tm.projection_from_matrix(np.identity(4))  # nothing is projected away
    with pytest.raises(ValueError):
        tm.projection_from_matrix(np.zeros((4, 4)), pseudo=True)


def test_clip_matrix_rejects_invalid_frustums():
    with pytest.raises(ValueError):
        tm.clip_matrix(1, -1, -1, 1, 1, 2)  # left >= right
    with pytest.raises(ValueError):
        tm.clip_matrix(-1, 1, 1, -1, 1, 2)  # bottom >= top
    with pytest.raises(ValueError):
        tm.clip_matrix(-1, 1, -1, 1, 0.0, 2, perspective=True)  # near <= 0


def test_shear_matrix_requires_orthogonal_direction_and_normal():
    with pytest.raises(ValueError):
        tm.shear_matrix(0.3, [1, 0, 0], [0, 0, 0], [1, 1, 0])


def test_shear_matrix_preserves_volume_and_round_trips():
    # Not an axis-aligned normal: the shear matrix is defective (eigenvalue 1
    # four times) and with an axis-aligned normal LAPACK may hand back a
    # near-infinite "point", which shear_from_matrix cannot recover from
    S0 = tm.shear_matrix(0.4, [1, 0, 0], [1, 2, 3], [0, 0.6, 0.8])
    assert np.isclose(np.linalg.det(S0), 1.0)
    S1 = tm.shear_matrix(*tm.shear_from_matrix(S0))
    assert tm.is_same_transform(S0, S1)


def test_shear_from_matrix_rejects_a_plain_rotation():
    with pytest.raises(ValueError):
        tm.shear_from_matrix(tm.rotation_matrix(math.pi / 2, [0, 0, 1]))


def test_decompose_matrix_error_paths():
    M = np.identity(4)
    M[3, 3] = 0.0
    with pytest.raises(ValueError):
        tm.decompose_matrix(M)
    S = np.identity(4)
    S[0, 0] = 0.0
    with pytest.raises(ValueError):
        tm.decompose_matrix(S)


def test_affine_matrix_from_points_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        tm.affine_matrix_from_points([[0, 1], [0, 1], [0, 1]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]])


# --- Euler angles and quaternions -----------------------------------------

ANGLES = (0.3, -0.8, 1.2)


@pytest.mark.parametrize("axes", sorted(tm._AXES2TUPLE))
def test_quaternion_from_euler_matches_euler_matrix_for_every_convention(axes):
    q = tm.quaternion_from_euler(*ANGLES, axes=axes)
    assert np.isclose(np.dot(q, q), 1.0)
    assert tm.is_same_transform(tm.quaternion_matrix(q), tm.euler_matrix(*ANGLES, axes=axes))
    # The tuple encoding of the same convention gives the same result
    tup = tm._AXES2TUPLE[axes]
    assert np.allclose(tm.quaternion_from_euler(*ANGLES, axes=tup), q)
    assert tm.is_same_transform(tm.euler_matrix(*ANGLES, axes=tup), tm.euler_matrix(*ANGLES, axes=axes))


@pytest.mark.parametrize("axes", sorted(tm._AXES2TUPLE))
def test_euler_from_matrix_round_trips_with_tuple_axes(axes):
    tup = tm._AXES2TUPLE[axes]
    R0 = tm.euler_matrix(*ANGLES, axes=axes)
    R1 = tm.euler_matrix(*tm.euler_from_matrix(R0, tup), axes=tup)
    assert tm.is_same_transform(R0, R1)


@pytest.mark.parametrize("axes", sorted(tm._AXES2TUPLE))
def test_euler_from_matrix_in_gimbal_lock(axes):
    # Repeated-axis conventions lock at a middle angle of 0, the others at pi/2
    middle = 0.0 if tm._AXES2TUPLE[axes][2] else math.pi / 2
    R0 = tm.euler_matrix(0.4, middle, 0.25, axes)
    angles = tm.euler_from_matrix(R0, axes)
    assert tm.is_same_transform(R0, tm.euler_matrix(*angles, axes=axes))


def test_unknown_euler_axes_are_rejected():
    with pytest.raises(KeyError):
        tm.euler_matrix(0, 0, 0, axes="abcd")
    with pytest.raises(KeyError):
        tm.euler_from_matrix(np.identity(4), axes=(5, 5, 5, 5))
    with pytest.raises(KeyError):
        tm.quaternion_from_euler(0, 0, 0, axes="nope")


def test_quaternion_matrix_of_a_zero_quaternion_is_identity():
    assert np.array_equal(tm.quaternion_matrix([0, 0, 0, 0]), np.identity(4))


def test_quaternion_inverse_and_conjugate():
    q = tm.quaternion_from_euler(*ANGLES)
    assert np.allclose(tm.quaternion_multiply(q, tm.quaternion_inverse(q)), [1, 0, 0, 0])
    # For unit quaternions the inverse is the conjugate
    assert np.allclose(tm.quaternion_inverse(q), tm.quaternion_conjugate(q))


def test_rotation_from_quaternion_recovers_angle_and_axis():
    axis = tm.unit_vector([1.0, 2.0, 2.0])
    q = tm.quaternion_about_axis(1.3, axis)
    angle, recovered = tm.rotation_from_quaternion(q)
    assert np.isclose(angle, 1.3)
    assert np.allclose(recovered, axis, atol=1e-6)
    assert recovered.dtype == np.float32


def test_rotation_from_quaternion_for_the_identity():
    angle, axis = tm.rotation_from_quaternion(np.array([1.0, 0.0, 0.0, 0.0]))
    assert angle == 0.0
    assert np.array_equal(axis, [0, 0, 0])


def test_rotation_from_quaternion_accepts_a_non_unit_quaternion():
    angle, axis = tm.rotation_from_quaternion(np.array([2.0, 0.0, 0.0, 0.0]))
    assert angle == 0.0


@pytest.mark.parametrize("angle, axis", [(0.3, [1, 2, 3]), (1.9, [0, 1, 1]), (2.0, [-1, 0, 2])])
def test_quaternion_from_matrix_precise_agrees_with_eigen_method_for_small_angles(angle, axis):
    R = tm.rotation_matrix(angle, axis)
    assert np.allclose(tm.quaternion_from_matrix(R, isprecise=True), tm.quaternion_from_matrix(R))


@pytest.mark.parametrize("angle, axis", [
    (math.pi, [1, 0, 0]), (math.pi, [0, 1, 0]), (math.pi, [0, 0, 1]), (2.5, [1, 2, 3])])
def test_quaternion_from_matrix_precise_for_large_angles(angle, axis):
    R = tm.rotation_matrix(angle, axis)
    q = tm.quaternion_from_matrix(R, isprecise=True)
    assert tm.is_same_transform(tm.quaternion_matrix(q), R)


def test_quaternion_slerp_edge_cases():
    q0 = tm.quaternion_about_axis(0.2, [0, 0, 1])
    # Identical quaternions: nothing to interpolate
    assert np.allclose(tm.quaternion_slerp(q0, q0, 0.5), q0)
    # q and -q are the same rotation: still nothing to interpolate
    assert np.allclose(tm.quaternion_slerp(q0, -q0, 0.5), q0)


def test_quaternion_slerp_takes_the_shortest_path():
    q0 = tm.quaternion_about_axis(0.0, [0, 0, 1])
    q1 = -tm.quaternion_about_axis(1.0, [0, 0, 1])  # same rotation, opposite hemisphere
    half = tm.quaternion_slerp(q0, q1, 0.5)
    assert np.allclose(half, tm.quaternion_about_axis(0.5, [0, 0, 1]))
    long_way = tm.quaternion_slerp(q0, q1, 0.5, shortestpath=False)
    angle, _ = tm.rotation_from_quaternion(long_way)
    assert not np.isclose(angle, 0.5)


# --- Arcball ---------------------------------------------------------------

def test_arcball_rejects_an_invalid_initial_orientation():
    with pytest.raises(ValueError):
        tm.Arcball(initial=[1, 2, 3])


def test_arcball_initial_matrix_and_quaternion():
    R = tm.rotation_matrix(0.5, [0, 1, 0])
    assert tm.is_same_transform(tm.Arcball(initial=R).matrix(), R)
    q = tm.quaternion_about_axis(0.5, [0, 1, 0])
    assert tm.is_same_transform(tm.Arcball(initial=2 * q).matrix(), R)  # normalised


def test_arcball_drag_without_movement_keeps_orientation():
    ball = tm.Arcball()
    ball.place([320, 320], 320)
    ball.down([400, 300])
    ball.drag([400, 300])
    assert np.allclose(ball.matrix(), np.identity(4))


def test_arcball_constrain_flag():
    ball = tm.Arcball()
    assert ball.getconstrain() is False
    ball.setconstrain(1)
    assert ball.getconstrain() is True


def test_arcball_constrained_drag_rotates_about_the_chosen_axis():
    ball = tm.Arcball()
    ball.place([0, 0], 50)
    ball.setaxes([0, 0, 1], [1, 0, 0])
    ball.setconstrain(True)
    ball.down([50, 0])  # on the equator: the arc of the z axis passes through it
    ball.drag([0, -50])  # a quarter turn along the equator
    # The arcball quaternion [cos, sin * axis] of the 90 degree arc between the
    # two sphere points is a half turn about z
    assert np.allclose(ball.matrix(), np.diag([-1.0, -1.0, 1.0, 1.0]))
    # Unconstrained, the same drag also stays a rotation about z here
    free = tm.Arcball()
    free.place([0, 0], 50)
    free.down([50, 0])
    free.drag([0, -50])
    assert np.allclose(free.matrix(), ball.matrix())
    # next() keeps spinning in the direction of the last drag
    before = ball.matrix()
    ball.next()
    assert not np.allclose(ball.matrix(), before)


def test_arcball_free_drag_rotates_twice_the_arc_about_its_normal():
    # Shoemake's arcball: dragging from sphere point v0 to v1 rotates by twice
    # the angle between them, about v0 x v1
    center, radius = [10.0, 20.0], 100.0
    start, end = [40.0, 0.0], [-30.0, 70.0]
    v0 = tm.arcball_map_to_sphere(start, center, radius)
    v1 = tm.arcball_map_to_sphere(end, center, radius)
    ball = tm.Arcball()
    ball.place(center, radius)
    ball.down(start)
    ball.drag(end)
    angle, direction, _ = tm.rotation_from_matrix(ball.matrix())
    axis = tm.unit_vector(np.cross(v0, v1))
    if np.dot(direction, axis) < 0:
        angle, direction = -angle, -direction
    assert np.allclose(direction, axis)
    assert np.isclose(angle, 2 * tm.angle_between_vectors(v0, v1))


def test_arcball_map_to_sphere_inside_and_outside():
    v = tm.arcball_map_to_sphere([0, 0], [0, 0], 10)
    assert np.allclose(v, [0, 0, 1])
    outside = tm.arcball_map_to_sphere([30, -40], [0, 0], 10)
    assert outside[2] == 0.0
    assert np.isclose(tm.vector_norm(outside), 1.0)
    assert np.allclose(outside, [0.6, 0.8, 0.0])


def test_arcball_constrain_to_axis():
    # Result lies on the great circle perpendicular to the axis, z >= 0
    v = tm.arcball_constrain_to_axis([0.3, 0.4, -0.866], [1, 0, 0])
    assert np.isclose(np.dot(v, [1, 0, 0]), 0.0)
    assert np.isclose(tm.vector_norm(v), 1.0)
    assert v[2] >= 0.0
    # A point on the axis itself has no projection: fixed fallbacks
    assert np.allclose(tm.arcball_constrain_to_axis([0, 0, 1], [0, 0, 1]), [1, 0, 0])
    fallback = tm.arcball_constrain_to_axis([1, 0, 0], [1, 0, 0])
    assert np.allclose(fallback, [0, 1, 0])


def test_arcball_nearest_axis():
    axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # A point near the north pole lies on the circles of the x and y axes
    assert tm.arcball_nearest_axis([0.0, 0.0, 1.0], axes) in (axes[0], axes[1])
    assert tm.arcball_nearest_axis([0.0, 0.70710678, 0.70710678], axes) is axes[0]
