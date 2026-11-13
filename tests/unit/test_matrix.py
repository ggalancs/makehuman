"""Tests for lib/matrix.py, the numpy.matrix helpers used by the OpenGL camera.

Each helper is checked against what it is meant to reproduce: the OpenGL
glOrtho/gluPerspective/glFrustum/gluLookAt/glViewport conventions, and, for
the rotations, the independent implementation in transformations.py.
"""

import math

import numpy as np
import pytest

import matrix
import transformations as tm

# matrix.py is built on numpy.matrix, which warns on every construction
pytestmark = pytest.mark.filterwarnings("ignore:the matrix subclass:PendingDeprecationWarning")


def ndc(m, point):
    """Project a 3D point with a 4x4 matrix and divide by w."""
    return matrix.transform3(m, point)


def test_transform_multiplies_a_column_vector():
    m = np.matrix(np.arange(16, dtype=float).reshape(4, 4))
    v = np.array([1.0, -2.0, 0.5, 1.0])
    out = matrix.transform(m, v)
    assert isinstance(out, np.ndarray) and not isinstance(out, np.matrix)
    assert out.shape == (4,)
    assert np.allclose(out, np.asarray(m) @ v)


def test_transform3_divides_by_w():
    m = np.matrix(np.diag([1.0, 1.0, 1.0, 2.0]))
    assert np.allclose(matrix.transform3(m, (2.0, 4.0, 6.0)), [1.0, 2.0, 3.0])


def test_magnitude_and_normalize():
    v = np.array([3.0, 4.0, 12.0])
    assert matrix.magnitude(v) == 13.0
    assert np.allclose(matrix.normalize(v), v / 13.0)
    zero = np.zeros(3)
    assert matrix.normalize(zero) is zero  # nothing to normalise: returned as is


def test_ortho_maps_the_box_onto_the_unit_cube():
    l, r, b, t, n, f = -2.0, 6.0, -1.0, 3.0, 0.5, 10.0
    m = matrix.ortho(l, r, b, t, n, f)
    # OpenGL looks down -z: near plane at z = -n maps to -1, far to +1
    assert np.allclose(ndc(m, (l, b, -n)), [-1, -1, -1])
    assert np.allclose(ndc(m, (r, t, -f)), [1, 1, 1])
    assert np.allclose(ndc(m, ((l + r) / 2, (b + t) / 2, -(n + f) / 2)), [0, 0, 0])


def test_perspective_matches_gluPerspective():
    fovy, aspect, n, f = 60.0, 1.5, 1.0, 100.0
    m = matrix.perspective(fovy, aspect, n, f)
    top = n * math.tan(math.radians(fovy) / 2)
    right = top * aspect
    assert np.allclose(ndc(m, (right, top, -n)), [1, 1, -1])
    assert np.allclose(ndc(m, (-right, -top, -n)), [-1, -1, -1])
    # A point on the far plane at the edge of the view volume
    assert np.allclose(ndc(m, (right * f / n, top * f / n, -f)), [1, 1, 1])


def test_frustum_maps_the_near_rectangle_and_far_plane():
    x0, x1, y0, y1, z0, z1 = -1.0, 2.0, -0.5, 1.5, 1.0, 50.0
    m = matrix.frustum(x0, x1, y0, y1, z0, z1)
    assert np.allclose(ndc(m, (x0, y0, -z0)), [-1, -1, -1])
    assert np.allclose(ndc(m, (x1, y1, -z0)), [1, 1, -1])
    assert np.isclose(ndc(m, (0.0, 0.0, -z1))[2], 1.0)
    # Same matrix as transformations.clip_matrix in perspective mode
    clip = tm.clip_matrix(x0, x1, y0, y1, z0, z1, perspective=True)
    p = (0.3, 0.2, -7.0)
    clipped = clip @ np.array([p[0], p[1], -p[2], 1.0])  # clip_matrix looks down +z
    assert np.allclose(ndc(m, p)[:2], (clipped[:3] / clipped[3])[:2])


def test_translate_and_scale():
    t = matrix.translate((1.0, -2.0, 3.0))
    s = matrix.scale((2.0, 3.0, 4.0))
    assert np.allclose(ndc(t, (1, 1, 1)), [2, -1, 4])
    assert np.allclose(ndc(s, (1, 1, 1)), [2, 3, 4])
    # Scale first, then translate
    assert np.allclose(ndc(t * s, (1, 1, 1)), [3, 1, 7])


@pytest.mark.parametrize("axis, fn", [((1, 0, 0), matrix.rotx), ((0, 1, 0), matrix.roty), ((0, 0, 1), matrix.rotz)])
@pytest.mark.parametrize("degrees", [0.0, 30.0, 90.0, -135.0])
def test_axis_rotations_agree_with_rotate_and_transformations(axis, fn, degrees):
    expected = tm.rotation_matrix(math.radians(degrees), axis)
    assert np.allclose(fn(degrees), expected)
    assert np.allclose(matrix.rotate(degrees, np.array(axis, dtype=float)), expected)


def test_rotate_about_an_arbitrary_axis():
    axis = np.array([1.0, 2.0, -2.0])
    m = matrix.rotate(70.0, axis)
    assert np.allclose(m, tm.rotation_matrix(math.radians(70.0), axis))
    r = np.asarray(m)[:3, :3]
    assert np.allclose(r @ r.T, np.identity(3))
    assert np.allclose(r @ (axis / 3.0), axis / 3.0)  # the axis does not move


def test_rotx_turns_y_into_z():
    assert np.allclose(ndc(matrix.rotx(90.0), (0, 1, 0)), [0, 0, 1])


def test_lookat_puts_the_eye_at_the_origin_looking_down_minus_z():
    eye = np.array([3.0, 2.0, 5.0])
    target = np.array([1.0, 2.0, -1.0])  # level with the eye, as MakeHuman's cameras are
    up = np.array([0.0, 1.0, 0.0])
    m = matrix.lookat(eye, target, up)
    assert np.allclose(ndc(m, eye), [0, 0, 0])
    distance = np.linalg.norm(target - eye)
    assert np.allclose(ndc(m, target), [0, 0, -distance])
    # The view matrix is a rigid motion that keeps "up" pointing up
    r = np.asarray(m)[:3, :3]
    assert np.allclose(r @ r.T, np.identity(3))
    assert np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(ndc(m, eye + up), [0, 1, 0])


def test_lookat_is_rigid_when_up_is_not_perpendicular_to_the_view():
    eye = np.array([3.0, 2.0, 5.0])
    target = np.array([1.0, 0.0, -1.0])
    m = matrix.lookat(eye, target, np.array([0.0, 1.0, 0.0]))
    r = np.asarray(m)[:3, :3]
    assert np.allclose(r @ r.T, np.identity(3))


def test_viewport_maps_ndc_to_window_coordinates():
    m = matrix.viewport(10, 20, 640, 480)
    assert np.allclose(ndc(m, (-1, -1, -1)), [10, 20, 0])
    assert np.allclose(ndc(m, (1, 1, 1)), [650, 500, 1])
    assert np.allclose(ndc(m, (0, 0, 0)), [330, 260, 0.5])
