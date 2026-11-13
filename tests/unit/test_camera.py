"""Camera maths (lib/camera.py) without the app: projections, stereo, conversions, orbiting."""

import math

import numpy as np
import pytest

import camera
from core import G


class FakeHuman(object):
    def __init__(self):
        self.rotation = [0.0, 0.0, 0.0]
        self.position = [0.0, 0.0, 0.0]

    def getRotation(self):
        return list(self.rotation)

    def setRotation(self, rot):
        self.rotation = list(rot)

    def getPosition(self):
        return list(self.position)

    def setPosition(self, pos):
        self.position = list(pos)


class FakeApp(object):
    def __init__(self):
        self.selectedHuman = FakeHuman()
        self.redraws = 0

    def redraw(self):
        self.redraws += 1


class FakeObject(object):
    """What getModelMatrix expects: obj.object.transform and lockRotation."""

    def __init__(self, transform=None, lock=False):
        self.object = self
        self.transform = transform if transform is not None else np.matrix(np.identity(4))
        self.lockRotation = lock


@pytest.fixture(autouse=True)
def app(monkeypatch):
    fake = FakeApp()
    monkeypatch.setattr(G, "app", fake)
    monkeypatch.setattr(G, "windowWidth", 800)
    monkeypatch.setattr(G, "windowHeight", 600)
    return fake


def test_property_round_trips_and_change_events():
    cam = camera.Camera()
    changes = []
    cam.onChanged = lambda event: changes.append(event)
    for name, value in (("projection", 1), ("fovAngle", 45.0), ("nearPlane", 0.5), ("farPlane", 50.0),
                        ("eyeX", 1.0), ("eyeY", 2.0), ("eyeZ", 3.0), ("focusX", 0.5), ("focusY", 0.25),
                        ("focusZ", -1.0), ("upX", 0.0), ("upY", 1.0), ("upZ", 0.0), ("stereoMode", 2)):
        setattr(cam, name, value)
        assert getattr(cam, name) == value, name
    assert tuple(cam.eye) == (1.0, 2.0, 3.0) and tuple(cam.focus) == (0.5, 0.25, -1.0) and tuple(cam.up) == (0.0, 1.0, 0.0)
    cam.eye = (4, 5, 6)
    cam.focus = (0, 0, 0)
    cam.up = (0, 0, 1)
    assert (cam.eyeX, cam.eyeY, cam.eyeZ) == (4, 5, 6) and cam.upZ == 1
    assert len(changes) >= 17


def test_scale_depends_on_distance_and_field_of_view():
    cam = camera.Camera()
    cam.eye, cam.focus, cam.fovAngle = (0, 0, 10), (0, 0, 0), 90.0
    assert cam.scale == pytest.approx(10.0)
    cam.fovAngle = 60.0
    assert cam.scale == pytest.approx(10.0 * math.tan(math.radians(30)))


def test_ortho_and_perspective_projections_differ():
    cam = camera.Camera()
    cam.eye, cam.focus, cam.up = (0, 0, 10), (0, 0, 0), (0, 1, 0)
    cam.switchToPerspective()
    assert cam.projection == 1 and cam.nearPlane == 0.1
    proj_p, view_p = cam.getMatrices(0)
    cam.switchToOrtho()
    assert cam.projection == 0 and cam.nearPlane == -100.0
    proj_o, view_o = cam.getMatrices(0)
    assert np.allclose(view_p, view_o)            # same view, different lens
    assert not np.allclose(proj_p, proj_o)
    assert np.asarray(proj_o)[3, 3] == pytest.approx(1.0)  # orthographic: no perspective divide
    assert np.asarray(proj_p)[3, 2] == pytest.approx(-1.0)


@pytest.mark.parametrize("mode", [1, 2])
def test_stereo_eyes_see_from_different_positions(mode):
    cam = camera.Camera()
    cam.eye, cam.focus, cam.up = (0, 0, 10), (0, 0, 0), (0, 1, 0)
    cam.switchToPerspective()
    cam.stereoMode = mode
    left = cam.getMatrices(1)
    right = cam.getMatrices(2)
    assert not np.allclose(np.asarray(left[0]) @ np.asarray(left[1]), np.asarray(right[0]) @ np.asarray(right[1]))


def test_screen_world_round_trip():
    cam = camera.Camera()
    cam.eye, cam.focus, cam.up = (0, 1, 8), (0, 1, 0), (0, 1, 0)
    cam.switchToPerspective()
    point = [0.3, 1.2, -0.4]
    sx, sy, sz = cam.convertToScreen(*point)
    assert 0 <= sx <= 800 and 0 <= sy <= 600
    assert np.allclose(cam.convertToWorld3D(sx, sy, sz), point, atol=1e-4)
    cx, cy, _ = cam.convertToScreen(0, 1, 0)
    assert (cx, cy) == pytest.approx((400, 300), abs=1e-3)   # the focus point is in the middle
    assert np.allclose(cam.getModelMatrix(FakeObject(np.matrix(np.diag([2, 2, 2, 1])))), np.diag([2, 2, 2, 1]))
    assert np.allclose(cam.getTransform(), np.asarray(cam.getMatrices(0)[1]).flatten())


def test_base_camera_moves_the_selected_human(app):
    cam = camera.Camera()
    cam.setRotation([0, 90, 0])
    assert app.selectedHuman.rotation == [0, 90, 0] and cam.isInLeftView() and cam.isInSideView()
    cam.addRotation(1, 180)
    assert cam.isInRightView() and cam.isInParallelView()
    cam.setRotation([0, 0, 0])
    assert cam.isInFrontView() and not cam.isInSideView()
    cam.addRotation(1, 180)
    assert cam.isInBackView()
    cam.setRotation([90, 0, 0])
    assert cam.isInTopView()
    cam.setRotation([270, 0, 0])
    assert cam.isInBottomView()
    cam.setRotation([10, 0, 0])
    assert not cam.isInParallelView()
    cam.addTranslation(0, 2.5)
    assert app.selectedHuman.position == [2.5, 0, 0]
    cam.eye, cam.focus, cam.up = (0, 0, 10), (0, 0, 0), (0, 1, 0)
    cam.switchToPerspective()
    cam.addXYTranslation(10, 0)
    assert app.selectedHuman.position[0] != 2.5
    z = cam.eyeZ
    cam.addZoom(1.5)
    assert cam.eyeZ == z + 1.5
    cam.updateCamera()
    assert cam.mousePickHumanCenter(1, 2) is None


def test_orbital_inclination_is_limited_unless_disabled():
    cam = camera.OrbitalCamera()
    cam.horizontalRotation = 370
    assert cam.horizontalRotation == 10
    cam.verticalInclination = 120
    assert cam.verticalInclination == 90
    cam.verticalInclination = 200
    assert cam.verticalInclination == 270
    cam.limitInclination = False
    cam.verticalInclination = 150
    assert cam.verticalInclination == 150 and not cam.limitInclination
    cam.limitInclination = True
    assert cam.verticalInclination == 90


def test_orbital_rotation_translation_and_zoom():
    cam = camera.OrbitalCamera()
    cam.noAutoScale = True
    cam.setRotation([30, 45, 0])
    assert cam.getRotation() == [30, 45, 0]
    cam.addRotation(1, 15)
    assert cam.getRotation()[1] == 60
    cam.addRotation(0, 10)
    assert cam.getRotation()[0] == 40
    # getPosition is the pan translation that setPosition sets (clamped to [-1, 1]); a second
    # definition used to shadow it with the eye position, breaking the reset-camera animation
    cam.setPosition([0.5, 2.0, -3.0])
    assert cam.getPosition() == [0.5, 1.0, -1.0]
    cam.setPosition([0.25, 0.0, 0.0])
    cam.addTranslation(0, 0.5)
    assert cam.getPosition()[0] != 0.25
    cam.addXYTranslation(5, 5)
    cam.setZoomFactor(100)
    assert cam.zoomFactor == cam.maxZoomFactor
    cam.setZoomFactor(0.001)
    assert cam.zoomFactor == cam.minZoomFactor
    cam.setZoomFactor(1.0)
    cam.addZoom(1)
    assert cam.zoomFactor != 1.0
    assert len(cam.getEye()) == 3 and cam.getEyeX() == cam.getEye()[0]
    assert (cam.getFocusX(), cam.getFocusY(), cam.getFocusZ()) == tuple(cam.focus)
    assert cam.getAspect() == pytest.approx(800 / 600)


def test_orbital_model_matrix_rotates_unless_locked():
    cam = camera.OrbitalCamera()
    cam.noAutoScale = True
    cam.setRotation([0, 90, 0])
    free = cam.getModelMatrix(FakeObject())
    locked = cam.getModelMatrix(FakeObject(lock=True))
    assert np.allclose(locked, np.identity(4))
    assert np.allclose(np.asarray(free)[:3, :3] @ [1, 0, 0], [0, 0, -1], atol=1e-6)


def test_orbital_matrices_for_every_projection_and_eye():
    cam = camera.OrbitalCamera()
    cam.noAutoScale = True
    for projection in (0, 1):
        cam.projection = projection
        for mode in (0, 1, 2):
            cam.stereoMode = mode
            for eye in (0, 1, 2):
                proj, view = cam.getMatrices(eye)
                assert np.isfinite(np.asarray(proj)).all() and np.isfinite(np.asarray(view)).all()
    assert cam.getScale() > 0
