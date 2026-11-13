"""qtui.centerOnScreenOf: the splash screen opens on the screen the main window
is restored to, not on the one under the mouse (headless, offscreen platform)."""

import pytest
from PyQt5 import QtCore, QtWidgets

import qtui

_KEEP = []


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _KEEP.append(app)
    return app


@pytest.fixture
def widget(qapp):
    w = QtWidgets.QWidget()
    w.resize(100, 60)
    w.move(3, 4)
    _KEEP.append(w)
    return w


def test_widget_is_centered_on_the_screen_of_the_saved_window(qapp, widget):
    screen = qapp.primaryScreen()
    area = screen.geometry()
    saved = {"x": area.x() + 10, "y": area.y() + 10, "width": 200, "height": 100,
             "state": ["normal geometry"]}
    assert qtui.centerOnScreenOf(widget, saved) is screen
    assert widget.frameGeometry().center() == screen.availableGeometry().center()


@pytest.mark.parametrize("saved", [
    {"x": 10 ** 6, "y": 10 ** 6, "width": 200, "height": 100, "state": []},  # on no screen
    "",  # the default when nothing was saved yet
    None,
    {"x": 0, "y": 0},  # incomplete
], ids=["off-screen", "empty", "none", "incomplete"])
def test_widget_stays_put_without_a_usable_geometry(qapp, widget, saved):
    assert qtui.centerOnScreenOf(widget, saved) is None
    assert widget.pos() == QtCore.QPoint(3, 4)
