"""
Unit tests for the parts of lib/qtui.py that need no OpenGL context, run
headless on the Qt offscreen platform.

* The main window (Frame) is built with qtui.Canvas replaced by a plain
  QWidget, so no OpenGL widget is ever created.
* Canvas and Application methods are called on small stand-in objects: their
  logic (mouse/wheel translation, timers, async events, start order) does not
  depend on a GL context or on a second QApplication.
* Native file dialogs are replaced with fakes.
"""

import os
import sys

import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from core import G
import events3d
import language
import log
import qtgui
import qtui

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MH_DIR = os.path.join(REPO, "makehuman")

_QAPP = []  # keeps the QApplication alive for the whole session
_KEEP = []  # QObjects a timer may still reach


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    assert isinstance(app, QtWidgets.QApplication)
    _QAPP.append(app)
    return app


class CurrentFile(object):
    modified = False


class FakeApp(object):
    """The parts of the MakeHuman application object that qtui touches."""

    def __init__(self):
        self.mainwin = None
        self.events = []
        self.prompts = []
        self.loaded = []
        self.async_calls = []
        self.failures = []
        self.redraws = 0
        self.buttons = 0
        self.currentFile = CurrentFile()

    def callEvent(self, name, event):
        self.events.append((name, event))

    def prompt(self, *args):
        self.prompts.append(args)

    def loadHumanMHM(self, path):
        self.loaded.append(path)

    def callAsync(self, func, *args, **kwargs):
        self.async_calls.append(func)

    def mouseButtons(self):
        return self.buttons

    def redraw(self):
        self.redraws += 1

    def progress(self, value, *args):
        self.failures.append(value)  # an event handler raised


@pytest.fixture(autouse=True)
def app(qapp, monkeypatch):
    fake = FakeApp()
    monkeypatch.setattr(G, "app", fake)
    monkeypatch.setattr(language.language, "languageStrings", None)
    yield fake
    assert fake.failures == [], "an event handler raised inside callEvent (see the log warning)"


class FakeCanvas(QtWidgets.QWidget):
    """Takes the place of the OpenGL canvas inside Frame."""

    def __init__(self, parent, app):
        super(FakeCanvas, self).__init__(parent)
        self.app = app
        self.updates = 0
        self.made_current = 0

    def update(self):
        self.updates += 1

    def makeCurrent(self):
        self.made_current += 1


@pytest.fixture
def frame(app, monkeypatch):
    monkeypatch.setattr(qtui, "Canvas", FakeCanvas)
    monkeypatch.chdir(MH_DIR)  # the window icon is found relative to the working directory
    win = qtui.Frame(app, (640, 480))
    app.mainwin = win
    yield win
    win.hide()


# --------------------------------------------------------------------------
# Frame
# --------------------------------------------------------------------------

def test_frame_builds_the_main_window(frame, app):
    assert frame.windowTitle() == qtui.Frame.title
    assert frame.size() == QtCore.QSize(640, 480)
    assert frame.acceptDrops()
    assert frame.app is app
    assert isinstance(frame.statusBar, qtgui.StatusBar)
    assert QtWidgets.QMainWindow.statusBar(frame) is frame.statusBar
    assert frame.centralWidget() is frame.central
    assert isinstance(frame.tabs, qtgui.Tabs)
    assert isinstance(frame.canvas, FakeCanvas) and frame.canvas.app is app
    assert frame.t_layout.indexOf(frame.canvas) >= 0
    assert isinstance(frame.progressBar, qtgui.ProgressBar)
    assert frame.bottom.indexOf(frame.progressBar) >= 0
    assert isinstance(frame.top, qtui.VLayout)
    icon = frame.windowIcon()
    assert not icon.isNull()
    if qtui.supportsSVG():
        assert icon.availableSizes() == []  # the scalable SVG icon
    if sys.platform.startswith('darwin'):
        assert frame.toolbar.isHidden()


def test_frame_uses_png_icon_without_svg_support(app, monkeypatch):
    monkeypatch.setattr(qtui, "Canvas", FakeCanvas)
    monkeypatch.setattr(qtui, "supportsSVG", lambda: False)
    monkeypatch.chdir(MH_DIR)
    win = qtui.Frame(app, (300, 200))
    sizes = win.windowIcon().availableSizes()
    assert sizes and not win.windowIcon().isNull()  # a bitmap icon


def test_frame_panels(frame):
    left, right = frame.addPanels()
    assert isinstance(left, qtui.TaskPanel) and isinstance(right, qtui.TaskPanel)
    assert (left.objectName(), right.objectName()) == ('LeftTaskPanel', 'RightTaskPanel')
    left2, right2 = frame.addPanels()
    frame.showPanels(left2, right2)
    assert frame.left_top.currentWidget() is left2
    assert frame.right_top.currentWidget() is right2
    frame.showPanels(left, right)
    assert frame.left_top.currentWidget() is left


def test_frame_top_widgets(frame):
    label = frame.addTopWidget(QtWidgets.QLabel('top'))
    assert frame.top.count() == 1 and frame.top.itemAt(0).widget() is label
    frame.removeTopWidget(label)
    assert frame.top.count() == 0


def test_closing_the_window_asks_the_app_instead(frame, app):
    event = QtGui.QCloseEvent()
    frame.closeEvent(event)
    assert not event.isAccepted()
    assert app.events == [('onQuit', None)]


def test_refresh_layout_visits_every_widget(frame):
    left, _ = frame.addPanels()
    nested = left.addWidget(QtWidgets.QLabel('deep'))
    visited = []
    original = qtui.Frame.refreshLayout

    def spy(widget=None):
        visited.append(widget)
        return original(frame, widget)
    frame.refreshLayout = spy
    frame.refreshLayout()
    assert visited[0] is None  # starts from the frame itself
    for widget in (frame.central, frame.tabs, frame.canvas, frame.statusBar, left, nested):
        assert any(v is widget for v in visited)


@pytest.mark.parametrize("flags, expected", [
    (QtCore.Qt.WindowNoState, {'normal geometry', 'normal'}),
    (QtCore.Qt.WindowMaximized, {'maximized'}),
    (QtCore.Qt.WindowMinimized, {'minimized', 'normal geometry'}),
    (QtCore.Qt.WindowMinimized | QtCore.Qt.WindowMaximized, {'minimized', 'maximized'}),
    (QtCore.Qt.WindowFullScreen | QtCore.Qt.WindowActive, {'fullscreen', 'active'}),
    (QtCore.Qt.WindowActive, {'active', 'normal geometry', 'normal'}),
])
def test_window_state_sets(flags, expected):
    assert qtui.Frame._asWindowStateSet(QtCore.Qt.WindowStates(flags)) == expected


def test_window_state_property(frame):
    assert frame.windowState == {'normal geometry', 'normal'}
    frame.windowState = {'maximized'}
    assert 'maximized' in frame.windowState
    frame.windowState = {'normal geometry'}
    assert frame.windowState == {'normal geometry', 'normal'}
    frame.windowState = {'minimized', 'normal geometry'}
    assert 'minimized' in frame.windowState


def test_restoring_a_fullscreen_window(frame):
    frame.restoreGeometry({'width': 500, 'height': 400, 'x': 10, 'y': 20, 'state': ['fullscreen']})
    assert 'fullscreen' in frame.windowState


def test_store_and_restore_geometry(frame):
    frame.restoreGeometry({'width': 500, 'height': 400, 'x': 10, 'y': 20,
                           'state': ['normal geometry', 'normal']})
    assert frame.size() == QtCore.QSize(500, 400)
    assert frame.isVisible()
    stored = frame.storeGeometry()
    assert (stored['width'], stored['height']) == (500, 400)
    assert set(stored['state']) == {'normal geometry', 'normal'}

    frame.normalStateGeometry = {'width': 321, 'height': 123, 'x': 1, 'y': 2}
    frame.windowState = {'maximized'}
    stored = frame.storeGeometry()
    assert (stored['width'], stored['height']) == (321, 123)  # the size to restore to
    assert 'maximized' in stored['state']
    assert 'state' not in frame.normalStateGeometry  # stored a copy


def test_resize_move_and_state_change_remember_the_normal_geometry(frame):
    frame.resize(410, 310)
    frame.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(410, 310), QtCore.QSize(640, 480)))
    assert (frame.normalStateGeometry['width'], frame.normalStateGeometry['height']) == (410, 310)
    frame.move(30, 40)
    frame.moveEvent(QtGui.QMoveEvent(QtCore.QPoint(30, 40), QtCore.QPoint(0, 0)))
    assert (frame.normalStateGeometry['x'], frame.normalStateGeometry['y']) == (frame.pos().x(), frame.pos().y())
    frame.resize(420, 320)
    frame.changeEvent(QtCore.QEvent(QtCore.QEvent.WindowStateChange))
    assert frame.normalStateGeometry['width'] == 420

    remembered = dict(frame.normalStateGeometry)
    frame.windowState = {'maximized'}
    frame.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(999, 999), QtCore.QSize(420, 320)))
    frame.moveEvent(QtGui.QMoveEvent(QtCore.QPoint(0, 0), QtCore.QPoint(30, 40)))
    frame.changeEvent(QtCore.QEvent(QtCore.QEvent.WindowStateChange))
    assert frame.normalStateGeometry == remembered  # maximized sizes are not kept
    frame.changeEvent(QtCore.QEvent(QtCore.QEvent.ActivationChange))


def mime(*paths):
    data = QtCore.QMimeData()
    if paths:
        data.setUrls([QtCore.QUrl.fromLocalFile(p) for p in paths])
    _KEEP.append(data)
    return data


def drag_enter(data):
    return QtGui.QDragEnterEvent(QtCore.QPoint(5, 5), QtCore.Qt.CopyAction, data,
                                 QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)


def drop(data):
    return QtGui.QDropEvent(QtCore.QPointF(5, 5), QtCore.Qt.CopyAction, data,
                            QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)


@pytest.mark.parametrize("paths, accepted", [
    (('/models/girl.mhm',), True),
    (('/models/GIRL.MHM',), True),
    (('/models/girl.obj',), False),
    ((), False),
])
def test_only_mhm_files_can_be_dragged_in(frame, paths, accepted):
    event = drag_enter(mime(*paths))
    event.ignore()
    frame.dragEnterEvent(event)
    assert event.isAccepted() == accepted


def test_dropping_a_model_loads_it(frame, app):
    frame.dropEvent(drop(mime('/models/girl.mhm')))
    assert app.loaded == ['/models/girl.mhm']
    frame.dropEvent(drop(mime('/models/girl.obj')))
    frame.dropEvent(drop(mime()))
    assert app.loaded == ['/models/girl.mhm'] and app.prompts == []


def test_dropping_a_model_over_unsaved_changes_asks_first(frame, app):
    app.currentFile.modified = True
    frame.dropEvent(drop(mime('/models/boy.mhm')))
    assert app.loaded == []
    title, text, yes, no, action = app.prompts[0]
    assert (title, yes, no) == ('Load', 'Yes', 'No')
    assert 'unsaved changes' in text
    action()  # the user answers Yes
    assert app.loaded == ['/models/boy.mhm']


# --------------------------------------------------------------------------
# VLayout, TaskPanel, CategoryPanel, LogWindow
# --------------------------------------------------------------------------

def capped(width, height):
    w = QtWidgets.QWidget()
    w.setMaximumSize(width, height)
    return w


def test_vlayout_stacks_visible_children():
    host = QtWidgets.QWidget()
    layout = qtui.VLayout(host)
    a, b, c = capped(80, 20), capped(60, 30), capped(70, 15)
    for w in (a, b, c):
        layout.addWidget(w)
    assert layout.count() == 3
    assert layout.itemAt(-1) is None and layout.itemAt(3) is None
    assert layout.expandingDirections() == QtCore.Qt.Vertical
    assert layout.sizeHint().height() == 20 + 30 + 15
    assert layout.maximumSize() == layout.sizeHint()

    b.hide()
    assert layout.sizeHint().height() == 20 + 15
    layout.setGeometry(QtCore.QRect(10, 5, 50, 100))
    assert a.geometry() == QtCore.QRect(10, 5, 50, 20)  # clipped to the layout width
    assert c.geometry() == QtCore.QRect(10, 25, 50, 15)
    layout.setGeometry(QtCore.QRect(0, 0, 100, 30))
    assert c.geometry().height() == 10  # clipped to the remaining height

    item = layout.takeAt(0)
    assert item.widget() is a and layout.count() == 2
    assert layout.takeAt(9) is None


def test_task_panel_holds_widgets_top_to_bottom():
    panel = qtui.TaskPanel()
    assert panel.minimumHeight() == 250
    assert panel.layout.direction() == QtWidgets.QBoxLayout.TopToBottom
    first = panel.addWidget(QtWidgets.QLabel('one'))
    second = panel.addWidget(QtWidgets.QLabel('two'))
    assert [panel.layout.itemAt(i).widget() for i in range(2)] == [first, second]
    panel.removeWidget(first)
    assert panel.layout.count() == 1 and panel.layout.itemAt(0).widget() is second


def test_category_panel_holds_widgets_bottom_to_top():
    panel = qtui.CategoryPanel()
    assert panel.layout.direction() == QtWidgets.QBoxLayout.BottomToTop
    label = panel.addWidget(QtWidgets.QLabel('x'))
    assert panel.layout.indexOf(label) == 0
    panel.removeWidget(label)
    assert panel.layout.count() == 0
    calls = []
    panel.attachEvent('onShow', lambda e: calls.append(e))
    panel.show()
    panel.hide()
    assert calls == [panel]  # it is a qtgui.Widget


def test_log_window_filters_and_colors_by_level():
    window = qtui.LogWindow()
    assert window.allowsMultipleSelection()
    window.addLogMessage('debug line', log.DEBUG)
    window.addLogMessage('warning line', log.WARNING)
    assert [i.text for i in window.getItems()] == ['debug line', 'warning line']
    assert window.getItemData(1) == log.WARNING
    assert window.item(1).foreground().color() == QtGui.QColor('darkorange')
    assert not window.item(0).isHidden()

    window.setLevel(log.WARNING)
    assert window.item(0).isHidden() and not window.item(1).isHidden()
    window.addLogMessage('another debug', log.DEBUG)
    assert window.item(2).isHidden()
    window.setItemColor(0, '#000000')
    window.setLevel(log.DEBUG)
    assert not any(i.isHidden() for i in window.getItems())
    assert window.item(0).foreground().color() == QtGui.QColor('grey')


# --------------------------------------------------------------------------
# Version and format helpers
# --------------------------------------------------------------------------

def test_qt_version_helpers():
    assert qtui.getQtVersionString() == QtCore.qVersion()
    version = qtui.getQtVersion()
    assert version == [int(n) for n in QtCore.qVersion().split('.')]
    assert version[0] == 5


def test_supported_image_formats(monkeypatch):
    formats = qtui.supportedImageFormats()
    assert 'png' in formats
    assert all(f == f.lower() for f in formats)
    assert qtui.supportsJPG() == ('jpg' in formats)
    assert qtui.supportsSVG() == ('svg' in formats)
    monkeypatch.setattr(qtui, "supportedImageFormats", lambda: ['png', 'svg'])
    assert qtui.supportsSVG() and not qtui.supportsJPG()
    monkeypatch.setattr(qtui, "supportedImageFormats", lambda: ['png', 'jpg'])
    assert not qtui.supportsSVG() and qtui.supportsJPG()


def test_supports_svg_on_a_new_major_version(monkeypatch):
    monkeypatch.setattr(qtui, "getQtVersionString", lambda: '6.0.0')
    monkeypatch.setattr(qtui, "supportedImageFormats", lambda: ['png', 'svg'])
    assert qtui.supportsSVG()


# --------------------------------------------------------------------------
# File dialog wrappers and small helpers
# --------------------------------------------------------------------------

@pytest.fixture
def dialogs(monkeypatch, app):
    app.mainwin = 'MAINWIN'
    seen = []
    answers = {}

    def fake(name):
        def dialog(parent, **kwargs):
            seen.append((name, parent, kwargs))
            return answers[name]
        return staticmethod(dialog)

    for name in ('getOpenFileName', 'getSaveFileName', 'getExistingDirectory'):
        monkeypatch.setattr(QtWidgets.QFileDialog, name, fake(name))
    return seen, answers


@pytest.mark.parametrize("answer, expected", [
    (('/x/scene', 'MakeHuman scene (*.mhscene)'), '/x/scene.mhscene'),
    (('/x/scene.MHSCENE', 'MakeHuman scene (*.mhscene)'), '/x/scene.MHSCENE'),
    (('/x/shot.png', 'JPEG Image (*.jpg)'), '/x/shot.png.jpg'),
    (('/x/scene', 'All files (*.*)'), '/x/scene'),
    (('/x/scene', 'No pattern here'), '/x/scene'),
    (('', 'PNG Image (*.png)'), ''),
])
def test_save_file_name_adds_the_chosen_extension(dialogs, answer, expected):
    seen, answers = dialogs
    answers['getSaveFileName'] = answer
    path, ftype = qtui.getSaveFileName('/start', 'PNG (*.png);;All files (*.*)')
    assert (path, ftype) == (expected, answer[1])
    assert seen == [('getSaveFileName', 'MAINWIN', {'directory': '/start', 'filter': 'PNG (*.png);;All files (*.*)'})]


def test_save_file_name_with_a_multi_pattern_filter(dialogs):
    seen, answers = dialogs
    answers['getSaveFileName'] = ('/x/shot', 'Images (*.png *.jpg)')
    assert qtui.getSaveFileName('/start')[0] == '/x/shot.png'


def test_open_file_name_and_directory(dialogs):
    seen, answers = dialogs
    answers['getOpenFileName'] = ('/x/a.mhm', 'MHM (*.mhm)')
    assert qtui.getOpenFileName('/start') == ('/x/a.mhm', 'MHM (*.mhm)')
    assert seen[-1] == ('getOpenFileName', 'MAINWIN', {'directory': '/start', 'filter': 'All files (*.*)'})
    answers['getExistingDirectory'] = '/x/dir'
    assert qtui.getExistingDirectory('/start') == '/x/dir'
    answers['getExistingDirectory'] = ('/x/other', 'ignored')
    assert qtui.getExistingDirectory('/start') == '/x/other'
    assert seen[-1] == ('getExistingDirectory', 'MAINWIN', {'directory': '/start'})


def test_set_shortcut():
    action = QtWidgets.QAction('Save', None)
    qtui.setShortcut(qtui.Modifiers.CTRL, qtui.Keys.s, action)
    assert action.shortcut() == QtGui.QKeySequence(int(QtCore.Qt.ControlModifier) + QtCore.Qt.Key_S)


def test_keys_table():
    assert qtui.Keys.a == QtCore.Qt.Key_A and qtui.Keys.F12 == QtCore.Qt.Key_F12
    assert QtCore.Qt.Key_Escape in qtui.Keys._all
    assert qtui.Buttons.RIGHT_MASK == QtCore.Qt.RightButton


def test_call_async_thread_posts_to_the_message_queue(app):
    posted = []

    class Messages(object):
        def post(self, event):
            posted.append(event)
    app.messages = Messages()
    func = lambda *a, **k: None
    qtui.callAsyncThread(func, 1, key=2)
    event = posted[0]
    assert isinstance(event, qtui.AsyncEvent)
    assert event.type() == QtCore.QEvent.User
    assert (event.callback, event.args, event.kwargs) == (func, (1,), {'key': 2})


# --------------------------------------------------------------------------
# Application methods (on a stand-in: there can be only one QApplication)
# --------------------------------------------------------------------------

class Logger(object):
    def __init__(self):
        self.lines = []

    def debug(self, msg, *args):
        self.lines.append(msg % args if args else msg)


class AppStandIn(object):
    def __init__(self):
        self.calls = []
        self.logger_async = Logger()
        self.logger_redraw = Logger()
        self.logger_event = Logger()
        self.eventHandlers = []
        self.mainwin = None
        self.log_window = None
        self.posted = []

    def callEvent(self, name, event):
        self.calls.append((name, event))

    def postEvent(self, receiver, event):
        self.posted.append((receiver, event))

    _postAsync = qtui.Application._postAsync
    callAsync = qtui.Application.callAsync


def test_started_announces_on_start():
    standin = AppStandIn()
    qtui.Application.started(standin)
    assert standin.calls == [('onStart', None)]


def test_start_runs_the_loop_then_shuts_gl_down_before_hiding(monkeypatch):
    order = []

    class Win(object):
        def __init__(self):
            self.canvas = self

        def makeCurrent(self):
            order.append('makeCurrent')

        def hide(self):
            order.append('hide')

    class Messages(object):
        def start(self):
            order.append('messages.start')

    standin = AppStandIn()
    standin.OnInit = lambda: (order.append('OnInit'), setattr(standin, 'mainwin', Win()),
                              setattr(standin, 'messages', Messages()))
    standin.callAsync = lambda func: order.append(('callAsync', func))
    standin.exec_ = lambda: order.append('exec_')
    standin.started = 'started'
    monkeypatch.setattr(qtui.gl, "OnExit", lambda: order.append('gl.OnExit'))
    qtui.Application.start(standin)
    assert order == ['OnInit', ('callAsync', 'started'), 'messages.start', 'exec_',
                     'makeCurrent', 'gl.OnExit', 'hide']

    order.clear()
    standin.OnInit = lambda: (order.append('OnInit'), setattr(standin, 'messages', Messages()))
    standin.mainwin = None
    qtui.Application.start(standin)
    assert order[-2:] == ['exec_', 'gl.OnExit']  # no window: nothing to make current or hide


def test_on_init_builds_the_main_window(monkeypatch):
    import debugdump
    import eventqueue
    dumped = []
    monkeypatch.setattr(debugdump.dump, "appendQt", lambda: dumped.append(True))
    monkeypatch.setattr(qtui, "Canvas", FakeCanvas)
    monkeypatch.setattr(G, "windowWidth", 500)
    monkeypatch.setattr(G, "windowHeight", 400)
    monkeypatch.chdir(MH_DIR)
    standin = AppStandIn()
    qtui.Application.OnInit(standin)
    try:
        assert dumped == [True]
        assert isinstance(standin.messages, eventqueue.Manager)
        assert isinstance(standin.mainwin, qtui.Frame)
        assert standin.mainwin.app is standin
        assert standin.mainwin.size() == QtCore.QSize(500, 400)
        assert standin.mainwin.isVisible()
        assert standin.statusBar is standin.mainwin.statusBar
        assert standin.progressBar is standin.mainwin.progressBar
        assert isinstance(standin.log_window, qtui.LogWindow)
    finally:
        standin.mainwin.hide()


def test_redraw_updates_the_canvas(frame):
    standin = AppStandIn()
    qtui.Application.redraw(standin)  # no window yet
    standin.mainwin = frame
    qtui.Application.redraw(standin)
    assert frame.canvas.updates == 1
    assert standin.logger_redraw.lines == ['Application.redraw'] * 2


def test_add_log_message_goes_to_the_log_window():
    standin = AppStandIn()
    qtui.Application.addLogMessage(standin, 'lost', log.ERROR)  # no window yet
    standin.log_window = qtui.LogWindow()
    qtui.Application.addLogMessage(standin, 'kept', log.ERROR)
    assert [i.text for i in standin.log_window.getItems()] == ['kept']


def test_async_events_are_posted_and_run():
    standin = AppStandIn()
    ran = []
    qtui.Application.callAsync(standin, lambda *a, **k: ran.append((a, k)), 1, b=2)
    receiver, event = standin.posted[0]
    assert receiver is standin and isinstance(event, qtui.AsyncEvent)
    assert ran == []  # only queued
    assert qtui.Application.event(standin, event) is True
    assert ran == [((1,), {'b': 2})]
    assert any('callAsync' in line for line in standin.logger_async.lines)


def test_event_filter_lets_everything_through():
    standin = AppStandIn()
    assert qtui.Application.eventFilter(standin, None, QtCore.QEvent(QtCore.QEvent.Show)) is False


def test_event_handlers_are_ordered_and_called():
    standin = AppStandIn()

    class Handler(events3d.EventHandler):
        def __init__(self, order):
            super(Handler, self).__init__()
            self.sortOrder = order
            self.got = []

        def onCustom(self, event):
            self.got.append(event)

    first, second = Handler(10), Handler(3)
    qtui.Application.addEventHandler(standin, first, 10)
    qtui.Application.addEventHandler(standin, second, 3)
    later = Handler(None)
    qtui.Application.addEventHandler(standin, later)
    assert later.sortOrder == 11  # after the highest order so far
    assert standin.eventHandlers == [first, second, later]
    qtui.Application.callEventHandlers(standin, 'onCustom', 'payload')
    assert [h.got for h in (first, second, later)] == [['payload']] * 3


class TimerHost(QtCore.QObject):
    """A QObject running Application's timer bookkeeping."""
    addTimer = qtui.Application.addTimer
    removeTimer = qtui.Application.removeTimer
    handleTimer = qtui.Application.handleTimer
    timerEvent = qtui.Application.timerEvent

    def __init__(self):
        super(TimerHost, self).__init__()
        self.g_timers = {}


def test_timers_fire_until_removed(qapp):
    from PyQt5.QtTest import QTest
    host = TimerHost()
    _KEEP.append(host)
    ticks = []
    timer_id = host.addTimer(5, lambda: ticks.append(1))
    assert host.g_timers == {timer_id: host.g_timers[timer_id]}
    for _ in range(100):
        if len(ticks) >= 2:
            break
        QTest.qWait(10)
    assert len(ticks) >= 2
    host.removeTimer(timer_id)
    assert host.g_timers == {}
    count = len(ticks)
    QTest.qWait(30)
    assert len(ticks) == count
    host.handleTimer(12345)  # unknown ids are ignored


# --------------------------------------------------------------------------
# Canvas input handling (on a stand-in: no OpenGL widget is created)
# --------------------------------------------------------------------------

class CanvasStandIn(object):
    def __init__(self, app):
        self.app = app
        self.updates = 0
        self.current = 0
        self.mapped = QtCore.QPoint(0, 0)
        self.blockRedraw = False

    def update(self):
        self.updates += 1

    def makeCurrent(self):
        self.current += 1

    def mapFromGlobal(self, point):
        return self.mapped

    handleMouse = qtui.Canvas.handleMouse
    mouseUpDownEvent = qtui.Canvas.mouseUpDownEvent


@pytest.fixture
def canvas(app, monkeypatch):
    for name in ('g_mouse_pos', 'gg_mouse_pos', 'g_mousewheel_t'):
        monkeypatch.setattr(qtui, name, None)
    monkeypatch.setattr(G, "windowWidth", 200)
    monkeypatch.setattr(G, "windowHeight", 100)
    return CanvasStandIn(app)


def mouse_event(kind, x, y, button=QtCore.Qt.LeftButton):
    return QtGui.QMouseEvent(kind, QtCore.QPointF(x, y), button, button, QtCore.Qt.NoModifier)


@pytest.mark.parametrize("point, expected", [
    ((10, 20), (10, 20)), ((200, 100), (200, 100)),
    ((-1, 20), None), ((201, 20), None), ((10, -1), None), ((10, 101), None),
])
def test_mouse_position_inside_the_canvas(canvas, point, expected):
    canvas.mapped = QtCore.QPoint(*point)
    assert qtui.Canvas.getMousePos(canvas) == expected


def test_mouse_buttons_become_app_events(canvas, app):
    qtui.Canvas.mousePressEvent(canvas, mouse_event(QtCore.QEvent.MouseButtonPress, 12, 34))
    qtui.Canvas.mouseReleaseEvent(canvas, mouse_event(QtCore.QEvent.MouseButtonRelease, 13, 35,
                                                      QtCore.Qt.RightButton))
    names = [n for n, _ in app.events]
    assert names == ['onMouseDownCallback', 'onMouseUpCallback']
    down, up = app.events[0][1], app.events[1][1]
    assert (down.button, down.x, down.y) == (QtCore.Qt.LeftButton, 12, 34)
    assert (up.button, up.x, up.y) == (QtCore.Qt.RightButton, 13, 35)
    assert qtui.gg_mouse_pos == (13, 35)
    assert canvas.updates == 2 and canvas.current >= 2


class Clock(object):
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now


def wheel_event(x, y, dy):
    point = QtCore.QPointF(x, y)
    return QtGui.QWheelEvent(point, point, QtCore.QPoint(0, 0), QtCore.QPoint(0, dy),
                             QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)


def test_wheel_picks_a_position_only_after_a_pause(canvas, app, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(qtui, "time", clock)
    qtui.Canvas.wheelEvent(canvas, wheel_event(10, 20, 120))
    first = app.events[-1][1]
    assert (first.wheelDelta, first.x, first.y) == (1, 10, 20)
    assert qtui.gg_mouse_pos == (10, 20) and canvas.updates == 1

    clock.now += 0.1  # still scrolling: keep the picked position
    qtui.Canvas.wheelEvent(canvas, wheel_event(50, 60, -120))
    second = app.events[-1][1]
    assert (second.wheelDelta, second.x, second.y) == (-1, None, None)
    assert qtui.gg_mouse_pos == (10, 20) and canvas.updates == 1

    clock.now += qtui.MOUSEWHEEL_PICK_TIMEOUT + 0.1
    qtui.Canvas.wheelEvent(canvas, wheel_event(70, 80, 120))
    assert (app.events[-1][1].x, app.events[-1][1].y) == (70, 80)
    assert canvas.updates == 2
    assert [n for n, _ in app.events] == ['onMouseWheelCallback'] * 3


def test_mouse_moves_are_coalesced_into_one_relative_move(canvas, app):
    qtui.Canvas.mouseMoveEvent(canvas, mouse_event(QtCore.QEvent.MouseMove, 10, 10, QtCore.Qt.NoButton))
    qtui.Canvas.mouseMoveEvent(canvas, mouse_event(QtCore.QEvent.MouseMove, 15, 12, QtCore.Qt.NoButton))
    qtui.Canvas.mouseMoveEvent(canvas, mouse_event(QtCore.QEvent.MouseMove, 18, 20, QtCore.Qt.NoButton))
    assert app.async_calls == [canvas.handleMouse]  # scheduled once
    assert qtui.g_mouse_pos == (18, 20)

    app.buttons = 0
    canvas.handleMouse()
    name, event = app.events[-1]
    assert name == 'onMouseMovedCallback'
    assert (event.button, event.x, event.y, event.dx, event.dy) == (0, 18, 20, 8, 10)
    assert qtui.g_mouse_pos is None and qtui.gg_mouse_pos == (18, 20)
    assert canvas.updates == 0  # no button held: no redraw

    canvas.handleMouse()  # nothing pending
    assert len(app.events) == 1

    qtui.Canvas.mouseMoveEvent(canvas, mouse_event(QtCore.QEvent.MouseMove, 20, 20, QtCore.Qt.NoButton))
    app.buttons = int(QtCore.Qt.LeftButton)
    canvas.handleMouse()
    event = app.events[-1][1]
    assert (event.button, event.dx, event.dy) == (int(QtCore.Qt.LeftButton), 2, 0)
    assert canvas.updates == 1


def test_gl_callbacks(canvas, app, monkeypatch):
    calls = []
    monkeypatch.setattr(qtui.gl, "OnInit", lambda: calls.append('init'))
    monkeypatch.setattr(qtui.gl, "renderToCanvas", lambda: calls.append('render'))
    monkeypatch.setattr(qtui.gl, "reshape", lambda w, h: calls.append(('reshape', w, h)))
    canvas.app.logger_redraw = Logger()
    qtui.Canvas.initializeGL(canvas)
    qtui.Canvas.paintGL(canvas)
    canvas.blockRedraw = True
    qtui.Canvas.paintGL(canvas)
    assert calls == ['init', 'render']
    assert canvas.app.logger_redraw.lines == ['paintGL', 'paintGL (blocked)']

    qtui.Canvas.resizeGL(canvas, 320, 240)
    assert (G.windowWidth, G.windowHeight) == (320, 240)
    assert calls[-1] == ('reshape', 320, 240)
    name, event = app.events[-1]
    assert name == 'onResizedCallback'
    assert (event.width, event.height, event.fullscreen) == (320, 240, False)


def test_canvas_create_configures_the_widget(monkeypatch):
    monkeypatch.setattr(G, "canvas", None)
    widget = QtWidgets.QWidget()
    qtui.Canvas.create(widget)
    assert G.canvas is widget
    assert widget.focusPolicy() == QtCore.Qt.TabFocus
    assert widget.hasMouseTracking()
    assert widget.minimumHeight() == 5
    assert widget.testAttribute(QtCore.Qt.WA_OpaquePaintEvent)
    assert not widget.testAttribute(QtCore.Qt.WA_KeyCompression)
    policy = widget.sizePolicy()
    assert (policy.horizontalPolicy(), policy.verticalPolicy()) == (QtWidgets.QSizePolicy.Ignored,) * 2
