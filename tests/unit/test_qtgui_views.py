"""
Unit tests for the composite widgets of lib/qtgui.py, run headless on the Qt
offscreen platform: dialogs, the file entry and browse button, splash screen,
status bar, the vertical scroll area, actions and icons, image views and the
color helpers.

Native dialogs are never opened: QFileDialog/QColorDialog static methods are
replaced, and modal dialogs are answered from a QTimer. Every widget a timer
can still reach is kept alive in _KEEP so a late timer never touches a
deleted C++ object (that would abort the whole test process).
"""

import os
import subprocess
import sys
import time

import pytest
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtTest import QTest

from core import G
import language
import qtgui

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MH_DIR = os.path.join(REPO, "makehuman")

_QAPP = []  # keeps the QApplication alive for the whole session
_KEEP = []  # widgets that a pending QTimer may still reach


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    assert isinstance(app, QtWidgets.QApplication)
    _QAPP.append(app)
    return app


class FakeApp(object):
    """The parts of the MakeHuman application object that qtgui touches."""

    def __init__(self):
        self.mainwin = None
        self.theme = None
        self.settings = {'invertMouseWheel': False}
        self.redraws = 0
        self.failures = []
        self.prompts = []

    def redraw(self):
        self.redraws += 1

    def progress(self, value, *args):
        # events3d.EventHandler.eventFailed calls this when a handler raised
        self.failures.append(value)

    def getSetting(self, name):
        return self.settings[name]

    def prompt(self, *args, **kwargs):
        self.prompts.append((args, kwargs))


@pytest.fixture(autouse=True)
def app(qapp, monkeypatch):
    fake = FakeApp()
    monkeypatch.setattr(G, "app", fake)
    monkeypatch.setattr(language.language, "languageStrings", None)
    yield fake
    assert fake.failures == [], "an event handler raised inside callEvent (see the log warning)"


def record(widget, *names):
    calls = []
    for name in names:
        widget.attachEvent(name, lambda event, name=name: calls.append((name, event)))
    return calls


def send(widget, event):
    return QtWidgets.QApplication.sendEvent(widget, event)


def wait_until(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        QTest.qWait(10)
    return condition()


def solid_image(width, height, color):
    img = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
    img.fill(QtGui.QColor(color))
    return img


def save_png(path, width, height, color):
    assert solid_image(width, height, color).save(str(path))
    return str(path)


# --------------------------------------------------------------------------
# Dialog.prompt (modal)
# --------------------------------------------------------------------------

def answer_later(dialog, accept=True, tick_dont_show_again=False):
    """Answer the modal dialog once its event loop is running."""
    _KEEP.append(dialog)

    def act():
        if tick_dont_show_again:
            dialog.check.setChecked(True)
        if accept:
            dialog.accept()
        else:
            dialog.reject()
    QtCore.QTimer.singleShot(0, act)


def test_prompt_accept_runs_first_action():
    dialog = qtgui.Dialog()
    done = []
    answer_later(dialog, accept=True)
    result = dialog.prompt('Title', 'Delete %s?', 'Yes', 'No',
                           button1Action=lambda: done.append(1),
                           button2Action=lambda: done.append(2), fmtArgs='model')
    assert result is True
    assert done == [1]
    assert dialog.windowTitle() == 'Title'
    assert dialog.text.text() == 'Delete model?'
    assert (dialog.button1.text(), dialog.button2.text()) == ('Yes', 'No')
    assert not dialog.button2.isHidden()
    assert dialog.check.isHidden()
    assert dialog.isModal()


def test_prompt_reject_runs_second_action():
    dialog = qtgui.Dialog()
    done = []
    answer_later(dialog, accept=False)
    result = dialog.prompt('T', '%s and %s', 'OK', 'Cancel',
                           button1Action=lambda: done.append(1),
                           button2Action=lambda: done.append(2), fmtArgs=['a', 'b'])
    assert result is False
    assert done == [2]
    assert dialog.text.text() == 'a and b'


def test_prompt_with_single_button_and_no_actions():
    dialog = qtgui.Dialog()
    answer_later(dialog, accept=True)
    assert dialog.prompt('Info', 'Done', 'OK') is True
    assert dialog.button2.isHidden()
    answer_later(dialog, accept=False)
    assert dialog.prompt('Info', 'Done', 'OK') is False


def test_dont_show_again_suppresses_the_prompt():
    dialog = qtgui.Dialog()
    answer_later(dialog, accept=True)  # not ticked: will ask again
    assert dialog.prompt('Tip', 'Hint', 'OK', helpId='tip1') is True
    assert not dialog.check.isHidden()
    assert 'tip1' not in dialog.helpIds

    answer_later(dialog, accept=True, tick_dont_show_again=True)
    assert dialog.prompt('Tip', 'Hint', 'OK', helpId='tip1') is True
    assert dialog.helpIds == {'tip1'}

    # Suppressed: returns at once without showing (no timer needed to answer)
    assert dialog.prompt('Tip', 'Hint', 'OK', helpId='tip1') is None


@pytest.mark.parametrize("platform", ['darwin', 'linux'])
def test_about_box_with_scrollbars_links_urls(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    parent = QtWidgets.QWidget()
    parent.resize(800, 600)
    box = qtgui.AboutBoxScrollbars(parent, 'About MakeHuman', 'Visit http://example.com/mh\nnow', 'v1.2')
    assert box.windowTitle() == 'About MakeHuman'
    assert box.isModal()
    assert box.testAttribute(QtCore.Qt.WA_DeleteOnClose)
    labels = {lbl.text(): lbl for lbl in box.findChildren(QtWidgets.QLabel)}
    body = [t for t in labels if 'example.com' in t]
    assert body == ['Visit <a href="http://example.com/mh" style="color: #ffa02f;">http://example.com/mh</a><br>now']
    assert 'v1.2' in labels and labels['v1.2'].font().bold()
    assert box.height() == int(0.8 * 600)
    assert box.width() <= 800
    label = labels[body[0]]
    if platform == 'darwin':
        assert label.contentsMargins().left() == 16
    else:
        assert label.indent() == 9
        assert box.contentsMargins().top() == 15
    ok = box.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.Ok)
    assert ok.isDefault()


# --------------------------------------------------------------------------
# FileEntryView
# --------------------------------------------------------------------------

def test_file_entry_paths_in_open_mode(tmp_path):
    fe = qtgui.FileEntryView('File:', 'Save')
    assert fe.mode == 'open' and fe.browse.mode == 'open'
    assert fe.label.text() == 'File:'
    assert fe.confirm.text() == 'Save'
    assert fe.layout.indexOf(fe.confirm) >= 0
    fe.directory = str(tmp_path)
    fe.text = 'model.mhm'
    assert fe.text == 'model.mhm'
    assert fe.path == os.path.normpath(os.path.join(str(tmp_path), 'model.mhm'))
    fe.path = os.path.join(str(tmp_path), 'sub', 'other.mhm')
    assert fe.directory == os.path.join(str(tmp_path), 'sub')
    assert fe.text == 'other.mhm'
    fe.filter = 'MakeHuman Models (*.mhm)'
    assert fe.filter == 'MakeHuman Models (*.mhm);;All Files (*.*)'
    assert fe.browse.filter == fe.filter


def test_file_entry_dir_mode_shows_the_directory(tmp_path):
    fe = qtgui.FileEntryView('Folder', mode='dir')
    assert fe.mode == 'dir'
    assert fe.layout.indexOf(fe.confirm) == -1  # no confirm button for folders
    fe.directory = str(tmp_path)
    assert fe.text == str(tmp_path)
    assert fe.path == str(tmp_path)
    fe.path = str(tmp_path / 'x')
    assert fe.directory == str(tmp_path / 'x')
    fe.mode = 'save'
    assert fe.browse.mode == 'save'


def test_file_entry_confirm_sources(tmp_path):
    fe = qtgui.FileEntryView('File', 'Go')
    calls = record(fe, 'onFileSelected', 'onChange')
    fe.directory = str(tmp_path)
    QTest.keyClicks(fe.edit, 'a')
    QTest.keyClick(fe.edit, QtCore.Qt.Key_Return)
    fe.confirm.click()
    assert calls[0] == ('onChange', 'a')
    selected = [event for name, event in calls if name == 'onFileSelected']
    assert [e.source for e in selected] == ['return', 'button']
    assert all(e.path == os.path.join(str(tmp_path), 'a') for e in selected)
    text = repr(selected[0])
    assert text.startswith('FileSelectedEvent: Path: ')
    assert repr(os.path.join(str(tmp_path), 'a')) in text and 'return' in text


def test_file_entry_line_edit_rejects_path_characters():
    fe = qtgui.FileEntryView()
    QTest.keyClicks(fe.edit, 'a/b*c')
    assert fe.text == 'abc'


def test_file_entry_with_empty_name_asks_for_one(app):
    fe = qtgui.FileEntryView('File', 'Go')
    calls = record(fe, 'onFileSelected')
    fe.confirm.click()
    assert calls == []
    assert len(app.prompts) == 1
    args, kwargs = app.prompts[0]
    assert args[0] == 'Missing Filename' and kwargs == {'helpId': 'noFileName'}


def test_file_entry_dir_mode_takes_the_typed_directory(tmp_path):
    fe = qtgui.FileEntryView('Folder', mode='dir')
    calls = record(fe, 'onFileSelected')
    fe.text = str(tmp_path)
    fe._confirm('return')
    assert fe.directory == str(tmp_path)
    assert calls[0][1].path == str(tmp_path) and calls[0][1].source == 'return'


def test_file_entry_focus_goes_to_the_line_edit():
    fe = qtgui.FileEntryView()
    focused = []
    fe.edit.setFocus = lambda: focused.append(True)
    send(fe, QtGui.QFocusEvent(QtCore.QEvent.FocusIn))
    assert focused == [True]


@pytest.fixture
def file_dialogs(monkeypatch):
    """Replace the native file dialogs; answers[name] is what they return."""
    seen = []
    answers = {}

    def fake(name):
        def dialog(parent=None, *args, **kwargs):
            seen.append((name, kwargs))
            return answers[name]
        return staticmethod(dialog)

    for name in ('getOpenFileName', 'getSaveFileName', 'getExistingDirectory'):
        monkeypatch.setattr(QtWidgets.QFileDialog, name, fake(name))
    return seen, answers


def test_file_entry_browse_selects_and_confirms(tmp_path, file_dialogs):
    seen, answers = file_dialogs
    chosen = str(tmp_path / 'picked.mhm')
    answers['getOpenFileName'] = (chosen, 'MHM (*.mhm)')
    fe = qtgui.FileEntryView('File', 'Load')
    fe.directory = str(tmp_path)
    calls = record(fe, 'onFileSelected')
    fe.browse.click()
    # with no text yet, the dialog opens in the entry's directory
    assert seen == [('getOpenFileName', {'directory': str(tmp_path), 'filter': ''})]
    assert fe.text == 'picked.mhm' and fe.directory == str(tmp_path)
    assert [(e.path, e.source) for _, e in calls] == [(chosen, 'browse')]

    # with a name typed, the browse button starts from the full path
    other = tmp_path / 'deeper'
    other.mkdir()
    fe.path = str(other / 'typed.mhm')
    answers['getOpenFileName'] = ('', '')  # cancelled
    fe.browse.click()
    assert fe.browse.directory == str(other) and fe.browse.filename == 'typed.mhm'
    assert len(calls) == 1  # cancelling selects nothing


# --------------------------------------------------------------------------
# BrowseButton
# --------------------------------------------------------------------------

def test_conform_filter_adds_all_files_once():
    assert qtgui.BrowseButton.conformFilter('Images (*.png)') == 'Images (*.png);;All Files (*.*)'
    assert qtgui.BrowseButton.conformFilter('All (*.*)') == 'All (*.*)'


def test_get_existing_path(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    (home / 'models').mkdir(parents=True)
    afile = home / 'models' / 'a.mhm'
    afile.write_text('x')
    monkeypatch.setattr(qtgui, "getPath", lambda sub='': str(home))
    monkeypatch.chdir(tmp_path)
    get = qtgui.BrowseButton.getExistingPath
    assert get(str(home / 'models')) == str(home / 'models')
    assert get(str(afile)) == str(afile)
    # inside the MakeHuman home: climb to the first folder that exists
    assert get(str(home / 'models' / 'new' / 'deeper' / 'b.mhm')) == str(home / 'models')
    # elsewhere: fall back to the working directory
    assert get(str(tmp_path / 'nowhere' / 'x' / 'c.mhm')) == os.path.normpath(os.getcwd())


def test_browse_button_properties(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    b = qtgui.BrowseButton()
    assert b.text() == '...'
    assert b.mode == 'open'
    assert os.path.realpath(b.directory) == os.path.realpath(str(tmp_path))
    assert qtgui.BrowseButton(mode='SAVE', label='Save').mode == 'save'
    with pytest.raises(RuntimeError):
        b.mode = 'bogus'
    b.path = os.path.join(str(tmp_path), 'd', 'f.txt')
    assert (b.directory, b.filename) == (os.path.join(str(tmp_path), 'd'), 'f.txt')
    assert b.path == os.path.join(str(tmp_path), 'd', 'f.txt')
    b.filter = 'Text (*.txt)'
    assert b.filter == 'Text (*.txt);;All Files (*.*)'
    assert b.retrunFilter() is False
    b.retrunFilter(True)
    assert b.retrunFilter() is True
    b.retrunFilter('yes')  # only booleans are accepted
    assert b.retrunFilter() is True


def test_bad_browse_mode_is_named_in_the_error():
    with pytest.raises(RuntimeError, match='bogus'):
        qtgui.BrowseButton(mode='bogus')


def test_browse_button_open_save_and_dir(tmp_path, file_dialogs):
    seen, answers = file_dialogs
    b = qtgui.BrowseButton('open')
    b.directory = str(tmp_path)
    b.filter = 'Poses (*.bvh)'
    calls = record(b, 'beforeBrowse', 'onClicked')
    answers['getOpenFileName'] = ('/x/walk.bvh', 'Poses (*.bvh)')
    b.click()
    assert calls == [('beforeBrowse', None), ('onClicked', '/x/walk.bvh')]
    assert seen[-1] == ('getOpenFileName', {'directory': str(tmp_path), 'filter': b.filter})

    b.retrunFilter(True)
    calls.clear()
    b.click()
    assert calls[-1] == ('onClicked', ('/x/walk.bvh', 'Poses (*.bvh)'))

    b.mode = 'save'
    b.retrunFilter(False)
    answers['getSaveFileName'] = ('/x/out.bvh', 'Poses (*.bvh)')
    calls.clear()
    b.click()
    assert calls[-1] == ('onClicked', '/x/out.bvh')
    assert seen[-1][0] == 'getSaveFileName'

    b.mode = 'dir'
    answers['getExistingDirectory'] = str(tmp_path / 'chosen')
    calls.clear()
    b.click()
    assert calls[-1] == ('onClicked', str(tmp_path / 'chosen'))
    assert b.directory == str(tmp_path / 'chosen')
    assert seen[-1] == ('getExistingDirectory', {'directory': str(tmp_path)})

    answers['getExistingDirectory'] = ''  # cancelled
    calls.clear()
    b.click()
    assert calls == [('beforeBrowse', None)]


def test_save_dialog_starts_with_the_current_filename(tmp_path, file_dialogs):
    seen, answers = file_dialogs
    answers['getSaveFileName'] = ('', '')
    b = qtgui.BrowseButton('save')
    b.path = str(tmp_path / 'mymodel.mhm')
    b.click()
    assert seen[-1][1]['directory'] == str(tmp_path / 'mymodel.mhm')


# --------------------------------------------------------------------------
# ColorPickButton and color helpers
# --------------------------------------------------------------------------

def test_color_conversions():
    import material
    red = qtgui.colorFromQColor(QtGui.QColor(255, 0, 0))
    assert isinstance(red, material.Color)
    assert red.asTuple() == (1.0, 0.0, 0.0)
    assert qtgui.colorFromQColor(QtGui.QColor()).asTuple() == (0.0, 0.0, 0.0)
    q = qtgui.qColorFromColor(material.Color(0.0, 0.5, 1.0))
    assert (q.red(), q.green(), q.blue()) == (0, 127, 255)
    q = qtgui.qColorFromColor((1.0, 1.0, 0.0))
    assert (q.red(), q.green(), q.blue()) == (255, 255, 0)


def test_color_pick_button_color_property():
    import material
    b = qtgui.ColorPickButton()
    assert b.text() == 'Pick'
    assert b.color.asTuple() == (0.0, 0.0, 0.0)
    c = material.Color(0.1, 0.2, 0.3)
    b.color = c
    assert b.color is c
    b.color = QtGui.QColor(0, 255, 0)
    assert b.color.asTuple() == (0.0, 1.0, 0.0)
    b.color = [0.5, 0.25, 0.0]
    assert b.color.asTuple() == (0.5, 0.25, 0.0)
    assert qtgui.ColorPickButton(c).color is c


def test_color_pick_button_click(monkeypatch):
    import material
    offered = []
    answer = [QtGui.QColor(0, 0, 255)]

    def get_color(initial, *args, **kwargs):
        offered.append(initial)
        return answer[0]
    monkeypatch.setattr(QtWidgets.QColorDialog, "getColor", staticmethod(get_color))
    b = qtgui.ColorPickButton(material.Color(1.0, 0.0, 0.0))
    calls = record(b, 'onClicked')
    b.click()
    assert offered[0] == QtGui.QColor(255, 0, 0)
    assert b.color.asTuple() == (0.0, 0.0, 1.0)
    assert [(n, c.asTuple()) for n, c in calls] == [('onClicked', (0.0, 0.0, 1.0))]

    answer[0] = QtGui.QColor()  # dialog cancelled
    b.click()
    assert len(calls) == 1
    assert b.color.asTuple() == (0.0, 0.0, 1.0)


# --------------------------------------------------------------------------
# Action, ActionGroup and icon lookup
# --------------------------------------------------------------------------

SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
       '<rect width="16" height="16" fill="%s"/></svg>')


@pytest.fixture
def icon_tree(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    (data / 'icons').mkdir(parents=True)
    (data / 'themes' / 'dark' / 'icons').mkdir(parents=True)
    icons, theme = data / 'icons', data / 'themes' / 'dark' / 'icons'
    save_png(icons / 'notfound.png', 16, 16, '#ff00ff')
    save_png(icons / 'save.png', 16, 16, '#ff0000')
    save_png(icons / 'save_disabled.png', 16, 16, '#ffff00')
    save_png(icons / 'save_active.png', 16, 16, '#00ffff')
    save_png(icons / 'save_selected.png', 16, 16, '#888888')
    save_png(theme / 'save.png', 16, 16, '#00ff00')
    save_png(icons / 'load.png', 16, 16, '#0000ff')
    (icons / 'grab.svg').write_text(SVG % '#123456')
    save_png(icons / 'grab.png', 16, 16, '#ff0000')
    (theme / 'grab.svg').write_text(SVG % '#654321')
    monkeypatch.setattr(qtgui, "getSysDataPath",
                        lambda sub='': os.path.join(str(data), sub) if sub else str(data))
    return data


def icon_color(icon, mode=QtGui.QIcon.Normal):
    return icon.pixmap(16, 16, mode).toImage().pixelColor(8, 8).name()


def test_icon_lookup_order(app, icon_tree):
    get = qtgui.Action.getIcon
    assert icon_color(get('load')) == '#0000ff'
    assert icon_color(get('missing')) == '#ff00ff'
    app.theme = 'dark'
    assert icon_color(get('save')) == '#00ff00'  # theme overrides
    assert icon_color(get('load')) == '#0000ff'  # not themed: default icon


def test_svg_icons_are_preferred(app, icon_tree):
    import qtui
    if not qtui.supportsSVG():
        pytest.skip("this Qt build has no SVG image plugin")
    assert icon_color(qtgui.Action.getIcon('grab')) == '#123456'  # svg over png
    app.theme = 'dark'
    assert icon_color(qtgui.Action.getIcon('grab')) == '#654321'  # theme svg


def test_icon_lookup_without_svg_support(app, icon_tree, monkeypatch):
    import qtui
    monkeypatch.setattr(qtui, "supportsSVG", lambda: False)
    assert icon_color(qtgui.Action.getIcon('grab')) == '#ff0000'
    app.theme = 'dark'  # the theme only has grab.svg: keep the default png
    assert icon_color(qtgui.Action.getIcon('grab')) == '#ff0000'


def test_icon_state_variants(icon_tree):
    icon = qtgui.Action.getIcon('save')
    assert icon_color(icon) == '#ff0000'
    assert icon_color(icon, QtGui.QIcon.Disabled) == '#ffff00'
    assert icon_color(icon, QtGui.QIcon.Active) == '#00ffff'
    assert icon_color(icon, QtGui.QIcon.Selected) == '#888888'


def test_action_triggers_its_method(icon_tree, monkeypatch):
    monkeypatch.setattr(qtgui.Action, "_groups", {})
    called = []
    action = qtgui.Action('save', 'Save', lambda: called.append('save'), tooltip='Save the model')
    assert action.name == 'save' and action.text == 'Save'
    assert action.toolTip() == 'Save the model'
    assert not action.isCheckable()
    action.trigger()
    assert called == ['save']

    toggle = qtgui.Action('load', 'Wire', lambda: None, toggle=True)
    assert toggle.isCheckable()


def test_actions_in_a_group_are_exclusive(icon_tree, monkeypatch):
    monkeypatch.setattr(qtgui.Action, "_groups", {})
    front = qtgui.Action('load', 'Front', lambda: None, group='view')
    back = qtgui.Action('load', 'Back', lambda: None, group='view')
    group = qtgui.Action.getGroup('view')
    assert isinstance(group, qtgui.ActionGroup)
    assert front.actionGroup() is group and back.actionGroup() is group
    assert front.isCheckable() and back.isCheckable()
    front.trigger()
    back.trigger()
    assert back.isChecked() and not front.isChecked()
    assert qtgui.Action.getGroup('other') is not group


# --------------------------------------------------------------------------
# SplashScreen and StatusBar
# --------------------------------------------------------------------------

def test_splash_progress_is_clamped():
    splash = qtgui.SplashScreen(QtGui.QPixmap(800, 600), '1.2')
    assert splash.getProgress() == 0.0
    splash.setProgress(0.4)
    assert splash.getProgress() == pytest.approx(0.4)
    splash.setProgress(-3)
    assert splash.getProgress() == 0.0
    splash.setProgress(7)
    assert splash.getProgress() == 1.0


def test_splash_log_message_is_escaped():
    splash = qtgui.SplashScreen(QtGui.QPixmap(800, 600))
    assert splash.escape('<a & b>') == '&lt;a &amp; b&gt;'
    splash.logMessage('Loading <plugins> & more')
    assert splash.message == 'Loading &lt;plugins&gt; &amp; more'


def paint_splash(splash):
    img = solid_image(800, 600, '#000000')
    painter = QtGui.QPainter(img)
    splash.drawContents(painter)
    painter.end()
    return img


def count_light(img, rect):
    return sum(1 for x in range(rect.left(), rect.right() + 1)
               for y in range(rect.top(), rect.bottom() + 1)
               if img.pixelColor(x, y).lightness() > 200)


def test_splash_draws_progress_bar_and_version():
    splash = qtgui.SplashScreen(QtGui.QPixmap(800, 600), '1.2')
    splash.setProgress(0.5)
    img = paint_splash(splash)
    bar = splash.progressBarRect
    mid_y = bar.y() + bar.height() // 2
    assert img.pixelColor(bar.x() + 50, mid_y).name() == '#f58220'
    assert img.pixelColor(bar.x() + bar.width() - 50, mid_y).name() == '#191718'
    assert count_light(img, splash.versionRect) > 0

    splash.setProgress(0)
    unversioned = qtgui.SplashScreen(QtGui.QPixmap(800, 600))
    img = paint_splash(unversioned)
    assert img.pixelColor(bar.x() + 50, mid_y).name() == '#191718'
    assert count_light(img, splash.versionRect) == 0


def test_status_bar_messages():
    bar = qtgui.StatusBar()
    _KEEP.append(bar)
    bar.showMessage('Saved %s', 'model')
    assert bar.currentMessage() == 'Saved model'
    bar.showMessage(['Loaded ', '%d files'], 3)
    assert bar.currentMessage() == 'Loaded 3 files'
    bar.setMessage('Ready')
    assert bar._perm.text() == 'Ready'
    bar.setMessage(['Mode: ', '%s'], 'edit')
    assert bar._perm.text() == 'Mode: edit'


def test_status_bar_temporary_message_restores_the_old_one():
    bar = qtgui.StatusBar()
    _KEEP.append(bar)  # the restore timer outlives this test if it fails
    bar.setMessage('Ready')
    bar.temporaryMessage('Busy %d', 1, msec=20)
    assert bar._perm.text() == 'Busy 1'
    bar.temporaryMessage(['Still ', 'busy'], msec=40)
    assert bar._perm.text() == 'Still busy'
    assert wait_until(lambda: bar._perm.text() == 'Ready')


# --------------------------------------------------------------------------
# VScrollLayout, Viewport, VScrollArea
# --------------------------------------------------------------------------

def test_empty_vscroll_layout():
    layout = qtgui.VScrollLayout()
    assert layout.count() == 0
    assert layout.itemAt(0) is None and layout.itemAt(1) is None
    assert layout.childHeight() == 0
    assert int(layout.expandingDirections()) == 0
    assert layout.hasHeightForWidth() is False
    assert layout.heightForWidth(100) == -1
    left, top, right, bottom = layout.getContentsMargins()
    assert layout.sizeHint() == QtCore.QSize(left + right, top + bottom)
    assert layout.minimumSize() == QtWidgets.QLayout.minimumSize(layout)
    assert layout.maximumSize() == QtWidgets.QLayout.maximumSize(layout)
    layout.setGeometry(QtCore.QRect(0, 0, 10, 10))  # nothing to place


def test_vscroll_layout_sizes_and_positions_its_child():
    host = QtWidgets.QWidget()
    layout = qtgui.VScrollLayout(host)
    child = QtWidgets.QWidget()
    child.setMinimumSize(80, 300)
    layout.addWidget(child)
    assert layout.count() == 1
    assert layout.itemAt(0).widget() is child
    assert layout.itemAt(1) is None
    assert layout.sizeHint() == QtCore.QSize(80, 300)
    assert layout.minimumSize() == QtCore.QSize(80, 0)  # can always scroll
    assert layout.maximumSize() == child.maximumSize()
    assert layout.childHeight() == 300
    assert layout.expandingDirections() == layout.itemAt(0).expandingDirections()
    assert layout.hasHeightForWidth() == layout.itemAt(0).hasHeightForWidth()
    assert layout.heightForWidth(50) == layout.itemAt(0).heightForWidth(50)

    layout.setPosition(20)
    layout.setGeometry(QtCore.QRect(0, 0, 100, 50))
    assert child.geometry() == QtCore.QRect(0, -20, 80, 300)

    child.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    layout.setGeometry(QtCore.QRect(0, 0, 100, 500))
    assert child.geometry() == QtCore.QRect(0, -20, 100, 500)

    # only one child: a second one is refused (called directly: a Python
    # exception raised while Qt calls addItem would abort the process)
    with pytest.raises(RuntimeError):
        layout.addItem(QtWidgets.QWidgetItem(QtWidgets.QWidget()))


def test_vscroll_layout_take_at_returns_the_child():
    host = QtWidgets.QWidget()
    layout = qtgui.VScrollLayout(host)
    layout.addWidget(QtWidgets.QLabel('x'))
    item = layout.takeAt(0)
    assert item is not None and layout.count() == 0


AREA_SCRIPT = r"""
import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
mh = sys.argv[1]
for sub in reversed(["", "lib", "apps", "shared", os.path.join("apps", "gui"), "core"]):
    sys.path.insert(0, os.path.join(mh, sub))
from core import G
G.args = {}
from PyQt5 import QtWidgets
app = QtWidgets.QApplication([])
import qtgui
area = qtgui.VScrollArea()
mode = sys.argv[2]
if mode == "show_empty":
    area.show()
    app.processEvents()
else:
    area.setWidget(QtWidgets.QLabel("first"))
    if mode == "clear":
        area.setWidget(None)
    else:
        area.setWidget(QtWidgets.QLabel("second"))
print("survived")
"""


def run_area_script(mode):
    """These cases abort the interpreter, so they run in a child process."""
    proc = subprocess.run([sys.executable, "-c", AREA_SCRIPT, MH_DIR, mode],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "survived"


@pytest.mark.parametrize("change", ["clear", "replace"])
def test_vscroll_area_widget_can_be_changed(change):
    run_area_script(change)


def test_empty_vscroll_area_can_be_shown():
    run_area_script("show_empty")


def test_vscroll_area_scrolls_its_child(qapp):
    area = qtgui.VScrollArea()
    _KEEP.append(area)
    assert area.getClassName() == 'VScrollArea'
    child = QtWidgets.QWidget()
    child.setMinimumSize(80, 300)
    area.setWidget(child)
    area.resize(100, 120)
    area.show()
    assert wait_until(lambda: area._viewport.height() == 120)
    send(area, QtGui.QResizeEvent(area.size(), area.size()))
    bar = area._scrollbar
    assert bar.maximum() == 300 - 120
    assert bar.pageStep() == 120

    bar.setValue(50)
    assert wait_until(lambda: child.y() == -50)

    # the child growing (any non-resize event of it) updates the range
    child.setMinimumSize(80, 500)
    assert area.eventFilter(child, QtCore.QEvent(QtCore.QEvent.LayoutRequest)) is False
    assert bar.maximum() == 500 - 120
    child.setMinimumSize(80, 400)
    area.eventFilter(child, QtCore.QEvent(QtCore.QEvent.Resize))  # ignored
    area.eventFilter(QtWidgets.QWidget(), QtCore.QEvent(QtCore.QEvent.Show))  # not the child
    assert bar.maximum() == 500 - 120
    area.hide()


# --------------------------------------------------------------------------
# ImageView and ZoomableImageView
# --------------------------------------------------------------------------

def test_get_pixmap_accepts_all_image_kinds(tmp_path):
    import image
    path = save_png(tmp_path / 'p.png', 6, 3, '#ff0000')
    pm = QtGui.QPixmap(2, 2)
    assert qtgui.getPixmap(pm) is pm
    assert qtgui.getPixmap(solid_image(4, 5, 'red')).size() == QtCore.QSize(4, 5)
    assert qtgui.getPixmap(path).size() == QtCore.QSize(6, 3)
    # An RGB image: RGBA ones come out with red and blue swapped by
    # image.Image.toQImage (already an xfail in test_image.py)
    rgb = QtGui.QImage(6, 3, QtGui.QImage.Format_RGB32)
    rgb.fill(QtGui.QColor('#ff0000'))
    rgb_path = str(tmp_path / 'rgb.png')
    assert rgb.save(rgb_path)
    from_image = qtgui.getPixmap(image.Image(rgb_path))
    assert from_image.size() == QtCore.QSize(6, 3)
    assert from_image.toImage().pixelColor(0, 0).name() == '#ff0000'


def test_image_view_without_image(tmp_path):
    view = qtgui.ImageView()
    assert view.sizeHint() == QtWidgets.QLabel.sizeHint(view)
    assert view.heightForWidth(70) == 70
    view.refreshImage()
    view.save(str(tmp_path / 'none'))
    assert not os.path.exists(str(tmp_path / 'none.png'))


def test_image_view_scales_down_to_its_size(tmp_path):
    view = qtgui.ImageView()
    view.setImage(save_png(tmp_path / 'wide.png', 200, 100, '#00ff00'))
    assert view.sizeHint() == QtCore.QSize(200, 100)
    assert view.heightForWidth(100) == 50
    assert view.pixmap().size() == QtCore.QSize(200, 100)

    view.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(100, 100), QtCore.QSize(200, 100)))
    assert view.workingSize == QtCore.QSize(100, 100)
    assert view.pixmap().size() == QtCore.QSize(100, 50)

    view.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(400, 400), QtCore.QSize(100, 100)))
    assert view.pixmap().size() == QtCore.QSize(200, 100)  # never scaled up

    view.save(str(tmp_path / 'copy'))
    assert QtGui.QImage(str(tmp_path / 'copy.png')).size() == QtCore.QSize(200, 100)
    view.save(str(tmp_path / 'copy.jpg'))
    assert os.path.isfile(str(tmp_path / 'copy.jpg'))


@pytest.fixture
def zoom_view(qapp):
    view = qtgui.ZoomableImageView()
    _KEEP.append(view)
    view.resize(200, 100)
    view.show()
    QTest.qWait(10)
    yield view
    view.hide()


def test_zoomable_view_fits_new_images_to_its_width(zoom_view):
    view = zoom_view
    assert view.heightForWidth(30) == 30  # no image yet
    view.refreshImage()
    view.setImage(solid_image(400, 200, 'blue'))
    assert view.ratio == pytest.approx(200 / 400)
    assert view.minratio == pytest.approx(100 / 400)
    assert view.imageLabel.size() == QtCore.QSize(200, 100)
    assert view.heightForWidth(100) == 50

    view.setImage(solid_image(50, 50, 'blue'))
    assert view.minratio == 1.0  # small images cannot be shrunk


def test_zoomable_view_follows_resizes(zoom_view):
    view = zoom_view
    view.setImage(solid_image(400, 200, 'blue'))
    view.workingWidth = None
    view.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(100, 100), QtCore.QSize(200, 100)))
    assert view.workingWidth == 100 and view.ratio == pytest.approx(0.5)
    view.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(150, 100), QtCore.QSize(100, 100)))
    assert view.ratio == pytest.approx(0.75)
    view.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(600, 100), QtCore.QSize(150, 100)))
    assert view.ratio == 1.0  # never larger than the image
    assert view.imageLabel.size() == QtCore.QSize(400, 200)


def wheel(view, dy, pos=(10, 10)):
    point = QtCore.QPointF(*pos)
    view.wheelEvent(QtGui.QWheelEvent(point, point, QtCore.QPoint(0, 0), QtCore.QPoint(0, dy),
                                      QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
                                      QtCore.Qt.NoScrollPhase, False))


def test_zoomable_view_wheel_zoom(zoom_view, app):
    view = zoom_view
    view.setImage(solid_image(400, 200, 'blue'))
    wheel(view, 120)
    assert view.ratio == pytest.approx(0.5 * (1 + 120 * 0.0007))
    app.settings['invertMouseWheel'] = True
    before = view.ratio
    wheel(view, 120)
    assert view.ratio == pytest.approx(before * (1 - 120 * 0.0007))
    for _ in range(40):
        wheel(view, 120)
    assert view.ratio == view.minratio
    app.settings['invertMouseWheel'] = False
    for _ in range(40):
        wheel(view, 120)
    assert view.ratio == 1.0
    assert view.imageLabel.size() == QtCore.QSize(400, 200)


def mouse(kind, pos, buttons):
    button = QtCore.Qt.NoButton if kind == QtCore.QEvent.MouseMove else buttons
    return QtGui.QMouseEvent(kind, QtCore.QPointF(*pos), button, buttons, QtCore.Qt.NoModifier)


def test_zoomable_view_drag_pans_and_right_drag_zooms(zoom_view, app):
    view = zoom_view
    view.setImage(solid_image(400, 200, 'blue'))
    view.ratio = 1.0
    view.refreshImage()
    QTest.qWait(10)
    h = view.horizontalScrollBar()
    assert h.maximum() > 0
    h.setValue(50)

    view.mouseMoveEvent(mouse(QtCore.QEvent.MouseMove, (5, 5), QtCore.Qt.LeftButton))
    assert view.mdown == QtCore.QPoint(5, 5)  # first move without press just records
    view.mousePressEvent(mouse(QtCore.QEvent.MouseButtonPress, (100, 50), QtCore.Qt.LeftButton))
    assert view.mdown == QtCore.QPoint(100, 50)
    view.mouseMoveEvent(mouse(QtCore.QEvent.MouseMove, (90, 50), QtCore.Qt.LeftButton))
    assert h.value() == 50 + 2 * 10
    assert view.mdown == QtCore.QPoint(90, 50)

    view.ratio = 0.5
    view.mousePressEvent(mouse(QtCore.QEvent.MouseButtonPress, (100, 50), QtCore.Qt.RightButton))
    view.mouseMoveEvent(mouse(QtCore.QEvent.MouseMove, (100, 60), QtCore.Qt.RightButton))
    assert view.ratio == pytest.approx(0.5 * (1 + 100 * 0.0007))
    app.settings['invertMouseWheel'] = True
    before = view.ratio
    view.mouseMoveEvent(mouse(QtCore.QEvent.MouseMove, (100, 70), QtCore.Qt.RightButton))
    assert view.ratio == pytest.approx(before * (1 - 100 * 0.0007))
    for y in range(80, 2000, 100):
        view.mouseMoveEvent(mouse(QtCore.QEvent.MouseMove, (100, y), QtCore.Qt.RightButton))
    assert view.ratio == view.minratio
    app.settings['invertMouseWheel'] = False  # dragging down zooms in
    for y in range(2100, 4000, 100):
        view.mouseMoveEvent(mouse(QtCore.QEvent.MouseMove, (100, y), QtCore.Qt.RightButton))
    assert view.ratio == 1.0


def test_zoomable_view_save(zoom_view, tmp_path):
    view = zoom_view
    view.save(str(tmp_path / 'empty'))
    assert not os.path.exists(str(tmp_path / 'empty.png'))
    view.setImage(solid_image(40, 20, 'red'))
    view.save(str(tmp_path / 'shot'))
    assert QtGui.QImage(str(tmp_path / 'shot.png')).size() == QtCore.QSize(40, 20)
    view.save(str(tmp_path / 'shot.thumb'), iformat='PNG')
    assert QtGui.QImage(str(tmp_path / 'shot.thumb'), 'PNG').size() == QtCore.QSize(40, 20)
