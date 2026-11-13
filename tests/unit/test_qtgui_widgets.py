"""
Unit tests for the basic widgets of lib/qtgui.py, run headless on the Qt
offscreen platform: tabs, group boxes, the slider and its style, buttons,
list views, text widgets, validators and small helpers.

MakeHuman widgets report user actions through events3d: a handler attached
with ``widget.mhEvent`` (or ``attachEvent``) is called by ``callEvent``.
callEvent swallows exceptions raised by handlers and reports them by calling
``G.app.progress(1)``; the FakeApp below records that call and the ``app``
fixture fails the test if it happened, so a broken handler cannot hide.
"""

import pytest
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtTest import QTest

from core import G
import language
import qtgui

_QAPP = []  # keeps the QApplication alive for the whole session


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
    """Attach handlers for the given events; return the list they append to."""
    calls = []
    for name in names:
        widget.attachEvent(name, lambda event, name=name: calls.append((name, event)))
    return calls


def send(widget, event):
    return QtWidgets.QApplication.sendEvent(widget, event)


# --------------------------------------------------------------------------
# Translation and the Widget base class
# --------------------------------------------------------------------------

def test_get_language_string_without_translation_table():
    assert qtgui.getLanguageString('') == ''
    assert qtgui.getLanguageString(None) is None
    assert qtgui.getLanguageString('Hello') == 'Hello'
    assert qtgui.getLanguageString('Size', ': 3') == 'Size: 3'
    assert qtgui.getLanguageString('Size', 3, ' %d cm') == 'Size 3 cm'


def test_widgets_translate_their_labels(monkeypatch):
    monkeypatch.setattr(language.language, "languageStrings", {'Hello': 'Hola', 'Sort': 'Ordenar'})
    assert qtgui.Button('Hello').getLabel() == 'Hola'
    assert qtgui.TextView('Hello').text() == 'Hola'
    assert qtgui.GroupBox('Sort').title() == 'Ordenar'


def test_show_hide_and_focus_reach_event_handlers():
    view = qtgui.TextView('x')
    calls = record(view, 'onShow', 'onHide', 'onFocus', 'onBlur')
    view.show()
    view.hide()
    send(view, QtGui.QFocusEvent(QtCore.QEvent.FocusIn))
    send(view, QtGui.QFocusEvent(QtCore.QEvent.FocusOut))
    assert [name for name, _ in calls] == ['onShow', 'onHide', 'onFocus', 'onBlur']
    assert all(event is view for _, event in calls)


def test_default_widget_handlers_are_no_ops():
    view = qtgui.TextView('x')
    for name in ('onFocus', 'onBlur', 'onShow', 'onHide'):
        assert getattr(qtgui.Widget, name)(view, None) is None


# --------------------------------------------------------------------------
# Tabs and TabBar
# --------------------------------------------------------------------------

def test_tabs_add_find_and_change():
    tabs = qtgui.Tabs()
    selected = record(tabs, 'onTabSelected')
    modelling = tabs.addTab('modelling', 'Modelling')
    files = tabs.addTab('files', 'Files')
    assert (modelling.idx, files.idx) == (0, 1)
    assert tabs.tabText(1) == 'Files'
    assert tabs.findTab('files') is files
    assert tabs.findTab('nope') is None
    clicked = record(files, 'onClicked')

    tabs.changeTab('files')
    assert tabs.currentIndex() == 1
    assert selected == [('onTabSelected', files)]
    assert clicked == [('onClicked', files)]

    tabs.changeTab('missing')  # unknown names are ignored
    assert tabs.currentIndex() == 1


def test_tab_selection_also_refreshes_the_task_bar_of_the_category():
    tabs = qtgui.Tabs()
    modelling = tabs.addTab('modelling', 'Modelling')
    tabs.addTab('files', 'Files')
    main = modelling.child.addTab('main', 'Main')
    modelling.child.addTab('gender', 'Gender')
    task_selected = record(modelling.child, 'onTabSelected')
    tabs.changeTab('files')
    tabs.changeTab('modelling')
    # Tabs.tabChanged re-announces the current task of the selected category
    assert task_selected == [('onTabSelected', main)]


def test_inserting_tabs_renumbers_the_existing_ones():
    tabs = qtgui.Tabs()
    a = tabs.addTab('a', 'A')
    b = tabs.addTab('b', 'B')
    c = tabs.addTab('c', 'C', 1)
    assert (a.idx, c.idx, b.idx) == (0, 1, 2)
    assert [tabs.tabText(i) for i in range(3)] == ['A', 'C', 'B']
    assert tabs._tabs_by_idx == {0: a, 1: c, 2: b}


def test_tabbar_add_insert_and_change():
    bar = qtgui.TabBar()
    assert bar.tabBar() is bar
    assert not bar.drawBase()
    selected = record(bar, 'onTabSelected')
    x = bar.addTab('x', 'X')
    y = bar.addTab('y', 'Y')
    w = bar.addTab('w', 'W', 0)
    assert (w.idx, x.idx, y.idx) == (0, 1, 2)
    assert [bar.tabText(i) for i in range(3)] == ['W', 'X', 'Y']
    selected.clear()
    bar.changeTab('y')
    assert selected == [('onTabSelected', y)]
    assert qtgui.TabsBase.onTabSelected(bar, None) is None
    assert qtgui.Tab.onClicked(y, None) is None


def test_inserting_at_index_zero_announces_the_inserted_tab():
    tabs = qtgui.Tabs()
    tabs.addTab('a', 'A')
    tabs.addTab('b', 'B')
    selected = record(tabs, 'onTabSelected')
    c = tabs.addTab('c', 'C', 0)
    assert tabs.currentIndex() == 0 and tabs.tabText(0) == 'C'
    assert selected[-1] == ('onTabSelected', c)


# --------------------------------------------------------------------------
# GroupBox
# --------------------------------------------------------------------------

def test_groupbox_add_remove_and_children():
    box = qtgui.GroupBox('Options')
    assert str(box).endswith(' - Options')
    first = box.addWidget(qtgui.Button('one'))
    second = box.addWidget(qtgui.Button('two'), tooltip='second button')
    assert box.count() == 2
    assert box.children == [first, second]
    assert box.itemAt(1).widget() is second
    assert second.toolTip() == 'second button'
    # rows default to the number of widgets already in the box
    assert box.layout.getItemPosition(1)[:2] == (1, 0)
    assert not first.isHidden()

    box.removeWidget(first)
    assert box.children == [second]
    assert first.parent() is None

    assert qtgui.GroupBox().title() == ''


# --------------------------------------------------------------------------
# SliderStyle and NarrowLineEdit
# --------------------------------------------------------------------------

class RecordingStyle(object):
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args))
            return ('result of', name)
        return method


SLIDERSTYLE_DELEGATES = [
    ('drawComplexControl', (1, 2, 3), (1, 2, 3, None)),
    ('drawControl', (1, 2, 3), (1, 2, 3, None)),
    ('drawItemPixmap', (1, 2, 3, 4), (1, 2, 3, 4)),
    ('drawItemText', (1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6, QtGui.QPalette.NoRole)),
    ('drawPrimitive', (1, 2, 3), (1, 2, 3, None)),
    ('generatedIconPixmap', (1, 2, 3), (1, 2, 3)),
    ('hitTestComplexControl', (1, 2, 3), (1, 2, 3, None)),
    ('itemPixmapRect', (1, 2, 3), (1, 2, 3)),
    ('itemTextRect', (1, 2, 3, 4, 5), (1, 2, 3, 4, 5)),
    ('pixelMetric', (1,), (1, None, None)),
    ('polish', (1,), (1,)),
    ('styleHint', (QtWidgets.QStyle.SH_Slider_StopMouseOverSlider,),
     (QtWidgets.QStyle.SH_Slider_StopMouseOverSlider, None, None, None)),
    ('subControlRect', (1, 2, 3), (1, 2, 3, None)),
    ('subElementRect', (1, 2), (1, 2, None)),
    ('unpolish', (1,), (1,)),
    ('sizeFromContents', (1, 2, 3), (1, 2, 3, None)),
]


@pytest.mark.parametrize("name, args, forwarded", SLIDERSTYLE_DELEGATES)
def test_sliderstyle_delegates_to_the_underlying_style(monkeypatch, name, args, forwarded):
    rec = RecordingStyle()
    # Older versions delegate to the style passed in, newer ones to the
    # current application style: route both to the recorder.
    monkeypatch.setattr(QtWidgets.QApplication, "style", staticmethod(lambda: rec))
    style = qtgui.SliderStyle(rec)
    assert getattr(style, name)(*args) == ('result of', name)
    assert rec.calls == [(name, forwarded)]


def test_sliderstyle_makes_every_mouse_button_jump_to_the_click(monkeypatch):
    rec = RecordingStyle()
    monkeypatch.setattr(QtWidgets.QApplication, "style", staticmethod(lambda: rec))
    style = qtgui.SliderStyle(rec)
    buttons = style.styleHint(QtWidgets.QStyle.SH_Slider_AbsoluteSetButtons)
    assert int(buttons) == int(QtCore.Qt.LeftButton | QtCore.Qt.MidButton | QtCore.Qt.RightButton)
    assert rec.calls == []


def test_sliderstyle_answers_like_the_application_style():
    slider = qtgui.Slider()
    style = slider.slider.style()
    assert isinstance(style, qtgui.SliderStyle)
    metric = QtWidgets.QStyle.PM_SliderLength
    assert style.pixelMetric(metric) == QtWidgets.QApplication.style().pixelMetric(metric)


def test_narrow_line_edit_width_follows_its_column_count():
    narrow = qtgui.NarrowLineEdit(2).sizeHint()
    wide = qtgui.NarrowLineEdit(12).sizeHint()
    assert wide.width() > narrow.width()
    assert wide.height() == narrow.height() > 0


# --------------------------------------------------------------------------
# Slider
# --------------------------------------------------------------------------

class CmConverter(object):
    units = 'cm'

    def dataToDisplay(self, value):
        return value * 100

    def displayToData(self, value):
        return value / 100.0


class PlainConverter(object):
    def dataToDisplay(self, value):
        return value * 10

    def displayToData(self, value):
        return value / 10.0


def test_slider_initial_state_and_label_formatting():
    s = qtgui.Slider(value=0.25, min=0.0, max=1.0, label='Weight %.2f', tooltip='tip')
    assert s.getValue() == pytest.approx(0.25)
    assert s.slider.value() == 250
    assert s.label.text() == 'Weight 0.25'
    assert s.edit.text() == '0.25'
    assert s.slider.toolTip() == 'tip'
    assert s.edit.isHidden() and s.units.isHidden()
    assert s.slider.orientation() == QtCore.Qt.Horizontal
    assert not s.slider.hasTracking()


def test_slider_without_label_hides_it_and_can_be_vertical():
    s = qtgui.Slider(vertical=True)
    assert s.label.isHidden()
    assert s.slider.orientation() == QtCore.Qt.Vertical


def test_slider_set_value_clamps_and_converts():
    s = qtgui.Slider(value=0.0, min=-1.0, max=1.0, label='Value %.1f')
    s.setValue(0.5)
    assert s.slider.value() == 750
    assert s.getValue() == pytest.approx(0.5)
    # QSlider clamps its integer position by itself; the Python clamp is what
    # keeps the edit box and the label inside the range
    s.setValue(7)
    assert s.getValue() == pytest.approx(1.0)
    assert s.edit.text() == '1.00' and s.label.text() == 'Value 1.0'
    s.setValue(-7)
    assert s.getValue() == pytest.approx(-1.0)
    assert s.edit.text() == '-1.00'


def test_set_value_does_not_emit_change_events():
    s = qtgui.Slider()
    calls = record(s, 'onChange', 'onChanging')
    s.setValue(0.4)
    s.setValue(0.4)  # same position: early return
    assert calls == []
    assert s.edit.text() == '0.40'


def test_reversed_slider_maps_values_backwards():
    s = qtgui.Slider(value=1.0, min=1.0, max=0.0)
    assert s.slider.value() == 0
    s.setValue(0.25)
    assert s.slider.value() == 750
    assert s.getValue() == pytest.approx(0.25)
    s.setValue(5)  # clamped to the virtual maximum
    assert s.getValue() == pytest.approx(1.0)
    assert s.edit.text() == '1.00'


def test_changing_min_or_max_keeps_the_value():
    s = qtgui.Slider(value=0.5)
    s.setMax(2.0)
    assert s.getValue() == pytest.approx(0.5)
    assert s.slider.value() == 250
    s.setMin(0.25)
    assert s.getValue() == pytest.approx(0.5, abs=1.75 / 1000)  # one slider step
    assert s.slider.value() == round(1000 * 0.25 / 1.75)


def test_value_converter_shows_units_and_edit_box():
    s = qtgui.Slider(value=0.5, valueConverter=CmConverter())
    assert not s.units.isHidden() and s.units.text() == 'cm'
    assert not s.edit.isHidden()
    assert s.edit.text() == '50.00'
    assert s.toDisplay(0.1) == pytest.approx(10)
    assert s.fromDisplay(10) == pytest.approx(0.1)

    s.valueConverter = PlainConverter()
    assert s.units.isHidden()
    assert not s.edit.isHidden()

    s.valueConverter = None
    assert s.units.isHidden()
    assert s.edit.isHidden()
    assert s.toDisplay(0.3) == 0.3 and s.fromDisplay(0.3) == 0.3


def test_units_label_follows_a_converter_whose_units_change():
    conv = CmConverter()
    s = qtgui.Slider(value=0.5, valueConverter=conv)
    conv.units = 'inch'
    s.setValue(0.6)
    assert s.units.text() == 'inch'


def test_edit_widget_visibility_toggle():
    s = qtgui.Slider()
    s.setEditWidgetVisibility(True)
    assert not s.edit.isHidden()
    s.setEditWidgetVisibility(False)
    assert s.edit.isHidden()


def test_typing_a_value_and_pressing_return_changes_the_slider():
    s = qtgui.Slider(value=0.5, valueConverter=CmConverter())
    calls = record(s, 'onChanging', 'onChange')
    s.edit.clear()
    QTest.keyClicks(s.edit, '25')
    QTest.keyClick(s.edit, QtCore.Qt.Key_Return)
    assert s.getValue() == pytest.approx(0.25)
    assert calls == [('onChanging', pytest.approx(0.25)), ('onChange', pytest.approx(0.25))]


def test_return_with_same_or_empty_value_emits_nothing():
    s = qtgui.Slider(value=0.5)
    calls = record(s, 'onChanging', 'onChange')
    s.setEditWidgetVisibility(True)
    s.edit.setText('0.5004')
    s._enter()
    s.edit.setText('')
    s._enter()
    assert calls == []
    assert s.getValue() == pytest.approx(0.5)


def test_dragging_the_slider_emits_press_changing_release_change():
    s = qtgui.Slider()
    calls = record(s, 'onPress', 'onChanging', 'onRelease', 'onChange')
    s.slider.setSliderDown(True)
    s.slider.setSliderPosition(300)
    s.slider.setSliderDown(False)
    assert [name for name, _ in calls] == ['onPress', 'onChanging', 'onRelease', 'onChange']
    assert calls[1][1] == pytest.approx(0.3)
    assert calls[3][1] == pytest.approx(0.3)
    assert calls[0][1] is s and calls[2][1] is s
    assert s.getValue() == pytest.approx(0.3)


def test_clicking_the_track_jumps_to_the_click_position():
    s = qtgui.Slider()
    calls = record(s, 'onChange')
    s.slider.resize(200, 20)
    press = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, QtCore.QPointF(150, 10),
                              QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    s.slider.mousePressEvent(press)
    assert 0.6 < s.getValue() < 0.9
    assert calls and calls[-1][1] == pytest.approx(s.getValue())


def test_slider_subclass_can_swallow_mouse_presses():
    class Locked(qtgui.Slider):
        def sliderMousePressEvent(self, event):
            self.seen = event.button()
            return False

    s = Locked()
    s.slider.resize(200, 20)
    press = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, QtCore.QPointF(150, 10),
                              QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    s.slider.mousePressEvent(press)
    assert s.seen == QtCore.Qt.LeftButton
    assert s.getValue() == 0.0
    assert qtgui.Slider().sliderMousePressEvent(press) is True


def test_slider_focus_events_come_from_the_slider_and_the_edit_box():
    s = qtgui.Slider()
    calls = record(s, 'onFocus', 'onBlur')
    send(s.slider, QtGui.QFocusEvent(QtCore.QEvent.FocusIn))
    send(s.edit, QtGui.QFocusEvent(QtCore.QEvent.FocusOut))
    assert calls == [('onFocus', s), ('onBlur', s)]
    # events of other objects are not filtered
    assert s.eventFilter(QtCore.QObject(), QtGui.QFocusEvent(QtCore.QEvent.FocusIn)) is False
    assert len(calls) == 2
    assert qtgui.Slider.onChange(s, 1) is None and qtgui.Slider.onChanging(s, 1) is None


def test_slider_images_follow_the_global_switch(monkeypatch, tmp_path):
    monkeypatch.setattr(qtgui.Slider, "_show_images", False)
    monkeypatch.setattr(qtgui.Slider, "_imageCache", {})
    path = str(tmp_path / 'slider.png')
    img = QtGui.QImage(8, 4, QtGui.QImage.Format_RGB32)
    img.fill(QtGui.QColor('red'))
    assert img.save(path)

    a = qtgui.Slider(image=path)
    b = qtgui.Slider(image=path)
    assert a.image.pixmap().size() == QtCore.QSize(8, 4)
    assert qtgui.Slider._getImage(path) is qtgui.Slider._imageCache[path]
    assert list(qtgui.Slider._imageCache) == [path]
    assert a.image.isHidden() and b.image.isHidden()

    qtgui.Slider.showImages(True)
    assert qtgui.Slider.imagesShown() is True
    assert not a.image.isHidden() and not b.image.isHidden()
    qtgui.Slider.showImages(False)
    assert a.image.isHidden() and b.image.isHidden()


# --------------------------------------------------------------------------
# Buttons
# --------------------------------------------------------------------------

def test_button_label_and_click_event():
    b = qtgui.Button('Go')
    assert b.getLabel() == 'Go'
    b.setLabel('Stop')
    assert b.text() == 'Stop'
    calls = record(b, 'onClicked')
    b.click()
    assert calls == [('onClicked', None)]
    assert qtgui.ButtonBase.onClicked(b, None) is None


def test_checkbox_selection():
    c = qtgui.CheckBox('Wire', selected=True)
    assert c.selected is True
    c.setSelected(False)
    assert c.selected is False and not c.isChecked()
    calls = record(c, 'onClicked')
    c.click()
    assert c.selected is True
    assert calls == [('onClicked', None)]


def test_radio_buttons_of_a_group_are_exclusive(monkeypatch):
    monkeypatch.setattr(qtgui.RadioButton, "groups", {})
    group = []
    small = qtgui.RadioButton(group, 'Small', selected=True)
    large = qtgui.RadioButton(group, 'Large')
    assert group == [small, large]
    assert small.getSelection() is small
    large.click()
    assert large.selected and not small.selected
    assert small.getSelection() is large

    other = []
    lonely = qtgui.RadioButton(other, 'Alone', selected=True)
    large.click()
    assert lonely.selected  # separate list, separate QButtonGroup


def test_radio_button_group_without_selection():
    group = []
    qtgui.RadioButton(group, 'a')
    qtgui.RadioButton(group, 'b')
    assert group[0].getSelection() is None


def test_removing_radio_buttons_from_their_group(monkeypatch):
    monkeypatch.setattr(qtgui.RadioButton, "groups", {})
    group = []
    a = qtgui.RadioButton(group, 'a')
    b = qtgui.RadioButton(group, 'b')
    qgroup = qtgui.RadioButton.groups[id(group)]
    a._removeFromGroup(group)
    assert qgroup.buttons() == [b]
    b._removeFromGroup(group)
    assert id(group) not in qtgui.RadioButton.groups
    b._removeFromGroup(group)  # unknown group: nothing to do
    assert qtgui.RadioButton.groups == {}


# --------------------------------------------------------------------------
# ListItem and ListView
# --------------------------------------------------------------------------

def test_list_item_user_data_text_and_checkbox():
    view = qtgui.ListView()
    item = view.addItem('alpha', data={'k': 1})
    assert item.text == 'alpha'
    assert item.getUserData() == {'k': 1}
    assert not item.hasCheckbox and not item.isChecked()

    item.setChecked(True)  # without checkbox: selection
    assert item.isSelected()
    item.setChecked(False)
    assert not item.isSelected()

    boxed = view.addItem('beta', checkbox=True)
    assert boxed.hasCheckbox
    assert boxed.flags() & QtCore.Qt.ItemIsUserCheckable
    boxed.setChecked(True)
    assert boxed.isChecked() and boxed.checkState() == QtCore.Qt.Checked
    boxed.setChecked(False)
    assert not boxed.isChecked()


def test_clicking_a_checkbox_item_reports_checked_and_unchecked():
    view = qtgui.ListView()
    calls = record(view, 'onItemChecked', 'onItemUnchecked', 'onClicked')
    item = view.addItem('beta', checkbox=True)
    item.setCheckState(QtCore.Qt.Checked)  # what the user's click does
    view.itemClicked.emit(item)
    item.setCheckState(QtCore.Qt.Unchecked)
    view.itemClicked.emit(item)
    assert calls == [('onItemChecked', item), ('onItemUnchecked', item)]
    # a click that does not change the check state is a plain click
    view.itemClicked.emit(item)
    assert calls[-1] == ('onClicked', item)


def test_clicks_in_single_and_multi_selection_mode():
    view = qtgui.ListView()
    calls = record(view, 'onClicked', 'onItemChecked', 'onItemUnchecked', 'onActivate')
    a = view.addItem('a')
    view.itemClicked.emit(a)
    assert calls == [('onClicked', a)]

    view.allowMultipleSelection(True)
    assert view.allowsMultipleSelection()
    calls.clear()
    a.setSelected(True)
    view.itemClicked.emit(a)
    a.setSelected(False)
    view.itemClicked.emit(a)
    assert calls == [('onItemChecked', a), ('onItemUnchecked', a)]

    view.itemActivated.emit(a)
    assert calls[-1] == ('onActivate', a)

    view.allowMultipleSelection(False)
    assert not view.allowsMultipleSelection()
    assert qtgui.ListView.onActivate(view, None) is None
    assert qtgui.ListView.onClicked(view, None) is None


def test_list_view_items_colors_data_and_visibility():
    view = qtgui.ListView()
    view.setData(['one', 'two'])
    assert [i.text for i in view.getItems()] == ['one', 'two']
    view.setData(['x', 'y', 'z'])
    assert [i.text for i in view.getItems()] == ['x', 'y', 'z']

    first = view.addItem('first', color='#ff0000', data=5, pos=0)
    assert view.item(0) is first
    assert view.getItemData(0) == 5
    assert first.foreground().color() == QtGui.QColor('#ff0000')
    assert qtgui.ListView.getBrush('#ff0000') is qtgui.ListView.getBrush('#ff0000')
    view.setItemColor(1, '#00ff00')
    assert view.item(1).foreground().color() == QtGui.QColor('#00ff00')

    assert view.rowCount() == 4
    view.showItem(1, False)
    assert view.item(1).isHidden()
    assert view.rowCount() == 3
    view.showItem(1, True)
    assert view.rowCount() == 4


def test_list_view_selection_helpers():
    view = qtgui.ListView()
    assert view.getSelectedItem() is None
    a = view.addItem('a')
    b = view.addItem('b')
    view.allowMultipleSelection(True)
    a.setSelected(True)
    b.setSelected(True)
    assert view.getSelectedItem() in ('a', 'b')
    assert sorted(view.getSelectedItems()) == ['a', 'b']


def test_clear_selection_unchecks_items_and_reports_it():
    view = qtgui.ListView()
    calls = record(view, 'onClearSelection')
    plain = view.addItem('plain')
    boxed = view.addItem('boxed', checkbox=True)
    plain.setSelected(True)
    boxed.setChecked(True)
    view.clearSelection()
    assert not plain.isSelected()
    assert not boxed.isChecked()
    assert calls == [('onClearSelection', None)]


def test_list_view_without_vertical_scrolling_is_as_tall_as_its_rows():
    view = qtgui.ListView()
    view.setVerticalScrollingEnabled(False)
    assert view.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff
    assert view.sizeHint().height() == 0
    for name in 'abc':
        view.addItem(name)
    row = max(view.sizeHintForRow(0), view.iconSize().height())
    assert view.sizeHint().height() == 3 * row
    view.showItem(0, False)
    assert view.sizeHint().height() == 2 * row

    view.setVerticalScrollingEnabled(True)
    assert view.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded
    assert view.sizeHint() == QtWidgets.QListWidget.sizeHint(view)


# --------------------------------------------------------------------------
# TextView, validators, TextEdit, DocumentEdit, ProgressBar
# --------------------------------------------------------------------------

def test_text_view_set_text_and_format():
    view = qtgui.TextView('start')
    view.setText('plain')
    assert view.text() == 'plain'
    view.setTextFormat('%d of %d', 3, 7)
    assert view.text() == '3 of 7'


def test_text_view_with_word_wrap_grows_to_fit_its_text():
    view = qtgui.TextView('word ' * 60)
    view.resize(80, 10)
    send(view, QtGui.QResizeEvent(QtCore.QSize(80, 10), QtCore.QSize(0, 0)))
    assert view.minimumHeight() == 0  # no word wrap: untouched
    view.setWordWrap(True)
    send(view, QtGui.QResizeEvent(QtCore.QSize(80, 10), QtCore.QSize(0, 0)))
    assert view.minimumHeight() == view.heightForWidth(80) > 10


@pytest.mark.parametrize("text, ok", [
    ('', True), ('12', True), ('-', True), ('-12', True),
    ('1.5', False), ('a', False), ('1-', False), ('--1', False)])
def test_int_validator(text, ok):
    assert qtgui.intValidator(text) == ok


@pytest.mark.parametrize("text, ok", [
    ('', True), ('12', True), ('1.5', True), ('.5', True), ('-', True), ('-1.5', True),
    ('1.2.3', False), ('-1.2.3', False), ('abc', False), ('1e5', False)])
def test_float_validator(text, ok):
    assert bool(qtgui.floatValidator(text)) == ok


@pytest.mark.parametrize("text, ok", [
    ('', True), ('my model', True), ('a/b', False), ('a\\b', False), ('a:b', False),
    ('a*', False), ('a?', False), ('"a"', False), ('<a>', False), ('a|b', False)])
def test_filename_validator(text, ok):
    assert qtgui.filenameValidator(text) == ok


def test_text_edit_uses_matching_qt_validators():
    edit = qtgui.TextEdit('x', qtgui.intValidator)
    assert isinstance(edit.validator(), QtGui.QIntValidator)
    assert edit.validateText('12') and not edit.validateText('x')
    edit.setValidator(qtgui.floatValidator)
    assert isinstance(edit.validator(), QtGui.QDoubleValidator)
    edit.setValidator(qtgui.filenameValidator)
    assert isinstance(edit.validator(), QtGui.QRegExpValidator)
    assert not edit.validateText('a/b')
    edit.setValidator(None)
    assert edit.validator() is None
    assert edit.validateText('anything at all')


def test_filename_validator_blocks_typing_forbidden_characters():
    edit = qtgui.TextEdit('', qtgui.filenameValidator)
    QTest.keyClicks(edit, 'a/b:c')
    assert edit.text == 'abc'


def test_text_edit_typing_and_return_events():
    edit = qtgui.TextEdit()
    calls = record(edit, 'onChange', 'onActivate')
    QTest.keyClicks(edit, 'ab')
    QTest.keyClick(edit, QtCore.Qt.Key_Return)
    assert calls == [('onChange', 'a'), ('onChange', 'ab'), ('onActivate', 'ab')]
    assert edit.text == 'ab' and edit.getText() == 'ab'
    assert qtgui.TextEdit.onChange(edit, None) is None


def test_text_edit_set_text_moves_cursor_to_end_without_change_event():
    edit = qtgui.TextEdit()
    calls = record(edit, 'onChange')
    edit.setText('hello')
    assert edit.cursorPosition() == 5
    assert calls == []


def test_text_edit_arrow_keys_and_modifiers():
    edit = qtgui.TextEdit('abc')
    calls = record(edit, 'onUpArrow', 'onDownArrow', 'onModifier')
    QTest.keyClick(edit, QtCore.Qt.Key_Up)
    QTest.keyClick(edit, QtCore.Qt.Key_Down)
    QTest.keyClick(edit, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    assert calls[:2] == [('onUpArrow', None), ('onDownArrow', None)]
    # QTest presses Ctrl itself first (already with the modifier), then A
    assert calls[-1] == ('onModifier', (int(QtCore.Qt.ControlModifier), QtCore.Qt.Key_A))
    assert {name for name, _ in calls[2:]} == {'onModifier'}
    assert edit.text == 'abc'  # Ctrl+A selects, it does not type


def test_text_edit_claims_up_and_down_before_shortcuts():
    edit = qtgui.TextEdit()
    up = QtGui.QKeyEvent(QtCore.QEvent.ShortcutOverride, QtCore.Qt.Key_Up, QtCore.Qt.NoModifier)
    up.ignore()
    assert edit.eventFilter(edit, up) is True
    assert up.isAccepted()
    other = QtGui.QKeyEvent(QtCore.QEvent.ShortcutOverride, QtCore.Qt.Key_A, QtCore.Qt.NoModifier)
    assert edit.eventFilter(edit, other) is False
    assert edit.eventFilter(QtCore.QObject(), up) is False


def test_text_edit_tab_handling():
    class Completing(qtgui.TextEdit):
        def tabPressed(self):
            self.tabs = getattr(self, 'tabs', 0) + 1
            return True

    edit = Completing('ab')
    tab = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Tab, QtCore.Qt.NoModifier)
    assert edit.event(tab) is True
    tab2 = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Tab, QtCore.Qt.NoModifier)
    tab2.ignore()
    edit.keyPressEvent(tab2)
    assert tab2.isAccepted()
    assert edit.tabs == 2
    assert edit.text == 'ab'
    assert qtgui.TextEdit().tabPressed() is False


def test_document_edit_text_round_trip():
    doc = qtgui.DocumentEdit('<b>first</b>')
    assert not doc.acceptRichText()
    doc.setText('line 1')
    doc.addText('\nline 2')
    assert doc.text == 'line 1\nline 2'
    assert doc.getText() == doc.toPlainText()
    assert qtgui.DocumentEdit.NoWrap == QtWidgets.QTextEdit.NoWrap


def test_progress_bar_maps_fraction_to_range():
    bar = qtgui.ProgressBar(visible=False)
    assert bar.isHidden()
    bar.setRange(10, 20)
    bar.setProgress(0.5)
    assert bar.value() == 15
    bar.setProgress(1)
    assert bar.value() == 20
    assert not qtgui.ProgressBar().isHidden()


# --------------------------------------------------------------------------
# ShortcutEdit and MouseActionEdit
# --------------------------------------------------------------------------

def label_of(mod, key):
    return QtGui.QKeySequence(key + mod).toString(QtGui.QKeySequence.NativeText)


def test_shortcut_edit_shows_and_changes_the_shortcut():
    ctrl = int(QtCore.Qt.ControlModifier)
    edit = qtgui.ShortcutEdit((ctrl, QtCore.Qt.Key_S))
    assert edit.text() == label_of(ctrl, QtCore.Qt.Key_S)
    assert qtgui.ShortcutEdit(None).text() == ''
    edit.setShortcut((0, QtCore.Qt.Key_F5))
    assert edit.text() == label_of(0, QtCore.Qt.Key_F5)


def test_shortcut_edit_records_key_presses_without_shift():
    edit = qtgui.ShortcutEdit(None)
    calls = record(edit, 'onChanged')
    mods = QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier
    edit.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Shift, mods))
    assert calls == []  # a bare modifier is not a shortcut
    press = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_K, mods)
    edit.keyPressEvent(press)
    ctrl = int(QtCore.Qt.ControlModifier)
    assert calls == [('onChanged', (ctrl, QtCore.Qt.Key_K))]
    assert edit.text() == label_of(ctrl, QtCore.Qt.Key_K)
    assert press.isAccepted()
    assert qtgui.ShortcutEdit.onChanged(edit, None) is None


def test_shortcut_edit_strips_the_keypad_modifier_bit():
    edit = qtgui.ShortcutEdit(None)
    assert edit.shortcutToLabel(0x20000000, QtCore.Qt.Key_5) == label_of(0, QtCore.Qt.Key_5)


def test_shortcut_edit_captures_shortcut_overrides_and_highlights_on_focus():
    edit = qtgui.ShortcutEdit(None)
    override = QtGui.QKeyEvent(QtCore.QEvent.ShortcutOverride, QtCore.Qt.Key_Q, QtCore.Qt.ControlModifier)
    override.ignore()
    assert edit.eventFilter(edit, override) is True and override.isAccepted()
    assert edit.eventFilter(edit, QtCore.QEvent(QtCore.QEvent.Enter)) is False
    send(edit, QtGui.QFocusEvent(QtCore.QEvent.FocusIn))
    assert edit.backgroundRole() == QtGui.QPalette.Highlight
    assert edit.foregroundRole() == QtGui.QPalette.HighlightedText
    send(edit, QtGui.QFocusEvent(QtCore.QEvent.FocusOut))
    assert edit.backgroundRole() == QtGui.QPalette.Window
    assert edit.foregroundRole() == QtGui.QPalette.WindowText


@pytest.mark.parametrize("modifiers, button, text", [
    (0, QtCore.Qt.LeftButton, 'Left'),
    (QtCore.Qt.ControlModifier, QtCore.Qt.MidButton, 'Ctrl+Middle'),
    (QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier, QtCore.Qt.RightButton, 'Alt+Shift+Right'),
    (QtCore.Qt.MetaModifier, QtCore.Qt.NoButton, 'Meta+[Unknown]'),
])
def test_mouse_action_labels(modifiers, button, text):
    edit = qtgui.MouseActionEdit((int(modifiers), button))
    assert edit.text() == text


def test_mouse_action_edit_records_clicks():
    edit = qtgui.MouseActionEdit((0, QtCore.Qt.LeftButton))
    calls = record(edit, 'onChanged')
    edit.mousePressEvent(QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress, QtCore.QPointF(1, 1), QtCore.Qt.RightButton,
        QtCore.Qt.RightButton, QtCore.Qt.ControlModifier))
    ctrl = int(QtCore.Qt.ControlModifier)
    assert calls == [('onChanged', (ctrl, QtCore.Qt.RightButton))]
    assert edit.text() == 'Ctrl+Right'
    edit.setShortcut((0, QtCore.Qt.MidButton))
    assert edit.text() == 'Middle'
    assert qtgui.MouseActionEdit.onChanged(edit, None) is None


# --------------------------------------------------------------------------
# StackedBox, SpinBox, TreeView, TableView, Actions container
# --------------------------------------------------------------------------

def test_stacked_box_wraps_switches_and_removes_widgets():
    box = qtgui.StackedBox()
    one = box.addWidget(QtWidgets.QLabel('one'))
    two = box.addWidget(QtWidgets.QLabel('two'))
    assert box.count() == 2
    box.showWidget(two)
    assert box.currentWidget() is two.parentWidget()
    box.removeWidget(one)
    assert box.count() == 1
    assert box.widget(0) is two.parentWidget()


def test_stacked_box_auto_resize_ignores_hidden_pages():
    box = qtgui.StackedBox()
    one = box.addWidget(QtWidgets.QLabel('one'))
    two = box.addWidget(QtWidgets.QLabel('two'))
    page = lambda w: w.parentWidget().sizePolicy().verticalPolicy()
    assert page(one) == page(two) == QtWidgets.QSizePolicy.Preferred
    box.setAutoResize(True)
    assert page(one) == QtWidgets.QSizePolicy.Preferred  # current page
    assert page(two) == QtWidgets.QSizePolicy.Ignored
    box.showWidget(two)
    assert page(one) == QtWidgets.QSizePolicy.Ignored
    assert page(two) == QtWidgets.QSizePolicy.Preferred
    box.setAutoResize(False)
    assert page(one) == page(two) == QtWidgets.QSizePolicy.Preferred


def test_spin_box_reports_user_changes_only():
    spin = qtgui.SpinBox(5)
    calls = record(spin, 'onChange')
    assert spin.value() == 5 and spin.maximum() == 99999
    spin.setValue(9)  # programmatic: silent
    assert spin.value() == 9 and calls == []
    spin.stepUp()  # like the user clicking the arrow
    assert calls == [('onChange', 10)]


def test_tree_view_items_and_events():
    tree = qtgui.TreeView()
    calls = record(tree, 'onActivate', 'onExpand')
    root = tree.addTopLevel('root')
    assert root.isDir and root.parent is None and root.text == 'root'
    assert root.childIndicatorPolicy() == QtWidgets.QTreeWidgetItem.ShowIndicator
    sub = root.addChild('sub', isDir=True)
    files = root.addChildren(['a.txt', 'b.txt'])
    assert sub.parent is root and [f.text for f in files] == ['a.txt', 'b.txt']
    assert not files[0].isDir
    assert files[0].childIndicatorPolicy() == QtWidgets.QTreeWidgetItem.DontShowIndicator
    assert root.childCount() == 3
    leaf = tree.addTopLevel('leaf.txt', isDir=False)

    tree.itemActivated.emit(root, 0)  # directories are expanded, not activated
    tree.itemActivated.emit(leaf, 0)
    tree.itemExpanded.emit(leaf)      # files cannot be expanded
    tree.itemExpanded.emit(root)
    assert calls == [('onActivate', leaf), ('onExpand', root)]


def test_table_view_items_keep_user_data():
    table = qtgui.TableView()
    table.setRowCount(1)
    table.setColumnCount(2)
    table.setItem(0, 0, 'name', data={'id': 7})
    table.setItem(0, 1, 'plain')
    assert table.item(0, 0).text == 'name'
    assert table.getItemData(0, 0) == {'id': 7}
    assert table.getItemData(0, 1) is None


def test_actions_container_keeps_declaration_order():
    actions = qtgui.Actions()
    actions.save = 's'
    actions.load = 'l'
    actions._private = 'p'
    assert list(actions) == ['s', 'l']
    assert actions._private == 'p'


def test_size_policy_aliases():
    assert qtgui.SizePolicy.Ignored == QtWidgets.QSizePolicy.Ignored
    assert qtgui.SizePolicy.MinimumExpanding == QtWidgets.QSizePolicy.MinimumExpanding
