"""
Unit tests for lib/filechooser.py, run headless on the Qt offscreen platform:
thumbnail cache, flow layout, file sorting, tag filter, file loaders and the
grid / list / icon-list file choosers working on real files in tmp_path.

Event handlers attached with mhEvent/attachEvent are called through
events3d.callEvent, which swallows exceptions and reports them with
G.app.progress(1); the FakeApp records that and the ``app`` fixture fails the
test, so a handler that raised cannot pass silently.
"""

import os

import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from core import G
import events3d
import filechooser as fc
import language

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MH_DIR = os.path.join(REPO, "makehuman")

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
    """The parts of the MakeHuman application object that filechooser touches."""

    def __init__(self):
        self.mainwin = None
        self.settings = {'tagFilterMode': 'OR'}
        self.redraws = 0
        self.failures = []
        self.async_calls = []

    def redraw(self):
        self.redraws += 1

    def progress(self, value, *args):
        self.failures.append(value)  # an event handler raised

    def getSetting(self, name):
        return self.settings[name]

    def callAsync(self, func, *args, **kwargs):
        self.async_calls.append(func)


@pytest.fixture(autouse=True)
def app(qapp, monkeypatch):
    fake = FakeApp()
    monkeypatch.setattr(G, "app", fake)
    monkeypatch.setattr(language.language, "languageStrings", None)
    # createTagFilter appends to this class-level list; keep tests independent
    monkeypatch.setattr(fc.FileChooserBase, "switchFuncList", [])
    yield fake
    assert fake.failures == [], "an event handler raised inside callEvent (see the log warning)"


def record(widget, *names):
    calls = []
    for name in names:
        widget.attachEvent(name, lambda event, name=name: calls.append((name, event)))
    return calls


def png(path, width=16, height=16, color='#ff0000'):
    img = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
    img.fill(QtGui.QColor(color))
    assert img.save(str(path), 'PNG')
    return str(path)


def touch(path, text='x'):
    with open(str(path), 'w') as f:
        f.write(text)
    return str(path)


# --------------------------------------------------------------------------
# ThumbnailCache and FileChooserRectangle
# --------------------------------------------------------------------------

def test_thumbnail_cache_scales_and_caches(tmp_path):
    cache = fc.ThumbnailCache((128, 128))
    big = png(tmp_path / 'big.png', 256, 256)
    small = png(tmp_path / 'small.png', 32, 32)
    pix = cache[big]
    assert pix.size() == QtCore.QSize(128, 128)
    assert cache[small].size() == QtCore.QSize(128, 128)
    assert cache[big] is pix  # unchanged file: served from the cache

    png(tmp_path / 'big.png', 64, 64, '#00ff00')  # file replaced
    again = cache[big]
    assert again is not pix
    assert again.toImage().pixelColor(5, 5).name() == '#00ff00'


def test_thumbnail_cache_crops_non_square_images(tmp_path):
    cache = fc.ThumbnailCache((128, 128))
    assert cache[png(tmp_path / 'wide.png', 256, 128)].size() == QtCore.QSize(128, 128)


class Owner(events3d.EventHandler):
    def __init__(self):
        super(Owner, self).__init__()
        self.selection = None


def test_file_chooser_rectangle_selects_its_file(tmp_path):
    owner = Owner()
    calls = record(owner, 'onFileSelected')
    rect = fc.FileChooserRectangle(owner, '/models/a.mhm', 'A model', png(tmp_path / 'a.png', 64, 64))
    assert rect.label.text() == 'A model'
    assert rect.preview.pixmap().size() == QtCore.QSize(128, 128)
    rect.click()
    assert owner.selection == '/models/a.mhm'
    assert calls == [('onFileSelected', '/models/a.mhm')]


# --------------------------------------------------------------------------
# FlowLayout
# --------------------------------------------------------------------------

def fixed(width, height):
    w = QtWidgets.QWidget()
    w.setFixedSize(width, height)
    return w


def test_flow_layout_wraps_rows():
    host = QtWidgets.QWidget()
    layout = fc.FlowLayout(host)
    assert layout.minimumSize() == QtCore.QSize(0, 0)
    a, b, c = fixed(50, 20), fixed(50, 30), fixed(50, 10)
    for w in (a, b, c):
        layout.addWidget(w)
    assert layout.count() == 3
    assert layout.itemAt(-1) is None and layout.itemAt(3) is None
    assert layout.hasHeightForWidth()
    assert layout.expandingDirections() == QtCore.Qt.Vertical
    assert layout.sizeHint() == QtCore.QSize(150, 30)
    assert layout.minimumSize() == QtCore.QSize(50, 20)
    assert layout.heightForWidth(200) == 30
    assert layout.heightForWidth(120) == 30 + 10
    assert layout.heightForWidth(50) == 20 + 30 + 10

    layout.setGeometry(QtCore.QRect(0, 0, 120, 100))
    assert [w.pos() for w in (a, b, c)] == [QtCore.QPoint(0, 0), QtCore.QPoint(50, 0), QtCore.QPoint(0, 30)]

    item = layout.takeAt(1)
    assert item.widget() is b
    assert layout.count() == 2
    assert layout.takeAt(5) is None


# --------------------------------------------------------------------------
# FileSort and FileSortRadioButton
# --------------------------------------------------------------------------

@pytest.fixture
def three_files(tmp_path):
    paths = []
    for name, size in (('Beta.mhm', 30), ('alpha.mhm', 10), ('gamma.mhm', 20)):
        paths.append(touch(tmp_path / name, 'x' * size))
    return paths


def names(paths):
    return [os.path.basename(p) for p in paths]


def test_file_sort_fields_and_orders(three_files):
    sort = fc.FileSort()
    assert sort.fields() == ['name', 'created', 'modified', 'size']
    assert sort.metaFields == []
    assert names(sort.sort('name', three_files)) == ['alpha.mhm', 'Beta.mhm', 'gamma.mhm']
    assert names(sort.sort('size', three_files)) == ['alpha.mhm', 'gamma.mhm', 'Beta.mhm']
    assert sort.getMethod('size')(three_files[0]) == 30
    with pytest.raises(KeyError):
        sort.getMethod('faces')


def test_file_sort_insert_keeps_order(three_files, tmp_path):
    sort = fc.FileSort()
    ordered = sort.sort('size', three_files)
    medium = touch(tmp_path / 'delta.mhm', 'x' * 25)
    result = sort.insert('size', medium, ordered)
    assert names(result) == ['alpha.mhm', 'gamma.mhm', 'delta.mhm', 'Beta.mhm']


def test_file_sort_reads_metadata_only_for_changed_files(three_files):
    class Counting(fc.FileSort):
        def __init__(self):
            super(Counting, self).__init__()
            self.reads = []

        def getMeta(self, filename):
            self.reads.append(os.path.basename(filename))
            return {'faces': len(filename)}

    sort = Counting()
    sort.sort('name', three_files)
    assert sorted(sort.reads) == ['Beta.mhm', 'alpha.mhm', 'gamma.mhm']
    assert sort._meta[three_files[0]]['modified'] == os.path.getmtime(three_files[0])
    sort.sort('name', three_files)
    assert len(sort.reads) == 3  # nothing changed: cached
    later = os.path.getmtime(three_files[1]) + 10
    os.utime(three_files[1], (later, later))
    sort.sort('name', three_files)
    assert sort.reads[3:] == ['alpha.mhm']


def test_file_sort_skips_files_it_cannot_read(tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(fc.log, "warning", lambda msg, *a: warnings.append(msg % a if a else msg))
    sort = fc.FileSort()
    missing = str(tmp_path / 'gone.mhm')
    sort.updateMeta([missing])  # must not raise
    assert 'modified' not in sort._meta.get(missing, {})
    assert len(warnings) == 1 and 'gone.mhm' in warnings[0]
    assert sort.getMeta(missing) == {}


def test_file_sort_with_meta_fields(three_files):
    class FacesSort(fc.FileSort):
        def __init__(self):
            super(FacesSort, self).__init__()
            self.metaFields = ['faces']

        def getMeta(self, filename):
            return {'faces': {'alpha.mhm': 3, 'Beta.mhm': 1, 'gamma.mhm': 2}[os.path.basename(filename)]}

    sort = FacesSort()
    assert sort.fields() == ['name', 'created', 'modified', 'size', 'faces']
    assert names(sort.sort('faces', three_files)) == ['Beta.mhm', 'gamma.mhm', 'alpha.mhm']


def test_sort_radio_button_resorts_the_chooser():
    class Chooser(object):
        sortBy = 'name'
        refreshed = 0

        def refresh(self):
            self.refreshed += 1

    chooser = Chooser()
    group = []
    by_name = fc.FileSortRadioButton(chooser, group, True, 'name')
    by_size = fc.FileSortRadioButton(chooser, group, False, 'size')
    assert by_size.text() == 'By size' and by_name.selected
    by_size.click()
    assert chooser.sortBy == 'size' and chooser.refreshed == 1
    assert by_size.selected and not by_name.selected


# --------------------------------------------------------------------------
# TagFilter
# --------------------------------------------------------------------------

class Item(object):
    def __init__(self, *tags):
        self.tags = set(tags)
        self.hidden = None

    def setHidden(self, hidden):
        self.hidden = hidden


@pytest.mark.parametrize("mode, label", [('OR', 'OR'), ('AND', 'AND'), ('NOR', 'NOT OR'), ('NAND', 'NOT AND')])
def test_tag_filter_title_names_the_mode(app, mode, label):
    app.settings['tagFilterMode'] = mode
    assert fc.TagFilter().title() == 'Tag Filter [Mode : %s]' % label


def test_tag_filter_title_follows_setting_changes(app):
    tf = fc.TagFilter()
    app.settings['tagFilterMode'] = 'AND'
    tf.labelSwitch()
    assert tf.title() == 'Tag Filter [Mode : AND]'
    app.settings['tagFilterMode'] = 'NAND'
    tf.show()  # onShow refreshes the title too
    assert tf.title() == 'Tag Filter [Mode : NOT AND]'
    tf.hide()


def test_tag_filter_tags_and_toggles():
    tf = fc.TagFilter()
    calls = record(tf, 'onTagsChanged')
    tf.setTags(['Casual', 'casual', 'Formal'])
    assert tf.getTags() == {'casual', 'formal'}
    assert tf.tags['formal'].text() == 'Formal'
    assert not tf.filterActive()

    tf.tags['casual'].click()
    assert tf.getSelectedTags() == {'casual'}
    assert tf.filterActive()
    assert calls == [('onTagsChanged', {'casual'})]
    tf.tags['casual'].click()
    assert tf.getSelectedTags() == set()

    tf.setTagState('unknown', True)  # not a tag of this filter
    assert tf.getSelectedTags() == set()
    assert len(calls) == 2

    tf.addTags(['Sport'])
    assert tf.getTags() == {'casual', 'formal', 'sport'}
    tf.clearAll()
    assert tf.getTags() == set() and tf.getSelectedTags() == set()


def test_show_tags_puts_sticky_tags_first_and_restores_selection():
    tf = fc.TagFilter()
    tf.setTags(['zeta', 'alpha', 'female', 'male'])
    tf.showTags(selection={'alpha', 'female'}, stickyTags=['Female', 'Male', 42])
    assert [w.tag for w in tf.children] == ['female', 'male', 'alpha', 'zeta']
    assert tf.getSelectedTags() == {'alpha', 'female'}
    assert tf.tags['female'].isChecked() and tf.tags['alpha'].isChecked()
    assert not tf.tags['zeta'].isChecked()

    tf.removeTags()
    assert tf.count() == 0


def test_show_tags_with_a_single_sticky_tag_and_no_selection():
    tf = fc.TagFilter()
    tf.setTags(['b', 'a', 'Top'])
    tf.showTags(stickyTags='top')
    assert [w.tag for w in tf.children] == ['top', 'a', 'b']
    assert tf.getSelectedTags() == set()


def test_rebuilding_tags_does_not_leak_widgets():
    tf = fc.TagFilter()
    tf.setTags(['a', 'b'])
    tf.showTags()
    tf.setTags(['c'])
    tf.showTags()
    assert tf.count() == 1


@pytest.mark.parametrize("mode, hidden", [
    ('OR', [False, False, True]),
    ('AND', [False, True, True]),
    ('NOR', [True, True, False]),
    ('NAND', [True, False, False]),
])
def test_tag_filter_modes(app, mode, hidden):
    app.settings['tagFilterMode'] = mode
    tf = fc.TagFilter()
    tf.setTags(['red', 'big'])
    tf.setTagState('red', True)
    tf.setTagState('big', True)
    items = [Item('red', 'big'), Item('red'), Item('blue')]
    tf.filter(items)
    assert [i.hidden for i in items] == hidden


def test_inactive_tag_filter_shows_everything():
    tf = fc.TagFilter()
    items = [Item('red'), Item()]
    tf.filter(items)
    assert [i.hidden for i in items] == [False, False]


# --------------------------------------------------------------------------
# FileHandler, TaggedFileLoader, MhmatFileLoader
# --------------------------------------------------------------------------

class FakeChooser(object):
    def __init__(self, extensions=('mhm',), previewExtensions=None, notFoundImage=None):
        self.extensions = list(extensions)
        self.previewExtensions = previewExtensions
        self.notFoundImage = notFoundImage
        self.added = []
        self.selectedTags = set()
        self.shown = None

    def addItem(self, file, label, preview, tags=None):
        self.added.append((os.path.basename(file), label, preview, tags))

    def getSelectedTags(self):
        return self.selectedTags

    def removeTags(self):
        self.selectedTags = set()

    def showTags(self, selection):
        self.shown = selection


def test_file_handler_labels(tmp_path):
    handler = fc.FileHandler()
    chooser = FakeChooser(notFoundImage='nf.thumb')
    handler.setFileChooser(chooser)
    handler.refresh([str(tmp_path / 'my_COOL_model.mhm'), str(tmp_path / 'x.mhm')])
    assert [(n, l) for n, l, _, _ in chooser.added] == [('my_COOL_model.mhm', 'My cool model'), ('x.mhm', 'X')]

    chooser.extensions = []
    chooser.added = []
    handler.refresh([str(tmp_path / 'data_file.txt')])
    assert chooser.added[0][1] == 'Data_file.txt'


def test_file_handler_preview_lookup(tmp_path):
    handler = fc.FileHandler()
    chooser = FakeChooser(previewExtensions=['thumb', 'png'], notFoundImage='nf.thumb')
    handler.setFileChooser(chooser)
    model = str(tmp_path / 'a.mhm')
    assert handler.getPreview(None) == 'nf.thumb'
    assert handler.getPreview(model) == 'nf.thumb'  # no preview at all
    png_preview = png(tmp_path / 'a.png')
    assert handler.getPreview(model) == png_preview
    thumb = png(tmp_path / 'a.thumb')
    assert handler.getPreview(model) == thumb  # first extension wins

    chooser.previewExtensions = None  # the file is its own preview
    assert handler.getPreview(thumb) == thumb
    chooser.notFoundImage = None
    assert handler.getPreview(model) == model


def test_file_handler_matching(tmp_path):
    handler = fc.FileHandler()
    item = Item()
    item.file = str(tmp_path / 'sub' / '..' / 'a.mhm')
    assert handler.getSelection(item) == item.file
    assert handler.matchesItem(item, str(tmp_path / 'a.mhm'))
    assert not handler.matchesItem(item, str(tmp_path / 'b.mhm'))
    assert handler.matchesItems(item, [str(tmp_path / 'b.mhm'), str(tmp_path / 'a.mhm')])
    assert not handler.matchesItems(item, [])
    assert fc.abspath(None) is None
    assert fc.abspath(str(tmp_path / 'x' / '..' / 'y')) == os.path.realpath(str(tmp_path / 'y'))


class FakeLibrary(object):
    def __init__(self, tags, names=None):
        self.tags = tags
        self.names = names or {}

    def getTags(self, filename):
        return self.tags[os.path.basename(filename)]

    def getName(self, filename):
        return self.names.get(os.path.basename(filename), '')


def test_tagged_loader_labels_tags_and_selection(tmp_path):
    library = FakeLibrary({'my_shirt.mhclo': {'top'}, 'jeans.mhclo': {'bottom'}},
                          names={'my_shirt.mhclo': 'Blue shirt'})
    loader = fc.TaggedFileLoader(library)
    chooser = FakeChooser(extensions=['mhclo'], notFoundImage='nf.thumb')
    chooser.selectedTags = {'top'}
    loader.setFileChooser(chooser)
    files = [str(tmp_path / 'my_shirt.mhclo'), str(tmp_path / 'jeans.mhclo')]
    loader.refresh(files)
    assert [(n, l, t) for n, l, _, t in chooser.added] == [
        ('my_shirt.mhclo', 'My shirt', {'top'}), ('jeans.mhclo', 'Jeans', {'bottom'})]
    assert chooser.shown == {'top'}  # selection from before the refresh

    loader.setNameTagsUsage(True)
    chooser.added = []
    loader.refresh(files)
    assert [l for _, l, _, _ in chooser.added] == ['Blue shirt', 'Jeans']  # no name: file name

    chooser.extensions = []
    chooser.added = []
    loader.setNameTagsUsage(False)
    loader.refresh(files[1:])
    assert chooser.added[0][1] == 'Jeans.mhclo'


def test_mhmat_loader_falls_back_to_the_diffuse_texture(tmp_path, monkeypatch):
    import material
    opened = []

    class Mat(object):
        def __init__(self, texture):
            self.diffuseTexture = texture

    textures = {'skin.mhmat': str(tmp_path / 'skin.png'), 'plain.mhmat': None}

    def from_file(path):
        opened.append(os.path.basename(path))
        return Mat(textures[os.path.basename(path)])

    monkeypatch.setattr(material, "fromFile", from_file)
    notfound = png(tmp_path / 'notfound.thumb')
    loader = fc.MhmatFileLoader()
    loader.setFileChooser(FakeChooser(extensions=['mhmat'], previewExtensions=['thumb'],
                                      notFoundImage=notfound))
    assert loader.getPreview(str(tmp_path / 'skin.mhmat')) == textures['skin.mhmat']
    assert loader.getPreview(str(tmp_path / 'plain.mhmat')) == notfound
    thumb = png(tmp_path / 'skin.thumb')
    assert loader.getPreview(str(tmp_path / 'skin.mhmat')) == thumb
    assert opened == ['skin.mhmat', 'plain.mhmat']  # not opened when a thumbnail exists


# --------------------------------------------------------------------------
# FileChooserBase behaviour (through ListFileChooser)
# --------------------------------------------------------------------------

@pytest.fixture
def library(tmp_path):
    root = tmp_path / 'lib'
    (root / 'sub').mkdir(parents=True)
    files = {
        'b_model': touch(root / 'b_model.mhm', 'x' * 20),
        'a_model': touch(root / 'a_model.mhm', 'x' * 30),
        'deep': touch(root / 'sub' / 'deep.mhm', 'x' * 10),
    }
    touch(root / 'ignored.txt')
    return root, files


def test_chooser_paths_extensions_and_search(library):
    root, files = library
    chooser = fc.ListFileChooser(str(root), 'mhm')
    assert chooser.getPaths() == [str(root)]
    assert chooser.extensions == ['mhm']
    assert sorted(chooser.search()) == sorted(files.values())
    flat = fc.ListFileChooser([str(root)], ['mhm'], doNotRecurse=True)
    assert sorted(flat.search()) == sorted([files['a_model'], files['b_model']])

    chooser.setPreviewExtensions(None)
    assert chooser.previewExtensions is None
    chooser.setPreviewExtensions('thumb')
    assert chooser.previewExtensions == ['thumb']
    chooser.setPreviewExtensions(['thumb', 'png'])
    assert chooser.previewExtensions == ['thumb', 'png']


def test_chooser_resize_schedules_a_scrollbar_update(app):
    chooser = fc.ListFileChooser('.', 'mhm')
    assert chooser.eventFilter(chooser, QtGui.QResizeEvent(QtCore.QSize(1, 1), QtCore.QSize(0, 0))) is False
    assert app.async_calls == [chooser._updateScrollBar]
    assert chooser.eventFilter(chooser, QtCore.QEvent(QtCore.QEvent.Show)) is False
    assert len(app.async_calls) == 1
    assert chooser._updateScrollBar() is None


def test_list_chooser_refresh_lists_sorted_files(library, app):
    root, files = library
    chooser = fc.ListFileChooser(str(root), 'mhm', name='Models')
    assert chooser.mainBox.title() == 'Models'
    calls = record(chooser, 'onRefresh')
    chooser.refresh()
    items = chooser.children.getItems()
    assert [i.text for i in items] == ['A model', 'B model', 'Deep']
    assert [i.file for i in items] == [files['a_model'], files['b_model'], files['deep']]
    assert items[0].toolTip() == 'A model<p>' + files['a_model']
    assert calls == [('onRefresh', chooser)]
    assert app.redraws >= 1

    chooser.sortBy = 'size'
    chooser.refresh()
    assert [i.text for i in chooser.children.getItems()] == ['Deep', 'B model', 'A model']
    assert chooser.contains(files['deep']) and not chooser.contains('/nope.mhm')


def test_sort_box_buttons_refresh_and_resort(library):
    root, files = library
    chooser = fc.ListFileChooser(str(root), 'mhm')
    box = chooser.createSortBox()
    radios = [w for w in box.children if isinstance(w, fc.FileSortRadioButton)]
    assert [r.field for r in radios] == ['name', 'created', 'modified', 'size']
    assert radios[0].selected
    assert chooser.children.count() == 0
    chooser.refreshButton.click()
    assert chooser.children.count() == 3
    radios[3].click()
    assert chooser.sortBy == 'size'
    assert chooser.children.item(0).file == files['deep']


def test_auto_refresh_on_show(library):
    root, _ = library
    chooser = fc.ListFileChooser(str(root), 'mhm')
    chooser.show()
    assert chooser.children.count() == 3
    chooser.hide()
    chooser.children.clear()
    chooser.enableAutoRefresh(False)
    chooser.show()
    assert chooser.children.count() == 0
    chooser.hide()


def test_tag_helpers_without_a_tag_filter(library):
    root, files = library
    chooser = fc.ListFileChooser(str(root), 'mhm')
    chooser.refresh()
    assert chooser.getSelectedTags() is None
    chooser.showTags({'x'})
    chooser.removeTags()
    chooser.applyTagFilter()
    assert fc.FileChooserBase.addItem(chooser, 'f', 'l', 'p', ['t']) is None

    chooser.setTags(files['a_model'], ['Red', 'Big'])
    assert chooser._getListItem(files['a_model']).tags == ['red', 'big']
    chooser.setTags(files['b_model'], ['blue'])
    chooser.addTags('/not/listed.mhm', ['x'])  # unknown files are ignored
    chooser.setTags('/not/listed.mhm', ['x'])
    assert chooser.getAllTags() == {'red', 'big', 'blue'}


def test_add_tags_extends_a_tag_set(library):
    root, files = library
    chooser = fc.ListFileChooser(str(root), 'mhm')
    chooser.addItem(files['a_model'], 'A', None, tags={'blue'})
    chooser.addTags(files['a_model'], ['Small'])
    assert chooser._getListItem(files['a_model']).tags == {'blue', 'small'}


def test_add_tags_to_a_plainly_loaded_item(library):
    root, files = library
    chooser = fc.ListFileChooser(str(root), 'mhm')
    chooser.refresh()
    chooser.addTags(files['a_model'], ['Small'])
    assert chooser._getListItem(files['a_model']).tags == {'small'}


def test_tagged_list_chooser_filters_and_keeps_tag_selection(library, app):
    root, files = library
    chooser = fc.IconListFileChooser(str(root), 'mhm', 'thumb', name='Models')
    tag_filter = chooser.createTagFilter()
    assert chooser.switchFuncList == [chooser.onTagFilterModeSwitched]
    lib = FakeLibrary({'a_model.mhm': {'casual'}, 'b_model.mhm': {'formal'}, 'deep.mhm': {'casual', 'sport'}})
    chooser.setFileLoadHandler(fc.TaggedFileLoader(lib))
    chooser.refresh()
    assert tag_filter.getTags() == {'casual', 'formal', 'sport'}

    def visible():
        return [os.path.basename(i.file) for i in chooser.children.getItems() if not i.isHidden()]

    tag_filter.tags['casual'].click()
    assert visible() == ['a_model.mhm', 'deep.mhm']
    assert chooser.getSelectedTags() == {'casual'}

    chooser.refresh()  # the selected tag survives the rebuild of the tag list
    assert chooser.getSelectedTags() == {'casual'}
    assert chooser.tagFilter.tags['casual'].isChecked()
    assert visible() == ['a_model.mhm', 'deep.mhm']

    app.settings['tagFilterMode'] = 'NOR'
    chooser.onTagFilterModeSwitched()
    assert tag_filter.title() == 'Tag Filter [Mode : NOT OR]'
    assert visible() == ['b_model.mhm']

    chooser.removeItem('/not/listed.mhm')  # unknown files are ignored
    assert chooser.children.count() == 3


# --------------------------------------------------------------------------
# ListFileChooser selection
# --------------------------------------------------------------------------

@pytest.fixture
def single(library):
    root, files = library
    chooser = fc.ListFileChooser(str(root), 'mhm')
    chooser.refresh()
    return chooser, files


@pytest.fixture
def multi(library):
    root, files = library
    chooser = fc.ListFileChooser(str(root), 'mhm', multiSelect=True)
    chooser.refresh()
    return chooser, files


def test_single_selection(single):
    chooser, files = single
    assert chooser.getSelectedItem() is None and chooser.getSelectedItems() == []
    chooser.selectItem(files['b_model'])
    assert chooser.isSelected(files['b_model'])
    assert chooser.getHighlightedItem() == files['b_model']
    assert chooser.getSelectedItems() == [files['b_model']]
    chooser.selectItem(files['a_model'])
    assert chooser.getSelectedItems() == [files['a_model']]
    assert not chooser.isSelected('/not/listed.mhm')

    chooser.setSelections([files['deep']])  # multi-selection only
    assert chooser.getSelectedItems() == [files['a_model']]

    chooser.deselectItem(files['b_model'])  # not selected: nothing happens
    assert chooser.getSelectedItems() == [files['a_model']]
    chooser.deselectItem(files['a_model'])
    assert chooser.getSelectedItems() == []


def test_single_selection_click_reports_the_file(single):
    chooser, files = single
    calls = record(chooser, 'onFileSelected')
    item = chooser._getListItem(files['deep'])
    chooser.children.itemClicked.emit(item)
    assert calls == [('onFileSelected', files['deep'])]


def test_highlight(single):
    chooser, files = single
    chooser.setHighlightedItem(files['deep'])
    assert chooser.children.currentItem().file == files['deep']
    chooser.setHighlightedItem('/not/listed.mhm')
    assert chooser.children.currentItem().file == files['deep']
    chooser.setHighlightedItem(None)
    assert chooser.children.currentItem() is None


def test_refresh_keeps_or_drops_the_selection(single):
    chooser, files = single
    chooser.selectItem(files['b_model'])
    chooser.refresh()
    assert chooser.getSelectedItems() == [files['b_model']]
    chooser.refresh(keepSelections=False)
    assert chooser.getSelectedItems() == []


def test_multi_selection(multi):
    chooser, files = multi
    assert chooser.children.allowsMultipleSelection()
    chooser.selectItem(files['a_model'])
    chooser.selectItem(files['deep'])
    chooser.selectItem(files['deep'])  # already selected
    assert sorted(chooser.getSelectedItems()) == sorted([files['a_model'], files['deep']])
    chooser.setSelection(files['b_model'])  # single-selection only
    assert not chooser.isSelected(files['b_model'])

    chooser.deselectItem(files['a_model'])
    assert chooser.getSelectedItems() == [files['deep']]

    chooser.setHighlightedItem(files['deep'])
    chooser.refresh()
    assert chooser.getSelectedItems() == [files['deep']]
    assert chooser.getHighlightedItem() == files['deep']


def test_multi_selection_click_events_and_deselect_all(multi):
    chooser, files = multi
    calls = record(chooser, 'onFileSelected', 'onFileDeselected', 'onDeselectAll')
    item = chooser._getListItem(files['a_model'])
    item.setSelected(True)
    chooser.children.itemClicked.emit(item)
    item.setSelected(False)
    chooser.children.itemClicked.emit(item)
    assert calls == [('onFileSelected', files['a_model']), ('onFileDeselected', files['a_model'])]

    box = chooser.createSortBox()
    deselect = [w for w in box.children if isinstance(w, QtWidgets.QPushButton) and w.text() == 'Deselect all']
    assert len(deselect) == 1
    chooser.selectItem(files['deep'])
    deselect[0].click()
    assert chooser.getSelectedItems() == []
    assert calls[-1] == ('onDeselectAll', None)


def test_list_chooser_misc(library):
    root, _ = library
    chooser = fc.ListFileChooser(str(root), 'mhm')
    assert chooser.children.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff
    chooser.setVerticalScrollingEnabled(True)
    assert chooser.children.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded
    focused = []
    chooser.children.setFocus = lambda: focused.append(True)
    chooser.setFocus()
    assert focused == [True]
    assert chooser.resizeEvent(None) is None
    item = chooser.addItem(None, None, None)
    assert item.toolTip() == ''
    first = chooser.addItem('/x.mhm', 'X', None, pos=0)
    assert chooser.children.item(0) is first
    chooser.clearList()
    assert chooser.children.count() == 0


def test_list_chooser_with_none_item(library):
    root, _ = library
    chooser = fc.ListFileChooser(str(root), 'mhm', noneItem=True)
    chooser.refresh()
    assert chooser.children.item(0).text == 'None'


# --------------------------------------------------------------------------
# IconListFileChooser
# --------------------------------------------------------------------------

def test_icon_list_icons_are_scaled_and_cached(tmp_path):
    root = tmp_path / 'lib'
    root.mkdir()
    touch(root / 'big.mhm')
    big_thumb = png(root / 'big.thumb', 256, 128, '#0000ff')
    notfound = png(tmp_path / 'notfound.thumb', 32, 32, '#ff00ff')
    chooser = fc.IconListFileChooser(str(root), 'mhm', 'thumb', notfound, stickyTags=['x'])
    assert chooser.children.iconSize() == QtCore.QSize(50, 50)
    assert chooser.children.wordWrap()
    assert chooser.stickyTags == ['x']
    chooser.refresh()
    item = chooser.children.item(0)
    assert item.preview == big_thumb
    assert max(item.icon().availableSizes()[0].width(), item.icon().availableSizes()[0].height()) == 128
    icon, mtime = chooser._iconCache[big_thumb]
    chooser.refresh()
    assert chooser._iconCache[big_thumb][0] is icon  # cached

    later = mtime + 10
    os.utime(big_thumb, (later, later))
    chooser.refresh()
    assert chooser._iconCache[big_thumb][0] is not icon
    chooser.setIconSize(20, 30)
    assert chooser.children.iconSize() == QtCore.QSize(20, 30)


def test_icon_list_none_item_icons(tmp_path, monkeypatch):
    root = tmp_path / 'lib'
    root.mkdir()
    touch(root / 'a.mhm')
    notfound = png(tmp_path / 'notfound.thumb')
    chooser = fc.IconListFileChooser(str(root), 'mhm', 'thumb', notfound, noneItem=True)
    chooser.refresh()
    none = chooser.children.item(0)
    assert (none.text, none.file) == ('None', None)
    assert none.preview == str(tmp_path / 'clear.thumb')  # next to the "not found" icon
    assert [i.text for i in chooser.children.getItems()] == ['None', 'A']

    clear = png(tmp_path / 'myclear.png')
    with_clear = fc.IconListFileChooser(str(root), 'mhm', 'thumb', notfound, clearImage=clear, noneItem=True)
    with_clear.refresh()
    assert with_clear.children.item(0).preview == clear

    monkeypatch.chdir(MH_DIR)  # system data is found relative to the working directory
    bare = fc.IconListFileChooser(str(root), 'mhm', 'thumb', None, noneItem=True)
    bare.refresh()
    assert bare.children.item(0).preview == 'data/clear.thumb'

    multi = fc.IconListFileChooser(str(root), 'mhm', 'thumb', notfound, noneItem=True, multiSelect=True)
    multi.refresh()
    assert [i.text for i in multi.children.getItems()] == ['A']  # no "None" when several are allowed


# --------------------------------------------------------------------------
# FileChooser (grid of thumbnails)
# --------------------------------------------------------------------------

def test_grid_chooser_uses_notfound_thumb_from_its_folder(tmp_path):
    root = tmp_path / 'lib'
    root.mkdir()
    own = png(root / 'notfound.thumb', 128, 128)
    chooser = fc.FileChooser(str(root), 'mhm', 'thumb')
    assert chooser.notFoundImage == os.path.join(str(root), 'notfound.thumb') == own
    assert chooser.location.text() == fc.abspath(str(root))


def test_grid_chooser_falls_back_to_system_notfound(tmp_path, monkeypatch):
    monkeypatch.chdir(MH_DIR)
    chooser = fc.FileChooser([str(tmp_path / 'a'), str(tmp_path / 'b')], 'mhm',
                             notFoundImage=str(tmp_path / 'missing.thumb'))
    assert chooser.notFoundImage == 'data/notfound.thumb'
    assert chooser.location.text() == str(tmp_path / 'a') + '  |  ' + str(tmp_path / 'b')


def test_grid_chooser_refresh_and_click(tmp_path):
    root = tmp_path / 'lib'
    root.mkdir()
    a = touch(root / 'a_model.mhm')
    b = touch(root / 'b.mhm')
    png(root / 'a_model.thumb', 64, 64, '#00ff00')
    png(root / 'notfound.thumb', 128, 128, '#ff00ff')
    chooser = fc.FileChooser(str(root), 'mhm', 'thumb')
    chooser.refresh()
    rects = [chooser.children.itemAt(i).widget() for i in range(chooser.children.count())]
    assert [(r.file, r.label.text()) for r in rects] == [(a, 'A model'), (b, 'B')]
    assert rects[0].preview.pixmap().toImage().pixelColor(5, 5).name() == '#00ff00'
    assert rects[1].preview.pixmap().toImage().pixelColor(5, 5).name() == '#ff00ff'

    chooser.refresh()  # old thumbnails are removed first
    assert chooser.children.count() == 2

    calls = record(chooser, 'onFileSelected')
    chooser.children.itemAt(1).widget().click()
    assert chooser.selection == b
    assert calls == [('onFileSelected', b)]
    added = chooser.addItem(a, 'A again', str(root / 'notfound.thumb'), tags=['t'])
    assert added.tags == ['t'] and chooser.children.count() == 3


@pytest.mark.xfail(strict=True, raises=AttributeError, reason=(
    "filechooser.FileChooserBase.removeItem cannot remove a listed file from either "
    "chooser: for ListFileChooser _removeListItem calls ListView.removeItem and "
    "QListWidgetItem.widget(), which do not exist; for the FileChooser grid _getListItem "
    "calls FlowLayout.getItems(), which does not exist (no app code calls removeItem today)"))
@pytest.mark.parametrize("kind", ["list", "grid"])
def test_remove_item(tmp_path, kind):
    root = tmp_path / 'lib'
    root.mkdir()
    a = touch(root / 'a.mhm')
    png(root / 'notfound.thumb', 128, 128)
    if kind == "list":
        chooser = fc.ListFileChooser(str(root), 'mhm')
    else:
        chooser = fc.FileChooser(str(root), 'mhm', 'thumb')
    chooser.refresh()
    chooser.removeItem(a)
    assert chooser.children.count() == 0
