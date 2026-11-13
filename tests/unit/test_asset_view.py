"""
Unit tests for plugins/8_asset_downloader/assetdownload.py (the "Download
assets" task of the Community tab) and its table model, built headless: the
task panels and top widgets live in a container that is never shown, message
boxes are recorded instead of opened, downloads go to the local http.server.
Buttons are pressed with QAbstractButton.click() and table rows through the
real ``clicked`` signal. Harness and fixtures: test_asset_harness.py.
"""

import json
import os
import types

import pytest
from PyQt5 import QtCore, QtWidgets

from test_asset_harness import (  # noqa: F401  (qapp and env are pytest fixtures)
    ASSETS_JSON, FILES, MH_DIR, SCREEN_SEED, THUMB_SEED, asset_json, assetdownload,
    clothes_files, env, pixel_color, png_bytes, qapp, tablemodel)

RED, GREEN, BLUE = "#ff0000", "#00ff00", "#0000ff"
TYPES = ["pose", "clothes", "target", "hair", "teeth", "eyebrows", "eyelashes",
         "skin", "proxy", "material", "model", "rig", "expression"]


def clothes(nid, title, category="Shirt", **kw):
    return asset_json(nid, title, category=category, files=clothes_files(nid, "item%d" % nid), **kw)


def catalog():
    hat = clothes(103, "Nodate Hat", category="Hat", author="Zed")
    del hat["changed"]
    return [
        clothes(101, "Red Shirt", author="bob", description="A red cotton shirt", changed="2021-05-06 07:08:09"),
        clothes(102, "Blue Dress", category="Dress", author="alice", license="CC-BY", description="Evening dress"),
        hat,
        asset_json(201, "Fedora Felt", type="material",
                   belongs_to={"belonging_is_assigned": True, "belongs_to_core_asset": "clothes/fedora01"},
                   files={"mhmat": FILES + "/assets/201/felt.mhmat"}),
    ]


def cache_picture(env, nid, name, color):
    folder = os.path.join(env.root, str(nid))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name), "wb") as f:
        f.write(png_bytes(color))


def make_view(env, monkeypatch):
    monkeypatch.chdir(MH_DIR)  # notfound.thumb is found relative to the working folder, as in the app
    return assetdownload.AssetDownloadTaskView(types.SimpleNamespace(name="Community"))


@pytest.fixture
def view(env, monkeypatch):
    env.write_remote_db(catalog())
    cache_picture(env, 101, "thumb.png", RED)
    cache_picture(env, 101, "screenshot.png", GREEN)
    cache_picture(env, 102, "thumb.png", BLUE)
    return make_view(env, monkeypatch)


@pytest.fixture
def empty_view(env, monkeypatch):
    return make_view(env, monkeypatch)


def items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def rows(view):
    model = view.tableView.model()
    return [[model.data(model.index(r, c), QtCore.Qt.DisplayRole) for c in range(model.columnCount(QtCore.QModelIndex()))]
            for r in range(model.rowCount(QtCore.QModelIndex()))]


def list_assets(view, kind="clothes", **controls):
    view.cbxTypes.setCurrentText(kind)
    for name, value in controls.items():
        widget = getattr(view, name)
        if isinstance(widget, QtWidgets.QComboBox):
            widget.setCurrentText(value)
        else:
            widget.setText(value)
    view.btnFilter.click()


def click_asset(view, nid):
    model = view.tableView.model()
    for r in range(model.rowCount(QtCore.QModelIndex())):
        if model.data(model.index(r, 0), QtCore.Qt.DisplayRole) == str(nid):
            view.tableView.selectRow(r)
            view.tableView.clicked.emit(model.index(r, 0))
            return
    raise AssertionError("asset %s is not listed" % nid)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def test_controls_and_table_after_construction(env, view):
    assert items(view.cbxTypes) == TYPES
    assert view.cbxTypes.getCurrentItem() == "pose"
    assert items(view.cbxSubTypes) == ["-- any --"]
    assert items(view.cbxAuthors) == ["-- any --", "alice", "bob", "Zed"]  # case-insensitive order
    assert items(view.cbxLicense) == ["-- any --", "CC0", "CC-BY", "AGPL"]
    assert items(view.cbxDownloaded) == ["-- any --", "yes", "no"]
    assert items(view.cbxUpdated) == ["-- any --", "One week", "One month", "Three months", "One year"]
    assert view.txtTitle.getText() == "" and view.txtDesc.getText() == ""

    assert rows(view) == [["No filter"]]
    model = view.tableView.model()
    assert isinstance(model, tablemodel.AssetTableModel)
    assert model.headerData(0, QtCore.Qt.Horizontal) == "Info"
    assert model.headerData(0, QtCore.Qt.Vertical) == "1"
    assert model.headerData(0, QtCore.Qt.Horizontal, QtCore.Qt.ToolTipRole) is None
    assert model.data(model.index(0, 0), QtCore.Qt.ToolTipRole) is None

    left, right = env.app.mainwin.panels[0]
    assert left.added == [view.filterBox]
    assert right.added == [view.selectBox, view.syncBox]
    assert env.app.mainwin.top == [view.tableView, view.detailsPanel]
    assert view.detailsPanel.isHidden()
    assert view.tableView.selectionMode() == QtWidgets.QAbstractItemView.SingleSelection
    assert view.tableView.selectionBehavior() == QtWidgets.QAbstractItemView.SelectRows
    assert view.tableView.isSortingEnabled()

    assert view.hasFilter is False and view.isShowingDetails is False
    assert view.currentlySelectedRemoteAsset is None
    assert view.notfound == os.path.join(MH_DIR, "data", "notfound.thumb")
    assert pixel_color(view.thumbnail) is not None  # the "not found" picture
    assert pixel_color(view.detailsRender) is not None
    assert view.fetchScreens.selected is False


def test_showing_the_task_without_a_database_explains_what_to_do(env, empty_view):
    empty_view.show()  # the app shows the task; View calls onShow
    assert len(env.messages) == 1
    box = env.messages[0]
    assert "has not been downloaded yet" in box.text
    assert "click 'synchronize'" in box.text
    assert box.title == "Information"
    assert box.icon == QtWidgets.QMessageBox.Information
    assert box.buttons == QtWidgets.QMessageBox.Ok


def test_showing_the_task_with_a_database_says_nothing(env, view):
    view.show()
    assert env.messages == []


@pytest.mark.parametrize("kind, expected", [
    ("clothes", ["-- any --", "dress", "hat", "shirt"]),
    ("material", ["-- any --", "for core asset"]),
    ("hair", ["-- any --"]),
])
def test_asset_type_sets_the_subtypes(view, kind, expected):
    view.cbxTypes.setCurrentText("rig")
    view.cbxSubTypes.addItem("left over")
    view.cbxTypes.setCurrentText(kind)
    assert items(view.cbxSubTypes) == expected


# --------------------------------------------------------------------------
# Listing assets
# --------------------------------------------------------------------------

def test_update_list_shows_the_matching_assets(view):
    list_assets(view, cbxAuthors="bob")
    assert rows(view) == [["101", "bob", "CC0", "Red Shirt", "A red cotton shirt"]]
    model = view.tableView.model()
    assert model is view.proxymodel and model.sourceModel() is view.model
    assert [model.headerData(c, QtCore.Qt.Horizontal) for c in range(5)] == \
        ["node id", "author", "license", "title", "description"]
    assert view.hasFilter is True
    assert not view.tableView.isHidden() and view.detailsPanel.isHidden()
    assert view.btnDetails.text() == "View details"


def test_update_list_hands_every_control_to_the_database(view, monkeypatch):
    calls = []
    monkeypatch.setattr(view.assetdb, "getFilteredAssets", lambda kind, **kw: calls.append((kind, kw)) or [])
    list_assets(view)  # everything left at "any"
    list_assets(view, "clothes", cbxSubTypes="shirt", cbxAuthors="bob", cbxLicense="CC-BY",
                cbxDownloaded="yes", cbxUpdated="One month", txtTitle="red", txtDesc="cotton")
    list_assets(view, "material", cbxSubTypes="for core asset")
    list_assets(view, "pose", cbxAuthors="-- any --", cbxLicense="-- any --", cbxDownloaded="no",
                cbxUpdated="-- any --", txtTitle="", txtDesc="")
    nothing = dict(author=None, subtype=None, title=None, isDownloaded=None, desc=None, changed=None, license=None)
    assert calls[0] == ("clothes", nothing)
    assert calls[1] == ("clothes", dict(author="bob", subtype="shirt", title="red", isDownloaded="yes",
                                        desc="cotton", changed="One month", license="CC-BY"))
    assert calls[2][0] == "material" and calls[2][1]["subtype"] == "for core asset"
    assert calls[3] == ("pose", dict(nothing, isDownloaded="no"))
    assert rows(view) == []


def test_row_click_shows_the_asset(view):
    list_assets(view)
    click_asset(view, 101)
    asset = view.currentlySelectedRemoteAsset
    assert asset is view.assetdb.remoteAssets["clothes"][101]
    assert pixel_color(view.thumbnail) == RED
    assert view.detailsName.text() == "<h1>Red Shirt</h1>"
    assert view.detailsDesc.text() == "<p>A red cotton shirt</p>"
    extras = view.detailsExtras.text()
    assert "<b>Author........: </b>bob<br />" in extras
    assert "<b>License.......: </b>CC0<br />" in extras
    assert "<b>Last changed..: </b>2021-05-06 07:08:09<br />" in extras
    assert pixel_color(view.detailsRender) == GREEN  # cached screenshot
    assert view.btnDownloadScreenshot.isHidden()

    click_asset(view, 102)  # no screenshot in the cache
    assert view.currentlySelectedRemoteAsset.getId() == 102
    assert pixel_color(view.thumbnail) == BLUE
    assert pixel_color(view.detailsRender) not in (None, GREEN)
    assert not view.btnDownloadScreenshot.isHidden()


def test_row_click_on_asset_without_thumbnail_keeps_the_previous_picture(env, view):
    view.assetdb.remoteAssets["clothes"][102].localFiles.pop("thumb")
    list_assets(view)
    click_asset(view, 101)
    click_asset(view, 102)
    assert view.currentlySelectedRemoteAsset.getId() == 102
    assert pixel_color(view.thumbnail) == RED


def test_row_click_before_any_list_does_nothing(view):
    view.tableView.selectRow(0)
    view.tableView.clicked.emit(view.tableView.model().index(0, 0))
    assert view.currentlySelectedRemoteAsset is None


def test_click_without_a_selected_row_does_nothing(view):
    list_assets(view)
    view.tableView.clearSelection()
    view.tableView.clicked.emit(view.tableView.model().index(0, 0))
    assert view.currentlySelectedRemoteAsset is None


def test_new_list_clears_the_selection(view):
    list_assets(view)
    click_asset(view, 101)
    list_assets(view, cbxAuthors="alice")
    assert view.currentlySelectedRemoteAsset is None
    assert pixel_color(view.thumbnail) not in (None, RED)  # back to "not found"


@pytest.mark.xfail(strict=True, raises=KeyError, reason=(
    "assetdownload.py:498 takes the asset type from the combo box as it is "
    "now, not the type the list was made for: list clothes, pick another type "
    "without pressing 'Update list', click a row, and line 502 "
    "(remoteAssets['hair'][101]) raises KeyError inside the Qt slot (PyQt "
    "aborts MakeHuman, no excepthook is installed)"))
def test_row_click_after_changing_the_type_selects_the_listed_asset(env, view):
    list_assets(view)
    view.cbxTypes.setCurrentText("hair")
    click_asset(view, 101)
    env.raise_errors()
    assert view.currentlySelectedRemoteAsset is view.assetdb.remoteAssets["clothes"][101]


@pytest.mark.xfail(strict=True, raises=TypeError, reason=(
    "assetdownload.py:521 concatenates getChanged() into the details, but "
    "RemoteAsset defaults it to None when assets.json has no 'changed' "
    "(remoteasset.py:103); clicking such an asset raises TypeError inside the "
    "Qt slot (PyQt aborts MakeHuman, no excepthook is installed)"))
def test_row_click_on_asset_without_change_date(env, view):
    list_assets(view)
    click_asset(view, 103)
    env.raise_errors()
    assert view.currentlySelectedRemoteAsset.getId() == 103


# --------------------------------------------------------------------------
# Details panel
# --------------------------------------------------------------------------

def test_details_need_a_selected_asset(view):
    list_assets(view)
    view.btnDetails.click()
    assert view.isShowingDetails is False
    assert view.detailsPanel.isHidden() and not view.tableView.isHidden()


def test_details_toggle(view):
    list_assets(view)
    click_asset(view, 101)
    view.btnDetails.click()
    assert view.isShowingDetails is True
    assert view.tableView.isHidden() and not view.detailsPanel.isHidden()
    assert view.btnDetails.text() == "Hide details"
    view.btnDetails.click()
    assert view.isShowingDetails is False
    assert not view.tableView.isHidden() and view.detailsPanel.isHidden()
    assert view.btnDetails.text() == "View details"


def test_new_list_closes_the_details(view):
    list_assets(view)
    click_asset(view, 101)
    view.btnDetails.click()
    list_assets(view)
    assert view.isShowingDetails is False and view.detailsPanel.isHidden()
    assert view.btnDetails.text() == "View details"


def test_download_screenshot_button(env, view):
    env.publish(clothes_files(102, "item102")["render"], png_bytes(GREEN))
    list_assets(view)
    click_asset(view, 102)
    view.btnDetails.click()
    view.btnDownloadScreenshot.click()
    env.pump_until(lambda: view.btnDownloadScreenshot.isHidden())
    assert env.fetcher.opened == [clothes_files(102, "item102")["render"]]
    assert pixel_color(view.detailsRender) == GREEN
    assert os.path.exists(os.path.join(env.root, "102", "screenshot.png"))


def test_download_screenshot_that_fails_keeps_the_button(env, view):
    list_assets(view)
    click_asset(view, 102)
    view.btnDownloadScreenshot.click()  # not on the server
    env.pump_until(lambda: env.threads and env.threads[0].isFinished() and view.screenshotDt.downloadThread is None)
    assert not view.btnDownloadScreenshot.isHidden()
    assert pixel_color(view.detailsRender) != GREEN


# --------------------------------------------------------------------------
# Downloading and synchronising
# --------------------------------------------------------------------------

def test_download_needs_a_synchronized_database(env, empty_view):
    empty_view.btnDownload.click()
    assert env.threads == [] and env.messages == []


def test_download_needs_a_selected_asset(env, view):
    view.btnDownload.click()  # no list yet
    list_assets(view)
    view.btnDownload.click()  # list, but no row chosen
    assert env.threads == [] and env.messages == []


def test_download_button_installs_the_asset(env, view):
    env.publish_files(catalog()[0])
    list_assets(view)
    click_asset(view, 101)
    view.btnDownload.click()
    env.pump_until(lambda: env.messages)
    assert [m.text for m in env.messages] == ["Finished downloading"]
    install = env.data_path("clothes", "Red_Shirt")
    assert sorted(os.listdir(install)) == ["item101.mhclo", "item101.obj", "item101.thumb"]


def test_download_button_explains_a_server_error(env, view):
    list_assets(view)
    click_asset(view, 101)
    view.btnDownload.click()  # nothing on the server
    env.pump_until(lambda: env.messages)
    text = env.messages[0].text
    assert "error code 404" in text
    assert "when trying to download " + clothes_files(101, "item101")["mhclo"] in text
    assert "reported to the author of the asset" in text


def test_synchronize_reports_an_unavailable_server(env, empty_view):
    # nothing published: every file answers 404, as makehumancommunity.org does today for
    # assets.json (a first synchronisation stops at the first failing file, a seed zip)
    empty_view.btnSync.click()
    env.pump_until(lambda: env.messages)
    assert len(env.messages) == 1
    text = env.messages[0].text
    assert text.startswith("Could not download the asset database (error 404 for http")
    assert "synchronized" not in text
    assert items(empty_view.cbxAuthors) == ["-- any --"]
    assert not empty_view.assetdb.isSynchronized


@pytest.mark.parametrize("screenshots", [False, True])
def test_synchronize_button(env, empty_view, screenshots):
    assets = catalog()
    env.publish(ASSETS_JSON, json.dumps(dict((str(a["nid"]), a) for a in assets)))
    for a in assets:
        env.publish_files(a)
    env.publish_zip(THUMB_SEED, {"101/thumb.png": png_bytes(RED)})
    env.publish_zip(SCREEN_SEED, {"101/screenshot.png": png_bytes(GREEN)})
    assert items(empty_view.cbxAuthors) == ["-- any --"]

    empty_view.fetchScreens.setChecked(screenshots)
    empty_view.btnSync.click()
    env.pump_until(lambda: env.messages)

    assert [m.text for m in env.messages] == ["Asset DB is now synchronized"]
    assert items(empty_view.cbxAuthors) == ["-- any --", "alice", "bob", "Zed"]
    assert empty_view.assetdb.isSynchronized
    assert (SCREEN_SEED in env.fetcher.opened) is screenshots
    assert os.path.exists(os.path.join(env.root, "102", "screenshot.png")) is screenshots
    list_assets(empty_view)
    assert sorted(r[0] for r in rows(empty_view)) == ["101", "102", "103"]


def test_callbacks_without_a_job(env, view):
    assert view._onSyncProgress(0.5) is None
    assert view._downloadFinished(0, None) is None
    assert env.messages == []
