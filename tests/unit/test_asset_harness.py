"""
Shared harness for the asset downloader tests (plugins/8_asset_downloader),
plus tests of the harness itself. The other test_asset_*.py files import their
fixtures and helpers from here.

Import: the plugin modules read ``gui3d.app.mhapi`` when they are imported.
The full mhapi API needs a human, so this module builds the four namespaces
the plugin uses (utility, locations, assets, ui) from the real 1_mhapi code,
imports the plugin with them and puts gui3d.app back as it was. The packages
are loaded by file location; sys.path is not changed.

Network: the plugin downloads through mhapi.utility.getCompatibleUrlFetcher().
Every test gets a LocalFetcher there that maps the plugin's hard-coded hosts
(www.makehumancommunity.org, download.tuxfamily.org) to an http.server on
127.0.0.1 serving files from tmp_path, and socket.create_connection refuses
any other host. Refusals are recorded and fail the test at teardown, because
DownloadThread.run has a bare ``except`` that would otherwise hide them.

Qt: offscreen platform only. Panels and top widgets of the task view live in a
container that is never shown, and QMessageBox is replaced by a recorder, so
no window is ever created.

Callbacks: DownloadTask's callbacks run on the main thread from queued
signals. An exception there goes to sys.excepthook, and with the default hook
PyQt 5.15 aborts the process (measured: exit 134). Each test installs a
recording hook; pump_until() re-raises what it caught, and so does teardown
for anything left over. Exceptions swallowed by events3d.callEvent (button
handlers) are caught the same way through EventHandler.eventFailed.
"""

import functools
import http.server
import importlib
import importlib.util
import json
import os
import socket
import sys
import threading
import time
import types
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from core import G
import events3d
import getpath
import gui3d
import language
import progress

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MH_DIR = os.path.join(REPO, "makehuman")
PLUGINS = os.path.join(MH_DIR, "plugins")
PLUGIN_DIR = os.path.join(PLUGINS, "8_asset_downloader")

SITE = "http://www.makehumancommunity.org"
FILES = SITE + "/sites/default/files"
ASSETS_JSON = FILES + "/assets.json"
TUX = "http://download.tuxfamily.org/makehuman/assets"
THUMB_SEED = TUX + "/asset-db-thumbnails.zip"
SCREEN_SEED = TUX + "/asset-db-screenshots.zip"
REMOTE_HOSTS = ("www.makehumancommunity.org", "download.tuxfamily.org")
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


# --------------------------------------------------------------------------
# Importing the plugin
# --------------------------------------------------------------------------

def _load_package(name):
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(PLUGINS, name)
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _build_mhapi():
    _load_package("1_mhapi")
    api = types.SimpleNamespace()
    api.utility = importlib.import_module("1_mhapi._utility").Utility(api)
    api.locations = importlib.import_module("1_mhapi._locations").Locations(api)
    api.assets = importlib.import_module("1_mhapi._assets").Assets(api)
    api.ui = importlib.import_module("1_mhapi._ui").UI(api)
    return api


_saved_app = gui3d.app
gui3d.app = types.SimpleNamespace(mhapi=_build_mhapi())
try:
    _load_package("8_asset_downloader")
    assetdb = importlib.import_module("8_asset_downloader.assetdb")
    remoteasset = importlib.import_module("8_asset_downloader.remoteasset")
    downloadtask = importlib.import_module("8_asset_downloader.downloadtask")
    assetcleaner = importlib.import_module("8_asset_downloader.assetcleaner")
    assetdownload = importlib.import_module("8_asset_downloader.assetdownload")
    tablemodel = importlib.import_module("8_asset_downloader.tablemodel")
finally:
    gui3d.app = _saved_app

mhapi = assetdb.mhapi  # the namespace object every plugin module captured


# --------------------------------------------------------------------------
# Stand-ins for the application
# --------------------------------------------------------------------------

class Panel(QtWidgets.QWidget):
    """Left/right task panel: just a layout that keeps what is added."""

    def __init__(self, parent):
        super(Panel, self).__init__(parent)
        self.box = QtWidgets.QVBoxLayout(self)
        self.added = []

    def addWidget(self, widget, *args, **kwargs):
        self.box.addWidget(widget)
        self.added.append(widget)
        return widget

    def removeWidget(self, widget):
        self.box.removeWidget(widget)
        self.added.remove(widget)


class FakeMainWin(object):
    """Parents every panel and top widget to a container that is never shown."""

    def __init__(self):
        self.container = QtWidgets.QWidget()
        self.box = QtWidgets.QVBoxLayout(self.container)
        self.panels = []
        self.top = []

    def addPanels(self):
        left, right = Panel(self.container), Panel(self.container)
        self.panels.append((left, right))
        return left, right

    def showPanels(self, left, right):
        pass

    def addTopWidget(self, widget, *args, **kwargs):
        self.box.addWidget(widget)
        self.top.append(widget)
        return widget

    def removeTopWidget(self, widget):
        self.box.removeWidget(widget)
        self.top.remove(widget)


class FakeApp(object):
    """The parts of the MakeHuman application object the plugin reaches."""

    def __init__(self, api):
        self.mhapi = api
        self.mainwin = FakeMainWin()
        self.progress_calls = []
        self.redraws = 0

    def progress(self, value, text="", *args):
        self.progress_calls.append((value, text))

    def redraw(self):
        self.redraws += 1


def message_box_class(record):
    class FakeMessageBox(object):
        Information = QtWidgets.QMessageBox.Information
        Ok = QtWidgets.QMessageBox.Ok

        def __init__(self):
            self.icon = self.text = self.title = self.buttons = None

        def setIcon(self, icon):
            self.icon = icon

        def setText(self, text):
            self.text = text

        def setWindowTitle(self, title):
            self.title = title

        def setStandardButtons(self, buttons):
            self.buttons = buttons

        def show(self):
            record.append(self)

    return FakeMessageBox


# --------------------------------------------------------------------------
# Local web server and URL fetcher
# --------------------------------------------------------------------------

class _Handler(http.server.SimpleHTTPRequestHandler):
    """Static files from the server root. Paths under /no-length/ are served
    without a Content-Length header (HTTP/1.0, body ends when the connection
    closes), as a streaming or chunked response would be."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.requests.append(self.path)
        if self.path.startswith("/no-length/"):
            rel = urllib.parse.unquote(self.path[len("/no-length/"):])
            with open(os.path.join(self.directory, rel), "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(data)
            return
        super(_Handler, self).do_GET()


class LocalServer(object):

    def __init__(self, root):
        self.root = str(root)
        os.makedirs(self.root, exist_ok=True)
        handler = functools.partial(_Handler, directory=self.root)
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.requests = []
        self.httpd.requests = self.requests
        self.base = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        # shutdown() waits for the next poll: keep it short, there is a server per test
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(10)


class LocalFetcher(object):
    """Takes the place of urllib.request inside the plugin."""

    def __init__(self, base):
        self.base = base
        self.opened = []
        self.refused = []
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def urlopen(self, url, *args, **kwargs):
        parts = urllib.parse.urlsplit(url)
        if parts.hostname in REMOTE_HOSTS:
            target = self.base + urllib.parse.urlunsplit(("", "", parts.path, parts.query, ""))
        elif parts.hostname in LOCAL_HOSTS:
            target = url
        else:
            self.refused.append(url)
            raise urllib.error.URLError("test harness: network access refused for " + url)
        self.opened.append(url)
        return self._opener.open(target, timeout=15)


def url_path(url):
    """Path a remote URL is served from, relative to the local server root."""
    return urllib.parse.unquote(urllib.parse.urlsplit(url).path).lstrip("/")


# --------------------------------------------------------------------------
# Fixture data
# --------------------------------------------------------------------------

def png_bytes(color="#ff0000", size=8):
    img = QtGui.QImage(size, size, QtGui.QImage.Format_RGB32)
    img.fill(QtGui.QColor(color))
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.WriteOnly)
    assert img.save(buf, "PNG")
    return bytes(buf.data())


def pixel_color(label):
    """Colour at the centre of the pixmap shown by a QLabel, or None."""
    pix = label.pixmap()
    if pix is None or pix.isNull():
        return None
    img = pix.toImage()
    return img.pixelColor(img.width() // 2, img.height() // 2).name()


def clothes_files(nid, name):
    base = "%s/assets/%d/%s" % (FILES, nid, name)
    return {"mhclo": base + ".mhclo", "obj": base + ".obj",
            "thumb": base + ".thumb", "render": base + ".png"}


def asset_json(nid, title, type="clothes", author="alice", license="CC0",
               changed="2020-01-01 10:00:00", description="", files=None, **extra):
    raw = {"nid": nid, "uid": 1000 + nid, "title": title, "type": type,
           "username": author, "license": license, "changed": changed,
           "created": "2019-01-01 10:00:00", "description": description}
    if files is not None:
        raw["files"] = files
    raw.update(extra)
    return raw


def make_zip(path, entries):
    with zipfile.ZipFile(str(path), "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return str(path)


# --------------------------------------------------------------------------
# Per-test environment
# --------------------------------------------------------------------------

_QAPP = []  # keeps the QApplication alive for the whole session


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    assert isinstance(app, QtWidgets.QApplication)
    # Never risk a visible window: the user works on another screen
    assert app.platformName() == "offscreen", app.platformName()
    _QAPP.append(app)
    return app


class Env(object):

    def __init__(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch

        # Isolated MakeHuman home for getpath (re-read on every getHomePath call)
        self.home = tmp_path / "home"
        self.home.mkdir()
        monkeypatch.setenv("MH_HOME_LOCATION", str(self.home))
        monkeypatch.setattr(getpath, "__home_path", None)
        monkeypatch.setattr(mhapi.utility, "logChannels", {})  # log files in this home

        self.app = FakeApp(mhapi)
        monkeypatch.setattr(G, "app", self.app)
        monkeypatch.setattr(gui3d, "app", self.app)
        monkeypatch.setattr(language.language, "languageStrings", None)
        monkeypatch.setattr(progress, "current_Progress_", None)

        self.slot_errors = []
        self.event_errors = []
        monkeypatch.setattr(sys, "excepthook", lambda t, v, tb: self.slot_errors.append(v))
        monkeypatch.setattr(events3d.EventHandler, "eventFailed",
                            lambda handler, level: self.event_errors.append(sys.exc_info()[1]))

        self.messages = []
        monkeypatch.setattr(assetdownload, "QMessageBox", message_box_class(self.messages))

        self.server = LocalServer(tmp_path / "www")
        self.fetcher = LocalFetcher(self.server.base)
        monkeypatch.setattr(mhapi.utility, "urlrequest", self.fetcher)

        self.socket_refused = []
        real_connect = socket.create_connection

        def guarded_connect(address, *args, **kwargs):
            if address[0] not in LOCAL_HOSTS:
                self.socket_refused.append(address)
                raise OSError("test harness: network access refused for %r" % (address,))
            return real_connect(address, *args, **kwargs)

        monkeypatch.setattr(socket, "create_connection", guarded_connect)

        self.threads = []
        threads = self.threads

        class RecordingDownloadThread(downloadtask.DownloadThread):
            def __init__(self, *args, **kwargs):
                super(RecordingDownloadThread, self).__init__(*args, **kwargs)
                threads.append(self)  # also keeps it alive until it has stopped

        monkeypatch.setattr(downloadtask, "DownloadThread", RecordingDownloadThread)

    # -- paths ------------------------------------------------------------

    @property
    def root(self):
        return mhapi.locations.getUserDataPath("community-assets")

    def data_path(self, *parts):
        return os.path.join(mhapi.locations.getUserDataPath(), *parts)

    # -- server content ---------------------------------------------------

    def publish(self, url, data):
        """Make ``url`` (on a remote host or the local one) answer with ``data``."""
        path = os.path.join(self.server.root, url_path(url))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data if isinstance(data, bytes) else data.encode("utf-8"))
        return path

    def publish_zip(self, url, entries):
        data_path = make_zip(self.tmp_path / ("zip-%d.zip" % len(os.listdir(str(self.tmp_path)))), entries)
        with open(data_path, "rb") as f:
            return self.publish(url, f.read())

    def local_url(self, rel):
        return self.server.base + "/" + rel

    def publish_files(self, raw, payloads=None):
        """Publish every file of an asset; ``payloads`` maps file keys to bytes."""
        payloads = payloads or {}
        for key, url in raw.get("files", {}).items():
            default = png_bytes() if key in ("thumb", "render", "illustration") else ("%s of %s" % (key, raw["title"])).encode()
            self.publish(url, payloads.get(key, default))

    # -- local database ---------------------------------------------------

    def write_remote_db(self, assets):
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, "remote.json"), "w") as f:
            json.dump(dict((str(a["nid"]), a) for a in assets), f)

    def write_local_db(self, local):
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, "local.json"), "w") as f:
            json.dump(local, f)

    def plugin_log(self):
        """What the plugin wrote to its own log channel (warnings and errors at the default level)."""
        path = os.path.join(mhapi.locations.getUserHomePath("plugin_logs"), "assetdownload.txt")
        with open(path, encoding="utf-8") as f:
            return f.read()

    def read_json(self, name):
        with open(os.path.join(self.root, name)) as f:
            return json.load(f)

    # -- Qt event loop ----------------------------------------------------

    def raise_errors(self):
        for errors in (self.slot_errors, self.event_errors):
            if errors:
                raise errors.pop(0)

    def pump_until(self, condition, timeout=20.0):
        deadline = time.monotonic() + timeout
        while True:
            QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
            self.raise_errors()
            if condition():
                return
            if time.monotonic() > deadline:
                raise AssertionError("timed out waiting for the download callbacks")
            time.sleep(0.005)

    def close(self):
        problems = []
        for t in self.threads:
            if not t.wait(20000):
                problems.append("a DownloadThread was still running after 20 s")
        QtCore.QCoreApplication.processEvents()  # deliver what the threads queued
        self.server.close()
        problems.extend("callback raised %r" % e for e in self.slot_errors)
        problems.extend("event handler raised %r" % e for e in self.event_errors)
        problems.extend("network access attempted: %s" % u for u in self.fetcher.refused)
        problems.extend("network access attempted: %r" % (a,) for a in self.socket_refused)
        assert not problems, problems


@pytest.fixture
def env(qapp, tmp_path, monkeypatch):
    e = Env(tmp_path, monkeypatch)
    yield e
    e.close()


# --------------------------------------------------------------------------
# Tests of the harness itself
# --------------------------------------------------------------------------

def test_plugin_modules_are_the_repository_files():
    for module in (assetdb, remoteasset, downloadtask, assetcleaner, assetdownload, tablemodel):
        assert os.path.dirname(os.path.abspath(module.__file__)) == PLUGIN_DIR
        assert module.mhapi is mhapi
    assert gui3d.app is _saved_app  # the import left the application global alone


def test_home_is_isolated_in_tmp_path(env):
    assert env.root.startswith(str(env.home))
    assert env.data_path("clothes").startswith(str(env.home))


def test_fetcher_maps_remote_hosts_to_the_local_server(env):
    env.publish(ASSETS_JSON, b'{"x": 1}')
    with env.fetcher.urlopen(ASSETS_JSON) as r:
        assert r.read() == b'{"x": 1}'
    assert env.fetcher.opened == [ASSETS_JSON]
    assert env.server.requests == ["/sites/default/files/assets.json"]


def test_fetcher_refuses_other_hosts(env):
    with pytest.raises(urllib.error.URLError):
        env.fetcher.urlopen("http://example.invalid/assets.json")
    assert env.fetcher.refused == ["http://example.invalid/assets.json"]
    env.fetcher.refused.clear()  # consumed: teardown would fail on it


def test_socket_guard_refuses_direct_connections(env):
    with pytest.raises(OSError):
        # .invalid never resolves (RFC 2606), so even a broken guard sends nothing anywhere
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open("http://example.invalid/", timeout=5)
    assert env.socket_refused == [("example.invalid", 80)]
    env.socket_refused.clear()


def test_no_length_route_omits_content_length(env):
    env.publish(env.local_url("blob.bin"), b"abc")
    with env.fetcher.urlopen(env.local_url("no-length/blob.bin")) as r:
        assert r.info().get("Content-Length") is None
        assert r.read() == b"abc"


def test_exception_in_a_queued_slot_is_recorded_and_reraised(env):
    class Emitter(QtCore.QThread):
        fired = QtCore.pyqtSignal(str)

        def run(self):
            self.fired.emit("x")

    calls = []

    def slot(value):
        calls.append(value)
        raise RuntimeError("boom in slot")

    thread = Emitter()
    thread.fired.connect(slot)
    thread.start()
    assert thread.wait(10000)
    with pytest.raises(RuntimeError, match="boom in slot"):
        env.pump_until(lambda: False, timeout=5)
    assert calls == ["x"]  # delivered on the main thread, process still alive


def test_exception_in_a_button_handler_is_reraised(env):
    import gui
    button = gui.Button("x")

    @button.mhEvent
    def onClicked(event):
        raise KeyError("handler")

    button.click()
    with pytest.raises(KeyError):
        env.raise_errors()
