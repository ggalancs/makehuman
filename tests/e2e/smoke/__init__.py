"""
End-to-end smoke test that runs inside a real MakeHuman process.

tests/e2e/run_smoke.py installs this package as a user plugin in an isolated
MakeHuman home and launches the app. Once the GUI has finished starting, it
drives the application through its own APIs, records one result per check in
the JSON file named by MH_SMOKE_RESULTS (rewritten after every check, so a hang
still leaves the partial results on disk) and quits.

MakeHuman catches many exceptions and only logs them. A logging handler
attributes every ERROR record to the check that was running, and such a check
counts as failed even though nothing was raised.
"""

import glob
import json
import logging
import os
import platform
import socket
import sys
import threading
import time
import traceback

import numpy as np
from PyQt5 import QtCore, QtWidgets

import getpath
import gui3d
import mh
from core import G

RESULTS_PATH = os.environ.get("MH_SMOKE_RESULTS")
OUTDIR = os.environ.get("MH_SMOKE_OUTDIR")
SOCKET_PORT = int(os.environ.get("MH_SMOKE_SOCKET_PORT", "0"))


class ErrorCollector(logging.Handler):
    """Errors, and warnings that carry an exception: MakeHuman logs exceptions raised
    inside GUI event handlers as warnings ("Exception during event ...")."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        if record.levelno < logging.ERROR and not record.exc_info:
            return
        try:
            text = logging.Formatter("%(name)s: %(message)s").format(record)
        except Exception:
            text = repr(record.msg)
        self.records.append(text)


def pump(seconds=0.0):
    """Let Qt process pending events (paints, queued signals) for a while."""
    end = time.time() + seconds
    while True:
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
        if time.time() >= end:
            return
        time.sleep(0.01)


class AssertionFailed(Exception):
    pass


def expect(condition, message):
    if not condition:
        raise AssertionFailed(message)


class Smoke(object):
    def __init__(self):
        self.results = []
        self.modals = []
        self.modal_accept = False  # a check sets this to answer the next dialogs with OK/Yes
        self.collector = ErrorCollector()
        # Root logger plus the non-propagating loggers from data/logging.ini
        for name in [None, "mh.callEvent", "mh.callAsync", "mh.redraw", "mh.event", "mh.load"]:
            logging.getLogger(name).addHandler(self.collector)

    # -- bookkeeping -------------------------------------------------------

    def flush(self, complete=False):
        if not RESULTS_PATH:
            return
        tmp = RESULTS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"complete": complete, "modals": self.modals, "results": self.results}, f, indent=2)
        os.replace(tmp, RESULTS_PATH)

    def allow_logged(self, substring):
        """Declare an error the current check provokes on purpose (matched by substring)."""
        self.allowed_errors.append(substring)

    def check(self, name, fn):
        self.allowed_errors = []
        self.modal_accept = False
        first_record = len(self.collector.records)
        entry = {"name": name, "running": True}
        self.results.append(entry)
        self.flush()  # if the check hangs, the harness knows which one
        start = time.time()
        try:
            detail = fn()
            entry["ok"] = True
            if detail is not None:
                entry["detail"] = detail
        except Exception:
            entry["ok"] = False
            entry["error"] = traceback.format_exc()
        pump()
        logged = [r for r in self.collector.records[first_record:]
                  if not any(allowed in r for allowed in self.allowed_errors)]
        if logged:
            entry["ok"] = False
            entry["logged_errors"] = logged
        entry["seconds"] = round(time.time() - start, 2)
        del entry["running"]
        self.flush()

    def close_modals(self):
        """Modal dialogs would block an unattended run: record and dismiss them."""
        w = QtWidgets.QApplication.activeModalWidget()
        if w is not None:
            text = w.windowTitle()
            if isinstance(w, QtWidgets.QMessageBox):
                text += ": " + w.text()
            elif hasattr(getattr(w, "text", None), "text"):  # MakeHuman's gui.Dialog
                text += ": " + w.text.text()
            self.modals.append(text)
            if self.modal_accept and hasattr(w, "accept"):
                w.accept()
            else:
                w.reject() if hasattr(w, "reject") else w.close()

    # -- helpers -----------------------------------------------------------

    @property
    def human(self):
        return G.app.selectedHuman

    def out(self, *parts):
        path = os.path.join(OUTDIR, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def first_asset(self, folder, pattern):
        found = sorted(glob.glob(os.path.join(getpath.getSysDataPath(folder), "*", pattern)))
        found = [f for f in found if "none" not in os.path.basename(f).lower()]
        expect(found, "no %s assets found in %s" % (pattern, getpath.getSysDataPath(folder)))
        return found

    # -- checks --------------------------------------------------------------

    def check_environment(self):
        from OpenGL import GL
        from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
        import OpenGL
        G.canvas.makeCurrent()
        return {
            "python": sys.version.split()[0],
            "machine": platform.machine(),
            "qt": QT_VERSION_STR,
            "pyqt": PYQT_VERSION_STR,
            "numpy": np.__version__,
            "pyopengl": OpenGL.__version__,
            "gl_renderer": GL.glGetString(GL.GL_RENDERER).decode(),
            "gl_version": GL.glGetString(GL.GL_VERSION).decode(),
            "canvas_logical": [G.windowWidth, G.windowHeight],
            "device_pixel_ratio": G.canvas.devicePixelRatioF(),
        }

    def check_visit_all_tasks(self):
        visited, failures = [], {}
        categories = sorted(G.app.categories.values(), key=lambda c: c.sortOrder)
        for category in categories:
            for task in list(category.tasks):
                n = len(self.collector.records)
                try:
                    mh.changeTask(category.name, task.name)
                    pump(0.05)
                except Exception:
                    failures["%s/%s" % (category.name, task.name)] = traceback.format_exc()
                if len(self.collector.records) > n:
                    failures["%s/%s" % (category.name, task.name)] = self.collector.records[n:]
                visited.append("%s/%s" % (category.name, task.name))
        mh.changeTask("Modelling", "Main")
        expect(not failures, "tasks failed: %s" % json.dumps(failures, indent=1))
        return {"visited": len(visited), "tasks": visited}

    def check_macro_modifiers(self):
        human = self.human
        before = human.meshData.coord.copy()
        human.setGender(1.0)
        human.setAge(0.7)
        human.setMuscle(0.8)
        human.setWeight(0.3)
        human.setHeight(0.9)
        human.applyAllTargets()
        after = human.meshData.coord
        expect(np.isfinite(after).all(), "non-finite coordinates after macro change")
        moved = float(np.abs(after - before).max())
        expect(moved > 0.1, "macro modifiers did not change the mesh (max delta %s)" % moved)
        return {"max_vertex_delta": round(moved, 3)}

    def check_modifier_undo_redo(self):
        import humanmodifier
        mod = self.human.getModifier("nose/nose-scale-horiz-decr|incr")
        old = mod.getValue()
        G.app.do(humanmodifier.ModifierAction(mod, old, 0.7, lambda: None))
        expect(abs(mod.getValue() - 0.7) < 1e-6, "do() did not apply")
        G.app.undo()
        expect(abs(mod.getValue() - old) < 1e-6, "undo() did not restore")
        G.app.redo()
        expect(abs(mod.getValue() - 0.7) < 1e-6, "redo() did not re-apply")

    def visible_body_points(self):
        """Screen positions (logical pixels) of front-most body vertices at several heights."""
        human = self.human
        mesh = human.meshData
        cam = G.app.modelCamera
        faces = mesh.getFacesForGroups(["body"])
        verts = np.unique(mesh.fvert[faces].ravel())
        coords = mesh.coord[verts]
        screen = np.array([cam.convertToScreen(*c, obj=human.mesh) for c in coords])
        points = []
        ys = screen[:, 1]
        for q in (0.15, 0.3, 0.5, 0.7):
            lo, hi = np.quantile(ys, [q - 0.03, q + 0.03])
            band = np.where((ys >= lo) & (ys <= hi))[0]
            # front-most vertex of the band (smallest depth), nudged off the silhouette
            i = band[np.argmin(screen[band, 2])]
            points.append((int(round(screen[i, 0])), int(round(screen[i, 1]))))
        return points

    def check_picking(self):
        """Hit-testing must find the body under its own projected vertices (fails if HiDPI scaling is wrong)."""
        import selection
        G.app.actions.front.trigger()
        G.app.redraw()
        pump(0.3)
        import glmodule
        G.canvas.makeCurrent()  # as Canvas.mouse*Event does before picking
        glmodule.markPickingBufferDirty()
        hits = []
        for x, y in self.visible_body_points():
            picked = mh.getPickedColor(x, y)
            fg = selection.selectionColorMap.getSelectedFaceGroup(picked)
            hits.append({"xy": [x, y], "group": fg.name if fg else None})
        corner = mh.getPickedColor(3, 3)
        background = selection.selectionColorMap.getSelectedFaceGroup(corner)
        missed = [h for h in hits if h["group"] is None]
        expect(not missed, "picking missed the body: %s" % hits)
        expect(background is None, "picking found %s in the empty corner" % background.name if background else "")
        return hits

    def check_depth_query(self):
        """Screen -> world through the depth buffer must land back on the vertex."""
        human = self.human
        cam = G.app.modelCamera
        (x, y) = self.visible_body_points()[2]
        G.canvas.makeCurrent()
        wx, wy, wz = cam.convertToWorld2D(x, y, obj=human.mesh)
        coords = human.meshData.coord
        dist = float(np.min(np.linalg.norm(coords - np.array([wx, wy, wz]), axis=1)))
        expect(dist < 0.5, "depth query maps to a point %.3f units away from any vertex" % dist)
        return {"distance_to_mesh": round(dist, 4)}

    def check_grab_screen(self):
        """The grab must contain the whole model, centred (a mis-scaled read returns one corner of it)."""
        G.app.actions.front.trigger()
        pump(0.2)
        import image
        path = self.out("grab.png")
        img = mh.grabScreen(0, 0, G.windowWidth, G.windowHeight, path)
        ratio = G.canvas.devicePixelRatioF()
        expected = [int(round(G.windowWidth * ratio)), int(round(G.windowHeight * ratio))]
        expect([img.width, img.height] == expected, "grab is %sx%s, expected %s" % (img.width, img.height, expected))
        result = self.analyse_grab(img)
        # The toolbar button goes through MHApplication.grabScreen without makeCurrent()
        grab_dir = mh.getPath("grab")
        before = set(os.listdir(grab_dir)) if os.path.isdir(grab_dir) else set()
        G.app.grabScreen()
        new = sorted(set(os.listdir(grab_dir)) - before)
        expect(len(new) == 1, "toolbar grab wrote %s" % new)
        result["toolbar"] = self.analyse_grab(image.Image(os.path.join(grab_dir, new[0])))
        result.update({"size": [img.width, img.height], "logical": [G.windowWidth, G.windowHeight], "file": path})
        return result

    def analyse_grab(self, img):
        data = img.data[..., :3].astype(int)
        h, w = data.shape[:2]
        # Skin is saturated red-brown; background and grid lines are grey (R == G == B)
        skin = (data[..., 0] - data[..., 2] > 40) & (data[..., 0] > 60)
        ys, xs = np.nonzero(skin)
        expect(len(xs) > 0.005 * w * h, "no skin pixels in the grab (%d)" % len(xs))
        cx = xs.mean() / w
        height = (ys.max() - ys.min() + 1) / h
        expect(abs(cx - 0.5) < 0.1, "model not centred horizontally in grab (centroid x %.2f)" % cx)
        expect(height > 0.6, "model only spans %.0f%% of the grab height" % (100 * height))
        return {"centroid_x": round(cx, 3), "model_height": round(height, 3)}

    def check_proxies(self):
        loaded = {}
        for task, folder, pattern in [("Eyes", "eyes", "*.mhclo"), ("Hair", "hair", "*.mhclo"),
                                      ("Eyebrows", "eyebrows", "*.mhclo"), ("Eyelashes", "eyelashes", "*.mhclo"),
                                      ("Teeth", "teeth", "*.mhclo"), ("Tongue", "tongue", "*.mhclo"),
                                      ("Clothes", "clothes", "*.mhclo")]:
            tv = G.app.getTask("Geometries", task)
            files = self.first_asset(folder, pattern)
            for f in files[: 2 if task == "Clothes" else 1]:
                tv.proxyFileSelected(f)
                pump(0.05)
            names = [p.name for p in tv.getSelection()]
            expect(names, "%s: nothing selected after proxyFileSelected" % task)
            loaded[task] = names
        objs = self.human.getObjects(excludeZeroFaceObjs=True)
        expect(len(objs) >= 8, "expected human + 8 proxies, got %d objects" % len(objs))
        return loaded

    def check_topology_proxy(self):
        tv = G.app.getTask("Geometries", "Topologies")
        f = self.first_asset("proxymeshes", "*.proxy")[0]
        tv.proxyFileSelected(f)
        pump(0.1)
        expect(self.human.isProxied(), "human not proxied after selecting %s" % f)
        tv.resetSelection()
        pump(0.1)
        expect(not self.human.isProxied(), "topology proxy not removed")
        return os.path.basename(f)

    def check_skin_material(self):
        tv = G.app.getTask("Materials", "Material")
        skins = sorted(glob.glob(os.path.join(getpath.getSysDataPath("skins"), "*", "*.mhmat")))
        expect(skins, "no skins installed")
        mh.changeTask("Materials", "Material")
        tv.filechooser.callEvent("onFileSelected", skins[0])
        pump(0.1)
        got = self.human.material.filename
        expect(os.path.realpath(got) == os.path.realpath(skins[0]), "skin is %s, expected %s" % (got, skins[0]))
        mh.changeTask("Modelling", "Main")
        return os.path.basename(skins[0])

    def check_skeleton_and_pose(self):
        human = self.human
        G.app.getTask("Pose/Animate", "Skeleton").chooseSkeleton(getpath.getSysDataPath("rigs/default.mhskel"))
        pump(0.1)
        expect(human.getSkeleton() is not None, "no skeleton after chooseSkeleton")
        before = human.meshData.coord.copy()
        G.app.getTask("Pose/Animate", "Pose").loadPose(getpath.getSysDataPath("poses/benchmark.bvh"))
        pump(0.2)
        moved = float(np.abs(human.meshData.coord - before).max())
        expect(moved > 0.1, "pose did not move the mesh (max delta %s)" % moved)
        return {"pose_max_delta": round(moved, 3)}

    def check_expression(self):
        human = self.human
        files = sorted(glob.glob(os.path.join(getpath.getSysDataPath("expressions"), "*.mhpose")))
        expect(len(files) >= 30, "expected the 32 standard expressions, found %d" % len(files))
        smile = [f for f in files if os.path.basename(f) == "smile01.mhpose"][0]
        before = human.meshData.coord.copy()
        G.app.getTask("Pose/Animate", "Expressions").chooseExpression(smile)
        pump(0.2)
        moved = float(np.abs(human.meshData.coord - before).max())
        expect(moved > 0.01, "expression did not move the face (max delta %s)" % moved)
        return {"expressions": len(files), "smile01_max_delta": round(moved, 3)}

    def check_game_engine_skeleton(self):
        human = self.human
        rigs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(getpath.getSysDataPath("rigs"), "*.mhskel")))
        expect({"game_engine.mhskel", "cmu_mb.mhskel", "default_no_toes.mhskel"} <= set(rigs), "rigs: %s" % rigs)
        G.app.getTask("Pose/Animate", "Skeleton").chooseSkeleton(getpath.getSysDataPath("rigs/game_engine.mhskel"))
        pump(0.1)
        skel = human.getSkeleton()
        expect(skel is not None and skel.name.lower().startswith("game"), "skeleton is %s" % (skel and skel.name))
        return {"rigs": rigs, "bones": skel.getBoneCount()}

    def check_smooth_and_wireframe(self):
        human = self.human
        G.app.actions.smooth.setChecked(True)
        G.app.toggleSubdivision()
        pump(0.2)
        expect(human.isSubdivided(), "not subdivided")
        G.app.actions.smooth.setChecked(False)
        G.app.toggleSubdivision()
        G.app.actions.wireframe.setChecked(True)
        G.app.toggleSolid()
        pump(0.1)
        G.app.actions.wireframe.setChecked(False)
        G.app.toggleSolid()
        for name in ("front", "back", "left", "right", "top", "bottom", "resetCam"):
            getattr(G.app.actions, name).trigger()
            pump(0.02)

    def check_random(self):
        rnd = G.app.modules["0_modeling_8_random"]
        rnd.randomize(self.human, symmetry=1.0, macro=True, height=True, face=True, body=True)
        expect(np.isfinite(self.human.meshData.coord).all(), "non-finite coordinates after randomize")

    def check_render(self):
        tv = G.app.getTask("Rendering", "Render")
        viewer = G.app.getTask("Rendering", "Viewer")
        mh.changeTask("Rendering", "Render")
        sizes = {}
        for aa in (False, True):
            tv.AAbox.setSelected(aa)
            tv.renderingWidth, tv.renderingHeight = 320, 240
            tv.renderButton.callEvent("onClicked", None)
            pump(0.2)
            pm = viewer.image.imageLabel.pixmap()
            expect(pm is not None and not pm.isNull(), "no rendered image in viewer (AA=%s)" % aa)
            sizes["AA" if aa else "noAA"] = [pm.width(), pm.height()]
        mh.changeTask("Modelling", "Main")
        return sizes

    def check_save_load_mhm(self):
        import guisave
        path = self.out("model.mhm")
        human = self.human
        values = {m.fullName: m.getValue() for m in human.modifiers}
        proxies = sorted(p.name for p in human.getProxies(includeHumanProxy=False))
        guisave.saveMHM(path)
        expect(os.path.getsize(path) > 0, "empty mhm")
        import image
        thumb = image.Image(os.path.splitext(path)[0] + ".thumb")
        expect(thumb.width <= 128 and thumb.height <= 128, "thumbnail is %dx%d" % (thumb.width, thumb.height))
        human.resetMeshValues()
        G.app.loadHumanMHM(path)
        pump(0.2)
        diff = {k: (v, human.getModifier(k).getValue()) for k, v in values.items()
                if abs(human.getModifier(k).getValue() - v) > 1e-4}
        expect(not diff, "modifiers differ after reload: %s" % list(diff.items())[:5])
        reloaded = sorted(p.name for p in human.getProxies(includeHumanProxy=False))
        expect(reloaded == proxies, "proxies differ after reload: %s vs %s" % (proxies, reloaded))
        return {"modifiers": len(values), "proxies": proxies}

    def check_exports(self):
        tv = G.app.getTask("Files", "Export")
        report, failures = {}, {}
        for exporter, _radio, _options in tv.formats:
            name = exporter.name
            folder = self.out("exports", name.replace(" ", "_"), "x")[:-2]
            n = len(self.collector.records)
            try:
                exporter.export(self.human, lambda ext, folder=folder: os.path.join(folder, "model." + ext.lstrip(".")))
                pump(0.05)
                files = [f for f in glob.glob(os.path.join(folder, "**"), recursive=True) if os.path.isfile(f)]
                size = sum(os.path.getsize(f) for f in files)
                report[name] = {"files": len(files), "bytes": size}
                if size == 0:
                    failures[name] = "no output"
            except Exception:
                failures[name] = traceback.format_exc()
            if len(self.collector.records) > n:
                failures[name] = self.collector.records[n:]
        expect(report, "no exporters registered")
        expect(not failures, "exports failed: %s" % json.dumps(failures, indent=1))
        return report

    def check_socket_server(self):
        expect(SOCKET_PORT, "MH_SMOKE_SOCKET_PORT not set")
        replies = {}

        def ask(function):
            with socket.create_connection(("127.0.0.1", SOCKET_PORT), timeout=10) as s:
                s.sendall(json.dumps({"function": function, "params": {}, "error": "", "data": ""}).encode())
                chunks = []
                while True:
                    b = s.recv(65536)
                    if not b:
                        break
                    chunks.append(b)
            replies[function] = json.loads(b"".join(chunks).decode())

        for function in ("getUserDir", "getBodyMeshInfo", "getAvailableModifierNames"):
            t = threading.Thread(target=ask, args=(function,), daemon=True)
            t.start()
            deadline = time.time() + 15
            while t.is_alive() and time.time() < deadline:
                pump(0.05)  # the server answers through a queued signal on this thread
            expect(function in replies, "no reply to %s within 15s" % function)
            expect(not replies[function].get("error"), "%s error: %s" % (function, replies[function].get("error")))
        return {k: (str(v.get("data"))[:80]) for k, v in replies.items()}

    # -- driver --------------------------------------------------------------

    def run(self):
        checks = [
            ("environment", self.check_environment),
            ("visit_all_tasks", self.check_visit_all_tasks),
            ("macro_modifiers", self.check_macro_modifiers),
            ("modifier_undo_redo", self.check_modifier_undo_redo),
            ("picking_hidpi", self.check_picking),
            ("depth_query", self.check_depth_query),
            ("grab_screen", self.check_grab_screen),
            ("proxies", self.check_proxies),
            ("topology_proxy", self.check_topology_proxy),
            ("skin_material", self.check_skin_material),
            ("skeleton_and_pose", self.check_skeleton_and_pose),
            ("expression", self.check_expression),
            ("smooth_wireframe_cameras", self.check_smooth_and_wireframe),
            ("random", self.check_random),
            ("render_opengl", self.check_render),
            ("save_load_mhm", self.check_save_load_mhm),
            ("game_engine_skeleton", self.check_game_engine_skeleton),
            ("exports", self.check_exports),
            ("socket_server", self.check_socket_server),
        ]
        from . import checks_api, checks_app, checks_plugins, checks_utilities
        for module in (checks_utilities, checks_api, checks_plugins, checks_app):
            checks += [(name, (lambda fn: lambda: fn(self))(fn)) for name, fn in module.CHECKS]
        only = os.environ.get("MH_SMOKE_ONLY")
        if only:
            keep = set(only.split(","))
            checks = [c for c in checks if c[0] in keep or c[0] == "environment"]
        for name, fn in checks:
            self.check(name, fn)
        self.flush(complete=True)
        log_quit = logging.getLogger()
        log_quit.info("smoke test finished, quitting")
        QtCore.QTimer.singleShot(200, G.app.stop)


_smoke = None
_modal_timer = None


def _wait_for_startup():
    # startupSequence sets splash to None once the main window is up
    if getattr(G.app, "splash", True) is not None or not G.app.mainwin.isVisible():
        QtCore.QTimer.singleShot(250, _wait_for_startup)
        return
    QtCore.QTimer.singleShot(500, _smoke.run)


def load(app):
    global _smoke, _modal_timer
    _smoke = Smoke()
    _modal_timer = QtCore.QTimer()
    _modal_timer.timeout.connect(_smoke.close_modals)
    _modal_timer.start(300)
    QtCore.QTimer.singleShot(250, _wait_for_startup)


def unload(app):
    pass
