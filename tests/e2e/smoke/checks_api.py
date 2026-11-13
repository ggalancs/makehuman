"""End-to-end checks for the plugin API (mhapi) and the socket server."""

import array
import glob
import json
import os
import socket
import threading
import time

import numpy as np

import getpath
from core import G

from . import SOCKET_PORT, expect, pump


def _api():
    return G.app.mhapi


def check_mhapi_basics(smoke):
    api = _api()
    expect(api.internals.getHuman() is G.app.selectedHuman and api.internals.getApp() is G.app, "internals")
    expect(api.internals.numpyTypecodeToPythonTypeCode("f") in ("f", "d"), "typecode f")
    expect(api.locations.getUnicodeAbsPath(".") == os.path.abspath("."), "abs path")
    expect(os.path.isdir(api.locations.getInstallationPath()), "installation path")
    expect(api.locations.getSystemDataPath("targets").endswith("targets"), "system data path")
    expect(api.locations.getUserHomePath("models").endswith("models"), "user home path")
    expect(api.locations.getUserDataPath("clothes").endswith(os.path.join("data", "clothes")), "user data path")
    v = api.version
    expect([int(x) for x in v.getVersionNumberAsArray()][:2] == [int(x) for x in v.getVersionNumberAsString().split(".")][:2],
           "version %s vs %s" % (v.getVersionNumberAsArray(), v.getVersionNumberAsString()))
    expect(isinstance(v.getFullVersion(), str) and v.getFullVersion(), "full version %r" % v.getFullVersion())
    expect(v.getBranch() is None or isinstance(v.getBranch(), str), "branch")
    for f in (v.getRevision, v.getRevisionId, v.getRevisionNumber):
        f()
    u = api.utility
    expect([u.getTypeAsString(x) for x in (1, 1.5, {}, [], "s", b"b")] ==
           ["int", "float", "dict", "list", "unicode", "bytes"], "getTypeAsString")
    expect(u.getTypeAsString(None) == str(type(None)) and u.getTypeAsString(array.array("f")) == "array(f)", "unlisted types")
    values = [u.getValueAsString(x) for x in ([1], "s", b"ok", b"\xff\xfe", 1.23456789, {"a": 1})]
    expect(values == ["[]", "s", "ok", "fffe", "1.2346", "{'a': 1}"], "getValueAsString %s" % values)
    expect(u.isPython3() and u.isPyQtAvailable() in (True, False) and u.isPySideAvailable() in (True, False), "python")
    expect(callable(getattr(u.getCompatibleUrlFetcher(), "urlopen", None)), "url fetcher")
    channel = u.getLogChannel("smoke", defaultLevel=5, mirrorToMHLog=True)
    for level in ("crash", "error", "warn", "info", "debug", "trace", "spam"):
        if level not in ("crash", "error"):
            getattr(channel, level)("smoke %s" % level, {"x": 1})
    u.resetDebugWriter("smoke")
    for content in ([1.5, "txt", b"raw"], "text", 7, b"bytes"):
        u.debugWrite(content, "smoke", "check_mhapi_basics")
    written = os.listdir(os.path.join(api.locations.getUserHomePath("debugWriter"), "smoke"))
    expect(len([f for f in written if f.startswith("bin-")]) == 4 and "textualContent.txt" in written, "debugWrite %s" % written)
    # NameSpace tracing prints where API calls come from
    api.mesh.tracing = True
    try:
        coords = api.mesh.getVertexCoordinates()
    finally:
        api.mesh.tracing = False
    expect(coords.shape[1] == 3 and len(coords) > 10000, "vertex coordinates")
    return {"version": v.getVersionNumberAsString()}


def check_mhapi_ui(smoke):
    ui = _api().ui
    expect("Modelling" in [c.name for c in ui.getAllCategories()], "categories")
    expect(len(ui.getAllTaskViews()) > 40, "task views")
    expect(ui.getTaskView("Files", "Export") is G.app.getTask("Files", "Export"), "getTaskView")
    expect(ui.getTaskView("Files", "Nope") is None and ui.getTaskView("Nope", "Nope") is None, "missing task view")
    widgets = [ui.createWidget(), ui.createGroupBox("g"), ui.createSlider(0.3, label="s"), ui.createButton("b"),
               ui.createCheckBox("c", selected=True), ui.createLabel("l"), ui.createTextEdit("t")]
    expect(all(w is not None for w in widgets) and widgets[4].selected, "widgets")
    changes = []
    combo = ui.createComboBox(["a", "b", "c"], onChange=lambda item: changes.append(item))
    expect(combo.rowCount() == 3, "combo rows")
    combo.setCurrentRow(1)
    expect(combo.getCurrentItem() == "b", "combo current row")
    combo.setCurrentItem("c")
    expect(combo.getCurrentItem() == "c", "combo current item")
    combo.setOnChange(lambda item: changes.append("again"))
    combo.setData(["x"])
    expect(combo.rowCount() == 1, "combo setData")
    lst = ui.createList(["one", "two"])
    expect(lst.count() == 2, "list")
    tab = ui.createTab(G.app.getCategory("Utilities"), "smoke_tab", "Smoke tab")
    expect(tab is not None, "tab")


def check_mhapi_assets(smoke, tmp=None):
    assets = _api().assets
    expect("material" in assets.getAssetTypes(), "asset types")
    expect(assets.assetTitleToDirName(" My__Hat (2)/x ") == "My_Hat_2x", "title to dir: %s" % assets.assetTitleToDirName(" My__Hat (2)/x "))
    for kind in ("clothes", "hair", "material", "model", "skin", "proxy", "target"):
        expect(assets.getAssetLocation("x y", kind), "location for %s" % kind)
    try:
        assets.getAssetLocation("x", "bogus")
        raise AssertionError("bogus asset type accepted")
    except ValueError:
        pass
    found = {}
    for name in ("Skins", "Hair", "Eyebrows", "Eyelashes", "Clothes"):
        system = getattr(assets, "getAvailableSystem" + name)()
        user = getattr(assets, "getAvailableUser" + name)()
        expect(system, "no system %s" % name)
        found[name] = len(system) + len(user)
    def stem(path):
        return os.path.splitext(os.path.basename(path or ""))[0]

    def other(available, equipped):
        """An asset that is not worn now: selecting the worn one again would toggle it off."""
        return next(a for a in available if stem(a) != stem(equipped))

    for kind in ("Hair", "Eyebrows", "Eyelashes"):
        pick = other(getattr(assets, "getAvailableSystem" + kind)(), getattr(assets, "getEquipped" + kind)())
        getattr(assets, "equip" + kind)(pick)
        expect(stem(getattr(assets, "getEquipped" + kind)()) == stem(pick), "equip%s" % kind)
        getattr(assets, "unequip" + kind)(pick)
    assets.unequipAllClothes()  # earlier checks may have dressed the model
    clothes = assets.getAvailableSystemClothes()[:2]
    for c in clothes:
        assets.equipClothes(c)
    expect(sorted(assets.getEquippedClothes()) == sorted(clothes), "equipClothes")
    assets.unequipClothes(clothes[0])
    expect(assets.getEquippedClothes() == [clothes[1]], "unequipClothes")
    assets.unequipAllClothes()
    expect(assets.getEquippedClothes() == [], "unequipAllClothes")

    # read and write asset files (on copies)
    out = smoke.out("assets", "x")[:-2]
    for path in (sorted(glob.glob(getpath.getSysDataPath("clothes/*/*.mhclo")))[0], assets.getAvailableSystemSkins()[0],
                 glob.glob(getpath.getSysDataPath("proxymeshes/*/*.proxy"))[0]):
        info = assets.openAssetFile(path)
        expect(info and os.path.samefile(info["absolute path"], path), "openAssetFile %s" % path)
        stripped = assets.openAssetFile(path, strip=True)
        copy = os.path.join(out, os.path.basename(path))
        with open(path, "rb") as src, open(copy, "wb") as dst:
            dst.write(src.read())
        info["absolute path"] = copy
        info["name"] = "smoke copy"
        expect(assets.writeAssetFile(info, createBackup=True) is not False, "writeAssetFile")
        expect("smoke copy" in open(copy, encoding="utf-8").read(), "written name")
        expect(stripped is not None, "strip")
    expect(assets.openAssetFile(os.path.join(out, "missing.mhclo")) is None, "missing asset")
    material_hash = assets.materialToHash(G.app.selectedHuman.material)
    expect(isinstance(material_hash, dict) and material_hash, "materialToHash")
    return found


def check_mhapi_modifiers_skeleton_exports(smoke):
    api = _api()
    m = api.modifiers
    names = m.getAvailableModifierNames()
    expect("macrodetails/Gender" in names, "modifier names")
    m.setGender(0.9)
    m.setAge(0.6)
    m.setMuscle(0.4)
    m.setWeight(0.6)
    m.setHeight(0.4)
    human = G.app.selectedHuman
    expect(abs(human.getGender() - 0.9) < 1e-6 and abs(human.getMuscle() - 0.4) < 1e-6, "macro setters")
    m.applyModifier("nose/nose-scale-horiz-decr|incr", 0.3)
    expect(abs(human.getModifier("nose/nose-scale-horiz-decr|incr").getValue() - 0.3) < 1e-6, "applyModifier")
    m.applyTarget("nose/nose-scale-vert-incr", 0.5)  # relative to data/targets, without extension
    expect(any("nose-scale-vert-incr" in t for t in m.getAppliedTargets()), "applyTarget")
    m.applyModifier("armslegs/r-hand-scale-decr|incr", 0.5)
    m.applySymmetryLeft()
    m.applySymmetryRight()

    s = api.skeleton
    G.app.getTask("Pose/Animate", "Skeleton").chooseSkeleton(getpath.getSysDataPath("rigs/default.mhskel"))
    expect(s.getSkeleton() is not None and s.getBaseSkeleton() is not None, "skeletons")
    G.app.getTask("Pose/Animate", "Pose").loadPose(getpath.getSysDataPath("poses/benchmark.bvh"))
    expect(s.getPoseAsBoneDict() is None, "getPoseAsBoneDict is an unimplemented stub upstream")
    bvh_text = s.getPoseAsBVH()
    expect(bvh_text is not None, "pose as BVH")
    expect(s.getPoseAsAnimation() is not None, "pose as animation")
    s.setExpressionFromFile(getpath.getSysDataPath("expressions/smile01.mhpose"))
    s.setPoseFromFile(getpath.getSysDataPath("poses/benchmark.bvh"))
    s.clearPoseAndExpression()

    e = api.exports
    expect(len(e.getExportFormats()) >= 6 and e.getOBJExporter() and e.getFBXExporter() and e.getDAEExporter(), "exporters")
    expect(e.getMHX2Exporter() is None, "no MHX2 exporter is bundled")
    out = smoke.out("api_exports", "x")[:-2]
    for name, fn in (("api.obj", e.exportAsOBJ), ("api.fbx", e.exportAsFBX), ("api.dae", e.exportAsDAE)):
        fn(os.path.join(out, name), useExportsDir=False)
        expect(os.path.getsize(os.path.join(out, name)) > 0, "%s not written" % name)
    e.exportAsOBJ("api_in_exports.obj")  # default: the user's exports folder
    expect(os.path.isfile(os.path.join(getpath.getPath("exports"), "api_in_exports.obj")), "export to exports dir")
    return {"modifiers": len(names)}


def _ask(function, params=None, binary=False):
    """One request per connection, as the socket plugin expects. Runs off the GUI thread."""
    box = {}

    def go():
        with socket.create_connection(("127.0.0.1", SOCKET_PORT), timeout=20) as s:
            s.sendall(json.dumps({"function": function, "params": params or {}, "error": "", "data": ""}).encode())
            chunks = []
            while True:
                b = s.recv(1 << 20)
                if not b:
                    break
                chunks.append(b)
        raw = b"".join(chunks)
        box["value"] = raw if binary else json.loads(raw.decode())

    t = threading.Thread(target=go, daemon=True)
    t.start()
    deadline = time.time() + 30
    while t.is_alive() and time.time() < deadline:
        pump(0.02)  # the server answers through a queued signal on the GUI thread
    expect("value" in box, "no answer to %s" % function)
    return box["value"]


def check_socket_all_functions(smoke):
    expect(SOCKET_PORT, "socket server not configured")
    human = G.app.selectedHuman
    G.app.getTask("Pose/Animate", "Skeleton").chooseSkeleton(getpath.getSysDataPath("rigs/default.mhskel"))
    shirt = sorted(glob.glob(getpath.getSysDataPath("clothes/*/*.mhclo")))[0]
    G.app.getTask("Geometries", "Clothes").proxyFileSelected(shirt)
    pump(0.1)

    info = _ask("getBodyMeshInfo")["data"]
    nverts = len(human.meshData.coord)
    verts = np.frombuffer(_ask("getBodyVerticesBinary", binary=True), dtype=np.float32)
    expect(verts.size == nverts * 3, "body vertices: %d floats for %d vertices" % (verts.size, nverts))
    faces = _ask("getBodyFacesBinary", binary=True)
    expect(len(faces) == human.meshData.fvert.nbytes, "body faces bytes")
    expect(len(_ask("getBodyTextureCoordsBinary", binary=True)) == human.meshData.texco.nbytes, "texture coords")
    expect(len(_ask("getBodyFaceUVMappingsBinary", binary=True)) == human.meshData.fuvs.nbytes, "uv mappings")
    expect(_ask("getBodyMaterialInfo")["data"], "material info")
    expect(_ask("getCoord")["data"], "getCoord")
    weights_info = _ask("getBodyWeightInfo")["data"]
    expect(weights_info, "weight info")
    expect(len(_ask("getBodyWeightsVertList", binary=True)) > 0, "weights vert list")
    expect(len(_ask("getBodyWeights", binary=True)) > 0, "weights")
    skeleton = _ask("getSkeleton")["data"]
    expect(skeleton and "bones" in json.dumps(skeleton), "skeleton")
    pose = _ask("getPose")["data"]
    expect(len(pose) > 50, "pose bones")
    expect(len(_ask("getPose", {"poseFilename": getpath.getSysDataPath("expressions/smile01.mhpose")})["data"]) > 50, "pose with expression")

    proxies = _ask("getProxiesInfo")["data"]
    expect(proxies, "proxies info")
    for p in proxies:
        uuid = p["uuid"]
        for fn in ("getProxyVerticesBinary", "getProxyFacesBinary", "getProxyTextureCoordsBinary",
                   "getProxyFaceUVMappingsBinary", "getProxyWeightsVertList", "getProxyWeights"):
            expect(len(_ask(fn, {"uuid": uuid}, binary=True)) > 0, "%s for %s" % (fn, p.get("name")))
        expect(_ask("getProxyMaterialInfo", {"uuid": uuid})["data"], "proxy material")
        expect(_ask("getProxyWeightInfo", {"uuid": uuid})["data"], "proxy weight info")

    names = _ask("getAvailableModifierNames")["data"]
    expect("macrodetails/Gender" in names, "modifier names over socket")
    _ask("applyModifier", {"modifier": "macrodetails/Gender", "power": 0.2})
    pump(0.2)
    expect(abs(human.getGender() - 0.2) < 1e-6, "applyModifier over socket")
    expect(_ask("getAppliedTargets")["data"], "applied targets")
    expect(_ask("getUserDir")["data"] and _ask("getSysDir")["data"], "dirs")
    expect("Unknown command" in _ask("noSuchFunction")["error"], "unknown command")
    G.app.getTask("Geometries", "Clothes").resetSelection()
    return {"proxies": len(proxies), "vertices": nverts, "mesh_info_keys": sorted(info)[:6]}


CHECKS = [
    ("mhapi_basics", check_mhapi_basics),
    ("mhapi_ui", check_mhapi_ui),
    ("mhapi_assets", check_mhapi_assets),
    ("mhapi_modifiers_skeleton_exports", check_mhapi_modifiers_skeleton_exports),
    ("socket_all_functions", check_socket_all_functions),
]
