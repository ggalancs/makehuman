"""End-to-end checks for the main window: camera navigation, actions, settings, i18n,
shortcuts, load/save handlers and scenes."""

import os

import events3d
import mh
from core import G

from . import expect, pump


def _camera_state():
    cam = G.app.modelCamera
    return (round(cam.horizontalRotation, 3), round(cam.verticalInclination, 3),
            round(cam.zoomFactor, 4), tuple(round(v, 4) for v in cam.translation))


CAMERA_PRESETS = ["setFaceCamera", "setLeftHandFrontCamera", "setLeftHandTopCamera", "setRightHandFrontCamera",
                  "setRightHandTopCamera", "setLeftFootFrontCamera", "setLeftFootLeftCamera",
                  "setRightFootFrontCamera", "setRightFootRightCamera", "setLeftArmFrontCamera",
                  "setLeftArmTopCamera", "setRightArmFrontCamera", "setRightArmTopCamera",
                  "setLeftLegFrontCamera", "setLeftLegLeftCamera", "setRightLegFrontCamera",
                  "setRightLegRightCamera"]


def check_camera_presets(smoke):
    """Every body-part camera ends in a distinct view; the global camera goes back."""
    G.app.setGlobalCamera()
    pump(0.4)
    home = _camera_state()
    states = {}
    for name in CAMERA_PRESETS:
        getattr(G.app, name)()
        pump(0.4)  # the camera moves through a short animation
        states[name] = _camera_state()
        expect(states[name] != home, "%s left the camera where it was" % name)
    expect(len(set(states.values())) >= len(CAMERA_PRESETS) - 2, "presets are not distinct: %s" % states)
    G.app.setGlobalCamera()
    pump(0.4)
    expect(_camera_state()[2] == home[2], "global camera zoom %s vs %s" % (_camera_state(), home))
    return {"presets": len(states)}


def check_camera_keys_and_mouse(smoke):
    G.app.setGlobalCamera()
    pump(0.4)
    moves = {}
    for name in ("rotateLeft", "rotateRight", "rotateUp", "rotateDown", "panUp", "panDown",
                 "panLeft", "panRight", "zoomIn", "zoomOut"):
        before = _camera_state()
        getattr(G.app, name)()
        moves[name] = before != _camera_state()
    expect(all(moves.values()), "camera keys without effect: %s" % [k for k, v in moves.items() if not v])
    for name in ("frontView", "backView", "leftView", "rightView", "topView", "bottomView", "resetView"):
        getattr(G.app, name)()
        pump(0.1)
    G.app.setGlobalCamera()
    pump(0.4)

    # Drag with each mouse button and use the wheel, entering where the canvas does
    x, y = G.windowWidth // 2, G.windowHeight // 3
    effects = {}
    for button in (mh.Buttons.LEFT_MASK, mh.Buttons.RIGHT_MASK, mh.Buttons.MIDDLE_MASK):
        before = _camera_state()
        G.canvas.makeCurrent()
        G.app.callEvent("onMouseDownCallback", events3d.MouseEvent(button, x, y))
        for step in range(5):
            G.app.callEvent("onMouseMovedCallback", events3d.MouseEvent(button, x + 8 * (step + 1), y + 4 * (step + 1), 8, 4))
        G.app.callEvent("onMouseUpCallback", events3d.MouseEvent(button, x + 40, y + 20))
        pump(0.1)
        effects[button] = before != _camera_state()
    expect(all(effects.values()), "mouse drags without effect: %s" % effects)
    for invert in (False, True):
        G.app.setSetting("invertMouseWheel", invert)
        for delta in (1, -1):
            before = _camera_state()
            G.app.callEvent("onMouseWheelCallback", events3d.MouseWheelEvent(delta, x, y))
            expect(before != _camera_state(), "wheel %s (inverted %s) did nothing" % (delta, invert))
    G.app.setSetting("invertMouseWheel", False)
    G.app.callEvent("onMouseMovedCallback", events3d.MouseEvent(0, x, y, 1, 1))  # hover only
    G.app.setGlobalCamera()
    pump(0.4)
    return {"keys": len(moves)}


def check_main_actions(smoke):
    human = G.app.selectedHuman
    for go, category in (("goToSave", "Files"), ("goToLoad", "Files"), ("goToExport", "Files"),
                         ("goToRendering", "Rendering"), ("goToHelp", "Help"), ("goToModelling", "Modelling")):
        getattr(G.app, go)()
        expect(G.app.currentCategory.name == category, "%s -> %s" % (go, G.app.currentCategory.name))
    for action in ("grid", "pose", "symmetry"):
        checkable = getattr(G.app.actions, action)
        for state in (not checkable.isChecked(), checkable.isChecked()):
            checkable.trigger()
            pump(0.05)
    mod = human.getModifier("armslegs/r-hand-scale-decr|incr")
    mirror = human.getModifier(mod.getSymmetricOpposite())
    mod.setValue(0.6)
    mirror.setValue(0.0)
    G.app.symmetryLeft()  # copies the left side onto the right... or the other way: one of them
    G.app.symmetryRight()
    expect(abs(mod.getValue() - mirror.getValue()) < 1e-6, "symmetry did not equalise %s/%s" % (mod.getValue(), mirror.getValue()))
    mod.setValue(0.0)
    mirror.setValue(0.0)
    human.applyAllTargets()

    # Save: without a file it goes to the Save tab; with one it saves in place
    G.app.currentFile.closed()
    G.app.doSave()
    expect(G.app.currentTask.name == "Save", "doSave without a file went to %s" % G.app.currentTask.name)
    import guisave
    path = smoke.out("main", "saved.mhm")
    guisave.saveMHM(path)
    human.getModifier("macrodetails/Gender").setValue(0.2)
    human.applyAllTargets()
    G.app.currentFile.changed()
    mtime = os.path.getmtime(path)
    pump(1.1)
    G.app.doSave()
    expect(os.path.getmtime(path) > mtime, "doSave did not save in place")
    G.app.updateFilenameCaption()
    expect("saved" in G.app.mainwin.windowTitle(), "caption: %s" % G.app.mainwin.windowTitle())
    mh.changeTask("Modelling", "Main")

    G.app.status("smoke %s", "status")
    G.app.statusPersist("smoke persistent")
    G.app.statusPersist("")
    G.app.progress(0.5, "half")
    G.app.progress(1.0)
    G.app.about()
    G.app.credits()
    pump(0.6)


def check_themes_languages_settings(smoke):
    styles = G.app.getLookAndFeelStyles()
    expect(styles, "no Qt styles")
    current = G.app.getLookAndFeel()
    G.app.setLookAndFeel(styles[0])
    G.app.setLookAndFeel(current)
    for theme in ("default", "makehuman"):
        G.app.setTheme(theme)
        pump(0.1)
    G.app.reloadIcons()
    expect(os.path.isfile(G.app.getThemeResource("images", "splash.png")), "theme resource")
    languages = G.app.getLanguages()
    expect(len(languages) > 5, "languages: %s" % languages)
    other = [l for l in languages if l != "english"][0]
    G.app.setLanguage(other)
    translated = G.app.getLanguageString("Modelling")
    G.app.setLanguage("english")
    expect(G.app.getLanguageString("Modelling") == "Modelling", "english string")
    expect(isinstance(translated, str) and translated, "translation in %s" % other)
    G.app.dumpMissingStrings()

    G.app.addSetting("smoke_setting", 3)
    expect(G.app.getSetting("smoke_setting") == 3 and G.app.getSettingDefault("smoke_setting") == 3, "addSetting")
    G.app.setSetting("smoke_setting", 4)
    expect(G.app.getSetting("smoke_setting") == 4, "setSetting")
    G.app.saveSettings()
    expect(os.path.isfile(os.path.join(mh.getPath(), "settings.ini")), "settings.ini not written")

    G.app.setShortcut(mh.Modifiers.CTRL, mh.Keys.F1, G.app.actions.front)
    expect(G.app.getShortcut(G.app.actions.front) == (mh.Modifiers.CTRL, mh.Keys.F1), "setShortcut")
    G.app.resetShortcuts()
    G.app.setMouseAction(mh.Modifiers.SHIFT, mh.Buttons.LEFT_MASK, G.app.mouseZoom)
    expect(G.app.getMouseAction(G.app.mouseZoom) == (mh.Modifiers.SHIFT, mh.Buttons.LEFT_MASK), "setMouseAction")
    G.app.setMouseAction(mh.Modifiers.SHIFT, mh.Buttons.LEFT_MASK, G.app.mouseRotate)
    return {"styles": len(styles), "languages": len(languages)}


def check_load_save_handlers_and_scene(smoke):
    seen = []
    G.app.addLoadHandler("smoketest", lambda human, values, strict: seen.append(values))
    expect(G.app.getLoadHandler("smoketest") is not None, "load handler")
    G.app.addSaveHandler(lambda human, f: f.write("smoketest hello world\n"), priority=99)
    import guisave
    path = smoke.out("main", "handlers.mhm")
    guisave.saveMHM(path)
    expect("smoketest hello world" in open(path).read(), "save handler output missing")
    G.app.loadHumanMHM(path)
    # handlers also get status lines around the load
    expect(["smoketest", "hello", "world"] in seen and ["status", "finished"] in seen, "load handler got %s" % seen)

    from scene import Scene
    original = G.app.getScene()
    scene = Scene()
    G.app.setScene(scene)
    expect(G.app.getScene() is scene, "setScene")
    scene.addLight()
    pump(0.1)
    G.app.setScene(original)
    return {"handler_calls": len(seen)}


CHECKS = [
    ("camera_presets", check_camera_presets),
    ("camera_keys_and_mouse", check_camera_keys_and_mouse),
    ("main_actions", check_main_actions),
    ("themes_languages_settings", check_themes_languages_settings),
    ("load_save_handlers_and_scene", check_load_save_handlers_and_scene),
]
