"""End-to-end checks for tool plugins: shell, save targets, background, scene editor,
skeleton debugging and mass production."""

import contextlib
import glob
import os

import getpath
import mh
from core import G
from PyQt5 import QtGui

from . import expect, pump


@contextlib.contextmanager
def file_dialogs(open_path=None, save_path=None):
    """Answer MakeHuman's native file dialogs, which would block an unattended run."""
    saved = mh.getOpenFileName, mh.getSaveFileName
    mh.getOpenFileName = lambda *a, **k: (open_path or "", "")
    mh.getSaveFileName = lambda *a, **k: (save_path or "", "")
    try:
        yield
    finally:
        mh.getOpenFileName, mh.getSaveFileName = saved


def check_shell(smoke):
    tv = G.app.getTask("Utilities", "Shell")
    expect(tv.console is None, "the IPython console is not installed, the plain shell is used")
    tv.clearText()
    tv.line.setText("x = 6 * 7")
    tv.line.callEvent("onActivate", "x = 6 * 7")
    tv.line.callEvent("onActivate", "print(x)")
    expect("42" in tv.text.getText(), "shell output: %r" % tv.text.getText())
    tv.line.callEvent("onActivate", "1/0")
    expect("ZeroDivisionError" in tv.text.getText(), "shell traceback")
    expect(tv.getSuggestions("G.ap")[:1] == ["G.app"], "completion: %s" % tv.getSuggestions("G.ap"))
    tv.line.setText("G.app.selectedHuman.getHe")
    tv.line.callEvent("onTabPressed", tv.line)
    expect(tv.line.text.startswith("G.app.selectedHuman.getHeight"), "tab completion: %r" % tv.line.text)
    tv.line.callEvent("onUpArrow", None)
    expect(tv.line.text == "1/0", "history up: %r" % tv.line.text)
    tv.line.callEvent("onUpArrow", None)
    tv.line.callEvent("onDownArrow", None)
    expect(tv.line.text == "1/0", "history down: %r" % tv.line.text)
    tv.line.callEvent("onDownArrow", None)
    expect(tv.line.text == "", "history end")
    tv.clear.callEvent("onClicked", None)
    expect(tv.text.getText() == "", "clear")
    tv.onThemeChanged(None)


def check_save_targets(smoke):
    tv = G.app.getTask("Utilities", "Save Targets")
    human = G.app.selectedHuman
    out = smoke.out("targets", "x")[:-2]
    modals = len(smoke.modals)
    smoke.allow_logged("Not a .target file")
    tv.saveAsButton.callEvent("onClicked", (os.path.join(out, "wrong.obj"), "Object (*.obj)"))
    pump(0.4)
    expect(any("Expected a path to a .target file" in m for m in smoke.modals[modals:]), "bad extension accepted")

    human.getModifier("nose/nose-scale-horiz-decr|incr").setValue(0.6)
    human.applyAllTargets()
    target = os.path.join(out, "my_shape")
    tv.saveAsButton.callEvent("onClicked", (target, "Target (*.target)"))
    pump(0.2)
    expect(os.path.isfile(target + ".target"), "Save As did not write my_shape.target")
    lines = [l for l in open(target + ".target") if l.strip() and not l.startswith("#")]
    expect(len(lines) > 50, "target has %d vertex lines" % len(lines))
    expect(tv.nameEdit.getText() == "my_shape.target", "name field: %r" % tv.nameEdit.getText())
    tv.saveButton.callEvent("onClicked", None)  # quick save to the same name
    expect(os.path.getsize(target + ".target") > 0, "quick save")

    modals = len(smoke.modals)
    if os.path.isfile(tv.metaFile):
        os.remove(tv.metaFile)
    tv.saveDiffButton.callEvent("onClicked", None)
    pump(0.4)
    expect(any("No Base Model" in m for m in smoke.modals[modals:]), "diff without base")
    tv.setBaseButton.callEvent("onClicked", None)
    expect(os.path.isfile(tv.metaFile), "Set Base wrote no base model")
    human.getModifier("nose/nose-scale-horiz-decr|incr").setValue(-0.6)
    human.applyAllTargets()
    tv.diffDirName = out
    tv.diffFileName = "diff.target"
    tv.saveDiffButton.callEvent("onClicked", None)
    pump(0.2)
    diff = [l for l in open(os.path.join(out, "diff.target")) if l.strip() and not l.startswith("#")]
    expect(0 < len(diff) < len(lines) + 1000, "diff target lines: %d" % len(diff))
    tv.saveDiffAsButton.callEvent("onClicked", (os.path.join(out, "diff_as"), "Target (*.target)"))
    expect(os.path.isfile(os.path.join(out, "diff_as.target")), "diff Save As")
    tv.clearButton.callEvent("onClicked", None)
    human.getModifier("nose/nose-scale-horiz-decr|incr").setValue(0)
    human.applyAllTargets()
    return {"target_lines": len(lines), "diff_lines": len(diff)}


def check_background(smoke):
    tv = G.app.getTask("Settings", "Background")
    # only backgrounds inside the backgrounds folder are stored in MHM files
    image = os.path.join(getpath.getPath("backgrounds"), "smoke_grid.png")
    os.makedirs(os.path.dirname(image), exist_ok=True)
    img = QtGui.QImage(400, 200, QtGui.QImage.Format_RGB32)
    img.fill(0xff3366cc)
    expect(img.save(image), "could not write test image")
    mh.changeTask("Settings", "Background")
    G.app.actions.front.trigger()
    tv.setSelectedSideCheckbox("front")
    tv.filechooser.callEvent("onFileSelected", image)
    pump(0.2)
    expect(tv.isBackgroundSet() and tv.isBackgroundShowing(), "background not shown")
    expect(tv.getCurrentBackground()[0] == image and abs(tv.getCurrentBackground()[1] - 2.0) < 1e-6, "aspect")
    tv.opacitySlider.callEvent("onChanging", 30)
    tv.opacitySlider.callEvent("onChange", 30)
    expect(tv.opacity == 30, "opacity")
    tv.setBackgroundScale(2.0)
    expect(abs(tv.getBackgroundScale() - 2.0) < 1e-6, "scale")
    tv.moveBackground(0.5, 0.25)
    tv.setBackgroundPosition(1.0, 2.0)
    tv.foregroundTggl.setSelected(True)
    tv.foregroundTggl.callEvent("onClicked", None)
    expect(tv.isShowBgInFront(), "foreground")
    tv.dragButton.setSelected(True)
    tv.dragButton.callEvent("onClicked", None)
    tv.dragButton.setSelected(False)
    tv.dragButton.callEvent("onClicked", None)
    for radio in tv.radioButtonGroup:
        radio.setSelected(True)
        radio.callEvent("onClicked", None)
    tv.setSelectedSideCheckbox("front")
    G.app.actions.front.trigger()

    # saved with the model and restored from it
    import guisave
    model = smoke.out("background", "with_background.mhm")
    guisave.saveMHM(model)
    text = open(model).read()
    expect("background front" in text, "background not saved in the MHM")
    G.app.undo()  # the BackgroundAction
    G.app.loadHumanMHM(model)
    pump(0.2)
    expect(tv.filenames["front"] and tv.filenames["front"][0].endswith("smoke_grid.png"), "background not loaded back")
    tv.toggleBackground()
    tv.toggleBackground()
    tv.filechooser.callEvent("onFileSelected", None)
    tv.foregroundTggl.setSelected(False)
    tv.foregroundTggl.callEvent("onClicked", None)
    mh.changeTask("Modelling", "Main")


def check_scene_editor(smoke):
    tv = G.app.getTask("Utilities", "Scene Editor")
    mh.changeTask("Utilities", "Scene Editor")
    scene = tv.scene
    lights = len(scene.lights)
    tv.adder.lightbtn.callEvent("onClicked", None)
    pump(0.1)
    expect(len(scene.lights) == lights + 1, "Add light")
    for row in range(tv.itemList.count()):
        tv.itemList.callEvent("onClicked", tv.itemList.item(row))
        props = tv.itemList.item(row).getUserData()
        for name in ("colbox", "posbox", "focbox", "specbox", "fov", "att", "soft", "size", "samples"):
            widget = getattr(props, name, None)
            if widget is not None:
                widget.callEvent("onChange", widget.getValue())
    path = smoke.out("scenes", "smoke.mhscene")
    with file_dialogs(save_path=path):
        tv.saveButton.callEvent("onClicked", None)  # no file yet: Save As
    expect(os.path.isfile(path), "scene not saved")
    tv.saveButton.callEvent("onClicked", None)      # saves to the same file
    last = tv.itemList.item(tv.itemList.count() - 1).getUserData()
    last.removebtn.callEvent("onClicked", None)
    expect(len(scene.lights) == lights, "Remove light")
    smoke.modal_accept = True  # "Your scene is unsaved. Are you sure?" -> Close
    with file_dialogs(open_path=path):
        tv.loadButton.callEvent("onClicked", None)
        pump(0.5)
    expect(len(scene.lights) == lights + 1, "scene loaded back: %d lights" % len(scene.lights))
    tv.adder.lightbtn.callEvent("onClicked", None)
    tv.closeButton.callEvent("onClicked", None)  # modified: confirmation, then reset
    pump(0.5)
    smoke.modal_accept = False
    tv.addButton.callEvent("onClicked", None)
    mh.changeTask("Modelling", "Main")
    return {"lights": len(scene.lights)}


def check_skeleton_debug(smoke):
    G.app.getTask("Pose/Animate", "Skeleton").chooseSkeleton(getpath.getSysDataPath("rigs/default.mhskel"))
    tasks = [t.name for t in G.app.getCategory("Utilities").tasks]
    debug = [t for t in G.app.getCategory("Utilities").tasks if "keleton" in t.name]
    expect(debug, "no skeleton debug task in %s" % tasks)
    tv = debug[0]
    mh.changeTask("Utilities", tv.name)
    pump(0.1)
    for toggle in (tv.showBonesTggl, tv.showAxisTggl, tv.showPlanesTggl, tv.showWeightsTggl):
        toggle.setSelected(True)
        toggle.callEvent("onClicked", None)
        pump(0.05)
    bones = list(tv.boneSelector) if hasattr(tv, "boneSelector") else []
    for radio in bones[:5]:
        radio.setSelected(True)
        radio.callEvent("onClicked", None)
    tv.removeBoneHighlights()
    for toggle in (tv.showBonesTggl, tv.showAxisTggl, tv.showPlanesTggl, tv.showWeightsTggl):
        toggle.setSelected(False)
        toggle.callEvent("onClicked", None)
    mh.changeTask("Modelling", "Main")
    return {"task": tv.name, "bones_clicked": len(bones[:5])}


def check_mass_produce(smoke):
    tv = G.app.getTask("Community", "Mass produce")
    s = tv.randomizationSettings
    s.getUI("output", "numfiles").setText("2")
    for name in ("upperClothes", "lowerClothes"):
        s.getUI("proxies", name).setSelected(True)
    models = getpath.getPath("models")
    produced = {}
    for fmt, folder, ext in (("MHM", models, ".mhm"), ("OBJ", getpath.getPath("exports"), ".obj")):
        s.getUI("output", "fileformat").setCurrentItem(fmt)
        s.getUI("output", "fnbase").setText("mass_" + fmt.lower() + "_")
        tv.produceButton.callEvent("onClicked", None)
        pump(0.2)
        files = sorted(glob.glob(os.path.join(folder, "mass_%s_*%s" % (fmt.lower(), ext))))
        expect(len(files) == 2, "%s: produced %s" % (fmt, files))
        produced[fmt] = [os.path.basename(f) for f in files]
    a, b = (open(f).read() for f in sorted(glob.glob(os.path.join(models, "mass_mhm_*.mhm"))))
    expect(a != b, "the two produced characters are identical")
    return produced


CHECKS = [
    ("shell", check_shell),
    ("save_targets", check_save_targets),
    ("background", check_background),
    ("scene_editor", check_scene_editor),
    ("skeleton_debug", check_skeleton_debug),
    ("mass_produce", check_mass_produce),
]
