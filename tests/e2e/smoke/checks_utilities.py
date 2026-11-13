"""End-to-end checks for the Utilities category (Scripting, Shell, ...)."""

import json
import os

from core import G
import mh

from . import expect, pump


def _scripting():
    return G.app.modules["7_scripting"]


def check_scripting_examples(smoke):
    """Every entry of the Examples list appends a documented snippet to the script."""
    view = _scripting().scriptingView
    view.scriptText.setText("")
    names = [view.listView.item(i).text for i in range(view.listView.count())]
    grew = []
    for row, name in enumerate(names):
        before = len(view.scriptText.getText())
        view.listView.setCurrentRow(row)
        view.insertButton.callEvent("onClicked", None)
        text = view.scriptText.getText()
        call = "MHScript." + name.split("(")[0]
        if len(text) > before and call in text[before:]:
            grew.append(name)
    missing = sorted(set(names) - set(grew))
    expect(not missing, "examples that append nothing (or the wrong call): %s" % missing)
    return {"examples": len(names)}


API_SCRIPT = '''
import json, os
out = {out!r}
r = {{}}
MHScript.setAge(0.3)
MHScript.setWeight(0.7)
r["height_cm"] = MHScript.getHeightCm()
MHScript.applyTarget({target!r}, 0.8)
r["target_applied"] = any({target!r} in t for t in MHScript.human.targetsDetailStack)
MHScript.printDetailStack()
MHScript.getModelingParameters()
MHScript.updateModelingParameter("macrodetails/Gender", 1.0)
MHScript.updateModelingParameters({{"macrodetails/Gender": 0.25, "macrodetails-universal/Muscle": 0.75}})
r["gender"] = MHScript.human.getGender()
r["muscle"] = MHScript.human.getMuscle()
MHScript.setHeadSquareness(0.5)
r["head_square"] = MHScript.human.getModifier("head/head-square").getValue()
MHScript.setMaterial({skin!r})
r["skin"] = MHScript.human.material.filename
for axis in "XYZ":
    getattr(MHScript, "setPosition" + axis)(1.5)
    getattr(MHScript, "modifyPosition" + axis)(0.5)
    r["pos" + axis] = getattr(MHScript, "getPosition" + axis)()
    getattr(MHScript, "setRotation" + axis)(10)
    getattr(MHScript, "modifyRotation" + axis)(5)
    r["rot" + axis] = getattr(MHScript, "getRotation" + axis)()
MHScript.printPositionInfo()
MHScript.printRotationInfo()
MHScript.printCameraInfo()
for axis in "XYZ":
    getattr(MHScript, "setPosition" + axis)(0)
    getattr(MHScript, "setRotation" + axis)(0)
MHScript.setZoom(1.5)
MHScript.modifyZoom(0.5)
r["zoom"] = MHScript.getZoom()
MHScript.setZoom(1.0)
r["names"] = [MHScript.incrementingFilename("shot"), MHScript.incrementingFilename("shot", suffix=".jpg", width=2)]
MHScript.saveModel("scripted", out)
MHScript.loadModel("scripted", out)
MHScript.saveObj("scripted", out)
MHScript.screenShot(os.path.join(out, "scripted.png"))
with open(os.path.join(out, "script_results.json"), "w") as f:
    json.dump(r, f, default=float)
'''


def check_scripting_api(smoke):
    """Run a script through the Execute tab that exercises every MHScript call."""
    out = smoke.out("scripting", "x")[:-2]
    target = "nose/nose-scale-horiz-incr"
    skin = mh.getSysDataPath("skins/young_caucasian_female/young_caucasian_female.mhmat")
    view = _scripting().scriptingView
    view.scriptText.setText(API_SCRIPT.format(out=out, target=target, skin=skin))
    modals = len(smoke.modals)
    G.app.getTask("Utilities", "Execute").execButton.callEvent("onClicked", None)
    pump(0.5)
    dialogs = smoke.modals[modals:]
    expect(any("Good job" in d for d in dialogs), "script did not report success: %s" % dialogs)
    with open(os.path.join(out, "script_results.json")) as f:
        r = json.load(f)
    expect(r["target_applied"], "applyTarget did not add the target")
    expect(abs(r["gender"] - 0.25) < 1e-6 and abs(r["muscle"] - 0.75) < 1e-6, "updateModelingParameters: %s" % r)
    expect(abs(r["head_square"] - 0.5) < 1e-6, "setHeadSquareness")
    expect(os.path.samefile(r["skin"], skin), "setMaterial")
    for axis in "XYZ":
        expect(abs(r["pos" + axis] - 2.0) < 1e-6, "position %s = %s" % (axis, r["pos" + axis]))
    # Scripting's Y rotation is the mesh's Z axis, which MakeHuman deliberately keeps at 0
    expect(abs(r["rotX"] - 15.0) < 1e-6 and abs(r["rotZ"] - 15.0) < 1e-6 and r["rotY"] == 0,
           "rotations %s" % {k: r[k] for k in ("rotX", "rotY", "rotZ")})
    expect(r["names"] == ["shot0001.png", "shot02.jpg"], "incrementingFilename: %s" % r["names"])
    for f in ("scripted.mhm", "scripted.obj", "scripted.png"):
        expect(os.path.getsize(os.path.join(out, f)) > 0, "%s not written" % f)
    expect(r["height_cm"] > 100, "getHeightCm")
    return {k: r[k] for k in ("height_cm", "zoom", "names")}


def check_scripting_errors_and_files(smoke):
    """A failing script is reported, and scripts can be saved and loaded again."""
    view = _scripting().scriptingView
    modals = len(smoke.modals)
    view.scriptText.setText("raise ValueError('expected failure')")
    smoke.allow_logged("expected failure")
    G.app.getTask("Utilities", "Execute").execButton.callEvent("onClicked", None)
    pump(0.5)
    expect(any("expected failure" in d for d in smoke.modals[modals:]), "error not reported: %s" % smoke.modals[modals:])

    path = smoke.out("scripting", "saved_script.py")
    view.scriptText.setText("x = 1  # saved")
    view.saveButton.callEvent("onClicked", path)
    pump(0.4)
    with open(path) as f:
        expect(f.read() == "x = 1  # saved", "script file content")
    view.scriptText.setText("")
    view.loadButton.callEvent("onClicked", path)
    pump(0.4)
    expect(view.scriptText.getText() == "x = 1  # saved", "loaded script text")
    expect(view.filename == path, "remembered file name")
    modals = len(smoke.modals)
    view.loadButton.callEvent("onClicked", path + ".missing")
    pump(0.4)
    expect(any("does not exist" in d for d in smoke.modals[modals:]), "missing file not reported")
    view.loadButton.callEvent("onClicked", "")  # cancelled dialog: nothing happens
    view.saveButton.callEvent("onClicked", "")
    mh.changeTask("Utilities", "Scripting")
    mh.changeTask("Modelling", "Main")


def check_scripting_canvas_size(smoke):
    """The Execute tab can read and fix the canvas size, and rejects bad input."""
    tab = G.app.getTask("Utilities", "Execute")
    tab.getButton.callEvent("onClicked", None)
    expect(tab.widthEdit.getText() == str(G.windowWidth), "Get button did not show the width")
    for width, height, message in (("abc", "300", "valid integers"), ("300", "x", "valid integers"),
                                   ("50", "300", "at least 100")):
        tab.widthEdit.setText(width)
        tab.heightEdit.setText(height)
        modals = len(smoke.modals)
        tab.setButton.callEvent("onClicked", None)
        pump(0.4)
        expect(any(message in d for d in smoke.modals[modals:]), "no error for %sx%s" % (width, height))
    central = G.app.mainwin.centralWidget()
    tab.widthEdit.setText("640")
    tab.heightEdit.setText("480")
    tab.setButton.callEvent("onClicked", None)
    pump(0.5)
    size = (G.windowWidth, G.windowHeight)
    central.setMinimumSize(0, 0)
    central.setMaximumSize(16777215, 16777215)
    pump(0.3)
    expect(size == (643, 483), "canvas is %s, expected 643x483 (requested size + 3 px border)" % (size,))
    return {"canvas": size}


CHECKS = [
    ("scripting_examples", check_scripting_examples),
    ("scripting_api", check_scripting_api),
    ("scripting_errors_and_files", check_scripting_errors_and_files),
    ("scripting_canvas_size", check_scripting_canvas_size),
]
