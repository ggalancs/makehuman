"""End-to-end checks: sub-surface scattering render, material editor,
expression mixer, target explorer, measurements and settings."""

import json
import os

import numpy as np

import getpath
import mh
from core import G
from PyQt5 import QtWidgets

from . import expect, pump


def _find_task(name):
    for category in G.app.categories.values():
        for task in category.tasks:
            if task.name == name:
                return category.name, task
    raise AssertionError("no task %s" % name)


def _arrow_keys(qslider, up=True, presses=5):
    """What the right (or left) arrow key does to a focused slider
    (QAbstractSlider::keyPressEvent): a single step that emits valueChanged without a press
    or a release. Real key events would need the window to be active, and the app runs in
    the background."""
    step = QtWidgets.QAbstractSlider.SliderSingleStepAdd if up else QtWidgets.QAbstractSlider.SliderSingleStepSub
    for _ in range(presses):
        qslider.triggerAction(step)


def check_render_subsurface_scattering(smoke):
    mh2opengl = G.app.modules["4_rendering_opengl"].mh2opengl
    G.app.getTask("Materials", "Material").filechooser.callEvent(
        "onFileSelected", getpath.getSysDataPath("skins/young_caucasian_male/young_caucasian_male.mhmat"))
    viewer = G.app.getTask("Rendering", "Viewer")
    for aa in (False, True):
        mh2opengl.Render({"scene": G.app.scene, "AA": aa, "dimensions": (200, 300), "lightmapSSS": True})
        pump(0.2)
        pm = viewer.image.imageLabel.pixmap()
        expect(pm is not None and (pm.width(), pm.height()) == (200, 300), "SSS render AA=%s" % aa)
    expect(G.app.selectedHuman.material.diffuseTexture is not None, "material restored after SSS render")
    mh.changeTask("Modelling", "Main")


def check_material_editor(smoke):
    mh.changeTask("Utilities", "Material Editor")
    tv = G.app.getTask("Utilities", "Material Editor")
    pump(0.1)
    shaders = [tv.shaderList.item(i) for i in range(tv.shaderList.count())]
    expect(len(shaders) >= 3, "shaders: %d" % len(shaders))
    for item in shaders:
        tv.shaderList.callEvent("onClicked", item)
        pump(0.05)
    path = smoke.out("material_editor", "edited")
    tv.saveMaterialBtn.callEvent("onClicked", path)
    expect(os.path.isfile(path + ".mhmat"), "material not saved")
    text = open(path + ".mhmat").read()
    expect("shader " in text and "diffuseColor" in text, "saved material content")
    tv.loadMaterialBtn.callEvent("onClicked", getpath.getSysDataPath("skins/default.mhmat"))
    pump(0.1)
    expect(os.path.basename(G.app.selectedHuman.material.filename) == "default.mhmat", "material editor load")
    mh.changeTask("Modelling", "Main")
    return {"shaders": len(shaders)}


def check_expression_mixer(smoke):
    mh.changeTask("Utilities", "Expression mixer")
    tv = G.app.getTask("Utilities", "Expression mixer")
    pump(0.1)
    names = list(tv.modifiers)[:3]
    expect(len(names) == 3, "pose units: %s" % list(tv.modifiers)[:5])
    for i, name in enumerate(names):
        tv.modifiers[name] = 0.3 * (i + 1)
    tv.updatePose()
    path = smoke.out("expression_mixer", "mixed")
    tv.saveBtn.callEvent("onClicked", path)
    data = json.load(open(path + ".mhpose"))
    expect(set(names) <= set(data.get("unit_poses", {})), "saved pose units: %s" % data)
    tv.resetBtn.callEvent("onClicked", None)
    expect(all(v == 0 for v in tv.modifiers.values()), "reset")
    tv.loadBtn.callEvent("onClicked", path + ".mhpose")
    expect(abs(tv.modifiers[names[2]] - 0.9) < 1e-6, "loaded pose units")
    tv.resetBtn.callEvent("onClicked", None)
    mh.changeTask("Modelling", "Main")
    return {"units": len(tv.modifiers)}


def check_target_explorer_and_measure(smoke):
    human = G.app.selectedHuman
    human.getModifier("nose/nose-scale-horiz-decr|incr").setValue(0.5)
    human.applyAllTargets()
    category, tv = _find_task("Targets")
    mh.changeTask(category, "Targets")
    pump(0.1)
    tv.clear.callEvent("onClicked", None)
    pump(0.1)
    # an ordinary modelling slider (no task-specific handler) follows the keyboard too
    mh.changeTask("Modelling", "Torso")
    pump(0.1)
    torso = G.app.getTask("Modelling", "Torso").sliders[0]
    torso.update()  # earlier checks may have left this modifier at any value
    start, entries = torso.modifier.getValue(), len(G.app.undoStack)
    _arrow_keys(torso.slider, up=start < 0.5)
    pump(0.3)
    expect(abs(torso.modifier.getValue() - torso.getValue()) < 1e-6 and abs(torso.getValue() - start) > 1e-3,
           "%s from %.3f: slider shows %.3f, model has %.3f"
           % (torso.modifier.fullName, start, torso.getValue(), torso.modifier.getValue()))
    expect(len(G.app.undoStack) > entries, "keyboard change not undoable")
    G.app.undo()
    pump(0.3)
    # the model comes back exactly, the slider to its nearest step
    one_step = (torso.max - torso.min) / torso.slider.maximum()
    expect(abs(torso.modifier.getValue() - start) < 1e-6 and abs(torso.getValue() - start) <= one_step,
           "undo of a keyboard change: model %.5f, slider %.5f, expected %.5f"
           % (torso.modifier.getValue(), torso.getValue(), start))

    mh.changeTask("Modelling", "Measure")
    measure = G.app.getTask("Modelling", "Measure")
    waist = "measure/measure-waist-circ-decr|incr"
    slider = next(s for s in measure.sliders if s.modifier.fullName == waist)
    commits = []
    real_commit = slider._onChange
    slider._onChange = lambda: (commits.append(1), real_commit())

    def user_does(action):
        """Waist after the action, undo entries and model updates it caused."""
        entries, applied = len(G.app.undoStack), len(commits)
        action()
        pump(0.3)
        return measure.getMeasure(waist), len(G.app.undoStack) - entries, len(commits) - applied

    # earlier checks may have left the waist anywhere: move away from where it is
    start = slider.modifier.getValue()
    typed_value = start + 0.3 if start <= 0.4 else start - 0.3
    dragged_value = start + 0.5 if start <= 0.4 else start - 0.5

    def type_value():
        slider.edit.setText("%.2f" % slider.toDisplay(typed_value))
        slider._enter()

    def drag():
        slider.slider.setSliderDown(True)
        slider.slider.setSliderPosition(slider._f2i(dragged_value))
        slider.slider.setSliderDown(False)

    try:
        before = measure.getMeasure(waist)
        typed, entries, applied = user_does(type_value)
        expect(abs(typed - before) > 1 and abs(slider.modifier.getValue() - typed_value) < 0.01,
               "typed value %.2f not applied: model %.3f, waist %.2f -> %.2f"
               % (typed_value, slider.modifier.getValue(), before, typed))
        expect((entries, applied) == (1, 1), "typed value: %d undo entries, %d updates" % (entries, applied))
        G.app.undo()
        pump(0.3)
        expect(abs(measure.getMeasure(waist) - before) < 0.01, "undo of a typed value")

        dragged, entries, applied = user_does(drag)
        expect(abs(dragged - before) > abs(typed - before), "drag: waist %.2f -> %.2f" % (before, dragged))
        expect((entries, applied) == (1, 1), "drag: %d undo entries, %d updates" % (entries, applied))

        keyed, entries, applied = user_does(lambda: _arrow_keys(slider.slider, up=dragged_value < 0.5))
        expect(abs(slider.modifier.getValue() - slider.getValue()) < 1e-6,
               "arrow keys: slider shows %.3f, model has %.3f" % (slider.getValue(), slider.modifier.getValue()))
        expect(abs(keyed - dragged) > 0.01 and entries >= 1, "arrow keys: waist %.2f, %d undo entries" % (keyed, entries))
    finally:
        del slider._onChange
        slider.modifier.setValue(start)
        human.applyAllTargets()
        slider.update()
    expect(abs(measure.getMeasure(waist) - before) < 0.01, "waist not restored")
    G.app.setSetting("units", "imperial")
    measure.syncSliders()
    G.app.setSetting("units", "metric")
    human.getModifier("nose/nose-scale-horiz-decr|incr").setValue(0)
    human.applyAllTargets()
    mh.changeTask("Modelling", "Main")
    return {"waist_cm": [round(v, 2) for v in (before, typed, dragged, keyed)]}


def check_settings_tabs(smoke):
    """Toggle every check box and move every slider of the Settings tabs, then restore them."""
    from qtgui import CheckBox, Slider
    changed = 0
    for task in list(G.app.getCategory("Settings").tasks):
        mh.changeTask("Settings", task.name)
        pump(0.05)
        # task views are not QWidgets: take the visible widgets of the window once the tab is shown
        for box in G.app.mainwin.findChildren(CheckBox):
            if not (box.isVisible() and box.isEnabled()):
                continue
            original = box.selected
            for state in (not original, original):
                box.setSelected(state)
                box.callEvent("onClicked", None)
            changed += 1
        for slider in [w for w in G.app.mainwin.findChildren(Slider) if w.isVisible()]:
            value = slider.getValue()
            slider.callEvent("onChange", value)
            changed += 1
    expect(changed > 15, "only %d settings exercised" % changed)
    mh.changeTask("Modelling", "Main")
    return {"widgets": changed}


def check_special_foot_pose(smoke):
    """A garment can ask for a foot pose (e.g. high heels): wearing it mixes that pose into
    the feet of the current pose, taking it off restores the pose that was there."""
    import shutil
    human = G.app.selectedHuman
    source = os.path.abspath(getpath.getSysDataPath("clothes/female_casualsuit01/female_casualsuit01.mhclo"))
    folder = getpath.getDataPath("clothes/smoke_heels")
    os.makedirs(folder, exist_ok=True)
    lines = []
    for line in open(source, encoding="utf-8"):
        key = line.split(" ", 1)[0]
        if key == "name":
            line = "name smoke_heels\n"
        elif key == "uuid":
            line = "uuid 0f6c3c7e-5b8a-4f7e-9f62-6d1e0c5a9b01\n"
        elif key in ("obj_file", "material"):  # the meshes and material of the original
            line = "%s %s\n" % (key, os.path.join(os.path.dirname(source), line.split()[1]))
        lines.append(line)
    lines.append("special_pose foot smoke_heels.bvh\n")
    garment = os.path.join(folder, "smoke_heels.mhclo")
    with open(garment, "w", encoding="utf-8") as f:
        f.writelines(lines)
    shutil.copy(getpath.getSysDataPath("poses/fight01.bvh"), getpath.getDataPath("special_poses/foot/smoke_heels.bvh"))

    G.app.getTask("Pose/Animate", "Skeleton").chooseSkeleton(getpath.getSysDataPath("rigs/default.mhskel"))
    G.app.getTask("Pose/Animate", "Expressions").chooseExpression(None)
    G.app.getTask("Pose/Animate", "Pose").loadPose(getpath.getSysDataPath("poses/benchmark.bvh"))
    pump(0.1)
    base = human.getActiveAnimation().name
    skel = human.getBaseSkeleton()  # poses are applied to it

    def pose_of(*bones):
        return [skel.getBone(b).matPose.copy() for b in bones]

    # the garment's foot pose, loaded on its own, for reference
    G.app.getTask("Pose/Animate", "Pose").loadPose(getpath.getSysDataPath("poses/fight01.bvh"))
    pump(0.1)
    foot_pose_feet = pose_of("foot.L", "toe1-1.R")
    G.app.getTask("Pose/Animate", "Pose").loadPose(getpath.getSysDataPath("poses/benchmark.bvh"))
    pump(0.1)
    feet_before, arm_before = pose_of("foot.L", "toe1-1.R"), pose_of("upperarm01.L")
    expect(any(not np.allclose(a, b) for a, b in zip(foot_pose_feet, feet_before)),
           "the two poses have the same feet: this check could not see the mixing")

    _category, clothes = _find_task("Clothes")
    clothes.proxyFileSelected(garment)
    pump(0.2)
    expect(human.getActiveAnimation().name == "special-foot-pose", "foot pose not applied: %s" % human.getActiveAnimation().name)
    feet, arm = pose_of("foot.L", "toe1-1.R"), pose_of("upperarm01.L")
    expect(all(np.allclose(a, b, atol=1e-5) for a, b in zip(feet, foot_pose_feet)), "the feet do not follow the garment's pose")
    expect(np.allclose(arm[0], arm_before[0]), "the foot pose moved an arm")

    # another pose keeps the feet of the garment
    G.app.getTask("Pose/Animate", "Pose").loadPose(getpath.getSysDataPath("poses/fight02.bvh"))
    pump(0.1)
    expect(human.getActiveAnimation().name == "special-foot-pose", "foot pose lost after a pose change")

    clothes.proxyFileDeselected(garment)
    pump(0.2)
    expect(human.getActiveAnimation().name != "special-foot-pose" and not human.hasAnimation("special-foot-pose"),
           "foot pose still active after taking the garment off: %s" % human.getActiveAnimation().name)
    G.app.getTask("Pose/Animate", "Pose").loadPose(None)
    return {"base_pose": base}


def check_custom_targets(smoke):
    """A target file put in the custom folder gets a slider after a rescan; deleting the
    file and rescanning removes the slider and its effect on the model."""
    import shutil
    human = G.app.selectedHuman
    category, tv = _find_task("Custom")
    mh.changeTask(category, "Custom")
    pump(0.1)
    rescan = next(w for w in tv.optionsBox.children if getattr(w, "getLabel", lambda: "")() == "Rescan targets' folder")
    folder = os.path.join(tv.targetsPath, "smoke")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, "wide-nose.target")
    shutil.copy(getpath.getSysDataPath("targets/nose/nose-scale-horiz-incr.target"), target)
    human.applyAllTargets()
    rest = human.meshData.coord.copy()

    rescan.callEvent("onClicked", None)
    pump(0.1)
    expect(len(tv.sliders) == 1 and len(tv.modifiers) == 1, "custom sliders: %d" % len(tv.sliders))
    slider = tv.sliders[0]
    slider.edit.setText("%.2f" % slider.toDisplay(1.0))
    slider._enter()
    pump(0.3)
    moved = np.abs(human.meshData.coord - rest).max()
    expect(moved > 0.01, "custom target did not change the mesh (%.4f)" % moved)

    tv.keepValues.setSelected(True)
    rescan.callEvent("onClicked", None)  # the value survives a rescan
    pump(0.1)
    expect(abs(list(tv.modifiers.values())[0].getValue() - 1.0) < 1e-6, "value lost on rescan")
    tv.keepValues.setSelected(False)

    os.remove(target)
    rescan.callEvent("onClicked", None)
    pump(0.2)
    expect(not tv.sliders and not tv.modifiers, "slider of a deleted target still there")
    back = np.abs(human.meshData.coord - rest).max()
    expect(back < 1e-5, "deleting the custom target left the model changed (%.5f)" % back)
    mh.changeTask("Modelling", "Main")
    return {"moved": round(float(moved), 4)}


def check_install_user_plugins(smoke):
    """Settings > User Plugins: install a zipped plugin and a plugin file (the file dialog
    answers with them), activate the zipped one, and refuse to install it twice."""
    import zipfile
    from PyQt5 import QtWidgets
    tv = G.app.getTask("Settings", "User Plugins")
    mh.changeTask("Settings", "User Plugins")
    pump(0.1)
    code = "import gui3d\n\ndef load(app):\n    app.%s = True\n\ndef unload(app):\n    pass\n"
    archive = smoke.out("user_plugins", "zipped.zip")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("9_smoke_zipped/__init__.py", code % "smoke_zipped_loaded")
    single = smoke.out("user_plugins", "9_smoke_file.py")
    with open(single, "w") as f:
        f.write(code % "smoke_file_loaded")
    plugins = getpath.getPath("plugins")
    answers = []
    real_dialog = QtWidgets.QFileDialog.getOpenFileName
    QtWidgets.QFileDialog.getOpenFileName = lambda *args, **kwargs: (answers.pop(0), "")
    try:
        answers.append(archive)
        tv.installZipButton.callEvent("onClicked", None)
        pump(0.1)
        expect(os.path.isfile(os.path.join(plugins, "9_smoke_zipped", "__init__.py")), "zipped plugin not unpacked")
        boxes = {box.name: box for box in tv.userPluginBox.children}
        expect("9_smoke_zipped" in boxes, "installed plugin not listed: %s" % sorted(boxes))

        boxes["9_smoke_zipped"].setSelected(True)
        tv.activateButton.callEvent("onClicked", None)
        pump(0.1)
        expect(getattr(G.app, "smoke_zipped_loaded", False) and "9_smoke_zipped" in G.app.modules,
               "activated plugin was not loaded")

        answers.append(archive)  # again: its files exist, nothing is overwritten
        tv.installZipButton.callEvent("onClicked", None)
        pump(0.1)

        answers.append(single)
        tv.installPyButton.callEvent("onClicked", None)
        pump(0.1)
        expect(os.path.isfile(os.path.join(plugins, "9_smoke_file.py")), "plugin file not copied")
        tv.reloadButton.callEvent("onClicked", None)
        expect("9_smoke_file" in [box.name for box in tv.userPluginBox.children], "plugin file not listed")
    finally:
        QtWidgets.QFileDialog.getOpenFileName = real_dialog
    mh.changeTask("Modelling", "Main")
    return {"listed": sorted(box.name for box in tv.userPluginBox.children)}


CHECKS = [
    ("render_subsurface_scattering", check_render_subsurface_scattering),
    ("material_editor", check_material_editor),
    ("expression_mixer", check_expression_mixer),
    ("target_explorer_and_measure", check_target_explorer_and_measure),
    ("settings_tabs", check_settings_tabs),
    ("special_foot_pose", check_special_foot_pose),
    ("custom_targets", check_custom_targets),
    ("install_user_plugins", check_install_user_plugins),
]
