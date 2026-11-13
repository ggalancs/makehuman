"""End-to-end checks for animation playback and every exporter option."""

import glob
import hashlib
import os

import numpy as np

import getpath
import mh
from core import G

from . import expect, pump


def _exporter(name):
    for exporter, _radio, _options in G.app.getTask("Files", "Export").formats:
        if exporter.name == name:
            return exporter
    raise AssertionError("no exporter %s" % name)


def _export(smoke, exporter, variant, ext):
    folder = smoke.out("export_options", variant, "x")[:-2]
    exporter.export(G.app.selectedHuman, lambda e, folder=folder: os.path.join(folder, "model." + e.lstrip(".")))
    pump(0.05)
    path = os.path.join(folder, "model." + ext)
    expect(os.path.isfile(path) and os.path.getsize(path) > 0, "%s: %s not written" % (variant, path))
    return path


def _set_scale(exporter, name):
    """The unit radio buttons belong to the Export task, shared by all exporters."""
    for button, button_name in G.app.getTask("Files", "Export").scaleButtons:
        button.setSelected(button_name == name)


def _make_animation(smoke, frames=12):
    """A multi-frame BVH for the default rig (the bundled animations use another rig): the
    benchmark pose, blended in from the rest pose."""
    text = open(getpath.getSysDataPath("poses/benchmark.bvh")).read()
    head, motion = text.split("MOTION")
    lines = [l for l in motion.strip().splitlines() if l.strip()]
    values = np.array([float(v) for v in lines[-1].split()])
    rows = []
    for i in range(frames):
        frame = values * (i / float(frames - 1))
        frame[:3] = values[:3]  # keep the root position
        rows.append(" ".join("%.6f" % v for v in frame))
    path = smoke.out("animation", "blend.bvh")
    with open(path, "w") as f:
        f.write(head + "MOTION\nFrames: %d\nFrame Time: 0.041667\n%s\n" % (frames, "\n".join(rows)))
    return path


def check_animation_playback(smoke):
    human = G.app.selectedHuman
    G.app.getTask("Pose/Animate", "Skeleton").chooseSkeleton(getpath.getSysDataPath("rigs/default.mhskel"))
    walk = _make_animation(smoke)
    # an expression left active by earlier checks would be merged into a one-frame pose
    G.app.getTask("Pose/Animate", "Expressions").chooseExpression(None)
    G.app.getTask("Pose/Animate", "Pose").loadPose(None)
    G.app.getTask("Pose/Animate", "Pose").loadPose(walk)
    mh.changeTask("Pose/Animate", "Animations")
    tv = G.app.getTask("Pose/Animate", "Animations")
    pump(0.1)
    expect(tv.nFrames > 10, "walk animation has %s frames" % tv.nFrames)
    first = human.meshData.coord.copy()
    tv.btn4.callEvent("onClicked", None)
    expect(tv.currentframe == tv.nFrames - 1, "last frame button")
    last = human.meshData.coord.copy()
    expect(np.abs(last - first).max() > 0.05, "frames do not move the mesh")
    tv.btn2.callEvent("onClicked", None)
    expect(tv.currentframe == tv.nFrames - 2, "one frame back")
    tv.btn3.callEvent("onClicked", None)
    expect(tv.currentframe == tv.nFrames - 1, "one frame forward")
    tv.btn1.callEvent("onClicked", None)
    expect(tv.currentframe == 0, "first frame button")
    tv.playbackSlider.callEvent("onChanging", 5)
    tv.playbackSlider.callEvent("onChange", 5)
    expect(tv.currentframe == 5, "slider")
    mh.changeTask("Modelling", "Main")
    return {"frames": tv.nFrames}


def check_export_options(smoke):
    """Each option changes the output the way it says (with the walk animation loaded)."""
    report = {}

    fbx = _exporter("Filmbox (fbx)")
    for binary in (True, False):
        fbx.binary.setSelected(binary)
        fbx.hiddenGeom.setSelected(not binary)
        head = open(_export(smoke, fbx, "fbx_binary_%s" % binary, "fbx"), "rb").read(20)
        expect(head.startswith(b"Kaydara FBX Binary") == binary, "FBX binary=%s starts with %r" % (binary, head))
    fbx.binary.setSelected(True)
    fbx.hiddenGeom.setSelected(False)

    stl = _exporter("Stereolithography (stl)")
    for binary in (True, False):
        stl.stlBinary.setSelected(binary)
        head = open(_export(smoke, stl, "stl_binary_%s" % binary, "stl"), "rb").read(5)
        expect((head == b"solid") != binary, "STL binary=%s starts with %r" % (binary, head))
    stl.stlBinary.setSelected(False)

    obj = _exporter("Wavefront obj")
    sizes = {}
    for normals, hidden, scale in ((False, False, "decimeter"), (True, False, "meter"), (False, True, "decimeter")):
        obj.useNormals.setSelected(normals)
        obj.hiddenGeom.setSelected(hidden)
        _set_scale(obj, scale)
        text = open(_export(smoke, obj, "obj_%s_%s_%s" % (normals, hidden, scale), "obj")).read()
        verts = [list(map(float, l.split()[1:4])) for l in text.splitlines() if l.startswith("v ")]
        expect(("\nvn " in text) == normals, "OBJ normals=%s" % normals)
        sizes[(normals, hidden, scale)] = (len(verts), np.ptp(np.array(verts)[:, 1]))
    expect(sizes[(False, True, "decimeter")][0] > sizes[(False, False, "decimeter")][0], "helper geometry adds vertices")
    ratio = sizes[(True, False, "meter")][1] / sizes[(False, False, "decimeter")][1]
    expect(abs(ratio - 0.1) < 0.01, "meters vs decimeters height ratio %.3f" % ratio)
    obj.useNormals.setSelected(False)
    obj.hiddenGeom.setSelected(False)
    _set_scale(obj, "decimeter")
    report["obj_vertices"] = {str(k): v[0] for k, v in sizes.items()}

    dae = _exporter("Collada (dae)")
    orientations = [dae.yUpFaceZ, dae.yUpFaceX, dae.zUpFaceNegY, dae.zUpFaceX]
    systems = [dae.localY, dae.localX, dae.localG]
    outputs = set()
    for i, orient in enumerate(orientations):
        orient.setSelected(True)
        systems[i % 3].setSelected(True)
        dae.facePoseUnits.setSelected(i == 1)
        dae.hiddenGeom.setSelected(i == 2)
        text = open(_export(smoke, dae, "dae_%d" % i, "dae"), encoding="utf-8").read()
        # Collada writes animations (the pose plus the face units) only with Facial pose-units on
        expect(("<animation" in text) == (i == 1), "Collada animations in variant %d: %s" % (i, "<animation" in text))
        outputs.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
    expect(len(outputs) == 4, "Collada variants produced identical files")
    dae.yUpFaceZ.setSelected(True)
    dae.localY.setSelected(True)
    dae.facePoseUnits.setSelected(False)
    dae.hiddenGeom.setSelected(False)

    bvh = _exporter("Biovision Hierarchy BVH")
    texts = []
    for feet in (True, False):
        bvh.feetOnGround.setSelected(feet)
        texts.append(open(_export(smoke, bvh, "bvh_feet_%s" % feet, "bvh")).read())
        expect("MOTION" in texts[-1] and "Frames:" in texts[-1], "BVH motion section")
    expect(texts[0] != texts[1], "feet on ground made no difference in BVH")
    bvh.feetOnGround.setSelected(True)

    ogre = _exporter("Ogre3D")
    for feet in (True, False):
        ogre.feetOnGround.setSelected(feet)
        _export(smoke, ogre, "ogre_feet_%s" % feet, "mesh.xml")
    ogre.feetOnGround.setSelected(True)
    skeletons = glob.glob(smoke.out("export_options", "ogre_feet_True", "x")[:-2] + "/*.skeleton.xml")
    expect(skeletons, "Ogre skeleton not exported")
    G.app.getTask("Pose/Animate", "Pose").loadPose(None)
    return report


CHECKS = [
    ("animation_playback", check_animation_playback),
    ("export_options", check_export_options),
]
