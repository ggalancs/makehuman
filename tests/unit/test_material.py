"""shared/material.py: .mhmat parsing and writing, shader configuration, textures, Color."""

import logging
import os
import sys
import types

import numpy as np
import pytest

import files3d
import getpath
import human
import image
import material
import progress
from core import G

MH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "makehuman")
SKIN_TEX = "data/skins/young_caucasian_female/young_lightskinned_female_diffuse.png"


@pytest.fixture(autouse=True)
def no_opengl_shaders(monkeypatch):
    """Unit tests have no OpenGL context. If another test file imported the real shader
    module, materials would try to compile GLSL (material.getShaderObj) and segfault."""
    shader = sys.modules.get("shader")
    if shader is not None and hasattr(shader, "Shader"):
        monkeypatch.setattr(shader.Shader, "_supported", False)



class FakeApp(object):
    def __init__(self, selectedHuman=None):
        self.selectedHuman = selectedHuman

    def progress(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def mh_env(monkeypatch):
    monkeypatch.chdir(MH_DIR)  # shader and texture paths resolve relative to makehuman/
    monkeypatch.setattr(progress, "current_Progress_", None)
    monkeypatch.setattr(G, "app", FakeApp(), raising=False)
    monkeypatch.setattr(material, "_autoSkinBlender", None)


def png(path, colour=0xff336699):
    img = image.Image(data=np.full((2, 2, 4), 200, dtype=np.uint8))
    img.save(str(path))
    return str(path)


# --- Color -------------------------------------------------------------------

def test_color_construction_and_clamping():
    c = material.Color(0.2, 1.5, -1)
    assert c.values == [0.2, 1.0, 0.0] and c.asTuple() == (0.2, 1.0, 0.0)
    assert material.Color([0.1, 0.2, 0.3]).values == [0.1, 0.2, 0.3]  # copy constructor
    c.setValues(0.3, 0.4, 0.5)
    assert (c.r, c.g, c.b) == (0.3, 0.4, 0.5) and c[1] == 0.4 and list(c) == [0.3, 0.4, 0.5]
    c.r = 7
    assert c.r == 1.0
    assert repr(material.Color(0.5, 0.25, 0)) == "Color(0.5 0.25 0.0)"
    assert material.Color(0.5, 0.25, 0).asStr() == "0.5 0.25 0.0"
    d = c.clone()
    assert d == c and d is not c


def test_color_values_property_setter():
    c = material.Color()
    c.values = (0.3, 0.4, 0.5)
    assert c.values == [0.3, 0.4, 0.5]


def test_color_comparisons():
    a, b = material.Color(0.1, 0.2, 0.3), material.Color(0.1, 0.2, 0.4)
    assert a < b and a <= b and b > a and b >= a and a != b and not (a == b)
    assert a == material.Color(0.1, 0.2, 0.3)


def test_color_arithmetic_clamps():
    a = material.Color(0.2, 0.4, 0.6)
    assert (a + (0.1, 0.1, 0.9)).values == pytest.approx([0.3, 0.5, 1.0])
    assert ((0.1, 0.1, 0.1) + a).values == pytest.approx([0.3, 0.5, 0.7])
    assert (a - (0.1, 0.5, 0.1)).values == pytest.approx([0.1, 0.0, 0.5])
    assert ((1, 1, 1) - a).values == pytest.approx([0.8, 0.6, 0.4])
    assert (a * 2).values == pytest.approx([0.4, 0.8, 1.0])
    assert (a * (0.5, 0.5, 2)).values == pytest.approx([0.1, 0.2, 1.0])
    assert (0.5 * a).values == pytest.approx([0.1, 0.2, 0.3])
    assert ((2, 0, 1) * a).values == pytest.approx([0.4, 0.0, 0.6])
    assert (a / 2).values == pytest.approx([0.1, 0.2, 0.3])
    assert (a / (2, 4, 6)).values == pytest.approx([0.1, 0.1, 0.1])
    assert (0.1 / a).values == pytest.approx([0.5, 0.25, 0.1 / 0.6])
    assert ((0.1, 0.1, 0.3) / a).values == pytest.approx([0.5, 0.25, 0.5])


def test_color_true_division():
    assert (material.Color(0.2, 0.4, 0.6) / 2).values == pytest.approx([0.1, 0.2, 0.3])


def test_intensity_helpers():
    assert material.getIntensity((1, 1, 1)) == pytest.approx(1.0)
    m = material.Material()
    m.diffuseColor = (1, 0, 0)
    m.specularColor = (0, 0, 1)
    assert m.diffuseIntensity == pytest.approx(0.2126)
    assert m.specularIntensity == pytest.approx(0.0722)
    assert material.isNumeric("1e3") and not material.isNumeric("abc") and not material.isNumeric(None)


# --- Parsing -----------------------------------------------------------------

FULL_MHMAT = """\
# a comment
// another comment
name Full
tag Shiny
tag Metal Things
description A full material
ambientColor 0.1 0.2 0.3
diffuseColor 0.4 0.5 0.6
diffuseIntensity 0.5
specularColor 0.7 0.8 0.9
specularIntensity 0.5
shininess 1.5
emissiveColor 0.01 0.02 0.03
viewPortColor 0.3 0.3 0.3
viewPortAlpha 0.5
opacity 0.75
translucency -1
shadeless true
wireframe yes
transparent enabled
alphaToCoverage false
backfaceCull False
depthless true
castShadows false
receiveShadows no
autoBlendSkin false
diffuseTexture diffuse.png
bumpmapTexture bump.png
bumpmapIntensity 0.3
normalmapTexture normal.png
normalmapIntensity 2
displacementmapTexture disp.png
displacementmapIntensity 0.4
specularmapTexture spec.png
specularmapIntensity 0.6
transparencymapTexture alpha.png
transparencymapIntensity 0.7
aomapTexture ao.png
aomapIntensity 0.8
sssEnabled true
sssRScale 4.0
sssGScale -2
sssBScale 1.0
shader data/shaders/glsl/litsphere
uvMap data/uvs/a7.mhuv
shaderParam gain 0.5
shaderParam tint 0.1 0.2 0.3
shaderParam lookup lut.png
shaderDefine CUSTOM_B
shaderDefine CUSTOM_A
shaderConfig diffuse false
shaderConfig bump false
shaderConfig normal true
shaderConfig displacement false
shaderConfig spec false
shaderConfig vertexColors false
shaderConfig transparency false
shaderConfig ambientOcclusion false
shaderConfig sparkles true
"""

TEXTURES = ["diffuse.png", "bump.png", "normal.png", "disp.png", "spec.png", "alpha.png", "ao.png", "lut.png"]


@pytest.fixture
def full(tmp_path):
    folder = tmp_path / "mat"
    folder.mkdir()
    for name in TEXTURES:
        png(folder / name)
    path = folder / "full.mhmat"
    path.write_text(FULL_MHMAT, encoding="utf-8")
    return str(path)


def test_parse_every_property(full, caplog):
    m = material.fromFile(full)
    folder = os.path.dirname(full)
    assert m.name == "Full" and m.description == "A full material"
    assert m.tags == {"shiny", "metal things"}
    assert m.filename == os.path.normpath(full) and m.filepath == folder
    assert m.ambientColor.values == pytest.approx([0.1, 0.2, 0.3])
    assert m.diffuseColor.values == pytest.approx([0.4, 0.5, 0.6])
    assert m.specularColor.values == pytest.approx([0.7, 0.8, 0.9])
    assert m.emissiveColor.values == pytest.approx([0.01, 0.02, 0.03])
    assert m.usesViewPortColor() and m.viewPortColor.values == pytest.approx([0.3] * 3) and m.viewPortAlpha == 0.5
    assert (m.shininess, m.opacity, m.translucency) == (1.0, 0.75, 0.0)  # clamped to 0..1
    assert (m.shadeless, m.wireframe, m.transparent, m.alphaToCoverage) == (True, True, True, False)
    assert (m.backfaceCull, m.depthless, m.castShadows, m.receiveShadows) == (False, True, False, False)
    assert m.autoBlendSkin is False
    for attr, name in [("diffuseTexture", "diffuse"), ("bumpMapTexture", "bump"), ("normalMapTexture", "normal"),
                       ("displacementMapTexture", "disp"), ("specularMapTexture", "spec"),
                       ("transparencyMapTexture", "alpha"), ("aoMapTexture", "ao")]:
        assert getattr(m, attr) == getpath.canonicalPath(os.path.join(folder, name + ".png")), attr
    assert (m.bumpMapIntensity, m.normalMapIntensity, m.displacementMapIntensity) == (0.3, 1.0, 0.4)
    assert (m.specularMapIntensity, m.transparencyMapIntensity, m.aoMapIntensity) == (0.6, 0.7, 0.8)
    assert (m.sssEnabled, m.sssRScale, m.sssGScale, m.sssBScale) == (True, 4.0, 0.0, 1.0)
    assert m.shader == getpath.canonicalPath("data/shaders/glsl/litsphere")
    assert m.uvMap.endswith("a7.mhuv")
    params = m.shaderParameters
    assert params["gain"] == 0.5 and params["tint"] == [0.1, 0.2, 0.3]
    assert params["lookup"] == getpath.canonicalPath(os.path.join(folder, "lut.png"))
    assert params["diffuse"] == pytest.approx([0.4, 0.5, 0.6, 0.75])
    assert params["specular"] == pytest.approx([0.7, 0.8, 0.9, 1.0])
    assert m.shaderConfig == {"diffuse": False, "bump": False, "normal": True, "displacement": False,
                              "spec": False, "vertexColors": False, "transparency": False,
                              "ambientOcclusion": False}
    # with a shader set, config defines are added next to the custom ones, sorted
    assert m.shaderDefines == ["CUSTOM_A", "CUSTOM_B", "NORMALMAP"]
    assert params["normalmapTexture"] == m.normalMapTexture
    for message in ('"diffuseIntensity"', '"specularIntensity"', "Unknown material shaderConfig property: sparkles"):
        assert message in caplog.text


def test_default_uv_map_is_stored_as_none(tmp_path):
    path = tmp_path / "m.mhmat"
    path.write_text("name m\nuvMap data/uvs/default.mhuv\n", encoding="utf-8")
    assert material.fromFile(str(path)).uvMap is None
    m = material.Material()
    m.uvMap = "data/uvs/a7.mhuv"
    assert m.uvMap == getpath.canonicalPath("data/uvs/a7.mhuv")
    m.uvMap = "data/uvs/default.mhuv"
    assert m.uvMap is None


def test_missing_file_is_reported(tmp_path, caplog):
    m = material.Material("keep")
    m.fromFile(str(tmp_path / "nope.mhmat"))
    assert m.name == "keep" and "Failed to load material" in caplog.text


def test_shipped_skin_material():
    m = material.fromFile("data/skins/default.mhmat")
    assert m.name == "DefaultSkin" and m.autoBlendSkin
    assert m.shader == getpath.canonicalPath("data/shaders/glsl/litsphere")
    assert m.shaderConfig["bump"] and not m.shaderConfig["diffuse"]
    assert (m.sssRScale, m.sssGScale, m.sssBScale) == (4.0, 2.0, 1.0)


# --- Writing -----------------------------------------------------------------

def test_write_and_read_back(full, tmp_path):
    original = material.fromFile(full)
    original.setShaderParameter("weights", ["1", "2"])
    out = str(tmp_path / "copy.mhmat")
    original.toFile(out, comments=["first note", "# already a comment", "// slashes"])
    text = open(out, encoding="utf-8").read()
    assert text.startswith("# Material definition for Full\n# first note\n# already a comment\n// slashes\n")
    assert "shaderDefine CUSTOM_A" in text and "shaderDefine NORMALMAP" not in text
    assert "shaderParam weights 1.0 2.0" in text and "shaderParam gain 0.5" in text
    assert "sssEnabled True" in text

    back = material.fromFile(out)
    for attr in ("name", "description", "shininess", "opacity", "translucency", "shadeless", "wireframe",
                 "transparent", "alphaToCoverage", "backfaceCull", "depthless", "castShadows", "receiveShadows",
                 "diffuseTexture", "bumpMapTexture", "normalMapTexture", "displacementMapTexture",
                 "specularMapTexture", "transparencyMapTexture", "aoMapTexture", "bumpMapIntensity",
                 "normalMapIntensity", "displacementMapIntensity", "specularMapIntensity",
                 "transparencyMapIntensity", "aoMapIntensity", "sssEnabled", "sssRScale", "sssGScale",
                 "sssBScale", "shader", "uvMap", "shaderConfig", "shaderDefines"):
        assert getattr(back, attr) == getattr(original, attr), attr
    for attr in ("ambientColor", "diffuseColor", "specularColor", "emissiveColor"):
        assert getattr(back, attr).values == pytest.approx(getattr(original, attr).values), attr
    assert back.shaderParameters["lookup"] == original.shaderParameters["lookup"]
    assert back.shaderParameters["weights"] == [1.0, 2.0]


def test_texture_paths_inside_data_are_portable(tmp_path):
    m = material.Material("portable")
    m.diffuseTexture = SKIN_TEX
    img = image.Image(SKIN_TEX)
    m.setShaderParameter("litsphereTexture", img)
    out = str(tmp_path / "p.mhmat")
    m.toFile(out)
    text = open(out, encoding="utf-8").read()
    assert "diffuseTexture skins/young_caucasian_female/young_lightskinned_female_diffuse.png" in text
    assert "shaderParam litsphereTexture skins/young_caucasian_female/" in text  # Image saved by its source
    assert material.fromFile(out).diffuseTexture == getpath.canonicalPath(SKIN_TEX)


def test_texture_path_helper(tmp_path, caplog):
    m = material.Material()
    outside = str(tmp_path / "x.png")
    assert m._texPath(outside, str(tmp_path)) == getpath.canonicalPath(outside)
    assert "Beware!" in caplog.text
    assert m._texPath("some\\\\file.png") == getpath.formatPath("some\\\\file.png")
    m.filepath = os.path.dirname(getpath.canonicalPath(SKIN_TEX))
    assert m._texPath(getpath.canonicalPath(SKIN_TEX)) == "young_lightskinned_female_diffuse.png"


def test_write_failure_is_reported(tmp_path, caplog):
    material.Material().toFile(str(tmp_path))  # a directory
    assert "Failed to open material file" in caplog.text


def test_peek_metadata(full, tmp_path, caplog):
    assert material.peekMetadata(full) == ("Full", {"shiny", "metal things"}, "A full material")
    bare = tmp_path / "bare.mhmat"
    bare.write_text("# nothing\n\nname Bare\n", encoding="utf-8")
    assert material.peekMetadata(str(bare)) == ("Bare", set(), "Bare material")
    assert material.peekMetadata(str(tmp_path / "missing.mhmat")) is None
    assert "Failed to load metadata" in caplog.text


# --- Properties and copies ---------------------------------------------------

def test_copy_constructor_and_clone(full):
    original = material.fromFile(full)
    copy = material.Material(original)
    clone = original.clone()
    for other in (copy, clone):
        assert other.name == "Full" and other.diffuseTexture == original.diffuseTexture
        assert other.shaderDefines == original.shaderDefines
        assert other.diffuseColor is not original.diffuseColor
        assert other.diffuseColor == original.diffuseColor
    copy.addShaderDefine("ONLY_COPY")
    assert "ONLY_COPY" not in original.shaderDefines


def test_setters_clamp_and_flag():
    m = material.Material("m")
    assert m.description == "m material" and not m.usesViewPortColor()
    m.shininess, m.opacity, m.translucency = 3, -1, 0.5
    assert (m.shininess, m.opacity, m.translucency) == (1.0, 0.0, 0.5)
    m.viewPortAlpha = 2
    assert m.viewPortAlpha == 1.0 and m.usesViewPortColor()
    m.enableViewPortColor(False)
    m.viewPortColor = (0.1, 0.2, 0.3)
    assert m.usesViewPortColor() and m.viewPortColor.values == pytest.approx([0.1, 0.2, 0.3])
    m.ambientColor, m.emissiveColor = (0.5, 0.5, 0.5), (0.2, 0.2, 0.2)
    assert m.ambientColor.values == [0.5] * 3 and m.emissiveColor.values == [0.2] * 3
    for prop in ("shadeless", "wireframe", "transparent", "alphaToCoverage", "backfaceCull", "depthless",
                 "castShadows", "receiveShadows", "sssEnabled"):
        setattr(m, prop, "flag")
        assert getattr(m, prop) == "flag"
    m.sssRScale, m.sssGScale, m.sssBScale = 1, 2, 3
    assert (m.sssRScale, m.sssGScale, m.sssBScale) == (1, 2, 3)
    m.autoBlendSkin = False
    assert m.autoBlendSkin is False


def test_tags():
    m = material.Material()
    m.addTag("Skin")
    m.addTag("SKIN")
    assert m.tags == {"skin"}
    m.removeTag("Skin")
    m.removeTag("absent")
    assert m.tags == set()


@pytest.mark.parametrize("prop", ["bumpMapIntensity", "normalMapIntensity", "displacementMapIntensity",
                                  "specularMapIntensity", "transparencyMapIntensity", "aoMapIntensity"])
def test_intensity_setters(prop):
    m = material.Material()
    setattr(m, prop, 0.25)
    assert getattr(m, prop) == 0.25


# --- Shader configuration ----------------------------------------------------

def textured(tmp_path, **textures):
    m = material.Material("tex")
    m.filepath = str(tmp_path)
    for attr, name in textures.items():
        setattr(m, attr, png(tmp_path / name))
    return m


def test_shader_config_defines_and_parameters(tmp_path):
    m = textured(tmp_path, diffuseTexture="d.png", bumpMapTexture="b.png", displacementMapTexture="x.png",
                 specularMapTexture="s.png", transparencyMapTexture="t.png", aoMapTexture="a.png")
    assert m.shaderDefines == []  # no shader: nothing configured yet
    m.shader = "data/shaders/glsl/phong"
    assert m.shader == getpath.canonicalPath("data/shaders/glsl/phong")
    assert m.shaderDefines == ["AOMAP", "BUMPMAP", "DIFFUSE", "DISPLACEMENT", "SPECULARMAP",
                               "TRANSPARENCYMAP", "VERTEX_COLOR"]
    params = m.shaderParameters
    assert params["diffuseTexture"] == m.diffuseTexture and params["aomapIntensity"] == 1.0
    m.normalMapTexture = png(tmp_path / "n.png")  # a normal map replaces the bump map
    assert "NORMALMAP" in m.shaderDefines and "BUMPMAP" not in m.shaderDefines
    m.configureShading(diffuse=False, normal=False, vertexColors=False, displacement=False, spec=False,
                       transparency=False, ambientOcclusion=False, bump=True)
    assert m.shaderDefines == ["BUMPMAP"]
    assert "diffuseTexture" not in m.shaderParameters
    assert m.supportsAo() == m.supportsAmbientOcclusion() is True


def test_shader_path_resolution():
    assert material.getShaderPath(None) is None
    assert material.getShaderPath("not/a/shader") == "not/a/shader"
    assert material.getShaderPath("litsphere", "data/shaders/glsl") == \
        getpath.canonicalPath("data/shaders/glsl/litsphere")
    assert material.getFilePath(None) is None and material.getFilePath(5) == 5


def test_protected_shader_parameters_and_defines():
    m = material.Material()
    with pytest.raises(RuntimeError, match="protected"):
        m.setShaderParameter("diffuseTexture", "x.png")
    with pytest.raises(RuntimeError, match="protected"):
        m.removeShaderParameter("diffuse")
    with pytest.raises(RuntimeError, match="protected"):
        m.addShaderDefine("DIFFUSE")
    with pytest.raises(RuntimeError, match="protected"):
        m.removeShaderDefine("NORMALMAP")


def test_custom_shader_parameters_and_defines():
    m = material.Material()
    m.setShaderParameter("gain", "2")
    m.setShaderParameter("offset", 3)
    assert m.shaderParameters["gain"] == 2.0 and m.shaderParameters["offset"] == 3
    m.removeShaderParameter("gain")
    m.removeShaderParameter("never-set")
    assert "gain" not in m.shaderParameters
    m.clearShaderParameters()
    assert set(m.shaderParameters) == {"ambient", "diffuse", "specular", "emissive"}
    m.addShaderDefine("B")
    m.addShaderDefine("A")
    m.addShaderDefine("A")
    assert m.shaderDefines == ["A", "B"]
    m.removeShaderDefine("A")
    m.removeShaderDefine("missing")
    assert m.shaderDefines == ["B"]
    changed = m.shaderChanged
    m.clearShaderDefines()
    assert m.shaderDefines == [] and m.shaderChanged >= changed


def test_clear_several_shader_defines():
    m = material.Material()
    for define in ("A", "B", "C"):
        m.addShaderDefine(define)
    m.clearShaderDefines()
    assert m.shaderDefines == []


class FakeShaderObj(object):
    def __init__(self, defineables, uniforms):
        self.defineables = defineables
        self._uniforms = uniforms
        self.glUniforms = ["gl_ModelViewMatrix"]

    def getUniforms(self):
        return [types.SimpleNamespace(name=n) for n in self._uniforms]


@pytest.fixture
def fake_shader(monkeypatch):
    """Stand-in for the OpenGL shader module (needs a GL context otherwise)."""
    state = types.SimpleNamespace(supported=True, obj=FakeShaderObj([], []), requests=[])
    module = types.ModuleType("shader")

    class Shader(object):
        @classmethod
        def supported(cls):
            return state.supported

    def getShader(path, defines):
        state.requests.append((path, defines))
        return state.obj

    module.Shader, module.getShader = Shader, getShader
    monkeypatch.setitem(sys.modules, "shader", module)
    return state


def test_supports_follow_the_shader_capabilities(tmp_path, fake_shader):
    m = textured(tmp_path, diffuseTexture="d.png", bumpMapTexture="b.png", normalMapTexture="n.png",
                 displacementMapTexture="x.png", specularMapTexture="s.png", transparencyMapTexture="t.png",
                 aoMapTexture="a.png")
    m._shader = "data/shaders/glsl/phong"
    checks = [m.supportsDiffuse, m.supportsBump, m.supportsNormal, m.supportsDisplacement,
              m.supportsSpecular, m.supportsTransparency, m.supportsAmbientOcclusion]
    assert [f() for f in checks] == [False] * 7  # textures set, but the shader uses none
    fake_shader.obj = FakeShaderObj(["DIFFUSE", "NORMALMAP", "AOMAP"], ["bumpmapTexture", "specularmapTexture"])
    assert [f() for f in checks] == [True, True, True, False, True, False, True]
    assert m.shaderUniforms == ["bumpmapTexture", "specularmapTexture", "gl_ModelViewMatrix"]
    assert fake_shader.requests[-1][0] == "data/shaders/glsl/phong"
    fake_shader.supported = False
    assert m.shaderObj is None and m.shaderUniforms == []
    fake_shader.supported = True
    m._shader = None
    assert m.getShaderObj() is None


def test_texture_dict(tmp_path, fake_shader):
    m = textured(tmp_path, diffuseTexture="d.png", aoMapTexture="a.png")
    lut = png(tmp_path / "lut.png")
    m.setShaderParameter("lut", lut)
    m.setShaderParameter("mem", image.Image(lut))
    m.setShaderParameter("gain", 2)
    assert list(m.getTextureDict()) == ["diffuseTexture", "aoMapTexture"]  # uniforms unused by the shader
    every = m.getTextureDict(includeUnused=True)
    assert every["lut"] == lut and isinstance(every["mem"], image.Image) and "gain" not in every
    assert "mem" not in m.getTextureDict(includeUnused=True, includeInMemory=False)
    assert list(m.getTextureDict(includeUniforms=False, includeUnused=True)) == ["diffuseTexture", "aoMapTexture"]


def test_export_textures(tmp_path, fake_shader, caplog):
    src = tmp_path / "src"
    src.mkdir()
    m = textured(src, diffuseTexture="d.png", aoMapTexture="a.png")
    m._specularMapTexture = str(src / "missing_spec.png")  # incomplete asset
    m.setShaderParameter("lut", png(src / "lut.png"))
    jpg_source = image.Image(png(src / "photo.png"))
    jpg_source.sourcePath = str(src / "photo.jpg")
    m.setShaderParameter("mem", jpg_source)
    m._shader = "data/shaders/glsl/phong"  # uniforms are only looked up when a shader is set
    fake_shader.obj = FakeShaderObj(["DIFFUSE", "AOMAP", "SPECULARMAP"], ["lut", "mem"])
    out = tmp_path / "export"
    result = m.exportTextures(str(out), excludeTextures=["aoMapTexture", "notThere"])
    assert result == {"diffuseTexture": str(out / "d.png"), "lut": str(out / "lut.png"),
                      "mem": str(out / "photo.jpg.png")}
    for path in result.values():
        assert os.path.isfile(path)
    assert "missing_spec.png" in caplog.text
    assert m.exportTextures(str(out), excludeUniforms=True) == \
        {"diffuseTexture": str(out / "d.png"), "aoMapTexture": str(out / "a.png")}


def test_in_memory_texture_is_kept_as_is():
    m = material.Material()
    img = image.Image(data=np.zeros((1, 1, 3), dtype=np.uint8))
    m.diffuseTexture = img
    assert m.diffuseTexture is img


def test_export_in_memory_texture_without_source(tmp_path, fake_shader, monkeypatch):
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(str(work))  # the bug writes into the working directory: keep it in tmp_path
    m = material.Material("mem")
    m.setShaderParameter("mem", image.Image(data=np.zeros((2, 2, 4), dtype=np.uint8)))
    m._shader = "data/shaders/glsl/phong"
    fake_shader.obj = FakeShaderObj([], ["mem"])
    out = tmp_path / "export"
    result = m.exportTextures(str(out))
    assert os.path.isfile(os.path.join(str(work), result["mem"]))  # what happens today
    assert os.path.dirname(os.path.abspath(result["mem"])) == str(out)


# --- Skin blending -----------------------------------------------------------

def test_auto_blend_skin_uses_the_human_ethnicity(monkeypatch):
    h = human.Human(files3d.loadMesh(getpath.getSysDataPath("3dobjs/base.obj"), maxFaces=5))
    monkeypatch.setattr(G, "app", FakeApp(h), raising=False)
    m = material.fromFile("data/skins/default.mhmat")
    blender = material.getSkinBlender()
    assert material.getSkinBlender() is blender  # created once
    expected = blender.getDiffuseColor()
    assert m.diffuseColor == expected
    assert m.diffuseColor.values != [1.0, 1.0, 1.0]
    lit = m.shaderParameters["litsphereTexture"]
    assert isinstance(lit, image.Image) and lit.sourcePath.endswith("adaptive_skin_tone.png")


# --- UV maps -----------------------------------------------------------------

def test_uv_obj_file(tmp_path):
    path = tmp_path / "uv.obj"
    path.write_text("# uvs\n\nv 0 0 0\nvt 0.0 0.5\nvt 1.0 0.25\nvt 0.5 1.0\nf 1/1 2/2 3/3\nf 1/3/1 2/2/1 3/1/1\n",
                    encoding="utf-8")
    uvs, fuvs = material.loadUvObjFile(str(path))
    assert uvs == [(0.0, 0.5), (1.0, 0.25), (0.5, 1.0)] and fuvs == [[0, 1, 2], [2, 1, 0]]
    uvmap = material.UVMap("alt")
    assert (uvmap.name, uvmap.type, uvmap.materialName, uvmap.uvs) == ("alt", "UvSet", "Default", None)
    uvmap.read(None, str(path))
    assert uvmap.filepath == str(path)
    assert uvmap.uvs.shape == (3, 2) and uvmap.fuvs.tolist() == [[0, 1, 2], [2, 1, 0]]
