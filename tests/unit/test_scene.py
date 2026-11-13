"""shared/scene.py: lights and environment, and the .mhscene file format."""

import logging
import os
import pickle

import pytest

import scene
from material import Color

MH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "makehuman")


class Recorder(object):
    def __init__(self):
        self.calls = 0

    def changed(self, *args, **kwargs):
        self.calls += 1


def colors_equal(c1, c2):
    return tuple(round(v, 5) for v in c1.values) == tuple(round(v, 5) for v in c2.values)


# --- version checks -----------------------------------------------------------------------

def test_check_versions():
    assert scene.checkVersions((5, 5, 5))
    assert scene.checkVersions((None, 1, None))
    assert not scene.checkVersions((5, 4, None))
    assert not scene.checkVersions((None, 6, 5))
    with pytest.raises(scene.FileVersionException) as old:
        scene.checkVersions((5, 4, 6), scene.FileVersionException)
    assert "no more supported" in str(old.value)
    with pytest.raises(scene.FileVersionException) as new:
        scene.checkVersions((5, 7, 6), scene.FileVersionException)
    assert "too old to open this file" in str(new.value)


def test_file_version_exception_messages():
    assert str(scene.FileVersionException()) == "Incompatible file version"
    assert str(scene.FileVersionException("custom")) == "'custom'"


# --- scene objects --------------------------------------------------------------------------

def test_light_defaults_and_change_notification():
    owner = Recorder()
    light = scene.Light(owner)
    assert light.position == (-10.99, 20.0, 20.0)
    assert light.getAttributes() == sorted(["position", "focus", "color", "specular", "fov",
                                            "attenuation", "areaLights", "areaLightSize"])
    light.fov = 180.0  # same value: no notification
    assert owner.calls == 0
    light.fov = 45.0
    assert light.fov == 45.0 and owner.calls == 1


def test_color_attributes_stay_colors():
    light = scene.Light(Recorder())
    light.color = (0.5, 0.25, 0.0)
    assert isinstance(light.color, Color)
    assert colors_equal(light.color, Color(0.5, 0.25, 0.0))


def test_plain_attributes_are_ordinary():
    light = scene.Light()
    light.note = "not a scene attribute"
    assert light.note == "not a scene attribute"
    light.changed()  # no scene attached: nothing to notify


def test_unknown_attribute_raises_attribute_error():
    assert not hasattr(scene.Light(), "brightness")


# --- Scene save and load -------------------------------------------------------------------------

def test_new_scene_has_one_default_light():
    s = scene.Scene()
    assert len(s.lights) == 1
    assert colors_equal(s.environment.ambience, Color(0.3, 0.3, 0.3))
    assert s.file.path is None and not s.file.modified


def test_save_and_load_round_trip(tmp_path):
    s = scene.Scene()
    s.environment.ambience = Color(0.1, 0.2, 0.3)
    s.environment.skybox = "sky.png"
    s.lights[0].position = (1.0, 2.0, 3.0)
    s.addLight()
    s.lights[1].color = Color(0.0, 1.0, 0.0)
    assert s.file.modified
    path = str(tmp_path / "test.mhscene")
    assert s.save(path) is True
    assert s.file.path == path and not s.file.modified

    loaded = scene.Scene(path)
    assert loaded.file.path == path and not loaded.file.modified
    assert loaded.filever == scene.mhscene_version
    assert len(loaded.lights) == 2
    assert loaded.lights[0].position == (1.0, 2.0, 3.0)
    assert colors_equal(loaded.lights[1].color, Color(0.0, 1.0, 0.0))
    assert isinstance(loaded.environment.ambience, Color)
    assert colors_equal(loaded.environment.ambience, Color(0.1, 0.2, 0.3))
    assert loaded.environment.skybox == "sky.png"


def test_shipped_scenes_load():
    for name in ("default.mhscene", "coldlights.mhscene"):
        s = scene.Scene(os.path.join(MH_DIR, "data", "scenes", name))
        assert s.file.path.endswith(name)
        assert len(s.lights) >= 1


def test_add_and_remove_lights_mark_the_scene_modified(tmp_path):
    s = scene.Scene()
    s.save(str(tmp_path / "s.mhscene"))
    s.addLight()
    assert len(s.lights) == 2 and s.file.modified
    s.save(str(tmp_path / "s.mhscene"))
    s.removeLight(s.lights[0])
    assert len(s.lights) == 1 and s.file.modified


def test_reset_restores_defaults(tmp_path):
    s = scene.Scene()
    s.addLight()
    s.environment.skybox = "x"
    s.reset()
    assert len(s.lights) == 1 and s.environment.skybox is None
    assert s.file.path is None


@pytest.mark.parametrize("version", [scene.mhscene_minversion - 1, scene.mhscene_version + 1])
def test_unsupported_versions_are_refused(tmp_path, version, caplog):
    path = tmp_path / "v.mhscene"
    with open(str(path), "wb") as f:
        pickle.dump(version, f)
    s = scene.Scene()
    with caplog.at_level(logging.WARNING):
        assert s.load(str(path)) is False
    assert len(s.lights) == 1  # untouched
    assert any(str(path) in r.getMessage() for r in caplog.records)


def test_save_reports_unpicklable_values(tmp_path, caplog):
    s = scene.Scene()
    s.environment.skybox = lambda: None  # cannot be pickled
    with caplog.at_level(logging.ERROR):
        assert s.save(str(tmp_path / "bad.mhscene")) is False
    assert any("Failed to save scene file" in r.getMessage() for r in caplog.records)
    assert s.file.path is None


def test_save_reports_unexpected_open_errors(tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        assert scene.Scene().save(str(tmp_path / "nul\0byte.mhscene")) is False
    assert any("Failed to save scene file" in r.getMessage() for r in caplog.records)


def test_load_reports_unexpected_open_errors(tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        assert scene.Scene().load(str(tmp_path / "nul\0byte.mhscene")) is False
    assert any("Failed to load scene file" in r.getMessage() for r in caplog.records)


def test_load_of_a_missing_file_returns_false(tmp_path):
    assert scene.Scene().load(str(tmp_path / "missing.mhscene")) is False


def test_save_into_a_folder_returns_false(tmp_path):
    assert scene.Scene().save(str(tmp_path)) is False


def test_load_of_a_corrupt_file_returns_false(tmp_path):
    path = tmp_path / "corrupt.mhscene"
    path.write_bytes(b"\x80\x02K\x05garbage")  # version 5, then junk
    s = scene.Scene()
    assert s.load(str(path)) is False
    assert len(s.lights) == 1


def test_attributes_newer_than_the_file_are_skipped():
    """Versioned attributes are only read from files that contain them."""
    owner = Recorder()

    class Probe(scene.SceneObject):
        def __init__(self, s=None):
            scene.SceneObject.__init__(self, s, attributes={
                "always": 1,
                "since6": [2, 6],
                "range": [3, (4, 5)],
                "ranges": [Color(0, 0, 0), [(1, 2), (5, None)]],
            })

    import io
    buf = io.BytesIO()
    for value in (10, (0.5, 0.5, 0.5), (0.2, 0.4, 0.6)):  # always, range, ranges; since6 skipped
        pickle.dump(value, buf)
    buf.seek(0)
    owner.filever = 5
    probe = Probe(owner)
    probe.load(buf)
    assert probe.always == 10
    assert probe.since6 == 2  # not in a version 5 file: default kept
    assert probe.range == (0.5, 0.5, 0.5)
    assert isinstance(probe.ranges, Color)
    assert colors_equal(probe.ranges, Color(0.2, 0.4, 0.6))
