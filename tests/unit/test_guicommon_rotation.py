"""guicommon used log without importing it: a Z rotation raised NameError instead of warning."""

import logging

import guicommon
import module3d


def test_z_rotation_is_refused_with_a_warning(caplog):
    obj = guicommon.Object(module3d.Object3D("probe"))
    with caplog.at_level(logging.WARNING):
        obj.setRotation([370.0, -30.0, 12.0])
    assert [float(v) for v in obj.getRotation()] == [10.0, 330.0, 0.0]
    assert any("rotation around Z" in r.getMessage() for r in caplog.records)
