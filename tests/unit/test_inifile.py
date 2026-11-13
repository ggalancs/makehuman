"""lib/inifile.py: JSON settings files, with a fallback for Python-literal syntax."""

import json

import pytest

import inifile


def test_parse_plain_json():
    assert inifile.parseINI('{"a": 1, "b": [true, null, "x"]}') == {"a": 1, "b": [True, None, "x"]}


def test_parse_bytes_are_decoded_as_utf8():
    raw = '{"name": "Ñandú"}'.encode("utf-8")
    assert inifile.parseINI(raw) == {"name": "Ñandú"}


def test_parse_python_literal_fallback():
    # Old settings files were written with repr(): single quotes and Python constants
    s = "{'sliderImages': True, 'units': 'metric', 'guiTheme': None, 'invertMouseWheel': False}"
    assert inifile.parseINI(s) == {"sliderImages": True, "units": "metric",
                                   "guiTheme": None, "invertMouseWheel": False}


def test_parse_applies_extra_replacements_before_the_builtin_ones():
    s = "{'speed': FAST}"
    assert inifile.parseINI(s, replace=[("FAST", "3")]) == {"speed": 3}


def test_parse_still_fails_on_garbage():
    with pytest.raises(ValueError):
        inifile.parseINI("this is not a settings file")


def test_nested_values_are_preserved():
    data = {"outer": {"inner": [1, {"k": "v"}, [2.5, "s"]]}, "n": 0}
    assert inifile.parseINI(json.dumps(data)) == data


def test_s2u_keys_become_str_and_scalars_pass_through():
    assert inifile._s2u({1: [b"raw", 2.0]}) == {"1": [b"raw", 2.0]}
    assert inifile._s2u(7) == 7


def test_format_round_trip_keeps_non_ascii():
    d = {"author": "José Núñez", "values": [1, 2, 3], "flag": False}
    text = inifile.formatINI(d)
    assert text.endswith("\n")
    assert "José Núñez" in text  # ensure_ascii=False: no \u escapes
    assert inifile.parseINI(text) == d
