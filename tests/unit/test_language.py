"""lib/language.py: loading translations, lookups and the list of missing strings."""

import json
import logging
import os

import pytest

import language


@pytest.fixture
def langdir(tmp_path, monkeypatch):
    """A working folder with data/languages, like makehuman/ has."""
    folder = tmp_path / "data" / "languages"
    folder.mkdir(parents=True)
    (folder / "pirate.json").write_text(json.dumps({
        "Hello": "Ahoy",
        "Age": "Years at sea: ",
        "Left": "",  # explicitly empty translation
        "__options__": {"rtl": True},
    }), encoding="utf-8")
    (folder / "plain.json").write_text(json.dumps({"Hello": "Hi"}), encoding="utf-8")
    (folder / "broken.json").write_text("{not json", encoding="utf-8")
    (folder / "notes.txt").write_text("not a language", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return folder


def test_set_language_loads_strings_and_options(langdir):
    lang = language.Language()
    lang.setLanguage("pirate")
    assert lang.language == "pirate"
    assert lang.rtl is True
    assert lang.getLanguageString("Hello") == "Ahoy"
    lang.setLanguage("plain")
    assert lang.rtl is False
    assert lang.getLanguageString("Hello") == "Hi"


def test_unknown_language_keeps_the_previous_name_but_drops_strings(langdir):
    lang = language.Language()
    lang.setLanguage("plain")
    lang.setLanguage("klingon")
    assert lang.languageStrings is None
    assert lang.language == "plain"
    assert lang.getLanguageString("Hello") == "Hello"


def test_broken_language_file_is_reported(langdir, caplog):
    lang = language.Language()
    with caplog.at_level(logging.ERROR):
        lang.setLanguage("broken")
    assert lang.languageStrings is None
    assert lang.language == "broken"
    assert any("Error in language file broken" in r.getMessage() for r in caplog.records)


def test_untranslated_lookups():
    lang = language.Language()
    assert lang.getLanguageString("") == ""
    assert lang.getLanguageString(None) is None
    assert lang.getLanguageString("%%s") == "%%s"
    assert lang.getLanguageString("Hello") == "Hello"
    assert lang.getLanguageString(["Weight: ", "70kg"]) == "Weight: 70kg"
    assert lang.getLanguageString(["Weight: ", 70.0, "%.1f kg"]) == "Weight: 70.0 kg"
    assert lang.getLanguageString("Size: ", "XL") == "Size: XL"
    assert not lang.missingStrings  # nothing is missing when no language is loaded


def test_translated_lookups_with_appended_data(langdir):
    lang = language.Language()
    lang.setLanguage("pirate")
    assert lang.getLanguageString(["Age", 30, "%d"]) == "Years at sea: 30"
    assert lang.getLanguageString("Age", "many") == "Years at sea: many"
    assert lang.getLanguageString("Left") == ""
    assert list(lang.missingStrings) == []


def test_missing_strings_are_remembered_in_order(langdir):
    lang = language.Language()
    lang.setLanguage("pirate")
    assert lang.getLanguageString("Goodbye") == "Goodbye"
    assert lang.getLanguageString("Ship") == "Ship"
    lang.getLanguageString("Goodbye")
    assert list(lang.missingStrings) == ["Goodbye", "Ship"]


def test_missing_translation_keeps_appended_data(langdir):
    lang = language.Language()
    lang.setLanguage("pirate")
    assert lang.getLanguageString(["Weight: ", "70kg"]) == "Weight: 70kg"


def test_dump_missing_strings_without_language_writes_nothing(langdir):
    lang = language.Language()
    lang.getLanguageString("whatever")
    lang.dumpMissingStrings()
    assert sorted(p.name for p in langdir.iterdir()) == \
        ["broken.json", "notes.txt", "pirate.json", "plain.json"]


@pytest.mark.parametrize("lang_name, expected", [
    ("pirate", '"Goodbye": "",\n"Line\\none": "",\n'),
    ("master", '"Goodbye": "Goodbye",\n"Line\\none": "Line\\none",\n'),
])
def test_dump_missing_strings_writes_json_lines(langdir, lang_name, expected):
    if lang_name == "master":
        (langdir / "master.json").write_text("{}", encoding="utf-8")
    lang = language.Language()
    lang.setLanguage(lang_name)
    lang.getLanguageString("Goodbye")
    lang.getLanguageString("Line\none")
    lang.dumpMissingStrings()
    assert (langdir / (lang_name + ".missing")).read_text(encoding="utf-8") == expected


def test_dump_missing_strings_creates_the_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lang = language.Language()
    lang.language = "newlang"
    lang.dumpMissingStrings()
    assert (tmp_path / "data" / "languages" / "newlang.missing").is_file()


def test_get_languages_lists_system_and_user_files(langdir, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("MH_HOME_LOCATION", str(home))
    import getpath
    user_langs = getpath.getDataPath("languages")
    assert language.getLanguages() == ["english", "broken", "pirate", "plain"]
    os.makedirs(user_langs)
    with open(os.path.join(user_langs, "elvish.json"), "w") as f:
        f.write("{}")
    assert language.getLanguages() == ["english", "broken", "elvish", "pirate", "plain"]


def test_shipped_languages_are_valid(monkeypatch):
    """Every bundled language file loads and is found by getLanguages()."""
    monkeypatch.chdir(os.path.join(os.path.dirname(__file__), "..", "..", "makehuman"))
    names = language.getLanguages()
    assert names[0] == "english" and "german_generic" in names and "master" in names
    for name in names[1:]:
        lang = language.Language()
        lang.setLanguage(name)
        assert isinstance(lang.languageStrings, dict), name
