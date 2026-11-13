"""Tests for lib/sorter.py, the decorate-sort-undecorate helper used by the
file choosers."""

import pytest

from sorter import Sorter


class Item(object):
    """Not orderable on purpose: sorting must never compare the items."""

    def __init__(self, name, size):
        self.name = name
        self.size = size

    def __repr__(self):
        return "Item(%r, %r)" % (self.name, self.size)


def make_sorter():
    s = Sorter()
    s.methods = [("name", lambda o: o.name.lower()), ("size", lambda o: o.size)]
    return s


ITEMS = [Item("b", 3), Item("A", 1), Item("c", 3), Item("a", 2), Item("B", 1)]


def test_methods_container():
    s = make_sorter()
    methods = s.getMethods()
    assert isinstance(methods, Sorter.Methods)
    assert list(s.fields()) == ["name", "size"]
    assert "name" in methods and "colour" not in methods
    assert methods["size"](Item("x", 7)) == 7
    methods.clear()
    assert list(s.fields()) == [] and "name" not in methods


def test_sort_by_field():
    s = make_sorter()
    assert [o.size for o in s.sort("size", ITEMS)] == [1, 1, 2, 3, 3]
    assert [o.name.lower() for o in s.sort("name", ITEMS)] == ["a", "a", "b", "b", "c"]
    assert [o.name for o in ITEMS] == ["b", "A", "c", "a", "B"]  # input untouched


def test_sort_is_stable_and_never_compares_the_objects():
    s = make_sorter()
    # Equal keys keep their input order; Item has no ordering, so comparing
    # two of them would raise TypeError
    by_size = s.sort("size", ITEMS)
    assert [o.name for o in by_size] == ["A", "B", "a", "b", "c"]


def test_sort_shortcut_attributes():
    s = make_sorter()
    assert s.sortSize(ITEMS) == s.sort("size", ITEMS)
    assert s.sortName(ITEMS) == s.sort("name", ITEMS)


def test_unknown_attributes_still_raise_attribute_error():
    s = make_sorter()
    with pytest.raises(AttributeError):
        s.colour
    assert not hasattr(s, "notASortMethod")


def test_unknown_sort_field_raises_key_error():
    with pytest.raises(KeyError):
        make_sorter().sortColour(ITEMS)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4])
def test_insert_keeps_the_list_sorted(size):
    s = make_sorter()
    objects = s.sort("size", ITEMS)
    new = Item("new", size)
    result = s.insert("size", new, objects)
    assert result is objects
    assert [o.size for o in result] == sorted(o.size for o in ITEMS + [new])
    # Like bisect_left: the new object goes before existing equal keys
    first_equal = next(i for i, o in enumerate(result) if o.size == size)
    assert result[first_equal] is new


def test_insert_into_an_empty_list():
    s = make_sorter()
    item = Item("only", 1)
    assert s.insert("name", item, []) == [item]


@pytest.mark.parametrize("assigned", [False, True], ids=["empty", "after-assign"])
def test_methods_extend(assigned):
    s = Sorter()
    if assigned:
        s.methods = [("name", lambda o: o.name)]
    s.methods.extend([("size", lambda o: o.size)])
    assert "size" in s.methods
    assert list(s.fields())[-1] == "size"
