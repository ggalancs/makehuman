"""lib/profiler.py: a thin wrapper around cProfile."""

import pytest

import profiler


@pytest.fixture(autouse=True)
def clean_profiler(monkeypatch):
    monkeypatch.setattr(profiler, "_profiler", None)
    monkeypatch.setattr(profiler, "_show", None)
    monkeypatch.setattr(profiler, "_sort", "cumulative")


def busy(n):
    return sum(i * i for i in range(n))


def test_run_executes_and_prints_stats(capsys):
    scope = {"busy": busy}
    profiler.run("result = busy(1000)", scope, scope)
    assert scope["result"] == busy(1000)
    out = capsys.readouterr().out
    assert "function calls" in out
    assert "Ordered by: cumulative time" in out
    assert "busy" in out


def test_run_shows_stats_even_when_the_command_fails():
    shown = []
    profiler.set_show(shown.append)
    with pytest.raises(ZeroDivisionError):
        profiler.run("1/0", {}, {})
    assert len(shown) == 1


def test_start_accumulate_stop():
    shown = []
    profiler.set_show(shown.append)
    scope = {"busy": busy, "calls": 0}
    profiler.accum("calls += 1", scope, scope)  # not active: ignored
    assert scope["calls"] == 0 and not profiler.active()

    profiler.start()
    first = profiler._profiler
    profiler.start()  # already running: same profiler
    assert profiler.active() and profiler._profiler is first
    profiler.accum("calls += 1; busy(10)", scope, scope)
    profiler.accum("calls += 1", scope, scope)
    assert scope["calls"] == 2

    profiler.stop()
    assert not profiler.active()
    assert shown == [first]
    stats = __import__("pstats").Stats(first)
    assert any(func[2] == "busy" for func in stats.stats)
    profiler.stop()  # stopping twice is harmless
    assert shown == [first]


def test_sort_order_is_used(capsys):
    profiler.set_sort("ncalls")
    profiler.run("busy(10)", {"busy": busy}, {})
    assert "Ordered by: call count" in capsys.readouterr().out


def test_show_ignores_type_errors_from_the_viewer():
    def picky(prof):
        raise TypeError("cannot show this")

    profiler.set_show(picky)
    profiler.show(object())  # must not raise


def test_show_does_not_hide_other_errors():
    def broken(prof):
        raise ValueError("real problem")

    profiler.set_show(broken)
    with pytest.raises(ValueError):
        profiler.show(object())
