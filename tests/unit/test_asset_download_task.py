"""
Unit tests for plugins/8_asset_downloader/downloadtask.py against a local
http.server: DownloadThread (run synchronously on the main thread, so its
signals reach the recorders directly) and DownloadTask (real QThread, callbacks
delivered through the Qt event loop). Harness and fixtures:
test_asset_harness.py.
"""

import os
import socket
import types

import pytest

from test_asset_harness import (  # noqa: F401  (qapp and env are pytest fixtures)
    FILES, downloadtask, env, qapp)

BIG = 2500 * 1000  # bytes; the thread reads files over 1 MB in 100 kB chunks


def remote(name):
    return FILES + "/assets/9/" + name


def closed_port_url(name="gone.mhclo"):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return "http://127.0.0.1:%d/%s" % (port, name)  # nothing listens there any more


def big_payload():
    return bytes(range(256)) * (BIG // 256) + b"tail"


def read(path):
    with open(path, "rb") as f:
        return f.read()


class Clock(object):
    """Stands in for the time module inside downloadtask; each call advances it."""

    def __init__(self, step):
        self.step = step
        self.now = 1000.0

    def time(self):
        self.now += self.step
        return self.now


def run_thread(tuples, steps=None, clock=None):
    thread = downloadtask.DownloadThread(tuples, overrideProgressSteps=steps)
    progress, finished = [], []
    thread.signalProgress.connect(lambda p: progress.append((clock.now if clock else None, p)))
    thread.signalFinished.connect(finished.append)
    thread.run()  # on this thread: the signals reach the lambdas directly
    return thread, [p for t, p in progress], finished, progress


# --------------------------------------------------------------------------
# DownloadThread
# --------------------------------------------------------------------------

def test_small_file_is_downloaded_into_a_new_folder(env, tmp_path):
    env.publish(remote("shirt.mhclo"), b"mhclo data")
    dest = str(tmp_path / "out" / "deep" / "shirt.mhclo")
    thread, progress, finished, _ = run_thread([(remote("shirt.mhclo"), dest)])
    assert read(dest) == b"mhclo data"
    assert finished == ["OK"]
    assert progress[0] == 0.0
    assert thread.exiting is True
    assert env.fetcher.opened == [remote("shirt.mhclo")]


def test_several_files_in_order(env, tmp_path):
    tuples = []
    for name in ("a.mhclo", "a.obj", "a.thumb"):
        env.publish(remote(name), name.encode())
        tuples.append((remote(name), str(tmp_path / name)))
    _, _, finished, _ = run_thread(tuples)
    assert finished == ["OK"]
    assert [read(l) for r, l in tuples] == [b"a.mhclo", b"a.obj", b"a.thumb"]
    assert env.fetcher.opened == [r for r, l in tuples]


def test_spaces_in_the_url_are_escaped(env, tmp_path):
    env.publish(remote("My Shirt.mhclo"), b"spaced")
    dest = str(tmp_path / "My Shirt.mhclo")
    _, _, finished, _ = run_thread([(remote("My Shirt.mhclo"), dest)])
    assert finished == ["OK"]
    assert read(dest) == b"spaced"
    assert env.fetcher.opened == [remote("My%20Shirt.mhclo")]


def test_large_file_is_read_in_chunks_with_progress(env, tmp_path, monkeypatch):
    clock = Clock(step=1.0)  # every progress check finds the half second elapsed
    monkeypatch.setattr(downloadtask, "time", clock)
    data = big_payload()
    env.publish(remote("big.obj"), data)
    dest = str(tmp_path / "big.obj")
    _, progress, finished, _ = run_thread([(remote("big.obj"), dest)], clock=clock)
    assert finished == ["OK"]
    assert read(dest) == data
    # 2.5 MB counts as 3 MB: whole megabytes received / 3, then 1.0 when the file is done
    assert progress[0] == 0.0
    assert progress[-1] == 1.0
    assert {1 / 3.0, 2 / 3.0} <= set(progress)
    assert progress == sorted(progress)


def test_override_progress_steps_scales_the_progress(env, tmp_path, monkeypatch):
    clock = Clock(step=1.0)
    monkeypatch.setattr(downloadtask, "time", clock)
    env.publish(remote("big.obj"), big_payload())
    _, progress, finished, _ = run_thread([(remote("big.obj"), str(tmp_path / "big.obj"))], steps=4, clock=clock)
    assert finished == ["OK"]
    assert progress[-1] == 1 / 4.0  # one file done out of four steps
    assert {1 / 12.0, 2 / 12.0} <= set(progress)


def test_small_files_report_progress_per_file(env, tmp_path, monkeypatch):
    clock = Clock(step=1.0)
    monkeypatch.setattr(downloadtask, "time", clock)
    tuples = []
    for name in ("a", "b", "c", "d"):
        env.publish(remote(name), name.encode())
        tuples.append((remote(name), str(tmp_path / name)))
    _, progress, _, _ = run_thread(tuples, clock=clock)
    assert progress == [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "downloadtask.py:137-138 throttles progress with `now = time() - 0.5; "
    "if now > lastReport: lastReport = now`; storing the shifted time means "
    "that after the first report every later chunk reports again, so the "
    "intended 0.5 s spacing only holds once (more UI updates than meant)"))
def test_progress_reports_are_at_least_half_a_second_apart(env, tmp_path, monkeypatch):
    clock = Clock(step=0.05)
    monkeypatch.setattr(downloadtask, "time", clock)
    env.publish(remote("big.obj"), big_payload())
    _, _, finished, timed = run_thread([(remote("big.obj"), str(tmp_path / "big.obj"))], clock=clock)
    assert finished == ["OK"]
    times = [t for t, p in timed[1:]]  # the 0.0 at the start is unconditional
    assert len(times) >= 2
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert min(gaps) >= 0.5 - 1e-9, gaps


def test_http_error_stops_the_batch(env, tmp_path):
    env.publish(remote("ok.mhclo"), b"ok")
    tuples = [(remote("missing.mhclo"), str(tmp_path / "missing.mhclo")),
              (remote("ok.mhclo"), str(tmp_path / "ok.mhclo"))]
    _, _, finished, _ = run_thread(tuples)
    assert finished == ["404;" + remote("missing.mhclo")]
    assert not os.path.exists(str(tmp_path / "missing.mhclo"))
    assert not os.path.exists(str(tmp_path / "ok.mhclo"))  # never attempted
    assert env.server.requests == ["/sites/default/files/assets/9/missing.mhclo"]
    assert "Caught http error HTTP Error 404" in env.plugin_log()


def test_connection_error_skips_the_file_and_goes_on(env, tmp_path):
    env.publish(remote("ok.mhclo"), b"ok")
    gone = closed_port_url()
    tuples = [(gone, str(tmp_path / "gone.mhclo")), (remote("ok.mhclo"), str(tmp_path / "ok.mhclo"))]
    run_thread(tuples)
    assert not os.path.exists(str(tmp_path / "gone.mhclo"))
    assert read(str(tmp_path / "ok.mhclo")) == b"ok"
    assert env.fetcher.opened == [gone, remote("ok.mhclo")]
    log = env.plugin_log()
    assert "Connection refused" in log
    assert "Could not download " + gone in log


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "downloadtask.py:153-155: a bare `except` logs any non-HTTP failure "
    "(connection refused or reset, timeout, bad URL) and the batch still ends "
    "with status 'OK', so callers believe a file arrived that is not on disk "
    "(see test_asset_db.py for the crash this leads to)"))
def test_connection_error_is_reported(env, tmp_path):
    _, _, finished, _ = run_thread([(closed_port_url(), str(tmp_path / "gone.mhclo"))])
    assert finished != ["OK"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "downloadtask.py:101 calls .strip() on info().get('Content-Length'), "
    "which is None when the server streams the body (chunked transfer or "
    "HTTP/1.0 without length); the AttributeError lands in the bare except at "
    "line 153, the file is never written and the status is still 'OK'"))
def test_response_without_content_length_is_saved(env, tmp_path):
    env.publish(env.local_url("stream.mhclo"), b"streamed")
    dest = str(tmp_path / "stream.mhclo")
    _, _, finished, _ = run_thread([(env.local_url("no-length/stream.mhclo"), dest)])
    assert os.path.exists(dest) and read(dest) == b"streamed"


def test_empty_batch_finishes_ok(env):
    _, progress, finished, _ = run_thread([])
    assert progress == [0.0]
    assert finished == ["OK"]


# --------------------------------------------------------------------------
# DownloadTask
# --------------------------------------------------------------------------

# Every test keeps the DownloadTask in a variable, as the plugin does
# (AssetDB._downloadTask, view.screenshotDt): PyQt holds bound-method slots
# weakly, so a task nobody references never gets its callbacks.

class Calls(object):
    def __init__(self):
        self.finished = []
        self.progress = []

    def on_finished(self, code=0, file=None):
        self.finished.append((code, file))

    def on_progress(self, prog):
        self.progress.append(prog)


def test_task_downloads_and_reports_success(env, tmp_path):
    env.publish(remote("a.mhclo"), b"A")
    env.publish(remote("a.obj"), b"B")
    tuples = [(remote("a.mhclo"), str(tmp_path / "a.mhclo")), (remote("a.obj"), str(tmp_path / "a.obj"))]
    calls = Calls()
    task = downloadtask.DownloadTask(None, tuples, calls.on_finished, calls.on_progress)
    env.pump_until(lambda: calls.finished)
    assert calls.finished == [(0, None)]
    assert read(str(tmp_path / "a.mhclo")) == b"A" and read(str(tmp_path / "a.obj")) == b"B"
    assert calls.progress[0] == 0.0
    assert env.app.progress_calls[0] == (0.0, "Downloading files...")
    assert env.app.progress_calls[-1] == (1.0, "Downloading files...")
    assert task.downloadThread is None

    thread = env.threads[0]
    assert thread.wait(10000)
    thread.signalFinished.emit("OK")  # the task disconnected itself
    thread.signalProgress.emit(0.5)
    env.pump_until(lambda: True)
    assert calls.finished == [(0, None)]
    assert 0.5 not in calls.progress


def test_task_reports_http_error_code_and_url(env, tmp_path):
    calls = Calls()
    task = downloadtask.DownloadTask(None, [(remote("nope.mhclo"), str(tmp_path / "nope.mhclo"))], calls.on_finished)
    env.pump_until(lambda: calls.finished)
    assert calls.finished == [(404, remote("nope.mhclo"))]


@pytest.mark.xfail(strict=True, raises=ValueError, reason=(
    "downloadtask.py:234 splits the 'code;url' status with split(';', 2), "
    "so a failing URL that contains ';' gives three parts and the unpacking "
    "raises ValueError inside the Qt slot (in the app, with no excepthook "
    "installed, PyQt then aborts MakeHuman)"))
def test_task_reports_http_error_for_url_with_semicolon(env, tmp_path):
    calls = Calls()
    bad = remote("nope;v=2.mhclo")
    task = downloadtask.DownloadTask(None, [(bad, str(tmp_path / "nope.mhclo"))], calls.on_finished)
    env.pump_until(lambda: calls.finished)
    assert calls.finished == [(404, bad)]


def test_task_without_callbacks(env, tmp_path):
    env.publish(remote("a.mhclo"), b"A")
    task = downloadtask.DownloadTask(None, [(remote("a.mhclo"), str(tmp_path / "a.mhclo"))])
    env.pump_until(lambda: task.downloadThread is None)
    assert read(str(tmp_path / "a.mhclo")) == b"A"
    assert env.app.progress_calls[-1] == (1.0, "Downloading files...")


def test_task_with_override_steps_leaves_the_progress_bar_to_the_caller(env, tmp_path):
    env.publish(remote("a.mhclo"), b"A")
    calls = Calls()
    task = downloadtask.DownloadTask(None, [(remote("a.mhclo"), str(tmp_path / "a.mhclo"))],
                                     calls.on_finished, calls.on_progress, overrideProgressSteps=3)
    env.pump_until(lambda: calls.finished)
    assert calls.finished == [(0, None)]
    assert (1.0, "Downloading files...") not in env.app.progress_calls
    assert env.app.progress_calls[0] == (0.0, "Downloading files...")
