"""lib/log.py: level names, filters, GUI log handlers and log.init()."""

import logging
import os
import warnings

import pytest

import log
from core import G

MH_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MH_DIR = os.path.join(MH_DIR, "makehuman")


def make_record(msg, args=(), level=logging.INFO):
    return logging.LogRecord("test", level, __file__, 1, msg, args, None)


# --- level names and colours -------------------------------------------------

@pytest.mark.parametrize("level, name", [
    (logging.DEBUG, "debug"), (logging.INFO, "info"), (log.NOTICE, "notice"),
    (logging.WARNING, "warning"), (logging.ERROR, "error"), (logging.CRITICAL, "critical"),
])
def test_known_levels_have_lowercase_names(level, name):
    assert log.logLevelToStr(level) == name


@pytest.mark.parametrize("level", [5, 15, 60])
def test_unknown_level_is_named_after_a_known_level(level):
    assert log.logLevelToStr(level) in ("DEBUG", "INFO", "NOTICE", "WARNING", "ERROR", "CRITICAL")


def test_level_names_are_registered_with_logging():
    assert logging.getLevelName(log.NOTICE) == "NOTICE"
    assert logging.getLevelName(log.MESSAGE) == "MESSAGE"


@pytest.mark.parametrize("level, color", [
    (logging.DEBUG, "grey"), (log.NOTICE, "blue"), (logging.WARNING, "darkorange"),
    (logging.ERROR, "red"), (logging.CRITICAL, "red"),
])
def test_level_colors(level, color):
    assert log.getLevelColor(level) == color


def test_unknown_level_color_defaults_to_red():
    assert log.getLevelColor(15) == "red"


# --- message helpers -----------------------------------------------------------

def test_to_unicode_formats_positional_and_dict_args():
    assert log._toUnicode("%s and %d", "a", 2) == "a and 2"
    assert log._toUnicode("%(x)s!", {"x": "hi"}) == "hi!"
    assert log._toUnicode(b"raw %s", b"bytes") == "raw bytes"  # bytes are decoded as UTF-8


def test_to_unicode_reraises_real_format_errors():
    with pytest.raises(TypeError):
        log._toUnicode("%s %s", "only one")


@pytest.mark.parametrize("func, level", [
    (log.debug, logging.DEBUG), (log.message, logging.INFO),
    (log.notice, log.NOTICE), (log.warning, logging.WARNING), (log.error, logging.ERROR),
])
def test_helpers_log_at_their_level(func, level, caplog):
    with caplog.at_level(logging.DEBUG):
        func("value %s", 42)
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [(level, "value 42")]


def test_notice_reports_the_caller_not_the_log_module(caplog):
    with caplog.at_level(logging.DEBUG):
        log.notice("from the test")
    assert caplog.records[0].filename == os.path.basename(__file__)


@pytest.mark.parametrize("name, level", [
    ("debug", logging.DEBUG), ("warning", logging.WARNING),
    ("error", logging.ERROR), ("message", logging.INFO),
])
def test_unicode_error_fallback_logs_the_formatted_message(name, level, monkeypatch, caplog):
    real = getattr(logging, {"message": "info"}.get(name, name))
    calls = []

    def flaky(msg, *args, **kwargs):
        calls.append(msg)
        if len(calls) == 1:
            raise UnicodeError("simulated")
        return real(msg, *args, **kwargs)

    monkeypatch.setattr(logging, {"message": "info"}.get(name, name), flaky)
    with caplog.at_level(logging.DEBUG):
        getattr(log, name)("value %s", "a")
    assert len(calls) == 2  # the fallback was taken
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [(level, "value a")]


# --- filters -------------------------------------------------------------------

def test_noise_filter_drops_trailing_traceback_argument():
    rec = make_record("OpenGL said:\n%s", ("first", "huge dump"))
    assert log.NoiseFilter().filter(rec) is True
    assert rec.msg == "OpenGL said"
    assert rec.args == ("first",)


def test_noise_filter_leaves_other_messages_and_survives_non_strings(capsys):
    rec = make_record("plain %s", ("x",))
    assert log.NoiseFilter().filter(rec) is True
    assert (rec.msg, rec.args) == ("plain %s", ("x",))
    odd = make_record(12345)
    assert log.NoiseFilter().filter(odd) is True  # still passes the record on
    assert "AttributeError" in capsys.readouterr().err


def test_downgrade_filter_caps_the_level():
    flt = log.DowngradeFilter(logging.DEBUG)
    rec = make_record("x", level=logging.ERROR)
    assert flt.filter(rec) is True
    assert (rec.levelno, rec.levelname) == (logging.DEBUG, "DEBUG")
    low = make_record("x", level=logging.DEBUG)
    flt.filter(low)
    assert low.levelname == "DEBUG"


def test_downgrade_filter_survives_broken_records(capsys):
    rec = make_record("x")
    rec.levelno = None
    assert log.DowngradeFilter(logging.DEBUG).filter(rec) is True
    assert "TypeError" in capsys.readouterr().err


# --- handlers that forward to the GUI -------------------------------------------

class Splash(object):
    def __init__(self):
        self.lines = []

    def logMessage(self, text):
        self.lines.append(text)


class StatusBar(object):
    def __init__(self):
        self.messages = []

    def temporaryMessage(self, fmt, msg, msec):
        self.messages.append((fmt % msg, msec))


class GuiApp(object):
    def __init__(self):
        self.splash = Splash()
        self.statusBar = StatusBar()
        self.log_window = object()
        self.logged = []

    def addLogMessage(self, text, level):
        self.logged.append((text, level))


def test_splash_handler_shows_first_line(monkeypatch):
    app = GuiApp()
    monkeypatch.setattr(G, "app", app)
    log.SplashLogHandler().emit(make_record("Loading\nsecond line"))
    assert app.splash.lines == ["Loading\n"]


def test_status_handler_only_reports_errors(monkeypatch):
    app = GuiApp()
    monkeypatch.setattr(G, "app", app)
    handler = log.StatusLogHandler()
    handler.emit(make_record("just info"))
    handler.emit(make_record("disk %s", ("full",), level=logging.ERROR))
    assert len(app.statusBar.messages) == 1
    text, msec = app.statusBar.messages[0]
    assert text.endswith("Error Message:  disk full")
    assert msec == 10000


def test_application_handler_forwards_formatted_record(monkeypatch):
    app = GuiApp()
    monkeypatch.setattr(G, "app", app)
    log.ApplicationLogHandler().emit(make_record("hello %s", ("you",), level=logging.WARNING))
    assert app.logged == [("hello you", logging.WARNING)]


def test_handlers_do_nothing_without_app(monkeypatch):
    monkeypatch.setattr(G, "app", None)
    for cls in (log.SplashLogHandler, log.StatusLogHandler, log.ApplicationLogHandler):
        cls().emit(make_record("ignored", level=logging.ERROR))  # must not raise


def test_logger_class_has_message_and_notice(caplog):
    logger = log.Logger("mh.test.logger")
    logger.addHandler(caplog.handler)
    logger.setLevel(logging.DEBUG)
    logger.message("m %d", 1)
    logger.notice("n %d", 2)
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.INFO, "m 1"), (log.NOTICE, "n 2")]


# --- log.init() -------------------------------------------------------------------

def _is_pytest_handler(handler):
    # pytest swaps its capture handlers on the root logger for every test phase
    return type(handler).__module__.startswith("_pytest")


@pytest.fixture
def isolated_logging():
    """log.init() reconfigures the global logging tree; put everything back."""
    root = logging.getLogger()
    saved_root = ([h for h in root.handlers if not _is_pytest_handler(h)], root.level, root.filters[:])
    saved = {}
    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(lg, logging.Logger):
            saved[name] = (lg.disabled, lg.level, lg.handlers[:], lg.propagate, lg.filters[:])
    logger_class = logging.getLoggerClass()
    yield
    logging.captureWarnings(False)
    logging.setLoggerClass(logger_class)
    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if not isinstance(lg, logging.Logger):
            continue
        for h in lg.handlers:
            if name not in saved or h not in saved[name][2]:
                h.close()
        if name in saved:
            lg.disabled, lg.level, lg.handlers[:], lg.propagate, lg.filters[:] = \
                saved[name][0], saved[name][1], saved[name][2], saved[name][3], saved[name][4]
        else:
            lg.disabled, lg.level, lg.handlers[:], lg.propagate, lg.filters[:] = \
                False, logging.NOTSET, [], True, []
    current_pytest = [h for h in root.handlers if _is_pytest_handler(h)]
    for h in root.handlers:
        if h not in saved_root[0] and not _is_pytest_handler(h):
            h.close()
    root.handlers[:] = saved_root[0] + current_pytest
    root.level, root.filters[:] = saved_root[1], saved_root[2]


@pytest.fixture
def user_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MH_HOME_LOCATION", str(tmp_path))
    import getpath
    user_dir = getpath.getPath("")
    return user_dir


def test_init_prefers_the_users_logging_ini(isolated_logging, user_home, tmp_path):
    os.makedirs(user_home)
    out = tmp_path / "user.log"
    with open(os.path.join(user_home, "logging.ini"), "w") as f:
        f.write("[loggers]\nkeys=root\n[handlers]\nkeys=f\n[formatters]\nkeys=\n"
                "[handler_f]\nclass=FileHandler\nlevel=DEBUG\nargs=(%r, 'w')\n"
                "[logger_root]\nlevel=DEBUG\nhandlers=f\n" % str(out))
    log.init()
    assert logging.getLoggerClass() is log.Logger
    log.message("hello user config")
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("captured warning")
    for h in logging.getLogger().handlers:
        h.flush()
    text = out.read_text()
    assert "hello user config" in text
    assert "captured warning" in text  # captureWarnings(True) routes warnings to logging
    assert any(isinstance(f, log.NoiseFilter) for f in logging.getLogger("OpenGL.formathandler").filters)
    assert any(isinstance(f, log.DowngradeFilter) for f in logging.getLogger("OpenGL.extensions").filters)


def test_init_falls_back_to_the_system_logging_ini(isolated_logging, user_home, monkeypatch):
    os.makedirs(user_home)
    with open(os.path.join(user_home, "logging.ini"), "w") as f:
        f.write("this is not an ini file")
    monkeypatch.chdir(MH_DIR)
    log.init()
    kinds = {type(h) for h in logging.getLogger().handlers}
    assert {log.SplashLogHandler, log.ApplicationLogHandler, log.StatusLogHandler,
            logging.FileHandler} <= kinds
    log.message("into the system log file")
    for h in logging.getLogger().handlers:
        h.flush()
    with open(os.path.join(user_home, "makehuman.log")) as f:
        assert "into the system log file" in f.read()
    assert logging.getLogger("mh.callEvent").level == logging.WARNING


def test_init_falls_back_to_basic_config(isolated_logging, user_home, tmp_path, monkeypatch):
    # No user folder (so neither ini can open its log file) and no data/logging.ini
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    root.handlers[:] = []
    root.setLevel(logging.WARNING)
    log.init()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1 and isinstance(root.handlers[0], logging.StreamHandler)


def test_init_survives_when_no_configuration_works(isolated_logging, user_home, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def broken(**kwargs):
        raise ValueError("no logging for you")

    monkeypatch.setattr(logging, "basicConfig", broken)
    log.init()  # must not raise: MakeHuman starts even without logging
    assert logging.getLoggerClass() is log.Logger


def test_splitpath_returns_every_component():
    assert log._splitpath("/usr/local/lib") == ["/", "usr", "local", "lib"]
    assert log._splitpath("a/b") == ["", "a", "b"]


def test_plain_notice_when_caller_transparent_version_cannot_compile(monkeypatch, caplog):
    """If compiling notice() into the logging module fails, log.py defines plain versions."""
    import code
    import runpy

    def refuse(*args, **kwargs):
        raise SyntaxError("simulated")

    monkeypatch.setattr(code, "compile_command", refuse)
    ns = runpy.run_path(log.__file__, run_name="log_without_compile")
    with caplog.at_level(logging.DEBUG):
        ns["notice"]("plain %s", "notice")
        logger = ns["Logger"]("mh.test.fallback")
        logger.addHandler(caplog.handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.notice("method %s", "notice")
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (log.NOTICE, "plain notice"), (log.NOTICE, "method notice")]
    # the plain version reports itself, not the caller: that is the price of the fallback
    assert caplog.records[0].filename == "log.py"
