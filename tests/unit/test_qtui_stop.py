"""Quitting must leave the Qt event loop instead of finalising Python inside a slot."""

import qtui


class FakeApp(object):
    def __init__(self):
        self.calls = []
        app = self

        class Messages(object):
            def stop(self):
                app.calls.append("messages.stop")

        self.messages = Messages()

    def callEvent(self, name, event):
        self.calls.append(name)

    def quit(self):
        self.calls.append("quit")


def test_stop_quits_the_event_loop_without_sys_exit():
    app = FakeApp()
    qtui.Application.stop(app)  # upstream raised SystemExit here
    assert app.calls == ["messages.stop", "onStop", "quit"]


def test_stop_runs_only_once():
    app = FakeApp()
    qtui.Application.stop(app)
    qtui.Application.stop(app)
    assert app.calls == ["messages.stop", "onStop", "quit"]
