"""Notifications about files written to a project directory."""

from unittest.mock import MagicMock, patch

import pytest

from napari_roxas_ai._utils import _project_files_events as events


@pytest.fixture(autouse=True)
def no_listeners(monkeypatch):
    """Isolate the tests from listeners of widgets created elsewhere."""
    monkeypatch.setattr(events, "_listeners", [])


class _Listener:
    def __init__(self):
        self.calls = []

    def on_changed(self, directory):
        self.calls.append(directory)


def test_listeners_are_told_the_directory(tmp_path):
    first, second = _Listener(), _Listener()
    events.add_project_files_listener(first.on_changed)
    events.add_project_files_listener(second.on_changed)

    events.notify_project_files_changed(tmp_path)

    assert first.calls == [str(tmp_path)]
    assert second.calls == [str(tmp_path)]


def test_the_source_is_not_told_about_its_own_files(tmp_path):
    source, other = _Listener(), _Listener()
    events.add_project_files_listener(source.on_changed)
    events.add_project_files_listener(other.on_changed)

    events.notify_project_files_changed(tmp_path, source=source)

    assert source.calls == []
    assert other.calls == [str(tmp_path)]


def test_without_directory_nobody_is_told():
    listener = _Listener()
    events.add_project_files_listener(listener.on_changed)

    events.notify_project_files_changed(None)

    assert listener.calls == []


def test_listeners_of_deleted_widgets_are_dropped(tmp_path):
    listener = _Listener()
    events.add_project_files_listener(listener.on_changed)
    del listener

    events.notify_project_files_changed(tmp_path)

    assert events._listeners == []


def test_a_failing_listener_does_not_stop_the_others(tmp_path):
    def failing(directory):
        raise RuntimeError("broken")

    listener = _Listener()
    events.add_project_files_listener(failing)
    events.add_project_files_listener(listener.on_changed)

    events.notify_project_files_changed(tmp_path)

    assert listener.calls == [str(tmp_path)]


def test_is_same_directory(tmp_path):
    assert events.is_same_directory(tmp_path, str(tmp_path / "sub" / ".."))
    assert not events.is_same_directory(tmp_path, tmp_path / "sub")
    assert not events.is_same_directory(None, tmp_path)


def test_preparation_tells_the_listeners_when_done(tmp_path):
    from napari_roxas_ai._preparation import _preparation_widget

    widget = MagicMock()
    widget.project_directory = str(tmp_path)
    widget.worker.should_stop = False
    with patch.object(_preparation_widget, "QMessageBox"), patch.object(
        _preparation_widget, "notify_project_files_changed"
    ) as notify:
        _preparation_widget.PreparationWidget._processing_finished(widget)

    notify.assert_called_once_with(str(tmp_path), source=widget)
