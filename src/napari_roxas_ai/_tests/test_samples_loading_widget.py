import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qtpy.QtWidgets import QAbstractItemView

from napari_roxas_ai._loading import _samples_loading_widget
from napari_roxas_ai._loading._samples_loading_widget import (
    MAX_PROJECT_DIRECTORY_CHARS,
    MAX_VISIBLE_IMAGE_ROWS,
    MIN_VISIBLE_IMAGE_ROWS,
    SamplesLoadingWidget,
    _elide_path,
)
from napari_roxas_ai._settings import DEFAULT_SETTINGS, SettingsManager


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """An isolated settings.json, so the tests never touch the real one."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(DEFAULT_SETTINGS, indent=4), encoding="utf-8")
    monkeypatch.setattr(SettingsManager, "_instance", None)
    monkeypatch.setattr(SettingsManager, "_settings_file", path)
    manager = SettingsManager()
    monkeypatch.setattr(_samples_loading_widget, "settings", manager)
    return manager


@pytest.fixture
def project(tmp_path):
    """A project directory with two prepared samples."""
    directory = tmp_path / "project"
    directory.mkdir()
    for name in ("sample_1", "sample_2"):
        (directory / f"{name}.metadata.json").write_text("{}")
    return directory


@pytest.fixture
def widget(make_napari_viewer, settings, project):
    settings.set("project_directory", str(project))
    return SamplesLoadingWidget(make_napari_viewer())


# --- SettingsManager listeners ---


def test_listener_is_called_with_the_new_value(settings):
    callback = MagicMock()
    settings.add_listener("project_directory", callback)

    settings.set("project_directory", "/a/b")

    callback.assert_called_once_with("/a/b")


def test_listener_is_not_called_when_the_value_stays_the_same(settings):
    settings.set("project_directory", "/a/b")
    callback = MagicMock()
    settings.add_listener("project_directory", callback)

    settings.set("project_directory", "/a/b")
    settings.set("processing.try_to_use_gpu", True)

    callback.assert_not_called()


def test_listener_follows_replace_and_reset(settings):
    callback = MagicMock()
    settings.add_listener("project_directory", callback)

    settings.replace({**settings.as_dict(), "project_directory": "/a/b"})
    settings.reset()

    assert [c.args for c in callback.call_args_list] == [("/a/b",), (None,)]


def test_failing_listener_does_not_stop_the_others(settings):
    broken = MagicMock(side_effect=RuntimeError("boom"))
    working = MagicMock()
    settings.add_listener("project_directory", broken)
    settings.add_listener("project_directory", working)

    settings.set("project_directory", "/a/b")

    working.assert_called_once_with("/a/b")


def test_listener_does_not_keep_its_owner_alive(settings):
    class Owner:
        calls = 0

        def on_change(self, value):
            Owner.calls += 1

    owner = Owner()
    settings.add_listener("project_directory", owner.on_change)
    del owner

    settings.set("project_directory", "/a/b")

    assert Owner.calls == 0


# --- Path eliding ---


def test_short_path_is_shown_as_is():
    assert _elide_path("C:/Users/vonarx/ROXAS AI") == "C:/Users/vonarx/ROXAS AI"


def test_long_path_is_elided_in_the_middle():
    path = "C:/Users/vonarx/Desktop/" + "x" * 80 + "/ROXAS AI"

    elided = _elide_path(path)

    assert len(elided) == MAX_PROJECT_DIRECTORY_CHARS
    assert elided.startswith("C:/Users/vonarx/")
    assert elided.endswith("/ROXAS AI")
    assert "…" in elided


# --- Widget ---


def test_widget_layout(widget, project):
    assert widget[0] is widget._project_directory_row
    assert widget._project_directory_row[0].value == "Project Directory:"
    assert widget._project_dialog_button.text == _elide_path(str(project))
    assert widget._project_dialog_button.tooltip == str(project)
    assert widget._samples_selection_row[0].value == "Available Images:"
    assert widget._load_samples_button.text == "Load Selected Image"
    assert not hasattr(widget, "_select_all_button")
    assert not hasattr(widget, "_reverse_selection_button")
    assert list(widget._sample_select_widget.choices) == ["sample_1", "sample_2"]


def test_long_project_directory_is_elided_with_full_tooltip(
    make_napari_viewer, settings, tmp_path
):
    directory = tmp_path / ("x" * 80)
    directory.mkdir()
    settings.set("project_directory", str(directory))

    widget = SamplesLoadingWidget(make_napari_viewer())

    assert len(widget._project_dialog_button.text) == MAX_PROJECT_DIRECTORY_CHARS
    assert widget._project_dialog_button.tooltip == str(directory)


def test_project_directory_changed_elsewhere_is_shown(widget, settings, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "sample_3.metadata.json").write_text("{}")

    settings.set("project_directory", str(other))

    assert widget.project_directory == str(other)
    assert widget._project_dialog_button.tooltip == str(other)
    assert list(widget._sample_select_widget.choices) == ["sample_3"]


def test_choosing_a_project_directory_stores_it_in_the_settings(
    widget, settings, tmp_path
):
    with patch.object(
        _samples_loading_widget.QFileDialog,
        "getExistingDirectory",
        return_value=str(tmp_path),
    ):
        widget._open_project_dialog()

    assert settings.get("project_directory") == str(tmp_path)
    assert widget.project_directory == str(tmp_path)
    on_disk = json.loads(settings.settings_file.read_text(encoding="utf-8"))
    assert on_disk["project_directory"] == str(tmp_path)


def test_only_one_image_can_be_selected(widget):
    assert (
        widget._sample_select_widget.native.selectionMode()
        == QAbstractItemView.SingleSelection
    )


def test_loading_is_refused_while_an_image_is_open(widget):
    widget._viewer.add_image(
        np.zeros((5, 5)), metadata={"sample_stem_path": "sample_1"}
    )
    widget._sample_select_widget.value = "sample_2"

    with patch.object(
        _samples_loading_widget.QMessageBox, "information"
    ) as mock_info, patch.object(_samples_loading_widget, "Worker") as mock_worker:
        widget._load_selected_samples()

    mock_info.assert_called_once()
    assert "sample_1" in mock_info.call_args.args[2]
    mock_worker.assert_not_called()


def test_loading_without_selection_does_nothing(widget):
    with patch.object(_samples_loading_widget, "Worker") as mock_worker:
        widget._load_selected_samples()

    mock_worker.assert_not_called()


def test_double_click_loads_the_image(widget):
    list_widget = widget._sample_select_widget.native
    widget._load_selected_samples = MagicMock()

    list_widget.itemDoubleClicked.emit(list_widget.item(0))

    widget._load_selected_samples.assert_called_once_with()


def test_image_list_is_at_least_five_rows_high(widget):
    list_widget = widget._sample_select_widget.native
    row_height = list_widget.sizeHintForRow(0)
    frame = list_widget.frameWidth()

    # Two images, still five rows
    assert list_widget.height() == 5 * row_height + 2 * frame
    assert (
        widget._available_images_label.native.height() == row_height + 2 * frame
    )


def test_image_list_grows_with_more_than_five_images(widget, settings, project):
    for i in range(3, 8):
        (project / f"sample_{i}.metadata.json").write_text("{}")
    widget._refresh_samples_list()
    list_widget = widget._sample_select_widget.native

    assert list_widget.count() == 7
    assert (
        list_widget.height()
        == 7 * list_widget.sizeHintForRow(0) + 2 * list_widget.frameWidth()
    )


def test_image_list_caps_at_max_visible_rows(widget, settings, project):
    for i in range(3, 16):
        (project / f"sample_{i}.metadata.json").write_text("{}")
    widget._refresh_samples_list()
    list_widget = widget._sample_select_widget.native

    assert list_widget.count() == 15
    assert (
        list_widget.height()
        == MAX_VISIBLE_IMAGE_ROWS * list_widget.sizeHintForRow(0)
        + 2 * list_widget.frameWidth()
    )


def test_project_directory_button_and_image_list_line_up(widget, qtbot):
    widget.native.resize(420, 300)
    widget.native.show()
    qtbot.waitExposed(widget.native)
    button = widget._project_dialog_button.native
    image_list = widget._sample_select_widget.native

    def left_and_width(box):
        return box.mapTo(widget.native, box.rect().topLeft()).x(), box.width()

    assert left_and_width(button) == left_and_width(image_list)
