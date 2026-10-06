"""The batch widgets show the project directory and keep it in sync."""

import json
from unittest.mock import MagicMock, patch

import pytest

from napari_roxas_ai._measurements import _batch_sample_measurements
from napari_roxas_ai._segmentation import _batch_sample_segmentation
from napari_roxas_ai._settings import DEFAULT_SETTINGS, SettingsManager


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """An isolated settings.json, so the tests never touch the real one."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(DEFAULT_SETTINGS, indent=4), encoding="utf-8")
    monkeypatch.setattr(SettingsManager, "_instance", None)
    monkeypatch.setattr(SettingsManager, "_settings_file", path)
    manager = SettingsManager()
    manager.set("project_directory", str(tmp_path))
    monkeypatch.setattr(_batch_sample_measurements, "settings", manager)
    monkeypatch.setattr(_batch_sample_segmentation, "settings", manager)
    return manager


def _measurements_widget():
    return _batch_sample_measurements.BatchSampleMeasurementsWidget(MagicMock())


def _segmentation_widget():
    module = _batch_sample_segmentation
    with patch.object(module, "check_assets_and_download"), patch.object(
        module.BatchSampleSegmentationWidget,
        "_get_model_files",
        return_value=("model.pth",),
    ):
        return module.BatchSampleSegmentationWidget(MagicMock())


@pytest.fixture(params=[_measurements_widget, _segmentation_widget])
def batch_widget(request, settings):
    return request.param()


def test_batch_widget_shows_the_project_directory(batch_widget, tmp_path):
    assert batch_widget.input_directory_path == str(tmp_path)
    assert (
        batch_widget._input_file_dialog_button.text
        == f"Project Directory: {tmp_path}"
    )


def test_batch_widget_follows_a_change_made_elsewhere(
    batch_widget, settings, tmp_path
):
    other = str(tmp_path / "other")

    settings.set("project_directory", other)

    assert batch_widget.input_directory_path == other
    assert (
        batch_widget._input_file_dialog_button.text
        == f"Project Directory: {other}"
    )


def test_batch_widget_choice_becomes_the_project_directory(
    batch_widget, settings, tmp_path
):
    other = str(tmp_path / "other")
    module = __import__(type(batch_widget).__module__, fromlist=["QFileDialog"])

    with patch.object(
        module.QFileDialog, "getExistingDirectory", return_value=other
    ):
        batch_widget._open_input_file_dialog()

    assert settings.get("project_directory") == other
    assert batch_widget.input_directory_path == other
    on_disk = json.loads(settings.settings_file.read_text(encoding="utf-8"))
    assert on_disk["project_directory"] == other
