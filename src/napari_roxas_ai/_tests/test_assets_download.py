"""
Tests for the lazy download of asset files (e.g. model weights).
"""

from unittest.mock import patch

from napari_roxas_ai._assets_files import _assets_dowloader
from napari_roxas_ai._assets_files._assets_dowloader import (
    check_assets_and_download,
)
from napari_roxas_ai._edition import _rings_layer_editor
from napari_roxas_ai._edition._rings_layer_editor import (
    RingsLayerEditorWidget,
)


def test_check_assets_skips_download_when_directory_has_files(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"weights")

    with patch.object(
        _assets_dowloader, "download_and_decompress_file"
    ) as mock_download:
        check_assets_and_download(str(tmp_path), "rings_models.zip")

    mock_download.assert_not_called()


def test_check_assets_downloads_when_directory_missing(tmp_path):
    directory = tmp_path / "_rings"

    with patch.object(
        _assets_dowloader, "get_asset_file_url", return_value="url"
    ), patch.object(
        _assets_dowloader, "download_and_decompress_file"
    ) as mock_download:
        check_assets_and_download(str(directory), "rings_models.zip")

    mock_download.assert_called_once_with("url", str(directory))


def test_check_assets_downloads_when_directory_empty(tmp_path):
    # An empty directory is left behind by an interrupted download
    with patch.object(
        _assets_dowloader, "get_asset_file_url", return_value="url"
    ), patch.object(
        _assets_dowloader, "download_and_decompress_file"
    ) as mock_download:
        check_assets_and_download(str(tmp_path), "rings_models.zip")

    mock_download.assert_called_once_with("url", str(tmp_path))


def test_rings_editor_model_files_downloads_missing_models(tmp_path):
    models_path = tmp_path / "_rings"

    def fake_download(directory, asset_name):
        models_path.mkdir()
        (models_path / "rings_model.pt").write_bytes(b"weights")

    with patch.object(
        _rings_layer_editor, "RINGS_MODELS_PATH", models_path
    ), patch.object(
        _rings_layer_editor,
        "check_assets_and_download",
        side_effect=fake_download,
    ):
        choices = RingsLayerEditorWidget._get_rings_model_files(None)

    assert choices == ("rings_model.pt",)


def test_rings_editor_model_files_survives_failed_download(tmp_path):
    models_path = tmp_path / "_rings"

    with patch.object(
        _rings_layer_editor, "RINGS_MODELS_PATH", models_path
    ), patch.object(
        _rings_layer_editor,
        "check_assets_and_download",
        side_effect=ConnectionError("offline"),
    ), patch.object(
        _rings_layer_editor, "show_warning"
    ) as mock_warning:
        choices = RingsLayerEditorWidget._get_rings_model_files(None)

    assert choices == ()
    mock_warning.assert_called_once()
