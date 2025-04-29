"""
Tests for the writer module functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from napari_roxas_ai._writer._writer import (
    save_image,
    update_metadata_file,
    write_cells_file,
    write_multiple_layers,
    write_rings_file,
    write_scan_file,
    write_single_layer,
)


# Helper function to create test metadata
def create_test_metadata(metadata=None):
    """Create test metadata dictionary."""
    if metadata is None:
        metadata = {
            "sample_name": "test_sample",
            "sample_scale": 2.5,
            "sample_type": "conifer",
            "scan_date": "2023-01-01",
            "cells_count": 100,
            "rings_count": 20,
        }
    return metadata


# Helper function to create test features dataframe
def create_test_features(rows=5):
    """Create test features dataframe."""
    data = {
        "area": np.random.randint(100, 500, size=rows),
        "perimeter": np.random.randint(40, 100, size=rows),
        "circularity": np.random.random(size=rows),
    }
    return pd.DataFrame(data)


# Fixture for temporary directory
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


# Mock SettingsManager for testing
@pytest.fixture
def mock_settings():
    """Create a mock settings manager."""
    with patch("napari_roxas_ai._writer._writer.settings") as mock_settings:
        # Configure mock settings
        mock_settings.get.side_effect = lambda key, default=None: {
            "file_extensions.scan_file_extension": [".scan"],
            "file_extensions.cells_file_extension": [".cells"],
            "file_extensions.rings_file_extension": [".rings"],
            "file_extensions.metadata_file_extension": [".metadata", ".json"],
            "file_extensions.cells_table_file_extension": [".cells", ".csv"],
            "file_extensions.rings_table_file_extension": [".rings", ".csv"],
            "tables.separator": "\t",
            "tables.index_column": "index",
        }.get(key, default)

        yield mock_settings


class TestWriterModule:
    """Tests for the writer module."""

    def test_update_metadata_file_new_file(self, temp_dir):
        """Test update_metadata_file when file doesn't exist."""
        # Setup test data
        metadata_path = temp_dir / "test_sample.metadata.json"
        metadata = create_test_metadata()

        # Test creating a new metadata file
        with patch("pathlib.Path.exists", return_value=False), patch(
            "builtins.open", create=True
        ), patch("json.dump") as mock_dump:

            result = update_metadata_file(
                str(metadata_path), metadata, "scan_"
            )

            # Verify results
            assert result == str(metadata_path)
            assert mock_dump.called

    def test_update_metadata_file_existing_file(self, temp_dir):
        """Test update_metadata_file when file already exists."""
        # Setup test data
        metadata_path = temp_dir / "test_sample.metadata.json"
        original_metadata = {
            "sample_name": "original_name",
            "sample_scale": 1.0,
        }

        # Mock file operations
        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", create=True
        ), patch("json.load", return_value=original_metadata), patch(
            "json.dump"
        ) as mock_dump:

            # New metadata to add
            new_metadata = {"scan_date": "2023-01-01", "sample_scale": 2.5}

            # Test updating an existing metadata file
            result = update_metadata_file(
                str(metadata_path), new_metadata, "scan_"
            )

            # Verify results
            assert result == str(metadata_path)
            assert mock_dump.called

    def test_save_image_without_rescale(self, temp_dir):
        """Test save_image function without rescaling."""
        # Setup test data
        image_path = temp_dir / "test_image.tif"
        image_data = np.ones((100, 100), dtype=np.uint8) * 128

        # Test saving image without rescale
        with patch("PIL.Image.fromarray") as mock_fromarray:
            mock_image = MagicMock()
            mock_fromarray.return_value = mock_image

            result = save_image(str(image_path), image_data, rescale=False)

            # Verify PIL.Image.fromarray was called with the correct array
            mock_fromarray.assert_called_once()
            # Verify save was called on the PIL Image
            mock_image.save.assert_called_once_with(str(image_path))

            assert result == str(image_path)

    def test_save_image_with_rescale(self, temp_dir):
        """Test save_image function with rescaling."""
        # Setup test data
        image_path = temp_dir / "test_image.tif"
        image_data = np.random.randint(0, 1000, (100, 100), dtype=np.int16)

        # Test saving image with rescale
        with patch("PIL.Image.fromarray") as mock_fromarray, patch(
            "numpy.min", return_value=0
        ), patch("numpy.max", return_value=1000):

            mock_image = MagicMock()
            mock_fromarray.return_value = mock_image

            result = save_image(str(image_path), image_data, rescale=True)

            # Verify PIL.Image.fromarray was called
            mock_fromarray.assert_called_once()
            # Verify save was called on the PIL Image
            mock_image.save.assert_called_once_with(str(image_path))

            assert result == str(image_path)

    def test_write_scan_file(self, temp_dir, mock_settings):
        """Test write_scan_file function."""
        # Setup test data
        path = temp_dir / "test_sample.scan"
        data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        meta = {
            "name": "test_scan",
            "metadata": {
                "sample_name": "test_sample",
                "sample_scale": 2.5,
                "scan_date": "2023-01-01",
            },
        }

        # Mock dependent functions
        with patch(
            "napari_roxas_ai._writer._writer.update_metadata_file"
        ) as mock_update_metadata, patch(
            "napari_roxas_ai._writer._writer.save_image"
        ) as mock_save_image:

            mock_update_metadata.return_value = str(
                temp_dir / "test_sample.metadata.json"
            )
            mock_save_image.return_value = str(temp_dir / "test_sample.scan")

            # Test writing scan file
            result = write_scan_file(str(path), data, meta)

            # Verify results
            assert (
                len(result) == 2
            )  # Should return paths to metadata and image files
            assert mock_update_metadata.called
            assert mock_save_image.called

    def test_write_cells_file(self, temp_dir, mock_settings):
        """Test write_cells_file function."""
        # Setup test data
        path = temp_dir / "test_sample.cells"
        data = np.random.randint(0, 10, (100, 100), dtype=np.uint8)
        features = create_test_features()
        meta = {
            "name": "test_cells",
            "features": features,
            "metadata": {
                "sample_name": "test_sample",
                "cells_count": 100,
            },
        }

        # Mock dependent functions
        with patch(
            "napari_roxas_ai._writer._writer.update_metadata_file"
        ) as mock_update_metadata, patch(
            "napari_roxas_ai._writer._writer.save_image"
        ) as mock_save_image:

            mock_update_metadata.return_value = str(
                temp_dir / "test_sample.metadata.json"
            )
            mock_save_image.return_value = str(temp_dir / "test_sample.cells")

            # Test writing cells file
            result = write_cells_file(str(path), data, meta)

            # Verify results
            assert (
                len(result) == 3
            )  # Should return paths to metadata, features and image files
            assert mock_update_metadata.called
            assert mock_save_image.called

    def test_write_rings_file(self, temp_dir, mock_settings):
        """Test write_rings_file function."""
        # Setup test data
        path = temp_dir / "test_sample.rings"
        data = np.random.randint(0, 10, (100, 100), dtype=np.uint8)
        features = create_test_features()
        meta = {
            "name": "test_rings",
            "features": features,
            "metadata": {
                "sample_name": "test_sample",
                "rings_count": 20,
            },
        }

        # Mock dependent functions
        with patch(
            "napari_roxas_ai._writer._writer.update_metadata_file"
        ) as mock_update_metadata, patch(
            "napari_roxas_ai._writer._writer.save_image"
        ) as mock_save_image:

            mock_update_metadata.return_value = str(
                temp_dir / "test_sample.metadata.json"
            )
            mock_save_image.return_value = str(temp_dir / "test_sample.rings")

            # Test writing rings file
            result = write_rings_file(str(path), data, meta)

            # Verify results
            assert (
                len(result) == 3
            )  # Should return paths to metadata, features and image files
            assert mock_update_metadata.called
            assert mock_save_image.called

    def test_write_single_layer_scan(self, temp_dir, mock_settings):
        """Test write_single_layer with scan layer."""
        # Setup test data
        path = temp_dir / "test_sample"
        data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        meta = {
            "name": "test_sample.scan",
            "metadata": {
                "sample_name": "test_sample",
                "scan_date": "2023-01-01",
            },
        }

        # Mock dependent functions
        with patch(
            "napari_roxas_ai._writer._writer.write_scan_file"
        ) as mock_write_scan:
            mock_write_scan.return_value = [
                str(temp_dir / "test_sample.metadata.json"),
                str(temp_dir / "test_sample.scan"),
            ]

            # Test writing single layer
            result = write_single_layer(str(path), data, meta)

            # Verify results
            assert len(result) == 2
            assert mock_write_scan.called

    def test_write_single_layer_cells(self, temp_dir, mock_settings):
        """Test write_single_layer with cells layer."""
        # Setup test data
        path = temp_dir / "test_sample"
        data = np.random.randint(0, 10, (100, 100), dtype=np.uint8)
        meta = {
            "name": "test_sample.cells",
            "features": create_test_features(),
            "metadata": {
                "sample_name": "test_sample",
                "cells_count": 100,
            },
        }

        # Mock dependent functions
        with patch(
            "napari_roxas_ai._writer._writer.write_cells_file"
        ) as mock_write_cells:
            mock_write_cells.return_value = [
                str(temp_dir / "test_sample.metadata.json"),
                str(temp_dir / "test_sample.cells.csv"),
                str(temp_dir / "test_sample.cells"),
            ]

            # Test writing single layer
            result = write_single_layer(str(path), data, meta)

            # Verify results
            assert len(result) == 3
            assert mock_write_cells.called

    def test_write_single_layer_rings(self, temp_dir, mock_settings):
        """Test write_single_layer with rings layer."""
        # Setup test data
        path = temp_dir / "test_sample"
        data = np.random.randint(0, 10, (100, 100), dtype=np.uint8)
        meta = {
            "name": "test_sample.rings",
            "features": create_test_features(),
            "metadata": {
                "sample_name": "test_sample",
                "rings_count": 20,
            },
        }

        # Mock dependent functions
        with patch(
            "napari_roxas_ai._writer._writer.write_rings_file"
        ) as mock_write_rings:
            mock_write_rings.return_value = [
                str(temp_dir / "test_sample.metadata.json"),
                str(temp_dir / "test_sample.rings.csv"),
                str(temp_dir / "test_sample.rings"),
            ]

            # Test writing single layer
            result = write_single_layer(str(path), data, meta)

            # Verify results
            assert len(result) == 3
            assert mock_write_rings.called

    def test_write_multiple_layers(self, temp_dir, mock_settings):
        """Test write_multiple_layers function."""
        # Setup test data
        path = temp_dir / "test_sample"

        # Create layer data
        scan_data = (
            np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            {
                "name": "test_sample.scan",
                "metadata": {
                    "sample_name": "test_sample",
                    "scan_date": "2023-01-01",
                },
            },
            "image",
        )

        cells_data = (
            np.random.randint(0, 10, (100, 100), dtype=np.uint8),
            {
                "name": "test_sample.cells",
                "features": create_test_features(),
                "metadata": {"sample_name": "test_sample", "cells_count": 100},
            },
            "labels",
        )

        layers_data = [scan_data, cells_data]

        # Mock dependent functions
        with patch(
            "napari_roxas_ai._writer._writer.write_single_layer"
        ) as mock_write_single:
            mock_write_single.side_effect = [
                [
                    str(temp_dir / "test_sample.metadata.json"),
                    str(temp_dir / "test_sample.scan"),
                ],
                [
                    str(temp_dir / "test_sample.metadata.json"),
                    str(temp_dir / "test_sample.cells.csv"),
                    str(temp_dir / "test_sample.cells"),
                ],
            ]

            # Test writing multiple layers
            result = write_multiple_layers(str(path), layers_data)

            # Verify results are unique and contain all paths
            assert len(result) > 0  # At least one path should be returned
            assert mock_write_single.call_count == 2


if __name__ == "__main__":
    pytest.main(["-v", __file__])
