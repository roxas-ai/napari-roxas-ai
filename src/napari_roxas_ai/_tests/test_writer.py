"""
Tests for the writer module functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from napari_roxas_ai._writer._writer import (
    interpolate_row_at_col,
    round_column,
    round_column_sci,
    save_annotated_scan_image,
    save_image,
    save_sample_annotated_image,
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
            "spatial_resolution": 2.5,
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
            "file_extensions.image_file_extensions": [".jpg", ".jpeg", ".png"],
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
            "spatial_resolution": 1.0,
        }

        # Write an actual existing metadata file
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(original_metadata, f)

        # New metadata to add
        new_metadata = {"scan_date": "2023-01-01", "spatial_resolution": 2.5}

        # Test updating an existing metadata file
        result = update_metadata_file(
            str(metadata_path), new_metadata, "scan_"
        )

        # Verify results
        assert result == str(metadata_path)

        # Read back and verify the file was updated
        with open(metadata_path, "r", encoding="utf-8") as f:
            saved_metadata = json.load(f)
        # spatial_resolution is not updated because it doesn't match prefix "scan_"
        assert saved_metadata["spatial_resolution"] == 1.0
        assert saved_metadata["scan_date"] == "2023-01-01"
        # sample_name is derived from filename when not in new_metadata
        assert saved_metadata["sample_name"] == "test_sample"

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
            mock_image.save.assert_called_once_with(
                str(image_path), compression="tiff_deflate"
            )

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
            mock_image.save.assert_called_once_with(
                str(image_path), compression="tiff_deflate"
            )

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
                "spatial_resolution": 2.5,
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

            # Only the metadata is written, the scan image stays untouched
            assert len(result) == 1
            assert mock_update_metadata.called
            assert not mock_save_image.called

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

    def test_round_column_sci(self):
        """Test round_column_sci formatting and handling of values."""
        df = pd.DataFrame({
            "col_sci": [0.0001234567, 123456.789, 0.000000005079, np.nan, "invalid"],
            "col_other": [1, 2, 3, 4, 5]
        })
        result = round_column_sci(df.copy(), "col_sci", 3)
        assert result.loc[0, "col_sci"] == "1.235E-04"
        assert result.loc[1, "col_sci"] == "1.235E+05"
        assert result.loc[2, "col_sci"] == "5.079E-09"
        assert np.isnan(result.loc[3, "col_sci"]) or pd.isna(result.loc[3, "col_sci"])
        assert np.isnan(result.loc[4, "col_sci"]) or pd.isna(result.loc[4, "col_sci"])
        # Column that doesn't exist should be ignored without error
        result_no_change = round_column_sci(df.copy(), "non_existent", 3)
        assert "non_existent" not in result_no_change.columns

    def test_save_annotated_scan_image_with_cells_default_100_percent(self, temp_dir):
        """Test save_annotated_scan_image overlays cells and defaults to 100% scale."""
        from PIL import Image

        scan_path = temp_dir / "sample.scan.jpg"
        annotated_path = temp_dir / "sample_annotated.jpg"

        # Create 200x200 RGB scan image (white background)
        scan_arr = np.full((200, 200, 3), 255, dtype=np.uint8)
        Image.fromarray(scan_arr).save(scan_path)

        # Create cells data with some cell lumina (1s in center)
        cells_data = np.zeros((200, 200), dtype=np.uint8)
        cells_data[80:120, 80:120] = 1

        # Create rings features
        rings_features = pd.DataFrame({
            "YEAR": [2020, 2021],
            "RBXY": [
                [[50, 0], [50, 100], [50, 199]],
                [[150, 0], [150, 100], [150, 199]],
            ],
            "cells_above": [5000, 15000],
        })

        save_annotated_scan_image(
            scan_path=str(scan_path),
            annotated_path=str(annotated_path),
            rings_features=rings_features,
            cells_data=cells_data,
        )

        assert annotated_path.exists()
        with Image.open(annotated_path) as img:
            # Default scale is 1.0 (100%) -> (200, 200)
            assert img.size == (200, 200)
            img_arr = np.array(img)
            # Center pixels should have cell fill color blended (cyan hue: high G and B, lower R)
            center_color = img_arr[100, 100]
            assert center_color[0] < center_color[1] or center_color[0] < center_color[2]

    def test_save_annotated_scan_image_with_custom_downscale(self, temp_dir):
        """Test save_annotated_scan_image respects custom scale_factor setting."""
        from PIL import Image
        from napari_roxas_ai._writer import _writer

        scan_path = temp_dir / "sample_scaled.scan.jpg"
        annotated_path = temp_dir / "sample_scaled_annotated.jpg"

        scan_arr = np.full((200, 200, 3), 255, dtype=np.uint8)
        Image.fromarray(scan_arr).save(scan_path)

        rings_features = pd.DataFrame({
            "YEAR": [2020],
            "RBXY": [[[50, 0], [50, 199]]],
            "cells_above": [5000],
        })

        with patch.object(_writer.settings, "get", side_effect=lambda k, d=None: 0.75 if k == "annotated_image.scale_factor" else d):
            save_annotated_scan_image(
                scan_path=str(scan_path),
                annotated_path=str(annotated_path),
                rings_features=rings_features,
            )

        assert annotated_path.exists()
        with Image.open(annotated_path) as img:
            assert img.size == (150, 150)

    def test_save_sample_annotated_image_auto_loads_cells_file(self, temp_dir):
        """Test save_sample_annotated_image loads <sample>.cells.png from disk."""
        from PIL import Image

        sample_stem = temp_dir / "tree1"
        scan_path = temp_dir / "tree1.scan.jpg"
        cells_path = temp_dir / "tree1.cells.png"
        annotated_path = temp_dir / "tree1_annotated.jpg"

        scan_arr = np.full((100, 100, 3), 200, dtype=np.uint8)
        Image.fromarray(scan_arr).save(scan_path)

        cells_arr = np.zeros((100, 100), dtype=np.uint8)
        cells_arr[40:60, 40:60] = 1
        Image.fromarray(cells_arr).save(cells_path)

        rings_features = pd.DataFrame({
            "YEAR": [2022],
            "RBXY": [[[50, 0], [50, 99]]],
            "cells_above": [5000],
        })

        result_path = save_sample_annotated_image(sample_stem, rings_features)
        assert result_path == str(annotated_path)
        assert annotated_path.exists()
        with Image.open(annotated_path) as img:
            assert img.size == (100, 100)  # 100% default scale


    def test_interpolate_row_at_col(self):
        """Test interpolate_row_at_col helper on horizontal, diagonal, and reversed polylines."""
        # Horizontal line at row=50 from col=0 to col=100
        line_h = [[50.0, 0.0], [50.0, 100.0]]
        assert interpolate_row_at_col(line_h, 50.0) == pytest.approx(50.0)
        assert interpolate_row_at_col(line_h, 0.0) == pytest.approx(50.0)
        assert interpolate_row_at_col(line_h, 100.0) == pytest.approx(50.0)

        # Diagonal line from (row=20, col=0) to (row=80, col=100)
        line_diag = [[20.0, 0.0], [80.0, 100.0]]
        assert interpolate_row_at_col(line_diag, 50.0) == pytest.approx(50.0)
        assert interpolate_row_at_col(line_diag, 25.0) == pytest.approx(35.0)

        # Reversed order (col=100 to col=0)
        line_rev = [[80.0, 100.0], [20.0, 0.0]]
        assert interpolate_row_at_col(line_rev, 50.0) == pytest.approx(50.0)

        # Out-of-bounds query should clamp to nearest vertex
        assert interpolate_row_at_col(line_diag, -10.0) == pytest.approx(20.0)
        assert interpolate_row_at_col(line_diag, 150.0) == pytest.approx(80.0)

        # Invalid/empty input
        assert np.isnan(interpolate_row_at_col([], 50.0))
        assert np.isnan(interpolate_row_at_col(None, 50.0))

    def test_save_annotated_scan_image_centers_calendar_years_with_diagonal_rings(
        self, temp_dir
    ):
        """Test save_annotated_scan_image horizontally centers year labels at w/2."""
        from PIL import Image

        scan_path = temp_dir / "sample_diag.scan.jpg"
        annotated_path = temp_dir / "sample_diag_annotated.jpg"

        # Create 200x200 RGB scan image
        scan_arr = np.full((200, 200, 3), 255, dtype=np.uint8)
        Image.fromarray(scan_arr).save(scan_path)

        # Create diagonal ring boundaries with asymmetric vertex distributions
        rings_features = pd.DataFrame({
            "YEAR": [2020, 2021],
            "RBXY": [
                # First boundary tilted: at col=100 (midpoint), row is 60
                [[40.0, 0.0], [50.0, 50.0], [60.0, 100.0], [80.0, 200.0]],
                # Second boundary tilted: at col=100 (midpoint), row is 140
                [[120.0, 0.0], [140.0, 100.0], [150.0, 150.0], [160.0, 200.0]],
            ],
            "cells_above": [6000, 14000],
        })

        placed_texts = []

        import cv2

        orig_put_text = cv2.putText

        def mock_put_text(img, text, org, font_face, font_scale, color, thickness=1, line_type=None):
            placed_texts.append({"text": text, "org": org})
            return orig_put_text(img, text, org, font_face, font_scale, color, thickness, line_type)

        with patch("cv2.putText", side_effect=mock_put_text):
            save_annotated_scan_image(
                scan_path=str(scan_path),
                annotated_path=str(annotated_path),
                rings_features=rings_features,
            )

        assert len(placed_texts) == 2
        # Center column is 200 / 2 = 100.
        # Year 2020 is placed between row 0 and row 60 -> center_r = 30
        # Year 2021 is placed between row 60 and row 140 -> center_r = 100
        # Both labels "2020" and "2021" have the exact same width `tw`, so their x coordinates must be identical!
        assert placed_texts[0]["text"] == "2020"
        assert placed_texts[1]["text"] == "2021"
        assert placed_texts[0]["org"][0] == placed_texts[1]["org"][0]
        # Check vertical placement: year 1 around y ~ 30 + th/2, year 2 around y ~ 100 + th/2
        assert placed_texts[0]["org"][1] < placed_texts[1]["org"][1]

    def test_save_annotated_scan_image_custom_year_label_size(self, temp_dir):
        """Test save_annotated_scan_image respects annotated_image.year_label_size setting."""
        from PIL import Image
        import cv2
        from napari_roxas_ai._writer import _writer

        scan_path = temp_dir / "sample_size.scan.jpg"
        annotated_path = temp_dir / "sample_size_annotated.jpg"

        # Create 200x200 RGB scan image
        scan_arr = np.full((200, 200, 3), 255, dtype=np.uint8)
        Image.fromarray(scan_arr).save(scan_path)

        # 2 rings with large width (band area 20000 -> rw = 100 px width)
        rings_features = pd.DataFrame({
            "YEAR": [2020, 2021],
            "RBXY": [
                [[100.0, 0.0], [100.0, 200.0]],
                [[200.0, 0.0], [200.0, 200.0]],
            ],
            "cells_above": [0, 20000],
        })

        captured_font_scales = []
        orig_put_text = cv2.putText

        def mock_put_text(img, text, org, font_face, font_scale, color, thickness=1, line_type=None):
            captured_font_scales.append(font_scale)
            return orig_put_text(img, text, org, font_face, font_scale, color, thickness, line_type)

        # Case 1: default (20%) -> 0.2 * 100 = 20 px target_h
        with patch("cv2.putText", side_effect=mock_put_text):
            with patch.object(_writer.settings, "get", side_effect=lambda k, d=None: 20 if k == "annotated_image.year_label_size" else d):
                save_annotated_scan_image(
                    scan_path=str(scan_path),
                    annotated_path=str(annotated_path),
                    rings_features=rings_features,
                )
        font_scale_20 = captured_font_scales[0]

        captured_font_scales.clear()

        # Case 2: custom 50% -> 0.5 * 100 = 50 px target_h
        with patch("cv2.putText", side_effect=mock_put_text):
            with patch.object(_writer.settings, "get", side_effect=lambda k, d=None: 50 if k == "annotated_image.year_label_size" else d):
                save_annotated_scan_image(
                    scan_path=str(scan_path),
                    annotated_path=str(annotated_path),
                    rings_features=rings_features,
                )
        font_scale_50 = captured_font_scales[0]

        # 50% should produce 2.5x larger font scale than 20%
        assert font_scale_50 == pytest.approx(font_scale_20 * 2.5, rel=1e-2)

    def test_write_rings_file_uses_in_memory_cells_data(self, temp_dir):
        """Test write_rings_file passes in-memory cells_data to save_sample_annotated_image."""
        from napari_roxas_ai._writer._writer import write_rings_file
        from PIL import Image

        sample_stem = temp_dir / "sample_mem"
        scan_path = temp_dir / "sample_mem.scan.jpg"
        scan_arr = np.full((100, 100, 3), 200, dtype=np.uint8)
        Image.fromarray(scan_arr).save(scan_path)

        cells_data_mem = np.zeros((100, 100), dtype=np.uint8)
        cells_data_mem[30:70, 30:70] = 1

        rings_features = pd.DataFrame({
            "YEAR": [2022],
            "RBXY": [[[50, 0], [50, 99]]],
            "cells_above": [5000],
        })

        meta = {
            "name": "sample_mem.rings",
            "features": rings_features,
            "metadata": {
                "sample_name": "sample_mem",
                "cells_data": cells_data_mem,
            },
        }

        rings_data = np.zeros((100, 100), dtype=np.uint8)
        written = write_rings_file(str(sample_stem), rings_data, meta)

        annotated_path = temp_dir / "sample_mem_annotated.jpg"
        assert str(annotated_path) in written
        assert annotated_path.exists()

        # Check that the in-memory cell mask was drawn without reading .cells.png from disk
        assert not (temp_dir / "sample_mem.cells.png").exists()
        with Image.open(annotated_path) as img:
            arr = np.array(img)
            # Cell pixels away from the ring line (e.g. at (40, 50)) should be cyan-tinted
            center = arr[40, 50]
            assert center[0] < center[1] or center[0] < center[2]

    def test_write_rings_file_uses_in_memory_scan_data(self, temp_dir):
        """Test write_rings_file passes in-memory scan_data to save_sample_annotated_image without scan file on disk."""
        from napari_roxas_ai._writer._writer import write_rings_file
        from PIL import Image

        sample_stem = temp_dir / "sample_mem_scan"
        # Notice we do NOT create sample_mem_scan.scan.jpg on disk!
        scan_data_mem = np.full((100, 100, 3), 200, dtype=np.uint8)

        rings_features = pd.DataFrame({
            "YEAR": [2022],
            "RBXY": [[[50, 0], [50, 99]]],
            "cells_above": [5000],
        })

        meta = {
            "name": "sample_mem_scan.rings",
            "features": rings_features,
            "metadata": {
                "sample_name": "sample_mem_scan",
                "scan_data": scan_data_mem,
            },
        }

        rings_data = np.zeros((100, 100), dtype=np.uint8)
        written = write_rings_file(str(sample_stem), rings_data, meta)

        annotated_path = temp_dir / "sample_mem_scan_annotated.jpg"
        assert str(annotated_path) in written
        assert annotated_path.exists()
        assert not (temp_dir / "sample_mem_scan.scan.jpg").exists()

    def test_save_image_rescaling(self, temp_dir):
        """Test save_image rescaling behavior for binary, uniform, and arbitrary arrays."""
        from napari_roxas_ai._writer._writer import save_image
        from PIL import Image

        # 1. Binary 0/1 array rescaled to 0/255
        p1 = temp_dir / "test_bin.png"
        arr1 = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        save_image(str(p1), arr1, rescale=True)
        with Image.open(p1) as im:
            loaded = np.array(im)
            assert np.array_equal(loaded, np.array([[0, 255], [255, 0]], dtype=np.uint8))

        # 2. Binary 0/255 array rescaled
        p2 = temp_dir / "test_255.png"
        arr2 = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        save_image(str(p2), arr2, rescale=True)
        with Image.open(p2) as im:
            loaded = np.array(im)
            assert np.array_equal(loaded, np.array([[0, 255], [255, 0]], dtype=np.uint8))

        # 3. Uniform constant array
        p3 = temp_dir / "test_const.png"
        arr3 = np.full((10, 10), 42, dtype=np.uint8)
        save_image(str(p3), arr3, rescale=True)
        with Image.open(p3) as im:
            loaded = np.array(im)
            assert np.array_equal(loaded, np.zeros((10, 10), dtype=np.uint8))

        # 4. Without rescale
        p4 = temp_dir / "test_no_rescale.png"
        arr4 = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        save_image(str(p4), arr4, rescale=False)
        with Image.open(p4) as im:
            loaded = np.array(im)
            assert np.array_equal(loaded, arr4)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
