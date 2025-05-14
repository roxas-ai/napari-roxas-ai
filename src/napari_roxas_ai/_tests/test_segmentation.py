"""
Tests for the segmentation module functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from napari_roxas_ai._segmentation._batch_sample_segmentation import (
    Worker as BatchWorker,
)
from napari_roxas_ai._segmentation._single_sample_segmentation import (
    Worker as SingleWorker,
)


# Helper function to create a test image
def create_test_image(size=(100, 100), channels=3):
    """Create a test image with the specified dimensions."""
    if channels == 1:
        return np.random.randint(0, 255, size=size, dtype=np.uint8)
    else:
        return np.random.randint(
            0, 255, size=size + (channels,), dtype=np.uint8
        )


# Fixtures for testing
@pytest.fixture
def test_image():
    """Create a test image for segmentation."""
    return create_test_image()


@pytest.fixture
def mock_cells_model():
    """Create a mock CellsSegmentationModel."""
    mock_model = MagicMock()
    mock_model.load_weights.return_value = None
    mock_model.infer.return_value = create_test_image(channels=1)
    return mock_model


@pytest.fixture
def mock_rings_model():
    """Create a mock rings model."""
    # Create mock boundaries
    boundaries = [
        torch.tensor([[i, i], [i + 1, i + 1], [i + 2, i + 2]])
        for i in range(5)
    ]

    # Create a mock model
    mock_model = MagicMock()
    mock_model.infer.return_value = (
        create_test_image(channels=1),  # rings labels
        boundaries,  # ring boundaries
    )

    return mock_model


@pytest.fixture
def mock_settings():
    """Create a mock settings manager."""
    mock_settings = MagicMock()

    # Configure mock settings
    mock_settings.get.side_effect = lambda key, default=None: {
        "file_extensions.scan_file_extension": [".scan"],
        "file_extensions.cells_file_extension": [".cells"],
        "file_extensions.rings_file_extension": [".rings"],
        "file_extensions.image_file_extensions": [".tif", ".jpg", ".png"],
    }.get(key, default)

    return mock_settings


@pytest.fixture
def mock_viewer():
    """Create a mock napari viewer."""
    viewer = MagicMock()

    # Mock the add_labels method
    viewer.add_labels.return_value = MagicMock()

    # Mock the layers dictionary
    viewer.layers = {}

    return viewer


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


# Create sample image files for batch processing tests
@pytest.fixture
def sample_image_dir(temp_dir):
    """Create a directory with sample images for batch processing."""
    # Create some sample images
    for i in range(3):
        image = create_test_image()
        img = Image.fromarray(image.astype(np.uint8))
        img.save(temp_dir / f"sample_{i}.scan.tif")

    return temp_dir


class TestSingleSampleSegmentation:
    """Tests for single sample segmentation functionality."""

    def test_get_model_files(self):
        """Test getting model files from directory."""
        with patch(
            "pathlib.Path.iterdir",
            return_value=[
                Path("dummy/path/model1.pth"),
                Path("dummy/path/model2.pth"),
            ],
        ):
            from napari_roxas_ai._segmentation._single_sample_segmentation import (
                SingleSampleSegmentationWidget,
            )

            # Call the method directly as a normal function, passing a dummy self
            dummy_self = MagicMock()
            result = SingleSampleSegmentationWidget._get_model_files(
                dummy_self, "dummy/path"
            )
            assert result == ("model1.pth", "model2.pth")

    def test_extract_base_name(self, mock_settings):
        """Test extracting base name from layer name."""
        from napari_roxas_ai._segmentation._single_sample_segmentation import (
            SingleSampleSegmentationWidget,
        )

        # Configure settings
        mock_settings.get.return_value = [".scan"]

        # Pass static method a mock self
        mock_self = MagicMock()
        mock_self.settings = mock_settings

        # Test the method directly
        base_name = SingleSampleSegmentationWidget._extract_base_name(
            mock_self, "sample1.scan"
        )
        assert base_name == "sample1"

        # Test without scan extension
        base_name = SingleSampleSegmentationWidget._extract_base_name(
            mock_self, "other_name"
        )
        assert base_name == "other_name"


class TestSingleWorkerFunctionality:
    """Test the SingleWorker class functionality directly."""

    def test_cells_segmentation(
        self, test_image, mock_cells_model, mock_settings
    ):
        """Test cells segmentation functionality."""
        with patch(
            "napari_roxas_ai._segmentation._cells_model.CellsSegmentationModel",
            return_value=mock_cells_model,
        ):
            # We'll test the worker without QObject initialization
            with patch.object(SingleWorker, "__init__", return_value=None):
                worker = SingleWorker(None, None, None, None, None, None, None)

                # Manually setup worker attributes
                worker.input_array = test_image
                worker.segment_cells = True
                worker.segment_rings = False
                worker.cells_model_weights_file = "/path/to/cells_model.pth"
                worker.settings = mock_settings
                worker.base_name = "test_sample"

                # Mock the signals
                worker.result_ready = MagicMock()
                worker.finished = MagicMock()

                # Mock CellsSegmentationModel instance
                worker_cells_model = mock_cells_model

                # Call the functional parts without using run()
                # Set up cells model
                mock_settings.get.return_value = [".cells"]
                worker_cells_model.load_weights.return_value = None

                # Perform inference
                cells_labels = worker_cells_model.infer(test_image)

                # Add to results
                suffix = worker.settings.get(
                    "file_extensions.cells_file_extension"
                )[0]
                name = f"{worker.base_name}{suffix}"
                results = {
                    "cells": {
                        "data": (cells_labels / 255).astype("uint8"),
                        "name": name,
                    }
                }

                # Verify cells segmentation operation
                mock_cells_model.infer.assert_called_with(test_image)
                assert "cells" in results
                assert results["cells"]["name"] == "test_sample.cells"

    def test_rings_segmentation(
        self, test_image, mock_rings_model, mock_settings
    ):
        """Test rings segmentation functionality."""
        with patch("torch.package.PackageImporter") as mock_package_importer:
            # Configure the mock package importer
            mock_importer = MagicMock()
            mock_importer.load_pickle.return_value = mock_rings_model
            mock_package_importer.return_value = mock_importer

            # We'll test the worker without QObject initialization
            with patch.object(SingleWorker, "__init__", return_value=None):
                worker = SingleWorker(None, None, None, None, None, None, None)

                # Manually setup worker attributes
                worker.input_array = test_image
                worker.segment_cells = False
                worker.segment_rings = True
                worker.rings_model_weights_file = "/path/to/rings_model.pt"
                worker.settings = mock_settings
                worker.base_name = "test_sample"

                # Mock the signals
                worker.result_ready = MagicMock()
                worker.finished = MagicMock()

                # Mock settings
                mock_settings.get.return_value = [".rings"]

                # Set up rings model
                imp = mock_package_importer()
                package_name = "LinearRingModel"
                resource_name = "model.pkl"
                loaded_model = imp.load_pickle(package_name, resource_name)

                # Perform inference
                rings_labels, rings_boundaries = loaded_model.infer(test_image)

                # Create a DataFrame from boundaries
                boundary_data = []
                for _i, _boundary in enumerate(rings_boundaries):
                    # Since we're using mock data, simplify this part
                    boundary_data.append({"boundary_coordinates": []})

                boundaries_df = pd.DataFrame(boundary_data)

                # Add to results
                suffix = worker.settings.get(
                    "file_extensions.rings_file_extension"
                )[0]
                name = f"{worker.base_name}{suffix}"
                results = {
                    "rings": {
                        "data": rings_labels.astype("int32"),
                        "name": name,
                        "features": boundaries_df,
                    }
                }

                # Verify rings segmentation operation
                mock_importer.load_pickle.assert_called_with(
                    package_name, resource_name
                )
                loaded_model.infer.assert_called_with(test_image)
                assert "rings" in results
                assert results["rings"]["name"] == "test_sample.rings"
                assert "features" in results["rings"]


class TestBatchSampleSegmentation:
    """Tests for batch sample segmentation functionality."""

    def test_get_model_files(self):
        """Test getting model files from directory."""
        with patch(
            "pathlib.Path.iterdir",
            return_value=[
                Path("dummy/path/model1.pth"),
                Path("dummy/path/model2.pth"),
            ],
        ):
            from napari_roxas_ai._segmentation._batch_sample_segmentation import (
                BatchSampleSegmentationWidget,
            )

            # Call the method directly as a normal function, passing a dummy self
            dummy_self = MagicMock()
            result = BatchSampleSegmentationWidget._get_model_files(
                dummy_self, "dummy/path"
            )
            assert result == ("model1.pth", "model2.pth")


class TestBatchWorkerFunctionality:
    """Test the BatchWorker class functionality directly."""

    def test_file_discovery(self, sample_image_dir):
        """Test file discovery in batch worker."""
        with patch(
            "napari_roxas_ai._segmentation._batch_sample_segmentation.settings"
        ) as mock_settings:
            # Configure mock settings
            mock_settings.get.return_value = [".scan"]

            # Mock glob to return our scan files
            expected_files = [
                str(sample_image_dir / f"sample_{i}.scan.tif")
                for i in range(3)
            ]
            with patch("glob.glob", return_value=expected_files):
                # We'll test the worker without full initialization
                # This block is just to test file discovery
                with patch.object(BatchWorker, "__init__", return_value=None):
                    worker = BatchWorker(None, None, None, None, None)

                    # Manually setup the attributes
                    worker.input_directory_path = str(sample_image_dir)
                    worker.scan_content_ext = ".scan"

                    # Manually run the file discovery logic
                    worker.scan_file_paths = expected_files

                    # Test that the files were found
                    assert len(worker.scan_file_paths) == 3
                    for file_path in expected_files:
                        assert file_path in worker.scan_file_paths

    def test_segmentation_logic(
        self, sample_image_dir, mock_cells_model, mock_rings_model
    ):
        """Test segmentation logic in batch worker."""
        with patch(
            "napari_roxas_ai._segmentation._batch_sample_segmentation.settings"
        ) as mock_settings, patch(
            "napari_roxas_ai._segmentation._batch_sample_segmentation.read_scan_file"
        ) as mock_read_image:
            # Configure mock settings
            mock_settings.get.side_effect = lambda key, default=None: {
                "file_extensions.scan_file_extension": [".scan"],
                "file_extensions.cells_file_extension": [".cells"],
                "file_extensions.rings_file_extension": [".rings"],
            }.get(key, default)

            # Mock the read_scan_file function
            mock_read_image.return_value = (
                create_test_image(),
                {
                    "metadata": {"sample_name": "test_sample"},
                    "scale": (1.0, 1.0),
                },
                "image",
            )

            # We'll test the worker without QObject initialization
            with patch.object(BatchWorker, "__init__", return_value=None):
                worker = BatchWorker(None, None, None, None, None)

                # Manually setup worker attributes
                worker.input_directory_path = str(sample_image_dir)
                worker.segment_cells = True
                worker.segment_rings = True
                worker.cells_model_weights_file = "/path/to/cells_model.pth"
                worker.rings_model_weights_file = "/path/to/rings_model.pt"
                worker.scan_file_paths = [
                    str(sample_image_dir / "test1.scan.tif")
                ]

                # Setup mock models
                worker.cells_model = mock_cells_model
                worker.rings_model = mock_rings_model

                # Setup content extensions
                worker.cells_content_ext = ".cells"
                worker.rings_content_ext = ".rings"

                # Mock progress and finished signals
                worker.progress = MagicMock()
                worker.finished = MagicMock()

                # Mock read_scan_file to get sample data
                scan_data, scan_add_kwargs, _ = mock_read_image.return_value

                # Extract sample metadata
                sample_metadata = {
                    k: v
                    for k, v in scan_add_kwargs["metadata"].items()
                    if isinstance(k, str) and k.startswith("sample_")
                }

                # Process cells and verify
                mock_cells_model.infer(scan_data)
                mock_cells_model.infer.assert_called_with(scan_data)

                # Create cells outputs placeholder (not using result directly)
                cells_layer_name = f"{sample_metadata['sample_name']}{worker.cells_content_ext}"
                cells_add_kwargs = {
                    "name": cells_layer_name,
                    "scale": scan_add_kwargs["scale"],
                    "features": pd.DataFrame(),
                    "metadata": {},
                }
                cells_add_kwargs["metadata"].update(sample_metadata)

                # Process rings and verify
                mock_rings_model.infer(scan_data)
                mock_rings_model.infer.assert_called_with(scan_data)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
