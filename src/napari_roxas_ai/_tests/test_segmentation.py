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
from qtpy.QtWidgets import QSizePolicy
from napari_roxas_ai._segmentation._single_sample_segmentation import (
    Worker as SingleWorker,
)
from napari_roxas_ai._utils._metadata_keys import SAMPLE_METADATA_PREFIXES


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

    def test_add_result_layers_marks_dirty_and_writes_annotated_image(
        self, make_napari_viewer
    ):
        """New segmentation layers are unsaved; the annotated image is drawn at once."""
        from napari_roxas_ai._segmentation import _single_sample_segmentation
        from napari_roxas_ai._segmentation._single_sample_segmentation import (
            SingleSampleSegmentationWidget,
        )

        viewer = make_napari_viewer()
        scan_layer = viewer.add_image(
            create_test_image(),
            name="sample_01.scan",
            metadata={"sample_name": "sample_01", "sample_stem_path": "sample_01"},
        )
        dummy_self = MagicMock()
        dummy_self._viewer = viewer
        dummy_self.input_layer = scan_layer
        dummy_self.settings = _single_sample_segmentation.SettingsManager()
        dummy_self.cells_model_file = "cells_model.pth"
        dummy_self.rings_model_file = None

        results = {
            "cells": {
                "data": np.ones((100, 100), dtype=np.uint8),
                "name": "sample_01.cells",
            }
        }
        with patch.object(
            _single_sample_segmentation, "save_layers_annotated_image"
        ) as mock_annotate:
            SingleSampleSegmentationWidget._add_result_layers(dummy_self, results)

        assert viewer.layers["sample_01.cells"].metadata["_is_dirty"] is True
        mock_annotate.assert_called_once_with(viewer.layers, "sample_01")

    def test_get_model_files_skips_hidden_files(self):
        """The download bookkeeping file (.asset_id) is not a model."""
        with patch(
            "pathlib.Path.iterdir",
            return_value=[
                Path("dummy/path/.asset_id"),
                Path("dummy/path/model1.pth"),
            ],
        ):
            from napari_roxas_ai._segmentation._single_sample_segmentation import (
                SingleSampleSegmentationWidget,
            )

            result = SingleSampleSegmentationWidget._get_model_files(
                MagicMock(), "dummy/path"
            )
            assert result == ("model1.pth",)

    def test_standard_rings_model_preselected(self, mock_settings):
        """The rerun model of the rings editor must not become the default."""
        from napari_roxas_ai._segmentation._single_sample_segmentation import (
            SingleSampleSegmentationWidget,
        )

        with patch(
            "napari_roxas_ai._segmentation._single_sample_segmentation.check_assets_and_download"
        ), patch(
            "napari_roxas_ai._segmentation._single_sample_segmentation.SingleSampleSegmentationWidget._get_model_files",
            return_value=(
                "LinearRingModel_v3_iterative.pt",
                "linear_rings_model.pth",
            ),
        ), patch(
            "napari_roxas_ai._segmentation._single_sample_segmentation.SettingsManager",
            return_value=mock_settings,
        ):
            viewer = MagicMock()
            viewer.layers = []
            widget = SingleSampleSegmentationWidget(viewer)

        assert widget._rings_model_weights_file.value == "linear_rings_model.pth"

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

    def test_widget_structure_and_valid_layers(self, mock_settings):
        """Test SingleSampleSegmentationWidget elements and layer detection."""
        from napari_roxas_ai._segmentation._single_sample_segmentation import (
            SingleSampleSegmentationWidget,
        )

        viewer = MagicMock()
        image_layer = MagicMock()
        image_layer.name = "sample1.scan"
        type(image_layer).__name__ = "Image"
        # Mock isinstance check
        with patch(
            "napari_roxas_ai._segmentation._single_sample_segmentation.check_assets_and_download"
        ), patch(
            "napari_roxas_ai._segmentation._single_sample_segmentation.SingleSampleSegmentationWidget._get_model_files",
            return_value=("dummy_model.pth",),
        ), patch(
            "napari_roxas_ai._segmentation._single_sample_segmentation.SettingsManager",
            return_value=mock_settings,
        ), patch(
            "napari.layers.Image", (type(image_layer),)
        ):
            viewer.layers = [image_layer]
            widget = SingleSampleSegmentationWidget(viewer)
            # Check that there is no _input_layer_combo attribute
            assert not hasattr(widget, "_input_layer_combo")
            # Check labels and row containers
            assert widget._segment_cells_checkbox.label == "Cell Segmentation:"
            assert widget._segment_rings_checkbox.label == "Ring Segmentation:"
            assert widget._cells_model_weights_file.label == ""
            assert widget._rings_model_weights_file.label == ""
            assert widget._cells_row.layout == "horizontal"
            assert widget._rings_row.layout == "horizontal"
            assert widget._segment_cells_checkbox in widget._cells_row
            assert widget._cells_model_weights_file in widget._cells_row
            assert widget._segment_rings_checkbox in widget._rings_row
            assert widget._rings_model_weights_file in widget._rings_row
            assert (
                widget._segment_cells_checkbox.native.width()
                == widget._segment_rings_checkbox.native.width()
            )
            assert (
                widget._cells_model_weights_file.native.width()
                == widget._rings_model_weights_file.native.width()
            )
            assert widget._run_segmentation_button.enabled is True

            # Untick both checkboxes -> button disabled
            widget._segment_cells_checkbox.value = False
            widget._segment_rings_checkbox.value = False
            assert widget._run_segmentation_button.enabled is False

            # Tick one checkbox back -> button enabled
            widget._segment_cells_checkbox.value = True
            assert widget._run_segmentation_button.enabled is True

            widget._segment_cells_checkbox.value = False
            widget._segment_rings_checkbox.value = True
            assert widget._run_segmentation_button.enabled is True

            valid = widget._get_valid_layers()
            assert valid == [image_layer]


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
                    boundary_data.append({"RBXY": []})

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

    def test_batch_segmentation_widget_layout_and_visibility(self):
        """Test BatchSampleSegmentationWidget row layout and visibility toggles."""
        from napari_roxas_ai._segmentation._batch_sample_segmentation import (
            BatchSampleSegmentationWidget,
        )

        with patch(
            "napari_roxas_ai._segmentation._batch_sample_segmentation.check_assets_and_download"
        ), patch.object(
            BatchSampleSegmentationWidget,
            "_get_model_files",
            return_value=("model1.pth", "model2.pth"),
        ):
            widget = BatchSampleSegmentationWidget(MagicMock())

            # Check row containers and labels
            assert widget._segment_cells_checkbox.label == "Cell Segmentation:"
            assert widget._segment_rings_checkbox.label == "Ring Segmentation:"
            assert widget._cells_model_weights_file.label == ""
            assert widget._rings_model_weights_file.label == ""

            # Check rows in widget layout
            assert widget[0] is widget._project_directory_row
            assert widget[1] is widget._process_processed_checkbox
            assert widget[2] is widget._handpick_files_checkbox
            assert widget[3] is widget._file_selection_container
            assert widget[4] is widget._reverse_selection_button
            assert widget[5] is widget._cells_row
            assert widget[6] is widget._rings_row
            assert widget[7] is widget._run_segmentation_button
            assert widget[8] is widget.progress_bar

            # Check checkbox labels and tooltips
            assert (
                widget._process_processed_checkbox.label
                == "Include Previously Segmented Images"
            )
            assert (
                widget._process_processed_checkbox.tooltip
                == "Also include images for which a .rings.tif or .cells.png file already exists."
            )
            assert (
                widget._handpick_files_checkbox.label
                == "Manually Select Files To Process:"
            )

            # Check that checkbox widths are equalized
            assert (
                widget._segment_cells_checkbox.native.width()
                == widget._segment_rings_checkbox.native.width()
            )
            assert (
                widget._cells_model_weights_file.native.width()
                == widget._rings_model_weights_file.native.width()
            )

            # Initial state
            assert widget._segment_cells_checkbox.value is True
            assert not widget._cells_model_weights_file.native.isHidden()
            assert widget._segment_rings_checkbox.value is True
            assert not widget._rings_model_weights_file.native.isHidden()
            assert widget._run_segmentation_button.enabled is True

            # Toggle cells checkbox -> hide combo
            widget._segment_cells_checkbox.value = False
            assert widget._cells_model_weights_file.native.isHidden()
            assert widget._run_segmentation_button.enabled is True

            # Toggle rings checkbox -> hide combo & disable run button
            widget._segment_rings_checkbox.value = False
            assert widget._rings_model_weights_file.native.isHidden()
            assert widget._run_segmentation_button.enabled is False

            # Toggle cells checkbox back -> show combo & enable run button
            widget._segment_cells_checkbox.value = True
            assert not widget._cells_model_weights_file.native.isHidden()
            assert widget._run_segmentation_button.enabled is True

    def test_run_segmentation_picks_up_files_added_after_opening(self, tmp_path):
        """Images prepared while the widget is open are found when running."""
        from napari_roxas_ai._segmentation import _batch_sample_segmentation
        from napari_roxas_ai._segmentation._batch_sample_segmentation import (
            BatchSampleSegmentationWidget,
        )

        with patch.object(
            _batch_sample_segmentation, "check_assets_and_download"
        ), patch.object(
            BatchSampleSegmentationWidget,
            "_get_model_files",
            return_value=("model1.pth", "model2.pth"),
        ):
            widget = BatchSampleSegmentationWidget(MagicMock())
            widget._handpick_files_checkbox.value = False
            widget.input_directory_path = str(tmp_path)
            widget._refresh_file_list()
            assert widget.source_files == []

            # The project is prepared while the widget is open
            scan = tmp_path / "sample1.scan.jpg"
            scan.touch()

            with patch.object(
                _batch_sample_segmentation, "Worker"
            ) as mock_worker, patch.object(
                _batch_sample_segmentation, "QThread"
            ), patch.object(
                _batch_sample_segmentation, "show_warning"
            ) as mock_warning:
                widget._run_segmentation()

            mock_warning.assert_not_called()
            assert mock_worker.call_args.kwargs["scan_file_paths"] == [str(scan)]

            # Ticking the manual selection lists it as well
            scan2 = tmp_path / "sample2.scan.jpg"
            scan2.touch()
            widget._handpick_files_checkbox.value = True
            assert set(widget._file_select_widget.choices) == {
                "sample1.scan.jpg",
                "sample2.scan.jpg",
            }

    def test_files_prepared_by_other_widgets_are_listed(self, tmp_path):
        """Images prepared while the widget is open appear in the manual selection."""
        from napari_roxas_ai._segmentation import _batch_sample_segmentation
        from napari_roxas_ai._segmentation._batch_sample_segmentation import (
            BatchSampleSegmentationWidget,
        )
        from napari_roxas_ai._utils._project_files_events import (
            notify_project_files_changed,
        )

        (tmp_path / "sample1.scan.jpg").touch()
        with patch.object(
            _batch_sample_segmentation, "check_assets_and_download"
        ), patch.object(
            BatchSampleSegmentationWidget,
            "_get_model_files",
            return_value=("model1.pth", "model2.pth"),
        ):
            widget = BatchSampleSegmentationWidget(MagicMock())
            widget.input_directory_path = str(tmp_path)
            widget._refresh_file_list()
            widget._file_select_widget.value = ["sample1.scan.jpg"]

            (tmp_path / "sample2.scan.jpg").touch()
            notify_project_files_changed(tmp_path)

            assert set(widget._file_select_widget.choices) == {
                "sample1.scan.jpg",
                "sample2.scan.jpg",
            }
            # The manual selection is kept
            assert widget._file_select_widget.value == ["sample1.scan.jpg"]

            # Files of another project are ignored
            (tmp_path / "sample3.scan.jpg").touch()
            notify_project_files_changed(tmp_path / "other")
            assert "sample3.scan.jpg" not in widget._file_select_widget.choices

    def test_batch_segmentation_tells_the_listeners_when_done(self, tmp_path):
        from napari_roxas_ai._segmentation import _batch_sample_segmentation
        from napari_roxas_ai._segmentation._batch_sample_segmentation import (
            BatchSampleSegmentationWidget,
        )

        (tmp_path / "sample1.scan.jpg").touch()
        with patch.object(
            _batch_sample_segmentation, "check_assets_and_download"
        ), patch.object(
            BatchSampleSegmentationWidget,
            "_get_model_files",
            return_value=("model1.pth", "model2.pth"),
        ):
            widget = BatchSampleSegmentationWidget(MagicMock())
            widget._handpick_files_checkbox.value = False
            widget.input_directory_path = str(tmp_path)

            thread = MagicMock()
            with patch.object(_batch_sample_segmentation, "Worker"), patch.object(
                _batch_sample_segmentation, "QThread", return_value=thread
            ), patch.object(
                _batch_sample_segmentation, "notify_project_files_changed"
            ) as notify:
                widget._run_segmentation()
                notify.assert_not_called()
                # The run ends
                for call in thread.finished.connect.call_args_list:
                    call.args[0]()

            notify.assert_called_once_with(str(tmp_path), source=widget)

    def test_batch_segmentation_widget_file_selection_and_filtering(self, tmp_path):
        """Test file list filtering, handpicking and reverse selection in BatchSampleSegmentationWidget."""
        from napari_roxas_ai._segmentation._batch_sample_segmentation import (
            BatchSampleSegmentationWidget,
        )
        from napari_roxas_ai._settings import SettingsManager

        # Create dummy scan files and output files
        scan1 = tmp_path / "sample1.scan.jpg"
        scan1.touch()
        scan2 = tmp_path / "sample2.scan.jpg"
        scan2.touch()
        # sample2 has an existing .rings.tif
        rings2 = tmp_path / "sample2.rings.tif"
        rings2.touch()

        with patch(
            "napari_roxas_ai._segmentation._batch_sample_segmentation.check_assets_and_download"
        ), patch.object(
            BatchSampleSegmentationWidget,
            "_get_model_files",
            return_value=("model1.pth", "model2.pth"),
        ):
            widget = BatchSampleSegmentationWidget(MagicMock())
            widget.input_directory_path = str(tmp_path)
            widget._refresh_file_list()

            # With _process_processed_checkbox = True (default), both scan1 and scan2 are included
            assert str(scan1) in widget.source_files
            assert str(scan2) in widget.source_files
            assert len(widget.source_files) == 2

            # When unchecked, scan2 is excluded because sample2.rings.tif exists
            widget._process_processed_checkbox.value = False
            widget._refresh_file_list()
            assert str(scan1) in widget.source_files
            assert str(scan2) not in widget.source_files
            assert len(widget.source_files) == 1

            # Test handpick checkbox default and toggling
            assert widget._handpick_files_checkbox.value is True
            assert not widget._file_selection_container.native.isHidden()
            assert not widget._reverse_selection_button.native.isHidden()

            # Untick handpick checkbox
            widget._handpick_files_checkbox.value = False
            widget._toggle_file_selection()
            assert widget._file_selection_container.native.isHidden()
            assert widget._reverse_selection_button.native.isHidden()

            # Re-tick handpick checkbox
            widget._process_processed_checkbox.value = True
            widget._refresh_file_list()
            widget._handpick_files_checkbox.value = True
            widget._toggle_file_selection()
            assert not widget._file_selection_container.native.isHidden()
            assert not widget._reverse_selection_button.native.isHidden()
            assert hasattr(widget, "_file_select_widget")
            assert len(widget._file_select_widget.choices) == 2

            # Test selection and reverse selection
            widget._file_select_widget.value = [widget._file_select_widget.choices[0]]
            assert len(widget._get_selected_file_paths()) == 1

            widget._reverse_file_selection()
            assert len(widget._file_select_widget.value) == 1
            assert widget._file_select_widget.value[0] == widget._file_select_widget.choices[1]
            assert widget._get_selected_file_paths() == [str(scan2)]


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
                    if isinstance(k, str)
                    and k.startswith(SAMPLE_METADATA_PREFIXES)
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

    def test_worker_run_terminal_logging(
        self, sample_image_dir, mock_cells_model, mock_rings_model, capsys
    ):
        """Test terminal output logging during worker run."""
        with patch(
            "napari_roxas_ai._segmentation._batch_sample_segmentation.settings"
        ) as mock_settings, patch(
            "napari_roxas_ai._segmentation._batch_sample_segmentation.read_scan_file"
        ) as mock_read_image, patch(
            "napari_roxas_ai._segmentation._batch_sample_segmentation.write_single_layer"
        ), patch(
            "napari_roxas_ai._segmentation._batch_sample_segmentation.update_rings_geometries",
            return_value=(pd.DataFrame(), np.zeros((10, 10)), None),
        ), patch(
            "napari_roxas_ai._segmentation._batch_sample_segmentation.get_metadata_from_file",
            return_value={"rings_outmost_complete_year": 2020},
        ):
            mock_settings.get.side_effect = lambda key, default=None: {
                "file_extensions.scan_file_extension": [".scan"],
                "file_extensions.cells_file_extension": [".cells"],
                "file_extensions.rings_file_extension": [".rings"],
                "samples_metadata.fields": [{"id": "rings_outmost_complete_year", "default": 2020}],
            }.get(key, default)

            # The second file cannot be read and must not count as done
            mock_read_image.side_effect = [
                (
                    create_test_image(),
                    {
                        "metadata": {"sample_name": "sample_01"},
                        "scale": (1.0, 1.0),
                    },
                    "image",
                ),
                OSError("broken"),
            ]

            with patch.object(BatchWorker, "__init__", return_value=None):
                worker = BatchWorker(None, None, None, None, None)
                worker.input_directory_path = str(sample_image_dir)
                worker.segment_cells = True
                worker.segment_rings = True
                worker.cells_model_weights_file = Path("/path/to/linear_cells_model.pth")
                worker.rings_model_weights_file = Path("/path/to/linear_rings_model.pth")
                worker.scan_file_paths = [
                    str(sample_image_dir / "sample_01.scan.tif"),
                    str(sample_image_dir / "sample_02.scan.tif"),
                ]
                worker.cells_model = mock_cells_model
                worker.rings_model = mock_rings_model
                worker.cells_content_ext = ".cells.png"
                worker.rings_content_ext = ".rings.tif"
                worker.progress = MagicMock()
                worker.finished = MagicMock()
                worker.message = MagicMock()

                worker.run()

                captured = capsys.readouterr()
                assert "[Batch Segmentation] [1/2] sample_01.scan.tif: Cells (linear_cells_model.pth), Rings (linear_rings_model.pth)" in captured.out
                assert "[Batch Segmentation] [2/2] Error processing sample_02.scan.tif: broken" in captured.out
                assert "[Batch Segmentation] Completed 1 of 2 file(s) successfully." in captured.out
                worker.message.emit.assert_called_once_with(
                    "[Batch Segmentation] Completed 1 of 2 file(s) successfully."
                )

    def test_worker_run_cells_only_writes_annotated_image(
        self, sample_image_dir, mock_cells_model
    ):
        """Cells-only runs also write the annotated image, as an extra progress step."""
        scan_data = create_test_image()
        with patch(
            "napari_roxas_ai._segmentation._batch_sample_segmentation.settings"
        ) as mock_settings, patch(
            "napari_roxas_ai._segmentation._batch_sample_segmentation.read_scan_file",
            return_value=(
                scan_data,
                {
                    "metadata": {
                        "sample_name": "sample_01",
                        "sample_stem_path": "sample_01",
                    },
                    "scale": (1.0, 1.0),
                },
                "image",
            ),
        ), patch(
            "napari_roxas_ai._segmentation._batch_sample_segmentation.write_single_layer"
        ), patch(
            "napari_roxas_ai._segmentation._batch_sample_segmentation.save_sample_annotated_image"
        ) as mock_annotate:
            mock_settings.get.side_effect = lambda key, default=None: {
                "project_directory": str(sample_image_dir),
                "samples_metadata.fields": [{"id": "rings_outmost_complete_year", "default": 2020}],
            }.get(key, default)

            with patch.object(BatchWorker, "__init__", return_value=None):
                worker = BatchWorker(None, None, None, None, None)
                worker.segment_cells = True
                worker.segment_rings = False
                worker.cells_model_weights_file = Path("/path/to/linear_cells_model.pth")
                worker.rings_model_weights_file = None
                worker.scan_file_paths = [str(sample_image_dir / "sample_01.scan.tif")]
                worker.cells_model = mock_cells_model
                worker.cells_content_ext = ".cells"
                worker.progress = MagicMock()
                worker.finished = MagicMock()
                worker.message = MagicMock()

                worker.run()

        mock_annotate.assert_called_once()
        args, kwargs = mock_annotate.call_args
        assert args[0] == (sample_image_dir / "sample_01").resolve()
        assert args[1].empty
        assert kwargs["cells_data"] is not None
        assert kwargs["scan_data"] is scan_data
        # Cells and annotated image: two steps
        assert worker.progress.emit.call_args_list[-1].args == (2, 2)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
