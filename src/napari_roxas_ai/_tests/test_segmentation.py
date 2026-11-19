"""
Tests for the segmentation module functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from napari_roxas_ai._jobs import Job, JobStatus, Message, run_job_sync


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
        ), patch(
            "napari_roxas_ai._segmentation._single_sample_segmentation.install_unsaved_changes_guard"
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
        ), patch(
            "napari_roxas_ai._segmentation._single_sample_segmentation.install_unsaved_changes_guard"
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


class FakeCellsModel:
    """Finds one cell in the middle of the image, in two patches."""

    def infer_steps(self, img):
        labels = np.zeros(img.shape[:2])
        labels[40:60, 40:60] = 255
        yield 1, 2
        yield 2, 2
        return labels

    def infer(self, img):
        steps = self.infer_steps(img)
        while True:
            try:
                next(steps)
            except StopIteration as stop:
                return stop.value


class FakeRingsModel:
    """Finds two horizontal ring boundaries."""

    def infer(self, img):
        width = img.shape[1]
        boundaries = [
            torch.tensor([[30, 0], [30, width - 1]]),
            [[60, 0], [60, width - 1]],
        ]
        return np.zeros(img.shape[:2], dtype=np.uint8), boundaries


@pytest.fixture
def fake_models(monkeypatch):
    """Replace the model loading by the fake models."""
    from napari_roxas_ai._segmentation import _operations

    monkeypatch.setattr(
        _operations, "load_cells_model", lambda path, try_gpu: FakeCellsModel()
    )
    monkeypatch.setattr(
        _operations, "load_rings_model", lambda path, try_gpu: FakeRingsModel()
    )


def _run(generator):
    """The events and the result of an operation."""
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            return events, stop.value


class TestSingleSegmentationOperation:
    def test_cells_and_rings_are_segmented(self, test_image, fake_models):
        from napari_roxas_ai._segmentation._operations import segment_sample

        _, results = _run(
            segment_sample(
                test_image,
                Path("cells.pth"),
                Path("rings.pth"),
                False,
                "s.cells",
                "s.rings",
            )
        )

        cells = results["cells"]
        assert cells["name"] == "s.cells"
        assert set(np.unique(cells["data"])) == {0, 1}
        rings = results["rings"]
        assert rings["name"] == "s.rings"
        assert rings["data"].dtype == np.int32
        # The dummy top boundary first, then the found ones as lists
        assert rings["features"]["RBXY"].tolist() == [
            [[0, 0], [0, 99]],
            [[30, 0], [30, 99]],
            [[60, 0], [60, 99]],
        ]

    def test_only_the_chosen_parts_are_segmented(
        self, test_image, fake_models
    ):
        from napari_roxas_ai._segmentation._operations import segment_sample

        _, results = _run(
            segment_sample(
                test_image,
                None,
                Path("rings.pth"),
                False,
                "s.cells",
                "s.rings",
            )
        )
        assert list(results) == ["rings"]


SINGLE = "napari_roxas_ai._segmentation._single_sample_segmentation"


@pytest.fixture
def single_widget(make_napari_viewer, sync_runner):
    """The widget with a scan in the viewer; the project is left alone."""
    from napari_roxas_ai._segmentation._single_sample_segmentation import (
        SingleSampleSegmentationWidget,
    )

    viewer = make_napari_viewer()
    viewer.add_image(
        create_test_image(),
        name="sample_01.scan",
        metadata={"sample_name": "sample_01"},
    )
    with (
        patch(f"{SINGLE}.check_assets_and_download"),
        patch.object(
            SingleSampleSegmentationWidget,
            "_get_model_files",
            return_value=("linear_rings_model.pth",),
        ),
    ):
        widget = SingleSampleSegmentationWidget(viewer)
    with (
        patch(f"{SINGLE}.fix_sample_stem_paths_in_project"),
        patch(f"{SINGLE}.save_layers_annotated_image"),
    ):
        yield viewer, widget


class TestSingleSegmentationWidget:
    def test_segmented_layers_are_added(self, single_widget, fake_models):
        viewer, widget = single_widget

        widget._run_segmentation_button.press()

        assert viewer.layers["sample_01.cells"].metadata["_is_dirty"] is True
        # The two found ring boundaries
        rings = viewer.layers["sample_01.rings"]
        assert len(rings.features) == 2
        assert widget._run_segmentation_button.enabled

    def test_failed_segmentation_reports_and_leaves_the_widget_usable(
        self, single_widget, monkeypatch
    ):
        from napari_roxas_ai._job_runner import _runner
        from napari_roxas_ai._segmentation import _operations

        viewer, widget = single_widget

        def broken(path, try_gpu):
            raise RuntimeError("no GPU memory")

        monkeypatch.setattr(_operations, "load_cells_model", broken)
        errors = []
        monkeypatch.setattr(_runner, "show_error", errors.append)

        widget._run_segmentation_button.press()

        assert errors == ["Segmentation failed: no GPU memory"]
        assert "sample_01.cells" not in viewer.layers
        assert widget._run_segmentation_button.enabled


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

    def test_run_segmentation_picks_up_files_added_after_opening(
        self, tmp_path, sync_runner
    ):
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

            segmented = []

            def fake_segment_samples(scan_file_paths, *args):
                segmented.extend(scan_file_paths)
                yield from ()

            with (
                patch.object(
                    _batch_sample_segmentation,
                    "segment_samples",
                    fake_segment_samples,
                ),
                patch.object(
                    _batch_sample_segmentation, "show_warning"
                ) as mock_warning,
            ):
                widget._run_segmentation_button.press()

            mock_warning.assert_not_called()
            assert segmented == [str(scan)]

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

    def test_failed_batch_tells_the_listeners_and_leaves_the_widget_usable(
        self, tmp_path, sync_runner, monkeypatch
    ):
        from napari_roxas_ai._job_runner import _runner
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

            def failing_segment_samples(*args):
                raise RuntimeError("no models")
                yield

            errors = []
            monkeypatch.setattr(_runner, "show_error", errors.append)
            with (
                patch.object(
                    _batch_sample_segmentation,
                    "segment_samples",
                    failing_segment_samples,
                ),
                patch.object(
                    _batch_sample_segmentation, "notify_project_files_changed"
                ) as notify,
            ):
                widget._run_segmentation_button.press()

            # Also told after a failure: some samples may have been written
            notify.assert_called_once_with(str(tmp_path), source=widget)
            assert errors == ["Batch segmentation failed: no models"]
            assert widget._run_segmentation_button.enabled

    def test_batch_segmentation_widget_file_selection_and_filtering(
        self, tmp_path
    ):
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


class TestBatchSegmentationOperation:
    """The batch operation, with fake models and file access."""

    @pytest.fixture
    def scans(self, sample_image_dir, fake_models):
        from napari_roxas_ai._segmentation import _operations

        written = []
        read = MagicMock(
            side_effect=[
                (
                    create_test_image(),
                    {
                        "metadata": {
                            "sample_name": "sample_01",
                            "sample_stem_path": "sample_01",
                        },
                        "scale": (1.0, 1.0),
                    },
                    "image",
                ),
                OSError("broken"),
                (
                    create_test_image(),
                    {
                        "metadata": {"sample_name": "sample_03"},
                        "scale": (1, 1),
                    },
                    "image",
                ),
            ]
        )
        with (
            patch.object(_operations, "read_scan_file", read),
            patch.object(
                _operations,
                "write_single_layer",
                side_effect=lambda path, data, meta, **kw: written.append(
                    meta["name"]
                ),
            ),
            patch.object(
                _operations,
                "get_metadata_from_file",
                return_value={"rings_outmost_complete_year": 2020},
            ),
            patch.object(
                _operations, "save_sample_annotated_image"
            ) as annotate,
        ):
            yield written, annotate

    def _segment(self, sample_image_dir, cells=True, rings=True):
        from napari_roxas_ai._segmentation._operations import segment_samples

        return segment_samples(
            [
                str(sample_image_dir / f"sample_0{i}.scan.tif")
                for i in (1, 2, 3)
            ],
            Path("/m/linear_cells_model.pth") if cells else None,
            Path("/m/linear_rings_model.pth") if rings else None,
            False,
            ".cells",
            ".rings",
            2020,
            str(sample_image_dir),
        )

    def test_samples_are_written_and_failures_skipped(
        self, sample_image_dir, scans, capsys
    ):
        written, annotate = scans

        events, _ = _run(self._segment(sample_image_dir))

        assert written == [
            "sample_01.cells",
            "sample_01.rings",
            "sample_03.cells",
            "sample_03.rings",
        ]
        assert annotate.call_count == 2
        out = capsys.readouterr().out
        assert (
            "[Batch Segmentation] [1/3] sample_01.scan.tif: "
            "Cells (linear_cells_model.pth), Rings (linear_rings_model.pth)"
        ) in out
        assert (
            "[Batch Segmentation] [2/3] Error processing sample_02.scan.tif: "
            "broken"
        ) in out
        # Only the summary is shown
        messages = [e.text for e in events if isinstance(e, Message)]
        assert messages == [
            "[Batch Segmentation] Completed 2 of 3 file(s) successfully."
        ]

    def test_cells_only_writes_the_annotated_image(
        self, sample_image_dir, scans
    ):
        written, annotate = scans

        _run(self._segment(sample_image_dir, rings=False))

        assert written == ["sample_01.cells", "sample_03.cells"]
        args, kwargs = annotate.call_args_list[0]
        assert args[0] == (sample_image_dir / "sample_01").resolve()
        assert args[1].empty
        assert kwargs["cells_data"] is not None

    def test_cancelling_stops_between_samples(self, sample_image_dir, scans):
        written, _ = scans

        outcome = run_job_sync(
            Job("Batch segmentation", lambda: self._segment(sample_image_dir)),
            should_cancel=lambda: len(written) > 0,
        )

        assert outcome.status is JobStatus.CANCELLED
        # The first sample is complete, the next ones are not started
        assert written == ["sample_01.cells", "sample_01.rings"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
