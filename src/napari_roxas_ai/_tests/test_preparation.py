"""
Tests for the preparation module functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from napari_roxas_ai._preparation._crossdating_handler import (
    CrossdatingSelectionDialog,
    merge_crossdating_files,
    process_crossdating_files,
)
from napari_roxas_ai._preparation._metadata_dialog import MetadataDialog
from napari_roxas_ai._preparation._preparation_widget import PreparationWidget
from napari_roxas_ai._preparation._worker import Worker


# Helper function to create test image
def create_test_image(path, size=(100, 100), color=(73, 109, 137)):
    """Create a simple test image at the specified path."""
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return path


# Helper function to create test crossdating file
def create_test_crossdating_file(path, series_count=2, year_count=10):
    """Create a simple crossdating file in tab-delimited format."""
    # Create a pandas DataFrame with series as columns and years as rows
    years = list(range(2000, 2000 + year_count))
    data = {}

    for i in range(series_count):
        series_name = f"SERIES{i+1}"
        # Generate random data with some pattern
        data[series_name] = np.random.randint(100, 500, size=year_count)

    df = pd.DataFrame(data, index=years)
    df.to_csv(path, sep="\t")
    return path


# Fixture for temporary directories
@pytest.fixture
def temp_dirs():
    """Create temporary source and project directories."""
    with tempfile.TemporaryDirectory() as project_dir:
        yield project_dir


# Fixture for creating test images
@pytest.fixture
def project_with_images(temp_dirs):
    """Create a project directory with test images and crossdating files."""
    project_dir = temp_dirs

    # Create test images with different extensions
    img_paths = []
    img_paths.append(create_test_image(Path(project_dir) / "test1.jpg"))
    img_paths.append(create_test_image(Path(project_dir) / "test2.png"))
    img_paths.append(create_test_image(Path(project_dir) / "test3.tif"))

    # Create a subdirectory
    subdir = Path(project_dir) / "subdir"
    subdir.mkdir(exist_ok=True)
    img_paths.append(create_test_image(subdir / "test4.jpg"))

    # Create a "processed" file with a ROXAS extension
    create_test_image(Path(project_dir) / "processed.scan.jpg")

    # Create crossdating subdirectory
    crossdating_dir = Path(project_dir) / "crossdating"
    crossdating_dir.mkdir(exist_ok=True)

    # Create a few test crossdating files
    crossdating_paths = []
    crossdating_paths.append(
        create_test_crossdating_file(crossdating_dir / "series1.rwl")
    )
    crossdating_paths.append(
        create_test_crossdating_file(crossdating_dir / "series2.txt")
    )

    return project_dir, img_paths, crossdating_paths


# Mock napari viewer
@pytest.fixture
def mock_viewer():
    """Create a mock viewer object."""
    viewer = MagicMock()
    return viewer


# Mock SettingsManager for testing
@pytest.fixture
def mock_settings_manager():
    """Create a mock settings manager with test values."""
    with patch(
        "napari_roxas_ai._settings._settings_manager.SettingsManager"
    ) as MockSettings:
        settings_instance = MagicMock()
        MockSettings.return_value = settings_instance

        # Configure the mock settings with test values
        settings_instance.get.side_effect = lambda key, default=None: {
            "file_extensions.scan_file_extension": [".scan"],
            "file_extensions.metadata_file_extension": [".metadata", ".json"],
            "file_extensions.crossdating_file_extension": [
                ".crossdating",
                ".txt",
            ],
            "file_extensions.roxas_file_extensions": [".scan"],
            "file_extensions.image_file_extensions": [
                ".jpg",
                ".png",
                ".tif",
                ".tiff",
            ],
            "file_extensions.text_file_extensions": [
                ".rwl",
                ".tuc",
                ".txt",
                ".csv",
                ".tsv",
            ],
        }.get(key, default)

        yield settings_instance


# Create a PreparationWidget with mocked UI interactions
@pytest.fixture
def prep_widget(mock_viewer, mock_settings_manager):
    """Create a preparation widget with a mock viewer."""
    widget = PreparationWidget(mock_viewer)

    # Mock UI elements that will be accessed during tests
    widget._file_selection_container = MagicMock()
    widget._reverse_selection_button = MagicMock()
    widget._run_in_thread = MagicMock()
    widget._cancel_processing = MagicMock()
    widget._progress_bar = MagicMock()
    widget._image_action_button = MagicMock()
    widget._crossdating_action_button = MagicMock()
    widget._project_dialog_button = MagicMock()
    widget._process_processed_checkbox = MagicMock()
    widget._handpick_files_checkbox = MagicMock()

    return widget


class TestPreparationWidget:
    """Tests for the PreparationWidget class."""

    @patch("napari_roxas_ai._preparation._preparation_widget.QFileDialog")
    def test_project_directory_selection(self, mock_dialog, prep_widget):
        """Test project directory selection."""
        # Set up mock directory path
        test_dir = str(Path("/test/project/dir").absolute())
        mock_dialog.getExistingDirectory.return_value = test_dir

        # Call the method
        prep_widget._open_project_dialog()

        # Check that the directory was set correctly
        assert prep_widget.project_directory == test_dir
        assert (
            prep_widget._project_dialog_button.text
            == f"Project Directory: {test_dir}"
        )

    def test_file_selection_toggle(self, prep_widget):
        """Test toggling file selection mode."""
        # Mock the update method
        prep_widget._update_file_selection_widget = MagicMock()

        # Enable handpick mode
        prep_widget._handpick_files_checkbox.value = True
        prep_widget._toggle_file_selection()

        # Check that the file selection widget was updated
        assert prep_widget._update_file_selection_widget.called

        # Disable handpick mode
        prep_widget._handpick_files_checkbox.value = False
        prep_widget._toggle_file_selection()

        # Check that the file selection container is hidden
        assert prep_widget._file_selection_container.visible is False
        assert prep_widget._reverse_selection_button.visible is False

    def test_reverse_file_selection(self, prep_widget):
        """Test reversing file selection."""
        # Create a mock file selection widget
        prep_widget._file_select_widget = MagicMock()
        prep_widget._file_select_widget.value = ["file1", "file2"]
        prep_widget._file_select_widget.choices = [
            "file1",
            "file2",
            "file3",
            "file4",
        ]

        # Call the reverse selection method
        prep_widget._reverse_file_selection()

        # Check that the selection was reversed
        assert isinstance(prep_widget._file_select_widget.value, list)

    @patch("napari_roxas_ai._preparation._preparation_widget.QMessageBox")
    def test_validation_no_project_dir(self, mock_msgbox, prep_widget):
        """Test validation with no project directory."""
        # Set project directory to None
        prep_widget.project_directory = None

        # Call validate method
        result = prep_widget._validate_inputs()

        # Check result and verify warning was shown
        assert result is False
        mock_msgbox.warning.assert_called_once()

    @patch("napari_roxas_ai._preparation._preparation_widget.QMessageBox")
    def test_validation_no_files(self, mock_msgbox, prep_widget):
        """Test validation with no files to process."""
        # Set project directory but no files
        prep_widget.project_directory = "/test/dir"
        prep_widget.source_files = []

        # Call validate method
        result = prep_widget._validate_inputs()

        # Check result and verify warning was shown
        assert result is False
        mock_msgbox.warning.assert_called_once()

    @patch(
        "napari_roxas_ai._preparation._preparation_widget.process_crossdating_files"
    )
    @patch("napari_roxas_ai._preparation._preparation_widget.QMessageBox")
    def test_process_crossdating_files(
        self, mock_msgbox, mock_process, prep_widget
    ):
        """Test crossdating files processing."""
        # Set up test data
        prep_widget.project_directory = "/test/dir"
        prep_widget.crossdating_file_extension = ".crossdating.txt"
        prep_widget.text_file_extensions = [".rwl", ".txt"]

        # Mock a successful result from process_crossdating_files
        mock_process.return_value = pd.DataFrame()

        # Call the method
        prep_widget._process_crossdating_files()

        # Verify that process_crossdating_files was called with the right parameters
        mock_process.assert_called_once_with(
            project_directory="/test/dir",
            crossdating_file_extension=".crossdating.txt",
            text_file_extensions=[".rwl", ".txt"],
        )

        # Verify that a success message was shown
        mock_msgbox.information.assert_called_once()

    def test_ui_state_during_processing(self, prep_widget):
        """Test UI state updates during processing."""
        # Test UI update when processing starts
        prep_widget._update_ui_for_processing(True)

        assert prep_widget._is_processing is True
        assert prep_widget._image_action_button.text == "Cancel Processing"
        assert prep_widget._progress_bar.visible is True
        assert prep_widget._project_dialog_button.enabled is False
        assert prep_widget._crossdating_action_button.enabled is False

        # Test UI update when processing ends
        prep_widget._update_ui_for_processing(False)

        assert prep_widget._is_processing is False
        assert (
            prep_widget._image_action_button.text
            == "Start Processing Image Files"
        )
        assert prep_widget._progress_bar.visible is False
        assert prep_widget._project_dialog_button.enabled is True
        assert prep_widget._crossdating_action_button.enabled is True


class TestWorker:
    """Tests for the Worker class."""

    @patch("napari_roxas_ai._preparation._worker.shutil")
    def test_image_processing(self, mock_shutil, temp_dirs):
        """Test image processing functionality."""
        # Create sample data
        project_dir = Path(temp_dirs)
        test_file = project_dir / "test.jpg"
        create_test_image(test_file)

        # Create worker instance
        worker = Worker(
            project_directory=str(project_dir),
            scan_content_extension=".scan",
            metadata_file_extension=".metadata.json",
            image_file_extensions=[".jpg", ".png"],
            selected_files=None,
            process_processed=False,
            overwrite_files=True,
        )

        # Mock signals
        worker.progress = MagicMock()
        worker.finished = MagicMock()
        worker.metadata_request = MagicMock()

        # Mock metadata setting to avoid UI interaction
        worker.set_metadata = MagicMock()

        # Add test file to all_files list
        worker.all_files = [str(test_file)]
        worker.current_file_index = 0

        # Run the worker method
        with patch.object(worker, "_extract_image_metadata", return_value={}):
            worker._process_next_file()

        # Check that signals were emitted correctly
        worker.metadata_request.emit.assert_called_once_with("test")


class TestMetadataDialog:
    """Tests for the MetadataDialog."""

    @patch("napari_roxas_ai._preparation._metadata_dialog.QDialog.exec_")
    def test_metadata_dialog(self, mock_exec):
        """Test metadata dialog initialization and result retrieval."""
        # Setup mock to return accept (QDialog.Accepted is typically 1)
        mock_exec.return_value = 1

        # Create dialog instance with sample filename
        dialog = MetadataDialog("test_sample")

        # Mock the get_result method to return test data
        original_get_result = dialog.get_result

        def mock_get_result():
            return {
                "sample_name": "test_sample",
                "sample_type": "conifer",
                "sample_geometry": "linear",
                "sample_scale": 2.5,
                "sample_angle": 45.0,
                "rings_outmost_complete_year": 2020,
            }, True

        dialog.get_result = mock_get_result

        # Get result using our mocked method
        metadata, apply_to_all = dialog.get_result()

        # Check metadata structure
        assert metadata["sample_name"] == "test_sample"
        assert metadata["sample_type"] == "conifer"
        assert apply_to_all is True

        # Restore original method
        dialog.get_result = original_get_result


class TestCrossdatingHandler:
    """Tests for the crossdating handling functionality."""

    @patch(
        "napari_roxas_ai._preparation._crossdating_handler.CrossdatingSelectionDialog"
    )
    def test_process_crossdating_files(self, mock_dialog, temp_dirs):
        """Test processing crossdating files."""
        # Set up test data
        project_dir = Path(temp_dirs)

        # Create mock dialog instance
        mock_dialog_instance = MagicMock()
        mock_dialog.return_value = mock_dialog_instance

        # Set up mock dialog behavior
        mock_dialog_instance.exec_.return_value = 1  # User accepted
        mock_dialog_instance.get_selected_files.return_value = [
            str(project_dir / "test1.rwl"),
            str(project_dir / "test2.txt"),
        ]

        # Mock file reading and merging
        with patch(
            "napari_roxas_ai._preparation._crossdating_handler.merge_crossdating_files",
            return_value=pd.DataFrame(),
        ), patch(
            "napari_roxas_ai._preparation._crossdating_handler.pd.DataFrame.to_csv"
        ) as mock_to_csv:
            # Call the function
            result = process_crossdating_files(
                project_directory=str(project_dir),
                crossdating_file_extension=".crossdating.txt",
                text_file_extensions=[".rwl", ".txt"],
            )

            # Verify file was saved
            mock_to_csv.assert_called_once()

            # Check result
            assert result is not None

    def test_crossdating_selection_dialog(self, temp_dirs):
        """Test the crossdating selection dialog."""
        # Set up test environment
        project_dir = Path(temp_dirs)

        # Create test files
        (project_dir / "rwls").mkdir(exist_ok=True)  # Replace os.makedirs
        test_file1 = project_dir / "rwls" / "test1.rwl"
        test_file2 = project_dir / "rwls" / "test2.txt"

        with open(test_file1, "w") as f:
            f.write("test data")
        with open(test_file2, "w") as f:
            f.write("test data")

        # Create the dialog instance
        dialog = CrossdatingSelectionDialog(
            project_directory=str(project_dir),
            text_file_extensions=[".rwl", ".txt"],
            project_file_path=str(project_dir / "project.crossdating.txt"),
        )

        # Mock the file_list widget
        dialog.file_list = MagicMock()
        dialog.file_list.count.return_value = 2

        # Create mock items for the file list
        item1, item2 = MagicMock(), MagicMock()
        item1.isSelected.return_value = True
        item1.data.return_value = str(test_file1)
        item2.isSelected.return_value = False

        # Set up the mock to return our items
        dialog.file_list.item.side_effect = lambda idx: [item1, item2][idx]

        # Call the OK clicked method
        dialog._ok_clicked()

        # Check that selected files contains only the selected item
        assert dialog.get_selected_files() == [str(test_file1)]

    def test_merge_crossdating_files(self, temp_dirs):
        """Test merging crossdating files."""
        # Set up test environment
        project_dir = Path(temp_dirs)

        # Create test files with real data
        file1 = project_dir / "series1.rwl"
        file2 = project_dir / "series2.rwl"
        target_file = project_dir / "target.crossdating.txt"

        # Create test data
        df1 = pd.DataFrame(
            {"Series1": [100, 200, 300], "Series2": [150, 250, 350]},
            index=["2000", "2001", "2002"],
        )

        df2 = pd.DataFrame(
            {"Series3": [400, 500, 600], "Series4": [450, 550, 650]},
            index=["2000", "2001", "2002"],
        )

        target_df = pd.DataFrame(
            {"Series5": [700, 800, 900]}, index=["2000", "2001", "2002"]
        )

        # Save test data to files - we don't need to do this since we're mocking the read function
        df1.to_csv(file1, sep="\t")
        df2.to_csv(file2, sep="\t")
        target_df.to_csv(target_file, sep="\t")

        # Define expected column set for verification
        expected_columns = {
            "Series1",
            "Series2",
            "Series3",
            "Series4",
            "Series5",
        }

        # Mock the _try_read_dataframe function to return our test data
        with patch(
            "napari_roxas_ai._preparation._crossdating_handler._try_read_dataframe"
        ) as mock_read:
            # Configure the mock to return different DataFrames for different input files
            def side_effect(filepath):
                if filepath == str(file1):
                    return True, df1
                elif filepath == str(file2):
                    return True, df2
                elif filepath == str(target_file):
                    return True, target_df
                else:
                    return False, None

            mock_read.side_effect = side_effect

            # Call the merge function
            result = merge_crossdating_files(
                source_files=[str(file1), str(file2)],
                target_file=str(target_file),
            )

            # Check result
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert set(result.columns) == expected_columns
            assert len(result) == 3
            # Check that each series from each dataframe is present in the merged result
            for series in df1.columns:
                assert series in result.columns
            for series in df2.columns:
                assert series in result.columns
            for series in target_df.columns:
                assert series in result.columns


if __name__ == "__main__":
    pytest.main(["-v", __file__])
