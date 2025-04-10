import glob
import os
from typing import TYPE_CHECKING, Dict, List, Optional

from magicgui.widgets import (
    CheckBox,
    Container,
    ProgressBar,
    PushButton,
    Select,
)
from PIL import Image
from qtpy.QtCore import QThread
from qtpy.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from .._settings._settings_manager import SettingsManager
from ._file_overwrite_dialog import FileOverwriteDialog
from ._metadata_dialog import MetadataDialog
from ._worker import Worker

if TYPE_CHECKING:
    import napari

# Disable PIL's decompression bomb warning for large images
Image.MAX_IMAGE_PIXELS = None


class PreparationWidget(Container):
    """
    Widget for preparing sample images and metadata for ROXAS analysis.

    This widget provides UI controls for selecting directories and
    initiating the image preparation process.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """
        Initialize the preparation widget.

        Args:
            viewer: The napari viewer instance
        """
        super().__init__()
        self._viewer = viewer

        # Load settings
        self._load_settings()

        # Create UI components
        self._create_ui_components()

        # Initialize state variables
        self.source_directory = None
        self.project_directory = None
        self.worker = None
        self.worker_thread = None
        self.metadata_dialog = None
        self.same_directory = False
        self.source_files = []  # List of files in the source directory
        self.selected_files = []  # List of files selected by the user

    def _load_settings(self):
        """Load settings from the settings manager."""
        # Initialize settings manager
        self.settings_manager = SettingsManager()

        # Get authorized sample types and geometries
        self.authorized_sample_types = self.settings_manager.get(
            "authorised_sample_types",
            ["conifer", "angiosperm"],
        )
        self.authorized_sample_geometries = self.settings_manager.get(
            "authorised_sample_geometries",
            ["linear", "circular"],
        )

        # Get default scale and angle
        self.default_scale = self.settings_manager.get(
            "default_scale",
            2.2675,  # Default value in pixels/μm
        )
        self.default_angle = self.settings_manager.get("default_angle", 0.0)

        # Get file extension settings
        scan_file_extension = self.settings_manager.get(
            "scan_file_extension", [".scan", ".jpg"]
        )
        self.scan_file_prefix = (
            scan_file_extension[0] if scan_file_extension else ".scan"
        )

        metadata_file_extension_parts = self.settings_manager.get(
            "metadata_file_extension", [".metadata", ".json"]
        )
        self.metadata_file_extension = "".join(metadata_file_extension_parts)

        # Get ROXAS file extensions to avoid processing already prepared files
        self.roxas_file_extensions = self.settings_manager.get(
            "roxas_file_extensions", []
        )

        # Get supported image file extensions
        self.image_file_extensions = self.settings_manager.get(
            "image_file_extensions",
            [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".jp2"],
        )

    def _create_ui_components(self):
        """Create and configure UI components."""
        # Source directory selector
        self._source_dialog_button = PushButton(text="Source Directory: None")
        self._source_dialog_button.changed.connect(self._open_source_dialog)

        # Project directory selector
        self._project_dialog_button = PushButton(
            text="Project Directory: None"
        )
        self._project_dialog_button.changed.connect(self._open_project_dialog)

        # File selection controls
        self._handpick_files_checkbox = CheckBox(
            value=False, label="Handpick files from source directory"
        )
        self._handpick_files_checkbox.changed.connect(
            self._toggle_file_selection
        )

        # File selection container (initially hidden)
        self._file_selection_container = Container(
            widgets=[], labels=False, layout="vertical", visible=False
        )

        # Reverse selection button (initially hidden)
        self._reverse_selection_button = PushButton(
            text="Reverse Selection", visible=False
        )
        self._reverse_selection_button.changed.connect(
            self._reverse_file_selection
        )

        # Progress bar (initially hidden)
        self._progress_bar = ProgressBar(
            value=0, min=0, max=100, visible=False, label="Processing Progress"
        )

        # Start button
        self._start_button = PushButton(text="Start Processing")
        self._start_button.changed.connect(self._start_processing)

        # Cancel button (initially hidden)
        self._cancel_button = PushButton(text="Cancel", visible=False)
        self._cancel_button.changed.connect(self._cancel_processing)

        # Append all widgets to the container
        self.extend(
            [
                self._source_dialog_button,
                self._project_dialog_button,
                self._handpick_files_checkbox,
                self._file_selection_container,
                self._reverse_selection_button,
                self._start_button,
                self._cancel_button,
                self._progress_bar,
            ]
        )

    def _open_source_dialog(self):
        """Open file dialog to select source directory and refresh file list."""
        self.source_directory = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select Source Directory with Images",
        )
        if self.source_directory:
            self._source_dialog_button.text = (
                f"Source Directory: {self.source_directory}"
            )

            # If project directory is not defined yet, set it to the source directory
            if not self.project_directory:
                self.project_directory = self.source_directory
                self._project_dialog_button.text = (
                    f"Project Directory: {self.project_directory}"
                )

            # Reset file selection when source directory changes
            self._refresh_file_list()

    def _refresh_file_list(self):
        """Refresh the list of files from the source directory."""
        # Reset selected files list
        self.selected_files = []

        # Get all image files in source directory
        self.source_files = []

        if self.source_directory:
            # Gather all files with supported image extensions
            for ext in self.image_file_extensions:
                self.source_files.extend(
                    glob.glob(
                        f"{self.source_directory}/**/*{ext}", recursive=True
                    )
                )
                self.source_files.extend(
                    glob.glob(
                        f"{self.source_directory}/**/*{ext.upper()}",
                        recursive=True,
                    )
                )

            # Filter out files that already have ROXAS extensions
            if self.roxas_file_extensions:
                filtered_files = []
                for file_path in self.source_files:
                    file_name = os.path.basename(file_path)
                    skip_file = False

                    # Check if the file name contains any ROXAS extension
                    for ext in self.roxas_file_extensions:
                        if ext in file_name:
                            skip_file = True
                            break

                    if not skip_file:
                        filtered_files.append(file_path)

                self.source_files = filtered_files

        # Sort files for consistent display
        self.source_files = sorted(self.source_files)

        # Update the file selection widget if it's visible
        if self._handpick_files_checkbox.value:
            self._update_file_selection_widget()

    def _update_file_selection_widget(self):
        """Update the file selection widget with current source files."""
        # Clear the container
        self._file_selection_container.clear()

        # Create a new file selection widget
        if self.source_files:
            # Convert absolute paths to display names (relative to source directory)
            display_names = []
            self._path_mapping = {}  # Map display names to absolute paths

            for file_path in self.source_files:
                # Use relative path if possible, otherwise use base name
                try:
                    rel_path = os.path.relpath(
                        file_path, self.source_directory
                    )
                    display_names.append(rel_path)
                    self._path_mapping[rel_path] = file_path
                except ValueError:
                    base_name = os.path.basename(file_path)
                    display_names.append(base_name)
                    self._path_mapping[base_name] = file_path

            # Create a proper select widget with multiple selection
            self._file_select_widget = Select(
                choices=display_names,
                allow_multiple=True,
                label="Select Files to Process",
            )

            # Add the widget to the container
            self._file_selection_container.append(self._file_select_widget)

            # Make container and reverse button visible
            self._file_selection_container.visible = True
            self._reverse_selection_button.visible = True
        else:
            # Hide container and reverse button if no files
            self._file_selection_container.visible = False
            self._reverse_selection_button.visible = False

    def _toggle_file_selection(self):
        """Toggle the file selection interface."""
        if self._handpick_files_checkbox.value:
            # Turn on file selection mode
            self._update_file_selection_widget()
        else:
            # Turn off file selection mode
            self._file_selection_container.visible = False
            self._reverse_selection_button.visible = False
            # Reset file selection
            self.selected_files = []

    def _reverse_file_selection(self):
        """Reverse the current file selection."""
        if hasattr(self, "_file_select_widget"):
            # Get currently selected files
            current_selection = set(self._file_select_widget.value)

            # Create a new selection with all files except the current selection
            all_display_names = set(self._file_select_widget.choices)
            new_selection = list(all_display_names - current_selection)

            # Update the widget
            self._file_select_widget.value = new_selection

    def _get_selected_file_paths(self) -> List[str]:
        """
        Get the absolute paths of selected files.

        If no files are specifically selected (handpick mode off),
        returns an empty list, which means "use all files".
        """
        if not self._handpick_files_checkbox.value:
            # If handpick mode is off, return empty list to indicate "all files"
            return []

        # If handpick mode is on but nothing is selected, return empty list
        if (
            not hasattr(self, "_file_select_widget")
            or not self._file_select_widget.value
        ):
            return []

        # Convert selected display names back to absolute paths using our mapping
        selected_paths = []
        for display_name in self._file_select_widget.value:
            if display_name in self._path_mapping:
                selected_paths.append(self._path_mapping[display_name])

        return selected_paths

    def _open_project_dialog(self):
        """Open file dialog to select project directory."""
        self.project_directory = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select Project Directory",
        )
        if self.project_directory:
            self._project_dialog_button.text = (
                f"Project Directory: {self.project_directory}"
            )

    def _start_processing(self):
        """Start the file processing in a separate thread."""
        # Validate inputs
        if not self._validate_inputs():
            return

        # Check if source and project are the same directory
        self.same_directory = os.path.normpath(
            self.source_directory
        ) == os.path.normpath(self.project_directory)

        # Show warning for same directory mode
        if self.same_directory and not self._confirm_same_directory():
            return

        # Get selected files if in handpick mode
        selected_files = self._get_selected_file_paths()

        # Update UI for processing
        self._update_ui_for_processing(True)

        # Start processing in a separate thread
        self._run_in_thread(
            self.source_directory,
            self.project_directory,
            None,  # No default metadata initially
            False,  # Not applying to all initially
            selected_files,
        )

    def _validate_inputs(self) -> bool:
        """Validate user inputs before processing."""
        if not self.source_directory:
            QMessageBox.warning(
                None, "Warning", "Please select a source directory."
            )
            return False

        if not self.project_directory:
            QMessageBox.warning(
                None, "Warning", "Please select a project directory."
            )
            return False

        # If handpick mode is on but no files are selected, show a warning
        if (
            self._handpick_files_checkbox.value
            and hasattr(self, "_file_select_widget")
            and not self._file_select_widget.value
        ):
            QMessageBox.warning(
                None, "Warning", "No files selected for processing."
            )
            return False

        return True

    def _confirm_same_directory(self) -> bool:
        """Show confirmation dialog for same-directory mode."""
        confirm = QMessageBox()
        confirm.setIcon(QMessageBox.Warning)
        confirm.setWindowTitle("Warning: Source Modification")
        confirm.setText("The source and project directories are the same.")
        confirm.setInformativeText(
            f"Original image files will be replaced with {self.scan_file_prefix}* files. "
            "This operation cannot be undone."
        )
        confirm.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        confirm.setDefaultButton(QMessageBox.Cancel)

        result = confirm.exec_()
        return result == QMessageBox.Ok

    def _update_ui_for_processing(self, is_processing: bool):
        """Update UI elements based on processing state."""
        self._start_button.visible = not is_processing
        self._cancel_button.visible = is_processing
        self._progress_bar.visible = is_processing

        if is_processing:
            self._progress_bar.value = 0

    def _cancel_processing(self):
        """Cancel the current processing job."""
        if self.worker:
            self.worker.stop()

    def _update_progress(self, current, total):
        """Update the progress bar."""
        if total > 0:
            percentage = int(100 * current / total)
            self._progress_bar.value = percentage

    def _request_metadata(self, filename):
        """
        Show dialog to request metadata for a file.

        Args:
            filename: Base filename (without extension) for the current file
        """
        # Create default metadata with sample name and defaults
        default_metadata = {
            "sample_name": filename,
            "sample_type": (
                self.authorized_sample_types[0]
                if self.authorized_sample_types
                else "conifer"
            ),
            "sample_geometry": (
                self.authorized_sample_geometries[0]
                if self.authorized_sample_geometries
                else "linear"
            ),
            "sample_scale": self.default_scale,
            "sample_angle": self.default_angle,
        }

        # Show dialog to get metadata
        self.metadata_dialog = MetadataDialog(
            default_metadata,
            self.authorized_sample_types,
            self.authorized_sample_geometries,
        )
        result = self.metadata_dialog.exec_()

        if result:
            # User confirmed the dialog
            metadata = self.metadata_dialog.get_metadata()
            apply_to_all = self.metadata_dialog.apply_to_all()

            # Send metadata back to worker
            self.worker.set_metadata(metadata, apply_to_all)
        else:
            # User cancelled - stop processing
            self._cancel_processing()

    def _request_overwrite_confirmation(self, filename):
        """
        Show dialog to confirm file overwrite.

        Args:
            filename: Path of the file that would be overwritten
        """
        # Create and show the overwrite dialog
        dialog = FileOverwriteDialog(filename)
        result = dialog.exec_()

        if result:
            # User made a choice
            overwrite = dialog.should_overwrite()
            apply_to_all = dialog.apply_to_all()

            # Send the choice back to worker
            self.worker.set_overwrite_choice(overwrite, apply_to_all)
        else:
            # Dialog was canceled - stop processing
            self._cancel_processing()

    def _processing_finished(self):
        """Handle the completion of processing."""
        # Reset UI
        self._update_ui_for_processing(False)

        # Show completion message
        QMessageBox.information(
            None, "Complete", "File processing completed successfully."
        )

    def _run_in_thread(
        self,
        source_directory: str,
        target_directory: str,
        default_metadata: Optional[Dict] = None,
        apply_to_all: bool = False,
        selected_files: Optional[List[str]] = None,
    ):
        """
        Run the processing in a separate thread.

        Args:
            source_directory: Directory containing source images
            target_directory: Directory to save processed files
            default_metadata: Optional default metadata to use
            apply_to_all: Whether to apply default metadata to all files
            selected_files: Optional list of specific files to process
        """
        # Create thread and worker
        self.worker_thread = QThread()
        self.worker = Worker(
            source_directory,
            target_directory,
            default_metadata,
            apply_to_all,
            self.authorized_sample_types,
            self.default_scale,
            self.default_angle,
            self.same_directory,
            self.scan_file_prefix,
            self.metadata_file_extension,
            self.roxas_file_extensions,
            self.image_file_extensions,
            selected_files,
        )
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._processing_finished)
        self.worker.progress.connect(self._update_progress)
        self.worker.metadata_request.connect(self._request_metadata)
        self.worker.overwrite_request.connect(
            self._request_overwrite_confirmation
        )

        # Start the thread
        self.worker_thread.start()
