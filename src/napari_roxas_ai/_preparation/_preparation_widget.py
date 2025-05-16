"""
Widget for preparing sample images and metadata for ROXAS analysis.
"""

import glob
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from magicgui.widgets import (
    CheckBox,
    Container,
    ProgressBar,
    PushButton,
    Select,
)
from qtpy.QtCore import QThread
from qtpy.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from napari_roxas_ai._settings._settings_manager import SettingsManager

from ._crossdating_handler import process_crossdating_files
from ._metadata_dialog import MetadataDialog
from ._worker import Worker

if TYPE_CHECKING:
    import napari


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

        # Initialize state variables
        self.project_directory = None
        self.worker = None
        self.worker_thread = None
        self.source_files = []  # List of files in the project directory
        self.selected_files = []  # List of files selected by the user

        # Load settings
        self._load_settings()

        # Create UI components
        self._create_ui_components()

    def _load_settings(self):
        """Load settings from the settings manager."""
        # Initialize settings manager
        self.settings_manager = SettingsManager()

        # Get file extension settings
        scan_file_extension = self.settings_manager.get(
            "file_extensions.scan_file_extension"
        )
        self.scan_content_extension = (
            scan_file_extension[0] if scan_file_extension else ".scan"
        )

        metadata_file_extension_parts = self.settings_manager.get(
            "file_extensions.metadata_file_extension"
        )
        self.metadata_file_extension = "".join(metadata_file_extension_parts)

        # Get crossdating file extension
        crossdating_file_extension = self.settings_manager.get(
            "file_extensions.crossdating_file_extension"
        )
        self.crossdating_file_extension = "".join(crossdating_file_extension)

        # Get supported image file extensions
        self.image_file_extensions = self.settings_manager.get(
            "file_extensions.image_file_extensions"
        )

        # Get supported text file extensions for crossdating
        self.text_file_extensions = self.settings_manager.get(
            "file_extensions.text_file_extensions"
        )

    def _create_ui_components(self):
        """Create and configure UI components."""
        # Project directory selector
        self.project_directory = self.settings_manager.get("project_directory")
        self._project_dialog_button = PushButton(
            text=f"Project Directory: {self.project_directory}"
        )
        self._project_dialog_button.changed.connect(self._open_project_dialog)

        # Overwrite files checkbox
        self._overwrite_files_checkbox = CheckBox(
            value=True,
            label="Overwrite original files (replace instead of copy)",
        )
        # Temporary disabling the checkbox as it produces unexpected behavior in windows (deletes files instead of copying)
        self._overwrite_files_checkbox.value = False
        self._overwrite_files_checkbox.visible = False

        # Process already processed files checkbox
        self._process_processed_checkbox = CheckBox(
            value=False,
            label=f"Process already processed files (with {self.scan_content_extension} extension)",
        )
        self._process_processed_checkbox.changed.connect(
            self._refresh_file_list
        )

        # File selection controls
        self._handpick_files_checkbox = CheckBox(
            value=False, label="Manually select files to process"
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

        # Image processing button
        self._image_action_button = PushButton(
            text="Start Processing Image Files"
        )
        self._image_action_button.changed.connect(
            self._toggle_image_processing
        )

        # Crossdating processing button
        self._crossdating_action_button = PushButton(
            text="Start Processing Crossdating Files"
        )
        self._crossdating_action_button.changed.connect(
            self._process_crossdating_files
        )

        self._is_processing = False

        # Append all widgets to the container
        self.extend(
            [
                self._project_dialog_button,
                self._overwrite_files_checkbox,
                self._process_processed_checkbox,
                self._handpick_files_checkbox,
                self._file_selection_container,
                self._reverse_selection_button,
                self._image_action_button,
                self._crossdating_action_button,
                self._progress_bar,
            ]
        )
        self._refresh_file_list()

    def _open_project_dialog(self):
        """Open file dialog to select project directory and refresh file list."""
        directory = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select Project Directory",
            directory=self.project_directory,
        )
        if directory:
            self.project_directory = directory
            self._project_dialog_button.text = (
                f"Project Directory: {directory}"
            )
            self._refresh_file_list()

    def _refresh_file_list(self):
        """Refresh the list of files from the project directory."""
        # Reset selected files list
        self.selected_files = []

        # Get all image files in project directory
        self.source_files = []

        if self.project_directory and Path(self.project_directory).exists():
            # Gather all files with supported image extensions
            for ext in self.image_file_extensions:
                self.source_files.extend(
                    glob.glob(
                        str(Path(self.project_directory) / "**" / f"*{ext}"),
                        recursive=True,
                    )
                )
                self.source_files.extend(
                    glob.glob(
                        str(
                            Path(self.project_directory)
                            / "**"
                            / f"*{ext.upper()}"
                        ),
                        recursive=True,
                    )
                )

            # Filter based on whether to include already processed files

            if not self._process_processed_checkbox.value:
                filtered_files = []
                for file_path in self.source_files:
                    basename = Path(file_path).name
                    if self.scan_content_extension not in basename:
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
            # Convert absolute paths to display names (relative to project directory)
            display_names = []
            self._path_mapping = {}  # Map display names to absolute paths

            for file_path in self.source_files:
                # Use relative path if possible, otherwise use base name
                try:
                    rel_path = str(
                        Path(file_path).relative_to(self.project_directory)
                    )
                    display_names.append(rel_path)
                    self._path_mapping[rel_path] = file_path
                except ValueError:
                    basename = Path(file_path).name
                    display_names.append(basename)
                    self._path_mapping[basename] = file_path

            # Create a select widget with multiple selection
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

    def _toggle_image_processing(self):
        """Toggle between starting and canceling image processing."""
        if not self._is_processing:
            self._start_image_processing()
        else:
            self._cancel_processing()

    def _start_image_processing(self):
        """Start the image file processing in a separate thread."""
        # Validate inputs
        if not self._validate_inputs():
            return

        # Update UI for processing
        self._update_ui_for_processing(True)

        # Get selected files if in handpick mode
        selected_files = self._get_selected_file_paths()

        # Start processing in a separate thread
        self._run_in_thread(
            self.project_directory,
            selected_files,
            self._process_processed_checkbox.value,
        )

    def _process_crossdating_files(self):
        """Handle the selection and processing of crossdating files."""
        if not self.project_directory:
            QMessageBox.warning(
                None, "Warning", "Please select a project directory."
            )
            return

        # Process crossdating files
        result_df = process_crossdating_files(
            project_directory=self.project_directory,
            crossdating_file_extension=self.crossdating_file_extension,
            text_file_extensions=self.text_file_extensions,
        )

        # Result_df will be None if user canceled or if there was an error
        if result_df is not None:
            crossdating_file_path = (
                Path(self.project_directory)
                / f"rings_series{self.crossdating_file_extension}"
            )
            QMessageBox.information(
                None,
                "Crossdating Files Processed",
                f"Crossdating data has been processed and saved to:\n{crossdating_file_path}",
            )

    def _validate_inputs(self) -> bool:
        """Validate user inputs before processing."""
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

        # Check if there are any files to process
        if not self.source_files:
            QMessageBox.warning(
                None, "Warning", "No image files found to process."
            )
            return False

        return True

    def _update_ui_for_processing(self, is_processing: bool):
        """Update UI elements based on processing state."""
        self._is_processing = is_processing

        if is_processing:
            self._image_action_button.text = "Cancel Processing"
            self._progress_bar.visible = True
            self._progress_bar.value = 0
            self._project_dialog_button.enabled = False
            self._process_processed_checkbox.enabled = False
            self._handpick_files_checkbox.enabled = False
            self._file_selection_container.enabled = False
            self._reverse_selection_button.enabled = False
            self._crossdating_action_button.enabled = False
        else:
            self._image_action_button.text = "Start Processing Image Files"
            self._progress_bar.visible = False
            self._project_dialog_button.enabled = True
            self._process_processed_checkbox.enabled = True
            self._handpick_files_checkbox.enabled = True
            self._file_selection_container.enabled = True
            self._reverse_selection_button.enabled = True
            self._crossdating_action_button.enabled = True

    def _cancel_processing(self):
        """Cancel the current processing job."""
        if self.worker:
            self.worker.stop()

            # Wait for the thread to finish properly
            if self.worker_thread and self.worker_thread.isRunning():
                self.worker_thread.quit()
                self.worker_thread.wait()

        self._update_ui_for_processing(False)

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
        # Show dialog to get metadata
        metadata_dialog = MetadataDialog(filename)
        result = metadata_dialog.exec_()

        if result:
            # User confirmed the dialog - get metadata and apply_to_all flag
            metadata, apply_to_all = metadata_dialog.get_result()

            # Send metadata back to worker
            self.worker.set_metadata(metadata, apply_to_all)
        else:
            # User cancelled - stop processing
            self._cancel_processing()

    def _processing_finished(self):
        """Handle the completion of processing."""
        # Reset UI
        self._update_ui_for_processing(False)

        # Show completion message if not canceled
        if not (self.worker and self.worker.should_stop):
            QMessageBox.information(
                None, "Complete", "File processing completed successfully."
            )

    def _run_in_thread(
        self,
        project_directory: str,
        selected_files: Optional[List[str]] = None,
        process_processed: bool = False,
    ):
        """
        Run the processing in a separate thread.

        Args:
            project_directory: Directory to process
            selected_files: Optional list of specific files to process
            process_processed: Whether to process files that are already processed
        """
        # Create thread and worker
        self.worker_thread = QThread()

        self.worker = Worker(
            project_directory=project_directory,
            scan_content_extension=self.scan_content_extension,
            metadata_file_extension=self.metadata_file_extension,
            image_file_extensions=self.image_file_extensions,
            selected_files=selected_files,
            process_processed=process_processed,
            overwrite_files=self._overwrite_files_checkbox.value,
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

        # Start the thread
        self.worker_thread.start()
