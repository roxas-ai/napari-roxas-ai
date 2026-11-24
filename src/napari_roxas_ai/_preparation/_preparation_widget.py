"""
Widget for preparing sample images and metadata for ROXAS analysis.
"""

import glob
from pathlib import Path
from typing import TYPE_CHECKING, List

from magicgui.widgets import (
    CheckBox,
    Container,
    Label,
    PushButton,
    Select,
)
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from napari_roxas_ai._job_runner import JobButton, JobHandle, get_runner
from napari_roxas_ai._jobs import Job, JobStatus, JobTag
from napari_roxas_ai._settings._settings_manager import SettingsManager
from napari_roxas_ai._utils._project_files_events import (
    notify_project_files_changed,
)
from napari_roxas_ai.shortcuts import prevent_duplicate_dock_if_already_present

from ._crossdating_handler import process_crossdating_files
from ._metadata_dialog import MetadataDialog
from ._operations import (
    ROXAS_CLASSIC_OUTPUT_SUFFIXES,
    Answer,
    discover_files,
    is_roxas_ai_output_file,
    prepare_files,
    sample_base_name,
)

if TYPE_CHECKING:
    import napari

# The file list is at least this many rows high and otherwise grows with
# its content, one row per file, up to MAX_VISIBLE_IMAGE_ROWS
MIN_VISIBLE_IMAGE_ROWS = 5
MAX_VISIBLE_IMAGE_ROWS = 7

# Gap between a row's label and its box. Set explicitly because the style's
# default depends on the widget pair (on macOS label-to-button is wider than
# label-to-list), which would make the button and the list start unevenly.
LABEL_SPACING = 10

# Longest project directory shown on the button, longer paths are elided in the
# middle (the full path is in the tooltip)
MAX_PROJECT_DIRECTORY_CHARS = 40


def _elide_path(path: str, max_chars: int = MAX_PROJECT_DIRECTORY_CHARS) -> str:
    """
    Shorten `path` to at most `max_chars` characters by cutting out its middle.

    The start (drive, user folder) and the end (the project folder itself) are
    what tells projects apart, so both are kept and the middle gives way.
    """
    if len(path) <= max_chars:
        return path
    keep = max_chars - 1  # room for the ellipsis
    head = keep // 2
    tail = keep - head
    return f"{path[:head]}…{path[-tail:]}"


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
        # The rows bring their own labels, a label column would only indent them
        super().__init__(labels=False)
        self._viewer = viewer
        prevent_duplicate_dock_if_already_present(
            self, viewer, "1 - Prepare project images for analysis"
        )

        # Initialize state variables
        self.project_directory = None
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

    def refresh_from_settings(self):
        """
        Pick up settings changed while this widget was open.

        Called by the settings widget after applying. The file extensions are
        copied into this widget when it is built, so they are re-read, and the
        checkbox that shows one of them is relabelled.

        The project directory is not handled here: the settings listener
        registered in _create_ui_components keeps it up to date.
        """
        self._load_settings()

        # A CheckBox is a button, so magicgui renders .text, not .label
        self._process_processed_checkbox.text = (
            "Process already processed files "
            f"(with {self.scan_content_extension} extension)"
        )

    def _create_ui_components(self):
        """Create and configure UI components."""
        # Project directory selector
        self.project_directory = self.settings_manager.get("project_directory")

        # "Project Directory:" label next to a button showing the (elided) path
        self._project_dialog_button = PushButton()
        self._project_dialog_button.changed.connect(self._open_project_dialog)
        # Without a stylesheet, macOS draws a push button larger than its layout
        # slot (room for the focus ring), so it would stick out to the left of
        # the image list below; lay it out by its drawn rect instead
        self._project_dialog_button.native.setAttribute(
            Qt.WA_LayoutUsesWidgetRect
        )
        self._project_directory_label = Label(value="Project Directory:")
        self._project_directory_row = Container(
            widgets=[self._project_directory_label, self._project_dialog_button],
            labels=False,
            layout="horizontal",
        )
        self._project_directory_row.margins = (0, 0, 0, 0)
        label_width = self._project_directory_label.native.sizeHint().width() + 20
        self._project_directory_label.native.setFixedWidth(label_width)
        self._project_directory_label.native.setAlignment(
            Qt.AlignLeft | Qt.AlignVCenter
        )
        self._project_directory_row.native.layout().setSpacing(LABEL_SPACING)
        self._update_project_dialog_button()

        # Follow the project directory wherever it is changed (other widgets,
        # the menu, the settings widget)
        self.settings_manager.add_listener(
            "project_directory", self._on_project_directory_changed
        )

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
            value=True,
            label=f"Process already processed files (with {self.scan_content_extension} extension)",
        )
        self._process_processed_checkbox.changed.connect(
            self._refresh_file_list
        )

        # File selection controls
        self._handpick_files_checkbox = CheckBox(
            value=True, label="Manually select files to process"
        )
        self._handpick_files_checkbox.changed.connect(
            self._toggle_file_selection
        )

        # File selection container (initially hidden)
        self._file_selection_container = Container(
            widgets=[], labels=False, layout="vertical", visible=False
        )
        self._file_selection_container.margins = (0, 0, 0, 0)
        self._file_selection_container.native.layout().setAlignment(Qt.AlignTop)

        # Reverse selection button (initially hidden)
        self._reverse_selection_button = PushButton(
            text="Reverse Selection", visible=False
        )
        self._reverse_selection_button.changed.connect(
            self._reverse_file_selection
        )

        # Image processing button: shows the progress while the files are
        # found and prepared
        self._image_action_button = JobButton(
            text="Start Processing Image Files",
            start=self._start_image_processing,
        )

        # Crossdating processing button
        self._crossdating_action_button = PushButton(
            text="Start Processing Crossdating Files"
        )
        self._crossdating_action_button.changed.connect(
            self._process_crossdating_files
        )

        # Append all widgets to the container
        self.extend(
            [
                self._project_directory_row,
                self._overwrite_files_checkbox,
                self._process_processed_checkbox,
                self._handpick_files_checkbox,
                self._file_selection_container,
                self._reverse_selection_button,
                self._image_action_button,
                self._crossdating_action_button,
            ]
        )
        self.native.layout().setAlignment(Qt.AlignTop)
        self.native.layout().setAlignment(
            self._file_selection_container.native, Qt.AlignTop
        )
        self._refresh_file_list()

    def _update_project_dialog_button(self):
        """Show the project directory on the button, elided, full path as tooltip."""
        if self.project_directory:
            self._project_dialog_button.text = _elide_path(self.project_directory)
            self._project_dialog_button.tooltip = self.project_directory
        else:
            self._project_dialog_button.text = "Not set"
            self._project_dialog_button.tooltip = None

    def _on_project_directory_changed(self, directory):
        """Show a project directory that was changed anywhere in the plugin."""
        self.project_directory = directory
        self._update_project_dialog_button()
        self._refresh_file_list()

    def _open_project_dialog(self):
        """
        Open file dialog to select the project directory.

        The choice is stored in the settings, whose listener (see
        _on_project_directory_changed) updates this and every other widget.
        """
        directory = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select Project Directory",
            directory=self.project_directory,
        )
        if directory:
            self.settings_manager.set("project_directory", directory)

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

            # ROXAS AI outputs (.cells/.rings rasters, annotated image) and
            # ROXAS classic outputs are never inputs
            self.source_files = [
                f
                for f in self.source_files
                if not is_roxas_ai_output_file(f)
                and not f.endswith(ROXAS_CLASSIC_OUTPUT_SUFFIXES)
            ]

            # Filter based on whether to include already processed files

            if not self._process_processed_checkbox.value:
                filtered_files = []
                for file_path in self.source_files:
                    basename = Path(file_path).name
                    if self.scan_content_extension not in basename:
                        filtered_files.append(file_path)
                self.source_files = filtered_files

        # Sort files for consistent display and remove duplicates (e.g. on case-insensitive filesystems)
        self.source_files = sorted(list(set(self.source_files)))

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
            self._fit_file_list_height()

            # Make container and reverse button visible
            self._file_selection_container.visible = True
            self._reverse_selection_button.visible = True
        else:
            # Hide container and reverse button if no files
            self._file_selection_container.visible = False
            self._reverse_selection_button.visible = False

    def _fit_file_list_height(self):
        """
        Make the file list as high as its entries need, bounded between
        MIN_VISIBLE_IMAGE_ROWS and MAX_VISIBLE_IMAGE_ROWS rows.
        """
        list_widget = self._file_select_widget.native
        row_height = list_widget.sizeHintForRow(0)
        frame = list_widget.frameWidth()
        rows = min(
            max(list_widget.count(), MIN_VISIBLE_IMAGE_ROWS),
            MAX_VISIBLE_IMAGE_ROWS,
        )
        list_widget.setFixedHeight(rows * row_height + 2 * frame)

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

    def _start_image_processing(self) -> JobHandle | None:
        """
        Prepare the image files: they are found in the background, then the
        metadata dialogs are answered, then the files are prepared in the
        background. (2 Jobs)
        """
        if not self._validate_inputs():
            return None

        self._update_ui_for_processing(True)

        def on_found(files: List[str]) -> None:
            if not files:
                self._processing_finished(completed=True)
                return
            answers = self._ask_metadata(files)
            if not answers:
                # Cancelled at the first dialog: nothing to prepare
                self._processing_finished(completed=False)
                return

            def on_prepared(handle: JobHandle) -> None:
                # Not completed if cancelled or a dialog was cancelled
                self._processing_finished(
                    completed=handle.status is JobStatus.SUCCEEDED
                    and handle.result
                )

            self._image_action_button.track(
                get_runner().submit(
                    Job(
                        "Preparation",
                        prepare_files,
                        args=(
                            files,
                            answers,
                            self.project_directory,
                            self.scan_content_extension,
                            self.metadata_file_extension,
                            self.settings_manager.get(
                                "file_extensions.cells_file_extension"
                            ),
                            self.settings_manager.get(
                                "file_extensions.rings_file_extension"
                            ),
                        ),
                        tags={JobTag.PREPARATION},
                    ),
                    on_finished=on_prepared,
                )
            )

        def on_search_finished(handle: JobHandle) -> None:
            # Failed or cancelled: nothing was prepared
            if handle.status is not JobStatus.SUCCEEDED:
                self._update_ui_for_processing(False)

        return get_runner().submit(
            Job(
                "Finding the image files",
                discover_files,
                args=(
                    self.project_directory,
                    self.image_file_extensions,
                    self.scan_content_extension,
                    self._get_selected_file_paths(),
                    self._process_processed_checkbox.value,
                ),
                tags={JobTag.PREPARATION},
            ),
            on_result=on_found,
            on_finished=on_search_finished,
        )

    def _ask_metadata(self, files: List[str]) -> List[Answer]:
        """
        Ask the metadata of the files, one dialog per file until one applies
        to all the remaining files.
        """
        answers = []
        for file_path in files:
            dialog = MetadataDialog(
                sample_base_name(file_path, self.scan_content_extension)
            )
            if not dialog.exec_():
                break
            answers.append(dialog.get_result())
            if answers[-1][1]:  # apply to all
                break
        return answers

    def _process_crossdating_files(self):
        """Handle the selection and processing of crossdating files."""
        if not self.project_directory:
            QMessageBox.warning(
                None, "Warning", "Please select a project directory."
            )
            return

        # Process crossdating files
        process_crossdating_files(
            project_directory=self.project_directory,
            crossdating_file_extension=self.crossdating_file_extension,
            text_file_extensions=self.text_file_extensions,
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
        """Lock the controls while the files are prepared."""
        for control in (
            self._project_dialog_button,
            self._process_processed_checkbox,
            self._handpick_files_checkbox,
            self._file_selection_container,
            self._reverse_selection_button,
            self._crossdating_action_button,
        ):
            control.enabled = not is_processing

    def _processing_finished(self, completed: bool):
        """Handle the end of the preparation (completed or not)."""
        # Reset UI
        self._update_ui_for_processing(False)

        # Let the widgets listing the project files (3B, 9B) rescan them,
        # also after a cancel: the files prepared until then are on disk
        notify_project_files_changed(self.project_directory, source=self)

        if completed:
            show_info("File processing completed successfully.")
