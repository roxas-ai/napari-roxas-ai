import glob
from pathlib import Path
from typing import TYPE_CHECKING, List

from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    Label,
    PushButton,
    Select,
)
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QMessageBox

from napari_roxas_ai._assets_files import check_assets_and_download
from napari_roxas_ai._job_runner import JobButton, JobHandle, get_runner
from napari_roxas_ai._jobs import Job, JobTag
from napari_roxas_ai._settings import SettingsManager
from napari_roxas_ai.shortcuts import prevent_duplicate_dock_if_already_present

from .._utils._fix_sample_stem_path import fix_sample_stem_paths_in_project
from .._utils._project_files_events import (
    add_project_files_listener,
    is_same_directory,
    notify_project_files_changed,
)
from ._operations import segment_samples

if TYPE_CHECKING:
    import napari

MODULE_PATH = Path(__file__).parent.absolute()
CELLS_MODELS_PATH = MODULE_PATH / "_models" / "_cells"
RINGS_MODELS_PATH = MODULE_PATH / "_models" / "_rings"
DEFAULT_RINGS_MODEL = "linear_rings_model.pth"

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


settings = SettingsManager()


class BatchSampleSegmentationWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        # The rows bring their own labels, a label column would only indent them
        super().__init__(labels=False)
        self._viewer = viewer
        prevent_duplicate_dock_if_already_present(
            self, viewer, "3B - Batch detect cells & rings"
        )

        # Ensure model weights are present before listing them below.
        # (Moved out of import time; only downloads if the dirs are missing.)
        check_assets_and_download(str(CELLS_MODELS_PATH), "cells_models.zip")
        check_assets_and_download(str(RINGS_MODELS_PATH), "rings_models.zip")

        # The project directory is the input directory
        self.input_directory_path = settings.get("project_directory")
        self.source_files = []
        self.selected_files = []
        self._path_mapping = {}

        # "Project Directory:" label next to a button showing the (elided) path
        self._input_file_dialog_button = PushButton()
        self._input_file_dialog_button.changed.connect(
            self._open_input_file_dialog
        )
        # Without a stylesheet, macOS draws a push button larger than its layout
        # slot (room for the focus ring), so it would stick out to the left of
        # the list below; lay it out by its drawn rect instead
        self._input_file_dialog_button.native.setAttribute(
            Qt.WA_LayoutUsesWidgetRect
        )
        self._project_directory_label = Label(value="Project Directory:")
        self._project_directory_row = Container(
            widgets=[
                self._project_directory_label,
                self._input_file_dialog_button,
            ],
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
        settings.add_listener(
            "project_directory", self._on_project_directory_changed
        )
        # Follow the files other widgets write to the project (preparation),
        # the list is otherwise only read when the widget opens
        add_project_files_listener(self._on_project_files_changed)

        # Process already processed files checkbox
        self._process_processed_checkbox = CheckBox(
            value=True,
            label="Include Previously Segmented Images",
            tooltip="Also include images for which a .rings.tif or .cells.png file already exists.",
        )
        self._process_processed_checkbox.changed.connect(
            self._refresh_file_list
        )

        # File selection controls
        self._handpick_files_checkbox = CheckBox(
            value=True, label="Manually Select Files To Process:"
        )
        self._handpick_files_checkbox.changed.connect(
            self._toggle_file_selection
        )

        # File selection container (initially hidden)
        self._file_selection_container = Container(
            widgets=[], labels=False, layout="vertical", visible=False
        )
        self._file_selection_container.margins = (0, 0, 0, 0)
        if hasattr(self._file_selection_container.native, "layout") and self._file_selection_container.native.layout() is not None:
            self._file_selection_container.native.layout().setAlignment(Qt.AlignTop)

        # Reverse selection button (initially hidden)
        self._reverse_selection_button = PushButton(
            text="Reverse Selection", visible=False
        )
        self._reverse_selection_button.changed.connect(
            self._reverse_file_selection
        )

        # Cells segmentation checkbox and model selection
        self._segment_cells_checkbox = CheckBox(
            value=True, label="Cell Segmentation:"
        )
        self._segment_cells_checkbox.changed.connect(
            self._update_cells_model_visibility
        )

        # Cells model selection
        self._cells_model_weights_file = ComboBox(
            choices=self._get_model_files(where=CELLS_MODELS_PATH),
            label="",
        )

        self._cells_row = Container(
            widgets=[
                self._segment_cells_checkbox,
                self._cells_model_weights_file,
            ],
            layout="horizontal",
            labels=True,
        )
        if hasattr(self._cells_row.native, "layout"):
            layout = self._cells_row.native.layout()
            if layout is not None:
                layout.setContentsMargins(0, 0, 0, 0)

        # Rings segmentation checkbox and model selection
        self._segment_rings_checkbox = CheckBox(
            value=True, label="Ring Segmentation:"
        )
        self._segment_rings_checkbox.changed.connect(
            self._update_rings_model_visibility
        )

        # Rings model selection
        self._rings_model_weights_file = ComboBox(
            choices=self._get_model_files(where=RINGS_MODELS_PATH),
            label="",
        )
        # The rings folder also holds the model for rerunning the editor's
        # segmentation from a given year; keep the standard model preselected
        if DEFAULT_RINGS_MODEL in self._rings_model_weights_file.choices:
            self._rings_model_weights_file.value = DEFAULT_RINGS_MODEL

        self._rings_row = Container(
            widgets=[
                self._segment_rings_checkbox,
                self._rings_model_weights_file,
            ],
            layout="horizontal",
            labels=True,
        )
        if hasattr(self._rings_row.native, "layout"):
            layout = self._rings_row.native.layout()
            if layout is not None:
                layout.setContentsMargins(0, 0, 0, 0)

        # Equalize checkbox widths so the model selection dropdowns align
        # and take equal width, with sufficient padding to prevent truncation
        checkbox_width = max(
            self._segment_cells_checkbox.native.sizeHint().width(),
            self._segment_rings_checkbox.native.sizeHint().width(),
        ) + 25
        self._segment_cells_checkbox.native.setFixedWidth(checkbox_width)
        self._segment_rings_checkbox.native.setFixedWidth(checkbox_width)

        # Equalize model selection dropdown widths based on all available choices
        # across cell and ring models
        all_model_choices = list(self._cells_model_weights_file.choices) + list(
            self._rings_model_weights_file.choices
        )
        font_metrics = self._cells_model_weights_file.native.fontMetrics()
        max_text_width = max(
            (
                font_metrics.horizontalAdvance(str(choice))
                for choice in all_model_choices
            ),
            default=0,
        )
        model_dropdown_width = max_text_width + 70
        self._cells_model_weights_file.native.setFixedWidth(model_dropdown_width)
        self._rings_model_weights_file.native.setFixedWidth(model_dropdown_width)

        if hasattr(self._cells_row.native, "layout") and self._cells_row.native.layout() is not None:
            cells_layout = self._cells_row.native.layout()
            cells_layout.setSpacing(LABEL_SPACING)
            cells_layout.addStretch()
        if hasattr(self._rings_row.native, "layout") and self._rings_row.native.layout() is not None:
            rings_layout = self._rings_row.native.layout()
            rings_layout.setSpacing(LABEL_SPACING)
            rings_layout.addStretch()

        # Shows the progress while it runs, can be cancelled between images
        self._run_segmentation_button = JobButton(
            text="Run Segmentation", start=self._run_segmentation
        )

        # Append the widgets to the container
        self.extend(
            [
                self._project_directory_row,
                self._process_processed_checkbox,
                self._handpick_files_checkbox,
                self._file_selection_container,
                self._reverse_selection_button,
                self._cells_row,
                self._rings_row,
                self._run_segmentation_button,
            ]
        )
        if hasattr(self.native, "layout") and self.native.layout() is not None:
            self.native.layout().setAlignment(Qt.AlignTop)
            self.native.layout().setAlignment(
                self._file_selection_container.native, Qt.AlignTop
            )

        # Initialize visibility of model selection widgets, button state, and file list
        self._update_cells_model_visibility()
        self._update_rings_model_visibility()
        self._update_run_button_state()
        self._refresh_file_list()

    def _update_project_dialog_button(self):
        """Show the project directory on the button, elided, full path as tooltip."""
        if self.input_directory_path:
            self._input_file_dialog_button.text = _elide_path(
                self.input_directory_path
            )
            self._input_file_dialog_button.tooltip = self.input_directory_path
        else:
            self._input_file_dialog_button.text = "Not set"
            self._input_file_dialog_button.tooltip = None

    def _on_project_directory_changed(self, directory):
        """Show a project directory that was changed anywhere in the plugin."""
        self.input_directory_path = directory
        self._update_project_dialog_button()
        self._refresh_file_list()

    def _on_project_files_changed(self, directory):
        """Rescan the file list after another widget wrote files to the project."""
        if not is_same_directory(directory, self.input_directory_path):
            return

        # Keep a manual selection, as far as the files are still listed
        selection = []
        if self._handpick_files_checkbox.value and hasattr(
            self, "_file_select_widget"
        ):
            selection = list(self._file_select_widget.value)

        self._refresh_file_list()

        if selection and self.source_files:
            choices = self._file_select_widget.choices
            self._file_select_widget.value = [
                name for name in selection if name in choices
            ]

    def _refresh_file_list(self):
        """Refresh the list of scan files from the project directory."""
        self.selected_files = []
        self.source_files = []

        if self.input_directory_path and Path(self.input_directory_path).exists():
            scan_content_ext = settings.get(
                "file_extensions.scan_file_extension"
            )[0]
            # Gather all files containing scan content extension
            raw_scan_files = glob.glob(
                str(
                    Path(self.input_directory_path)
                    / "**"
                    / f"*{scan_content_ext}.*"
                ),
                recursive=True,
            )

            # Filter out non-scan files and output files
            cells_ext = "".join(settings.get("file_extensions.cells_file_extension"))
            rings_ext = "".join(settings.get("file_extensions.rings_file_extension"))

            valid_scan_files = []
            for file_path in raw_scan_files:
                p = Path(file_path)
                # Ignore cells and rings files in case extension matching picked them up
                if p.name.lower().endswith(cells_ext.lower()) or p.name.lower().endswith(rings_ext.lower()):
                    continue
                valid_scan_files.append(file_path)

            if not self._process_processed_checkbox.value:
                # Exclude files that already have corresponding .rings.tif or .cells.png
                unprocessed_files = []
                for file_path in valid_scan_files:
                    p = Path(file_path)
                    parent = p.parent
                    # Derive sample stem from the scan file path
                    # e.g., name is sample.scan.jpg -> stem without extensions or matching stem
                    name = p.name
                    if scan_content_ext in name:
                        stem = name.split(scan_content_ext)[0]
                    else:
                        stem = p.stem

                    cells_file = parent / f"{stem}{cells_ext}"
                    rings_file = parent / f"{stem}{rings_ext}"

                    if not (cells_file.exists() or rings_file.exists()):
                        unprocessed_files.append(file_path)
                self.source_files = unprocessed_files
            else:
                self.source_files = valid_scan_files

        # Sort files for consistent display and remove duplicates
        self.source_files = sorted(list(set(self.source_files)))

        # Update the file selection widget if it's visible
        if self._handpick_files_checkbox.value:
            self._update_file_selection_widget()

    def _update_file_selection_widget(self):
        """Update the file selection widget with current source files."""
        self._file_selection_container.clear()

        if self.source_files:
            display_names = []
            self._path_mapping = {}

            for file_path in self.source_files:
                try:
                    rel_path = str(
                        Path(file_path).relative_to(self.input_directory_path)
                    )
                    display_names.append(rel_path)
                    self._path_mapping[rel_path] = file_path
                except ValueError:
                    basename = Path(file_path).name
                    display_names.append(basename)
                    self._path_mapping[basename] = file_path

            self._file_select_widget = Select(
                choices=display_names,
                allow_multiple=True,
                label="Select Files to Process",
            )

            self._file_selection_container.append(self._file_select_widget)
            self._fit_file_list_height()

            self._file_selection_container.visible = True
            self._reverse_selection_button.visible = True
        else:
            self._file_selection_container.visible = False
            self._reverse_selection_button.visible = False

    def _fit_file_list_height(self):
        """
        Make the file list as high as its entries need, bounded between
        MIN_VISIBLE_IMAGE_ROWS and MAX_VISIBLE_IMAGE_ROWS rows.
        """
        if not hasattr(self, "_file_select_widget"):
            return
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
            # Rescan the project directory, it may have changed on disk
            self._refresh_file_list()
        else:
            self._file_selection_container.visible = False
            self._reverse_selection_button.visible = False
            self.selected_files = []

    def _reverse_file_selection(self):
        """Reverse the current file selection."""
        if hasattr(self, "_file_select_widget"):
            current_selection = set(self._file_select_widget.value)
            all_display_names = set(self._file_select_widget.choices)
            new_selection = list(all_display_names - current_selection)
            self._file_select_widget.value = new_selection

    def _get_selected_file_paths(self) -> List[str]:
        """
        Get the absolute paths of selected files.

        Returns:
            List of absolute file paths
        """
        if self._handpick_files_checkbox.value and hasattr(
            self, "_file_select_widget"
        ):
            selected_display_names = self._file_select_widget.value
            return [
                self._path_mapping[name]
                for name in selected_display_names
                if name in self._path_mapping
            ]
        return self.source_files

    def _open_input_file_dialog(self):
        """
        Open a file dialog to select the project directory.

        The choice is stored in the settings, whose listener (see
        _on_project_directory_changed) updates this and every other widget.
        """
        directory = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select Project Directory",
            directory=self.input_directory_path,
        )
        if directory:
            settings.set("project_directory", directory)

    def _get_model_files(self, where: str) -> tuple:
        """Get available model weight files from the weights directory."""
        return tuple(
            path.name
            for path in Path(where).iterdir()
            if not path.name.startswith(".")
        )

    def _update_cells_model_visibility(self) -> None:
        """Update visibility of cells model selection based on checkbox."""
        self._cells_model_weights_file.visible = (
            self._segment_cells_checkbox.value
        )
        self._update_run_button_state()

    def _update_rings_model_visibility(self) -> None:
        """Update visibility of rings model selection based on checkbox."""
        self._rings_model_weights_file.visible = (
            self._segment_rings_checkbox.value
        )
        self._update_run_button_state()

    def _update_run_button_state(self) -> None:
        """Update enabled state of run button based on checkboxes."""
        self._run_segmentation_button.enabled = (
            self._segment_cells_checkbox.value
            or self._segment_rings_checkbox.value
        )

    def _run_segmentation(self) -> JobHandle | None:
        """Run the segmentation in the background."""
        # Check that there is an input directory
        if self.input_directory_path is None:
            QMessageBox.warning(
                None, "Error", "Please select an input directory"
            )
            return

        # Check if at least one segmentation method is selected
        if not (
            self._segment_cells_checkbox.value
            or self._segment_rings_checkbox.value
        ):
            QMessageBox.warning(
                None, "Error", "Please select at least one segmentation method"
            )
            return

        # Validate cells model if cells segmentation is selected
        self.cells_model_file = None
        if self._segment_cells_checkbox.value:
            self.cells_model_file = self._cells_model_weights_file.value
            if not self.cells_model_file:
                QMessageBox.warning(
                    None, "Error", "Please select a cells model file"
                )
                return

        # Validate rings model if rings segmentation is selected
        self.rings_model_file = None
        if self._segment_rings_checkbox.value:
            self.rings_model_file = self._rings_model_weights_file.value
            if not self.rings_model_file:
                QMessageBox.warning(
                    None, "Error", "Please select a rings model file"
                )
                return

        # The project directory may have changed on disk since the list was
        # built (e.g. images prepared while this widget was open). A manual
        # selection is kept as it is, refreshing would clear it.
        if not self._handpick_files_checkbox.value:
            self._refresh_file_list()

        # Check if files are available to process
        target_files = self._get_selected_file_paths()
        if not target_files:
            show_warning("No files found or selected to process.")
            return

        # --- FIX OUTDATED METADATA STEMS (silent, filesystem-truth based) ---
        if self.input_directory_path:
            fix_sample_stem_paths_in_project(self.input_directory_path)

        default_rings_year = [
            field["default"]
            for field in settings.get("samples_metadata.fields")
            if field["id"] == "rings_outmost_complete_year"
        ][0]

        def on_finished(_handle):
            # Let the widgets listing the project files (9B) rescan them
            notify_project_files_changed(
                self.input_directory_path, source=self
            )

        show_info("Segmentation in progress...")
        return get_runner().submit(
            Job(
                "Batch segmentation",
                segment_samples,
                args=(
                    sorted(target_files),
                    (
                        CELLS_MODELS_PATH / self.cells_model_file
                        if self.cells_model_file
                        else None
                    ),
                    (
                        RINGS_MODELS_PATH / self.rings_model_file
                        if self.rings_model_file
                        else None
                    ),
                    settings.get("processing.try_to_use_gpu"),
                    settings.get("file_extensions.cells_file_extension")[0],
                    settings.get("file_extensions.rings_file_extension")[0],
                    default_rings_year,
                    settings.get("project_directory"),
                ),
                tags={JobTag.SEGMENTATION},
            ),
            on_cancelled=lambda: show_info(
                "Segmentation cancelled (finished images are saved)"
            ),
            on_finished=on_finished,
        )

