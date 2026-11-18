import glob
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from magicgui.widgets import (
    CheckBox,
    Container,
    Label,
    PushButton,
    Select,
)
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QMessageBox

from napari_roxas_ai._job_runner import JobButton, JobHandle, get_runner
from napari_roxas_ai._jobs import Job, JobTag
from napari_roxas_ai._settings import SettingsManager
from napari_roxas_ai._settings._settings_widget import SettingsWidget
from napari_roxas_ai.shortcuts import prevent_duplicate_dock_if_already_present

from .._utils._project_files_events import (
    add_project_files_listener,
    is_same_directory,
)
from ._batch_operations import measure_samples

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

# Label of the manual selection checkbox, which ends in a colon while the file
# list below it is shown
MANUAL_SELECTION_LABEL = "Manually Select Files To Process"


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


def find_samples(directory: str) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Find the samples of a project that have a cells or a rings file.

    Returns
    -------
    dict
        Sample stem (sample directory / sample name, without extensions) ->
        {"cells": cells file path or None, "rings": rings file path or None}.
    """
    samples: Dict[str, Dict[str, Optional[str]]] = {}
    if not directory or not Path(directory).exists():
        return samples

    for kind in ("cells", "rings"):
        content_ext = settings.get(f"file_extensions.{kind}_file_extension")[0]
        for file_path in glob.glob(
            str(Path(directory) / "**" / f"*{content_ext}.*"), recursive=True
        ):
            # "<sample>.rings.tif" -> "<sample>"
            stem = str(Path(file_path).parent / Path(Path(file_path).stem).stem)
            samples.setdefault(stem, {"cells": None, "rings": None})[kind] = (
                file_path
            )

    return dict(sorted(samples.items()))


class BatchSampleMeasurementsWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        # The rows bring their own labels, a label column would only indent them
        super().__init__(labels=False)
        self._viewer = viewer
        prevent_duplicate_dock_if_already_present(
            self, viewer, "9B - Batch measure & update"
        )

        # The project directory is the input directory
        self.input_directory_path = settings.get("project_directory")
        self.samples: Dict[str, Dict[str, Optional[str]]] = {}
        self._path_mapping: Dict[str, str] = {}

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
        # Follow the files other widgets write to the project (preparation,
        # batch segmentation), the list is otherwise only read when the
        # widget opens
        add_project_files_listener(self._on_project_files_changed)

        # Measurement checkboxes
        self._measure_cells_checkbox = CheckBox(
            value=True, label="Measure Cells"
        )
        self._measure_rings_checkbox = CheckBox(
            value=True, label="Measure Rings"
        )

        # Update checkboxes. Measuring keeps both outputs in line with the
        # measurements, so they are locked on while a measurement is selected.
        self._update_annotated_checkbox = CheckBox(
            value=True,
            label="Update Annotated image",
            tooltip="Redraw and save the annotated images from the rings tables",
        )
        self._update_reference_checkbox = CheckBox(
            value=True,
            label="Update Crossdating Reference Files",
            tooltip=(
                "Redraw and save the crossdating reference plots, using the "
                "reference series and outmost complete ring year of each "
                "sample's metadata (the best-matching series if none is "
                "stored; samples without a crossdating file are skipped)"
            ),
        )
        for checkbox in (
            self._measure_cells_checkbox,
            self._measure_rings_checkbox,
        ):
            checkbox.changed.connect(self._sync_update_checkboxes)
        for checkbox in (
            self._update_annotated_checkbox,
            self._update_reference_checkbox,
        ):
            checkbox.changed.connect(self._update_run_button_state)

        # File selection controls
        self._handpick_files_checkbox = CheckBox(
            value=True, label=f"{MANUAL_SELECTION_LABEL}:"
        )
        self._handpick_files_checkbox.changed.connect(
            self._toggle_file_selection
        )

        # File selection list, built once and only refilled afterwards: a list
        # replaced on every refresh would be left to Python's garbage collector,
        # which may destroy it in the worker thread while its Qt timers are still
        # registered with the main thread, crashing napari
        self._file_select_widget = Select(
            choices=[],
            allow_multiple=True,
            label="Select Files to Process",
        )
        self._file_select_widget.reset_choices = (
            self._update_file_selection_widget
        )
        self._file_selection_container = Container(
            widgets=[self._file_select_widget],
            labels=False,
            layout="vertical",
            visible=False,
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

        # Button to open settings focused on measurements
        self._check_settings_button = PushButton(text="Check Settings")
        self._check_settings_button.changed.connect(self._open_settings)

        # Create a button to launch the analysis ("&&" shows a literal "&",
        # a single one would mark a keyboard shortcut)
        # Shows the progress while it runs, can be cancelled between samples
        self._run_analysis_button = JobButton(
            text="Measure && Update", start=self._run_analysis
        )

        # Append the widgets to the container
        self.extend(
            [
                self._project_directory_row,
                self._handpick_files_checkbox,
                self._file_selection_container,
                self._reverse_selection_button,
                self._measure_cells_checkbox,
                self._measure_rings_checkbox,
                self._update_annotated_checkbox,
                self._update_reference_checkbox,
                self._check_settings_button,
                self._run_analysis_button,
            ]
        )
        if hasattr(self.native, "layout") and self.native.layout() is not None:
            self.native.layout().setAlignment(Qt.AlignTop)
            self.native.layout().setAlignment(
                self._file_selection_container.native, Qt.AlignTop
            )

        self._sync_update_checkboxes()
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

    def _open_settings(self):
        """Open SettingsWidget as a floating window centered on the active screen with measurements expanded."""
        SettingsWidget.open_floating(self._viewer, section_name="measurements")

    def _sync_update_checkboxes(self):
        """
        Lock the update checkboxes on while a measurement is selected, so the
        annotated images and reference plots stay in line with the
        measurements. Without a measurement they can be chosen freely.
        """
        measuring = (
            self._measure_cells_checkbox.value
            or self._measure_rings_checkbox.value
        )
        for checkbox in (
            self._update_annotated_checkbox,
            self._update_reference_checkbox,
        ):
            if measuring:
                checkbox.value = True
            checkbox.enabled = not measuring
        self._update_run_button_state()

    def _update_run_button_state(self):
        """Enable the run button only if there is something to do."""
        self._run_analysis_button.enabled = any(
            checkbox.value
            for checkbox in (
                self._measure_cells_checkbox,
                self._measure_rings_checkbox,
                self._update_annotated_checkbox,
                self._update_reference_checkbox,
            )
        )

    def _on_project_files_changed(self, directory):
        """Rescan the sample list after another widget wrote files to the project."""
        if not is_same_directory(directory, self.input_directory_path):
            return

        # Keep a manual selection, as far as the samples are still listed
        selection = []
        if self._handpick_files_checkbox.value:
            selection = list(self._file_select_widget.value)

        self._refresh_file_list()

        if selection and self.samples:
            choices = self._file_select_widget.choices
            self._file_select_widget.value = [
                name for name in selection if name in choices
            ]

    def _refresh_file_list(self):
        """Refresh the list of samples from the project directory."""
        self.samples = find_samples(self.input_directory_path)
        if self._handpick_files_checkbox.value:
            self._update_file_selection_widget()

    def _update_file_selection_widget(self):
        """Update the file selection widget with the current samples."""
        self._path_mapping = {}

        if not self.samples:
            self._file_select_widget.choices = []
            self._file_selection_container.visible = False
            self._reverse_selection_button.visible = False
            return

        display_names = []
        for stem in self.samples:
            try:
                display_name = Path(stem).relative_to(
                    self.input_directory_path
                ).as_posix()
            except ValueError:
                display_name = Path(stem).name
            display_names.append(display_name)
            self._path_mapping[display_name] = stem

        self._file_select_widget.choices = display_names
        self._fit_file_list_height()

        self._file_selection_container.visible = True
        self._reverse_selection_button.visible = True

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
        """Show or hide the file selection interface."""
        selecting = self._handpick_files_checkbox.value
        self._handpick_files_checkbox.text = (
            f"{MANUAL_SELECTION_LABEL}:" if selecting else MANUAL_SELECTION_LABEL
        )
        if selecting:
            self._update_file_selection_widget()
        else:
            self._file_selection_container.visible = False
            self._reverse_selection_button.visible = False

    def _reverse_file_selection(self):
        """Reverse the current file selection."""
        current_selection = set(self._file_select_widget.value)
        self._file_select_widget.value = [
            name
            for name in self._file_select_widget.choices
            if name not in current_selection
        ]

    def _get_selected_samples(self) -> Dict[str, Dict[str, Optional[str]]]:
        """The samples to process: the selected ones, or all of them."""
        if self._handpick_files_checkbox.value:
            selected = {
                self._path_mapping[name]
                for name in self._file_select_widget.value
                if name in self._path_mapping
            }
            return {
                stem: files
                for stem, files in self.samples.items()
                if stem in selected
            }
        return self.samples

    def _run_analysis(self) -> JobHandle | None:
        """Run the measurements and updates in the background."""
        # Check that there is an input directory
        if self.input_directory_path is None:
            QMessageBox.warning(
                None, "Error", "Please select an input directory"
            )
            return None

        # Without a manual selection, take the project as it is now
        if not self._handpick_files_checkbox.value:
            self.samples = find_samples(self.input_directory_path)

        samples = self._get_selected_samples()
        if not samples:
            show_warning("No files found or selected to process.")
            return None

        config = {
            "cluster_dbl_cwt_threshold": settings.get(
                "measurements.cluster_dbl_cwt_threshold"
            ),
            "smoothing_kernel_size": settings.get(
                "measurements.cells_smoothing_kernel_size"
            ),
            "relwidth_cwt_integration": settings.get(
                "measurements.relwidth_cwt_integration"
            ),
            "tangential_angle": settings.get(
                "measurements.cells_tangential_angle"
            ),
            "lower_limit_cwt_iqr_multiplier": settings.get(
                "measurements.lower_limit_cwt_iqr_multiplier"
            ),
            "upper_limit_cwt_iqr_multiplier": settings.get(
                "measurements.upper_limit_cwt_iqr_multiplier"
            ),
            "opposite_cwt_ratio_limit": settings.get(
                "measurements.opposite_cwt_ratio_limit"
            ),
            "opposite_cwt_percentile_exception": settings.get(
                "measurements.opposite_cwt_percentile_exception"
            ),
            "adjacent_cwt_ratio_limit": settings.get(
                "measurements.adjacent_cwt_ratio_limit"
            ),
            "sample_interval": settings.get(
                "cwt_visualization.sample_interval", 25
            ),
        }

        show_info("Batch measure & update in progress...")
        return get_runner().submit(
            Job(
                "Batch measuring",
                measure_samples,
                args=(config, dict(samples)),
                kwargs={
                    "measure_cells": self._measure_cells_checkbox.value,
                    "measure_rings": self._measure_rings_checkbox.value,
                    "update_annotated_images": (
                        self._update_annotated_checkbox.value
                    ),
                    "update_reference_plots": (
                        self._update_reference_checkbox.value
                    ),
                },
                tags={JobTag.MEASUREMENTS},
            ),
            on_result=lambda _: show_info("Batch measure & update complete"),
            on_cancelled=lambda: show_info(
                "Batch measure & update cancelled (finished samples are saved)"
            ),
        )
