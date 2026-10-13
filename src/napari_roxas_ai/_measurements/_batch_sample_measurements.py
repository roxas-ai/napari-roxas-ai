import gc
import glob
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    Container,
    Label,
    ProgressBar,
    PushButton,
    Select,
)
from napari.utils.notifications import show_info, show_warning
from PIL import Image
from qtpy.QtCore import QObject, Qt, QThread, Signal
from qtpy.QtWidgets import QFileDialog, QMessageBox

from napari_roxas_ai._crossdating._cross_dating_plotter import (
    build_crossdating_plot_df,
    compute_alignment_candidates,
    find_crossdating_files,
    reference_plot_path,
    ring_width_series,
    save_crossdating_plot,
)
from napari_roxas_ai._reader import read_cells_file, read_rings_file
from napari_roxas_ai._reader._crossdating_reader import read_crossdating_file
from napari_roxas_ai._settings import SettingsManager
from napari_roxas_ai._settings._settings_widget import SettingsWidget
from napari_roxas_ai._writer import write_single_layer
from napari_roxas_ai._writer._writer import save_sample_annotated_image

from .._utils._metadata_keys import (
    MEASUREMENT_PARAMETER_KEYS,
    NO_REFERENCE_SERIES,
)
from .._utils._version_utils import (
    get_measurement_operator,
    get_software_version,
)
from ._sample_measurer import SampleAnalyzer

if TYPE_CHECKING:
    import napari

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None

# The file list is at least this many rows high and otherwise grows with
# its content, one row per file, up to MAX_VISIBLE_IMAGE_ROWS
MIN_VISIBLE_IMAGE_ROWS = 5
MAX_VISIBLE_IMAGE_ROWS = 10

# Gap between a row's label and its box. Set explicitly because the style's
# default depends on the widget pair (on macOS label-to-button is wider than
# label-to-list), which would make the button and the list start unevenly.
LABEL_SPACING = 10

# Longest project directory shown on the button, longer paths are elided in the
# middle (the full path is in the tooltip)
MAX_PROJECT_DIRECTORY_CHARS = 35

# Label of the manual selection checkbox, which ends in a colon while the file
# list below it is shown
MANUAL_SELECTION_LABEL = "Manually Select Files To Process"

# Default outmost complete ring year of an undated sample (project convention)
UNDATED_OUTMOST_YEAR = 9999


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


class Worker(QObject):
    finished = Signal()
    progress = Signal(int, int)  # current, total

    def __init__(
        self,
        config: Dict[str, Any],
        samples: Dict[str, Dict[str, Optional[str]]],
        measure_cells: bool = True,
        measure_rings: bool = True,
        update_annotated_images: bool = True,
        update_reference_plots: bool = True,
    ):
        super().__init__()
        self.config = config
        self.samples = samples
        self.measure_cells = measure_cells
        self.measure_rings = measure_rings
        self.update_annotated_images = update_annotated_images
        self.update_reference_plots = update_reference_plots

        # Crossdating files read so far, as most samples of a project share one
        self._crossdating_dataframes: Dict[Path, pd.DataFrame] = {}

    def run(self):
        total = len(self.samples)

        # Constant for the whole batch run
        self._sw_version = get_software_version()
        self._meas_by = get_measurement_operator()

        processed = 0
        for file_idx, (stem, files) in enumerate(self.samples.items(), 1):
            self.progress.emit(file_idx - 1, total)
            name = Path(stem).name
            try:
                # The rings as read or measured, shared by the later steps
                rings = None
                rings_written = False
                tasks = []

                if self.measure_cells or self.measure_rings:
                    rings, rings_written, task = self._measure(files)
                    tasks.append(task)

                if self.update_annotated_images:
                    if rings_written:
                        # Writing a rings file already updates the annotated image
                        tasks.append("Annotated image")
                    else:
                        rings = rings or self._read_rings(files)
                        tasks.append(self._update_annotated_image(stem, rings))

                if self.update_reference_plots:
                    rings = rings or self._read_rings(files)
                    tasks.append(self._update_reference_plot(stem, rings))

                print(
                    f"[Batch Measurements] [{file_idx}/{total}] {name}: {', '.join(tasks)}"
                )
                processed += 1
            except Exception as e:
                print(
                    f"[Batch Measurements] [{file_idx}/{total}] Error processing {name}: {e}"
                )

        if total > 0:
            print(
                f"[Batch Measurements] Completed {processed} of {total} file(s) successfully."
            )

        self.progress.emit(total, total)
        self.finished.emit()

    def _measure(
        self, files: Dict[str, Optional[str]]
    ) -> Tuple[Optional[Tuple[np.ndarray, dict]], bool, str]:
        """
        Measure one sample and save the results.

        Returns the rings (data, add_kwargs) of the sample if they were read,
        whether the rings file was written, and what was done (for the log).
        """
        cells_file_path = files["cells"]
        rings_file_path = files["rings"]

        if self.measure_cells and cells_file_path is None:
            return None, False, "Measurement skipped (no cells file)"
        if self.measure_rings and rings_file_path is None:
            return None, False, "Measurement skipped (no rings file)"

        if self.measure_cells and self.measure_rings:
            # Read the files
            cells_data, cells_add_kwargs, _ = read_cells_file(cells_file_path)
            rings_data, rings_add_kwargs, _ = read_rings_file(rings_file_path)

            self.config["pixels_per_um"] = cells_add_kwargs["metadata"][
                "spatial_resolution"
            ]
            self.config["sample_type"] = (
                (cells_add_kwargs.get("metadata") or {}).get("sample_type")
                or (rings_add_kwargs.get("metadata") or {}).get("sample_type")
                or "conifer"
            )

            analyzer = SampleAnalyzer(
                self.config,
                cells_data.astype("uint8") * 255,
                rings_add_kwargs["features"],
            )
            cells_table, rings_table = analyzer.analyze_sample()

            cells_add_kwargs["features"] = cells_table
            rings_add_kwargs["features"] = rings_table

            # One timestamp per sample, shared by its cells and rings output
            meas_created_at = datetime.now().isoformat()
            for add_kwargs in (cells_add_kwargs, rings_add_kwargs):
                add_kwargs["metadata"]["meas_created_at"] = meas_created_at
                add_kwargs["metadata"]["sw_version"] = self._sw_version
                add_kwargs["metadata"]["meas_by"] = self._meas_by
            # Cells-only parameters, recorded so the run can be reproduced
            for key in MEASUREMENT_PARAMETER_KEYS:
                cells_add_kwargs["metadata"][key] = self.config[key]

            # Save to file (the file extension in the path argument is ignored)
            write_single_layer(
                path=cells_file_path,
                data=cells_data,
                meta=cells_add_kwargs,
            )
            write_single_layer(
                path=rings_file_path,
                data=rings_data,
                meta=rings_add_kwargs,
            )
            return (rings_data, rings_add_kwargs), True, "Cells, Rings"

        if self.measure_cells:
            # Read the files
            cells_data, cells_add_kwargs, _ = read_cells_file(cells_file_path)
            self.config["pixels_per_um"] = cells_add_kwargs["metadata"][
                "spatial_resolution"
            ]
            self.config["sample_type"] = (
                (cells_add_kwargs.get("metadata") or {}).get("sample_type")
                or "conifer"
            )
            analyzer = SampleAnalyzer(
                self.config,
                cells_data.astype("uint8") * 255,
                pd.DataFrame(),
            )
            cells_table = analyzer.analyze_cells()

            cells_add_kwargs["features"] = cells_table
            cells_add_kwargs["metadata"][
                "meas_created_at"
            ] = datetime.now().isoformat()
            cells_add_kwargs["metadata"]["sw_version"] = self._sw_version
            cells_add_kwargs["metadata"]["meas_by"] = self._meas_by
            # Cells-only parameters, recorded so the run can be reproduced
            for key in MEASUREMENT_PARAMETER_KEYS:
                cells_add_kwargs["metadata"][key] = self.config[key]
            # Save to file (the file extension in the path argument is ignored)
            write_single_layer(
                path=cells_file_path,
                data=cells_data,
                meta=cells_add_kwargs,
            )
            return None, False, "Cells"

        # Rings only
        rings_data, rings_add_kwargs, _ = read_rings_file(rings_file_path)
        self.config["pixels_per_um"] = rings_add_kwargs["metadata"][
            "spatial_resolution"
        ]
        self.config["sample_type"] = (
            (rings_add_kwargs.get("metadata") or {}).get("sample_type")
            or "conifer"
        )
        analyzer = SampleAnalyzer(
            self.config,
            np.zeros_like(rings_data),
            rings_add_kwargs["features"],
        )
        rings_table = analyzer.analyze_rings()

        rings_add_kwargs["features"] = rings_table
        rings_add_kwargs["metadata"][
            "meas_created_at"
        ] = datetime.now().isoformat()
        rings_add_kwargs["metadata"]["sw_version"] = self._sw_version
        rings_add_kwargs["metadata"]["meas_by"] = self._meas_by
        # Save to file (the file extension in the path argument is ignored)
        write_single_layer(
            path=rings_file_path,
            data=rings_data,
            meta=rings_add_kwargs,
        )
        return (rings_data, rings_add_kwargs), True, "Rings"

    @staticmethod
    def _read_rings(
        files: Dict[str, Optional[str]]
    ) -> Optional[Tuple[np.ndarray, dict]]:
        """Read the rings file of a sample, None if it has none."""
        if files["rings"] is None:
            return None
        rings_data, rings_add_kwargs, _ = read_rings_file(files["rings"])
        return rings_data, rings_add_kwargs

    @staticmethod
    def _rings_features(
        rings: Optional[Tuple[np.ndarray, dict]]
    ) -> Optional[pd.DataFrame]:
        """The rings table, None if there is none or it is empty."""
        if rings is None:
            return None
        features = rings[1].get("features")
        if features is None or features.empty:
            return None
        return features

    def _update_annotated_image(
        self, stem: str, rings: Optional[Tuple[np.ndarray, dict]]
    ) -> str:
        """
        Redraw the annotated image of a sample from its rings table.

        Returns what was done (for the log).
        """
        features = self._rings_features(rings)
        if features is None:
            return "Annotated image skipped (no rings table)"
        if save_sample_annotated_image(stem, features) is None:
            return "Annotated image skipped (no scan file)"
        return "Annotated image"

    def _crossdating_dataframe(self, path: Path) -> pd.DataFrame:
        """Read a crossdating file and add its average series (cached)."""
        if path not in self._crossdating_dataframes:
            dataframe = read_crossdating_file(str(path))
            dataframe["average"] = dataframe.mean(axis=1, skipna=True)
            self._crossdating_dataframes[path] = dataframe
        return self._crossdating_dataframes[path]

    def _update_reference_plot(
        self, stem: str, rings: Optional[Tuple[np.ndarray, dict]]
    ) -> str:
        """
        Redraw the crossdating reference plot of a sample.

        The sample is plotted as dated by the outmost complete ring year of its
        metadata, against its stored reference series. A sample without one
        ("NA") is plotted against the series that matches it best, as found by
        the automatic alignment of the visual crossdating. Samples without a
        crossdating file are skipped.

        Returns what was done (for the log).
        """
        name = Path(stem).name
        features = self._rings_features(rings)
        if (
            features is None
            or "YEAR" not in features.columns
            or "cells_above" not in features.columns
        ):
            return "Reference plot skipped (no rings table)"

        rings_data, rings_add_kwargs = rings
        metadata = rings_add_kwargs.get("metadata") or {}
        if "spatial_resolution" not in metadata:
            return "Reference plot skipped (no spatial resolution)"

        crossdating_files = find_crossdating_files(Path(stem).parent)
        if not crossdating_files:
            return "Reference plot skipped (no crossdating file)"

        width_series = ring_width_series(
            features,
            metadata.get("rings_outmost_complete_year", UNDATED_OUTMOST_YEAR),
            rings_data.shape[1],
            metadata["spatial_resolution"],
        )

        plot = self._stored_reference_plot_df(
            metadata.get("reference_series"), crossdating_files, width_series
        )
        best_match = plot is None
        if best_match:
            plot = self._best_matching_plot_df(crossdating_files, width_series)
        if plot is None:
            return "Reference plot skipped (no matching reference series)"

        reference_name, plot_df = plot
        sample_name = metadata.get("sample_name") or name
        save_crossdating_plot(
            plot_df,
            reference_name,
            sample_name,
            reference_plot_path(Path(stem).parent, sample_name),
        )
        if best_match:
            return f"Reference plot ({reference_name}, best match)"
        return f"Reference plot ({reference_name})"

    def _stored_reference_plot_df(
        self,
        reference_series: Optional[str],
        crossdating_files: List[Path],
        width_series: pd.Series,
    ) -> Optional[Tuple[str, pd.DataFrame]]:
        """Plot data against the stored reference series, if it is found."""
        if (
            not isinstance(reference_series, str)
            or reference_series == NO_REFERENCE_SERIES
        ):
            return None
        for path in crossdating_files:
            dataframe = self._crossdating_dataframe(path)
            if reference_series in dataframe.columns:
                return reference_series, build_crossdating_plot_df(
                    dataframe, reference_series, width_series
                )
        return None

    def _best_matching_plot_df(
        self, crossdating_files: List[Path], width_series: pd.Series
    ) -> Optional[Tuple[str, pd.DataFrame]]:
        """
        Plot data against the series with the best alignment correlation over
        all crossdating files.
        """
        best = None
        best_corr = -np.inf
        for path in crossdating_files:
            dataframe = self._crossdating_dataframe(path)
            for column in dataframe.columns.drop("average"):
                plot_df = build_crossdating_plot_df(
                    dataframe, column, width_series
                )
                candidates = compute_alignment_candidates(plot_df, top_k=1)
                if candidates and candidates[0]["corr"] > best_corr:
                    best_corr = candidates[0]["corr"]
                    best = (column, plot_df)
        return best


class BatchSampleMeasurementsWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        # The rows bring their own labels, a label column would only indent them
        super().__init__(labels=False)
        self._viewer = viewer

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
            value=False, label=MANUAL_SELECTION_LABEL
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
        self._run_analysis_button = PushButton()
        self._run_analysis_button.text = "Measure && Update"
        self._run_analysis_button.changed.connect(self._run_analysis)

        # Add a progress bar with a description
        self.progress_bar = ProgressBar(
            value=0, min=0, max=100, visible=False, label="Progress"
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
                self.progress_bar,
            ]
        )
        if hasattr(self.native, "layout") and self.native.layout() is not None:
            self.native.layout().setAlignment(Qt.AlignTop)
            self.native.layout().setAlignment(
                self._file_selection_container.native, Qt.AlignTop
            )
            # Spare height goes below the widgets instead of stretching the controls
            self.native.layout().addStretch()

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

    def _update_progress(self, current, total):
        """Update the progress bar."""
        if total > 0:
            percentage = int(100 * current / total)
            self.progress_bar.value = percentage
            self.progress_bar.visible = True

    def _run_analysis(self) -> None:
        """Run the measurements and updates in a separate thread."""
        # Check that there is an input directory
        if self.input_directory_path is None:
            QMessageBox.warning(
                None, "Error", "Please select an input directory"
            )
            return

        # Without a manual selection, take the project as it is now
        if not self._handpick_files_checkbox.value:
            self.samples = find_samples(self.input_directory_path)

        samples = self._get_selected_samples()
        if not samples:
            show_warning("No files found or selected to process.")
            return

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
            "adjacent_cwt_ratio_limit": settings.get(
                "measurements.adjacent_cwt_ratio_limit"
            ),
        }

        # Prepare worker to run the analysis in a separate thread
        self.worker_thread = QThread()
        self.worker = Worker(
            config,
            samples,
            self._measure_cells_checkbox.value,
            self._measure_rings_checkbox.value,
            self._update_annotated_checkbox.value,
            self._update_reference_checkbox.value,
        )
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # Connect progress signal
        self.worker.progress.connect(self._update_progress)

        # Disable the run button while processing
        self._run_analysis_button.enabled = False

        def on_finished():
            self._update_run_button_state()
            show_info("Batch measure & update complete")

        self.worker_thread.finished.connect(on_finished)

        show_info("Batch measure & update in progress...")

        # Destroy unreferenced Qt objects here in the main thread. Left to the
        # garbage collector, they could be destroyed in the worker thread while
        # their timers are still registered with the main thread, which crashes
        # napari.
        gc.collect()

        # Run the analysis in a separate thread
        self.worker_thread.start()
