import glob
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    Container,
    FloatSpinBox,
    ProgressBar,
    PushButton,
    SpinBox,
)
from PIL import Image
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import QFileDialog, QMessageBox

from napari_roxas_ai._reader import read_cells_file, read_rings_file
from napari_roxas_ai._settings import SettingsManager
from napari_roxas_ai._writer import write_single_layer

from .._utils._metadata_keys import MEASUREMENT_PARAMETER_KEYS
from .._utils._version_utils import (
    get_measurement_operator,
    get_software_version,
)
from ._sample_measurer import SampleAnalyzer

if TYPE_CHECKING:
    import napari

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None

settings = SettingsManager()


class Worker(QObject):
    finished = Signal()
    progress = Signal(int, int)  # current, total

    def __init__(
        self,
        config: Dict[str, Any],
        input_directory_path: str,
        measure_cells: bool = True,
        measure_rings: bool = True,
    ):
        super().__init__()
        self.config = config
        self.input_directory_path = input_directory_path
        self.measure_cells = measure_cells
        self.measure_rings = measure_rings

        self.cells_content_ext = settings.get(
            "file_extensions.cells_file_extension"
        )[0]
        self.rings_content_ext = settings.get(
            "file_extensions.rings_file_extension"
        )[0]

        if self.measure_cells and self.measure_rings:
            self.measurement = "both"
            cells_files = sorted(
                glob.glob(
                    str(
                        Path(self.input_directory_path)
                        / "**"
                        / f"*{self.cells_content_ext}.*"
                    ),
                    recursive=True,
                )
            )
            rings_files = sorted(
                glob.glob(
                    str(
                        Path(self.input_directory_path)
                        / "**"
                        / f"*{self.rings_content_ext}.*"
                    ),
                    recursive=True,
                )
            )
            cells_samples = [Path(Path(f).stem).stem for f in cells_files]
            rings_samples = [Path(Path(f).stem).stem for f in rings_files]
            measured_samples = set(cells_samples).intersection(
                set(rings_samples)
            )
            self.cells_file_paths = [
                f
                for f in cells_files
                if cells_samples[cells_files.index(f)] in measured_samples
            ]
            self.rings_file_paths = [
                f
                for f in rings_files
                if rings_samples[rings_files.index(f)] in measured_samples
            ]

        elif self.measure_cells and not self.measure_rings:
            self.measurement = "cells"
            self.cells_file_paths = sorted(
                glob.glob(
                    str(
                        Path(self.input_directory_path)
                        / "**"
                        / f"*{self.cells_content_ext}.*"
                    ),
                    recursive=True,
                )
            )

        elif self.measure_rings and not self.measure_cells:
            self.measurement = "rings"
            self.rings_file_paths = sorted(
                glob.glob(
                    str(
                        Path(self.input_directory_path)
                        / "**"
                        / f"*{self.rings_content_ext}.*"
                    ),
                    recursive=True,
                )
            )

    def run(self):

        # factor = int(self.measure_cells) + int(self.measure_rings)
        # total = len(self.cells_file_paths) * factor if self.measure_cells else len(self.rings_file_paths) * factor
        total = (
            len(self.cells_file_paths)
            if self.measure_cells
            else len(self.rings_file_paths)
        )
        i = 0

        # Constant for the whole batch run
        sw_version = get_software_version()
        meas_by = get_measurement_operator()

        if self.measurement == "both":
            for cells_file_path, rings_file_path in zip(
                self.cells_file_paths, self.rings_file_paths
            ):
                # Emit progress signal
                self.progress.emit(i, total)
                i += 1
                # Read the files
                cells_data, cells_add_kwargs, _ = read_cells_file(
                    cells_file_path
                )
                rings_data, rings_add_kwargs, _ = read_rings_file(
                    rings_file_path
                )

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
                    add_kwargs["metadata"]["sw_version"] = sw_version
                    add_kwargs["metadata"]["meas_by"] = meas_by
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

        elif self.measurement == "cells":
            for cells_file_path in self.cells_file_paths:
                # Emit progress signal
                self.progress.emit(i, total)
                i += 1
                # Read the files
                cells_data, cells_add_kwargs, _ = read_cells_file(
                    cells_file_path
                )
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
                cells_add_kwargs["metadata"]["sw_version"] = sw_version
                cells_add_kwargs["metadata"]["meas_by"] = meas_by
                # Cells-only parameters, recorded so the run can be reproduced
                for key in MEASUREMENT_PARAMETER_KEYS:
                    cells_add_kwargs["metadata"][key] = self.config[key]
                # Save to file (the file extension in the path argument is ignored)
                write_single_layer(
                    path=cells_file_path,
                    data=cells_data,
                    meta=cells_add_kwargs,
                )

        elif self.measurement == "rings":
            for rings_file_path in self.rings_file_paths:
                # Emit progress signal
                self.progress.emit(i, total)
                i += 1
                # Read the files
                rings_data, rings_add_kwargs, _ = read_rings_file(
                    rings_file_path
                )
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
                rings_add_kwargs["metadata"]["sw_version"] = sw_version
                rings_add_kwargs["metadata"]["meas_by"] = meas_by
                # Save to file (the file extension in the path argument is ignored)
                write_single_layer(
                    path=rings_file_path,
                    data=rings_data,
                    meta=rings_add_kwargs,
                )

        self.progress.emit(total, total)
        self.finished.emit()


class BatchSampleMeasurementsWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # Get input directory
        self.input_directory_path = settings.get("project_directory")
        self._input_file_dialog_button = PushButton(
            text=f"Input Directory: {self.input_directory_path}",
        )
        self._input_file_dialog_button.changed.connect(
            self._open_input_file_dialog
        )

        # Cells measurement checkbox and settings (currently hidden as we want the user to compute cells first anyway)
        self._measure_cells_checkbox = CheckBox(
            value=True, label="Measure Cells"
        )
        self._measure_cells_checkbox.changed.connect(
            self._update_cells_settings_visibility
        )
        # self._measure_cells_checkbox.visible = False

        # Create input fields for the config parameters
        self._cluster_dbl_cwt_threshold = FloatSpinBox(
            value=settings.get("measurements.cluster_dbl_cwt_threshold"),
            label="Cluster DBL CWT Threshold (µm)",
        )
        self._smoothing_kernel_size = SpinBox(
            value=settings.get("measurements.cells_smoothing_kernel_size"),
            label="Smoothing Kernel Size (1 to disable)",
        )
        self._relwidth_cwt_integration = FloatSpinBox(
            value=settings.get("measurements.relwidth_cwt_integration"),
            label="Wall Fraction for Thickness Measurement",
        )

        # Create a container for the cells measurements settings
        self._cells_measurements_settings = Container()
        self._cells_measurements_settings.extend(
            [
                self._cluster_dbl_cwt_threshold,
                self._smoothing_kernel_size,
                self._relwidth_cwt_integration,
            ]
        )

        # Rings measurement checkbox and settings
        self._measure_rings_checkbox = CheckBox(
            value=True, label="Measure Rigns"
        )

        # Create a button to launch the analysis
        self._run_analysis_button = PushButton(text="Run Analysis")
        self._run_analysis_button.changed.connect(self._run_analysis)

        # Add a progress bar with a description
        self.progress_bar = ProgressBar(
            value=0, min=0, max=100, visible=False, label="Progress"
        )

        # Append the widgets to the container
        self.extend(
            [
                self._input_file_dialog_button,
                self._measure_cells_checkbox,
                self._cells_measurements_settings,
                self._measure_rings_checkbox,
                self._run_analysis_button,
                self.progress_bar,
            ]
        )

    def _open_input_file_dialog(self):
        """Open a file dialog to select the input directory path."""
        directory = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select Input Directory",
            directory=self.input_directory_path,
        )
        if directory:
            self.input_directory_path = directory
            self._input_file_dialog_button.text = (
                f"Input Directory: {self.input_directory_path}"
            )

    def _update_cells_settings_visibility(self):
        self._cells_measurements_settings.visible = (
            self._measure_cells_checkbox.value
        )

    def _update_progress(self, current, total):
        """Update the progress bar."""
        if total > 0:
            percentage = int(100 * current / total)
            self.progress_bar.value = percentage
            self.progress_bar.visible = True

    def _run_analysis(self) -> None:
        """Run the analysis analysis in a separate thread."""
        # Check that there is an input directory
        if self.input_directory_path is None:
            QMessageBox.warning(
                None, "Error", "Please select an input directory"
            )
            return

        # Check if at least one analysis method is selected
        if not (
            self._measure_cells_checkbox.value
            or self._measure_rings_checkbox.value
        ):
            QMessageBox.warning(
                None, "Error", "Please select at least one measurement method"
            )
            return

        # Get other parameters
        config = {
            # Rounded to drop the float noise that spin box stepping produces
            # (e.g. 3.5000000000000004), since this value is also recorded in
            # the sample metadata.
            "cluster_dbl_cwt_threshold": round(
                self._cluster_dbl_cwt_threshold.value, 6
            ),
            "smoothing_kernel_size": self._smoothing_kernel_size.value,
            "relwidth_cwt_integration": round(
                self._relwidth_cwt_integration.value, 6
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
            self.input_directory_path,
            self._measure_cells_checkbox.value,
            self._measure_rings_checkbox.value,
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
        self.worker_thread.finished.connect(
            lambda: setattr(self._run_analysis_button, "enabled", True)
        )

        # Run the analysis in a separate thread
        self.worker_thread.start()
